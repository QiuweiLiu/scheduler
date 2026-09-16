#!/usr/bin/env bash
set -euo pipefail

# Run after the resource completion cohort has finished.  The script keeps the
# 640-core fixed artifacts immutable, selects exactly the missing resource
# reservoir slots, then builds one 768-template workload from both layers.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/scheduler}"
PYTHON="${PYTHON:-/root/miniconda3/envs/finetooling/bin/python}"
STAMP="${STAMP:-20260804}"
cd "${PROJECT_ROOT}"

CORE_DERIVED="results/processed/trace_enrichment_core_fixed_${STAMP}"
RESOURCE_DERIVED="results/processed/trace_enrichment_resource_completion32_${STAMP}"
RESOURCE_EXISTING="results/processed/collection_index_resource_batch_candidates_20260803.jsonl"
RESOURCE_COMPLETION="${RESOURCE_DERIVED}/candidate_snapshot_v0_1.jsonl"
RESOURCE_SELECTED="results/processed/collection_index_resource_fixed_${STAMP}.jsonl"

"${PYTHON}" -m tracing.analysis.build_trace_enrichment \
  --root results/raw/phase3_resource_completion32_20260804 \
  --split-manifest configs/phase3_video_split_48_8_8.jsonl \
  --output-dir "${RESOURCE_DERIVED}"

"${PYTHON}" -m tracing.analysis.select_resource_completion \
  --existing "${RESOURCE_EXISTING}" \
  --completion "${RESOURCE_COMPLETION}" \
  --output "${RESOURCE_SELECTED}" \
  --rejected-output "${RESOURCE_SELECTED}.rejected.jsonl" \
  --expected 128

"${PYTHON}" -m tracing.analysis.build_collection_index \
  --core "${CORE_DERIVED}/candidate_snapshot_v0_1.jsonl" \
  --resource "${RESOURCE_SELECTED}" \
  --output "results/processed/collection_index_core_resource_fixed_${STAMP}.jsonl"

"${PYTHON}" - "results/processed/collection_index_core_resource_fixed_${STAMP}.jsonl.summary.json" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
if not summary.get("formal_gate") or summary.get("core_count") != 640 or summary.get("resource_count") != 128:
    raise SystemExit(f"collection gate failed: {summary}")
PY

# Keep the compute view aligned with the selected run IDs.  Existing resource
# rows point back to their source snapshot; completion rows point to the new
# resource-derived snapshot.  No node is synthesized here.
COMBINED_COMPUTE="results/processed/compute_events_core_resource_fixed_${STAMP}.jsonl"
"${PYTHON}" - "${CORE_DERIVED}/compute_events_v0_1.jsonl" "${RESOURCE_SELECTED}" "${RESOURCE_DERIVED}" "${COMBINED_COMPUTE}" <<'PY'
import json
import sys
from pathlib import Path

core_path, resource_index, resource_derived, output = map(Path, sys.argv[1:])
rows = [json.loads(line) for line in core_path.read_text(encoding="utf-8").splitlines() if line.strip()]
seen = {str(row.get("run_id")) for row in rows}
resource_rows = [json.loads(line) for line in resource_index.read_text(encoding="utf-8").splitlines() if line.strip()]
for row in resource_rows:
    source = Path(str(row.get("source_snapshot", "")))
    if source.name == "candidate_snapshot_v0_1.jsonl":
        source = source.with_name("compute_events_v0_1.jsonl")
    elif row.get("run_id") in {str(item.get("run_id")) for item in rows}:
        continue
    if not source.is_file():
        source = resource_derived / "compute_events_v0_1.jsonl"
    if not source.is_file():
        raise SystemExit(f"missing compute source for {row.get('run_id')}: {source}")
    found = False
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        run_id = str(event.get("run_id"))
        if run_id == str(row.get("run_id")) and run_id not in seen:
            rows.append(event)
            found = True
    if not found:
        raise SystemExit(f"no compute events found for selected resource run: {row.get('run_id')}")
    seen.add(str(row.get("run_id")))
Path(output).write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
PY

"${PYTHON}" -m tracing.workloads.build_workload \
  --snapshot "results/processed/collection_index_core_resource_fixed_${STAMP}.jsonl" \
  --compute "${COMBINED_COMPUTE}" \
  --templates "results/processed/job_templates_core_resource_fixed_${STAMP}.jsonl"

"${PYTHON}" - "results/processed/job_templates_core_resource_fixed_${STAMP}.jsonl.summary.json" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
if summary.get("templates") != 768:
    raise SystemExit(f"template count failed: {summary}")
PY

"${PYTHON}" -m tracing.workloads.build_workload \
  --templates "results/processed/job_templates_core_resource_fixed_${STAMP}.jsonl" \
  --split train --episodes "results/processed/workload_train_core_resource_fixed_${STAMP}.jsonl" \
  --count 20000 --seed 20260804
"${PYTHON}" -m tracing.workloads.build_workload \
  --templates "results/processed/job_templates_core_resource_fixed_${STAMP}.jsonl" \
  --split validation --episodes "results/processed/workload_validation_core_resource_fixed_${STAMP}.jsonl" \
  --count 1000 --seed 20260805
"${PYTHON}" -m tracing.workloads.build_workload \
  --templates "results/processed/job_templates_core_resource_fixed_${STAMP}.jsonl" \
  --split test --episodes "results/processed/workload_test_core_resource_fixed_${STAMP}.jsonl" \
  --count 1000 --seed 20260806

"${PYTHON}" - "results/processed/workload_train_core_resource_fixed_${STAMP}.jsonl.summary.json" "results/processed/workload_validation_core_resource_fixed_${STAMP}.jsonl.summary.json" "results/processed/workload_test_core_resource_fixed_${STAMP}.jsonl.summary.json" <<'PY'
import json
import sys
for path in sys.argv[1:]:
    summary = json.load(open(path, encoding="utf-8"))
    if not summary.get("split_isolation_gate") or summary.get("split_isolation_errors"):
        raise SystemExit(f"workload isolation failed: {summary}")
PY
