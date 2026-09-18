#!/usr/bin/env python3
"""Build predictor anchors from S_* scheduler traces (inference only).

Reuses the exact field derivations of the P9d builder (role/action/family,
task/stack context, history tokens, causal-input audit) so that the frozen J
predictor sees the same input schema it was trained on.  Emits one anchor per
trace event (documented choice) with ``model_input`` only; no labels.

Output: features_sstar.jsonl.gz + a coverage report (value frequencies and
out-of-vocabulary rates against the frozen J training vocabulary).
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import j_series_common as common  # noqa: E402

BUILDER_PATH = ROOT / "scripts" / "build_p9d_topology_dataset.py"
_spec = importlib.util.spec_from_file_location("p9d_builder", BUILDER_PATH)
builder = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(builder)  # type: ignore[union-attr]


def read_manifest_groups(path: Path) -> Dict[str, str]:
    groups: Dict[str, str] = {}
    if not path.is_file():
        return groups
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task_id = str(row.get("task_id") or "")
            group = str(row.get("group") or "")
            if task_id and group:
                groups[task_id] = group
    return groups


def video_id_from(run_manifest: Mapping[str, Any], run_id: str) -> str:
    video_id = str(run_manifest.get("video_id") or "")
    if video_id:
        return video_id
    path = str(run_manifest.get("video_path") or "")
    if path:
        return Path(path).stem
    return run_id.split("_", 1)[0]


# --- Phase 21: task-context pipeline repair ---------------------------------
# Pure join logic lives in scripts/preprocess/sstar_task_context.py (unit tested
# without numpy/torch); this module only wires it into anchor construction.
sys.path.insert(0, str(ROOT / "scripts" / "preprocess"))
from sstar_task_context import (  # noqa: E402
    REGISTRY_UNAVAILABLE_FIELDS,
    TASK_CONTEXT_FIELDS,
    UNKNOWN,
    enrich_sample_context,
    is_missing,
    load_task_registry,
    read_video_allowlist,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vocab-source", type=Path, required=True, help="J dataset train file used to rebuild the frozen vocabulary")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--task-registry", type=Path, default=None, help="Phase 21: benchmark registry joined by video_id to restore task context")
    parser.add_argument("--video-allowlist", type=Path, default=None, help="optional newline-delimited video_id subset (paired pilot)")
    parser.add_argument("--mask-registry-task-context", action="store_true", help="Phase 21 control: write the registry fields as 'unknown'")
    args = parser.parse_args()

    registry, registry_sha = load_task_registry(args.task_registry)
    allowlist = read_video_allowlist(args.video_allowlist)

    groups = read_manifest_groups(args.manifest)
    runs = sorted(p for p in args.runs_root.iterdir() if p.is_dir() and not p.name.startswith("._"))
    if args.limit:
        runs = runs[: args.limit]

    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    for run in runs:
        trace_path = run / "trace.jsonl"
        manifest_path = run / "run_manifest.json"
        if not trace_path.is_file() or not manifest_path.is_file():
            skipped.append({"run": run.name, "reason": "missing_trace_or_manifest"})
            continue
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = str(run_manifest.get("run_id") or run.name)
        video_id = video_id_from(run_manifest, run_id)
        if allowlist is not None and video_id not in allowlist:
            skipped.append({"run": run.name, "reason": "outside_video_allowlist"})
            continue
        registry_row = registry.get(video_id) if registry else None
        if registry and registry_row is None:
            raise ValueError(f"video_id missing from the task registry: {video_id}")
        enriched_sample, task_audit = enrich_sample_context(run_manifest, registry_row, args.mask_registry_task_context)
        task_id = str(run_manifest.get("task_id") or "")
        group = groups.get(task_id, "s_train" if "s_train" in task_id else ("s_val" if "s_val" in task_id else "unknown"))
        events: List[Dict[str, Any]] = []
        with trace_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        if not events:
            skipped.append({"run": run.name, "reason": "empty_trace"})
            continue
        for index, event in enumerate(events):
            event_id = str(event.get("event_id") or f"{run_id}:event:{index}")
            anchor = {
                "sample_row": enriched_sample,
                "sample_id": f"{run_id}::{event_id}",
                "video_id": video_id,
                "registry_video_id": video_id,
                "run_id": run_id,
                "split": group.lower(),
                "current_node_id": event_id,
                "current_event_index": index,
            }
            feature = builder.make_feature(anchor, events, index, enriched_sample)
            feature["scheduler_group"] = group
            feature["task_context_audit"] = task_audit
            rows.append(feature)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

    # coverage report against the frozen J vocabulary
    train_rows = list(common.read_jsonl_gz(args.vocab_source))
    vocabs = common.build_vocabs(train_rows)
    coverage: Dict[str, Any] = {"anchors": len(rows), "runs": len(runs), "skipped": skipped}
    field_stats: Dict[str, Any] = {}
    for field in ("event_type", "node_type", "raw_action", "action_family", "model_id"):
        seen = Counter()
        oov = Counter()
        for row in rows:
            for token in row["model_input"]["history"]:
                value = str(token.get(field))
                seen[value] += 1
                if value not in vocabs.history[field].index:
                    oov[value] += 1
        field_stats[field] = {
            "distinct": len(seen),
            "total": sum(seen.values()),
            "oov_tokens": sum(oov.values()),
            "oov_rate": sum(oov.values()) / max(1, sum(seen.values())),
            "top_values": seen.most_common(12),
            "oov_values": oov.most_common(12),
        }
    registry_backed = set(TASK_CONTEXT_FIELDS)
    for field, vocab in (("answer_type", vocabs.context["answer_type"]), ("domain", vocabs.context["domain"]), ("official_task_type", vocabs.context["official_task_type"]), ("question_type", vocabs.context["question_type"]), ("sub_category", vocabs.context["sub_category"]), ("temporal_scope", vocabs.context["temporal_scope"]), ("baseline", vocabs.context["baseline"]), ("model_stack_id", vocabs.context["model_stack_id"]), ("planner_model_id", vocabs.context["planner_model_id"])):
        seen = Counter()
        oov_all = 0
        oov_non_missing = 0
        non_missing = 0
        missing = 0
        from_registry = 0
        bad_type = 0
        for row in rows:
            task_context = row["model_input"]["task_context"]
            value = task_context.get(field) if field in task_context else row["model_input"]["stack_context"].get(field)
            if is_missing(value):
                missing += 1
            else:
                non_missing += 1
            rendered = str(value)
            seen[rendered] += 1
            if rendered not in vocab.index:
                oov_all += 1
                if not is_missing(value):
                    oov_non_missing += 1
            if field in registry_backed and row.get("task_context_audit", {}).get("recovered_fields"):
                if field in row["task_context_audit"]["recovered_fields"]:
                    from_registry += 1
            if field == "required_modalities" and value is not None and not isinstance(value, list):
                bad_type += 1
        field_stats[field] = {
            "distinct": len(seen),
            "total": len(rows),
            "missing": missing,
            "missing_rate": missing / max(1, len(rows)),
            "unknown_count": seen.get(UNKNOWN, 0),
            "unknown_rate": seen.get(UNKNOWN, 0) / max(1, len(rows)),
            "oov_rate_all": oov_all / max(1, len(rows)),
            "oov_rate_nonmissing": oov_non_missing / max(1, non_missing),
            "source_coverage": (from_registry / max(1, len(rows))) if field in registry_backed else None,
            "bad_type_count": bad_type,
            "top_values": seen.most_common(8),
        }
        # ``required_modalities`` is list-valued and encoded through the modalities
        # vocabulary (not the context vocabulary), so it is audited separately and
        # stored with the same schema the gate below expects.
        seen = Counter()
        oov_all = 0
        missing = 0
        bad_type = 0
        for row in rows:
            value = row["model_input"]["task_context"].get("required_modalities")
            if is_missing(value) or str(value).strip() == UNKNOWN:
                # the frozen fallback carries the literal "unknown" marker here,
                # which is a documented missing value rather than a type error
                missing += 1
                rendered = UNKNOWN
            elif not isinstance(value, list):
                bad_type += 1
                rendered = str(value)
            else:
                rendered = "|".join(str(item) for item in value)
            seen[rendered] += 1
            if rendered not in vocabs.modalities.index:
                oov_all += 1
        non_missing_rows = max(1, len(rows) - missing)
        field_stats["required_modalities"] = {
            "distinct": len(seen),
            "total": len(rows),
            "missing": missing,
            "missing_rate": missing / max(1, len(rows)),
            "unknown_count": seen.get(UNKNOWN, 0),
            "unknown_rate": seen.get(UNKNOWN, 0) / max(1, len(rows)),
            "oov_rate_all": oov_all / max(1, len(rows)),
            "oov_rate_nonmissing": oov_all / non_missing_rows,
            "source_coverage": None,
            "bad_type_count": bad_type,
            "top_values": seen.most_common(8),
        }
    # every anchor must carry at least one populated task-context field
    all_unknown_rows = 0
    for row in rows:
        task_context = row["model_input"]["task_context"]
        values = [str(task_context.get(field)) for field in TASK_CONTEXT_FIELDS + REGISTRY_UNAVAILABLE_FIELDS]
        if all(value in ("", UNKNOWN) for value in values):
            all_unknown_rows += 1
    coverage["fields"] = field_stats
    coverage["task_context_gate"] = {
        "registry": str(args.task_registry) if args.task_registry else None,
        "registry_sha256": registry_sha,
        "registry_videos": len(registry),
        "masked_control": bool(args.mask_registry_task_context),
        "all_unknown_rows": all_unknown_rows,
        "unique_sample_ids": len({str(row.get("sample_id")) for row in rows}),
    }
    failures: List[str] = []
    if registry and not args.mask_registry_task_context:
        for field in TASK_CONTEXT_FIELDS:
            stats = field_stats[field]
            if stats["unknown_rate"] > 0.0:
                failures.append(f"{field} still unknown for {stats['unknown_count']} rows")
            if stats["missing_rate"] > 0.0:
                failures.append(f"{field} missing for {stats['missing']} rows")
        if all_unknown_rows:
            failures.append(f"{all_unknown_rows} rows have every task-context field unknown")
        if field_stats["required_modalities"]["bad_type_count"]:
            failures.append(f"required_modalities has {field_stats['required_modalities']['bad_type_count']} non-list values")
    coverage["task_context_gate"]["failures"] = failures
    common.write_json(args.output.parent / "coverage_report.json", coverage)
    print(json.dumps({
        "anchors": len(rows),
        "runs": len(runs),
        "skipped": len(skipped),
        "oov_rates": {
            k: round(v.get("oov_rate_all", v.get("oov_rate", 0.0)), 4) for k, v in field_stats.items()
        },
        "task_context": {
            field: {"unknown_rate": round(field_stats[field]["unknown_rate"], 4), "source_coverage": field_stats[field]["source_coverage"]}
            for field in TASK_CONTEXT_FIELDS
        },
        "task_context_failures": failures,
        "output": str(args.output),
    }, ensure_ascii=False, indent=1))
    if failures:
        raise SystemExit("task-context gate failed: " + "; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
