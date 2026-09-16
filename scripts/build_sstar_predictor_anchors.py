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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vocab-source", type=Path, required=True, help="J dataset train file used to rebuild the frozen vocabulary")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

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
                "sample_row": run_manifest,
                "sample_id": f"{run_id}::{event_id}",
                "video_id": video_id,
                "registry_video_id": video_id,
                "run_id": run_id,
                "split": group.lower(),
                "current_node_id": event_id,
                "current_event_index": index,
            }
            feature = builder.make_feature(anchor, events, index, run_manifest)
            feature["scheduler_group"] = group
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
    for field, vocab in (("answer_type", vocabs.context["answer_type"]), ("domain", vocabs.context["domain"]), ("official_task_type", vocabs.context["official_task_type"]), ("question_type", vocabs.context["question_type"]), ("sub_category", vocabs.context["sub_category"]), ("temporal_scope", vocabs.context["temporal_scope"]), ("baseline", vocabs.context["baseline"]), ("model_stack_id", vocabs.context["model_stack_id"]), ("planner_model_id", vocabs.context["planner_model_id"])):
        oov = 0
        seen = Counter()
        for row in rows:
            value = str(row["model_input"]["task_context"].get(field) if field in row["model_input"]["task_context"] else row["model_input"]["stack_context"].get(field))
            seen[value] += 1
            if value not in vocab.index:
                oov += 1
        field_stats[field] = {"distinct": len(seen), "total": len(rows), "oov_tokens": oov, "oov_rate": oov / max(1, len(rows)), "top_values": seen.most_common(8)}
    coverage["fields"] = field_stats
    common.write_json(args.output.parent / "coverage_report.json", coverage)
    print(json.dumps({
        "anchors": len(rows),
        "runs": len(runs),
        "skipped": len(skipped),
        "oov_rates": {k: round(v["oov_rate"], 4) for k, v in field_stats.items()},
        "output": str(args.output),
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
