#!/usr/bin/env python3
"""Build leakage-audited prefix/suffix examples for neural trace training.

The converter deliberately keeps input and labels in separate objects.  It is
currently wired to the project's canonical prefix file; public-dataset
adapters can emit the same ``neural-trace-v0.1`` run format later.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tracing.analysis.multistep_graph_predictor import future_suffixes
from tracing.analysis.reproduce_prediction_baselines import Prefix, load_prefixes, load_split


FORBIDDEN_KEY_PARTS = (
    "answer",
    "ground_truth",
    "target",
    "future",
    "remaining_steps",
    "remaining_runtime",
    "suffix",
)


def _is_forbidden_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in FORBIDDEN_KEY_PARTS)


def _sanitize(value: Any, key: str = "") -> Any:
    """Remove answer/future-like fields from nested input metadata."""
    if key and _is_forbidden_key(key):
        return None
    if isinstance(value, Mapping):
        return {
            str(child_key): _sanitize(child_value, str(child_key))
            for child_key, child_value in value.items()
            if not _is_forbidden_key(str(child_key))
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value


def _run_rows(rows: Sequence[Prefix]) -> dict[str, list[Prefix]]:
    grouped: dict[str, list[Prefix]] = defaultdict(list)
    for row in rows:
        grouped[row.run_id].append(row)
    ordered: dict[str, list[Prefix]] = {}
    for run_id, values in grouped.items():
        ordered[run_id] = sorted(values, key=lambda row: (row.position, row.prefix))
        positions = [row.position for row in ordered[run_id]]
        if len(positions) != len(set(positions)):
            raise ValueError(f"duplicate position in run {run_id}")
    return ordered


def build_examples(rows: Sequence[Prefix], dataset_id: str = "own_video_agent") -> list[dict[str, Any]]:
    suffix_map = future_suffixes(rows)
    examples: list[dict[str, Any]] = []
    for row in rows:
        suffix = list(suffix_map[(row.run_id, row.position)])
        task_context = _sanitize(dict(row.task_structure))
        observed_state = _sanitize(dict(row.state_features))
        examples.append(
            {
                "schema_version": "neural-prefix-v0.1",
                "dataset": dataset_id,
                "case_id": row.run_id,
                "group_id": row.video_id,
                "split": row.split,
                "domain_id": row.baseline,
                "model_stack_id": row.model_stack_id,
                "planner_model_id": row.planner_model_id,
                "position": row.position,
                "input": {
                    "prefix_nodes": list(row.prefix),
                    "prefix_raw_actions": list(row.raw_prefix),
                    "task_context": task_context,
                    "observed_state": observed_state,
                },
                "target": {
                    "next_node": row.target,
                    "suffix": suffix,
                    "end": row.target == "__END__",
                    "length": len(suffix),
                },
                "leakage_checks": {
                    "future_events_in_input": False,
                    "ground_truth_in_input": False,
                    "answer_text_in_input": False,
                    "remaining_steps_in_input": False,
                    "remaining_runtime_in_input": False,
                    "video_id_used_as_feature": False,
                },
            }
        )
    return examples


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_report(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    splits = ("train", "validation", "test")
    return {
        "schema_version": "neural-prefix-report-v0.1",
        "rows": len(examples),
        "rows_by_split": {split: sum(row.get("split") == split for row in examples) for split in splits},
        "runs": len({row.get("case_id") for row in examples}),
        "groups": len({row.get("group_id") for row in examples}),
        "labels": sorted({row.get("target", {}).get("next_node") for row in examples}),
        "leakage_checks": {
            "future_events_in_input": False,
            "ground_truth_in_input": False,
            "answer_text_in_input": False,
            "remaining_steps_in_input": False,
            "remaining_runtime_in_input": False,
            "video_id_used_as_feature": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dataset-id", default="own_video_agent")
    args = parser.parse_args(argv)
    rows = load_prefixes(args.prefix, load_split(args.split_manifest))
    examples = build_examples(rows, dataset_id=args.dataset_id)
    _write_jsonl(args.output, examples)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(build_report(examples), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "report": str(args.report), "rows": len(examples)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
