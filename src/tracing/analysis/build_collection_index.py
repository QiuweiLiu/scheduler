#!/usr/bin/env python3
"""Freeze a deduplicated collection index without changing raw traces.

The command always writes an auditable candidate index and reports whether the
640-core/128-resource gate is actually satisfied.  It never pads a missing
slot with a DVD run, a workload episode, or an unknown synthetic row.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _text(value: Any, default: str = "unknown") -> str:
    text = str(value or "").strip()
    return text or default


def _slot(row: Mapping[str, Any], layer: str) -> tuple[str, ...]:
    base = (
        _text(row.get("dataset")),
        _text(row.get("video_id")),
        _text(row.get("base_task_id")),
        _text(row.get("baseline")),
        _text(row.get("model_stack_id")),
    )
    if layer == "resource":
        # Resource/recovery samples intentionally preserve repeated runs when
        # their measurement condition differs; they are not semantic slots.
        return base + (_text(row.get("seed")), _text(row.get("yolo_batch")), _text(row.get("trace_sha256")))
    return base


def build(core_paths: Iterable[Path], resource_paths: Iterable[Path], output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for layer, paths in (("core", core_paths), ("resource", resource_paths)):
        for path in paths:
            for row in _read_jsonl(path):
                if _text(row.get("status")) != "success" or _text(row.get("validator")) != "valid":
                    rejected.append({"run_id": row.get("run_id"), "layer": layer, "reason": "not_success_valid", "source": str(path)})
                    continue
                if layer == "core" and _text(row.get("baseline")) == "dvd_local":
                    rejected.append({"run_id": row.get("run_id"), "layer": layer, "reason": "dvd_is_independent_extension", "source": str(path)})
                    continue
                enriched = dict(row)
                enriched["collection_layer"] = layer
                enriched["source_snapshot"] = str(path)
                rows.append(enriched)
    by_slot: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        by_slot.setdefault(_slot(row, str(row.get("collection_layer"))), []).append(row)
    selected: list[dict[str, Any]] = []
    duplicates = 0
    for slot, candidates in sorted(by_slot.items()):
        candidates.sort(key=lambda row: (str(row.get("run_id")), str(row.get("source_snapshot"))))
        selected.append(candidates[0])
        duplicates += max(0, len(candidates) - 1)
    selected.sort(key=lambda row: (row.get("collection_layer"), row.get("video_id"), row.get("base_task_id"), row.get("baseline"), row.get("model_stack_id"), row.get("run_id")))
    _write_rows = output
    output.parent.mkdir(parents=True, exist_ok=True)
    with _write_rows.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    core_count = sum(row.get("collection_layer") == "core" for row in selected)
    resource_count = sum(row.get("collection_layer") == "resource" for row in selected)
    core_videos = {row.get("video_id") for row in selected if row.get("collection_layer") == "core"}
    summary = {
        "schema_version": "collection-index-v0.1",
        "output": str(output),
        "selected_total": len(selected),
        "core_count": core_count,
        "resource_count": resource_count,
        "core_expected": 640,
        "resource_expected": 128,
        "formal_gate": core_count == 640 and resource_count == 128,
        "core_videos": len(core_videos),
        "baseline_counts": dict(sorted(Counter(row.get("baseline") for row in selected).items())),
        "model_stack_counts": dict(sorted(Counter(row.get("model_stack_id") for row in selected).items())),
        "duplicate_slot_rows": duplicates,
        "rejected_rows": len(rejected),
        "rejected_reason_counts": dict(sorted(Counter(row["reason"] for row in rejected).items())),
        "limitations": ["DVD rows are deliberately independent and never fill a missing core/resource slot."],
    }
    output.with_suffix(output.suffix + ".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".rejected.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rejected) + ("\n" if rejected else ""), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", action="append", type=Path, default=[])
    parser.add_argument("--resource", action="append", type=Path, default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = build(args.core, args.resource, args.output)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
