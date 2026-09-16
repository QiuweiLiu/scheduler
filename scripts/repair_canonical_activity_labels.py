#!/usr/bin/env python3
"""Repair canonical activity labels in an existing prefix-sample view.

The raw trace and its hashes are immutable.  This migration is used when a
canonicalization bug is found after enrichment: it derives the corrected
activity sequence only from the already-recorded raw actions, keeps the raw
actions and source hashes, and writes a new versioned JSONL view.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tracing.collectors.structured_state import canonical_action


MIGRATION_VERSION = "canonical-activity-repair-v0.1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        rows.append(value)
    return rows


def _copy_state_features(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result = dict(value)
    prefix = value.get("prefix")
    result["prefix"] = dict(prefix) if isinstance(prefix, Mapping) else {}
    return result


def repair_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repaired: list[dict[str, Any]] = []
    mapping_counts: Counter[tuple[str, str]] = Counter()
    changed_rows = 0
    for row in rows:
        raw_prefix = [str(value) for value in (row.get("prefix_raw_actions") or [])]
        canonical_prefix = [canonical_action(value) for value in raw_prefix]
        if len(canonical_prefix) != len(raw_prefix):
            raise ValueError(f"prefix/raw length mismatch: {row.get('prefix_id')}")
        raw_target = str(row.get("target_next_raw_action") or "__END__")
        canonical_target = "__END__" if raw_target == "__END__" else canonical_action(raw_target)
        previous_prefix = list(row.get("prefix_activities") or [])
        previous_target = str(row.get("target_next_activity") or "__END__")
        for raw, new in zip(raw_prefix, canonical_prefix):
            if new != canonical_action(raw):
                raise AssertionError("canonical action must be deterministic")
            # Count only the legacy labels that this migration changes.
            if raw in {"image-grid-selector", "patch-zoomer"}:
                mapping_counts[(raw, new)] += 1
        if raw_target in {"image-grid-selector", "patch-zoomer"}:
            mapping_counts[(raw_target, canonical_target)] += 1
        if previous_prefix != canonical_prefix or previous_target != canonical_target:
            changed_rows += 1

        new_row = dict(row)
        new_row["prefix_activities"] = canonical_prefix
        new_row["target_next_activity"] = canonical_target
        new_row["canonicalization_version"] = MIGRATION_VERSION

        state = _copy_state_features(row.get("state_features"))
        state_prefix = state.get("prefix")
        if state_prefix:
            state_prefix["last_actions"] = canonical_prefix[-3:]
            state_prefix["action_histogram"] = dict(sorted(Counter(canonical_prefix).items()))
            state["prefix"] = state_prefix
            new_row["state_features"] = state
        repaired.append(new_row)
    summary = {
        "schema_version": MIGRATION_VERSION,
        "input_rows": len(rows),
        "output_rows": len(repaired),
        "changed_rows": changed_rows,
        "mapping_counts": {f"{raw}->{new}": count for (raw, new), count in sorted(mapping_counts.items())},
        "source_hashes_preserved": all(bool(row.get("source_trace_sha256")) for row in repaired),
        "future_event_markers_preserved": all(row.get("future_events_included_in_input") is False for row in repaired),
        "ground_truth_markers_preserved": all(row.get("ground_truth_included_in_input") is False for row in repaired),
    }
    return repaired, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()
    rows, summary = repair_rows(_read_jsonl(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
