#!/usr/bin/env python3
"""Select a bounded resource/recovery reservoir without changing raw traces.

The existing batch pilot is a measured 112-row resource reservoir.  A later
oversample may contain more valid rows than the remaining formal slots; this
utility chooses the missing number deterministically and records every
discarded candidate, rather than silently truncating a JSONL file.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select(existing_path: Path, completion_path: Path, output: Path, rejected_output: Path, expected: int = 128) -> dict[str, Any]:
    existing = _read(existing_path)
    existing_ids = {str(row.get("run_id")) for row in existing}
    invalid = [
        {"run_id": row.get("run_id"), "reason": "not_success_valid", "row": row}
        for row in _read(completion_path)
        if row.get("status") != "success" or row.get("validator") != "valid"
    ]
    completion = [
        row for row in _read(completion_path)
        if row.get("status") == "success" and row.get("validator") == "valid"
    ]
    need = max(0, expected - len(existing))
    completion.sort(key=lambda row: (
        str(row.get("baseline")),
        str(row.get("yolo_batch")),
        str(row.get("video_id")),
        str(row.get("base_task_id")),
        str(row.get("run_id")),
    ))
    unique_completion: list[dict[str, Any]] = []
    duplicate_rejected: list[dict[str, Any]] = []
    seen_ids = set(existing_ids)
    for row in completion:
        run_id = str(row.get("run_id"))
        if run_id in seen_ids:
            duplicate_rejected.append({"run_id": row.get("run_id"), "reason": "duplicate_run_id", "row": row})
            continue
        seen_ids.add(run_id)
        unique_completion.append(row)

    # Round-robin baseline/batch buckets so a formal completion cannot be
    # filled by one policy simply because its name sorts first.
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in unique_completion:
        buckets[(str(row.get("baseline")), str(row.get("yolo_batch")))].append(row)
    selected_completion: list[dict[str, Any]] = []
    bucket_keys = sorted(buckets)
    while len(selected_completion) < need and bucket_keys:
        progressed = False
        for key in bucket_keys:
            if buckets[key] and len(selected_completion) < need:
                selected_completion.append(buckets[key].pop(0))
                progressed = True
        if not progressed:
            break
    selected_ids = {str(row.get("run_id")) for row in selected_completion}
    rejected = invalid + duplicate_rejected + [
        {"run_id": row.get("run_id"), "reason": "oversample_after_formal_slots", "row": row}
        for row in unique_completion
        if str(row.get("run_id")) not in selected_ids
    ]
    combined = existing + selected_completion
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in combined) + "\n", encoding="utf-8")
    rejected_output.parent.mkdir(parents=True, exist_ok=True)
    rejected_output.write_text(
        "\n".join(json.dumps({"run_id": row.get("run_id"), "reason": "oversample_after_formal_slots", "row": row}, ensure_ascii=False, sort_keys=True) for row in rejected)
        + ("\n" if rejected else ""),
        encoding="utf-8",
    )
    return {
        "existing": len(existing),
        "completion_valid": len(completion),
        "completion_invalid": len(invalid),
        "completion_duplicate_run_ids": len(duplicate_rejected),
        "selected_completion": len(selected_completion),
        "rejected_completion": len(rejected),
        "selected_total": len(combined),
        "expected": expected,
        "formal_gate": len(combined) == expected,
        "baseline_counts_completion": dict(sorted(Counter(row.get("baseline") for row in selected_completion).items())),
        "output": str(output),
        "rejected_output": str(rejected_output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", required=True, type=Path)
    parser.add_argument("--completion", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rejected-output", required=True, type=Path)
    parser.add_argument("--expected", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(select(args.existing, args.completion, args.output, args.rejected_output, args.expected), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
