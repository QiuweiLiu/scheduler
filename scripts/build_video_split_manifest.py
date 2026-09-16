#!/usr/bin/env python3
"""Create a deterministic video-level 48/8/8 split sidecar.

The source manifests may contain several tasks per video.  This utility
deduplicates video IDs, sorts them deterministically, and writes one split row
per video.  It never edits the source manifests or trace data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _video_id(row: dict[str, Any]) -> str:
    value = row.get("video_id") or row.get("source_video_id")
    if value:
        return str(value).strip()
    path = str(row.get("video_path") or "").strip()
    return Path(path).stem


def build(paths: Iterable[Path], output: Path, train_count: int = 48, validation_count: int = 8, test_count: int = 8) -> dict[str, Any]:
    sources: dict[str, set[str]] = {}
    for path in paths:
        for row in _read_jsonl(path):
            video_id = _video_id(row)
            if video_id:
                sources.setdefault(video_id, set()).add(str(path))
    video_ids = sorted(sources)
    expected = train_count + validation_count + test_count
    if len(video_ids) != expected:
        raise ValueError(f"expected exactly {expected} unique videos, found {len(video_ids)}")
    assignments: list[dict[str, Any]] = []
    for ordinal, video_id in enumerate(video_ids):
        if ordinal < train_count:
            split = "train"
        elif ordinal < train_count + validation_count:
            split = "validation"
        else:
            split = "test"
        assignments.append({
            "schema_version": "video-split-v0.1",
            "video_id": video_id,
            "split": split,
            "ordinal": ordinal,
            "source_manifest_paths": sorted(sources[video_id]),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in assignments:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema_version": "video-split-v0.1",
        "video_count": len(video_ids),
        "counts": {"train": train_count, "validation": validation_count, "test": test_count},
        "source_paths": [str(path) for path in paths],
        "video_id_order_sha256": hashlib.sha256("\n".join(video_ids).encode("utf-8")).hexdigest(),
        "output": str(output),
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
