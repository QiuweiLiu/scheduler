#!/usr/bin/env python3
"""Evaluate a small, auditable resource-prediction adapter.

This is the resource-side companion to the Future Predictor.  It uses only
past/task/resource metadata and never feeds measured runtime or VRAM back into
the input row being predicted.  The ``park_song_style`` label means a
predictor-only local adapter, not a claim of reproducing the original paper's
full scheduler or private model.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


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


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) and number >= 0.0 else default


def _split_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in _read_jsonl(path):
        video_id = _text(row.get("video_id"))
        split = _text(row.get("split"), "unassigned")
        if video_id and split != "unassigned":
            mapping[video_id] = split
    return mapping


def _load_rows(snapshot_path: Path, compute_path: Path, split_path: Path) -> list[dict[str, Any]]:
    snapshot = {str(row["run_id"]): row for row in _read_jsonl(snapshot_path) if row.get("selected_for_snapshot", True)}
    split_map = _split_map(split_path)
    result: list[dict[str, Any]] = []
    for row in _read_jsonl(compute_path):
        run_id = str(row.get("run_id"))
        source = snapshot.get(run_id)
        if source is None or _text(source.get("status")) != "success":
            continue
        video_id = _text(source.get("video_id"))
        split = split_map.get(video_id, _text(source.get("split"), "unassigned"))
        resource = {
            "runtime_ms": _number(row.get("runtime_ms")),
            "peak_vram_mb": max(_number(row.get("peak_reserved_mb")), _number(row.get("peak_allocated_mb"))),
        }
        result.append({
            "run_id": run_id,
            "video_id": video_id,
            "split": split,
            "activity": _text(row.get("activity")),
            "raw_action": _text(row.get("raw_action")),
            "model_id": _text(row.get("model_id")),
            "gpu_model": _text(row.get("gpu_model")),
            "input_scale": row.get("input_scale") or {},
            "runtime_ms": resource["runtime_ms"],
            "peak_vram_mb": resource["peak_vram_mb"],
        })
    if not result:
        raise ValueError("no selected compute rows found")
    return result


def _key(row: Mapping[str, Any], level: str) -> tuple[Any, ...]:
    scale = row.get("input_scale") or {}
    if level == "global":
        return tuple()
    if level == "activity":
        return (_text(row.get("activity")),)
    if level == "model_activity":
        return (_text(row.get("model_id")), _text(row.get("activity")))
    return (
        _text(row.get("model_id")),
        _text(row.get("activity")),
        _text(row.get("gpu_model")),
        int(scale.get("yolo_batch") or 0),
        min(int(scale.get("frame_count") or 0), 32),
    )


class MedianPredictor:
    def __init__(self, target: str, levels: Sequence[str]) -> None:
        self.target = target
        self.levels = list(levels)
        self.values: dict[str, dict[tuple[Any, ...], list[float]]] = defaultdict(lambda: defaultdict(list))

    def fit(self, rows: Sequence[Mapping[str, Any]]) -> None:
        for row in rows:
            value = _number(row.get(self.target))
            for level in self.levels:
                self.values[level][_key(row, level)].append(value)

    def predict(self, row: Mapping[str, Any]) -> float:
        for level in reversed(self.levels):
            values = self.values[level].get(_key(row, level))
            if values:
                return statistics.median(values)
        return 0.0


def _metrics(rows: Sequence[tuple[float, float]]) -> dict[str, float]:
    if not rows:
        return {"n": 0}
    errors = [prediction - truth for truth, prediction in rows]
    abs_errors = sorted(abs(error) for error in errors)
    p95 = abs_errors[min(len(abs_errors) - 1, math.ceil(0.95 * len(abs_errors)) - 1)]
    return {
        "n": len(rows),
        "mae": statistics.fmean(abs(error) for error in errors),
        "rmse": math.sqrt(statistics.fmean(error * error for error in errors)),
        "p95_absolute_error": p95,
    }


def build_report(snapshot_path: Path, compute_path: Path, split_path: Path) -> dict[str, Any]:
    rows = _load_rows(snapshot_path, compute_path, split_path)
    videos = defaultdict(set)
    for row in rows:
        videos[row["video_id"]].add(row["split"])
    leakage = {video: sorted(splits) for video, splits in videos.items() if len(splits) != 1}
    train = [row for row in rows if row["split"] == "train"]
    metrics: dict[str, Any] = {}
    for split in ("validation", "test"):
        held_out = [row for row in rows if row["split"] == split]
        split_metrics: dict[str, Any] = {}
        for name, levels in {
            "global_median": ("global",),
            "activity_median": ("global", "activity"),
            "model_activity_median": ("global", "activity", "model_activity"),
            "park_song_style": ("global", "activity", "model_activity", "structured_scale"),
        }.items():
            runtime = MedianPredictor("runtime_ms", levels)
            memory = MedianPredictor("peak_vram_mb", levels)
            runtime.fit(train)
            memory.fit(train)
            runtime_rows = [(row["runtime_ms"], runtime.predict(row)) for row in held_out]
            memory_rows = [(row["peak_vram_mb"], memory.predict(row)) for row in held_out]
            split_metrics[name] = {
                "implementation": "predictor_only" if name == "park_song_style" else "local_median_baseline",
                "runtime": _metrics(runtime_rows),
                "peak_vram": _metrics(memory_rows),
            }
        metrics[split] = split_metrics
    video_counts = Counter(next(iter(splits)) for splits in videos.values() if len(splits) == 1)
    fixed_gate = dict(video_counts) == {"train": 48, "validation": 8, "test": 8}
    return {
        "schema_version": "resource-predictor-v0.1",
        "snapshot": str(snapshot_path),
        "compute": str(compute_path),
        "rows": len(rows),
        "videos": len({row["video_id"] for row in rows}),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "leakage_conflicts": leakage,
        "video_split_counts": dict(sorted(video_counts.items())),
        "expected_video_split_counts": {"train": 48, "validation": 8, "test": 8},
        "formal_split_gate": not leakage and fixed_gate,
        "metrics": metrics,
        "input_contract": "candidate/task/model/GPU/input-scale only; measured runtime and VRAM are labels",
        "limitations": [
            "Resident model weight memory is unknown in current traces; peak VRAM is treated as a measured workspace-inclusive label.",
            "This is a predictor-only adapter and not a complete resource-optimization reproduction.",
        ],
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--compute", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = build_report(args.snapshot, args.compute, args.split_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": report["rows"], "formal_split_gate": report["formal_split_gate"], "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
