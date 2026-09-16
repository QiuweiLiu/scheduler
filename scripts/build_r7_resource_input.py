#!/usr/bin/env python3
"""Build the R7 resource-predictor input from existing 300-video events.

This creates a versioned derived input directory.  Source snapshots and event
logs remain untouched; only runs covered by the authoritative P_dev split are
included.  Metadata is reduced to the static task/video fields consumed by the
resource predictor.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


STATIC_TASK_KEYS = {
    "domain",
    "question_type",
    "sub_category",
    "required_modalities",
    "answer_type",
    "temporal_scope",
    "question_chars",
    "question_tokens",
    "option_count",
    "option_chars_mean",
}
STATIC_VIDEO_KEYS = {"duration_s", "fps"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def split_map(path: Path) -> dict[str, str]:
    return {str(row["video_id"]): str(row["split"]) for row in read_jsonl(path)}


def safe_metadata(row: dict[str, Any]) -> dict[str, Any]:
    task_structure = row.get("task_structure") if isinstance(row.get("task_structure"), dict) else {}
    state = row.get("state_features") if isinstance(row.get("state_features"), dict) else {}
    task = state.get("task") if isinstance(state.get("task"), dict) else {}
    video = state.get("video") if isinstance(state.get("video"), dict) else {}
    features = row.get("features") if isinstance(row.get("features"), dict) else {}
    categorical = features.get("categorical") if isinstance(features.get("categorical"), dict) else {}
    numerical = features.get("numeric") if isinstance(features.get("numeric"), dict) else {}
    task_out = {key: task_structure[key] for key in STATIC_TASK_KEYS if key in task_structure}
    task_out.update({key: task[key] for key in STATIC_TASK_KEYS if key in task and key not in task_out})
    categorical_out = {key: categorical[key] for key in STATIC_TASK_KEYS if key in categorical}
    numerical_out = {key: numerical[key] for key in STATIC_TASK_KEYS if key in numerical}
    video_out = {key: video[key] for key in STATIC_VIDEO_KEYS if key in video}
    return {
        "schema_version": "r7-resource-static-metadata-v1",
        "run_id": str(row.get("run_id")),
        "video_id": str(row.get("video_id")),
        "task_structure": task_out,
        "state_features": {"task": task_out, "video": video_out},
        "features": {"categorical": categorical_out, "numeric": numerical_out},
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    splits = split_map(args.split)
    if len(splits) != 300 or set(splits.values()) != {"train", "validation", "test"}:
        raise ValueError(f"P_dev split must contain 300 rows and train/validation/test, got {len(splits)}")
    snapshot_rows = read_jsonl(args.core_snapshot) + read_jsonl(args.expansion_snapshot)
    selected_snapshot: dict[str, dict[str, Any]] = {}
    for row in snapshot_rows:
        if not row.get("selected_for_snapshot", True) or str(row.get("status")) != "success":
            continue
        run_id = str(row.get("run_id"))
        video_id = str(row.get("video_id"))
        if video_id not in splits:
            raise ValueError(f"snapshot video outside P_dev: {video_id}")
        if run_id in selected_snapshot:
            raise ValueError(f"duplicate selected run_id: {run_id}")
        selected = dict(row)
        selected["split"] = splits[video_id]
        selected_snapshot[run_id] = selected
    if len({row["video_id"] for row in selected_snapshot.values()}) != 300:
        raise ValueError("selected snapshot does not cover 300 P_dev videos")

    compute_rows = read_jsonl(args.core_compute) + read_jsonl(args.expansion_compute)
    filtered_compute = [row for row in compute_rows if str(row.get("run_id")) in selected_snapshot]
    unknown_runs = sorted({str(row.get("run_id")) for row in filtered_compute} - set(selected_snapshot))
    if unknown_runs:
        raise ValueError(f"unknown compute run IDs: {unknown_runs[:3]}")
    compute_videos = {str(row.get("video_id")) for row in filtered_compute}
    if not compute_videos <= set(splits):
        raise ValueError("compute contains videos outside P_dev")
    metadata_rows = read_jsonl(args.metadata)
    metadata_by_run: dict[str, dict[str, Any]] = {}
    metadata_by_video: dict[str, dict[str, Any]] = {}
    snapshot_video_ids = {item["video_id"] for item in selected_snapshot.values()}
    for row in metadata_rows:
        run_id = str(row.get("run_id"))
        video_id = str(row.get("video_id"))
        safe = safe_metadata(row)
        if run_id in selected_snapshot and run_id not in metadata_by_run:
            metadata_by_run[run_id] = safe
        if video_id in snapshot_video_ids and video_id not in metadata_by_video:
            metadata_by_video[video_id] = safe

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    snapshot_out = sorted(selected_snapshot.values(), key=lambda row: str(row["run_id"]))
    compute_out = sorted(filtered_compute, key=lambda row: (str(row.get("run_id")), int(row.get("event_index") or 0), str(row.get("derived_event_id"))))
    metadata_out = sorted(metadata_by_run.values(), key=lambda row: row["run_id"])
    write_jsonl(output / "snapshot_p_dev.jsonl", snapshot_out)
    write_jsonl(output / "compute_p_dev.jsonl", compute_out)
    write_jsonl(output / "static_metadata_p_dev.jsonl", metadata_out)
    summary = {
        "schema_version": "r7-resource-input-v1",
        "source": {
            "core_snapshot": str(args.core_snapshot),
            "expansion_snapshot": str(args.expansion_snapshot),
            "core_compute": str(args.core_compute),
            "expansion_compute": str(args.expansion_compute),
            "metadata": str(args.metadata),
            "split": str(args.split),
        },
        "p_dev": {
            "videos": len({row["video_id"] for row in snapshot_out}),
            "runs": len(snapshot_out),
            "split_counts": dict(sorted(Counter(row["split"] for row in snapshot_out).items())),
            "compute_rows": len(compute_out),
            "compute_runs": len({str(row.get("run_id")) for row in compute_out}),
            "compute_videos": len(compute_videos),
            "metadata_runs": len(metadata_out),
            "metadata_videos": len(metadata_by_video),
            "metadata_video_fallback_runs": len(selected_snapshot) - len(metadata_by_run),
            "status_counts": dict(sorted(Counter(str(row.get("status")) for row in compute_out).items())),
        },
        "contract": {
            "fit_split": "train",
            "validation_split": "validation",
            "diagnostic_split": "test",
            "measured_runtime_and_memory_are_labels": True,
            "metadata_excludes_answer_and_future_targets": True,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-snapshot", type=Path, required=True)
    parser.add_argument("--expansion-snapshot", type=Path, required=True)
    parser.add_argument("--core-compute", type=Path, required=True)
    parser.add_argument("--expansion-compute", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=root / "data/manifests/predictor_development_split_v1_normalized.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = build(args)
    print(json.dumps(summary["p_dev"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
