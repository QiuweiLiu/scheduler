#!/usr/bin/env python3
"""Build the frozen R7 S_train/S_val trace-collection manifests.

The registry is the only source of group membership.  Public Video-MME
question metadata is joined by video_id, and the resulting rows are expanded
over the two already-validated dynamic model stacks and two dynamic baselines.
No answer field is copied into the collection manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


STACKS: tuple[dict[str, Any], ...] = (
    {
        "suffix": "stack_a",
        "model_stack_id": "stack_a_qwen3_vl8b_yolo11x",
        "planner_mode": "local_qwen",
        "model_name": "Qwen3-VL-8B-Instruct",
        "qwen_model": "/root/autodl-tmp/Qwen3-VL-8B-Instruct",
        "qwen_max_new_tokens": 256,
        "yolo_model": "/root/autodl-tmp/upload/models/yolo11x.pt",
        "yolo_batch": 8,
        "yolo_preobserve": False,
        "planner_model_id": "Qwen3-VL-8B-Instruct",
        "visual_model_id": "Qwen3-VL-8B-Instruct",
        "answer_model_id": "Qwen3-VL-8B-Instruct",
        "detector_model_id": "yolo11x.pt",
        "yolo_python": "/root/miniconda3/envs/finetooling/bin/python",
    },
    {
        "suffix": "stack_b",
        "model_stack_id": "stack_b_qwen3_4b_qwen25vl3b_yolo26n",
        "planner_mode": "local_split",
        "model_name": "Qwen2.5-VL-3B-Instruct",
        "qwen_model": "/root/autodl-tmp/scheduler/Qwen2.5-VL-3B-Instruct",
        "planner_model": "/root/autodl-tmp/scheduler/Qwen3-4B",
        "answer_model": "/root/autodl-tmp/scheduler/Qwen2.5-VL-3B-Instruct",
        "qwen_max_new_tokens": 96,
        "planner_max_new_tokens": 96,
        "answer_max_new_tokens": 96,
        "planner_constrained_json": True,
        "yolo_model": "/root/autodl-tmp/upload/models/yolo26n.pt",
        "yolo_batch": 1,
        "yolo_preobserve": False,
        "planner_model_id": "Qwen3-4B",
        "visual_model_id": "Qwen2.5-VL-3B-Instruct",
        "answer_model_id": "Qwen2.5-VL-3B-Instruct",
        "detector_model_id": "YOLO26n",
        "yolo_python": "/root/miniconda3/envs/finetooling/bin/python",
    },
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        rows.append(value)
    return rows


def duration_rank(value: Any) -> int:
    return {"short": 0, "medium": 1, "long": 2}.get(str(value), 3)


def build_rows(
    registry: dict[str, Any],
    source_rows: list[dict[str, Any]],
    remote_video_root: str,
    groups: tuple[str, ...],
) -> list[dict[str, Any]]:
    source_by_id = {str(row["video_id"]): row for row in source_rows}
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for group in groups:
        video_ids = [str(value) for value in registry["groups"][group]["video_ids"]]
        for video_id in video_ids:
            if video_id in seen_ids:
                raise ValueError(f"video assigned twice across selected groups: {video_id}")
            seen_ids.add(video_id)
            source = source_by_id.get(video_id)
            if source is None:
                raise ValueError(f"missing public question metadata for {video_id}")
            for stack in STACKS:
                for baseline in ("star", "langgraph_react"):
                    task_id = f"r7_{group.lower()}_{video_id}_{stack['suffix']}"
                    row: dict[str, Any] = {
                        "schema_version": "r7-trace-task-v1",
                        "group": group,
                        "video_id": video_id,
                        "source_task_id": str(source.get("task_id") or f"videomme_{video_id}"),
                        "task_id": f"{task_id}_{baseline}",
                        "dataset": "videomme",
                        "video_path": f"{remote_video_root.rstrip('/')}/{video_id}.mp4",
                        "question": str(source["question"]),
                        "options": list(source.get("options") or []),
                        "domain": source.get("domain"),
                        "official_task_type": source.get("official_task_type"),
                        "sub_category": source.get("sub_category"),
                        "baselines": [baseline],
                        "repetitions": 1,
                    }
                    row.update({key: value for key, value in stack.items() if key != "suffix"})
                    rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=ROOT / "data/manifests/video_split_registry_r7_v1.json")
    parser.add_argument("--source", type=Path, default=ROOT / "data/manifests/videomme_600_source_v2.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/manifests/r7_trace_manifest_v1.jsonl")
    parser.add_argument("--pilot-output", type=Path, default=ROOT / "data/manifests/r7_trace_pilot_manifest_v1.jsonl")
    parser.add_argument("--remote-video-root", default="/root/autodl-tmp/scheduler/data/phase3/public/videos/videomme")
    parser.add_argument("--pilot-videos-per-group", type=int, default=1)
    args = parser.parse_args()
    if args.pilot_videos_per_group <= 0:
        raise ValueError("pilot-videos-per-group must be positive")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    source_rows = read_jsonl(args.source)
    full = build_rows(registry, source_rows, args.remote_video_root, ("S_train", "S_val"))
    source_by_id = {str(row["video_id"]): row for row in source_rows}
    pilot_ids: set[str] = set()
    for group in ("S_train", "S_val"):
        candidates = [source_by_id[str(value)] for value in registry["groups"][group]["video_ids"]]
        candidates.sort(key=lambda row: (duration_rank(row.get("duration")), str(row["video_id"])))
        pilot_ids.update(str(row["video_id"]) for row in candidates[: args.pilot_videos_per_group])
    pilot = [row for row in full if str(row["video_id"]) in pilot_ids]
    for path, rows in ((args.output, full), (args.pilot_output, pilot)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    print(json.dumps({"full_rows": len(full), "pilot_rows": len(pilot), "full_videos": 160, "pilot_videos": len(pilot_ids), "stacks": [stack["model_stack_id"] for stack in STACKS], "baselines": ["star", "langgraph_react"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
