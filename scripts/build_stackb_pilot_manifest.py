#!/usr/bin/env python3
"""Build the bounded Stack B pilot manifest from public Video-MME rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = _read_jsonl(args.source)
    by_video: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        video_id = str(row.get("video_id") or Path(str(row["video_path"])).stem)
        by_video.setdefault(video_id, []).append(row)
    selected: list[dict[str, Any]] = []
    for video_id, video_rows in by_video.items():
        if len(video_rows) < 2:
            continue
        selected.extend(video_rows[:2])
        if len(selected) >= args.videos * 2:
            break
    selected = selected[: args.videos * 2]
    if len(selected) != args.videos * 2:
        raise ValueError(f"expected {args.videos * 2} rows with two questions per video, got {len(selected)}")
    output: list[dict[str, Any]] = []
    for row in selected:
        item = dict(row)
        # Keep evaluation labels in the separate annotation source; they must
        # not enter the runtime prompt or trace manifest.
        item.pop("answer", None)
        video_id = str(item.get("video_id") or Path(str(item["video_path"])).stem)
        item["video_id"] = video_id
        item["video_path"] = str(args.video_root / f"{video_id}.mp4")
        item["baselines"] = ["star", "langgraph_react"]
        item["repetitions"] = 1
        item["model_stack_id"] = args.model_stack_id
        item["planner_model"] = str(args.planner_model)
        item["planner_model_id"] = args.planner_model_id
        item["planner_python"] = args.python
        item["planner_max_new_tokens"] = args.planner_max_new_tokens
        item["visual_model_id"] = args.visual_model_id
        item["answer_model"] = str(args.answer_model)
        item["answer_model_id"] = args.answer_model_id
        item["answer_python"] = args.python
        item["answer_max_new_tokens"] = args.answer_max_new_tokens
        item["detector_model_id"] = args.detector_model_id
        item["yolo_batch"] = args.yolo_batch
        output.append(item)
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("configs/phase3_videomme_expansion32_source.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--videos", type=int, default=8)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--model-stack-id", default="stack_b_qwen3_4b_qwen25vl3b_yolo26n")
    parser.add_argument("--planner-model", type=Path, required=True)
    parser.add_argument("--planner-model-id", default="Qwen3-4B")
    parser.add_argument("--visual-model-id", default="Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--answer-model", type=Path, required=True)
    parser.add_argument("--answer-model-id", default="Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--python", required=True)
    parser.add_argument("--planner-max-new-tokens", type=int, default=96)
    parser.add_argument("--answer-max-new-tokens", type=int, default=96)
    parser.add_argument("--detector-model-id", default="YOLO26n")
    parser.add_argument("--yolo-batch", type=int, default=1)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    rows = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rows": len(rows), "videos": len(rows) // 2, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
