#!/usr/bin/env python3
"""Build the 32-video Phase 3 source, runtime, and evaluation manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tracing.collectors.structured_state import structured_task_record, write_jsonl


BASELINES = ["st_fixed", "star", "langgraph_react"]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build(project_root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    legacy = _read_jsonl(project_root / "configs/phase2_public_videomme_prelim.jsonl")
    expansion = _read_jsonl(project_root / "configs/phase3_videomme_expansion24_source.jsonl")
    source: list[dict] = []
    for record in legacy + expansion:
        item = dict(record)
        video_id = str(item.get("video_id") or Path(str(item["video_path"])).stem)
        if video_id in {str(row.get("video_id")) for row in source}:
            raise ValueError(f"duplicate Video-MME video: {video_id}")
        if video_id in {Path(str(row["video_path"])).stem for row in legacy}:
            item["video_path"] = str(
                project_root / "data/phase2/public/videos/videomme" / f"{video_id}.mp4"
            )
        item["video_id"] = video_id
        item["baselines"] = list(BASELINES)
        item["repetitions"] = 3
        item["source"] = (
            "Video-MME official metadata; one question per independent public video; "
            "Phase 3 structured-state cohort"
        )
        source.append(item)

    if len(source) != 32 or len({str(item["video_id"]) for item in source}) != 32:
        raise ValueError("expected exactly 32 unique source videos")
    runtime = [structured_task_record(item) for item in source]
    annotations = [
        {
            "task_id": item["task_id"],
            "video_id": item["video_id"],
            "question_id": item.get("question_id"),
            "answer": item.get("answer"),
            "options": item.get("options"),
            "domain": item.get("domain"),
            "sub_category": item.get("sub_category"),
            "official_task_type": item.get("official_task_type"),
        }
        for item in source
    ]
    return source, runtime, annotations


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    root = args.project_root.resolve()
    source, runtime, annotations = build(root)
    write_jsonl(root / "configs/phase3_videomme_32_source.jsonl", source)
    write_jsonl(root / "configs/phase3_structured_videomme_32_local.jsonl", runtime)
    write_jsonl(root / "configs/phase3_videomme_32_eval_annotations.jsonl", annotations)
    print(json.dumps({"source": len(source), "runtime": len(runtime), "annotations": len(annotations)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
