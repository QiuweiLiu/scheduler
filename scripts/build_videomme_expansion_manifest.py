#!/usr/bin/env python3
"""Build a balanced, non-overlapping Video-MME expansion manifest.

The Hugging Face datasets-server ``rows`` endpoint is intentionally fetched
outside this script.  It reads the cached JSON pages, selects 32 new videos
from a deterministic domain-balanced cohort, and writes one download manifest
plus two official questions per selected video.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DOMAIN_TARGETS = {
    "Knowledge": 7,
    "Sports Competition": 7,
    "Film & Television": 6,
    "Artistic Performance": 6,
    "Life Record": 6,
}


def _rows(rows_dir: Path, min_offset: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in sorted(rows_dir.glob("videomme_rows_*.json"), key=lambda p: int(p.stem.rsplit("_", 1)[-1])):
        offset = int(path.stem.rsplit("_", 1)[-1])
        if offset < min_offset:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        output.extend(item["row"] for item in payload.get("rows", []))
    if not output:
        raise RuntimeError(f"no Video-MME rows found in {rows_dir} at offset >= {min_offset}")
    return output


def _select(rows: list[dict[str, Any]], existing: set[str]) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        video_id = str(row["videoID"])
        if video_id not in existing:
            grouped[video_id].append(row)
    for video_rows in grouped.values():
        video_rows.sort(key=lambda row: str(row["question_id"]))

    selected: list[tuple[str, list[dict[str, Any]]]] = []
    used_types: Counter[str] = Counter()
    used_subcategories: Counter[str] = Counter()
    for domain, target in DOMAIN_TARGETS.items():
        candidates = [
            (video_id, video_rows)
            for video_id, video_rows in grouped.items()
            if str(video_rows[0].get("domain")) == domain
        ]
        candidates.sort(
            key=lambda item: (
                used_types[str(item[1][0].get("task_type"))],
                used_subcategories[str(item[1][0].get("sub_category"))],
                item[0],
            )
        )
        if len(candidates) < target:
            raise RuntimeError(f"domain {domain!r} has only {len(candidates)} candidates; need {target}")
        for video_id, video_rows in candidates[:target]:
            selected.append((video_id, video_rows))
            used_types[str(video_rows[0].get("task_type"))] += 1
            used_subcategories[str(video_rows[0].get("sub_category"))] += 1
    return selected


def _two_questions(video_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    first_type: str | None = None
    for row in video_rows:
        task_type = str(row.get("task_type") or "unknown")
        if not chosen:
            chosen.append(row)
            first_type = task_type
        elif task_type != first_type:
            chosen.append(row)
            break
    if len(chosen) < 2:
        chosen = video_rows[:2]
    if len(chosen) < 2:
        raise RuntimeError(f"video {video_rows[0].get('videoID')} has fewer than two official questions")
    return chosen[:2]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def build(project_root: Path, rows_dir: Path, min_offset: int) -> dict[str, Any]:
    existing = {path.stem for path in project_root.glob("data/**/videos/videomme/*.mp4")}
    selected = _select(_rows(rows_dir, min_offset), existing)
    video_root = project_root / "data/phase3/public/videos/videomme"
    download_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for index, (video_id, video_rows) in enumerate(selected, start=1):
        first = video_rows[0]
        download_rows.append(
            {
                "dataset": "videomme",
                "video_id": video_id,
                "video_path": str(video_root / f"{video_id}.mp4"),
                "domain": first.get("domain"),
                "sub_category": first.get("sub_category"),
                "duration": first.get("duration"),
                "source_url": first.get("url"),
                "source": "Video-MME official rows API; balanced 32-video expansion",
            }
        )
        for question_index, row in enumerate(_two_questions(video_rows), start=1):
            source_rows.append(
                {
                    "task_id": f"videomme_expansion32_{index:03d}_{video_id}_q{question_index}",
                    "dataset": "videomme",
                    "video_path": str(video_root / f"{video_id}.mp4"),
                    "video_id": video_id,
                    "question_id": row["question_id"],
                    "question": row["question"],
                    "options": row["options"],
                    "answer": row["answer"],
                    "domain": row.get("domain"),
                    "sub_category": row.get("sub_category"),
                    "official_task_type": row.get("task_type"),
                    "duration": row.get("duration"),
                    "source_url": row.get("url"),
                    "source": "Video-MME official rows API; balanced 32-video expansion; two task types per video",
                    "baselines": ["st_fixed", "star", "langgraph_react"],
                    "repetitions": 1,
                }
            )

    download_path = project_root / "data/phase3/public/videomme_expansion32_download_manifest.jsonl"
    source_path = project_root / "configs/phase3_videomme_expansion32_source.jsonl"
    _write_jsonl(download_path, download_rows)
    _write_jsonl(source_path, source_rows)
    summary = {
        "video_count": len(download_rows),
        "task_count": len(source_rows),
        "domain_counts": dict(Counter(str(row["domain"]) for row in download_rows)),
        "video_ids": [row["video_id"] for row in download_rows],
        "download_manifest": str(download_path),
        "source_manifest": str(source_path),
    }
    (project_root / "data/phase3/public/videomme_expansion32_selection.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--rows-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--min-offset", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve(), args.rows_dir.resolve(), args.min_offset), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
