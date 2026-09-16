#!/usr/bin/env python3
"""Build a deterministic 600-video Video-MME source and download manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_rows(rows_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(rows_dir.glob("rows_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(item["row"] for item in payload.get("rows", []))
    if len(rows) != 2700:
        raise ValueError(f"expected 2700 Video-MME rows, got {len(rows)}")
    return rows


def read_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def choose_question(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return dict(sorted(rows, key=lambda row: str(row.get("question_id", "")))[0])


def balanced_new_ids(candidates: dict[str, dict[str, Any]], needed: int) -> list[str]:
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for video_id, row in candidates.items():
        key = (str(row.get("domain", "")), str(row.get("sub_category", "")))
        buckets[key].append(video_id)
    for values in buckets.values():
        values.sort()
    selected: list[str] = []
    keys = sorted(buckets)
    while len(selected) < needed:
        progressed = False
        for key in keys:
            values = buckets[key]
            if values:
                selected.append(values.pop(0))
                progressed = True
                if len(selected) == needed:
                    break
        if not progressed:
            raise ValueError(f"only {len(selected)} new videos available, need {needed}")
    return selected


def source_row(row: dict[str, Any], *, existing: bool, project_root: Path) -> dict[str, Any]:
    video_id = str(row["videoID"])
    return {
        "collection_id": "trace_collection_v2_expanded",
        "dataset": "videomme",
        "video_id": video_id,
        "video_path": str(project_root / "data/phase3/public/videos/videomme" / f"{video_id}.mp4"),
        "question_id": str(row["question_id"]),
        "question": row["question"],
        "options": list(row.get("options") or []),
        "answer": row.get("answer"),
        "domain": row.get("domain"),
        "sub_category": row.get("sub_category"),
        "official_task_type": row.get("task_type"),
        "duration": row.get("duration"),
        "source_url": row.get("url"),
        "source": "Video-MME official datasets-server metadata",
        "existing_before_r5": existing,
        "download_required": not existing,
        "baselines": ["st_fixed", "star", "langgraph_react"],
        "repetitions": 1,
    }


def manifest_hash(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_rows(args.rows_dir)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["videoID"])].append(row)
    representative = {video_id: choose_question(items) for video_id, items in grouped.items()}

    remote_ids = read_ids(args.remote_ids)
    local_ids = {path.stem for path in args.local_video_dir.glob("*.mp4")}
    existing_ids = set(representative) & (remote_ids | local_ids)
    if len(existing_ids) >= args.target:
        selected_ids = sorted(existing_ids)[: args.target]
    else:
        needed = args.target - len(existing_ids)
        candidates = {video_id: row for video_id, row in representative.items() if video_id not in existing_ids}
        selected_ids = sorted(existing_ids) + balanced_new_ids(candidates, needed)
    selected_ids = sorted(selected_ids)
    selected = [source_row(representative[video_id], existing=video_id in existing_ids, project_root=args.project_root) for video_id in selected_ids]
    download = [row for row in selected if row["download_required"]]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.download_output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in selected) + "\n", encoding="utf-8")
    args.download_output.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in download) + "\n", encoding="utf-8")
    report = {
        "schema_version": "videomme-600-source-v0.1",
        "target_video_count": args.target,
        "official_metadata_rows": len(rows),
        "official_unique_video_count": len(representative),
        "existing_remote_or_local_count": len(existing_ids),
        "selected_video_count": len(selected),
        "download_required_count": len(download),
        "source_manifest_sha256": manifest_hash(selected),
        "download_manifest_sha256": manifest_hash(download),
        "existing_id_sources": {"remote_file_ids": len(remote_ids), "local_mp4_ids": len(local_ids)},
        "selection": "all existing official videos plus deterministic domain/sub_category round-robin for new videos",
        "domain_counts": dict(sorted(_counts(selected, "domain").items())),
        "sub_category_counts": dict(sorted(_counts(selected, "sub_category").items())),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get(field, ""))] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--rows-dir", type=Path, default=root / "data/external/videomme/rows")
    parser.add_argument("--remote-ids", type=Path, default=root / "data/external/videomme/remote_video_file_ids_all_20260813.txt")
    parser.add_argument("--local-video-dir", type=Path, default=root / "data/phase3/public/videos/videomme")
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--target", type=int, default=600)
    parser.add_argument("--output", type=Path, default=root / "data/manifests/videomme_600_source_v2.jsonl")
    parser.add_argument("--download-output", type=Path, default=root / "data/manifests/videomme_600_download_v2.jsonl")
    parser.add_argument("--report", type=Path, default=root / "data/manifests/videomme_600_selection_report_v2.json")
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
