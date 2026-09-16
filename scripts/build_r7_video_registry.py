#!/usr/bin/env python3
"""Build the deterministic R7 six-group video registry.

The predictor split is authoritative.  Short video IDs are normalized only
when there is exactly one source-manifest prefix match, except for the two
ambiguous IDs whose remote run_id evidence is recorded below.  No video ID is
silently inferred from ordering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


MANUAL_REPAIRS = {
    "9": {
        "canonical": "9_M4bNOxsYs",
        "evidence": "remote role_event_samples.jsonl run_id=9_M4bNOxsYs_langgraph_react_1785889999709",
    },
    "B": {
        "canonical": "B_OL2TSrpKM",
        "evidence": "remote role_event_samples.jsonl run_id=B_OL2TSrpKM_langgraph_react_1785892852692",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["video_id"])].append(row)
    return {video_id: sorted(items, key=lambda row: str(row.get("question_id", "")))[0] for video_id, items in grouped.items()}


def normalize_ids(raw_ids: set[str], source_ids: set[str]) -> tuple[dict[str, str], list[dict[str, str]]]:
    mapping: dict[str, str] = {}
    repairs: list[dict[str, str]] = []
    for raw in sorted(raw_ids):
        if raw in source_ids:
            mapping[raw] = raw
            continue
        if raw in MANUAL_REPAIRS:
            canonical = MANUAL_REPAIRS[raw]["canonical"]
            if canonical not in source_ids:
                raise ValueError(f"manual repair {raw}->{canonical} is absent from source manifest")
            repairs.append({"raw": raw, "canonical": canonical, "method": "run_id_evidence", "evidence": MANUAL_REPAIRS[raw]["evidence"]})
            mapping[raw] = canonical
            continue
        candidates = sorted(video_id for video_id in source_ids if video_id.startswith(raw + "_"))
        if len(candidates) != 1:
            raise ValueError(f"cannot uniquely normalize {raw!r}: candidates={candidates}")
        mapping[raw] = candidates[0]
        repairs.append({"raw": raw, "canonical": candidates[0], "method": "unique_source_prefix"})
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("normalized predictor IDs are not unique")
    return mapping, repairs


def balanced_order(rows: dict[str, dict[str, Any]], candidate_ids: set[str]) -> list[str]:
    buckets: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for video_id in candidate_ids:
        row = rows[video_id]
        key = (str(row.get("domain", "")), str(row.get("duration", "")), str(row.get("official_task_type", "")))
        buckets[key].append(video_id)
    for values in buckets.values():
        values.sort()
    ordered: list[str] = []
    for key in sorted(buckets):
        buckets[key].sort()
    keys = sorted(buckets)
    while True:
        progressed = False
        for key in keys:
            if buckets[key]:
                ordered.append(buckets[key].pop(0))
                progressed = True
        if not progressed:
            break
    return ordered


def build(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    rows = source_rows(args.source)
    source_ids = set(rows)
    dev_raw = json.loads(args.dev_split.read_text(encoding="utf-8"))
    raw_dev_split = {str(video_id): str(split) for video_id, split in dev_raw["split"].items()}
    holdout_rows = read_jsonl(args.holdout_split)
    raw_holdout = {str(row["video_id"]): str(row["split"]) for row in holdout_rows}
    dev_map, repairs = normalize_ids(set(raw_dev_split), source_ids)
    holdout_map, holdout_repairs = normalize_ids(set(raw_holdout), source_ids)
    if holdout_repairs:
        raise ValueError(f"holdout split unexpectedly needs repairs: {holdout_repairs}")
    dev_ids = set(dev_map.values())
    holdout_ids = set(holdout_map.values())
    if len(dev_ids) != 300 or len(holdout_ids) != 40:
        raise ValueError(f"unexpected predictor counts dev={len(dev_ids)} holdout={len(holdout_ids)}")
    if dev_ids & holdout_ids:
        raise ValueError("development and holdout split overlap")
    unseen = source_ids - dev_ids - holdout_ids
    if len(unseen) != 260:
        raise ValueError(f"expected 260 predictor-unseen videos, got {len(unseen)}")
    ordered = balanced_order(rows, unseen)
    group_sizes = [("S_train", 120), ("S_val", 40), ("T_final", 80), ("T_backup", 20)]
    groups: dict[str, list[str]] = {}
    offset = 0
    for name, size in group_sizes:
        groups[name] = ordered[offset : offset + size]
        offset += size
    if offset != len(ordered):
        raise ValueError("scheduler group sizes do not cover unseen pool")
    assignments: dict[str, str] = {video_id: "P_dev" for video_id in dev_ids}
    assignments.update({video_id: "P_holdout_diag" for video_id in holdout_ids})
    for group, video_ids in groups.items():
        assignments.update({video_id: group for video_id in video_ids})
    if len(assignments) != 600:
        raise ValueError(f"expected 600 assignments, got {len(assignments)}")
    normalized_dev = {dev_map[raw]: split for raw, split in raw_dev_split.items()}
    normalized_rows = [
        {"schema_version": "video-split-r7-predictor-v1", "video_id": video_id, "split": split}
        for video_id, split in sorted(normalized_dev.items())
    ]
    registry = {
        "schema_version": "video-split-registry-r7-v1",
        "status": "id_resolved_boundary_gate_pending_trace_collection",
        "collection_id": "trace_collection_v2_expanded",
        "source_manifest": {"path": project_path(args.source, root), "sha256": sha256(args.source), "video_count": len(source_ids)},
        "predictor_split_sources": {
            "development": {"path": project_path(args.dev_split, root), "sha256": sha256(args.dev_split), "raw_video_count": len(raw_dev_split)},
            "holdout": {"path": project_path(args.holdout_split, root), "sha256": sha256(args.holdout_split), "raw_video_count": len(raw_holdout)},
        },
        "groups": {
            "P_dev": {"count": 300, "video_ids": sorted(dev_ids), "subsplit": {split: sorted(video_id for video_id, value in normalized_dev.items() if value == split) for split in ("train", "validation", "test")}},
            "P_holdout_diag": {"count": 40, "video_ids": sorted(holdout_ids)},
            "S_train": {"count": 120, "video_ids": groups["S_train"]},
            "S_val": {"count": 40, "video_ids": groups["S_val"]},
            "T_final": {"count": 80, "video_ids": groups["T_final"], "sealed_until_models_frozen": True},
            "T_backup": {"count": 20, "video_ids": groups["T_backup"], "replacement_only": True},
        },
        "normalization": {
            "raw_development_to_canonical": dev_map,
            "holdout_identity_map": holdout_map,
            "repairs": sorted(repairs + holdout_repairs, key=lambda item: item["raw"]),
            "repair_count": len(repairs) + len(holdout_repairs),
        },
        "invariants": {
            "total_count": len(assignments),
            "predictor_count": len(dev_ids | holdout_ids),
            "scheduler_count": len(unseen),
            "all_source_ids_assigned_once": len(assignments) == len(source_ids) and set(assignments) == source_ids,
            "predictor_scheduler_overlap": len((dev_ids | holdout_ids) & unseen),
            "development_holdout_overlap": len(dev_ids & holdout_ids),
        },
        "policy": {
            "behavior_and_resource_share_p_dev": True,
            "scheduler_never_fits_on_predictor_pool": True,
            "r6_legacy_compatibility_only": True,
            "final_test_sealed_until_freeze": True,
        },
    }
    args.output.write_text(json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.normalized_dev_output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in normalized_rows) + "\n",
        encoding="utf-8",
    )
    return {"output": str(args.output), "counts": {name: len(ids) for name, ids in [("P_dev", dev_ids), ("P_holdout_diag", holdout_ids), *groups.items()]}, "repair_count": len(repairs), "overlap": registry["invariants"]["predictor_scheduler_overlap"]}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "data/manifests/videomme_600_source_v2.jsonl")
    parser.add_argument("--dev-split", type=Path, default=root / "data/manifests/predictor_development_split_v1.json")
    parser.add_argument("--holdout-split", type=Path, default=root / "data/manifests/predictor_holdout_split_v1.jsonl")
    parser.add_argument("--output", type=Path, default=root / "data/manifests/video_split_registry_r7_v1.json")
    parser.add_argument("--normalized-dev-output", type=Path, default=root / "data/manifests/predictor_development_split_v1_normalized.jsonl")
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
