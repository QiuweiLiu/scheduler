"""Phase 21 — task-context restoration for S_* predictor anchors (pure logic).

The S_* traces carry no benchmark task metadata, so the frozen predictor's
``task_context`` block collapsed to ``"unknown"`` for every anchor even though it
was trained with it populated. These helpers join the benchmark registry by
``video_id`` and restore only the three categorical fields the frozen encoder
consumes (``domain``, ``official_task_type``, ``sub_category``); everything else
is copied verbatim so a masked/fixed comparison isolates exactly those inputs.

Kept dependency-free so the join logic can be unit-tested without numpy/torch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Set, Tuple

TASK_CONTEXT_FIELDS = ("domain", "official_task_type", "sub_category")
REGISTRY_UNAVAILABLE_FIELDS = ("answer_type", "question_type", "required_modalities", "temporal_scope")
# fields the registry deliberately must NOT be copied into the model input
FORBIDDEN_REGISTRY_FIELDS = ("question", "answer", "options", "duration", "question_id")
UNKNOWN = "unknown"


def is_missing(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def load_task_registry(path: Path | None) -> Tuple[Dict[str, Dict[str, Any]], str | None]:
    """video_id -> registry row, plus the registry SHA256. Duplicates are fatal."""

    if path is None:
        return {}, None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_video: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        video_id = str(row.get("video_id") or "")
        if not video_id:
            continue
        if video_id in by_video:
            raise ValueError(f"duplicate video_id in task registry: {video_id}")
        by_video[video_id] = row
    return by_video, hashlib.sha256(path.read_bytes()).hexdigest()


def read_video_allowlist(path: Path | None) -> Set[str] | None:
    if path is None:
        return None
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def deterministic_pilot_videos(video_ids: Any, count: int) -> list[str]:
    """A pre-registered, order-independent pilot subset (sha256 of the video id)."""

    return sorted({str(value) for value in video_ids}, key=lambda value: hashlib.sha256(value.encode()).hexdigest())[
        : int(count)
    ]


def enrich_sample_context(
    run_manifest: Mapping[str, Any],
    registry_row: Mapping[str, Any] | None,
    mask: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Restore registry-backed task context on a copy of the run manifest."""

    enriched = dict(run_manifest)
    audit: Dict[str, Any] = {"join_key": "video_id", "recovered_fields": [], "masked": bool(mask)}
    if registry_row is None:
        audit["registry_joined"] = False
        return enriched, audit
    audit["registry_joined"] = True
    audit["question_id"] = str(registry_row.get("question_id") or "")
    for field in TASK_CONTEXT_FIELDS:
        new_value = str(registry_row.get(field) or "")
        old_value = str(enriched.get(field) or "")
        if mask:
            enriched[field] = UNKNOWN
            audit.setdefault("masked_fields", []).append(field)
            continue
        if old_value and old_value != UNKNOWN:
            if new_value and old_value != new_value:
                raise ValueError(
                    f"task registry disagrees with the trace for {field}: {old_value!r} vs {new_value!r}"
                )
            continue
        if new_value:
            enriched[field] = new_value
            audit["recovered_fields"].append(field)
    # nothing outside the three registry fields may enter the sample row; the audit
    # may carry question_id as provenance (documented) but never the task payload
    for forbidden in FORBIDDEN_REGISTRY_FIELDS:
        if forbidden in enriched and forbidden not in run_manifest:
            raise AssertionError(f"forbidden registry field leaked into the sample row: {forbidden}")
    for forbidden in ("question", "answer", "options", "duration"):
        if forbidden in audit:
            raise AssertionError(f"forbidden registry field leaked into the audit: {forbidden}")
    return enriched, audit
