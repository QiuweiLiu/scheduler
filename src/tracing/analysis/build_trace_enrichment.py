#!/usr/bin/env python3
"""Build auditable derived views from Dynamic Video Agent traces.

The raw ``trace.jsonl`` files are immutable.  This compiler creates four
versioned JSONL artifacts:

* ``candidate_snapshot_v0_1.jsonl``: one auditable row per run candidate;
* ``semantic_events_v0_1.jsonl``: logical agent/tool events for future-path
  prediction;
* ``compute_events_v0_1.jsonl``: all measured execution events for resource
  prediction and workload compilation;
* ``trace_enrichment_v0_1.jsonl`` and ``prefix_samples_v0_1.jsonl``.

It deliberately keeps unknown provenance as ``unknown`` and never uses a
future event as a prefix feature.  The script is dependency-free apart from
the repository's existing trace validator and canonical action helper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tracing.collectors.structured_state import canonical_action
from tracing.validators.validate_trace import validate_trace


DERIVATION_VERSION = "trace-enrichment-v0.1"
STATE_FEATURE_VERSION = "state-features-v0.2"
SEMANTIC_VERSION = "semantic-events-v0.1"
COMPUTE_VERSION = "compute-events-v0.1"
SNAPSHOT_VERSION = "candidate-snapshot-v0.1"
PREFIX_VERSION = "prefix-samples-v0.1"
TASK_RUN_RE = re.compile(r"_r\d+$")
TERMINAL_ACTIONS = {
    "answer",
    "finish",
    "baseline_finish",
    "run_success",
}
START_ACTIONS = {
    "baseline_start",
    "run",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(payload)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _number(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed == parsed and parsed >= 0.0 else default


def _base_task_id(task_id: str) -> str:
    return TASK_RUN_RE.sub("", task_id)


def _utc_sort_key(value: Any) -> str:
    return _text(value, "9999-12-31T23:59:59Z")


def _load_split(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    mapping: dict[str, str] = {}
    for row in _read_jsonl(path):
        video_id = _text(row.get("video_id") or row.get("source_video_id"))
        split = _text(row.get("split"))
        if video_id and split:
            mapping[video_id] = split
    return mapping


def _manifest_value(manifest: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in manifest and manifest[name] not in (None, ""):
            return manifest[name]
    stack = manifest.get("model_stack")
    if isinstance(stack, Mapping):
        for name in names:
            if name in stack and stack[name] not in (None, ""):
                return stack[name]
    return None


def _event_value(events: Sequence[Mapping[str, Any]], key: str, *nested: str) -> Any:
    for event in events:
        value = event.get(key)
        if value not in (None, ""):
            return value
        input_data = event.get("input")
        if isinstance(input_data, Mapping):
            for name in nested:
                value = input_data.get(name)
                if value not in (None, ""):
                    return value
    return None


def _video_id(manifest: Mapping[str, Any], task_id: str) -> str:
    video_path = _text(manifest.get("video_path"))
    return Path(video_path).stem if video_path else _base_task_id(task_id)


def _run_status(run_dir: Path, manifest: Mapping[str, Any]) -> str:
    status_path = run_dir / "run_status.json"
    if status_path.is_file():
        try:
            status = _text(_read_json(status_path).get("status"))
            if status:
                return status
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return _text(manifest.get("status"), "unknown")


def _trace_events(trace_path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(trace_path)


def _state_snapshot(run_dir: Path, position: int) -> dict[str, Any] | None:
    """Load the immutable state_t snapshot for a prefix, when available."""
    path = run_dir / "states" / f"state_{position:04d}.json"
    if not path.is_file():
        return None
    try:
        payload = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _state_features(state: Mapping[str, Any] | None, position: int) -> dict[str, Any]:
    """Project state_t to scheduler features without answer/future fields."""
    if not state:
        return {
            "schema_version": STATE_FEATURE_VERSION,
            "state_present": False,
            "state_step_id": position,
            "future_events_excluded": True,
            "ground_truth_excluded": True,
        }
    prefix = state.get("prefix") if isinstance(state.get("prefix"), Mapping) else {}
    evidence = state.get("evidence") if isinstance(state.get("evidence"), Mapping) else {}
    video = state.get("video") if isinstance(state.get("video"), Mapping) else {}
    task = state.get("task") if isinstance(state.get("task"), Mapping) else {}
    leakage = state.get("leakage_guard") if isinstance(state.get("leakage_guard"), Mapping) else {}
    return {
        "schema_version": STATE_FEATURE_VERSION,
        "state_present": True,
        "state_step_id": int(state.get("step_id") if state.get("step_id") is not None else position),
        "task": dict(task),
        "video": {
            "duration_s": video.get("duration_s"),
            "fps": video.get("fps"),
            "width": video.get("width"),
            "height": video.get("height"),
            "shot_count": video.get("shot_count"),
            "subtitle_available": video.get("subtitle_available"),
        },
        "prefix": {
            "observed_step_count": prefix.get("observed_step_count", position),
            "progress_ratio": prefix.get("progress_ratio"),
            "max_steps": prefix.get("max_steps"),
            "last_actions": list(prefix.get("last_actions") or [])[-3:],
            "action_histogram": dict(prefix.get("action_histogram") or {}),
            "retry_count": prefix.get("retry_count", 0),
            "error_count": prefix.get("error_count", 0),
            "local_runtime_ms": prefix.get("local_runtime_ms", 0.0),
            "api_wait_ms": prefix.get("api_wait_ms", 0.0),
        },
        "evidence": {
            "coverage_ratio": evidence.get("coverage_ratio"),
            "frames_seen": evidence.get("frames_seen", 0),
            "modality_counts": dict(evidence.get("modality_counts") or {}),
            "object_counts": dict(evidence.get("object_counts") or {}),
            "ocr_chars": evidence.get("ocr_chars", 0),
            "temporal_relation_count": evidence.get("temporal_relation_count", 0),
            "confidence_mean": evidence.get("confidence_mean"),
            "observed_interval_count": len(evidence.get("observed_intervals") or []),
        },
        "leakage_guard": {
            "future_events_excluded": leakage.get("future_events_excluded") is True,
            "ground_truth_excluded": leakage.get("ground_truth_excluded") is True,
            "answer_text_used_as_feature": leakage.get("answer_text_used_as_feature") is True,
            "video_id_used_as_feature": leakage.get("video_id_used_as_feature") is True,
        },
        "future_events_excluded": True,
        "ground_truth_excluded": True,
    }


def _merge_task_structure(base: Mapping[str, Any], state_features: Mapping[str, Any]) -> dict[str, Any]:
    """Fill missing manifest task fields from the pre-action state snapshot."""
    merged = dict(base)
    task = state_features.get("task") if isinstance(state_features.get("task"), Mapping) else {}
    for key, value in task.items():
        if key not in merged or merged[key] in (None, "", "unknown", "unspecified", 0, []):
            merged[key] = value
    return merged


def _event_runtime(event: Mapping[str, Any]) -> float:
    resource = event.get("resource")
    return _number(resource.get("runtime_ms")) if isinstance(resource, Mapping) else 0.0


def _event_activity(event: Mapping[str, Any]) -> str:
    action = _text(event.get("action"), "unknown")
    return canonical_action(action)


def _is_semantic_event(event: Mapping[str, Any]) -> bool:
    event_type = _text(event.get("event_type"))
    action = _text(event.get("action"), "unknown")
    node_type = _text(event.get("node_type"), "")
    if event_type == "action":
        return node_type != "run_control"
    if event_type == "run" and action.lower() in TERMINAL_ACTIONS:
        return True
    return False


def _is_start_event(event: Mapping[str, Any]) -> bool:
    action = _text(event.get("action"), "").lower()
    return _text(event.get("event_type")) == "run" and action in START_ACTIONS


def _semantic_events(
    events: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    video_id: str,
    baseline: str,
    workflow_type_id: str,
    source_trace_sha256: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event_index, event in enumerate(events):
        if not _is_semantic_event(event):
            continue
        event_id = _text(event.get("event_id"), f"{run_id}:unknown:{event_index}")
        status = _text(event.get("status"), "unknown")
        action = _text(event.get("action"), "unknown")
        input_data = event.get("input") if isinstance(event.get("input"), Mapping) else {}
        standard_call = input_data.get("standard_tool_call") if isinstance(input_data, Mapping) else None
        rows.append(
            {
                "schema_version": SEMANTIC_VERSION,
                "derived_event_id": f"{run_id}:semantic:{len(rows)}",
                "source_event_ids": [event_id],
                "source_trace_sha256": source_trace_sha256,
                "derivation_version": DERIVATION_VERSION,
                "value_source": "measured",
                "run_id": run_id,
                "video_id": video_id,
                "baseline": baseline,
                "workflow_type_id": workflow_type_id,
                "event_index": len(rows),
                "source_event_index": event_index,
                "event_type": _text(event.get("event_type"), "unknown"),
                "activity": _event_activity(event),
                "raw_action": action,
                "node_type": _text(event.get("node_type"), "unknown"),
                "status": status,
                "retry_of": event.get("retry_of"),
                "parent_step_ids": event.get("parent_step_ids") or [],
                "runtime_ms": _event_runtime(event),
                "timestamp_start": event.get("timestamp_start"),
                "timestamp_end": event.get("timestamp_end"),
                "standard_tool_name": standard_call.get("name") if isinstance(standard_call, Mapping) else None,
                "parameter_provenance": "unknown",
            }
        )
    return rows


def _compute_events(
    events: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    video_id: str,
    baseline: str,
    workflow_type_id: str,
    source_trace_sha256: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event_index, event in enumerate(events):
        event_id = _text(event.get("event_id"), f"{run_id}:unknown:{event_index}")
        resource = event.get("resource") if isinstance(event.get("resource"), Mapping) else {}
        input_data = event.get("input") if isinstance(event.get("input"), Mapping) else {}
        row = {
            "schema_version": COMPUTE_VERSION,
            "derived_event_id": f"{run_id}:compute:{len(rows)}",
            "source_event_ids": [event_id],
            "source_trace_sha256": source_trace_sha256,
            "derivation_version": DERIVATION_VERSION,
            "value_source": "measured",
            "run_id": run_id,
            "video_id": video_id,
            "baseline": baseline,
            "workflow_type_id": workflow_type_id,
            "event_index": len(rows),
            "source_event_index": event_index,
            "event_type": _text(event.get("event_type"), "unknown"),
            "activity": _event_activity(event),
            "raw_action": _text(event.get("action"), "unknown"),
            "node_type": _text(event.get("node_type"), "unknown"),
            "status": _text(event.get("status"), "unknown"),
            "retry_of": event.get("retry_of"),
            "runtime_ms": _event_runtime(event),
            "api_wait_ms": _number(resource.get("api_wait_ms")),
            "queue_ms": resource.get("queue_ms"),
            "load_ms": resource.get("load_ms"),
            "peak_allocated_mb": resource.get("peak_allocated_mb"),
            "peak_reserved_mb": resource.get("peak_reserved_mb"),
            "gpu_id": resource.get("gpu_id"),
            "gpu_model": resource.get("gpu_model"),
            "model_id": event.get("model_id"),
            "model_resident_before": event.get("model_resident_before"),
            "input_scale": {
                "frame_count": input_data.get("frame_count_after"),
                "frame_count_before": input_data.get("frame_count_before"),
                "yolo_batch": input_data.get("yolo_batch"),
            "qwen_image_count": resource.get("qwen_image_count")
            or (
                input_data.get("tool_metrics", {}).get("qwen_image_count")
                if isinstance(input_data.get("tool_metrics"), Mapping)
                else None
            ),
            },
            "timestamp_start": event.get("timestamp_start"),
            "timestamp_end": event.get("timestamp_end"),
        }
        rows.append(row)
    return rows


def _workflow_type_id(manifest: Mapping[str, Any], events: Sequence[Mapping[str, Any]], baseline: str) -> str:
    explicit = _text(_manifest_value(manifest, "workflow_type_id", "workflow_type"))
    if explicit:
        return explicit
    dataset = _text(manifest.get("dataset")) or _text(_event_value(events, "dataset"), "unknown")
    return f"{dataset}.{baseline or 'unknown'}"


def _run_row(
    run_dir: Path,
    *,
    split_map: Mapping[str, str],
    include_errors: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    trace_path = run_dir / "trace.jsonl"
    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json(manifest_path)
    events = _trace_events(trace_path)
    run_id = _text(manifest.get("run_id"), run_dir.name)
    task_id = _text(manifest.get("task_id")) or _text(_event_value(events, "task_id"), run_id)
    status = _run_status(run_dir, manifest)
    if status != "success" and not include_errors:
        raise ValueError("run is not successful")
    trace_hash = _sha256(trace_path)
    manifest_hash = _sha256(manifest_path)
    video_id = _video_id(manifest, task_id)
    baseline = _text(manifest.get("baseline")) or _text(_event_value(events, "baseline", "baseline"), "unknown")
    model_stack_id = _text(_manifest_value(manifest, "model_stack_id")) or _text(_event_value(events, "model_stack_id", "model_stack_id"), "unknown")
    planner_model_id = _text(_manifest_value(manifest, "planner_model_id", "model_name")) or _text(_event_value(events, "planner_model_id", "planner_model_id"), "unknown")
    task_structure = manifest.get("task_structure") if isinstance(manifest.get("task_structure"), Mapping) else {}
    workflow_type_id = _workflow_type_id(manifest, events, baseline)
    split = _text(split_map.get(video_id), "unassigned")
    validation_errors = validate_trace(trace_path)
    semantic = _semantic_events(
        events,
        run_id=run_id,
        video_id=video_id,
        baseline=baseline,
        workflow_type_id=workflow_type_id,
        source_trace_sha256=trace_hash,
    )
    compute = _compute_events(
        events,
        run_id=run_id,
        video_id=video_id,
        baseline=baseline,
        workflow_type_id=workflow_type_id,
        source_trace_sha256=trace_hash,
    )
    terminal = bool(semantic and semantic[-1]["raw_action"].lower() in TERMINAL_ACTIONS)
    retries = sum(1 for event in events if event.get("retry_of"))
    errors = sum(1 for event in events if event.get("status") == "error")
    row = {
        "schema_version": SNAPSHOT_VERSION,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "dataset": _text(manifest.get("dataset"), "unknown"),
        "framework": _text(manifest.get("framework"), "unknown"),
        "task_id": task_id,
        "base_task_id": _base_task_id(task_id),
        "video_id": video_id,
        "split": split,
        "baseline": baseline,
        "workflow_type_id": workflow_type_id,
        "model_stack_id": model_stack_id,
        "planner_model_id": planner_model_id,
        "task_structure": dict(task_structure),
        "video_sha256": manifest.get("video_sha256"),
        "question_sha256": manifest.get("question_sha256"),
        "seed": manifest.get("seed"),
        "yolo_batch": manifest.get("yolo_batch"),
        "status": status,
        "validator": "valid" if not validation_errors else "invalid",
        "validator_errors": validation_errors,
        "has_terminal": terminal,
        "event_count": len(events),
        "semantic_event_count": len(semantic),
        "compute_event_count": len(compute),
        "retry_count": retries,
        "error_count": errors,
        "trace_sha256": trace_hash,
        "manifest_sha256": manifest_hash,
        "derivation_version": DERIVATION_VERSION,
        "value_source": "measured",
    }
    enrichment = {
        "schema_version": DERIVATION_VERSION,
        "run_id": run_id,
        "source_trace_sha256": trace_hash,
        "source_manifest_sha256": manifest_hash,
        "source_event_ids": [str(event.get("event_id")) for event in events if event.get("event_id")],
        "workflow": {
            "workflow_type_id": workflow_type_id,
            "workflow_id": run_id,
            "agent_id_source": "event.action",
        },
        "activity": {
            "canonicalization": "tracing.collectors.structured_state.canonical_action",
            "unknown_count": sum(1 for event in semantic if event["activity"] == "other"),
        },
        "dependencies": {
            "parameter_provenance": "unknown",
            "control_flow_alternatives": "unknown",
            "speculation_lifecycle": "unknown",
        },
        "quality": {
            "status": status,
            "validator": row["validator"],
            "has_terminal": terminal,
            "split": split,
        },
        "derivation_version": DERIVATION_VERSION,
        "value_source": "derived",
    }
    prefixes: list[dict[str, Any]] = []
    for position in range(len(semantic) + 1):
        prefix = semantic[:position]
        target = semantic[position] if position < len(semantic) else None
        state_features = _state_features(_state_snapshot(run_dir, position), position)
        enriched_task_structure = _merge_task_structure(task_structure, state_features)
        prefixes.append(
            {
                "schema_version": PREFIX_VERSION,
                "prefix_id": f"{run_id}:semantic:{position}",
                "run_id": run_id,
                "video_id": video_id,
                "split": split,
                "dataset": _text(manifest.get("dataset"), "unknown"),
                "task_id": task_id,
                "base_task_id": _base_task_id(task_id),
                "baseline": baseline,
                "workflow_type_id": workflow_type_id,
                "model_stack_id": model_stack_id,
                "planner_model_id": planner_model_id,
                "task_structure": enriched_task_structure,
                "state_features": state_features,
                "state_present": bool(state_features.get("state_present")),
                "position": position,
                "prefix_activities": [event["activity"] for event in prefix],
                "prefix_raw_actions": [event["raw_action"] for event in prefix],
                "prefix_source_event_ids": [event["source_event_ids"][0] for event in prefix],
                "target_next_activity": target["activity"] if target else "__END__",
                "target_next_raw_action": target["raw_action"] if target else "__END__",
                "target_source_event_id": target["source_event_ids"][0] if target else None,
                "remaining_steps": len(semantic) - position,
                "remaining_runtime_ms": sum(_number(event.get("runtime_ms")) for event in semantic[position:]),
                "future_events_included_in_input": False,
                "ground_truth_included_in_input": False,
                "source_trace_sha256": trace_hash,
                "derivation_version": DERIVATION_VERSION,
                "value_source": "derived",
            }
        )
    return row, semantic, compute, enrichment, prefixes


def _discover_run_dirs(roots: Sequence[Path]) -> list[Path]:
    found: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for trace_path in root.rglob("trace.jsonl"):
            if (trace_path.parent / "run_manifest.json").is_file():
                found.add(trace_path.parent)
    return sorted(found)


def _slot_key(row: Mapping[str, Any]) -> str:
    return "|".join(
        [
            _text(row.get("dataset"), "unknown"),
            _text(row.get("video_id"), "unknown"),
            _text(row.get("base_task_id"), "unknown"),
            _text(row.get("baseline"), "unknown"),
            _text(row.get("model_stack_id"), "unknown"),
        ]
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build(
    roots: Sequence[Path],
    output_dir: Path,
    *,
    split_manifest: Path | None = None,
    include_errors: bool = False,
) -> dict[str, Any]:
    split_map = _load_split(split_manifest)
    candidates: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    compute_rows: list[dict[str, Any]] = []
    enrichment_rows: list[dict[str, Any]] = []
    prefix_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for run_dir in _discover_run_dirs(roots):
        try:
            row, semantic, compute, enrichment, prefixes = _run_row(
                run_dir,
                split_map=split_map,
                include_errors=include_errors,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            rejected.append({"run_dir": str(run_dir), "reason": f"{type(exc).__name__}: {exc}"})
            continue
        candidates.append(row)
        semantic_rows.extend(semantic)
        compute_rows.extend(compute)
        enrichment_rows.append(enrichment)
        prefix_rows.extend(prefixes)

    by_slot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_slot[_slot_key(row)].append(row)
    selected_ids: set[str] = set()
    for slot, rows in by_slot.items():
        rows.sort(
            key=lambda row: (
                row["status"] != "success",
                row["validator"] != "valid",
                row["run_id"],
                row["run_dir"],
            )
        )
        if rows:
            selected_ids.add(rows[0]["run_id"])
        for rank, row in enumerate(rows, 1):
            row["slot_key"] = slot
            row["dedupe_rank"] = rank
            row["selected_for_snapshot"] = row["run_id"] in selected_ids
    selected = [row for row in candidates if row.get("selected_for_snapshot")]
    selected.sort(key=lambda row: (row["video_id"], row["base_task_id"], row["baseline"], row["model_stack_id"], row["run_id"]))

    selected_run_ids = {row["run_id"] for row in selected}
    semantic_rows = [row for row in semantic_rows if row["run_id"] in selected_run_ids]
    compute_rows = [row for row in compute_rows if row["run_id"] in selected_run_ids]
    enrichment_rows = [row for row in enrichment_rows if row["run_id"] in selected_run_ids]
    prefix_rows = [row for row in prefix_rows if row["run_id"] in selected_run_ids]

    _write_jsonl(output_dir / "candidate_snapshot_v0_1.jsonl", selected)
    _write_jsonl(output_dir / "candidate_candidates_v0_1.jsonl", candidates)
    _write_jsonl(output_dir / "semantic_events_v0_1.jsonl", semantic_rows)
    _write_jsonl(output_dir / "compute_events_v0_1.jsonl", compute_rows)
    _write_jsonl(output_dir / "trace_enrichment_v0_1.jsonl", enrichment_rows)
    _write_jsonl(output_dir / "prefix_samples_v0_1.jsonl", prefix_rows)
    _write_jsonl(output_dir / "rejected_runs_v0_1.jsonl", rejected)
    summary = {
        "schema_version": DERIVATION_VERSION,
        "roots": [str(root) for root in roots],
        "split_manifest": str(split_manifest) if split_manifest else None,
        "include_errors": include_errors,
        "discovered_candidates": len(candidates),
        "selected_snapshot_runs": len(selected),
        "rejected_runs": len(rejected),
        "semantic_events": len(semantic_rows),
        "compute_events": len(compute_rows),
        "prefix_samples": len(prefix_rows),
        "state_present_prefix_samples": sum(bool(row.get("state_present")) for row in prefix_rows),
        "validator_valid_selected": sum(row["validator"] == "valid" for row in selected),
        "unassigned_split_selected": sum(row["split"] == "unassigned" for row in selected),
        "duplicate_candidate_runs": len(candidates) - len(selected),
        "status_selected": dict(Counter(row["status"] for row in selected)),
        "baseline_selected": dict(Counter(row["baseline"] for row in selected)),
        "model_stack_selected": dict(Counter(row["model_stack_id"] for row in selected)),
        "derivation_version": DERIVATION_VERSION,
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    (output_dir / "enrichment_summary_v0_1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, type=Path, help="Raw results root; repeatable")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--include-errors", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    summary = build(
        args.root,
        args.output_dir,
        split_manifest=args.split_manifest,
        include_errors=args.include_errors,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
