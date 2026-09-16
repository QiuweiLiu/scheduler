#!/usr/bin/env python3
"""Dependency-free validator for trace schema v0.1 JSONL files."""

from __future__ import annotations

import argparse
import datetime as datetime_module
import json
from pathlib import Path
from typing import Any, Iterable, Optional


SCHEMA_VERSION = "0.1"
REQUIRED_EVENT_FIELDS = {
    "schema_version",
    "event_id",
    "event_type",
    "run_id",
    "framework",
    "dataset",
    "task_id",
    "step_id",
    "parent_step_ids",
    "action",
    "node_type",
    "model_id",
    "model_resident_before",
    "input",
    "resource",
    "status",
    "retry_of",
    "output_summary_ref",
    "timestamp_start",
    "timestamp_end",
    "error",
}
_NONNEGATIVE_RESOURCE_FIELDS = {
    "queue_ms",
    "decode_ms",
    "load_ms",
    "local_runtime_ms",
    "peak_allocated_mb",
    "peak_reserved_mb",
    "system_gpu_memory_used_mib",
    "system_gpu_memory_total_mib",
    "system_gpu_utilization_pct",
}


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _is_nonnegative_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0


def _parse_iso_datetime(value: Any) -> Optional[datetime_module.datetime]:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime_module.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _validate_optional_nonnegative(
    errors: list[str],
    prefix: str,
    resource: dict[str, Any],
    name: str,
) -> None:
    value = resource.get(name)
    if value is not None and not _is_nonnegative_number(value):
        errors.append(f"{prefix} resource.{name} must be a non-negative number or null")


def validate_event(event: Any, line_number: int) -> list[str]:
    prefix = f"line {line_number}:"
    if not isinstance(event, dict):
        return [f"{prefix} event must be an object"]

    errors: list[str] = []
    missing = sorted(REQUIRED_EVENT_FIELDS - set(event))
    if missing:
        errors.append(f"{prefix} missing fields: {', '.join(missing)}")
        return errors

    if event["schema_version"] != SCHEMA_VERSION:
        errors.append(f"{prefix} unsupported schema_version {event['schema_version']!r}")
    if not _is_nonempty_string(event["event_id"]):
        errors.append(f"{prefix} event_id must be a non-empty string")
    if event["event_type"] not in {"action", "api_call", "run"}:
        errors.append(f"{prefix} invalid event_type")
    for name in ("run_id", "framework", "dataset", "task_id", "action", "node_type"):
        if not _is_nonempty_string(event[name]):
            errors.append(f"{prefix} {name} must be a non-empty string")
    if not isinstance(event["step_id"], int) or isinstance(event["step_id"], bool) or event["step_id"] < 0:
        errors.append(f"{prefix} step_id must be a non-negative integer")
    if not isinstance(event["parent_step_ids"], list) or not all(
        isinstance(step, int) and not isinstance(step, bool) and step >= 0
        for step in event["parent_step_ids"]
    ):
        errors.append(f"{prefix} parent_step_ids must be a list of non-negative integers")
    if event["model_id"] is not None and not _is_nonempty_string(event["model_id"]):
        errors.append(f"{prefix} model_id must be a non-empty string or null")
    if event["model_resident_before"] is not None and not isinstance(
        event["model_resident_before"], bool
    ):
        errors.append(f"{prefix} model_resident_before must be a boolean or null")
    if not isinstance(event["input"], dict):
        errors.append(f"{prefix} input must be an object")

    resource = event["resource"]
    if not isinstance(resource, dict):
        errors.append(f"{prefix} resource must be an object")
    else:
        for name in ("runtime_ms", "api_wait_ms"):
            if not _is_nonnegative_number(resource.get(name)):
                errors.append(f"{prefix} resource.{name} must be a non-negative number")
        for name in _NONNEGATIVE_RESOURCE_FIELDS:
            _validate_optional_nonnegative(errors, prefix, resource, name)
        gpu_id = resource.get("gpu_id")
        if gpu_id is not None and (
            not isinstance(gpu_id, int) or isinstance(gpu_id, bool) or gpu_id < 0
        ):
            errors.append(f"{prefix} resource.gpu_id must be a non-negative integer or null")
        for name in ("gpu_model", "gpu_driver_version"):
            value = resource.get(name)
            if value is not None and not _is_nonempty_string(value):
                errors.append(f"{prefix} resource.{name} must be a non-empty string or null")

        runtime_ms = resource.get("runtime_ms")
        api_wait_ms = resource.get("api_wait_ms")
        local_runtime_ms = resource.get("local_runtime_ms")
        if _is_nonnegative_number(runtime_ms) and _is_nonnegative_number(api_wait_ms):
            if api_wait_ms > runtime_ms + 0.001:
                errors.append(f"{prefix} resource.api_wait_ms cannot exceed runtime_ms")
        if _is_nonnegative_number(runtime_ms) and _is_nonnegative_number(local_runtime_ms):
            if local_runtime_ms > runtime_ms + 0.001:
                errors.append(f"{prefix} resource.local_runtime_ms cannot exceed runtime_ms")

    if event["status"] not in {"success", "error", "cancelled"}:
        errors.append(f"{prefix} invalid status")
    retry_of = event["retry_of"]
    if retry_of is not None and not _is_nonempty_string(retry_of):
        errors.append(f"{prefix} retry_of must be a non-empty string or null")
    output_ref = event["output_summary_ref"]
    if output_ref is not None and not _is_nonempty_string(output_ref):
        errors.append(f"{prefix} output_summary_ref must be a non-empty string or null")
    error = event["error"]
    if error is not None and not _is_nonempty_string(error):
        errors.append(f"{prefix} error must be a non-empty string or null")
    if event["status"] == "error" and not _is_nonempty_string(error):
        errors.append(f"{prefix} error status requires a non-empty error message")

    start_time = _parse_iso_datetime(event["timestamp_start"])
    end_time = _parse_iso_datetime(event["timestamp_end"])
    if start_time is None:
        errors.append(f"{prefix} timestamp_start must be timezone-aware ISO-8601")
    if end_time is None:
        errors.append(f"{prefix} timestamp_end must be timezone-aware ISO-8601")
    if start_time is not None and end_time is not None and end_time < start_time:
        errors.append(f"{prefix} timestamp_end must not precede timestamp_start")
    return errors


def validate_trace(trace_path: Path) -> list[str]:
    errors: list[str] = []
    seen = 0
    seen_event_ids: set[str] = set()
    with trace_path.open(encoding="utf-8") as trace_file:
        for line_number, line in enumerate(trace_file, start=1):
            if not line.strip():
                errors.append(f"line {line_number}: blank lines are not permitted")
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON ({exc.msg})")
                continue

            errors.extend(validate_event(event, line_number))
            if isinstance(event, dict):
                event_id = event.get("event_id")
                if _is_nonempty_string(event_id):
                    if event_id in seen_event_ids:
                        errors.append(f"line {line_number}: duplicate event_id {event_id!r}")
                    retry_of = event.get("retry_of")
                    if _is_nonempty_string(retry_of) and retry_of not in seen_event_ids:
                        errors.append(
                            f"line {line_number}: retry_of must reference an earlier event_id"
                        )
                    seen_event_ids.add(event_id)
            seen += 1
    if seen == 0:
        errors.append("trace contains no events")
    return errors


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_path", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    errors = validate_trace(args.trace_path)
    if errors:
        print("INVALID")
        print("\n".join(errors))
        return 1
    print("VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
