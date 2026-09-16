#!/usr/bin/env python3
"""Build auditable node-level workload v0.2 from frozen VideoSeek traces.

The v0.1 builder remains untouched.  This version keeps raw traces immutable,
restores parent edges from the original trace when they exist, marks missing
edges explicitly, separates scheduler estimates from simulator truth, and
generates fixed workload cells for Phase 4.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TEMPLATE_VERSION = "job-template-v0.2"
EPISODE_VERSION = "workload-episode-v0.2"
PRESSURES = (0.50, 0.70, 0.85, 0.95, 1.05)
TOPOLOGIES = ((32760.0, 32760.0), (24576.0, 24576.0), (32760.0, 24576.0))
INITIAL_STATES = ("cold", "warm", "skewed")
MAIN_ARRIVALS = ("alibaba_replay", "poisson", "burst")
DIAGNOSTIC_ARRIVALS = ("staggered",)
JOB_COUNTS = (16, 32, 64)

# These are conservative capacity proxies used only to make cache states
# physically checkable.  They are not claimed to be measured resident weights.
# Every use is recorded with this provenance and must be replaced after the
# contention microbenchmark.
MODEL_RESIDENT_MB = {
    "Qwen3-VL-8B-Instruct": 17000.0,
    "Qwen3-4B": 7600.0,
    "Qwen2.5-VL-3B-Instruct": 7100.0,
    "yolo11x.pt": 512.0,
    "yolo26n.pt": 128.0,
    "cpu-metadata-adapter-v1": 0.0,
    "finish_argument": None,
}
RESIDENT_PROVENANCE = "controlled_peak_proxy_pending_contention_calibration"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _text(value: Any, default: str = "unknown") -> str:
    text = str(value or "").strip()
    return text or default


def _number(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) and parsed >= 0.0 else default


def _optional_number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0.0 else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile(values: Sequence[float], q: float) -> float | None:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)) and float(value) >= 0.0)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return clean[lower]
    fraction = position - lower
    return clean[lower] * (1.0 - fraction) + clean[upper] * fraction


def _resolve_raw_path(project_root: Path, snapshot: Mapping[str, Any]) -> Path | None:
    run_dir = _text(snapshot.get("run_dir"), "")
    if not run_dir:
        return None
    candidate = project_root / run_dir / "trace.jsonl"
    return candidate if candidate.exists() else None


def _raw_index(project_root: Path, snapshot: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], str | None]:
    path = _resolve_raw_path(project_root, snapshot)
    if path is None:
        return {}, defaultdict(list), None
    by_event: dict[str, dict[str, Any]] = {}
    by_step: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            event_id = _text(event.get("event_id"), "")
            if event_id:
                by_event[event_id] = event
            if event.get("step_id") is not None and event_id:
                by_step[str(event["step_id"])].append(event_id)
    return by_event, by_step, str(path)


def _effective_model_id(event: Mapping[str, Any], snapshot: Mapping[str, Any]) -> str:
    model = _text(event.get("model_id"))
    if model != "finish_argument":
        return model
    stack = _text(snapshot.get("model_stack_id"))
    if stack.startswith("stack_a"):
        return "Qwen3-VL-8B-Instruct"
    if stack.startswith("stack_b"):
        return "Qwen2.5-VL-3B-Instruct"
    return model


def _execution_lane(event: Mapping[str, Any], model_id: str) -> str:
    resource = event.get("resource") if isinstance(event.get("resource"), Mapping) else {}
    peak = max(_number(resource.get("peak_reserved_mb")), _number(resource.get("peak_allocated_mb")))
    if model_id.startswith("openai/") or model_id.startswith("api/"):
        return "api"
    if model_id.startswith("cpu-") or (peak <= 0.0 and _text(event.get("node_type")) in {"videotool_temporal", "videotool_generalist"}):
        return "cpu"
    return "gpu"


def _raw_metadata(event: Mapping[str, Any], raw_by_event: Mapping[str, Mapping[str, Any]]) -> tuple[list[str], list[str], list[str]]:
    source_ids = [str(value) for value in event.get("source_event_ids") or []]
    raw_events = [raw_by_event[source_id] for source_id in source_ids if source_id in raw_by_event]
    step_ids = sorted({str(raw["step_id"]) for raw in raw_events if raw.get("step_id") is not None}, key=lambda value: (len(value), value))
    parents = sorted({str(parent) for raw in raw_events for parent in (raw.get("parent_step_ids") or [])})
    return source_ids, step_ids, parents


def _resident_model(model_id: str) -> tuple[float | None, str]:
    if model_id in MODEL_RESIDENT_MB:
        return MODEL_RESIDENT_MB[model_id], "known_zero_for_cpu" if MODEL_RESIDENT_MB[model_id] == 0.0 else RESIDENT_PROVENANCE
    return None, "unknown_resident_model"


def _node_from_event(
    event: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    raw_by_event: Mapping[str, Mapping[str, Any]],
    index: int,
) -> dict[str, Any]:
    resource = event.get("resource") if isinstance(event.get("resource"), Mapping) else {}
    source_ids, step_ids, parents = _raw_metadata(event, raw_by_event)
    event_id = source_ids[0] if source_ids else _text(event.get("derived_event_id"), f"event:{index}")
    model_id = _effective_model_id(event, snapshot)
    lane = _execution_lane(event, model_id)
    peak_reserved = _optional_number(event.get("peak_reserved_mb"))
    peak_allocated = _optional_number(event.get("peak_allocated_mb"))
    if peak_reserved is None:
        peak_reserved = _optional_number(resource.get("peak_reserved_mb"))
    if peak_allocated is None:
        peak_allocated = _optional_number(resource.get("peak_allocated_mb"))
    workspace = max(value for value in (peak_reserved, peak_allocated) if value is not None) if any(value is not None for value in (peak_reserved, peak_allocated)) else None
    resident, resident_provenance = _resident_model(model_id)
    return {
        "node_id": event_id,
        "sequence_index": index,
        "source_event_ids": source_ids,
        "source_step_ids": step_ids,
        "event_type": _text(event.get("event_type")),
        "activity": _text(event.get("activity")),
        "raw_action": _text(event.get("raw_action") or event.get("action")),
        "node_type": _text(event.get("node_type")),
        "execution_lane": lane,
        "status": _text(event.get("status")),
        "retry_of": event.get("retry_of"),
        "parent_step_ids": parents,
        "predecessor_node_ids": [],
        "edge_provenance": "unresolved",
        "model_id": model_id,
        "runtime_ms": _number(event.get("runtime_ms") or resource.get("runtime_ms")),
        "load_ms": _optional_number(event.get("load_ms")) if event.get("load_ms") is not None else _optional_number(resource.get("load_ms")),
        "queue_ms": event.get("queue_ms") if event.get("queue_ms") is not None else resource.get("queue_ms"),
        "api_wait_ms": _number(event.get("api_wait_ms") or resource.get("api_wait_ms")),
        "workspace_peak_mb": workspace,
        "resident_model_mb": resident,
        "memory_provenance": resident_provenance,
        "gpu_id": event.get("gpu_id") if event.get("gpu_id") is not None else resource.get("gpu_id"),
        "gpu_model": event.get("gpu_model") if event.get("gpu_model") is not None else resource.get("gpu_model"),
        "source_trace_sha256": event.get("source_trace_sha256"),
    }


def _shares_source_step(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_steps = {str(value) for value in left.get("source_step_ids") or []}
    right_steps = {str(value) for value in right.get("source_step_ids") or []}
    return bool(left_steps & right_steps)


def _resolve_edges(nodes: list[dict[str, Any]], has_raw_parent_metadata: bool) -> list[str]:
    by_step: dict[str, list[str]] = defaultdict(list)
    by_id = {str(node["node_id"]): node for node in nodes}
    errors: list[str] = []
    for node in nodes:
        for step_id in node.get("source_step_ids") or []:
            by_step[str(step_id)].append(str(node["node_id"]))
    for index, node in enumerate(nodes):
        parents = node.get("parent_step_ids") or []
        if parents:
            predecessors: list[str] = []
            for parent in parents:
                candidates = by_step.get(str(parent), [])
                if not candidates:
                    errors.append(f"{node['node_id']}:missing_parent_step:{parent}")
                predecessors.extend(candidates)
            node["predecessor_node_ids"] = sorted(set(predecessors), key=lambda value: by_id[value]["sequence_index"] if value in by_id else 0)
            node["edge_provenance"] = "raw_parent_step_ids"
        elif index == 0:
            node["predecessor_node_ids"] = []
            node["edge_provenance"] = "root"
        elif _shares_source_step(nodes[index - 1], node):
            node["predecessor_node_ids"] = [str(nodes[index - 1]["node_id"])]
            node["edge_provenance"] = "same_step_event_order"
        elif not has_raw_parent_metadata:
            node["predecessor_node_ids"] = [str(nodes[index - 1]["node_id"])]
            node["edge_provenance"] = "sequence_fallback_no_parent_data"
        else:
            node["predecessor_node_ids"] = []
            node["edge_provenance"] = "invalid_missing_parent"
            errors.append(f"{node['node_id']}:missing_parent_across_steps")
    for node in nodes:
        if any(pred == node["node_id"] for pred in node["predecessor_node_ids"]):
            errors.append(f"{node['node_id']}:self_edge")
        if any(by_id.get(pred, {}).get("sequence_index", -1) >= node["sequence_index"] for pred in node["predecessor_node_ids"]):
            errors.append(f"{node['node_id']}:non_forward_edge")
    return errors


def _read_snapshot(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    selected = [row for row in rows if row.get("selected_for_snapshot", True)]
    run_ids = [str(row.get("run_id")) for row in selected]
    if not selected or len(run_ids) != len(set(run_ids)):
        raise ValueError("snapshot is empty or contains duplicate run_id values")
    return selected


def compile_templates(project_root: Path, snapshot_path: Path, compute_path: Path, output_path: Path) -> dict[str, Any]:
    snapshots = _read_snapshot(snapshot_path)
    selected_ids = {str(row["run_id"]) for row in snapshots}
    compute_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in _read_jsonl(compute_path):
        run_id = _text(event.get("run_id"))
        if run_id not in selected_ids:
            continue
        if event.get("event_type") == "action" or _number(event.get("runtime_ms")) > 0.0:
            compute_by_run[run_id].append(event)
    templates: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    edge_counts: Counter[str] = Counter()
    lane_counts: Counter[str] = Counter()
    unknown_resident: Counter[str] = Counter()
    for snapshot in snapshots:
        run_id = str(snapshot["run_id"])
        raw_by_event, _, raw_path = _raw_index(project_root, snapshot)
        events = sorted(compute_by_run.get(run_id, []), key=lambda row: int(row.get("event_index") or 0))
        if not events:
            invalid.append({"run_id": run_id, "reason": "no_compute_events"})
            continue
        nodes = [_node_from_event(event, snapshot, raw_by_event, index) for index, event in enumerate(events)]
        has_raw_parent_metadata = any(bool(raw.get("parent_step_ids")) for raw in raw_by_event.values())
        edge_errors = _resolve_edges(nodes, has_raw_parent_metadata)
        for node in nodes:
            edge_counts[str(node["edge_provenance"])] += 1
            lane_counts[str(node["execution_lane"])] += 1
            if node["resident_model_mb"] is None and node["execution_lane"] == "gpu":
                unknown_resident[str(node["model_id"])] += 1
        if edge_errors:
            invalid.append({"run_id": run_id, "reason": "invalid_edges", "errors": edge_errors[:20]})
            continue
        templates.append({
            "schema_version": TEMPLATE_VERSION,
            "template_id": run_id,
            "run_id": run_id,
            "video_id": _text(snapshot.get("video_id")),
            "split": _text(snapshot.get("split")),
            "dataset": _text(snapshot.get("dataset")),
            "task_id": _text(snapshot.get("task_id")),
            "base_task_id": _text(snapshot.get("base_task_id")),
            "baseline": _text(snapshot.get("baseline")),
            "workflow_type_id": _text(snapshot.get("workflow_type_id")),
            "model_stack_id": _text(snapshot.get("model_stack_id")),
            "planner_model_id": _text(snapshot.get("planner_model_id")),
            "task_structure": dict(snapshot.get("task_structure") or {}),
            "source_trace_sha256": snapshot.get("trace_sha256"),
            "source_manifest_sha256": snapshot.get("manifest_sha256"),
            "source_raw_trace_path": raw_path,
            "status": _text(snapshot.get("status")),
            "validator": _text(snapshot.get("validator")),
            "nodes": nodes,
            "node_count": len(nodes),
            "estimated_runtime_ms": sum(float(node["runtime_ms"]) for node in nodes),
            "estimated_peak_workspace_mb": max((float(node["workspace_peak_mb"] or 0.0) for node in nodes), default=0.0),
            "edge_provenance_counts": dict(Counter(str(node["edge_provenance"]) for node in nodes)),
            "derivation_version": TEMPLATE_VERSION,
            "value_source": "measured_trace_events_with_raw_parent_edges",
        })
    if invalid:
        _write_json(output_path.with_suffix(output_path.suffix + ".invalid.jsonl"), invalid)
    if not templates:
        raise ValueError("no valid job templates could be compiled")
    _write_jsonl(output_path, templates)
    summary = {
        "schema_version": TEMPLATE_VERSION,
        "snapshot": str(snapshot_path),
        "compute_events": str(compute_path),
        "output": str(output_path),
        "templates": len(templates),
        "invalid_templates": len(invalid),
        "videos": len({row["video_id"] for row in templates}),
        "nodes": sum(int(row["node_count"]) for row in templates),
        "split_counts": dict(sorted(Counter(row["split"] for row in templates).items())),
        "baseline_counts": dict(sorted(Counter(row["baseline"] for row in templates).items())),
        "edge_provenance_counts": dict(sorted(edge_counts.items())),
        "execution_lane_counts": dict(sorted(lane_counts.items())),
        "unknown_gpu_resident_models": dict(sorted(unknown_resident.items())),
        "source_snapshot_sha256": _sha256(snapshot_path),
        "source_compute_sha256": _sha256(compute_path),
        "resident_memory_provenance": RESIDENT_PROVENANCE,
    }
    _write_json(output_path.with_suffix(output_path.suffix + ".summary.json"), summary)
    return summary


def _load_arrival_gaps(path: Path | None) -> dict[str, list[float]]:
    if path is None or not path.exists():
        return {split: [] for split in ("train", "validation", "test")}
    starts: list[float] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 5:
                continue
            value = _optional_number(row[4])
            if value is not None:
                starts.append(value)
    starts = sorted(starts)
    if len(starts) < 4:
        return {split: [] for split in ("train", "validation", "test")}
    # Alibaba start_time is seconds. Preserve duplicate submissions as zero gaps
    # and convert the interval representation to the scheduler's milliseconds.
    gaps = [max(0.0, (right - left) * 1000.0) for left, right in zip(starts, starts[1:])]
    n = len(gaps)
    train_end = max(1, int(n * 0.60))
    validation_end = max(train_end + 1, int(n * 0.80))
    cuts = {
        "train": (0, train_end),
        "validation": (train_end, min(validation_end, n)),
        "test": (min(validation_end, n), n),
    }
    return {split: gaps[start:end] for split, (start, end) in cuts.items()}


def build_arrival_manifest(path: Path | None, output: Path) -> dict[str, Any]:
    gaps = _load_arrival_gaps(path)
    manifest = {
        "schema_version": "arrival-source-v0.2",
        "source": str(path) if path else None,
        "source_sha256": _sha256(path) if path and path.exists() else None,
        "source_time_unit": "seconds",
        "derived_gap_time_unit": "milliseconds",
        "split_gap_counts": {split: len(values) for split, values in gaps.items()},
        "split_zero_gap_counts": {split: sum(1 for value in values if value == 0.0) for split, values in gaps.items()},
        "split_gap_medians_ms": {split: _quantile(values, 0.50) for split, values in gaps.items()},
        "split_gap_p95_ms": {split: _quantile(values, 0.95) for split, values in gaps.items()},
        "time_block_policy": "sorted_start_time_contiguous_60_20_20",
        "replay_policy": "sample_one_contiguous_gap_block_without_replacement",
        "usage": "arrival_intervals_only_resource_demand_not_reused",
    }
    _write_json(output, manifest)
    return manifest


def _load_resource_contract(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        raise ValueError("a frozen --resource-contract path is required for workload generation")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != "scheduler-resource-contract-v1":
        raise ValueError(f"unsupported resource contract schema: {contract.get('schema_version')}")
    required = {
        "runtime_p50_ms",
        "runtime_p90_ms",
        "load_p50_ms",
        "peak_memory_p95_mb",
        "uncertainty",
    }
    outputs = set(contract.get("outputs") or [])
    missing = sorted(required - outputs)
    if missing:
        raise ValueError(f"resource contract is missing outputs: {missing}")
    return {
        "schema_version": contract["schema_version"],
        "path": str(path),
        "sha256": _sha256(path),
        "outputs": sorted(outputs),
        "unknown_policy": contract.get("unknown_policy"),
    }


def _resource_stats(templates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    def add(key: str, node: Mapping[str, Any]) -> None:
        runtime = _number(node.get("runtime_ms"))
        values[key]["runtime"].append(runtime)
        if node.get("load_ms") is not None:
            values[key]["load"].append(_number(node.get("load_ms")))
        if node.get("workspace_peak_mb") is not None:
            values[key]["memory"].append(_number(node.get("workspace_peak_mb")))

    for template in templates:
        for node in template.get("nodes") or []:
            lane = _text(node.get("execution_lane"))
            model = _text(node.get("model_id"))
            activity = _text(node.get("activity"))
            node_type = _text(node.get("node_type"))
            add("|".join((model, activity, node_type, lane)), node)
            add("|".join((model, activity, "*", lane)), node)
            add("|".join((model, "*", "*", lane)), node)
            add("|".join(("*", activity, "*", lane)), node)
            add("|".join(("*", "*", "*", lane)), node)
    return {
        key: {
            "runtime_p50_ms": _quantile(item["runtime"], 0.50),
            "runtime_p90_ms": _quantile(item["runtime"], 0.90),
            "load_p50_ms": _quantile(item["load"], 0.50),
            "peak_memory_p95_mb": _quantile(item["memory"], 0.95),
            "count": len(item["runtime"]),
        }
        for key, item in values.items()
    }


def _estimate_node(node: Mapping[str, Any], stats: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    lane = _text(node.get("execution_lane"))
    model = _text(node.get("model_id"))
    activity = _text(node.get("activity"))
    node_type = _text(node.get("node_type"))
    keys = (
        "|".join((model, activity, node_type, lane)),
        "|".join((model, activity, "*", lane)),
        "|".join((model, "*", "*", lane)),
        "|".join(("*", activity, "*", lane)),
        "|".join(("*", "*", "*", lane)),
    )
    estimate = next((stats[key] for key in keys if key in stats and stats[key].get("count", 0) >= 3), None)
    if estimate is None:
        raise ValueError(f"no train-only resource estimate for model={model} activity={activity} node_type={node_type} lane={lane}")
    source = "train_template_quantiles"
    if lane in {"cpu", "api"}:
        memory = 0.0
    else:
        memory = estimate.get("peak_memory_p95_mb")
        if memory is None:
            memory = _optional_number(node.get("workspace_peak_mb"))
    return {
        "runtime_p50_ms": estimate.get("runtime_p50_ms"),
        "runtime_p90_ms": estimate.get("runtime_p90_ms"),
        "load_p50_ms": estimate.get("load_p50_ms"),
        "peak_memory_p95_mb": memory,
        "uncertainty": "empirical_quantile",
        "source": source,
    }


def _job_estimates(template: Mapping[str, Any], stats: Mapping[str, Mapping[str, Any]]) -> tuple[float, float, float]:
    estimates = [_estimate_node(node, stats) for node in template.get("nodes") or []]
    total_p50 = sum(_number(item.get("runtime_p50_ms")) for item in estimates)
    total_p90 = sum(_number(item.get("runtime_p90_ms")) for item in estimates)
    gpu_p50 = sum(_number(item.get("runtime_p50_ms")) for node, item in zip(template.get("nodes") or [], estimates) if node.get("execution_lane") == "gpu")
    return total_p50, total_p90, gpu_p50


def _select_templates(rng: random.Random, templates: Sequence[Mapping[str, Any]], count: int) -> list[Mapping[str, Any]]:
    if count > len(templates):
        raise ValueError(f"split has {len(templates)} templates but episode requests {count} without replacement")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for template in templates:
        grouped[f"{template.get('baseline')}|{template.get('model_stack_id')}"] .append(template)
    groups = list(grouped.values())
    for group in groups:
        rng.shuffle(group)
    selected: list[Mapping[str, Any]] = []
    cursor = 0
    while len(selected) < count:
        group = groups[cursor % len(groups)]
        if group:
            selected.append(group.pop())
        cursor += 1
        if cursor > count * max(2, len(groups)) and len(selected) < count:
            remaining = [row for group in groups for row in group]
            selected.extend(remaining[: count - len(selected)])
    return selected


def _normalised_positions(
    rng: random.Random,
    count: int,
    pattern: str,
    gaps: Sequence[float],
) -> tuple[list[float], dict[str, Any]]:
    if count <= 1:
        return [0.0], {"source": "single_job", "gap_count": 0, "scale_to_episode_window_ms": True}
    if pattern == "burst":
        prefix = max(1, count // 4)
        tail = max(1, count - prefix)
        tail_positions = [index / max(1, tail - 1) for index in range(tail)]
        positions = [0.0] * prefix + [0.15 + 0.85 * value for value in tail_positions]
        return positions[:count], {
            "source": "synthetic_burst",
            "gap_count": count - 1,
            "scale_to_episode_window_ms": True,
        }
    if pattern == "alibaba_replay":
        if len(gaps) < count - 1:
            raise ValueError(f"Alibaba replay split has {len(gaps)} gaps but needs {count - 1}")
        start_index = rng.randrange(0, len(gaps) - count + 2)
        chosen = [float(value) for value in gaps[start_index : start_index + count - 1]]
        metadata = {
            "source": "alibaba_v2020_pai_job_table",
            "gap_count": count - 1,
            "gap_start_index": start_index,
            "gap_end_index": start_index + count - 1,
            "scale_to_episode_window_ms": True,
        }
    elif pattern == "poisson":
        chosen = [rng.expovariate(1.0) for _ in range(count - 1)]
        metadata = {
            "source": "synthetic_poisson",
            "gap_count": count - 1,
            "scale_to_episode_window_ms": True,
        }
    else:
        chosen = [0.5 + 0.25 * (index % 3) for index in range(count - 1)]
        metadata = {
            "source": "synthetic_staggered",
            "gap_count": count - 1,
            "scale_to_episode_window_ms": True,
        }
    cumulative = [0.0]
    for gap in chosen:
        cumulative.append(cumulative[-1] + max(0.0, float(gap)))
    last = cumulative[-1] or 1.0
    return [value / last for value in cumulative], metadata


def _initial_residency(selected: Sequence[Mapping[str, Any]], state: str, capacities: Sequence[float]) -> tuple[list[list[str]], list[list[float]]]:
    if state == "cold":
        return ([[] for _ in capacities], [[] for _ in capacities])
    frequency = Counter(
        str(node.get("model_id"))
        for template in selected
        for node in template.get("nodes") or []
        if node.get("execution_lane") == "gpu"
    )
    models = [model for model, _ in frequency.most_common() if MODEL_RESIDENT_MB.get(model) is not None and MODEL_RESIDENT_MB.get(model, 0.0) > 0.0]
    result: list[list[str]] = [[] for _ in capacities]
    memory: list[list[float]] = [[] for _ in capacities]
    if not models:
        return result, memory
    if state == "skewed":
        candidate = models[0]
        mb = float(MODEL_RESIDENT_MB[candidate])
        if mb <= capacities[0] * 0.70:
            result[0] = [candidate]
            memory[0] = [mb]
        return result, memory
    for gpu_index, capacity in enumerate(capacities):
        used = 0.0
        for model in models:
            mb = float(MODEL_RESIDENT_MB[model])
            if used + mb > capacity * 0.70:
                continue
            result[gpu_index].append(model)
            memory[gpu_index].append(mb)
            used += mb
    return result, memory


def _load_gap_map(path: Path | None) -> dict[str, list[float]]:
    return _load_arrival_gaps(path)


def generate_split(
    template_path: Path,
    output_path: Path,
    *,
    split: str,
    count: int,
    seed: int,
    arrival_source: Path | None,
    resource_contract: Path | None,
    fixed_test_matrix: bool = False,
) -> dict[str, Any]:
    templates = [row for row in _read_jsonl(template_path) if _text(row.get("split")) == split]
    if not templates:
        raise ValueError(f"no templates for split={split}")
    rng = random.Random(seed)
    all_templates = _read_jsonl(template_path)
    train_templates = [row for row in all_templates if _text(row.get("split")) == "train"]
    stats = _resource_stats(train_templates or templates)
    contract_binding = _load_resource_contract(resource_contract)
    by_id = {str(row["template_id"]): row for row in templates}
    gaps = _load_gap_map(arrival_source).get(split, [])
    episodes: list[dict[str, Any]] = []
    cells: Counter[str] = Counter()
    for episode_index in range(count):
        if fixed_test_matrix:
            cell_index, repetition = divmod(episode_index, 50)
            arrival_pattern = MAIN_ARRIVALS[cell_index // (len(PRESSURES) * len(TOPOLOGIES) * len(INITIAL_STATES))]
            rest = cell_index % (len(PRESSURES) * len(TOPOLOGIES) * len(INITIAL_STATES))
            pressure = PRESSURES[rest // (len(TOPOLOGIES) * len(INITIAL_STATES))]
            rest %= len(TOPOLOGIES) * len(INITIAL_STATES)
            topology = TOPOLOGIES[rest // len(INITIAL_STATES)]
            initial_state = INITIAL_STATES[rest % len(INITIAL_STATES)]
            job_count = JOB_COUNTS[repetition % len(JOB_COUNTS)]
        else:
            cell_count = len(MAIN_ARRIVALS) * len(PRESSURES) * len(TOPOLOGIES) * len(INITIAL_STATES)
            cell_index = episode_index % cell_count
            repetition = episode_index // cell_count
            arrival_pattern = MAIN_ARRIVALS[cell_index // (len(PRESSURES) * len(TOPOLOGIES) * len(INITIAL_STATES))]
            rest = cell_index % (len(PRESSURES) * len(TOPOLOGIES) * len(INITIAL_STATES))
            pressure = PRESSURES[rest // (len(TOPOLOGIES) * len(INITIAL_STATES))]
            rest %= len(TOPOLOGIES) * len(INITIAL_STATES)
            topology = TOPOLOGIES[rest // len(INITIAL_STATES)]
            initial_state = INITIAL_STATES[rest % len(INITIAL_STATES)]
            job_count = JOB_COUNTS[repetition % len(JOB_COUNTS)]
            if arrival_pattern == "alibaba_replay" and not gaps:
                arrival_pattern = "poisson"
        selected = _select_templates(rng, templates, job_count)
        estimates = {str(row["template_id"]): _job_estimates(row, stats) for row in selected}
        total_gpu_runtime = sum(value[2] for value in estimates.values())
        window_ms = max(1.0, total_gpu_runtime / (pressure * max(1, len(topology))))
        positions, arrival_metadata = _normalised_positions(rng, job_count, arrival_pattern, gaps)
        arrivals = [round(position * window_ms, 3) for position in positions]
        realized_load = total_gpu_runtime / (window_ms * max(1, len(topology)))
        deadline_multiplier = (1.5, 2.0, 3.0)[episode_index % 3]
        jobs: list[dict[str, Any]] = []
        for job_index, template in enumerate(selected):
            total_p50, total_p90, gpu_p50 = estimates[str(template["template_id"])]
            jobs.append({
                "job_instance_id": f"episode_{split}_{episode_index:06d}_job_{job_index:03d}",
                "template_id": template["template_id"],
                "arrival_ms": arrivals[job_index],
                "service_class": "normal" if job_index % 4 else "priority",
                "deadline_multiplier": deadline_multiplier,
                "deadline_ms": round(arrivals[job_index] + deadline_multiplier * total_p90, 3),
                "scheduler_visible": {
                    "predicted_job_runtime_p50_ms": round(total_p50, 3),
                    "predicted_job_runtime_p90_ms": round(total_p90, 3),
                    "predicted_gpu_runtime_p50_ms": round(gpu_p50, 3),
                },
            })
        residency, residency_memory = _initial_residency(selected, initial_state, topology)
        cell = f"{arrival_pattern}|{pressure:.2f}|{','.join(str(int(value)) for value in topology)}|{initial_state}"
        cells[cell] += 1
        episodes.append({
            "schema_version": EPISODE_VERSION,
            "episode_id": f"{split}_{episode_index:06d}",
            "split": split,
            "seed": seed,
            "arrival_source": "alibaba_v2020_pai_job_table" if arrival_pattern == "alibaba_replay" else "synthetic",
            "arrival_pattern": arrival_pattern,
            "arrival_metadata": arrival_metadata,
            "arrival_span_ms": round(max(arrivals) - min(arrivals), 3) if arrivals else 0.0,
            "target_offered_compute_load": pressure,
            "realized_offered_compute_load": round(realized_load, 6),
            "predicted_total_gpu_runtime_ms": round(total_gpu_runtime, 3),
            "episode_window_ms": round(window_ms, 3),
            "gpu_topology_mb": list(topology),
            "initial_state": initial_state,
            "initial_residency_hint": residency,
            "initial_residency_memory_mb": residency_memory,
            "residency_provenance": RESIDENT_PROVENANCE,
            "resource_contract": contract_binding,
            "deadline_multiplier": deadline_multiplier,
            "jobs": jobs,
            "source_template_ids": sorted({str(row["template_id"]) for row in selected}),
            "source_video_ids": sorted({str(row["video_id"]) for row in selected}),
            "source_trace_sha256": sorted({str(row["source_trace_sha256"]) for row in selected if row.get("source_trace_sha256")}),
            "scenario_cell": cell,
            "derivation_version": EPISODE_VERSION,
            "value_source": "measured_template_sampling_plus_arrival_process_plus_train_resource_quantiles",
        })
    _write_jsonl(output_path, episodes)
    split_errors = []
    for episode in episodes:
        if len(episode["source_template_ids"]) != len(episode["jobs"]):
            split_errors.append(f"{episode['episode_id']}:duplicate_template")
        for job in episode["jobs"]:
            if str(by_id.get(str(job["template_id"]), {}).get("split")) != split:
                split_errors.append(f"{episode['episode_id']}:{job['template_id']}:split")
    summary = {
        "schema_version": EPISODE_VERSION,
        "output": str(output_path),
        "split": split,
        "episodes": len(episodes),
        "jobs": sum(len(row["jobs"]) for row in episodes),
        "arrival_source_counts": dict(Counter(row["arrival_source"] for row in episodes)),
        "arrival_pattern_counts": dict(sorted(Counter(row["arrival_pattern"] for row in episodes).items())),
        "pressure_counts": dict(sorted(Counter(row["target_offered_compute_load"] for row in episodes).items())),
        "initial_state_counts": dict(sorted(Counter(row["initial_state"] for row in episodes).items())),
        "gpu_topology_counts": {
            ",".join(str(value) for value in topology): count
            for topology, count in sorted(Counter(tuple(row["gpu_topology_mb"]) for row in episodes).items())
        },
        "job_count_counts": dict(sorted(Counter(len(row["jobs"]) for row in episodes).items())),
        "scenario_cell_counts": dict(sorted(cells.items())),
        "scenario_matrix_expected_cells": len(MAIN_ARRIVALS) * len(PRESSURES) * len(TOPOLOGIES) * len(INITIAL_STATES),
        "scenario_matrix_complete": len(cells) == len(MAIN_ARRIVALS) * len(PRESSURES) * len(TOPOLOGIES) * len(INITIAL_STATES) and all(value > 0 for value in cells.values()),
        "resource_contract": contract_binding,
        "split_isolation_errors": split_errors,
        "split_isolation_gate": not split_errors,
        "residency_capacity_gate": all(
            sum(memory) <= capacity * 0.70 + 1e-6
            for row in episodes
            for capacity_row, memory_row in zip(row["gpu_topology_mb"], row["initial_residency_memory_mb"])
            for capacity, memory in [(capacity_row, memory_row)]
        ),
    }
    _write_json(output_path.with_suffix(output_path.suffix + ".summary.json"), summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--compute", type=Path, required=True)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--arrival-source", type=Path)
    parser.add_argument("--arrival-manifest", type=Path)
    parser.add_argument("--resource-contract", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--episodes", type=Path)
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--fixed-test-matrix", action="store_true")
    args = parser.parse_args()
    if args.arrival_manifest:
        build_arrival_manifest(args.arrival_source, args.arrival_manifest)
    if args.snapshot and args.compute:
        summary = compile_templates(args.project_root, args.snapshot, args.compute, args.templates)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if args.split:
        if args.episodes is None or args.count <= 0:
            raise ValueError("--split requires --episodes and positive --count")
        summary = generate_split(
            args.templates,
            args.episodes,
            split=args.split,
            count=args.count,
            seed=args.seed,
            arrival_source=args.arrival_source,
            resource_contract=args.resource_contract,
            fixed_test_matrix=args.fixed_test_matrix,
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
