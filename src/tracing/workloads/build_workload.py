#!/usr/bin/env python3
"""Compile immutable trace-derived job templates and workload episodes.

The compiler keeps measured node sequences intact and only synthesizes job
arrival times, GPU topology, cache initial state, and episode sampling.  An
episode stores template IDs rather than duplicating node payloads, so the
20k/1k/1k workload remains small and can be reconstructed from the same
versioned template JSONL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TEMPLATE_VERSION = "job-template-v0.1"
EPISODE_VERSION = "workload-episode-v0.1"
PRESSURES = (0.50, 0.70, 0.85, 0.95, 1.05)
TOPOLOGIES = ((32760.0, 32760.0), (24576.0, 24576.0), (32760.0, 24576.0))
INITIAL_STATES = ("cold", "hot", "skewed")
ARRIVAL_PATTERNS = ("poisson", "burst", "staggered")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
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


def _text(value: Any, default: str = "unknown") -> str:
    text = str(value or "").strip()
    return text or default


def _number(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) and parsed >= 0.0 else default


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_snapshot(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    selected = [row for row in rows if row.get("selected_for_snapshot", True)]
    if not selected:
        raise ValueError(f"snapshot has no selected rows: {path}")
    run_ids = [str(row.get("run_id")) for row in selected]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("candidate snapshot contains duplicate run_id values")
    return selected


def _event_node(event: Mapping[str, Any], index: int) -> dict[str, Any]:
    runtime = _number(event.get("runtime_ms"))
    peak_reserved = event.get("peak_reserved_mb")
    peak_allocated = event.get("peak_allocated_mb")
    workspace = max(_number(peak_reserved), _number(peak_allocated))
    event_id = _text(event.get("source_event_ids", [None])[0] if event.get("source_event_ids") else None, f"event:{index}")
    parent_step_ids = event.get("parent_step_ids") if isinstance(event.get("parent_step_ids"), list) else []
    return {
        "node_id": event_id,
        "sequence_index": index,
        "source_event_ids": list(event.get("source_event_ids") or [event_id]),
        "event_type": _text(event.get("event_type")),
        "activity": _text(event.get("activity")),
        "raw_action": _text(event.get("raw_action")),
        "node_type": _text(event.get("node_type")),
        "status": _text(event.get("status")),
        "retry_of": event.get("retry_of"),
        "parent_step_ids": parent_step_ids,
        "model_id": event.get("model_id"),
        "runtime_ms": runtime,
        "load_ms": _number(event.get("load_ms")),
        "queue_ms": event.get("queue_ms"),
        "api_wait_ms": _number(event.get("api_wait_ms")),
        "workspace_peak_mb": workspace,
        "resident_model_mb": None,
        "memory_provenance": "resident_model_unknown_workspace_peak_measured",
        "gpu_id": event.get("gpu_id"),
        "gpu_model": event.get("gpu_model"),
        "source_trace_sha256": event.get("source_trace_sha256"),
    }


def compile_templates(snapshot_path: Path, compute_path: Path, output_path: Path) -> dict[str, Any]:
    snapshots = _read_snapshot(snapshot_path)
    selected_ids = {str(row["run_id"]) for row in snapshots}
    compute_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in _read_jsonl(compute_path):
        run_id = _text(event.get("run_id"))
        if run_id in selected_ids:
            # Run-control bookkeeping with no measured work is not a schedulable
            # node; all action/error/retry compute remains in the DAG.
            if event.get("event_type") == "action" or _number(event.get("runtime_ms")) > 0.0:
                compute_by_run[run_id].append(event)
    templates: list[dict[str, Any]] = []
    for snapshot in snapshots:
        run_id = str(snapshot["run_id"])
        events = sorted(compute_by_run.get(run_id, []), key=lambda row: int(row.get("event_index") or 0))
        if not events:
            continue
        nodes = [_event_node(event, index) for index, event in enumerate(events)]
        for index, node in enumerate(nodes):
            if index and not node["parent_step_ids"]:
                node["predecessor_node_ids"] = [nodes[index - 1]["node_id"]]
            else:
                node["predecessor_node_ids"] = []
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
            "status": _text(snapshot.get("status")),
            "validator": _text(snapshot.get("validator")),
            "nodes": nodes,
            "node_count": len(nodes),
            "estimated_runtime_ms": sum(_number(node.get("runtime_ms")) for node in nodes),
            "estimated_peak_workspace_mb": max((_number(node.get("workspace_peak_mb")) for node in nodes), default=0.0),
            "derivation_version": TEMPLATE_VERSION,
            "value_source": "measured_trace_events",
        })
    if not templates:
        raise ValueError("no job templates could be compiled")
    _write_jsonl(output_path, templates)
    summary = {
        "schema_version": TEMPLATE_VERSION,
        "snapshot": str(snapshot_path),
        "compute_events": str(compute_path),
        "output": str(output_path),
        "templates": len(templates),
        "videos": len({row["video_id"] for row in templates}),
        "nodes": sum(int(row["node_count"]) for row in templates),
        "split_counts": dict(sorted(Counter(row["split"] for row in templates).items())),
        "baseline_counts": dict(sorted(Counter(row["baseline"] for row in templates).items())),
        "source_snapshot_sha256": _sha256(snapshot_path),
        "source_compute_sha256": _sha256(compute_path),
    }
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _sample_arrivals(rng: random.Random, count: int, pattern: str, pressure: float, base_ms: float) -> list[float]:
    gap = max(1.0, base_ms / max(pressure, 0.01))
    arrivals: list[float] = []
    now = 0.0
    for index in range(count):
        if pattern == "burst":
            increment = 0.0 if index < max(1, count // 4) else gap
        elif pattern == "staggered":
            increment = gap * (0.5 + 0.5 * (index % 3) / 2.0)
        else:
            increment = rng.expovariate(1.0 / gap)
        now += increment
        arrivals.append(round(now, 3))
    return arrivals


def _initial_residency(templates: Sequence[Mapping[str, Any]], state: str, capacity_count: int) -> list[list[str]]:
    models = sorted({
        _text(node.get("model_id"))
        for template in templates
        for node in (template.get("nodes") or [])
        if node.get("model_id")
    })
    if state == "cold" or not models:
        return [[] for _ in range(capacity_count)]
    if state == "hot":
        return [models[:] for _ in range(capacity_count)]
    return [models[:1]] + [[] for _ in range(max(0, capacity_count - 1))]


def generate_episodes(
    template_path: Path,
    output_path: Path,
    *,
    split: str,
    count: int,
    seed: int,
    jobs_per_episode: Sequence[int] = (16, 32, 64),
) -> dict[str, Any]:
    templates = [row for row in _read_jsonl(template_path) if _text(row.get("split")) == split]
    if not templates:
        raise ValueError(f"no templates for split={split}")
    rng = random.Random(seed)
    episodes: list[dict[str, Any]] = []
    for episode_index in range(count):
        pressure = PRESSURES[episode_index % len(PRESSURES)]
        pattern = ARRIVAL_PATTERNS[(episode_index // len(PRESSURES)) % len(ARRIVAL_PATTERNS)]
        initial_state = INITIAL_STATES[(episode_index // (len(PRESSURES) * len(ARRIVAL_PATTERNS))) % len(INITIAL_STATES)]
        topology = TOPOLOGIES[episode_index % len(TOPOLOGIES)]
        job_count = jobs_per_episode[episode_index % len(jobs_per_episode)]
        selected = [templates[rng.randrange(len(templates))] for _ in range(job_count)]
        base_ms = statistics.median([max(1.0, _number(row.get("estimated_runtime_ms"))) for row in selected])
        arrivals = _sample_arrivals(rng, job_count, pattern, pressure, base_ms)
        deadline_multiplier = (1.5, 2.0, 3.0)[episode_index % 3]
        jobs = [
            {
                "job_instance_id": f"episode_{split}_{episode_index:06d}_job_{index:03d}",
                "template_id": row["template_id"],
                "arrival_ms": arrivals[index],
                "deadline_ms": arrivals[index] + max(1.0, _number(row.get("estimated_runtime_ms"))) * deadline_multiplier,
            }
            for index, row in enumerate(selected)
        ]
        episodes.append({
            "schema_version": EPISODE_VERSION,
            "episode_id": f"{split}_{episode_index:06d}",
            "split": split,
            "seed": seed,
            "pressure": pressure,
            "arrival_pattern": pattern,
            "initial_state": initial_state,
            "gpu_topology_mb": list(topology),
            "deadline_multiplier": deadline_multiplier,
            "jobs": jobs,
            "source_template_ids": sorted({row["template_id"] for row in selected}),
            "source_video_ids": sorted({row["video_id"] for row in selected}),
            "source_trace_sha256": sorted({row["source_trace_sha256"] for row in selected if row.get("source_trace_sha256")}),
            "initial_residency_hint": _initial_residency(selected, initial_state, len(topology)),
            "derivation_version": EPISODE_VERSION,
            "value_source": "measured_template_sampling_plus_synthetic_arrivals",
        })
    _write_jsonl(output_path, episodes)
    template_split = {str(row["template_id"]): _text(row.get("split")) for row in _read_jsonl(template_path)}
    isolation_errors: list[str] = []
    for episode in episodes:
        for job in episode["jobs"]:
            template_id = str(job["template_id"])
            if template_split.get(template_id) != split:
                isolation_errors.append(f"{episode['episode_id']}:{template_id}")
    summary = {
        "schema_version": EPISODE_VERSION,
        "output": str(output_path),
        "split": split,
        "episodes": len(episodes),
        "jobs": sum(len(row["jobs"]) for row in episodes),
        "pressure_counts": dict(sorted(Counter(row["pressure"] for row in episodes).items())),
        "arrival_pattern_counts": dict(sorted(Counter(row["arrival_pattern"] for row in episodes).items())),
        "initial_state_counts": dict(sorted(Counter(row["initial_state"] for row in episodes).items())),
        "gpu_topology_counts": {
            ",".join(str(value) for value in topology): count
            for topology, count in sorted(Counter(tuple(row["gpu_topology_mb"]) for row in episodes).items())
        },
        "split_isolation_errors": isolation_errors,
        "split_isolation_gate": not isolation_errors,
    }
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--compute", type=Path)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--episodes", type=Path)
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    if args.snapshot and args.compute:
        print(json.dumps(compile_templates(args.snapshot, args.compute, args.templates), ensure_ascii=False, sort_keys=True))
    if args.split:
        if args.episodes is None or args.count <= 0:
            raise ValueError("--split requires --episodes and positive --count")
        print(json.dumps(generate_episodes(args.templates, args.episodes, split=args.split, count=args.count, seed=args.seed), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
