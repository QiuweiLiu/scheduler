#!/usr/bin/env python3
"""Replay saved Video Agent traces as a deterministic discrete-event scheduler.

This is the first Phase 4 gate.  It does not launch a second GPU and it does
not invent future actions: each job replays the action sequence observed in a
successful trace.  The simulator separates measured action compute time from
the measured cold-load field, tracks per-GPU model residency and compares
simple policies with an offline look-ahead Oracle.  The ``predictive`` policy
uses only baseline/last-action transition statistics from *other videos* for
each job, so the counterfactual comparison still has a video-level leakage
boundary.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracing.collectors.structured_state import canonical_action  # noqa: E402


POLICIES = ("round_robin", "least_loaded", "myopic", "static_template", "predictive", "oracle")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _number(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _text(value: Any, default: str = "unknown") -> str:
    text = str(value or "").strip()
    return text or default


@dataclass(frozen=True)
class Action:
    raw_name: str
    canonical: str
    model: str
    compute_ms: float
    load_ms: float
    peak_vram_mb: float


@dataclass(frozen=True)
class JobTemplate:
    job_id: str
    video_id: str
    baseline: str
    actions: tuple[Action, ...]


@dataclass
class JobState:
    template: JobTemplate
    arrival_ms: float
    next_index: int = 0
    running: bool = False
    completed_at: float | None = None
    assigned_gpus: list[int] = field(default_factory=list)
    load_overhead_ms: float = 0.0


@dataclass
class GPUState:
    index: int
    capacity_mb: float
    busy_until: float = 0.0
    busy_time_ms: float = 0.0
    load_time_ms: float = 0.0
    resident: dict[str, float] = field(default_factory=dict)
    peak_resident_mb: float = 0.0
    evictions: int = 0


def _model_key(event: Mapping[str, Any], action: str) -> str:
    input_data = event.get("input") or {}
    if isinstance(input_data, Mapping):
        replacement = input_data.get("replacement_model_id")
        if replacement:
            return _text(replacement)
    return _text(event.get("model_id"), canonical_action(action))


def load_jobs(root: Path) -> list[JobTemplate]:
    jobs: list[JobTemplate] = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        status_path = run_dir / "run_status.json"
        trace_path = run_dir / "trace.jsonl"
        if not status_path.is_file() or not trace_path.is_file():
            continue
        status = _read_json(status_path)
        if str(status.get("status")) != "success":
            continue
        manifest_path = run_dir / "run_manifest.json"
        manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
        events = _read_jsonl(trace_path)
        actions: list[Action] = []
        for event in events:
            if event.get("event_type") != "action":
                continue
            raw_name = _text(event.get("action"), "other")
            resource = event.get("resource") or {}
            runtime = max(0.1, _number(resource.get("runtime_ms"), 0.1))
            measured_load = max(0.0, _number(resource.get("load_ms")))
            peak = max(_number(resource.get("peak_reserved_mb")), _number(resource.get("peak_allocated_mb")))
            # Trace runtime includes the measured load.  The simulator adds a
            # cold load only when a model is absent on the chosen GPU.
            compute = max(0.1, runtime - measured_load)
            actions.append(
                Action(
                    raw_name=raw_name,
                    canonical=canonical_action(raw_name),
                    model=_model_key(event, raw_name),
                    compute_ms=compute,
                    load_ms=measured_load,
                    peak_vram_mb=max(0.0, peak),
                )
            )
        if not actions:
            continue
        task_id = _text(manifest.get("task_id"), run_dir.name)
        # The run id is unique but the video id is used only as a split group,
        # never as a policy feature.  Read it from the manifest because the
        # ``langgraph_react`` baseline itself contains an underscore.
        video_path = str(manifest.get("video_path", ""))
        video_id = Path(video_path).stem if video_path else ""
        if not video_id:
            video_id = task_id
        baseline = _text(manifest.get("baseline"), "unknown")
        jobs.append(JobTemplate(task_id + "@" + run_dir.name, video_id, baseline, tuple(actions)))
    if not jobs:
        raise ValueError(f"no successful action traces found under {root}")
    return jobs


def cold_load_priors(jobs: Sequence[JobTemplate]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for job in jobs:
        for action in job.actions:
            if action.load_ms > 0.0:
                values[action.model].append(action.load_ms)
    return {model: statistics.median(samples) for model, samples in values.items()}


def transition_priors(jobs: Sequence[JobTemplate]) -> dict[tuple[str, str], Counter[str]]:
    result: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for job in jobs:
        for previous, current in zip((None,) + tuple(action.canonical for action in job.actions), job.actions):
            key = (job.baseline, previous or "START")
            result[key][current.model] += 1
    return result


def _model_memory(action: Action) -> float:
    return max(1.0, action.peak_vram_mb)


def _can_fit(gpu: GPUState, action: Action) -> bool:
    if action.model in gpu.resident:
        return True
    return sum(gpu.resident.values()) + _model_memory(action) <= gpu.capacity_mb


def _project_action_cost(gpu: GPUState, action: Action, priors: Mapping[str, float]) -> tuple[float, bool]:
    if action.model in gpu.resident:
        return action.compute_ms, False
    return action.compute_ms + max(action.load_ms, _number(priors.get(action.model))), True


def _ensure_model(gpu: GPUState, action: Action, priors: Mapping[str, float]) -> tuple[float, bool]:
    if action.model in gpu.resident:
        return 0.0, False
    memory = _model_memory(action)
    if sum(gpu.resident.values()) + memory > gpu.capacity_mb:
        gpu.resident.clear()
        gpu.evictions += 1
    if memory > gpu.capacity_mb:
        raise RuntimeError(f"action model {action.model} requires {memory:.1f}MB > GPU {gpu.index} capacity {gpu.capacity_mb:.1f}MB")
    gpu.resident[action.model] = memory
    load = max(action.load_ms, _number(priors.get(action.model)))
    gpu.load_time_ms += load
    gpu.peak_resident_mb = max(gpu.peak_resident_mb, sum(gpu.resident.values()))
    return load, True


def _policy_gpu(
    policy: str,
    job: JobState,
    gpus: Sequence[GPUState],
    free_indices: Sequence[int],
    priors: Mapping[str, float],
    transitions: Mapping[tuple[str, str], Counter[str]],
    round_robin_cursor: int,
) -> int:
    action = job.template.actions[job.next_index]
    candidates = [gpus[index] for index in free_indices]
    fitting = [gpu for gpu in candidates if _can_fit(gpu, action) or _model_memory(action) <= gpu.capacity_mb]
    if not fitting:
        fitting = candidates
    if policy == "round_robin":
        return min(fitting, key=lambda gpu: ((gpu.index - round_robin_cursor) % len(gpus), gpu.index)).index
    if policy == "static_template":
        return min(fitting, key=lambda gpu: gpu.index).index
    if policy in {"least_loaded", "myopic"}:
        def current_score(gpu: GPUState) -> tuple[float, int]:
            cost, _ = _project_action_cost(gpu, action, priors)
            return (gpu.busy_until + (cost if policy == "myopic" else 0.0), gpu.index)
        return min(fitting, key=current_score).index
    if policy == "oracle":
        def lookahead(gpu: GPUState) -> tuple[float, int]:
            resident = dict(gpu.resident)
            total = 0.0
            for future in job.template.actions[job.next_index :]:
                if future.model not in resident:
                    resident.clear() if sum(resident.values()) + _model_memory(future) > gpu.capacity_mb else None
                    resident[future.model] = _model_memory(future)
                    total += max(future.load_ms, _number(priors.get(future.model)))
                total += future.compute_ms
            return (gpu.busy_until + total, gpu.index)
        return min(fitting, key=lookahead).index
    if policy == "predictive":
        history = tuple((job.template.actions[index].canonical for index in range(job.next_index)))
        key = (job.template.baseline, history[-1] if history else "START")
        counts = transitions.get(key) or Counter()
        total = sum(counts.values())
        if total <= 0:
            counts = Counter({action.model: 1})
            total = 1
        def predictive_score(gpu: GPUState) -> tuple[float, int]:
            current, _ = _project_action_cost(gpu, action, priors)
            expected_next_load = 0.0
            for model, count in counts.items():
                probability = count / total
                if model not in gpu.resident and model != action.model:
                    expected_next_load += probability * _number(priors.get(model))
            return (gpu.busy_until + current + expected_next_load, gpu.index)
        return min(fitting, key=predictive_score).index
    raise ValueError(f"unknown policy: {policy}")


def simulate(
    jobs: Sequence[JobTemplate],
    policy: str,
    capacities: Sequence[float],
    seed: int,
    arrival_interval_ms: float,
) -> dict[str, Any]:
    rng = random.Random(seed)
    ordered = list(jobs)
    rng.shuffle(ordered)
    states = [JobState(job, index * max(0.0, arrival_interval_ms)) for index, job in enumerate(ordered)]
    gpus = [GPUState(index, float(capacity)) for index, capacity in enumerate(capacities)]
    priors = cold_load_priors(jobs)
    transitions = transition_priors([job for job in jobs if job.video_id not in {state.template.video_id for state in states}])
    # The set above is empty if written literally; build an all-job prior for
    # the shared policy, then replace it per job in the scheduling branch.
    all_transitions = transition_priors(jobs)
    arrival_index = 0
    ready: list[JobState] = []
    events: list[tuple[float, int, int, int]] = []
    sequence = 0
    now = 0.0
    rr_cursor = 0

    def release_arrivals() -> None:
        nonlocal arrival_index
        while arrival_index < len(states) and states[arrival_index].arrival_ms <= now + 1e-9:
            ready.append(states[arrival_index])
            arrival_index += 1

    release_arrivals()
    while ready or events or arrival_index < len(states):
        # Complete every action ending at the current event time.
        while events and events[0][0] <= now + 1e-9:
            finish, _order, gpu_index, state_index = heapq.heappop(events)
            state = states[state_index]
            state.running = False
            state.next_index += 1
            if state.next_index >= len(state.template.actions):
                state.completed_at = finish
            else:
                ready.append(state)
        release_arrivals()
        scheduled = False
        while ready:
            free = [gpu.index for gpu in gpus if gpu.busy_until <= now + 1e-9]
            if not free:
                break
            state = ready.pop(0)
            job_transitions = transition_priors([job for job in jobs if job.video_id != state.template.video_id])
            gpu_index = _policy_gpu(policy, state, gpus, free, priors, job_transitions or all_transitions, rr_cursor)
            gpu = gpus[gpu_index]
            action = state.template.actions[state.next_index]
            load, _loaded = _ensure_model(gpu, action, priors)
            duration = action.compute_ms + load
            start = max(now, gpu.busy_until, state.arrival_ms)
            finish = start + duration
            gpu.busy_until = finish
            gpu.busy_time_ms += duration
            state.running = True
            state.assigned_gpus.append(gpu_index)
            state.load_overhead_ms += load
            sequence += 1
            heapq.heappush(events, (finish, sequence, gpu_index, states.index(state)))
            if policy == "round_robin":
                rr_cursor = (gpu_index + 1) % len(gpus)
            scheduled = True
        if scheduled:
            continue
        next_finish = events[0][0] if events else float("inf")
        next_arrival = states[arrival_index].arrival_ms if arrival_index < len(states) else float("inf")
        next_time = min(next_finish, next_arrival)
        if not math.isfinite(next_time):
            break
        now = max(now, next_time)

    completions = [state.completed_at - state.arrival_ms for state in states if state.completed_at is not None]
    makespan = max((state.completed_at or 0.0) for state in states) if states else 0.0
    mean_completion = statistics.fmean(completions) if completions else 0.0
    sorted_completion = sorted(completions)
    def percentile(value: float) -> float:
        if not sorted_completion:
            return 0.0
        position = min(len(sorted_completion) - 1, max(0, math.ceil(value * len(sorted_completion)) - 1))
        return sorted_completion[position]
    total_load = sum(state.load_overhead_ms for state in states)
    total_actions = sum(len(state.template.actions) for state in states)
    return {
        "policy": policy,
        "seed": seed,
        "jobs": len(states),
        "actions": total_actions,
        "capacities_mb": list(capacities),
        "mean_completion_ms": mean_completion,
        "p95_completion_ms": percentile(0.95),
        "p99_completion_ms": percentile(0.99),
        "makespan_ms": makespan,
        "throughput_jobs_per_s": len(states) / (makespan / 1000.0) if makespan > 0 else 0.0,
        "total_load_overhead_ms": total_load,
        "gpu_utilization_mean": statistics.fmean(gpu.busy_time_ms / makespan for gpu in gpus) if makespan > 0 else 0.0,
        "gpu_peak_memory_utilization_mean": statistics.fmean((gpu.peak_resident_mb / gpu.capacity_mb) for gpu in gpus),
        "gpu_evictions": sum(gpu.evictions for gpu in gpus),
        "completed_jobs": len(completions),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy"])].append(row)
    output: list[dict[str, Any]] = []
    numeric = (
        "mean_completion_ms",
        "p95_completion_ms",
        "p99_completion_ms",
        "makespan_ms",
        "throughput_jobs_per_s",
        "total_load_overhead_ms",
        "gpu_utilization_mean",
        "gpu_peak_memory_utilization_mean",
        "gpu_evictions",
    )
    for policy, values in sorted(grouped.items()):
        result: dict[str, Any] = {"policy": policy, "scenarios": len(values)}
        for key in numeric:
            result[key] = statistics.fmean(_number(value.get(key)) for value in values)
        result["completed_jobs"] = min(_number(value.get("completed_jobs")) for value in values)
        output.append(result)
    oracle = next((row for row in output if row["policy"] == "oracle"), None)
    if oracle:
        oracle_mean = max(oracle["mean_completion_ms"], 1e-9)
        for row in output:
            row["relative_gap_to_oracle"] = (row["mean_completion_ms"] - oracle_mean) / oracle_mean
    return output


def run_experiment(jobs: Sequence[JobTemplate], capacities: Sequence[float], seeds: Sequence[int], arrival_interval_ms: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for policy in POLICIES:
            rows.append(simulate(jobs, policy, capacities, seed, arrival_interval_ms))
    aggregate = _aggregate(rows)
    myopic = next((row for row in aggregate if row["policy"] == "myopic"), None)
    oracle = next((row for row in aggregate if row["policy"] == "oracle"), None)
    oracle_gap = None
    if myopic and oracle:
        oracle_gap = {
            "myopic_minus_oracle_mean_completion_ms": myopic["mean_completion_ms"] - oracle["mean_completion_ms"],
            "myopic_relative_gap": myopic["relative_gap_to_oracle"],
            "interpretation": "future_information_has_counterfactual_value" if myopic["mean_completion_ms"] > oracle["mean_completion_ms"] * 1.01 else "oracle_gap_small",
        }
    return {
        "schema_version": "phase4-replay-0.1",
        "task": "offline discrete-event replay of measured Video Agent traces",
        "data": {
            "runs": len(jobs),
            "videos": len({job.video_id for job in jobs}),
            "baselines": dict(sorted(Counter(job.baseline for job in jobs).items())),
            "actions": sum(len(job.actions) for job in jobs),
            "models": dict(sorted(Counter(action.model for job in jobs for action in job.actions).items())),
        },
        "assumptions": {
            "trace_replay": "action sequences are fixed to observed successful traces; no synthetic future action is created",
            "cold_load": "measured load_ms is subtracted from action runtime and re-added only when model is absent on the chosen GPU; median positive load is fallback prior",
            "gpu_capacities_mb": list(capacities),
            "arrival_interval_ms": arrival_interval_ms,
            "predictive_leakage_control": "transition prior for a job excludes all runs sharing that job's video_id",
            "oracle": "full remaining action sequence is available only to the oracle look-ahead scorer",
            "real_gpu_claim": False,
        },
        "scenarios": rows,
        "aggregate": aggregate,
        "oracle_gap": oracle_gap,
        "phase4_gate": (
            "proceed_to_real_multigpu_calibration" if oracle_gap and oracle_gap["interpretation"] == "future_information_has_counterfactual_value" else "stop_or_refine_workload"
        ),
        "limitations": [
            "This is a counterfactual replay on one-GPU traces, not a measurement on two physical GPUs.",
            "The trace fixes the action sequence; it tests scheduling value conditional on observed paths, not policy-induced path changes.",
            "Model unload cost and inter-GPU transfer are not measured in the current traces and are not fabricated.",
        ],
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--capacities", default="32760,24576", help="comma-separated GPU capacities in MiB")
    parser.add_argument("--seeds", default="0,1,2,3,4", help="comma-separated deterministic scenario seeds")
    parser.add_argument("--arrival-interval-ms", type=float, default=0.0)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    capacities = [float(item) for item in args.capacities.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    if not capacities or not seeds:
        raise ValueError("capacities and seeds must not be empty")
    jobs = load_jobs(args.root.expanduser().resolve())
    report = run_experiment(jobs, capacities, seeds, max(0.0, args.arrival_interval_ms))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(args.csv, report["aggregate"])
    print(json.dumps({
        "runs": report["data"]["runs"],
        "videos": report["data"]["videos"],
        "policies": len(report["aggregate"]),
        "oracle_gap": report["oracle_gap"],
        "phase4_gate": report["phase4_gate"],
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
