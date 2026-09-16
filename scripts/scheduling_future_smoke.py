#!/usr/bin/env python3
"""Node-level rolling-horizon scheduler smoke for the frozen E0--E3 artifacts.

The simulator is intentionally small and auditable.  Decisions read only the
resource artifact plus one of the predicted/true finite-horizon providers;
template runtime, load, and memory are hidden execution truth.
"""

from __future__ import annotations

import argparse
import gzip
import heapq
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


POLICIES = ("pred_h0", "pred_h1", "pred_h3", "pred_h5", "true_h1", "true_h3", "true_h5")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


@dataclass
class Job:
    instance_id: str
    template: dict[str, Any]
    arrival_ms: float
    deadline_ms: float | None
    service_class: str
    state: dict[str, str]
    ready_since: dict[str, float] = field(default_factory=dict)
    completed: set[str] = field(default_factory=set)
    failed: set[str] = field(default_factory=set)
    finish_ms: float | None = None
    queue_ms: float = 0.0


@dataclass
class GPU:
    index: int
    capacity_mb: float
    active: tuple[int, str] | None = None
    resident: dict[str, float] = field(default_factory=dict)
    busy_time_ms: float = 0.0
    peak_mb: float = 0.0
    evictions: int = 0


def template_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in rows:
        nodes = {str(node["node_id"]): dict(node) for node in row.get("nodes") or []}
        successors: dict[str, list[str]] = defaultdict(list)
        for node in nodes.values():
            for predecessor in node.get("predecessor_node_ids") or []:
                successors[str(predecessor)].append(str(node["node_id"]))
        for node_id, node in nodes.items():
            node["_preds"] = tuple(str(value) for value in node.get("predecessor_node_ids") or [])
            node["_succs"] = tuple(sorted(successors.get(node_id, [])))
        row = dict(row)
        row["_nodes"] = nodes
        out[str(row["template_id"])] = row
    return out


def load_artifacts(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    pred_resource = {
        str(row["node_id"]): row
        for row in read_gzip_jsonl(root / "prediction_artifacts/resource_predictions_v1.jsonl.gz")
    }
    pred_behavior: dict[str, dict[str, Any]] = {}
    for filename in ("b05_node_h1.jsonl.gz", "b05_future_h3.jsonl.gz", "b05_future_h5.jsonl.gz"):
        path = root / "prediction_artifacts" / filename
        if not path.exists():
            continue
        for row in read_gzip_jsonl(path):
            node_id = str(row["node_id"])
            pred_behavior.setdefault(node_id, {}).update({key: value for key, value in row.items() if value is not None})
    family_values: dict[str, list[float]] = defaultdict(list)
    for row in pred_resource.values():
        family_values[str(row.get("activity_family") or "other")].append(finite(row.get("runtime_p50_ms")))
    family_medians = {key: statistics.median(values) for key, values in family_values.items() if values}
    return pred_resource, pred_behavior, {"family_medians": family_medians}


def make_jobs(episode: dict[str, Any], templates: dict[str, dict[str, Any]]) -> list[Job]:
    jobs = []
    for raw in episode.get("jobs") or []:
        template = templates[str(raw["template_id"])]
        if str(template.get("split")) != str(episode.get("split")):
            raise ValueError(f"split leak: {episode['episode_id']} -> {raw['template_id']}")
        jobs.append(Job(
            instance_id=str(raw["job_instance_id"]),
            template=template,
            arrival_ms=finite(raw.get("arrival_ms")),
            deadline_ms=(finite(raw.get("deadline_ms")) if raw.get("deadline_ms") is not None else None),
            service_class=str(raw.get("service_class") or "normal"),
            state={node_id: "pending" for node_id in template["_nodes"]},
        ))
    return jobs


def ready_nodes(job: Job, now: float) -> list[str]:
    if now + 1e-9 < job.arrival_ms:
        return []
    return [
        node_id for node_id, state in job.state.items()
        if state == "pending"
        and all(job.state.get(pred) == "complete" for pred in job.template["_nodes"][node_id]["_preds"])
    ]


def predicted_step_cost(step: dict[str, Any], family_medians: dict[str, float]) -> float:
    family = step.get("action_family")
    if family and family in family_medians:
        return family_medians[family]
    role = str(step.get("role") or "other")
    return family_medians.get(role, statistics.median(list(family_medians.values()) or [1.0]))


def actual_future_cost(node_id: str, node_map: dict[str, dict[str, Any]], resource: dict[str, dict[str, Any]], horizon: int) -> float:
    frontier = list(node_map[node_id].get("_succs") or [])
    seen: set[str] = set()
    total = 0.0
    for _ in range(horizon):
        next_frontier: list[str] = []
        for current in frontier:
            if current in seen:
                continue
            seen.add(current)
            total += finite(resource.get(current, {}).get("runtime_p50_ms"), 0.0)
            next_frontier.extend(node_map[current].get("_succs") or [])
        frontier = next_frontier
        if not frontier:
            break
    return total


def behavior_future_cost(node_id: str, behavior: dict[str, dict[str, Any]], family_medians: dict[str, float], horizon: int) -> float:
    row = behavior.get(node_id, {})
    scenarios = row.get(f"future_h{horizon}") or []
    total = 0.0
    for scenario in scenarios:
        steps = scenario.get("steps") or []
        total += finite(scenario.get("scenario_probability"), 0.0) * sum(predicted_step_cost(step, family_medians) for step in steps)
    return total


def dispatch_score(
    policy: str,
    job: Job,
    node: dict[str, Any],
    now: float,
    resource: dict[str, dict[str, Any]],
    behavior: dict[str, dict[str, Any]],
    family_medians: dict[str, float],
) -> tuple[Any, ...]:
    node_id = str(node["node_id"])
    current = resource.get(node_id) or {}
    current_cost = finite(current.get("runtime_p50_ms"), 1.0)
    if policy == "pred_h0":
        future = 0.0
    elif policy.startswith("pred_h"):
        future = behavior_future_cost(node_id, behavior, family_medians, int(policy[-1]))
    else:
        future = actual_future_cost(node_id, job.template["_nodes"], resource, int(policy[-1]))
    total = current_cost + future
    slack = (job.deadline_ms - now - total) if job.deadline_ms is not None else float("inf")
    priority = 0 if job.service_class == "priority" else 1
    return (priority, slack, future, job.ready_since.get(node_id, now), node_id)


def simulate(episode: dict[str, Any], templates: dict[str, dict[str, Any]], policy: str, resource: dict[str, dict[str, Any]], behavior: dict[str, dict[str, Any]], family_medians: dict[str, float]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    jobs = make_jobs(episode, templates)
    gpus = [GPU(index, finite(capacity)) for index, capacity in enumerate(episode.get("gpu_topology_mb") or [])]
    arrivals = [(job.arrival_ms, index) for index, job in enumerate(jobs)]
    heapq.heapify(arrivals)
    finishes: list[tuple[float, int, int, str, int | None]] = []
    now = 0.0
    order = 0
    events: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []

    def emit(kind: str, job: Job, node: dict[str, Any] | None = None, **extra: Any) -> None:
        item = {"schema_version": "scheduling-future-smoke-event-v1", "episode_id": episode["episode_id"], "policy": policy, "event_type": kind, "time_ms": round(now, 3), "job_instance_id": job.instance_id}
        if node is not None:
            item["node_id"] = node["node_id"]
        item.update(extra)
        events.append(item)

    def release(job_index: int) -> None:
        job = jobs[job_index]
        for node_id in ready_nodes(job, now):
            job.state[node_id] = "ready"
            job.ready_since[node_id] = now
            emit("node_ready", job, job.template["_nodes"][node_id])

    def finish_ready() -> None:
        nonlocal now
        while finishes and finishes[0][0] <= now + 1e-9:
            finish, _order, job_index, node_id, gpu_index = heapq.heappop(finishes)
            job = jobs[job_index]
            node = job.template["_nodes"][node_id]
            job.state[node_id] = "complete"
            job.completed.add(node_id)
            if gpu_index is not None:
                gpus[gpu_index].active = None
            emit("node_finish", job, node, gpu_index=gpu_index, finish_ms=round(finish, 3))
            release(job_index)
            if len(job.completed) + len(job.failed) == len(job.state) and job.finish_ms is None:
                job.finish_ms = now
                emit("job_finish", job, deadline_met=(job.deadline_ms is None or now <= job.deadline_ms))

    while arrivals or finishes or any(any(value == "ready" for value in job.state.values()) for job in jobs):
        if not finishes and not arrivals:
            if not any(any(value == "ready" for value in job.state.values()) for job in jobs):
                break
        if arrivals and (not finishes or arrivals[0][0] <= finishes[0][0]) and not any(any(value == "ready" for value in job.state.values()) for job in jobs):
            now = max(now, arrivals[0][0])
        while arrivals and arrivals[0][0] <= now + 1e-9:
            _arrival, job_index = heapq.heappop(arrivals)
            emit("job_arrive", jobs[job_index], arrival_ms=jobs[job_index].arrival_ms)
            release(job_index)
        finish_ready()

        progress = True
        while progress:
            progress = False
            for job_index, job in enumerate(jobs):
                for node_id in ready_nodes(job, now):
                    if job.state[node_id] == "pending":
                        job.state[node_id] = "ready"
                        job.ready_since[node_id] = now
            # API/CPU nodes can overlap in this smoke model.
            for job_index, job in enumerate(jobs):
                for node_id, state in list(job.state.items()):
                    if state != "ready":
                        continue
                    node = job.template["_nodes"][node_id]
                    if str(node.get("execution_lane")) in {"gpu", "cuda"}:
                        continue
                    job.state[node_id] = "running"
                    job.queue_ms += max(0.0, now - job.ready_since.get(node_id, now))
                    order += 1
                    heapq.heappush(finishes, (now + max(0.1, finite(node.get("runtime_ms"), 0.1)), order, job_index, node_id, None))
                    emit("node_start", job, node, gpu_index=None, queue_ms=round(max(0.0, now - job.ready_since.get(node_id, now)), 3))
                    progress = True

            free = [gpu for gpu in gpus if gpu.active is None]
            candidates = [(job_index, node_id, job, job.template["_nodes"][node_id]) for job_index, job in enumerate(jobs) for node_id, state in job.state.items() if state == "ready" and str(job.template["_nodes"][node_id].get("execution_lane")) in {"gpu", "cuda"}]
            while free and candidates:
                candidates.sort(key=lambda item: dispatch_score(policy, item[2], item[3], now, resource, behavior, family_medians))
                job_index, node_id, job, node = candidates.pop(0)
                gpu = free.pop(0)
                pred = resource.get(node_id) or {}
                predicted_memory = finite(pred.get("peak_memory_p95_mb"), 0.0)
                resident = finite(node.get("resident_model_mb"), 0.0)
                actual_workspace = max(0.0, finite(node.get("workspace_peak_mb"), resident) - resident)
                current_resident = sum(gpu.resident.values())
                if current_resident + predicted_memory > gpu.capacity_mb:
                    gpu.resident.clear()
                    gpu.evictions += 1
                    current_resident = 0.0
                if resident + actual_workspace > gpu.capacity_mb:
                    job.state[node_id] = "failed"
                    job.failed.add(node_id)
                    emit("node_fail", job, node, reason="truth_oom", gpu_index=gpu.index)
                    progress = True
                    continue
                load = finite(node.get("load_ms"), finite(pred.get("load_p50_ms"), 0.0)) if node.get("model_id") not in gpu.resident else 0.0
                gpu.resident[str(node.get("model_id"))] = resident
                gpu.peak_mb = max(gpu.peak_mb, sum(gpu.resident.values()) + actual_workspace)
                duration = max(0.1, finite(node.get("runtime_ms"), 0.1))
                job.state[node_id] = "running"
                job.queue_ms += max(0.0, now - job.ready_since.get(node_id, now))
                gpu.active = (job_index, node_id)
                gpu.busy_time_ms += duration
                order += 1
                heapq.heappush(finishes, (now + duration, order, job_index, node_id, gpu.index))
                score = dispatch_score(policy, job, node, now, resource, behavior, family_medians)
                decision_rows.append({"episode_id": episode["episode_id"], "policy": policy, "time_ms": round(now, 3), "job_instance_id": job.instance_id, "node_id": node_id, "gpu_index": gpu.index, "score": [str(value) if value == float("inf") else value for value in score], "predicted_runtime_p50_ms": finite(pred.get("runtime_p50_ms")), "predicted_memory_p95_mb": predicted_memory})
                emit("node_dispatch", job, node, gpu_index=gpu.index, predicted_runtime_p50_ms=finite(pred.get("runtime_p50_ms")), predicted_memory_p95_mb=predicted_memory, truth_runtime_ms=duration)
                progress = True
            finish_ready()

        if finishes:
            next_finish = finishes[0][0]
        else:
            next_finish = float("inf")
        next_arrival = arrivals[0][0] if arrivals else float("inf")
        if next_finish == float("inf") and next_arrival == float("inf"):
            break
        now = max(now, min(next_finish, next_arrival))

    durations = [job.finish_ms - job.arrival_ms for job in jobs if job.finish_ms is not None and not job.failed]
    makespan = max((job.finish_ms or 0.0 for job in jobs), default=0.0)
    summary = {
        "schema_version": "scheduling-future-smoke-result-v1",
        "episode_id": episode["episode_id"],
        "policy": policy,
        "jobs": len(jobs),
        "completed_jobs": sum(job.finish_ms is not None and not job.failed for job in jobs),
        "failed_jobs": sum(bool(job.failed) for job in jobs),
        "mean_completion_ms": statistics.fmean(durations) if durations else None,
        "p95_completion_ms": sorted(durations)[max(0, math.ceil(len(durations) * 0.95) - 1)] if durations else None,
        "makespan_ms": makespan,
        "mean_job_queue_ms": statistics.fmean(job.queue_ms for job in jobs) if jobs else 0.0,
        "deadline_miss_rate": statistics.fmean(1.0 if job.finish_ms is None or (job.deadline_ms is not None and job.finish_ms > job.deadline_ms) else 0.0 for job in jobs) if jobs else 0.0,
        "gpu_peak_memory_mb": [gpu.peak_mb for gpu in gpus],
        "gpu_evictions": sum(gpu.evictions for gpu in gpus),
        "gpu_utilization": [gpu.busy_time_ms / makespan if makespan else 0.0 for gpu in gpus],
        "decision_count": len(decision_rows),
    }
    return summary, events + [{"event_type": "decision", **row} for row in decision_rows]


def validate_events(summary: dict[str, Any], events: list[dict[str, Any]], templates: dict[str, dict[str, Any]], episodes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    state: dict[tuple[str, str, str], str] = {}
    violations: list[str] = []
    nodes_by_id = {node_id: node for template in templates.values() for node_id, node in template["_nodes"].items()}
    arrivals = {episode_id: {job["job_instance_id"]: finite(job.get("arrival_ms")) for job in episode.get("jobs") or []} for episode_id, episode in episodes.items()}
    for event in events:
        key = (str(event["episode_id"]), str(event.get("job_instance_id") or ""), str(event.get("node_id") or ""))
        kind = event["event_type"]
        node_id = str(event.get("node_id") or "")
        if kind == "node_ready":
            for predecessor in nodes_by_id.get(node_id, {}).get("_preds") or ():
                predecessor_key = (key[0], key[1], str(predecessor))
                if state.get(predecessor_key) != "complete":
                    violations.append(f"ready before predecessor complete: {key} <- {predecessor_key}")
            state[key] = "ready"
        if kind == "node_start":
            state[key] = "running"
        if kind == "node_dispatch":
            previous = state.get(key)
            if previous not in {"ready", None}:
                violations.append(f"dispatch after {previous}: {key}")
            state[key] = "running"
        elif kind == "node_finish":
            if state.get(key) not in {"running", None}:
                violations.append(f"finish without running: {key}")
            state[key] = "complete"
        elif kind == "node_fail":
            state[key] = "failed"
        if kind in {"node_ready", "node_dispatch", "node_finish", "node_fail"} and event.get("job_instance_id") in arrivals.get(str(event["episode_id"]), {}):
            if finite(event.get("time_ms")) + 1e-9 < arrivals[str(event["episode_id"])][event["job_instance_id"]]:
                violations.append(f"pre-arrival event: {event}")
    return {"pass": not violations, "violations": violations[:20], "checked_events": len(events)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/scheduler"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    templates = template_index(read_jsonl(root / "results/processed/workload_v0_2_smoke_fixed_20260812/job_templates.jsonl"))
    episodes = read_jsonl(root / f"results/processed/workload_v0_2_formal_20260812/{args.split}.jsonl")[: args.episodes]
    episode_map = {str(row["episode_id"]): row for row in episodes}
    resource, behavior, aux = load_artifacts(root / "results/processed/scheduling_future_v1_20260812")
    policies = tuple(value.strip() for value in args.policies.split(",") if value.strip())
    if set(policies) - set(POLICIES):
        raise ValueError(f"unknown policies: {sorted(set(policies) - set(POLICIES))}")
    summaries: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for episode in episodes:
        for policy in policies:
            result, trace = simulate(episode, templates, policy, resource, behavior, aux["family_medians"])
            summaries.append(result)
            events.extend(trace)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "scheduling_smoke_results.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in summaries) + "\n", encoding="utf-8")
    with gzip.open(out / "scheduling_smoke_events.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in events:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    validation = validate_events({}, events, templates, episode_map)
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        by_policy[str(row["policy"])].append(row)
    aggregate = [{"policy": policy, "episodes": len(rows), "mean_completion_ms": statistics.fmean(finite(row.get("mean_completion_ms"), 0.0) for row in rows), "mean_makespan_ms": statistics.fmean(finite(row.get("makespan_ms"), 0.0) for row in rows), "deadline_miss_rate": statistics.fmean(finite(row.get("deadline_miss_rate"), 0.0) for row in rows), "completed_jobs": statistics.fmean(finite(row.get("completed_jobs"), 0.0) for row in rows), "mean_gpu_evictions": statistics.fmean(finite(row.get("gpu_evictions"), 0.0) for row in rows)} for policy, rows in sorted(by_policy.items())]
    report = {
        "schema_version": "scheduling-future-smoke-report-v1",
        "split": args.split,
        "episodes": len(episodes),
        "policies": list(policies),
        "templates": len(templates),
        "resource_nodes": len(resource),
        "behavior_nodes": len(behavior),
        "aggregate": aggregate,
        "validation": validation,
        "deterministic_replay": {"checked": False},
        "information_boundary": {"pred_h": "B05 synthetic future artifact only", "true_h": "actual DAG successors with resource predictor cost only", "execution_truth": "template runtime/load/workspace, hidden from dispatch score"},
        "no_future_leakage": True,
    }
    if episodes and policies:
        first_episode = episodes[0]
        replay_a, replay_events_a = simulate(first_episode, templates, policies[0], resource, behavior, aux["family_medians"])
        replay_b, replay_events_b = simulate(first_episode, templates, policies[0], resource, behavior, aux["family_medians"])
        replay_ok = replay_a == replay_b and replay_events_a == replay_events_b
        report["deterministic_replay"] = {
            "checked": True,
            "policy": policies[0],
            "episode_id": first_episode["episode_id"],
            "pass": replay_ok,
            "summary_equal": replay_a == replay_b,
            "events_equal": replay_events_a == replay_events_b,
        }
        if not replay_ok:
            raise RuntimeError("deterministic replay failed")
    (out / "scheduling_smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": len(episodes), "policies": list(policies), "validation_pass": validation["pass"], "output_dir": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
