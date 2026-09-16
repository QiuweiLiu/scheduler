#!/usr/bin/env python3
"""Audit action ranking on common simulator states.

This is a diagnostic evaluator.  It replays a fixed reference policy to obtain
common decision states, then evaluates every strict-feasible ``ready_node x
free_gpu`` action without changing the simulator or any policy.  The first
version uses the existing TrueOpt-H local suffix value: the next ``H`` DAG
layers for the candidate node.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import tracing.analysis.workload_v02_simulator as simulator
from tracing.analysis.workload_v02_simulator import (
    GPU,
    Job,
    Node,
    build_jobs,
    estimate,
    fits_gpu,
    limited_future_truth_cost,
    load_future_artifacts,
    load_templates,
    plan_gpu_admission,
    predicted_future_cost,
    read_jsonl,
    simulate_episode,
    train_resource_stats,
)


AUDIT_SCHEMA_VERSION = "action-value-audit-v0.1"
POLICIES = ("myopic", "predopt_h5", "predopt_v2_h5", "trueopt_h5")
AUDIT_POLICY_CHOICES = POLICIES + tuple(simulator.ALIGNED_H5_POLICIES)
TRUTH_CONTRACTS = ("legacy", "aligned_h5")
SCORE_SCALE_MS = 100000.0
TIE_EPSILON_MS = 1e-9


@dataclass(frozen=True)
class Candidate:
    job_index: int
    job_instance_id: str
    node_id: str
    gpu_index: int
    item: tuple[float, float, int, str]
    node: Node
    estimate_row: Mapping[str, Any]
    gpu: GPU
    raw_feasible: bool
    model_feasible: bool
    strict_feasible: bool

    @property
    def action_key(self) -> tuple[int, str, int]:
        return (self.job_index, self.node_id, self.gpu_index)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(state: Mapping[str, Any]) -> str:
    payload = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _action_payload(candidate: Candidate) -> dict[str, Any]:
    return {
        "job_instance_id": candidate.job_instance_id,
        "node_id": candidate.node_id,
        "gpu_index": candidate.gpu_index,
    }


def _width_bucket(width: int) -> str:
    if width <= 2:
        return "2"
    if width <= 4:
        return "4"
    if width <= 8:
        return "8"
    if width >= 12:
        return "12+"
    return str(width)


def _validate_policies(policies: Sequence[str]) -> None:
    if not policies:
        raise ValueError("at least one audit policy is required")
    unknown = sorted(set(policies) - set(AUDIT_POLICY_CHOICES))
    if unknown:
        raise ValueError(f"unsupported audit policy(s): {', '.join(unknown)}")


def _ranked(keys: Iterable[tuple[int, str, int]], scores: Mapping[tuple[int, str, int], Any]) -> list[tuple[int, str, int]]:
    return sorted(keys, key=lambda key: scores[key])


def _rank_of(action_key: tuple[int, str, int], ordered: Sequence[tuple[int, str, int]]) -> int | None:
    try:
        return ordered.index(action_key) + 1
    except ValueError:
        return None


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[ordered[position][0]] = average
        index = end
    return ranks


def spearman(values_x: Sequence[float], values_y: Sequence[float]) -> float | None:
    if len(values_x) != len(values_y) or len(values_x) < 2:
        return None
    ranks_x = _average_ranks(values_x)
    ranks_y = _average_ranks(values_y)
    mean_x = statistics.fmean(ranks_x)
    mean_y = statistics.fmean(ranks_y)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(ranks_x, ranks_y))
    denominator_x = math.sqrt(sum((x - mean_x) ** 2 for x in ranks_x))
    denominator_y = math.sqrt(sum((y - mean_y) ** 2 for y in ranks_y))
    if denominator_x == 0.0 or denominator_y == 0.0:
        return None
    return numerator / (denominator_x * denominator_y)


def _model_memory_by_id(templates: Mapping[str, Any]) -> dict[str, float]:
    memories: dict[str, float] = {}
    for template in templates.values():
        for node in template.nodes:
            if node.lane != "gpu" or node.resident_model_mb is None:
                continue
            memory = float(node.resident_model_mb)
            previous = memories.get(node.model_id)
            if previous is not None and abs(previous - memory) > 1e-9:
                raise ValueError(f"inconsistent resident memory for model {node.model_id}")
            memories[node.model_id] = memory
    return memories


def _reconstruct_state(
    episode: Mapping[str, Any],
    templates: Mapping[str, Any],
    scheduler_state: Mapping[str, Any],
) -> tuple[list[Job], list[GPU], list[tuple[float, float, int, str]]]:
    jobs = build_jobs(episode, templates)
    job_indices = {job.job_instance_id: index for index, job in enumerate(jobs)}
    for job in jobs:
        for node_id in scheduler_state.get("completed_prefix", {}).get(job.job_instance_id, []):
            if node_id in job.template.by_id:
                job.node_state[node_id] = "complete"
                job.completed.add(node_id)

    ready_items: list[tuple[float, float, int, str]] = []
    for raw in scheduler_state.get("ready_nodes") or []:
        job_id = str(raw["job_instance_id"])
        node_id = str(raw["node_id"])
        if job_id not in job_indices:
            raise ValueError(f"scheduler state references unknown job {job_id}")
        job_index = job_indices[job_id]
        job = jobs[job_index]
        if node_id not in job.template.by_id:
            raise ValueError(f"scheduler state references unknown node {job_id}:{node_id}")
        ready_since = float(raw.get("ready_since_ms") or scheduler_state.get("time_ms") or 0.0)
        job.node_state[node_id] = "ready"
        job.ready_since[node_id] = ready_since
        priority = 0.0 if job.service_class == "priority" else 1.0
        ready_items.append((priority, ready_since, job_index, node_id))

    memories = _model_memory_by_id(templates)
    gpus: list[GPU] = []
    for raw in scheduler_state.get("gpus") or []:
        gpu = GPU(int(raw["index"]), float(raw["capacity_mb"]))
        if bool(raw.get("busy")):
            gpu.active_node = (-1, "__busy__")
        for model_id in raw.get("resident_models") or []:
            model = str(model_id)
            if model not in memories:
                raise ValueError(f"resident model has no measured memory: {model}")
            gpu.resident[model] = memories[model]
        gpus.append(gpu)
    return jobs, gpus, ready_items


def _build_candidates(
    jobs: Sequence[Job],
    gpus: Sequence[GPU],
    ready_items: Sequence[tuple[float, float, int, str]],
    train_stats: Mapping[str, Mapping[str, Any]],
) -> list[Candidate]:
    free_gpus = [gpu for gpu in gpus if gpu.active_node is None]
    candidates: list[Candidate] = []
    for item in ready_items:
        _priority, _ready_since, job_index, node_id = item
        job = jobs[job_index]
        node = job.template.by_id[node_id]
        if node.lane != "gpu":
            continue
        row = estimate(node, train_stats)
        for gpu in free_gpus:
            raw_feasible = float(row["memory_p95_mb"]) <= gpu.capacity_mb + 1e-9
            model_feasible = fits_gpu(gpu, node, row)
            strict_feasible, _evictions, _model, _workspace, _projected = plan_gpu_admission(gpu, node, row)
            candidates.append(
                Candidate(
                    job_index=job_index,
                    job_instance_id=job.job_instance_id,
                    node_id=node_id,
                    gpu_index=gpu.index,
                    item=item,
                    node=node,
                    estimate_row=row,
                    gpu=gpu,
                    raw_feasible=raw_feasible,
                    model_feasible=model_feasible,
                    strict_feasible=strict_feasible,
                )
            )
    return candidates


def _score_tuple(
    policy: str,
    candidate: Candidate,
    jobs: Sequence[Job],
    train_stats: Mapping[str, Mapping[str, Any]],
    future_artifacts: Mapping[str, Mapping[str, Any]],
    horizon: int,
    model_memories: Mapping[str, float] | None = None,
) -> tuple[Any, ...]:
    item = candidate.item
    row = candidate.estimate_row
    load = 0.0 if candidate.node.model_id in candidate.gpu.resident else float(row["load_p50_ms"])
    current = float(row["runtime_p50_ms"]) + load
    future = predicted_future_cost(candidate.node_id, future_artifacts, train_stats, horizon)
    if policy == "myopic":
        reuse = 0.0 if candidate.node.model_id in candidate.gpu.resident else 1.0
        return (float(item[0]), current, reuse, float(item[1]), item[2], 0, item[3], candidate.gpu_index)
    if policy == "predopt_h5":
        return (float(item[0]), current + future, future, float(item[1]), item[2], 0, item[3], candidate.gpu_index)
    if policy == "predopt_v2_h5":
        memory_ratio = float(row["memory_p95_mb"]) / max(1.0, float(candidate.gpu.capacity_mb))
        total = (
            simulator.PREDOPT_V2_PRIORITY_WEIGHT * float(item[0])
            + current / SCORE_SCALE_MS
            + simulator.PREDOPT_V2_FUTURE_WEIGHT * future / SCORE_SCALE_MS
            + simulator.PREDOPT_V2_MEMORY_WEIGHT * memory_ratio
        )
        return (total, future / SCORE_SCALE_MS, current / SCORE_SCALE_MS, float(item[1]), item[2], item[3], candidate.gpu_index)
    if policy == "trueopt_h5":
        future_truth = limited_future_truth_cost(jobs[candidate.job_index], candidate.node_id, candidate.gpu, train_stats, horizon)
        return (future_truth, float(item[0]), float(item[1]), item[2], item[3], candidate.gpu_index)
    if policy in simulator.ALIGNED_H5_POLICIES:
        source = (
            "predicted_layer"
            if policy == "aligned_predopt_h5_layer"
            else "predicted"
            if policy == "aligned_predopt_h5"
            else "truth"
        )
        score = simulator.aligned_h5_score(
            source,
            jobs[candidate.job_index],
            candidate.node_id,
            candidate.gpu,
            train_stats,
            future_artifacts if source in {"predicted", "predicted_layer"} else None,
            horizon,
            model_memories or simulator.model_memory_by_model(job.template for job in jobs),
        )
        return (
            float(item[0]),
            score["total_ms"],
            score["future_ms"],
            score["current_ms"],
            float(item[1]),
            int(item[2]),
            str(item[3]),
            candidate.gpu_index,
        )
    raise ValueError(f"unsupported audit policy: {policy}")


def _true_value(
    candidate: Candidate,
    jobs: Sequence[Job],
    train_stats: Mapping[str, Mapping[str, Any]],
    horizon: int,
    truth_contract: str = "legacy",
    model_memories: Mapping[str, float] | None = None,
) -> float:
    if truth_contract == "aligned_h5":
        score = simulator.aligned_h5_score(
            "truth",
            jobs[candidate.job_index],
            candidate.node_id,
            candidate.gpu,
            train_stats,
            None,
            horizon,
            model_memories or simulator.model_memory_by_model(job.template for job in jobs),
        )
        return score["total_ms"]
    if truth_contract != "legacy":
        raise ValueError(f"unsupported truth contract: {truth_contract}")
    return limited_future_truth_cost(jobs[candidate.job_index], candidate.node_id, candidate.gpu, train_stats, horizon)


def _actual_policy_actions(
    policies: Sequence[str],
    ready_items: Sequence[tuple[float, float, int, str]],
    jobs: Sequence[Job],
    free_gpus: Sequence[GPU],
    train_stats: Mapping[str, Mapping[str, Any]],
    future_artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[int, str, int]]:
    actions: dict[str, tuple[int, str, int]] = {}
    for policy in policies:
        ready_item, gpu_index, _row, _candidate_count, _feasible_count = simulator.choose_action(
            policy,
            ready_items,
            jobs,
            free_gpus,
            train_stats,
            0,
            future_artifacts,
        )
        actions[policy] = (int(ready_item[2]), str(ready_item[3]), int(gpu_index))
    return actions


def audit_dispatch_event(
    episode: Mapping[str, Any],
    templates: Mapping[str, Any],
    event: Mapping[str, Any],
    train_stats: Mapping[str, Mapping[str, Any]],
    future_artifacts: Mapping[str, Mapping[str, Any]],
    horizon: int = 5,
    *,
    policies: Sequence[str] = POLICIES,
    truth_contract: str = "legacy",
) -> dict[str, Any]:
    """Evaluate one dispatch state without mutating simulator state."""

    scheduler_state = event.get("scheduler_state")
    if not isinstance(scheduler_state, Mapping):
        raise ValueError("node_dispatch event has no scheduler_state")
    jobs, gpus, ready_items = _reconstruct_state(episode, templates, scheduler_state)
    candidates = _build_candidates(jobs, gpus, ready_items, train_stats)
    strict = [candidate for candidate in candidates if candidate.strict_feasible]
    if not strict:
        raise ValueError(f"no strict-feasible action at {episode['episode_id']} decision {scheduler_state.get('decision_index')}")
    policy_names = tuple(policies)
    _validate_policies(policy_names)
    if truth_contract not in TRUTH_CONTRACTS:
        raise ValueError(f"unsupported truth contract: {truth_contract}")
    if truth_contract == "aligned_h5" and "aligned_trueopt_h5" not in policy_names:
        raise ValueError("aligned_h5 truth contract requires aligned_trueopt_h5 in --policies")
    model_memories = simulator.model_memory_by_model(job.template for job in jobs)
    free_gpus = [gpu for gpu in gpus if gpu.active_node is None]
    actual_actions = _actual_policy_actions(policy_names, ready_items, jobs, free_gpus, train_stats, future_artifacts)
    candidate_keys = [candidate.action_key for candidate in strict]
    score_maps = {
        policy: {
            candidate.action_key: _score_tuple(
                policy,
                candidate,
                jobs,
                train_stats,
                future_artifacts,
                horizon,
                model_memories,
            )
            for candidate in strict
        }
        for policy in policy_names
    }
    truth_scores = {
        candidate.action_key: _true_value(
            candidate,
            jobs,
            train_stats,
            horizon,
            truth_contract,
            model_memories,
        )
        for candidate in strict
    }
    if truth_contract == "aligned_h5":
        # The aligned contract treats priority as a hard first-level rule.
        # The reference optimum must therefore use the same tuple as the
        # privileged aligned_trueopt_h5 policy, rather than minimizing the
        # numeric cost across priority classes.
        truth_key_map = score_maps["aligned_trueopt_h5"]
        truth_order = _ranked(candidate_keys, truth_key_map)
        best_truth = truth_scores[truth_order[0]]
        best_truth_key = truth_key_map[truth_order[0]]
        truth_ties = {
            key
            for key in candidate_keys
            if truth_key_map[key][0] == best_truth_key[0]
            and abs(float(truth_key_map[key][1]) - float(best_truth_key[1])) <= TIE_EPSILON_MS
        }
        same_priority_values = sorted(set(
            float(truth_key_map[key][1])
            for key in candidate_keys
            if truth_key_map[key][0] == best_truth_key[0]
        ))
        second_gap = same_priority_values[1] - same_priority_values[0] if len(same_priority_values) > 1 else 0.0
    else:
        truth_order = _ranked(candidate_keys, {key: (truth_scores[key], key) for key in candidate_keys})
        best_truth = min(truth_scores.values())
        truth_ties = {key for key, value in truth_scores.items() if abs(value - best_truth) <= TIE_EPSILON_MS}
        unique_truth = sorted(set(truth_scores.values()))
        second_gap = unique_truth[1] - unique_truth[0] if len(unique_truth) > 1 else 0.0
    records_by_key = {candidate.action_key: candidate for candidate in candidates}
    policies: dict[str, Any] = {}
    for policy in policy_names:
        score_order = _ranked(candidate_keys, score_maps[policy])
        score_rank_values = {key: float(index) for index, key in enumerate(score_order)}
        truth_rank_values = {key: float(index) for index, key in enumerate(truth_order)}
        chosen = actual_actions[policy]
        chosen_candidate = records_by_key.get(chosen)
        chosen_in_strict = chosen in truth_scores
        chosen_truth = truth_scores.get(chosen)
        chosen_priority_violation = False
        if truth_contract == "aligned_h5" and chosen in truth_key_map:
            chosen_priority_violation = truth_key_map[chosen][0] != best_truth_key[0]
        chosen_regret = (
            (chosen_truth - best_truth)
            if chosen_truth is not None and not chosen_priority_violation
            else None
        )
        chosen_job = jobs[chosen[0]] if 0 <= chosen[0] < len(jobs) else None
        policies[policy] = {
            "chosen_action": _action_payload(chosen_candidate) if chosen_candidate else {
                "job_instance_id": jobs[chosen[0]].job_instance_id if 0 <= chosen[0] < len(jobs) else None,
                "node_id": chosen[1],
                "gpu_index": chosen[2],
            },
            "chosen_in_strict_feasible": chosen_in_strict,
            "chosen_true_value_ms": chosen_truth,
            "true_regret_ms": chosen_regret,
            "priority_violation": chosen_priority_violation,
            "true_top1_agreement": bool(chosen in truth_ties),
            "predicted_score_top1_agreement": bool(score_order[0] in truth_ties),
            "chosen_score_rank": _rank_of(chosen, score_order),
            "chosen_true_rank": _rank_of(chosen, truth_order),
            "chosen_priority": float(chosen_candidate.item[0]) if chosen_candidate else None,
            "chosen_service_class": chosen_job.service_class if chosen_job else None,
            "chosen_model_id": chosen_candidate.node.model_id if chosen_candidate else None,
            "chosen_role": chosen_candidate.node.role if chosen_candidate else None,
            "chosen_action_family": chosen_candidate.node.action_family if chosen_candidate else None,
            "chosen_cache_hit": (
                bool(chosen_candidate.node.model_id in chosen_candidate.gpu.resident)
                if chosen_candidate
                else None
            ),
            "chosen_deadline_slack_ms": (
                float(chosen_job.deadline_ms) - float(scheduler_state.get("time_ms") or 0.0)
                if chosen_job is not None and chosen_job.deadline_ms is not None
                else None
            ),
            "predicted_top1_action": _action_payload(records_by_key[score_order[0]]),
            "true_top1_actions": [_action_payload(records_by_key[key]) for key in truth_ties],
            "spearman_predicted_vs_true": spearman(
                [score_rank_values[key] for key in candidate_keys],
                [truth_rank_values[key] for key in candidate_keys],
            ),
        }

    candidate_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row: dict[str, Any] = {
            "action": _action_payload(candidate),
            "model_id": candidate.node.model_id,
            "role": candidate.node.role,
            "action_family": candidate.node.action_family,
            "service_class": jobs[candidate.job_index].service_class,
            "cache_hit": bool(candidate.node.model_id in candidate.gpu.resident),
            "deadline_slack_ms": (
                float(jobs[candidate.job_index].deadline_ms) - float(scheduler_state.get("time_ms") or 0.0)
                if jobs[candidate.job_index].deadline_ms is not None
                else None
            ),
            "raw_feasible": candidate.raw_feasible,
            "model_feasible": candidate.model_feasible,
            "strict_feasible": candidate.strict_feasible,
        }
        if candidate.strict_feasible:
            row["true_h5_suffix_value_ms"] = truth_scores[candidate.action_key]
            row["scores"] = {
                policy: list(score_maps[policy][candidate.action_key])
                for policy in policy_names
            }
            row["truth_value_ms"] = truth_scores[candidate.action_key]
        candidate_rows.append(row)

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "episode_id": str(episode["episode_id"]),
        "decision_index": int(scheduler_state.get("decision_index") or 0),
        "time_ms": float(scheduler_state.get("time_ms") or 0.0),
        "reference_policy": str(event.get("policy") or "unknown"),
        "state_hash": state_hash(scheduler_state),
        "action_definition": "(job_instance_id, node_id, gpu_index)",
        "q_definition": (
            "aligned_h5_score(truth): current action plus next H DAG layers with shared cache contract"
            if truth_contract == "aligned_h5"
            else "limited_future_truth_cost: next H DAG layers, current TrueOpt-H semantics"
        ),
        "truth_contract": truth_contract,
        "policy_names": list(policy_names),
        "raw_candidate_width": len(candidates),
        "model_feasible_width": sum(candidate.model_feasible for candidate in candidates),
        "strict_feasible_width": len(strict),
        "strict_width_bucket": _width_bucket(len(strict)),
        "ready_gpu_node_count": len(ready_items),
        "free_gpu_count": len(free_gpus),
        "resident_model_count": sum(len(gpu.resident) for gpu in gpus),
        "ready_service_class_counts": dict(Counter(jobs[item[2]].service_class for item in ready_items)),
        "first_second_true_value_gap_ms": second_gap,
        "truth_best_value_ms": best_truth,
        "policies": policies,
        "candidates": candidate_rows,
    }


def _mean_or_none(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _deadline_bucket(value: Any) -> str:
    if value is None:
        return "no_deadline"
    return "overdue" if float(value) < 0.0 else "nonnegative"


def _strata_metrics(records: Sequence[Mapping[str, Any]], policy: str) -> dict[str, Any]:
    fields = {
        "model": lambda row: str(row.get("chosen_model_id")),
        "role": lambda row: str(row.get("chosen_role")),
        "cache_hit": lambda row: str(row.get("chosen_cache_hit")),
        "service_class": lambda row: str(row.get("chosen_service_class")),
        "deadline_sign": lambda row: _deadline_bucket(row.get("chosen_deadline_slack_ms")),
    }
    result: dict[str, Any] = {}
    for field, getter in fields.items():
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in records:
            groups[getter(record["policies"][policy])].append(record["policies"][policy])
        result[field] = {}
        for value, rows in sorted(groups.items()):
            regrets = [float(row["true_regret_ms"]) for row in rows if row.get("true_regret_ms") is not None]
            result[field][value] = {
                "decisions": len(rows),
                "error_count": sum(not row["true_top1_agreement"] for row in rows),
                "error_rate": _mean_or_none([1.0 if not row["true_top1_agreement"] else 0.0 for row in rows]),
                "true_top1_agreement_rate": _mean_or_none([1.0 if row["true_top1_agreement"] else 0.0 for row in rows]),
                "mean_true_regret_ms": _mean_or_none(regrets),
            }
    return result


def _aggregate_records(
    records: Sequence[Mapping[str, Any]],
    policies: Sequence[str] | None = None,
) -> dict[str, Any]:
    policy_names = tuple(policies or (records[0].get("policies") or {}).keys() or POLICIES)
    metrics: dict[str, Any] = {}
    for policy in policy_names:
        rows = [record["policies"][policy] for record in records]
        regrets = [float(row["true_regret_ms"]) for row in rows if row.get("true_regret_ms") is not None]
        correlations = [float(row["spearman_predicted_vs_true"]) for row in rows if row.get("spearman_predicted_vs_true") is not None]
        metrics[policy] = {
            "decisions": len(rows),
            "true_top1_agreement_rate": _mean_or_none([1.0 if row["true_top1_agreement"] else 0.0 for row in rows]),
            "predicted_score_top1_agreement_rate": _mean_or_none([1.0 if row["predicted_score_top1_agreement"] else 0.0 for row in rows]),
            "chosen_in_strict_feasible_rate": _mean_or_none([1.0 if row["chosen_in_strict_feasible"] else 0.0 for row in rows]),
            "priority_violation_rate": _mean_or_none([1.0 if row["priority_violation"] else 0.0 for row in rows]),
            "mean_true_regret_ms": _mean_or_none(regrets),
            "p95_true_regret_ms": quantile(regrets, 0.95),
            "mean_spearman_predicted_vs_true": _mean_or_none(correlations),
        }
        metrics[policy]["strata"] = _strata_metrics(records, policy)
    width_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        width_groups[str(record["strict_width_bucket"])].append(record)
    by_width: dict[str, Any] = {}
    for width, group in sorted(width_groups.items()):
        by_width[width] = {
            "decisions": len(group),
            "mean_first_second_true_value_gap_ms": _mean_or_none([float(row["first_second_true_value_gap_ms"]) for row in group]),
            "policies": {
                policy: {
                    "true_top1_agreement_rate": _mean_or_none([
                        1.0 if row["policies"][policy]["true_top1_agreement"] else 0.0 for row in group
                    ]),
                    "mean_true_regret_ms": _mean_or_none([
                        float(row["policies"][policy]["true_regret_ms"])
                        for row in group
                        if row["policies"][policy].get("true_regret_ms") is not None
                    ]),
                }
                for policy in policy_names
            },
        }
    return {"policies": metrics, "by_strict_width": by_width}


def quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def run_audit(
    episodes: Sequence[Mapping[str, Any]],
    templates: Mapping[str, Any],
    future_artifacts: Mapping[str, Mapping[str, Any]],
    train_stats: Mapping[str, Mapping[str, Any]],
    *,
    limit: int = 100,
    reference_policy: str = "myopic",
    horizon: int = 5,
    policies: Sequence[str] = POLICIES,
    truth_contract: str = "legacy",
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if reference_policy not in {"myopic", "predopt_h5"}:
        raise ValueError(f"unsupported reference policy: {reference_policy}")
    policy_names = tuple(policies)
    _validate_policies(policy_names)
    if truth_contract not in TRUTH_CONTRACTS:
        raise ValueError(f"unsupported truth contract: {truth_contract}")
    if truth_contract == "aligned_h5" and "aligned_trueopt_h5" not in policy_names:
        raise ValueError("aligned_h5 truth contract requires aligned_trueopt_h5 in --policies")
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    missing_artifacts: Counter[str] = Counter()
    selected_episodes = list(episodes[:limit])
    for episode in selected_episodes:
        summary, events = simulate_episode(
            episode,
            templates,
            reference_policy,
            future_artifacts=future_artifacts,
            train_stats=train_stats,
            collect_events=True,
        )
        summaries.append(dict(summary))
        for event in events:
            if event.get("event_type") != "node_dispatch":
                continue
            state = event.get("scheduler_state") or {}
            for ready in state.get("ready_nodes") or []:
                node_id = str(ready.get("node_id") or "")
                if node_id and node_id not in future_artifacts:
                    missing_artifacts[node_id] += 1
            records.append(
                audit_dispatch_event(
                    episode,
                    templates,
                    event,
                    train_stats,
                    future_artifacts,
                    horizon,
                    policies=policy_names,
                    truth_contract=truth_contract,
                )
            )

    failed_jobs = sum(int(summary.get("failed_jobs") or 0) for summary in summaries)
    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "passed" if selected_episodes and records and failed_jobs == 0 and not missing_artifacts else "failed",
        "reference_policy": reference_policy,
        "future_horizon": horizon,
        "q_definition": (
            "Q_aligned_H5 = aligned_h5_score(truth): current action plus next H DAG layers under shared cache contract; lower is better"
            if truth_contract == "aligned_h5"
            else "Q_local_H5 = limited_future_truth_cost over next H DAG layers; lower is better"
        ),
        "truth_contract": truth_contract,
        "policies": list(policy_names),
        "action_definition": "(job_instance_id, node_id, gpu_index)",
        "episodes": len(selected_episodes),
        "decisions": len(records),
        "failed_jobs": failed_jobs,
        "missing_future_artifact_nodes": dict(sorted(missing_artifacts.items())),
        "information_boundary": "reference scheduler state is scheduler-safe; true suffix values are audit-only and never passed to policy",
        "metrics": _aggregate_records(records, policy_names),
    }
    return report, records, summaries


def _write_metrics_csv(path: Path, report: Mapping[str, Any]) -> None:
    rows = []
    for policy, values in (report.get("metrics") or {}).get("policies", {}).items():
        row = {"policy": policy}
        row.update({key: value for key, value in values.items() if not isinstance(value, (dict, list))})
        rows.append(row)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--future-artifacts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, choices=(1, 3, 5), default=5)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--reference-policy", choices=("myopic", "predopt_h5"), default="myopic")
    parser.add_argument(
        "--policies",
        default=",".join(POLICIES),
        help="comma-separated audit policies; aligned H5 policies including aligned_predopt_h5_layer are opt-in",
    )
    parser.add_argument("--truth-contract", choices=TRUTH_CONTRACTS, default="legacy")
    args = parser.parse_args()

    policy_names = tuple(value.strip() for value in args.policies.split(",") if value.strip())
    _validate_policies(policy_names)

    templates = load_templates(args.templates)
    episodes = read_jsonl(args.episodes)
    future_artifacts = load_future_artifacts(args.future_artifacts)
    train_stats = train_resource_stats(templates)
    for node_id, artifact in future_artifacts.items():
        for horizon in (1, 3, 5):
            artifact[f"_cost_h{horizon}"] = predicted_future_cost(node_id, future_artifacts, train_stats, horizon)

    report, records, summaries = run_audit(
        episodes,
        templates,
        future_artifacts,
        train_stats,
        limit=args.limit,
        reference_policy=args.reference_policy,
        horizon=args.horizon,
        policies=policy_names,
        truth_contract=args.truth_contract,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    simulator.write_jsonl(args.output_dir / "decision_records.jsonl", records)
    simulator.write_jsonl(args.output_dir / "reference_summaries.jsonl", summaries)
    simulator.write_json(args.output_dir / "metrics.json", report)
    _write_metrics_csv(args.output_dir / "metrics.csv", report)
    manifest = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "templates": str(args.templates),
        "templates_sha256": sha256(args.templates),
        "episodes": str(args.episodes),
        "episodes_sha256": sha256(args.episodes),
        "future_artifacts": str(args.future_artifacts),
        "future_artifact_manifest_sha256": (
            sha256(args.future_artifacts / "b05_artifact_manifest.json")
            if (args.future_artifacts / "b05_artifact_manifest.json").is_file()
            else None
        ),
        "reference_policy": args.reference_policy,
        "horizon": args.horizon,
        "limit": args.limit,
        "policies": list(policy_names),
        "truth_contract": args.truth_contract,
        "q_definition": report["q_definition"],
    }
    simulator.write_json(args.output_dir / "run_manifest.json", manifest)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
