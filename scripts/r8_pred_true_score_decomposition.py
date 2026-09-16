#!/usr/bin/env python3
"""Compare Pred/True aligned-H5 scores on the same recorded decisions.

This is a post-hoc, read-only diagnostic over the existing common-state action
audit.  It does not replay an episode and it never passes truth values to a
policy.  For every recorded decision it verifies the candidate filter, then
emits one row per candidate with the predicted and truth score components,
ranks, pairwise order changes, and top-1 consequences.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "pred-true-score-decomposition-v0.1"
PRED_POLICY = "aligned_predopt_h5"
TRUE_POLICY = "aligned_trueopt_h5"
TIE_EPSILON_MS = 1e-9
SCORE_INDEX = {
    "priority": 0,
    "total_ms": 1,
    "future_ms": 2,
    "current_ms": 3,
    "ready_since_ms": 4,
    "job_index": 5,
    "node_id": 6,
    "gpu_index": 7,
}
COMPONENTS = ("current_ms", "future_ms", "total_ms")
GROUP_FIELDS = (
    "model_id",
    "role",
    "service_class",
    "cache_hit",
    "strict_width_bucket",
    "deadline_bucket",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_write(handle: Any, row: Mapping[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    handle.write("\n")


def as_action_key(action: Any) -> Optional[Tuple[str, str, int]]:
    if not isinstance(action, Mapping):
        return None
    try:
        return (
            str(action["job_instance_id"]),
            str(action["node_id"]),
            int(action["gpu_index"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def action_payload(action: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(action, Mapping):
        return None
    return {
        "job_instance_id": str(action.get("job_instance_id")),
        "node_id": str(action.get("node_id")),
        "gpu_index": int(action.get("gpu_index")),
    }


def action_from_key(key: Optional[Tuple[str, str, int]]) -> Optional[Dict[str, Any]]:
    if key is None:
        return None
    return {
        "job_instance_id": key[0],
        "node_id": key[1],
        "gpu_index": key[2],
    }


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def score_payload(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) < 8:
        raise ValueError("aligned score must be a sequence with at least 8 fields")
    values = list(raw)
    return {
        "raw": values,
        "priority": safe_float(values[SCORE_INDEX["priority"]]),
        "total_ms": safe_float(values[SCORE_INDEX["total_ms"]]),
        "future_ms": safe_float(values[SCORE_INDEX["future_ms"]]),
        "current_ms": safe_float(values[SCORE_INDEX["current_ms"]]),
        "ready_since_ms": safe_float(values[SCORE_INDEX["ready_since_ms"]]),
        "job_index": int(values[SCORE_INDEX["job_index"]]),
        "node_id": str(values[SCORE_INDEX["node_id"]]),
        "gpu_index": int(values[SCORE_INDEX["gpu_index"]]),
    }


def filter_stage(candidate: Mapping[str, Any]) -> str:
    if not bool(candidate.get("raw_feasible")):
        return "raw_rejected"
    if not bool(candidate.get("model_feasible")):
        return "model_rejected"
    if not bool(candidate.get("strict_feasible")):
        return "strict_rejected"
    return "strict_feasible"


def deadline_bucket(value: Any) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return "no_deadline"
    return "overdue" if numeric < 0.0 else "nonnegative"


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": quantile(values, 0.50),
        "p95": quantile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def average_ranks(values: Sequence[float]) -> List[float]:
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


def spearman(values_x: Sequence[float], values_y: Sequence[float]) -> Optional[float]:
    if len(values_x) != len(values_y) or len(values_x) < 2:
        return None
    ranks_x = average_ranks(values_x)
    ranks_y = average_ranks(values_y)
    mean_x = statistics.fmean(ranks_x)
    mean_y = statistics.fmean(ranks_y)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(ranks_x, ranks_y))
    denominator_x = math.sqrt(sum((x - mean_x) ** 2 for x in ranks_x))
    denominator_y = math.sqrt(sum((y - mean_y) ** 2 for y in ranks_y))
    if denominator_x == 0.0 or denominator_y == 0.0:
        return None
    return numerator / (denominator_x * denominator_y)


def new_component_stats() -> Dict[str, Any]:
    result: Dict[str, Any] = {"count": 0}
    for component in COMPONENTS:
        result["sum_" + component] = 0.0
        result["sum_abs_" + component] = 0.0
    result["current_dominant"] = 0
    result["future_dominant"] = 0
    result["balanced"] = 0
    return result


def update_component_stats(stats: MutableMapping[str, Any], deltas: Mapping[str, float]) -> None:
    stats["count"] += 1
    for component in COMPONENTS:
        value = float(deltas[component])
        stats["sum_" + component] += value
        stats["sum_abs_" + component] += abs(value)
    current = abs(float(deltas["current_ms"]))
    future = abs(float(deltas["future_ms"]))
    if current > future:
        stats["current_dominant"] += 1
    elif future > current:
        stats["future_dominant"] += 1
    else:
        stats["balanced"] += 1


def finalize_component_stats(stats: Mapping[str, Any]) -> Dict[str, Any]:
    count = int(stats.get("count") or 0)
    result: Dict[str, Any] = {"count": count}
    for component in COMPONENTS:
        result["mean_signed_" + component] = (
            float(stats["sum_" + component]) / count if count else None
        )
        result["mean_abs_" + component] = (
            float(stats["sum_abs_" + component]) / count if count else None
        )
    result["error_dominance"] = {
        "current": int(stats.get("current_dominant") or 0),
        "future": int(stats.get("future_dominant") or 0),
        "balanced": int(stats.get("balanced") or 0),
    }
    return result


def new_decision_group_stats() -> Dict[str, Any]:
    return {
        "decisions": 0,
        "exact_top1_agreement": 0,
        "truth_tie_aware_agreement": 0,
        "priority_match": 0,
        "priority_mismatch": 0,
        "same_priority_action_mismatch": 0,
        "filter_mismatch": 0,
        "sum_truth_regret_ms": 0.0,
        "sum_abs_pred_top1_total_error_ms": 0.0,
        "sum_abs_pred_top1_current_error_ms": 0.0,
        "sum_abs_pred_top1_future_error_ms": 0.0,
        "sum_inversion_rate": 0.0,
    }


def update_decision_group_stats(
    stats: MutableMapping[str, Any],
    decision: Mapping[str, Any],
) -> None:
    top = decision["top1_comparison"]
    error = decision["top1_score_error"]
    ranking = decision["ranking"]
    stats["decisions"] += 1
    stats["exact_top1_agreement"] += int(bool(top["exact_action_agreement"]))
    stats["truth_tie_aware_agreement"] += int(bool(top["truth_tie_aware_agreement"]))
    stats["priority_match"] += int(bool(top["priority_match"]))
    stats["priority_mismatch"] += int(bool(top["priority_mismatch"]))
    stats["same_priority_action_mismatch"] += int(bool(top["same_priority_action_mismatch"]))
    stats["filter_mismatch"] += int(bool(decision["candidate_filter"]["any_mismatch"]))
    regret = safe_float(top.get("raw_truth_total_gap_ms"))
    if regret is not None:
        stats["sum_truth_regret_ms"] += regret
    for field, error_key in (
        ("sum_abs_pred_top1_total_error_ms", "total_ms"),
        ("sum_abs_pred_top1_current_error_ms", "current_ms"),
        ("sum_abs_pred_top1_future_error_ms", "future_ms"),
    ):
        value = safe_float(error.get("pred_top1", {}).get("abs_" + error_key))
        if value is not None:
            stats[field] += value
    inversion_rate = safe_float(ranking.get("inversion_rate"))
    if inversion_rate is not None:
        stats["sum_inversion_rate"] += inversion_rate


def finalize_decision_group_stats(stats: Mapping[str, Any]) -> Dict[str, Any]:
    count = int(stats.get("decisions") or 0)
    return {
        "decisions": count,
        "exact_top1_agreement_rate": (
            float(stats["exact_top1_agreement"]) / count if count else None
        ),
        "truth_tie_aware_agreement_rate": (
            float(stats["truth_tie_aware_agreement"]) / count if count else None
        ),
        "priority_match_rate": float(stats["priority_match"]) / count if count else None,
        "priority_mismatch_rate": float(stats["priority_mismatch"]) / count if count else None,
        "same_priority_action_mismatch_rate": (
            float(stats["same_priority_action_mismatch"]) / count if count else None
        ),
        "filter_mismatch_rate": float(stats["filter_mismatch"]) / count if count else None,
        "mean_raw_truth_total_gap_ms": (
            float(stats["sum_truth_regret_ms"]) / count if count else None
        ),
        "mean_abs_pred_top1_error_ms": {
            "total": float(stats["sum_abs_pred_top1_total_error_ms"]) / count if count else None,
            "current": float(stats["sum_abs_pred_top1_current_error_ms"]) / count if count else None,
            "future": float(stats["sum_abs_pred_top1_future_error_ms"]) / count if count else None,
        },
        "mean_inversion_rate": float(stats["sum_inversion_rate"]) / count if count else None,
    }


def update_group(
    group_map: MutableMapping[str, MutableMapping[str, Dict[str, Any]]],
    field: str,
    value: Any,
    deltas: Mapping[str, float],
) -> None:
    key = str(value)
    groups = group_map[field]
    if key not in groups:
        groups[key] = new_component_stats()
    update_component_stats(groups[key], deltas)


def update_decision_groups(
    group_map: MutableMapping[str, MutableMapping[str, Dict[str, Any]]],
    decision: Mapping[str, Any],
) -> None:
    top = decision["pred_top1_candidate"]
    for field in GROUP_FIELDS:
        if field == "strict_width_bucket":
            value = decision["strict_width_bucket"]
        elif field == "deadline_bucket":
            value = deadline_bucket(top.get("deadline_slack_ms"))
        else:
            value = top.get(field)
        key = str(value)
        if key not in group_map[field]:
            group_map[field][key] = new_decision_group_stats()
        update_decision_group_stats(group_map[field][key], decision)


def action_candidate_key(candidate: Mapping[str, Any]) -> Tuple[str, str, int]:
    action = candidate.get("action")
    key = as_action_key(action)
    if key is None:
        raise ValueError("candidate is missing a valid action")
    return key


def score_sort_key(candidate: Mapping[str, Any], policy: str) -> Tuple[Any, ...]:
    score = candidate["scores"][policy]
    return tuple(score)


def score_delta(pred: Mapping[str, Any], truth: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "current_ms": float(pred["current_ms"]) - float(truth["current_ms"]),
        "future_ms": float(pred["future_ms"]) - float(truth["future_ms"]),
        "total_ms": float(pred["total_ms"]) - float(truth["total_ms"]),
    }


def error_payload(deltas: Mapping[str, float]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for component in COMPONENTS:
        value = float(deltas[component])
        result[component] = value
        result["abs_" + component] = abs(value)
    return result


def compare_decision(
    record: Mapping[str, Any],
    candidate_handle: Any,
    top_k: int,
    global_state: MutableMapping[str, Any],
) -> Dict[str, Any]:
    candidates = list(record.get("candidates") or [])
    strict_candidates = [candidate for candidate in candidates if bool(candidate.get("strict_feasible"))]
    if not strict_candidates:
        raise ValueError("decision has no strict-feasible candidate")

    strict_keys = {action_candidate_key(candidate) for candidate in strict_candidates}
    pred_scored_keys = {
        action_candidate_key(candidate)
        for candidate in candidates
        if bool(candidate.get("strict_feasible"))
        and isinstance(candidate.get("scores"), Mapping)
        and PRED_POLICY in candidate["scores"]
    }
    true_scored_keys = {
        action_candidate_key(candidate)
        for candidate in candidates
        if bool(candidate.get("strict_feasible"))
        and isinstance(candidate.get("scores"), Mapping)
        and TRUE_POLICY in candidate["scores"]
    }
    score_presence_mismatch = sum(
        1
        for candidate in strict_candidates
        if not isinstance(candidate.get("scores"), Mapping)
        or PRED_POLICY not in candidate["scores"]
        or TRUE_POLICY not in candidate["scores"]
    )

    scored_candidates: List[Mapping[str, Any]] = []
    for candidate in strict_candidates:
        if not isinstance(candidate.get("scores"), Mapping):
            continue
        if PRED_POLICY not in candidate["scores"] or TRUE_POLICY not in candidate["scores"]:
            continue
        scored_candidates.append(candidate)

    pred_order = sorted(scored_candidates, key=lambda item: score_sort_key(item, PRED_POLICY))
    true_order = sorted(scored_candidates, key=lambda item: score_sort_key(item, TRUE_POLICY))
    pred_keys = [action_candidate_key(candidate) for candidate in pred_order]
    true_keys = [action_candidate_key(candidate) for candidate in true_order]
    pred_rank = {key: index + 1 for index, key in enumerate(pred_keys)}
    true_rank = {key: index + 1 for index, key in enumerate(true_keys)}

    pred_top = pred_order[0]
    true_top = true_order[0]
    pred_top_key = action_candidate_key(pred_top)
    true_top_key = action_candidate_key(true_top)
    true_top_score = score_payload(true_top["scores"][TRUE_POLICY])
    true_top_tie_keys = []
    for candidate in true_order:
        score = score_payload(candidate["scores"][TRUE_POLICY])
        if (
            score["priority"] == true_top_score["priority"]
            and abs(float(score["total_ms"]) - float(true_top_score["total_ms"])) <= TIE_EPSILON_MS
        ):
            true_top_tie_keys.append(action_candidate_key(candidate))

    pred_score_top = score_payload(pred_top["scores"][PRED_POLICY])
    pred_top_truth_score = score_payload(pred_top["scores"][TRUE_POLICY])
    true_top_pred_score = score_payload(true_top["scores"][PRED_POLICY])
    true_top_truth_score = score_payload(true_top["scores"][TRUE_POLICY])
    pred_top_delta = score_delta(pred_score_top, pred_top_truth_score)
    true_top_delta = score_delta(true_top_pred_score, true_top_truth_score)

    pair_count = len(scored_candidates) * (len(scored_candidates) - 1) // 2
    inversion_count = 0
    for left in range(len(pred_keys)):
        for right in range(left + 1, len(pred_keys)):
            if true_rank[pred_keys[left]] > true_rank[pred_keys[right]]:
                inversion_count += 1
    inversion_rate = float(inversion_count) / pair_count if pair_count else 0.0
    tau = 1.0 - 2.0 * inversion_rate if pair_count else None
    common_keys = pred_keys
    pred_positions = [float(pred_rank[key]) for key in common_keys]
    true_positions = [float(true_rank[key]) for key in common_keys]

    filter_counts = Counter(filter_stage(candidate) for candidate in candidates)
    recorded_pred_action = as_action_key(
        (record.get("policies") or {}).get(PRED_POLICY, {}).get("chosen_action")
    )
    recorded_true_action = as_action_key(
        (record.get("policies") or {}).get(TRUE_POLICY, {}).get("chosen_action")
    )
    candidate_filter = {
        "raw_candidate_width": int(record.get("raw_candidate_width") or 0),
        "model_feasible_width": int(record.get("model_feasible_width") or 0),
        "strict_feasible_width": int(record.get("strict_feasible_width") or 0),
        "derived_raw_candidate_width": len(candidates),
        "derived_model_feasible_width": sum(bool(c.get("model_feasible")) for c in candidates),
        "derived_strict_feasible_width": len(strict_candidates),
        "filter_stage_counts": dict(sorted(filter_counts.items())),
        "pred_scored_width": len(pred_scored_keys),
        "true_scored_width": len(true_scored_keys),
        "strict_set_equals_pred_scored_set": strict_keys == pred_scored_keys,
        "strict_set_equals_true_scored_set": strict_keys == true_scored_keys,
        "pred_true_scored_set_equal": pred_scored_keys == true_scored_keys,
        "score_presence_mismatch_count": score_presence_mismatch,
    }
    filter_mismatch_reasons = []
    if candidate_filter["raw_candidate_width"] != candidate_filter["derived_raw_candidate_width"]:
        filter_mismatch_reasons.append("raw_width")
    if candidate_filter["model_feasible_width"] != candidate_filter["derived_model_feasible_width"]:
        filter_mismatch_reasons.append("model_width")
    if candidate_filter["strict_feasible_width"] != candidate_filter["derived_strict_feasible_width"]:
        filter_mismatch_reasons.append("strict_width")
    if not candidate_filter["strict_set_equals_pred_scored_set"]:
        filter_mismatch_reasons.append("pred_scored_set")
    if not candidate_filter["strict_set_equals_true_scored_set"]:
        filter_mismatch_reasons.append("true_scored_set")
    if not candidate_filter["pred_true_scored_set_equal"]:
        filter_mismatch_reasons.append("pred_true_scored_set")
    if score_presence_mismatch:
        filter_mismatch_reasons.append("score_presence")
    candidate_filter["mismatch_reasons"] = filter_mismatch_reasons
    candidate_filter["any_mismatch"] = bool(filter_mismatch_reasons)

    top1_comparison = {
        "pred_top1_action": action_payload(pred_top.get("action")),
        "true_canonical_top1_action": action_payload(true_top.get("action")),
        "true_top1_tie_actions": [action_from_key(key) for key in true_top_tie_keys],
        "exact_action_agreement": pred_top_key == true_top_key,
        "truth_tie_aware_agreement": pred_top_key in set(true_top_tie_keys),
        "priority_match": pred_score_top["priority"] == true_top_score["priority"],
        "priority_mismatch": pred_score_top["priority"] != true_top_score["priority"],
        "same_priority_action_mismatch": (
            pred_top_key != true_top_key and pred_score_top["priority"] == true_top_score["priority"]
        ),
        "pred_top1_pred_total_ms": pred_score_top["total_ms"],
        "pred_top1_truth_total_ms": pred_top_truth_score["total_ms"],
        "true_top1_pred_total_ms": true_top_pred_score["total_ms"],
        "true_top1_truth_total_ms": true_top_truth_score["total_ms"],
        "raw_truth_total_gap_ms": float(pred_top_truth_score["total_ms"])
        - float(true_top_truth_score["total_ms"]),
        "comparable_truth_regret_ms": (
            float(pred_top_truth_score["total_ms"]) - float(true_top_truth_score["total_ms"])
            if pred_score_top["priority"] == true_top_score["priority"]
            else None
        ),
        "predicted_margin_to_true_top1_ms": float(true_top_pred_score["total_ms"])
        - float(pred_score_top["total_ms"]),
        "raw_truth_priority_pred_top1": pred_top_truth_score["priority"],
        "truth_priority_true_top1": true_top_score["priority"],
        "recorded_pred_action": action_from_key(recorded_pred_action),
        "recorded_true_action": action_from_key(recorded_true_action),
        "recorded_pred_matches_recomputed_top1": recorded_pred_action == pred_top_key,
        "recorded_true_matches_recomputed_top1": recorded_true_action == true_top_key,
    }

    score_error = {
        "pred_top1": error_payload(pred_top_delta),
        "true_top1": error_payload(true_top_delta),
    }
    score_validation = {
        "pred_total_minus_current_plus_future_ms": float(pred_score_top["total_ms"])
        - float(pred_score_top["current_ms"])
        - float(pred_score_top["future_ms"]),
        "true_total_minus_current_plus_future_ms": float(true_top_score["total_ms"])
        - float(true_top_score["current_ms"])
        - float(true_top_score["future_ms"]),
        "shared_priority_fields": sum(
            1
            for candidate in scored_candidates
            if score_payload(candidate["scores"][PRED_POLICY])["priority"]
            == score_payload(candidate["scores"][TRUE_POLICY])["priority"]
        ),
        "shared_tie_fields": sum(
            1
            for candidate in scored_candidates
            if score_payload(candidate["scores"][PRED_POLICY])["ready_since_ms"]
            == score_payload(candidate["scores"][TRUE_POLICY])["ready_since_ms"]
            and score_payload(candidate["scores"][PRED_POLICY])["job_index"]
            == score_payload(candidate["scores"][TRUE_POLICY])["job_index"]
            and score_payload(candidate["scores"][PRED_POLICY])["node_id"]
            == score_payload(candidate["scores"][TRUE_POLICY])["node_id"]
            and score_payload(candidate["scores"][PRED_POLICY])["gpu_index"]
            == score_payload(candidate["scores"][TRUE_POLICY])["gpu_index"]
        ),
    }
    score_validation["all_shared_priority_fields"] = (
        score_validation["shared_priority_fields"] == len(scored_candidates)
    )
    score_validation["all_shared_tie_fields"] = (
        score_validation["shared_tie_fields"] == len(scored_candidates)
    )

    decision = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": str(record.get("episode_id")),
        "decision_index": int(record.get("decision_index") or 0),
        "time_ms": float(record.get("time_ms") or 0.0),
        "state_hash": str(record.get("state_hash")),
        "reference_policy": str(record.get("reference_policy")),
        "strict_width_bucket": str(record.get("strict_width_bucket")),
        "ready_gpu_node_count": int(record.get("ready_gpu_node_count") or 0),
        "free_gpu_count": int(record.get("free_gpu_count") or 0),
        "resident_model_count": int(record.get("resident_model_count") or 0),
        "ready_service_class_counts": dict(record.get("ready_service_class_counts") or {}),
        "candidate_filter": candidate_filter,
        "top1_comparison": top1_comparison,
        "top1_score_error": score_error,
        "score_validation": score_validation,
        "ranking": {
            "scored_candidate_count": len(scored_candidates),
            "spearman_pred_vs_true_order": spearman(pred_positions, true_positions),
            "pair_count": pair_count,
            "inversion_count": inversion_count,
            "inversion_rate": inversion_rate,
            "kendall_tau_like": tau,
            "top_k_overlap": {
                str(k): len(set(pred_keys[:k]) & set(true_keys[:k]))
                for k in (1, 2, 4, 8)
                if len(scored_candidates) >= k
            },
        },
        "pred_top1_candidate": {
            "action": action_payload(pred_top.get("action")),
            "action_key": list(pred_top_key),
            "model_id": pred_top.get("model_id"),
            "role": pred_top.get("role"),
            "action_family": pred_top.get("action_family"),
            "service_class": pred_top.get("service_class"),
            "cache_hit": pred_top.get("cache_hit"),
            "deadline_slack_ms": pred_top.get("deadline_slack_ms"),
        },
        "true_top1_candidate": {
            "action": action_payload(true_top.get("action")),
            "action_key": list(true_top_key),
            "model_id": true_top.get("model_id"),
            "role": true_top.get("role"),
            "action_family": true_top.get("action_family"),
            "service_class": true_top.get("service_class"),
            "cache_hit": true_top.get("cache_hit"),
            "deadline_slack_ms": true_top.get("deadline_slack_ms"),
        },
    }
    if top1_comparison["exact_action_agreement"]:
        decision["issue_label"] = "same_top1"
    elif top1_comparison["priority_mismatch"]:
        decision["issue_label"] = "priority_mismatch"
    elif top1_comparison["truth_tie_aware_agreement"]:
        decision["issue_label"] = "truth_tie_only"
    else:
        decision["issue_label"] = "same_priority_cost_ranking_mismatch"

    # Emit every candidate, including rejected candidates with null scores.
    for ordinal, candidate in enumerate(candidates):
        key = action_candidate_key(candidate)
        stage = filter_stage(candidate)
        row: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": decision["episode_id"],
            "decision_index": decision["decision_index"],
            "time_ms": decision["time_ms"],
            "state_hash": decision["state_hash"],
            "candidate_ordinal": ordinal,
            "action": action_payload(candidate.get("action")),
            "action_key": list(key),
            "model_id": candidate.get("model_id"),
            "role": candidate.get("role"),
            "action_family": candidate.get("action_family"),
            "service_class": candidate.get("service_class"),
            "cache_hit": candidate.get("cache_hit"),
            "deadline_slack_ms": candidate.get("deadline_slack_ms"),
            "deadline_bucket": deadline_bucket(candidate.get("deadline_slack_ms")),
            "raw_feasible": bool(candidate.get("raw_feasible")),
            "model_feasible": bool(candidate.get("model_feasible")),
            "strict_feasible": bool(candidate.get("strict_feasible")),
            "filter_stage": stage,
            "pred_score": None,
            "true_score": None,
            "score_error": None,
            "ranking": {
                "pred_rank": None,
                "true_rank": None,
                "rank_delta_pred_minus_true": None,
                "pred_top1": False,
                "true_canonical_top1": False,
                "true_top1_tie": False,
            },
        }
        if stage == "strict_feasible" and isinstance(candidate.get("scores"), Mapping):
            if PRED_POLICY in candidate["scores"] and TRUE_POLICY in candidate["scores"]:
                pred_score = score_payload(candidate["scores"][PRED_POLICY])
                true_score = score_payload(candidate["scores"][TRUE_POLICY])
                deltas = score_delta(pred_score, true_score)
                pred_additivity_residual = (
                    float(pred_score["total_ms"])
                    - float(pred_score["current_ms"])
                    - float(pred_score["future_ms"])
                )
                true_additivity_residual = (
                    float(true_score["total_ms"])
                    - float(true_score["current_ms"])
                    - float(true_score["future_ms"])
                )
                row["pred_score"] = pred_score
                row["true_score"] = true_score
                row["score_error"] = {
                    "pred_minus_true": error_payload(deltas),
                    "total_additivity_residual_ms": deltas["total_ms"]
                    - deltas["current_ms"]
                    - deltas["future_ms"],
                    "pred_total_additivity_residual_ms": pred_additivity_residual,
                    "true_total_additivity_residual_ms": true_additivity_residual,
                }
                row["ranking"] = {
                    "pred_rank": pred_rank.get(key),
                    "true_rank": true_rank.get(key),
                    "rank_delta_pred_minus_true": (
                        pred_rank.get(key) - true_rank.get(key)
                        if key in pred_rank and key in true_rank
                        else None
                    ),
                    "pred_top1": key == pred_top_key,
                    "true_canonical_top1": key == true_top_key,
                    "true_top1_tie": key in set(true_top_tie_keys),
                }
                global_state["candidate_rows_scored"] += 1
                update_component_stats(global_state["candidate_component_stats"], deltas)
                global_state["candidate_pred_additivity"].append(pred_additivity_residual)
                global_state["candidate_true_additivity"].append(true_additivity_residual)
                global_state["candidate_delta_additivity"].append(
                    deltas["total_ms"] - deltas["current_ms"] - deltas["future_ms"]
                )
                global_state["candidate_priority_mismatch_rows"] += int(
                    pred_score["priority"] != true_score["priority"]
                )
                global_state["candidate_tie_field_mismatch_rows"] += int(
                    not (
                        pred_score["ready_since_ms"] == true_score["ready_since_ms"]
                        and pred_score["job_index"] == true_score["job_index"]
                        and pred_score["node_id"] == true_score["node_id"]
                        and pred_score["gpu_index"] == true_score["gpu_index"]
                    )
                )
                for field in GROUP_FIELDS:
                    if field == "strict_width_bucket":
                        value = record.get("strict_width_bucket")
                    else:
                        value = row.get(field)
                    update_group(global_state["candidate_groups"], field, value, deltas)
                for component in COMPONENTS:
                    global_state["candidate_signed"][component].append(deltas[component])
                    global_state["candidate_abs"][component].append(abs(deltas[component]))
                if row["ranking"]["pred_top1"]:
                    global_state["pred_top1_candidate_error"]["current_ms"].append(deltas["current_ms"])
                    global_state["pred_top1_candidate_error"]["future_ms"].append(deltas["future_ms"])
                    global_state["pred_top1_candidate_error"]["total_ms"].append(deltas["total_ms"])
                if row["ranking"]["true_canonical_top1"]:
                    global_state["true_top1_candidate_error"]["current_ms"].append(deltas["current_ms"])
                    global_state["true_top1_candidate_error"]["future_ms"].append(deltas["future_ms"])
                    global_state["true_top1_candidate_error"]["total_ms"].append(deltas["total_ms"])
                metric = abs(deltas["total_ms"])
                entry = (metric, global_state["sequence"], row)
                global_state["sequence"] += 1
                heapq.heappush(global_state["top_candidate_heap"], entry)
                if len(global_state["top_candidate_heap"]) > top_k:
                    heapq.heappop(global_state["top_candidate_heap"])
        jsonl_write(candidate_handle, row)

    global_state["decisions"] += 1
    global_state["candidate_rows"] += len(candidates)
    global_state["strict_candidate_rows"] += len(strict_candidates)
    global_state["filter_stage_counts"].update(filter_counts)
    global_state["filter_mismatch_decisions"] += int(bool(candidate_filter["any_mismatch"]))
    global_state["pred_true_scored_set_mismatch_decisions"] += int(
        not candidate_filter["pred_true_scored_set_equal"]
    )
    global_state["score_presence_mismatch_rows"] += score_presence_mismatch
    global_state["exact_top1_agreement"] += int(top1_comparison["exact_action_agreement"])
    global_state["truth_tie_aware_agreement"] += int(top1_comparison["truth_tie_aware_agreement"])
    global_state["priority_match"] += int(top1_comparison["priority_match"])
    global_state["priority_mismatch"] += int(top1_comparison["priority_mismatch"])
    global_state["same_priority_action_mismatch"] += int(top1_comparison["same_priority_action_mismatch"])
    spearman_value = decision["ranking"]["spearman_pred_vs_true_order"]
    if spearman_value is not None:
        global_state["spearman_values"].append(float(spearman_value))
    global_state["inversion_counts"] += inversion_count
    global_state["pair_counts"] += pair_count
    global_state["inversion_rates"].append(inversion_rate)
    for k, overlap in decision["ranking"]["top_k_overlap"].items():
        global_state["top_k_overlap"][k].append(float(overlap))
    global_state["raw_truth_total_gaps"].append(float(top1_comparison["raw_truth_total_gap_ms"]))
    comparable_regret = top1_comparison.get("comparable_truth_regret_ms")
    if comparable_regret is not None:
        global_state["comparable_truth_regrets"].append(float(comparable_regret))
    global_state["pred_top1_error"]["current_ms"].append(pred_top_delta["current_ms"])
    global_state["pred_top1_error"]["future_ms"].append(pred_top_delta["future_ms"])
    global_state["pred_top1_error"]["total_ms"].append(pred_top_delta["total_ms"])
    global_state["true_top1_error"]["current_ms"].append(true_top_delta["current_ms"])
    global_state["true_top1_error"]["future_ms"].append(true_top_delta["future_ms"])
    global_state["true_top1_error"]["total_ms"].append(true_top_delta["total_ms"])
    global_state["score_additivity_pred"].append(score_validation["pred_total_minus_current_plus_future_ms"])
    global_state["score_additivity_true"].append(score_validation["true_total_minus_current_plus_future_ms"])
    global_state["shared_priority_field_counts"].append(score_validation["shared_priority_fields"])
    global_state["shared_tie_field_counts"].append(score_validation["shared_tie_fields"])
    global_state["shared_priority_field_incomplete_decisions"] += int(
        not score_validation["all_shared_priority_fields"]
    )
    global_state["shared_tie_field_incomplete_decisions"] += int(
        not score_validation["all_shared_tie_fields"]
    )
    global_state["recorded_pred_matches_recomputed_top1"] += int(
        top1_comparison["recorded_pred_matches_recomputed_top1"]
    )
    global_state["recorded_true_matches_recomputed_top1"] += int(
        top1_comparison["recorded_true_matches_recomputed_top1"]
    )
    update_decision_groups(global_state["decision_groups"], decision)
    mismatch_metric = max(
        abs(float(top1_comparison["raw_truth_total_gap_ms"])),
        abs(float(pred_top_delta["total_ms"])),
    )
    if not top1_comparison["exact_action_agreement"]:
        entry = (mismatch_metric, global_state["decision_sequence"], decision)
        global_state["decision_sequence"] += 1
        heapq.heappush(global_state["top_decision_heap"], entry)
        if len(global_state["top_decision_heap"]) > top_k:
            heapq.heappop(global_state["top_decision_heap"])
    return decision


def fresh_state() -> Dict[str, Any]:
    return {
        "decisions": 0,
        "candidate_rows": 0,
        "strict_candidate_rows": 0,
        "candidate_rows_scored": 0,
        "filter_stage_counts": Counter(),
        "filter_mismatch_decisions": 0,
        "pred_true_scored_set_mismatch_decisions": 0,
        "score_presence_mismatch_rows": 0,
        "exact_top1_agreement": 0,
        "truth_tie_aware_agreement": 0,
        "priority_match": 0,
        "priority_mismatch": 0,
        "same_priority_action_mismatch": 0,
        "spearman_values": [],
        "inversion_counts": 0,
        "pair_counts": 0,
        "inversion_rates": [],
        "top_k_overlap": defaultdict(list),
        "raw_truth_total_gaps": [],
        "comparable_truth_regrets": [],
        "pred_top1_error": defaultdict(list),
        "true_top1_error": defaultdict(list),
        "pred_top1_candidate_error": defaultdict(list),
        "true_top1_candidate_error": defaultdict(list),
        "score_additivity_pred": [],
        "score_additivity_true": [],
        "candidate_pred_additivity": [],
        "candidate_true_additivity": [],
        "candidate_delta_additivity": [],
        "candidate_priority_mismatch_rows": 0,
        "candidate_tie_field_mismatch_rows": 0,
        "shared_priority_field_counts": [],
        "shared_tie_field_counts": [],
        "shared_priority_field_incomplete_decisions": 0,
        "shared_tie_field_incomplete_decisions": 0,
        "recorded_pred_matches_recomputed_top1": 0,
        "recorded_true_matches_recomputed_top1": 0,
        "candidate_component_stats": new_component_stats(),
        "candidate_signed": defaultdict(list),
        "candidate_abs": defaultdict(list),
        "candidate_groups": {field: {} for field in GROUP_FIELDS},
        "decision_groups": {field: {} for field in GROUP_FIELDS},
        "top_candidate_heap": [],
        "top_decision_heap": [],
        "sequence": 0,
        "decision_sequence": 0,
    }


def finalize_groups(groups: Mapping[str, Mapping[str, Mapping[str, Any]]], decision: bool = False) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for field, values in groups.items():
        result[field] = {}
        for value, stats in sorted(values.items()):
            result[field][value] = (
                finalize_decision_group_stats(stats)
                if decision
                else finalize_component_stats(stats)
            )
    return result


def state_metrics(state: Mapping[str, Any]) -> Dict[str, Any]:
    decisions = int(state["decisions"])
    strict_rows = int(state["strict_candidate_rows"])
    scored_rows = int(state["candidate_rows_scored"])
    candidate_error_distributions: Dict[str, Any] = {}
    for component in COMPONENTS:
        candidate_error_distributions[component] = {
            "signed": distribution(state["candidate_signed"][component]),
            "absolute": distribution(state["candidate_abs"][component]),
        }
    pred_top1_error = {
        component: distribution(state["pred_top1_error"][component]) for component in COMPONENTS
    }
    true_top1_error = {
        component: distribution(state["true_top1_error"][component]) for component in COMPONENTS
    }
    pred_top1_candidate_error = {
        component: distribution(state["pred_top1_candidate_error"][component]) for component in COMPONENTS
    }
    true_top1_candidate_error = {
        component: distribution(state["true_top1_candidate_error"][component]) for component in COMPONENTS
    }
    additivity_tolerance = 1e-6
    return {
        "counts": {
            "decisions": decisions,
            "candidate_rows": int(state["candidate_rows"]),
            "strict_candidate_rows": strict_rows,
            "scored_candidate_rows": scored_rows,
            "mean_candidates_per_decision": float(state["candidate_rows"]) / decisions if decisions else None,
            "mean_strict_candidates_per_decision": float(strict_rows) / decisions if decisions else None,
        },
        "candidate_filtering": {
            "filter_stage_counts": dict(sorted(state["filter_stage_counts"].items())),
            "filter_mismatch_decisions": int(state["filter_mismatch_decisions"]),
            "filter_mismatch_rate": (
                float(state["filter_mismatch_decisions"]) / decisions if decisions else None
            ),
            "pred_true_scored_set_mismatch_decisions": int(state["pred_true_scored_set_mismatch_decisions"]),
            "score_presence_mismatch_rows": int(state["score_presence_mismatch_rows"]),
            "interpretation": (
                "The source audit stores one shared strict-feasibility filter; this audit verifies that "
                "both aligned score maps are present on exactly that same action set."
            ),
        },
        "top1": {
            "pred_vs_true_canonical_exact_agreement_rate": (
                float(state["exact_top1_agreement"]) / decisions if decisions else None
            ),
            "pred_top1_in_true_tie_agreement_rate": (
                float(state["truth_tie_aware_agreement"]) / decisions if decisions else None
            ),
            "priority_match_rate": float(state["priority_match"]) / decisions if decisions else None,
            "priority_mismatch_rate": float(state["priority_mismatch"]) / decisions if decisions else None,
            "same_priority_action_mismatch_rate": (
                float(state["same_priority_action_mismatch"]) / decisions if decisions else None
            ),
            "recorded_policy_top1_reproduction": {
                "pred": {
                    "matches": int(state["recorded_pred_matches_recomputed_top1"]),
                    "rate": (
                        float(state["recorded_pred_matches_recomputed_top1"]) / decisions
                        if decisions
                        else None
                    ),
                },
                "true": {
                    "matches": int(state["recorded_true_matches_recomputed_top1"]),
                    "rate": (
                        float(state["recorded_true_matches_recomputed_top1"]) / decisions
                        if decisions
                        else None
                    ),
                },
            },
            "raw_truth_total_gap_ms": distribution(state["raw_truth_total_gaps"]),
            "comparable_truth_regret_ms": distribution(state["comparable_truth_regrets"]),
        },
        "ranking": {
            "mean_spearman_pred_vs_true_order": mean_or_none(state["spearman_values"]),
            "spearman_distribution": distribution(state["spearman_values"]),
            "pair_count": int(state["pair_counts"]),
            "inversion_count": int(state["inversion_counts"]),
            "pooled_inversion_rate": (
                float(state["inversion_counts"]) / state["pair_counts"]
                if state["pair_counts"]
                else None
            ),
            "mean_decision_inversion_rate": mean_or_none(state["inversion_rates"]),
            "top_k_overlap_mean": {
                key: mean_or_none(values) for key, values in sorted(state["top_k_overlap"].items())
            },
        },
        "score_error": {
            "candidate_level": candidate_error_distributions,
            "pred_top1_candidate": pred_top1_candidate_error,
            "true_top1_candidate": true_top1_candidate_error,
            "pred_top1_decision": pred_top1_error,
            "true_top1_decision": true_top1_error,
            "candidate_component_summary": finalize_component_stats(state["candidate_component_stats"]),
        },
        "score_contract_validation": {
            "pred_additivity_residual_ms": distribution(state["score_additivity_pred"]),
            "true_additivity_residual_ms": distribution(state["score_additivity_true"]),
            "max_abs_pred_additivity_residual_ms": max(
                (abs(value) for value in state["score_additivity_pred"]), default=0.0
            ),
            "max_abs_true_additivity_residual_ms": max(
                (abs(value) for value in state["score_additivity_true"]), default=0.0
            ),
            "additivity_pass": all(
                abs(value) <= additivity_tolerance for value in state["score_additivity_pred"]
            )
            and all(abs(value) <= additivity_tolerance for value in state["score_additivity_true"]),
            "candidate_pred_additivity_residual_ms": distribution(
                state["candidate_pred_additivity"]
            ),
            "candidate_true_additivity_residual_ms": distribution(
                state["candidate_true_additivity"]
            ),
            "candidate_delta_additivity_residual_ms": distribution(
                state["candidate_delta_additivity"]
            ),
            "candidate_additivity_pass": all(
                abs(value) <= additivity_tolerance for value in state["candidate_pred_additivity"]
            )
            and all(
                abs(value) <= additivity_tolerance for value in state["candidate_true_additivity"]
            ),
            "candidate_priority_field_mismatch_rows": int(
                state["candidate_priority_mismatch_rows"]
            ),
            "candidate_tie_field_mismatch_rows": int(
                state["candidate_tie_field_mismatch_rows"]
            ),
            "shared_priority_field_rows": distribution(
                [float(value) for value in state["shared_priority_field_counts"]]
            ),
            "shared_tie_field_rows": distribution(
                [float(value) for value in state["shared_tie_field_counts"]]
            ),
            "shared_priority_field_incomplete_decisions": int(
                state["shared_priority_field_incomplete_decisions"]
            ),
            "shared_tie_field_incomplete_decisions": int(
                state["shared_tie_field_incomplete_decisions"]
            ),
        },
        "by_candidate": finalize_groups(state["candidate_groups"], decision=False),
        "by_decision_top1_context": finalize_groups(state["decision_groups"], decision=True),
    }


def clean_heap(heap: Sequence[Tuple[float, int, Mapping[str, Any]]]) -> List[Mapping[str, Any]]:
    return [entry[2] for entry in sorted(heap, key=lambda item: (-item[0], item[1]))]


def read_records(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("invalid JSON at line %d: %s" % (line_number, exc))
            if not isinstance(row, Mapping):
                raise ValueError("record at line %d is not an object" % line_number)
            yield row


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if not args.audit_records.exists():
        parser.error("audit records do not exist: %s" % args.audit_records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    state = fresh_state()
    input_hash = sha256(args.audit_records)
    decision_path = args.output_dir / "decision_comparisons.jsonl"
    candidate_path = args.output_dir / "candidate_score_comparisons.jsonl"
    try:
        with decision_path.open("w", encoding="utf-8") as decision_handle, candidate_path.open(
            "w", encoding="utf-8"
        ) as candidate_handle:
            for record in read_records(args.audit_records):
                decision = compare_decision(record, candidate_handle, args.top_k, state)
                jsonl_write(decision_handle, decision)
    except Exception:
        # A partial output is not presented as a completed experiment.
        raise

    if state["decisions"] == 0:
        raise ValueError("no decisions found in audit records")
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "passed"
            if state["filter_mismatch_decisions"] == 0
            and state["pred_true_scored_set_mismatch_decisions"] == 0
            and state["score_presence_mismatch_rows"] == 0
            and state_metrics(state)["score_contract_validation"]["additivity_pass"]
            and state_metrics(state)["score_contract_validation"]["candidate_additivity_pass"]
            and state_metrics(state)["score_contract_validation"][
                "candidate_priority_field_mismatch_rows"
            ]
            == 0
            and state_metrics(state)["score_contract_validation"][
                "candidate_tie_field_mismatch_rows"
            ]
            == 0
            else "failed"
        ),
        "source": {
            "audit_records": str(args.audit_records),
            "audit_records_sha256": input_hash,
            "pred_policy": PRED_POLICY,
            "true_policy": TRUE_POLICY,
            "post_hoc_only": True,
            "truth_values_passed_to_policy": False,
        },
        "metrics": state_metrics(state),
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    top_decisions = clean_heap(state["top_decision_heap"])
    top_candidates = clean_heap(state["top_candidate_heap"])
    with (args.output_dir / "top_decision_mismatches.jsonl").open("w", encoding="utf-8") as handle:
        for row in top_decisions:
            jsonl_write(handle, row)
    with (args.output_dir / "top_candidate_score_errors.jsonl").open("w", encoding="utf-8") as handle:
        for row in top_candidates:
            jsonl_write(handle, row)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": metrics["status"],
        "audit_records": str(args.audit_records),
        "audit_records_sha256": input_hash,
        "output_files": {
            "metrics": str(args.output_dir / "metrics.json"),
            "decision_comparisons": str(decision_path),
            "candidate_score_comparisons": str(candidate_path),
            "top_decision_mismatches": str(args.output_dir / "top_decision_mismatches.jsonl"),
            "top_candidate_score_errors": str(args.output_dir / "top_candidate_score_errors.jsonl"),
            "topology_metrics": str(args.output_dir / "topology_metrics.json"),
            "topology_node_summaries": str(args.output_dir / "topology_node_summaries.jsonl"),
        },
        "counts": metrics["metrics"]["counts"],
        "top_k": args.top_k,
    }
    with (args.output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps({"status": metrics["status"], "counts": metrics["metrics"]["counts"]}, ensure_ascii=False))
    return 0 if metrics["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
