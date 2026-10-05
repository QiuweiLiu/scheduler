"""PDRS — Prefix-Conditioned Distributional Receding-Horizon Scheduling.

Two consumptions of the frozen per-instance future belief, both WITHOUT any
machine-response table (no measured slowdown / interference / batching values):

* :func:`suffix_expected_cost` — survival-weighted expected remaining work used
  for node ordering:

      W_i = sum_{k=1..H} q_k * [ E[T_k] + p_load_k * p95(L_k) ],   q_k = P(L >= k)

  ``q_k`` comes from the frozen chain-length distribution, ``E[T_k]`` from the
  predicted runtime mean, ``p_load_k`` / ``p95(L_k)`` from the predicted load
  occurrence probability and load-duration p95.

* :func:`placement_affinity` — the H=1 next-model residency affinity used for
  GPU placement:

      A(i,g) = q_1 * p_load_1 * p95(L_1) * [1 - P_hit(i,g)]
      P_hit  = sum_m P(M_1 = m) * 1[m in resident_after(i,g)]

  i.e. the expected avoidable loading cost of the NEXT step given the models
  that GPU ``g`` holds after this placement.

Every field is read fail-closed: a row that carries future steps but is missing
a required field raises instead of silently degrading into a different consumer.
Rows without future steps (chain tails) legitimately contribute zero.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

HORIZON = 5


def _row(future_artifacts: Mapping[str, Mapping[str, Any]], node_id: str) -> Mapping[str, Any]:
    return future_artifacts.get(str(node_id)) or {}


def _steps(row: Mapping[str, Any], horizon: int) -> list[Mapping[str, Any]]:
    scenarios = row.get(f"future_h{int(horizon)}") or []
    if not scenarios:
        return []
    return list(scenarios[0].get("steps") or [])


def survival_weights(row: Mapping[str, Any], horizon: int = HORIZON) -> list[float]:
    """q_k = P(L >= k) for k = 1..horizon from the frozen length distribution."""

    probs = [float(value) for value in (row.get("length_probabilities") or [])]
    if not probs:
        raise ValueError("PDRS requires length_probabilities")
    if len(probs) < horizon + 1:
        raise ValueError(
            f"PDRS requires length_probabilities for 0..{horizon}, got {len(probs)} values"
        )
    return [sum(probs[k:]) for k in range(1, horizon + 1)]


def _required_number(step: Mapping[str, Any], key: str) -> float:
    resource = step.get("resource") or {}
    value = resource.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"PDRS requires resource.{key}; got {value!r}")
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")) or value < 0.0:
        raise ValueError(f"PDRS requires a finite non-negative resource.{key}; got {value!r}")
    return value


def _load_p95(step: Mapping[str, Any]) -> float:
    resource = step.get("resource") or {}
    value = (resource.get("load_duration_ms_quantiles") or {}).get("p95")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"PDRS requires load_duration_ms_quantiles.p95; got {value!r}")
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")) or value < 0.0:
        raise ValueError(f"PDRS requires a finite non-negative load p95; got {value!r}")
    return value


def suffix_expected_cost(
    future_artifacts: Mapping[str, Mapping[str, Any]],
    node_id: str,
    horizon: int = HORIZON,
) -> float:
    """Survival-weighted expected remaining work W_i (ordering term)."""

    row = _row(future_artifacts, node_id)
    steps = _steps(row, horizon)
    if not steps:
        return 0.0
    weights = survival_weights(row, horizon)
    total = 0.0
    for index, step in enumerate(steps[:horizon]):
        runtime_mean = _required_number(step, "runtime_mean_ms")
        load_prob = _required_number(step, "load_occurrence_probability")
        total += weights[index] * (runtime_mean + load_prob * _load_p95(step))
    return total


def next_model_distribution(
    future_artifacts: Mapping[str, Mapping[str, Any]],
    node_id: str,
    horizon: int = HORIZON,
) -> dict[str, float]:
    """P(M_1 = m): the predicted next-step model distribution."""

    row = _row(future_artifacts, node_id)
    steps = _steps(row, horizon)
    if not steps:
        return {}
    raw = steps[0].get("model_probabilities") or {}
    if not raw:
        raise ValueError("PDRS requires model_probabilities on the first future step")
    return {str(model): float(prob) for model, prob in raw.items()}


def placement_affinity_for_distribution(
    future_artifacts: Mapping[str, Mapping[str, Any]],
    node_id: str,
    distribution: Mapping[str, float],
    resident_after: Iterable[str],
    horizon: int = HORIZON,
) -> float:
    """A(i,g) for an explicit next-model distribution (used by the oracle arm)."""

    row = _row(future_artifacts, node_id)
    steps = _steps(row, horizon)
    if not steps or not distribution:
        return 0.0
    weights = survival_weights(row, horizon)
    step = steps[0]
    load_prob = _required_number(step, "load_occurrence_probability")
    load_cost = _load_p95(step)
    if load_prob <= 0.0 or load_cost <= 0.0:
        return 0.0
    residents = {str(model) for model in resident_after}
    hit = sum(float(prob) for model, prob in distribution.items() if str(model) in residents)
    return weights[0] * load_prob * load_cost * (1.0 - hit)


def placement_affinity(
    future_artifacts: Mapping[str, Mapping[str, Any]],
    node_id: str,
    resident_after: Iterable[str],
    horizon: int = HORIZON,
) -> float:
    """H=1 expected avoidable next-step loading cost A(i,g) (placement term)."""

    distribution = next_model_distribution(future_artifacts, node_id, horizon)
    return placement_affinity_for_distribution(
        future_artifacts, node_id, distribution, resident_after, horizon
    )

def build_prior_artifacts(
    future_artifacts: Mapping[str, Mapping[str, Any]],
    horizon: int = HORIZON,
) -> dict[str, dict[str, Any]]:
    """Unconditional-prior control: pooled marginals replace instance beliefs.

    Keeps every per-step runtime/load field untouched; replaces only the
    instance-specific ``model_probabilities`` of the first future step and the
    chain ``length_probabilities`` with pooled averages over all frozen rows.
    This is the matched control for "instance-conditioned vs prior" (the
    SAGA/LLMSched-style historical/aggregate alternative).
    """

    model_acc: dict[str, float] = {}
    model_n = 0
    length_acc: list[float] | None = None
    length_n = 0
    for row in future_artifacts.values():
        steps = _steps(row, horizon)
        if steps:
            distribution = steps[0].get("model_probabilities") or {}
            if distribution:
                for model, probability in distribution.items():
                    model_acc[str(model)] = model_acc.get(str(model), 0.0) + float(probability)
                model_n += 1
        probs = row.get("length_probabilities") or []
        if probs:
            if length_acc is None:
                length_acc = [0.0] * len(probs)
            for index, probability in enumerate(probs):
                length_acc[index] += float(probability)
            length_n += 1
    prior_model = {model: value / model_n for model, value in model_acc.items()} if model_n else {}
    prior_length = [value / length_n for value in length_acc] if (length_acc and length_n) else None

    result: dict[str, dict[str, Any]] = {}
    for node_id, row in future_artifacts.items():
        new_row = dict(row)
        scenarios = row.get(f"future_h{horizon}") or []
        if scenarios and prior_model:
            steps = [dict(step) for step in (scenarios[0].get("steps") or [])]
            if steps:
                steps[0] = {**steps[0], "model_probabilities": dict(prior_model)}
                new_row[f"future_h{horizon}"] = [{**scenarios[0], "steps": steps}]
        if prior_length is not None:
            new_row["length_probabilities"] = list(prior_length)
        result[str(node_id)] = new_row
    return result


def build_shuffled_artifacts(
    future_artifacts: Mapping[str, Mapping[str, Any]],
    seed: int = 11,
) -> dict[str, dict[str, Any]]:
    """Shuffled-belief control: same rows, wrong instance.

    Every node receives another node's row (deterministic permutation), which
    preserves the pooled distribution shape while destroying prefix-specific
    information.
    """

    import random

    node_ids = sorted(str(node_id) for node_id in future_artifacts)
    permutation = list(node_ids)
    random.Random(seed).shuffle(permutation)
    return {target: future_artifacts[source] for target, source in zip(node_ids, permutation)}
