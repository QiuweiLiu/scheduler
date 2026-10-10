"""Residency belief encoder and policy actions (round-6 residency experiment).

Unified belief interface (GPT round-6 design): for instance ``j``, future step
``k``, model ``m``, the demand quality is

    D(j, k, m) = q_k * p_k(m) * l_k

and the two consumptions differ ONLY in ``(q, p, l)``:

* ``f0point``: q=1, p=one-hot(argmax model_probabilities),
  l = 1[load_occurrence_probability >= 0.5] * p95(load)
* ``pdrs``:    q=P(L >= k), p=full model_probabilities,
  l = load_occurrence_probability * p95(load)

All residency action rules are IDENTICAL across consumptions; only the demand
values differ:

* prefetch demand   U(m) = sum_j D(j, 1, m)
* residency value   V(m) = sum_j sum_k D(j, k, m)
* eviction subset   exact enumeration: min total V, then min excess freed
                    memory, then lexicographic model-id tuple
* prefetch target   argmax U; device = max free memory among no-eviction fits

Every read is fail-closed with the same policy as ``pdrs_methods``: a row that
carries future steps but is missing a required field raises instead of silently
degrading into a different consumer.  Rows without future steps (chain tails)
legitimately contribute zero.
"""

from __future__ import annotations

import itertools
from typing import Any, Iterable, Mapping, Sequence

from tracing.analysis.pdrs_methods import (
    HORIZON,
    _load_p95,
    _required_number,
    _row,
    _steps,
    survival_weights,
)

MODES = ("f0point", "pdrs", "oracle")

LOAD_THRESHOLD = 0.5


def _model_probabilities(step: Mapping[str, Any]) -> dict[str, float]:
    raw = step.get("model_probabilities") or {}
    if not raw:
        raise ValueError("residency belief requires model_probabilities on the future step")
    return {str(model): float(prob) for model, prob in raw.items()}


def step_demand(
    step: Mapping[str, Any],
    mode: str,
    survival_q: float,
) -> dict[str, float]:
    """D(k, m) for one future step under the requested consumption."""

    if mode not in MODES:
        raise ValueError(f"unknown residency belief mode: {mode!r}")
    if mode == "oracle":
        raise ValueError("oracle demand needs the realized step identity; use oracle_demand()")
    load_prob = _required_number(step, "load_occurrence_probability")
    load_cost = _load_p95(step)
    if mode == "f0point":
        if load_prob < LOAD_THRESHOLD:
            return {}
        distribution = _model_probabilities(step)
        top_model = max(sorted(distribution), key=lambda model: (distribution[model], model))
        value = load_cost
        if value <= 0.0:
            return {}
        return {top_model: value}
    # pdrs: survival-weighted full distribution with soft load occurrence
    distribution = _model_probabilities(step)
    scale = survival_q * load_prob * load_cost
    if scale <= 0.0:
        return {}
    return {model: scale * prob for model, prob in distribution.items()}


def demand_quality(
    future_artifacts: Mapping[str, Mapping[str, Any]],
    node_id: str,
    mode: str,
    horizon: int = HORIZON,
) -> list[dict[str, float]]:
    """Per future step k=1..H: {model: D(k, m)} for one instance."""

    row = _row(future_artifacts, node_id)
    steps = _steps(row, horizon)
    if not steps:
        return []
    weights = survival_weights(row, horizon)
    return [step_demand(step, mode, weights[index]) for index, step in enumerate(steps[:horizon])]


def oracle_demand(
    future_artifacts: Mapping[str, Mapping[str, Any]],
    node_id: str,
    true_steps: Sequence[Mapping[str, float]],
    horizon: int = HORIZON,
) -> list[dict[str, float]]:
    """Realized-identity demand: true one-hot per step, frozen predicted load scale.

    ``true_steps`` carries the realized future model distributions (one-hot on a
    serial chain).  The load value still comes from the frozen predictor p95 /
    occurrence probability, so the oracle isolates the value of perfect IDENTITY
    knowledge without injecting execution-duration truth.
    """

    row = _row(future_artifacts, node_id)
    steps = _steps(row, horizon)
    out: list[dict[str, float]] = []
    for index in range(min(horizon, len(steps))):
        if index >= len(true_steps) or not true_steps[index]:
            out.append({})
            continue
        step = steps[index]
        load_prob = _required_number(step, "load_occurrence_probability")
        load_cost = _load_p95(step)
        scale = load_prob * load_cost
        if scale <= 0.0:
            out.append({})
            continue
        out.append(
            {str(model): scale * float(prob) for model, prob in true_steps[index].items()}
        )
    return out


def totals_from_demand(
    demand: Sequence[Mapping[str, float]],
    next_steps: int = 1,
) -> tuple[dict[str, float], dict[str, float]]:
    """(V totals over all steps, U totals over the first ``next_steps`` steps).

    ``next_steps`` defaults to the frozen single-step next-demand; the
    sensitivity variant uses 2 (prefetch target scored on the cumulative
    demand of the next two steps).
    """

    values: dict[str, float] = {}
    nexts: dict[str, float] = {}
    window = max(1, int(next_steps))
    for index, step in enumerate(demand):
        for model, value in step.items():
            values[model] = values.get(model, 0.0) + value
            if index < window:
                nexts[model] = nexts.get(model, 0.0) + value
    return values, nexts


def choose_eviction_subset(
    evictable_memory_mb: Mapping[str, float],
    values: Mapping[str, float],
    required_free_mb: float,
) -> tuple[str, ...]:
    """Exact minimal-V eviction subset that frees ``required_free_mb``.

    Tie-breaks: min total value, then min excess freed memory, then
    lexicographic model-id tuple.  When no subset frees enough, the full
    evictable set is returned and the caller's projection check decides
    admission (never silently succeeds).
    """

    if required_free_mb <= 1e-9:
        return ()
    models = sorted(str(model) for model in evictable_memory_mb)
    if not models:
        return ()
    best_key: tuple[float, float, tuple[str, ...]] | None = None
    best_subset: tuple[str, ...] = ()
    for size in range(1, len(models) + 1):
        for subset in itertools.combinations(models, size):
            freed = sum(float(evictable_memory_mb[model]) for model in subset)
            if freed < required_free_mb - 1e-9:
                continue
            key = (
                sum(float(values.get(model, 0.0)) for model in subset),
                freed - required_free_mb,
                subset,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_subset = subset
    if best_key is None:
        return tuple(models)
    return best_subset


def choose_prefetch_gpu(
    free_memory_mb: Mapping[int, float],
    model_mb: float,
) -> int | None:
    """Device with the most free memory among no-eviction fits; tie -> gpu id."""

    fits = [
        (int(index), float(free))
        for index, free in free_memory_mb.items()
        if float(free) >= float(model_mb) - 1e-9
    ]
    if not fits:
        return None
    return min(fits, key=lambda item: (-item[1], item[0]))[0]


def eviction_penalty(
    evicted: Iterable[str],
    values: Mapping[str, float],
) -> float:
    """Total V of a planned eviction (used for the residency placement rule)."""

    return sum(float(values.get(str(model), 0.0)) for model in evicted)
