"""Hermes-PDGraph-adapted helpers (frozen 2026-10-04; revised after review 2424f67).

Adaptation of Hermes (ACM TACO 2026) to the single-GPU node-level substrate.

Review-driven design (see the round-2 review of 2424f67):

* the PDGraph is conditioned on the **workflow/application type** (the raw
  ``workflow_type_id``), not on a global mixture of all applications;
* historical tuples are kept **aligned**: for each (workflow, revealed_count,
  last_role) state we store ``(prefix_demand, suffix_demand, next_role,
  next_model)`` so a completed node's *observed* demand can filter the tuples;
* refinement is **gated by the Pearson correlation** between prefix and suffix
  demand (only rho > 0.5 lets completed observations enter the conditioning);
  below the gate the unconditional state distribution is used;
* the downstream unit for prewarm comes from the **conditional joint**
  (next_role, next_model) distribution of the selected tuples, not from a
  global role->model majority;
* the Gittins index is the paper's exact empirical ``G(D, 0)`` (lower = higher
  priority); the earlier equal-mass 10-point grid is kept only as an appendix
  sensitivity.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

PEARSON_GATE = 0.5
REFINEMENT_TOLERANCE = 0.25
MIN_REFINED_TUPLES = 3


def _pearson(pairs: Sequence[tuple[float, float]]) -> float | None:
    if len(pairs) < 5:
        return None
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    covariance = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    variance_x = sum((x - mx) ** 2 for x in xs)
    variance_y = sum((y - my) ** 2 for y in ys)
    if variance_x <= 0.0 or variance_y <= 0.0:
        return None
    return covariance / math.sqrt(variance_x * variance_y)


def build_pdgraph(templates: Mapping[str, Any]) -> dict[str, Any]:
    """Workflow-conditioned PDGraph over aligned historical demand tuples."""

    states: dict[tuple[str, int, str], list[tuple[float, float, str, str]]] = defaultdict(list)
    by_workflow_count: dict[tuple[str, int], list[tuple[float, float, str, str]]] = defaultdict(list)
    by_count: dict[int, list[tuple[float, float, str, str]]] = defaultdict(list)
    global_tuples: list[tuple[float, float, str, str]] = []
    role_models: dict[str, Counter[str]] = defaultdict(Counter)
    for template in templates.values():
        if str(getattr(template, "split", "")) != "train":
            continue
        nodes = list(template.nodes)
        if not nodes:
            continue
        workflow = str(getattr(template, "workflow_type_id", "") or "")
        durations = [float(node.runtime_ms) for node in nodes]
        # State with nothing revealed yet: the remaining demand is the FULL
        # template and the next unit is the first node.
        total = sum(durations)
        first = nodes[0]
        zero_entry = (0.0, total, str(first.role), str(first.model_id))
        states[(workflow, 0, "")].append(zero_entry)
        by_workflow_count[(workflow, 0)].append(zero_entry)
        by_count[0].append(zero_entry)
        global_tuples.append(zero_entry)
        for index, node in enumerate(nodes):
            prefix = sum(durations[: index + 1])
            suffix = sum(durations[index + 1:])
            if index + 1 < len(nodes):
                next_node = nodes[index + 1]
                entry = (prefix, suffix, str(next_node.role), str(next_node.model_id))
            else:
                entry = (prefix, suffix, "", "")
            state = (workflow, index + 1, str(node.role))
            states[state].append(entry)
            by_workflow_count[(workflow, index + 1)].append(entry)
            by_count[index + 1].append(entry)
            global_tuples.append(entry)
            role_models[str(node.role)][str(node.model_id)] += 1
    rho: dict[tuple, float | None] = {}
    for key, entries in states.items():
        rho[("state",) + key] = _pearson([(e[0], e[1]) for e in entries])
    for key, entries in by_workflow_count.items():
        rho[("wf",) + key] = _pearson([(e[0], e[1]) for e in entries])
    for key, entries in by_count.items():
        rho[("count", key)] = _pearson([(e[0], e[1]) for e in entries])
    rho[("global",)] = _pearson([(e[0], e[1]) for e in global_tuples])
    return {
        "states": {key: list(entries) for key, entries in states.items()},
        "by_workflow_count": {key: list(entries) for key, entries in by_workflow_count.items()},
        "by_count": {key: list(entries) for key, entries in by_count.items()},
        "global": global_tuples,
        "role_models": {key: dict(counter) for key, counter in role_models.items()},
        "rho": rho,
    }


def _lookup(
    graph: Mapping[str, Any],
    workflow: str,
    revealed_count: int,
    last_role: str,
) -> tuple[list[tuple[float, float, str, str]], tuple | None]:
    """Return (tuples, rho_key) with the fallback chain exact -> workflow+count
    -> count -> global."""

    states = graph.get("states") or {}
    key = (str(workflow), int(revealed_count), str(last_role))
    if states.get(key):
        return list(states[key]), ("state",) + key
    wf = (graph.get("by_workflow_count") or {}).get((str(workflow), int(revealed_count)))
    if wf:
        return list(wf), ("wf", str(workflow), int(revealed_count))
    by_count = (graph.get("by_count") or {}).get(int(revealed_count))
    if by_count:
        return list(by_count), ("count", int(revealed_count))
    return list(graph.get("global") or []), ("global",)


def _select(
    entries: list[tuple[float, float, str, str]],
    rho: float | None,
    observed_prefix_ms: float | None,
) -> list[tuple[float, float, str, str]]:
    """Apply the Pearson gate and the observed-demand tuple filter."""

    if observed_prefix_ms is None or rho is None or rho <= PEARSON_GATE:
        return entries
    tolerance = max(1.0, REFINEMENT_TOLERANCE * float(observed_prefix_ms))
    refined = [e for e in entries if abs(e[0] - float(observed_prefix_ms)) <= tolerance]
    if len(refined) < MIN_REFINED_TUPLES:
        return entries
    return refined


def conditional_tuples(
    graph: Mapping[str, Any],
    workflow: str,
    revealed_count: int,
    last_role: str,
    observed_prefix_ms: float | None = None,
) -> tuple[list[tuple[float, float, str, str]], bool]:
    """Return (selected tuples, refined?) for the state and observed prefix."""

    entries, rho_key = _lookup(graph, workflow, revealed_count, last_role)
    rho = (graph.get("rho") or {}).get(rho_key) if rho_key is not None else None
    selected = _select(entries, rho, observed_prefix_ms)
    return selected, len(selected) != len(entries)


def remaining_samples(
    graph: Mapping[str, Any],
    workflow: str,
    revealed_count: int,
    last_role: str,
    observed_prefix_ms: float | None = None,
) -> list[float]:
    """Conditional remaining-demand samples (suffix sums of the selected tuples)."""

    selected, _refined = conditional_tuples(
        graph, workflow, revealed_count, last_role, observed_prefix_ms
    )
    return [entry[1] for entry in selected]


def downstream_demand(
    graph: Mapping[str, Any],
    workflow: str,
    revealed_count: int,
    last_role: str,
    observed_prefix_ms: float | None = None,
) -> tuple[float, str | None]:
    """``(p_s, model_id)`` from the conditional JOINT next-unit distribution.

    The unit is the most probable (next_role, next_model) pair among the
    selected tuples; ``p_s`` is its share.  An empty or single-unit state
    returns ``(0.0, None)`` rather than a guess.
    """

    selected, _refined = conditional_tuples(
        graph, workflow, revealed_count, last_role, observed_prefix_ms
    )
    joint: Counter[tuple[str, str]] = Counter()
    for entry in selected:
        if entry[2]:
            joint[(entry[2], entry[3])] += 1
    total = sum(joint.values())
    if total <= 0:
        return (0.0, None)
    (next_role, model_id), count = max(joint.items(), key=lambda item: (item[1], item[0]))
    return (count / total, str(model_id))


def gittins_index(samples: Sequence[float]) -> float | None:
    """The paper's exact empirical Gittins index ``G(D, 0)`` (lower = higher priority).

    ``G = inf_{delta>0} E[min(X, delta)] / P(X <= delta)`` over the empirical
    distribution.  The infimum can only be attained at a support value: between
    two support points the denominator is constant while the numerator grows, so
    every distinct sample value is checked.  Prefix sums make this O(n) after
    sorting.
    """

    xs = sorted(float(value) for value in samples)
    n = len(xs)
    if n == 0:
        return None
    prefix = [0.0] * (n + 1)
    for index, value in enumerate(xs):
        prefix[index + 1] = prefix[index] + value
    best: float | None = None
    index = 0
    while index < n:
        value = xs[index]
        end = index + 1
        while end < n and xs[end] == value:
            end += 1
        count = end
        candidate = (prefix[end] + (n - end) * value) / count
        if best is None or candidate < best:
            best = candidate
        index = end
    return best


def gittins_index_bucketed(samples: Sequence[float], *, buckets: int = 10) -> float | None:
    """Appendix sensitivity only: the earlier equal-mass 10-point grid."""

    xs = sorted(float(value) for value in samples)
    n = len(xs)
    if n == 0:
        return None
    best: float | None = None
    for bucket in range(1, max(2, int(buckets)) + 1):
        position = (n - 1) * (bucket / float(max(2, int(buckets))))
        low = int(position)
        high = min(n - 1, low + 1)
        fraction = position - low
        delta = xs[low] * (1.0 - fraction) + xs[high] * fraction
        below = sum(1 for value in xs if value <= delta + 1e-12)
        probability = below / n
        if probability <= 0.0:
            continue
        expectation = sum(min(value, delta) for value in xs) / n
        index = expectation / probability
        if best is None or index < best:
            best = index
    return best


def model_load_estimate_ms(
    train_stats: Mapping[str, Mapping[str, Any]],
    model_id: str,
) -> float:
    """Train-only load-duration estimate (median over the model's rows)."""

    values = []
    prefix = f"{model_id}|"
    for key, row in train_stats.items():
        if not str(key).startswith(prefix):
            continue
        value = row.get("load_p50_ms")
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return statistics.median(values) if values else 0.0
