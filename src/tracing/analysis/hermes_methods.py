"""Hermes-PDGraph-adapted helpers (frozen 2026-10-04).

Adaptation of Hermes (ACM TACO 2026) to the single-GPU node-level substrate:
a train-only probabilistic demand graph over revealed prefixes, the paper's
exact Gittins rank, and the analytical online prewarm trigger.

Review constraints (see docs/research/2026-10-03_main_table_baselines_manifest.md):

* use the paper's Gittins formula with LOWER index = HIGHER priority;
* evaluate it on 10-bucket support boundaries (paper default), not an invented
  horizon;
* condition on the remaining-demand distribution (a = 0 form), which is exact
  and node-boundary friendly;
* prewarm uses ``p_e = p_s * P(t_c > t_s + t_p)`` with K = 0.5 and may fire
  while the current node is still executing (never only on idle);
* never mix deadline terms into the Gittins rank (Hermes-DDL is a separate
  variant and is not part of the main table).
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence


def build_pdgraph(templates: Mapping[str, Any]) -> dict[str, Any]:
    """Train-only probabilistic demand graph over (revealed_count, last_role).

    For every train template, after each node i the state is
    ``(i + 1 revealed nodes, role of node i)``; the graph records the next role
    distribution and an empirical sample of the remaining intrinsic work
    (sum of the unexecuted suffix durations).  Coarser fallbacks aggregate over
    roles (per count) and over everything (global) so an unseen state still has
    a distribution instead of a guess.
    """

    next_role: dict[tuple[int, str], Counter[str]] = defaultdict(Counter)
    remaining: dict[tuple[int, str], list[float]] = defaultdict(list)
    remaining_by_count: dict[int, list[float]] = defaultdict(list)
    global_remaining: list[float] = []
    role_models: dict[str, Counter[str]] = defaultdict(Counter)
    for template in templates.values():
        if str(getattr(template, "split", "")) != "train":
            continue
        nodes = list(template.nodes)
        if not nodes:
            continue
        # Initial state: nothing revealed yet -> the first role and the full
        # remaining work are the train-only priors for a freshly ready job.
        next_role[(0, "")][str(nodes[0].role)] += 1
        total = sum(float(node.runtime_ms) for node in nodes)
        remaining_by_count[0].append(total)
        global_remaining.append(total)
        for index, node in enumerate(nodes):
            state = (index + 1, str(node.role))
            if index + 1 < len(nodes):
                next_role[state][str(nodes[index + 1].role)] += 1
            tail = sum(float(other.runtime_ms) for other in nodes[index + 1:])
            remaining[state].append(tail)
            remaining_by_count[index + 1].append(tail)
            global_remaining.append(tail)
            role_models[str(node.role)][str(node.model_id)] += 1
    return {
        "next_role": {key: dict(counter) for key, counter in next_role.items()},
        "remaining": {key: list(values) for key, values in remaining.items()},
        "remaining_by_count": {key: list(values) for key, values in remaining_by_count.items()},
        "global_remaining": global_remaining,
        "role_models": {key: dict(counter) for key, counter in role_models.items()},
    }


def state_for_job(graph: Mapping[str, Any], revealed_count: int, last_role: str) -> tuple[int, str]:
    return (int(revealed_count), str(last_role))


def remaining_samples(
    graph: Mapping[str, Any],
    revealed_count: int,
    last_role: str,
) -> list[float]:
    """Conditional remaining-demand samples with coarse fallbacks."""

    state = state_for_job(graph, revealed_count, last_role)
    samples = list((graph.get("remaining") or {}).get(state) or [])
    if samples:
        return samples
    samples = list((graph.get("remaining_by_count") or {}).get(int(revealed_count)) or [])
    if samples:
        return samples
    return list(graph.get("global_remaining") or [])


def gittins_index(samples: Sequence[float], *, buckets: int = 10) -> float | None:
    """The paper's Gittins index ``G(D, 0)`` (LOWER means HIGHER priority).

    ``G = inf_delta E[min(X, delta)] / P(X <= delta)`` evaluated on the
    ``buckets`` support boundaries of the empirical distribution.
    """

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


def downstream_demand(
    graph: Mapping[str, Any],
    revealed_count: int,
    last_role: str,
) -> tuple[float, str | None]:
    """``(p_s, model_id)`` for the most probable next downstream unit.

    ``p_s`` is the conditional probability of the most likely next role from the
    current state; the model is the train-only majority model of that role.  An
    unseen state returns ``(0.0, None)`` rather than a guess.
    """

    state = state_for_job(graph, revealed_count, last_role)
    counter = (graph.get("next_role") or {}).get(state)
    if not counter:
        return (0.0, None)
    total = sum(counter.values())
    if total <= 0:
        return (0.0, None)
    next_role, count = max(counter.items(), key=lambda item: (item[1], item[0]))
    model_counter = (graph.get("role_models") or {}).get(next_role) or {}
    if not model_counter:
        return (count / total, None)
    model_id = max(model_counter.items(), key=lambda item: (item[1], item[0]))[0]
    return (count / total, str(model_id))


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
