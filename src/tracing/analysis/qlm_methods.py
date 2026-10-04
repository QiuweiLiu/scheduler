"""QLM-queue-adapted helpers (frozen 2026-10-04).

Adaptation of QLM (SoCC'24) to the single-GPU node-level substrate: a train-only
empirical duration distribution per ``(model_id, role)``, a fixed S=64
common-random-number scenario set, and stochastic queue assignment whose
objective keeps QLM's SLO orientation:

1. maximize the number of requests that satisfy the chance-SLO constraint
   ``P(C_i <= d_i) > delta`` (minimize chance violations);
2. then minimize the expected SLO penalty ``sum_i E[(C_i - d_i)_+]``;
3. then minimize ``sum_i E[C_i]``.

Review constraints (see docs/research/2026-10-03_main_table_baselines_manifest.md):

* no online same-key Bayesian posterior invention -- the distribution is
  offline/train-only;
* model transition/load enters the completion time directly (no separate
  warm-start heuristic weight);
* ``delta = 0.90`` is an adaptation hyperparameter, not a paper default;
* when every chance constraint is infeasible the lexicographic fallback above
  still yields a legal action (no autoscaling in this substrate).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence


def build_duration_bank(templates: Mapping[str, Any]) -> dict[str, Any]:
    """Train-only empirical duration bank: ``(model_id, role)`` / model / global."""

    by_model_role: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_model: dict[str, list[float]] = defaultdict(list)
    global_values: list[float] = []
    for template in templates.values():
        if str(getattr(template, "split", "")) != "train":
            continue
        for node in template.nodes:
            value = float(node.runtime_ms)
            by_model_role[(str(node.model_id), str(node.role))].append(value)
            by_model[str(node.model_id)].append(value)
            global_values.append(value)
    return {
        "by_model_role": {key: list(values) for key, values in by_model_role.items()},
        "by_model": {key: list(values) for key, values in by_model.items()},
        "global": global_values,
    }


def samples_for(
    bank: Mapping[str, Any],
    model_id: str,
    role: str,
    rng: Any,
    scenarios: int,
) -> list[float]:
    """Draw ``scenarios`` samples with the fallback chain ``(model, role) -> model -> global``."""

    pool = list((bank.get("by_model_role") or {}).get((str(model_id), str(role))) or [])
    if not pool:
        pool = list((bank.get("by_model") or {}).get(str(model_id)) or [])
    if not pool:
        pool = list(bank.get("global") or [])
    if not pool:
        raise ValueError(f"QLM duration bank has no samples for {model_id!r}/{role!r}")
    return [float(rng.choice(pool)) for _ in range(int(scenarios))]


def qlm_saa_choice(
    keys: Sequence[tuple[int, str]],
    samples: Mapping[tuple[int, str], Sequence[float]],
    deadlines: Mapping[tuple[int, str], float | None],
    loads: Mapping[tuple[int, str], float],
    *,
    delta: float = 0.90,
) -> int:
    """Return the index (into ``keys``) of the request to run first.

    For every candidate first-slot choice the rest of the queue continues in
    EDD order (earliest deadline first; ties by queue position) under the same
    common-random-number scenarios, and the lexicographic objective above is
    evaluated on that trial order.  The EDD continuation is the tractable,
    documented approximation of the paper's stochastic program on one queue.
    """

    order_keys = list(keys)
    scenario_count = len(samples[order_keys[0]]) if order_keys else 0
    if scenario_count <= 0:
        raise ValueError("QLM SAA requires at least one scenario")

    def trial_order(first_index: int) -> list[tuple[int, str]]:
        first = order_keys[first_index]
        rest = [key for index, key in enumerate(order_keys) if index != first_index]
        rest.sort(
            key=lambda key: (
                deadlines.get(key) if deadlines.get(key) is not None else float("inf"),
                order_keys.index(key),
            )
        )
        return [first] + rest

    best: tuple[Any, ...] | None = None
    best_index = 0
    for first_index in range(len(order_keys)):
        order = trial_order(first_index)
        completions: dict[tuple[int, str], list[float]] = {key: [] for key in order}
        for scenario in range(scenario_count):
            clock = 0.0
            for key in order:
                clock += float(samples[key][scenario]) + float(loads.get(key, 0.0))
                completions[key].append(clock)
        violations = 0
        penalty = 0.0
        total = 0.0
        for key in order:
            values = completions[key]
            deadline = deadlines.get(key)
            total += sum(values) / scenario_count
            if deadline is None:
                continue
            on_time = sum(1 for value in values if value <= float(deadline) + 1e-9) / scenario_count
            if on_time <= delta:
                violations += 1
            penalty += sum(max(0.0, value - float(deadline)) for value in values) / scenario_count
        key_tuple = (violations, penalty, total, first_index)
        if best is None or key_tuple < best:
            best = key_tuple
            best_index = first_index
    return best_index
