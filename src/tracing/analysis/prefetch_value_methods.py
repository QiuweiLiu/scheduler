"""Prefetch decision methods A (analytic net benefit) and B (scenario rollout).

Frozen design: docs/research/2026-10-10_exp_g_prefetch_value_methods.md

Method A (analytic): for each legal target model m,

    G(m) = sum_{j,k} [ q_k * p_k(m) * load_prob_k ] * min(load_ms(m), max(0, t_need(j,k)))
           - interference(m, g*)            # measured F4 additive extra, covered cells only
           - lambda_mem * memory_mb(m)      # frozen on dev (ms per MB)

t_need(j,k) = current frontier node's predicted remaining + sum of predicted mean step
costs of the preceding predicted steps (decision-time information only).  Picks
argmax G(m) if positive, else no action.

Method B (rollout): candidates = {no-prefetch} + top-2 by G; future scenarios are
sampled from the SAME frozen per-step marginals (assumption, disclosed): each
frontier node's chain length ~ its length distribution, each future step's model ~
its step marginal, step durations = predicted mean costs, nodes/steps independent,
no future arrivals.  Each action is scored by a bounded list-scheduling rollout of
the frontier + sampled steps over cloned device states (loads, residency,
light eviction); cost = sum of per-job completion times, averaged over the 8
scenarios; the action with the lowest average cost wins (ties -> no prefetch).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from tracing.analysis.pdrs_methods import _load_p95, _required_number, _steps, survival_weights

HORIZON = 5
SCENARIO_COUNT = 8
ROLLOUT_TOP_ACTIONS = 2
DEFAULT_LAMBDA_MEM_MS_PER_MB = 0.05
SCENARIO_SEED = 11


@dataclass
class FrontierItem:
    key: str
    row: Mapping[str, Any]
    current_model: str
    current_remaining_ms: float
    current_mb: float


@dataclass
class DeviceState:
    capacity_mb: float
    resident: Dict[str, float]
    busy_until: float


def _step_mean_ms(step: Mapping[str, Any]) -> float:
    runtime = _required_number(step, "runtime_mean_ms")
    load_prob = _required_number(step, "load_occurrence_probability")
    return runtime + load_prob * _load_p95(step)


def _row_distribution(row: Mapping[str, Any]) -> List[float]:
    probs = [float(value) for value in (row.get("length_probabilities") or [])]
    return probs


def benefit_feature_table(
    items: Sequence[FrontierItem],
    load_ms_of: Callable[[str], float],
    horizon: int = HORIZON,
) -> Dict[str, Dict[str, float]]:
    """Per-model interpretable features used by Method A and the B1 comparison.

    Features (frozen): timing-capped expected saved load time (benefit_ms),
    uncapped expected load time (uncapped_ms), first-step demand U (u1_ms),
    demand probability mass (demand_mass), earliest weighted need time
    (first_need_ms).
    """

    table: Dict[str, Dict[str, float]] = {}
    for item in items:
        steps = _steps(item.row, horizon)
        if not steps:
            continue
        weights = survival_weights(item.row, horizon)
        offsets: List[float] = []
        offset = max(0.0, float(item.current_remaining_ms))
        for step in steps[:horizon]:
            offsets.append(offset)
            offset += _step_mean_ms(step)
        for index, step in enumerate(steps[:horizon]):
            distribution = step.get("model_probabilities") or {}
            load_prob = _required_number(step, "load_occurrence_probability")
            load_p95 = _load_p95(step)
            t_need = offsets[index]
            q = weights[index]
            for raw_model, raw_prob in distribution.items():
                model = str(raw_model)
                probability = float(raw_prob)
                if probability <= 0.0 or q <= 0.0 or load_prob <= 0.0:
                    continue
                load_ms = float(load_ms_of(model) or 0.0)
                if load_ms <= 0.0:
                    continue
                entry = table.setdefault(model, {
                    "benefit_ms": 0.0, "uncapped_ms": 0.0, "u1_ms": 0.0,
                    "demand_mass": 0.0, "first_need_ms": float("inf"),
                })
                mass = q * probability * load_prob
                entry["demand_mass"] += mass
                entry["uncapped_ms"] += mass * load_ms
                if t_need > 0.0:
                    entry["benefit_ms"] += mass * min(load_ms, t_need)
                    if mass > 0.01 and t_need < entry["first_need_ms"]:
                        entry["first_need_ms"] = t_need
                if index == 0:
                    entry["u1_ms"] += q * probability * load_prob * load_p95
    for entry in table.values():
        if entry["first_need_ms"] == float("inf"):
            entry["first_need_ms"] = 0.0
    return table


def analytic_benefit_table(
    items: Sequence[FrontierItem],
    load_ms_of: Callable[[str], float],
    horizon: int = HORIZON,
) -> Dict[str, Dict[str, float]]:
    """Backward-compatible subset view (benefit_ms + u1_ms)."""

    full = benefit_feature_table(items, load_ms_of, horizon)
    return {model: {"benefit_ms": e["benefit_ms"], "u1_ms": e["u1_ms"]}
            for model, e in full.items()}


# Feature order for the distilled linear model (frozen; intercept last).
FEATURE_NAMES: tuple[str, ...] = (
    "benefit_ms",
    "uncapped_ms",
    "u1_ms",
    "demand_mass",
    "first_need_ms",
    "load_ms",
    "memory_mb",
    "mem_pressure",
    "interference_ms",
    "intercept",
)


def candidate_feature_vector(
    entry: Mapping[str, float],
    model: str,
    device_index: int,
    devices: Sequence[DeviceState],
    memory_mb_of: Callable[[str], float],
    interference_ms: float,
    load_ms_of: Callable[[str], float],
) -> Dict[str, float]:
    """Decision-time features of a (model, device) prefetch action (frozen order)."""

    device = devices[device_index]
    memory_mb = float(memory_mb_of(model) or 0.0)
    workspace_guard = 0.0
    projected = sum(device.resident.values()) + workspace_guard + memory_mb
    mem_pressure = projected / device.capacity_mb if device.capacity_mb > 0 else 1.0
    return {
        "benefit_ms": float(entry.get("benefit_ms", 0.0)),
        "uncapped_ms": float(entry.get("uncapped_ms", 0.0)),
        "u1_ms": float(entry.get("u1_ms", 0.0)),
        "demand_mass": float(entry.get("demand_mass", 0.0)),
        "first_need_ms": float(entry.get("first_need_ms", 0.0)),
        "load_ms": float(load_ms_of(model) or 0.0),
        "memory_mb": memory_mb,
        "mem_pressure": float(mem_pressure),
        "interference_ms": float(interference_ms),
        "intercept": 1.0,
    }


def distilled_score(features: Mapping[str, float], beta: Mapping[str, float]) -> float:
    return sum(float(beta.get(name, 0.0)) * float(features.get(name, 0.0)) for name in FEATURE_NAMES)


def choose_net_benefit(
    table: Mapping[str, Mapping[str, float]],
    candidate_models: Sequence[str],
    interference_of: Callable[[str], float],
    memory_mb_of: Callable[[str], float],
    lambda_mem_ms_per_mb: float,
) -> Tuple[Optional[str], Dict[str, Dict[str, float]]]:
    """argmax G(m) over legal candidates; None when the best G <= 0."""

    scored: Dict[str, Dict[str, float]] = {}
    best_model: Optional[str] = None
    best_g = 0.0
    for model in sorted(candidate_models):
        entry = table.get(model) or {"benefit_ms": 0.0, "u1_ms": 0.0}
        interference = float(interference_of(model))
        memory_cost = float(lambda_mem_ms_per_mb) * float(memory_mb_of(model))
        g = float(entry["benefit_ms"]) - interference - memory_cost
        scored[model] = {
            "benefit_ms": float(entry["benefit_ms"]),
            "u1_ms": float(entry["u1_ms"]),
            "interference_ms": interference,
            "memory_cost_ms": memory_cost,
            "G_ms": g,
        }
        if g > best_g + 1e-12:
            best_g = g
            best_model = model
    return best_model, scored


# --------------------------------------------------------------------------- #
# Method B: scenario generation + bounded rollout
# --------------------------------------------------------------------------- #
def build_scenarios(
    items: Sequence[FrontierItem],
    horizon: int = HORIZON,
    count: int = SCENARIO_COUNT,
    seed: int = SCENARIO_SEED,
) -> List[List[List[Tuple[str, float]]]]:
    """Deterministic base scenario + sampled scenarios from the frozen marginals.

    Scenario = per frontier item, the list of future steps AFTER the current node:
    (model, predicted mean duration ms).  Sampling assumption (disclosed): chain
    length and per-step models are drawn independently from the node's own
    marginal distributions; no cross-node or cross-step coupling is assumed.
    """

    rng = random.Random(seed)
    scenarios: List[List[List[Tuple[str, float]]]] = []
    for index in range(count):
        per_item: List[List[Tuple[str, float]]] = []
        for item in items:
            steps = _steps(item.row, horizon)
            probs = _row_distribution(item.row)
            if not steps:
                per_item.append([])
                continue
            if index == 0:
                length = max(range(len(probs)), key=lambda k: (probs[k], -k)) if probs else len(steps)
            else:
                length = _sample_index(probs, rng) if probs else len(steps)
            chain: List[Tuple[str, float]] = []
            for step in steps[: max(0, min(length, horizon))]:
                distribution = step.get("model_probabilities") or {}
                items_list = sorted(distribution)
                if not items_list:
                    continue
                if index == 0:
                    model = max(items_list, key=lambda m: (float(distribution[m]), m))
                else:
                    weights = [float(distribution[m]) for m in items_list]
                    model = items_list[_sample_index(weights, rng)]
                chain.append((str(model), _step_mean_ms(step)))
            per_item.append(chain)
        scenarios.append(per_item)
    return scenarios


def _sample_index(weights: Sequence[float], rng: random.Random) -> int:
    total = sum(weights)
    if total <= 0.0:
        return 0
    target = rng.random() * total
    acc = 0.0
    for index, weight in enumerate(weights):
        acc += float(weight)
        if target <= acc:
            return index
    return len(weights) - 1


def rollout_cost(
    items: Sequence[FrontierItem],
    scenario: Sequence[Sequence[Tuple[str, float]]],
    devices: Sequence[DeviceState],
    action: Optional[Tuple[str, int]],
    load_ms_of: Callable[[str], float],
    memory_mb_of: Callable[[str], float],
    now: float,
    first_task: Optional[int] = None,
    first_gpu: Optional[int] = None,
    eviction_mode: str = "least_needed",
) -> float:
    """Bounded list-scheduling rollout; returns the sum of per-job completion times.

    Rules (pre-registered): tasks dispatch shortest-current-duration first; a task
    runs on a device that already holds its model (max free) or, failing that, on
    the device with the most free memory that fits without eviction; when nothing
    fits, the least-needed resident model (not required by any remaining scenario
    task; tie: largest) is evicted.  Loads occupy their device serially.  Only the
    frontier items and their sampled steps are simulated.
    """

    state = [
        DeviceState(capacity_mb=float(device.capacity_mb),
                    resident=dict(device.resident),
                    busy_until=max(float(now), float(device.busy_until)))
        for device in devices
    ]
    if action is not None:
        action_model, action_device = action
        device = state[action_device]
        load_ms = float(load_ms_of(action_model) or 0.0)
        device.busy_until = max(device.busy_until, float(now) + load_ms)
        device.resident[action_model] = float(memory_mb_of(action_model) or 0.0)

    jobs: List[Dict[str, Any]] = []
    for position, item in enumerate(items):
        queue: List[Tuple[str, float]] = [(item.current_model, float(item.current_remaining_ms))]
        queue.extend((str(model), float(duration)) for model, duration in scenario[position])
        jobs.append({"queue": list(queue), "finish": None})
    remaining = len(jobs)
    guard = 0
    while remaining > 0 and guard < 4096:
        guard += 1
        ready = [(job_index, job["queue"][0]) for job_index, job in enumerate(jobs) if job["queue"] and job["finish"] is None]
        if not ready:
            break
        forced_gpu = None
        if first_task is not None and any(pair[0] == first_task for pair in ready):
            job_index, (model, duration) = next(pair for pair in ready if pair[0] == first_task)
            forced_gpu = first_gpu
            first_task = None
        else:
            job_index, (model, duration) = min(ready, key=lambda pair: (pair[1][1], pair[0]))
        needed = float(memory_mb_of(model) or 0.0)
        required = 0.0 if any(model in device.resident for device in state) else needed
        # future use map for eviction decisions
        future_use: Dict[str, int] = {}
        for other_index, job in enumerate(jobs):
            if job["finish"] is not None:
                continue
            queue = job["queue"]
            for position, (future_model, _duration) in enumerate(queue):
                if other_index == job_index and position == 0:
                    continue
                if future_model not in future_use:
                    future_use[future_model] = other_index * 100 + position
        def _eviction_candidates(device):
            if eviction_mode == "largest_first":
                return sorted(device.resident, key=lambda m: (-device.resident[m], m))
            return sorted(
                device.resident,
                key=lambda resident: (
                    -future_use.get(resident, 10 ** 9),
                    -device.resident[resident],
                    resident,
                ),
            )

        choice = None
        if forced_gpu is not None and 0 <= int(forced_gpu) < len(state):
            device = state[int(forced_gpu)]
            if sum(device.resident.values()) + required <= device.capacity_mb + 1e-9:
                choice = int(forced_gpu)
            else:
                for resident in _eviction_candidates(device):
                    if resident == model:
                        continue
                    device.resident.pop(resident, None)
                    if sum(device.resident.values()) + required <= device.capacity_mb + 1e-9:
                        break
                if sum(device.resident.values()) + required <= device.capacity_mb + 1e-9:
                    choice = int(forced_gpu)
        if choice is None:
            # earliest-start device among devices that fit without eviction; a missing
            # model pays its load serially on that device (no resident-affinity pull)
            best_key = None
            for device_index, device in enumerate(state):
                fits_now = sum(device.resident.values()) + required <= device.capacity_mb + 1e-9
                if not fits_now:
                    continue
                start = device.busy_until
                if model not in device.resident:
                    start += float(load_ms_of(model) or 0.0)
                key = (start, device_index)
                if best_key is None or key < best_key:
                    best_key = key
                    choice = device_index
        if choice is None:
            # evict per the selected mode on the roomiest device
            device_index = max(range(len(state)),
                               key=lambda i: (state[i].capacity_mb - sum(state[i].resident.values()), -i))
            device = state[device_index]
            for resident in _eviction_candidates(device):
                if resident == model:
                    continue
                device.resident.pop(resident, None)
                if sum(device.resident.values()) + required <= device.capacity_mb + 1e-9:
                    break
            if sum(device.resident.values()) + required <= device.capacity_mb + 1e-9:
                choice = device_index
        if choice is None:
            # cannot fit anywhere even after eviction: charge the task to the roomiest device
            choice = max(range(len(state)), key=lambda i: state[i].capacity_mb - sum(state[i].resident.values()))
        device = state[choice]
        start = device.busy_until
        if model not in device.resident:
            start += float(load_ms_of(model) or 0.0)
            device.resident[model] = needed
        finish = start + float(duration)
        device.busy_until = finish
        job = jobs[job_index]
        job["queue"].pop(0)
        if not job["queue"]:
            job["finish"] = finish
            remaining -= 1
    total = 0.0
    for job in jobs:
        total += float(job["finish"]) if job["finish"] is not None else float("inf")
    return total


def choose_joint_rollout_action(
    items: Sequence[FrontierItem],
    devices: Sequence[DeviceState],
    dispatch_options: Sequence[Tuple[int, int]],
    prefetch_options: Sequence[Optional[Tuple[str, int]]],
    load_ms_of: Callable[[str], float],
    memory_mb_of: Callable[[str], float],
    now: float,
    scenarios: List[List[List[Tuple[str, float]]]],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """Joint (ordering, prefetch, eviction) selection by mean rollout cost.

    ``dispatch_options`` = (item_index, gpu_index) of the shortlisted dispatch
    candidates (caller keeps the top-k by the frozen F0 key); ``prefetch_options``
    = candidate prefetch actions including None; eviction modes are the frozen
    pair {least_needed, largest_first}.  The default joint action = first F0
    candidate + no prefetch + least_needed.
    """

    eviction_modes = ("least_needed", "largest_first")
    option_list: List[Dict[str, Any]] = []
    if dispatch_options:
        default_item, default_gpu = dispatch_options[0]
        option_list.append({"first_item": default_item, "first_gpu": default_gpu,
                            "prefetch": None, "eviction_mode": "least_needed"})
    for item_index, gpu_index in dispatch_options:
        for prefetch in prefetch_options:
            for mode in eviction_modes:
                entry = {"first_item": item_index, "first_gpu": gpu_index,
                         "prefetch": prefetch, "eviction_mode": mode}
                if entry in option_list:
                    continue
                option_list.append(entry)
    costs: Dict[str, Dict[str, float]] = {}
    best_entry = option_list[0]
    best_cost = float("inf")
    for position, entry in enumerate(option_list):
        total = 0.0
        for scenario in scenarios:
            total += rollout_cost(
                items, scenario, devices, entry["prefetch"], load_ms_of,
                memory_mb_of, now, first_task=entry["first_item"],
                first_gpu=entry["first_gpu"], eviction_mode=entry["eviction_mode"],
            )
        average = total / len(scenarios)
        key = "j%02d" % position
        costs[key] = {"first_item": entry["first_item"], "first_gpu": entry["first_gpu"],
                      "prefetch": ("%s@%d" % entry["prefetch"]) if entry["prefetch"] else "__none__",
                      "eviction_mode": entry["eviction_mode"], "cost": average}
        if average < best_cost - 1e-9:
            best_cost = average
            best_entry = entry
    info = {"best_cost": best_cost, "n_options": len(option_list)}
    return dict(best_entry, cost=best_cost), info


def choose_rollout_action(
    items: Sequence[FrontierItem],
    devices: Sequence[DeviceState],
    actions: Sequence[Optional[Tuple[str, int]]],
    load_ms_of: Callable[[str], float],
    memory_mb_of: Callable[[str], float],
    now: float,
    scenarios: Optional[List[List[List[Tuple[str, float]]]]] = None,
) -> Tuple[Optional[Tuple[str, int]], Dict[str, float]]:
    """Score {none + candidate (model, device) actions} by mean rollout cost."""

    if scenarios is None:
        scenarios = build_scenarios(items)
    candidates = [None] + [tuple(action) for action in actions if action is not None]
    averages: Dict[str, float] = {}
    costs_none: Optional[float] = None
    best_action: Optional[Tuple[str, int]] = None
    best_cost = float("inf")
    for action in candidates:
        total = 0.0
        for scenario in scenarios:
            total += rollout_cost(items, scenario, devices, action, load_ms_of, memory_mb_of, now)
        average = total / len(scenarios)
        label = "%s@%d" % action if action is not None else "__none__"
        averages[label] = average
        if action is None:
            costs_none = average
        if average < best_cost - 1e-9:
            best_cost = average
            best_action = action
    result: Dict[str, float] = {"best_cost": best_cost}
    if costs_none is not None:
        result["cost_none"] = costs_none
        result["improvement_ms"] = costs_none - best_cost
    return best_action, result
