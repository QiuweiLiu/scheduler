"""Pythia-adapted back end: the FULL Algorithm 3 priority the earlier port only half-built.

Algorithm 3's priority is

    base_priority = omega1 * S_completion + omega2 * S_unblock

and the local worker re-scores each scheduling window with an aging factor on accumulated
waiting time.  Both halves are restored here.

  * ``S_completion = 1 / (1 + V(role))`` with V = expected remaining DISTANCE IN STEPS.
  * ``S_unblock`` is DownstreamIdleRisk: for every future agent ``a`` whose deployment is
    currently IDLE, the paper adds ``1 / E[D(current, a)]`` -- the CLOSER a downstream agent
    is, the more it is worth unblocking ("one step away" beats "ten steps away").  The
    distance is therefore ``current -> a``, NOT ``a -> terminal``.  (An earlier version used
    ``a -> terminal``, which inverted the ordering; the gate ``test_distance_direction``
    pins the correct direction.)
  * aging: ``S_eff = S_base + lambda * (wait / tau)``, dimensionless.

Information boundary: future roles come from the TRAIN-ONLY PFA
(``reachable_future_roles``), the role -> deployment mapping from train history
(``profiler["role_model"]``), and the demand state from the live scheduler-visible queue.
The realized template is never read.

Adaptations recorded (NOT the paper's exact mechanism):
  * queue-demand proxy: ``D_m`` = count of scheduler-visible ready+running GPU requests for
    model m; a model with ``D_m == 0`` has zero VISIBLE demand (not literally an idle
    replica -- a never-resident model also reads as zero demand);
  * only future roles whose train-majority LANE is ``gpu`` enter S_unblock (a CPU/control
    role has no model-serving demand to unblock);
  * S_unblock is the RAW SUM over such roles (the paper's form); its scale relative to
    S_completion is set by the frozen ``omega2``, never by a candidate-wise mean;
  * ``omega1``, ``omega2``, ``lambda`` and ``tau`` are PRE-REGISTERED, data-independent
    adaptation constants -- the paper gives the form but no values.  They are frozen now
    and must not be re-tuned against the evaluation episodes.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence

from tracing.analysis.pythia_profiler import (
    END,
    expected_remaining_steps,
    reachable_future_roles,
    s_completion,
)

PYTHIA_SCHEMA = "pythia-algorithm3-priority-v1"

# Pre-registered, data-independent adaptation constants.  The paper states the priority
# FORM but gives no values; these are declared constants from the start, not tuned on any
# evaluation episode.
PYTHIA_OMEGA1 = 1.0
PYTHIA_OMEGA2 = 1.0
PYTHIA_AGING_WEIGHT = 1.0
PYTHIA_AGING_SCALE_MS = 30000.0


def visible_model_demand(jobs: Sequence[Any]) -> Dict[str, int]:
    """Scheduler-visible GPU demand per model: ready + running requests.

    The proxy for the paper's model-replica queue depth.  Residency is deliberately NOT
    consulted (resident != busy), and no planned/predicted future demand is read.
    """

    demand: Dict[str, int] = {}
    for job in jobs:
        for node_id, state in job.node_state.items():
            if state not in ("ready", "running"):
                continue
            node = job.template.by_id[node_id]
            if str(getattr(node, "lane", "gpu")) != "gpu":
                continue
            model = str(node.model_id)
            demand[model] = demand.get(model, 0) + 1
    return demand


def expected_distance_to_role(profiler: Mapping[str, Any], current_role: str,
                              target_role: str, horizon: int | None = None):
    """``E[first-hit distance from current_role to target_role]`` on the pruned PFA.

    Returns ``None`` when ``target_role`` is not reached within the bounded horizon (then it
    must not contribute).  This is the paper's ``E[D(current, a)]``: the distance the CURRENT
    agent is away from a downstream agent, not the downstream agent's distance to terminal.
    """

    if profiler.get("schema") != "pythia-role-pfa-v3":
        raise ValueError("Pythia profiler schema mismatch: %r" % (profiler.get("schema"),))
    steps = int(profiler["horizon"] if horizon is None else horizon)
    edges = profiler["edge_prob"]
    if str(target_role) == str(current_role):
        return 0.0

    front: Dict[str, float] = {str(current_role): 1.0}
    hit_prob = 0.0
    hit_weighted = 0.0
    for step in range(1, max(0, steps) + 1):
        nxt: Dict[str, float] = {}
        for role, prob in front.items():
            for following, p in edges.get(role, {}).items():
                if following == END:
                    continue
                mass = prob * float(p)
                if following == str(target_role):
                    hit_prob += mass
                    hit_weighted += step * mass
                else:
                    nxt[following] = nxt.get(following, 0.0) + mass
        front = nxt
        if not front:
            break
    if hit_prob <= 0.0:
        return None
    return hit_weighted / hit_prob


def unblock_contributions(profiler: Mapping[str, Any], role: str,
                          model_has_zero_visible_demand: Callable[[str], bool],
                          horizon: int | None = None):
    """``[(future_role, 1 / E[D(current, future_role)])]`` for the idle, GPU future set."""

    futures = reachable_future_roles(profiler, role, horizon=horizon)
    role_model = profiler.get("role_model") or {}
    role_model_dist = profiler.get("role_model_dist") or {}
    role_lane = profiler.get("role_lane") or {}
    out = []
    for future in futures:
        if str(role_lane.get(future, "gpu")) != "gpu":
            continue  # a CPU/control role has no model-serving demand to unblock
        dist = role_model_dist.get(future)
        if not dist:
            # Deterministic fallback for a hand-built profiler without the distribution.
            model = role_model.get(future)
            dist = {model: 1.0} if model is not None else {}
        # The paper's per-agent idle indicator, MARGINALISED over the role's deployment
        # distribution: with a low-purity mapping (e.g. a planner split 0.53/0.47 across two
        # models) the argmax alone would over-state or under-state the idle risk.
        idle_weight = sum(p for m, p in dist.items() if model_has_zero_visible_demand(m))
        if idle_weight <= 0.0:
            continue
        distance = expected_distance_to_role(profiler, role, future, horizon=horizon)
        if distance is None or distance <= 0.0:
            continue
        out.append((future, float(idle_weight) / distance))
    return out


def unblock_score(profiler: Mapping[str, Any], role: str,
                  model_has_zero_visible_demand: Callable[[str], bool],
                  horizon: int | None = None) -> float:
    """DownstreamIdleRisk = SUM over idle, GPU future roles of ``1 / E[D(current, a)]``."""

    return float(sum(contribution for _future, contribution
                     in unblock_contributions(profiler, role, model_has_zero_visible_demand,
                                              horizon=horizon)))


def pythia_base_priority(profiler: Mapping[str, Any], role: str,
                         model_has_zero_visible_demand: Callable[[str], bool],
                         *, omega1: float = PYTHIA_OMEGA1,
                         omega2: float = PYTHIA_OMEGA2,
                         horizon: int | None = None) -> float:
    """``omega1 * S_completion + omega2 * S_unblock``; larger is better."""

    completion = s_completion(expected_remaining_steps(profiler, role))
    unblock = unblock_score(profiler, role, model_has_zero_visible_demand, horizon=horizon)
    return float(omega1) * completion + float(omega2) * unblock


def pythia_effective_priority(base_priority: float, wait_ms: float,
                              aging_weight: float = PYTHIA_AGING_WEIGHT,
                              aging_scale_ms: float = PYTHIA_AGING_SCALE_MS) -> float:
    """``S_eff = S_base + lambda * (wait / tau)``; larger is better.

    Both terms are dimensionless SCORES.  Reusing the classical aging key would add a
    millisecond quantity to a score -- a unit error -- so aging is a dimensionless ratio.
    """

    if aging_scale_ms <= 0.0:
        raise ValueError("aging_scale_ms must be positive")
    return float(base_priority) + float(aging_weight) * (
        max(0.0, float(wait_ms)) / float(aging_scale_ms))
