"""Pythia-adapted back end: the FULL Algorithm 3 priority the earlier port only half-built.

The first port implemented ``S_completion`` and dropped the other half of the paper's
priority plus the worker-side aging.  Pythia Algorithm 3 is

    base_priority = omega1 * S_completion + omega2 * S_unblock

and the local worker re-scores each scheduling window with an aging factor that grows with
accumulated waiting time.  Both are restored here so the baseline is not a weakened
``completion only`` variant:

  * ``S_unblock`` is the DownstreamIdleRisk term: a future agent is worth unblocking when
    the deployment it needs is currently IDLE.  The paper reads model-replica queue depth;
    this simulator has no replica queue, so the term uses a QUEUE-DEMAND PROXY -- the number
    of scheduler-visible ready+running requests for that model.  A model with no visible
    demand is idle-risk.
  * the future roles come from the TRAIN-ONLY PFA (``reachable_future_roles``), and the
    role -> deployment mapping from train history (``profiler["role_model"]``).  The
    realized template is never read, so this remains a prediction, not a peek.

Adaptations recorded (NOT the paper's exact mechanism):
  * queue-demand proxy replaces replica queue depth;
  * S_unblock is the MEAN idle-risk over reachable future roles, bounded in [0, 1] so it is
    commensurate with S_completion (the paper's raw sum has an unstated scale);
  * aging is a dimensionless re-score ``S + lambda * (wait / tau)`` on our score, not the
    paper's worker loop.  ``omega1``, ``omega2``, ``lambda`` and ``tau`` are frozen
    constants (pre-registered; the paper does not give values).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from tracing.analysis.pythia_profiler import (
    expected_remaining_steps,
    reachable_future_roles,
    s_completion,
)

PYTHIA_SCHEMA = "pythia-algorithm3-priority-v1"

# Omega and aging constants.  The paper states the base_priority FORM but gives no values,
# so these are PRE-REGISTERED defaults pending a single dev-grid selection, then frozen.
# They must never be tuned after seeing the formal results.
PYTHIA_OMEGA1 = 1.0
PYTHIA_OMEGA2 = 1.0
PYTHIA_AGING_WEIGHT = 1.0
PYTHIA_AGING_SCALE_MS = 30000.0


def unblock_score(profiler: Mapping[str, Any], role: str,
                  model_is_idle: Callable[[str], bool],
                  horizon: int | None = None) -> float:
    """DownstreamIdleRisk: mean idle-risk over the PFA-reachable future roles.

    Each reachable future role contributes ``S_completion(V(a)) = 1 / (1 + V(a))`` when the
    deployment that role maps to is currently idle (no visible ready/running demand).  The
    mean keeps the term in [0, 1]; the paper's raw sum has an unstated scale.
    """

    futures = reachable_future_roles(profiler, role, horizon=horizon)
    if not futures:
        return 0.0
    role_model = profiler.get("role_model") or {}
    total = 0.0
    for future in futures:
        model = role_model.get(future)
        if model is None:
            continue
        if model_is_idle(model):
            total += s_completion(expected_remaining_steps(profiler, future))
    return float(total) / float(len(futures))


def pythia_base_priority(profiler: Mapping[str, Any], role: str,
                         model_is_idle: Callable[[str], bool],
                         *, omega1: float = PYTHIA_OMEGA1,
                         omega2: float = PYTHIA_OMEGA2,
                         horizon: int | None = None) -> float:
    """``omega1 * S_completion + omega2 * S_unblock``; larger is better."""

    completion = s_completion(expected_remaining_steps(profiler, role))
    unblock = unblock_score(profiler, role, model_is_idle, horizon=horizon)
    return float(omega1) * completion + float(omega2) * unblock


def pythia_effective_priority(base_priority: float, wait_ms: float,
                              aging_weight: float = PYTHIA_AGING_WEIGHT,
                              aging_scale_ms: float = PYTHIA_AGING_SCALE_MS) -> float:
    """``S_eff = S_base + lambda * (wait / tau)``; larger is better.

    Both terms are dimensionless SCORES.  Reusing the classical aging key would add a
    millisecond quantity to a score in (0, 1] -- a unit error -- so aging is expressed as a
    dimensionless ratio here.
    """

    if aging_scale_ms <= 0.0:
        raise ValueError("aging_scale_ms must be positive")
    return float(base_priority) + float(aging_weight) * (
        max(0.0, float(wait_ms)) / float(aging_scale_ms))
