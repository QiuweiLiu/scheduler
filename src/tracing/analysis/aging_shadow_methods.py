"""Shadow method for the aging-augmented remaining-work arm.

The plain arm and the aging arm must be scored on the *same* candidate pool with
the *same* future cost, so that the only difference between them is the ranking
key.  This module provides both halves of that pair:

  * ``RemainingWorkShadowMethod``  - the plain F0 same-shape key
  * ``AgingRemainingWorkShadowMethod`` - the identical future cost, aging key

Both read the H=5 p95 chain cost out of the F0 artifact pack, so the shadow never
re-trains a predictor and never touches the test split.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

from tracing.analysis.decision_trace import (
    ArtifactQuantileMethod,
    CandidateView,
    DecisionContext,
    build_aging_scheduler_key,
)
from tracing.analysis.workload_v02_simulator import _legacy_p95_load_cost

# The live ``sameshape_h5_p95`` consumer is ``_q95_step_cost``, i.e. the predicted
# p95 runtime *plus* the frozen p95 load surcharge.  A shadow method that omitted
# the load term would score a different future cost and pick a different winner,
# which the per-decision trace guard correctly refuses.  Defaulting to the same
# load rule keeps the shadow a faithful replay of the trajectory policy.
DEFAULT_LOAD_COST = _legacy_p95_load_cost


class RemainingWorkShadowMethod(ArtifactQuantileMethod):
    """The plain arm: F0 H=5 p95 predicted remaining work with the same-shape key."""

    method_id = "sameshape_h5_p95_F0"
    consumer_policy_id = "sameshape_h5_p95"
    predictor_artifact_id = "f0_seed11"


class AgingRemainingWorkShadowMethod(RemainingWorkShadowMethod):
    """The aging arm: identical future cost, aging-augmented key.

    Deliberately inherits the future cost so the pair differs in exactly one
    place.  The key is (priority, R - wait, R, -wait, ...) where R is the
    predicted remaining work and the wait is measured from the node's own ready
    time, so this is node-level aging rather than whole-job starvation age.
    """

    method_id = "sameshape_h5_p95_aging"
    consumer_policy_id = "sameshape_h5_p95_aging"

    def scheduler_key(
        self, candidate: CandidateView, future_cost_ms: float, context: DecisionContext
    ) -> Tuple[Any, ...]:
        return build_aging_scheduler_key(candidate, future_cost_ms, context)


def build_aging_pair(
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    horizon: int = 5,
    statistic: str = "p95",
    load_cost=DEFAULT_LOAD_COST,
    predictor_artifact_id: str = "f0_seed11",
) -> Tuple[RemainingWorkShadowMethod, AgingRemainingWorkShadowMethod]:
    """Return the plain/aging pair sharing one artifact pack and one horizon.

    The load rule defaults to the frozen legacy surcharge so the plain half is a
    faithful replay of the live consumer; overriding it would silently change the
    future cost and break the trace guard.
    """

    kwargs = dict(
        field="runtime_ms_quantiles",
        statistic=statistic,
        horizon=horizon,
        load_cost=load_cost,
        predictor_artifact_id=predictor_artifact_id,
    )
    plain = RemainingWorkShadowMethod("sameshape_h5_p95_F0", artifacts, **kwargs)
    aged = AgingRemainingWorkShadowMethod("sameshape_h5_p95_aging", artifacts, **kwargs)
    return plain, aged
