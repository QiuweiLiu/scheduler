"""Attribute-mixture shadow methods, appended to decision_trace.py.

The artifact's predicted ``step_offset`` (1..H inside the predicted chain) is NOT
the training ``sequence_index`` (absolute workflow position), so the ``exact``
lookup level is unusable here.  The review flagged exactly this
(``sequence_index_semantics``): when the semantics do not match, the lookup must
fall back to ``model|lane|*`` and ``*|lane|*`` rather than force an
index-shaped value into the key.  ``SEQUENCE_INDEX_SEMANTICS`` records the verdict.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tracing.analysis.attribute_stats_mixture import (  # noqa: E402
    AttributeLookupMixture,
    GroupRuntime,
    build_lookup_mixture,
    canonical_views_from_atoms,
    canonical_views_from_probs,
    mix_atomic_runtime_distribution,
    mix_empirical_runtime_distribution,
)
from tracing.analysis.decision_trace import (  # noqa: E402
    CandidateView,
    DecisionContext,
    FutureScoreResult,
    ShadowMethod,
)

SEQUENCE_INDEX_SEMANTICS = "incompatible: predicted step_offset vs training absolute sequence_index"


class StatsBankLookup:
    """Walk the simulator's own hierarchy over a prebuilt histogram bank.

    Level order matches ``_hierarchical_train_row``: ``model|lane|*`` then
    ``*|lane|*``.  The ``exact`` level is skipped by construction because the
    predicted position is not the training position.
    """

    def __init__(self, bank: Mapping[str, Mapping[str, Any]], *, minimum_count: int = 3) -> None:
        self.bank = bank
        self.minimum_count = minimum_count
        self.path_counts: Dict[str, int] = {"model_lane": 0, "lane": 0, "missing": 0}

    def __call__(self, model_id: str, lane: str) -> GroupRuntime | None:
        if model_id and model_id != "unknown":
            entry = self.bank.get("%s|%s|*|model_lane" % (model_id, lane))
            if entry is not None and int(entry.get("n") or 0) >= self.minimum_count:
                self.path_counts["model_lane"] += 1
                return self._to_group(entry, "model_lane")
        entry = self.bank.get("*|%s|*|lane" % lane)
        if entry is not None and int(entry.get("n") or 0) >= self.minimum_count:
            self.path_counts["lane"] += 1
            return self._to_group(entry, "lane")
        self.path_counts["missing"] += 1
        return None

    @staticmethod
    def _to_group(entry: Mapping[str, Any], level: str) -> GroupRuntime:
        histogram = entry.get("runtime_bin_probs")
        return GroupRuntime(
            lookup_key=str(entry.get("lookup_key")),
            lookup_level=level,
            count=int(entry.get("n") or 0),
            runtime_p50_ms=float(entry.get("runtime_p50_ms") or 0.0),
            load_p50_ms=float(entry.get("load_p50_ms") or 0.0),
            runtime_bin_probs=tuple(float(p) for p in histogram) if histogram else None,
        )


def _steps(artifacts: Mapping[str, Mapping[str, Any]], node_id: str, horizon: int) -> List[Mapping[str, Any]]:
    row = artifacts.get(str(node_id))
    if row is None:
        raise ValueError("no artifact row for node %s" % node_id)
    scenarios = row.get("future_h%d" % horizon) or []
    if len(scenarios) != 1:
        raise ValueError("expected exactly 1 scenario for %s, got %d" % (node_id, len(scenarios)))
    return list((scenarios[0].get("steps") or [])[:horizon])


def _mixture_load(mixture: AttributeLookupMixture) -> float:
    return sum(weight * float(group.load_p50_ms) for _name, weight, group in mixture.entries)


class AttrArgmaxStatsMethod(ShadowMethod):
    """Legacy baseline: argmax attribute -> single point statistic lookup."""

    method_id = "attr_argmax_stats_v1"
    consumer_policy_id = "legacy_argmax_point_stats_v1"
    predictor_artifact_id = "attribute_argmax_stats"

    def __init__(self, artifacts: Mapping[str, Mapping[str, Any]], lookup: StatsBankLookup,
                 *, horizon: int = 5, legacy_load: Callable[[Mapping[str, Any]], float] | None = None) -> None:
        self.artifacts = artifacts
        self.lookup = lookup
        self.horizon = horizon
        self.legacy_load = legacy_load or (lambda _step: 0.0)

    def future_cost(self, candidate: CandidateView, context: DecisionContext) -> FutureScoreResult:
        total = 0.0
        levels: List[str] = []
        for step in _steps(self.artifacts, candidate.node_id, self.horizon):
            group = self.lookup(str(step.get("model_id") or "unknown"),
                                str(step.get("execution_lane") or "unknown"))
            if group is None:
                raise ValueError("attr_argmax_stats_v1 found no lookup group for %s" % candidate.node_id)
            levels.append(group.lookup_level)
            total += group.runtime_p50_ms + group.load_p50_ms + float(self.legacy_load(step))
        return FutureScoreResult(
            future_cost_ms=total,
            runtime_statistic="point_p50",
            lookup_level="|".join(sorted(set(levels))),
            extra={"sequence_index_semantics": SEQUENCE_INDEX_SEMANTICS},
        )


class AttrMixAtomMethod(ShadowMethod):
    """Attribute uncertainty only: probability-weighted mixture of point statistics."""

    method_id = "attrmix_stats_atom_v1"
    consumer_policy_id = "sum_attribute_mixture_atom_mean_plus_legacy_load_v1"
    predictor_artifact_id = "attribute_stats_mixture_atom"

    def __init__(self, artifacts: Mapping[str, Mapping[str, Any]], lookup: StatsBankLookup,
                 *, horizon: int = 5, legacy_load: Callable[[Mapping[str, Any]], float] | None = None) -> None:
        self.artifacts = artifacts
        self.lookup = lookup
        self.horizon = horizon
        self.legacy_load = legacy_load or (lambda _step: 0.0)

    def future_cost(self, candidate: CandidateView, context: DecisionContext) -> FutureScoreResult:
        total = 0.0
        entropies: List[float] = []
        dropped = 0
        for step in _steps(self.artifacts, candidate.node_id, self.horizon):
            mixture = build_lookup_mixture(step, lookup_group=self.lookup)
            views = canonical_views_from_atoms(mix_atomic_runtime_distribution(mixture))
            total += views["runtime_mean_ms"] + _mixture_load(mixture) + float(self.legacy_load(step))
            entropies.append(mixture.attribute_entropy.get("model_id", 0.0))
            dropped += mixture.dropped_categories
        return FutureScoreResult(
            future_cost_ms=total,
            runtime_statistic="mixture_mean",
            lookup_level="mixture",
            attribute_entropy=(sum(entropies) / len(entropies)) if entropies else None,
            extra={
                "sequence_index_semantics": SEQUENCE_INDEX_SEMANTICS,
                "dropped_attribute_categories": dropped,
            },
        )


class AttrMixHist16Method(ShadowMethod):
    """Attribute uncertainty plus within-group empirical variability.

    The scheduler-facing statistic defaults to p95 so it is directly comparable
    with the legacy consumer; every canonical view is recorded in ``extra`` so the
    analysis never has to re-derive them.
    """

    method_id = "attrmix_stats_hist16_v1"
    consumer_policy_id = "sum_attribute_mixture_hist16_plus_legacy_load_v1"
    predictor_artifact_id = "attribute_stats_mixture_hist16"

    def __init__(self, artifacts: Mapping[str, Mapping[str, Any]], lookup: StatsBankLookup,
                 representatives: Sequence[float], *, statistic: str = "p95", horizon: int = 5,
                 legacy_load: Callable[[Mapping[str, Any]], float] | None = None) -> None:
        self.artifacts = artifacts
        self.lookup = lookup
        self.representatives = [float(r) for r in representatives]
        self.statistic = statistic
        self.horizon = horizon
        self.legacy_load = legacy_load or (lambda _step: 0.0)

    def future_cost(self, candidate: CandidateView, context: DecisionContext) -> FutureScoreResult:
        total = 0.0
        probs_out: List[float] = []
        missing_mass = 0.0
        last_views: Dict[str, float] = {}
        for step in _steps(self.artifacts, candidate.node_id, self.horizon):
            mixture = build_lookup_mixture(step, lookup_group=self.lookup)
            probs, absent = mix_empirical_runtime_distribution(
                mixture, n_bins=len(self.representatives)
            )
            missing_mass += absent
            views = canonical_views_from_probs(probs, self.representatives)
            last_views = views
            total += views[self.statistic] + _mixture_load(mixture) + float(self.legacy_load(step))
            probs_out.extend(probs)
        return FutureScoreResult(
            future_cost_ms=total,
            runtime_statistic=self.statistic,
            runtime_probs=tuple(probs_out),
            lookup_level="mixture_hist16",
            extra={
                "sequence_index_semantics": SEQUENCE_INDEX_SEMANTICS,
                "missing_histogram_mass": missing_mass,
                "last_step_views": last_views,
            },
        )


def load_stats_bank(path: Path) -> Dict[str, Dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
