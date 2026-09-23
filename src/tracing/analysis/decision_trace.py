"""Decision-level tracing and shadow scoring for the v03 scheduler.

Why this exists
---------------
Every scheduling result so far is a per-episode scalar.  We could never see *why*
two predictors chose differently, because the candidate pool, the per-candidate
scores and the residency-dependent ``current`` term were never persisted.  The
planning review therefore asked for an independent decision trace: one line per
real scheduler decision state, carrying the whole candidate pool and the score
every registered method would have given it.

Design rules (from the review)
------------------------------
* **Shadow only.**  Exactly one method actually decides; every other method is
  scored on the same candidate pool so ``current``, residency, priority and GPU
  assignment are identical across methods.  That is what makes the comparison
  clean - six already-diverged trajectories are not comparable.
* **Persist the exact scheduler key**, never a reconstruction from fields, so a
  later tie-break change cannot silently invalidate an old trace.
* **Fail closed.**  A missing candidate, a non-finite score or a malformed key
  raises; it never lands in the trace as a gap.
* **Zero training, zero predictor rerun.**  Only the simulator runs again.
* **Do not fix cost semantics here.**  The legacy double-counted load and the
  missing eviction term are recorded and labelled
  (``legacy_runtime_plus_load_no_eviction_v1``), not repaired: otherwise the
  argmax-vs-mixture comparison would be confounded by a cost-semantics change.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

SCHEMA_VERSION = "scheduler-decision-trace-v1"
COST_SEMANTICS = "legacy_runtime_plus_load_no_eviction_v1"
TRACE_TOLERANCE = 1e-9


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CandidateView:
    """One scheduling candidate: a (job, node, gpu) triple with its current cost."""

    candidate_id: str
    job_instance_id: str
    node_id: str
    model_id: str
    gpu_index: int
    priority: float
    model_resident: bool
    current_runtime_p50_ms: float
    current_load_ms: float
    legacy_tiebreak_1: Any = None
    legacy_tiebreak_2: Any = None
    legacy_tiebreak_3: Any = None
    # position in the live candidate pool; Python's min() keeps the first on an
    # exact key tie, so a stable sort over this index is the only faithful replay
    pool_index: int = 0

    @property
    def current_cost_ms(self) -> float:
        return self.current_runtime_p50_ms + self.current_load_ms


@dataclass
class FutureScoreResult:
    """One method's answer for one candidate."""

    future_cost_ms: float
    runtime_statistic: str
    runtime_probs: Tuple[float, ...] | None = None
    lookup_level: str | None = None
    attribute_entropy: float | None = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionContext:
    episode_id: str
    decision_index: int
    time_ms: float
    trajectory_policy_id: str
    candidates: List[CandidateView]
    truth_future_cost: Callable[[CandidateView], float] | None = None


def candidate_id(job_instance_id: str, node_id: str, gpu_index: int) -> str:
    """The same node on two GPUs is two distinct candidates."""

    return "%s|%s|%d" % (job_instance_id, node_id, gpu_index)


def build_scheduler_key(
    candidate: CandidateView, future_cost_ms: float
) -> Tuple[Any, ...]:
    """The exact lexicographic key the same-shape consumer builds.

    Kept in lockstep with ``sameshape_score`` in the simulator; the regression
    guard asserts that this reconstruction picks the same winner as the live
    ``min(pool, key=...)`` call.
    """

    return (
        float(candidate.priority),
        candidate.current_cost_ms + float(future_cost_ms),
        float(future_cost_ms),
        candidate.legacy_tiebreak_1,
        candidate.legacy_tiebreak_2,
        0,
        candidate.legacy_tiebreak_3,
        candidate.gpu_index,
    )


# --------------------------------------------------------------------------- #
class ShadowMethod:
    """Base class: a named way to turn a future step into a scheduler score."""

    method_id = "unnamed"
    consumer_policy_id = "unnamed"
    predictor_artifact_id = "unnamed"

    def future_cost(self, candidate: CandidateView, context: DecisionContext) -> FutureScoreResult:
        raise NotImplementedError

    def scheduler_key(
        self, candidate: CandidateView, future_cost_ms: float, context: DecisionContext
    ) -> Tuple[Any, ...]:
        """Ranking key.  Default is the same-shape key; aging arms override this."""

        return build_scheduler_key(candidate, future_cost_ms)

    def describe(self) -> Dict[str, Any]:
        return {
            "method_id": self.method_id,
            "consumer_policy_id": self.consumer_policy_id,
            "predictor_artifact_id": self.predictor_artifact_id,
        }


class ArtifactQuantileMethod(ShadowMethod):
    """Read a precomputed canonical view out of an artifact pack (fail-closed)."""

    def __init__(
        self,
        method_id: str,
        artifacts: Mapping[str, Mapping[str, Any]],
        *,
        field: str,
        statistic: str,
        horizon: int = 5,
        consumer_policy_id: str = "legacy_sum_q95_v1",
        predictor_artifact_id: str = "unknown",
        load_cost: Callable[[Mapping[str, Any]], float] | None = None,
    ) -> None:
        self.method_id = method_id
        self.artifacts = artifacts
        self.field = field
        self.statistic = statistic
        self.horizon = horizon
        self.consumer_policy_id = consumer_policy_id
        self.predictor_artifact_id = predictor_artifact_id
        self._load_cost = load_cost or (lambda _step: 0.0)

    def _read(self, step: Mapping[str, Any]) -> float | None:
        resource = step.get("resource") or {}
        if self.field == "runtime_ms_quantiles":
            return (resource.get("runtime_ms_quantiles") or {}).get(self.statistic)
        return resource.get(self.statistic)

    def future_cost(self, candidate: CandidateView, context: DecisionContext) -> FutureScoreResult:
        row = self.artifacts.get(str(candidate.node_id))
        if row is None:
            raise ValueError("%s has no artifact row for node %s" % (self.method_id, candidate.node_id))
        scenarios = row.get("future_h%d" % self.horizon) or []
        if len(scenarios) != 1:
            raise ValueError(
                "%s expected exactly 1 scenario for %s, got %d"
                % (self.method_id, candidate.node_id, len(scenarios))
            )
        total = 0.0
        for step in (scenarios[0].get("steps") or [])[: self.horizon]:
            value = self._read(step)
            if value is None:
                raise ValueError(
                    "%s is missing %s.%s on node %s; refusing to fall back"
                    % (self.method_id, self.field, self.statistic, candidate.node_id)
                )
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("%s read a non-finite/negative %s" % (self.method_id, self.statistic))
            total += value + float(self._load_cost(step))
        return FutureScoreResult(future_cost_ms=total, runtime_statistic=self.statistic)


class TruthReferenceMethod(ShadowMethod):
    """Truth-informed H=5 successor walk, for the mechanism reference only."""

    method_id = "truth_h5_v1"
    consumer_policy_id = "samekey_true_future_h5_v1"
    predictor_artifact_id = "oracle_truth_v03"

    def future_cost(self, candidate: CandidateView, context: DecisionContext) -> FutureScoreResult:
        if context.truth_future_cost is None:
            raise ValueError("truth reference requested but no truth walk was supplied")
        return FutureScoreResult(
            future_cost_ms=float(context.truth_future_cost(candidate)),
            runtime_statistic="truth_h5",
        )


# --------------------------------------------------------------------------- #
class DecisionTraceWriter:
    """Streaming gzip JSONL writer for decision states."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = gzip.GzipFile(filename=str(self.path), mode="wb", mtime=0)
        self._text = None
        self.decisions = 0
        self.candidates = 0
        self._chosen_sequence: List[str] = []

    def _writer(self):
        if self._text is None:
            import io

            self._text = io.TextIOWrapper(self._handle, encoding="utf-8", newline="\n")
        return self._text

    def write(self, record: Mapping[str, Any]) -> None:
        if record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("decision trace record has the wrong schema_version")
        for candidate in record.get("candidates") or []:
            if not math.isfinite(float(candidate["current_cost_ms"])):
                raise ValueError("non-finite current_cost_ms for %s" % candidate["candidate_id"])
            for method_id, score in (candidate.get("scores") or {}).items():
                if not math.isfinite(float(score["total_cost_ms"])):
                    raise ValueError("non-finite total_cost_ms for %s/%s" % (method_id, candidate["candidate_id"]))
        chosen = record.get("actual_choice") or {}
        if not chosen.get("candidate_id"):
            raise ValueError("decision trace record without an actual choice")
        self._chosen_sequence.append(str(chosen["candidate_id"]))
        self.decisions += 1
        self.candidates += len(record.get("candidates") or [])
        self._writer().write(
            json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        )

    def chosen_sequence_sha256(self) -> str:
        digest = hashlib.sha256()
        for item in self._chosen_sequence:
            digest.update(item.encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    def close(self) -> Dict[str, Any]:
        if self._text is not None:
            self._text.flush()
            self._text.close()
        else:
            self._handle.close()
        return {
            "path": str(self.path),
            "decisions": self.decisions,
            "candidates": self.candidates,
            "chosen_sequence_sha256": self.chosen_sequence_sha256(),
        }


# --------------------------------------------------------------------------- #
def build_aging_scheduler_key(
    candidate: CandidateView, future_cost_ms: float, context: DecisionContext
) -> Tuple[Any, ...]:
    """The exact lexicographic key of the aging arm.

    Mirrors ``srtf_aging_key`` in the simulator plus the identity tie-breakers:
    (priority, R - wait, R, -wait, ready_since, job_index, node_id, gpu_index).
    The wait is measured from the node's own ready time, so this is node-level
    aging and a successor restarts its age when it is unlocked.
    """

    wait = float(context.time_ms) - float(candidate.legacy_tiebreak_1)
    if wait < -1e-9:
        raise ValueError("negative ready age: %s" % wait)
    wait = max(0.0, wait)
    remaining = candidate.current_cost_ms + float(future_cost_ms)
    return (
        float(candidate.priority),
        remaining - wait,
        remaining,
        -wait,
        candidate.legacy_tiebreak_1,
        candidate.legacy_tiebreak_2,
        candidate.legacy_tiebreak_3,
        candidate.gpu_index,
    )


def score_decision(
    context: DecisionContext,
    methods: Sequence[ShadowMethod],
    *,
    method_timings: Mapping[str, Mapping[str, int]] | None = None,
) -> Dict[str, Any]:
    """Score every method on the identical candidate pool and rank each of them."""

    if not context.candidates:
        raise ValueError("decision %d has an empty candidate pool" % context.decision_index)
    for index, candidate in enumerate(context.candidates):
        if candidate.pool_index != index:
            raise ValueError(
                "candidate %s carries pool_index %d but sits at %d"
                % (candidate.candidate_id, candidate.pool_index, index)
            )
    ids = [c.candidate_id for c in context.candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate ids in decision %d" % context.decision_index)

    competitive_priority = min(c.priority for c in context.candidates)
    competitive = [c for c in context.candidates if c.priority == competitive_priority]

    scores_by_method: Dict[str, Dict[str, FutureScoreResult]] = {}
    timings: Dict[str, Dict[str, int]] = {}
    for method in methods:
        per_candidate: Dict[str, FutureScoreResult] = {}
        started = time.perf_counter_ns()
        for candidate in context.candidates:
            per_candidate[candidate.candidate_id] = method.future_cost(candidate, context)
        scored_ns = time.perf_counter_ns() - started
        scores_by_method[method.method_id] = per_candidate
        timings[method.method_id] = {
            "score_compute_wall_ns": scored_ns,
            "candidates_scored": len(context.candidates),
            "per_candidate_wall_ns": scored_ns // max(1, len(context.candidates)),
        }

    candidate_records: List[Dict[str, Any]] = []
    for candidate in context.candidates:
        entry: Dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "job_instance_id": candidate.job_instance_id,
            "node_id": candidate.node_id,
            "model_id": candidate.model_id,
            "gpu_index": candidate.gpu_index,
            "priority": float(candidate.priority),
            "model_resident": bool(candidate.model_resident),
            "current_runtime_p50_ms": float(candidate.current_runtime_p50_ms),
            "current_load_ms": float(candidate.current_load_ms),
            "current_cost_ms": float(candidate.current_cost_ms),
            "legacy_tiebreak_1": candidate.legacy_tiebreak_1,
            "legacy_tiebreak_2": candidate.legacy_tiebreak_2,
            "legacy_tiebreak_3": candidate.legacy_tiebreak_3,
            "scores": {},
        }
        for method in methods:
            result = scores_by_method[method.method_id][candidate.candidate_id]
            entry["scores"][method.method_id] = {
                "future_cost_ms": float(result.future_cost_ms),
                "total_cost_ms": float(candidate.current_cost_ms + result.future_cost_ms),
                "runtime_statistic": result.runtime_statistic,
                "lookup_level": result.lookup_level,
                "attribute_entropy": result.attribute_entropy,
                "scheduler_key": list(
                    method.scheduler_key(candidate, result.future_cost_ms, context)
                ),
                "runtime_probs": list(result.runtime_probs) if result.runtime_probs is not None else None,
                **result.extra,
            }
        candidate_records.append(entry)

    # rank each method inside the competitive priority tier
    by_id: Dict[str, Dict[str, Any]] = {entry["candidate_id"]: entry for entry in candidate_records}
    for method in methods:
        method_id = method.method_id
        started = time.perf_counter_ns()
        # sorted() is stable and the input is in pool order, so an exact key tie
        # resolves to the earliest pool entry - exactly what min() does.  The
        # previous (key, candidate_id) tie-break changed the winner on exact ties.
        ordered = sorted(
            competitive,
            key=lambda c: method.scheduler_key(
                c, scores_by_method[method_id][c.candidate_id].future_cost_ms, context
            ),
        )
        for rank, candidate in enumerate(ordered):
            by_id[candidate.candidate_id]["scores"][method_id]["scheduler_key_rank"] = rank
            by_id[candidate.candidate_id]["scores"][method_id]["would_choose"] = rank == 0
        timings[method_id]["rank_select_wall_ns"] = time.perf_counter_ns() - started
        timings[method_id]["total_wall_ns"] = (
            timings[method_id]["score_compute_wall_ns"] + timings[method_id]["rank_select_wall_ns"]
        )

    return {
        "competitive_priority": float(competitive_priority),
        "competitive_candidate_count": len(competitive),
        "candidate_records": candidate_records,
        "method_timings": {**timings, **(method_timings or {})},
    }
