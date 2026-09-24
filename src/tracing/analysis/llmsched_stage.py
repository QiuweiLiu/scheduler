"""LLM-2 of the LLMSched-adapted port: canonical stage ontology + duration state.

Paper: LLMSched: Uncertainty-Aware Workload Scheduling for Compound LLM
Applications (ICDCS 2025).  Its front end is a DAG in which every stage carries
BOTH structural uncertainty (does this stage occur at all?) and duration
uncertainty (how long does it take?).  This module builds that alphabet.

Two things it deliberately does NOT do, both of which were the review's findings:

  * It does not use ``sequence_index`` as a stage identity.  ``sequence_index == 4``
    in one trace is not the same semantic stage as ``sequence_index == 4`` in
    another: the corpus contains looping workflows, so the same position hosts
    different stages and the same stage recurs at different positions.

  * It does not use ``template.baseline`` as an application family.  ``baseline``
    is the workflow-family label the collection recorded; treating it as the
    application identity would smuggle a grouping the scheduler is not entitled to
    at admission time.  The first version models **one application BN** over
    VideoSeek instead, which is the strictly weaker assumption.

The alphabet is derived only from ontology the scheduler may legally see:
``lane``, ``role``, ``action_family`` and ``raw_action``.  Recurrence is resolved
with a PREFIX-ONLY occurrence counter, so a stage identity never depends on how
much future work happens to exist.

This module also owns the frozen duration discretizer.  Its bin edges are computed
on TRAIN ONLY and are then immutable: validation and test durations are mapped
through the frozen edges, never re-binned.  ``K`` is an adaptation hyperparameter,
not a value read from the paper, and is recorded as such in the artifact manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

STAGE_SCHEMA = "llmsched-canonical-stage-v2"

# The number of duration bins.  The paper's own K could not be confirmed from the
# text, so this is recorded as an ADAPTATION HYPERPARAMETER and frozen rather than
# searched.  It is deliberately small: the corpus has 640 workflows.
DEFAULT_N_BINS = 6

# The ABSENT state shares the stage alphabet with the duration bins.
ABSENT = "ABSENT"


# --------------------------------------------------------------------------- #
# canonical stage identity
# --------------------------------------------------------------------------- #
def canonical_stage_base(node: Any) -> str:
    """The semantic identity of a node, with no positional information.

    ``role`` and ``action_family`` are the two ontology fields that survive the
    scheduler projection and that name what the node *is*.  ``raw_action`` is more
    specific than ``action_family`` in some cases (one family covers several raw
    tools), so it is folded in to keep genuinely different tools apart.
    """

    role = str(getattr(node, "role", "") or "other")
    family = str(getattr(node, "action_family", "") or "other")
    raw = str(getattr(node, "raw_action", "") or "other")
    return "%s:%s:%s" % (role, family, raw)


def canonical_stage_key(node: Any, prefix_state: Mapping[str, int]) -> str:
    """``<base>#<occurrence>`` where the occurrence counts only the PREFIX.

    ``prefix_state`` maps a stage base to how many nodes with that base have
    ALREADY been observed in this job's canonical order.  Passing anything derived
    from the not-yet-executed future would reintroduce exactly the leak this
    function exists to prevent, so callers must build it from completed work only.
    """

    base = canonical_stage_base(node)
    seen = int((prefix_state or {}).get(base, 0))
    if seen < 0:
        raise ValueError("prefix occurrence for %r is negative: %r" % (base, seen))
    return "%s#%d" % (base, seen)


def advance_prefix(prefix_state: Mapping[str, int], node: Any) -> Dict[str, int]:
    """Return the prefix state after ``node`` has been observed."""

    base = canonical_stage_base(node)
    out = dict(prefix_state or {})
    out[base] = int(out.get(base, 0)) + 1
    return out


def canonical_order(template: Any) -> List[Any]:
    """The template's nodes in causal order.

    The v3.1 projection is a verified serial control-flow chain, so ``sequence_index``
    is a valid ORDER here.  That is the only thing it is used for: order, never
    identity.
    """

    return sorted(template.nodes, key=lambda n: (n.sequence_index, n.node_id))


def stage_sequence(template: Any) -> List[str]:
    """The full canonical stage sequence of a template (train-time use)."""

    prefix: Dict[str, int] = {}
    out: List[str] = []
    for node in canonical_order(template):
        out.append(canonical_stage_key(node, prefix))
        prefix = advance_prefix(prefix, node)
    return out


def intrinsic_duration_ms(node: Any) -> float:
    """The node's intrinsic service duration.

    For a merged CPU+GPU composite this is the node's own ``runtime_ms``, which the
    v3.1 contract defines as ``R_pre + R_nested + R_post``.  It is explicitly NOT
    ``finish_ms - node_start_ms`` taken from a simulation: that difference also
    contains the simulator's GPU queue delay, which is a property of the schedule
    rather than of the workload.  Training the BN on the scheduling artefact and
    then scoring schedules with it would be circular.
    """

    value = getattr(node, "runtime_ms", None)
    if value is None:
        raise ValueError("node %r has no runtime_ms" % getattr(node, "node_id", None))
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            "node %r has a non-finite or negative intrinsic duration: %r"
            % (getattr(node, "node_id", None), value)
        )
    return value


# --------------------------------------------------------------------------- #
# frozen train-only duration discretizer
# --------------------------------------------------------------------------- #
def _quantile_edges(values: Sequence[float], n_bins: int) -> List[float]:
    """Quantile bin edges over ``log1p`` durations.

    Quantiles rather than equal width: intrinsic durations here span roughly three
    orders of magnitude, so equal-width bins would collapse almost every sample
    into the top bin and destroy the duration signal.
    """

    if n_bins < 2:
        raise ValueError("n_bins must be >= 2, got %r" % n_bins)
    xs = sorted(math.log1p(float(v)) for v in values)
    n = len(xs)
    if n == 0:
        raise ValueError("cannot fit a discretizer on zero durations")
    edges: List[float] = []
    for i in range(1, n_bins):
        pos = (n - 1) * i / float(n_bins)
        lo = int(math.floor(pos))
        hi = min(n - 1, lo + 1)
        frac = pos - lo
        edges.append(xs[lo] * (1.0 - frac) + xs[hi] * frac)
    return edges


class DurationDiscretizer:
    """Maps an intrinsic duration to one of ``K`` frozen bins.

    Edges live in ``log1p`` space and are computed on TRAIN ONLY.  Every later
    mapping is a pure lookup, so validation and test durations cannot move the
    boundaries.
    """

    def __init__(self, edges: Sequence[float], n_bins: int, train_sample_count: int,
                 stage_support: Mapping[str, int] | None = None) -> None:
        self.edges = [float(e) for e in edges]
        self.n_bins = int(n_bins)
        if len(self.edges) != self.n_bins - 1:
            raise ValueError(
                "expected %d edges for %d bins, got %d"
                % (self.n_bins - 1, self.n_bins, len(self.edges))
            )
        if any(b < a for a, b in zip(self.edges, self.edges[1:])):
            raise ValueError("duration bin edges must be non-decreasing")
        self.train_sample_count = int(train_sample_count)
        self.stage_support = {str(k): int(v) for k, v in (stage_support or {}).items()}
        # how many non-train templates fit() refused to look at
        self.skipped_non_train: int = 0

    @classmethod
    def fit(cls, templates: Mapping[str, Any], *,
            n_bins: int = DEFAULT_N_BINS) -> "DurationDiscretizer":
        """Fit on TRAIN templates only.

        Non-train templates are SKIPPED, never folded in: the whole point of a frozen
        discretizer is that validation and test durations cannot move the boundaries.
        The number skipped is recorded so the train-only claim is checkable rather
        than asserted.

        Also records per-stage support, which is what ``Range(Y_i)`` is read from:
        the range of a stage is the span of its TRAIN support, so it is a property
        of the frozen model rather than of whatever validation data follows.
        """

        durations: List[float] = []
        support: Dict[str, int] = {}
        skipped = 0
        for tpl in templates.values():
            if getattr(tpl, "split", None) != "train":
                skipped += 1
                continue
            for stage, node in _stage_node_pairs(tpl):
                durations.append(intrinsic_duration_ms(node))
                support[stage] = support.get(stage, 0) + 1
        if not durations:
            raise ValueError(
                "no train templates among the %d supplied; refusing to fit a "
                "duration discretizer" % len(templates)
            )
        edges = _quantile_edges(durations, n_bins)
        instance = cls(edges, n_bins, len(durations), support)
        instance.skipped_non_train = int(skipped)
        return instance

    def bin_of(self, duration_ms: float) -> int:
        """The 0-based bin index for a duration."""

        value = float(duration_ms)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("cannot bin a non-finite or negative duration: %r" % duration_ms)
        x = math.log1p(value)
        lo, hi = 0, len(self.edges)
        while lo < hi:
            mid = (lo + hi) // 2
            if x < self.edges[mid]:
                hi = mid
            else:
                lo = mid + 1
        return int(lo)

    def state_of(self, duration_ms: float) -> str:
        return "D%d" % self.bin_of(duration_ms)

    def state_representative_ms(self, state: str) -> float:
        """A representative duration for a bin, in ms (bin midpoint in log space)."""

        k = self.bin_index(state)
        lo = math.expm1(self.edges[k - 1]) if k > 0 else 0.0
        hi = math.expm1(self.edges[k]) if k < len(self.edges) else None
        if hi is None:
            # the top bin is open ended; use the last edge as a floor
            return float(lo) if lo > 0.0 else 0.0
        if lo <= 0.0:
            return float(hi)
        return float(math.sqrt(max(lo, 1e-9) * max(hi, 1e-9)))

    def bin_index(self, state: str) -> int:
        if state == ABSENT:
            raise ValueError("ABSENT is not a duration bin")
        text = str(state)
        if not text.startswith("D") or not text[1:].isdigit():
            raise ValueError("unknown duration state %r" % state)
        k = int(text[1:])
        if not 0 <= k < self.n_bins:
            raise ValueError("duration state %r is outside the frozen vocabulary" % state)
        return k

    def states(self) -> List[str]:
        return [ABSENT] + ["D%d" % k for k in range(self.n_bins)]

    def support(self, stage: str) -> int:
        return int(self.stage_support.get(str(stage), 0))

    def to_json(self) -> Dict[str, Any]:
        return {
            "schema": STAGE_SCHEMA,
            "n_bins": self.n_bins,
            "bin_edges_log1p": list(self.edges),
            "train_sample_count": self.train_sample_count,
            "stage_support": dict(sorted(self.stage_support.items())),
            "n_bins_provenance": "adaptation_hyperparameter",
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "DurationDiscretizer":
        schema = payload.get("schema")
        if schema != STAGE_SCHEMA:
            raise ValueError(
                "duration discretizer schema mismatch: expected %r, got %r"
                % (STAGE_SCHEMA, schema)
            )
        return cls(payload["bin_edges_log1p"], int(payload["n_bins"]),
                   int(payload["train_sample_count"]), payload.get("stage_support"))

    def sha256(self) -> str:
        blob = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _stage_node_pairs(template: Any) -> Iterable[Tuple[str, Any]]:
    prefix: Dict[str, int] = {}
    for node in canonical_order(template):
        stage = canonical_stage_key(node, prefix)
        prefix = advance_prefix(prefix, node)
        yield stage, node


def build_stage_table(templates: Mapping[str, Any]) -> Dict[str, Any]:
    """Encode every TRAIN workflow as one row over the canonical stage alphabet.

    Returns ``{"rows": {trace_id: {stage: intrinsic_ms}}, "support": {stage: n},
    "n_train": n, "vocabulary": [stage, ...]}``.

    A stage that does not occur in a workflow is simply absent from that row, which
    is the encoding of the ABSENT state.  The occurrence counter is prefix-only, so
    a looped stage gets ``#0`` on its first pass and ``#1`` on its second without the
    encoder ever consulting how long the workflow turns out to be.
    """

    rows: Dict[str, Dict[str, float]] = {}
    support: Dict[str, int] = {}
    vocabulary: set[str] = set()
    n_train = 0
    for tid, tpl in templates.items():
        if getattr(tpl, "split", None) != "train":
            continue
        n_train += 1
        row: Dict[str, float] = {}
        prefix_state: Dict[str, int] = {}
        for node in canonical_order(tpl):
            stage = canonical_stage_key(node, prefix_state)
            prefix_state = advance_prefix(prefix_state, node)
            if stage in row:
                raise ValueError(
                    "canonical stage %r occurs twice in template %r; the prefix "
                    "occurrence counter should have made it unique" % (stage, tid)
                )
            row[stage] = intrinsic_duration_ms(node)
            support[stage] = support.get(stage, 0) + 1
            vocabulary.add(stage)
        rows[str(tid)] = row
    return {
        "rows": rows,
        "support": support,
        "vocabulary": sorted(vocabulary),
        "n_train": n_train,
    }
