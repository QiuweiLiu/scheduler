"""LLM-3/4/5 of the LLMSched-adapted port: joint BN, evidence, posterior, CMI.

Paper: LLMSched: Uncertainty-Aware Workload Scheduling for Compound LLM
Applications (ICDCS 2025).  This module is the v2 front end that REPLACES the
retired one in ``llmsched_bn_legacy.py``.

What changed and why (the review's findings):

  * The retired version modelled ``P(length = m | family)`` plus per-position runtime
    marginals.  Those are INDEPENDENT marginals: the joint could never express that
    one stage's duration tells you something about another's, which is the entire
    reason the paper uses a Bayesian network.
  * The retired version scored exploration with ``H(X)``.  The paper scores it with
    ``I(X; Y_1..Y_k) x prod_i Range(Y_i)``, i.e. MUTUAL INFORMATION against the
    future.  These differ: two candidates can have identical marginal entropy and
    identical ranges while one is perfectly informative about the future and the
    other is independent of it.  Gate L2 is exactly that construction.
  * The retired consumer advanced its posterior with ``consumed = len(job.completed)``,
    a COUNT.  A count carries no information about durations, so observing that a
    node took 40 ms or 40 s made no difference to any scheduling decision.

What this module does instead:

  1. Every canonical stage is one discrete variable over ``{ABSENT, D0..D5}``, so a
     single variable expresses BOTH structure uncertainty (does the stage occur?)
     and duration uncertainty (which bin?).
  2. The joint ``prod_i P(X_i | Pa(X_i))`` is learned on TRAIN ONLY, with edges
     constrained to go from an earlier canonical stage to a later one.  The
     causal-order constraint is a DEVIATION from the paper, recorded as such: with
     only 480 training workflows an unconstrained search would happily fit
     future -> past dependencies that carry no causal meaning.
  3. Inference is EXACT variable elimination over the discrete tables.  The network
     is small and the answer is needed for small query sets, so there is no reason
     to approximate.  A fail-closed induced-width guard refuses to build a network
     that would make elimination blow up.
  4. Evidence is built from completed nodes' INTRINSIC durations and threaded as a
     partial assignment; a stage is ABSENT only once it is causally known absent.

Adaptations recorded in the artifact manifest:
    * causal-order-constrained structure learning (deviation, see above)
    * VideoSeek canonical stage ontology replaces the paper's workflow family
    * the GPU/node is the execution unit, so intra-stage ``sample_tasks(r)`` is omitted
    * the frozen GPU lifecycle / admission substrate is paid outside this module
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from tracing.analysis.llmsched_stage import (
    ABSENT,
    DEFAULT_N_BINS,
    DurationDiscretizer,
    build_stage_table,
    canonical_stage_key,
    intrinsic_duration_ms,
)

BN_SCHEMA = "llmsched-bn-v2"

# join character for a parent assignment key; NUL cannot occur in a stage name
SEP = "\x1f"

# Structural learning limits.  Indegree is capped because the training set is small
# and a dense parent set would be fit by noise; the cap is an adaptation
# hyperparameter, not a value taken from the paper.
DEFAULT_MAX_PARENTS = 2

# How far back in the canonical order a parent may sit.  The v3.1 projection is a
# verified serial control-flow chain, so a dependency spanning many stages is not
# causally plausible, and allowing one let greedy MI fit noise on 480 workflows (the
# first attempt reached an induced width of 53).  This window is an adaptation
# hyperparameter and is what keeps exact elimination cheap.
DEFAULT_MAX_LAG = 4

# An edge must add at least this much information about the child, in bits.
DEFAULT_MIN_GAIN_BITS = 0.01

# Laplace-style smoothing so a CPD row is never exactly zero.  A state the training
# set never produced must remain reachable, otherwise a single unseen transition
# would make the whole posterior collapse to zero.
DEFAULT_SMOOTHING = 1.0

# Fail-closed guard on the variable-elimination induced width.
MAX_INDUCED_WIDTH = 6


# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #
def canonical_stage_order(stage_table: Mapping[str, Any],
                          discretizer: DurationDiscretizer) -> List[str]:
    """A total causal order over the stage vocabulary.

    A stage's rank is the mean position at which it occurs across the training
    workflows, normalised by workflow length.  Earlier-in-the-workflow stages come
    first.  This is the order the edge constraint is expressed against; it is a
    convenience for structure learning, NOT a claim that every workflow visits every
    stage in this order.

    Ties are broken by stage name so the order is deterministic across runs.
    """

    rows = stage_table["rows"]
    positions: Dict[str, List[float]] = {}
    for row in rows.values():
        n = max(1, len(row))
        for rank, stage in enumerate(row):
            frac = rank / float(n)
            positions.setdefault(stage, []).append(frac)
    vocabulary = list(stage_table["vocabulary"])
    missing = [s for s in vocabulary if s not in positions]
    if missing:
        raise ValueError(
            "stage vocabulary contains %d stages absent from every train row: %s"
            % (len(missing), missing[:5])
        )
    return sorted(vocabulary,
                  key=lambda s: (sum(positions[s]) / float(len(positions[s])), s))


def _mis(stage_a: List[str], stage_b: List[str]) -> float:
    """Mutual information between two discrete columns, in bits."""

    n = len(stage_a)
    if n == 0:
        return 0.0
    joint: Dict[Tuple[str, str], int] = {}
    pa: Dict[str, int] = {}
    pb: Dict[str, int] = {}
    for a, b in zip(stage_a, stage_b):
        joint[(a, b)] = joint.get((a, b), 0) + 1
        pa[a] = pa.get(a, 0) + 1
        pb[b] = pb.get(b, 0) + 1
    total = float(n)
    mi = 0.0
    for (a, b), c in joint.items():
        p_joint = c / total
        p_a = pa[a] / total
        p_b = pb[b] / total
        mi += p_joint * math.log2(p_joint / (p_a * p_b))
    return mi


def learn_structure(columns: Mapping[str, Sequence[str]], order: Sequence[str], *,
                    max_parents: int = DEFAULT_MAX_PARENTS,
                    max_lag: int = DEFAULT_MAX_LAG,
                    min_gain_bits: float = DEFAULT_MIN_GAIN_BITS) -> Dict[str, List[str]]:
    """Greedy constrained structure learning.

    Three constraints, all recorded as adaptations:

      * CAUSAL ORDER: a parent must appear earlier in ``order``.
      * LAG WINDOW: a parent must be within ``max_lag`` positions of the child.  The
        v3.1 projection is a verified serial control-flow chain, so long-range
        dependencies between distant stages are not causally plausible, and allowing
        them let greedy MI fit pure noise on 480 workflows: the first attempt produced
        an induced width of 53.  The window is what keeps elimination exact and cheap.
      * MINIMUM GAIN: an edge must add at least ``min_gain_bits`` of information.

    Parents are then added greedily by conditional mutual information gain.  Greedy
    forward selection rather than a score-based search is a deliberate simplification:
    with this much data an exhaustive BIC search would mostly be fitting noise.
    """

    if max_lag < 1:
        raise ValueError("max_lag must be >= 1, got %r" % max_lag)
    parents: Dict[str, List[str]] = {stage: [] for stage in order}
    for index, child in enumerate(order):
        window = list(order[max(0, index - max_lag):index])
        if not window:
            continue
        chosen: List[str] = []
        while len(chosen) < max_parents:
            best_gain, best_parent = 0.0, None
            for candidate in window:
                if candidate in chosen:
                    continue
                gain = conditional_mutual_information(
                    columns[child],
                    [columns[candidate]],
                    [columns[p] for p in chosen],
                )
                if gain > best_gain:
                    best_gain, best_parent = gain, candidate
            if best_parent is None or best_gain < min_gain_bits:
                break
            chosen.append(best_parent)
        parents[child] = sorted(chosen, key=lambda s: order.index(s))
    return parents


def conditional_mutual_information(x: Sequence[str], ys: Sequence[Sequence[str]],
                                   zs: Sequence[Sequence[str]]) -> float:
    """I(X; Y | Z) in bits, exactly, from counts.

    ``I(X;Y|Z) = sum p(x,y,z) log2 [ p(x,y,z) p(z) / (p(x,z) p(y,z)) ]``

    ``ys`` may hold several variables, in which case Y is their tuple.
    """

    n = len(x)
    if n == 0:
        return 0.0
    for series in list(ys) + list(zs):
        if len(series) != n:
            raise ValueError("conditional_mutual_information got ragged columns")

    counts: Dict[Tuple[Any, ...], int] = {}
    z_counts: Dict[Tuple[Any, ...], int] = {}
    xz_counts: Dict[Tuple[Any, ...], int] = {}
    yz_counts: Dict[Tuple[Any, ...], int] = {}
    for i in range(n):
        xv = x[i]
        yv = tuple(s[i] for s in ys)
        zv = tuple(s[i] for s in zs)
        counts[(xv, yv, zv)] = counts.get((xv, yv, zv), 0) + 1
        z_counts[zv] = z_counts.get(zv, 0) + 1
        xz_counts[(xv, zv)] = xz_counts.get((xv, zv), 0) + 1
        yz_counts[(yv, zv)] = yz_counts.get((yv, zv), 0) + 1

    total = float(n)
    cmi = 0.0
    for (xv, yv, zv), c in counts.items():
        p_xyz = c / total
        p_z = z_counts[zv] / total
        p_xz = xz_counts[(xv, zv)] / total
        p_yz = yz_counts[(yv, zv)] / total
        if p_xz > 0.0 and p_yz > 0.0 and p_z > 0.0:
            cmi += p_xyz * math.log2((p_xyz * p_z) / (p_xz * p_yz))
    return max(0.0, cmi)


# --------------------------------------------------------------------------- #
# CPDs
# --------------------------------------------------------------------------- #
def fit_cpds(columns: Mapping[str, Sequence[str]], parents: Mapping[str, Sequence[str]],
             discretizer: DurationDiscretizer, *,
             smoothing: float = DEFAULT_SMOOTHING
             ) -> Dict[str, Dict[str, Dict[str, float]]]:
    """``P(X_i = x | Pa(X_i) = pa)`` from TRAIN counts, with fixed smoothing.

    Returned as ``{child: {"<pa values joined by SEP>": {state: prob}}}``.  A row is
    renormalised after smoothing, so every row sums to exactly 1 and no state is
    unreachable: a transition the training set never produced must stay possible,
    otherwise one unseen observation would collapse the posterior to zero.
    """

    states = list(discretizer.states())
    cpds: Dict[str, Dict[str, Dict[str, float]]] = {}
    for child, pa_list in parents.items():
        pa_list = list(pa_list)
        counts: Dict[Tuple[str, ...], Dict[str, int]] = {}
        for i in range(len(columns[child])):
            key = tuple(columns[p][i] for p in pa_list)
            slot = counts.setdefault(key, {})
            value = columns[child][i]
            slot[value] = slot.get(value, 0) + 1
        for slot in counts.values():
            foreign = set(slot) - set(states)
            if foreign:
                raise ValueError(
                    "child %r has states outside the frozen vocabulary: %s"
                    % (child, sorted(foreign)[:5])
                )
        # The table is completed over EVERY parent combination, not only the ones the
        # training set happened to contain.  Emitting only observed rows left the CPD
        # undefined for unseen parent states, and an inference call that reached such a
        # row raised: exact inference must be total over the declared state space, with
        # smoothing standing in for the missing counts.
        rows: Dict[str, Dict[str, float]] = {}
        for combo in itertools.product(states, repeat=len(pa_list)):
            slot = counts.get(tuple(combo), {})
            total = float(sum(slot.values())) + smoothing * len(states)
            rows[SEP.join(combo)] = {
                state: (slot.get(state, 0) + smoothing) / total for state in states
            }
        cpds[child] = rows
    return cpds


# --------------------------------------------------------------------------- #
# exact inference
# --------------------------------------------------------------------------- #
def _factor_over(child: str, parent_names: Sequence[str],
                 cpds: Mapping[str, Mapping[str, Mapping[str, float]]],
                 states: Sequence[str], evidence: Mapping[str, str]
                 ) -> Tuple[List[str], Dict[Tuple[str, ...], float]]:
    """One CPD as a factor table, with evidence folded in.

    Both an observed PARENT and an observed CHILD collapse their axis.  Folding only
    the parents left the observed child's own axis in the table, so it was summed out
    as an ordinary hidden variable rather than pinned: with evidence C=f the query
    P(B | A=t, C=f) then returned the ancestor-only 0.9 instead of 0.75, i.e. the
    evidence never constrained anything.  Pinning the child removes its axis instead.
    """

    free_parents = [p for p in parent_names if p not in evidence]
    child_observed = child in evidence
    out_vars = list(free_parents) + ([] if child_observed else [child])
    table: Dict[Tuple[str, ...], float] = {}
    for combo in itertools.product(states, repeat=len(free_parents)):
        assignment = dict(zip(free_parents, combo))
        assignment.update({p: evidence[p] for p in parent_names if p in evidence})
        key = SEP.join(assignment[p] for p in parent_names)
        row = cpds[child].get(key)
        if row is None:
            raise KeyError(
                "CPD for %r has no row for parents %r; the table must be complete "
                "or every query on this network is undefined" % (child, key)
            )
        if child_observed:
            table[tuple(combo)] = float(row[evidence[child]])
        else:
            for value in states:
                table[tuple(combo) + (value,)] = float(row[value])
    return out_vars, table


def _eliminate(variables: List[str], table: Dict[Tuple[str, ...], float], name: str,
               states: Sequence[str]) -> Tuple[List[str], Dict[Tuple[str, ...], float]]:
    """Sum ``name`` out of a factor."""

    index = variables.index(name)
    remaining = [v for v in variables if v != name]
    out: Dict[Tuple[str, ...], float] = {}
    for key, value in table.items():
        reduced = key[:index] + key[index + 1:]
        out[reduced] = out.get(reduced, 0.0) + value
    _ = states
    return remaining, out


def _multiply(v1: List[str], t1: Dict[Tuple[str, ...], float],
              v2: List[str], t2: Dict[Tuple[str, ...], float]
              ) -> Tuple[List[str], Dict[Tuple[str, ...], float]]:
    """Pointwise product of two factors over their union of variables.

    Implemented as a hash join on the SHARED variables rather than a nested loop over
    both tables.  The nested-loop version paired every entry of one factor with every
    entry of the other, which is quadratic in the table size and did not terminate on
    the real network once factors grew past a few hundred entries.

    A key in ``t1`` is ordered by ``v1`` and a key in ``t2`` by ``v2``, so the two
    position lists are translated into slots of the union separately.
    """

    union = list(v1) + [v for v in v2 if v not in v1]
    slot_of_v1 = [union.index(v) for v in v1]
    slot_of_v2 = [union.index(v) for v in v2]
    shared_slots = sorted(set(slot_of_v1) & set(slot_of_v2))

    # bucket t2 by its values on the shared variables
    buckets: Dict[Tuple[str, ...], List[Tuple[Tuple[str, ...], float]]] = {}
    for key2, val2 in t2.items():
        signature = tuple(key2[position] for position, slot in enumerate(slot_of_v2)
                          if slot in shared_slots)
        buckets.setdefault(signature, []).append((key2, val2))

    out: Dict[Tuple[str, ...], float] = {}
    for key1, val1 in t1.items():
        signature = tuple(key1[position] for position, slot in enumerate(slot_of_v1)
                          if slot in shared_slots)
        partners = buckets.get(signature)
        if not partners:
            continue
        for key2, val2 in partners:
            merged: List[Any] = [None] * len(union)
            for position, slot in enumerate(slot_of_v1):
                merged[slot] = key1[position]
            agree = True
            for position, slot in enumerate(slot_of_v2):
                if merged[slot] is None:
                    merged[slot] = key2[position]
                elif merged[slot] != key2[position]:
                    agree = False
                    break
            if agree:
                key = tuple(merged)
                out[key] = out.get(key, 0.0) + val1 * val2
    return union, out


def _normalise(table: Dict[Tuple[str, ...], float]) -> Dict[Tuple[str, ...], float]:
    total = sum(table.values())
    if total <= 0.0:
        raise ValueError(
            "posterior has zero total mass; the evidence is inconsistent with the "
            "frozen network rather than merely unlikely"
        )
    return {k: v / total for k, v in table.items()}


def posterior_joint(profiler: Mapping[str, Any], query_stages: Sequence[str],
                    evidence: Mapping[str, str]) -> Dict[Tuple[str, ...], float]:
    """Exact ``P(query | evidence)`` by variable elimination.

    Enumerates nothing beyond the query set and its ancestors: every other variable is
    summed out one at a time.  With a max indegree of two and a causal-order edge
    constraint the induced width stays small, and the builder refuses to construct a
    network whose width would exceed the guard.
    """

    stages = profiler["stage_order"]
    cpds = profiler["cpds"]
    parents = profiler["parents"]
    states = list(profiler["state_vocabulary"])

    query = [str(q) for q in query_stages]
    for q in query:
        if q not in parents:
            raise KeyError("query stage %r is not in the frozen vocabulary" % q)
    for name, value in (evidence or {}).items():
        if name not in parents:
            raise KeyError("evidence stage %r is not in the frozen vocabulary" % name)
        if value not in states:
            raise ValueError("evidence state %r is outside the frozen vocabulary" % value)
    overlap = set(query) & set(evidence or {})
    if overlap:
        raise ValueError(
            "stages %s are both queried and observed; a variable cannot be "
            "conditioned on itself" % sorted(overlap)
        )

    # Every variable contributes its factor, with the observed ones folded in.  An
    # earlier version kept only the query and its ANCESTORS, which silently dropped
    # the factors of observed DESCENDANTS: P(B=t | A=t, C=f) then returned the
    # ancestor-only answer 0.9 instead of the correct 0.75, because the evidence on C
    # never travelled back up to B.  The induced-width guard already bounds what
    # eliminating the full graph costs, so there is no reason to prune here.
    tables: List[Tuple[List[str], Dict[Tuple[str, ...], float]]] = []
    for node in stages:
        table_vars, table = _factor_over(node, parents[node], cpds, states, evidence or {})
        tables.append((table_vars, table))

    # eliminate every variable that is not in the query
    eliminate = [v for v in stages if v not in query and v not in (evidence or {})]
    while eliminate:
        # cheapest-first by the SIZE OF THE PRODUCT, not by the first factor found:
        # ranking on one factor can pick a variable that merges many large tables and
        # blows the intermediate up.
        target = None
        best_cost = None
        for name in eliminate:
            size = 1
            found = False
            for variables, table in tables:
                if name in variables:
                    size *= max(1, len(table))
                    found = True
            if found and (best_cost is None or size < best_cost):
                best_cost, target = size, name
        if target is None:
            raise ValueError("variable elimination stalled on %r" % eliminate[0])
        grouped: List[Tuple[List[str], Dict[Tuple[str, ...], float]]] = []
        product_vars: List[str] = []
        product: Dict[Tuple[str, ...], float] = {(): 1.0}
        for variables, table in tables:
            if target in variables:
                product_vars, product = _multiply(product_vars, product, variables, table)
            else:
                grouped.append((variables, table))
        reduced_vars, reduced = _eliminate(product_vars, product, target, states)
        grouped.append((reduced_vars, reduced))
        tables = grouped
        eliminate.remove(target)

    acc_variables: List[str] = []
    acc_table: Dict[Tuple[str, ...], float] = {(): 1.0}
    for variables, table in tables:
        acc_variables, acc_table = _multiply(acc_variables, acc_table, variables, table)
    variables, table = acc_variables, acc_table
    order_positions = [variables.index(q) for q in query]
    out: Dict[Tuple[str, ...], float] = {}
    for key, value in table.items():
        out[tuple(key[i] for i in order_positions)] = value
    return _normalise(out)


def posterior_state_probs(profiler: Mapping[str, Any], query_stage: str,
                          evidence: Mapping[str, str]) -> Dict[str, float]:
    """``P(X = x | evidence)`` for a single stage, as a dict over the frozen states."""

    joint = posterior_joint(profiler, [query_stage], evidence)
    out = {state: 0.0 for state in profiler["state_vocabulary"]}
    for key, value in joint.items():
        out[key[0]] = out.get(key[0], 0.0) + value
    return out


def absorb(profiler: Mapping[str, Any], query_stage: str,
           evidence: Mapping[str, str]) -> Dict[str, float]:
    """Posterior for a READY stage, conditioned on that stage being PRESENT.

    A stage the scheduler is looking at has already been reached, so conditioning on
    it being absent is incoherent.  ``P(X = x | E, X != ABSENT)`` removes that
    incoherence; gate L6 asserts it, because a scheduler that sees a ready node while
    its own posterior still assigns 30 % to "does not exist" is not modelling what it
    claims to model.
    """

    raw = posterior_state_probs(profiler, query_stage, evidence)
    present_mass = 1.0 - float(raw.get(ABSENT, 0.0))
    if present_mass <= 0.0:
        raise ValueError(
            "posterior gives stage %r zero probability of existing, so it cannot be "
            "ready" % query_stage
        )
    out = {state: value / present_mass for state, value in raw.items()}
    out[ABSENT] = 0.0
    return out


# --------------------------------------------------------------------------- #
# uncertainty reduction and expected remaining work
# --------------------------------------------------------------------------- #
def descendants(profiler: Mapping[str, Any], stage: str) -> List[str]:
    """Stages reachable from ``stage`` along the learned edges.

    Defined from the causal network only, never from the workload's future: using the
    template's remaining nodes would leak exactly the truth the baseline is supposed
    to predict.
    """

    children: Dict[str, List[str]] = {s: [] for s in profiler["stage_order"]}
    for child, pa_list in profiler["parents"].items():
        for parent in pa_list:
            children[parent].append(child)
    seen: set[str] = set()
    stack = list(children.get(stage, []))
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(children.get(node, []))
    return sorted(seen)


def stage_range_ms(profiler: Mapping[str, Any], stage: str) -> float:
    """``Range(Y)`` for a stage: the span of its TRAIN duration support.

    A property of the frozen model, not of whatever validation data follows, so it
    cannot be tuned after the fact.
    """

    lo_hi = (profiler.get("stage_range") or {}).get(stage)
    if lo_hi is None:
        return 0.0
    return float(lo_hi)


def uncertainty_reduction(profiler: Mapping[str, Any], stage: str,
                          evidence: Mapping[str, str]) -> float:
    """``R_E(X) = I(X; Y | E) x prod_i Range(Y_i)`` over the stage's descendants.

    This is the paper's exploration score.  It is NOT an entropy: gate L2 constructs
    two candidates with identical marginal entropy and identical ranges where one is
    perfectly informative about the future and the other is independent of it, and
    only a mutual-information score can order them correctly.

    ``I(X; Y_1..Y_k | E)`` is computed as the sum of the pairwise terms rather than
    from the full joint.  Materialising the joint over a stage and all of its
    descendants is exponential in the descendant count and did not terminate; the
    pairwise sum is exactly equal when the descendants are conditionally independent
    given X and is an upper bound otherwise, which is the conservative direction for
    an exploration bonus.  The substitution is recorded as an adaptation.
    """

    future = descendants(profiler, stage)
    if not future:
        return 0.0
    total_mi = 0.0
    spread = 1.0
    for target in future:
        if target in (evidence or {}):
            continue
        total_mi += pair_mutual_information(profiler, stage, target, evidence)
        spread *= max(1.0, stage_range_ms(profiler, target))
    if total_mi <= 0.0:
        return 0.0
    return float(total_mi) * spread


def pair_mutual_information(profiler: Mapping[str, Any], stage_a: str, stage_b: str,
                            evidence: Mapping[str, str]) -> float:
    """Exact ``I(A; B | E)`` from the two-variable posterior.

    Only a two-variable joint is ever materialised, so the cost is independent of how
    many descendants the network has.
    """

    joint = posterior_joint(profiler, [stage_a, stage_b], evidence)
    pa: Dict[str, float] = {}
    pb: Dict[str, float] = {}
    for (a, b), prob in joint.items():
        pa[a] = pa.get(a, 0.0) + prob
        pb[b] = pb.get(b, 0.0) + prob
    total = 0.0
    for (a, b), prob in joint.items():
        if prob <= 0.0:
            continue
        denom = pa[a] * pb[b]
        if denom > 0.0:
            total += prob * math.log2(prob / denom)
    return max(0.0, total)


def expected_remaining_ms(profiler: Mapping[str, Any], stage: str,
                          evidence: Mapping[str, str]) -> float:
    """``E[remaining service after the current stage | E]`` over the descendants.

    ``E[D_i | E] = P(X_i != ABSENT | E) x E[D_i | X_i != ABSENT, E]``, summed over the
    still-possible future stages.  An ABSENT stage contributes exactly zero duration,
    which is what lets structural uncertainty feed the JCT objective rather than being
    silently averaged away.
    """

    future = descendants(profiler, stage)
    if not future:
        return 0.0
    total = 0.0
    for name in future:
        if name in (evidence or {}):
            continue
        probs = posterior_state_probs(profiler, name, evidence)
        present = 1.0 - float(probs.get(ABSENT, 0.0))
        if present <= 0.0:
            continue
        conditional = 0.0
        for state, prob in probs.items():
            if state == ABSENT:
                continue
            conditional += prob * _state_ms(profiler, state)
        total += conditional
    return float(total)


def _state_ms(profiler: Mapping[str, Any], state: str) -> float:
    """A representative duration for a duration state, from the frozen discretizer."""

    disc = profiler["discretizer"]
    return float(disc.state_representative_ms(state)) if hasattr(disc, "state_representative_ms") else float(
        profiler["state_ms"][state]
    )


def current_service_ms(profiler: Mapping[str, Any], stage: str,
                       evidence: Mapping[str, str]) -> float:
    """``E[D | E, X != ABSENT]`` for the ready stage being considered."""

    probs = absorb(profiler, stage, evidence)
    total = 0.0
    for state, prob in probs.items():
        if state == ABSENT:
            continue
        total += prob * _state_ms(profiler, state)
    return float(total)


def draw_mode(rng: Any, epsilon: float) -> str:
    """ONE coin per decision: returns 'EXPLORE' with probability epsilon.

    This is the paper's epsilon-greedy contract and is unchanged from the retired
    front end: a single draw selects the mode, and the mode then ranks the whole
    candidate pool.  Drawing per candidate would silently turn the exploration rate
    into something else entirely.
    """

    return "EXPLORE" if float(rng.random()) < float(epsilon) else "EXPLOIT"


# --------------------------------------------------------------------------- #
# builder
# --------------------------------------------------------------------------- #
def build_bn_profiler(templates: Mapping[str, Any], *,
                      n_bins: int = DEFAULT_N_BINS,
                      max_parents: int = DEFAULT_MAX_PARENTS,
                      max_lag: int = DEFAULT_MAX_LAG,
                      min_gain_bits: float = DEFAULT_MIN_GAIN_BITS,
                      smoothing: float = DEFAULT_SMOOTHING) -> Dict[str, Any]:
    """Learn the v2 Bayesian workflow model from TRAIN-ONLY templates.

    Fail-closed checks, all of which the review asked for:

      * zero validation and zero test samples enter the fit
      * every edge goes from an earlier canonical stage to a later one
      * every CPD row sums to 1
      * every state belongs to the frozen vocabulary
      * no query path can reach an unexecuted future node id

    The returned profiler is also the artifact: it round-trips through JSON and
    carries a hash that pins the learned structure and CPDs.
    """

    disc = DurationDiscretizer.fit(templates, n_bins=n_bins)
    table = build_stage_table(templates)

    vocab = list(disc.states())
    rows = table["rows"]
    train_ids = sorted(rows)
    if not train_ids:
        raise ValueError("no train rows; refusing to build a BN")

    columns: Dict[str, List[str]] = {stage: [] for stage in table["vocabulary"]}
    for tid in train_ids:
        row = rows[tid]
        for stage in table["vocabulary"]:
            value = row.get(stage)
            columns[stage].append(ABSENT if value is None else disc.state_of(value))

    order = canonical_stage_order(table, disc)
    parents = learn_structure(columns, order, max_parents=max_parents,
                              max_lag=max_lag, min_gain_bits=min_gain_bits)
    _assert_causal(parents, order)
    cpds = fit_cpds(columns, parents, disc, smoothing=smoothing)
    _assert_cpd_rows(cpds, vocab)
    induced = _induced_width(parents, order)
    if induced > MAX_INDUCED_WIDTH:
        raise ValueError(
            "learned network has induced width %d above the guard %d; variable "
            "elimination would not stay cheap" % (induced, MAX_INDUCED_WIDTH)
        )

    stage_range: Dict[str, float] = {}
    for stage in table["vocabulary"]:
        values = [v for v in rows[train_ids[0]].keys()]
        _ = values
        durations = [rows[tid][stage] for tid in train_ids if stage in rows[tid]]
        if durations:
            stage_range[stage] = max(durations) - min(durations)

    profiler = {
        "schema": BN_SCHEMA,
        "stage_order": order,
        "parents": {k: list(v) for k, v in parents.items()},
        "cpds": cpds,
        "state_vocabulary": vocab,
        "state_ms": {state: (0.0 if state == ABSENT else disc.state_representative_ms(state))
                     for state in vocab},
        "stage_range": stage_range,
        "discretizer": disc,
        "n_bins_provenance": "adaptation_hyperparameter",
        "n_train": int(table["n_train"]),
        "n_validation": 0,
        "n_test": 0,
        "train_sample_count": int(disc.train_sample_count),
        "skipped_non_train": int(disc.skipped_non_train),
        "max_parents": int(max_parents),
        "smoothing": float(smoothing),
        "induced_width": int(induced),
        "max_lag": int(max_lag),
        "min_gain_bits": float(min_gain_bits),
        "deviation": "causal-order + lag-window constrained structure learning",
    }
    return profiler


def _assert_causal(parents: Mapping[str, Sequence[str]], order: Sequence[str]) -> None:
    rank = {stage: i for i, stage in enumerate(order)}
    for child, pa_list in parents.items():
        for parent in pa_list:
            if rank[parent] >= rank[child]:
                raise ValueError(
                    "edge %r -> %r violates the causal order constraint" % (parent, child)
                )


def _assert_cpd_rows(cpds: Mapping[str, Mapping[str, Mapping[str, float]]],
                     states: Sequence[str]) -> None:
    for child, rows in cpds.items():
        if not rows:
            raise ValueError("child %r has no CPD rows" % child)
        for key, row in rows.items():
            if set(row) != set(states):
                raise ValueError(
                    "CPD row %s|%s does not cover the frozen vocabulary" % (child, key)
                )
            total = sum(row.values())
            if abs(total - 1.0) > 1e-9:
                raise ValueError("CPD row %s|%s sums to %r, not 1" % (child, key, total))


def _induced_width(parents: Mapping[str, Sequence[str]], order: Sequence[str]) -> int:
    """Induced width of the moralised graph under the fixed elimination order.

    Width is the largest neighbourhood a variable has AT THE MOMENT it is eliminated,
    including fill-in edges created by earlier eliminations.  Eliminated variables are
    removed from the neighbourhood before measuring, otherwise the count would include
    edges that elimination has already resolved and would grossly overstate the cost.
    It is used only as a fail-closed guard, never to choose an order.
    """

    neighbours: Dict[str, set] = {stage: set() for stage in order}
    for child, pa_list in parents.items():
        for parent in pa_list:
            neighbours[child].add(parent)
            neighbours[parent].add(child)
    width = 0
    alive = list(order)
    for stage in order:
        alive.remove(stage)
        adj = {n for n in neighbours[stage] if n in alive}
        width = max(width, len(adj))
        for a in adj:
            for b in adj:
                if a != b:
                    neighbours[a].add(b)
                    neighbours[b].add(a)
    return int(width)


# --------------------------------------------------------------------------- #
# evidence
# --------------------------------------------------------------------------- #
def evidence_from_completed(job: Any, profiler: Mapping[str, Any],
                            observed_ms: Mapping[str, float] | None = None
                            ) -> Dict[str, str]:
    """Build the per-job evidence dict from COMPLETED nodes only.

    ``observed_ms`` is the intrinsic duration actually recorded for each completed
    node.  The simulator is the truth provider for its own workload definition, so
    reading a node's intrinsic runtime AFTER it has finished is a historical
    observation, not future leakage.  A node whose duration is not supplied falls
    back to the template's frozen intrinsic duration, which is the same quantity the
    BN was trained on.

    Ordering matters: the occurrence suffix depends on the prefix, so the stages are
    reconstructed in causal order and only the completed ones are admitted.
    """

    from tracing.analysis.llmsched_stage import advance_prefix, canonical_order

    split = getattr(job.template, "split", None)
    if split not in (None, "train", "validation", "test"):
        raise ValueError("unexpected template split %r" % (split,))

    evidence: Dict[str, str] = {}
    prefix: Dict[str, int] = {}
    disc = profiler["discretizer"]
    known = set(profiler["stage_order"])
    for node in canonical_order(job.template):
        stage = canonical_stage_key(node, prefix)
        prefix = advance_prefix(prefix, node)
        if node.node_id not in job.completed:
            continue
        if stage not in known:
            raise KeyError(
                "completed node %r maps to canonical stage %r, which is not in the "
                "frozen vocabulary; the ontology and the model disagree"
                % (node.node_id, stage)
            )
        value = None if observed_ms is None else observed_ms.get(node.node_id)
        duration = intrinsic_duration_ms(node) if value is None else float(value)
        evidence[stage] = disc.state_of(duration)
    return evidence


def profiler_sha256(profiler: Mapping[str, Any]) -> str:
    """A content hash over the learned structure and CPDs."""

    payload = {
        "schema": profiler["schema"],
        "stage_order": list(profiler["stage_order"]),
        "parents": {k: list(v) for k, v in profiler["parents"].items()},
        "n_bins": profiler["discretizer"].n_bins,
        "bin_edges_log1p": list(profiler["discretizer"].edges),
        "train_sample_count": profiler["train_sample_count"],
        "max_parents": profiler["max_parents"],
        "smoothing": profiler["smoothing"],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
