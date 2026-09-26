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

import numpy as np
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

# Hard cap on the future set when a caller does not supply the workflow's own
# remaining stages.  A stage on the 55-stage vocabulary can reach about thirty
# descendants, and multiplying that many ranges together overflows the score.
# The size of the truncated correlated future set.  Equation (6) uses this ONE set in
# both the information term and the range factor, and the cost of the exact joint is
# one table of 7^(M+1) entries: M = 4 gives 16807, M = 5 gives 117649.  M is a fixed
# computational adaptation chosen from the discrete state-space budget and the
# profiling latency, NOT from scheduling performance, and it is not a value taken from
# the paper.
MAX_JOINT_FUTURE = 4


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
                 ) -> Tuple[List[str], "np.ndarray"]:
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
    shape = [len(states)] * len(out_vars)
    table = np.zeros(shape, dtype=np.float64) if shape else np.zeros((), dtype=np.float64)
    # indices, not state strings: the table axes are positional
    for combo in itertools.product(range(len(states)), repeat=len(free_parents)):
        assignment = {name: states[index] for name, index in zip(free_parents, combo)}
        assignment.update({p: evidence[p] for p in parent_names if p in evidence})
        key = SEP.join(assignment[p] for p in parent_names)
        row = cpds[child].get(key)
        if row is None:
            raise KeyError(
                "CPD for %r has no row for parents %r; the table must be complete "
                "or every query on this network is undefined" % (child, key)
            )
        if child_observed:
            table[combo] = float(row[evidence[child]])
        else:
            for value_index, value in enumerate(states):
                table[combo + (value_index,)] = float(row[value])
    return out_vars, table


def _eliminate(variables: List[str], table: "np.ndarray", name: str,
               states: Sequence[str]) -> Tuple[List[str], "np.ndarray"]:
    """Sum ``name`` out of a factor."""

    index = variables.index(name)
    remaining = [v for v in variables if v != name]
    return remaining, table.sum(axis=index)


def _multiply(v1: List[str], t1: "np.ndarray",
              v2: List[str], t2: "np.ndarray"
              ) -> Tuple[List[str], "np.ndarray"]:
    """Pointwise product of two factors over their union of variables.

    Factors are dense arrays whose axis i is ``v[i]`` over the frozen state
    vocabulary.  Arguments are aligned by transposing the second factor's shared axes
    to the front of the first factor's, then broadcasting and taking the outer product
    on the axes each factor alone owns.

    The dict version of this function multiplied 7^6 = 117649-entry tables in pure
    Python and was the whole cost of a query.  The algebra here is the same.
    """

    union = list(v1) + [v for v in v2 if v not in v1]
    index_of = {name: i for i, name in enumerate(union)}

    # align t1 into the union axis order
    a_axes = list(range(len(v1)))
    a_perm = sorted(a_axes, key=lambda ax: index_of[v1[ax]])
    a = t1.transpose(a_perm)
    a_order = [v1[ax] for ax in a_perm]

    # align t2 into the union axis order: shared axes first, then its own
    shared = [name for name in v2 if name in v1]
    own = [name for name in v2 if name not in v1]
    # after transposing to (shared + own), the shared axes must line up with t1's
    target_shared = [name for name in a_order if name in set(shared)]
    b_axes = [v2.index(name) for name in target_shared + own]
    b = t2.transpose(b_axes)

    n_shared = len(target_shared)
    # put b's shared axes in the same slots as they occupy in a
    slot_of_shared = {name: a_order.index(name) for name in target_shared}
    full_rank = len(union)
    b_shape = [1] * full_rank
    b_perm_back = sorted(range(n_shared), key=lambda k: slot_of_shared[target_shared[k]])
    b = b.transpose(b_perm_back + list(range(n_shared, len(target_shared + own))))
    for axis, name in enumerate(target_shared):
        b_shape[slot_of_shared[name]] = b.shape[axis]
    for offset, name in enumerate(own):
        b_shape[index_of[name]] = b.shape[n_shared + offset]
    b = b.reshape([dim if dim != 1 else 1 for dim in b_shape])

    a_shape = [1] * full_rank
    for axis, name in enumerate(a_order):
        a_shape[index_of[name]] = a.shape[axis]
    a = a.reshape(a_shape)

    return union, (a * b)


def _normalise(table: "np.ndarray") -> "np.ndarray":
    total = float(table.sum())
    if total <= 0.0:
        raise ValueError(
            "posterior has zero total mass; the evidence is inconsistent with the "
            "frozen network rather than merely unlikely"
        )
    return table / total


def _relevant_variables(parents: Mapping[str, Sequence[str]], query: Sequence[str],
                        evidence: Mapping[str, str]) -> set:
    """The query, the evidence, and every ancestor of either.

    Everything else is conditionally irrelevant and sums to one, so omitting it is
    exact.  An observed variable that is a DESCENDANT of the query is reached through
    its own ancestors, which is why evidence is folded into the same closure: that is
    what lets evidence on a later stage travel back to an earlier query.
    """

    relevant: set = set()
    stack = [str(name) for name in query] + [str(name) for name in evidence]
    while stack:
        node = stack.pop()
        if node in relevant:
            continue
        if node not in parents:
            raise KeyError("variable %r is not in the frozen vocabulary" % node)
        relevant.add(node)
        stack.extend(parents[node])
    return relevant


def posterior_joint(profiler: Mapping[str, Any], query_stages: Sequence[str],
                    evidence: Mapping[str, str]) -> Tuple[List[str], "np.ndarray"]:
    """Exact ``P(query | evidence)`` by variable elimination.

    Returns ``(variables, array)`` where ``variables`` is the query order and axis i of
    the array is ``variables[i]``; entry ``[k, l]`` is P(query[0] = states[k],
    query[1] = states[l] | evidence).

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

    # A posterior depends only on the frozen network and on the evidence, so an
    # evidence-keyed memo makes repeated queries within one scheduling decision free.
    # The cache lives beside the profiler rather than in a global keyed by id(), which
    # would be unsound once a profiler is collected and another reuses its address.
    # The cache is keyed by the identity of the network it was built against, not only by
    # the query: a shallow copy of a profiler SHARES this cache object, so keying on the
    # query alone let a copied-and-then-mutated profiler return the original's posteriors.
    # id() of the CPD table changes whenever the table is replaced, and the two lengths
    # make an accidental address reuse implausible.
    cache = profiler.setdefault("_posterior_cache", {})
    cache_key = (
        id(profiler["cpds"]),
        id(profiler["stage_order"]),
        len(profiler["cpds"]),
        len(profiler["stage_order"]),
        tuple(query),
        tuple(sorted((evidence or {}).items())),
    )
    if cache_key in cache:
        return cache[cache_key]
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
    # Only the ancestors of the query and of the evidence can affect the answer.
    # A variable that is neither is the root of a sub-tree that sums to one for every
    # configuration of its parents, so dropping it is exact rather than an
    # approximation: by induction its whole sub-tree marginalises away.  On this
    # network that cuts the 55 variables down to a small window, which is what makes a
    # query cheap enough for a scheduler inner loop.
    relevant = _relevant_variables(parents, query, evidence or {})

    tables: List[Tuple[List[str], "np.ndarray"]] = []
    for node in stages:
        if node not in relevant:
            continue
        table_vars, table = _factor_over(node, parents[node], cpds, states, evidence or {})
        tables.append((table_vars, table))

    # Eliminate in CANONICAL ORDER.  The builder already refuses any network whose
    # induced width for this order exceeds MAX_INDUCED_WIDTH, so the order is known to
    # be cheap; the previous cheapest-first scan ranked variables by ``len(table)``,
    # which on a numpy array is only the first axis and not the table size, and its
    # O(variables^2) rescan per step was itself most of the 2 s a query used to cost.
    eliminate = [v for v in stages
                 if v in relevant and v not in query and v not in (evidence or {})]
    for target in eliminate:
        grouped: List[Tuple[List[str], "np.ndarray"]] = []
        product_vars: List[str] = []
        product = np.ones((), dtype=np.float64)
        for variables, table in tables:
            if target in variables:
                product_vars, product = _multiply(product_vars, product, variables, table)
            else:
                grouped.append((variables, table))
        reduced_vars, reduced = _eliminate(product_vars, product, target, states)
        grouped.append((reduced_vars, reduced))
        tables = grouped

    acc_variables: List[str] = []
    acc_table = np.ones((), dtype=np.float64)
    for variables, table in tables:
        acc_variables, acc_table = _multiply(acc_variables, acc_table, variables, table)
    variables, table = acc_variables, acc_table

    # reorder to the caller's query order, then renormalise
    order = sorted(range(len(variables)), key=lambda axis: query.index(variables[axis])
                   if variables[axis] in query else len(query) + axis)
    table = table.transpose(order)
    variables = [variables[axis] for axis in order]
    table = _normalise(table)
    result = (variables, table)
    if len(cache) < 200000:
        cache[cache_key] = result
    return result


def posterior_state_probs(profiler: Mapping[str, Any], query_stage: str,
                          evidence: Mapping[str, str]) -> Dict[str, float]:
    """``P(X = x | evidence)`` for a single stage, as a dict over the frozen states."""

    _variables, table = posterior_joint(profiler, [query_stage], evidence)
    states = list(profiler["state_vocabulary"])
    return {state: float(table[index]) for index, state in enumerate(states)}


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


def conditional_state_probs(profiler: Mapping[str, Any], query_stage: str,
                            condition_stage: str,
                            evidence: Mapping[str, str]) -> Dict[str, float]:
    """``P(query = x | E, condition_stage != ABSENT)`` for a single stage.

    The v1 EXPLOIT path had two information-state defects, and this is the fix for both:

      * it asked for a target's marginal under the completed evidence ALONE, while the
        ready candidate it was scoring is KNOWN to exist.  Leaving mass on
        ``condition = ABSENT`` while the scheduler is looking at that node is incoherent
        and it biases every target the ready stage correlates with.
      * the conditioning must PROPAGATE through the network, so this is a joint marginal
        (``condition`` is dropped from the absent slice, then marginalised out), not a
        renormalised single-variable posterior.
    """

    states = list(profiler["state_vocabulary"])
    if str(query_stage) == str(condition_stage):
        return absorb(profiler, query_stage, evidence)

    variables, joint = posterior_joint(profiler, [condition_stage, query_stage], evidence)
    if variables[0] != condition_stage:
        joint = joint.transpose()
    absent = states.index(ABSENT)
    present = np.delete(joint, absent, axis=0)
    total = float(present.sum())
    if total <= 0.0:
        # the condition has zero probability: an incoherent state, not a plausible zero
        raise ValueError(
            "conditioning stage %r has zero probability of existing, so a target cannot "
            "be scored against it" % condition_stage
        )
    marginal = present.sum(axis=0) / total
    return {state: float(marginal[index]) for index, state in enumerate(states)}


def conditional_state_probs_for_set(profiler: Mapping[str, Any], query_stage: str,
                                    present_stages: Sequence[str],
                                    evidence: Mapping[str, str]) -> Dict[str, float]:
    """``P(query | E, every stage in present_stages != ABSENT)``.

    The whole-job interval must use ONE information state: the ready stages are known to
    exist, and that condition has to PROPAGATE to every other unresolved stage, not just
    be renormalised locally on the ready stage itself.  The audit's earlier note was that
    the interval was a wider "conservative envelope" than the scheduler's actual state.
    """

    present = [str(s) for s in present_stages if str(s) != str(query_stage)]
    if not present:
        return posterior_state_probs(profiler, query_stage, evidence)

    states = list(profiler["state_vocabulary"])
    variables, joint = posterior_joint(profiler, [query_stage] + present, evidence)
    variables = list(variables)
    arr = joint
    absent = states.index(ABSENT)
    for stage in present:
        axis = variables.index(stage)
        arr = np.delete(arr, absent, axis=axis)
    total = float(arr.sum())
    if total <= 0.0:
        raise ValueError(
            "the known-present condition has zero probability; a ready stage cannot be "
            "scored against it"
        )
    query_axis = variables.index(query_stage)
    marginal = arr.sum(axis=tuple(i for i in range(arr.ndim) if i != query_axis))
    marginal = marginal / total
    return {state: float(marginal[index]) for index, state in enumerate(states)}


# --------------------------------------------------------------------------- #
# uncertainty reduction and expected remaining work
# --------------------------------------------------------------------------- #
def _rank_future_by_proximity(profiler: Mapping[str, Any], stage: str,
                              future: Sequence[str]) -> List[str]:
    """Order future stages by how close they sit to the candidate in the network.

    Distance is the length of the shortest directed path in the learned graph, with the
    canonical order as a deterministic tie-break.  A stage the candidate is adjacent to
    is the one whose duration it most plausibly informs, which is the relevance the
    paper's Y is drawn from.  Nothing here reads the workload's template.
    """

    children: Dict[str, List[str]] = {s: [] for s in profiler["stage_order"]}
    for child, pa_list in profiler["parents"].items():
        for parent in pa_list:
            children[parent].append(child)
    rank = {s: i for i, s in enumerate(profiler["stage_order"])}

    distance: Dict[str, int] = {stage: 0}
    frontier = [stage]
    while frontier:
        nxt: List[str] = []
        for node in frontier:
            for child in children.get(node, []):
                if child not in distance:
                    distance[child] = distance[node] + 1
                    nxt.append(child)
        frontier = nxt
    return sorted(future, key=lambda s: (distance.get(s, 10 ** 6), rank.get(s, 10 ** 6)))


def _future_from_network(profiler: Mapping[str, Any], stage: str,
                         evidence: Mapping[str, str]) -> List[str]:
    """The stages a candidate can tell you about, from the NETWORK only.

    Reachability in the learned Bayesian network, minus the stages already observed
    (their durations are settled, so executing anything else cannot reveal them).
    Nothing here reads the workload's realized template, so an unexecuted node's
    identity, action, duration or existence cannot enter the score.  The leaking
    helper that did read it has been deleted rather than left unused, so a future
    call site cannot reach it by accident.
    """

    observed = set(evidence or {})
    return [s for s in descendants(profiler, stage) if s not in observed]


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

    ranges = profiler.get("stage_range") or {}
    if stage not in ranges:
        raise KeyError(
            "stage %r has no entry in stage_range; an artifact that lost the field "
            "must fail closed rather than return a plausible zero" % (stage,)
        )
    # a genuine range of zero is legitimate and stays zero
    return float(ranges[stage])


def uncertainty_reduction(profiler: Mapping[str, Any], stage: str,
                          evidence: Mapping[str, str],
                          future_stages: Sequence[str] | None = None) -> float:
    """``R_E(X) = I(X; Y_1..Y_M | E) x SUM_m Range(Y_m)``.

    This is the paper's exploration score.  It is NOT an entropy: gate L2 constructs
    two candidates with identical marginal entropy and identical ranges where one is
    perfectly informative about the future and the other is independent of it, and
    only a mutual-information score can order them correctly.

    Two corrections over the first version of this function:

    * Range is SUMMED, not multiplied.  The paper aggregates the duration ranges of
      the correlated future stages; multiplying is not the published formula and it is
      what drove the score to 1e72 once Y covered every reachable descendant.
    * Y comes from the LEARNED NETWORK plus what has already been observed, never from
      the workload's realized template.  An earlier version read the true unexecuted
      suffix of the job's template, which is exactly the structure uncertainty this
      baseline is supposed to be resolving: LLMSched's premise is that a job's exact
      stages and dependencies are not known before execution.  Reading the template
      produced entirely reasonable numbers while leaking the answer.

    The information term is an exact JOINT over the truncated set, not a sum of
    pairwise terms: with Y_1 = Y_2 = X the joint is H(X) while the pairwise sum is
    2 H(X), so the two are not interchangeable.  ``MAX_JOINT_FUTURE`` is a fixed
    computational adaptation chosen from the discrete state-space budget and the
    profiling latency, not from scheduling performance.
    """

    future = _future_from_network(profiler, stage, evidence)
    if future_stages is not None:
        allowed = {str(s) for s in future_stages}
        future = [s for s in future if s in allowed]
    if not future:
        return 0.0

    # Equation (6) uses ONE set Y_1..Y_M in both the information term and the range
    # factor.  An earlier version truncated only the information term and summed the
    # ranges of every network descendant, so a candidate with four strongly informative
    # near stages still collected the ranges of the two dozen distant ones it had said
    # nothing about.  Both terms now use the same truncated set.
    #
    # Why a cap at all: measured on this vocabulary the reachable set reaches 35 with a
    # mean of 16.8, and an exact joint over 7-state variables is 7^(1+M).  M = 4 gives
    # 16807 entries, which keeps exact inference usable inside a scheduler inner loop.
    # M is a FIXED COMPUTATIONAL ADAPTATION chosen from the state-space budget and the
    # profiling latency, NOT from scheduling performance and not a value from the paper.
    # The paper's correlation notion is simply "a directed path exists"; ranking by
    # shortest path is ours, and is deterministic and train-learned only.
    ranked = _rank_future_by_proximity(profiler, stage, future)
    selected = ranked[:MAX_JOINT_FUTURE]
    info = joint_mutual_information(profiler, stage, selected, evidence)

    spread = 0.0
    for target in selected:
        # deliberately no max(1.0, ...) floor: a genuine range of zero must contribute
        # zero rather than being promoted into a normal-looking positive score
        spread += stage_range_ms(profiler, target)
    if info <= 0.0 or spread <= 0.0:
        return 0.0
    return float(info) * spread


def joint_mutual_information(profiler: Mapping[str, Any], stage: str,
                             futures: Sequence[str],
                             evidence: Mapping[str, str],
                             *, condition_present: bool = True) -> float:
    """Exact ``I(X; Y_1..Y_M | E)`` from the joint over ``[X] + futures``.

    This is the paper's information term.  It is NOT the sum of the pairwise terms:
    with Y_1 = Y_2 = X the joint mutual information is H(X) while the pairwise sum is
    2 H(X), so substituting one for the other changes the ranking and not merely the
    scale.  The joint is materialised directly, which is why the caller must keep
    ``futures`` small.
    """

    futures = [str(s) for s in futures if str(s) != str(stage) and str(s) not in (evidence or {})]
    if not futures:
        return 0.0
    variables, joint = posterior_joint(profiler, [stage] + futures, evidence)
    if variables[0] != stage:
        joint = joint.transpose()

    states = list(profiler["state_vocabulary"])
    if condition_present and ABSENT in states:
        # A ready candidate has already been reached, so X != ABSENT is settled.  The
        # unconditional I(X;Y|E) keeps mass on X = ABSENT, and because every stage
        # variable carries an ABSENT state, a stage that usually does not occur could
        # earn a large exploration bonus through correlation with the future structure
        # even while the scheduler is looking at it.  Drop the absent slice and
        # renormalise, which is exactly I(X;Y | E, X != ABSENT).
        #
        # A fixture whose vocabulary has no ABSENT state (the two-state test networks)
        # has nothing to condition on, so this is a no-op there rather than an error.
        absent = states.index(ABSENT)
        present = np.delete(joint, absent, axis=0)
        total = float(present.sum())
        if total <= 0.0:
            return 0.0
        joint = present / total
    # I(X; Y) = sum p(x,y) log2 [ p(x,y) / (p(x) p(y)) ]
    px = joint.sum(axis=0, keepdims=True)
    rest = tuple(range(1, joint.ndim))
    py = joint.sum(axis=rest, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(joint > 0.0, joint / (px * py), 1.0)
        terms = np.where(joint > 0.0, joint * np.log2(ratio), 0.0)
    return max(0.0, float(terms.sum()))


def pair_mutual_information(profiler: Mapping[str, Any], stage_a: str, stage_b: str,
                            evidence: Mapping[str, str]) -> float:
    """Exact ``I(A; B | E)`` from the two-variable posterior.

    Only a two-variable joint is ever materialised, so the cost is independent of how
    many descendants the network has.
    """

    variables, joint = posterior_joint(profiler, [stage_a, stage_b], evidence)
    if variables[0] != stage_a:
        joint = joint.transpose()
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(joint > 0.0, joint / (pa * pb), 1.0)
        terms = np.where(joint > 0.0, joint * np.log2(ratio), 0.0)
    return max(0.0, float(terms.sum()))


def expected_remaining_ms(profiler: Mapping[str, Any], stage: str,
                          evidence: Mapping[str, str],
                          future_stages: Sequence[str] | None = None) -> float:
    """``E[remaining service after the current stage | E]`` over the descendants.

    ``E[D_i | E] = P(X_i != ABSENT | E) x E[D_i | X_i != ABSENT, E]``, summed over the
    still-possible future stages.  An ABSENT stage contributes exactly zero duration,
    which is what lets structural uncertainty feed the JCT objective rather than being
    silently averaged away.

    Y comes from the LEARNED NETWORK plus what has already been observed; see
    ``uncertainty_reduction`` for why the realized template must not be read.
    """

    future = _future_from_network(profiler, stage, evidence)
    if future_stages is not None:
        allowed = {str(s) for s in future_stages}
        future = [s for s in future if s in allowed]
    if not future:
        return 0.0
    total = 0.0
    for name in future:
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


def expected_job_remaining_ms(profiler: Mapping[str, Any], evidence: Mapping[str, str],
                              current_stage: str) -> float:
    """``E[remaining service AFTER the ready candidate | E, candidate != ABSENT]``.

    The whole JOB's unresolved work, not just the ready candidate's network descendants.
    ``expected_remaining_ms`` walked ``_future_from_network`` -- the descendants of one
    candidate -- so a stage with no directed path from the candidate was silently
    dropped even though the job must still execute it.  That makes EXPLOIT
    systematically too optimistic about jobs whose remaining stages sit outside one
    candidate's ancestor/descendant chain, and it disagreed with
    ``job_duration_interval_ms``, which already covers every unresolved stage.

    Every posterior is conditioned on the ready stage being present -- the information
    state the scheduler is actually in -- so EXPLORE (which conditions ``X != ABSENT``),
    current service (``absorb``) and remaining work now consume ONE posterior state.
    The candidate's own duration is NOT included: the caller adds ``current_service_ms``.

    Equation (6)'s exploration score is deliberately untouched: it stays the correlated
    descendants plus the top-``MAX_JOINT_FUTURE`` of them.
    """

    observed = set(evidence or {})
    total = 0.0
    for name in profiler["stage_order"]:
        if name in observed or name == current_stage:
            continue
        probs = conditional_state_probs(profiler, name, current_stage, evidence)
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


def job_duration_interval_ms(profiler: Mapping[str, Any],
                             evidence: Mapping[str, str],
                             known_present: Sequence[str] | None = None) -> Tuple[float, float]:
    """The SUPPORT of a job's remaining intrinsic work, from the network.

    This is a support interval of the remaining-duration random variable, not an
    expectation: each unresolved stage can contribute either nothing at all (it is
    absent) or one of its duration states, so the per-stage contribution spans
    ``{0} union {durations}``.  Weighting a single min/max by ``P(present)`` collapses
    that support and is simply wrong -- with P(ABSENT) = P(D = 100) = 0.5 the true
    support is [0, 100] while the weighted form reports [50, 50], a point interval for a
    distribution that is anything but certain.

    The interval covers EVERY model-side unresolved stage, not just the correlated
    descendants of one candidate: it describes the JOB's remaining work, which is what
    Algorithm 1 compares between jobs.  Nothing here reads the workload's realized
    template.

    ``known_present`` names stages the scheduler is currently LOOKING AT (a ready
    candidate), i.e. stages known to exist.  EVERY unresolved stage is conditioned on that
    set being present -- the condition propagates through the network, not just a local
    renormalisation of the ready stage.  This puts the interval, the EXPLOIT expectation
    and current service in the SAME information state.  ``known_present is None`` keeps the
    unconditioned support for callers that have no ready stage (offline analysis).
    """

    known = [str(stage) for stage in (known_present or ())]
    known_set = set(known)
    lower = 0.0
    upper = 0.0
    for name in profiler["stage_order"]:
        if name in (evidence or {}):
            continue
        if known_set:
            probs = conditional_state_probs_for_set(profiler, name, known, evidence)
        else:
            probs = posterior_state_probs(profiler, name, evidence)
        present = 1.0 - float(probs.get(ABSENT, 0.0))
        if present <= 0.0:
            # the stage is settled absent: it contributes exactly nothing
            continue
        durations = [_state_ms(profiler, state) for state in probs if state != ABSENT]
        if not durations:
            continue
        # a stage that may still be absent contributes a possible zero to the support;
        # one that is certainly present -- or known present because it is ready -- cannot
        if present < 1.0 - 1e-12 and name not in known_set:
            lower += 0.0
        else:
            lower += min(durations)
        upper += max(durations)
    return float(lower), float(upper)


def non_overlapping_sets(intervals: Mapping[Any, Tuple[float, float]]) -> Dict[Any, int]:
    """Group jobs into duration intervals that do not overlap, ordered by lower bound.

    Returns ``{job_key: set_index}``.  Jobs are visited by ascending lower bound; a job
    joins the running set while its interval still overlaps that set's span, and starts
    a new set once it lies strictly after.  Within a set the durations are genuinely
    ambiguous, which is where uncertainty reduction is the deciding information; across
    sets the ordering is already determined by the bounds, so R(X) must not override it.
    """

    keys = sorted(intervals, key=lambda k: (float(intervals[k][0]), float(intervals[k][1]), str(k)))
    out: Dict[Any, int] = {}
    index = -1
    span_hi = None
    for key in keys:
        lo, hi = (float(intervals[key][0]), float(intervals[key][1]))
        if span_hi is None or lo > span_hi + 1e-9:
            index += 1
            span_hi = hi
        else:
            span_hi = max(span_hi, hi)
        out[key] = index
    return out


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
                            observed_ms: Mapping[str, float] | None = None, *,
                            on_unknown: str = "raise",
                            stats: Dict[str, Any] | None = None
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
            # The network has no variable for this stage.  A TRAIN mismatch means the
            # ontology and the model disagree and must fail closed; at INFERENCE on a
            # validation/test split an unseen occurrence index (the vocabulary is train-only)
            # is expected, so the stage is SKIPPED rather than crashing the arm -- the
            # posterior simply cannot condition on a stage the model does not contain.
            if on_unknown == "raise":
                raise KeyError(
                    "completed node %r maps to canonical stage %r, which is not in the "
                    "frozen vocabulary; the ontology and the model disagree"
                    % (node.node_id, stage)
                )
            elif on_unknown == "skip":
                if stats is not None:
                    stats["oov_skipped"] = int(stats.get("oov_skipped", 0)) + 1
                continue
            else:
                raise ValueError("unknown on_unknown policy %r" % (on_unknown,))
        # The duration must be an OBSERVATION, not the template's frozen value.  The
        # two are numerically equal here because the simulator's truth provider is
        # seeded from the template, so the point is not the number: it is that the
        # evidence chain proves a duration is only known once the node has finished,
        # instead of relying on a convention about when the template happens to be read.
        value = None if observed_ms is None else observed_ms.get(node.node_id)
        if value is None:
            raise KeyError(
                "node %r is marked completed but carries no observed intrinsic "
                "duration; LLMSched evidence must come from an observation"
                % node.node_id
            )
        evidence[stage] = disc.state_of(float(value))
    return evidence


def profiler_sha256(profiler: Mapping[str, Any]) -> str:
    """A content hash over the learned structure and CPDs."""

    payload = {
        "schema": profiler["schema"],
        "stage_order": list(profiler["stage_order"]),
        "parents": {k: list(v) for k, v in profiler["parents"].items()},
        # The docstring already claimed the hash covered the structure and the CPDs,
        # but the payload omitted the CPDs entirely: two networks with identical
        # structure and completely different conditional tables hashed the same.
        "cpds": {
            child: {key: {state: round(float(prob), 12) for state, prob in row.items()}
                    for key, row in rows.items()}
            for child, rows in profiler["cpds"].items()
        },
        "state_vocabulary": list(profiler["state_vocabulary"]),
        "stage_range": {k: float(v) for k, v in profiler["stage_range"].items()},
        "n_bins": profiler["discretizer"].n_bins,
        "bin_edges_log1p": list(profiler["discretizer"].edges),
        "stage_support": dict(profiler["discretizer"].stage_support),
        "train_sample_count": profiler["train_sample_count"],
        "skipped_non_train": profiler["skipped_non_train"],
        "n_train": profiler["n_train"],
        "max_parents": profiler["max_parents"],
        "max_lag": profiler["max_lag"],
        "min_gain_bits": profiler["min_gain_bits"],
        "smoothing": profiler["smoothing"],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
