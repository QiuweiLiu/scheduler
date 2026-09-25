"""Pythia-adapted: a role-level probabilistic workflow profiler + completion priority.

Paper: Pythia: Exploiting Workflow Predictability for Efficient Agent-Native LLM
Serving (arXiv:2604.25899).  Its front end is a workflow profiler / probabilistic finite
automaton built from historical agent execution traces **over a role alphabet**, which
is pruned to a bounded set of probable futures and used to derive an expected remaining
distance.  Its back end turns that into a completion-aware priority.

The review's finding on the first version: it used ``template.baseline`` as the
alphabet.  ``baseline`` is the workflow-family label the COLLECTION recorded, not a role
alphabet, so the profiler was conditioning on a grouping the scheduler is not entitled
to at admission time and was not modelling roles at all.  This version builds a genuine
PFA over the roles the scheduler can legally observe.

Chain migrated:
    train-only traces -> role-alphabet PFA -> pruned bounded probable future
        -> E[remaining distance] -> completion-aware priority

NOT migrated, stated as omissions rather than faked:
  * cache routing, prefix caching, model-replica idleness and autoscaling are outside
    this GPU execution abstraction;
  * the downstream-idle term ``S_unblock`` is omitted because the simulator has no
    model-server queue to attach it to.  The review preferred an honest omission over
    inventing one, so the priority is ``omega1 * S_completion`` with omega2 = 0.

Deviation, recorded: the paper derives the automaton from its own trace format.  Here the
alphabet is the VideoSeek role ontology, and the expected remaining distance is computed
by a FINITE-HORIZON recursion rather than by solving the stationary system, because real
workflows loop (a planner is re-entered many times) and the undiscounted expectation over
a cyclic automaton does not converge.

That finite horizon is OUR approximation and must not be described as the paper's bound.
Pythia bounds its automaton with low-probability transition pruning PLUS repetition bounds
on looping roles (its regex form is `engineer^{3,6}`), whereas `horizon` here is a
fixed, train-independent step limit.  The two are not the same mechanism and the paper's
form is not reproduced.

Adapted rather than literal: the paper's alphabet entry is the CURRENT `agent_id`
(`engineer`).  A bare VideoSeek role is too coarse to separate the tools that actually
differ in cost, so the key composes three prefix-observable, stable ontology fields
(`role:action_family:raw_action`).  Eleven states over 480 train runs shows the key has
not degenerated into an instance identifier.  The occurrence suffix is deliberately
DROPPED, which is the faithful choice: Pythia expresses a loop as a REPEATED role
(`(engineer)^{3,6}`), so appending `#0/#1/#2` would split one looping role into
several positional states and destroy exactly the structure the automaton is for.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Tuple

from tracing.analysis.llmsched_stage import canonical_stage_base, intrinsic_duration_ms

PROFILER_SCHEMA = "pythia-role-pfa-v2"

# Transitions rarer than this are pruned, which is the paper's "bounded probable
# future": the automaton keeps only the paths that history says actually happen often.
DEFAULT_MIN_PROB = 0.05

# How many steps of probable future the expected remaining distance looks ahead.  A finite
# horizon is required because the automaton is cyclic.  This is OUR fixed,
# train-independent computational approximation and is NOT the paper's bound, which prunes
# transitions and applies repetition bounds to looping roles.
DEFAULT_HORIZON = 6

END = "<end>"


def role_alphabet(node: Any) -> str:
    """The role of a node, from the ontology the scheduler can legally see.

    ``canonical_stage_base`` folds ``role``, ``action_family`` and ``raw_action``; the
    occurrence suffix is deliberately dropped, because a PFA state is a ROLE and not a
    position, which is what lets the automaton model loops.
    """

    return canonical_stage_base(node)


def build_pythia_profiler(templates: Mapping[str, Any], *,
                          min_prob: float = DEFAULT_MIN_PROB,
                          horizon: int = DEFAULT_HORIZON) -> Dict[str, Any]:
    """Build the role-alphabet PFA from TRAIN-ONLY templates.

    For each role the profiler records the transition distribution over the next role
    (with a terminal ``<end>`` outcome), the mean intrinsic service duration of entering
    that role, and the expected remaining distance computed by a finite-horizon
    recursion over the PRUNED automaton.
    """

    transitions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    durations: Dict[str, List[float]] = defaultdict(list)
    starts: Dict[str, int] = defaultdict(int)
    n_runs = 0

    for tpl in templates.values():
        if getattr(tpl, "split", None) != "train":
            continue
        n_runs += 1
        order = sorted(tpl.nodes, key=lambda n: n.sequence_index)
        if not order:
            continue
        starts[role_alphabet(order[0])] += 1
        for index, node in enumerate(order):
            here = role_alphabet(node)
            durations[here].append(intrinsic_duration_ms(node))
            following = role_alphabet(order[index + 1]) if index + 1 < len(order) else END
            transitions[here][following] += 1

    if n_runs == 0:
        raise ValueError("no train templates among the %d supplied; refusing to build a "
                         "Pythia profiler" % len(templates))

    mean_duration = {role: sum(v) / len(v) for role, v in durations.items()}
    raw_edges: Dict[str, Dict[str, float]] = {}
    for role, counts in transitions.items():
        total = float(sum(counts.values()))
        raw_edges[role] = {nxt: c / total for nxt, c in counts.items()}

    # prune, then renormalise, so every retained row still sums to 1
    pruned: Dict[str, Dict[str, float]] = {}
    pruned_away: List[Tuple[str, str, float]] = []
    for role, edges in raw_edges.items():
        kept = {nxt: p for nxt, p in edges.items() if p >= min_prob}
        if not kept:
            best = max(edges.items(), key=lambda kv: kv[1])
            kept = {best[0]: best[1]}
        for nxt, p in edges.items():
            if nxt not in kept:
                pruned_away.append((role, nxt, p))
        z = sum(kept.values())
        pruned[role] = {nxt: p / z for nxt, p in kept.items()}

    total_starts = float(sum(starts.values()))
    start_prob = {role: c / total_starts for role, c in starts.items()}

    # Finite-horizon expected remaining DISTANCE: the expected NUMBER OF FUTURE ROLE
    # TRANSITIONS, not a duration.  Pythia Algorithm 3's ``D_remaining`` is the expected
    # distance to the terminal on the probabilistic regex, and the paper's own
    # DownstreamIdleRisk is phrased as "one step vs ten steps".  The previous version
    # accumulated per-role millisecond durations here, which turned ``S_completion`` into
    # a duration-weighted (SRTF-flavoured) priority rather than Pythia's.  A transition
    # to END is not a role, so it is not counted: a last role has distance 0 and the
    # ``+1`` in ``s_completion`` accounts for the current step itself.
    value: Dict[str, float] = {role: 0.0 for role in pruned}
    for _ in range(int(horizon)):
        nxt_value: Dict[str, float] = {}
        for role, edges in pruned.items():
            acc = 0.0
            for following, prob in edges.items():
                if following == END:
                    continue
                acc += prob * (1.0 + value.get(following, 0.0))
            nxt_value[role] = acc
        value = nxt_value

    return {
        "schema": PROFILER_SCHEMA,
        "alphabet": sorted(pruned),
        "edge_prob": pruned,
        # Descriptive only: the train intrinsic durations of each role.  The completion
        # priority must NOT read this field -- it is the distance in STEPS that ranks.
        "mean_duration_ms": {k: float(v) for k, v in mean_duration.items()},
        "start_prob": start_prob,
        "expected_remaining_steps_by_role": {k: float(v) for k, v in value.items()},
        "pruned_edges": [{"from": a, "to": b, "prob": p} for a, b, p in sorted(pruned_away)],
        "n_runs": n_runs,
        "min_prob": float(min_prob),
        "horizon": int(horizon),
        "alphabet_source": "VideoSeek role ontology (role:action_family:raw_action)",
        "deviation": "finite-horizon recursion replaces the stationary solve, because the "
                     "automaton is cyclic and the undiscounted expectation does not converge. "
                     "This is our fixed, train-independent step limit and is NOT the paper's "
                     "bound, which prunes transitions and applies repetition bounds to looping "
                     "roles",
        "adapted": [
            "role key composed from role:action_family:raw_action; the paper's alphabet entry "
            "is the current agent_id and a bare VideoSeek role is too coarse",
            "the horizon is a fixed computational approximation rather than the paper's "
            "pruning-plus-repetition-bound automaton",
        ],
        "omissions": [
            "cache routing, prefix caching, model-replica idleness, autoscaling",
            "S_unblock (no model-server queue in this simulator); omega2 = 0",
        ],
    }


def expected_remaining_steps(profiler: Mapping[str, Any], role: str) -> float:
    """V(role): the expected NUMBER OF FUTURE ROLE TRANSITIONS after a role.

    V(role) already EXCLUDES the role it is indexed by: it counts the roles that follow.
    Indexing by the last COMPLETED role instead would use the wrong state and count the
    current candidate twice.  Because this is a DISTANCE IN STEPS, the caller must not
    add the current node's millisecond duration; ``s_completion`` applies the ``+1`` that
    accounts for the current step.  This is Pythia Algorithm 3's ``D_remaining``; the
    previous millisecond-valued version was a duration-weighted priority, not Pythia's.
    """

    if profiler.get("schema") != PROFILER_SCHEMA:
        raise ValueError("Pythia profiler schema mismatch: %r" % (profiler.get("schema"),))
    table = profiler["expected_remaining_steps_by_role"]
    if role not in table:
        raise KeyError(
            "role %r is not in the frozen role alphabet; the ontology and the profiler "
            "disagree" % (role,)
        )
    return float(table[role])


def current_role(job: Any, node_id: str) -> str:
    """The role of the node the scheduler is about to dispatch.

    This is the quantity Pythia exposes on a request as ``agent_id``: the node is already
    ready, so its role is known and this is not a future leak.  It is deliberately NOT the
    last completed role, because ``V`` excludes the role it is indexed by and pairing the
    two would count the current candidate twice.
    """

    node = job.template.by_id.get(node_id)
    if node is None:
        raise KeyError("node %r is not in the job's template" % (node_id,))
    return role_alphabet(node)


def s_completion(remaining_steps: float) -> float:
    """``S_completion = 1 / (1 + E[D_remaining])``; larger is better, so the key uses -S.

    ``D_remaining`` is a DISTANCE IN STEPS (see ``expected_remaining_steps``).  The ``+1``
    is the current step itself, so a ready final role scores 1/1 = 1 and a role with two
    further transitions scores 1/3.  A millisecond argument here would be a unit error.
    """

    steps = max(0.0, float(remaining_steps))
    return 1.0 / (1.0 + steps)
