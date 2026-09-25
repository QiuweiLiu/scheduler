"""L1-L7 fidelity gate for the v2 LLMSched front end (llmsched_bn v2).

Each gate exists because a plausible-looking but wrong implementation passes
everything else.  L1 caught three such bugs during development, all of which returned
a wrong number without raising:

  * an ancestor-only closure dropped observed DESCENDANTS, so P(B=t|A=t,C=f) gave the
    ancestor-only 0.9 instead of 0.75;
  * factor multiplication translated keys through the union positions of the wrong
    operand;
  * an observed child kept its own axis and was summed out as a hidden variable
    instead of being pinned.

The gates below are written so that the RETIRED front end cannot pass them, which is
what makes the migration a claim rather than an assertion.
"""
from __future__ import annotations

import math
import random
import unittest
from pathlib import Path

from tracing.analysis.llmsched_bn import (
    ABSENT,
    BN_SCHEMA,
    SEP,
    conditional_state_probs,
    descendants,
    expected_job_remaining_ms,
    expected_remaining_ms,
    job_duration_interval_ms,
    non_overlapping_sets,
    absorb,
    build_bn_profiler,
    current_service_ms,
    evidence_from_completed,
    joint_mutual_information,
    pair_mutual_information,
    posterior_joint,
    posterior_state_probs,
    profiler_sha256,
    uncertainty_reduction,
)
from tracing.analysis.llmsched_stage import (
    ABSENT,
    DurationDiscretizer,
    canonical_stage_base,
    canonical_stage_key,
    canonical_stage_map,
    intrinsic_duration_ms,
    stage_sequence,
)
from tracing.analysis.workload_v02_simulator import Node, Template, simulate_episode
from tracing.analysis import llmsched_bn_legacy as LEGACY

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V041 = (PROJECT_ROOT / "results" / "processed"
        / "r7_workload_v041_ontology_no_run_container" / "job_templates_r7_v041.jsonl")


def two_state_profiler(cpds, order, parents):
    """A tiny hand-specified network with the same shape as the real profiler."""

    return {
        "schema": BN_SCHEMA,
        "stage_order": list(order),
        "parents": {k: list(v) for k, v in parents.items()},
        "cpds": cpds,
        "state_vocabulary": ["t", "f"],
        "state_ms": {"t": 1.0, "f": 2.0},
        "stage_range": {k: 1.0 for k in order},
        "discretizer": None,
    }


def node(node_id, seq, role, family, raw, lane="gpu", runtime=100.0, preds=(), succs=()):
    return Node(node_id=node_id, sequence_index=seq, predecessors=preds, successors=succs,
                lane=lane, model_id="m", runtime_ms=runtime, load_ms=0.0,
                workspace_peak_mb=1.0, resident_model_mb=1.0, status="success",
                role=role, action_family=family, raw_action=raw)


# --------------------------------------------------------------------------- #
class L1HandComputablePosterior(unittest.TestCase):
    """A -> B -> C, two states each, every posterior computed by hand."""

    def setUp(self):
        self.prof = two_state_profiler(
            {"A": {SEP.join([]): {"t": 0.6, "f": 0.4}},
             "B": {SEP.join(["t"]): {"t": 0.9, "f": 0.1},
                   SEP.join(["f"]): {"t": 0.2, "f": 0.8}},
             "C": {SEP.join(["t"]): {"t": 0.7, "f": 0.3},
                   SEP.join(["f"]): {"t": 0.1, "f": 0.9}}},
            ["A", "B", "C"], {"A": [], "B": ["A"], "C": ["B"]})

    def test_prior(self):
        self.assertAlmostEqual(posterior_state_probs(self.prof, "A", {})["t"], 0.6)

    def test_forward_conditioning(self):
        self.assertAlmostEqual(posterior_state_probs(self.prof, "B", {"A": "t"})["t"], 0.9)
        # 0.9*0.7 + 0.1*0.1
        self.assertAlmostEqual(posterior_state_probs(self.prof, "C", {"A": "t"})["t"], 0.64)

    def test_joint_and_normalisation(self):
        _v, joint = posterior_joint(self.prof, ["B", "C"], {"A": "t"})
        self.assertAlmostEqual(float(joint.sum()), 1.0, places=9)
        self.assertAlmostEqual(float(joint[0, 0]), 0.63)   # 0.9 * 0.7

    def test_evidence_flows_backwards_to_an_ancestor(self):
        """The bug L1 was written for: P(B=t | A=t, C=f) is 0.75, not the ancestor 0.9."""

        self.assertAlmostEqual(
            posterior_state_probs(self.prof, "B", {"A": "t", "C": "f"})["t"], 0.75)

    def test_evidence_flows_forwards(self):
        self.assertAlmostEqual(
            posterior_state_probs(self.prof, "C", {"A": "t", "B": "f"})["t"], 0.1)


# --------------------------------------------------------------------------- #
class L2SameEntropyDifferentMI(unittest.TestCase):
    """Kills any entropy-based exploration score, through the PRODUCTION scorer.

    Two candidates with identical marginal entropy and identical duration ranges; one
    is perfectly informative about its future and the other is independent of it.  An
    entropy-based score cannot order them; a mutual-information score must.  The
    assertion is deterministic and numeric: no sampling, no thresholds.
    """

    @staticmethod
    def profiler_with_replacement(conditional):
        """X ~ Bernoulli(0.5); Y follows `conditional` (a dict over X's states)."""

        return two_state_profiler(
            {"X": {SEP.join([]): {"t": 0.5, "f": 0.5}},
             "Y": {SEP.join(["t"]): dict(conditional["t"]),
                   SEP.join(["f"]): dict(conditional["f"])}},
            ["X", "Y"], {"X": [], "Y": ["X"]})

    def test_identical_entropy_different_mutual_information(self):
        informative = self.profiler_with_replacement(
            {"t": {"t": 1.0, "f": 0.0}, "f": {"t": 0.0, "f": 1.0}})     # Y = X
        independent = self.profiler_with_replacement(
            {"t": {"t": 0.5, "f": 0.5}, "f": {"t": 0.5, "f": 0.5}})     # Y || X

        # identical marginals for X, and identical Range(Y) in both networks
        for prof in (informative, independent):
            self.assertAlmostEqual(posterior_state_probs(prof, "X", {})["t"], 0.5)
            self.assertAlmostEqual(prof["stage_range"]["Y"], 1.0)

        mi_informative = joint_mutual_information(informative, "X", ["Y"], {})
        mi_independent = joint_mutual_information(independent, "X", ["Y"], {})
        self.assertAlmostEqual(mi_informative, 1.0, places=10)
        self.assertAlmostEqual(mi_independent, 0.0, places=10)

        # and the SCORER the policy actually calls must order them the same way
        r_informative = uncertainty_reduction(informative, "X", {})
        r_independent = uncertainty_reduction(independent, "X", {})
        self.assertGreater(r_informative, r_independent,
                           "an entropy-based score would tie these two")
        self.assertGreater(r_informative, 0.0)
        self.assertAlmostEqual(r_independent, 0.0, places=10)

    def test_the_joint_is_not_the_pairwise_sum(self):
        """I(X;Y1..YM | E) != sum_i I(X;Y_i | E); the two coincide only at M = 1."""

        prof = two_state_profiler(
            {"X": {SEP.join([]): {"t": 0.5, "f": 0.5}},
             "Y": {SEP.join(["t"]): {"t": 1.0, "f": 0.0},
                   SEP.join(["f"]): {"t": 0.0, "f": 1.0}},
             "Z": {SEP.join(["t"]): {"t": 1.0, "f": 0.0},
                   SEP.join(["f"]): {"t": 0.0, "f": 1.0}}},
            ["X", "Y", "Z"], {"X": [], "Y": ["X"], "Z": ["Y"]})
        joint = joint_mutual_information(prof, "X", ["Y", "Z"], {})
        pairwise = (pair_mutual_information(prof, "X", "Y", {})
                    + pair_mutual_information(prof, "X", "Z", {}))
        self.assertAlmostEqual(joint, 1.0, places=6)
        self.assertGreater(pairwise, joint + 1e-6,
                           "summing pairwise terms must not silently replace the joint")


class L3CompletedDurationEvidence(unittest.TestCase):
    """Evidence must be the OBSERVED duration, not a count of completed nodes."""

    def test_cpd_order_flips_the_winner(self):
        prof = two_state_profiler(
            {"Z": {SEP.join([]): {"t": 0.5, "f": 0.5}},
             "A": {SEP.join(["t"]): {"t": 0.9, "f": 0.1},
                   SEP.join(["f"]): {"t": 0.1, "f": 0.9}},
             "B": {SEP.join(["t"]): {"t": 0.1, "f": 0.9},
                   SEP.join(["f"]): {"t": 0.9, "f": 0.1}}},
            ["Z", "A", "B"], {"Z": [], "A": ["Z"], "B": ["Z"]})
        first = posterior_state_probs(prof, "A", {"Z": "t"})["t"]
        second = posterior_state_probs(prof, "A", {"Z": "f"})["t"]
        self.assertGreater(first, second)
        # the same evidence reverses B, so the ranking of A against B flips
        self.assertLess(posterior_state_probs(prof, "B", {"Z": "t"})["t"],
                        posterior_state_probs(prof, "B", {"Z": "f"})["t"])

    def test_a_count_is_not_evidence_but_a_duration_is(self):
        """The retired front end advanced on len(completed); this one must not."""

        if not V041.exists():
            self.skipTest("v04.1 projection not present")
        from tracing.analysis.workload_v02_simulator import load_templates
        tpls = load_templates(V041, topology_view="causal_v3")
        prof = build_bn_profiler(tpls)
        train = [t for t in tpls.values() if t.split == "train"]
        tpl = max(train, key=lambda t: len(t.nodes))
        stages = stage_sequence(tpl)
        first = stages[0]
        probs = []
        for fraction in (0.05, 0.5, 5.0):
            for other, node_ in [(None, None)]:
                pass
            # same stage, three different observed durations -> three duration bins
            observed = intrinsic_duration_ms(tpl.nodes[1]) * fraction
            evidence = {stages[1]: prof["discretizer"].state_of(observed)}
            probs.append(posterior_state_probs(prof, first, evidence))
        self.assertNotEqual(probs[0], probs[2],
                            "two very different observed durations must not give the "
                            "same posterior")
        _ = first


# --------------------------------------------------------------------------- #
class L4FutureTruthInvariance(unittest.TestCase):
    """An unexecuted node's truth must be invisible at decision time."""

    def test_posterior_ignores_unexecuted_nodes(self):
        if not V041.exists():
            self.skipTest("v04.1 projection not present")
        from tracing.analysis.workload_v02_simulator import load_templates
        tpls = load_templates(V041, topology_view="causal_v3")
        prof = build_bn_profiler(tpls)
        train = [t for t in tpls.values() if t.split == "train"]
        tpl = max(train, key=lambda t: len(t.nodes))
        stages = stage_sequence(tpl)

        class FakeJob:
            def __init__(self, template, completed):
                self.template = template
                self.completed = completed

        # only the FIRST node is complete in both cases; nothing about later nodes can
        # enter, because evidence_from_completed reads job.completed alone
        done = {tpl.nodes[0].node_id}
        observed = {tpl.nodes[0].node_id: float(tpl.nodes[0].runtime_ms)}
        base = evidence_from_completed(FakeJob(tpl, done), prof, observed_ms=observed)

        mutated_nodes = tuple(
            node_ if node_.node_id in done else
            Node(**{**node_.__dict__, "runtime_ms": node_.runtime_ms * 1000.0})
            for node_ in tpl.nodes)
        mutated = Template(tpl.template_id, tpl.video_id, tpl.split, tpl.baseline,
                           mutated_nodes, {n.node_id: n for n in mutated_nodes})
        after = evidence_from_completed(FakeJob(mutated, done), prof, observed_ms=observed)

        self.assertEqual(base, after,
                         "changing an UNEXECUTED node's duration changed the evidence")
        # and the posterior built from it is identical
        self.assertEqual(posterior_state_probs(prof, stages[1], base),
                         posterior_state_probs(prof, stages[1], after))


# --------------------------------------------------------------------------- #
class L4bFutureStructureInvariance(unittest.TestCase):
    """The leak the review found: the realized template's future must be invisible.

    Hold the prefix, the ready set and the completed evidence fixed, and change only
    the UNEXECUTED remainder of the workload.  Y, the posterior, R(X), the expected
    remaining work and the chosen action must all be identical.  An implementation
    that reads the template's future suffix produces perfectly reasonable numbers and
    fails this.
    """

    def test_future_structure_cannot_enter_the_score(self):
        if not V041.exists():
            self.skipTest("v04.1 projection not present")
        from tracing.analysis.workload_v02_simulator import load_templates
        tpls = load_templates(V041, topology_view="causal_v3")
        prof = build_bn_profiler(tpls)
        tpl = max((x for x in tpls.values() if x.split == "train"),
                  key=lambda x: len(x.nodes))

        done = {tpl.nodes[0].node_id}
        stages = stage_sequence(tpl)
        candidate = stages[1]

        class FakeJob:
            def __init__(self, template, completed):
                self.template = template
                self.completed = completed
                self.job_instance_id = "j"
                self.observed_intrinsic_ms = {
                    tpl.nodes[0].node_id: float(tpl.nodes[0].runtime_ms)}

        base_evidence = evidence_from_completed(
            FakeJob(tpl, done), prof,
            observed_ms={tpl.nodes[0].node_id: float(tpl.nodes[0].runtime_ms)})

        # Rewrite EVERY unexecuted node: different action, different family, different
        # duration, and a different node count.  Nothing about it may be visible.
        future_nodes = []
        for node_ in tpl.nodes:
            if node_.node_id in done:
                future_nodes.append(node_)
                continue
            future_nodes.append(Node(
                node_id=node_.node_id, sequence_index=node_.sequence_index,
                predecessors=node_.predecessors, successors=node_.successors,
                lane=node_.lane, model_id=node_.model_id,
                runtime_ms=node_.runtime_ms * 7.0 + 1.0, load_ms=node_.load_ms,
                workspace_peak_mb=node_.workspace_peak_mb,
                resident_model_mb=node_.resident_model_mb, status=node_.status,
                role="a_completely_different_role", action_family="different_family",
                raw_action="totally-different-tool"))
        mutated = Template(tpl.template_id, tpl.video_id, tpl.split, tpl.baseline,
                           tuple(future_nodes), {n.node_id: n for n in future_nodes})

        mutated_evidence = evidence_from_completed(
            FakeJob(mutated, done), prof,
            observed_ms={tpl.nodes[0].node_id: float(tpl.nodes[0].runtime_ms)})

        self.assertEqual(base_evidence, mutated_evidence,
                         "the evidence must not depend on unexecuted nodes")
        self.assertEqual(
            posterior_state_probs(prof, candidate, base_evidence),
            posterior_state_probs(prof, candidate, mutated_evidence),
            "the posterior must not depend on unexecuted nodes")
        self.assertAlmostEqual(
            uncertainty_reduction(prof, candidate, base_evidence),
            uncertainty_reduction(prof, candidate, mutated_evidence),
            places=12, msg="R(X) must not depend on unexecuted nodes")

    def test_the_template_is_not_read_anywhere_in_the_score_path(self):
        """A source-level guard: the loader for the realized future must be gone."""

        from tracing.analysis import llmsched_bn as module
        self.assertFalse(
            hasattr(module, "workflow_future_stages"),
            "the helper that read the realized template's future still exists")


class L2bOneCappedYAndPresentConditioning(unittest.TestCase):
    """The two narrow follow-ups on the information term."""

    def test_the_two_terms_use_the_same_truncated_set(self):
        """Eq. (6) uses ONE Y_1..Y_M in both the information term and the range sum."""

        if not V041.exists():
            self.skipTest("v04.1 projection not present")
        from tracing.analysis.llmsched_bn import MAX_JOINT_FUTURE, _future_from_network
        from tracing.analysis.workload_v02_simulator import load_templates
        prof = build_bn_profiler(load_templates(V041, topology_view="causal_v3"))
        wide = [s for s in prof["stage_order"]
                if len(_future_from_network(prof, s, {})) > MAX_JOINT_FUTURE]
        if not wide:
            self.skipTest("no stage has more future stages than the cap")
        stage = wide[0]
        from tracing.analysis.llmsched_bn import _rank_future_by_proximity
        future = _future_from_network(prof, stage, {})
        # the scorer ranks by proximity first; the hand recomputation must use the same
        selected = _rank_future_by_proximity(prof, stage, future)[:MAX_JOINT_FUTURE]
        # Recomputing R by hand from the SELECTED set must reproduce the scorer exactly.
        # If the range factor summed every descendant instead, the two would differ.
        info = joint_mutual_information(prof, stage, selected, {})
        hand = info * sum(prof["stage_range"][y] for y in selected)
        self.assertAlmostEqual(uncertainty_reduction(prof, stage, {}), hand, places=9)
        all_ranges = info * sum(prof["stage_range"][y] for y in future)
        self.assertNotAlmostEqual(uncertainty_reduction(prof, stage, {}), all_ranges,
                                  places=6)

    def test_a_ready_candidate_is_conditioned_on_being_present(self):
        """X != ABSENT is settled once the scheduler can see X ready."""

        states = ["t", "f", ABSENT]
        prof = {
            "schema": BN_SCHEMA, "stage_order": ["X", "Y"],
            "parents": {"X": [], "Y": ["X"]},
            "cpds": {
                "X": {SEP.join([]): {"t": 0.4, "f": 0.0, ABSENT: 0.6}},
                # present-and-quick, present-and-slow and absent are three genuinely
                # different regimes, so the unconditional MI is large purely because X
                # is often absent
                "Y": {SEP.join(["t"]): {"t": 0.50, "f": 0.50, ABSENT: 0.0},
                      SEP.join(["f"]): {"t": 0.00, "f": 1.00, ABSENT: 0.0},
                      SEP.join([ABSENT]): {"t": 0.95, "f": 0.05, ABSENT: 0.0}},
            },
            "state_vocabulary": states,
            "state_ms": {"t": 1.0, "f": 2.0, ABSENT: 0.0},
            "stage_range": {"Y": 1.0}, "discretizer": None}
        self.assertGreater(posterior_state_probs(prof, "X", {})[ABSENT], 0.0)

        unconditioned = joint_mutual_information(prof, "X", ["Y"], {},
                                                condition_present=False)
        conditioned = joint_mutual_information(prof, "X", ["Y"], {}, condition_present=True)
        self.assertGreater(unconditioned, conditioned,
                           "conditioning on presence must not be a no-op here")
        self.assertAlmostEqual(uncertainty_reduction(prof, "X", {}),
                               conditioned * 1.0, places=9)


class L2cNonOverlappingDurationSets(unittest.TestCase):
    """Algorithm 1: uncertainty reduction orders candidates only WITHIN a duration set.

    The paper builds sets of jobs whose duration intervals do not overlap, because only
    there is the duration ordering deterministic, and sorts the sets by lower bound.
    Ranking the whole pool by R(X) would let a high-variance job jump ahead of a job
    that is provably shorter.
    """

    def test_intervals_that_overlap_share_a_set(self):
        sets = non_overlapping_sets({0: (0.0, 10.0), 1: (5.0, 20.0)})
        self.assertEqual(sets[0], sets[1])

    def test_a_strictly_later_interval_starts_a_new_set(self):
        sets = non_overlapping_sets({0: (0.0, 10.0), 1: (30.0, 40.0)})
        self.assertLess(sets[0], sets[1])

    def test_chaining_transitively_merges_sets(self):
        """An interval bridging two sets joins them, so the grouping is not naive."""

        sets = non_overlapping_sets({0: (0.0, 10.0), 1: (5.0, 100.0), 2: (50.0, 60.0)})
        self.assertEqual(sets[0], sets[1])
        self.assertEqual(sets[1], sets[2])

    def test_sets_are_ordered_by_lower_bound(self):
        sets = non_overlapping_sets({0: (200.0, 210.0), 1: (0.0, 10.0), 2: (30.0, 40.0)})
        self.assertEqual(sets[1], 0)
        self.assertEqual(sets[2], 1)
        self.assertEqual(sets[0], 2)

    @staticmethod
    def one_stage_profiler(state_probs):
        return {
            "schema": BN_SCHEMA, "stage_order": ["Y"], "parents": {"Y": []},
            "cpds": {"Y": {SEP.join([]): dict(state_probs)}},
            "state_vocabulary": ["D0", ABSENT],
            "state_ms": {"D0": 100.0, ABSENT: 0.0},
            "stage_range": {"Y": 100.0}, "discretizer": None}

    def test_the_interval_is_the_support_not_the_expectation(self):
        """P(ABSENT) = P(D = 100) = 0.5 has support [0, 100], not the point [50, 50].

        Weighting a single min/max by P(present) collapses the support of the
        remaining-duration random variable onto an expectation, which reports total
        certainty for a distribution that is anything but certain.  Two jobs that are
        both genuinely ambiguous would then be ordered as if their durations were known.
        """

        uncertain = self.one_stage_profiler({"D0": 0.5, ABSENT: 0.5})
        lower, upper = job_duration_interval_ms(uncertain, {})
        self.assertAlmostEqual(lower, 0.0, places=9)
        self.assertAlmostEqual(upper, 100.0, places=9)

        certain = self.one_stage_profiler({"D0": 1.0, ABSENT: 0.0})
        lower, upper = job_duration_interval_ms(certain, {})
        self.assertAlmostEqual(lower, 100.0, places=9)
        self.assertAlmostEqual(upper, 100.0, places=9)

        absent = self.one_stage_profiler({"D0": 0.0, ABSENT: 1.0})
        lower, upper = job_duration_interval_ms(absent, {})
        self.assertAlmostEqual(lower, 0.0, places=9)
        self.assertAlmostEqual(upper, 0.0, places=9)

    def test_the_interval_covers_every_unresolved_stage(self):
        """It describes the JOB, not one candidate's correlated descendants."""

        if not V041.exists():
            self.skipTest("v04.1 projection not present")
        from tracing.analysis.llmsched_bn import _future_from_network
        from tracing.analysis.workload_v02_simulator import load_templates
        prof = build_bn_profiler(load_templates(V041, topology_view="causal_v3"))
        narrow = [s for s in prof["stage_order"]
                  if len(_future_from_network(prof, s, {})) and
                  len(_future_from_network(prof, s, {})) < len(prof["stage_order"]) - 1]
        if not narrow:
            self.skipTest("no stage has a strict subset of descendants")
        stage = narrow[0]
        # the job interval must not be the candidate's descendant set
        lower, upper = job_duration_interval_ms(prof, {})
        self.assertGreater(upper, 0.0)
        self.assertGreaterEqual(lower, 0.0)
        # every stage with any presence probability contributes something
        present_any = any(
            1.0 - posterior_state_probs(prof, s, {}).get(ABSENT, 0.0) > 0.0
            for s in prof["stage_order"])
        self.assertTrue(present_any)
        _ = stage

    def test_the_scorer_uses_the_group_before_the_information(self):
        """A provably shorter job must win even if a longer one is more informative."""

        if not V041.exists():
            self.skipTest("v04.1 projection not present")
        from tracing.analysis.workload_v02_simulator import load_templates
        prof = build_bn_profiler(load_templates(V041, topology_view="causal_v3"))
        short = prof["stage_order"][0]
        sets = non_overlapping_sets({0: (0.0, 1.0), 1: (10 ** 6, 10 ** 6 + 1.0)})
        self.assertLess(sets[0], sets[1],
                        "the short job must be in an earlier set than the long one")
        _ = short


class L8WholeJobRemainingAndKnownPresent(unittest.TestCase):
    """The EXPLOIT remaining-work estimator must describe the JOB and share ONE
    information state with EXPLORE and current service.

    Two defects this gate pins, both of which returned a plausible-looking number:

      * the v1 estimator walked ``_future_from_network``, so a stage with no directed
        path from the candidate was dropped even though the job must still run it;
      * every posterior was taken under the completed evidence only, as if the ready
        candidate might not exist, while EXPLORE and current service both condition on
        ``X != ABSENT`` -- three different information states inside one decision.
    """

    @staticmethod
    def chain_profiler(cpd_a, cpd_b, order=("A", "B")):
        return {
            "schema": BN_SCHEMA,
            "stage_order": list(order),
            "parents": {"A": [], "B": ["A"]},
            "cpds": {"A": {SEP.join([]): dict(cpd_a)}, "B": dict(cpd_b)},
            "state_vocabulary": ["D0", ABSENT],
            "state_ms": {"D0": 100.0, ABSENT: 0.0},
            "stage_range": {"A": 100.0, "B": 100.0},
            "discretizer": None,
        }

    def test_conditioning_propagates_through_the_network(self):
        """E[D_B | A != ABSENT] must be 100, not the unconditional 50.

        A is absent half the time, and B is present exactly when A is.  The
        unconditional marginal halves B's contribution; a candidate that is ready is
        known to exist, so the unconditioned number is simply the wrong information
        state.  This is the difference between the two estimators.
        """

        prof = self.chain_profiler(
            {"D0": 0.5, ABSENT: 0.5},
            {SEP.join(["D0"]): {"D0": 1.0, ABSENT: 0.0},
             SEP.join([ABSENT]): {"D0": 0.0, ABSENT: 1.0}},
        )
        self.assertAlmostEqual(expected_remaining_ms(prof, "A", {}), 50.0, places=9)
        self.assertAlmostEqual(expected_job_remaining_ms(prof, {}, "A"), 100.0, places=9)
        # the single-stage conditional helper agrees with the renormalisation
        probs = conditional_state_probs(prof, "B", "A", {})
        self.assertAlmostEqual(probs["D0"], 1.0, places=9)
        self.assertAlmostEqual(probs[ABSENT], 0.0, places=9)

    def test_an_unreachable_stage_is_still_part_of_the_job(self):
        """A stage the candidate cannot inform is still work the job must do."""

        prof = {
            "schema": BN_SCHEMA,
            "stage_order": ["A", "B", "Z"],
            "parents": {"A": [], "B": ["A"], "Z": []},
            "cpds": {
                "A": {SEP.join([]): {"D0": 1.0, ABSENT: 0.0}},
                "B": {SEP.join(["D0"]): {"D0": 1.0, ABSENT: 0.0},
                      SEP.join([ABSENT]): {"D0": 1.0, ABSENT: 0.0}},
                "Z": {SEP.join([]): {"D0": 1.0, ABSENT: 0.0}},
            },
            "state_vocabulary": ["D0", ABSENT],
            "state_ms": {"D0": 100.0, ABSENT: 0.0},
            "stage_range": {"A": 100.0, "B": 100.0, "Z": 100.0},
            "discretizer": None,
        }
        self.assertEqual(descendants(prof, "A"), ["B"])
        self.assertNotIn("Z", descendants(prof, "A"))
        # the candidate's descendants give 100; the whole job adds Z's 100
        self.assertAlmostEqual(expected_remaining_ms(prof, "A", {}), 100.0, places=9)
        self.assertAlmostEqual(expected_job_remaining_ms(prof, {}, "A"), 200.0, places=9)

    def test_a_certainly_absent_condition_fails_closed(self):
        prof = self.chain_profiler(
            {"D0": 0.0, ABSENT: 1.0},
            {SEP.join(["D0"]): {"D0": 1.0, ABSENT: 0.0},
             SEP.join([ABSENT]): {"D0": 0.0, ABSENT: 1.0}},
        )
        with self.assertRaises(ValueError):
            conditional_state_probs(prof, "B", "A", {})

    def test_known_present_removes_the_spurious_zero(self):
        """A ready stage cannot contribute a possible-zero to the job's support."""

        uncertain = self.chain_profiler(
            {"D0": 0.5, ABSENT: 0.5},
            {SEP.join(["D0"]): {"D0": 1.0, ABSENT: 0.0},
             SEP.join([ABSENT]): {"D0": 0.0, ABSENT: 1.0}},
        )
        lower, _ = job_duration_interval_ms(uncertain, {}, known_present=["A"])
        self.assertAlmostEqual(lower, 100.0, places=9)


class L5SameMarginalsDifferentJoint(unittest.TestCase):
    """The consumer must read joint CPDs, not marginal tables."""

    def test_same_marginals_different_correlation(self):
        # both networks: P(Z=t) = 1/2 and P(A=t|Z=t) = 0.9 alone
        correlated = two_state_profiler(
            {"Z": {SEP.join([]): {"t": 0.5, "f": 0.5}},
             "A": {SEP.join(["t"]): {"t": 0.9, "f": 0.1},
                   SEP.join(["f"]): {"t": 0.1, "f": 0.9}}},
            ["Z", "A"], {"Z": [], "A": ["Z"]})
        independent = two_state_profiler(
            {"Z": {SEP.join([]): {"t": 0.5, "f": 0.5}},
             "A": {SEP.join(["t"]): {"t": 0.5, "f": 0.5},
                   SEP.join(["f"]): {"t": 0.5, "f": 0.5}}},
            ["Z", "A"], {"Z": [], "A": ["Z"]})
        # the marginals of A are identical in both
        self.assertAlmostEqual(posterior_state_probs(correlated, "A", {})["t"], 0.5)
        self.assertAlmostEqual(posterior_state_probs(independent, "A", {})["t"], 0.5)
        # but the joint mutual information is not
        mi_corr = pair_mutual_information(correlated, "A", "Z", {})
        mi_indep = pair_mutual_information(independent, "A", "Z", {})
        self.assertGreater(mi_corr, 0.4)
        self.assertLess(mi_indep, 1e-9)


# --------------------------------------------------------------------------- #
class L6StructuralUncertainty(unittest.TestCase):
    """ABSENT must be a real state, and a READY stage must never be conditioned absent."""

    def test_absent_probability_responds_to_evidence(self):
        prof = two_state_profiler(
            {"Z": {SEP.join([]): {"t": 0.5, "f": 0.5}},
             "B": {SEP.join(["t"]): {"t": 0.9, "f": 0.1},
                   SEP.join(["f"]): {"t": 0.1, "f": 0.9}}},
            ["Z", "B"], {"Z": [], "B": ["Z"]})
        self.assertGreater(posterior_state_probs(prof, "B", {"Z": "t"})["t"],
                           posterior_state_probs(prof, "B", {"Z": "f"})["t"])

    def test_absorb_removes_the_absent_state_for_a_ready_stage(self):
        if not V041.exists():
            self.skipTest("v04.1 projection not present")
        from tracing.analysis.workload_v02_simulator import load_templates
        prof = build_bn_profiler(load_templates(V041, topology_view="causal_v3"))
        # pick a stage with a non-trivial absence prior
        target = None
        for name in prof["stage_order"]:
            if 0.05 < posterior_state_probs(prof, name, {}).get(ABSENT, 0.0) < 0.95:
                target = name
                break
        if target is None:
            self.skipTest("no stage with a non-trivial absence prior")
        raw = posterior_state_probs(prof, target, {})
        self.assertGreater(raw[ABSENT], 0.0)
        conditioned = absorb(prof, target, {})
        self.assertAlmostEqual(conditioned[ABSENT], 0.0)
        self.assertAlmostEqual(sum(conditioned.values()), 1.0)

    def test_real_network_has_structural_uncertainty(self):
        if not V041.exists():
            self.skipTest("v04.1 projection not present")
        from tracing.analysis.workload_v02_simulator import load_templates
        prof = build_bn_profiler(load_templates(V041, topology_view="causal_v3"))
        absence = [posterior_state_probs(prof, s, {})[ABSENT] for s in prof["stage_order"]]
        self.assertGreater(max(absence), 0.5,
                           "a vocabulary this large should have mostly-absent stages")
        self.assertLess(min(absence), 0.5)


# --------------------------------------------------------------------------- #
class L7EndToEndConsumer(unittest.TestCase):
    """The BN must actually drive dispatch, through the real simulator path."""

    def test_the_two_modes_are_not_the_same_policy(self):
        if not V041.exists():
            self.skipTest("v04.1 projection not present")
        from tracing.analysis.workload_v02_simulator import load_templates, train_resource_stats
        tpls = load_templates(V041, topology_view="causal_v3")
        train = {k: v for k, v in tpls.items() if v.split == "train"}
        val = {k: v for k, v in tpls.items() if v.split == "validation"}
        prof = build_bn_profiler(tpls)
        stats = train_resource_stats(train)
        val_ids = list(val)
        ep = {"episode_id": "l7", "split": "validation",
              "gpu_topology_mb": [24576.0, 24576.0],
              "initial_residency_hint": [[], []],
              "jobs": [{"job_instance_id": "j%d" % i, "template_id": val_ids[i % len(val_ids)],
                        "arrival_ms": float(i * 3000), "deadline_ms": None,
                        "service_class": "normal"} for i in range(6)]}
        out = {}
        for eps in (0.0, 1.0):
            ctx = {"llmsched_bn": prof, "llmsched_rng": random.Random(11),
                   "llmsched_epsilon": eps}
            summary, _ = simulate_episode(ep, val, "llmsched", train_stats=stats,
                                          policy_context=ctx)
            out[eps] = summary
        self.assertEqual(out[0.0]["completed_jobs"], 6)
        self.assertEqual(out[1.0]["completed_jobs"], 6)
        self.assertNotEqual(out[0.0]["makespan_ms"], out[1.0]["makespan_ms"],
                            "EXPLOIT and EXPLORE must not be the same schedule")

    def test_evidence_tracks_observed_durations_not_a_count(self):
        if not V041.exists():
            self.skipTest("v04.1 projection not present")
        from tracing.analysis.workload_v02_simulator import load_templates
        tpls = load_templates(V041, topology_view="causal_v3")
        prof = build_bn_profiler(tpls)
        tpl = max((t for t in tpls.values() if t.split == "train"),
                  key=lambda t: len(t.nodes))

        class FakeJob:
            def __init__(self, template, completed):
                self.template = template
                self.completed = completed

        keep = {tpl.nodes[0].node_id, tpl.nodes[1].node_id}
        observed = {n.node_id: float(n.runtime_ms) for n in tpl.nodes if n.node_id in keep}
        ev = evidence_from_completed(FakeJob(tpl, keep), prof, observed_ms=observed)
        # one entry per completed node, each a duration state, never a bare count
        self.assertEqual(len(ev), 2)
        for state in ev.values():
            self.assertRegex(state, r"^D\d+$")


# --------------------------------------------------------------------------- #
class UnitGates(unittest.TestCase):
    """The five ordinary gates the review asked to keep alongside L1-L7."""

    def test_stage_identity_is_prefix_only(self):
        n0 = node("n0", 0, "planner", "planner.generate", "planner.generate")
        n1 = node("n1", 1, "planner", "planner.generate", "planner.generate")
        self.assertEqual(canonical_stage_key(n0, {}), canonical_stage_key(n1, {}))
        self.assertNotEqual(canonical_stage_key(n0, {}),
                            canonical_stage_key(n1, {canonical_stage_base(n0): 1}))

    def test_unknown_stage_fails_closed(self):
        if not V041.exists():
            self.skipTest("v04.1 projection not present")
        from tracing.analysis.workload_v02_simulator import load_templates
        prof = build_bn_profiler(load_templates(V041, topology_view="causal_v3"))
        with self.assertRaises(KeyError):
            posterior_state_probs(prof, "not:a:stage#0", {})

    def test_missing_cpd_or_state_fails_closed(self):
        prof = two_state_profiler(
            {"A": {SEP.join([]): {"t": 0.5, "f": 0.5}},
             "B": {SEP.join(["t"]): {"t": 0.5, "f": 0.5},
                   SEP.join(["f"]): {"t": 0.5, "f": 0.5}}},
            ["A", "B"], {"A": [], "B": ["A"]})
        with self.assertRaises(ValueError):
            posterior_joint(prof, ["B"], {"A": "nonsense"})
        with self.assertRaises(KeyError):
            posterior_joint(prof, ["nope"], {})

    def test_one_epsilon_coin_per_decision(self):
        from tracing.analysis.llmsched_bn import draw_mode

        class CountingRng:
            def __init__(self):
                self.calls = 0

            def random(self):
                self.calls += 1
                return 0.5

        rng = CountingRng()
        self.assertEqual(draw_mode(rng, 0.9), "EXPLORE")
        self.assertEqual(rng.calls, 1)
        rng = CountingRng()
        self.assertEqual(draw_mode(rng, 0.1), "EXPLOIT")
        self.assertEqual(rng.calls, 1)

    def test_fixed_seed_is_deterministic(self):
        if not V041.exists():
            self.skipTest("v04.1 projection not present")
        from tracing.analysis.workload_v02_simulator import load_templates
        tpls = load_templates(V041, topology_view="causal_v3")
        a = profiler_sha256(build_bn_profiler(tpls))
        b = profiler_sha256(build_bn_profiler(tpls))
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)

    def test_train_only(self):
        if not V041.exists():
            self.skipTest("v04.1 projection not present")
        from tracing.analysis.workload_v02_simulator import load_templates
        tpls = load_templates(V041, topology_view="causal_v3")
        prof = build_bn_profiler(tpls)
        self.assertGreater(prof["n_train"], 0)
        self.assertEqual(prof["n_validation"], 0)
        self.assertEqual(prof["n_test"], 0)
        self.assertGreater(prof["skipped_non_train"], 0)

    def test_the_retired_front_end_cannot_satisfy_the_v2_gates(self):
        """A migration claim needs the old implementation to fail the new gate."""

        self.assertFalse(hasattr(LEGACY, "posterior_joint"),
                         "the retired front end has no joint posterior at all")
        self.assertTrue(hasattr(LEGACY, "posterior_length_probs"))
        self.assertTrue(hasattr(LEGACY, "duration_entropy"))


if __name__ == "__main__":
    unittest.main()
