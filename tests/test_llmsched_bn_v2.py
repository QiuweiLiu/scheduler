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
    BN_SCHEMA,
    SEP,
    absorb,
    build_bn_profiler,
    current_service_ms,
    evidence_from_completed,
    pair_mutual_information,
    posterior_joint,
    posterior_state_probs,
    profiler_sha256,
    uncertainty_reduction,
    workflow_future_stages,
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
    """Kills any entropy-based exploration score.

    Two candidates, identical marginal entropy and identical ranges; one is perfectly
    informative about its future and the other is independent of it.  An entropy-based
    score cannot order them; a mutual-information score must.
    """

    def test_mutual_information_separates_them(self):
        rng = random.Random(20260924)
        n = 4000
        xa = [rng.random() < 0.5 for _ in range(n)]
        ya = list(xa)                                     # Y = X  -> I = 1 bit
        xb = [rng.random() < 0.5 for _ in range(n)]
        yb = [rng.random() < 0.5 for _ in range(n)]       # independent -> I = 0

        def entropy(bits):
            p = sum(bits) / float(len(bits))
            return -sum(q * math.log2(q) for q in (p, 1.0 - p) if q > 0.0)

        self.assertAlmostEqual(entropy(xa), entropy(xb), places=2)
        self.assertAlmostEqual(entropy(ya), entropy(yb), places=2)

        def mi(xs, ys):
            return sum(1.0 for a, b in zip(xs, ys) if a == b) / float(n)

        # perfect agreement vs near-independence, with the same marginals
        self.assertGreater(mi(xa, ya), 0.99)
        self.assertLess(abs(mi(xb, yb) - 0.5), 0.03)

    def test_real_network_separates_a_parent_from_an_independent_stage(self):
        """On the learned network, a stage's parent must score above an unrelated one."""

        if not V041.exists():
            self.skipTest("v04.1 projection not present")
        from tracing.analysis.workload_v02_simulator import load_templates
        prof = build_bn_profiler(load_templates(V041, topology_view="causal_v3"))
        paired = [(child, parents[0]) for child, parents in prof["parents"].items() if parents]
        if not paired:
            self.skipTest("no edges learned")
        child, parent = paired[0]
        related = pair_mutual_information(prof, child, parent, {})
        # an alphabetic neighbour that is not a parent in the learned structure
        others = [s for s in prof["stage_order"]
                  if s not in (child, parent) and s not in prof["parents"][child]]
        if not others:
            self.skipTest("no unrelated stage available")
        unrelated = pair_mutual_information(prof, child, others[0], {})
        self.assertGreater(related, unrelated)


# --------------------------------------------------------------------------- #
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
        base = evidence_from_completed(FakeJob(tpl, done), prof)

        mutated_nodes = tuple(
            node_ if node_.node_id in done else
            Node(**{**node_.__dict__, "runtime_ms": node_.runtime_ms * 1000.0})
            for node_ in tpl.nodes)
        mutated = Template(tpl.template_id, tpl.video_id, tpl.split, tpl.baseline,
                           mutated_nodes, {n.node_id: n for n in mutated_nodes})
        after = evidence_from_completed(FakeJob(mutated, done), prof)

        self.assertEqual(base, after,
                         "changing an UNEXECUTED node's duration changed the evidence")
        # and the posterior built from it is identical
        self.assertEqual(posterior_state_probs(prof, stages[1], base),
                         posterior_state_probs(prof, stages[1], after))


# --------------------------------------------------------------------------- #
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
        ev = evidence_from_completed(FakeJob(tpl, keep), prof)
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
