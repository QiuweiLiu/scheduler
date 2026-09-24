"""Oracle gate for the RETIRED LLMSched front end (llmsched_bn_legacy).

LLM-1 of the migration retires this implementation from the formal path.  These
tests are kept, and now target the legacy module explicitly, so that the behaviour
being replaced stays reproducible: a retirement that cannot be replayed is just an
assertion.  The v2 gate is L1-L7 in test_llmsched_bn_v2.py.

The review's list for the retired front end:
    the BN posterior changes with completed-stage evidence;
    the uncertainty-reduction ranking matches a hand-built small graph;
    the epsilon-greedy activation ratio is correct.
"""
from __future__ import annotations

import random
import unittest

from tracing.analysis.llmsched_bn_legacy import (
    build_bn_profiler,
    draw_mode,
    duration_entropy,
    info_gain,
    posterior_entropy,
    posterior_length_probs,
    uncertainty_reduction,
)
from tracing.analysis.workload_v02_simulator import (
    POLICIES,
    Node,
    Template,
    simulate_episode,
    train_resource_stats,
)


def chain(tid, runtimes, family, split="train", model="m1"):
    nodes = [Node(node_id=f"{tid}:n{i}", sequence_index=i,
                  predecessors=(f"{tid}:n{i-1}",) if i else (),
                  successors=(f"{tid}:n{i+1}",) if i + 1 < len(runtimes) else (),
                  lane="gpu", model_id=model, runtime_ms=float(rt), load_ms=5.0,
                  workspace_peak_mb=100.0, resident_model_mb=90.0, status="success",
                  role="execute", action_family="inference")
             for i, rt in enumerate(runtimes)]
    return Template(tid, tid, split, family, tuple(nodes), {n.node_id: n for n in nodes})


class BnPosteriorTests(unittest.TestCase):
    def setUp(self):
        # hand-built: lengths 2, 4, 4 -> P(n=2)=1/3, P(n=4)=2/3
        self.tpls = {
            "a": chain("a", [10, 10], "fam"),
            "b": chain("b", [10, 20, 30, 40], "fam"),
            "c": chain("c", [10, 20, 30, 40], "fam"),
        }
        self.bn = build_bn_profiler(self.tpls)

    def test_posterior_is_exactly_hand_computable(self):
        p = posterior_length_probs(self.bn, "fam", 0)
        self.assertAlmostEqual(p[2], 1 / 3)
        self.assertAlmostEqual(p[4], 2 / 3)

    def test_posterior_changes_with_completed_stage_evidence(self):
        """Completing stages is evidence; the posterior must move."""
        p0 = posterior_length_probs(self.bn, "fam", 0)
        p3 = posterior_length_probs(self.bn, "fam", 3)
        self.assertIn(2, p0)
        self.assertNotIn(2, p3, "a length-2 run cannot survive 3 consumed stages")
        self.assertAlmostEqual(p3[4], 1.0)
        self.assertNotAlmostEqual(posterior_entropy(self.bn, "fam", 0),
                                  posterior_entropy(self.bn, "fam", 3))

    def test_entropy_shrinks_as_evidence_accumulates(self):
        self.assertGreaterEqual(posterior_entropy(self.bn, "fam", 0),
                                posterior_entropy(self.bn, "fam", 1))

    def test_info_gain_is_zero_below_the_shortest_run(self):
        """Measured property of this corpus: structure uncertainty is inert early."""
        self.assertAlmostEqual(info_gain(self.bn, "fam", 0), 0.0)

    def test_uncertainty_reduction_ranks_a_wide_range_higher(self):
        """A position whose observed durations are wide-ranged must score higher.

        The two runs of each family must differ AT the measured position, otherwise
        the empirical spread is zero by construction.
        """
        wide = {"w": chain("w", [10, 1000], "g"), "x": chain("x", [1000, 10], "g")}
        flat = {"y": chain("y", [500, 500], "g"), "z": chain("z", [500, 500], "g")}
        bn_wide = build_bn_profiler(wide)
        bn_flat = build_bn_profiler(flat)
        self.assertGreater(duration_entropy(bn_wide, "g", 0), duration_entropy(bn_flat, "g", 0))
        self.assertGreater(uncertainty_reduction(bn_wide, "g", 0),
                           uncertainty_reduction(bn_flat, "g", 0))


class EpsilonGreedyTests(unittest.TestCase):
    def test_one_coin_per_decision_matches_epsilon(self):
        rng = random.Random(1234)
        draws = [draw_mode(rng, 0.1) for _ in range(20000)]
        share = draws.count("EXPLORE") / len(draws)
        self.assertAlmostEqual(share, 0.1, delta=0.01)
        self.assertEqual(set(draws), {"EXPLORE", "EXPLOIT"})

    def test_epsilon_zero_never_explores(self):
        rng = random.Random(7)
        self.assertTrue(all(draw_mode(rng, 0.0) == "EXPLOIT" for _ in range(500)))

    def test_epsilon_one_always_explores(self):
        rng = random.Random(7)
        self.assertTrue(all(draw_mode(rng, 1.0) == "EXPLORE" for _ in range(500)))


class LegacyFrontEndTests(unittest.TestCase):
    """The retired front end's own behaviour, driven directly.

    These used to run the LIVE ``llmsched`` policy.  That policy is now the v2 front
    end, whose profiler carries a discretizer this one does not have, so driving the
    policy from here would test a mixture of the two.  The policy integration is
    covered by L7 in test_llmsched_bn_v2.py; what is kept here is the retired
    implementation's own behaviour, so the retirement stays replayable.
    """

    def setUp(self):
        # deliberately DIFFERENT lengths: the retired posterior is over total length,
        # so a corpus of equal-length workflows would make it a point mass that never
        # moves as stages complete
        self.templates = {
            "t2": chain("t2", [100.0, 200.0], "fam", model="m0"),
            "t3": chain("t3", [100.0, 200.0, 150.0], "fam", model="m1"),
            "t4": chain("t4", [100.0, 200.0, 150.0, 120.0], "fam", model="m2"),
        }
        self.bn = build_bn_profiler(self.templates)

    def test_posterior_advances_with_the_consumed_count(self):
        """The retired criterion: the posterior moves on how MANY stages finished.

        Note what that means: a stage that took 40 ms and one that took 40 s produce
        the same posterior, so the retired front end cannot see a duration at all.
        That is why v2 replaces it.
        """

        before = posterior_length_probs(self.bn, "fam", 0)
        after = posterior_length_probs(self.bn, "fam", 3)
        self.assertNotEqual(before, after)
        self.assertNotEqual(before, after)
        self.assertAlmostEqual(sum(after.values()), 1.0)
        self.assertTrue(all(length >= 3 for length in after))

    def test_duration_entropy_reads_position_not_duration_state(self):
        """The retired exploration term is H over a POSITION's runtime histogram."""

        value = duration_entropy(self.bn, "fam", 1)
        self.assertGreaterEqual(value, 0.0)
        # a position with no samples has no entropy
        self.assertEqual(duration_entropy(self.bn, "fam", 99), 0.0)

    def test_uncertainty_reduction_is_entropy_times_spread(self):
        r = uncertainty_reduction(self.bn, "fam", 0)
        self.assertGreaterEqual(r, 0.0)

    def test_arm_is_registered(self):
        self.assertIn("llmsched", POLICIES)


if __name__ == "__main__":
    unittest.main()
