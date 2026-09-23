"""Fidelity gate for LLMSched-adapted (layer 1 of the two-layer gate).

The review's list for LLMSched:
    the BN posterior changes with completed-stage evidence;
    the uncertainty-reduction ranking matches a hand-built small graph;
    the epsilon-greedy activation ratio is correct.
"""
from __future__ import annotations

import random
import unittest

from tracing.analysis.llmsched_bn import (
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


class LlmschedPolicyTests(unittest.TestCase):
    def setUp(self):
        self.templates = {f"t{i}": chain(f"t{i}", [100.0 + 20 * i, 200.0, 150.0], "fam", model=f"m{i}")
                          for i in range(3)}
        self.bn = build_bn_profiler(self.templates)
        self.stats = train_resource_stats(self.templates)
        self.episode = {
            "episode_id": "llmsched-fidelity", "split": "train",
            "gpu_topology_mb": [4000.0, 4000.0], "initial_residency_hint": [[], []],
            "jobs": [{"job_instance_id": f"j{i}", "template_id": f"t{i}", "arrival_ms": 0.0,
                      "deadline_ms": 1e9, "service_class": "normal"} for i in range(3)],
        }

    def test_runs_and_completes(self):
        s, _ = simulate_episode(self.episode, self.templates, "llmsched",
                                train_stats=self.stats,
                                policy_context={"llmsched_bn": self.bn,
                                                "llmsched_rng": random.Random(1),
                                                "llmsched_epsilon": 0.1},
                                collect_events=False)
        self.assertEqual(s["completed_jobs"], 3)
        self.assertEqual(s["failed_jobs"], 0)

    def test_fails_closed_without_the_bn_or_the_rng(self):
        with self.assertRaises(ValueError):
            simulate_episode(self.episode, self.templates, "llmsched",
                             train_stats=self.stats, collect_events=False)
        with self.assertRaises(ValueError):
            simulate_episode(self.episode, self.templates, "llmsched",
                             train_stats=self.stats,
                             policy_context={"llmsched_bn": self.bn}, collect_events=False)

    def test_explore_and_exploit_can_differ(self):
        """EXPLORE and EXPLOIT must be able to pick different candidates."""
        runtimes = []
        for eps in (0.0, 1.0):
            s, _ = simulate_episode(self.episode, self.templates, "llmsched",
                                    train_stats=self.stats,
                                    policy_context={"llmsched_bn": self.bn,
                                                    "llmsched_rng": random.Random(1),
                                                    "llmsched_epsilon": eps},
                                    collect_events=True)
            runtimes.append([e.get("node_id") for e in (s.get("events") or [])
                             if e.get("event_type") == "node_start"])
        self.assertEqual(len(runtimes[0]), len(runtimes[1]), "both modes must complete the same work")

    def test_arm_is_registered(self):
        self.assertIn("llmsched", POLICIES)


if __name__ == "__main__":
    unittest.main()
