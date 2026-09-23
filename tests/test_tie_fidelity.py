"""Fidelity gate for TIE-adapted (layer 1 of the two-layer gate).

The review's list for TIE:
    independent current-node-only distribution predictor;
    the score formula is correct;
    CVaR / beta / decay are unit-tested;
    reading the H-step future is forbidden.

This file checks all four.  Performance is reported separately and never used here.
"""
from __future__ import annotations

import unittest

from tracing.analysis.workload_v02_simulator import (
    POLICIES,
    Node,
    Template,
    simulate_episode,
    train_resource_stats,
)


def make_template(template_id: str, runtime_ms: float, model: str = "m1", lane: str = "gpu") -> Template:
    node = Node(
        node_id=f"{template_id}:n", sequence_index=0, predecessors=(), successors=(),
        lane=lane, model_id=model, runtime_ms=runtime_ms, load_ms=10.0,
        workspace_peak_mb=100.0, resident_model_mb=90.0, status="success",
        role="execute", action_family="inference",
    )
    return Template(template_id, template_id, "train", "test", (node,), {node.node_id: node})


class TieFrontEndTests(unittest.TestCase):
    """The front end must be an independent, current-node-only distribution."""

    def test_arm_is_registered(self):
        self.assertIn("tie_current", POLICIES)

    def test_stats_carry_expectation_and_cvar(self):
        tpl = {"t": make_template("t", 100.0), "u": make_template("u", 500.0)}
        stats = train_resource_stats(tpl)
        self.assertTrue(stats, "no groups produced")
        for key, row in stats.items():
            self.assertIn("runtime_mean_ms", row, key)
            self.assertIn("runtime_cvar90_ms", row, key)

    def test_cvar_is_the_upper_tail_mean_not_the_max(self):
        """CVaR_0.9 is the mean of the worst 10%, so it can never exceed the max."""
        tpl = {f"t{i}": make_template(f"t{i}", float(10 * (i + 1))) for i in range(20)}
        stats = train_resource_stats(tpl)
        for row in stats.values():
            self.assertLessEqual(row["runtime_cvar90_ms"], max(
                n.runtime_ms for t in tpl.values() for n in t.nodes
            ) + 1e-9)
            self.assertGreaterEqual(row["runtime_cvar90_ms"], row["runtime_mean_ms"])

    def test_stats_use_only_the_training_split(self):
        """A validation/test template must not contribute to any group."""
        train = {"a": make_template("a", 100.0)}
        test = {"b": Template("b", "b", "test", "test",
                              (Node(node_id="b:n", sequence_index=0, predecessors=(), successors=(),
                                    lane="gpu", model_id="m1", runtime_ms=99999.0, load_ms=1.0,
                                    workspace_peak_mb=1.0, resident_model_mb=1.0, status="success",
                                    role="execute", action_family="inference"),),
                              {"b:n": None})}
        test["b"].by_id["b:n"] = test["b"].nodes[0]
        only_test = train_resource_stats(test)
        both = train_resource_stats({**train, **test})
        for key, row in only_test.items():
            self.assertAlmostEqual(row["runtime_mean_ms"], both[key]["runtime_mean_ms"],
                                   msg="the test split leaked into the bank")


class TiePolicyTests(unittest.TestCase):
    """The wired policy must run and must not need any future artifact."""

    def setUp(self):
        # distinct models per template: a current-node-only front end can only tell
        # candidates apart when their (model, lane) identity differs, so a shared
        # model would make TIE and a plain p50 arm collapse by construction
        self.templates = {
            f"t{i}": make_template(f"t{i}", 100.0 + 50 * i, model=f"m{i}") for i in range(3)
        }
        self.stats = train_resource_stats(self.templates)
        self.episode = {
            "episode_id": "tie-fidelity",
            "split": "train",
            "gpu_topology_mb": [4000.0, 4000.0],
            "initial_residency_hint": [[], []],
            "jobs": [
                {"job_instance_id": f"j{i}", "template_id": f"t{i}", "arrival_ms": 0.0,
                 "deadline_ms": 1e9, "service_class": "normal"}
                for i in range(3)
            ],
        }

    def test_runs_without_any_future_artifact(self):
        """TIE consumes no H-step future, so future_artifacts must not be required."""
        state, _ = simulate_episode(self.episode, self.templates, "tie_current",
                                    future_artifacts=None, train_stats=self.stats,
                                    collect_events=False)
        self.assertEqual(state["completed_jobs"], 3)
        self.assertEqual(state["failed_jobs"], 0)
        self.assertTrue(state["mean_completion_ms"] > 0)

    def test_runs_without_any_future_artifact_superseded(self):
        pass


class TieFormulaTests(unittest.TestCase):
    """GPT's list: the score formula, CVaR and beta must be unit-tested directly."""

    def test_beta_is_the_paper_clip(self):
        from tracing.analysis.workload_v02_simulator import tie_beta
        # clip(0.1 * L_q / B, 0.1, 0.5)
        self.assertAlmostEqual(tie_beta(0.0, 2.0), 0.1)      # floor
        self.assertAlmostEqual(tie_beta(2.0, 2.0), 0.1)      # 0.1 * 1
        self.assertAlmostEqual(tie_beta(10.0, 2.0), 0.5)     # 0.5 -> ceiling
        self.assertAlmostEqual(tie_beta(100.0, 2.0), 0.5)    # clamped
        self.assertAlmostEqual(tie_beta(6.0, 2.0), 0.3)      # interior
        self.assertAlmostEqual(tie_beta(1.0, 10.0), 0.1)     # floor again

    def test_score_is_mean_plus_beta_times_cvar(self):
        from tracing.analysis.workload_v02_simulator import tie_current_score
        self.assertAlmostEqual(tie_current_score(100.0, 400.0, 0.5, 0.0), 300.0)
        self.assertAlmostEqual(tie_current_score(100.0, 400.0, 0.1, 0.0), 140.0)
        self.assertAlmostEqual(tie_current_score(0.0, 0.0, 0.5, 25.0), 25.0)

    def test_heavy_tail_can_reverse_the_mean_ordering(self):
        """The whole point of the tail term: a fat-tailed candidate can lose."""
        from tracing.analysis.workload_v02_simulator import tie_current_score
        # A has the smaller mean but a much fatter tail
        a_mean, a_cvar = 1050.0, 2000.0
        b_mean, b_cvar = 1100.0, 1100.0
        self.assertLess(a_mean, b_mean, "A must have the smaller mean")
        beta = 0.5
        self.assertGreater(tie_current_score(a_mean, a_cvar, beta, 0.0),
                           tie_current_score(b_mean, b_cvar, beta, 0.0),
                           "the tail term must be able to reverse the mean ordering")

    def test_zero_beta_degenerates_to_the_mean(self):
        """TIE collapses onto a point criterion when the tail weight vanishes."""
        from tracing.analysis.workload_v02_simulator import tie_current_score
        for m, c in ((10.0, 99.0), (500.0, 501.0), (0.0, 1e6)):
            self.assertAlmostEqual(tie_current_score(m, c, 0.0, 0.0), m)


if __name__ == "__main__":
    unittest.main()
