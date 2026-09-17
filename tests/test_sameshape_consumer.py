from __future__ import annotations

import unittest

from tracing.analysis.workload_v02_simulator import (
    POLICIES,
    SAMESHAPE_POLICIES,
    Node,
    Template,
    sameshape_future_cost,
    simulate_episode,
    train_resource_stats,
)


def make_template(template_id: str, runtime_ms: float, load_ms: float) -> Template:
    node = Node(
        node_id=f"{template_id}:gpu",
        sequence_index=0,
        predecessors=(),
        successors=(),
        lane="gpu",
        model_id="model-a",
        runtime_ms=runtime_ms,
        load_ms=load_ms,
        workspace_peak_mb=2100.0,
        resident_model_mb=2000.0,
        status="success",
        role="execute",
        action_family="inference",
    )
    return Template(template_id, template_id, "train", "test", (node,), {node.node_id: node})


def make_episode() -> dict[str, object]:
    return {
        "episode_id": "sameshape-consumer-smoke",
        "split": "train",
        "gpu_topology_mb": [4000.0, 4000.0],
        "initial_residency_hint": [[], []],
        "jobs": [
            {
                "job_instance_id": "priority",
                "template_id": "t0",
                "arrival_ms": 0.0,
                "deadline_ms": 100000.0,
                "service_class": "priority",
            },
            {
                "job_instance_id": "normal-a",
                "template_id": "t1",
                "arrival_ms": 0.0,
                "deadline_ms": 100000.0,
                "service_class": "normal",
            },
            {
                "job_instance_id": "normal-b",
                "template_id": "t2",
                "arrival_ms": 0.0,
                "deadline_ms": 100000.0,
                "service_class": "normal",
            },
        ],
    }


def make_artifacts(node_ids: tuple[str, ...], p50: float, p90: float, p95: float, horizon: int = 5) -> dict:
    step = {
        "model_id": "model-a",
        "execution_lane": "gpu",
        "resource": {
            "runtime_ms_quantiles": {"p50": p50, "p90": p90, "p95": p95},
            "load_occurrence_probability": 0.0,
        },
    }
    return {
        node_id: {
            f"future_h{horizon}": [
                {"scenario_probability": 1.0, "steps": [dict(step) for _ in range(horizon)]}
            ]
        }
        for node_id in node_ids
    }


class SameShapeConsumerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.templates = {
            f"t{index}": make_template(f"t{index}", 10.0 + 5.0 * index, 5.0) for index in range(3)
        }
        self.episode = make_episode()
        self.train_stats = train_resource_stats(self.templates)
        self.artifacts = make_artifacts(("t0:gpu", "t1:gpu", "t2:gpu"), 100.0, 300.0, 400.0)

    def test_arms_are_registered_policies(self) -> None:
        for policy in SAMESHAPE_POLICIES:
            self.assertIn(policy, POLICIES)

    def test_p95_arm_is_exactly_the_q95_champion_consumer(self) -> None:
        """sameshape_h5_p95 differs from predopt_h5_q95 in name only."""

        champion, _champion_events = simulate_episode(
            self.episode,
            self.templates,
            "predopt_h5_q95",
            future_artifacts=self.artifacts,
            train_stats=self.train_stats,
            collect_events=False,
        )
        candidate, _candidate_events = simulate_episode(
            self.episode,
            self.templates,
            "sameshape_h5_p95",
            future_artifacts=self.artifacts,
            train_stats=self.train_stats,
            collect_events=False,
        )
        self.assertEqual(
            {key: value for key, value in champion.items() if key != "policy"},
            {key: value for key, value in candidate.items() if key != "policy"},
        )

    def test_future_statistic_is_monotone_and_truth_arm_runs(self) -> None:
        node_id = "t0:gpu"
        p50_cost = sameshape_future_cost(node_id, self.artifacts, self.train_stats, 5, "p50")
        p95_cost = sameshape_future_cost(node_id, self.artifacts, self.train_stats, 5, "p95")
        self.assertAlmostEqual(p50_cost, 500.0, places=6)
        self.assertAlmostEqual(p95_cost, 2000.0, places=6)
        self.assertLess(p50_cost, p95_cost)

        truth_summary, _events = simulate_episode(
            self.episode,
            self.templates,
            "sameshape_h5_truth",
            future_artifacts=self.artifacts,
            train_stats=self.train_stats,
            collect_events=False,
        )
        self.assertEqual(truth_summary["completed_jobs"], 3)
        self.assertEqual(truth_summary["failed_jobs"], 0)

    def test_arms_require_future_artifacts(self) -> None:
        with self.assertRaises(ValueError):
            simulate_episode(
                self.episode,
                self.templates,
                "sameshape_h5_p50",
                train_stats=self.train_stats,
                collect_events=False,
            )


if __name__ == "__main__":
    unittest.main()
