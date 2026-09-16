from __future__ import annotations

import unittest

from tracing.analysis.workload_v02_simulator import Node, Template, simulate_episode


def make_template(template_id: str) -> Template:
    node = Node(
        node_id=f"{template_id}:gpu",
        sequence_index=0,
        predecessors=(),
        successors=(),
        lane="gpu",
        model_id="model-a",
        runtime_ms=10.0,
        load_ms=5.0,
        workspace_peak_mb=2100.0,
        resident_model_mb=2000.0,
        status="success",
        role="execute",
        action_family="inference",
    )
    return Template(template_id, template_id, "train", "test", (node,), {node.node_id: node})


class RiskAwareScoreTests(unittest.TestCase):
    def test_risk_aware_is_registered_and_records_normalized_features(self) -> None:
        templates = {f"t{index}": make_template(f"t{index}") for index in range(3)}
        episode = {
            "episode_id": "risk-aware-score-smoke",
            "split": "train",
            "gpu_topology_mb": [4000.0],
            "initial_residency_hint": [["model-a"]],
            "jobs": [
                {
                    "job_instance_id": "priority",
                    "template_id": "t0",
                    "arrival_ms": 0.0,
                    "deadline_ms": 20.0,
                    "service_class": "priority",
                },
                {
                    "job_instance_id": "normal-a",
                    "template_id": "t1",
                    "arrival_ms": 0.0,
                    "deadline_ms": 1000.0,
                    "service_class": "normal",
                },
                {
                    "job_instance_id": "normal-b",
                    "template_id": "t2",
                    "arrival_ms": 0.0,
                    "deadline_ms": 1000.0,
                    "service_class": "normal",
                },
            ],
        }
        summary, _events = simulate_episode(episode, templates, "risk_aware", collect_events=False)

        self.assertEqual(summary["failed_jobs"], 0)
        self.assertEqual(summary["completed_jobs"], 3)
        self.assertGreater(summary["risk_aware_decision_count"], 0)
        self.assertEqual(summary["risk_aware_strict_feasible_rate"], 1.0)
        self.assertGreaterEqual(summary["risk_aware_mean_resource_risk"], 0.0)
        self.assertLessEqual(summary["risk_aware_mean_resource_risk"], 1.0)


if __name__ == "__main__":
    unittest.main()
