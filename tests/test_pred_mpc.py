from __future__ import annotations

import unittest

from tracing.analysis.workload_v02_simulator import Node, Template, simulate_episode, train_resource_stats


def make_templates() -> dict[str, Template]:
    nodes: list[Node] = []
    for prefix, tail_runtime in (("a", 20.0), ("b", 80.0)):
        root = Node(
            node_id=f"{prefix}0",
            sequence_index=0,
            predecessors=(),
            successors=(f"{prefix}1",),
            lane="gpu",
            model_id=f"model-{prefix}",
            runtime_ms=10.0,
            load_ms=0.0,
            workspace_peak_mb=2100.0,
            resident_model_mb=2000.0,
            status="success",
            role="plan",
            action_family="planner",
        )
        tail = Node(
            node_id=f"{prefix}1",
            sequence_index=1,
            predecessors=(f"{prefix}0",),
            successors=(),
            lane="gpu",
            model_id=f"model-{prefix}",
            runtime_ms=tail_runtime,
            load_ms=0.0,
            workspace_peak_mb=2100.0,
            resident_model_mb=2000.0,
            status="success",
            role="answer",
            action_family="answer",
        )
        nodes.extend((root, tail))
    return {
        f"template-{prefix}": Template(
            template_id=f"template-{prefix}",
            video_id=f"video-{prefix}",
            split="train",
            baseline="test",
            nodes=(nodes[index], nodes[index + 1]),
            by_id={nodes[index].node_id: nodes[index], nodes[index + 1].node_id: nodes[index + 1]},
        )
        for index, prefix in ((0, "a"), (2, "b"))
    }


class PredMpcTests(unittest.TestCase):
    def test_pred_mpc_completes_with_predicted_rollout_only(self) -> None:
        templates = make_templates()
        episode = {
            "episode_id": "pred-mpc-smoke-0",
            "split": "train",
            "gpu_topology_mb": [24000.0, 24000.0],
            "initial_residency_hint": [[], []],
            "jobs": [
                {"job_instance_id": "job-a", "template_id": "template-a", "arrival_ms": 0.0, "service_class": "normal"},
                {"job_instance_id": "job-b", "template_id": "template-b", "arrival_ms": 0.0, "service_class": "normal"},
            ],
        }
        stats = train_resource_stats(templates)
        artifacts = {node_id: {"future_h3": [], "future_h5": []} for node_id in ("a0", "a1", "b0", "b1")}
        summary, _events = simulate_episode(
            episode,
            templates,
            "pred_mpc_h3",
            future_artifacts=artifacts,
            train_stats=stats,
            collect_events=False,
        )
        self.assertEqual(summary["failed_jobs"], 0)
        self.assertEqual(summary["completed_jobs"], 2)
        self.assertGreater(summary["pred_mpc_rollout_calls"], 0)
        self.assertGreaterEqual(summary["pred_mpc_mean_predicted_events"], 1.0)


if __name__ == "__main__":
    unittest.main()
