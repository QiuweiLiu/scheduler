from __future__ import annotations

import unittest

from tracing.analysis.workload_v02_simulator import Node, Template, simulate_episode


class WorkloadStateAdapterTests(unittest.TestCase):
    def test_unique_event_loop_emits_scheduler_state_without_truth(self) -> None:
        first = Node(
            node_id="node-0",
            sequence_index=0,
            predecessors=(),
            successors=("node-1",),
            lane="gpu",
            model_id="model-a",
            runtime_ms=10.0,
            load_ms=0.0,
            workspace_peak_mb=1100.0,
            resident_model_mb=1000.0,
            status="success",
            role="plan",
            action_family="planner",
        )
        second = Node(
            node_id="node-1",
            sequence_index=1,
            predecessors=("node-0",),
            successors=(),
            lane="cpu",
            model_id="cpu",
            runtime_ms=2.0,
            load_ms=0.0,
            workspace_peak_mb=None,
            resident_model_mb=0.0,
            status="success",
            role="aggregate",
            action_family="answer",
        )
        template = Template(
            template_id="template-0",
            video_id="video-0",
            split="train",
            baseline="test",
            nodes=(first, second),
            by_id={first.node_id: first, second.node_id: second},
        )
        episode = {
            "episode_id": "episode-0",
            "split": "train",
            "gpu_topology_mb": [24000.0],
            "jobs": [
                {
                    "job_instance_id": "job-0",
                    "template_id": "template-0",
                    "arrival_ms": 0.0,
                    "service_class": "normal",
                }
            ],
        }
        templates = {
            f"template-{index}": Template(
                template_id=f"template-{index}",
                video_id=f"video-{index}",
                split="train",
                baseline="test",
                nodes=template.nodes,
                by_id=template.by_id,
            )
            for index in range(3)
        }
        summary, events = simulate_episode(episode, templates, "myopic")
        self.assertEqual(summary["completed_jobs"], 1)
        dispatch = next(event for event in events if event["event_type"] == "node_dispatch")
        state = dispatch["scheduler_state"]
        self.assertEqual(state["schema_version"], "scheduler-state-view-v1")
        self.assertEqual(state["future"]["job-0:node-0"]["scenarios"], [])
        self.assertNotIn("runtime_ms", str(state))
        self.assertNotIn("workspace_peak_mb", str(state))


if __name__ == "__main__":
    unittest.main()
