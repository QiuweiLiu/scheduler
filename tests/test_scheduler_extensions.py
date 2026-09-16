from __future__ import annotations

import unittest

from tracing.analysis.workload_v02_simulator import Node, Template, simulate_episode


def make_template(template_id: str, runtime_ms: float = 10.0) -> Template:
    node = Node(
        node_id=f"{template_id}:yolo",
        sequence_index=0,
        predecessors=(),
        successors=(),
        lane="gpu",
        model_id="yolo11x.pt",
        runtime_ms=runtime_ms,
        load_ms=1.0,
        workspace_peak_mb=1200.0,
        resident_model_mb=500.0,
        status="success",
        role="videotool_spatial",
        action_family="object_detection",
        raw_action="yolo-tracker",
    )
    return Template(template_id, template_id, "train", "test", (node,), {node.node_id: node})


class SchedulerExtensionTests(unittest.TestCase):
    def test_batch_action_expands_candidate_width_and_records_choice(self) -> None:
        templates = {f"t{index}": make_template(f"t{index}") for index in range(6)}
        episode = {
            "episode_id": "batch-episode",
            "split": "train",
            "gpu_topology_mb": [4000.0],
            "jobs": [
                {"job_instance_id": "j0", "template_id": "t0", "arrival_ms": 0.0, "service_class": "normal"},
                {"job_instance_id": "j1", "template_id": "t1", "arrival_ms": 0.0, "service_class": "normal"},
            ],
        }
        extension = {
            "batch_enabled": True,
            "batch_options": [1, 8],
            "batch_profiles": {
                "yolo11x.pt": {
                    "1": {"runtime_ms": 10.0, "load_ms": 1.0, "peak_memory_mb": 1200.0},
                    "8": {"runtime_ms": 20.0, "load_ms": 1.0, "peak_memory_mb": 2200.0},
                }
            },
        }
        summary, events = simulate_episode(episode, templates, "batch_myopic", extension_config=extension)
        dispatch = next(event for event in events if event["event_type"] == "node_dispatch")
        self.assertEqual(dispatch["candidate_count"], 4)
        self.assertEqual(dispatch["batch_size"], 1)
        self.assertEqual(summary["batch_action_count"], 2)
        self.assertEqual(summary["batch_choice_counts"], {"1": 2})

    def test_prefetch_is_paid_and_can_be_used_without_second_model_load(self) -> None:
        templates = {f"t{index}": make_template(f"t{index}") for index in range(3)}
        episode = {
            "episode_id": "prefetch-episode",
            "split": "train",
            "gpu_topology_mb": [4000.0],
            "jobs": [{"job_instance_id": "j0", "template_id": "t0", "arrival_ms": 0.0, "service_class": "normal"}],
        }
        summary, events = simulate_episode(
            episode,
            templates,
            "myopic",
            extension_config={"prefetch_plan": [{"gpu_index": 0, "model_id": "yolo11x.pt"}]},
        )
        self.assertEqual(summary["prefetch_count"], 1)
        self.assertAlmostEqual(summary["prefetch_load_ms"], 1.0)
        self.assertEqual(summary["wasted_prefetches"], 0)
        self.assertEqual([event["event_type"] for event in events if event["event_type"].startswith("prefetch_")], ["prefetch_start", "prefetch_end"])
        node_start = next(event for event in events if event["event_type"] == "node_start")
        self.assertEqual(node_start["load_ms"], 0.0)

    def test_preemption_recomputes_an_active_node(self) -> None:
        templates = {f"long{index}": make_template(f"long{index}", runtime_ms=100.0) for index in range(3)}
        templates.update({f"short{index}": make_template(f"short{index}", runtime_ms=5.0) for index in range(3)})
        episode = {
            "episode_id": "preempt-episode",
            "split": "train",
            "gpu_topology_mb": [4000.0],
            "jobs": [
                {"job_instance_id": "normal", "template_id": "long0", "arrival_ms": 0.0, "service_class": "normal"},
                {"job_instance_id": "priority", "template_id": "short0", "arrival_ms": 1.0, "service_class": "priority"},
            ],
        }
        summary, events = simulate_episode(
            episode,
            templates,
            "myopic_preempt",
            extension_config={
                "preemption_enabled": True,
                "preempt_min_queue_age_ms": 10000.0,
                "preempt_deadline_slack_ms": 0.0,
            },
        )
        self.assertEqual(summary["failed_jobs"], 0)
        self.assertGreaterEqual(summary["preemptions"], 1)
        self.assertTrue(any(event["event_type"] == "node_preempt" and event["recompute"] for event in events))


if __name__ == "__main__":
    unittest.main()
