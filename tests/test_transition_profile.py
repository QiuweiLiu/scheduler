from __future__ import annotations

import unittest

from tracing.analysis.workload_v02_simulator import Node, Template, simulate_episode


def make_template(template_id: str, model_id: str, runtime_ms: float = 10.0) -> Template:
    node = Node(
        node_id=f"{template_id}:gpu",
        sequence_index=0,
        predecessors=(),
        successors=(),
        lane="gpu",
        model_id=model_id,
        runtime_ms=runtime_ms,
        load_ms=1.0,
        workspace_peak_mb=1200.0,
        resident_model_mb=500.0,
        status="success",
        role="videotool_spatial",
        action_family="object_detection",
        raw_action="yolo-tracker" if model_id.startswith("yolo") else "model-call",
    )
    return Template(template_id, template_id, "train", "test", (node,), {node.node_id: node})


def profile_config(*, gpu_capacity_mb: float | None = None) -> dict[str, object]:
    profile: dict[str, object] = {
        "enabled": True,
        "strict": True,
        "models": {
            "model-a": {"cold_load_ms": 20.0, "evict_proxy_ms": 7.0, "checkpoint_supported": False},
            "model-b": {"cold_load_ms": 15.0, "evict_proxy_ms": 5.0, "checkpoint_supported": False},
            "yolo11x.pt": {"cold_load_ms": 9.0, "evict_proxy_ms": 4.0, "checkpoint_supported": False},
        },
    }
    if gpu_capacity_mb is not None:
        profile["gpu_capacity_mb"] = gpu_capacity_mb
    return {"transition_profile": profile}


class TransitionProfileTests(unittest.TestCase):
    def test_profile_replaces_cold_load_and_charges_eviction(self) -> None:
        templates = {
            **{f"a{index}": make_template(f"a{index}", "model-a") for index in range(3)},
            **{f"b{index}": make_template(f"b{index}", "model-b") for index in range(3)},
        }
        episode = {
            "episode_id": "transition-episode",
            "split": "train",
            "gpu_topology_mb": [1300.0],
            "jobs": [
                {"job_instance_id": "a", "template_id": "a0", "arrival_ms": 0.0, "service_class": "normal"},
                {"job_instance_id": "b", "template_id": "b0", "arrival_ms": 100.0, "service_class": "normal"},
            ],
        }
        summary, events = simulate_episode(
            episode,
            templates,
            "myopic",
            extension_config=profile_config(),
        )
        self.assertEqual(summary["failed_jobs"], 0)
        self.assertTrue(summary["transition_profile_enabled"])
        self.assertAlmostEqual(summary["transition_profile_load_ms"], 35.0)
        self.assertAlmostEqual(summary["transition_profile_evict_ms"], 7.0)
        evict = next(event for event in events if event["event_type"] == "model_evict")
        self.assertAlmostEqual(evict["evict_cost_ms"], 7.0)
        starts = [event for event in events if event["event_type"] == "node_start"]
        self.assertEqual([start["load_source"] for start in starts], ["transition_profile", "transition_profile"])
        self.assertAlmostEqual(starts[1]["eviction_ms"], 7.0)

    def test_profile_prefetch_is_not_loaded_again_on_first_use(self) -> None:
        templates = {f"t{index}": make_template(f"t{index}", "yolo11x.pt") for index in range(3)}
        episode = {
            "episode_id": "transition-prefetch",
            "split": "train",
            "gpu_topology_mb": [4000.0],
            "jobs": [{"job_instance_id": "j0", "template_id": "t0", "arrival_ms": 0.0, "service_class": "normal"}],
        }
        summary, events = simulate_episode(
            episode,
            templates,
            "myopic",
            extension_config={
                **profile_config(),
                "prefetch_plan": [{"gpu_index": 0, "model_id": "yolo11x.pt"}],
            },
        )
        self.assertAlmostEqual(summary["transition_profile_load_ms"], 9.0)
        self.assertEqual(summary["transition_profile_load_hits"], 1)
        self.assertEqual(len([event for event in events if event["event_type"] == "model_load_start"]), 0)
        node_start = next(event for event in events if event["event_type"] == "node_start")
        self.assertEqual(node_start["load_ms"], 0.0)

    def test_profile_rejects_unmatched_gpu_capacity(self) -> None:
        templates = {f"t{index}": make_template(f"t{index}", "model-a") for index in range(3)}
        episode = {
            "episode_id": "transition-capacity-mismatch",
            "split": "train",
            "gpu_topology_mb": [4000.0],
            "jobs": [{"job_instance_id": "j0", "template_id": "t0", "arrival_ms": 0.0, "service_class": "normal"}],
        }
        with self.assertRaisesRegex(ValueError, "GPU capacity mismatch"):
            simulate_episode(
                episode,
                templates,
                "myopic",
                extension_config=profile_config(gpu_capacity_mb=32760.0),
            )


if __name__ == "__main__":
    unittest.main()
