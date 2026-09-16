from __future__ import annotations

import unittest

from tracing.scheduling.event_engine import (
    ExecutionTruth,
    ExecutionTruthProvider,
    FutureProvider,
    ResourceEstimate,
    build_scheduler_state,
)


class EventEngineContractTests(unittest.TestCase):
    def _inputs(self):
        node = {
            "job_instance_id": "job-0",
            "node_id": "node-1",
            "sequence_index": 1,
            "role": "execute",
            "action_family": "spatial",
            "model_id": "qwen3-vl-8b",
            "lane": "gpu",
            "ready_since_ms": 10.0,
        }
        resource = {
            ("job-0", "node-1"): ResourceEstimate(100.0, 150.0, 20.0, 600.0)
        }
        gpus = [{"index": 0, "capacity_mb": 24000.0, "busy": False, "resident_models": ()}]
        return node, resource, gpus

    def test_none_mode_does_not_open_future_artifact(self):
        node, resource, gpus = self._inputs()
        provider = FutureProvider("none", predicted={("job-0", "node-1", 3): [{"scenario_probability": 1.0}]})
        state = build_scheduler_state(
            episode_id="ep-0",
            decision_index=0,
            time_ms=10.0,
            ready_nodes=[node],
            completed_prefix={"job-0": ("node-0",)},
            gpus=gpus,
            resource_predictions=resource,
            future_provider=provider,
            horizon=3,
        )
        reveal = state.future[("job-0", "node-1")]
        self.assertEqual(reveal.mode, "none")
        self.assertEqual(reveal.scenarios, ())

    def test_predicted_future_has_no_execution_truth(self):
        node, resource, gpus = self._inputs()
        predicted = {
            ("job-0", "node-1", 1): [
                {
                    "scenario_probability": 0.8,
                    "steps": [{"role": "execute", "action_family": "temporal", "model_id": "m", "lane": "gpu"}],
                },
                {
                    "scenario_probability": 0.2,
                    "steps": [{"role": "aggregate", "action_family": "answer", "lane": "cpu"}],
                },
            ]
        }
        state = build_scheduler_state(
            episode_id="ep-0",
            decision_index=1,
            time_ms=10.0,
            ready_nodes=[node],
            completed_prefix={"job-0": ("node-0",)},
            gpus=gpus,
            resource_predictions=resource,
            future_provider=FutureProvider("pred_h", predicted=predicted),
            horizon=1,
        )
        payload = state.to_scheduler_dict()
        self.assertNotIn("runtime_ms", str(payload))
        self.assertNotIn("workspace_peak_mb", str(payload))
        self.assertEqual(payload["future"]["job-0:node-1"]["supported_probability_mass"], 1.0)

    def test_current_truth_is_not_state(self):
        node, resource, gpus = self._inputs()
        truth = ExecutionTruthProvider(
            {("job-0", "node-1"): ExecutionTruth(101.0, 21.0, 610.0, "success")}
        )
        state = build_scheduler_state(
            episode_id="ep-0",
            decision_index=2,
            time_ms=10.0,
            ready_nodes=[node],
            completed_prefix={"job-0": ()},
            gpus=gpus,
            resource_predictions=resource,
            future_provider=FutureProvider("none"),
            horizon=0,
        )
        self.assertEqual(truth.get("job-0", "node-1").runtime_ms, 101.0)
        self.assertNotIn("101.0", str(state.to_scheduler_dict()))

    def test_forbidden_future_truth_is_rejected(self):
        node, resource, gpus = self._inputs()
        provider = FutureProvider(
            "pred_h",
            predicted={
                ("job-0", "node-1", 1): [
                    {"scenario_probability": 1.0, "steps": [{"role": "execute", "runtime_ms": 3.0}]}
                ]
            },
        )
        with self.assertRaises(ValueError):
            build_scheduler_state(
                episode_id="ep-0",
                decision_index=0,
                time_ms=0.0,
                ready_nodes=[node],
                completed_prefix={},
                gpus=gpus,
                resource_predictions=resource,
                future_provider=provider,
                horizon=1,
            )


if __name__ == "__main__":
    unittest.main()
