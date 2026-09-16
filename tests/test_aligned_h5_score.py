from __future__ import annotations

import unittest

from tracing.analysis import workload_v02_simulator as simulator


def make_fixture() -> tuple[simulator.Template, simulator.Job, simulator.GPU, dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    first = simulator.Node(
        node_id="n0",
        sequence_index=0,
        predecessors=(),
        successors=("n1",),
        lane="gpu",
        model_id="model-a",
        runtime_ms=100.0,
        load_ms=30.0,
        workspace_peak_mb=1200.0,
        resident_model_mb=1000.0,
        status="success",
        role="plan",
        action_family="planner",
    )
    second = simulator.Node(
        node_id="n1",
        sequence_index=1,
        predecessors=("n0",),
        successors=(),
        lane="gpu",
        model_id="model-a",
        runtime_ms=200.0,
        load_ms=50.0,
        workspace_peak_mb=1200.0,
        resident_model_mb=1000.0,
        status="success",
        role="answer",
        action_family="answer",
    )
    template = simulator.Template(
        "template-a",
        "video-a",
        "train",
        "test",
        (first, second),
        {"n0": first, "n1": second},
    )
    job = simulator.Job("job-a", template, 0.0, None, "normal")
    gpu = simulator.GPU(0, 24000.0)
    stats = {
        "model-a|gpu|*|model_lane": {
            "runtime_p50_ms": 120.0,
            "runtime_p90_ms": 120.0,
            "load_p50_ms": 30.0,
            "memory_p95_mb": 1200.0,
            "count": 3,
        }
    }
    artifacts = {
        "n0": {
            "future_h5": [
                {
                    "scenario_probability": 1.0,
                    "steps": [
                        {
                            "model_id": "model-a",
                            "execution_lane": "gpu",
                            "step_offset": 1,
                        }
                    ],
                }
            ]
        }
    }
    return template, job, gpu, stats, artifacts


class AlignedH5ScoreTests(unittest.TestCase):
    def test_predicted_and_truth_share_contract_but_not_values(self) -> None:
        template, job, gpu, stats, artifacts = make_fixture()
        memories = simulator.model_memory_by_model((template,))

        predicted = simulator.aligned_h5_score(
            "predicted", job, "n0", gpu, stats, artifacts, 5, memories
        )
        truth = simulator.aligned_h5_score(
            "truth", job, "n0", gpu, stats, None, 5, memories
        )

        self.assertEqual(predicted, {"current_ms": 120.0, "future_ms": 90.0, "total_ms": 210.0})
        self.assertEqual(truth, {"current_ms": 100.0, "future_ms": 150.0, "total_ms": 250.0})
        self.assertNotEqual(predicted["total_ms"], truth["total_ms"])

    def test_truth_aligned_contract_includes_current_action(self) -> None:
        template, job, gpu, stats, _artifacts = make_fixture()
        aligned = simulator.aligned_h5_score("truth", job, "n0", gpu, stats, None, 5)
        legacy = simulator.limited_future_truth_cost(job, "n0", gpu, stats, 5)

        self.assertEqual(aligned["total_ms"], 250.0)
        self.assertEqual(legacy, 200.0)
        self.assertGreater(aligned["total_ms"], legacy)

    def test_predicted_score_requires_future_artifacts(self) -> None:
        _template, job, gpu, stats, _artifacts = make_fixture()
        with self.assertRaises(ValueError):
            simulator.aligned_h5_score("predicted", job, "n0", gpu, stats)

    def test_layer_score_counts_multiple_nodes_in_one_layer(self) -> None:
        template, job, gpu, stats, _artifacts = make_fixture()
        memories = simulator.model_memory_by_model((template,))
        layer_artifacts = {
            "n0": {
                "future_h5_layers": [
                    {
                        "scenario_id": "s00",
                        "scenario_probability": 1.0,
                        "layers": [
                            {
                                "layer_offset": 1,
                                "nodes": [{"model_id": "model-a", "execution_lane": "gpu"}],
                            },
                            {
                                "layer_offset": 2,
                                "nodes": [
                                    {"model_id": "model-a", "execution_lane": "gpu"},
                                    {"model_id": "model-a", "execution_lane": "gpu"},
                                ],
                            },
                        ],
                    }
                ]
            }
        }

        score = simulator.aligned_h5_score(
            "predicted_layer", job, "n0", gpu, stats, layer_artifacts, 5, memories
        )

        self.assertEqual(score, {"current_ms": 120.0, "future_ms": 270.0, "total_ms": 390.0})

    def test_layer_score_rejects_missing_layer_sidecar(self) -> None:
        template, job, gpu, stats, _artifacts = make_fixture()
        with self.assertRaises(ValueError):
            simulator.aligned_h5_score(
                "predicted_layer", job, "n0", gpu, stats, {"n0": {"future_h5": []}}, 5
            )

    def test_aligned_policies_are_opt_in(self) -> None:
        self.assertIn("aligned_predopt_h5", simulator.ALIGNED_H5_POLICIES)
        self.assertIn("aligned_predopt_h5_layer", simulator.ALIGNED_H5_POLICIES)
        self.assertIn("aligned_trueopt_h5", simulator.ALIGNED_H5_POLICIES)
        self.assertNotIn("aligned_predopt_h5", simulator.POLICIES)
        self.assertNotIn("aligned_predopt_h5_layer", simulator.POLICIES)
        self.assertNotIn("aligned_trueopt_h5", simulator.POLICIES)


if __name__ == "__main__":
    unittest.main()
