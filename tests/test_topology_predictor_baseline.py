from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from scripts.r8_topology_predictor_baseline import (
    condition_key,
    fit_empirical_models,
    future_layers,
    predict_scenarios,
    split_templates,
    topology_signature,
)
from tracing.scheduling.future_topology import validate_layer_scenarios


def template(task_id: str, width: int) -> dict[str, object]:
    nodes: list[dict[str, object]] = [
        {
            "node_id": f"{task_id}:root",
            "sequence_index": 0,
            "predecessor_node_ids": [],
            "node_type": "planner",
            "raw_action": "planner.generate",
            "activity": "other",
            "model_id": "model-a",
            "execution_lane": "gpu",
        }
    ]
    for index in range(width):
        nodes.append(
            {
                "node_id": f"{task_id}:child:{index}",
                "sequence_index": index + 1,
                "predecessor_node_ids": [f"{task_id}:root"],
                "node_type": "videotool_temporal",
                "raw_action": "frame-selector",
                "activity": "select_frames",
                "model_id": "cpu-metadata-adapter-v1",
                "execution_lane": "cpu",
            }
        )
    return {
        "task_id": task_id,
        "template_id": task_id,
        "video_id": task_id.split("_")[-1],
        "model_stack_id": "stack-a",
        "baseline": "star",
        "nodes": nodes,
    }


class TopologyPredictorBaselineTests(unittest.TestCase):
    def test_split_rejects_unclassified_templates(self) -> None:
        with self.assertRaises(ValueError):
            split_templates([template("r7_s_other_x", 1)])

    def test_reconstructs_layers_from_predecessor_ids(self) -> None:
        row = template("r7_s_train_x", 2)
        layers = future_layers(row, "r7_s_train_x:root")
        self.assertEqual([[node["node_id"] for node in layer] for layer in layers], [[
            "r7_s_train_x:child:0",
            "r7_s_train_x:child:1",
        ]])

    def test_empirical_prediction_is_multinode_and_identity_free(self) -> None:
        fit_row = template("r7_s_train_fit", 2)
        predict_row = template("r7_s_val_predict", 1)
        fit, predict = split_templates([fit_row, predict_row])
        models = fit_empirical_models(fit)
        key = condition_key(predict[0], predict[0]["nodes"][0])
        scenarios = predict_scenarios(key, models, max_scenarios=3)
        validate_layer_scenarios(scenarios, 5)
        self.assertEqual(sum(float(item["scenario_probability"]) for item in scenarios), 1.0)
        self.assertEqual(len(scenarios[0]["layers"][0]["nodes"]), 2)
        for scenario in scenarios:
            for layer in scenario["layers"]:
                for node in layer["nodes"]:
                    self.assertNotIn("node_id", node)
                    self.assertNotIn("predecessor_node_ids", node)
                    self.assertNotIn("runtime_ms", node)

    def test_signature_is_deterministic(self) -> None:
        row = template("r7_s_train_x", 2)
        self.assertEqual(
            topology_signature(row, "r7_s_train_x:root"),
            topology_signature(row, "r7_s_train_x:root"),
        )

    def test_output_sidecar_can_be_read_as_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sidecar.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps({"future_h5_layers": []}) + "\n")
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                self.assertEqual(json.loads(handle.readline())["future_h5_layers"], [])


if __name__ == "__main__":
    unittest.main()
