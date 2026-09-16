from __future__ import annotations

import unittest

from scripts.p9d_topology_tabular import (
    MAX_WIDTH,
    PROTOTYPE_FIELDS,
    TabularModel,
    _shape_targets,
    context_key,
    evaluate,
    topology_signature,
)


def make_feature(sample_id: str = "sample") -> dict:
    current = {
        "position": 0,
        "event_type": "run",
        "node_type": "run_control",
        "role": "init",
        "raw_action": "baseline_start",
        "action_family": "other",
        "model_id": "stack-a",
    }
    return {
        "sample_id": sample_id,
        "video_id": "video-a",
        "run_id": "run-a",
        "split": "train",
        "model_input": {
            "history": [current],
            "current_node": current,
            "task_context": {
                "answer_type": "multiple_choice",
                "domain": "science",
                "question_type": "unknown",
                "required_modalities": ["scene"],
                "temporal_scope": "unspecified",
                "sub_category": "general",
                "official_task_type": "reasoning",
            },
            "stack_context": {
                "baseline": "star",
                "model_stack_id": "stack-a",
                "planner_model_id": "model-a",
            },
        },
    }


def make_label(sample_id: str = "sample", width: int = 1) -> dict:
    return {
        "sample_id": sample_id,
        "video_id": "video-a",
        "run_id": "run-a",
        "split": "train",
        "future_horizon": 5,
        "future_layers": [
            {
                "layer_offset": 1,
                "nodes": [
                    {
                        "node_id": f"node-{index}",
                        "node_type": "planner",
                        "raw_action": "planner.generate",
                        "model_id": "model-a",
                        "execution_lane": "unknown",
                        "action_family": "other",
                    }
                    for index in range(width)
                ],
            }
        ],
    }


class FakeHead:
    def __init__(self, probabilities: dict[int, float]) -> None:
        self._probabilities = probabilities

    def probabilities(self, _x) -> dict[int, float]:
        return dict(self._probabilities)


class P9dTopologyTabularTests(unittest.TestCase):
    def test_signature_is_identity_free_and_shape_targets_keep_horizon_padding(self) -> None:
        signature = topology_signature(make_label(width=2))
        self.assertNotIn("node-0", repr(signature))
        self.assertEqual(_shape_targets(signature), (1, (2, 0, 0, 0, 0)))

    def test_context_key_uses_only_causal_model_input(self) -> None:
        key = context_key(make_feature()["model_input"])
        text = repr(key)
        self.assertNotIn("video-a", text)
        self.assertNotIn("run-a", text)
        self.assertIn("history_length", text)

    def test_fake_tabular_heads_decode_exact_identity_free_shape(self) -> None:
        pairs = [(make_feature(), make_label(width=1))]
        model = TabularModel({}, 0)
        model.encoder.fit(pairs)
        model.prototype_decoder.fit(pairs)
        model.heads = {
            "layer_count": FakeHead({1: 1.0}),
            "width_1": FakeHead({1: 1.0}),
            "width_2": FakeHead({0: 1.0}),
            "width_3": FakeHead({0: 1.0}),
            "width_4": FakeHead({0: 1.0}),
            "width_5": FakeHead({0: 1.0}),
        }
        metrics, rows = evaluate(pairs, model)
        self.assertEqual(metrics["top1_exact_shape_coverage"], 1.0)
        self.assertEqual(metrics["top1_exact_signature_coverage"], 1.0)
        self.assertEqual(metrics["prototype_field_accuracy"], 1.0)
        node = rows[0]["top_scenarios"][0]["layers"][0]["nodes"][0]
        self.assertNotIn("node_id", node)
        self.assertNotIn("predecessor_node_ids", node)

    def test_width_limit_is_explicit(self) -> None:
        with self.assertRaises(ValueError):
            topology_signature(make_label(width=MAX_WIDTH + 1))


if __name__ == "__main__":
    unittest.main()
