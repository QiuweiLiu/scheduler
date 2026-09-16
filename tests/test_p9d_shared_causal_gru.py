import unittest

import numpy as np

from scripts.p9d_shared_causal_gru import (
    ALL_TOPOLOGY_FIELDS,
    InputEncoder,
    MAX_WIDTH,
    ROLE_LABELS,
    SharedCausalGRU,
    TargetCodec,
    audit_model_input,
    build_target_arrays,
    loss_components,
    shape_targets,
    topology_signature,
)


def feature(sample_id="sample"):
    current = {
        "event_type": "api_call",
        "node_type": "planner",
        "role": "plan",
        "raw_action": "planner.generate",
        "action_family": "other",
        "model_id": "model-a",
    }
    return {
        "sample_id": sample_id,
        "video_id": "video-a",
        "run_id": "run-a",
        "current_node_id": "current-a",
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
                "baseline": "langgraph_react",
                "model_stack_id": "stack-a",
                "planner_model_id": "model-a",
            },
        },
    }


def label(sample_id="sample", width=2):
    return {
        "sample_id": sample_id,
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
                        "role": "plan",
                    }
                    for index in range(width)
                ],
            }
        ],
    }


class SharedCausalGRUContractTests(unittest.TestCase):
    def test_signature_ignores_node_identity_and_preserves_width(self):
        signature = topology_signature(label(width=2))
        self.assertNotIn("node-0", repr(signature))
        self.assertEqual(shape_targets(signature), (1, (2, 0, 0, 0, 0)))

    def test_model_input_audit_rejects_future_and_truth_keys(self):
        bad = {"history": [], "future_events": [], "runtime_ms": 1.0}
        violations = audit_model_input(bad)
        self.assertEqual(violations, ["model_input.future_events", "model_input.runtime_ms"])

    def test_train_only_encoders_and_target_shapes(self):
        pairs = [(feature(), label())]
        encoder = InputEncoder(max_history=4).fit(pairs)
        encoded = encoder.transform(pairs)
        self.assertEqual(encoded["tokens"].shape, (1, 4, 6))
        self.assertEqual(int(encoded["lengths"][0]), 1)
        codec = TargetCodec().fit(pairs)
        targets = build_target_arrays(
            pairs,
            [{"role": "execute", "family": "select_frames", "family_mask": True}],
            codec,
        )
        self.assertEqual(targets["prototype_targets"].shape, (1, 5, MAX_WIDTH, 5))
        self.assertEqual(targets["topology_role_targets"].shape, (1, 5, MAX_WIDTH))
        self.assertEqual(int(targets["topology_role_targets"][0, 0, 0]), ROLE_LABELS.index("plan"))
        self.assertTrue(np.all(targets["topology_role_targets"][0, 0, 2:] == -1))
        self.assertEqual(int(targets["topology_family_targets"][0, 0, 0]), 1)
        self.assertTrue(np.all(targets["topology_family_targets"][0, 0, 2:] == -1))
        self.assertEqual(tuple(ALL_TOPOLOGY_FIELDS), ("node_type", "raw_action", "model_id", "execution_lane", "action_family", "role"))

    @unittest.skipUnless(SharedCausalGRU is not None, "torch is unavailable in the local environment")
    def test_forward_loss_and_consistency_are_finite(self):
        import torch

        pairs = [(feature(), label())]
        encoder = InputEncoder(max_history=4).fit(pairs)
        codec = TargetCodec().fit(pairs)
        encoded = encoder.transform(pairs)
        encoded.update(
            build_target_arrays(
                pairs,
                [{"role": "execute", "family": "select_frames", "family_mask": True}],
                codec,
            )
        )
        history_sizes, context_sizes = encoder.vocab_sizes()
        model = SharedCausalGRU(history_sizes, context_sizes, codec.field_sizes(), codec.role_size(), dropout=0.0)
        batch = {
            key: torch.from_numpy(encoded[key])
            for key in ("tokens", "lengths", "context", "behavior_role_targets", "behavior_family_targets", "behavior_family_mask", "layer_targets", "width_targets", "prototype_targets", "topology_role_targets", "topology_family_targets")
        }
        outputs = model({key: batch[key] for key in ("tokens", "lengths", "context")})
        losses = loss_components(
            outputs,
            batch,
            {"role": 1.0, "family": 1.0, "layer": 1.0, "width": 1.0, "prototype": 0.5, "topology_role": 0.25, "topology_family": 0.25, "consistency": 0.05},
        )
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None))


if __name__ == "__main__":
    unittest.main()
