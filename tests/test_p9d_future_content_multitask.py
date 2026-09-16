import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.p9d_future_content_multitask import (
    CONTENT_FIELDS,
    ContentCodec,
    FutureContentGRU,
    _best_assignment,
    _content_signature,
    build_target_arrays,
    loss_components,
    stage0_audit,
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
                "baseline": "langgraph_react",
                "model_stack_id": "stack-a",
                "planner_model_id": "model-a",
            },
        },
    }


def label(sample_id="sample", width=2):
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
                        "node_type": "planner" if index == 0 else "videotool_temporal",
                        "raw_action": "planner.generate" if index == 0 else "temporal-qa",
                        "model_id": "model-a",
                        "execution_lane": "unknown",
                        "action_family": "other" if index == 0 else "temporal_ops",
                        "role": "plan" if index == 0 else "execute",
                    }
                    for index in range(width)
                ],
            }
        ],
        "label_summary": {"truncated_at_horizon": False},
        "label_contract": {
            "future_unit": "event_level_dag_bfs_layer",
            "parent_source": "raw_trace.parent_step_ids",
        },
        "source_trace_sha256": "trace-a",
    }


class FutureContentContractTests(unittest.TestCase):
    def test_joint_assignment_is_rectangular_and_content_signature_ignores_ids(self):
        cost = np.asarray([[4.0, 1.0], [0.5, 3.0]])
        self.assertEqual(_best_assignment(cost), [(0, 1), (1, 0)])
        signature = _content_signature(label(width=2))
        self.assertEqual(len(signature[0]), 2)
        self.assertNotIn("node-0", repr(signature))

    def test_codec_is_train_only_and_target_shapes_are_slotwise(self):
        pairs = [(feature(), label())]
        codec = ContentCodec().fit(pairs)
        targets = build_target_arrays(
            pairs,
            [{"role": "execute", "family": "select_frames", "family_mask": True}],
            codec,
        )
        self.assertEqual(tuple(CONTENT_FIELDS), ("node_type", "raw_action", "model_id", "action_family"))
        self.assertEqual(targets["content_targets"].shape, (1, 5, 5, 4))
        self.assertEqual(int(targets["width_targets"][0, 0]), 2)
        self.assertTrue(np.all(targets["content_targets"][0, 0, 2:] == -1))

    @unittest.skipUnless(FutureContentGRU is not None, "torch is unavailable in the local environment")
    def test_forward_and_joint_losses_are_finite(self):
        import torch

        pairs = [(feature(), label())]
        codec = ContentCodec().fit(pairs)
        encoded = {
            "tokens": np.zeros((1, 4, 6), dtype=np.int64),
            "lengths": np.asarray([1], dtype=np.int64),
            "context": np.zeros((1, 10), dtype=np.int64),
        }
        encoded.update(
            build_target_arrays(
                pairs,
                [{"role": "execute", "family": "select_frames", "family_mask": True}],
                codec,
            )
        )
        model = FutureContentGRU([3] * 6, [3] * 10, codec.field_sizes(), dropout=0.0)
        batch = {key: torch.from_numpy(value) for key, value in encoded.items()}
        outputs = model({key: batch[key] for key in ("tokens", "lengths", "context")})
        losses = loss_components(
            outputs,
            batch,
            {"next": 1.0, "structure": 1.0, "content": 1.0},
        )
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertTrue(
            all(
                torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
                if parameter.grad is not None
            )
        )

    def test_stage0_rejects_missing_termination_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "schema_version": "topology-predictor-p9d-v1",
                "source": {
                    "scheduler_trace_groups_used_for_fit": False,
                    "s_train_s_val_t_final_used": False,
                },
            }
            (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            for split in ("train", "validation", "test", "holdout"):
                item_feature = feature(f"{split}-sample")
                item_feature["split"] = split
                item_feature["video_id"] = f"{split}-video"
                item_label = label(f"{split}-sample")
                item_label["split"] = split
                item_label["video_id"] = f"{split}-video"
                item_label["source_trace_sha256"] = f"trace-{split}"
                with gzip.open(root / f"features_{split}.jsonl.gz", "wt", encoding="utf-8") as handle:
                    handle.write(json.dumps(item_feature) + "\n")
                with gzip.open(root / f"labels_{split}.jsonl.gz", "wt", encoding="utf-8") as handle:
                    item_label["label_summary"].pop("truncated_at_horizon")
                    handle.write(json.dumps(item_label) + "\n")
            result = stage0_audit(root)
            self.assertFalse(result["passed"])
            self.assertTrue(any("termination/truncation" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
