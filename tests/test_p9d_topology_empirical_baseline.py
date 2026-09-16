from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from scripts.p9d_topology_empirical_baseline import (
    SCHEMA_VERSION,
    condition_key,
    evaluate,
    fit_model,
    prediction_rows,
    run,
    topology_signature,
)


def feature(sample_id: str, current_action: str = "planner.generate", history_size: int = 1) -> dict:
    history = []
    for index in range(history_size):
        history.append(
            {
                "position": index,
                "event_type": "api_call" if index else "run",
                "node_type": "planner" if index else "run_control",
                "role": "plan" if index else "init",
                "raw_action": current_action if index else "baseline_start",
                "action_family": "other",
                "model_id": "model-a",
            }
        )
    return {
        "sample_id": sample_id,
        "video_id": "video-a",
        "run_id": "run-a",
        "split": "train",
        "model_input": {
            "history": history,
            "current_node": history[-1],
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


def label(sample_id: str, widths: list[int], split: str = "train") -> dict:
    layers = []
    for offset, width in enumerate(widths, 1):
        layers.append(
            {
                "layer_offset": offset,
                "nodes": [
                    {
                        "node_id": f"future-{offset}-{index}",
                        "predecessor_node_ids": [],
                        "node_type": "planner" if index == 0 else "videotool_temporal",
                        "raw_action": "planner.generate" if index == 0 else "frame-selector",
                        "model_id": "model-a" if index == 0 else "tool-a",
                        "execution_lane": "unknown",
                        "action_family": "other" if index == 0 else "select_frames",
                    }
                    for index in range(width)
                ],
            }
        )
    return {
        "sample_id": sample_id,
        "video_id": "video-a",
        "run_id": "run-a",
        "split": split,
        "future_horizon": 5,
        "future_layers": layers,
    }


def write_split(root: Path, split: str, pairs: list[tuple[dict, dict]]) -> None:
    with gzip.open(root / f"features_{split}.jsonl.gz", "wt", encoding="utf-8") as feature_handle:
        with gzip.open(root / f"labels_{split}.jsonl.gz", "wt", encoding="utf-8") as label_handle:
            for feature_row, label_row in pairs:
                feature_row = dict(feature_row)
                feature_row["split"] = split
                label_row = dict(label_row)
                label_row["split"] = split
                feature_handle.write(json.dumps(feature_row) + "\n")
                label_handle.write(json.dumps(label_row) + "\n")


class P9dTopologyEmpiricalBaselineTests(unittest.TestCase):
    def test_signature_is_identity_free_and_order_invariant(self) -> None:
        first = label("s1", [2, 1])
        second = label("s1", [2, 1])
        second["future_layers"][0]["nodes"].reverse()
        second["future_layers"][0]["nodes"][0]["node_id"] = "different-id"
        self.assertEqual(topology_signature(first), topology_signature(second))
        self.assertNotIn("future-1-0", json.dumps(topology_signature(first)))

    def test_condition_schemes_are_causal_and_history_changes_only_history_scheme(self) -> None:
        first = feature("s1", history_size=2)
        second = feature("s2", current_action="different.action", history_size=2)
        full_first = condition_key(first["model_input"], "current_full")
        full_second = condition_key(second["model_input"], "current_full")
        structural_first = condition_key(first["model_input"], "current_structural")
        structural_second = condition_key(second["model_input"], "current_structural")
        history_first = condition_key(first["model_input"], "history2_full")
        history_second = condition_key(second["model_input"], "history2_full")
        self.assertNotEqual(full_first, full_second)
        self.assertEqual(structural_first, structural_second)
        self.assertNotEqual(history_first, history_second)
        self.assertNotIn("video-a", " ".join(full_first))

    def test_empirical_fit_and_metrics_use_global_fallback(self) -> None:
        train = [(feature("a"), label("a", [1])), (feature("b"), label("b", [1]))]
        model = fit_model(train, "current_full")
        validation_feature = feature("v", current_action="unseen.action")
        validation_feature["split"] = "validation"
        validation_feature["model_input"]["current_node"]["raw_action"] = "unseen.action"
        validation = [(validation_feature, label("v", [1], split="validation"))]
        metrics = evaluate(validation, model, "current_full", 1e-12)
        self.assertEqual(metrics["sample_count"], 1)
        self.assertEqual(metrics["seen_condition_key_rate"], 0.0)
        self.assertEqual(metrics["top1_exact_signature_coverage"], 1.0)

    def test_prediction_rows_do_not_emit_node_identity_or_edges(self) -> None:
        train = [(feature("a"), label("a", [2]))]
        model = fit_model(train, "current_full")
        rows = list(prediction_rows(train, model, "current_full", "train"))
        payload = json.dumps(rows)
        self.assertNotIn("node_id", payload)
        self.assertNotIn("predecessor_node_ids", payload)
        self.assertIn("top_scenarios", rows[0])

    def test_end_to_end_run_writes_frozen_split_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            experiment = root / "experiment"
            dataset.mkdir()
            train = [(feature("train-a"), label("train-a", [1])), (feature("train-b"), label("train-b", [2]))]
            for split in ("train", "validation", "test", "holdout"):
                pairs = [(dict(feature(f"{split}-a")), label(f"{split}-a", [1], split=split))]
                if split == "train":
                    pairs = train
                write_split(dataset, split, pairs)
            manifest = {
                "schema_version": "topology-predictor-p9d-v1",
                "source": {
                    "scheduler_trace_groups_used_for_fit": False,
                    "s_train_s_val_t_final_used": False,
                },
            }
            (dataset / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (dataset / "alignment_report.json").write_text("{}", encoding="utf-8")
            config = {
                "schema_version": SCHEMA_VERSION,
                "experiment_id": "EXP-test-p9d",
                "experiment_root": str(experiment),
                "dataset_root": str(dataset),
                "horizon": 5,
                "max_scenarios": 3,
                "seed": 0,
                "nll_floor": 1e-12,
                "candidate_schemes": ["current_full", "current_structural", "history2_full"],
                "selection": {"split": "validation", "primary_metric": "nll_mean"},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            metrics = run(config_path)
            self.assertEqual(metrics["status"], "passed_empirical_baseline")
            self.assertIn(metrics["fit_contract"]["selected_scheme"], ("current_full", "current_structural", "history2_full"))
            self.assertTrue((experiment / "metrics.json").exists())
            self.assertTrue((experiment / "artifacts" / "predictions_holdout.jsonl.gz").exists())


if __name__ == "__main__":
    unittest.main()
