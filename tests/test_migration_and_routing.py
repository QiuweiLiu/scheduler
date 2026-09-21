"""Stage 1.4b acceptance: the migration contract and the F0/F1 routing."""
from __future__ import annotations

import gzip
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

import j_series_common as common  # noqa: E402
import j_series_train_eval as te  # noqa: E402

J_ROOT = PROJECT_ROOT / "results/processed/j_series_dataset_v1"
J3_CHECKPOINT = (
    PROJECT_ROOT / "experiments/EXP-20260911_p9d_j_predictor_acceptance/artifacts/predictor/J3_seed11.pt"
)


def read_rows(path: Path, limit: int):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


class MigrationAndRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vocabs = common.build_vocabs(read_rows(J_ROOT / "j_train.jsonl.gz", 10**9))
        cls.bins = te.load_runtime_bins()
        cls.payload = torch.load(J3_CHECKPOINT, map_location="cpu", weights_only=False)
        cls.config = {
            "hidden": 128,
            "history_embedding_dim": 16,
            "context_embedding_dim": 12,
            "slot_embedding_dim": 16,
            "dropout": 0.1,
            "runtime_bins": cls.bins,
            "runtime_head_hidden": 64,
        }

    def build(self):
        return common.JSeriesModel(self.vocabs, self.config, horizon=5, duration_mode="shared")

    def test_f0_and_f1_differ_in_exactly_one_field(self) -> None:
        f0 = te.DEFAULT_VARIANTS["F0"]
        f1 = te.DEFAULT_VARIANTS["F1"]
        self.assertFalse(f0["use_history_telemetry"])
        self.assertTrue(f1["use_history_telemetry"])
        differing = {k for k in set(f0) | set(f1) if f0.get(k) != f1.get(k)}
        self.assertEqual(differing, {"use_history_telemetry"},
                         "F0 and F1 must differ in exactly one field")
        self.assertTrue(f0["runtime_distribution_head"])
        self.assertTrue(f1["runtime_distribution_head"])

    def test_j3_migrates_into_the_distribution_model(self) -> None:
        model = self.build()
        report = te.load_j3_compatible(model, self.payload["model_state"])
        self.assertEqual(report["unexpected_missing_keys"], [])
        self.assertEqual(report["unexpected_shape_mismatch"], [])
        self.assertEqual(report["extra_in_checkpoint"], [])
        self.assertIn("head_runtime.weight", report["replaced_head_keys"])
        self.assertEqual(report["replaced_head_shapes"]["head_runtime.weight"], [3, 128])
        self.assertTrue(all(k.startswith(("hist_res_proj", "hist_status_emb", "head_runtime.net"))
                            for k in report["expected_missing_keys"]))

    def test_a_wrong_shaped_replaced_head_is_rejected(self) -> None:
        model = self.build()
        damaged = dict(self.payload["model_state"])
        damaged["head_runtime.weight"] = torch.zeros(7, 128)
        with self.assertRaises(SystemExit):
            te.load_j3_compatible(model, damaged)

    def test_f0_and_f1_share_one_initial_state_per_seed(self) -> None:
        """Matched initialisation: both arms clone the same seed-specific state."""

        def build_state():
            torch.manual_seed(11)
            model = self.build()
            te.load_j3_compatible(model, self.payload["model_state"])
            for module in (model.hist_res_proj, model.hist_status_emb, model.head_runtime):
                for parameter in module.parameters():
                    with torch.no_grad():
                        parameter.normal_(0.0, 0.02)
            return {k: v.detach().clone() for k, v in model.state_dict().items()}

        first = build_state()
        second = build_state()
        self.assertEqual(set(first), set(second))
        for key in first:
            self.assertTrue(torch.equal(first[key], second[key]), "initial state differs at %s" % key)
        self.assertTrue(torch.equal(first["gru.weight_ih_l0"], second["gru.weight_ih_l0"]))


if __name__ == "__main__":
    unittest.main()
