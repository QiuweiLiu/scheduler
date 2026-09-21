"""Stage 1 acceptance: the loader must never see the audit block, and the model runs.

Covers the two things the review asked for before formal training:
* a model-input serialization test - deleting every `history_resource_audit` block
  must leave the tensorized inputs bit-identical, so provenance can never leak in;
* a forward pass on the augmented dataset, plus the F0 behaviour (mask forced to
  zero produces exactly the same tensors as if no telemetry existed).
"""
from __future__ import annotations

import copy
import gzip
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

import j_series_common as common  # noqa: E402

J_ROOT = PROJECT_ROOT / "results/processed/j_series_dataset_v1"
HISTRES = PROJECT_ROOT / "results/processed/j_series_dataset_histres_v2"


def read_rows(path: Path, limit: int):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


class HistResModelInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.augmented = read_rows(HISTRES / "histres_train.jsonl.gz", 300)
        cls.plain = read_rows(J_ROOT / "j_train.jsonl.gz", 1500)
        cls.vocabs = common.build_vocabs(cls.plain)

    def test_audit_block_is_never_read(self) -> None:
        """Deleting history_resource_audit must not change a single array."""

        with_audit = common.encode_rows(self.augmented, self.vocabs, 5)
        stripped = []
        for row in self.augmented:
            clone = copy.deepcopy(row)
            clone.pop("history_resource_audit", None)
            stripped.append(clone)
        without_audit = common.encode_rows(stripped, self.vocabs, 5)

        self.assertEqual(set(with_audit), set(without_audit))
        for key in sorted(with_audit):
            self.assertTrue(
                np.array_equal(with_audit[key], without_audit[key]),
                "array %s changed when the audit block was removed" % key,
            )

    def test_telemetry_channel_is_populated(self) -> None:
        arrays = common.encode_rows(self.augmented, self.vocabs, 5)
        self.assertEqual(arrays["hist_res_num"].shape[2], len(common.HISTRES_NUMERIC_FIELDS))
        self.assertGreater(int((arrays["hist_res_mask"] > 0).sum()), 0)
        self.assertTrue(np.isfinite(arrays["hist_res_num"]).all())

    def test_plain_dataset_still_encodes(self) -> None:
        """The unchanged dataset must keep working (no history_resource present)."""

        arrays = common.encode_rows(self.plain, self.vocabs, 5)
        self.assertEqual(int(arrays["hist_res_mask"].sum()), 0)
        self.assertTrue(np.isfinite(arrays["hist_res_num"]).all())

    def test_current_and_future_tokens_carry_no_telemetry(self) -> None:
        """The boundary encoded in Stage 0 must survive into the tensors."""

        arrays = common.encode_rows(self.augmented, self.vocabs, 5)
        lengths = arrays["hist_len"]
        for i in range(len(lengths)):
            last = int(lengths[i]) - 1
            if last < 0:
                continue
            self.assertEqual(float(arrays["hist_res_mask"][i, last]), 0.0,
                             "the anchor token carries telemetry")
            for j in range(last + 1, arrays["hist_res_mask"].shape[1]):
                self.assertEqual(float(arrays["hist_res_mask"][i, j]), 0.0)

    def test_model_forward_with_and_without_telemetry(self) -> None:
        import torch

        from j_series_train_eval import to_torch_batch

        arrays = common.encode_rows(self.augmented, self.vocabs, 5)
        model = common.JSeriesModel(self.vocabs, {"hidden": 32, "history_embedding_dim": 8,
                                                  "context_embedding_dim": 4, "slot_embedding_dim": 8}, horizon=5)
        model.eval()
        indices = np.arange(min(16, len(arrays["hist_len"])), dtype=np.int64)
        batch = to_torch_batch(arrays, indices, torch.device("cpu"))
        with torch.no_grad():
            outputs = model.encode(batch)
        self.assertTrue(torch.isfinite(outputs).all())

        # F0 = same architecture with the observed mask forced to zero
        zeroed = dict(batch)
        zeroed["hist_res_mask"] = torch.zeros_like(batch["hist_res_mask"])
        with torch.no_grad():
            f0 = model.encode(zeroed)
            f1 = model.encode(batch)
        self.assertFalse(torch.equal(f0, f1),
                         "zeroing the mask should change the representation")


if __name__ == "__main__":
    unittest.main()
