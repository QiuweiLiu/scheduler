"""Stage 1.2 acceptance: a compatible-loaded J3 with telemetry off must reproduce
the frozen encoder, verified against an independent reimplementation of the old
encode path (not against the new code itself).
"""
from __future__ import annotations

import gzip
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import j_series_common as common  # noqa: E402
import j_series_train_eval as te  # noqa: E402

J_ROOT = PROJECT_ROOT / "results/processed/j_series_dataset_v1"
HISTRES = PROJECT_ROOT / "results/processed/j_series_dataset_histres_v2"
J3_CHECKPOINT = (
    PROJECT_ROOT / "experiments/EXP-20260911_p9d_j_predictor_acceptance/artifacts/predictor/J3_seed11.pt"
)
J3_SHA256 = "0ee8ded4f92553853026ee24a3c320f21d524f9d2c60de841091430d14949c77"


def read_rows(path: Path, limit: int):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def legacy_encode(model, batch):
    """The pre-Stage-1 token embedding + GRU, reimplemented from the frozen contract."""

    device = next(model.parameters()).device
    hist_ids = {field: batch["hist_%s" % field] for field in common.HISTORY_FIELDS}
    lengths = batch["hist_len"]
    steps = hist_ids[common.HISTORY_FIELDS[0]].shape[1]
    emb = None
    for field in common.HISTORY_FIELDS:
        value = model.hist_emb[field](hist_ids[field])
        emb = value if emb is None else emb + value
    positions = torch.arange(steps, device=device).unsqueeze(0).clamp(max=common.POSITION_BUCKETS - 1)
    emb = emb + model.pos_emb(positions)
    packed = nn.utils.rnn.pack_padded_sequence(
        emb, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
    )
    _, hidden = model.gru(packed)
    h = hidden[-1]
    ctx = None
    for idx, field in enumerate(common.CONTEXT_FIELDS):
        value = model.ctx_emb[field](batch["ctx_ids"][:, idx])
        ctx = value if ctx is None else ctx + value
    ctx = ctx + model.mod_proj(batch["mod_vec"])
    return torch.tanh(model.repr(h) + model.repr_ctx(ctx))


class J3CompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # the frozen checkpoint was trained on the full train split, so the vocab sizes
        # must come from the same rows or ctx_emb shapes will not match
        cls.vocabs = common.build_vocabs(read_rows(J_ROOT / "j_train.jsonl.gz", 10**9))
        cls.augmented = read_rows(HISTRES / "histres_train.jsonl.gz", 200)
        cls.payload = torch.load(J3_CHECKPOINT, map_location="cpu", weights_only=False)
        cls.config = {"hidden": 128, "history_embedding_dim": 16, "context_embedding_dim": 12,
                      "slot_embedding_dim": 16, "dropout": 0.1}

    def build(self):
        return common.JSeriesModel(self.vocabs, self.config, horizon=5, duration_mode="shared")

    def test_checkpoint_sha_is_the_frozen_one(self) -> None:
        self.assertEqual(te.common.sha256_file(J3_CHECKPOINT), J3_SHA256)

    def test_strict_load_would_fail_and_compatible_load_succeeds(self) -> None:
        model = self.build()
        with self.assertRaises(RuntimeError):
            model.load_state_dict(self.payload["model_state"])
        report = te.load_j3_compatible(model, self.payload["model_state"])
        self.assertEqual(report["unexpected_missing_keys"], [])
        self.assertEqual(report["unexpected_shape_mismatch"], [])
        self.assertEqual(report["extra_in_checkpoint"], [])
        self.assertTrue(all(k.startswith(("hist_res_proj", "hist_status_emb"))
                            for k in report["expected_missing_keys"]))
        self.assertGreater(report["loaded_keys"], 0)

    def test_compatible_load_rejects_a_missing_backbone_tensor(self) -> None:
        model = self.build()
        damaged = dict(self.payload["model_state"])
        victim = next(k for k in damaged if k.startswith("hist_emb"))
        damaged.pop(victim)
        with self.assertRaises(SystemExit):
            te.load_j3_compatible(model, damaged)

    def test_telemetry_off_reproduces_the_frozen_encoder(self) -> None:
        """The new architecture must not silently change the frozen baseline."""

        model = self.build()
        te.load_j3_compatible(model, self.payload["model_state"])
        model.eval()

        arrays = common.encode_rows(self.augmented, self.vocabs, 5)
        indices = np.arange(min(64, len(arrays["hist_len"])), dtype=np.int64)
        batch = te.to_torch_batch(arrays, indices, torch.device("cpu"))

        with torch.no_grad():
            new = model.encode(batch, use_history_telemetry=False)
            reference = legacy_encode(model, batch)
        max_abs = float((new - reference).abs().max())
        self.assertLess(max_abs, 1e-6,
                        "telemetry-off encode differs from the legacy path by %g" % max_abs)

    def test_telemetry_on_changes_the_encoder(self) -> None:
        model = self.build()
        te.load_j3_compatible(model, self.payload["model_state"])
        model.eval()
        arrays = common.encode_rows(self.augmented, self.vocabs, 5)
        indices = np.arange(min(64, len(arrays["hist_len"])), dtype=np.int64)
        batch = te.to_torch_batch(arrays, indices, torch.device("cpu"))
        with torch.no_grad():
            off = model.encode(batch, use_history_telemetry=False)
            on = model.encode(batch, use_history_telemetry=True)
        self.assertFalse(torch.equal(off, on))

    def test_matched_initialisation_is_deterministic(self) -> None:
        """F0 and F1 must start from the same state for a given seed."""

        states = []
        for _ in range(2):
            torch.manual_seed(11)
            model = self.build()
            te.load_j3_compatible(model, self.payload["model_state"])
            # the new branch is what differs per seed; re-init it deterministically
            for module in (model.hist_res_proj, model.hist_status_emb):
                for parameter in module.parameters():
                    with torch.no_grad():
                        parameter.normal_(0.0, 0.02)
            states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        first, second = states
        self.assertEqual(set(first), set(second))
        for key in first:
            self.assertTrue(torch.equal(first[key], second[key]), "state differs at %s" % key)


if __name__ == "__main__":
    unittest.main()
