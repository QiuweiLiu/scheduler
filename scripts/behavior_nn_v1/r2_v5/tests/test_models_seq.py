#!/usr/bin/env python3
"""Tests for shared config contract + model forward correctness."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import r2b  # noqa: F401  # 注入 scripts_behavior_r2_v5 到 sys.path
from r2 import constants
from r2b import models_ft, models_meta, models_residual, models_seq, shared
from r2.run_artifacts import RunConfigError


class ConfigContractTest(unittest.TestCase):
    def test_formal_configs_locked(self):
        for model_id, builder in (
            ("B04", lambda s: models_ft.build_ft_config("B04_x", s)),
            ("B05", lambda s: models_seq.build_seq_config("B05_x", "B05", s)),
            ("B07", lambda s: models_seq.build_seq_config("B07_x", "B07", s)),
            ("B09", lambda s: models_residual.build_residual_config("B09_x", "B09", s)),
            ("B10", lambda s: models_residual.build_residual_config("B10_x", "B10", s)),
        ):
            cfg = builder(11)
            self.assertEqual(cfg["model_id"], model_id)
            self.assertEqual(cfg["selection_split"], "validation")
            self.assertFalse(cfg["allow_test_selection"])
            with self.assertRaises(shared.ModelError):
                builder(11).__class__  # noqa: B018
                models_ft.build_ft_config("B04_bad", 11, hyperparameter_overrides={"width": 8})
            with self.assertRaises(RunConfigError):
                builder(0)  # NN 不允许 seed 0

    def test_meta_seed_zero(self):
        cfg = models_meta.build_meta_config("B08_x")
        self.assertEqual(cfg["seed"], 0)
        self.assertEqual(cfg["model_id"], "B08")


class SequenceModelForwardTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.vocab = [8, 9, 20, 6]
        self.static_dim = 6

    def _batch(self, tokens_list, lengths, static=None):
        T = max(map(len, tokens_list))
        B = len(tokens_list)
        tokens = np.zeros((B, T, 4), dtype=np.int64)
        mask = np.zeros((B, T), dtype=bool)
        for i, toks in enumerate(tokens_list):
            L = lengths[i]
            for t, tk in enumerate(toks[:L]):
                tokens[i, t] = tk
                mask[i, t] = True
        if static is None:
            static = torch.randn(B, self.static_dim)
        return {
            "static": static,
            "tokens": torch.from_numpy(tokens),
            "lengths": torch.from_numpy(np.asarray(lengths, dtype=np.int64)),
            "mask": torch.from_numpy(mask),
        }

    def test_rnn_shapes_and_len0(self):
        for cell in ("gru", "lstm"):
            model = models_seq.SequenceRNN(self.vocab, self.static_dim, 8, 0.0, 4, cell=cell)
            batch = self._batch([[[1, 2, 3, 4]], []], [1, 0])
            out = model(batch)
            self.assertEqual(tuple(out.shape), (2, 4))
            self.assertTrue(torch.isfinite(out).all())

    def test_rnn_pad_invariance_and_order(self):
        model = models_seq.SequenceRNN(self.vocab, self.static_dim, 8, 0.0, 4, cell="gru")
        fixed_static = torch.randn(2, self.static_dim)
        b1 = self._batch([[[1, 2, 3, 4], [5, 5, 5, 5]], [[4, 4, 3, 3]]], [2, 1], static=fixed_static)
        b2 = self._batch([[[1, 2, 3, 4], [5, 5, 5, 5], [0, 0, 0, 0], [0, 0, 0, 0]],
                          [[4, 4, 3, 3], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], [2, 1], static=fixed_static)
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(b1), model(b2), atol=1e-5))

    def test_causal_transformer_direction(self):
        """追加未来 token 不得改变 last-valid 输出(causal 方向性)。"""
        model = models_seq.CausalTransformer(self.vocab, self.static_dim, 16, 0.0, 4)
        fixed_static = torch.randn(1, self.static_dim)
        base = self._batch([[[1, 2, 3, 4], [5, 5, 5, 5]]], [2], static=fixed_static)
        # 未来 token 只存在于矩阵(被 padding mask 屏蔽),last-valid 仍为位置 1
        extra = self._batch(
            [[[1, 2, 3, 4], [5, 5, 5, 5], [4, 4, 4, 4], [3, 3, 3, 3]]], [2], static=fixed_static
        )
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(base), model(extra), atol=1e-5))

    def test_transformer_allpad_finite(self):
        model = models_seq.CausalTransformer(self.vocab, self.static_dim, 16, 0.0, 4)
        batch = self._batch([[]], [0])
        out = model(batch)
        self.assertTrue(torch.isfinite(out).all())
        out.sum().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))


if __name__ == "__main__":
    unittest.main()
