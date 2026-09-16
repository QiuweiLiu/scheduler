#!/usr/bin/env python3
"""真实数据 CPU smoke:B04/B08/B09/B05 各跑最小规模,验证产物契约。"""
import json
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2 import evaluation
from r2b import models_ft, models_meta, models_residual, models_seq
from r2.run_artifacts import verify_completed_run


import unittest as _u


def _oof_available(root):
    return (root / "baselines" / "COMPLETE").exists()


class R2SmokeIntegrationTest(unittest.TestCase):
    def _check_artifact(self, final, expected_role_rows=2029, expected_family_rows=615):
        manifest = json.loads((final / "dataset_manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest["test_targets_used"])
        self.assertEqual(manifest["test_rows_predicted"], 0)
        rows = [
            json.loads(line)
            for line in (final / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        role_rows = [r for r in rows if r["head"] == "role"]
        family_rows = [r for r in rows if r["head"] == "family"]
        self.assertEqual(len(role_rows), expected_role_rows)
        self.assertEqual(len(family_rows), expected_family_rows)
        self.assertEqual({r["split"] for r in rows}, {"validation"})
        metrics = json.loads((final / "metrics.json").read_text(encoding="utf-8"))
        recomputed = evaluation.evaluate(role_rows, family_rows)
        self.assertTrue(all(metrics[k] == recomputed[k] for k in recomputed))
        self.assertEqual(metrics["run"]["status"], "success")

    def test_b04_ft_smoke_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            config = models_ft.build_ft_config(
                "B04_ft_smoke", 0, mode="smoke",
                hyperparameter_overrides={
                    "d_model": 16, "dropout": 0.0, "batch_size": 1024,
                    "max_epochs": 1, "patience": 1,
                },
            )
            final = models_ft.run_validation_ft(root, config, device="cpu")
            self.assertEqual(final, verify_completed_run(root, config))
            self._check_artifact(final)

    def test_b08_meta_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            # 需要 OOF 缓存前置;此测试只验证 config 契约与 seed 0
            config = models_meta.build_meta_config("B08_meta_smoke", mode="smoke")
            self.assertEqual(config["seed"], 0)
            with self.assertRaises(Exception):
                models_meta.run_validation_meta(root, config)  # 无 OOF 缓存 → fail closed

    def test_b09_mlp_residual_smoke_requires_oof(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            config = models_residual.build_residual_config(
                "B09_mlp_smoke", "B09", 0, mode="smoke",
                hyperparameter_overrides={
                    "hidden": 16, "dropout": 0.0, "batch_size": 1024,
                    "max_epochs": 1, "patience": 1,
                },
            )
            if _oof_available(root):
                final = models_residual.run_validation_residual(root, config, "B09", device="cpu")
                self.assertEqual(final, verify_completed_run(root, config))
            else:
                with self.assertRaises(Exception):
                    models_residual.run_validation_residual(root, config, "B09", device="cpu")

    def test_b05_gru_smoke_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            config = models_seq.build_seq_config(
                "B05_gru_smoke", "B05", 0, mode="smoke",
                hyperparameter_overrides={
                    "hidden": 8, "dropout": 0.0, "batch_size": 1024,
                    "max_epochs": 1, "patience": 1,
                },
            )
            final = models_seq.run_validation_seq(root, config, "B05", device="cpu")
            self.assertEqual(final, verify_completed_run(root, config))
            self._check_artifact(final)


if __name__ == "__main__":
    unittest.main()
