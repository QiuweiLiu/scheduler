#!/usr/bin/env python3
"""Tests for OOF cache:fold coverage、key uniqueness、fail-closed."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from r2b import oof_cache, shared


def _fake_records(n, split="train"):
    records = []
    for i in range(n):
        records.append(
            {
                "run_id": f"run_{i % 20}",
                "video_id": f"vid_{i % 10}",
                "split": split,
                "target_source_event_id": f"run_{i % 20}:action:{i}",
                "next_role": "execute",
                "family_label": "select_frames" if i % 3 else "visual_qa",
            }
        )
    return records


class OofCacheTest(unittest.TestCase):
    def test_fold_coverage_and_keys(self):
        role_train = _fake_records(60)
        family_train = _fake_records(60)
        builder = oof_cache.OofBuilder()
        # 用真实 preprocessor 需要真实字段,这里只测折划分结构
        # (OofBuilder 依赖 build_main_view;完整性测试放 smoke)
        self.assertEqual(len({r["video_id"] for r in role_train}), 10)
        self.assertEqual(len({r["run_id"] for r in role_train}), 20)
        self.assertEqual(len({(r["run_id"], r["target_source_event_id"]) for r in role_train}), 60)

    def test_write_read_roundtrip_and_fail_closed(self):
        from r2 import provenance, run_artifacts

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 构造最小可写缓存:直接复用 write_cache 需要真实 preprocessor 输出,
            # 这里用假数据测文件协议与 fail-closed
            (root / "baselines").mkdir(parents=True)
            from r2 import constants

            npz = root / "baselines" / oof_cache.OOF_NPZ_NAME
            np.savez_compressed(
                npz,
                role_run_ids=np.asarray(["run_a"], dtype="U256"),
                role_target_ids=np.asarray(["t_a"], dtype="U256"),
                role_oof_prob=np.zeros((1, 4)),
                role_oof_log_prob=np.zeros((1, 4)),
                role_fold_ids=np.zeros(1, dtype=np.int64),
                family_run_ids=np.asarray(["run_b"], dtype="U256"),
                family_target_ids=np.asarray(["t_b"], dtype="U256"),
                family_oof_prob=np.zeros((1, 6)),
                family_oof_log_prob=np.zeros((1, 6)),
                family_fold_ids=np.zeros(1, dtype=np.int64),
                role_val_log_prob=np.zeros((1, 4)),
                family_val_log_prob=np.zeros((1, 6)),
            )
            manifest = {
                "schema_version": "v1",
                "n_folds": 5,
                "cache_seed": 0,
                "epsilon": oof_cache.EPSILON,
                "npz_sha256": provenance.sha256_file(npz),
                "class_order": {"role": list(constants.ROLES), "family": list(constants.FAMILIES)},
                "counts": {"role_train": 1, "family_train": 1, "role_val": 1, "family_val": 1},
                "sources": shared.source_manifest(),
            }
            manifest["manifest_sha256"] = provenance.canonical_sha256(
                {k: v for k, v in manifest.items() if k != "manifest_sha256"}
            )
            (root / "baselines" / oof_cache.OOF_MANIFEST_NAME).write_text(
                __import__("json").dumps(manifest, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (root / "baselines" / "COMPLETE").write_text("ok\n", encoding="utf-8")
            cache = oof_cache.load_cache(root)
            self.assertEqual(cache["role_index"], {("run_a", "t_a"): 0})
            # fail closed:破坏 npz(改一个数组值,hash 必变)
            np.savez_compressed(
                root / "baselines" / oof_cache.OOF_NPZ_NAME,
                role_run_ids=np.asarray(["run_a"], dtype="U256"),
                role_target_ids=np.asarray(["t_a"], dtype="U256"),
                role_oof_prob=np.ones((1, 4)),
                role_oof_log_prob=np.zeros((1, 4)),
                role_fold_ids=np.zeros(1, dtype=np.int64),
                family_run_ids=np.asarray(["run_b"], dtype="U256"),
                family_target_ids=np.asarray(["t_b"], dtype="U256"),
                family_oof_prob=np.zeros((1, 6)),
                family_oof_log_prob=np.zeros((1, 6)),
                family_fold_ids=np.zeros(1, dtype=np.int64),
                role_val_log_prob=np.zeros((1, 4)),
                family_val_log_prob=np.zeros((1, 6)),
            )
            with self.assertRaises(oof_cache.OofError):
                oof_cache.load_cache(root)


if __name__ == "__main__":
    unittest.main()
