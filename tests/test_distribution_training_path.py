"""Stage 1.4 acceptance: the distribution branch must actually be exercised.

The review's core complaint was that the new head was installed but the old training
path had not switched over, so nothing would have caught a wrong loss branch, wrong
CDF keys or wrong pseudo-Huber parameters until the first real run.
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

import j_series_common as common  # noqa: E402
import j_series_train_eval as te  # noqa: E402

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


class DistributionTrainingPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vocabs = common.build_vocabs(read_rows(J_ROOT / "j_train.jsonl.gz", 10**9))
        cls.rows = read_rows(HISTRES / "histres_train.jsonl.gz", 96)
        cls.bins = te.load_runtime_bins()
        cls.config = {
            "hidden": 64,
            "history_embedding_dim": 8,
            "context_embedding_dim": 4,
            "slot_embedding_dim": 8,
            "runtime_bins": cls.bins,
            "runtime_head_hidden": 64,
        }

    def build(self):
        return common.JSeriesModel(self.vocabs, self.config, horizon=5)

    def test_phase_r_scale_is_verified_from_the_train_split(self) -> None:
        """The frozen constant must match a recomputation, not just be asserted."""

        train = common.encode_rows(read_rows(J_ROOT / "j_train.jsonl.gz", 10**9), self.vocabs, 5)
        report = te.verify_phase_r_point_scale(train)
        self.assertTrue(report["matches"])
        # the median is a float64 reduction, so compare within the same relative
        # tolerance the implementation uses rather than to six decimal places
        self.assertAlmostEqual(report["median_positive_runtime_ms"] / te.PHASE_R_SCALE_MS,
                               1.0, delta=1e-7)
        self.assertEqual(te.PHASE_R_DELTA, 2.0)
        self.assertEqual(self.bins["scale"], te.PHASE_R_SCALE_MS)
        self.assertEqual(self.bins["delta"], te.PHASE_R_DELTA)

    def test_head_width_fails_closed(self) -> None:
        bad = dict(self.config)
        bad.pop("runtime_head_hidden")
        with self.assertRaises(SystemExit):
            common.JSeriesModel(self.vocabs, bad, horizon=5)

    def test_one_training_batch_runs_through_the_distribution_loss(self) -> None:
        """forward -> loss_terms -> backward -> optimizer.step on the real path."""

        model = self.build()
        arrays = common.encode_rows(self.rows, self.vocabs, 5)
        indices = np.arange(32, dtype=np.int64)
        batch = te.to_torch_batch(arrays, indices, torch.device("cpu"))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        encoded = model.encode(batch)
        resource = model.resource(encoded, None)
        self.assertIn("runtime_logits", resource)
        self.assertEqual(tuple(resource["runtime_logits"].shape), (32, 5, 16))

        mask = (torch.arange(5).unsqueeze(0) < batch["length"].unsqueeze(1)).float()
        loss = common.resource_loss(resource, batch, mask, bins=model.runtime_bins)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(model.head_runtime.net[-1].weight.grad.norm()), 0.0)
        optimizer.step()

    def test_loss_terms_passes_the_bins_so_the_distribution_branch_is_reached(self) -> None:
        model = self.build()
        arrays = common.encode_rows(self.rows, self.vocabs, 5)
        batch = te.to_torch_batch(arrays, np.arange(16, dtype=np.int64), torch.device("cpu"))
        outputs = {
            "structure": model.structure(model.encode(batch)),
            "attributes": model.attribute_logits(model.encode(batch)),
            "behavior": model.behavior(model.encode(batch)),
            "resource": model.resource(model.encode(batch), None),
        }
        terms = te.loss_terms(outputs, batch, 5, model=model)
        self.assertTrue(torch.isfinite(terms["resource"]))
        self.assertIn("runtime_logits", outputs["resource"])

    def test_cdf_views_use_the_q_ms_keys(self) -> None:
        model = self.build()
        arrays = common.encode_rows(self.rows, self.vocabs, 5)
        batch = te.to_torch_batch(arrays, np.arange(16, dtype=np.int64), torch.device("cpu"))
        resource = model.resource(model.encode(batch), None)
        import j_series_resource_dist as dist

        reps = np.asarray(model.runtime_bins["representatives_ms"], dtype=np.float64)
        with torch.no_grad():
            probs = torch.softmax(resource["runtime_logits"], dim=-1)
            views = dist.derive_from_probs(probs, reps)
        for key in ("q50_ms", "q90_ms", "q95_ms", "mean_ms"):
            self.assertIn(key, views, "derive_from_probs no longer exposes %s" % key)
            self.assertTrue(np.isfinite(np.asarray(views[key])).all())

    def test_metrics_path_runs_with_the_discretised_head(self) -> None:
        """per_row_metrics must not KeyError once the head is swapped."""

        from j_series_train_eval import Ctx

        model = self.build()
        arrays = common.encode_rows(self.rows, self.vocabs, 5)
        ctx = Ctx(
            config={"model": self.config, "variants": {}},
            run_root=PROJECT_ROOT,
            exp_dir=PROJECT_ROOT,
            device=torch.device("cpu"),
            horizon=5,
            batch_size=16,
            vocabs=self.vocabs,
            train=arrays,
            validation=arrays,
            test=arrays,
            val_draws=np.zeros((1, 1), dtype=np.int64),
            test_draws=np.zeros((1, 1), dtype=np.int64),
        )
        metrics = te.per_row_metrics(model, arrays, "J3", ctx, max_rows=16)
        # the discretised head feeds the same runtime_qscore pipeline as the quantile
        # head; reaching it without a KeyError is the point of this test
        self.assertIn("runtime_qscore", metrics)
        # rows with no valid future slot legitimately carry NaN, so require that the
        # pipeline produced finite scores where a target exists
        finite = np.isfinite(metrics["runtime_qscore"])
        self.assertGreater(int(finite.sum()), 0, "no finite runtime_qscore was produced")
        self.assertTrue((metrics["runtime_qscore"][finite] >= 0).all())


if __name__ == "__main__":
    unittest.main()
