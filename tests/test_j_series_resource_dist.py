"""Unit tests for the resource-v2 discrete-distribution line.

Covers bin construction, bin/band indexing, the four loss terms, quantile/mean
derivation, the ordering property of the ranked probability score, and the
gate evaluation.  CPU-only, no checkpoint, no dataset needed.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import j_series_resource_dist as dist  # noqa: E402

try:
    import torch  # noqa: E402

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


SAMPLE = np.asarray([0.1] * 40 + [0.5, 3.0, 12.0, 250.0, 900.0, 2500.0, 6000.0, 9500.0,
                                  12000.0, 18000.0, 22000.0, 40000.0], dtype=np.float64)


class BinTests(unittest.TestCase):
    def test_edges_are_increasing_and_overflow_last(self):
        edges = dist.build_bin_edges(SAMPLE, n_bins=12, near_zero_edge_ms=1.0, upper_quantile=0.999)
        self.assertEqual(len(edges), 13)
        self.assertEqual(edges[0], 0.0)
        self.assertEqual(edges[1], 1.0)
        self.assertTrue(math.isinf(edges[-1]))
        self.assertTrue(np.all(np.diff(edges[:-1]) > 0))

    def test_bin_index_handles_zero_and_overflow(self):
        edges = dist.build_bin_edges(SAMPLE, n_bins=6, near_zero_edge_ms=1.0, upper_quantile=0.999)
        idx = dist.bin_index(np.asarray([0.0, 0.05, 0.99, 1.0, 1e9]), edges)
        self.assertEqual(idx[0], 0)  # 0 ms lands in the near-zero bin
        self.assertEqual(idx[1], 0)  # 0.05 ms also
        self.assertEqual(idx[2], 0)  # just below the near-zero edge
        self.assertGreaterEqual(idx[3], 1)  # exactly at the edge moves up
        self.assertEqual(idx[4], len(edges) - 2)  # huge value -> overflow bin

    def test_representatives_are_positive_and_ordered(self):
        edges = dist.build_bin_edges(SAMPLE, n_bins=10)
        reps = dist.bin_representatives(edges)
        self.assertTrue(np.all(reps > 0))
        self.assertTrue(np.all(np.diff(reps) > 0))

    def test_band_index_boundaries(self):
        band = dist.band_index(np.asarray([0.5, 1.0, 999.0, 1000.0, 4999.0, 5000.0, 1e6]))
        self.assertEqual(list(band), [0, 1, 1, 2, 2, 3, 3])


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class LossTests(unittest.TestCase):
    def setUp(self):
        self.edges = dist.build_bin_edges(SAMPLE, n_bins=16)
        self.n_bins = len(self.edges) - 1

    def _onehot_logits(self, bin_idx: int, scale: float = 8.0):
        logits = torch.full((1, self.n_bins), -scale)
        logits[0, bin_idx] = scale
        return logits

    def test_nll_is_lower_for_the_correct_bin(self):
        good = self._onehot_logits(5)
        bad = self._onehot_logits(1)
        mask = torch.ones(1)
        tgt = torch.tensor([5])
        self.assertLess(float(dist.nll_loss(good, tgt, mask)), float(dist.nll_loss(bad, tgt, mask)))

    def test_rps_respects_ordering(self):
        """Predicting 7 s when the truth is 8 s must cost less than predicting 0.1 ms."""

        edges = dist.build_bin_edges(np.linspace(1, 40000, 400), n_bins=24)
        true_bin = int(dist.bin_index(np.asarray([8000.0]), edges)[0])
        near_bin = int(dist.bin_index(np.asarray([7000.0]), edges)[0])
        far_bin = 0
        mask = torch.ones(1)
        tgt = torch.tensor([true_bin])
        near_cost = float(dist.rps_loss(self._logits_for(edges, near_bin), tgt, mask, edges))
        far_cost = float(dist.rps_loss(self._logits_for(edges, far_bin), tgt, mask, edges))
        self.assertLess(near_cost, far_cost)

    @staticmethod
    def _logits_for(edges, bin_idx, scale=10.0):
        logits = torch.full((1, len(edges) - 1), -scale)
        logits[0, bin_idx] = scale
        return logits

    def test_pseudo_huber_is_bounded_per_unit_scale(self):
        mask = torch.ones(4)
        small = float(dist.pseudo_huber_point(torch.tensor([100.0]), torch.tensor([110.0]), 100.0, 2.0, torch.ones(1)))
        huge = float(dist.pseudo_huber_point(torch.tensor([100.0]), torch.tensor([100000.0]), 100.0, 2.0, torch.ones(1)))
        self.assertLess(small, huge)
        # a Huber-like term must grow sub-quadratically, so a 1000x residual must not cost 1e6
        self.assertLess(huge, 1e4)
        del mask

    def test_derived_quantiles_are_monotone_and_mean_between(self):
        probs = torch.softmax(torch.randn(3, self.n_bins), dim=-1)
        derived = dist.derive_from_probs(probs, self.edges)
        self.assertTrue(torch.all(derived["q50_ms"] <= derived["q90_ms"] + 1e-6))
        self.assertTrue(torch.all(derived["q90_ms"] <= derived["q95_ms"] + 1e-6))
        reps = dist.bin_representatives(self.edges)
        self.assertTrue(float(torch.min(derived["mean_ms"])) >= reps.min() - 1e-6)
        self.assertTrue(float(torch.max(derived["mean_ms"])) <= reps.max() + 1e-6)

    def test_coarse_band_ce_prefers_the_right_band(self):
        edges = dist.build_bin_edges(SAMPLE, n_bins=16)
        good = torch.full((1, len(edges) - 1), 0.0)
        good[0, 0] = 12.0  # near-zero bin => ultra_cheap
        bad = torch.full((1, len(edges) - 1), 0.0)
        bad[0, len(edges) - 2] = 12.0  # overflow bin => expensive
        mask = torch.ones(1)
        band = torch.tensor([0])
        self.assertLess(float(dist.coarse_band_ce(good, band, mask, edges)),
                        float(dist.coarse_band_ce(bad, band, mask, edges)))


class MetricTests(unittest.TestCase):
    def _perfect_probs(self, values, edges):
        rows = []
        for value in values:
            probs = np.zeros(len(edges) - 1)
            probs[int(dist.bin_index(np.asarray([value]), edges)[0])] = 1.0
            rows.append(probs)
        return np.asarray(rows)

    def test_perfect_distribution_reports_zero_crossing(self):
        edges = dist.build_bin_edges(SAMPLE, n_bins=16)
        truth = np.asarray([0.1, 12.0, 900.0, 9500.0, 40000.0])
        report = dist.evaluate_distribution(self._perfect_probs(truth, edges), truth, edges)
        self.assertEqual(report["A_distribution_calibration"]["quantile_crossing_rate"], 0.0)
        self.assertEqual(report["n_slot_pairs"], 5)
        # a one-hot distribution predicts the bin representative, so the pinball must be
        # small relative to the truth scale even though mean/truth is not exactly 1
        self.assertLess(report["B_distribution_accuracy"]["q50"]["normalized_pinball"], 0.5)
        self.assertGreater(report["D_scheduler_consumption"]["sum_mean_over_true"], 0.2)
        self.assertLess(report["D_scheduler_consumption"]["sum_mean_over_true"], 5.0)

    def test_quantile_report_matches_manual_pinball(self):
        q50 = np.asarray([1.0, 10.0, 100.0])
        q90 = np.asarray([2.0, 20.0, 200.0])
        q95 = np.asarray([3.0, 30.0, 300.0])
        truth = np.asarray([1.5, 25.0, 150.0])
        report = dist.evaluate_quantiles(q50, q90, q95, truth)
        manual = float(np.mean(dist.pinball(q95, truth, 0.95)))
        self.assertAlmostEqual(report["B_distribution_accuracy"]["q95"]["pinball_ms"], manual, places=9)
        self.assertEqual(report["A_distribution_calibration"]["quantile_crossing_rate"], 0.0)

    def test_gates_fail_when_nothing_changes(self):
        q50 = np.asarray([1.0, 10.0, 100.0, 1000.0])
        truth = np.asarray([2.0, 12.0, 120.0, 1200.0])
        base_report = dist.evaluate_quantiles(q50, q50 * 2, q50 * 3, truth)
        gates = dist.evaluate_gates(base_report, base_report)
        self.assertFalse(gates["headline"]["pass"])

    def test_tail_recall_bounds(self):
        truth = np.asarray([0.1, 0.2, 0.3, 500.0, 900.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
        self.assertEqual(dist.tail_recall(truth.copy(), truth), 1.0)
        self.assertEqual(dist.tail_recall(-truth, truth), 0.0)


if __name__ == "__main__":
    unittest.main()
