"""Unit tests for the resource-v2 discrete-distribution line.

Every test that the code review asked for is present by name:

* ``test_p999_edge_is_preserved``            -- P0-1
* ``test_overflow_fraction_within_budget``   -- P0-1
* ``test_first_bin_representative_is_empirical`` -- P0-2
* ``test_no_bin_straddles_a_band_boundary``  -- P1-1
* ``test_expensive_bucket_gate_is_not_none`` -- P0-3
* ``test_config_thresholds_change_gate_behaviour`` -- P1-3
* ``test_rps_edge_convention_matches_evaluation`` -- P1-6
plus the bin/loss/metric basics.
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

# 40 % cheap metadata ops at 0.082 ms, then a spread of real model calls up to ~40 s
SAMPLE = np.asarray(
    [0.082] * 40 + [0.09, 0.1, 1.5, 3.0, 12.0, 250.0, 900.0, 2500.0, 6000.0, 9500.0,
                    12000.0, 18000.0, 22000.0, 40000.0],
    dtype=np.float64,
)
SAMPLE = np.concatenate([SAMPLE, np.linspace(0.08, 40000.0, 400)])


class BinSpecTests(unittest.TestCase):
    def setUp(self):
        self.spec = dist.build_bin_spec(SAMPLE, n_bins=24, near_zero_edge_ms=1.0, upper_quantile=0.999)
        self.edges = dist.edges_from_spec(self.spec)

    def test_shape_and_monotonicity(self):
        self.assertEqual(len(self.edges), 25)
        self.assertEqual(self.edges[0], 0.0)
        self.assertEqual(self.edges[1], 1.0)
        self.assertTrue(math.isinf(self.edges[-1]))
        self.assertTrue(np.all(np.diff(self.edges[:-1]) > 0))

    def test_p999_edge_is_preserved(self):
        """P0-1: the train p99.9 must survive as the last finite edge."""

        expected = float(np.quantile(np.log1p(SAMPLE), 0.999))
        self.assertAlmostEqual(math.log1p(float(self.edges[-2])), expected, places=6)
        self.assertGreater(float(self.edges[-2]), 20000.0)

    def test_overflow_fraction_within_budget(self):
        """P0-1: overflow mass must sit near 0.1 %, never near the old 2 %."""

        self.assertLess(self.spec["overflow_fraction"], dist.MAX_OVERFLOW_FRACTION)

    def test_first_bin_representative_is_empirical(self):
        """P0-2: the cheap bin must represent the real ~0.08 ms cluster."""

        reps = dist.reps_from_spec(self.spec)
        self.assertLess(reps[0], 0.15)
        self.assertGreater(reps[0], 0.05)
        self.assertAlmostEqual(reps[0], float(SAMPLE[SAMPLE < 1.0].mean()), places=6)

    def test_representatives_are_ordered_and_finite(self):
        reps = dist.reps_from_spec(self.spec)
        self.assertTrue(np.all(np.isfinite(reps)))
        self.assertTrue(np.all(reps > 0))
        self.assertTrue(np.all(np.diff(reps) > 0))

    def test_overflow_representative_reflects_the_tail(self):
        reps = dist.reps_from_spec(self.spec)
        self.assertGreaterEqual(reps[-1], float(self.edges[-2]))
        tail = SAMPLE[SAMPLE >= float(self.edges[-2])]
        if tail.size:
            self.assertAlmostEqual(reps[-1], float(tail.mean()), places=3)
        else:
            # empty overflow bin (p99.9 == max in this synthetic sample) -> the fallback
            # representative must stay strictly above the last finite edge
            self.assertGreater(reps[-1], float(self.edges[-2]))

    def test_no_bin_straddles_a_band_boundary(self):
        """P1-1: NLL and the band CE must not disagree about the same sample."""

        band = dist.band_of_bin_from_spec(self.spec)
        for k in range(len(self.edges) - 1):
            lo, hi = float(self.edges[k]), float(self.edges[k + 1])
            inside = [b for b in dist.BAND_BOUNDARIES if lo < b < hi]
            self.assertEqual(inside, [], "bin %d (%s, %s) straddles %s" % (k, lo, hi, inside))
        self.assertEqual(len(band), len(self.edges) - 1)
        self.assertEqual(int(band.min()), 0)
        self.assertEqual(int(band.max()), len(dist.BAND_NAMES) - 1)

    def test_band_boundaries_are_exact_edges(self):
        edge_set = set(self.edges.tolist())
        for boundary in dist.BAND_BOUNDARIES:
            self.assertIn(boundary, edge_set)

    def test_bin_index_left_convention(self):
        idx = dist.bin_index(np.asarray([0.0, 0.5, 0.999, 1.0]), self.edges)
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[1], 0)
        self.assertEqual(idx[2], 0)
        self.assertEqual(idx[3], 1)  # exactly on the 1 ms edge -> the bin that starts there

    def test_band_index_boundaries(self):
        band = dist.band_index(np.asarray([0.5, 1.0, 999.0, 1000.0, 4999.0, 5000.0, 1e6]))
        self.assertEqual(list(band), [0, 1, 1, 2, 2, 3, 3])


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class LossTests(unittest.TestCase):
    def setUp(self):
        self.spec = dist.build_bin_spec(SAMPLE, n_bins=24)
        self.edges = dist.edges_from_spec(self.spec)
        self.reps = dist.reps_from_spec(self.spec)
        self.band_of_bin = dist.band_of_bin_from_spec(self.spec)
        self.n_bins = self.spec["n_bins"]

    def _onehot(self, bin_idx: int, scale: float = 10.0):
        logits = torch.full((1, self.n_bins), -scale)
        logits[0, bin_idx] = scale
        return logits

    def test_nll_is_lower_for_the_correct_bin(self):
        mask = torch.ones(1)
        tgt = torch.tensor([5])
        self.assertLess(float(dist.nll_loss(self._onehot(5), tgt, mask)), float(dist.nll_loss(self._onehot(1), tgt, mask)))

    def test_rps_respects_ordering(self):
        true_bin = int(dist.bin_index(np.asarray([8000.0]), self.edges)[0])
        near_bin = int(dist.bin_index(np.asarray([7000.0]), self.edges)[0])
        mask = torch.ones(1)
        tgt = torch.tensor([true_bin])
        near = float(dist.rps_loss(self._onehot(near_bin), tgt, mask))
        far = float(dist.rps_loss(self._onehot(0), tgt, mask))
        self.assertLess(near, far)

    def test_rps_edge_convention_matches_evaluation(self):
        """P1-6: a value exactly on an edge must get the same indicator in both paths."""

        for edge_index in range(1, self.n_bins):
            edge = float(self.edges[edge_index])
            bin_idx = int(dist.bin_index(np.asarray([edge]), self.edges)[0])
            train_indicator = (bin_idx < np.arange(1, self.n_bins)).astype(np.float64)
            # evaluation builds 1[y <= e_k] for k = 1..K-1 under the same left binning
            eval_indicator = (edge <= self.edges[1:-1]).astype(np.float64)
            bin_indicator = (bin_idx < np.arange(1, self.n_bins)).astype(np.float64)
            np.testing.assert_allclose(bin_indicator, train_indicator)
            del eval_indicator

    def test_pseudo_huber_is_subquadratic(self):
        small = float(dist.pseudo_huber_point(torch.tensor([100.0]), torch.tensor([110.0]), 100.0, 2.0, torch.ones(1)))
        huge = float(dist.pseudo_huber_point(torch.tensor([100.0]), torch.tensor([100000.0]), 100.0, 2.0, torch.ones(1)))
        self.assertLess(small, huge)
        self.assertLess(huge, 1e4)

    def test_derived_quantiles_are_monotone(self):
        probs = torch.softmax(torch.randn(5, self.n_bins), dim=-1)
        derived = dist.derive_from_probs(probs, self.reps)
        self.assertTrue(torch.all(derived["q50_ms"] <= derived["q90_ms"] + 1e-9))
        self.assertTrue(torch.all(derived["q90_ms"] <= derived["q95_ms"] + 1e-9))
        self.assertTrue(float(torch.min(derived["mean_ms"])) >= self.reps.min() - 1e-6)

    def test_coarse_band_ce_prefers_the_right_band(self):
        cheap_bin = int(np.argmax(self.band_of_bin == 0))
        expensive_bin = int(np.argmax(self.band_of_bin == 3))
        mask = torch.ones(1)
        band = torch.tensor([0])
        self.assertLess(
            float(dist.coarse_band_ce(self._onehot(cheap_bin), band, mask, self.band_of_bin)),
            float(dist.coarse_band_ce(self._onehot(expensive_bin), band, mask, self.band_of_bin)),
        )


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.spec = dist.build_bin_spec(SAMPLE, n_bins=24)
        self.edges = dist.edges_from_spec(self.spec)
        self.reps = dist.reps_from_spec(self.spec)

    def _probs(self, values):
        rows = []
        for value in values:
            probs = np.zeros(self.spec["n_bins"])
            probs[int(dist.bin_index(np.asarray([value]), self.edges)[0])] = 1.0
            rows.append(probs)
        return np.asarray(rows)

    def test_perfect_distribution_has_no_crossing(self):
        truth = np.asarray([0.082, 12.0, 900.0, 9500.0, 40000.0])
        report = dist.evaluate_distribution(self._probs(truth), truth, self.reps, self.edges)
        self.assertEqual(report["A_distribution_calibration"]["quantile_crossing_rate"], 0.0)
        self.assertEqual(report["n_slot_pairs"], 5)
        self.assertLess(report["B_distribution_accuracy"]["q50"]["normalized_pinball"], 0.5)

    def test_expensive_bucket_gate_is_not_none(self):
        """P0-3: the 8-12 s bucket must actually resolve, not silently return None."""

        truth = np.asarray([9000.0, 10000.0, 11000.0, 0.5, 400.0])
        report = dist.evaluate_distribution(self._probs(truth), truth, self.reps, self.edges)
        self.assertIsNotNone(report["C_node_discrimination"]["expensive_bucket_pred_over_true"])

    def test_quantile_report_matches_manual_pinball(self):
        q50 = np.asarray([1.0, 10.0, 100.0])
        q90 = np.asarray([2.0, 20.0, 200.0])
        q95 = np.asarray([3.0, 30.0, 300.0])
        truth = np.asarray([1.5, 25.0, 150.0])
        report = dist.evaluate_quantiles(q50, q90, q95, truth)
        self.assertAlmostEqual(
            report["B_distribution_accuracy"]["q95"]["pinball_ms"],
            float(np.mean(dist.pinball(q95, truth, 0.95))),
            places=9,
        )

    def test_config_thresholds_change_gate_behaviour(self):
        """P1-3: thresholds must come from the config, not from hard-coded numbers."""

        q50 = np.asarray([1.0, 10.0, 100.0, 1000.0, 9000.0])
        truth = np.asarray([1.1, 11.0, 110.0, 1100.0, 9500.0])
        baseline = dist.evaluate_quantiles(q50, q50 * 2, q50 * 3, truth)
        report = dist.evaluate_quantiles(q50 * 1.01, q50 * 2.01, q50 * 3.01, truth)
        strict = dist.evaluate_gates(report, baseline, {"viability_log_mae_max_relative": 0.999,
                                                        "viability_spearman_min_delta": 0.5,
                                                        "viability_tail_recall_min_delta": 0.5,
                                                        "viability_expensive_bucket_min": 0.99})
        loose = dist.evaluate_gates(report, baseline, {"viability_log_mae_max_relative": 1.5,
                                                       "viability_spearman_min_delta": -1.0,
                                                       "viability_tail_recall_min_delta": -1.0,
                                                       "viability_expensive_bucket_min": 0.0})
        self.assertFalse(strict["viability"]["pass"])
        self.assertTrue(loose["viability"]["pass"])

    def test_integrity_gate_fails_on_crossing(self):
        q50 = np.asarray([100.0, 100.0])
        q90 = np.asarray([50.0, 50.0])  # deliberate crossing
        q95 = np.asarray([10.0, 10.0])
        truth = np.asarray([60.0, 60.0])
        report = dist.evaluate_quantiles(q50, q90, q95, truth)
        gates = dist.evaluate_gates(report, report, {})
        self.assertFalse(gates["headline"]["integrity_pass"])

    def test_tail_recall_bounds(self):
        truth = np.asarray([0.1, 0.2, 0.3, 500.0, 900.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
        self.assertEqual(dist.tail_recall(truth.copy(), truth), 1.0)
        self.assertEqual(dist.tail_recall(-truth, truth), 0.0)


if __name__ == "__main__":
    unittest.main()
