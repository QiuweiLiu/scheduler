import unittest

import numpy as np

from scripts.p9f_predictor_acceptance_audit import (
    classification_metrics,
    cost_error_metrics,
    fit_temperature,
    future_node_count_bin,
    profile_lookup,
)


class PredictorAcceptanceAuditTests(unittest.TestCase):
    def test_classification_metrics_are_finite(self):
        logits = np.asarray([[4.0, 0.0], [0.0, 4.0], [2.0, 1.0]])
        targets = np.asarray([0, 1, 1])
        metrics = classification_metrics(logits, targets)
        self.assertEqual(metrics["sample_count"], 3)
        self.assertTrue(np.isfinite(metrics["nll"]))
        self.assertTrue(np.isfinite(metrics["brier"]))
        self.assertTrue(np.isfinite(metrics["ece"]))

    def test_temperature_grid_returns_valid_candidate(self):
        logits = np.asarray([[8.0, 0.0], [0.0, 8.0], [8.0, 0.0], [0.0, 8.0]])
        targets = np.asarray([0, 1, 1, 0])
        result = fit_temperature(logits, targets, [0.5, 1.0, 2.0])
        self.assertIn(result["temperature"], {0.5, 1.0, 2.0})
        self.assertEqual(len(result["grid"]), 3)

    def test_future_node_bins_and_profile_fallback(self):
        self.assertEqual(future_node_count_bin(0), "0")
        self.assertEqual(future_node_count_bin(2), "1-2")
        self.assertEqual(future_node_count_bin(5), "3-5")
        self.assertEqual(future_node_count_bin(6), "6+")
        profiles = {
            "execute|select_frames": {"p50_ms": 1.0},
            "execute|*": {"p50_ms": 2.0},
            "*|*": {"p50_ms": 3.0},
        }
        self.assertEqual(profile_lookup(profiles, "execute", "select_frames"), (1.0, "role_family"))
        self.assertEqual(profile_lookup(profiles, "execute", "detect"), (2.0, "role"))
        self.assertEqual(profile_lookup(profiles, "plan", "other"), (3.0, "global"))

    def test_cost_error_metrics(self):
        metrics = cost_error_metrics([(10.0, 8.0), (20.0, 25.0)])
        self.assertEqual(metrics["sample_count"], 2)
        self.assertEqual(metrics["mae_ms"], 3.5)
        self.assertEqual(metrics["bias_ms"], 1.5)
        self.assertEqual(metrics["underprediction_rate"], 0.5)

        count_metrics = cost_error_metrics([(2.0, 1.0), (4.0, 5.0)], unit="count")
        self.assertEqual(count_metrics["mae_count"], 1.0)
        self.assertEqual(count_metrics["bias_count"], 0.0)
        self.assertNotIn("mae_ms", count_metrics)


if __name__ == "__main__":
    unittest.main()
