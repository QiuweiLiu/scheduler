"""Unit tests for the PDRS consumers (ordering + placement), all table-free."""
from __future__ import annotations

import unittest

from tracing.analysis.pdrs_methods import (
    next_model_distribution,
    placement_affinity,
    suffix_expected_cost,
    survival_weights,
)


def row(length_probs=(0.0, 0.0, 0.0, 0.0, 0.0, 1.0), steps=None):
    return {
        "length_probabilities": list(length_probs),
        "future_h5": [{"steps": list(steps or [])}],
    }


def step(runtime_mean=1000.0, load_prob=0.5, load_p95=200.0, models=None):
    return {
        "model_probabilities": dict(models or {"m1": 1.0}),
        "resource": {
            "runtime_mean_ms": runtime_mean,
            "load_occurrence_probability": load_prob,
            "load_duration_ms_quantiles": {"p50": 100.0, "p95": load_p95},
        },
    }


class PdrsConsumerTests(unittest.TestCase):
    def test_survival_weights_follow_the_length_distribution(self) -> None:
        # P(L=1)=0.25, P(L=2)=0.25, P(L=3)=0.5 -> P(L>=1)=1.0, P(L>=2)=0.75, P(L>=3)=0.5
        r = row((0.0, 0.25, 0.25, 0.5, 0.0, 0.0))
        self.assertEqual(survival_weights(r), [1.0, 0.75, 0.5, 0.0, 0.0])

    def test_suffix_cost_weights_each_step_by_survival(self) -> None:
        r = row((0.0, 1.0, 0.0, 0.0, 0.0, 0.0), steps=[step(1000.0, 0.5, 200.0)])
        # q1 = 1.0 -> 1000 + 0.5*200
        self.assertAlmostEqual(suffix_expected_cost({"n": r}, "n"), 1100.0)

    def test_suffix_cost_is_zero_without_future_steps(self) -> None:
        r = row(steps=[])
        self.assertEqual(suffix_expected_cost({"n": r}, "n"), 0.0)

    def test_placement_affinity_is_zero_when_the_next_model_is_resident(self) -> None:
        r = row(steps=[step(models={"m1": 0.7, "m2": 0.3})])
        self.assertEqual(placement_affinity({"n": r}, "n", {"m1", "m2"}), 0.0)

    def test_placement_affinity_scales_with_the_missing_probability_mass(self) -> None:
        r = row(steps=[step(load_prob=0.5, load_p95=200.0, models={"m1": 0.7, "m2": 0.3})])
        # q1=1.0, p_load=0.5, p95=200, hit=0.7 -> 100 * 0.3
        self.assertAlmostEqual(placement_affinity({"n": r}, "n", {"m1"}), 30.0)

    def test_next_model_distribution_reads_the_first_future_step(self) -> None:
        r = row(steps=[step(models={"m1": 0.6, "m2": 0.4})])
        self.assertEqual(next_model_distribution({"n": r}, "n"), {"m1": 0.6, "m2": 0.4})

    def test_missing_runtime_mean_fails_closed(self) -> None:
        bad = step()
        del bad["resource"]["runtime_mean_ms"]
        r = row(steps=[bad])
        with self.assertRaisesRegex(ValueError, "runtime_mean_ms"):
            suffix_expected_cost({"n": r}, "n")

    def test_missing_model_probabilities_fails_closed(self) -> None:
        bad = step()
        del bad["model_probabilities"]
        r = row(steps=[bad])
        with self.assertRaisesRegex(ValueError, "model_probabilities"):
            placement_affinity({"n": r}, "n", set())

    def test_missing_length_probabilities_fails_closed(self) -> None:
        r = {"future_h5": [{"steps": [step()]}]}
        with self.assertRaisesRegex(ValueError, "length_probabilities"):
            suffix_expected_cost({"n": r}, "n")


if __name__ == "__main__":
    unittest.main()
