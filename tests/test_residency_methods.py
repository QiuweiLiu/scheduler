"""Unit tests for the round-6 residency belief encoder and action rules."""
from __future__ import annotations

import unittest

from tracing.analysis.residency_methods import (
    choose_eviction_subset,
    choose_prefetch_gpu,
    demand_quality,
    eviction_penalty,
    oracle_demand,
    step_demand,
    totals_from_demand,
)


def row(length_probs=(0.0, 0.0, 0.0, 0.0, 0.0, 1.0), steps=None):
    return {
        "length_probabilities": list(length_probs),
        "future_h5": [{"steps": list(steps or [])}],
    }


def step(load_prob=0.5, load_p95=200.0, models=None):
    return {
        "model_probabilities": dict(models or {"m1": 1.0}),
        "resource": {
            "runtime_mean_ms": 1000.0,
            "load_occurrence_probability": load_prob,
            "load_duration_ms_quantiles": {"p50": 100.0, "p95": load_p95},
        },
    }


class StepDemandTests(unittest.TestCase):
    def test_f0point_is_one_hot_top1_with_hard_load_threshold(self) -> None:
        s = step(load_prob=0.6, load_p95=200.0, models={"m1": 0.7, "m2": 0.3})
        self.assertEqual(step_demand(s, "f0point", 1.0), {"m1": 200.0})

    def test_f0point_is_empty_below_the_load_threshold(self) -> None:
        s = step(load_prob=0.49, models={"m1": 1.0})
        self.assertEqual(step_demand(s, "f0point", 1.0), {})

    def test_pdrs_is_survival_weighted_full_distribution_with_soft_load(self) -> None:
        s = step(load_prob=0.5, load_p95=200.0, models={"m1": 0.7, "m2": 0.3})
        # q=0.25, p_load=0.5, p95=200 -> scale=25 -> {m1: 17.5, m2: 7.5}
        demand = step_demand(s, "pdrs", 0.25)
        self.assertAlmostEqual(demand["m1"], 17.5)
        self.assertAlmostEqual(demand["m2"], 7.5)

    def test_pdrs_survival_uses_the_length_distribution_per_step(self) -> None:
        r = row((0.0, 0.5, 0.5, 0.0, 0.0, 0.0), steps=[step(), step()])
        demand = demand_quality({"n": r}, "n", "pdrs")
        # q1 = 1.0 -> 100; q2 = 0.5 -> 50
        self.assertAlmostEqual(demand[0]["m1"], 100.0)
        self.assertAlmostEqual(demand[1]["m1"], 50.0)

    def test_missing_model_probabilities_fails_closed(self) -> None:
        bad = step()
        del bad["model_probabilities"]
        with self.assertRaisesRegex(ValueError, "model_probabilities"):
            step_demand(bad, "pdrs", 1.0)

    def test_missing_length_probabilities_fails_closed(self) -> None:
        r = {"future_h5": [{"steps": [step()]}]}
        with self.assertRaisesRegex(ValueError, "length_probabilities"):
            demand_quality({"n": r}, "n", "pdrs")

    def test_oracle_uses_true_identity_with_frozen_load_scale(self) -> None:
        r = row(steps=[step(load_prob=0.5, load_p95=200.0, models={"m1": 1.0})])
        demand = oracle_demand({"n": r}, "n", [{"m9": 1.0}])
        self.assertAlmostEqual(demand[0]["m9"], 100.0)
        self.assertNotIn("m1", demand[0])

    def test_oracle_stops_at_the_realized_chain_end(self) -> None:
        r = row(steps=[step(), step()])
        demand = oracle_demand({"n": r}, "n", [{"m9": 1.0}])
        self.assertEqual(demand[1], {})

    def test_totals_split_value_and_next_step_demand(self) -> None:
        demand = [{"m1": 10.0}, {"m1": 5.0, "m2": 7.0}]
        values, nexts = totals_from_demand(demand)
        self.assertEqual(values, {"m1": 15.0, "m2": 7.0})
        self.assertEqual(nexts, {"m1": 10.0})


class EvictionSubsetTests(unittest.TestCase):
    def test_returns_empty_when_no_memory_must_be_freed(self) -> None:
        self.assertEqual(choose_eviction_subset({"a": 100.0}, {"a": 1.0}, 0.0), ())

    def test_picks_the_minimal_value_subset_that_fits(self) -> None:
        subset = choose_eviction_subset(
            {"a": 100.0, "b": 200.0, "c": 300.0},
            {"a": 5.0, "b": 1.0, "c": 9.0},
            250.0,
        )
        self.assertEqual(subset, ("a", "b"))

    def test_ties_prefer_the_smallest_excess_freed_memory(self) -> None:
        subset = choose_eviction_subset(
            {"a": 300.0, "b": 250.0},
            {"a": 1.0, "b": 1.0},
            200.0,
        )
        self.assertEqual(subset, ("b",))

    def test_falls_back_to_all_when_nothing_frees_enough(self) -> None:
        subset = choose_eviction_subset({"a": 10.0, "b": 20.0}, {"a": 1.0, "b": 2.0}, 100.0)
        self.assertEqual(subset, ("a", "b"))

    def test_penalty_sums_planned_values(self) -> None:
        self.assertAlmostEqual(
            eviction_penalty(("a", "c"), {"a": 5.0, "b": 1.0, "c": 9.0}), 14.0
        )


class PrefetchGpuTests(unittest.TestCase):
    def test_picks_max_free_memory_then_gpu_id(self) -> None:
        self.assertEqual(choose_prefetch_gpu({0: 500.0, 1: 700.0}, 400.0), 1)
        self.assertEqual(choose_prefetch_gpu({0: 700.0, 1: 700.0}, 400.0), 0)

    def test_returns_none_when_no_device_fits_without_eviction(self) -> None:
        self.assertIsNone(choose_prefetch_gpu({0: 300.0}, 400.0))
        self.assertIsNone(choose_prefetch_gpu({}, 1.0))


if __name__ == "__main__":
    unittest.main()
