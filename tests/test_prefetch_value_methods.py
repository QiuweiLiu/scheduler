"""Tests for exp G prefetch value methods (analytic net benefit + rollout)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracing.analysis.prefetch_value_methods import (  # noqa: E402
    FEATURE_NAMES,
    DeviceState,
    FrontierItem,
    _step_mean_ms,
    analytic_benefit_table,
    benefit_feature_table,
    build_scenarios,
    candidate_feature_vector,
    choose_net_benefit,
    choose_rollout_action,
    distilled_score,
    rollout_cost,
)


def step(models, load_prob=1.0, load_p95=4000.0, runtime_mean=1000.0):
    return {
        "model_probabilities": dict(models),
        "resource": {
            "runtime_mean_ms": runtime_mean,
            "load_occurrence_probability": load_prob,
            "load_duration_ms_quantiles": {"p50": load_p95, "p90": load_p95, "p95": load_p95},
        },
    }


def row(steps, length_probs=(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)):
    return {"length_probabilities": list(length_probs),
            "future_h5": [{"steps": [step(**kwargs) for kwargs in steps]}]}


def item(key, current_remaining, row_data, current_model="X"):
    return FrontierItem(key=key, row=row_data, current_model=current_model,
                        current_remaining_ms=current_remaining, current_mb=100.0)


class AnalyticTests(unittest.TestCase):
    def test_step_mean_cost(self) -> None:
        self.assertAlmostEqual(_step_mean_ms(step({"m": 1.0}, load_prob=0.5, load_p95=4000.0,
                                                  runtime_mean=1000.0)), 1000.0 + 0.5 * 4000.0)

    def test_benefit_uses_timing_cap_and_probability(self) -> None:
        items = [
            item("n1", 10000.0, row([{"models": {"A": 0.8}}])),
            item("n2", 500.0, row([{"models": {"B": 0.6}}])),
        ]
        table = analytic_benefit_table(items, load_ms_of=lambda m: {"A": 4000.0, "B": 4000.0}[m])
        # A: far use, full load saving: 1.0 * 0.8 * 1.0 * min(4000, 10000) = 3200
        self.assertAlmostEqual(table["A"]["benefit_ms"], 3200.0)
        # B: near use, capped at 500: 0.6 * 500 = 300
        self.assertAlmostEqual(table["B"]["benefit_ms"], 300.0)
        # u1 is the first-step demand with load p95 scale: 0.6 * 1.0 * 4000
        self.assertAlmostEqual(table["B"]["u1_ms"], 2400.0)

    def test_choose_respects_memory_and_interference_costs(self) -> None:
        table = {"A": {"benefit_ms": 3200.0, "u1_ms": 3200.0},
                 "B": {"benefit_ms": 300.0, "u1_ms": 2400.0}}
        chosen, scored = choose_net_benefit(
            table, ["A", "B"], interference_of=lambda m: 0.0,
            memory_mb_of=lambda m: {"A": 17000.0, "B": 7100.0}[m],
            lambda_mem_ms_per_mb=0.0)
        self.assertEqual(chosen, "A")
        self.assertAlmostEqual(scored["A"]["G_ms"], 3200.0)
        # heavy memory price makes every action unprofitable
        chosen2, scored2 = choose_net_benefit(
            table, ["A", "B"], interference_of=lambda m: 0.0,
            memory_mb_of=lambda m: {"A": 17000.0, "B": 7100.0}[m],
            lambda_mem_ms_per_mb=0.2)
        self.assertIsNone(chosen2)
        self.assertLess(scored2["A"]["G_ms"], 0.0)
        # interference alone can flip the winner
        chosen3, _ = choose_net_benefit(
            table, ["A", "B"], interference_of=lambda m: 3300.0 if m == "A" else 0.0,
            memory_mb_of=lambda m: 0.0, lambda_mem_ms_per_mb=0.0)
        self.assertEqual(chosen3, "B")


class ScenarioTests(unittest.TestCase):
    def test_scenarios_deterministic_and_bounded(self) -> None:
        items = [item("n1", 0.0, row(
            [{"models": {"A": 0.6, "B": 0.4}}, {"models": {"C": 1.0}}],
            length_probs=(0.0, 0.5, 0.5, 0.0, 0.0, 0.0)))]
        first = build_scenarios(items, count=8, seed=11)
        second = build_scenarios(items, count=8, seed=11)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertEqual(first[0][0], [("A", 1000.0 + 4000.0)])  # deterministic base: length argmax=1, model argmax=A
        for scenario in first:
            for chain in scenario:
                self.assertLessEqual(len(chain), 5)
                for model, duration in chain:
                    self.assertIn(model, {"A", "B", "C"})
                    self.assertGreater(duration, 0.0)


class RolloutTests(unittest.TestCase):
    def test_prefetch_of_needed_model_lowers_cost(self) -> None:
        items = [item("n1", 1000.0, row([{"models": {"A": 1.0}}]))]
        devices = [DeviceState(capacity_mb=40000.0, resident={}, busy_until=5000.0)]
        chosen, info = choose_rollout_action(
            items, devices, [("A", 0)], load_ms_of=lambda m: 2000.0,
            memory_mb_of=lambda m: 7000.0, now=0.0)
        self.assertEqual(chosen, ("A", 0))
        self.assertGreater(info["improvement_ms"], 0.0)

    def test_useless_prefetch_ties_to_none(self) -> None:
        items = [item("n1", 1000.0, row([{"models": {"A": 1.0}}]))]
        devices = [DeviceState(capacity_mb=40000.0, resident={}, busy_until=5000.0)]
        chosen, info = choose_rollout_action(
            items, devices, [("Z", 0)], load_ms_of=lambda m: 2000.0,
            memory_mb_of=lambda m: 7000.0, now=0.0)
        self.assertIsNone(chosen)

    def test_rollout_cost_components(self) -> None:
        items = [item("n1", 1000.0, row([{"models": {"A": 1.0}}]), current_model="A")]
        devices = [DeviceState(capacity_mb=40000.0, resident={}, busy_until=5000.0)]
        none_cost = rollout_cost(items, [[]], devices, None,
                                 lambda m: 2000.0, lambda m: 7000.0, now=0.0)
        prefetch_cost = rollout_cost(items, [[]], devices, ("A", 0),
                                     lambda m: 2000.0, lambda m: 7000.0, now=0.0)
        self.assertAlmostEqual(none_cost, 8000.0)
        self.assertAlmostEqual(prefetch_cost, 6000.0)


class FeatureTests(unittest.TestCase):
    def test_feature_table_and_vector(self) -> None:
        items = [
            item("n1", 10000.0, row([{"models": {"A": 0.8}}])),
            item("n2", 500.0, row([{"models": {"B": 0.6}}])),
        ]
        table = benefit_feature_table(items, load_ms_of=lambda m: 4000.0)
        self.assertAlmostEqual(table["A"]["benefit_ms"], 3200.0)
        self.assertAlmostEqual(table["A"]["demand_mass"], 0.8)
        self.assertAlmostEqual(table["A"]["first_need_ms"], 10000.0)
        self.assertAlmostEqual(table["B"]["benefit_ms"], 300.0)
        devices = [DeviceState(capacity_mb=40000.0, resident={"X": 5000.0}, busy_until=0.0)]
        features = candidate_feature_vector(
            table["A"], "A", 0, devices, memory_mb_of=lambda m: 7000.0,
            interference_ms=250.0, load_ms_of=lambda m: 4000.0)
        self.assertEqual(tuple(features.keys()), FEATURE_NAMES)
        self.assertAlmostEqual(features["mem_pressure"], (5000.0 + 7000.0) / 40000.0)
        beta = {name: 0.0 for name in FEATURE_NAMES}
        beta["benefit_ms"] = 1.0
        beta["interference_ms"] = -1.0
        self.assertAlmostEqual(distilled_score(features, beta), 3200.0 - 250.0)


class EndToEndTests(unittest.TestCase):
    def test_new_policies_run_end_to_end(self) -> None:
        from tracing.analysis.workload_v02_simulator import Node, Template, simulate_episode

        def gpu_node(node_id, model, successors=(), predecessors=()):
            return Node(node_id=node_id, sequence_index=0, predecessors=tuple(predecessors),
                        successors=tuple(successors), lane="gpu", model_id=model,
                        runtime_ms=100.0, load_ms=0.0, workspace_peak_mb=100.0,
                        resident_model_mb=200.0, status="success")

        first = gpu_node("a:1", "model-a", successors=("a:2",))
        second = gpu_node("a:2", "model-b", predecessors=("a:1",))
        templates = {"a": Template("a", "video", "train", "test", (first, second),
                                   {first.node_id: first, second.node_id: second})}
        stats = {}
        for model in ("model-a", "model-b"):
            stats[f"{model}|gpu|*|model_lane"] = {
                "runtime_p50_ms": 100.0, "runtime_p90_ms": 100.0,
                "load_p50_ms": 50.0, "memory_p95_mb": 350.0, "count": 100,
            }
        episode = {
            "episode_id": "exp-g", "split": "train", "gpu_topology_mb": [20000.0, 20000.0],
            "jobs": [{"job_instance_id": "j0", "template_id": "a", "arrival_ms": 0.0,
                      "deadline_ms": 100000.0, "service_class": "normal"}],
        }
        pack = {
            "a:1": row([{"models": {"model-b": 1.0}}], (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
            "a:2": row([{"models": {"model-a": 1.0}}], (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
        }
        for policy in ("pdrs_resident_g", "pdrs_resident_rollout", "pdrs_resident_da",
                       "pdrs_resident_hb", "pdrs_resident_probe"):
            context = {"activation": {}, "probe_states": [], "rollout_scenarios": 8}
            summary, _ = simulate_episode(
                episode, templates, policy, train_stats=stats, policy_context=context,
                future_artifacts=pack, future_horizon=5,
                extension_config={"prefetch_overlap": True}, collect_events=False)
            self.assertEqual(summary["failed_jobs"], 0, policy)
            self.assertGreaterEqual(context["activation"].get("prefetch_plan_calls", 0), 1, policy)


if __name__ == "__main__":
    unittest.main()
