"""Tests for the matched-consumer arms (exp B): pack-derived Gittins samples,
Hermes-style pack prewarm trigger, Torpor eviction swap, registration, and
leakage invariance of the new decision rules.
"""
from __future__ import annotations

import copy
import unittest

from tracing.analysis.hermes_methods import gittins_index
from tracing.analysis.workload_v02_simulator import (
    RESIDENCY_POLICIES,
    RESIDENCY_PREFETCH_POLICIES,
    Node,
    Template,
    pack_gittins_samples,
    simulate_episode,
)


def step(models, load_prob=1.0, load_p95=50.0, runtime_p95=100.0):
    return {
        "model_probabilities": dict(models),
        "resource": {
            "runtime_mean_ms": runtime_p95,
            "runtime_ms_quantiles": {"p50": runtime_p95, "p90": runtime_p95, "p95": runtime_p95},
            "load_occurrence_probability": load_prob,
            "load_duration_ms_quantiles": {"p50": load_p95, "p95": load_p95},
        },
    }


def row(steps, length_probs=(0.0, 0.5, 0.5, 0.0, 0.0, 0.0)):
    return {
        "length_probabilities": list(length_probs),
        "future_h5": [{"steps": [step(**kwargs) for kwargs in steps]}],
    }


def gpu_node(node_id, model, successors=(), predecessors=()):
    return Node(
        node_id=node_id,
        sequence_index=0,
        predecessors=tuple(predecessors),
        successors=tuple(successors),
        lane="gpu",
        model_id=model,
        runtime_ms=100.0,
        load_ms=0.0,
        workspace_peak_mb=100.0,
        resident_model_mb=200.0,
        status="success",
    )


def serial_template(template_id, first_model, second_model):
    first = gpu_node(f"{template_id}:1", first_model, successors=(f"{template_id}:2",))
    second = gpu_node(f"{template_id}:2", second_model, predecessors=(f"{template_id}:1",))
    return Template(template_id, "video", "train", "test", (first, second),
                    {first.node_id: first, second.node_id: second})


def stats_fixture(load_p50=50.0):
    stats = {}
    for model in ("model-a", "model-b"):
        stats[f"{model}|gpu|0|exact"] = {
            "runtime_p50_ms": 100.0,
            "runtime_p90_ms": 100.0,
            "load_p50_ms": load_p50,
            "memory_p95_mb": 350.0,
            "count": 100,
        }
    return stats


def fixture(gpu_topology=(500.0, 500.0), job_templates=("a",), pack=None):
    templates = {
        "a": serial_template("a", "model-a", "model-b"),
        "b": serial_template("b", "model-b", "model-a"),
    }
    episode = {
        "episode_id": "matched-consumer",
        "split": "train",
        "gpu_topology_mb": list(gpu_topology),
        "gpu_identity": "synthetic-gpu",
        "jobs": [
            {"job_instance_id": f"j{template_id}", "template_id": template_id,
             "arrival_ms": 0.0, "deadline_ms": 10000.0, "service_class": "normal"}
            for template_id in job_templates
        ],
    }
    if pack is None:
        pack = {
            "a:1": row([{"models": {"model-b": 1.0}}], (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
            "a:2": row([{"models": {"model-a": 1.0}}], (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
            "b:1": row([{"models": {"model-a": 1.0}}], (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
            "b:2": row([{"models": {"model-b": 1.0}}], (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
        }
    return episode, templates, episode_stats(), pack


def episode_stats():
    return stats_fixture()


class PackGittinsSamplesTests(unittest.TestCase):
    def test_cumulative_values_and_masses(self) -> None:
        pack = {"n": row([{"models": {"m1": 1.0}}, {"models": {"m1": 1.0}}],
                         (0.0, 0.5, 0.5, 0.0, 0.0, 0.0))}
        samples = pack_gittins_samples(pack, "n", stats_fixture(load_p50=0.0), horizon=5)
        self.assertEqual(len(samples), 200)
        # W1 = 100 (runtime p95) + load surcharge 0 -> 100 ; W2 = 200
        self.assertEqual(sorted(set(samples)), [100.0, 200.0])
        self.assertEqual(samples.count(100.0), 100)
        self.assertEqual(samples.count(200.0), 100)
        self.assertIsNotNone(gittins_index(samples))

    def test_tail_mass_lands_on_horizon_point(self) -> None:
        pack = {"n": row([{"models": {"m1": 1.0}}],
                         (0.0, 1.0, 0.0, 0.0, 0.0, 0.0))}
        samples = pack_gittins_samples(pack, "n", stats_fixture(load_p50=0.0), horizon=5)
        self.assertEqual(set(samples), {100.0})

    def test_empty_and_tail_rows_return_empty(self) -> None:
        self.assertEqual(pack_gittins_samples({}, "n", stats_fixture()), [])
        chain_tail = {"n": {"length_probabilities": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                            "future_h5": [{"steps": []}]}}
        self.assertEqual(pack_gittins_samples(chain_tail, "n", stats_fixture()), [])

    def test_short_steps_clamp_to_last_cumulative(self) -> None:
        pack = {"n": row([{"models": {"m1": 1.0}}],
                         (0.0, 0.25, 0.25, 0.25, 0.25, 0.0))}
        samples = pack_gittins_samples(pack, "n", stats_fixture(load_p50=0.0), horizon=5)
        self.assertEqual(set(samples), {100.0})


class RegistrationTests(unittest.TestCase):
    def test_arms_registered(self) -> None:
        for arm in ("hermes_order_mainline", "hermes_prefetch_mainline", "torpor_evict_mainline"):
            self.assertIn(arm, RESIDENCY_POLICIES)
        self.assertIn("hermes_order_mainline", RESIDENCY_PREFETCH_POLICIES)
        self.assertIn("torpor_evict_mainline", RESIDENCY_PREFETCH_POLICIES)
        self.assertNotIn("hermes_prefetch_mainline", RESIDENCY_PREFETCH_POLICIES)


class HermesPrefetchTriggerTests(unittest.TestCase):
    def _run(self, pack, policy_context=None):
        # Capacity must fit model-b's prefetch beside the running model-a work
        # (the swapped rule prewarms on the RUNNING device, like the faithful arm).
        episode, templates, stats, _ = fixture(gpu_topology=(800.0, 800.0), pack=pack)
        context = policy_context if policy_context is not None else {"activation": {}}
        summary, _ = simulate_episode(
            episode, templates, "hermes_prefetch_mainline",
            train_stats=stats, policy_context=context,
            future_artifacts=pack, future_horizon=5,
            extension_config={"prefetch_overlap": True},
            collect_events=False,
        )
        return summary, context

    def test_high_probability_next_unit_triggers_prewarm(self) -> None:
        pack = {
            "a:1": row([{"models": {"model-b": 1.0}}], (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
            "a:2": row([{"models": {"model-a": 1.0}}], (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
        }
        summary, context = self._run(pack)
        self.assertGreaterEqual(context["activation"].get("hermes_style_prewarm_trigger", 0), 1)
        self.assertGreaterEqual(int(summary.get("prefetch_count") or 0), 1)

    def test_low_probability_next_unit_does_not_trigger(self) -> None:
        pack = {
            "a:1": row([{"models": {"model-a": 0.4, "model-b": 0.35, "model-c": 0.25}}],
                       (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
            "a:2": row([{"models": {"model-a": 1.0}}], (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
        }
        summary, context = self._run(pack)
        self.assertEqual(context["activation"].get("hermes_style_prewarm_trigger", 0), 0)
        self.assertEqual(int(summary.get("prefetch_count") or 0), 0)

    def test_missing_row_fails_closed_to_no_action(self) -> None:
        summary, context = self._run({})
        self.assertEqual(context["activation"].get("hermes_style_prewarm_trigger", 0), 0)
        self.assertEqual(summary["failed_jobs"], 0)

    def test_tight_capacity_blocks_same_device_prewarm_and_is_counted(self) -> None:
        # The swapped rule prewarms on the RUNNING device; when that device lacks
        # room the substrate rejects the load (prefetch_fail capacity) and the
        # summary must expose the rejection count for the mechanism audit.
        pack = {
            "a:1": row([{"models": {"model-b": 1.0}}], (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
            "a:2": row([{"models": {"model-a": 1.0}}], (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
        }
        episode, templates, stats, _ = fixture(gpu_topology=(500.0, 500.0), pack=pack)
        context = {"activation": {}}
        summary, _ = simulate_episode(
            episode, templates, "hermes_prefetch_mainline",
            train_stats=stats, policy_context=context,
            future_artifacts=pack, future_horizon=5,
            extension_config={"prefetch_overlap": True},
            collect_events=False,
        )
        self.assertGreaterEqual(context["activation"].get("hermes_style_prewarm_trigger", 0), 1)
        self.assertEqual(int(summary.get("prefetch_count") or 0), 0)
        self.assertGreaterEqual(int(summary.get("prefetch_capacity_failures") or 0), 1)


class SwapArmRunTests(unittest.TestCase):
    def test_hermes_order_runs_end_to_end(self) -> None:
        episode, templates, stats, pack = fixture()
        context = {"activation": {}}
        summary, _ = simulate_episode(
            episode, templates, "hermes_order_mainline",
            train_stats=stats, policy_context=context,
            future_artifacts=pack, future_horizon=5,
            extension_config={"prefetch_overlap": True},
            collect_events=False,
        )
        self.assertEqual(summary["failed_jobs"], 0)
        self.assertEqual(summary["completed_jobs"], 1)

    def test_torpor_evict_runs_and_evicts_under_pressure(self) -> None:
        episode, templates, stats, pack = fixture(
            gpu_topology=(400.0,), job_templates=("a", "b"))
        context = {"activation": {}}
        summary, _ = simulate_episode(
            episode, templates, "torpor_evict_mainline",
            train_stats=stats, policy_context=context,
            future_artifacts=pack, future_horizon=5,
            extension_config={"prefetch_overlap": True},
            collect_events=False,
        )
        self.assertEqual(summary["failed_jobs"], 0)
        self.assertGreaterEqual(int(summary.get("gpu_evictions") or 0), 1)


class TruthLeakageInvarianceTests(unittest.TestCase):
    def _poison(self, pack):
        poisoned = copy.deepcopy(pack)
        for r in poisoned.values():
            r["true_runtime_ms"] = 9.0e9
            r["finish_ms"] = 9.0e9
            for scenario in r.get("future_h5") or []:
                for s in scenario["steps"]:
                    s["observed_intrinsic_ms"] = 9.0e9
                    s["resource"]["measured_slowdown"] = 99.0
        return poisoned

    def _summary(self, policy, pack):
        episode, templates, stats, _ = fixture(pack=pack)
        summary, _ = simulate_episode(
            episode, templates, policy,
            train_stats=stats, policy_context={"activation": {}},
            future_artifacts=pack, future_horizon=5,
            extension_config={"prefetch_overlap": True},
            collect_events=False,
        )
        return summary

    def test_poisoned_truth_fields_do_not_change_swap_arms(self) -> None:
        pack = {
            "a:1": row([{"models": {"model-b": 1.0}}], (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
            "a:2": row([{"models": {"model-a": 1.0}}], (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
            "b:1": row([{"models": {"model-a": 1.0}}], (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
            "b:2": row([{"models": {"model-b": 1.0}}], (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
        }
        poisoned = self._poison(pack)
        for policy in ("hermes_order_mainline", "hermes_prefetch_mainline", "torpor_evict_mainline"):
            clean_summary = self._summary(policy, pack)
            poisoned_summary = self._summary(policy, poisoned)
            for field in ("mean_completion_ms", "p95_completion_ms", "gpu_evictions",
                          "prefetch_count", "cold_loads_on_demand", "evicted_then_reloaded"):
                self.assertEqual(clean_summary[field], poisoned_summary[field],
                                 f"{policy}:{field} changed under poisoned truth fields")


if __name__ == "__main__":
    unittest.main()
