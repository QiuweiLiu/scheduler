"""Leakage guards for the round-6 residency action domain.

The residency arms (eviction subset + prefetch) may consume ONLY:

1. the frozen per-instance prediction pack (predicted fields: model
   probabilities, chain-length distribution, load occurrence probability and
   load-duration p95, per-step runtime quantiles);
2. scheduler-visible state (ready/running node states, GPU residency and
   loading sets, free memory);
3. static physical facts (model memory footprint, device capacity).

They must NOT consume execution truth (true runtimes, observed intrinsic
durations, finish times), unrevealed-suffix truth (template successor walks --
reserved for the explicitly-labelled oracle ceiling arm), or the measured
machine tables (colocation slowdowns, prefetch interference, batching).

These tests poison every truth-like and machine-table-like field with absurd
values and require the belief outputs and the end-to-end episode results to be
byte-identical, so a future refactor that starts reading hidden truth fails.
"""
from __future__ import annotations

import copy
import unittest

from tracing.analysis.residency_methods import demand_quality, oracle_demand
from tracing.analysis.workload_v02_simulator import (
    Node,
    Template,
    simulate_episode,
)

POISON = {
    "truth": {"next_models": ["poisoned"], "runtime_ms": 9.0e9},
    "actual_future_models": ["poisoned"],
    "true_runtime_ms": 9.0e9,
    "observed_intrinsic_ms": 9.0e9,
    "finish_ms": 9.0e9,
    "measured_slowdown": 99.0,
    "interference_extra_ms": 9.0e9,
    "machine_table": {"slowdown": 99.0},
}


def poison_row(row):
    """Deep-copy a prediction row and inject truth-like fields everywhere."""

    out = copy.deepcopy(row)
    out.update(POISON)
    for scenario in out.get("future_h5") or []:
        scenario.update(POISON)
        for step in scenario.get("steps") or []:
            step.update(POISON)
            resource = step.get("resource")
            if isinstance(resource, dict):
                resource.update(POISON)
    return out


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


def prediction_row(next_model, load_prob=1.0, load_p95=50.0, runtime_p95=100.0):
    return {
        "length_probabilities": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "future_h5": [{
            "steps": [{
                "model_probabilities": {next_model: 1.0},
                "resource": {
                    "runtime_mean_ms": runtime_p95,
                    "runtime_ms_quantiles": {"p50": runtime_p95, "p90": runtime_p95, "p95": runtime_p95},
                    "load_occurrence_probability": load_prob,
                    "load_duration_ms_quantiles": {"p50": load_p95, "p95": load_p95},
                },
            }],
        }],
    }


def fixture(gpu_topology=(500.0, 500.0), job_templates=("a", "b")):
    """Synthetic residency fixture, parameterized by topology and job mix.

    * ``(500, 500)`` + jobs ``("a",)``: the single job's next-step model is not
      resident anywhere and fits the idle device -> prefetch fires.
    * ``(400,)`` + jobs ``("a", "b")``: the second job's model cannot fit
      without eviction -> eviction fires (the placement cannot avoid it).
    """

    templates = {
        "a": serial_template("a", "model-a", "model-b"),
        "b": serial_template("b", "model-b", "model-a"),
    }
    stats = {}
    for model in ("model-a", "model-b"):
        stats[f"{model}|gpu|0|exact"] = {
            "runtime_p50_ms": 100.0,
            "runtime_p90_ms": 100.0,
            "load_p50_ms": 0.0,
            "memory_p95_mb": 350.0,
            "count": 100,
        }
    episode = {
        "episode_id": "leakage",
        "split": "train",
        "gpu_topology_mb": list(gpu_topology),
        "gpu_identity": "synthetic-gpu",
        "jobs": [
            {"job_instance_id": f"j{template_id}", "template_id": template_id,
             "arrival_ms": 0.0, "deadline_ms": 10000.0, "service_class": "normal"}
            for template_id in job_templates
        ],
    }
    pack = {
        "a:1": prediction_row("model-b"),
        "a:2": prediction_row("model-a"),
        "b:1": prediction_row("model-a"),
        "b:2": prediction_row("model-b"),
    }
    return episode, templates, stats, pack


class BeliefPoisonTests(unittest.TestCase):
    def test_poisoned_truth_fields_do_not_change_f0point_demand(self) -> None:
        clean = {"n": prediction_row("model-b")}
        poisoned = {"n": poison_row(prediction_row("model-b"))}
        self.assertEqual(
            demand_quality(clean, "n", "f0point"),
            demand_quality(poisoned, "n", "f0point"),
        )

    def test_poisoned_truth_fields_do_not_change_pdrs_demand(self) -> None:
        clean = {"n": prediction_row("model-b")}
        poisoned = {"n": poison_row(prediction_row("model-b"))}
        self.assertEqual(
            demand_quality(clean, "n", "pdrs"),
            demand_quality(poisoned, "n", "pdrs"),
        )

    def test_poisoned_truth_fields_do_not_change_oracle_demand(self) -> None:
        clean = {"n": prediction_row("model-b")}
        poisoned = {"n": poison_row(prediction_row("model-b"))}
        true_steps = [{"model-a": 1.0}]
        self.assertEqual(
            oracle_demand(clean, "n", true_steps),
            oracle_demand(poisoned, "n", true_steps),
        )


INVARIANT_FIELDS = ("mean_completion_ms", "p95_completion_ms", "gpu_evictions",
                    "prefetch_count", "cold_loads_on_demand", "evicted_then_reloaded")


class EpisodePoisonTests(unittest.TestCase):
    def _run(self, policy, pack, extension_config,
             gpu_topology=(500.0, 500.0), job_templates=("a",)):
        episode, templates, stats, _ = fixture(gpu_topology, job_templates)
        summary, _events = simulate_episode(
            episode, templates, policy,
            train_stats=stats,
            future_artifacts=pack,
            future_horizon=5,
            extension_config=extension_config,
            collect_events=False,
        )
        return summary

    def _assert_poison_invariant(self, policy, gpu_topology=(500.0, 500.0),
                                 job_templates=("a",)):
        _, _, _, pack = fixture(gpu_topology, job_templates)
        poisoned = {node_id: poison_row(row) for node_id, row in pack.items()}
        extension_config = {"prefetch_overlap": True}
        clean_summary = self._run(policy, pack, extension_config, gpu_topology, job_templates)
        poisoned_summary = self._run(policy, poisoned, extension_config, gpu_topology, job_templates)
        for field_name in INVARIANT_FIELDS:
            self.assertEqual(clean_summary[field_name], poisoned_summary[field_name],
                             f"{field_name} changed under poisoned truth fields")

    def test_prefetch_fixture_exercises_prefetch(self) -> None:
        _, _, _, pack = fixture()
        summary = self._run("pdrs_resident", pack, {"prefetch_overlap": True})
        self.assertGreater(summary["prefetch_count"], 0, "fixture must exercise prefetch")

    def test_single_gpu_fixture_exercises_eviction(self) -> None:
        _, _, _, pack = fixture((400.0,), ("a", "b"))
        summary = self._run("pdrs_resident", pack, {"prefetch_overlap": True},
                            (400.0,), ("a", "b"))
        self.assertGreater(summary["gpu_evictions"], 0, "fixture must exercise eviction")

    def test_poisoned_pack_does_not_change_residency_episode(self) -> None:
        # exercises prefetch (single job, idle second device)
        self._assert_poison_invariant("pdrs_resident")

    def test_poisoned_pack_does_not_change_residency_episode_under_eviction(self) -> None:
        self._assert_poison_invariant("pdrs_resident", gpu_topology=(400.0,),
                                      job_templates=("a", "b"))

    def test_poisoned_pack_does_not_change_f0point_episode(self) -> None:
        self._assert_poison_invariant("f0point_resident")


if __name__ == "__main__":
    unittest.main()
