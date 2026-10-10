"""Tests for the round-7 information-ladder prior arm (pdrs_resident_prior).

Covers: build_prior_artifacts(all_steps=True) pooling semantics + default-path
compatibility, policy registration, policy-scoped pack resolution, prefix
independence of the pooled belief, truth-leakage invariance, and an end-to-end
synthetic episode run.
"""
from __future__ import annotations

import copy
import unittest

from tracing.analysis.pdrs_methods import build_prior_artifacts
from tracing.analysis.residency_methods import demand_quality
from tracing.analysis.workload_v02_simulator import (
    RESIDENCY_POLICIES,
    RESIDENCY_PREFETCH_POLICIES,
    Node,
    Template,
    residency_artifacts_for_policy,
    residency_mode_for_policy,
    simulate_episode,
)


def step(models, load_prob=1.0, load_p95=50.0, runtime_mean=100.0):
    return {
        "model_probabilities": dict(models),
        "resource": {
            "runtime_mean_ms": runtime_mean,
            "runtime_ms_quantiles": {"p50": runtime_mean, "p90": runtime_mean, "p95": runtime_mean},
            "load_occurrence_probability": load_prob,
            "load_duration_ms_quantiles": {"p50": load_p95, "p95": load_p95},
        },
    }


def row(steps, length_probs=(0.0, 0.5, 0.5, 0.0, 0.0, 0.0)):
    return {
        "length_probabilities": list(length_probs),
        "future_h5": [{"steps": [step(**kwargs) for kwargs in steps]}],
    }


def two_instance_pack():
    return {
        "n1": row([{"models": {"m1": 0.8, "m2": 0.2}}, {"models": {"m1": 0.1, "m2": 0.9}}],
                  (0.0, 0.6, 0.4, 0.0, 0.0, 0.0)),
        "n2": row([{"models": {"m1": 0.2, "m2": 0.8}}],
                  (0.0, 0.4, 0.6, 0.0, 0.0, 0.0)),
    }


class PriorPackBuildTests(unittest.TestCase):
    def test_default_behavior_replaces_only_first_step(self) -> None:
        pack = two_instance_pack()
        out = build_prior_artifacts(pack)
        # step 0 pooled: (0.8+0.2)/2=0.5 each; step 1 of n1 stays instance-specific
        self.assertAlmostEqual(out["n1"]["future_h5"][0]["steps"][0]["model_probabilities"]["m1"], 0.5)
        step1 = out["n1"]["future_h5"][0]["steps"][1]["model_probabilities"]
        self.assertAlmostEqual(step1["m1"], 0.1)
        self.assertAlmostEqual(step1["m2"], 0.9)
        # length pooled
        self.assertAlmostEqual(out["n1"]["length_probabilities"][1], 0.5)

    def test_all_steps_pools_every_step_marginal(self) -> None:
        pack = two_instance_pack()
        out = build_prior_artifacts(pack, all_steps=True)
        step0 = out["n1"]["future_h5"][0]["steps"][0]["model_probabilities"]
        step1 = out["n1"]["future_h5"][0]["steps"][1]["model_probabilities"]
        self.assertAlmostEqual(step0["m1"], 0.5)
        # step 1 pooled only over rows that carry a step 1 (n1 alone here)
        self.assertAlmostEqual(step1["m1"], 0.1)
        # n2 has a single step -> its step 0 is the pooled step-0 marginal
        step0_n2 = out["n2"]["future_h5"][0]["steps"][0]["model_probabilities"]
        self.assertAlmostEqual(step0_n2["m1"], 0.5)

    def test_all_steps_belief_is_prefix_independent(self) -> None:
        pack = two_instance_pack()
        out = build_prior_artifacts(pack, all_steps=True)
        d1 = demand_quality(out, "n1", "pdrs")
        d2 = demand_quality(out, "n2", "pdrs")
        # n1 and n2 have identical step-0 demand; n2 simply has fewer steps
        self.assertEqual(d1[0], d2[0])

    def test_input_pack_is_not_mutated(self) -> None:
        pack = two_instance_pack()
        before = copy.deepcopy(pack)
        out = build_prior_artifacts(pack, all_steps=True)
        self.assertEqual(pack, before)
        # output steps must be fresh dicts, not aliases of the input
        self.assertIsNot(out["n1"]["future_h5"][0]["steps"][0], pack["n1"]["future_h5"][0]["steps"][0])

    def test_empty_pack_returns_empty(self) -> None:
        self.assertEqual(build_prior_artifacts({}, all_steps=True), {})


class PolicyRegistrationTests(unittest.TestCase):
    def test_registered_in_both_policy_sets(self) -> None:
        self.assertIn("pdrs_resident_prior", RESIDENCY_POLICIES)
        self.assertIn("pdrs_resident_prior", RESIDENCY_PREFETCH_POLICIES)

    def test_mode_is_pdrs_consumption(self) -> None:
        self.assertEqual(residency_mode_for_policy("pdrs_resident_prior"), "pdrs")

    def test_pack_resolution_and_caching(self) -> None:
        pack = two_instance_pack()
        context: dict = {}
        first = residency_artifacts_for_policy("pdrs_resident_prior", pack, context)
        self.assertIs(first, context["_residency_prior_pack"])
        self.assertIsNot(first, pack)
        step0 = first["n1"]["future_h5"][0]["steps"][0]["model_probabilities"]
        self.assertAlmostEqual(step0["m1"], 0.5)
        second = residency_artifacts_for_policy("pdrs_resident_prior", pack, context)
        self.assertIs(first, second)

    def test_preseeded_pack_is_used_verbatim(self) -> None:
        pack = two_instance_pack()
        seeded = {"x": {"length_probabilities": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]}}
        context = {"_residency_prior_pack": seeded}
        self.assertIs(residency_artifacts_for_policy("pdrs_resident_prior", pack, context), seeded)

    def test_other_arms_keep_the_real_pack(self) -> None:
        pack = two_instance_pack()
        context: dict = {}
        self.assertIs(residency_artifacts_for_policy("pdrs_resident", pack, context), pack)
        self.assertIs(residency_artifacts_for_policy("pdrs_resident_oracle", pack, context), pack)

    def test_missing_context_falls_back_to_real_pack(self) -> None:
        pack = two_instance_pack()
        self.assertIs(residency_artifacts_for_policy("pdrs_resident_prior", pack, None), pack)


class TruthLeakageInvarianceTests(unittest.TestCase):
    def test_prior_pack_ignores_truth_like_fields(self) -> None:
        pack = two_instance_pack()
        poisoned = copy.deepcopy(pack)
        for r in poisoned.values():
            r["true_runtime_ms"] = 9.0e9
            for scenario in r["future_h5"]:
                for s in scenario["steps"]:
                    s["observed_intrinsic_ms"] = 9.0e9
                    s["resource"]["measured_slowdown"] = 99.0
        clean_out = build_prior_artifacts(pack, all_steps=True)
        poisoned_out = build_prior_artifacts(poisoned, all_steps=True)
        # Unrelated truth-like fields are carried through by the row copy, but the
        # pooled channels and the consumed belief must be byte-identical.
        for node in ("n1", "n2"):
            self.assertEqual(clean_out[node]["length_probabilities"],
                             poisoned_out[node]["length_probabilities"])
            clean_steps = clean_out[node]["future_h5"][0]["steps"]
            poisoned_steps = poisoned_out[node]["future_h5"][0]["steps"]
            for clean_step, poisoned_step in zip(clean_steps, poisoned_steps):
                self.assertEqual(clean_step["model_probabilities"],
                                 poisoned_step["model_probabilities"])
            self.assertEqual(demand_quality(clean_out, node, "pdrs"),
                             demand_quality(poisoned_out, node, "pdrs"))


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


def fixture():
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
        "episode_id": "info-ladder",
        "split": "train",
        "gpu_topology_mb": [500.0, 500.0],
        "gpu_identity": "synthetic-gpu",
        "jobs": [
            {"job_instance_id": f"j{template_id}", "template_id": template_id,
             "arrival_ms": 0.0, "deadline_ms": 10000.0, "service_class": "normal"}
            for template_id in ("a",)
        ],
    }
    pack = {
        "a:1": row([{"models": {"model-b": 1.0}}], (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
        "a:2": row([{"models": {"model-a": 1.0}}], (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
        "b:1": row([{"models": {"model-a": 1.0}}], (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
        "b:2": row([{"models": {"model-b": 1.0}}], (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
    }
    return episode, templates, stats, pack


class EpisodeRunTests(unittest.TestCase):
    def test_prior_arm_runs_end_to_end_with_context(self) -> None:
        episode, templates, stats, pack = fixture()
        policy_context: dict = {"activation": {}}
        summary, _ = simulate_episode(
            episode, templates, "pdrs_resident_prior",
            train_stats=stats,
            policy_context=policy_context,
            future_artifacts=pack,
            future_horizon=5,
            extension_config={"prefetch_overlap": True},
            collect_events=False,
        )
        self.assertEqual(summary["failed_jobs"], 0)
        self.assertEqual(summary["completed_jobs"], len(episode["jobs"]))
        self.assertIsInstance(summary["mean_completion_ms"], float)

    def test_lazy_build_without_preseeded_pack(self) -> None:
        episode, templates, stats, pack = fixture()
        policy_context: dict = {"activation": {}}
        summary, _ = simulate_episode(
            episode, templates, "pdrs_resident_prior",
            train_stats=stats,
            policy_context=policy_context,
            future_artifacts=pack,
            future_horizon=5,
            extension_config={"prefetch_overlap": True},
            collect_events=False,
        )
        self.assertEqual(summary["failed_jobs"], 0)
        self.assertIn("_residency_prior_pack", policy_context)


if __name__ == "__main__":
    unittest.main()
