#!/usr/bin/env python3
from pathlib import Path
from dataclasses import replace
import json
import unittest

from tracing.analysis.workload_v02_simulator import (
    GPU,
    Job,
    Node,
    RequestSplit,
    Template,
    choose_action,
    colocation_slowdowns,
    load_colocation_profile,
    load_measured_engine_profile,
    load_prefetch_interference_profile,
    phase_decomposition,
    plan_gpu_admission,
    request_split_phase,
    simulate_episode,
    validate_colocation_profile,
    validate_prefetch_interference_profile,
    validate_batching_engine_profile,
    validate_request_preemption_profile,
)


def make_node(model: str, resident_mb: float) -> Node:
    return Node(
        node_id="node",
        sequence_index=0,
        predecessors=(),
        successors=(),
        lane="gpu",
        model_id=model,
        runtime_ms=100.0,
        load_ms=0.0,
        workspace_peak_mb=19430.0,
        resident_model_mb=resident_mb,
        status="success",
    )


def coloc_node(node_id: str, model: str, runtime: float, shape: str, workspace: float = 100.0, load: float = 0.0) -> Node:
    return Node(
        node_id=node_id,
        sequence_index=0,
        predecessors=(),
        successors=(),
        lane="gpu",
        model_id=model,
        runtime_ms=runtime,
        load_ms=load,
        workspace_peak_mb=workspace,
        resident_model_mb=100.0,
        status="success",
        workload_shape=shape,
    )


def simple_template(template_id: str, node: Node) -> Template:
    return Template(template_id, "video", "train", "test", (node,), {node.node_id: node})


def coloc_profile(cells):
    return {
        "enabled": True,
        "engine": "hf_substrate",
        "provenance_kind": "synthetic",
        "source_artifact": "synthetic-f1.json",
        "source_experiment": "synthetic-f1",
        "gpu_identity": "synthetic-gpu",
        "probe_source_artifact": "synthetic-probe.json",
        "probe_metadata": {"probe": "synthetic"},
        "cells": cells,
    }


class AdmissionTests(unittest.TestCase):
    def test_phase_profile_exposes_coefficients_without_invented_counts(self) -> None:
        node = make_node("Qwen3-VL-8B-Instruct", 17000.0)
        result = phase_decomposition(
            node,
            {},
            {
                "phase_profile": {
                    "enabled": True,
                    "models": {
                        "Qwen3-VL-8B-Instruct": {
                            "prefill_us_per_token": 190.01,
                            "prefill_intercept_ms": 9.8,
                            "tpot_ms": 28.423,
                            "kv_mb_per_token_measured_corrected": 0.1485,
                            "vision": {
                                "image_tokens_added": 222.0,
                                "prefill_ms_per_image": 53.0,
                                "mem_mb_per_image": 102.8,
                            },
                        }
                    },
                }
            },
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["prefill_us_per_token"], 190.01)
        self.assertEqual(result["vision_image_tokens_per_image"], 222.0)
        self.assertIsNone(result["input_token_count"])
        self.assertIsNone(result["output_token_count"])
        self.assertIsNone(result["image_count"])
        self.assertFalse(result["phase_durations_available"])
        self.assertNotIn("attributed_prefill_ms", result)

    def test_measured_colocation_dispatches_second_task_with_asymmetric_slowdown(self) -> None:
        first = coloc_node("a:n", "model-a", 100.0, "short")
        second = coloc_node("b:n", "model-b", 100.0, "short")
        profile = coloc_profile(
            [{
                "model_a": "model-a", "shape_a": "short",
                "model_b": "model-b", "shape_b": "short",
                "slowdown_a": 2.0, "slowdown_b": 3.0, "feasible": True,
            }]
        )
        stats = {
            "model-a|gpu|0|exact": {"runtime_p50_ms": 100.0, "runtime_p90_ms": 100.0, "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
            "model-b|gpu|0|exact": {"runtime_p50_ms": 100.0, "runtime_p90_ms": 100.0, "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
        }
        episode = {
            "episode_id": "coloc",
            "split": "train",
            "gpu_topology_mb": [1000.0],
            "gpu_identity": "synthetic-gpu",
            "initial_residency_hint": [["model-a", "model-b"]],
            "jobs": [
                {"job_instance_id": "ja", "template_id": "a", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
                {"job_instance_id": "jb", "template_id": "b", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
            ],
        }
        summary, events = simulate_episode(
            episode,
            {"a": simple_template("a", first), "b": simple_template("b", second)},
            "fcfs",
            train_stats=stats,
            extension_config={
                "colocation_profile": profile,
                "transition_profile": {
                    "models": {
                        model: {"cold_load_ms": 999.0, "evict_proxy_ms": 1.0, "checkpoint_supported": False}
                        for model in ("model-a", "model-b")
                    }
                },
            },
            collect_events=True,
        )
        starts = [event for event in events if event.get("event_type") == "node_start"]
        finishes = [event for event in events if event.get("event_type") == "node_finish"]
        self.assertEqual(len(starts), 2)
        self.assertEqual([event["start_ms"] for event in starts], [0.0, 0.0])
        self.assertTrue(starts[1]["colocation_active"])
        self.assertEqual(starts[1]["colocation_slowdown"], 3.0)
        dispatch = [event for event in events if event.get("event_type") == "node_dispatch"]
        self.assertEqual(dispatch[1]["scheduler_state"]["gpus"][0]["active_task_count"], 1)
        self.assertEqual(dispatch[1]["scheduler_state"]["gpus"][0]["allowed_concurrency"], 2)
        self.assertEqual(dispatch[1]["scheduler_state"]["ready_nodes"][0]["dispatchable_gpu_indices"], [0])
        self.assertEqual([event["finish_ms"] for event in finishes], [200.0, 233.333])
        self.assertLessEqual(max(summary["gpu_utilization"]), 1.0)
        self.assertEqual(summary["colocation_policy_semantics"],
                         "colocation_unaware_control_not_faithful_concurrent_baseline")
        self.assertFalse(any(e["event_type"] == "node_wait" and e["time_ms"] == 0.0 for e in events))

        loader = coloc_node("loader:n", "loader-model", 10.0, "short", load=20.0)
        deferred_stats = {**stats, "loader-model|gpu|0|exact": {
            "runtime_p50_ms": 10.0, "runtime_p90_ms": 10.0,
            "load_p50_ms": 20.0, "memory_p95_mb": 100.0, "count": 100,
        }}
        deferred_summary, deferred_events = simulate_episode(
            episode,
            {"a": simple_template("a", first), "b": simple_template("b", second),
             "loader": simple_template("loader", loader)},
            "fcfs", train_stats=deferred_stats,
            extension_config={"colocation_profile": profile, "prefetch_overlap": True,
                              "prefetch_plan": [{"gpu_index": 0, "model_id": "loader-model"}]},
        )
        last_node_end = max(e["finish_ms"] for e in deferred_events if e["event_type"] == "node_finish")
        actual_load = next(e for e in deferred_events if e["event_type"] == "prefetch_end")
        self.assertEqual(actual_load["start_ms"], last_node_end)
        self.assertTrue(any(e["event_type"] == "prefetch_reschedule" for e in deferred_events))
        self.assertAlmostEqual(deferred_summary["gpu_utilization"][0], 1.0)

    def test_stale_dispatchable_set_waits_instead_of_crashing_when_coverage_moved(self) -> None:
        """A busy GPU is dispatchable only for items with measured coverage.

        When the item that made the GPU dispatchable is dispatched to ANOTHER
        device, the surviving items must not be offered the stale busy GPU: the
        dispatch loop has to recompute the set and let them wait.  Before the
        fix this raised "choose_action called without a ready GPU node".
        """
        ta = coloc_node("a:n", "model-a", 1000.0, "short")
        tb = coloc_node("b:n", "model-b", 1000.0, "short")
        tz = coloc_node("z:n", "model-z", 100.0, "short")
        ty = coloc_node("y:n", "model-y", 100.0, "short")
        stats = {
            f"{model}|gpu|0|exact": {
                "runtime_p50_ms": runtime, "runtime_p90_ms": runtime,
                "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100,
            }
            for model, runtime in (("model-a", 1000.0), ("model-b", 1000.0),
                                   ("model-z", 100.0), ("model-y", 100.0))
        }
        # model-z is covered against BOTH busy models; model-y against neither.
        profile = coloc_profile([
            {"model_a": "model-a", "shape_a": "short", "model_b": "model-z",
             "shape_b": "short", "slowdown_a": 1.5, "slowdown_b": 1.5, "feasible": True},
            {"model_a": "model-b", "shape_a": "short", "model_b": "model-z",
             "shape_b": "short", "slowdown_a": 1.5, "slowdown_b": 1.5, "feasible": True},
        ])
        episode = {
            "episode_id": "stale-dispatchable",
            "split": "train",
            "gpu_topology_mb": [1000.0, 1000.0],
            "gpu_identity": "synthetic-gpu",
            "initial_residency_hint": [
                ["model-a", "model-z", "model-y"],
                ["model-b", "model-z", "model-y"],
            ],
            "jobs": [
                {"job_instance_id": "ja", "template_id": "a", "arrival_ms": 0.0,
                 "deadline_ms": 10000.0, "service_class": "normal"},
                {"job_instance_id": "jb", "template_id": "b", "arrival_ms": 0.0,
                 "deadline_ms": 10000.0, "service_class": "normal"},
                {"job_instance_id": "jz", "template_id": "z", "arrival_ms": 10.0,
                 "deadline_ms": 10000.0, "service_class": "normal"},
                {"job_instance_id": "jy", "template_id": "y", "arrival_ms": 10.0,
                 "deadline_ms": 10000.0, "service_class": "normal"},
            ],
        }
        summary, events = simulate_episode(
            episode,
            {"a": simple_template("a", ta), "b": simple_template("b", tb),
             "z": simple_template("z", tz), "y": simple_template("y", ty)},
            "fcfs",
            train_stats=stats,
            extension_config={"colocation_profile": profile},
            collect_events=True,
        )
        starts = {event["node_id"]: event for event in events if event.get("event_type") == "node_start"}
        # z took the lowest-index busy GPU and co-located; y had no covered busy GPU
        # left and had to wait for a free device instead of crashing.
        self.assertTrue(starts["z:n"]["colocation_active"])
        self.assertEqual(starts["z:n"]["start_ms"], 10.0)
        self.assertGreater(starts["y:n"]["start_ms"], 10.0)
        self.assertEqual(summary["completed_jobs"], 4)

    def test_colocation_memory_counts_shared_model_once_and_protects_different_models(self) -> None:
        shared_a = coloc_node("a:n", "model-a", 100.0, "short", workspace=200.0)
        shared_b = coloc_node("b:n", "model-a", 100.0, "short", workspace=200.0)
        stats = {
            "model-a|gpu|0|exact": {"runtime_p50_ms": 100.0, "runtime_p90_ms": 100.0, "load_p50_ms": 0.0, "memory_p95_mb": 200.0, "count": 100},
        }
        profile = coloc_profile([{
            "model_a": "model-a", "shape_a": "short",
            "model_b": "model-a", "shape_b": "short",
            "slowdown_a": 2.0, "slowdown_b": 2.0, "feasible": True,
        }])
        episode = {
            "episode_id": "shared-memory",
            "split": "train",
            "gpu_topology_mb": [300.0],
            "gpu_identity": "synthetic-gpu",
            "initial_residency_hint": [["model-a"]],
            "jobs": [
                {"job_instance_id": "ja", "template_id": "a", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
                {"job_instance_id": "jb", "template_id": "b", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
            ],
        }
        _summary, events = simulate_episode(
            episode,
            {"a": simple_template("a", shared_a), "b": simple_template("b", shared_b)},
            "fcfs",
            train_stats=stats,
            extension_config={"colocation_profile": profile},
            collect_events=True,
        )
        self.assertEqual(len([event for event in events if event.get("event_type") == "node_start"]), 2)

        different = coloc_node("b:n", "model-b", 100.0, "short", workspace=200.0)
        different_stats = dict(stats)
        different_stats["model-b|gpu|0|exact"] = dict(stats["model-a|gpu|0|exact"])
        different_profile = coloc_profile([{
            "model_a": "model-a", "shape_a": "short",
            "model_b": "model-b", "shape_b": "short",
            "slowdown_a": 2.0, "slowdown_b": 2.0, "feasible": True,
        }])
        different_episode = dict(episode)
        different_episode["episode_id"] = "different-memory"
        _summary, different_events = simulate_episode(
            different_episode,
            {"a": simple_template("a", shared_a), "b": simple_template("b", different)},
            "fcfs",
            train_stats=different_stats,
            extension_config={"colocation_profile": different_profile},
            collect_events=True,
        )
        starts = [event for event in different_events if event.get("event_type") == "node_start"]
        self.assertEqual(len(starts), 2)
        self.assertEqual(starts[0]["node_id"], "a:n")
        self.assertGreaterEqual(starts[1]["start_ms"], 100.0)

    def test_measured_engine_artifacts_are_explicitly_unsupported(self) -> None:
        cases = (
            (
                "experiments/EXP-20260929_vllm_batching_v1/artifacts/v5c_continuous.json",
                "unsupported_missing_request_metadata",
            ),
            (
                "experiments/EXP-20260929_vllm_prefix_cache_v1/artifacts/v6c_prefix_cache.json",
                "unsupported_missing_trace_prefix_identity",
            ),
            (
                "experiments/EXP-20260929_vllm_preemption_v1/artifacts/v7_preemption.json",
                "unsupported_preemption_count_unavailable",
            ),
        )
        for path, status in cases:
            profile = load_measured_engine_profile(Path(path), "vllm")
            self.assertFalse(profile["supports_execution"])
            self.assertEqual(profile["status"], status)
        with self.assertRaises(ValueError):
            load_measured_engine_profile(
                Path("experiments/EXP-20260929_vllm_batching_v1/artifacts/v5c_continuous.json"),
                "hf_substrate",
            )

    def test_pred_mpc_rejects_busy_colocation_without_exact_rollout_support(self) -> None:
        node = coloc_node("candidate", "model-b", 10.0, "short")
        template = simple_template("candidate-template", node)
        job = Job(
            "job",
            template,
            0.0,
            None,
            "normal",
            {"candidate": "ready"},
        )
        gpu = GPU(index=0, capacity_mb=1000.0, resident={"model-a": 100.0, "model-b": 100.0})
        gpu.add_active_task(
            "active", 0.0, 100.0, owner=(0, "active"), model_id="model-a", workload_shape="short"
        )
        profile = coloc_profile([{
            "model_a": "model-a", "shape_a": "short",
            "model_b": "model-b", "shape_b": "short",
            "slowdown_a": 2.0, "slowdown_b": 2.0, "feasible": True,
        }])
        stats = {
            "model-b|gpu|0|exact": {
                "runtime_p50_ms": 10.0, "runtime_p90_ms": 10.0,
                "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100,
            }
        }
        with self.assertRaises(ValueError):
            choose_action(
                "pred_mpc_h3",
                [(0.0, 0.0, 0, "candidate")],
                [job],
                [gpu],
                stats,
                0,
                future_artifacts={},
                extension_config={"colocation_profile": profile},
            )

    def test_f1_loader_preserves_provenance_and_explicit_model_alias(self) -> None:
        profile = load_colocation_profile(
            Path("experiments/EXP-20260929_substrate_colocation_surface_v1/artifacts/colocation_cost_table_v1.json")
        )
        self.assertEqual(profile["engine"], "hf_substrate")
        self.assertEqual(profile["source_experiment"], "EXP-20260929_substrate_colocation_surface_v1")
        self.assertEqual(profile["gpu_identity"], "NVIDIA GeForce RTX 4080 SUPER")
        self.assertIn("Qwen3-4B", profile["probe_metadata"])
        self.assertEqual(profile["model_aliases"]["Qwen3-VL-8B"], "Qwen3-VL-8B-Instruct")
        self.assertEqual(len(profile["cells"]), 45)
        validate_colocation_profile(
            {"colocation_profile": profile},
            {"gpu_model": "NVIDIA GeForce RTX 4080 SUPER"},
        )

    def test_colocation_lookup_reverses_measured_arm_factors(self) -> None:
        profile = coloc_profile([{
            "model_a": "model-a", "shape_a": "medium",
            "model_b": "model-b", "shape_b": "medium",
            "slowdown_a": 2.0, "slowdown_b": 3.0, "feasible": True,
        }])
        active = GPU(index=0, capacity_mb=1000.0)
        task_id = active.add_active_task(
            "b", 0.0, 100.0, model_id="model-b", workload_shape="medium"
        )
        task = active.active_tasks[task_id]
        candidate = coloc_node("a:n", "model-a", 100.0, "medium")
        self.assertEqual(colocation_slowdowns(profile, task, candidate), (3.0, 2.0))

    def test_measured_colocation_rejects_missing_or_mismatched_gpu_identity(self) -> None:
        profile = coloc_profile([{
            "model_a": "model-a", "shape_a": "medium",
            "model_b": "model-b", "shape_b": "medium",
            "slowdown_a": 2.0, "slowdown_b": 2.0, "feasible": True,
        }])
        profile["provenance_kind"] = "measured"
        profile["gpu_identity"] = "RTX-4080-SUPER"
        profile["probe_metadata"] = {"probe": "measured"}
        profile["probe_source_artifact"] = "probe.json"
        with self.assertRaises(ValueError):
            validate_colocation_profile({"colocation_profile": profile}, {"gpu_model": "other-gpu"})

    def test_f4_loader_preserves_copy_and_full_modes(self) -> None:
        profile = load_prefetch_interference_profile(
            Path("experiments/EXP-20260929_substrate_overlap_matrix_v1/artifacts/f4_overlap_matrix.json")
        )
        self.assertEqual(profile["engine"], "hf_substrate")
        self.assertEqual(profile["source_experiment"], "EXP-20260929_substrate_overlap_matrix_v1")
        self.assertEqual(profile["gpu_identity"], "NVIDIA GeForce RTX 4080 SUPER")
        self.assertIn("probe_text_tokens", profile["probe_metadata"])
        self.assertEqual(len(profile["pairs"]), 6)
        self.assertIn("copy_slowdown", profile["pairs"][0])
        self.assertIn("full_slowdown", profile["pairs"][0])
        validate_prefetch_interference_profile(
            {"prefetch_interference": profile, "prefetch_interference_mode": "full"},
            {"gpu_model": "NVIDIA GeForce RTX 4080 SUPER"},
        )
        with self.assertRaises(ValueError):
            validate_prefetch_interference_profile(
                {"prefetch_interference": profile, "prefetch_interference_mode": "copy"},
                {"gpu_model": "NVIDIA GeForce RTX 4080 SUPER"},
            )

    def test_canonical_f1_loader_activates_warm_measured_pair(self) -> None:
        profile = load_colocation_profile(
            Path("experiments/EXP-20260929_substrate_colocation_surface_v1/artifacts/colocation_cost_table_v1.json")
        )
        first = coloc_node("first:n", "Qwen3-4B", 100.0, "short")
        second = coloc_node("second:n", "Qwen2.5-VL-3B-Instruct", 100.0, "short")
        stats = {
            f"{node.model_id}|gpu|0|exact": {
                "runtime_p50_ms": 100.0, "runtime_p90_ms": 100.0,
                "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100,
            }
            for node in (first, second)
        }
        episode = {
            "episode_id": "canonical-f1-warm-smoke",
            "split": "train",
            "gpu_topology_mb": [32760.0],
            "gpu_identity": "NVIDIA GeForce RTX 4080 SUPER",
            "initial_residency_hint": [["Qwen3-4B", "Qwen2.5-VL-3B-Instruct"]],
            "jobs": [
                {"job_instance_id": "j1", "template_id": "first", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
                {"job_instance_id": "j2", "template_id": "second", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
            ],
        }
        summary, events = simulate_episode(
            episode,
            {"first": simple_template("first", first), "second": simple_template("second", second)},
            "fcfs",
            train_stats=stats,
            extension_config={"colocation_profile": profile},
        )
        starts = [event for event in events if event.get("event_type") == "node_start"]
        self.assertEqual(len(starts), 2)
        self.assertTrue(starts[1]["colocation_active"])
        self.assertGreater(starts[1]["colocation_slowdown"], 1.0)
        self.assertEqual(summary["colocation_profile_status"], "enabled")

    def test_prefetch_interference_slows_only_overlapping_inference(self) -> None:
        infer = coloc_node("infer:n", "infer-model", 100.0, "short")
        loader = coloc_node("load:n", "load-model", 10.0, "short", load=20.0)
        templates = {
            "infer": simple_template("infer", infer),
            "load": simple_template("load", loader),
        }
        stats = {
            "infer-model|gpu|0|exact": {"runtime_p50_ms": 100.0, "runtime_p90_ms": 100.0, "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
            "load-model|gpu|0|exact": {"runtime_p50_ms": 10.0, "runtime_p90_ms": 10.0, "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
        }
        interference = {
            "enabled": True,
            "engine": "hf_substrate",
            "provenance_kind": "synthetic",
            "source_artifact": "synthetic-f4.json",
            "source_experiment": "synthetic-f4",
            "gpu_identity": "synthetic-gpu",
            "probe_source_artifact": "synthetic-f4-probe.json",
            "probe_metadata": {"probe": "synthetic"},
            "supported_workload_shape": "short",
            "pairs": [{"infer": "infer-model", "load": "load-model", "copy_slowdown": 2.0, "copy_dilation": 1.0, "full_slowdown": 2.0, "full_dilation": 1.0}],
        }
        episode = {
            "episode_id": "prefetch-overlap",
            "split": "train",
            "gpu_topology_mb": [1000.0],
            "gpu_identity": "synthetic-gpu",
            "initial_residency_hint": [[]],
            "jobs": [{"job_instance_id": "j", "template_id": "infer", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"}],
        }
        config = {
            "prefetch_plan": [{"gpu_index": 0, "model_id": "load-model"}],
            "prefetch_overlap": True,
            "prefetch_interference": interference,
            "prefetch_interference_mode": "copy",
        }
        summary, events = simulate_episode(episode, templates, "fcfs", train_stats=stats, extension_config=config)
        self.assertEqual([event["finish_ms"] for event in events if event.get("event_type") == "node_finish"], [110.0])
        self.assertEqual(summary["mean_completion_ms"], 110.0)
        self.assertEqual([event["time_ms"] for event in events if event.get("event_type") == "node_finish"], [110.0])
        self.assertEqual(summary["prefetch_interference_events"], {"copy": 1})
        self.assertEqual(len([event for event in events if event.get("event_type") == "node_interference_start"]), 1)

        no_overlap_config = dict(config)
        no_overlap_config["prefetch_overlap"] = False
        _summary, no_overlap_events = simulate_episode(
            episode, templates, "fcfs", train_stats=stats, extension_config=no_overlap_config
        )
        self.assertEqual(
            [event["finish_ms"] for event in no_overlap_events if event.get("event_type") == "node_finish"],
            [120.0],
        )
        self.assertEqual(
            [event for event in no_overlap_events if event.get("event_type") == "node_interference_start"],
            [],
        )

        unsupported_profile = dict(interference)
        unsupported_profile["supported_workload_shape"] = "medium"
        unsupported_config = dict(config)
        unsupported_config["prefetch_interference"] = unsupported_profile
        unsupported_summary, unsupported_events = simulate_episode(
            episode, templates, "fcfs", train_stats=stats, extension_config=unsupported_config
        )
        self.assertEqual(unsupported_summary["prefetch_interference_unsupported"], {"unsupported_workload_shape": 1})
        self.assertTrue(
            any(event.get("event_type") == "prefetch_interference_unsupported" for event in unsupported_events)
        )

        missing_pair = dict(interference)
        missing_pair["pairs"] = [{"infer": "other-model", "load": "other-load", "copy_slowdown": 2.0, "copy_dilation": 1.0}]
        missing_pair_config = dict(config)
        missing_pair_config["prefetch_interference"] = missing_pair
        missing_pair_summary, missing_pair_events = simulate_episode(
            episode, templates, "fcfs", train_stats=stats, extension_config=missing_pair_config
        )
        self.assertEqual(missing_pair_summary["prefetch_interference_events"], {})
        self.assertEqual(missing_pair_summary["prefetch_interference_unsupported"], {"missing_measured_pair": 1})
        self.assertTrue(any(event.get("deferred_reason") == "missing_measured_pair" for event in missing_pair_events if event.get("event_type") == "prefetch_start"))

        missing_dilation = dict(interference)
        missing_dilation["pairs"] = [{"infer": "infer-model", "load": "load-model", "copy_slowdown": 2.0}]
        missing_dilation_config = dict(config)
        missing_dilation_config["prefetch_interference"] = missing_dilation
        missing_dilation_summary, missing_dilation_events = simulate_episode(
            episode, templates, "fcfs", train_stats=stats, extension_config=missing_dilation_config
        )
        self.assertEqual(missing_dilation_summary["prefetch_interference_events"], {})
        self.assertEqual(
            missing_dilation_summary["prefetch_interference_unsupported"],
            {"missing_or_invalid_dilation": 1},
        )
        self.assertFalse(
            next(event for event in missing_dilation_events if event.get("event_type") == "prefetch_start")["residency_committed"]
        )
        self.assertTrue(any(event.get("deferred_reason") == "missing_or_invalid_dilation" for event in missing_dilation_events if event.get("event_type") == "prefetch_start"))

    def test_multi_process_profile_rejects_same_model_cells(self) -> None:
        profile = {
            "enabled": True,
            "engine": "hf_substrate",
            "deployment": "multi_process",
            "provenance_kind": "synthetic",
            "source_artifact": "synthetic-mp.json",
            "source_experiment": "synthetic-mp",
            "gpu_identity": "synthetic-gpu",
            "probe_source_artifact": "synthetic-mp-probe.json",
            "probe_metadata": {"probe": "synthetic"},
            "cells": [{
                "model_a": "model-a", "shape_a": "medium",
                "model_b": "model-a", "shape_b": "medium",
                "slowdown_a": 1.02, "slowdown_b": 1.02, "feasible": True,
            }],
        }
        with self.assertRaisesRegex(ValueError, "same-model"):
            validate_colocation_profile(
                {"colocation_profile": profile}, {"gpu_identity": "synthetic-gpu"}
            )

    def test_multi_process_colocation_serializes_same_model_and_admits_cross_model(self) -> None:
        node_a = coloc_node("a:n", "model-a", 100.0, "medium")
        node_b = coloc_node("b:n", "model-b", 100.0, "medium")
        profile = {
            "enabled": True,
            "engine": "hf_substrate",
            "deployment": "multi_process",
            "provenance_kind": "synthetic",
            "source_artifact": "synthetic-mp.json",
            "source_experiment": "synthetic-mp",
            "gpu_identity": "synthetic-gpu",
            "probe_source_artifact": "synthetic-mp-probe.json",
            "probe_metadata": {"probe": "synthetic"},
            "cells": [{
                "model_a": "model-a", "shape_a": "medium",
                "model_b": "model-b", "shape_b": "medium",
                "slowdown_a": 1.24, "slowdown_b": 1.27, "feasible": True,
            }],
        }
        stats = {
            "model-a|gpu|0|exact": {"runtime_p50_ms": 100.0, "runtime_p90_ms": 100.0, "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
            "model-b|gpu|0|exact": {"runtime_p50_ms": 100.0, "runtime_p90_ms": 100.0, "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
        }
        transitions = {
            "models": {
                model: {"cold_load_ms": 999.0, "evict_proxy_ms": 1.0, "checkpoint_supported": False}
                for model in ("model-a", "model-b")
            }
        }
        # Same-model jobs serialize: the second node waits for a free GPU.
        same_episode = {
            "episode_id": "mp-same",
            "split": "train",
            "gpu_topology_mb": [1000.0],
            "gpu_identity": "synthetic-gpu",
            "initial_residency_hint": [["model-a"]],
            "jobs": [
                {"job_instance_id": "ja", "template_id": "a", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
                {"job_instance_id": "jb", "template_id": "a", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
            ],
        }
        same_summary, same_events = simulate_episode(
            same_episode,
            {"a": simple_template("a", node_a)},
            "fcfs",
            train_stats=stats,
            extension_config={"colocation_profile": profile, "transition_profile": transitions},
            collect_events=True,
        )
        same_starts = [event for event in same_events if event.get("event_type") == "node_start"]
        self.assertEqual([event["start_ms"] for event in same_starts], [0.0, 100.0])
        self.assertFalse(any(event.get("colocation_active") for event in same_starts))

        # Cross-model jobs co-locate with the measured multi-process slowdowns.
        cross_episode = {
            "episode_id": "mp-cross",
            "split": "train",
            "gpu_topology_mb": [1000.0],
            "gpu_identity": "synthetic-gpu",
            "initial_residency_hint": [["model-a", "model-b"]],
            "jobs": [
                {"job_instance_id": "ja", "template_id": "a", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
                {"job_instance_id": "jb", "template_id": "b", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
            ],
        }
        cross_summary, cross_events = simulate_episode(
            cross_episode,
            {"a": simple_template("a", node_a), "b": simple_template("b", node_b)},
            "fcfs",
            train_stats=stats,
            extension_config={"colocation_profile": profile, "transition_profile": transitions},
            collect_events=True,
        )
        cross_starts = [event for event in cross_events if event.get("event_type") == "node_start"]
        self.assertEqual([event["start_ms"] for event in cross_starts], [0.0, 0.0])
        self.assertEqual(cross_starts[1]["colocation_slowdown"], 1.27)
        finishes = [event["finish_ms"] for event in cross_events if event.get("event_type") == "node_finish"]
        # 100*1.24 = 124; the 1.27-side survivor speeds up when the first task ends:
        # 124 + (100 - 124/1.27) = 126.362.
        self.assertAlmostEqual(finishes[0], 124.0, places=3)
        self.assertAlmostEqual(finishes[1], 126.362, places=3)

    def test_additive_prefetch_interference_adds_measured_extra_time(self) -> None:
        infer = coloc_node("infer:n", "infer-model", 100.0, "medium")
        loader = coloc_node("load:n", "load-model", 10.0, "medium", load=20.0)
        templates = {
            "infer": simple_template("infer", infer),
            "load": simple_template("load", loader),
        }
        stats = {
            "infer-model|gpu|0|exact": {"runtime_p50_ms": 100.0, "runtime_p90_ms": 100.0, "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
            "load-model|gpu|0|exact": {"runtime_p50_ms": 10.0, "runtime_p90_ms": 10.0, "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
        }
        interference = {
            "enabled": True,
            "engine": "hf_substrate",
            "deployment": "multi_process",
            "interference_model": "additive_extra_ms",
            "provenance_kind": "synthetic",
            "source_artifact": "synthetic-mp-f4.json",
            "source_experiment": "synthetic-mp-f4",
            "gpu_identity": "synthetic-gpu",
            "probe_source_artifact": "synthetic-mp-f4-probe.json",
            "probe_metadata": {"probe": "synthetic"},
            "cells": [
                {"infer_model": "infer-model", "infer_shape": "medium", "load": "load-model", "extra_ms": 150.0}
            ],
        }
        episode = {
            "episode_id": "prefetch-additive",
            "split": "train",
            "gpu_topology_mb": [1000.0],
            "gpu_identity": "synthetic-gpu",
            "initial_residency_hint": [[]],
            "jobs": [{"job_instance_id": "j", "template_id": "infer", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"}],
        }
        config = {
            "prefetch_plan": [{"gpu_index": 0, "model_id": "load-model"}],
            "prefetch_overlap": True,
            "prefetch_interference": interference,
        }
        summary, events = simulate_episode(episode, templates, "fcfs", train_stats=stats, extension_config=config)
        # Additive model: the running task's finish grows by exactly extra_ms.
        self.assertEqual(
            [event["finish_ms"] for event in events if event.get("event_type") == "node_finish"],
            [250.0],
        )
        self.assertEqual(summary["mean_completion_ms"], 250.0)
        self.assertEqual(summary["prefetch_interference_events"], {"additive": 1})
        start_events = [event for event in events if event.get("event_type") == "node_interference_start"]
        self.assertEqual(len(start_events), 1)
        self.assertEqual(start_events[0]["extra_ms"], 150.0)
        self.assertEqual(start_events[0]["interference_model"], "additive_extra_ms")
        # Additive interference is a one-shot charge: nothing is reset when the load ends.
        self.assertFalse(any(event.get("event_type") == "node_interference_end" for event in events))
        # The load side keeps its measured duration (no dilation).
        prefetch_end = next(event for event in events if event.get("event_type") == "prefetch_end")
        self.assertEqual(prefetch_end["load_ms"], 20.0)

    def test_additive_prefetch_interference_fails_closed_outside_measured_cells(self) -> None:
        infer = coloc_node("infer:n", "infer-model", 100.0, "medium")
        loader = coloc_node("load:n", "load-model", 10.0, "medium", load=20.0)
        templates = {
            "infer": simple_template("infer", infer),
            "load": simple_template("load", loader),
        }
        stats = {
            "infer-model|gpu|0|exact": {"runtime_p50_ms": 100.0, "runtime_p90_ms": 100.0, "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
            "load-model|gpu|0|exact": {"runtime_p50_ms": 10.0, "runtime_p90_ms": 10.0, "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
        }
        # The only measured cell covers infer_shape=long, but the running call is
        # medium: the same loaded model spans up to 35x across shapes, so the
        # load must not reuse another cell's median.
        interference = {
            "enabled": True, "engine": "hf_substrate", "deployment": "multi_process",
            "interference_model": "additive_extra_ms", "provenance_kind": "synthetic",
            "source_artifact": "synthetic-mp-f4.json", "source_experiment": "synthetic-mp-f4",
            "gpu_identity": "synthetic-gpu",
            "probe_source_artifact": "synthetic-mp-f4-probe.json", "probe_metadata": {"probe": "synthetic"},
            "cells": [
                {"infer_model": "infer-model", "infer_shape": "long", "load": "load-model", "extra_ms": 150.0}
            ],
        }
        episode = {
            "episode_id": "prefetch-additive-uncovered",
            "split": "train",
            "gpu_topology_mb": [1000.0],
            "gpu_identity": "synthetic-gpu",
            "initial_residency_hint": [[]],
            "jobs": [{"job_instance_id": "j", "template_id": "infer", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"}],
        }
        config = {
            "prefetch_plan": [{"gpu_index": 0, "model_id": "load-model"}],
            "prefetch_overlap": True,
            "prefetch_interference": interference,
        }
        summary, events = simulate_episode(episode, templates, "fcfs", train_stats=stats, extension_config=config)
        self.assertEqual(summary["prefetch_interference_events"], {})
        self.assertEqual(summary["prefetch_interference_unsupported"], {"missing_measured_cell": 1})
        self.assertTrue(
            any(
                event.get("event_type") == "prefetch_start"
                and event.get("deferred_reason") == "missing_measured_cell"
                for event in events
            )
        )
        # The inference is not charged an unmeasured extra time.
        self.assertEqual(
            [event["finish_ms"] for event in events if event.get("event_type") == "node_finish"],
            [100.0],
        )

    def test_batching_engine_admits_homogeneous_pair_at_measured_latency_factor(self) -> None:
        def batch_node(node_id, model, role, runtime):
            return Node(
                node_id=node_id, sequence_index=0, predecessors=(), successors=(),
                lane="gpu", model_id=model, runtime_ms=runtime, load_ms=0.0,
                workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
                workload_shape="medium", role=role,
            )

        node = batch_node("a:n", "model-a", "planner", 100.0)
        stats = {"model-a|gpu|0|exact": {"runtime_p50_ms": 100.0, "runtime_p90_ms": 100.0,
                                         "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100}}
        batching = {
            "enabled": True, "engine": "vllm_batched", "provenance_kind": "synthetic",
            "source_artifact": "synthetic-batch.json", "source_experiment": "synthetic-batch",
            "gpu_identity": "synthetic-gpu",
            "probe_source_artifact": "synthetic-batch-probe.json", "probe_metadata": {"probe": "synthetic"},
            # Per-request latency factor f(2)=1.05 (wall ratio), NOT the throughput speedup.
            "same_model_curves": {"model-a|planner": {"2": 1.05}},
            "homogeneity_tolerance": 1.25,
            "kv_pools_tokens": {"model-a": 100000},
            "layers": {"model-a|planner": {"in_tokens": 100, "out_tokens": 10}},
            "cross_model_policy": "mp_table_proxy",
        }
        episode = {
            "episode_id": "batch-pair",
            "split": "train",
            "gpu_topology_mb": [1000.0],
            "gpu_identity": "synthetic-gpu",
            "initial_residency_hint": [["model-a"]],
            "jobs": [
                {"job_instance_id": "ja", "template_id": "a", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
                {"job_instance_id": "jb", "template_id": "b", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
            ],
        }
        summary, events = simulate_episode(
            episode,
            {"a": simple_template("a", node), "b": simple_template("b", node)},
            "fcfs",
            train_stats=stats,
            extension_config={"batching_engine": batching},
            collect_events=True,
        )
        # Both requests share the measured B=2 batch interval: 100 * 1.05 = 105 ms.
        starts = [event["start_ms"] for event in events if event.get("event_type") == "node_start"]
        finishes = [event["finish_ms"] for event in events if event.get("event_type") == "node_finish"]
        self.assertEqual(starts, [0.0, 0.0])
        self.assertEqual(finishes, [105.0, 105.0])
        self.assertTrue(summary["batching_engine_enabled"])

        # Without the engine the same pair serializes (multi-process rule).
        serial_summary, serial_events = simulate_episode(
            episode,
            {"a": simple_template("a", node), "b": simple_template("b", node)},
            "fcfs",
            train_stats=stats,
            extension_config={},
            collect_events=True,
        )
        serial_starts = [event["start_ms"] for event in serial_events if event.get("event_type") == "node_start"]
        self.assertEqual(serial_starts, [0.0, 100.0])
        self.assertFalse(serial_summary["batching_engine_enabled"])

    def test_batching_engine_serializes_heterogeneous_pair(self) -> None:
        def batch_node(node_id, model, role, runtime, sequence_index=0):
            return Node(
                node_id=node_id, sequence_index=sequence_index, predecessors=(), successors=(),
                lane="gpu", model_id=model, runtime_ms=runtime, load_ms=0.0,
                workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
                workload_shape="medium", role=role,
            )

        short = batch_node("a:n", "model-a", "planner", 100.0)
        long = batch_node("b:n", "model-a", "planner", 200.0, sequence_index=1)
        stats = {
            "model-a|gpu|0|exact": {"runtime_p50_ms": 100.0, "runtime_p90_ms": 200.0,
                                    "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
            "model-a|gpu|1|exact": {"runtime_p50_ms": 200.0, "runtime_p90_ms": 200.0,
                                    "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
        }
        batching = {
            "enabled": True, "engine": "vllm_batched", "provenance_kind": "synthetic",
            "source_artifact": "synthetic-batch.json", "source_experiment": "synthetic-batch",
            "gpu_identity": "synthetic-gpu",
            "probe_source_artifact": "synthetic-batch-probe.json", "probe_metadata": {"probe": "synthetic"},
            "same_model_curves": {"model-a|planner": {"2": 1.05}},
            "homogeneity_tolerance": 1.25,
            "kv_pools_tokens": {"model-a": 100000},
            "layers": {"model-a|planner": {"in_tokens": 100, "out_tokens": 10}},
        }
        episode = {
            "episode_id": "batch-heterogeneous",
            "split": "train",
            "gpu_topology_mb": [1000.0],
            "gpu_identity": "synthetic-gpu",
            "initial_residency_hint": [["model-a"]],
            "jobs": [
                {"job_instance_id": "ja", "template_id": "a", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
                {"job_instance_id": "jb", "template_id": "b", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
            ],
        }
        _summary, events = simulate_episode(
            episode,
            {"a": simple_template("a", short), "b": simple_template("b", long)},
            "fcfs",
            train_stats=stats,
            extension_config={"batching_engine": batching},
            collect_events=True,
        )
        # The B2 probe is homogeneous-only; a pair whose scheduler-visible
        # predictions differ by 2x is unmeasured and must serialize.
        starts = [event["start_ms"] for event in events if event.get("event_type") == "node_start"]
        self.assertEqual(starts, [0.0, 100.0])

    def test_batching_engine_gate_uses_visible_predictions_not_true_runtimes(self) -> None:
        def batch_node(node_id, model, role, runtime):
            return Node(
                node_id=node_id, sequence_index=0, predecessors=(), successors=(),
                lane="gpu", model_id=model, runtime_ms=runtime, load_ms=0.0,
                workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
                workload_shape="medium", role=role,
            )

        # Identical scheduler-visible predictions, different hidden runtimes:
        # the admission must depend on the visible side only, otherwise the
        # dispatchable-GPU view would leak future durations.
        a = batch_node("a:n", "model-a", "planner", 100.0)
        b = batch_node("b:n", "model-a", "planner", 200.0)
        stats = {
            "model-a|gpu|0|exact": {"runtime_p50_ms": 100.0, "runtime_p90_ms": 100.0,
                                    "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
        }
        batching = {
            "enabled": True, "engine": "vllm_batched", "provenance_kind": "synthetic",
            "source_artifact": "synthetic-batch.json", "source_experiment": "synthetic-batch",
            "gpu_identity": "synthetic-gpu",
            "probe_source_artifact": "synthetic-batch-probe.json", "probe_metadata": {"probe": "synthetic"},
            "same_model_curves": {"model-a|planner": {"2": 1.05}},
            "homogeneity_tolerance": 1.25,
            "kv_pools_tokens": {"model-a": 100000},
            "layers": {"model-a|planner": {"in_tokens": 100, "out_tokens": 10}},
        }
        episode = {
            "episode_id": "batch-visible-gate",
            "split": "train",
            "gpu_topology_mb": [1000.0],
            "gpu_identity": "synthetic-gpu",
            "initial_residency_hint": [["model-a"]],
            "jobs": [
                {"job_instance_id": "ja", "template_id": "a", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
                {"job_instance_id": "jb", "template_id": "b", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
            ],
        }
        _summary, events = simulate_episode(
            episode,
            {"a": simple_template("a", a), "b": simple_template("b", b)},
            "fcfs",
            train_stats=stats,
            extension_config={"batching_engine": batching},
            collect_events=True,
        )
        starts = [event["start_ms"] for event in events if event.get("event_type") == "node_start"]
        self.assertEqual(starts, [0.0, 0.0])

    def test_request_phase_does_not_rewind_under_additive_interference(self) -> None:
        gpu = GPU(index=0, capacity_mb=1000.0)
        task_id = gpu.add_active_task(
            "t", 0.0, 200.0,
            model_id="model-a", role="planner", work_ms=200.0, slowdown=1.0,
            request_split=RequestSplit(
                prefill_work_ms=100.0, decode_step_work_ms=2.0,
                tokens_base=0, tokens_total=50, context_base=1000,
            ),
        )
        phase = request_split_phase(gpu, task_id, 150.0)
        self.assertEqual(phase["phase"], "decode")
        self.assertEqual(phase["tokens_done"], 25)
        # Mirror add_node_work(): additive interference enters the stall budget.
        task = gpu.active_tasks[task_id]
        task.remaining_work_ms += 100.0
        task.stall_remaining_ms = float(task.stall_remaining_ms or 0.0) + 100.0
        # Mid-stall: intrinsic progress is paused, never rewound to prefill.
        mid = request_split_phase(gpu, task_id, 160.0)
        self.assertEqual(mid["phase"], "decode")
        self.assertEqual(mid["tokens_done"], 25)
        # After the stall budget is consumed the phase clock resumes.
        later = request_split_phase(gpu, task_id, 260.0)
        self.assertEqual(later["tokens_done"], 30)

    def test_batching_engine_resets_survivor_after_partner_finishes(self) -> None:
        def batch_node(node_id, model, role, runtime):
            return Node(
                node_id=node_id, sequence_index=0, predecessors=(), successors=(),
                lane="gpu", model_id=model, runtime_ms=runtime, load_ms=0.0,
                workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
                workload_shape="medium", role=role,
            )

        a = batch_node("a:n", "model-a", "planner", 100.0)
        b = batch_node("b:n", "model-a", "planner", 110.0)
        stats = {
            "model-a|gpu|0|exact": {"runtime_p50_ms": 105.0, "runtime_p90_ms": 110.0,
                                    "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
        }
        batching = {
            "enabled": True, "engine": "vllm_batched", "provenance_kind": "synthetic",
            "source_artifact": "synthetic-batch.json", "source_experiment": "synthetic-batch",
            "gpu_identity": "synthetic-gpu",
            "probe_source_artifact": "synthetic-batch-probe.json", "probe_metadata": {"probe": "synthetic"},
            "same_model_curves": {"model-a|planner": {"2": 1.05}},
            "homogeneity_tolerance": 1.25,
            "kv_pools_tokens": {"model-a": 100000},
            "layers": {"model-a|planner": {"in_tokens": 100, "out_tokens": 10}},
        }
        episode = {
            "episode_id": "batch-survivor",
            "split": "train",
            "gpu_topology_mb": [1000.0],
            "gpu_identity": "synthetic-gpu",
            "initial_residency_hint": [["model-a"]],
            "jobs": [
                {"job_instance_id": "ja", "template_id": "a", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
                {"job_instance_id": "jb", "template_id": "b", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
            ],
        }
        _summary, events = simulate_episode(
            episode,
            {"a": simple_template("a", a), "b": simple_template("b", b)},
            "fcfs",
            train_stats=stats,
            extension_config={"batching_engine": batching},
            collect_events=True,
        )
        # Batched phase: A consumes 100 work in 105 ms.  A leaves; B has 10 work
        # left and finishes solo at 115 ms (no double counting).
        starts = [event["start_ms"] for event in events if event.get("event_type") == "node_start"]
        finishes = [event["finish_ms"] for event in events if event.get("event_type") == "node_finish"]
        self.assertEqual(starts, [0.0, 0.0])
        self.assertEqual(finishes, [105.0, 115.0])

    def test_batching_engine_kv_pool_cap_refuses_oversized_batch(self) -> None:
        def batch_node(node_id, model, role, runtime):
            return Node(
                node_id=node_id, sequence_index=0, predecessors=(), successors=(),
                lane="gpu", model_id=model, runtime_ms=runtime, load_ms=0.0,
                workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
                workload_shape="medium", role=role,
            )

        node = batch_node("a:n", "model-a", "planner", 100.0)
        stats = {"model-a|gpu|0|exact": {"runtime_p50_ms": 100.0, "runtime_p90_ms": 100.0,
                                         "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100}}
        base = {
            "enabled": True, "engine": "vllm_batched", "provenance_kind": "synthetic",
            "source_artifact": "synthetic-batch.json", "source_experiment": "synthetic-batch",
            "gpu_identity": "synthetic-gpu",
            "probe_source_artifact": "synthetic-batch-probe.json", "probe_metadata": {"probe": "synthetic"},
            "same_model_curves": {"model-a|planner": {"2": 1.05}},
            "homogeneity_tolerance": 1.25,
            "layers": {"model-a|planner": {"in_tokens": 100, "out_tokens": 10}},
        }
        episode = {
            "episode_id": "batch-cap",
            "split": "train",
            "gpu_topology_mb": [1000.0],
            "gpu_identity": "synthetic-gpu",
            "initial_residency_hint": [["model-a"]],
            "jobs": [
                {"job_instance_id": "ja", "template_id": "a", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
                {"job_instance_id": "jb", "template_id": "a", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
            ],
        }
        # context = 110 tokens per request; a 100-token pool cannot hold two requests.
        small_pool = {**base, "kv_pools_tokens": {"model-a": 100}}
        _summary, events = simulate_episode(
            episode, {"a": simple_template("a", node)}, "fcfs",
            train_stats=stats, extension_config={"batching_engine": small_pool}, collect_events=True,
        )
        starts = [event["start_ms"] for event in events if event.get("event_type") == "node_start"]
        self.assertEqual(starts, [0.0, 100.0])

        # Missing KV metadata must fail closed, not fail open: a layer without
        # token priors cannot produce a cap, so the pair serializes.
        missing_layer_tokens = {**base, "kv_pools_tokens": {"model-a": 100000}, "layers": {"model-a|planner": {}}}
        _summary2, events2 = simulate_episode(
            episode, {"a": simple_template("a", node)}, "fcfs",
            train_stats=stats, extension_config={"batching_engine": missing_layer_tokens}, collect_events=True,
        )
        starts2 = [event["start_ms"] for event in events2 if event.get("event_type") == "node_start"]
        self.assertEqual(starts2, [0.0, 100.0])

    def test_batching_engine_mixed_role_pair_serializes(self) -> None:
        def batch_node(node_id, model, role, runtime):
            return Node(
                node_id=node_id, sequence_index=0, predecessors=(), successors=(),
                lane="gpu", model_id=model, runtime_ms=runtime, load_ms=0.0,
                workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
                workload_shape="medium", role=role,
            )

        planner = batch_node("a:n", "model-a", "planner", 100.0)
        answer = batch_node("b:n", "model-a", "answer_generation", 100.0)
        stats = {"model-a|gpu|0|exact": {"runtime_p50_ms": 100.0, "runtime_p90_ms": 100.0,
                                         "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100}}
        batching = {
            "enabled": True, "engine": "vllm_batched", "provenance_kind": "synthetic",
            "source_artifact": "synthetic-batch.json", "source_experiment": "synthetic-batch",
            "gpu_identity": "synthetic-gpu",
            "probe_source_artifact": "synthetic-batch-probe.json", "probe_metadata": {"probe": "synthetic"},
            # Only the homogeneous planner x planner condition was measured.
            "same_model_curves": {"model-a|planner": {"2": 1.05}},
            "homogeneity_tolerance": 1.25,
            "kv_pools_tokens": {"model-a": 100000},
            "layers": {
                "model-a|planner": {"in_tokens": 100, "out_tokens": 10},
                "model-a|answer_generation": {"in_tokens": 100, "out_tokens": 10},
            },
        }
        episode = {
            "episode_id": "batch-mixed-role",
            "split": "train",
            "gpu_topology_mb": [1000.0],
            "gpu_identity": "synthetic-gpu",
            "initial_residency_hint": [["model-a"]],
            "jobs": [
                {"job_instance_id": "ja", "template_id": "a", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
                {"job_instance_id": "jb", "template_id": "b", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
            ],
        }
        _summary, events = simulate_episode(
            episode,
            {"a": simple_template("a", planner), "b": simple_template("b", answer)},
            "fcfs",
            train_stats=stats,
            extension_config={"batching_engine": batching},
            collect_events=True,
        )
        # planner x answer_generation was never measured: serialize.
        starts = [event["start_ms"] for event in events if event.get("event_type") == "node_start"]
        self.assertEqual(starts, [0.0, 100.0])

    def test_batching_engine_validator_rejects_invalid_profiles(self) -> None:
        base = {
            "enabled": True, "engine": "vllm_batched", "provenance_kind": "synthetic",
            "source_artifact": "synthetic-batch.json", "source_experiment": "synthetic-batch",
            "gpu_identity": "synthetic-gpu",
            "same_model_curves": {"model-a|planner": {"2": 1.05}},
            "homogeneity_tolerance": 1.25,
            "kv_pools_tokens": {"model-a": 1000},
            "layers": {"model-a|planner": {"in_tokens": 100, "out_tokens": 10}},
        }
        # A throughput speedup used as a latency factor (< 1) is rejected.
        bad_factor = {**base, "same_model_curves": {"model-a|planner": {"2": 0.5}}}
        with self.assertRaisesRegex(ValueError, "per-request latency"):
            validate_batching_engine_profile({"batching_engine": bad_factor})
        # A curve without a matching token-prior layer is rejected.
        missing_layer = {**base, "layers": {}}
        with self.assertRaisesRegex(ValueError, "token-prior layer"):
            validate_batching_engine_profile({"batching_engine": missing_layer})
        # A curve without a matching KV pool is rejected.
        missing_pool = {**base, "kv_pools_tokens": {}}
        with self.assertRaisesRegex(ValueError, "KV pool"):
            validate_batching_engine_profile({"batching_engine": missing_pool})
        # vllm_batched cannot be combined with a single-process co-location table.
        single_process = {
            "colocation_profile": {
                "enabled": True, "engine": "hf_substrate", "deployment": "single_process",
                "provenance_kind": "synthetic",
                "source_artifact": "synthetic-f1.json", "source_experiment": "synthetic-f1",
                "gpu_identity": "synthetic-gpu",
                "cells": [{"model_a": "model-a", "shape_a": "medium", "model_b": "model-b",
                           "shape_b": "medium", "slowdown_a": 1.1, "slowdown_b": 1.1}],
            },
            "batching_engine": base,
        }
        with self.assertRaisesRegex(ValueError, "multi-process"):
            validate_batching_engine_profile(single_process)

    def test_parrot_appfifo_keeps_the_oldest_application_running(self) -> None:
        def chain_node(node_id, model, runtime, predecessors=()):
            return Node(
                node_id=node_id, sequence_index=0, predecessors=predecessors, successors=(),
                lane="gpu", model_id=model, runtime_ms=runtime, load_ms=0.0,
                workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
                workload_shape="short", role="planner",
            )

        a1 = chain_node("a1", "model-a", 50.0)
        a2 = chain_node("a2", "model-a", 10.0, ("a1",))
        b1 = chain_node("b1", "model-b", 10.0)
        templates = {
            "a": Template("a", "video", "train", "test", (a1, a2), {"a1": a1, "a2": a2}),
            "b": simple_template("b", b1),
        }
        stats = {
            "model-a|gpu|0|exact": {"runtime_p50_ms": 50.0, "runtime_p90_ms": 50.0, "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
            "model-b|gpu|0|exact": {"runtime_p50_ms": 10.0, "runtime_p90_ms": 10.0, "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
        }
        episode = {
            "episode_id": "parrot-appfifo",
            "split": "train",
            "gpu_topology_mb": [1000.0],
            "gpu_identity": "synthetic-gpu",
            "initial_residency_hint": [["model-a", "model-b"]],
            "jobs": [
                {"job_instance_id": "ja", "template_id": "a", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
                {"job_instance_id": "jb", "template_id": "b", "arrival_ms": 10.0, "deadline_ms": 1000.0, "service_class": "normal"},
            ],
        }

        def starts(policy):
            _summary, events = simulate_episode(
                episode, templates, policy, train_stats=stats, collect_events=True
            )
            return [event["node_id"] for event in events if event.get("event_type") == "node_start"]

        # Request-order FCFS serves the longest-waiting node (b1, ready at 10) at
        # t=50; Parrot keeps the OLDEST application (arrival 0) running: a2 first.
        self.assertEqual(starts("fcfs"), ["a1", "b1", "a2"])
        self.assertEqual(starts("parrot_appfifo"), ["a1", "a2", "b1"])

    def test_parrot_ignores_invisible_suffix_runtime(self) -> None:
        def chain_node(node_id, model, runtime, predecessors=()):
            return Node(
                node_id=node_id, sequence_index=0, predecessors=predecessors, successors=(),
                lane="gpu", model_id=model, runtime_ms=runtime, load_ms=0.0,
                workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
                workload_shape="short", role="planner",
            )

        def run(a2_runtime):
            a1 = chain_node("a1", "model-a", 50.0)
            a2 = chain_node("a2", "model-a", a2_runtime, ("a1",))
            b1 = chain_node("b1", "model-b", 10.0)
            templates = {
                "a": Template("a", "video", "train", "test", (a1, a2), {"a1": a1, "a2": a2}),
                "b": simple_template("b", b1),
            }
            stats = {
                "model-a|gpu|0|exact": {"runtime_p50_ms": 50.0, "runtime_p90_ms": 50.0, "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
                "model-b|gpu|0|exact": {"runtime_p50_ms": 10.0, "runtime_p90_ms": 10.0, "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
            }
            episode = {
                "episode_id": "parrot-suffix",
                "split": "train",
                "gpu_topology_mb": [1000.0],
                "gpu_identity": "synthetic-gpu",
                "initial_residency_hint": [["model-a", "model-b"]],
                "jobs": [
                    {"job_instance_id": "ja", "template_id": "a", "arrival_ms": 0.0, "deadline_ms": 5000.0, "service_class": "normal"},
                    {"job_instance_id": "jb", "template_id": "b", "arrival_ms": 10.0, "deadline_ms": 5000.0, "service_class": "normal"},
                ],
            }
            _summary, events = simulate_episode(
                episode, templates, "parrot_appfifo", train_stats=stats, collect_events=True
            )
            return [event["node_id"] for event in events if event.get("event_type") == "node_start"]

        # The unexecuted suffix runtime must not change the decision.
        self.assertEqual(run(10.0), ["a1", "a2", "b1"])
        self.assertEqual(run(500.0), ["a1", "a2", "b1"])

    def test_torpor_lifecycle_is_fcfs_without_runtime_sjf(self) -> None:
        a = Node(node_id="a:n", sequence_index=0, predecessors=(), successors=(),
                 lane="gpu", model_id="model-a", runtime_ms=100.0, load_ms=0.0,
                 workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
                 workload_shape="short", role="planner")
        b = Node(node_id="b:n", sequence_index=0, predecessors=(), successors=(),
                 lane="gpu", model_id="model-b", runtime_ms=10.0, load_ms=0.0,
                 workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
                 workload_shape="short", role="planner")
        stats = {
            "model-a|gpu|0|exact": {"runtime_p50_ms": 100.0, "runtime_p90_ms": 100.0, "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
            "model-b|gpu|0|exact": {"runtime_p50_ms": 10.0, "runtime_p90_ms": 10.0, "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
        }
        episode = {
            "episode_id": "torpor-fcfs",
            "split": "train",
            "gpu_topology_mb": [1000.0],
            "gpu_identity": "synthetic-gpu",
            "initial_residency_hint": [["model-a", "model-b"]],
            "jobs": [
                {"job_instance_id": "ja", "template_id": "a", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
                {"job_instance_id": "jb", "template_id": "b", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
            ],
        }

        def starts(policy):
            _summary, events = simulate_episode(
                episode,
                {"a": simple_template("a", a), "b": simple_template("b", b)},
                policy,
                train_stats=stats,
                collect_events=True,
            )
            return [event["node_id"] for event in events if event.get("event_type") == "node_start"]

        # Torpor has no runtime-SJF: the older request runs first even though b is
        # 10x shorter.  myopic is the runtime-ordering contrast that reorders them.
        self.assertEqual(starts("torpor_lifecycle"), ["a:n", "b:n"])
        self.assertEqual(starts("myopic"), ["b:n", "a:n"])

    def test_torpor_eviction_prefers_lowest_swap_burden(self) -> None:
        a = Node(node_id="a:n", sequence_index=0, predecessors=(), successors=(),
                 lane="gpu", model_id="model-a", runtime_ms=10.0, load_ms=0.0,
                 workspace_peak_mb=500.0, resident_model_mb=500.0, status="success",
                 workload_shape="short", role="planner")
        b = Node(node_id="b:n", sequence_index=0, predecessors=(), successors=(),
                 lane="gpu", model_id="model-b", runtime_ms=10.0, load_ms=0.0,
                 workspace_peak_mb=300.0, resident_model_mb=300.0, status="success",
                 workload_shape="short", role="planner")
        c = Node(node_id="c:n", sequence_index=0, predecessors=(), successors=(),
                 lane="gpu", model_id="model-c", runtime_ms=10.0, load_ms=0.0,
                 workspace_peak_mb=300.0, resident_model_mb=300.0, status="success",
                 workload_shape="short", role="planner")
        # a is expensive to bring back (load 400), b is cheap (load 50).
        stats = {
            "model-a|gpu|0|exact": {"runtime_p50_ms": 10.0, "runtime_p90_ms": 10.0, "load_p50_ms": 400.0, "memory_p95_mb": 500.0, "count": 100},
            "model-b|gpu|0|exact": {"runtime_p50_ms": 10.0, "runtime_p90_ms": 10.0, "load_p50_ms": 50.0, "memory_p95_mb": 300.0, "count": 100},
            "model-c|gpu|0|exact": {"runtime_p50_ms": 10.0, "runtime_p90_ms": 10.0, "load_p50_ms": 60.0, "memory_p95_mb": 300.0, "count": 100},
        }
        episode = {
            "episode_id": "torpor-evict",
            "split": "train",
            "gpu_topology_mb": [1000.0],
            "gpu_identity": "synthetic-gpu",
            "initial_residency_hint": [["model-a", "model-b"]],
            "jobs": [
                {"job_instance_id": "jc", "template_id": "c", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
            ],
        }

        def evicted(policy):
            _summary, events = simulate_episode(
                episode,
                {"a": simple_template("a", a), "b": simple_template("b", b), "c": simple_template("c", c)},
                policy,
                train_stats=stats,
                collect_events=True,
            )
            rows = [event for event in events if event.get("event_type") == "model_evict"]
            return [tuple(event.get("models") or ()) for event in rows]

        # Default engine evicts everything evictable; Torpor evicts only the
        # cheapest-to-restore resident (model-b) that makes the plan fit.
        self.assertEqual(evicted("fcfs"), [("model-a", "model-b")])
        self.assertEqual(evicted("torpor_lifecycle"), [("model-b",)])

    def test_hermes_gittins_prefers_the_lower_index_application(self) -> None:
        from tracing.analysis.hermes_methods import build_pdgraph

        def train_node(node_id, model, role, runtime):
            return Node(
                node_id=node_id, sequence_index=0, predecessors=(), successors=(),
                lane="gpu", model_id=model, runtime_ms=runtime, load_ms=0.0,
                workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
                workload_shape="short", role=role,
            )

        # Train-only bank: model-a/planner = {10, 190} (mean 100, Gittins 20),
        # model-b/planner = {95} (mean 95, Gittins 95).
        train_templates = {
            "ta1": simple_template("ta1", train_node("ta1:n", "model-a", "planner", 10.0)),
            "ta2": simple_template("ta2", train_node("ta2:n", "model-a", "planner", 190.0)),
            "tb1": simple_template("tb1", train_node("tb1:n", "model-b", "planner", 95.0)),
        }
        graph = build_pdgraph(train_templates)

        def eval_node(node_id, model, runtime):
            return Node(
                node_id=node_id, sequence_index=0, predecessors=(), successors=(),
                lane="gpu", model_id=model, runtime_ms=runtime, load_ms=0.0,
                workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
                workload_shape="short", role="planner",
            )

        stats = {
            "model-a|gpu|0|exact": {"runtime_p50_ms": 100.0, "runtime_p90_ms": 100.0, "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
            "model-b|gpu|0|exact": {"runtime_p50_ms": 95.0, "runtime_p90_ms": 95.0, "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
        }
        episode = {
            "episode_id": "hermes-gittins",
            "split": "train",
            "gpu_topology_mb": [1000.0],
            "gpu_identity": "synthetic-gpu",
            "initial_residency_hint": [["model-a", "model-b"]],
            "jobs": [
                {"job_instance_id": "ja", "template_id": "a", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
                {"job_instance_id": "jb", "template_id": "b", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
            ],
        }
        templates = {
            "a": simple_template("a", eval_node("a:n", "model-a", 100.0)),
            "b": simple_template("b", eval_node("b:n", "model-b", 95.0)),
        }

        def starts(policy, context=None):
            _summary, events = simulate_episode(
                episode, templates, policy, train_stats=stats,
                policy_context=context, collect_events=True,
            )
            return [event["node_id"] for event in events if event.get("event_type") == "node_start"]

        # Gittins (lower is better) picks the riskier-but-valuable a; the mean-based
        # myopic ordering picks the shorter b.
        self.assertEqual(starts("hermes_gittins", {"hermes_pdgraph": graph}), ["a:n", "b:n"])
        self.assertEqual(starts("myopic"), ["b:n", "a:n"])

    def test_hermes_pdgraph_conditions_on_the_revealed_prefix(self) -> None:
        from tracing.analysis.hermes_methods import (
            build_pdgraph,
            downstream_demand,
            remaining_samples,
        )

        def node(node_id, model, role, runtime):
            return Node(
                node_id=node_id, sequence_index=0, predecessors=(), successors=(),
                lane="gpu", model_id=model, runtime_ms=runtime, load_ms=0.0,
                workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
                workload_shape="short", role=role,
            )

        a1 = node("a1", "model-a", "planner", 10.0)
        a2 = node("a2", "model-b", "videotool_spatial", 50.0)
        template = Template("t", "video", "train", "test", (a1, a2), {"a1": a1, "a2": a2})
        graph = build_pdgraph({"t": template})

        # Before anything is revealed: full remaining work and the first role.
        self.assertEqual(remaining_samples(graph, "", 0, ""), [60.0])
        p_s, model_id = downstream_demand(graph, "", 0, "")
        self.assertEqual((p_s, model_id), (1.0, "model-a"))
        # After planner completes: the conditional demand changes to the suffix.
        self.assertEqual(remaining_samples(graph, "", 1, "planner"), [50.0])
        p_s2, model_id2 = downstream_demand(graph, "", 1, "planner")
        self.assertEqual((p_s2, model_id2), (1.0, "model-b"))

    def test_hermes_prewarm_triggers_online_with_covered_interference(self) -> None:
        from tracing.analysis.hermes_methods import build_pdgraph

        def node(node_id, model, role, runtime, shape="medium"):
            return Node(
                node_id=node_id, sequence_index=0, predecessors=(), successors=(),
                lane="gpu", model_id=model, runtime_ms=runtime, load_ms=0.0,
                workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
                workload_shape=shape, role=role,
            )

        a1 = node("a1", "model-a", "planner", 200.0)
        a2 = node("a2", "model-b", "videotool_spatial", 50.0)
        chain = Template("chain", "video", "train", "test", (a1, a2), {"a1": a1, "a2": a2})
        graph = build_pdgraph({"chain": chain})
        stats = {
            "model-a|gpu|0|exact": {"runtime_p50_ms": 200.0, "runtime_p90_ms": 200.0, "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
            "model-b|gpu|0|exact": {"runtime_p50_ms": 50.0, "runtime_p90_ms": 50.0, "load_p50_ms": 40.0, "memory_p95_mb": 100.0, "count": 100},
        }
        interference = {
            "enabled": True, "engine": "hf_substrate", "deployment": "multi_process",
            "interference_model": "additive_extra_ms", "provenance_kind": "synthetic",
            "source_artifact": "synthetic-mp-f4.json", "source_experiment": "synthetic-mp-f4",
            "gpu_identity": "synthetic-gpu",
            "probe_source_artifact": "synthetic-mp-f4-probe.json", "probe_metadata": {"probe": "synthetic"},
            "cells": [
                {"infer_model": "model-a", "infer_shape": "medium", "load": "model-b", "extra_ms": 30.0}
            ],
        }
        episode = {
            "episode_id": "hermes-prewarm",
            "split": "train",
            "gpu_topology_mb": [1000.0],
            "gpu_identity": "synthetic-gpu",
            "initial_residency_hint": [["model-a"]],
            "jobs": [
                {"job_instance_id": "j", "template_id": "chain", "arrival_ms": 0.0, "deadline_ms": 5000.0, "service_class": "normal"},
            ],
        }
        config = {"prefetch_overlap": True, "prefetch_interference": interference}
        summary, events = simulate_episode(
            episode, {"chain": chain}, "hermes_gittins", train_stats=stats,
            policy_context={"hermes_pdgraph": graph}, extension_config=config, collect_events=True,
        )
        prewarm = [
            event for event in events
            if event.get("event_type") == "prefetch_start" and event.get("model_id") == "model-b"
        ]
        self.assertEqual(len(prewarm), 1)
        # The prewarm fires while the current node is still executing and its
        # measured overlap charge lands on that node.
        self.assertEqual(prewarm[0]["time_ms"], 0.0)
        interference_events = [event for event in events if event.get("event_type") == "node_interference_start"]
        self.assertEqual(len(interference_events), 1)
        self.assertEqual(interference_events[0]["extra_ms"], 30.0)
        finishes = [event["finish_ms"] for event in events if event.get("event_type") == "node_finish"]
        self.assertEqual(finishes[0], 230.0)

    def test_hermes_prewarm_requires_k_threshold(self) -> None:
        from tracing.analysis.hermes_methods import build_pdgraph

        def node(node_id, model, role, runtime):
            return Node(
                node_id=node_id, sequence_index=0, predecessors=(), successors=(),
                lane="gpu", model_id=model, runtime_ms=runtime, load_ms=0.0,
                workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
                workload_shape="medium", role=role,
            )

        # Three equally likely next roles from (1, planner): top p_s = 1/3 < 0.5.
        train = {}
        for index, role in enumerate(("videotool_spatial", "answer_generation", "other")):
            n1 = node(f"t{index}a", "model-a", "planner", 100.0)
            n2 = node(f"t{index}b", f"model-{role}", role, 50.0)
            train[f"t{index}"] = Template(f"t{index}", "video", "train", "test", (n1, n2), {n1.node_id: n1, n2.node_id: n2})
        graph = build_pdgraph(train)

        a1 = node("a1", "model-a", "planner", 200.0)
        a2 = node("a2", "model-videotool_spatial", "videotool_spatial", 50.0)
        chain = Template("chain", "video", "train", "test", (a1, a2), {"a1": a1, "a2": a2})
        stats = {
            "model-a|gpu|0|exact": {"runtime_p50_ms": 200.0, "runtime_p90_ms": 200.0, "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
            "model-videotool_spatial|gpu|0|exact": {"runtime_p50_ms": 50.0, "runtime_p90_ms": 50.0, "load_p50_ms": 40.0, "memory_p95_mb": 100.0, "count": 100},
        }
        interference = {
            "enabled": True, "engine": "hf_substrate", "deployment": "multi_process",
            "interference_model": "additive_extra_ms", "provenance_kind": "synthetic",
            "source_artifact": "synthetic-mp-f4.json", "source_experiment": "synthetic-mp-f4",
            "gpu_identity": "synthetic-gpu",
            "probe_source_artifact": "synthetic-mp-f4-probe.json", "probe_metadata": {"probe": "synthetic"},
            "cells": [
                {"infer_model": "model-a", "infer_shape": "medium", "load": "model-videotool_spatial", "extra_ms": 30.0}
            ],
        }
        episode = {
            "episode_id": "hermes-prewarm-k",
            "split": "train",
            "gpu_topology_mb": [1000.0],
            "gpu_identity": "synthetic-gpu",
            "initial_residency_hint": [["model-a"]],
            "jobs": [
                {"job_instance_id": "j", "template_id": "chain", "arrival_ms": 0.0, "deadline_ms": 5000.0, "service_class": "normal"},
            ],
        }
        _summary, events = simulate_episode(
            episode, {"chain": chain}, "hermes_gittins", train_stats=stats,
            policy_context={"hermes_pdgraph": graph},
            extension_config={"prefetch_overlap": True, "prefetch_interference": interference},
            collect_events=True,
        )
        # p_s = 1/3 < K = 0.5 -> no prewarm, no interference charge.
        self.assertFalse(any(event.get("event_type") == "prefetch_start" for event in events))
        finishes = [event["finish_ms"] for event in events if event.get("event_type") == "node_finish"]
        self.assertEqual(finishes[0], 200.0)

    def test_qlm_saa_prefers_the_lower_variance_first_slot(self) -> None:
        import random as _random

        from tracing.analysis.qlm_methods import build_duration_bank

        def node(node_id, model, role, runtime):
            return Node(
                node_id=node_id, sequence_index=0, predecessors=(), successors=(),
                lane="gpu", model_id=model, runtime_ms=runtime, load_ms=0.0,
                workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
                workload_shape="short", role=role,
            )

        # Same means (100), different variances: a = {10, 190}, b = {100, 100}.
        train = {
            "ta1": simple_template("ta1", node("ta1:n", "model-a", "planner", 10.0)),
            "ta2": simple_template("ta2", node("ta2:n", "model-a", "planner", 190.0)),
            "tb1": simple_template("tb1", node("tb1:n", "model-b", "planner", 100.0)),
            "tb2": simple_template("tb2", node("tb2:n", "model-b", "planner", 100.0)),
        }
        bank = build_duration_bank(train)

        stats = {
            "model-a|gpu|0|exact": {"runtime_p50_ms": 100.0, "runtime_p90_ms": 190.0, "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
            "model-b|gpu|0|exact": {"runtime_p50_ms": 100.0, "runtime_p90_ms": 100.0, "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
        }
        episode = {
            "episode_id": "qlm-variance",
            "split": "train",
            "gpu_topology_mb": [1000.0],
            "gpu_identity": "synthetic-gpu",
            "initial_residency_hint": [["model-a", "model-b"]],
            "jobs": [
                {"job_instance_id": "ja", "template_id": "a", "arrival_ms": 0.0, "deadline_ms": 150.0, "service_class": "normal"},
                {"job_instance_id": "jb", "template_id": "b", "arrival_ms": 0.0, "deadline_ms": 150.0, "service_class": "normal"},
            ],
        }
        templates = {
            "a": simple_template("a", node("a:n", "model-a", "planner", 100.0)),
            "b": simple_template("b", node("b:n", "model-b", "planner", 100.0)),
        }

        def starts(policy, context=None):
            _summary, events = simulate_episode(
                episode, templates, policy, train_stats=stats,
                policy_context=context, collect_events=True,
            )
            return [event["node_id"] for event in events if event.get("event_type") == "node_start"]

        context = {"qlm_duration_bank": bank, "qlm_rng": _random.Random(7)}
        # The chance-SLO objective prefers serving the low-variance b first (its own
        # completion is safe, and a still has a chance after it); the equal-mean
        # deterministic ordering (myopic) keeps the queue order.
        self.assertEqual(starts("qlm_queue", context), ["b:n", "a:n"])
        self.assertEqual(starts("myopic"), ["a:n", "b:n"])

    def test_causal_contract_marker_blocks_legacy_load(self) -> None:
        """A causal projection (even with a prefixed contract string) must never
        load as legacy: the v041 hole silently produced an edgeless graph."""

        import json as _json
        import tempfile

        from tracing.analysis.workload_v02_simulator import load_templates

        row = {
            "template_id": "t",
            "video_id": "v",
            "split": "train",
            "baseline": "b",
            "topology_contract": "scheduler_projection_of_verified_serial_control_flow_v3_1",
            "nodes": [
                {
                    "node_id": "t:n0", "sequence_index": 0, "execution_lane": "gpu",
                    "model_id": "m", "runtime_ms": 10.0, "resource_applicable": True,
                    "causal_predecessor_node_ids": [],
                },
                {
                    "node_id": "t:n1", "sequence_index": 1, "execution_lane": "gpu",
                    "model_id": "m", "runtime_ms": 10.0, "resource_applicable": True,
                    "causal_predecessor_node_ids": ["t:n0"],
                },
            ],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            handle.write(_json.dumps(row) + "\n")
            path = Path(handle.name)
        try:
            with self.assertRaisesRegex(ValueError, "causal_v3"):
                load_templates(path)
            loaded = load_templates(path, topology_view="causal_v3")
            self.assertEqual(len(loaded["t"].by_id["t:n1"].predecessors), 1)
        finally:
            path.unlink()

    def test_main_table_baselines_ignore_the_invisible_suffix(self) -> None:
        """Leakage gate: the first decision must not move when the unexecuted
        suffix changes (runtime + role + model of the pending second nodes)."""

        import random as _random

        from tracing.analysis.hermes_methods import build_pdgraph
        from tracing.analysis.qlm_methods import build_duration_bank

        def chain_node(node_id, model, role, runtime, predecessors=()):
            return Node(
                node_id=node_id, sequence_index=0, predecessors=predecessors, successors=(),
                lane="gpu", model_id=model, runtime_ms=runtime, load_ms=0.0,
                workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
                workload_shape="short", role=role,
            )

        def world(mutate: bool):
            a1 = chain_node("a1", "model-a", "planner", 100.0)
            b1 = chain_node("b1", "model-b", "planner", 100.0)
            if mutate:
                a2 = chain_node("a2", "model-c", "answer_generation", 900.0, ("a1",))
                b2 = chain_node("b2", "model-c", "answer_generation", 700.0, ("b1",))
            else:
                a2 = chain_node("a2", "model-a", "videotool_spatial", 50.0, ("a1",))
                b2 = chain_node("b2", "model-b", "videotool_spatial", 50.0, ("b1",))
            templates = {
                "a": Template("a", "video", "train", "test", (a1, a2), {"a1": a1, "a2": a2}),
                "b": Template("b", "video", "train", "test", (b1, b2), {"b1": b1, "b2": b2}),
            }
            stats = {
                "model-a|gpu|0|exact": {"runtime_p50_ms": 100.0, "runtime_p90_ms": 100.0, "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
                "model-b|gpu|0|exact": {"runtime_p50_ms": 100.0, "runtime_p90_ms": 100.0, "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
                "model-c|gpu|0|exact": {"runtime_p50_ms": 900.0, "runtime_p90_ms": 900.0, "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
            }
            episode = {
                "episode_id": "leak-suffix",
                "split": "train",
                "gpu_topology_mb": [1000.0],
                "gpu_identity": "synthetic-gpu",
                "initial_residency_hint": [["model-a", "model-b"]],
                "jobs": [
                    {"job_instance_id": "ja", "template_id": "a", "arrival_ms": 0.0, "deadline_ms": 5000.0, "service_class": "normal"},
                    {"job_instance_id": "jb", "template_id": "b", "arrival_ms": 0.0, "deadline_ms": 5000.0, "service_class": "normal"},
                ],
            }
            return episode, templates, stats

        bank = build_duration_bank({
            "ta": simple_template("ta", chain_node("ta:n", "model-a", "planner", 100.0)),
            "tb": simple_template("tb", chain_node("tb:n", "model-b", "planner", 100.0)),
        })
        graph = build_pdgraph({
            "ga": Template("ga", "video", "train", "test", (), {}),
        }) if False else build_pdgraph({
            "ta": simple_template("ta", chain_node("ta:n", "model-a", "planner", 100.0)),
            "tb": simple_template("tb", chain_node("tb:n", "model-b", "planner", 100.0)),
        })
        contexts = {
            "parrot_appfifo": None,
            "qlm_queue": {"qlm_duration_bank": bank, "qlm_rng": _random.Random(7)},
            "hermes_gittins": {"hermes_pdgraph": graph},
            "torpor_lifecycle": None,
        }

        def first_choice(policy, context, mutate):
            episode, templates, stats = world(mutate)
            _summary, events = simulate_episode(
                episode, templates, policy, train_stats=stats,
                policy_context=dict(context) if context else None, collect_events=True,
            )
            starts = [event["node_id"] for event in events if event.get("event_type") == "node_start"]
            return starts[0]

        for policy, context in contexts.items():
            before = first_choice(policy, context, False)
            after = first_choice(policy, context, True)
            self.assertEqual(before, after, msg=f"{policy} moved on an invisible suffix change")

    def test_hermes_prewarm_does_not_read_the_true_runtime(self) -> None:
        """P1-1 regression: the prewarm trigger must use the frozen prediction,
        not the engine's true finish time."""

        from tracing.analysis.hermes_methods import build_pdgraph

        def node(node_id, model, role, runtime):
            return Node(
                node_id=node_id, sequence_index=0, predecessors=(), successors=(),
                lane="gpu", model_id=model, runtime_ms=runtime, load_ms=0.0,
                workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
                workload_shape="medium", role=role,
            )

        a2 = node("a2", "model-b", "videotool_spatial", 50.0)
        graph = build_pdgraph({
            "chain": Template("chain", "video", "train", "test", (), {}),
        }) if False else None
        # Build the graph from a real chain so the (1, planner) state exists.
        template = Template(
            "chain", "video", "train", "test",
            (node("a1", "model-a", "planner", 200.0), a2),
            {"a1": node("a1", "model-a", "planner", 200.0), "a2": a2},
        )
        graph = build_pdgraph({"chain": template})
        stats = {
            # The scheduler-visible prediction is 500 ms regardless of the truth.
            "model-a|gpu|0|exact": {"runtime_p50_ms": 500.0, "runtime_p90_ms": 500.0, "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
            "model-b|gpu|0|exact": {"runtime_p50_ms": 50.0, "runtime_p90_ms": 50.0, "load_p50_ms": 40.0, "memory_p95_mb": 100.0, "count": 100},
        }
        interference = {
            "enabled": True, "engine": "hf_substrate", "deployment": "multi_process",
            "interference_model": "additive_extra_ms", "provenance_kind": "synthetic",
            "source_artifact": "synthetic-mp-f4.json", "source_experiment": "synthetic-mp-f4",
            "gpu_identity": "synthetic-gpu",
            "probe_source_artifact": "synthetic-mp-f4-probe.json", "probe_metadata": {"probe": "synthetic"},
            "cells": [{"infer_model": "model-a", "infer_shape": "medium", "load": "model-b", "extra_ms": 30.0}],
        }

        def prewarm_count(true_runtime: float) -> int:
            a1 = node("a1", "model-a", "planner", true_runtime)
            chain = Template("chain", "video", "train", "test", (a1, a2), {"a1": a1, "a2": a2})
            episode = {
                "episode_id": "hermes-truth",
                "split": "train",
                "gpu_topology_mb": [1000.0],
                "gpu_identity": "synthetic-gpu",
                "initial_residency_hint": [["model-a"]],
                "jobs": [
                    {"job_instance_id": "j", "template_id": "chain", "arrival_ms": 0.0, "deadline_ms": 5000.0, "service_class": "normal"},
                ],
            }
            _summary, events = simulate_episode(
                episode, {"chain": chain}, "hermes_gittins", train_stats=stats,
                policy_context={"hermes_pdgraph": graph},
                extension_config={"prefetch_overlap": True, "prefetch_interference": interference},
                collect_events=True,
            )
            return sum(
                1 for event in events
                if event.get("event_type") == "prefetch_start" and event.get("model_id") == "model-b"
            )

        # Same visible prediction, wildly different true runtimes: same action.
        self.assertEqual(prewarm_count(30.0), 1)
        self.assertEqual(prewarm_count(1000.0), 1)

    def test_qlm_saa_clock_starts_at_the_decision_time(self) -> None:
        """P1-2 regression: waiting time must enter C_i; an absolute deadline of
        1000 at now=900 with a 200 ms sample is already violated."""

        from tracing.analysis.qlm_methods import qlm_saa_choice

        keys = [(0, "a"), (1, "b")]
        # a: 500 ms, deadline 1150 ; b: 600 ms, deadline 300.
        samples = {(0, "a"): [500.0], (1, "b"): [600.0]}
        deadlines = {(0, "a"): 1150.0, (1, "b"): 300.0}
        loads = {(0, "a"): 0.0, (1, "b"): 0.0}
        # now=0: b is already late either way, and serving b first is cheaper, so
        # the SAA picks b.
        self.assertEqual(qlm_saa_choice(keys, samples, deadlines, loads, now_ms=0.0), 1)
        # now=600: serving a first leaves only b violating (1); serving b first
        # pushes BOTH past their deadlines (2) -> the waiting time flips the slot.
        self.assertEqual(qlm_saa_choice(keys, samples, deadlines, loads, now_ms=600.0), 0)

    def test_gittins_index_is_exact_over_the_empirical_support(self) -> None:
        """P1-3 regression: the exact infimum, not the 10-point quantile grid."""

        from tracing.analysis.hermes_methods import gittins_index, gittins_index_bucketed

        # At delta=10: E[min(X,10)]/P(X<=10) = 10/0.5 = 20.
        self.assertAlmostEqual(gittins_index([10.0, 190.0]), 20.0, places=6)
        # The old grid misses it (first candidate delta ~28).
        self.assertNotAlmostEqual(gittins_index_bucketed([10.0, 190.0]), 20.0, places=3)
        self.assertLessEqual(gittins_index([10.0, 190.0]), gittins_index_bucketed([10.0, 190.0]))

    def test_hermes_pdgraph_is_workflow_conditioned_and_refines_on_observation(self) -> None:
        """P1-4 regression: same (count, role) state in two workflows must not
        share a mixture; a correlated state must filter tuples by observation."""

        from tracing.analysis.hermes_methods import (
            build_pdgraph,
            conditional_tuples,
            remaining_samples,
        )

        def node(node_id, model, role, runtime):
            return Node(
                node_id=node_id, sequence_index=0, predecessors=(), successors=(),
                lane="gpu", model_id=model, runtime_ms=runtime, load_ms=0.0,
                workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
                workload_shape="short", role=role,
            )

        def chain(tid, workflow, durations):
            first = node(f"{tid}:n0", "m", "planner", durations[0])
            second = node(f"{tid}:n1", "m", "tool", durations[1])
            return Template(
                tid, "v", "train", "b", (first, second),
                {first.node_id: first, second.node_id: second},
                workflow_type_id=workflow,
            )

        templates = {
            # Workflow A: planner 10 -> tool 100 ; Workflow B: planner 10 -> tool 1000.
            "a1": chain("a1", "wf.A", (10.0, 100.0)),
            "a2": chain("a2", "wf.A", (12.0, 120.0)),
            "b1": chain("b1", "wf.B", (10.0, 1000.0)),
            "b2": chain("b2", "wf.B", (12.0, 1200.0)),
        }
        graph = build_pdgraph(templates)
        a_samples = remaining_samples(graph, "wf.A", 1, "planner")
        b_samples = remaining_samples(graph, "wf.B", 1, "planner")
        self.assertEqual(sorted(a_samples), [100.0, 120.0])
        self.assertEqual(sorted(b_samples), [1000.0, 1200.0])

        # Correlated state (suffix = 2x prefix): an observed prefix filters tuples.
        correlated = {}
        for index in range(6):
            prefix = 10.0 * (index + 1)
            tid = f"c{index}"
            correlated[tid] = chain(tid, "wf.C", (prefix, prefix * 2.0))
        graph_c = build_pdgraph(correlated)
        all_samples = remaining_samples(graph_c, "wf.C", 1, "planner")
        self.assertEqual(len(all_samples), 6)
        refined_samples = remaining_samples(graph_c, "wf.C", 1, "planner", observed_prefix_ms=40.0)
        self.assertEqual(len(refined_samples), 3)  # prefixes 30/40/50 within +/-25%
        selected, refined = conditional_tuples(graph_c, "wf.C", 1, "planner", 40.0)
        self.assertTrue(refined)

        # Uncorrelated state: the Pearson gate blocks refinement.
        uncorrelated = {}
        for index in range(6):
            tid = f"u{index}"
            uncorrelated[tid] = chain(tid, "wf.U", (10.0 * (index + 1), 100.0 if index % 2 == 0 else 10.0))
        graph_u = build_pdgraph(uncorrelated)
        _selected_u, refined_u = conditional_tuples(graph_u, "wf.U", 1, "planner", 40.0)
        self.assertFalse(refined_u)

    def test_torpor_uses_the_canonical_f4_triple_for_coverage(self) -> None:
        """P1-5 regression: coverage is decided by the canonical
        (infer_model, infer_shape, load_model) matcher, never by a model-level
        set.  The busy-device interference branch is structurally unreachable in
        the current admission topology (cold loads are only admitted on idle
        devices), so the composition is pinned directly."""

        from tracing.analysis.torpor_methods import torpor_placement_rank
        from tracing.analysis.workload_v02_simulator import prefetch_interference_extra_ms

        profile = {
            "enabled": True, "engine": "hf_substrate", "deployment": "multi_process",
            "interference_model": "additive_extra_ms", "provenance_kind": "synthetic",
            "source_artifact": "synthetic-mp-f4.json", "source_experiment": "synthetic-mp-f4",
            "gpu_identity": "synthetic-gpu",
            "probe_source_artifact": "synthetic-mp-f4-probe.json", "probe_metadata": {"probe": "synthetic"},
            "cells": [{"infer_model": "model-a", "infer_shape": "medium", "load": "model-b", "extra_ms": 50.0}],
        }

        class _Gpu:
            resident: dict = {}

        # A model seen in ANOTHER (infer, shape) cell must not count as covered.
        uncovered_extra = prefetch_interference_extra_ms(profile, "model-c", "short", "model-b")
        self.assertIsNone(uncovered_extra)
        rank, cost = torpor_placement_rank(
            "model-b", _Gpu(), 40.0, uncovered_extra is not None, uncovered_extra
        )
        self.assertEqual(rank, 2)

        # The exact measured triple is covered and its cost enters the ranking.
        covered_extra = prefetch_interference_extra_ms(profile, "model-a", "medium", "model-b")
        self.assertEqual(covered_extra, 50.0)
        rank2, cost2 = torpor_placement_rank(
            "model-b", _Gpu(), 40.0, covered_extra is not None, covered_extra
        )
        self.assertEqual(rank2, 1)
        self.assertAlmostEqual(cost2, 90.0, places=6)

    def test_torpor_counts_cold_loads_on_idle_devices_as_covered(self) -> None:
        """A cold load on an idle device has no overlap interference: it is a
        covered rank-1 placement with zero interference cost."""

        def node(node_id, model, role, runtime):
            return Node(
                node_id=node_id, sequence_index=0, predecessors=(), successors=(),
                lane="gpu", model_id=model, runtime_ms=runtime, load_ms=0.0,
                workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
                workload_shape="short", role=role,
            )

        stats = {
            "model-c|gpu|0|exact": {"runtime_p50_ms": 200.0, "runtime_p90_ms": 200.0, "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100},
            "model-b|gpu|0|exact": {"runtime_p50_ms": 100.0, "runtime_p90_ms": 100.0, "load_p50_ms": 40.0, "memory_p95_mb": 100.0, "count": 100},
        }
        interference = {
            "enabled": True, "engine": "hf_substrate", "deployment": "multi_process",
            "interference_model": "additive_extra_ms", "provenance_kind": "synthetic",
            "source_artifact": "synthetic-mp-f4.json", "source_experiment": "synthetic-mp-f4",
            "gpu_identity": "synthetic-gpu",
            "probe_source_artifact": "synthetic-mp-f4-probe.json", "probe_metadata": {"probe": "synthetic"},
            "cells": [{"infer_model": "model-c", "infer_shape": "short", "load": "model-b", "extra_ms": 50.0}],
        }
        episode = {
            "episode_id": "torpor-idle-load",
            "split": "train",
            "gpu_topology_mb": [1000.0],
            "gpu_identity": "synthetic-gpu",
            "initial_residency_hint": [[]],
            "jobs": [
                {"job_instance_id": "jb", "template_id": "b", "arrival_ms": 0.0, "deadline_ms": 5000.0, "service_class": "normal"},
            ],
        }
        activation: dict = {}
        simulate_episode(
            episode,
            {"b": simple_template("b", node("b:n", "model-b", "planner", 100.0))},
            "torpor_lifecycle",
            train_stats=stats,
            policy_context={"activation": activation},
            extension_config={"prefetch_interference": interference},
            collect_events=True,
        )
        self.assertEqual(activation.get("covered_load_choice"), 1)
        self.assertIsNone(activation.get("uncovered_load_choice"))
        self.assertIsNone(activation.get("interference_aware_choice"))

    def test_request_preemption_resumes_from_token_breakpoint(self) -> None:
        def req_node(node_id, model, role, runtime):
            return Node(
                node_id=node_id, sequence_index=0, predecessors=(), successors=(),
                lane="gpu", model_id=model, runtime_ms=runtime, load_ms=0.0,
                workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
                workload_shape="short", role=role,
            )

        victim = req_node("a:n", "model-a", "planner", 200.0)
        target = req_node("c:n", "model-c", "planner", 10.0)
        stats = {
            f"{model}|gpu|0|exact": {"runtime_p50_ms": r, "runtime_p90_ms": r,
                                     "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100}
            for model, r in (("model-a", 200.0), ("model-c", 10.0))
        }
        request_profile = {
            "enabled": True,
            "mode": "REQUEST_RECOMPUTE",
            "engine": "hf_substrate",
            "provenance_kind": "synthetic",
            "source_artifact": "synthetic-rp.json",
            "source_experiment": "synthetic-rp",
            "gpu_identity": "synthetic-gpu",
            "probe_source_artifact": "synthetic-rp-probe.json",
            "probe_metadata": {"probe": "synthetic"},
            "layers": {"model-a|planner": {"in_tokens": 1000, "out_tokens": 50}},
            "models": {"model-a": {"prefill_us_per_token": 100.0, "prefill_intercept_ms": 0.0, "tpot_ms": 4.0}},
        }
        episode = {
            "episode_id": "request-preempt",
            "split": "train",
            "gpu_topology_mb": [1000.0],
            "gpu_identity": "synthetic-gpu",
            "initial_residency_hint": [["model-a", "model-c"]],
            "jobs": [
                {"job_instance_id": "ja", "template_id": "a", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
                {"job_instance_id": "jc", "template_id": "c", "arrival_ms": 150.0, "deadline_ms": 1000.0, "service_class": "priority"},
            ],
        }
        config = {
            "preemption_enabled": True,
            "max_preemptions": 1,
            "request_preemption": request_profile,
        }
        summary, events = simulate_episode(
            episode,
            {"a": simple_template("a", victim), "c": simple_template("c", target)},
            "myopic_preempt",
            train_stats=stats,
            extension_config=config,
            collect_events=True,
        )
        # Prefill = 100ms; decode = 2ms/token; preempted at t=150 → 25 tokens done.
        preempts = [event for event in events if event.get("event_type") == "node_preempt"]
        self.assertEqual(summary["preemptions"], 1)
        self.assertEqual(len(preempts), 1)
        self.assertTrue(preempts[0]["request_mode"])
        self.assertEqual(preempts[0]["tokens_done"], 25)
        self.assertEqual(preempts[0]["n_ctx"], 1025)
        self.assertAlmostEqual(preempts[0]["rm_ms"], 102.5, places=3)
        # Resume = R_m (102.5) + 25 remaining tokens * 2ms = 152.5ms from t=160.
        a_finishes = [
            event["finish_ms"]
            for event in events
            if event.get("event_type") == "node_finish" and event.get("node_id") == "a:n"
        ]
        self.assertEqual(len(a_finishes), 1)
        self.assertAlmostEqual(a_finishes[0], 312.5, places=3)
        self.assertAlmostEqual(summary["preempt_recompute_ms"], 102.5, places=3)
        # Leak guard: token/phase truth never reaches the scheduler-visible state.
        for event in events:
            if event.get("event_type") == "node_dispatch" and event.get("scheduler_state"):
                dumped = json.dumps(event["scheduler_state"], ensure_ascii=False)
                for forbidden in ("tokens_done", "rm_ms", "prefill_end", "n_ctx", "request_split"):
                    self.assertNotIn(forbidden, dumped)

    def test_request_preemption_blocks_during_prefill_then_fires_at_boundary(self) -> None:
        def req_node(node_id, model, role, runtime):
            return Node(
                node_id=node_id, sequence_index=0, predecessors=(), successors=(),
                lane="gpu", model_id=model, runtime_ms=runtime, load_ms=0.0,
                workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
                workload_shape="short", role=role,
            )

        victim = req_node("a:n", "model-a", "planner", 200.0)
        target = req_node("c:n", "model-c", "planner", 10.0)
        stats = {
            f"{model}|gpu|0|exact": {"runtime_p50_ms": r, "runtime_p90_ms": r,
                                     "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100}
            for model, r in (("model-a", 200.0), ("model-c", 10.0))
        }
        request_profile = {
            "enabled": True, "mode": "REQUEST_RECOMPUTE", "engine": "hf_substrate",
            "provenance_kind": "synthetic", "source_artifact": "synthetic-rp.json",
            "source_experiment": "synthetic-rp", "gpu_identity": "synthetic-gpu",
            "probe_source_artifact": "synthetic-rp-probe.json", "probe_metadata": {"probe": "synthetic"},
            "layers": {"model-a|planner": {"in_tokens": 1000, "out_tokens": 50}},
            "models": {"model-a": {"prefill_us_per_token": 100.0, "prefill_intercept_ms": 0.0, "tpot_ms": 4.0}},
        }
        episode = {
            "episode_id": "request-preempt-prefill",
            "split": "train",
            "gpu_topology_mb": [1000.0],
            "gpu_identity": "synthetic-gpu",
            "initial_residency_hint": [["model-a", "model-c"]],
            "jobs": [
                {"job_instance_id": "ja", "template_id": "a", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
                {"job_instance_id": "jc", "template_id": "c", "arrival_ms": 50.0, "deadline_ms": 1000.0, "service_class": "priority"},
            ],
        }
        summary, events = simulate_episode(
            episode,
            {"a": simple_template("a", victim), "c": simple_template("c", target)},
            "myopic_preempt",
            train_stats=stats,
            extension_config={"preemption_enabled": True, "max_preemptions": 1, "request_preemption": request_profile},
            collect_events=True,
        )
        # At t=50 the victim is mid-prefill (atomic): the action is refused and a
        # wake is scheduled at the prefill boundary (t=100, 0 decode tokens done).
        self.assertEqual(summary["preemptions"], 1)
        self.assertIn("victim_prefill_in_progress", summary["preemption_blocked_reasons"])
        preempts = [event for event in events if event.get("event_type") == "node_preempt"]
        self.assertEqual(len(preempts), 1)
        self.assertAlmostEqual(preempts[0]["time_ms"], 100.0, places=3)
        self.assertEqual(preempts[0]["tokens_done"], 0)
        c_starts = [
            event["start_ms"]
            for event in events
            if event.get("event_type") == "node_start" and event.get("node_id") == "c:n"
        ]
        self.assertEqual(c_starts, [100.0])
        a_finishes = [
            event["finish_ms"]
            for event in events
            if event.get("event_type") == "node_finish" and event.get("node_id") == "a:n"
        ]
        # Resume at t=110: R_m(1000)=100 + 50 remaining * 2ms = 200 work.
        self.assertEqual(len(a_finishes), 1)
        self.assertAlmostEqual(a_finishes[0], 310.0, places=3)

    def test_request_preemption_defers_to_next_token_boundary(self) -> None:
        def req_node(node_id, model, role, runtime):
            return Node(
                node_id=node_id, sequence_index=0, predecessors=(), successors=(),
                lane="gpu", model_id=model, runtime_ms=runtime, load_ms=0.0,
                workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
                workload_shape="short", role=role,
            )

        victim = req_node("a:n", "model-a", "planner", 200.0)
        target = req_node("c:n", "model-c", "planner", 10.0)
        stats = {
            f"{model}|gpu|0|exact": {"runtime_p50_ms": r, "runtime_p90_ms": r,
                                     "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100}
            for model, r in (("model-a", 200.0), ("model-c", 10.0))
        }
        request_profile = {
            "enabled": True, "mode": "REQUEST_RECOMPUTE", "engine": "hf_substrate",
            "provenance_kind": "synthetic", "source_artifact": "synthetic-rp.json",
            "source_experiment": "synthetic-rp", "gpu_identity": "synthetic-gpu",
            "probe_source_artifact": "synthetic-rp-probe.json", "probe_metadata": {"probe": "synthetic"},
            "layers": {"model-a|planner": {"in_tokens": 1000, "out_tokens": 50}},
            "models": {"model-a": {"prefill_us_per_token": 100.0, "prefill_intercept_ms": 0.0, "tpot_ms": 4.0}},
        }
        episode = {
            "episode_id": "request-preempt-mid-token",
            "split": "train",
            "gpu_topology_mb": [1000.0],
            "gpu_identity": "synthetic-gpu",
            "initial_residency_hint": [["model-a", "model-c"]],
            "jobs": [
                {"job_instance_id": "ja", "template_id": "a", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
                # 151 ms sits inside the token spanning [150, 152); the legal
                # boundary is 152 ms.
                {"job_instance_id": "jc", "template_id": "c", "arrival_ms": 151.0, "deadline_ms": 1000.0, "service_class": "priority"},
            ],
        }
        summary, events = simulate_episode(
            episode,
            {"a": simple_template("a", victim), "c": simple_template("c", target)},
            "myopic_preempt",
            train_stats=stats,
            extension_config={"preemption_enabled": True, "max_preemptions": 1, "request_preemption": request_profile},
            collect_events=True,
        )
        self.assertEqual(summary["preemptions"], 1)
        self.assertIn("victim_deferred_to_token_boundary", summary["preemption_blocked_reasons"])
        preempts = [event for event in events if event.get("event_type") == "node_preempt"]
        self.assertEqual(len(preempts), 1)
        # The in-flight token completes: preemption snaps to the 152 ms boundary
        # with 26 tokens done (100 ms prefill + 26 * 2 ms).
        self.assertAlmostEqual(preempts[0]["time_ms"], 152.0, places=3)
        self.assertEqual(preempts[0]["tokens_done"], 26)
        c_starts = [
            event["start_ms"]
            for event in events
            if event.get("event_type") == "node_start" and event.get("node_id") == "c:n"
        ]
        self.assertEqual(c_starts, [152.0])

    def test_request_preemption_phase_uses_batching_rate(self) -> None:
        def req_node(node_id, model, role, runtime):
            return Node(
                node_id=node_id, sequence_index=0, predecessors=(), successors=(),
                lane="gpu", model_id=model, runtime_ms=runtime, load_ms=0.0,
                workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
                workload_shape="short", role=role,
            )

        victim = req_node("a:n", "model-a", "planner", 200.0)
        partner = req_node("b:n", "model-a", "planner", 200.0)
        target = req_node("c:n", "model-c", "planner", 10.0)
        stats = {
            f"{model}|gpu|0|exact": {"runtime_p50_ms": r, "runtime_p90_ms": r,
                                     "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100}
            for model, r in (("model-a", 200.0), ("model-c", 10.0))
        }
        request_profile = {
            "enabled": True, "mode": "REQUEST_RECOMPUTE", "engine": "hf_substrate",
            "provenance_kind": "synthetic", "source_artifact": "synthetic-rp.json",
            "source_experiment": "synthetic-rp", "gpu_identity": "synthetic-gpu",
            "probe_source_artifact": "synthetic-rp-probe.json", "probe_metadata": {"probe": "synthetic"},
            "layers": {"model-a|planner": {"in_tokens": 1000, "out_tokens": 50}},
            "models": {"model-a": {"prefill_us_per_token": 100.0, "prefill_intercept_ms": 0.0, "tpot_ms": 4.0}},
        }
        batching = {
            "enabled": True, "engine": "vllm_batched", "provenance_kind": "synthetic",
            "source_artifact": "synthetic-batch.json", "source_experiment": "synthetic-batch",
            "gpu_identity": "synthetic-gpu",
            "probe_source_artifact": "synthetic-batch-probe.json", "probe_metadata": {"probe": "synthetic"},
            "same_model_curves": {"model-a|planner": {"2": 1.05}},
            "homogeneity_tolerance": 1.25,
            "kv_pools_tokens": {"model-a": 100000},
            "layers": {"model-a|planner": {"in_tokens": 1000, "out_tokens": 50}},
        }
        colocation = {
            "enabled": True, "engine": "hf_substrate", "deployment": "multi_process",
            "provenance_kind": "synthetic", "source_artifact": "synthetic-f1.json",
            "source_experiment": "synthetic-f1", "gpu_identity": "synthetic-gpu",
            "cells": [
                {"model_a": "model-a", "shape_a": "short", "model_b": "model-c", "shape_b": "short",
                 "slowdown_a": 1.0, "slowdown_b": 1.0, "feasible": True},
            ],
        }
        episode = {
            "episode_id": "request-preempt-batched-phase",
            "split": "train",
            "gpu_topology_mb": [1000.0],
            "gpu_identity": "synthetic-gpu",
            "initial_residency_hint": [["model-a", "model-c"]],
            "jobs": [
                {"job_instance_id": "ja", "template_id": "a", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
                {"job_instance_id": "jb", "template_id": "b", "arrival_ms": 50.0, "deadline_ms": 1000.0, "service_class": "normal"},
                {"job_instance_id": "jc", "template_id": "c", "arrival_ms": 52.0, "deadline_ms": 1000.0, "service_class": "priority"},
            ],
        }
        config = {
            "preemption_enabled": True,
            "max_preemptions": 1,
            "request_preemption": request_profile,
            "batching_engine": batching,
            "colocation_profile": colocation,
        }
        summary, events = simulate_episode(
            episode,
            {
                "a": simple_template("a", victim),
                "b": simple_template("b", partner),
                "c": simple_template("c", target),
            },
            "myopic_preempt",
            train_stats=stats,
            extension_config=config,
            collect_events=True,
        )
        # a runs solo [0,50) = 50 work; then batched at f=1.05.  Its prefill
        # (100 work) completes at 50 + 50*1.05 = 102.5 ms; the deferred wake
        # fires there with 0 decode tokens done.
        self.assertEqual(summary["preemptions"], 1)
        self.assertIn("victim_prefill_in_progress", summary["preemption_blocked_reasons"])
        preempts = [event for event in events if event.get("event_type") == "node_preempt"]
        self.assertEqual(len(preempts), 1)
        self.assertAlmostEqual(preempts[0]["time_ms"], 102.5, places=3)
        self.assertEqual(preempts[0]["tokens_done"], 0)
        c_starts = [
            event["start_ms"]
            for event in events
            if event.get("event_type") == "node_start" and event.get("node_id") == "c:n"
        ]
        self.assertEqual(c_starts, [102.5])

    def test_request_preemption_profile_requires_enabled_flag(self) -> None:
        profile = {
            "enabled": True, "mode": "REQUEST_RECOMPUTE", "engine": "hf_substrate",
            "provenance_kind": "synthetic", "source_artifact": "synthetic-rp.json",
            "source_experiment": "synthetic-rp", "gpu_identity": "synthetic-gpu",
            "layers": {"model-a|planner": {"in_tokens": 1000, "out_tokens": 50}},
            "models": {"model-a": {"prefill_us_per_token": 100.0, "prefill_intercept_ms": 0.0, "tpot_ms": 4.0}},
        }
        with self.assertRaisesRegex(ValueError, "preemption_enabled"):
            validate_request_preemption_profile({"request_preemption": profile})
        validate_request_preemption_profile(
            {"request_preemption": profile, "preemption_enabled": True},
            {"gpu_identity": "synthetic-gpu"},
        )

    def test_preemption_removes_one_live_task_and_keeps_survivor_progress(self) -> None:
        nodes = {
            model: coloc_node(f"{model}:n", model, 100.0, "short")
            for model in ("model-a", "model-b", "model-c")
        }
        stats = {
            f"{model}|gpu|0|exact": {
                "runtime_p50_ms": 100.0,
                "runtime_p90_ms": 100.0,
                "load_p50_ms": 0.0,
                "memory_p95_mb": 100.0,
                "count": 100,
            }
            for model in nodes
        }
        cells = []
        for active, candidate in (("model-a", "model-b"), ("model-b", "model-c"), ("model-c", "model-a")):
            cells.append({
                "model_a": active, "shape_a": "short",
                "model_b": candidate, "shape_b": "short",
                "slowdown_a": 2.0, "slowdown_b": 2.0, "feasible": True,
            })
        episode = {
            "episode_id": "preempt-coloc",
            "split": "train",
            "gpu_topology_mb": [1000.0],
            "gpu_identity": "synthetic-gpu",
            "initial_residency_hint": [["model-a", "model-b", "model-c"]],
            "jobs": [
                {"job_instance_id": "ja", "template_id": "a", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
                {"job_instance_id": "jb", "template_id": "b", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
                {"job_instance_id": "jc", "template_id": "c", "arrival_ms": 50.0, "deadline_ms": 1000.0, "service_class": "priority"},
            ],
        }
        summary, events = simulate_episode(
            episode,
            {model[-1]: simple_template(model[-1], node) for model, node in nodes.items()},
            "myopic_preempt",
            train_stats=stats,
            extension_config={
                "preemption_enabled": True,
                "max_preemptions": 1,
                "colocation_profile": coloc_profile(cells),
            },
            collect_events=True,
        )
        preempts = [event for event in events if event.get("event_type") == "node_preempt"]
        self.assertEqual(summary["preemptions"], 1)
        self.assertEqual(len(preempts), 1)
        self.assertEqual(preempts[0]["node_id"], "model-a:n")
        starts = [event for event in events if event.get("event_type") == "node_start"]
        self.assertIn("model-b:n", [event["node_id"] for event in starts])
        self.assertIn("model-c:n", [event["node_id"] for event in starts])
        b_finishes = [event for event in events if event.get("event_type") == "node_finish" and event.get("node_id") == "model-b:n"]
        self.assertEqual(len(b_finishes), 1)

        cold_episode = dict(episode)
        cold_episode["initial_residency_hint"] = [["model-a", "model-b"]]
        cold_summary, cold_events = simulate_episode(
            cold_episode,
            {model[-1]: simple_template(model[-1], node) for model, node in nodes.items()},
            "myopic_preempt", train_stats=stats,
            extension_config={"preemption_enabled": True, "max_preemptions": 1,
                              "colocation_profile": coloc_profile(cells)},
        )
        self.assertEqual(cold_summary["preemptions"], 0)
        self.assertIn("target_not_dispatchable_with_survivor", cold_summary["preemption_blocked_reasons"])
        self.assertFalse(any(e["event_type"] == "node_preempt" for e in cold_events))

    def test_prefetch_pending_blocks_preemption_before_model_is_resident(self) -> None:
        infer = coloc_node("infer:n", "infer-model", 500.0, "short")
        loader = coloc_node("load:n", "load-model", 10.0, "short", load=1000.0)
        target = coloc_node("target:n", "target-model", 10.0, "short")
        templates = {
            "infer": simple_template("infer", infer),
            "load": simple_template("load", loader),
            "target": simple_template("target", target),
        }
        stats = {
            f"{model}|gpu|0|exact": {
                "runtime_p50_ms": 500.0 if model == "infer-model" else 10.0,
                "runtime_p90_ms": 500.0 if model == "infer-model" else 10.0,
                "load_p50_ms": 0.0,
                "memory_p95_mb": 100.0,
                "count": 100,
            }
            for model in ("infer-model", "load-model", "target-model")
        }
        config = {
            "preemption_enabled": True,
            "max_preemptions": 1,
            "prefetch_plan": [{"gpu_index": 0, "model_id": "load-model"}],
            "prefetch_overlap": True,
            "prefetch_interference_mode": "copy",
            "prefetch_interference": {
                "engine": "hf_substrate",
                "provenance_kind": "synthetic",
                "source_artifact": "synthetic-f4.json",
                "source_experiment": "synthetic-f4",
                "gpu_identity": "synthetic-gpu",
                "probe_source_artifact": "synthetic-f4-probe.json",
                "probe_metadata": {"probe": "synthetic"},
                "supported_workload_shape": "short",
                "pairs": [{"infer": "infer-model", "load": "load-model", "copy_slowdown": 2.0, "copy_dilation": 1.0}],
            },
        }
        episode = {
            "episode_id": "prefetch-preempt-guard",
            "split": "train",
            "gpu_topology_mb": [1000.0],
            "gpu_identity": "synthetic-gpu",
            "initial_residency_hint": [[]],
            "jobs": [
                {"job_instance_id": "ji", "template_id": "infer", "arrival_ms": 0.0, "deadline_ms": 1000.0, "service_class": "normal"},
                {"job_instance_id": "jt", "template_id": "target", "arrival_ms": 50.0, "deadline_ms": 1000.0, "service_class": "priority"},
            ],
        }
        summary, events = simulate_episode(
            episode, templates, "myopic_preempt", train_stats=stats, extension_config=config
        )
        self.assertEqual(summary["preemptions"], 0)
        self.assertIn("prefetch_pending_preemption_unsupported", summary["preemption_blocked_reasons"])
        self.assertEqual([event for event in events if event.get("event_type") == "node_preempt"], [])

    def test_gpu_ledger_unions_overlapping_intervals(self) -> None:
        gpu = GPU(index=0, capacity_mb=100.0)
        first = gpu.add_active_task("first", 0.0, 10.0, owner=(0, "first"))
        survivor = gpu.add_active_task("survivor", 5.0, 15.0, owner=(1, "survivor"))

        self.assertEqual(gpu.busy_until, 15.0)
        self.assertEqual(gpu.busy_time_ms, 15.0)
        gpu.remove_active_task(first, 10.0)
        self.assertEqual(gpu.active_node, (1, "survivor"))
        self.assertEqual(gpu.busy_until, 15.0)
        self.assertEqual(gpu.busy_time_ms, 15.0)
        self.assertLessEqual(gpu.busy_time_ms / gpu.busy_end_ms, 1.0)
        gpu.remove_active_task(survivor, 15.0)
        self.assertEqual(gpu.busy_time_ms, 15.0)

    def test_transition_completion_opens_colocation_without_unrelated_events(self) -> None:
        cold = coloc_node("a:n", "model-a", 200.0, "short", load=100.0)
        warm = coloc_node("b:n", "model-b", 100.0, "short")
        stats = {
            f"{node.model_id}|gpu|0|exact": {
                "runtime_p50_ms": node.runtime_ms, "runtime_p90_ms": node.runtime_ms,
                "load_p50_ms": node.load_ms, "memory_p95_mb": 100.0, "count": 100,
            } for node in (cold, warm)
        }
        episode = {
            "episode_id": "cold-transition-boundary", "split": "train",
            "gpu_topology_mb": [1000.0], "gpu_identity": "synthetic-gpu",
            "initial_residency_hint": [["model-b"]],
            "jobs": [
                {"job_instance_id": "ja", "template_id": "a", "arrival_ms": 0.0, "service_class": "normal"},
                {"job_instance_id": "jb", "template_id": "b", "arrival_ms": 0.0, "service_class": "normal"},
            ],
        }
        profile = coloc_profile([{
            "model_a": "model-a", "shape_a": "short", "model_b": "model-b", "shape_b": "short",
            "slowdown_a": 2.0, "slowdown_b": 2.0, "feasible": True,
        }])
        templates = {"a": simple_template("a", cold), "b": simple_template("b", warm)}
        summary, events = simulate_episode(episode, templates, "fcfs", train_stats=stats,
                                          extension_config={"colocation_profile": profile})
        b_dispatch = next(e for e in events if e["event_type"] == "node_dispatch" and e["node_id"] == "b:n")
        self.assertEqual(b_dispatch["time_ms"], 100.0)
        first_dispatch = next(e for e in events if e["event_type"] == "node_dispatch")
        self.assertEqual(first_dispatch["scheduler_state"]["gpus"][0]["resident_models"], ["model-b"])
        self.assertEqual(summary["mean_completion_ms"], 300.0)

        priority_episode = {**episode, "jobs": [episode["jobs"][0],
            {**episode["jobs"][1], "arrival_ms": 50.0, "service_class": "priority"}]}
        priority_summary, priority_events = simulate_episode(
            priority_episode, templates, "myopic_preempt", train_stats=stats,
            extension_config={"preemption_enabled": True, "max_preemptions": 1},
        )
        preempt = next(e for e in priority_events if e["event_type"] == "node_preempt")
        self.assertEqual(preempt["time_ms"], 100.0)
        self.assertIn("victim_transition_in_progress", priority_summary["preemption_blocked_reasons"])
        self.assertEqual(priority_summary["preempt_recompute_ms"], 0.0)

    def test_prefetch_completion_releases_successor_at_actual_finish(self) -> None:
        infer = replace(coloc_node("infer:n", "infer-model", 100.0, "short"), successors=("after:n",))
        after = replace(coloc_node("after:n", "cpu-model", 5.0, "short"),
                        predecessors=("infer:n",), lane="cpu", sequence_index=1)
        loader = coloc_node("load:n", "load-model", 10.0, "short", load=20.0)
        templates = {
            "infer": Template("infer", "video", "train", "test", (infer, after), {n.node_id: n for n in (infer, after)}),
            "load": simple_template("load", loader),
        }
        stats = {f"{n.model_id}|{n.lane}|{n.sequence_index}|exact": {
            "runtime_p50_ms": n.runtime_ms, "runtime_p90_ms": n.runtime_ms,
            "load_p50_ms": n.load_ms, "memory_p95_mb": 100.0, "count": 100,
        } for n in (infer, after, loader)}
        episode = {"episode_id": "prefetch-successor", "split": "train", "gpu_topology_mb": [1000.0],
                   "gpu_identity": "synthetic-gpu", "initial_residency_hint": [["infer-model"]],
                   "jobs": [{"job_instance_id": "j", "template_id": "infer", "arrival_ms": 0.0, "service_class": "normal"}]}
        config = {"prefetch_plan": [{"gpu_index": 0, "model_id": "load-model"}], "prefetch_overlap": True,
                  "prefetch_interference_mode": "full", "prefetch_interference": {
                      "engine": "hf_substrate", "provenance_kind": "synthetic", "source_artifact": "synthetic",
                      "source_experiment": "synthetic", "gpu_identity": "synthetic-gpu",
                      "probe_source_artifact": "synthetic", "probe_metadata": {"probe": "synthetic"},
                      "supported_workload_shape": "short",
                      "pairs": [{"infer": "infer-model", "load": "load-model", "full_slowdown": 2.0, "full_dilation": 1.0}],
                  }}
        summary, events = simulate_episode(episode, templates, "fcfs", train_stats=stats, extension_config=config)
        after_start = next(e for e in events if e["event_type"] == "node_start" and e["node_id"] == "after:n")
        self.assertEqual(after_start["time_ms"], 110.0)
        self.assertEqual(summary["mean_completion_ms"], 115.0)

    def test_transition_time_never_consumes_inference_progress(self) -> None:
        gpu = GPU(index=0, capacity_mb=1000.0)
        task_id = gpu.add_active_task(
            "cold", 0.0, 150.0, owner=(0, "cold"),
            work_ms=100.0, work_start_ms=50.0,
        )
        task = gpu.update_task_progress(task_id, 25.0)
        self.assertEqual(task.remaining_work_ms, 100.0)
        self.assertEqual(task.finish_ms, 150.0)
        task = gpu.update_task_progress(task_id, 75.0)
        self.assertEqual(task.remaining_work_ms, 75.0)
        task = gpu.set_task_slowdown(task_id, 2.0, 75.0)
        self.assertEqual(task.finish_ms, 225.0)

    def test_all_gpu_identities_must_match_measured_profile(self) -> None:
        profile = coloc_profile([{
            "model_a": "a", "shape_a": "short",
            "model_b": "b", "shape_b": "short",
            "slowdown_a": 2.0, "slowdown_b": 2.0, "feasible": True,
        }])
        with self.assertRaises(ValueError):
            validate_colocation_profile(
                {"colocation_profile": profile},
                {"gpu_identity": ["synthetic-gpu", "unmeasured-gpu"]},
            )
    def test_gpu_ledger_keeps_composite_reservation_and_removes_only_victim(self) -> None:
        gpu = GPU(index=0, capacity_mb=100.0)
        victim = gpu.add_active_task("victim", 0.0, 100.0, owner=(0, "victim"))
        gpu.add_active_task("composite", 40.0, 80.0, kind="composite", owner=(1, "c"))

        gpu.remove_active_task(victim, 20.0)
        self.assertIsNone(gpu.active_node)
        self.assertEqual(gpu.busy_until, 80.0)
        self.assertEqual(gpu.busy_time_ms, 60.0)

    def test_gpu_ledger_rejects_invalid_intervals(self) -> None:
        gpu = GPU(index=0, capacity_mb=100.0)
        with self.assertRaises(ValueError):
            gpu.add_active_task("negative", -1.0, 1.0)
        with self.assertRaises(ValueError):
            gpu.add_active_task("reversed", 2.0, 1.0)

    def test_new_model_admission_evicts_before_ledger_check(self) -> None:
        gpu = GPU(
            index=0,
            capacity_mb=32760.0,
            resident={"Qwen3-4B": 7600.0, "Qwen2.5-VL-3B-Instruct": 7100.0},
        )
        node = make_node("Qwen3-VL-8B-Instruct", 17000.0)
        admitted, evicted, model_mb, workspace_mb, total_mb = plan_gpu_admission(
            gpu, node, {"memory_p95_mb": 19430.0}
        )
        self.assertTrue(admitted)
        self.assertEqual(evicted, ("Qwen2.5-VL-3B-Instruct", "Qwen3-4B"))
        self.assertEqual(model_mb, 17000.0)
        self.assertEqual(workspace_mb, 2430.0)
        self.assertEqual(total_mb, 19430.0)
        self.assertEqual(
            gpu.resident,
            {"Qwen3-4B": 7600.0, "Qwen2.5-VL-3B-Instruct": 7100.0},
        )

    def test_resident_target_does_not_evict_itself(self) -> None:
        gpu = GPU(
            index=0,
            capacity_mb=20000.0,
            resident={"Qwen3-VL-8B-Instruct": 17000.0, "Qwen3-4B": 7600.0},
        )
        node = make_node("Qwen3-VL-8B-Instruct", 17000.0)
        admitted, evicted, _, _, total_mb = plan_gpu_admission(
            gpu, node, {"memory_p95_mb": 19430.0}
        )
        self.assertTrue(admitted)
        self.assertEqual(evicted, ("Qwen3-4B",))
        self.assertEqual(total_mb, 19430.0)
        self.assertIn("Qwen3-VL-8B-Instruct", gpu.resident)

    def test_model_larger_than_gpu_is_oom_without_eviction_plan(self) -> None:
        gpu = GPU(
            index=0,
            capacity_mb=16000.0,
            resident={"Qwen3-4B": 7600.0},
        )
        node = make_node("Qwen3-VL-8B-Instruct", 17000.0)
        admitted, evicted, _, _, total_mb = plan_gpu_admission(
            gpu, node, {"memory_p95_mb": 19430.0}
        )
        self.assertFalse(admitted)
        self.assertEqual(evicted, ())
        self.assertGreater(total_mb, gpu.capacity_mb)
        self.assertEqual(gpu.resident, {"Qwen3-4B": 7600.0})


    def test_myopic_joint_action_selects_shorter_ready_node(self) -> None:
        def make_job(job_id: str, node_id: str, model_id: str) -> Job:
            node = Node(
                node_id=node_id,
                sequence_index=0,
                predecessors=(),
                successors=(),
                lane="gpu",
                model_id=model_id,
                runtime_ms=100.0,
                load_ms=0.0,
                workspace_peak_mb=1100.0,
                resident_model_mb=1000.0,
                status="success",
            )
            template = Template(
                template_id=job_id,
                video_id="video",
                split="train",
                baseline="test",
                nodes=(node,),
                by_id={node_id: node},
            )
            return Job(
                job_instance_id=job_id,
                template=template,
                arrival_ms=0.0,
                deadline_ms=None,
                service_class="normal",
                node_state={node_id: "ready"},
            )

        slow = make_job("job-slow", "node-slow", "slow-model")
        fast = make_job("job-fast", "node-fast", "fast-model")
        stats = {
            "slow-model|gpu|0|exact": {
                "runtime_p50_ms": 100.0,
                "runtime_p90_ms": 120.0,
                "load_p50_ms": 0.0,
                "memory_p95_mb": 1100.0,
                "count": 3,
            },
            "fast-model|gpu|0|exact": {
                "runtime_p50_ms": 10.0,
                "runtime_p90_ms": 20.0,
                "load_p50_ms": 0.0,
                "memory_p95_mb": 1100.0,
                "count": 3,
            },
        }
        items = [(0.0, 0.0, 0, "node-slow"), (0.0, 0.0, 1, "node-fast")]
        selected, gpu_index, _, candidate_count, feasible_count = choose_action(
            "myopic",
            items,
            [slow, fast],
            [GPU(index=0, capacity_mb=32760.0)],
            stats,
            0,
        )
        self.assertEqual(selected[3], "node-fast")
        self.assertEqual(gpu_index, 0)
        self.assertEqual(candidate_count, 2)
        self.assertEqual(feasible_count, 2)

    def test_round_robin_joint_action_keeps_oldest_ready_node(self) -> None:
        node_a = make_node("model-a", 1000.0)
        node_b = make_node("model-b", 1000.0)
        template_a = Template("job-a", "video", "train", "test", (node_a,), {"node": node_a})
        template_b = Template("job-b", "video", "train", "test", (node_b,), {"node": node_b})
        job_a = Job("job-a", template_a, 0.0, None, "normal", {"node": "ready"})
        job_b = Job("job-b", template_b, 0.0, None, "normal", {"node": "ready"})
        stats = {
            "model-a|gpu|0|exact": {"runtime_p50_ms": 100.0, "runtime_p90_ms": 100.0, "load_p50_ms": 0.0, "memory_p95_mb": 1100.0, "count": 3},
            "model-b|gpu|0|exact": {"runtime_p50_ms": 1.0, "runtime_p90_ms": 1.0, "load_p50_ms": 0.0, "memory_p95_mb": 1100.0, "count": 3},
        }
        selected, _, _, candidate_count, _ = choose_action(
            "round_robin",
            [(0.0, 0.0, 0, "node"), (0.0, 0.0, 1, "node")],
            [job_a, job_b],
            [GPU(index=0, capacity_mb=32760.0)],
            stats,
            0,
        )
        self.assertEqual(selected[2], 0)
        self.assertEqual(candidate_count, 2)


if __name__ == "__main__":
    unittest.main()
