#!/usr/bin/env python3
from pathlib import Path
from dataclasses import replace
import json
import unittest

from tracing.analysis.workload_v02_simulator import (
    GPU,
    Job,
    Node,
    Template,
    choose_action,
    colocation_slowdowns,
    load_colocation_profile,
    load_measured_engine_profile,
    load_prefetch_interference_profile,
    phase_decomposition,
    plan_gpu_admission,
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
            "supported_workload_shape": "any",
            "loads": [{"load": "load-model", "extra_ms": 150.0}],
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

    def test_batching_engine_admits_same_model_pair_with_measured_speedup(self) -> None:
        def batch_node(node_id, model, role, runtime):
            return Node(
                node_id=node_id, sequence_index=0, predecessors=(), successors=(),
                lane="gpu", model_id=model, runtime_ms=runtime, load_ms=0.0,
                workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
                workload_shape="medium", role=role,
            )

        short = batch_node("a:n", "model-a", "planner", 100.0)
        long = batch_node("b:n", "model-a", "planner", 200.0)
        stats = {"model-a|gpu|0|exact": {"runtime_p50_ms": 100.0, "runtime_p90_ms": 200.0,
                                         "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100}}
        batching = {
            "enabled": True, "engine": "vllm_batched", "provenance_kind": "synthetic",
            "source_artifact": "synthetic-batch.json", "source_experiment": "synthetic-batch",
            "gpu_identity": "synthetic-gpu",
            "probe_source_artifact": "synthetic-batch-probe.json", "probe_metadata": {"probe": "synthetic"},
            "same_model_curves": {"model-a|planner": {"2": 2.0}},
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
            {"a": simple_template("a", short), "b": simple_template("b", long)},
            "fcfs",
            train_stats=stats,
            extension_config={"batching_engine": batching},
            collect_events=True,
        )
        # Batch factor 1/2: both start together; the short call finishes at 50;
        # the survivor speeds back up (100 remaining work) and finishes at 150.
        starts = [event["start_ms"] for event in events if event.get("event_type") == "node_start"]
        finishes = [event["finish_ms"] for event in events if event.get("event_type") == "node_finish"]
        self.assertEqual(starts, [0.0, 0.0])
        self.assertEqual(finishes, [50.0, 150.0])
        self.assertTrue(summary["batching_engine_enabled"])

        # Without the engine the same pair serializes (multi-process rule).
        serial_summary, serial_events = simulate_episode(
            episode,
            {"a": simple_template("a", short), "b": simple_template("b", long)},
            "fcfs",
            train_stats=stats,
            extension_config={},
            collect_events=True,
        )
        serial_starts = [event["start_ms"] for event in serial_events if event.get("event_type") == "node_start"]
        self.assertEqual(serial_starts, [0.0, 100.0])
        self.assertFalse(serial_summary["batching_engine_enabled"])

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
        batching = {
            "enabled": True, "engine": "vllm_batched", "provenance_kind": "synthetic",
            "source_artifact": "synthetic-batch.json", "source_experiment": "synthetic-batch",
            "gpu_identity": "synthetic-gpu",
            "same_model_curves": {"model-a|planner": {"2": 2.0}},
            # context = 110 tokens per request; a 100-token pool cannot hold two requests.
            "kv_pools_tokens": {"model-a": 100},
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
        _summary, events = simulate_episode(
            episode,
            {"a": simple_template("a", node)},
            "fcfs",
            train_stats=stats,
            extension_config={"batching_engine": batching},
            collect_events=True,
        )
        starts = [event["start_ms"] for event in events if event.get("event_type") == "node_start"]
        self.assertEqual(starts, [0.0, 100.0])

    def test_batching_engine_validator_rejects_invalid_curves(self) -> None:
        profile = {
            "enabled": True, "engine": "vllm_batched", "provenance_kind": "synthetic",
            "source_artifact": "synthetic-batch.json", "source_experiment": "synthetic-batch",
            "gpu_identity": "synthetic-gpu",
            "same_model_curves": {"model-a|planner": {"2": 0.5}},
            "kv_pools_tokens": {"model-a": 1000},
        }
        with self.assertRaisesRegex(ValueError, "speedup"):
            validate_batching_engine_profile({"batching_engine": profile})

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

    def test_request_preemption_blocks_during_prefill(self) -> None:
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
        self.assertEqual(summary["preemptions"], 0)
        self.assertIn("victim_prefill_in_progress", summary["preemption_blocked_reasons"])
        c_starts = [
            event["start_ms"]
            for event in events
            if event.get("event_type") == "node_start" and event.get("node_id") == "c:n"
        ]
        # The victim is never interrupted mid-prefill: the target waits for the GPU.
        self.assertEqual(c_starts, [200.0])

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
