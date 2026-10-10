"""Tests for exp D (counterfactual lib) and exp E (prefetch sensitivity arms)."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from counterfactual_lib import (  # noqa: E402
    ALL_RESIDENT_CAPACITY_MB,
    SINGLE_MODEL_ID,
    build_all_resident_cell,
    single_model_transform,
)
from tracing.analysis.residency_methods import totals_from_demand  # noqa: E402
from tracing.analysis.workload_v02_simulator import (  # noqa: E402
    RESIDENCY_POLICIES,
    RESIDENCY_PREFETCH_POLICIES,
    Node,
    Template,
    residency_next_steps_for_policy,
    residency_states_for_policy,
    simulate_episode,
)


def gpu_node(node_id, model, memory, load_ms, successors=(), predecessors=()):
    return Node(
        node_id=node_id,
        sequence_index=0,
        predecessors=tuple(predecessors),
        successors=tuple(successors),
        lane="gpu",
        model_id=model,
        runtime_ms=100.0,
        load_ms=load_ms,
        workspace_peak_mb=100.0,
        resident_model_mb=memory,
        status="success",
    )


def fixture_templates():
    three = gpu_node("t1:1", "Qwen2.5-VL-3B-Instruct", 7100.0, 2800.0)
    four = gpu_node("t2:1", "Qwen3-4B", 7600.0, 2800.0)
    eight_a = gpu_node("t3:1", "Qwen3-VL-8B-Instruct", 17000.0, 5200.0)
    eight_b = gpu_node("t4:1", "Qwen3-VL-8B-Instruct", 17000.0, 5191.0)
    yolo = gpu_node("t5:1", "yolo11x.pt", 512.0, 300.0)
    templates = {}
    for tid, node in (("t1", three), ("t2", four), ("t3", eight_a), ("t4", eight_b), ("t5", yolo)):
        templates[tid] = Template(tid, "video", "train", "test", (node,), {node.node_id: node})
    return templates


def episode_fixture():
    return {
        "episode_id": "ep_d",
        "split": "train",
        "gpu_topology_mb": [32760.0, 32760.0],
        "episode_window_ms": 5000.0,
        "predicted_total_gpu_runtime_ms": 1000.0,
        "target_offered_compute_load": 0.25,
        "realized_offered_compute_load": 0.25,
        "jobs": [{"job_instance_id": "j0", "template_id": "t1", "arrival_ms": 0.0,
                  "deadline_ms": 100000.0, "service_class": "normal"}],
        "initial_residency_hint": [["Qwen3-4B", "Qwen3-VL-8B-Instruct"], ["yolo11x.pt"]],
        "initial_residency_memory_mb": [[7600.0, 17000.0], [512.0]],
    }


def pack_fixture():
    step = {
        "model_probabilities": {"Qwen2.5-VL-3B-Instruct": 0.3, "cpu-metadata-adapter-v1": 0.7},
        "resource": {
            "runtime_mean_ms": 100.0,
            "runtime_ms_quantiles": {"p50": 100.0, "p90": 100.0, "p95": 100.0},
            "load_occurrence_probability": 1.0,
            "load_duration_ms_quantiles": {"p50": 50.0, "p95": 50.0},
        },
    }
    return {"t1:1": {"length_probabilities": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                     "future_h5": [{"steps": [step]}]}}


class SingleModelTransformTests(unittest.TestCase):
    def test_relabel_memory_load_and_pack(self) -> None:
        templates = fixture_templates()
        before = copy.deepcopy(templates)
        episodes = [episode_fixture()]
        pack = pack_fixture()
        new_templates, new_episodes, new_pack = single_model_transform(templates, episodes, pack)
        self.assertEqual(new_templates["t1"].nodes[0].model_id, SINGLE_MODEL_ID)
        self.assertEqual(new_templates["t1"].nodes[0].resident_model_mb, 17000.0)
        self.assertEqual(new_templates["t1"].nodes[0].load_ms, 5195.5)  # median of 8B loads
        self.assertEqual(new_templates["t4"].nodes[0].model_id, SINGLE_MODEL_ID)
        self.assertEqual(new_templates["t5"].nodes[0].model_id, "yolo11x.pt")
        # input not mutated
        self.assertEqual(templates["t1"].nodes[0].model_id, before["t1"].nodes[0].model_id)
        # episode hints merged and deduped
        self.assertEqual(new_episodes[0]["initial_residency_hint"],
                         [[SINGLE_MODEL_ID], ["yolo11x.pt"]])
        self.assertEqual(new_episodes[0]["initial_residency_memory_mb"], [[17000.0], [512.0]])
        # pack LLM mass merged onto the single model; CPU mass preserved
        distribution = new_pack["t1:1"]["future_h5"][0]["steps"][0]["model_probabilities"]
        self.assertAlmostEqual(distribution[SINGLE_MODEL_ID], 0.3)
        self.assertAlmostEqual(distribution["cpu-metadata-adapter-v1"], 0.7)
        self.assertNotIn("Qwen2.5-VL-3B-Instruct", distribution)

    def test_all_resident_cell_pins_capacity(self) -> None:
        cell = build_all_resident_cell([episode_fixture()])
        self.assertEqual(cell[0]["gpu_topology_mb"], list(ALL_RESIDENT_CAPACITY_MB))
        self.assertEqual(cell[0]["episode_id"], "ep_d")
        self.assertEqual(cell[0]["counterfactual"], "all_resident_v1")


class SensitivityPolicyTests(unittest.TestCase):
    def test_totals_from_demand_next_steps(self) -> None:
        demand = [{"a": 1.0}, {"a": 10.0, "b": 5.0}, {"a": 100.0}]
        values, nexts1 = totals_from_demand(demand)
        self.assertEqual(values, {"a": 111.0, "b": 5.0})
        self.assertEqual(nexts1, {"a": 1.0})
        _values, nexts2 = totals_from_demand(demand, next_steps=2)
        self.assertEqual(nexts2, {"a": 11.0, "b": 5.0})

    def test_policy_helpers(self) -> None:
        self.assertEqual(residency_states_for_policy("pdrs_resident"), ("ready", "running"))
        self.assertEqual(residency_states_for_policy("pdrs_resident_readyonly"), ("ready",))
        self.assertEqual(residency_next_steps_for_policy("pdrs_resident"), 1)
        self.assertEqual(residency_next_steps_for_policy("pdrs_resident_k2"), 2)

    def test_registration(self) -> None:
        for arm in ("pdrs_resident_k2", "pdrs_resident_inflight2", "pdrs_resident_readyonly"):
            self.assertIn(arm, RESIDENCY_POLICIES)
        self.assertIn("pdrs_resident_k2", RESIDENCY_PREFETCH_POLICIES)
        self.assertIn("pdrs_resident_readyonly", RESIDENCY_PREFETCH_POLICIES)
        self.assertNotIn("pdrs_resident_inflight2", RESIDENCY_PREFETCH_POLICIES)


def run_fixture_episode(policy):
    templates = fixture_templates()
    episode = {
        "episode_id": "ep_e",
        "split": "train",
        "gpu_topology_mb": [20000.0, 20000.0],
        "jobs": [{"job_instance_id": "j0", "template_id": "t3", "arrival_ms": 0.0,
                  "deadline_ms": 100000.0, "service_class": "normal"}],
    }
    stats = {}
    for model, memory in (("Qwen3-VL-8B-Instruct", 350.0),):
        stats[f"{model}|gpu|*|model_lane"] = {
            "runtime_p50_ms": 100.0, "runtime_p90_ms": 100.0,
            "load_p50_ms": 40.0, "memory_p95_mb": memory, "count": 100,
        }
    pack = {"t3:1": {"length_probabilities": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                     "future_h5": [{"steps": []}]}}
    context = {"activation": {}}
    summary, _ = simulate_episode(
        episode, templates, policy, train_stats=stats, policy_context=context,
        future_artifacts=pack, future_horizon=5,
        extension_config={"prefetch_overlap": True}, collect_events=False)
    return summary, context


class VariantRunTests(unittest.TestCase):
    def test_variant_arms_run_end_to_end(self) -> None:
        for policy in ("pdrs_resident_k2", "pdrs_resident_inflight2", "pdrs_resident_readyonly"):
            summary, _ = run_fixture_episode(policy)
            self.assertEqual(summary["failed_jobs"], 0, policy)
            self.assertIn("mean_completion_ms", summary)


if __name__ == "__main__":
    unittest.main()
