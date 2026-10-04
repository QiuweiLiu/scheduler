"""Unit tests for the frozen-workload profile contract (evidence-based derivation)."""
from __future__ import annotations

import unittest

from tracing.analysis.profile_contract import (
    apply_real_workload_profile_contract,
    covered_strata,
)
from tracing.analysis.workload_v02_simulator import Node, Template


def node(node_id, model, role, lane="gpu", gpu_model="NVIDIA GeForce RTX 4080 SUPER", shape=""):
    return Node(
        node_id=node_id, sequence_index=0, predecessors=(), successors=(),
        lane=lane, model_id=model, runtime_ms=100.0, load_ms=0.0,
        workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
        role=role, workload_shape=shape, gpu_model=gpu_model,
    )


def template(template_id, nodes):
    return Template(template_id, template_id, "train", "test", tuple(nodes),
                    {n.node_id: n for n in nodes})


def config(identity="NVIDIA GeForce RTX 4080"):
    return {
        "colocation_profile": {
            "gpu_identity": identity,
            "cells": [{"model_a": "m8", "shape_a": "planner",
                       "model_b": "m4", "shape_b": "planner",
                       "slowdown_a": 1.2, "slowdown_b": 1.1, "feasible": True}],
        },
        "prefetch_interference": {
            "gpu_identity": identity,
            "cells": [{"infer_model": "m8", "infer_shape": "planner", "load": "m4",
                       "extra_ms": 10.0, "n": 3}],
        },
    }


class ProfileContractTests(unittest.TestCase):
    def test_derives_shape_from_role_and_identity_from_node_evidence(self) -> None:
        tpl = template("t", [node("n1", "m8", "planner"),
                             node("n2", "m4", "planner"),
                             node("cpu", "cpu-model", "videotool_temporal", lane="cpu", gpu_model=""),
                             node("yolo", "yolo11x.pt", "videotool_spatial")])
        templates, episodes = apply_real_workload_profile_contract(
            {"t": tpl}, [{"episode_id": "e1"}], config())
        self.assertEqual(templates["t"].by_id["n1"].workload_shape, "planner")
        self.assertEqual(templates["t"].by_id["n2"].workload_shape, "planner")
        # uncovered models and non-GPU lanes keep their empty shape
        self.assertEqual(templates["t"].by_id["yolo"].workload_shape, "")
        self.assertEqual(templates["t"].by_id["cpu"].workload_shape, "")
        self.assertEqual(episodes[0]["gpu_identity"], "NVIDIA GeForce RTX 4080 SUPER")
        # the original template object is untouched (frozen derivation)
        self.assertEqual(tpl.by_id["n1"].workload_shape, "")

    def test_uncovered_stratum_of_a_covered_model_fails_closed(self) -> None:
        tpl = template("t", [node("n1", "m8", "videotool_temporal")])
        with self.assertRaisesRegex(ValueError, "has no measured cell"):
            apply_real_workload_profile_contract({"t": tpl}, [{"episode_id": "e1"}], config())

    def test_partially_missing_gpu_evidence_fails_closed(self) -> None:
        """One identified GPU node must not silently cover a missing one."""
        tpl = template("t", [node("n1", "m8", "planner"),
                             node("n2", "m4", "planner", gpu_model="")])
        with self.assertRaisesRegex(ValueError, "without gpu_model evidence"):
            apply_real_workload_profile_contract({"t": tpl}, [{"episode_id": "e1"}], config())

    def test_mixed_gpu_evidence_fails_closed(self) -> None:
        tpl = template("t", [node("n1", "m8", "planner"),
                             node("n2", "m4", "planner", gpu_model="NVIDIA GeForce RTX 4090")])
        with self.assertRaisesRegex(ValueError, "uniform node gpu_model"):
            apply_real_workload_profile_contract({"t": tpl}, [{"episode_id": "e1"}], config())

    def test_different_gpu_class_fails_closed(self) -> None:
        tpl = template("t", [node("n1", "m8", "planner", gpu_model="NVIDIA GeForce RTX 4090")])
        with self.assertRaisesRegex(ValueError, "does not match the profile"):
            apply_real_workload_profile_contract({"t": tpl}, [{"episode_id": "e1"}], config())

    def test_existing_episode_identity_must_match(self) -> None:
        tpl = template("t", [node("n1", "m8", "planner")])
        with self.assertRaisesRegex(ValueError, "declares"):
            apply_real_workload_profile_contract(
                {"t": tpl}, [{"episode_id": "e1", "gpu_identity": "NVIDIA GeForce RTX 4090"}],
                config())

    def test_covered_strata_unions_both_profiles(self) -> None:
        covered = covered_strata(config())
        self.assertEqual(covered["m8"], {"planner"})
        self.assertEqual(covered["m4"], {"planner"})


if __name__ == "__main__":
    unittest.main()
