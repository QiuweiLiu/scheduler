"""Tests for the pressure-grid transforms (exp C): arrival scaling, capacity
pinning, provenance, determinism, and the frozen-builder reuse.
"""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pressure_grid_lib import (  # noqa: E402
    CAPACITY_MB,
    build_pressure_cell,
    cell_aggregate_sha256,
    cell_id,
    cell_ids,
    episode_sha256,
)


def episode_fixture():
    return {
        "episode_id": "ep_x",
        "scenario_cell": "cell_x",
        "gpu_topology_mb": [32760.0, 24576.0],
        "episode_window_ms": 100000.0,
        "arrival_span_ms": 90000.0,
        "predicted_total_gpu_runtime_ms": 50000.0,
        "target_offered_compute_load": 0.25,
        "realized_offered_compute_load": 0.25,
        "jobs": [
            {"job_instance_id": "j0", "template_id": "t0", "arrival_ms": 0.0,
             "deadline_ms": 150000.0, "service_class": "p1"},
            {"job_instance_id": "j1", "template_id": "t1", "arrival_ms": 30000.0,
             "deadline_ms": 180000.0, "service_class": "p2"},
            {"job_instance_id": "j2", "template_id": "t2", "arrival_ms": 90000.0,
             "deadline_ms": 240000.0, "service_class": "p3"},
        ],
    }


class TransformTests(unittest.TestCase):
    def test_alpha_scales_arrivals_and_preserves_budget(self) -> None:
        original = episode_fixture()
        before = copy.deepcopy(original)
        out = build_pressure_cell([original], 0.8, "cap2424")[0]
        arrivals_o = [j["arrival_ms"] for j in before["jobs"]]
        arrivals_n = [j["arrival_ms"] for j in out["jobs"]]
        for old, new in zip(arrivals_o, arrivals_n):
            self.assertAlmostEqual(new, round(old * 0.8, 3))
        for j_old, j_new in zip(before["jobs"], out["jobs"]):
            budget_old = j_old["deadline_ms"] - j_old["arrival_ms"]
            budget_new = j_new["deadline_ms"] - j_new["arrival_ms"]
            self.assertAlmostEqual(budget_new, budget_old, places=6)
        self.assertAlmostEqual(out["episode_window_ms"], 80000.0)
        self.assertFalse(build_pressure_cell([before], 0.8, "cap2424")[0]["jobs"][0]["arrival_ms"]
                         != arrivals_n[0])

    def test_capacity_is_pinned_and_capacity_values_real(self) -> None:
        original = episode_fixture()
        for key, values in CAPACITY_MB.items():
            out = build_pressure_cell([original], 1.0, key)[0]
            self.assertEqual(out["gpu_topology_mb"], [float(v) for v in values])
            self.assertEqual(out["pressure_capacity_mb"], [float(v) for v in values])
        # natural fleet values only
        for values in CAPACITY_MB.values():
            self.assertIn(values, ((24576.0, 24576.0), (32760.0, 24576.0), (32760.0, 32760.0)))

    def test_service_and_membership_fields_untouched(self) -> None:
        original = episode_fixture()
        out = build_pressure_cell([original], 0.9, "cap3224")[0]
        self.assertEqual(out["predicted_total_gpu_runtime_ms"],
                         original["predicted_total_gpu_runtime_ms"])
        self.assertEqual([j["template_id"] for j in out["jobs"]],
                         [j["template_id"] for j in original["jobs"]])
        self.assertEqual([j["job_instance_id"] for j in out["jobs"]],
                         [j["job_instance_id"] for j in original["jobs"]])

    def test_provenance_fields(self) -> None:
        original = episode_fixture()
        out = build_pressure_cell([original], 1.0, "cap3232")[0]
        self.assertEqual(out["episode_id"], "ep_x")
        self.assertEqual(out["parent_episode_sha256"], episode_sha256(original))
        self.assertEqual(out["pressure_alpha"], 1.0)
        self.assertEqual(out["pressure_capacity_key"], "cap3232")
        self.assertTrue(out["scenario_cell"].endswith("pressure-grid-v1"))

    def test_input_not_mutated(self) -> None:
        original = episode_fixture()
        before = copy.deepcopy(original)
        build_pressure_cell([original], 0.8, "cap2424")
        self.assertEqual(original, before)

    def test_determinism_and_alpha_sensitivity(self) -> None:
        original = episode_fixture()
        a = cell_aggregate_sha256(build_pressure_cell([original], 0.8, "cap2424"))
        b = cell_aggregate_sha256(build_pressure_cell([original], 0.8, "cap2424"))
        c = cell_aggregate_sha256(build_pressure_cell([original], 0.9, "cap2424"))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_invalid_capacity_key_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_pressure_cell([episode_fixture()], 1.0, "cap9999")


class GridDefinitionTests(unittest.TestCase):
    def test_nine_unique_cells(self) -> None:
        ids = cell_ids()
        self.assertEqual(len(ids), 9)
        self.assertEqual(len(set(ids)), 9)
        self.assertEqual(cell_id(0.8, "cap2424"), "a080_cap2424")
        for cell in ids:
            alpha = float(cell.split("_")[0][1:]) / 100.0
            self.assertIn(alpha, (1.0, 0.9, 0.8))


if __name__ == "__main__":
    unittest.main()
