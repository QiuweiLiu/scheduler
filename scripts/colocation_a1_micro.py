#!/usr/bin/env python3
"""Colocation Phase A1: micro two-task endpoint calibration (advisor 2026-10-16).

For each of the 11 measured cells, build a minimal two-task co-located GPU
interval using the simulator's OWN machinery (GPU.add_active_task,
GPU.set_task_slowdown, GPU.update_task_progress, set_node_slowdown) and the
REAL lookup function colocation_slowdowns(); record both completion times and
compare against the physically measured coloc endpoints.

Pre-registered tolerance (frozen BEFORE looking at results, per advisor):
  tol_ms = max(0.03 * measured_ms, 30.0)   # 3-round median dispersion budget
Endpoint deviations beyond tol are reported as systematic-bias findings;
tolerances are NOT adjusted to hide them.

A1a = table lookup exactness (1e-9); A1b = endpoint reproduction within tol.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracing.analysis.workload_v02_simulator import (  # noqa: E402
    GPU,
    _colocation_profile,
    colocation_slowdowns,
)

CAL = ROOT / "experiments/EXP-20261004_real_strata_calibration_v1/artifacts"
OUT = ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/colocation_a1_micro.json"
TOL_REL, TOL_ABS = 0.03, 30.0


class _NodeStub:
    """Duck-typed Node carrying only the fields colocation_slowdowns reads."""

    def __init__(self, model_id: str, workload_shape: str) -> None:
        self.model_id = model_id
        self.workload_shape = workload_shape


def main() -> int:
    v7 = json.loads((CAL / "substrate_extension_v7.json").read_text(encoding="utf-8"))
    profile = _colocation_profile(v7)
    cells = profile["cells"]
    report = {"schema": "colocation-a1-micro", "tol": {"rel": TOL_REL, "abs_ms": TOL_ABS},
              "cells": [], "failures": []}

    for cell in cells:
        cid = cell["cell_id"]
        a = _NodeStub(cell["model_a"], cell["shape_a"])
        b = _NodeStub(cell["model_b"], cell["shape_b"])
        rec = {"cell_id": cid, "solo_a": cell["solo_a_ms"], "solo_b": cell["solo_b_ms"],
               "measured_coloc_a": cell["coloc_a_ms"], "measured_coloc_b": cell["coloc_b_ms"]}

        # --- real lookup, both directions ---
        gpu = GPU(index=0, capacity_mb=32230.8)
        id_a = gpu.add_active_task("A", 0.0, cell["solo_a_ms"], model_id=cell["model_a"],
                                   workload_shape=cell["shape_a"], work_ms=cell["solo_a_ms"])
        id_b = gpu.add_active_task("B", 0.0, cell["solo_b_ms"], model_id=cell["model_b"],
                                   workload_shape=cell["shape_b"], work_ms=cell["solo_b_ms"])
        fwd = colocation_slowdowns(profile, gpu.active_tasks[id_a], b)
        rev = colocation_slowdowns(profile, gpu.active_tasks[id_b], a)
        rec["lookup_fwd"] = fwd
        rec["lookup_rev"] = rev
        exp = (cell["slowdown_a"], cell["slowdown_b"])
        ok_a = fwd is not None and abs(fwd[0] - exp[0]) < 1e-9 and abs(fwd[1] - exp[1]) < 1e-9
        ok_b = rev is not None and abs(rev[0] - exp[1]) < 1e-9 and abs(rev[1] - exp[0]) < 1e-9
        rec["lookup_exact"] = bool(ok_a and ok_b)
        if not rec["lookup_exact"]:
            report["failures"].append(f"{cid}: lookup mismatch fwd={fwd} rev={rev} exp={exp}")

        # --- two-task execution via simulator methods ---
        gpu.set_task_slowdown(id_a, float(cell["slowdown_a"]), 0.0)
        gpu.set_task_slowdown(id_b, float(cell["slowdown_b"]), 0.0)
        ta, tb = gpu.active_tasks[id_a], gpu.active_tasks[id_b]
        fin_a, fin_b = ta.finish_ms, tb.finish_ms
        t1 = min(fin_a, fin_b)
        first = id_a if fin_a <= fin_b else id_b
        survivor = id_b if first == id_a else id_a
        gpu.update_task_progress(id_a, t1)
        gpu.update_task_progress(id_b, t1)
        if gpu.active_tasks[first].remaining_work_ms > 1e-9:
            report["failures"].append(f"{cid}: first task has leftover work at its finish")
        gpu.set_task_slowdown(survivor, 1.0, t1)
        pred_a = t1 if first == id_a else gpu.active_tasks[id_a].finish_ms
        pred_b = t1 if first == id_b else gpu.active_tasks[id_b].finish_ms
        rec["pred_coloc_a"], rec["pred_coloc_b"] = round(pred_a, 3), round(pred_b, 3)
        dev_a = pred_a - cell["coloc_a_ms"]
        dev_b = pred_b - cell["coloc_b_ms"]
        tol_a = max(TOL_REL * cell["coloc_a_ms"], TOL_ABS)
        tol_b = max(TOL_REL * cell["coloc_b_ms"], TOL_ABS)
        rec["dev_a"], rec["dev_b"] = round(dev_a, 3), round(dev_b, 3)
        rec["dev_rel_a"] = round(dev_a / cell["coloc_a_ms"], 4)
        rec["dev_rel_b"] = round(dev_b / cell["coloc_b_ms"], 4)
        rec["within_tol"] = bool(abs(dev_a) <= tol_a and abs(dev_b) <= tol_b)
        if not rec["within_tol"]:
            report["failures"].append(f"{cid}: endpoint deviation a={dev_a:+.1f}ms b={dev_b:+.1f}ms")
        report["cells"].append(rec)

    n_lookup = sum(1 for c in report["cells"] if c["lookup_exact"])
    n_endpoint = sum(1 for c in report["cells"] if c["within_tol"])
    report["A1a_lookup"] = f"{n_lookup}/{len(cells)}"
    report["A1b_endpoints"] = f"{n_endpoint}/{len(cells)}"
    report["A1a_status"] = "PASS" if n_lookup == len(cells) else "FAIL"
    report["A1b_status"] = "PASS" if n_endpoint == len(cells) else "FAIL"
    report["A1_status"] = "PASS" if (report["A1a_status"] == "PASS" and report["A1b_status"] == "PASS") else "FAIL"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for c in report["cells"]:
        print(f"{c['cell_id'][:44]:44s} pred {c['pred_coloc_a']:>7.1f}/{c['pred_coloc_b']:>7.1f} "
              f"meas {c['measured_coloc_a']:>7.1f}/{c['measured_coloc_b']:>7.1f} "
              f"dev {c['dev_a']:>+7.1f}/{c['dev_b']:>+7.1f}  tol={'OK' if c['within_tol'] else 'EXCEED'}")
    print("A1a:", report["A1a_status"], report["A1a_lookup"], "| A1b:", report["A1b_status"], report["A1b_endpoints"])
    if report["failures"]:
        print("failures:", len(report["failures"]))
    print("evidence:", OUT)
    return 0 if report["A1_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
