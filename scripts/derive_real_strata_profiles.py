#!/usr/bin/env python
"""Derive the DES profile artifacts from the real-strata calibration run.

Input : the raw artifact produced by .scratch/real_strata_calibration.py on the
        measurement instance (downloaded under experiments/).
Output: (1) real_strata_colocation_table_v1.json  (schema colocation-cost-table-v1)
        (2) real_strata_interference_additive_v1.json (schema prefetch-interference-additive-v2)

Both keep the EXACT strata vocabulary of the frozen workload: shape = node role
(planner / videotool_spatial / answer_generation), measured at the
request_preemption_profile_v1 token medians.  Model names stay in the short form
the sim's alias table already normalizes.

Usage:
  python scripts/derive_real_strata_profiles.py \
      --raw experiments/EXP-20261004_real_strata_calibration_v1/artifacts/real_strata_calibration.json \
      --out-dir experiments/EXP-20261004_real_strata_calibration_v1/artifacts
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

EXPERIMENT_ID = "EXP-20261004_real_strata_calibration_v1"


def _median(values):
    return round(statistics.median(values), 1) if values else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--probe-source", default=None,
                    help="path recorded as probe_source_artifact (defaults to --raw)")
    args = ap.parse_args()
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    probe_source = args.probe_source or str(args.raw)
    gpu = raw.get("gpu")
    protocol = raw.get("protocol") or {}

    # ---- co-location table -------------------------------------------------
    cells = []
    for cell in raw["cells"]["A"]:
        if "error" in cell:
            continue
        cells.append({
            "cell_id": cell["cell_id"],
            "model_a": cell["model_a"],
            "shape_a": cell["tier_a"],
            "model_b": cell["model_b"],
            "shape_b": cell["tier_b"],
            "in_a": cell["in_a"], "out_a": cell["out_a"], "images_a": cell.get("images_a", 0),
            "in_b": cell["in_b"], "out_b": cell["out_b"], "images_b": cell.get("images_b", 0),
            "solo_a_ms": _median(cell["solo_a_ms"]),
            "solo_b_ms": _median(cell["solo_b_ms"]),
            "coloc_a_ms": _median(cell["coloc_a_ms"]),
            "coloc_b_ms": _median(cell["coloc_b_ms"]),
            "slowdown_a": cell["slowdown_a"],
            "slowdown_b": cell["slowdown_b"],
            "feasible": True,
        })
    coloc = {
        "schema": "colocation-cost-table-v1",
        "experiment_id": EXPERIMENT_ID,
        "gpu": gpu,
        "deployment": "multi_process",
        "note": ("real-workload strata (shape = node role) measured at the frozen token medians; "
                 "worker reuse across cells within a model pair; same-model pairs are serial by "
                 "deployment semantics and are not part of this table"),
        "probe_metadata": protocol,
        "probe_source_artifact": probe_source,
        "cells": cells,
    }
    coloc_path = out / "real_strata_colocation_table_v1.json"
    coloc_path.write_text(json.dumps(coloc, ensure_ascii=False, indent=1))
    print(f"WROTE {coloc_path}  cells={len(cells)}")

    # ---- additive interference table --------------------------------------
    icells = []
    for cell in raw["cells"]["B"]:
        if "error" in cell:
            continue
        raw_extra = float(cell["extra_ms"])
        icells.append({
            "infer_model": cell["infer_model"],
            "infer_shape": cell["infer_tier"],
            "infer_in": cell["infer_in"], "infer_out": cell["infer_out"],
            "infer_images": cell.get("infer_images", 0),
            "load": cell["load_model"],
            # The additive model is non-negative by construction; a negative
            # median delta is measurement noise at the 3-round resolution and is
            # clamped to the 0 ms floor.  The measured value is kept for audit.
            "extra_ms": max(0.0, raw_extra),
            "extra_ms_raw": raw_extra,
            "extra_ms_clamped_to_zero": raw_extra < 0.0,
            "dilation": cell["dilation"],
            "n": len(cell["overlap_infer_ms"]),
            "solo_infer_median_ms": _median(cell["solo_infer_ms"]),
            "overlap_infer_median_ms": _median(cell["overlap_infer_ms"]),
            "source_cell_id": cell["cell_id"],
        })
    interf = {
        "schema": "prefetch-interference-additive-v2",
        "experiment_id": EXPERIMENT_ID,
        "gpu": gpu,
        "deployment": "multi_process",
        "interference_model": "additive_extra_ms",
        "note": ("real-workload infer strata x the other two load models; same-model loads are "
                 "structurally unreachable (a running model is resident) and are not measured"),
        "probe_metadata": protocol,
        "probe_source_artifact": probe_source,
        "cells": icells,
    }
    interf_path = out / "real_strata_interference_additive_v1.json"
    interf_path.write_text(json.dumps(interf, ensure_ascii=False, indent=1))
    print(f"WROTE {interf_path}  cells={len(icells)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
