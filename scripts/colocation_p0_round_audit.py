#!/usr/bin/env python3
"""Colocation Phase P0 (advisor 2026-10-16): audit the original per-round data.

Read-only. For every cell: raw 3-round solo/coloc values for both sides, medians,
min-max and dispersion, slowdown recomputation, the A1b piecewise-rate endpoint
deviations, and the theoretical structural-deviation term
    T_hat_L - T_L_measured = -(sL - 1) * (TL - tS / sL)
to confirm the sign is structural (math) rather than new physical evidence.
The frozen physical table, v7 config and A1b pre-registered thresholds are NOT
modified.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "experiments/EXP-20261004_real_strata_calibration_v1/artifacts"
A1 = ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/colocation_a1_micro.json"
OUT = ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/colocation_p0_round_audit.json"


def med(xs):
    return float(st.median(xs))


def main() -> int:
    cal = json.loads((CAL / "real_strata_calibration.json").read_text(encoding="utf-8"))
    a1 = json.loads(A1.read_text(encoding="utf-8"))
    a1_by_id = {c["cell_id"]: c for c in a1["cells"]}

    records = []
    phase_b = []
    for phase, cells in cal["cells"].items():
        for c in cells:
            cid = c["cell_id"]
            if "solo_a_ms" not in c:
                phase_b.append({"cell_id": cid, "extra_ms": c.get("extra_ms"), "dilation": c.get("dilation"),
                                "solo_infer_ms": c.get("solo_infer_ms"), "overlap_infer_ms": c.get("overlap_infer_ms"),
                                "solo_load_ms": c.get("solo_load_ms"), "overlap_load_ms": c.get("overlap_load_ms"),
                                "is_round_array": isinstance(c.get("extra_ms"), list)})
                continue
            a1c = a1_by_id.get(cid)
            rec = {"cell_id": cid, "phase": phase}
            for side in ("a", "b"):
                solo = [float(x) for x in c[f"solo_{side}_ms"]]
                coloc = [float(x) for x in c[f"coloc_{side}_ms"]]
                rec[f"solo_{side}"] = {"raw": solo, "median": med(solo), "min": min(solo), "max": max(solo),
                                       "range": max(solo) - min(solo), "rel_range": (max(solo) - min(solo)) / med(solo)}
                rec[f"coloc_{side}"] = {"raw": coloc, "median": med(coloc), "min": min(coloc), "max": max(coloc),
                                        "range": max(coloc) - min(coloc), "rel_range": (max(coloc) - min(coloc)) / med(coloc)}
                rec[f"slowdown_{side}_recomputed"] = med(coloc) / med(solo)
                rec[f"slowdown_{side}_frozen"] = c[f"slowdown_{side}"]
            if a1c is not None:
                # piecewise predictions vs raw round intervals (long side focus)
                for side in ("a", "b"):
                    pred = a1c[f"pred_coloc_{side}"]
                    coloc = rec[f"coloc_{side}"]
                    rec[f"piecewise_{side}"] = {
                        "pred": pred,
                        "median": coloc["median"],
                        "dev_vs_median": pred - coloc["median"],
                        "inside_raw_range": coloc["min"] - 1e-9 <= pred <= coloc["max"] + 1e-9,
                        "rel_dev_vs_median": (pred - coloc["median"]) / coloc["median"],
                    }
                # structural-deviation math check on the long side
                sa, sb = c["slowdown_a"], c["slowdown_b"]
                ma, mb = med([float(x) for x in c["solo_a_ms"]]), med([float(x) for x in c["solo_b_ms"]])
                ta, tb = ma * sa, mb * sb  # short side finish
                if ta <= tb:
                    sL, TL, tS = sb, mb, ta
                    long_side = "b"
                else:
                    sL, TL, tS = sa, ma, tb
                    long_side = "a"
                rec["structural"] = {
                    "long_side": long_side, "sL": sL, "TL": TL, "tS": tS,
                    "term_ms": -(sL - 1.0) * (TL - tS / sL),
                    "observed_dev_ms": rec[f"piecewise_{long_side}"]["dev_vs_median"],
                }
                rec["structural"]["math_ok"] = abs(rec["structural"]["term_ms"] - rec["structural"]["observed_dev_ms"]) < 0.5
            records.append(rec)

    n_inside = sum(1 for r in records for side in ("a", "b")
                   if f"piecewise_{side}" in r and r[f"piecewise_{side}"]["inside_raw_range"])
    n_long_inside = sum(1 for r in records if r["piecewise_" + r["structural"]["long_side"]]["inside_raw_range"])
    out = {"schema": "colocation-p0-round-audit", "n_cells": len(records), "phase_b_probes": phase_b,
           "n_piecewise_preds_inside_raw_range": f"{n_inside}/{2 * len(records)}",
           "n_long_side_inside_raw_range": f"{n_long_inside}/{len(records)}",
           "records": records}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    for r in records:
        ls = r["structural"]["long_side"]
        lc = r[f"coloc_{ls}"]
        print(f"{r['cell_id'][:44]:44s} long={ls} rounds[{lc['min']:.0f},{lc['max']:.0f}] "
              f"med {lc['median']:.1f} | piecewise dev {r['structural']['observed_dev_ms']:+.1f} "
              f"term {r['structural']['term_ms']:+.1f} math={'OK' if r['structural']['math_ok'] else 'MISMATCH'} "
              f"inside={'Y' if r['piecewise_' + ls]['inside_raw_range'] else 'N'} | coloc rel-range {lc['rel_range']:.3f}")
    print(f"piecewise preds inside raw round range: {out['n_piecewise_preds_inside_raw_range']}; "
          f"long-side inside: {out['n_long_side_inside_raw_range']}")
    print("evidence:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
