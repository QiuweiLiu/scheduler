#!/usr/bin/env python3
"""Fast Action-Value v1 evaluation (advisor P0-P4, 2026-10-16).

Loads the expanded label dataset and runs:
  P0: delta-hat gating (relative to the model's own PDRS prediction), actual
      switch counting, partition-qualified states.
  P2: frozen-architecture 4-way ablation F0 / Prior / H5 / Prior+H5 (GBDT
      primary, ridge secondary), same samples/candidates/states.
  P3: theta calibrated on TRAIN out-of-fold predictions (GroupKFold by episode),
      pre-registered grid {0,100,250,500,1000}; selected theta applied to dev.
  P4: H5 feature diagnostics (nonzero rates, variance, per-state spread).
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/av_v1_dataset.json"
OUT = ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/av_v1_report.json"

F0_COLS = ["cand_runtime_ms", "cand_load_ms", "cand_mem_mb", "cand_wait_ms", "cand_seq_idx",
           "gpu_queue_ms", "gpu_free_mb", "gpu_model_resident", "n_ready_gpu", "n_ready_total", "f0_rank"]
PRIOR_COLS = ["prior_remaining_ms", "prior_terminated_frac"]
H5_COLS = ["h5_chain_compute_ms", "h5_chain_len", "h5_model_switches", "h5_load_sum_ms"]
VARIANTS = {"AV-F0": F0_COLS, "AV-Prior": F0_COLS + PRIOR_COLS,
            "AV-H5": F0_COLS + H5_COLS, "AV-Prior+H5": F0_COLS + PRIOR_COLS + H5_COLS}
THETA_GRID = [0.0, 100.0, 250.0, 500.0, 1000.0]


def matrix(rows, cols):
    return np.asarray([[r[c] for c in cols] for r in rows], dtype=float)


def simulate_decisions(rows, preds, theta):
    """delta-hat gating per state; returns list of per-state decisions."""
    out = []
    for st in sorted({r["state"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["state"] == st]
        pdrs_idx = next((i for i in idx if rows[i]["is_pdrs"]), None)
        if pdrs_idx is None:
            continue
        q_pdrs = preds[pdrs_idx]
        cand = [i for i in idx if rows[i]["in_top5"]]
        if not cand:
            continue
        best = min(cand, key=lambda i: preds[i] - q_pdrs)
        delta_hat = preds[best] - q_pdrs
        if delta_hat < -theta:
            choice = best
            switched = True
        else:
            choice = pdrs_idx
            switched = False
        out.append({"state": st, "episode": rows[choice]["episode"], "switched": switched,
                    "dj": float(rows[choice]["label_dj"]),
                    "pred_best_delta": float(delta_hat)})
    return out


def agg_decisions(decisions):
    djs = [d["dj"] for d in decisions]
    return {
        "states": len(decisions),
        "mean_dj_ms": round(statistics.fmean(djs), 1) if djs else None,
        "improved": sum(1 for d in djs if d < -1e-9),
        "worsened": sum(1 for d in djs if d > 1e-9),
        "tied": sum(1 for d in djs if abs(d) <= 1e-9),
        "actual_switches": sum(1 for d in decisions if d["switched"]),
        "worst_deterioration_ms": round(max(djs), 1) if djs else None,
        "best_improvement_ms": round(min(djs), 1) if djs else None,
        "mean_deterioration_ms": round(statistics.fmean([d for d in djs if d > 1e-9]), 1) if any(d > 1e-9 for d in djs) else 0.0,
        "worst_decile_mean_ms": (round(statistics.fmean(sorted(djs)[-max(1, len(djs) // 10):]), 1) if djs else None),
    }


def main() -> int:
    rows = json.loads(DATASET.read_text(encoding="utf-8"))
    train = [r for r in rows if r["partition"] == "train"]
    dev = [r for r in rows if r["partition"] == "dev"]
    print(f"dataset: train {len(train)} rows / {len({r['state'] for r in train})} states / "
          f"{len({r['episode'] for r in train})} episodes | dev {len(dev)} rows / "
          f"{len({r['state'] for r in dev})} states / {len({r['episode'] for r in dev})} episodes")

    # P4: H5 feature diagnostics
    p4 = {}
    for c in H5_COLS + PRIOR_COLS:
        vals = np.asarray([r[c] for r in rows], dtype=float)
        nz = float(np.mean(np.abs(vals) > 1e-9))
        p4[c] = {"nonzero_frac": round(nz, 3), "mean": round(float(vals.mean()), 2),
                 "std": round(float(vals.std()), 2)}
    spread = {}
    for c in H5_COLS:
        s = []
        for st in {r["state"] for r in dev}:
            v = [r[c] for r in dev if r["state"] == st]
            if v:
                s.append(float(np.std(v)))
        spread[c] = round(statistics.fmean(s), 2) if s else None
    p4["_per_state_std_mean"] = spread
    print("P4 H5 diagnostics:", json.dumps({k: v for k, v in p4.items() if k != "_per_state_std_mean"}, ensure_ascii=False))

    results = {}
    for vname, cols in VARIANTS.items():
        Xtr_raw, Xdv_raw = matrix(train, cols), matrix(dev, cols)
        ytr = np.asarray([r["label_dj"] for r in train], dtype=float)
        groups = np.asarray([r["episode"] for r in train])
        scaler = StandardScaler().fit(Xtr_raw)
        Xtr, Xdv = scaler.transform(Xtr_raw), scaler.transform(Xdv_raw)

        for mname in ("gbdt", "ridge"):
            ctor = (lambda: HistGradientBoostingRegressor(max_iter=200, max_depth=3,
                                                          learning_rate=0.05, random_state=0)) \
                if mname == "gbdt" else (lambda: Ridge(alpha=10.0))
            # P3: train-side OOF theta calibration (GroupKFold by episode)
            oof = np.zeros(len(train))
            gkf = GroupKFold(n_splits=4)
            for tr_idx, te_idx in gkf.split(Xtr_raw, ytr, groups):
                sc = StandardScaler().fit(Xtr_raw[tr_idx])
                m = ctor()
                m.fit(sc.transform(Xtr_raw[tr_idx]), ytr[tr_idx])
                oof[te_idx] = m.predict(sc.transform(Xtr_raw[te_idx]))
            theta_curve = {}
            for th in THETA_GRID:
                dec = simulate_decisions(train, oof, th)
                a = agg_decisions(dec)
                theta_curve[th] = {"mean_dj_ms": a["mean_dj_ms"], "actual_switches": a["actual_switches"],
                                   "worsened": a["worsened"], "worst_deterioration_ms": a["worst_deterioration_ms"]}
            # P0 fix (advisor): negative dJ = improvement -> choose the MINIMUM
            # OOF mean; "always PDRS" (never switch, mean 0) is a candidate too.
            candidates = {th: theta_curve[th]["mean_dj_ms"] for th in THETA_GRID
                          if theta_curve[th]["mean_dj_ms"] is not None}
            candidates["always_pdrs"] = 0.0
            best_key = min(candidates, key=lambda k: candidates[k])
            best_theta = float("inf") if best_key == "always_pdrs" else float(best_key)
            # final model on all train, evaluate on dev with the selected theta
            model = ctor()
            model.fit(Xtr, ytr)
            t0 = time.perf_counter()
            pred_dev = model.predict(Xdv)
            inf_ms = (time.perf_counter() - t0) / max(1, len({r["state"] for r in dev})) * 1000.0
            dec = simulate_decisions(dev, pred_dev, best_theta)
            a = agg_decisions(dec)
            per_ep = {}
            for d in dec:
                per_ep.setdefault(d["episode"], []).append(d["dj"])
            a["per_episode_mean_dj"] = {k: round(statistics.fmean(v), 1) for k, v in sorted(per_ep.items())}
            a["theta"] = ("always_pdrs" if best_theta == float("inf") else best_theta)
            a["theta_selection_oof"] = {str(k): v for k, v in candidates.items()}
            a["theta_curve_train_oof"] = theta_curve
            a["inference_ms_per_state"] = round(inf_ms, 3)
            results[f"{vname}|{mname}"] = a
            print(f"{vname}|{mname}: theta*={best_theta:.0f} dev Δmean={a['mean_dj_ms']}ms "
                  f"({a['improved']}/{a['worsened']}/{a['tied']}) switches={a['actual_switches']}/{a['states']} "
                  f"worst={a['worst_deterioration_ms']} inf={inf_ms:.2f}ms")

    # reference: ceiling on dev states
    ceilings = []
    for st in {r["state"] for r in dev}:
        idx = [i for i, r in enumerate(dev) if r["state"] == st and r["in_top5"]]
        if idx:
            ceilings.append(min(dev[i]["label_dj"] for i in idx))
    report = {
        "schema": "fast-action-value-v1-eval",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "train_states": len({r["state"] for r in train}), "dev_states": len({r["state"] for r in dev}),
        "oracle_top5_ceiling_mean": round(statistics.fmean(ceilings), 1) if ceilings else None,
        "oracle_top5_improving_states": sum(1 for c in ceilings if c < -1e-9),
        "p4_diagnostics": p4,
        "results": results,
    }
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("evidence:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
