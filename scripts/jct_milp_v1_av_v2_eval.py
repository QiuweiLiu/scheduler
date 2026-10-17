#!/usr/bin/env python3
"""AV step-2 final evaluation (advisor 2026-10-16): risk gating + H5-v1
comparison + shuffle control.

Variants (all GBDT, same samples/labels/screening):
  Prior / Prior+H5v0 / Prior+H5v1 / Prior+shuffled-H5
Risk gating: theta (incl. always-PDRS) selected on TRAIN out-of-fold
predictions by minimizing  mean dJ + lambda * CVaR90(max(dJ,0));
primary lambda = 0.5 (pre-registered); lambda=0 reported for the risk-gating
ablation on Prior+H5v0.  Dev used once per variant, never for selection.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/av_v2_dataset.json"
OUT = ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/av_v2_report.json"

F0 = ["cand_runtime_ms", "cand_load_ms", "cand_mem_mb", "cand_wait_ms", "cand_seq_idx",
      "gpu_queue_ms", "gpu_free_mb", "gpu_model_resident", "n_ready_gpu", "n_ready_total", "f0_rank"]
PRIOR = ["prior_remaining_ms", "prior_terminated_frac"]
H5V0 = ["h5_chain_compute_ms", "h5_chain_len", "h5_model_switches", "h5_load_sum_ms"]
H5V1 = ["next3_not_resident", "next3_coldload_ms", "next3_resident_bytes",
        "comp_models", "comp_overlap", "comp_not_resident_any", "terminates_within3"]
THETA_GRID = [0.0, 100.0, 250.0, 500.0, 1000.0]
LAMBDAS = [0.0, 0.5]


def cvar90_positive(djs):
    pos = sorted([d for d in djs if d > 1e-9], reverse=True)
    if not pos:
        return 0.0
    k = max(1, int(np.ceil(0.1 * len(djs))))
    return float(np.mean(pos[:k]))


def decisions_for(rows, preds, theta):
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
        if preds[best] - q_pdrs < -theta:
            out.append({"state": st, "episode": rows[best]["episode"], "switched": True,
                        "dj": float(rows[best]["label_dj"])})
        else:
            out.append({"state": st, "episode": rows[pdrs_idx]["episode"], "switched": False,
                        "dj": 0.0})
    return out


def agg(dec):
    djs = [d["dj"] for d in dec]
    return {
        "states": len(dec),
        "mean_dj_ms": round(statistics.fmean(djs), 1) if djs else None,
        "improved": sum(1 for d in djs if d < -1e-9),
        "worsened": sum(1 for d in djs if d > 1e-9),
        "tied": sum(1 for d in djs if abs(d) <= 1e-9),
        "actual_switches": sum(1 for d in dec if d["switched"]),
        "worst_deterioration_ms": round(max(djs), 1) if djs else None,
        "worst_decile_loss_ms": round(cvar90_positive(djs), 1),
        "per_episode_mean_dj": {k: round(statistics.fmean(v), 1)
                                for k, v in sorted({e: [d["dj"] for d in dec if d["episode"] == e]
                                                    for e in {d["episode"] for d in dec}}.items())},
    }


def main() -> int:
    rows = json.loads(DATASET.read_text(encoding="utf-8"))
    train = [r for r in rows if r["partition"] == "train"]
    dev = [r for r in rows if r["partition"] == "dev"]
    print(f"dataset v2: train {len(train)} rows/{len({r['state'] for r in train})} states | "
          f"dev {len(dev)} rows/{len({r['state'] for r in dev})} states")

    # shuffle control: permute the H5 block at STATE level (fixed seed), applied
    # to both partitions consistently
    rng = np.random.default_rng(7)
    h5_cols = H5V0 + H5V1
    states_all = sorted({r["state"] for r in rows})
    perm = rng.permutation(len(states_all))
    mapping = {states_all[i]: states_all[perm[i]] for i in range(len(states_all))}
    by_state = {}
    for r in rows:
        by_state.setdefault(r["state"], []).append(r)
    for r in rows:
        src_state = mapping[r["state"]]
        # match by action within the source state when possible, else first row
        cands = [x for x in by_state[src_state] if x["action"] == r["action"]]
        src = cands[0] if cands else by_state[src_state][0]
        for c in h5_cols:
            r[f"shuf_{c}"] = src[c]

    variants = {
        "Prior": F0 + PRIOR,
        "Prior+H5v0": F0 + PRIOR + H5V0,
        "Prior+H5v1": F0 + PRIOR + H5V0 + H5V1,
        "Prior+H5shuf": F0 + PRIOR + [f"shuf_{c}" for c in h5_cols],
    }
    results = {}
    for vname, cols in variants.items():
        Xtr_raw = np.asarray([[r[c] for c in cols] for r in train], dtype=float)
        Xdv_raw = np.asarray([[r[c] for c in cols] for r in dev], dtype=float)
        ytr = np.asarray([r["label_dj"] for r in train], dtype=float)
        groups = np.asarray([r["episode"] for r in train])
        oof = np.zeros(len(train))
        for tr_idx, te_idx in GroupKFold(n_splits=4).split(Xtr_raw, ytr, groups):
            sc = StandardScaler().fit(Xtr_raw[tr_idx])
            m = HistGradientBoostingRegressor(max_iter=200, max_depth=3, learning_rate=0.05,
                                              random_state=0)
            m.fit(sc.transform(Xtr_raw[tr_idx]), ytr[tr_idx])
            oof[te_idx] = m.predict(sc.transform(Xtr_raw[te_idx]))
        sc_full = StandardScaler().fit(Xtr_raw)
        model = HistGradientBoostingRegressor(max_iter=200, max_depth=3, learning_rate=0.05,
                                              random_state=0)
        model.fit(sc_full.transform(Xtr_raw), ytr)
        pred_dev = model.predict(sc_full.transform(Xdv_raw))
        for lam in LAMBDAS:
            sel = {"always_pdrs": 0.0}
            for th in THETA_GRID:
                dec = decisions_for(train, oof, th)
                a = agg(dec)
                sel[th] = a["mean_dj_ms"] + lam * a["worst_decile_loss_ms"]
            best_key = min(sel, key=lambda k: sel[k])
            theta = float("inf") if best_key == "always_pdrs" else float(best_key)
            dec_dev = decisions_for(dev, pred_dev, theta)
            a = agg(dec_dev)
            a["theta"] = ("always_pdrs" if theta == float("inf") else theta)
            a["lambda"] = lam
            a["train_oof_objective"] = {str(k): round(v, 1) for k, v in sel.items()}
            results[f"{vname}|lam{lam}"] = a
            print(f"{vname:14s} lam={lam}: theta={a['theta']} dev Δmean={a['mean_dj_ms']}ms "
                  f"({a['improved']}/{a['worsened']}/{a['tied']}) switches={a['actual_switches']}/{a['states']} "
                  f"worst={a['worst_deterioration_ms']} decile={a['worst_decile_loss_ms']}")

    report = {
        "schema": "fast-action-value-v2-risk-h5v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "train_states": len({r["state"] for r in train}), "dev_states": len({r["state"] for r in dev}),
        "primary_lambda": 0.5, "shuffle_seed": 7,
        "results": results,
    }
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("evidence:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
