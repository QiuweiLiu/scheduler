#!/usr/bin/env python3
"""AV v3 final diagnostic (advisor 2026-10-16): True-H5 oracle control +
corrected shuffle.

Variants (GBDT, same samples/labels/screening, theta from train OOF):
  Prior | Prior+PredH5(fixed) | Prior+TrueH5(oracle)
at lambda in {0, 0.5}.  Shuffle control: same-capacity (the True-H5 block),
within-partition state-level permutation, 5 fixed seeds, distribution reported.
True-H5 is an OFFLINE information oracle only.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/av_v3_dataset.json"
OUT = ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/av_v3_report.json"

F0 = ["cand_runtime_ms", "cand_load_ms", "cand_mem_mb", "cand_wait_ms", "cand_seq_idx",
      "gpu_queue_ms", "gpu_free_mb", "gpu_model_resident", "n_ready_gpu", "n_ready_total", "f0_rank"]
PRIOR = ["prior_remaining_ms", "prior_terminated_frac"]
PRED_H5 = ["h5_chain_compute_ms", "h5_chain_len", "h5_model_switches", "h5_load_sum_ms",
           "p_comp_models_fixed", "p_comp_overlap_fixed"]
TRUE_H5 = ["t_next3_not_resident", "t_next3_coldload_ms", "t_next3_compute_ms",
           "t_terminates_within3", "t_comp_models", "t_comp_overlap"]
THETA_GRID = [0.0, 100.0, 250.0, 500.0, 1000.0]
SEEDS = [1, 2, 3, 4, 5]


def cvar90_pos(djs):
    # advisor fix: worst 10% of ALL states (zero losses included as zeros)
    if not djs:
        return 0.0
    k = max(1, int(np.ceil(0.1 * len(djs))))
    worst = sorted([max(d, 0.0) for d in djs], reverse=True)[:k]
    return float(np.mean(worst))


def decisions(rows, preds, theta):
    out = []
    for st in sorted({r["state"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["state"] == st]
        p_idx = next((i for i in idx if rows[i]["is_pdrs"]), None)
        if p_idx is None:
            continue
        q_p = preds[p_idx]
        cand = [i for i in idx if rows[i]["in_top5"]]
        if not cand:
            continue
        best = min(cand, key=lambda i: preds[i] - q_p)
        if preds[best] - q_p < -theta:
            out.append({"state": st, "episode": rows[best]["episode"], "switched": True,
                        "dj": float(rows[best]["label_dj"])})
        else:
            out.append({"state": st, "episode": rows[p_idx]["episode"], "switched": False, "dj": 0.0})
    return out


def agg(dec):
    djs = [d["dj"] for d in dec]
    return {"states": len(dec),
            "mean_dj_ms": round(statistics.fmean(djs), 1) if djs else None,
            "improved": sum(1 for d in djs if d < -1e-9),
            "worsened": sum(1 for d in djs if d > 1e-9),
            "tied": sum(1 for d in djs if abs(d) <= 1e-9),
            "actual_switches": sum(1 for d in dec if d["switched"]),
            "worst_deterioration_ms": round(max(djs), 1) if djs else None,
            "worst_decile_loss_ms": round(cvar90_pos(djs), 1)}


def run_variant(name, cols, train, dev, extra=None):
    def mat(rows):
        return np.asarray([[r[c] for c in cols] for r in rows], dtype=float)
    Xtr_raw, Xdv_raw = mat(train), mat(dev)
    ytr = np.asarray([r["label_dj"] for r in train], dtype=float)
    groups = np.asarray([r["episode"] for r in train])
    oof = np.zeros(len(train))
    for tr_idx, te_idx in GroupKFold(n_splits=4).split(Xtr_raw, ytr, groups):
        sc = StandardScaler().fit(Xtr_raw[tr_idx])
        m = HistGradientBoostingRegressor(max_iter=200, max_depth=3, learning_rate=0.05, random_state=0)
        m.fit(sc.transform(Xtr_raw[tr_idx]), ytr[tr_idx])
        oof[te_idx] = m.predict(sc.transform(Xtr_raw[te_idx]))
    sc_full = StandardScaler().fit(Xtr_raw)
    model = HistGradientBoostingRegressor(max_iter=200, max_depth=3, learning_rate=0.05, random_state=0)
    model.fit(sc_full.transform(Xtr_raw), ytr)
    pred_dev = model.predict(sc_full.transform(Xdv_raw))
    # ranking quality on dev
    rhos = []
    for st in sorted({r["state"] for r in dev}):
        idx = [i for i, r in enumerate(dev) if r["state"] == st and r["in_top5"]]
        if len(idx) >= 3 and len({dev[i]["label_dj"] for i in idx}) > 1:
            rho, _ = spearmanr(pred_dev[idx], [dev[i]["label_dj"] for i in idx])
            if not np.isnan(rho):
                rhos.append(float(rho))
    res = {"n_features": len(cols), "dev_rank_corr_mean": round(statistics.fmean(rhos), 3) if rhos else None}
    for lam in (0.0, 0.5):
        sel = {"always_pdrs": 0.0}
        for th in THETA_GRID:
            a = agg(decisions(train, oof, th))
            sel[th] = a["mean_dj_ms"] + lam * a["worst_decile_loss_ms"]
        key = min(sel, key=lambda k: sel[k])
        theta = float("inf") if key == "always_pdrs" else float(key)
        a = agg(decisions(dev, pred_dev, theta))
        a["theta"] = "always_pdrs" if theta == float("inf") else theta
        res[f"lambda_{lam}"] = a
    if extra:
        res.update(extra)
    return res


def main() -> int:
    rows = json.loads(DATASET.read_text(encoding="utf-8"))
    train = [r for r in rows if r["partition"] == "train"]
    dev = [r for r in rows if r["partition"] == "dev"]
    print(f"dataset v3: train {len(train)}/{len({r['state'] for r in train})} states | "
          f"dev {len(dev)}/{len({r['state'] for r in dev})} states")

    results = {}
    results["Prior"] = run_variant("Prior", F0 + PRIOR, train, dev)
    results["Prior+PredH5"] = run_variant("Prior+PredH5", F0 + PRIOR + PRED_H5, train, dev)
    results["Prior+TrueH5"] = run_variant("Prior+TrueH5", F0 + PRIOR + TRUE_H5, train, dev)
    for k, v in results.items():
        l0, l5 = v["lambda_0.0"], v["lambda_0.5"]
        print(f"{k:16s} rho={v['dev_rank_corr_mean']} | lam0: theta={l0['theta']} Δ={l0['mean_dj_ms']} "
              f"({l0['improved']}/{l0['worsened']}/{l0['tied']}) sw={l0['actual_switches']} | "
              f"lam.5: theta={l5['theta']} Δ={l5['mean_dj_ms']} sw={l5['actual_switches']}")

    # corrected shuffle: same capacity (TRUE_H5 block), within-partition, 5 seeds
    shuf_dev_means = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        tr_rows = [dict(r) for r in train]
        dv_rows = [dict(r) for r in dev]
        for part_rows in (tr_rows, dv_rows):
            states = sorted({r["state"] for r in part_rows})
            perm = rng.permutation(len(states))
            mp = {states[i]: states[perm[i]] for i in range(len(states))}
            by_st = {}
            for r in part_rows:
                by_st.setdefault(r["state"], []).append(r)
            for r in part_rows:
                src_list = by_st[mp[r["state"]]]
                src = next((x for x in src_list if x["action"] == r["action"]), src_list[0])
                for c in TRUE_H5:
                    r[f"shuf_{c}"] = src[c]
        v = run_variant("shuf", F0 + PRIOR + [f"shuf_{c}" for c in TRUE_H5], tr_rows, dv_rows)
        shuf_dev_means.append(v["lambda_0.0"]["mean_dj_ms"])
    results["shuffle_trueh5_multiseed"] = {
        "seeds": SEEDS, "lambda0_dev_mean_dj": shuf_dev_means,
        "mean": round(statistics.fmean(shuf_dev_means), 1) if shuf_dev_means else None,
        "min": min(shuf_dev_means) if shuf_dev_means else None,
        "max": max(shuf_dev_means) if shuf_dev_means else None}
    print("shuffle(trueH5) multi-seed lam0 dev Δ:", shuf_dev_means)

    report = {"schema": "fast-action-value-v3-trueh5", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "train_states": len({r["state"] for r in train}), "dev_states": len({r["state"] for r in dev}),
              "results": results}
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("evidence:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
