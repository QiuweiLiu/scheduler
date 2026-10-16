#!/usr/bin/env python3
"""Fast Action-Value v0 pipeline (advisor route-3, P1+P2, 2026-10-16).

P1: label generation.  States from the formal workload r150_b15 (train-source
episodes = TRAIN partition, validation-source episodes = DEV partition ->
strict template isolation).  Per state: probe pool + PDRS baseline, screen
Top-5 (PDRS kept), per-pool-action forced rollout cost -> label
dJ = cost(a) - cost(PDRS) (negative = better than PDRS).  Features use only
decision-time visible information; three feature blocks: F0 (current state),
Prior (historical aggregate remaining work), H5 (instance predicted chain).

P2: models Ridge / Huber / GBDT per feature block; regression metrics on DEV;
decision simulation on DEV states (Top-5 candidates, switch only when the
predicted dJ < 0, PDRS fallback otherwise); oracle-in-Top5 ceiling reference.

No evaluation truth is used by any decision rule; labels come from the
offline counterfactual rollouts (forced action + PDRS continuation).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import milp_v0_tests as T  # noqa: E402
import milp_v0_performance as P  # noqa: E402
from tracing.analysis.des_rollout import build_planning_episode, run_planning  # noqa: E402
from tracing.analysis.profile_contract import apply_real_workload_profile_contract  # noqa: E402
from tracing.analysis.workload_v02_simulator import read_jsonl, simulate_episode  # noqa: E402
from tracing.scheduling.milp_v0 import _screen_anchors  # noqa: E402

FORMAL = ROOT / "experiments/EXP-20261014_burst64_v1/high_contention"
TAIL_TABLE = ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/tail_estimator_train_table.json"
FRACTIONS = [0.45, 0.55, 0.65, 0.75, 0.85]
KA = 5


def load_tail_table():
    data = json.loads(TAIL_TABLE.read_text(encoding="utf-8"))
    meta = {}
    for line in (ROOT / "results/processed/r7_workload_v041_ontology_no_run_container"
                 / "job_templates_r7_v041.jsonl").read_text(encoding="utf-8").strip().split("\n"):
        row = json.loads(line)
        if str(row.get("split")) != "train":
            continue
        for node in row.get("nodes") or []:
            if node.get("execution_lane") == "gpu" and node.get("model_id"):
                meta.setdefault(str(node["model_id"]), str(row.get("model_stack_id") or ""))
    return {"table": data["table"], "model_to_stack": meta}


def collect_partition(episodes, partition, tm, stats, artifacts, config, tail_table):
    rows = []
    for ep in episodes:
        summary, _ev = simulate_episode(ep, tm, "pdrs_resident", train_stats=stats,
                                        future_artifacts=artifacts, future_horizon=5,
                                        extension_config=config,
                                        policy_context={"activation": {}}, collect_events=False)
        for frac in FRACTIONS:
            at = round(frac * float(summary["makespan_ms"]), 1)
            capture = {"at_ms_list": [at], "snapshots": []}
            simulate_episode(ep, tm, "pdrs_resident", train_stats=stats,
                             future_artifacts=artifacts, future_horizon=5,
                             extension_config=config,
                             policy_context={"activation": {}, "des_capture": capture},
                             collect_events=False)
            snaps = capture.get("snapshots") or []
            if not snaps:
                continue
            snap = snaps[0]
            jobs_meta = [{"job_instance_id": j["job_instance_id"], "template_id": j["template_id"]}
                         for j in ep["jobs"]]
            plan_episode = build_planning_episode(ep, snap, include_arrivals=True)
            _s, _e, pctx = run_planning(
                plan_episode, tm, future_artifacts=artifacts, train_stats=stats,
                extension_config=config, policy="des_rollout", forced_action=None,
                collect_events=False, extra_context={"des_probe_pool": True})
            pool = sorted({(str(r[1]), int(r[2])) for r in (pctx.get("des_first_pool") or [])})
            baseline = tuple(pctx.get("des_first_baseline") or ())
            if not pool or not baseline:
                continue
            # P0 guard: the probe's first decision must be the captured state
            probe_now = float((pctx.get("des_first_state") or {}).get("now_ms", -1.0))
            if abs(probe_now - float(snap["now_ms"])) > 1e-6:
                continue
            obs = T.obs_from_snapshot(snap, jobs_meta, tm, stats, artifacts)
            masked = replace(obs, legal_first=frozenset(pool))
            screened = _screen_anchors(masked, KA, frozenset({baseline[0]}))
            assert baseline in pool, "PDRS action missing from pool"
            assert baseline[0] in screened, "PDRS anchor missing from screen"

            costs = {}
            for node, gpu in pool:
                s2, _e2, _c = run_planning(
                    plan_episode, tm, future_artifacts=artifacts, train_stats=stats,
                    extension_config=config, policy="des_rollout",
                    forced_action={"node_key": node, "gpu_index": gpu}, collect_events=False)
                costs[(node, gpu)] = float(s2["mean_completion_ms"])
            pdrs_cost = costs[baseline]
            ready_by_id = {str(it.node_id): it for it in masked.ready}
            n_ready_gpu = sum(1 for it in masked.ready if it.lane == "gpu")
            n_ready_total = len(masked.ready)
            f0_order = sorted([it for it in masked.ready if it.lane == "gpu"],
                              key=lambda a: (a.priority, a.ready_since_ms, a.node_id))
            f0_rank = {str(it.node_id): i + 1 for i, it in enumerate(f0_order)}

            for (node, gpu) in pool:
                it = ready_by_id.get(node)
                if it is None:
                    continue
                gpu_row = masked.gpus[int(gpu)]
                resident_mb = sum(gpu_row.resident.values())
                stack = tail_table["model_to_stack"].get(str(it.model_id))
                prior_rem, prior_term = None, None
                if stack:
                    row_t = tail_table["table"].get(f"{stack}|pos{int(it.sequence_index) // 5 * 5}") or {}
                    prior_rem = (row_t.get("remaining_after_d") or {}).get("0")
                    prior_term = row_t.get("terminated_within_h5_frac")
                chain = masked.chains.get(it.node_id, ())
                h5_compute = sum(max(0.1, s.runtime_ms - s.load_ms) for s in chain)
                h5_load = sum(float(s.load_ms) for s in chain)
                h5_models = {str(s.model_id) for s in chain if s.model_id}
                feat = {
                    "cand_runtime_ms": float(it.runtime_p50_ms),
                    "cand_load_ms": float(it.load_p50_ms),
                    "cand_mem_mb": float(it.memory_p95_mb),
                    "cand_wait_ms": float(masked.now_ms - it.ready_since_ms),
                    "cand_seq_idx": float(it.sequence_index),
                    "gpu_queue_ms": float(max(0.0, gpu_row.available_at_ms - masked.now_ms)),
                    "gpu_free_mb": float(gpu_row.capacity_mb - resident_mb),
                    "gpu_model_resident": 1.0 if str(it.model_id) in gpu_row.resident else 0.0,
                    "n_ready_gpu": float(n_ready_gpu),
                    "n_ready_total": float(n_ready_total),
                    "f0_rank": float(f0_rank.get(node, 0)),
                    "prior_remaining_ms": float(prior_rem) if prior_rem is not None else -1.0,
                    "prior_terminated_frac": float(prior_term) if prior_term is not None else -1.0,
                    "h5_chain_compute_ms": float(h5_compute),
                    "h5_chain_len": float(len(chain)),
                    "h5_model_switches": float(len(h5_models)),
                    "h5_load_sum_ms": float(h5_load),
                }
                rows.append({
                    "partition": partition, "episode": ep["episode_id"], "state": f"{ep['episode_id']}@{frac}",
                    "now_ms": snap["now_ms"], "action": [node, int(gpu)], "is_pdrs": (node, int(gpu)) == baseline,
                    "in_top5": node in screened,
                    "label_dj": round(costs[(node, gpu)] - pdrs_cost, 3),
                    **feat,
                })
            print(f"  [{partition}] {ep['episode_id'][-4:]}@{frac}: pool={len(pool)} rows={sum(1 for r in rows if r['state'].endswith(f'@{frac}') and r['episode'] == ep['episode_id'])}")
    return rows


F0_COLS = ["cand_runtime_ms", "cand_load_ms", "cand_mem_mb", "cand_wait_ms", "cand_seq_idx",
           "gpu_queue_ms", "gpu_free_mb", "gpu_model_resident", "n_ready_gpu", "n_ready_total", "f0_rank"]
PRIOR_COLS = ["prior_remaining_ms", "prior_terminated_frac"]
H5_COLS = ["h5_chain_compute_ms", "h5_chain_len", "h5_model_switches", "h5_load_sum_ms"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-episodes", type=int, default=8)
    ap.add_argument("--dev-episodes", type=int, default=4)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/av_v0_report.json")
    ap.add_argument("--dataset", type=Path,
                    default=ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/av_v0_dataset.json")
    args = ap.parse_args()

    config_path = ROOT / "experiments/EXP-20261012_milp_v0/artifacts/serial_prefetch_v1_extension.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _serial, templates, _n, stats, artifacts = P.load_split("confirm", 1, config_path)
    tm = {k: replace(v, split="train") for k, v in templates.items()}
    tail_table = load_tail_table()

    t0 = time.perf_counter()
    train_eps = read_jsonl(FORMAL / "workload_burst64_r150_b15_train.jsonl")[: args.train_episodes]
    dev_eps = read_jsonl(FORMAL / "workload_burst64_r150_b15_validation.jsonl")[: args.dev_episodes]
    for row in train_eps + dev_eps:
        row["split"] = "train"
    tm, train_eps = apply_real_workload_profile_contract(tm, train_eps, config)
    tm, dev_eps = apply_real_workload_profile_contract(tm, dev_eps, config)

    print("P1: generating labels (train-source episodes)...")
    train_rows = collect_partition(train_eps, "train", tm, stats, artifacts, config, tail_table)
    print(f"P1: train rows={len(train_rows)}")
    print("P1: generating labels (validation-source episodes)...")
    dev_rows = collect_partition(dev_eps, "dev", tm, stats, artifacts, config, tail_table)
    print(f"P1: dev rows={len(dev_rows)}")
    print(f"label generation wall: {time.perf_counter() - t0:.0f}s")

    all_rows = train_rows + dev_rows
    args.dataset.write_text(json.dumps(all_rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    def matrix(rows, cols):
        return np.asarray([[r[c] for c in cols] for r in rows], dtype=float)

    def labels(rows):
        return np.asarray([r["label_dj"] for r in rows], dtype=float)

    variants = {"AV-F0": F0_COLS, "AV-Prior": F0_COLS + PRIOR_COLS, "AV-H5": F0_COLS + H5_COLS}
    models = {
        "ridge": lambda: Ridge(alpha=10.0),
        "huber": lambda: HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=500),
        "gbdt": lambda: HistGradientBoostingRegressor(max_iter=200, max_depth=3, learning_rate=0.05,
                                                      random_state=0),
    }
    results = {}
    for vname, cols in variants.items():
        Xtr_raw, Xdv_raw = matrix(train_rows, cols), matrix(dev_rows, cols)
        ytr, ydv = labels(train_rows), labels(dev_rows)
        scaler = StandardScaler().fit(Xtr_raw)
        Xtr, Xdv = scaler.transform(Xtr_raw), scaler.transform(Xdv_raw)
        for mname, ctor in models.items():
            model = ctor()
            model.fit(Xtr, ytr)
            pred_dev = model.predict(Xdv)
            # regression metrics on dev samples
            mae = float(np.mean(np.abs(pred_dev - ydv)))
            rho_by_state = []
            for st in sorted({r["state"] for r in dev_rows}):
                idx = [i for i, r in enumerate(dev_rows) if r["state"] == st]
                if len(idx) >= 3 and len(set(ydv[idx])) > 1 and len(set(pred_dev[idx])) > 1:
                    rho, _ = spearmanr(pred_dev[idx], ydv[idx])
                    if not np.isnan(rho):
                        rho_by_state.append(float(rho))
            # decision simulation on dev states (Top-5, switch only if pred < 0)
            decisions = []
            for st in sorted({r["state"] for r in dev_rows}):
                idx = [i for i, r in enumerate(dev_rows) if r["state"] == st and r["in_top5"]]
                if not idx:
                    continue
                p = pred_dev[idx]
                j = idx[int(np.argmin(p))]
                if p.min() < 0.0:
                    decisions.append({"state": st, "switched": True, "dj": float(ydv[j])})
                else:
                    decisions.append({"state": st, "switched": False, "dj": 0.0})
            djs = [d["dj"] for d in decisions]
            # inference timing: predict for 5 candidates
            t_inf = time.perf_counter()
            for _ in range(20):
                model.predict(Xdv[:5])
            inf_ms = (time.perf_counter() - t_inf) / 20 * 1000.0
            results[f"{vname}|{mname}"] = {
                "features": cols, "mae_dev": round(mae, 2),
                "spearman_mean": round(statistics.fmean(rho_by_state), 3) if rho_by_state else None,
                "spearman_states": len(rho_by_state),
                "decisions": len(decisions),
                "mean_dj_ms": round(statistics.fmean(djs), 1) if djs else None,
                "improved": sum(1 for d in djs if d < -1e-9),
                "worsened": sum(1 for d in djs if d > 1e-9),
                "tied": sum(1 for d in djs if abs(d) <= 1e-9),
                "switched_states": sum(1 for d in decisions if d["switched"]),
                "inference_ms_5cand": round(inf_ms, 3),
            }
            print(f"  {vname}|{mname}: MAE={mae:.0f} rho={results[f'{vname}|{mname}']['spearman_mean']} "
                  f"decisions Δmean={results[f'{vname}|{mname}']['mean_dj_ms']}ms "
                  f"({results[f'{vname}|{mname}']['improved']}/{results[f'{vname}|{mname}']['worsened']}/"
                  f"{results[f'{vname}|{mname}']['tied']}) inf={inf_ms:.2f}ms")

    # reference: PDRS (Δ=0) and oracle-in-Top5 ceiling on dev states
    ceilings, best_states = [], 0
    for st in sorted({r["state"] for r in dev_rows}):
        idx = [i for i, r in enumerate(dev_rows) if r["state"] == st and r["in_top5"]]
        if not idx:
            continue
        best = min(ydv[i] for i in idx)
        ceilings.append(best)
        if best < -1e-9:
            best_states += 1
    report = {
        "schema": "fast-action-value-v0-v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "train_rows": len(train_rows), "dev_rows": len(dev_rows),
        "train_states": len({r["state"] for r in train_rows}), "dev_states": len({r["state"] for r in dev_rows}),
        "train_episodes": args.train_episodes, "dev_episodes": args.dev_episodes,
        "template_isolation": "train=480-pool episodes, dev=160-pool episodes (disjoint sources)",
        "label_gen_wall_s": round(time.perf_counter() - t0, 0),
        "oracle_top5_ceiling": {"mean_dj_ms": round(statistics.fmean(ceilings), 1) if ceilings else None,
                                "improving_states": best_states, "states": len(ceilings)},
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False)[:2600])
    print("evidence:", args.out, "| dataset:", args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
