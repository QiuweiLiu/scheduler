#!/usr/bin/env python3
"""Per-action cost-prediction check (advisor-facing diagnosis).

For selected deep states: force every (screened) legal first action in the
screened MILP, record the planned objective; compute each action's TRUE cost
via forced rollouts; then compare rankings (Spearman, argmin agreement, the
chosen action's predicted vs true rank).  Explains WHY the MILP picks actions
whose true cost is worse than PDRS even when better ones are in the candidate
set.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import milp_v0_tests as T  # noqa: E402
import milp_v0_performance as P  # noqa: E402
from tracing.analysis.des_rollout import build_planning_episode, run_planning  # noqa: E402
from tracing.analysis.profile_contract import apply_real_workload_profile_contract  # noqa: E402
from tracing.analysis.workload_v02_simulator import read_jsonl, simulate_episode  # noqa: E402
from tracing.scheduling.milp_v0 import _screen_anchors, solve_v1  # noqa: E402

FORMAL = ROOT / "experiments/EXP-20261014_burst64_v1/high_contention"
TAIL_TABLE = ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/tail_estimator_train_table.json"


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=0.5)
    ap.add_argument("--ka", type=int, default=5)
    ap.add_argument("--kc", type=int, default=2)
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/cost_check_states.json")
    args = ap.parse_args()

    config_path = ROOT / "experiments/EXP-20261012_milp_v0/artifacts/serial_prefetch_v1_extension.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _serial, templates, _n, stats, artifacts = P.load_split("confirm", 1, config_path)
    tm = {k: replace(v, split="train") for k, v in templates.items()}
    episodes = read_jsonl(FORMAL / "workload_burst64_r150_b15_train.jsonl")
    for row in episodes:
        row["split"] = "train"
    tm, episodes = apply_real_workload_profile_contract(tm, episodes, config)
    tail_table = load_tail_table()

    # select the states to check (episode, fraction) pairs
    checks = [(args.episode, args.fraction)]
    report = {"schema": "jct-milp-v1-cost-check-v1",
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "states": []}
    for ep_idx, frac in checks:
        ep = episodes[ep_idx]
        summary, _ev = simulate_episode(ep, tm, "pdrs_resident", train_stats=stats,
                                        future_artifacts=artifacts, future_horizon=5,
                                        extension_config=config,
                                        policy_context={"activation": {}}, collect_events=False)
        at = round(frac * float(summary["makespan_ms"]), 1)
        capture = {"at_ms_list": [at], "snapshots": []}
        simulate_episode(ep, tm, "pdrs_resident", train_stats=stats,
                         future_artifacts=artifacts, future_horizon=5,
                         extension_config=config,
                         policy_context={"activation": {}, "des_capture": capture}, collect_events=False)
        snap = capture["snapshots"][0]
        jobs_meta = [{"job_instance_id": j["job_instance_id"], "template_id": j["template_id"]}
                     for j in ep["jobs"]]
        plan_episode = build_planning_episode(ep, snap, include_arrivals=True)
        _s, _e, pctx = run_planning(
            plan_episode, tm, future_artifacts=artifacts, train_stats=stats,
            extension_config=config, policy="des_rollout", forced_action=None,
            collect_events=False, extra_context={"des_probe_pool": True})
        pool = sorted({(str(r[1]), int(r[2])) for r in (pctx.get("des_first_pool") or [])})
        baseline = tuple(pctx.get("des_first_baseline") or ())
        obs = T.obs_from_snapshot(snap, jobs_meta, tm, stats, artifacts)
        masked = replace(obs, legal_first=frozenset(pool))
        required = frozenset({baseline[0]}) if baseline else frozenset()
        screened_ids = _screen_anchors(masked, args.ka, required)

        true_cost = {}
        for node, gpu in pool:
            s2, _ev2, _c = run_planning(
                plan_episode, tm, future_artifacts=artifacts, train_stats=stats,
                extension_config=config, policy="des_rollout",
                forced_action={"node_key": node, "gpu_index": gpu}, collect_events=False)
            true_cost[(node, gpu)] = float(s2["mean_completion_ms"])

        pred = {}
        for node, gpu in pool:
            if node not in screened_ids:
                continue
            res = solve_v1(masked, chain_depth=3, k_chains=args.kc, tail=True,
                           tail_table=tail_table, time_limit_s=args.budget,
                           k_candidates=args.ka, required_node_ids=required,
                           force_first_action=(node, gpu))
            if not res.fallback and res.objective is not None:
                pred[(node, gpu)] = float(res.objective)

        # unforced (normal) choice for reference
        res_free = solve_v1(masked, chain_depth=3, k_chains=args.kc, tail=True,
                            tail_table=tail_table, time_limit_s=args.budget,
                            k_candidates=args.ka, required_node_ids=required)
        free_choice = tuple(res_free.first_action) if (not res_free.fallback and res_free.first_action) else None

        common = sorted(set(pred) & set(true_cost))
        def ranks(d):
            order = sorted(d, key=lambda a: d[a])
            return {a: i + 1 for i, a in enumerate(order)}
        rp, rt = ranks(pred), ranks(true_cost)
        n = len(common)
        if n >= 3:
            xs = [rp[a] for a in common]
            ys = [rt[a] for a in common]
            mx, my = sum(xs) / n, sum(ys) / n
            cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            vx = sum((x - mx) ** 2 for x in xs) ** 0.5
            vy = sum((y - my) ** 2 for y in ys) ** 0.5
            rho = cov / (vx * vy) if vx * vy > 0 else None
        else:
            rho = None
        best_true = min(true_cost, key=lambda a: true_cost[a])
        best_pred = min(pred, key=lambda a: pred[a]) if pred else None
        entry = {
            "episode": ep["episode_id"], "fraction": frac, "now_ms": snap["now_ms"],
            "pool": len(pool), "screened": sorted(screened_ids),
            "pdrs_action": list(baseline) if baseline else None,
            "free_choice": list(free_choice) if free_choice else None,
            "spearman_pred_vs_true": round(rho, 3) if rho is not None else None,
            "n_common": n,
            "true_best": {"action": list(best_true), "cost": round(true_cost[best_true], 1),
                          "pred_rank": rp.get(best_true), "true_rank": rt.get(best_true)},
            "pred_best": {"action": list(best_pred), "pred_rank": 1,
                          "true_rank": rt.get(best_pred), "true_cost": round(true_cost[best_pred], 1)} if best_pred else None,
            "free_choice_detail": None,
            "actions": [
                {"action": list(a), "true_cost": round(true_cost[a], 1),
                 "pred": round(pred[a], 1) if a in pred else None,
                 "pred_rank": rp.get(a), "true_rank": rt.get(a),
                 "in_screened": a[0] in screened_ids}
                for a in sorted(true_cost, key=lambda x: true_cost[x])
            ],
        }
        if free_choice and free_choice in true_cost:
            entry["free_choice_detail"] = {
                "action": list(free_choice), "true_rank": rt.get(free_choice),
                "pred_rank": rp.get(free_choice), "true_cost": round(true_cost[free_choice], 1),
                "is_pdrs": free_choice == baseline,
            }
        report["states"].append(entry)
        print(f"{entry['episode']}@{frac}: pool={len(pool)} screened={len(screened_ids)} "
              f"spearman={entry['spearman_pred_vs_true']} | true_best pred_rank="
              f"{entry['true_best']['pred_rank']}/{n} | free_choice true_rank="
              f"{entry['free_choice_detail']['true_rank'] if entry['free_choice_detail'] else None}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("evidence:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
