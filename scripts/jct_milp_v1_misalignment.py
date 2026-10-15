#!/usr/bin/env python3
"""Objective-misalignment test: does the current MILP's plan value rank actions
like the true cost does?

Per state, per legal action: (a) plan value = solve_v0a with legal_first={action}
(the model's own valuation of forcing that first action), (b) true cost = the
forced-action rollout with main-line continuation.  Compare rankings.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path("/Volumes/Lenovo/scheduler")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import milp_v0_tests as T  # noqa: E402
import milp_v0_performance as P  # noqa: E402
from tracing.analysis.des_rollout import build_planning_episode, run_planning  # noqa: E402
from tracing.analysis.profile_contract import apply_real_workload_profile_contract  # noqa: E402
from tracing.analysis.workload_v02_simulator import read_jsonl  # noqa: E402
from tracing.scheduling.milp_v0 import solve_v0a  # noqa: E402


def rankdata(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    for position, index in enumerate(order):
        ranks[index] = float(position)
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    n = len(a)
    mean = (n - 1) / 2.0
    num = sum((x - mean) * (y - mean) for x, y in zip(ra, rb))
    den = sum((x - mean) ** 2 for x in ra) ** 0.5 * sum((y - mean) ** 2 for y in rb) ** 0.5
    return num / den if den else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--burst", action="store_true", help="use Burst64 states instead of natural")
    ap.add_argument("--states", type=int, default=12)
    ap.add_argument("--budget", type=float, default=0.05)
    args = ap.parse_args()

    config_path = ROOT / "experiments/EXP-20261012_milp_v0/artifacts/serial_prefetch_v1_extension.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    serial, templates, _n, stats, artifacts = P.load_split("confirm", 1, config_path)
    if args.burst:
        tm = {k: replace(v, split="train") for k, v in templates.items()}
        episodes = read_jsonl(ROOT / "experiments/EXP-20261014_burst64_v1/artifacts"
                              / "workload_burst64_r110_b10.jsonl")
        for episode in episodes:
            episode["split"] = "train"
        tm, episodes = apply_real_workload_profile_contract(tm, episodes, config)
    else:
        tm, episodes = templates, P.load_split("confirm", 6, config_path)[2]

    states = T.capture_states(episodes[:4], tm, stats, artifacts, config, cap=6)
    rows = []
    for ep, jobs_meta, snap in states:
        if len(rows) >= args.states:
            break
        plan_episode = build_planning_episode(ep, snap, include_arrivals=True)
        _s, _e, pctx = run_planning(
            plan_episode, tm, future_artifacts=artifacts, train_stats=stats,
            extension_config=config, policy="des_rollout", forced_action=None,
            collect_events=False, extra_context={"des_probe_pool": True})
        if abs(float((pctx.get("des_first_state") or {}).get("now_ms", -1)) - float(snap["now_ms"])) > 1e-6:
            continue
        pool = sorted({(str(r[1]), int(r[2])) for r in (pctx.get("des_first_pool") or [])})
        if len(pool) < 2:
            continue
        obs = T.obs_from_snapshot(snap, jobs_meta, tm, stats, artifacts)
        plan_vals, true_costs = [], []
        for node, gpu in pool:
            res = solve_v0a(replace(obs, legal_first=frozenset({(node, gpu)})), time_limit_s=args.budget)
            plan_vals.append(float(res.objective) if res.objective is not None else None)
            summary, _ev, _c = run_planning(
                plan_episode, tm, future_artifacts=artifacts, train_stats=stats,
                extension_config=config, policy="des_rollout",
                forced_action={"node_key": node, "gpu_index": gpu}, collect_events=False)
            true_costs.append(float(summary["mean_completion_ms"]))
        valid = [i for i, v in enumerate(plan_vals) if v is not None]
        if len(valid) < 2:
            continue  # need >=2 solvable actions to rank
        plan_sub = [plan_vals[i] for i in valid]
        true_sub = [true_costs[i] for i in valid]
        rho = spearman(plan_sub, true_sub)
        best_true = min(range(len(valid)), key=lambda i: true_sub[i])
        best_plan = min(range(len(valid)), key=lambda i: plan_sub[i])
        plan_vals, true_costs = plan_sub, true_sub
        pool = [pool[i] for i in valid]
        rows.append({
            "episode": ep["episode_id"], "now_ms": snap["now_ms"], "actions": len(pool),
            "spearman_plan_vs_true": round(rho, 3),
            "plan_argmin_is_true_argmin": bool(best_plan == best_true),
            "plan_argmin_true_regret": round(true_costs[best_plan] - true_costs[best_true], 1),
            "true_best_regret_of_worst": round(max(true_costs) - true_costs[best_true], 1),
        })
    n = len(rows)
    report = {
        "schema": "jct-milp-v1-misalignment-test-v1",
        "source": "burst64_r110_b10" if args.burst else "natural_confirm",
        "states": n, "budget_s": args.budget,
        "mean_spearman": round(sum(r["spearman_plan_vs_true"] for r in rows) / max(1, n), 3),
        "plan_argmin_matches_true_frac": round(sum(1 for r in rows if r["plan_argmin_is_true_argmin"]) / max(1, n), 3),
        "mean_plan_argmin_regret_ms": round(sum(r["plan_argmin_true_regret"] for r in rows) / max(1, n), 1),
        "rows": rows,
    }
    out = ROOT / ("experiments/EXP-20261015_jct_milp_v1/artifacts/misalignment_"
                  + ("burst64" if args.burst else "natural") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print("evidence:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
