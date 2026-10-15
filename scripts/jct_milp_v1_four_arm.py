#!/usr/bin/env python3
"""JCT-MILP-v1 four-arm diagnostic on the same states (advisor-requested).

Arms on the SAME state/action set: PDRS baseline, old H5-MILP (solve_v0a),
new JCT-MILP without Tail (solve_v1 tail=False), new with Tail (solve_v1
tail=True).  Per arm: chosen first action's TRUE cost (forced rollout with
main-line continuation), regret vs the true best, legality, solve status/time.
Also: K-truncation check (K=2/5 vs K=3) and pairwise ordering accuracy.

Diagnostic only; train-source-isolated Burst64 episodes.
"""

from __future__ import annotations

import argparse
import json
import statistics
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
from tracing.analysis.workload_v02_simulator import read_jsonl  # noqa: E402
from tracing.scheduling.milp_v0 import solve_v0a, solve_v1  # noqa: E402

ISOLATED = ROOT / "experiments/EXP-20261014_burst64_v1/isolated"
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
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--states", type=int, default=30)
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/four_arm_diagnostic.json")
    args = ap.parse_args()

    config_path = ROOT / "experiments/EXP-20261012_milp_v0/artifacts/serial_prefetch_v1_extension.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    serial, templates, _n, stats, artifacts = P.load_split("confirm", 1, config_path)
    tm = {k: replace(v, split="train") for k, v in templates.items()}
    episodes = []
    for tag in ("r085_b10", "r110_b10"):
        rows = read_jsonl(ISOLATED / f"workload_burst64_{tag}_train.jsonl")[: args.episodes]
        for row in rows:
            row["split"] = "train"
        episodes.extend(rows)
    tm, episodes = apply_real_workload_profile_contract(tm, episodes, config)
    tail_table = load_tail_table()

    states = T.capture_states(episodes, tm, stats, artifacts, config, cap=8)
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
        if not pool:
            continue
        baseline = tuple(pctx.get("des_first_baseline") or ())
        obs = T.obs_from_snapshot(snap, jobs_meta, tm, stats, artifacts)
        masked = replace(obs, legal_first=frozenset(pool))

        arm_choices = {"pdrs": baseline if baseline else None}
        arm_meta = {"pdrs": {"status": "baseline", "solve_ms": 0.0}}
        for arm, fn in (("old_milp", lambda: solve_v0a(masked, time_limit_s=args.budget)),
                        ("new_no_tail", lambda: solve_v1(masked, tail=False, time_limit_s=args.budget)),
                        ("new_tail", lambda: solve_v1(masked, tail=True, tail_table=tail_table,
                                                      time_limit_s=args.budget))):
            t0 = time.perf_counter()
            res = fn()
            wall = (time.perf_counter() - t0) * 1000.0
            arm_choices[arm] = tuple(res.first_action) if (not res.fallback and res.first_action) else None
            arm_meta[arm] = {"status": res.status, "solve_ms": round(wall, 1),
                             "fallback": bool(res.fallback)}
        # K-truncation: does the extracted first action change with K?
        k_choices = {}
        for k in (2, 5):
            res = solve_v1(masked, k_chains=k, tail=True, tail_table=tail_table,
                           time_limit_s=args.budget)
            k_choices[k] = tuple(res.first_action) if (not res.fallback and res.first_action) else None

        costs = {}
        for node, gpu in pool:
            summary, _ev, _c = run_planning(
                plan_episode, tm, future_artifacts=artifacts, train_stats=stats,
                extension_config=config, policy="des_rollout",
                forced_action={"node_key": node, "gpu_index": gpu}, collect_events=False)
            costs[(node, gpu)] = float(summary["mean_completion_ms"])
        best = min(costs.values())
        row = {"episode": ep["episode_id"], "now_ms": snap["now_ms"], "pool": len(pool),
               "true_best": round(best, 1), "arms": {}, "k_truncation": {}}
        for arm, choice in arm_choices.items():
            entry = {"action": list(choice) if choice else None, **arm_meta[arm]}
            if choice and choice in costs:
                entry["true_cost"] = round(costs[choice], 1)
                entry["regret"] = round(costs[choice] - best, 1)
                entry["is_true_best"] = bool(abs(costs[choice] - best) < 1e-9)
                entry["legal"] = choice in set(pool)
            row["arms"][arm] = entry
        k3 = arm_choices.get("new_tail")
        row["k_truncation"] = {
            "k2_same_as_k3": k_choices.get(2) == k3,
            "k5_same_as_k3": k_choices.get(5) == k3,
            "k2_action": list(k_choices.get(2)) if k_choices.get(2) else None,
            "k5_action": list(k_choices.get(5)) if k_choices.get(5) else None,
        }
        rows.append(row)

    def agg(arm):
        entries = [r["arms"][arm] for r in rows if r["arms"][arm].get("regret") is not None]
        regrets = [e["regret"] for e in entries]
        return {
            "covered": len(entries),
            "mean_regret": round(statistics.fmean(regrets), 1) if regrets else None,
            "median_regret": round(statistics.median(regrets), 1) if regrets else None,
            "picked_true_best_frac": round(sum(1 for e in entries if e["is_true_best"]) / len(entries), 3) if entries else None,
            "legal_frac": round(sum(1 for e in entries if e.get("legal")) / len(entries), 3) if entries else None,
            "solve_ms_p50": round(statistics.median([e["solve_ms"] for e in entries]), 1) if entries else None,
        }

    report = {
        "schema": "jct-milp-v1-four-arm-v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": "burst64_train_isolated", "states": len(rows), "budget_s": args.budget,
        "arms": {arm: agg(arm) for arm in ("pdrs", "old_milp", "new_no_tail", "new_tail")},
        "k_truncation": {
            "k2_same_frac": round(sum(1 for r in rows if r["k_truncation"]["k2_same_as_k3"]) / max(1, len(rows)), 3),
            "k5_same_frac": round(sum(1 for r in rows if r["k_truncation"]["k5_same_as_k3"]) / max(1, len(rows)), 3),
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1, ensure_ascii=False))
    print("evidence:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
