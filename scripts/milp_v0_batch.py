#!/usr/bin/env python3
"""MILP-v0a batch validation (advisor gate: batch A targeted + batch B coverage).

Batch A: ~30 targeted diagnostic states covering the just-fixed areas
(immediate dispatch, cold load, CPU/GPU mixed chains, busy GPUs).
Batch B: 100-300 pool states with coverage / feasibility / fallback / solve-time
statistics and stratified breakdowns.

Records per state: strata, solver status/gap/solve_ms, first action, whether the
first action is in the DES candidate pool (per-pair legality), fallback reason.
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
from tracing.analysis.des_rollout import build_planning_episode, run_planning  # noqa: E402
from tracing.scheduling.milp_v0 import _build_nodes, solve_v0a  # noqa: E402

EVIDENCE = ROOT / "experiments/EXP-20261012_milp_v0/artifacts/batch_validation.json"


def classify(obs, res, dump, nodes):
    ready = [r for r in obs.ready if r.lane == "gpu"]
    now = obs.now_ms
    free_now = [g for g in obs.gpus if g.available_at_ms <= now + 1e-9]
    # first action model residency (cold load if the model is resident nowhere)
    cold = None
    if res.first_action:
        node_key = res.first_action[0]
        model = next((n.model_id for n in nodes if n.key == node_key), None)
        if model:
            cold = not any(model in g.resident for g in obs.gpus)
    chain_has_cpu = any(step.lane == "cpu" for steps in obs.chains.values() for step in steps)
    h5_len = max((len(steps) for steps in obs.chains.values()), default=0)
    mem_used = max((sum(g.resident.values()) / max(1.0, g.capacity_mb) for g in obs.gpus), default=0.0)
    return {
        "ready_gpu": len(ready),
        "free_gpus_now": len(free_now),
        "cold_load": cold,
        "chain_has_cpu": chain_has_cpu,
        "h5_len": h5_len,
        "mem_pressure": round(mem_used, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--cap", type=int, default=4)
    parser.add_argument("--batch-a", type=int, default=30)
    parser.add_argument("--budget", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=EVIDENCE)
    args = parser.parse_args()

    serial, templates, episodes, stats, artifacts = T.load_fixtures(args.episodes)
    states = T.capture_states(episodes, templates, stats, artifacts, serial, cap=args.cap)
    print(f"captured states: {len(states)} (episodes={args.episodes}, cap={args.cap})")

    rows = []
    for ep, jobs_meta, snap in states:
        obs = T.obs_from_snapshot(snap, jobs_meta, templates, stats, artifacts)
        if not [r for r in obs.ready if r.lane == "gpu"]:
            continue
        # Advisor P0: the DES probe runs FIRST so its candidate pool becomes the
        # legality mask for the immediate dispatch; the PDRS baseline first
        # action is recorded on the same pool for change-type classification.
        plan_episode = build_planning_episode(ep, snap, include_arrivals=True)
        _s, _e, pctx = run_planning(
            plan_episode, templates, future_artifacts=artifacts, train_stats=stats,
            extension_config=serial, policy="des_rollout", forced_action=None,
            collect_events=False, extra_context={"des_probe_pool": True})
        probe_now = float((pctx.get("des_first_state") or {}).get("now_ms", -1.0))
        pool = {(str(r[1]), int(r[2])) for r in (pctx.get("des_first_pool") or [])}
        if abs(probe_now - float(snap["now_ms"])) > 1e-6:
            pool = set()  # no dispatchable moment at the snapshot time
        obs = replace(obs, legal_first=frozenset(pool))
        nodes = _build_nodes(obs)
        dump: dict = {}
        res = solve_v0a(obs, time_limit_s=args.budget, debug_dump=dump)
        strata = classify(obs, res, dump, nodes)
        row = {
            "episode": ep["episode_id"], "now_ms": snap["now_ms"],
            "status": res.status, "fallback": res.fallback, "reason": res.reason,
            "solve_ms": round(res.solve_ms, 2), "gap": res.gap,
            "objective": res.objective, "first_action": list(res.first_action) if res.first_action else None,
            "planned_nodes": res.planned_nodes,
            "mask_size": len(pool),
            **strata,
        }
        # per-pair legality of the first action via the DES probe pool
        if not res.fallback and res.first_action:
            row["first_in_des_pool"] = tuple(res.first_action) in pool
            baseline = pctx.get("des_first_baseline")
            if baseline:
                milp_node, milp_gpu = str(res.first_action[0]), int(res.first_action[1])
                base_node, base_gpu = str(baseline[0]), int(baseline[1])
                if milp_node != base_node:
                    row["change_type"] = "ordering"
                elif milp_gpu != base_gpu:
                    row["change_type"] = "placement"
                else:
                    row["change_type"] = "same"
                row["baseline_action"] = [base_node, base_gpu]
        rows.append(row)

    # batch A: targeted selection (cover strata)
    categories = {
        "both_busy": [r for r in rows if r["free_gpus_now"] == 0],
        "one_free": [r for r in rows if r["free_gpus_now"] == 1],
        "both_free": [r for r in rows if r["free_gpus_now"] >= 2],
        "cold_load": [r for r in rows if r.get("cold_load")],
        "cpu_chain": [r for r in rows if r.get("chain_has_cpu")],
        "fallback": [r for r in rows if r["fallback"]],
    }
    batch_a = []
    seen = set()
    cat_iters = {name: iter(cat_rows) for name, cat_rows in categories.items()}
    remaining = list(categories)
    while len(batch_a) < args.batch_a and remaining:
        for name in list(remaining):
            row = next(cat_iters[name], None)
            if row is None:
                remaining.remove(name)
                continue
            key = (row["episode"], row["now_ms"])
            if key in seen:
                continue
            seen.add(key)
            batch_a.append({"category": name, **row})
            if len(batch_a) >= args.batch_a:
                break

    # aggregate stats
    eligible = len(rows)
    fallbacks = [r for r in rows if r["fallback"]]
    executed = [r for r in rows if not r["fallback"]]
    solve_times = [r["solve_ms"] for r in rows]
    illegal = [r for r in executed if r.get("first_in_des_pool") is False]
    status_counts: dict = {}
    for r in rows:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    reason_counts: dict = {}
    for r in fallbacks:
        reason_counts[r["reason"]] = reason_counts.get(r["reason"], 0) + 1
    strata_summary = {
        "ready_gpu": statistics.fmean(r["ready_gpu"] for r in rows) if rows else 0.0,
        "free_gpus_now_mean": statistics.fmean(r["free_gpus_now"] for r in rows) if rows else 0.0,
        "cold_load_rate": (sum(1 for r in rows if r.get("cold_load")) / eligible) if eligible else 0.0,
        "cpu_chain_rate": (sum(1 for r in rows if r.get("chain_has_cpu")) / eligible) if eligible else 0.0,
        "h5_len_mean": statistics.fmean(r["h5_len"] for r in rows) if rows else 0.0,
        "mem_pressure_p95": (sorted(r["mem_pressure"] for r in rows)[int(0.95 * (len(rows) - 1))] if rows else 0.0),
    }
    solve_p50 = statistics.median(solve_times) if solve_times else 0.0
    solve_p95 = (sorted(solve_times)[int(0.95 * (len(solve_times) - 1))] if solve_times else 0.0)
    change_counts: dict = {}
    for r in executed:
        ct = r.get("change_type", "unclassified")
        change_counts[ct] = change_counts.get(ct, 0) + 1
    # Advisor release-gate item: separate "no legal action at now" from
    # "legal action exists but the v0a no-eviction plan is infeasible".
    infeasible_split = {
        "no_legal_action_at_now": sum(1 for r in rows if r["fallback"] and r["mask_size"] == 0),
        "plan_infeasible_with_legal_action": sum(1 for r in rows if r["fallback"] and r["mask_size"] > 0),
    }
    evidence = {
        "schema": "milp-v0a-batch-v3", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "episodes": args.episodes, "cap": args.cap, "budget_s": args.budget,
        "eligible_states": eligible,
        "status_counts": status_counts,
        "fallback_counts": len(fallbacks), "fallback_reasons": reason_counts,
        "executed": len(executed),
        "first_action_illegal": len(illegal),
        "change_type_counts": change_counts,
        "infeasible_split": infeasible_split,
        "solve_ms_p50": round(solve_p50, 2), "solve_ms_p95": round(solve_p95, 2),
        "strata": strata_summary,
        "batch_a": batch_a,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"eligible={eligible} executed={len(executed)} fallback={len(fallbacks)} illegal={len(illegal)}")
    print(f"status: {status_counts}")
    print(f"fallback reasons: {reason_counts}")
    print(f"solve ms p50={solve_p50:.1f} p95={solve_p95:.1f}")
    print(f"strata: {strata_summary}")
    print(f"change types: {change_counts}")
    print(f"infeasible split: {infeasible_split}")
    print(f"batch A selected: {len(batch_a)} | categories: {sorted({r['category'] for r in batch_a})}")
    print(f"evidence: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
