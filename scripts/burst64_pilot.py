#!/usr/bin/env python3
"""Burst64-v1 structural pilot (diagnostics only; no scheduler/predictor changes).

Part A: Myopic + PDRS structural metrics over the 4 conditions x 10 episodes
  (decisions, distinct ready jobs, legal action counts, competing fraction,
  queue length, GPU utilization, evictions/cold loads/prefetch mechanics).
  Reported per-episode macro AND pooled.
Part B: PDRS Headroom on a fixed state sample of the most contended condition
  (r110_b15): real DES first-action counterfactuals with main-line continuation.

The mixed-source workload requires an in-memory template split override
("structural_mixed") because the simulator guards source isolation; the files
themselves keep per-template provenance.  Diagnostics only.
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

WORKLOADS = ROOT / "experiments/EXP-20261014_burst64_v1/artifacts"
TAGS = ("r085_b10", "r110_b10", "r085_b15", "r110_b15")


def mean(vals):
    return round(statistics.fmean(vals), 3) if vals else None


def dist(vals):
    out = {}
    for v in sorted(set(int(x) for x in vals)):
        out[str(v)] = sum(1 for x in vals if int(x) == v)
    return out


def part_a(serial, templates_mixed, stats, artifacts):
    from tracing.analysis.workload_v02_simulator import read_jsonl, simulate_episode

    report = {}
    for tag in TAGS:
        episodes = read_jsonl(WORKLOADS / f"workload_burst64_{tag}.jsonl")
        for policy in ("myopic", "pdrs_resident"):
            pooled = {"ready_jobs": [], "legal": [], "queue": [], "competing": 0,
                      "util": [], "evict": [], "cold": [], "prefetch": [], "decisions": 0}
            macro = []
            for ep in episodes:
                summary, events = simulate_episode(
                    ep, templates_mixed, policy, train_stats=stats, future_artifacts=artifacts,
                    future_horizon=5, extension_config=serial, policy_context={"activation": {}},
                    collect_events=True)
                starts = {}
                busy = 0.0
                ep_ready, ep_legal, ep_comp = [], [], 0
                for event in events or []:
                    et = event.get("event_type")
                    if et == "node_dispatch":
                        state = event.get("scheduler_state") or {}
                        nodes = state.get("ready_nodes") or []
                        jobs = {str(n.get("job_instance_id")) for n in nodes}
                        raw = int(event.get("candidate_count") or 0)
                        feas = int(event.get("feasible_candidate_count") or 0)
                        legal = feas if feas > 0 else raw
                        pooled["decisions"] += 1
                        pooled["ready_jobs"].append(len(jobs))
                        pooled["legal"].append(legal)
                        pooled["queue"].append(len(nodes))
                        if len(jobs) >= 2:
                            pooled["competing"] += 1
                        ep_ready.append(len(jobs)); ep_legal.append(legal)
                        if len(jobs) >= 2:
                            ep_comp += 1
                    elif et == "node_start":
                        starts[str(event.get("node_id"))] = float(event.get("time_ms") or 0.0)
                    elif et == "node_finish":
                        key = str(event.get("node_id"))
                        if key in starts:
                            busy += float(event.get("time_ms") or 0.0) - starts.pop(key)
                makespan = float(summary.get("makespan_ms") or 1.0)
                util = busy / (2.0 * makespan)
                pooled["util"].append(util)
                pooled["evict"].append(float(summary.get("gpu_evictions") or 0))
                pooled["cold"].append(float(summary.get("cold_loads_on_demand") or 0))
                pooled["prefetch"].append(float(summary.get("prefetch_count") or 0))
                macro.append({
                    "episode": ep["episode_id"],
                    "mean_completion_ms": round(float(summary["mean_completion_ms"]), 1),
                    "completed": int(summary["completed_jobs"]), "jobs": int(summary["jobs"]),
                    "decisions": len(ep_ready),
                    "ready_jobs_mean": mean(ep_ready),
                    "legal_mean": mean(ep_legal),
                    "competing_frac": round(ep_comp / max(1, len(ep_ready)), 3),
                    "util": round(util, 3),
                })
            n = max(1, pooled["decisions"])
            report[f"{tag}|{policy}"] = {
                "macro_mean_completion_ms": mean([m["mean_completion_ms"] for m in macro]),
                "macro_ready_jobs": mean([m["ready_jobs_mean"] for m in macro]),
                "macro_legal": mean([m["legal_mean"] for m in macro]),
                "macro_competing_frac": mean([m["competing_frac"] for m in macro]),
                "macro_util": mean([m["util"] for m in macro]),
                "all_completed": all(m["completed"] == m["jobs"] for m in macro),
                "pooled": {
                    "decisions": pooled["decisions"],
                    "ready_jobs_mean": mean(pooled["ready_jobs"]),
                    "ready_jobs_dist": dist(pooled["ready_jobs"]),
                    "legal_mean": mean(pooled["legal"]),
                    "legal_dist": dist(pooled["legal"]),
                    "legal_single_frac": round(sum(1 for x in pooled["legal"] if x == 1) / n, 3),
                    "competing_frac": round(pooled["competing"] / n, 3),
                    "queue_mean": mean(pooled["queue"]),
                    "util_mean": mean(pooled["util"]),
                    "evictions_mean": mean(pooled["evict"]),
                    "cold_loads_mean": mean(pooled["cold"]),
                    "prefetch_mean": mean(pooled["prefetch"]),
                },
            }
    return report


def part_b(serial, templates_mixed, stats, artifacts, condition="r110_b15", max_states=24):
    import milp_v0_tests as T
    from tracing.analysis.des_rollout import build_planning_episode, run_planning
    from tracing.analysis.workload_v02_simulator import read_jsonl

    episodes = read_jsonl(WORKLOADS / f"workload_burst64_{condition}.jsonl")
    states = T.capture_states(episodes, templates_mixed, stats, artifacts, serial, cap=12)
    rows = []
    for ep, jobs_meta, snap in states:
        if len(rows) >= max_states:
            break
        plan_episode = build_planning_episode(ep, snap, include_arrivals=True)
        _s, _e, pctx = run_planning(
            plan_episode, templates_mixed, future_artifacts=artifacts, train_stats=stats,
            extension_config=serial, policy="des_rollout", forced_action=None,
            collect_events=False, extra_context={"des_probe_pool": True})
        probe_now = float((pctx.get("des_first_state") or {}).get("now_ms", -1.0))
        if abs(probe_now - float(snap["now_ms"])) > 1e-6:
            continue
        pool = sorted({(str(r[1]), int(r[2])) for r in (pctx.get("des_first_pool") or [])})
        baseline = tuple(pctx.get("des_first_baseline") or ())
        if not pool or not baseline:
            continue
        costs = {}
        for node, gpu in pool:
            summary, _ev, _c = run_planning(
                plan_episode, templates_mixed, future_artifacts=artifacts, train_stats=stats,
                extension_config=serial, policy="des_rollout",
                forced_action={"node_key": node, "gpu_index": gpu}, collect_events=False)
            costs[(node, gpu)] = float(summary["mean_completion_ms"])
        best = min(costs.values())
        base = costs.get(baseline)
        if base is None:
            continue
        rows.append({
            "episode": ep["episode_id"], "now_ms": snap["now_ms"], "pool": len(pool),
            "headroom": round(base - best, 1),
            "baseline_is_best": abs(base - best) < 1e-9,
            "spread": round(max(costs.values()) - best, 1),
        })
    heads = [r["headroom"] for r in rows]
    nz = [h for h in heads if h > 1e-9]
    return {
        "condition": condition, "states": len(rows),
        "pool_mean": mean([r["pool"] for r in rows]),
        "headroom_mean": mean(heads), "headroom_mean_nonzero": mean(nz),
        "headroom_zero_frac": round(sum(1 for h in heads if h <= 1e-9) / max(1, len(heads)), 3),
        "baseline_is_best_frac": round(sum(1 for r in rows if r["baseline_is_best"]) / max(1, len(rows)), 3),
        "headroom_p90": round(sorted(heads)[int(0.9 * (len(heads) - 1))], 1) if heads else None,
        "headroom_max": round(max(heads), 1) if heads else None,
        "spread_mean": mean([r["spread"] for r in rows]),
        "details": rows,
    }


def main() -> int:
    import milp_v0_performance as P

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=WORKLOADS / "burst64_v1_pilot.json")
    ap.add_argument("--skip-headroom", action="store_true")
    args = ap.parse_args()

    serial, templates, _nat, stats, artifacts = P.load_split("confirm", 1)
    templates_mixed = {k: replace(v, split="structural_mixed") for k, v in templates.items()}

    report = {"schema": "burst64-v1-pilot-v1", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "part_a_structural": part_a(serial, templates_mixed, stats, artifacts)}
    if not args.skip_headroom:
        report["part_b_headroom"] = part_b(serial, templates_mixed, stats, artifacts)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print("evidence:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
