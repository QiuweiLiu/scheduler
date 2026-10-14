#!/usr/bin/env python3
"""Root-cause decomposition: why are there so few legal actions per decision?

Uses the existing node_dispatch instrumentation (scheduler_state, candidate
counts) over ALL decisions of sampled confirm episodes -- no sampling bias.
Decomposes: ready GPU nodes x dispatchable GPUs -> raw candidates -> fitting
-> pool (fitting-preferred); plus GPU busy fraction and arrival pacing.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tracing.analysis.workload_v02_simulator import simulate_episode  # noqa: E402


def dist(vals):
    out = {}
    for v in sorted(set(vals)):
        out[str(v)] = vals.count(v)
    return out


def run_env(name, config, episodes, templates, stats, artifacts):
    ready_counts, raw_counts, feas_counts, pool_sizes, free_gpus = [], [], [], [], []
    busy_fracs = []
    for ep in episodes:
        summary, events = simulate_episode(
            ep, templates, "pdrs_resident", train_stats=stats, future_artifacts=artifacts,
            future_horizon=5, extension_config=config, policy_context={"activation": {}},
            collect_events=True)
        busy = {}
        for event in events or []:
            if event.get("event_type") == "node_dispatch":
                state = event.get("scheduler_state") or {}
                nodes = state.get("ready_nodes") or []
                ready_counts.append(len(nodes))
                raw = int(event.get("candidate_count") or 0)
                feas = int(event.get("feasible_candidate_count") or 0)
                raw_counts.append(raw)
                feas_counts.append(feas)
                pool_sizes.append(feas if feas > 0 else raw)
                dev = set()
                for node in nodes:
                    dev.update(int(g) for g in (node.get("dispatchable_gpu_indices") or ()))
                free_gpus.append(len(dev))
            elif event.get("event_type") in ("node_start", "node_finish"):
                pass
        # GPU busy fraction from node_start/node_finish per gpu
        starts = {}
        intervals = {}
        for event in events or []:
            g = event.get("gpu_index")
            if g is None:
                continue
            if event.get("event_type") == "node_start":
                starts[g] = float(event.get("time_ms") or 0.0)
            elif event.get("event_type") == "node_finish" and g in starts:
                intervals[g] = intervals.get(g, 0.0) + (float(event.get("time_ms") or 0.0) - starts.pop(g))
        makespan = float(summary.get("makespan_ms") or 0.0)
        if makespan > 0 and intervals:
            busy_fracs.append(sum(intervals.values()) / (makespan * max(1, len(intervals))))
    n = len(pool_sizes)
    return {
        "environment": name,
        "episodes": len(episodes),
        "decisions": n,
        "ready_gpu_nodes_mean": round(statistics.fmean(ready_counts), 2) if n else None,
        "ready_gpu_nodes_dist": dist(ready_counts),
        "free_gpus_mean": round(statistics.fmean(free_gpus), 2) if n else None,
        "free_gpus_dist": dist(free_gpus),
        "raw_candidates_mean": round(statistics.fmean(raw_counts), 2) if n else None,
        "fitting_mean": round(statistics.fmean(feas_counts), 2) if n else None,
        "pool_mean": round(statistics.fmean(pool_sizes), 2) if n else None,
        "pool_dist": dist(pool_sizes),
        "pool_single_frac": round(sum(1 for p in pool_sizes if p == 1) / n, 3) if n else None,
        "fitting_zero_frac": round(sum(1 for f in feas_counts if f == 0) / n, 3) if n else None,
        "gpu_busy_fraction_mean": round(statistics.fmean(busy_fracs), 3) if busy_fracs else None,
    }


def main() -> int:
    import milp_v0_performance as P

    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "experiments/EXP-20261012_milp_v0/artifacts/legal_action_rootcause.json")
    args = ap.parse_args()

    serial, templates, episodes, stats, artifacts = P.load_split("confirm", args.episodes)
    v7 = json.loads((ROOT / "experiments/EXP-20261004_real_strata_calibration_v1/artifacts"
                     / "substrate_extension_v7.json").read_text(encoding="utf-8"))
    # v7 needs its own contract-applied fixtures
    from tracing.analysis.profile_contract import apply_real_workload_profile_contract
    from tracing.analysis.workload_v02_simulator import train_resource_stats
    t7, e7 = apply_real_workload_profile_contract(
        {k: v for k, v in templates.items()}, list(episodes), v7)
    stats7 = train_resource_stats({k: v for k, v in t7.items() if v.split == "train"})

    report = {
        "schema": "legal-action-rootcause-v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "split": "confirm",
        "serial_v0": run_env("serial_v0", serial, episodes, templates, stats, artifacts),
        "v7_full": run_env("v7_full", v7, e7, t7, stats7, artifacts),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print("evidence:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
