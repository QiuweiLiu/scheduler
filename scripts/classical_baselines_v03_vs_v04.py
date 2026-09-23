"""Run the classical scheduling baselines on the corrected v04 causal workload.

Our own predictor-driven policies are deliberately excluded here: the point is to
establish the reference floor on the repaired execution graph, and to quantify how
much of the old v03 numbers came from the fake parallelism.

Baselines: fcfs, myopic, round_robin, sjf_pred.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(r"F:\scheduler")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tracing.analysis.workload_v02_simulator import (
    load_future_artifacts,
    load_templates,
    read_jsonl,
    simulate_episode,
    train_resource_stats,
)

V03 = ROOT / "results/processed/r7_workload_v03_no_run_container/job_templates_r7_v03.jsonl"
V04 = ROOT / "results/processed/r7_workload_v04_causal_v31_no_run_container/job_templates_r7_v04.jsonl"
EPISODES = ROOT / "results/processed/r7_workload_v03_no_run_container/episodes/workload_validation_r7_v03.jsonl"

BASELINES = ("fcfs", "myopic", "round_robin", "sjf_pred")
SUMMARY_KEYS = ("mean_completion_ms", "p95_completion_ms", "makespan_ms", "deadline_miss_rate",
                "gpu_evictions", "gpu_utilization", "completed_jobs", "failed_jobs")


def run_all(templates, episodes, stats, artifacts, tag: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for policy in BASELINES:
        started = time.time()
        per_ep = []
        for ep in episodes:
            summary, _ = simulate_episode(ep, templates, policy,
                                          future_artifacts=artifacts, train_stats=stats,
                                          collect_events=False)
            per_ep.append({k: summary.get(k) for k in SUMMARY_KEYS})
        def mean(key):
            vals = [r[key] for r in per_ep if isinstance(r.get(key), (int, float))]
            return statistics.fmean(vals) if vals else None
        util = []
        for r in per_ep:
            u = r.get("gpu_utilization")
            if isinstance(u, (list, tuple)):
                util.extend(float(x) for x in u if isinstance(x, (int, float)))
            elif isinstance(u, (int, float)):
                util.append(float(u))
        out[policy] = {
            "mean_completion_ms": mean("mean_completion_ms"),
            "p95_completion_ms": mean("p95_completion_ms"),
            "makespan_ms": mean("makespan_ms"),
            "deadline_miss_rate": mean("deadline_miss_rate"),
            "gpu_utilization": statistics.fmean(util) if util else None,
            "evictions": sum(int(r.get("gpu_evictions") or 0) for r in per_ep),
            "completed_jobs": sum(int(r.get("completed_jobs") or 0) for r in per_ep),
            "failed_jobs": sum(int(r.get("failed_jobs") or 0) for r in per_ep),
            "wall_seconds": round(time.time() - started, 1),
            "workload": tag,
        }
        m = out[policy]
        print("   %-12s [%s] mean=%10.1f  p95=%10.1f  util=%.3f  evict=%4d  failed=%d  (%.0fs)"
              % (policy, tag, m["mean_completion_ms"], m["p95_completion_ms"],
                 m["gpu_utilization"], m["evictions"], m["failed_jobs"], m["wall_seconds"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "experiments/EXP-20260921_scheduler_replication_v1/artifacts/classical_baselines_v03_vs_v04.json")
    args = ap.parse_args()

    stats = train_resource_stats(load_templates(V03))
    artifacts = load_future_artifacts(
        ROOT / "outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts"
    )
    print("base (frozen J3) artifacts: %d nodes" % len(artifacts))
    episodes = read_jsonl(EPISODES)[: args.limit]
    print("episodes: %d | policies: %s" % (len(episodes), ", ".join(BASELINES)))

    print("\n--- v03 raw step-expansion graph (the old, fake-parallel one) ---")
    v3 = run_all(load_templates(V03), episodes, stats, artifacts, "v03_raw")

    print("\n--- v04 causal v3.1 graph (corrected) ---")
    v4 = run_all(load_templates(V04, topology_view="causal_v3"), episodes, stats, artifacts, "v04_causal")

    print("\n" + "=" * 80)
    print("DELTA mean_completion_ms  (v04_causal - v03_raw)")
    print("=" * 80)
    for policy in BASELINES:
        a, b = v3[policy]["mean_completion_ms"], v4[policy]["mean_completion_ms"]
        print("   %-12s %10.1f -> %10.1f   %+9.1f  (%+.1f%%)"
              % (policy, a, b, b - a, 100.0 * (b - a) / a))

    args.out.write_text(json.dumps({"episodes": len(episodes), "baselines": list(BASELINES),
                                    "v03_raw": v3, "v04_causal": v4},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
