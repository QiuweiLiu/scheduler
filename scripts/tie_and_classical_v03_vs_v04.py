"""Run TIE-adapted plus the classical references on v03 (raw) vs v04 (causal).

TIE-adapted is the first of the four paper-reproduction joint baselines.  It is a
joint baseline in the sense that it brings its OWN front end: a train-only,
current-node-only runtime distribution taken from the (model, lane) histogram, with
no use of our future-structure predictor.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict

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
BASE_PACK = ROOT / "outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts"

POLICIES = ("tie_current", "fcfs", "myopic", "round_robin", "sjf_pred")
KEYS = ("mean_completion_ms", "p95_completion_ms", "makespan_ms", "deadline_miss_rate",
        "gpu_evictions", "gpu_utilization", "completed_jobs", "failed_jobs")


def run(templates, episodes, stats, artifacts, tag: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for policy in POLICIES:
        started = time.time()
        rows = []
        for ep in episodes:
            s, _ = simulate_episode(ep, templates, policy, future_artifacts=artifacts,
                                    train_stats=stats, collect_events=False)
            rows.append({k: s.get(k) for k in KEYS})
        def mean(key):
            v = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
            return statistics.fmean(v) if v else None
        util = []
        for r in rows:
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
            "evictions": sum(int(r.get("gpu_evictions") or 0) for r in rows),
            "completed_jobs": sum(int(r.get("completed_jobs") or 0) for r in rows),
            "failed_jobs": sum(int(r.get("failed_jobs") or 0) for r in rows),
            "wall_seconds": round(time.time() - started, 1),
            "workload": tag,
        }
        m = out[policy]
        print("   %-13s [%-10s] mean=%10.1f p95=%10.1f util=%.3f evict=%4d failed=%d (%.0fs)"
              % (policy, tag, m["mean_completion_ms"], m["p95_completion_ms"],
                 m["gpu_utilization"], m["evictions"], m["failed_jobs"], m["wall_seconds"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "experiments/EXP-20260921_scheduler_replication_v1/artifacts/tie_and_classical_v03_vs_v04.json")
    a = ap.parse_args()

    stats = train_resource_stats(load_templates(V03))
    artifacts = load_future_artifacts(BASE_PACK)
    episodes = read_jsonl(EPISODES)[: a.limit]
    print("episodes: %d | policies: %s | train-stat groups: %d"
          % (len(episodes), ", ".join(POLICIES), len(stats)))

    print("\n--- v03 raw step-expansion graph ---")
    v3 = run(load_templates(V03), episodes, stats, artifacts, "v03_raw")
    print("\n--- v04 causal v3.1 graph (corrected) ---")
    v4 = run(load_templates(V04, topology_view="causal_v3"), episodes, stats, artifacts, "v04_causal")

    print("\n" + "=" * 78)
    print("DELTA mean_completion_ms  (v04_causal - v03_raw)")
    print("=" * 78)
    for p in POLICIES:
        x, y = v3[p]["mean_completion_ms"], v4[p]["mean_completion_ms"]
        print("   %-13s %10.1f -> %10.1f  %+9.1f (%+.1f%%)" % (p, x, y, y - x, 100 * (y - x) / x))

    print("\n" + "=" * 78)
    print("v04 causal: ranking against fcfs")
    print("=" * 78)
    base = v4["fcfs"]["mean_completion_ms"]
    for p in POLICIES:
        d = v4[p]["mean_completion_ms"] - base
        print("   %-13s %+10.1f ms  (%+.1f%%)" % (p, d, 100 * d / base))

    a.out.write_text(json.dumps({"episodes": len(episodes), "policies": list(POLICIES),
                                 "v03_raw": v3, "v04_causal": v4}, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    print("\nwrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
