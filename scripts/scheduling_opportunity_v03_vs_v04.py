"""Scheduling-opportunity quantification: v03 (raw step-expansion) vs v04 (causal v3.1).

The review's pre-registered question: after repairing the topology, is there still a
real scheduling problem?  If the candidate pools collapse to a single action, the
scheduler has nothing to decide and the workload pressure must be redesigned - but
NOT by restoring the fake parallelism.

Reports the metrics the review named:
  Pr(feasible_actions >= 2), Pr(ready_GPU_nodes >= 2), candidate count p50/p90,
  competitive decision rate, active jobs per decision, GPU utilization,
  eviction/load counts.
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(r"F:\scheduler")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tracing.analysis import decision_trace as dt
from tracing.analysis.workload_v02_simulator import (
    load_resource_v2_overlay,
    load_templates,
    read_jsonl,
    set_decision_trace,
    simulate_episode,
    train_resource_stats,
)

BASE_PACK = ROOT / "outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts"
F0_ROOT = ROOT / "outputs/resource_v2_artifacts/f0_seed11"
V03 = ROOT / "results/processed/r7_workload_v03_no_run_container/job_templates_r7_v03.jsonl"
V04 = ROOT / "results/processed/r7_workload_v04_causal_v31_no_run_container/job_templates_r7_v04.jsonl"
EPISODES = ROOT / "results/processed/r7_workload_v03_no_run_container/episodes/workload_validation_r7_v03.jsonl"


def iter_decisions(path: Path) -> Iterable[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as h:
        for line in h:
            if line.strip():
                yield json.loads(line)




def _mean_util(per_episode):
    """gpu_utilization may be a scalar or a per-GPU list."""
    vals = []
    for r in per_episode:
        u = r.get("gpu_utilization")
        if u is None:
            continue
        if isinstance(u, (list, tuple)):
            vals.extend(float(x) for x in u if isinstance(x, (int, float)))
        elif isinstance(u, (int, float)):
            vals.append(float(u))
    return statistics.fmean(vals) if vals else None

def run(label: str, templates, episodes, artifacts, stats, out_dir: Path, policy: str) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = out_dir / f"opportunity_trace_{label}.jsonl.gz"
    writer = dt.DecisionTraceWriter(trace)
    set_decision_trace(writer, methods=[], truth_cost=None, trajectory_method_id=None)

    per_episode = []
    started = time.time()
    for ep in episodes:
        summary, _ = simulate_episode(ep, templates, policy,
                                      future_artifacts=artifacts, train_stats=stats,
                                      collect_events=False)
        per_episode.append({
            "episode_id": ep["episode_id"],
            "mean_completion_ms": summary.get("mean_completion_ms"),
            "gpu_utilization": summary.get("gpu_utilization"),
            "gpu_evictions": summary.get("gpu_evictions"),
            "prefetch_count": summary.get("prefetch_count"),
            "completed_jobs": summary.get("completed_jobs"),
            "failed_jobs": summary.get("failed_jobs"),
        })
    report = writer.close()
    set_decision_trace(None)

    # ---- opportunity metrics over the traced decisions -------------------- #
    cand_counts: List[int] = []
    feas_counts: List[int] = []
    competitive = 0
    decisions = 0
    for d in iter_decisions(trace):
        decisions += 1
        cand_counts.append(int(d.get("candidate_count") or 0))
        fc = d.get("competitive_candidate_count")
        feas_counts.append(int(fc if fc is not None else 0))
        if int(d.get("competitive_candidate_count") or 0) >= 2:
            competitive += 1


    def pct_at_least(values: List[int], k: int) -> float:
        return 100.0 * sum(1 for v in values if v >= k) / max(1, len(values))

    def q(values, p):
        if not values:
            return None
        s = sorted(values)
        return s[int(p * (len(s) - 1))]

    metrics = {
        "decisions": decisions,
        "candidates_total": sum(cand_counts),
        "Pr_feasible_actions_ge2_pct": round(pct_at_least(feas_counts, 2), 2),
        "Pr_ready_GPU_nodes_ge2_pct": round(pct_at_least(cand_counts, 2), 2),
        "competitive_decision_rate_pct": round(100.0 * competitive / max(1, decisions), 2),
        "candidate_count_p50": q(cand_counts, 0.50),
        "candidate_count_p90": q(cand_counts, 0.90),
        "candidate_count_max": max(cand_counts) if cand_counts else None,
        "competitive_count_p50": q(feas_counts, 0.50),
        "competitive_count_p90": q(feas_counts, 0.90),
        "mean_completion_ms": statistics.fmean(
            [r["mean_completion_ms"] for r in per_episode if r["mean_completion_ms"] is not None]
        ),
        "gpu_utilization_mean": _mean_util(per_episode),
        "evictions_total": sum(int(r["gpu_evictions"] or 0) for r in per_episode),
        "prefetch_total": sum(int(r["prefetch_count"] or 0) for r in per_episode),
        "failed_jobs_total": sum(int(r["failed_jobs"] or 0) for r in per_episode),
        "episodes": len(per_episode),
        "wall_seconds": round(time.time() - started, 1),
        "trace": report,
    }
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--policy", default="sameshape_h5_p95")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "experiments/EXP-20260921_scheduler_replication_v1/artifacts/opportunity_v03_vs_v04.json")
    args = ap.parse_args()

    stats = train_resource_stats(load_templates(V03))
    artifacts, pre = load_resource_v2_overlay(BASE_PACK, F0_ROOT)
    episodes = read_jsonl(EPISODES)[: args.limit]
    print("episodes: %d | F0 pack view_err=%.2e" % (len(episodes), pre["canonical_view_max_abs_error"]))

    results = {}
    for label, path, view in (("v03_raw", V03, "legacy"), ("v04_causal", V04, "causal_v3")):
        templates = load_templates(path, topology_view=view)
        m = run(label, templates, episodes, artifacts, stats, args.out.parent, args.policy)
        results[label] = m
        print("\n--- %s ---" % label)
        for k in ("decisions", "candidates_total", "Pr_feasible_actions_ge2_pct",
                  "Pr_ready_GPU_nodes_ge2_pct", "competitive_decision_rate_pct",
                  "candidate_count_p50", "candidate_count_p90", "candidate_count_max",
                  "mean_completion_ms", "gpu_utilization_mean", "evictions_total"):
            print("   %-34s %s" % (k, m[k]))

    v3, v4 = results["v03_raw"], results["v04_causal"]
    delta = {
        "mean_completion_ms": v4["mean_completion_ms"] - v3["mean_completion_ms"],
        "Pr_ready_GPU_nodes_ge2_pp": v4["Pr_ready_GPU_nodes_ge2_pct"] - v3["Pr_ready_GPU_nodes_ge2_pct"],
        "competitive_decision_rate_pp": v4["competitive_decision_rate_pct"] - v3["competitive_decision_rate_pct"],
        "candidate_count_p50_delta": (v4["candidate_count_p50"] or 0) - (v3["candidate_count_p50"] or 0),
        "decisions_delta": v4["decisions"] - v3["decisions"],
    }
    print("\n" + "=" * 70)
    print("DELTA  (v04_causal - v03_raw)")
    print("=" * 70)
    for k, v in delta.items():
        print("   %-34s %+.2f" % (k, v))

    args.out.write_text(json.dumps({"policy": args.policy, "episodes": len(episodes),
                                    "results": results, "delta": delta},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote", args.out)
    return 0


main_ = main
if __name__ == "__main__":
    raise SystemExit(main())
