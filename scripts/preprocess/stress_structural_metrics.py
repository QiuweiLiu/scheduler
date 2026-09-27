#!/usr/bin/env python
"""Structural activation metrics for a workload regime -- NO performance outcomes.

Reports only construction/activation structure, so stress intensity can be chosen
without consulting any method's latency or completion time.

Scope is GPU decisions only: a decision is a ``node_dispatch`` event whose ``lane`` is
``"gpu"``.  Events without a ``lane`` cause an error (never a silent skip).  All required
fields are validated; missing data raises instead of defaulting to 0.

Primary aggregation is EPISODE-MACRO (each episode counts once); decision-pooled values
are reported as explicit ``*_pooled`` telemetry.  A 95% bootstrap CI over episodes is
reported for the macro metrics, and ``--pair-with`` produces a *paired* episode-level
bootstrap CI for the delta against another run (this is the only valid way to compare two
regimes on the same episodes).

Metric definitions:

  * ``competitive_macro``   = mean over episodes of Pr(#distinct ready GPU jobs >= 2)
  * ``competitive_pooled``  = Pr(#distinct ready GPU jobs >= 2) pooled over decisions
  * ``ready_jobs_p50_macro``= median over episodes of the per-episode ready-jobs median
  * ``ready_jobs_p50_pooled`` = quantile of #distinct ready GPU jobs pooled over decisions
  * ``ordering_retention_macro`` = mean over episodes of
        Pr(#distinct FEASIBLE GPU jobs >= 2 | #distinct ready GPU jobs >= 2)

``ordering_retention`` is a CAPACITY PROXY: a ready GPU job is "feasible" iff at least one
non-busy GPU has ``capacity_mb >= peak_memory_p95_mb`` (the same capacity test the simulator
uses to build its ``fitting`` pool).  It does NOT run ``plan_gpu_admission`` (residency /
eviction / model state), so it must not be read as a full admission or ordering guarantee.
It counts DISTINCT JOBS, never (job, node, gpu) placements.

Usage::

    python scripts/preprocess/stress_structural_metrics.py \
        --templates <projection.jsonl> --episodes <episodes.jsonl> --policy myopic \
        --label alpha080 --out <report.json> --per-episode-out <episodes.jsonl> \
        [--pair-with <other_per_episode.jsonl>]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tracing.analysis.workload_v02_simulator import (  # noqa: E402
    load_templates,
    train_resource_stats,
    simulate_episode,
)

BOOTSTRAP_SEED = 20260927
BOOTSTRAP_N = 2000


def quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def bootstrap_ci(values, statistic, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """Percentile bootstrap 95% CI over the episode-level values."""
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    k = len(values)
    stats = []
    for _ in range(n):
        sample = [values[rng.randrange(k)] for _ in range(k)]
        stats.append(statistic(sample))
    stats.sort()
    return [round(stats[max(0, int(round(0.025 * (n - 1))))], 4),
            round(stats[min(n - 1, int(round(0.975 * (n - 1))))], 4)]


def paired_bootstrap_ci(pairs, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """Percentile bootstrap 95% CI of mean(a - b) over matched (a, b) episode pairs."""
    if len(pairs) < 2:
        return None
    rng = random.Random(seed)
    k = len(pairs)
    stats = []
    for _ in range(n):
        sample = [pairs[rng.randrange(k)] for _ in range(k)]
        stats.append(statistics.fmean(a - b for a, b in sample))
    stats.sort()
    lo = stats[max(0, int(round(0.025 * (n - 1))))]
    hi = stats[min(n - 1, int(round(0.975 * (n - 1))))]
    return [round(lo, 4), round(hi, 4)]


def _require(node, key, ctx):
    if key not in node or node[key] is None:
        raise KeyError("scheduler_state %s missing %r" % (ctx, key))
    return node[key]


def gpu_decisions(events):
    """Yield scheduler_state for GPU dispatch decisions only (lane must be present)."""
    for e in events:
        if e.get("event_type") != "node_dispatch":
            continue
        lane = e.get("lane")
        if lane is None:
            raise KeyError("node_dispatch event lacks 'lane'")
        if lane != "gpu":
            continue
        state = e.get("scheduler_state")
        if not state:
            raise KeyError("gpu node_dispatch event lacks scheduler_state")
        if "ready_nodes" not in state or "gpus" not in state:
            raise KeyError("scheduler_state lacks ready_nodes/gpus")
        yield state


def ready_gpu_jobs(state):
    jobs = set()
    for r in state["ready_nodes"]:
        if _require(r, "lane", "ready_node") != "gpu":
            continue
        jobs.add(_require(r, "job_instance_id", "ready_node"))
    return jobs


def feasible_gpu_jobs(state):
    """Distinct ready GPU jobs with >=1 placement on a non-busy GPU that fits by capacity."""
    free_caps = []
    for gp in state["gpus"]:
        busy = _require(gp, "busy", "gpu")
        if not busy:
            free_caps.append(float(_require(gp, "capacity_mb", "gpu")))
    feasible = set()
    for r in state["ready_nodes"]:
        if _require(r, "lane", "ready_node") != "gpu":
            continue
        res = _require(r, "resource", "ready_node")
        mem = float(_require(res, "peak_memory_p95_mb", "ready_node.resource"))
        if any(mem <= cap + 1e-9 for cap in free_caps):
            feasible.add(_require(r, "job_instance_id", "ready_node"))
    return feasible


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--policy", default="myopic")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--per-episode-out", default="")
    ap.add_argument("--pair-with", default="", help="another --per-episode-out jsonl to pair against")
    args = ap.parse_args()

    tpls = load_templates(Path(args.templates), topology_view="causal_v3")
    stats = train_resource_stats({k: v for k, v in tpls.items() if v.split == "train"})
    episodes = [json.loads(l) for l in Path(args.episodes).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        episodes = episodes[: args.limit]

    per_episode = []
    ready_counts = []           # pooled over GPU decisions
    for ep in episodes:
        summary, events = simulate_episode(ep, tpls, args.policy, train_stats=stats, collect_events=True)
        if "failed_jobs" not in summary:
            raise KeyError("simulate_episode summary lacks failed_jobs for %r" % ep.get("episode_id"))
        ep_ready, ep_multi_ready, ep_multi_feasible = [], 0, 0
        for state in gpu_decisions(events):
            n_ready = len(ready_gpu_jobs(state))
            ep_ready.append(n_ready)
            ready_counts.append(n_ready)
            if n_ready >= 2:
                ep_multi_ready += 1
                if len(feasible_gpu_jobs(state)) >= 2:
                    ep_multi_feasible += 1
        util_series = summary.get("gpu_utilization")
        if util_series is None:
            raise KeyError("simulate_episode summary lacks gpu_utilization for %r" % ep.get("episode_id"))
        per_episode.append({
            "key": ep.get("parent_episode_id") or ep.get("episode_id"),
            "episode_id": ep.get("episode_id"),
            "decisions": len(ep_ready),
            "competitive_rate": (sum(1 for n in ep_ready if n >= 2) / len(ep_ready)) if ep_ready else None,
            "ready_jobs_median": quantile(ep_ready, 0.5),
            "multi_ready_decisions": ep_multi_ready,
            "multi_feasible_decisions": ep_multi_feasible,
            "ordering_retention": (ep_multi_feasible / ep_multi_ready) if ep_multi_ready else None,
            "gpu_utilization": (statistics.fmean(util_series) if util_series else None),
            "failed_jobs": int(summary["failed_jobs"]),
        })

    ep_comp = [r["competitive_rate"] for r in per_episode if r["competitive_rate"] is not None]
    ep_util = [r["gpu_utilization"] for r in per_episode if r["gpu_utilization"] is not None]
    ep_order = [r["ordering_retention"] for r in per_episode if r["ordering_retention"] is not None]
    ep_p50 = [r["ready_jobs_median"] for r in per_episode if r["ready_jobs_median"] is not None]
    total_multi_ready = sum(r["multi_ready_decisions"] for r in per_episode)
    total_multi_feasible = sum(r["multi_feasible_decisions"] for r in per_episode)

    report = {
        "label": args.label or None,
        "policy": args.policy,
        "episodes": len(episodes),
        "gpu_decisions": len(ready_counts),
        "metrics_definition": {
            "scope": "GPU decisions only (node_dispatch events with lane == 'gpu')",
            "competitive": "Pr(#distinct ready GPU jobs >= 2)",
            "ordering_retention": "Pr(#distinct feasible GPU jobs >= 2 | #distinct ready GPU jobs >= 2); "
                                  "feasible = CAPACITY PROXY (>=1 non-busy GPU with capacity >= peak_memory_p95_mb); "
                                  "does NOT model residency/eviction/admission; counts distinct JOBS",
            "aggregation": "primary = episode-macro; decision-pooled values are explicit *_pooled telemetry",
        },
        "competitive_macro": round(statistics.fmean(ep_comp), 4) if ep_comp else None,
        "competitive_macro_ci95": bootstrap_ci(ep_comp, statistics.fmean),
        "competitive_pooled": round(sum(1 for n in ready_counts if n >= 2) / len(ready_counts), 4) if ready_counts else None,
        "ready_jobs_p50_macro": round(statistics.fmean(ep_p50), 4) if ep_p50 else None,
        "ready_jobs_p50_pooled": quantile(ready_counts, 0.5),
        "ready_jobs_p90_pooled": quantile(ready_counts, 0.9),
        "gpu_utilization_macro": round(statistics.fmean(ep_util), 4) if ep_util else None,
        "gpu_utilization_macro_ci95": bootstrap_ci(ep_util, statistics.fmean),
        "ordering_retention_macro": round(statistics.fmean(ep_order), 4) if ep_order else None,
        "ordering_retention_macro_ci95": bootstrap_ci(ep_order, statistics.fmean),
        "ordering_retention_pooled": (round(total_multi_feasible / total_multi_ready, 4) if total_multi_ready else None),
        "multi_ready_decisions": total_multi_ready,
        "multi_feasible_decisions": total_multi_feasible,
        "failed_jobs": sum(r["failed_jobs"] for r in per_episode),
        "episodes_with_multi_ready": sum(1 for r in per_episode if r["multi_ready_decisions"] > 0),
    }

    if args.pair_with:
        other = {}
        for line in Path(args.pair_with).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                other[row["key"]] = row
        pairs_c = [(r["competitive_rate"], other[r["key"]]["competitive_rate"])
                   for r in per_episode
                   if r["key"] in other and r["competitive_rate"] is not None
                   and other[r["key"]]["competitive_rate"] is not None]
        pairs_u = [(r["gpu_utilization"], other[r["key"]]["gpu_utilization"])
                   for r in per_episode
                   if r["key"] in other and r["gpu_utilization"] is not None
                   and other[r["key"]]["gpu_utilization"] is not None]
        report["paired_vs"] = args.pair_with
        report["paired_episodes"] = len(pairs_c)
        report["paired_delta_competitive"] = round(statistics.fmean(a - b for a, b in pairs_c), 4) if pairs_c else None
        report["paired_delta_competitive_ci95"] = paired_bootstrap_ci(pairs_c)
        report["paired_delta_util"] = round(statistics.fmean(a - b for a, b in pairs_u), 4) if pairs_u else None
        report["paired_delta_util_ci95"] = paired_bootstrap_ci(pairs_u)

    text = json.dumps(report, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    if args.per_episode_out:
        with Path(args.per_episode_out).open("w", encoding="utf-8") as handle:
            for row in per_episode:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
