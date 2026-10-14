#!/usr/bin/env python3
"""PDRS oracle ceiling: the main line with TRUE future identities vs the
frozen GRU predictions (same ordering / placement / eviction / prefetch rules).

The oracle arm is a clairvoyant upper-bound reference (not deployable).
Paired over the same episodes; pooled P95 and per-episode P95 both reported.
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

ARMS = {"pdrs_pred": "pdrs_resident", "pdrs_oracle": "pdrs_resident_oracle"}


def bootstrap_ci(values, seed=11, b=2000):
    if len(values) < 2:
        return (float(values[0]), float(values[0])) if values else (0.0, 0.0)
    import random as _r
    rng = _r.Random(seed)
    means = []
    for _ in range(b):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(statistics.fmean(sample))
    means.sort()
    return (means[int(0.025 * b)], means[int(0.975 * b) - 1])


def quantile(values, q):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]


def main() -> int:
    import milp_v0_performance as P

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--split", choices=("train", "confirm"), default="confirm")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    serial, templates, episodes, stats, artifacts = P.load_split(args.split, args.episodes, args.config)
    per_arm = {name: {"mean": [], "p95": [], "dur": [], "cold": [], "evict": [], "prefetch": []}
               for name in ARMS}
    for ep in episodes:
        arrivals = {str(row["job_instance_id"]): float(row["arrival_ms"])
                    for row in (ep.get("jobs") or [])}
        for name, policy in ARMS.items():
            summary, events = simulate_episode(
                ep, templates, policy, train_stats=stats, future_artifacts=artifacts,
                future_horizon=5, extension_config=serial, policy_context={"activation": {}},
                collect_events=True)
            row = per_arm[name]
            row["mean"].append(float(summary["mean_completion_ms"]))
            row["p95"].append(float(summary["p95_completion_ms"]))
            row["cold"].append(float(summary["cold_loads_on_demand"]))
            row["evict"].append(float(summary["gpu_evictions"]))
            row["prefetch"].append(float(summary.get("prefetch_count") or 0))
            for event in events or []:
                if event.get("event_type") == "job_finish":
                    jid = str(event.get("job_instance_id"))
                    if jid in arrivals:
                        row["dur"].append(float(event["time_ms"]) - arrivals[jid])

    report = {"schema": "pdrs-oracle-ceiling-v1", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "episodes": len(episodes), "split": args.split,
              "config": str(args.config) if args.config else "serial_v0_extension.json",
              "arms": {}}
    for name, row in per_arm.items():
        report["arms"][name] = {
            "mean": round(statistics.fmean(row["mean"]), 2),
            "p95_pooled": round(quantile(row["dur"], 0.95), 2),
            "p95_mean_of_episodes": round(statistics.fmean(row["p95"]), 2),
            "cold_loads": round(statistics.fmean(row["cold"]), 2),
            "evictions": round(statistics.fmean(row["evict"]), 2),
            "prefetch_count": round(statistics.fmean(row["prefetch"]), 2),
        }
    diffs = [x - y for x, y in zip(per_arm["pdrs_oracle"]["mean"], per_arm["pdrs_pred"]["mean"])]
    p95_diffs = [x - y for x, y in zip(per_arm["pdrs_oracle"]["p95"], per_arm["pdrs_pred"]["p95"])]
    report["oracle_minus_pred"] = {
        "mean_delta": round(statistics.fmean(diffs), 2),
        "mean_ci95": [round(v, 2) for v in bootstrap_ci(diffs)],
        "p95_delta_mean_of_episodes": round(statistics.fmean(p95_diffs), 2),
        "p95_ci95": [round(v, 2) for v in bootstrap_ci(p95_diffs)],
        "episodes_oracle_better": sum(1 for d in diffs if d < 0),
        "episodes_pred_better": sum(1 for d in diffs if d > 0),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print("evidence:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
