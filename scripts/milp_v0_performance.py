#!/usr/bin/env python3
"""MILP-v0a paired performance evaluation (serial_v0 environment).

Arms: M0 = F0 reference (sameshape_h5_p95), M1 = PDRS (pdrs_resident),
M2 = milp_v0a (online MILP, first action if legal else PDRS fallback).

Metrics per arm: mean/p95 completion, makespan, cold loads, evictions.
M2 additionally: decisions, executed/fallback/prediction-gap counts, solve-time
P50/P95 and EXPOSED latency (decisions where the chosen device was idle at the
decision moment), reported separately per the review.
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

import milp_v0_tests as T  # noqa: E402
from tracing.analysis.workload_v02_simulator import simulate_episode  # noqa: E402

ARMS = {
    "M0_f0": "sameshape_h5_p95",
    "M1_pdrs": "pdrs_resident",
    "M2_milp": "milp_v0a",
}
EVIDENCE = ROOT / "experiments/EXP-20261012_milp_v0/artifacts/performance_eval.json"


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
    lo = means[int(0.025 * b)]
    hi = means[int(0.975 * b) - 1]
    return (lo, hi)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--budget", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=EVIDENCE)
    args = parser.parse_args()

    serial, templates, episodes, stats, artifacts = T.load_fixtures(args.episodes)
    per_arm: dict = {name: {"mean": [], "p95": [], "makespan": [], "cold": [], "evict": [], "wall": []} for name in ARMS}
    milp_totals: dict = {}
    for ep in episodes:
        for name, policy in ARMS.items():
            ctx = {"activation": {}}
            if policy == "milp_v0a":
                ctx["milp_budget_s"] = args.budget
            t0 = time.perf_counter()
            summary, _events = simulate_episode(
                ep, templates, policy, train_stats=stats, future_artifacts=artifacts,
                future_horizon=5, extension_config=serial, policy_context=ctx,
                collect_events=False)
            wall = time.perf_counter() - t0
            row = per_arm[name]
            row["mean"].append(float(summary["mean_completion_ms"]))
            row["p95"].append(float(summary["p95_completion_ms"]))
            row["makespan"].append(float(summary["makespan_ms"]))
            row["cold"].append(float(summary["cold_loads_on_demand"]))
            row["evict"].append(float(summary["gpu_evictions"]))
            row["wall"].append(wall)
            if policy == "milp_v0a":
                act = ctx.get("activation") or {}
                for key in ("milp_decisions", "milp_executed", "milp_fallback", "milp_gap_fallback",
                            "milp_solve_ms", "milp_exposed_ms"):
                    milp_totals[key] = float(milp_totals.get(key, 0.0)) + float(act.get(key, 0.0))

    report: dict = {"schema": "milp-v0a-performance-v1", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "episodes": len(episodes), "budget_s": args.budget, "arms": {}}
    for name, row in per_arm.items():
        report["arms"][name] = {
            "mean": round(statistics.fmean(row["mean"]), 2),
            "p95": round(statistics.fmean(row["p95"]), 2),
            "makespan": round(statistics.fmean(row["makespan"]), 2),
            "cold_loads": round(statistics.fmean(row["cold"]), 3),
            "evictions": round(statistics.fmean(row["evict"]), 3),
            "wall_s_per_episode": round(statistics.fmean(row["wall"]), 3),
        }
    deltas = {}
    for a, b in (("M2_milp", "M1_pdrs"), ("M2_milp", "M0_f0"), ("M1_pdrs", "M0_f0")):
        diffs = [x - y for x, y in zip(per_arm[a]["mean"], per_arm[b]["mean"])]
        p95_diffs = [x - y for x, y in zip(per_arm[a]["p95"], per_arm[b]["p95"])]
        deltas[f"{a}_minus_{b}"] = {
            "mean_delta": round(statistics.fmean(diffs), 2),
            "mean_ci95": [round(v, 2) for v in bootstrap_ci(diffs)],
            "p95_delta": round(statistics.fmean(p95_diffs), 2),
            "p95_ci95": [round(v, 2) for v in bootstrap_ci(p95_diffs)],
        }
    report["deltas"] = deltas
    decisions = max(1.0, milp_totals.get("milp_decisions", 0.0))
    report["milp_online"] = {
        "decisions": int(milp_totals.get("milp_decisions", 0)),
        "executed": int(milp_totals.get("milp_executed", 0)),
        "fallback": int(milp_totals.get("milp_fallback", 0)),
        "prediction_gap_fallback": int(milp_totals.get("milp_gap_fallback", 0)),
        "solve_ms_total": round(milp_totals.get("milp_solve_ms", 0.0), 1),
        "solve_ms_per_decision": round(milp_totals.get("milp_solve_ms", 0.0) / decisions, 2),
        "exposed_ms_total": round(milp_totals.get("milp_exposed_ms", 0.0), 1),
        "exposed_ms_per_episode": round(milp_totals.get("milp_exposed_ms", 0.0) / max(1, len(episodes)), 2),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "deltas"}, indent=1, ensure_ascii=False))
    print("deltas:", json.dumps(deltas, indent=1, ensure_ascii=False))
    print(f"evidence: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
