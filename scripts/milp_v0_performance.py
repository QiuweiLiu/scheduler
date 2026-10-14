#!/usr/bin/env python3
"""MILP-v0a paired performance evaluation (serial_v0 environment).

Arms: M0 = F0 reference (sameshape_h5_p95), M1 = PDRS (pdrs_resident),
M2 = milp_v0a (online MILP, first action if legal else PDRS fallback).

Splits: --split train (dev pool, seen during tuning) or confirm (frozen
confirm300, workflow-source isolated; loaded via
data/manifests/validation_split_dev700_confirm300.json).

Reporting split (advisor ruling 2026-10-13): zero-overhead algorithm gain
(mean, pooled P95, mean-of-episode-P95, makespan) and independent online cost
(per-decision end-to-end wall time P50/P95, solve statuses, fallback rates,
exposed latency) are reported SEPARATELY; net JCT is NOT measured (no
event-level latency injection yet) and must not be claimed.
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
CONFIRM_SPLIT = ROOT / "data/manifests/validation_split_dev700_confirm300.json"
CONFIRM_EPISODES = (ROOT / "results/processed/r7_workload_v03_no_run_container"
                    / "episodes/workload_validation_r7_v03.jsonl")
TRAIN_EPISODES = (ROOT / "results/processed/r7_workload_20260817"
                  / "episodes/workload_train_r7.jsonl")


def load_split(split: str, episode_limit: int, config_path=None):
    from tracing.analysis.profile_contract import apply_real_workload_profile_contract
    from tracing.analysis.workload_v02_simulator import (
        load_resource_v2_overlay, load_templates, read_jsonl, train_resource_stats)

    serial = json.loads((config_path or T.SERIAL).read_text(encoding="utf-8"))
    templates = load_templates(
        ROOT / "results/processed/r7_workload_v041_ontology_no_run_container/job_templates_r7_v041.jsonl",
        topology_view="causal_v3")
    if split == "confirm":
        manifest = json.loads(CONFIRM_SPLIT.read_text(encoding="utf-8"))
        wanted = set(manifest["confirm"])
        by_id = {str(e.get("episode_id")): e for e in read_jsonl(CONFIRM_EPISODES)}
        episodes = [by_id[eid] for eid in sorted(wanted) if eid in by_id]
    else:
        episodes = read_jsonl(TRAIN_EPISODES)
    episodes = episodes[:episode_limit]
    templates, episodes = apply_real_workload_profile_contract(templates, episodes, serial)
    stats = train_resource_stats({k: v for k, v in templates.items() if v.split == "train"})
    artifacts, _ = load_resource_v2_overlay(
        ROOT / "outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts",
        ROOT / "outputs/resource_v2_artifacts/f0_seed11")
    return serial, templates, episodes, stats, artifacts


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


def quantile(values, q):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--budget", type=float, default=0.5)
    parser.add_argument("--split", choices=("train", "confirm"), default="train")
    parser.add_argument("--config", type=Path, default=None,
                        help="extension config path (default: serial_v0_extension.json)")
    parser.add_argument("--burst64", action="store_true",
                        help="load Burst64-v1 episodes (mixed-source; in-memory split override)")
    parser.add_argument("--burst-limit", type=int, default=0, help="cap episodes in burst64 mode (0=all)")
    parser.add_argument("--out", type=Path, default=EVIDENCE)
    args = parser.parse_args()

    if args.burst64:
        from dataclasses import replace as _replace
        from tracing.analysis.workload_v02_simulator import read_jsonl as _read_jsonl
        burst_dir = ROOT / "experiments/EXP-20261014_burst64_v1/artifacts"
        config_path = args.config or (ROOT / "experiments/EXP-20261012_milp_v0/artifacts"
                                      / "serial_prefetch_v1_extension.json")
        serial, templates, _nat, stats, artifacts = load_split("confirm", 1, config_path)
        # Diagnostic-only relabel: the simulator guards source isolation and the
        # MILP reads the train-only resource catalog by split; both sides are
        # relabeled to "train" for the Burst64 structural runs.  Provenance is
        # preserved in the workload files/manifest.
        templates = {k: _replace(v, split="train") for k, v in templates.items()}
        episodes = []
        for tag in ("r085_b10", "r110_b10", "r085_b15", "r110_b15"):
            episodes.extend(_read_jsonl(burst_dir / f"workload_burst64_{tag}.jsonl"))
        for episode in episodes:
            episode["split"] = "train"
        from tracing.analysis.profile_contract import apply_real_workload_profile_contract
        templates, episodes = apply_real_workload_profile_contract(templates, episodes, serial)
        if args.burst_limit:
            episodes = episodes[: args.burst_limit]
        print(f"burst64: {len(episodes)} episodes, config={config_path.name}")
    else:
        serial, templates, episodes, stats, artifacts = load_split(args.split, args.episodes, args.config)
    per_arm: dict = {name: {"mean": [], "p95": [], "makespan": [], "cold": [], "evict": [],
                            "wall": [], "dur_pool": []} for name in ARMS}
    milp_totals: dict = {}
    milp_solve_list: list = []
    for ep in episodes:
        arrivals = {str(row["job_instance_id"]): float(row["arrival_ms"])
                    for row in (ep.get("jobs") or [])}
        for name, policy in ARMS.items():
            ctx = {"activation": {}}
            if policy == "milp_v0a":
                ctx["milp_budget_s"] = args.budget
            t0 = time.perf_counter()
            summary, events = simulate_episode(
                ep, templates, policy, train_stats=stats, future_artifacts=artifacts,
                future_horizon=5, extension_config=serial, policy_context=ctx,
                collect_events=True)
            wall = time.perf_counter() - t0
            row = per_arm[name]
            row["mean"].append(float(summary["mean_completion_ms"]))
            row["p95"].append(float(summary["p95_completion_ms"]))
            row["makespan"].append(float(summary["makespan_ms"]))
            row["cold"].append(float(summary["cold_loads_on_demand"]))
            row["evict"].append(float(summary["gpu_evictions"]))
            row["wall"].append(wall)
            cond = str(ep.get("episode_id", "")).split("_ep")[0]
            per_arm[name].setdefault("by_cond", {}).setdefault(cond, []).append(
                float(summary["mean_completion_ms"]))
            for event in events or []:
                if event.get("event_type") == "job_finish":
                    jid = str(event.get("job_instance_id"))
                    if jid in arrivals:
                        row["dur_pool"].append(float(event["time_ms"]) - arrivals[jid])
            if policy == "milp_v0a":
                act = ctx.get("activation") or {}
                for key in ("milp_decisions", "milp_executed", "milp_fallback", "milp_gap_fallback",
                            "milp_solve_ms", "milp_exposed_ms", "milp_time_limit", "milp_optimal"):
                    milp_totals[key] = float(milp_totals.get(key, 0.0)) + float(act.get(key, 0.0))
                milp_solve_list.extend(act.get("milp_solve_ms_list") or [])

    report: dict = {
        "schema": "milp-v0a-performance-v2",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "episodes": len(episodes), "budget_s": args.budget, "split": args.split,
        "net_jct_note": ("net JCT NOT measured: zero-overhead gain and independent online "
                         "cost are reported separately (advisor ruling 2026-10-13); no "
                         "event-level latency injection in this run"),
        "arms": {},
    }
    for name, row in per_arm.items():
        report["arms"][name] = {
            "mean": round(statistics.fmean(row["mean"]), 2),
            "p95_pooled": round(quantile(row["dur_pool"], 0.95), 2),
            "p95_mean_of_episodes": round(statistics.fmean(row["p95"]), 2),
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
            "p95_delta_mean_of_episodes": round(statistics.fmean(p95_diffs), 2),
            "p95_ci95_mean_of_episodes": [round(v, 2) for v in bootstrap_ci(p95_diffs)],
        }
    report["deltas"] = deltas
    if args.burst64:
        report["by_condition"] = {}
        for cond in sorted(per_arm["M0_f0"]["by_cond"]):
            entry = {}
            for name in ARMS:
                entry[name] = round(statistics.fmean(per_arm[name]["by_cond"][cond]), 1)
            entry["M2_minus_M1"] = round(entry["M2_milp"] - entry["M1_pdrs"], 1)
            entry["M2_minus_M0"] = round(entry["M2_milp"] - entry["M0_f0"], 1)
            entry["M1_minus_M0"] = round(entry["M1_pdrs"] - entry["M0_f0"], 1)
            report["by_condition"][cond] = entry
    decisions = max(1.0, milp_totals.get("milp_decisions", 0.0))
    report["milp_online_cost"] = {
        "decisions": int(milp_totals.get("milp_decisions", 0)),
        "executed": int(milp_totals.get("milp_executed", 0)),
        "fallback": int(milp_totals.get("milp_fallback", 0)),
        "prediction_gap_fallback": int(milp_totals.get("milp_gap_fallback", 0)),
        "time_limit": int(milp_totals.get("milp_time_limit", 0)),
        "optimal": int(milp_totals.get("milp_optimal", 0)),
        "decision_wall_ms_p50": round(quantile(milp_solve_list, 0.5), 2) if milp_solve_list else None,
        "decision_wall_ms_p95": round(quantile(milp_solve_list, 0.95), 2) if milp_solve_list else None,
        "decision_wall_ms_mean": round(statistics.fmean(milp_solve_list), 2) if milp_solve_list else None,
        "decision_wall_ms_max": round(max(milp_solve_list), 2) if milp_solve_list else None,
        "solve_ms_total": round(milp_totals.get("milp_solve_ms", 0.0), 1),
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
