#!/usr/bin/env python3
"""Run a bounded R8 CP-RHO matrix on frozen scheduler episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from tracing.analysis.workload_v02_simulator import (
    load_future_artifacts,
    load_templates,
    predicted_future_cost,
    read_jsonl,
    simulate_episode,
    train_resource_stats,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(len(ordered) - 1, low + 1)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--future-artifacts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--horizons", default="5")
    parser.add_argument("--time-limit-s", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 1 or args.time_limit_s <= 0.0:
        parser.error("--limit must be positive and --time-limit-s must be positive")
    horizons = tuple(int(value) for value in args.horizons.split(",") if value.strip())
    if not horizons or any(value not in (1, 3, 5) for value in horizons):
        parser.error("--horizons must contain only 1, 3 or 5")

    templates = load_templates(args.templates)
    episodes = read_jsonl(args.episodes)[: args.limit]
    artifacts = load_future_artifacts(args.future_artifacts)
    train_stats = train_resource_stats(templates)
    for node_id, artifact in artifacts.items():
        for horizon in (1, 3, 5):
            artifact[f"_cost_h{horizon}"] = predicted_future_cost(node_id, artifacts, train_stats, horizon)

    policies = tuple(f"cp_rho_h{horizon}" for horizon in horizons) + tuple(
        f"predopt_v2_h{horizon}" for horizon in horizons
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for episode_index, episode in enumerate(episodes, 1):
        for policy in policies:
            context = {
                "cp_rho_time_limit_s": args.time_limit_s,
                "cp_rho_seed": args.seed,
                "cp_rho_num_workers": 1,
            }
            summary, _events = simulate_episode(
                episode,
                templates,
                policy,
                future_artifacts=artifacts,
                train_stats=train_stats,
                collect_events=False,
                policy_context=context,
            )
            results.append(summary)
        if episode_index % 5 == 0 or episode_index == len(episodes):
            print(json.dumps({"progress_episodes": episode_index, "target_episodes": len(episodes)}, ensure_ascii=False), flush=True)

    result_path = args.output_dir / "scheduler_results.jsonl"
    result_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in results) + "\n",
        encoding="utf-8",
    )
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_policy[str(row["policy"])].append(row)
    aggregate: dict[str, dict[str, Any]] = {}
    for policy, rows in sorted(by_policy.items()):
        completions = [float(row["mean_completion_ms"]) for row in rows if row.get("mean_completion_ms") is not None]
        solve_times = [float(row.get("cp_rho_mean_solve_ms") or 0.0) for row in rows if policy.startswith("cp_rho_")]
        aggregate[policy] = {
            "episodes": len(rows),
            "failed_jobs": sum(int(row.get("failed_jobs") or 0) for row in rows),
            "mean_completion_ms": statistics.fmean(completions) if completions else None,
            "p95_completion_ms": quantile(completions, 0.95),
            "mean_deadline_miss_rate": statistics.fmean(float(row.get("deadline_miss_rate") or 0.0) for row in rows),
            "mean_gpu_evictions": statistics.fmean(float(row.get("gpu_evictions") or 0.0) for row in rows),
            "mean_cp_rho_solve_ms": statistics.fmean(solve_times) if solve_times else None,
            "max_cp_rho_p95_solve_ms": max((float(row.get("cp_rho_p95_solve_ms") or 0.0) for row in rows), default=None) if solve_times else None,
            "cp_rho_fallbacks": sum(int(row.get("cp_rho_fallbacks") or 0) for row in rows),
            "cp_rho_timeouts": sum(int(row.get("cp_rho_timeouts") or 0) for row in rows),
        }
    report = {
        "schema_version": "r8-cp-rho-matrix-v0.1",
        "status": "passed" if len(results) == len(episodes) * len(policies) and all(value["failed_jobs"] == 0 for value in aggregate.values()) else "failed",
        "episodes": len(episodes),
        "policies": list(policies),
        "result_rows": len(results),
        "expected_result_rows": len(episodes) * len(policies),
        "aggregate": aggregate,
        "solver": {
            "time_limit_s": args.time_limit_s,
            "num_workers": 1,
            "seed": args.seed,
            "solver_model": "current-ready assignment + per-GPU optional intervals/NoOverlap; predictor future cost coefficient",
        },
        "source_templates_sha256": sha256(args.templates),
        "source_episodes_sha256": sha256(args.episodes),
        "future_artifacts_manifest_sha256": sha256(args.future_artifacts / "b05_artifact_manifest.json"),
        "information_boundary": "deployable current ready nodes plus train-only resource estimates and frozen finite-horizon artifacts; execution truth remains engine-only",
    }
    (args.output_dir / "cp_rho_matrix_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "episodes": len(episodes), "result_rows": len(results), "aggregate": aggregate}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
