#!/usr/bin/env python3
"""Run the opt-in predicted rolling-horizon smoke/matrix."""

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


DEFAULT_POLICIES = (
    "myopic",
    "predopt_h5",
    "cp_rho_h5",
    "pred_mpc_h3",
    "pred_mpc_h5",
    "pred_mpc_h5_no_terminal",
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
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--future-artifacts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    policies = tuple(value.strip() for value in args.policies.split(",") if value.strip())
    if not policies:
        parser.error("--policies must not be empty")

    templates = load_templates(args.templates)
    episodes = read_jsonl(args.episodes)[: args.limit]
    artifacts = load_future_artifacts(args.future_artifacts)
    train_stats = train_resource_stats(templates)
    for node_id, artifact in artifacts.items():
        for horizon in (1, 3, 5):
            artifact[f"_cost_h{horizon}"] = predicted_future_cost(node_id, artifacts, train_stats, horizon)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for episode_index, episode in enumerate(episodes, 1):
        for policy in policies:
            policy_context: dict[str, Any] = {}
            if policy.startswith("cp_rho_h"):
                policy_context.update({
                    "cp_rho_time_limit_s": 0.25,
                    "cp_rho_seed": 0,
                    "cp_rho_num_workers": 1,
                })
            summary, _events = simulate_episode(
                episode,
                templates,
                policy,
                future_artifacts=artifacts,
                train_stats=train_stats,
                collect_events=False,
                policy_context=policy_context,
            )
            results.append(summary)
        print(
            json.dumps(
                {"progress_episodes": episode_index, "target_episodes": len(episodes)},
                ensure_ascii=False,
            ),
            flush=True,
        )

    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_policy[str(row["policy"])].append(row)
    aggregate: dict[str, dict[str, Any]] = {}
    for policy, rows in sorted(by_policy.items()):
        completions = [float(row["mean_completion_ms"]) for row in rows if row.get("mean_completion_ms") is not None]
        aggregate[policy] = {
            "episodes": len(rows),
            "failed_jobs": sum(int(row.get("failed_jobs") or 0) for row in rows),
            "completed_jobs": sum(int(row.get("completed_jobs") or 0) for row in rows),
            "mean_completion_ms": statistics.fmean(completions) if completions else None,
            "p95_completion_ms": quantile(completions, 0.95),
            "mean_deadline_miss_rate": statistics.fmean(float(row.get("deadline_miss_rate") or 0.0) for row in rows),
            "mean_job_queue_ms": statistics.fmean(float(row.get("mean_job_queue_ms") or 0.0) for row in rows),
            "mean_gpu_evictions": statistics.fmean(float(row.get("gpu_evictions") or 0.0) for row in rows),
            "mean_pred_mpc_rollout_ms": statistics.fmean(
                float(row.get("pred_mpc_mean_rollout_ms") or 0.0) for row in rows
            ) if any("pred_mpc_mean_rollout_ms" in row for row in rows) else None,
            "mean_pred_mpc_terminal_ms": statistics.fmean(
                float(row.get("pred_mpc_mean_terminal_ms") or 0.0) for row in rows
            ) if any("pred_mpc_mean_terminal_ms" in row for row in rows) else None,
        }

    report = {
        "schema_version": "r8-pred-mpc-matrix-v0.1",
        "status": "passed" if len(results) == len(episodes) * len(policies) and all(
            value["failed_jobs"] == 0 for value in aggregate.values()
        ) else "failed",
        "episodes": len(episodes),
        "policies": list(policies),
        "result_rows": len(results),
        "expected_result_rows": len(episodes) * len(policies),
        "aggregate": aggregate,
        "objective": "mean job completion time; active-job-area identity is the diagnostic global objective",
        "information_boundary": "current ready GPU nodes plus train-only resource estimates and frozen finite-horizon artifacts; execution truth remains engine-only",
        "rollout_boundary": "PredMPC simulates only the current ready-node candidate window for known jobs; continuation cost comes from frozen future artifacts, not template suffix nodes",
        "source_templates_sha256": sha256(args.templates),
        "source_episodes_sha256": sha256(args.episodes),
        "future_artifacts_manifest_sha256": sha256(args.future_artifacts / "b05_artifact_manifest.json"),
    }
    (args.output_dir / "scheduler_results.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in results) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "r8-pred-mpc-matrix-v0.1",
                "templates": str(args.templates),
                "episodes": str(args.episodes),
                "future_artifacts": str(args.future_artifacts),
                "limit": args.limit,
                "policies": list(policies),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "episodes": len(episodes), "aggregate": aggregate}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
