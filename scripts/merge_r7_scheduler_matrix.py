#!/usr/bin/env python3
"""Merge independently replayed R7 policy summaries into one report."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from r7_scheduler_matrix import quantile, sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--future-artifacts", type=Path, required=True)
    parser.add_argument("--policies", required=True)
    args = parser.parse_args()
    policies = tuple(value.strip() for value in args.policies.split(",") if value.strip())
    episodes = [json.loads(line) for line in args.episodes.read_text(encoding="utf-8").splitlines() if line.strip()]
    episode_by_id = {str(row["episode_id"]): row for row in episodes}
    rows: list[dict[str, Any]] = []
    policy_counts: dict[str, int] = {}
    duplicate_keys: list[tuple[str, str]] = []
    for policy in policies:
        path = args.input_root / policy / "scheduler_results.jsonl"
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        policy_counts[policy] = len(values)
        seen: set[tuple[str, str]] = set()
        for row in values:
            key = (str(row.get("episode_id")), str(row.get("policy")))
            if key in seen:
                duplicate_keys.append(key)
            seen.add(key)
            rows.append(row)
    rows.sort(key=lambda row: (str(row.get("episode_id")), policies.index(str(row.get("policy"))) if str(row.get("policy")) in policies else len(policies)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "scheduler_results.jsonl"
    result_path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")

    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    capacity_violations: list[dict[str, Any]] = []
    for row in rows:
        policy = str(row["policy"])
        by_policy[policy].append(row)
        episode = episode_by_id.get(str(row["episode_id"]), {})
        capacities = [float(value) for value in episode.get("gpu_topology_mb") or []]
        peaks = [float(value) for value in row.get("gpu_peak_memory_mb") or []]
        if len(capacities) != len(peaks) or any(peak > cap + 1e-6 for peak, cap in zip(peaks, capacities)):
            capacity_violations.append({"policy": policy, "episode_id": row["episode_id"], "peaks": peaks, "capacities": capacities})

    aggregate: dict[str, dict[str, Any]] = {}
    for policy, values in sorted(by_policy.items()):
        completion = [float(row["mean_completion_ms"]) for row in values if row.get("mean_completion_ms") is not None]
        queue = [float(row["mean_job_queue_ms"]) for row in values]
        deadline = [float(row["deadline_miss_rate"]) for row in values]
        evictions = [int(row["gpu_evictions"]) for row in values]
        utilization = [float(value) for row in values for value in row.get("gpu_utilization") or []]
        peaks = [float(value) for row in values for value in row.get("gpu_peak_memory_mb") or []]
        aggregate[policy] = {
            "episodes": len(values),
            "completed_jobs": sum(int(row.get("completed_jobs") or 0) for row in values),
            "failed_jobs": sum(int(row.get("failed_jobs") or 0) for row in values),
            "mean_completion_ms": statistics.fmean(completion) if completion else None,
            "p95_completion_ms": quantile(completion, 0.95),
            "mean_job_queue_ms": statistics.fmean(queue) if queue else None,
            "mean_deadline_miss_rate": statistics.fmean(deadline) if deadline else None,
            "mean_gpu_evictions": statistics.fmean(evictions) if evictions else None,
            "mean_gpu_utilization": statistics.fmean(utilization) if utilization else None,
            "max_gpu_peak_memory_mb": max(peaks) if peaks else None,
        }
    oracle_gaps: dict[str, Any] = {}
    if "oracle" in aggregate:
        oracle = aggregate["oracle"]["mean_completion_ms"]
        for policy, values in aggregate.items():
            if policy == "oracle" or oracle in (None, 0) or values["mean_completion_ms"] is None:
                continue
            gap = values["mean_completion_ms"] - oracle
            oracle_gaps[policy] = {"minus_oracle_mean_completion_ms": gap, "relative_gap": gap / oracle}
    expected_rows = len(episodes) * len(policies)
    report = {
        "schema_version": "r7-scheduler-matrix-v0.1",
        "status": "passed" if len(rows) == expected_rows and not duplicate_keys and not capacity_violations and all(policy_counts.get(policy) == len(episodes) for policy in policies) and all(values["failed_jobs"] == 0 for values in aggregate.values()) else "failed",
        "episodes": len(episodes),
        "policies": list(policies),
        "result_rows": len(rows),
        "expected_result_rows": expected_rows,
        "policy_row_counts": policy_counts,
        "duplicate_keys": duplicate_keys[:10],
        "aggregate": aggregate,
        "oracle_gaps": oracle_gaps,
        "capacity_violation_count": len(capacity_violations),
        "capacity_violation_examples": capacity_violations[:5],
        "source_templates_sha256": sha256(args.templates),
        "source_episodes_sha256": sha256(args.episodes),
        "future_artifacts_manifest_sha256": sha256(args.future_artifacts / "b05_artifact_manifest.json"),
        "event_logging": "disabled_in_matrix_runner; event invariants covered by sim_smoke_50",
        "information_boundary": {
            "myopic_optimizer": "current ready nodes plus train-only resource estimates",
            "predopt_h": "B05 current-prefix artifact and P_dev-train-only finite-horizon rollout",
            "trueopt_h": "actual DAG successor identities and engine truth for reference only",
            "oracle": "full suffix truth reference only",
        },
    }
    (args.output_dir / "scheduler_matrix_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "result_rows": len(rows), "policy_row_counts": policy_counts, "aggregate": aggregate}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
