#!/usr/bin/env python3
"""Paired matrix for opt-in batch, paid-prefetch, and node-preemption semantics.

The legacy R7 matrix is intentionally untouched.  This runner supplies an
experiment configuration to ``simulate_episode`` and writes one compact row
per (episode, policy), plus optional event evidence for a small pilot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from tracing.analysis.workload_v02_simulator import (
    POLICIES,
    load_future_artifacts,
    load_templates,
    read_jsonl,
    simulate_episode,
    train_resource_stats,
)


EXTRA_POLICIES = {
    "batch_myopic",
    "batch_max_fit",
    "batch_throughput",
    "batch_round_robin",
    "myopic_preempt",
    "batch_myopic_preempt",
}


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
    parser.add_argument("--future-artifacts", type=Path)
    parser.add_argument("--extension-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--policies", required=True)
    parser.add_argument("--collect-events", action="store_true")
    args = parser.parse_args()

    policies = tuple(value.strip() for value in args.policies.split(",") if value.strip())
    unknown = sorted(set(policies) - (set(POLICIES) | EXTRA_POLICIES))
    if unknown:
        raise ValueError(f"unknown policies: {unknown}")
    extension_config = json.loads(args.extension_config.read_text(encoding="utf-8"))
    if not isinstance(extension_config, dict):
        raise ValueError("extension config must be a JSON object")
    templates = load_templates(args.templates)
    episodes = read_jsonl(args.episodes)[: args.limit]
    artifacts = load_future_artifacts(args.future_artifacts) if args.future_artifacts else {}
    train_stats = train_resource_stats(templates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "scheduler_results.jsonl"
    events_path = args.output_dir / "event_samples.jsonl"
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    with results_path.open("w", encoding="utf-8") as result_handle:
        event_handle = events_path.open("w", encoding="utf-8") if args.collect_events else None
        try:
            for episode_index, episode in enumerate(episodes):
                for policy in policies:
                    summary, episode_events = simulate_episode(
                        episode,
                        templates,
                        policy,
                        future_artifacts=artifacts,
                        train_stats=train_stats,
                        collect_events=args.collect_events,
                        extension_config=extension_config,
                    )
                    result_handle.write(json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n")
                    result_handle.flush()
                    rows.append(summary)
                    if event_handle is not None:
                        for event in episode_events:
                            event_handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
                        event_handle.flush()
                if (episode_index + 1) % 25 == 0 or episode_index + 1 == len(episodes):
                    print(json.dumps({"progress_episodes": episode_index + 1, "target_episodes": len(episodes), "policies": len(policies)}), flush=True)
        finally:
            if event_handle is not None:
                event_handle.close()

    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_policy[str(row["policy"])].append(row)
    aggregate: dict[str, dict[str, Any]] = {}
    capacity_violations = 0
    episode_by_id = {str(row["episode_id"]): row for row in episodes}
    for policy, values in sorted(by_policy.items()):
        completion = [float(row["mean_completion_ms"]) for row in values if row.get("mean_completion_ms") is not None]
        queues = [float(row.get("mean_job_queue_ms") or 0.0) for row in values]
        misses = [float(row.get("deadline_miss_rate") or 0.0) for row in values]
        evictions = [float(row.get("gpu_evictions") or 0.0) for row in values]
        preemptions = [float(row.get("preemptions") or 0.0) for row in values]
        recompute = [float(row.get("preempt_recompute_ms") or 0.0) for row in values]
        prefetch_loads = [float(row.get("prefetch_load_ms") or 0.0) for row in values]
        wasted_prefetches = [float(row.get("wasted_prefetches") or 0.0) for row in values]
        action_widths = [float(row["batch_action_width_p50"]) for row in values if row.get("batch_action_width_p50") is not None]
        peaks = [float(value) for row in values for value in row.get("gpu_peak_memory_mb") or []]
        for row in values:
            episode = episode_by_id[str(row["episode_id"])]
            capacities = [float(value) for value in episode.get("gpu_topology_mb") or []]
            row_peaks = [float(value) for value in row.get("gpu_peak_memory_mb") or []]
            if len(capacities) != len(row_peaks) or any(peak > cap + 1e-6 for peak, cap in zip(row_peaks, capacities)):
                capacity_violations += 1
        aggregate[policy] = {
            "episodes": len(values),
            "completed_jobs": sum(int(row.get("completed_jobs") or 0) for row in values),
            "failed_jobs": sum(int(row.get("failed_jobs") or 0) for row in values),
            "mean_completion_ms": statistics.fmean(completion) if completion else None,
            "p95_completion_ms": quantile(completion, 0.95),
            "mean_job_queue_ms": statistics.fmean(queues) if queues else None,
            "mean_deadline_miss_rate": statistics.fmean(misses) if misses else None,
            "mean_gpu_evictions": statistics.fmean(evictions) if evictions else None,
            "mean_preemptions": statistics.fmean(preemptions) if preemptions else None,
            "mean_preempt_recompute_ms": statistics.fmean(recompute) if recompute else None,
            "mean_prefetch_load_ms": statistics.fmean(prefetch_loads) if prefetch_loads else None,
            "mean_wasted_prefetches": statistics.fmean(wasted_prefetches) if wasted_prefetches else None,
            "mean_batch_action_width_p50": statistics.fmean(action_widths) if action_widths else None,
            "max_gpu_peak_memory_mb": max(peaks) if peaks else None,
        }
    expected_rows = len(episodes) * len(policies)
    report = {
        "schema_version": "r8-scheduler-extensions-matrix-v0.1",
        "status": "passed" if len(rows) == expected_rows and capacity_violations == 0 and all(row["failed_jobs"] == 0 for row in aggregate.values()) else "failed",
        "episodes": len(episodes),
        "policies": list(policies),
        "result_rows": len(rows),
        "expected_result_rows": expected_rows,
        "aggregate": aggregate,
        "capacity_violation_count": capacity_violations,
        "event_logging": "enabled" if args.collect_events else "disabled",
        "extension_config": extension_config,
        "source_templates_sha256": sha256(args.templates),
        "source_episodes_sha256": sha256(args.episodes),
        "source_extension_config_sha256": sha256(args.extension_config),
        "source_runner_sha256": sha256(Path(__file__).resolve()),
        "source_simulator_sha256": sha256(Path(__file__).resolve().parents[1] / "src/tracing/analysis/workload_v02_simulator.py"),
        "future_artifacts_manifest_sha256": sha256(args.future_artifacts / "b05_artifact_manifest.json") if args.future_artifacts else None,
        "execution_context": {
            "cwd": os.getcwd(),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "information_boundary": {
            "batch": "measured per-yolo-tool batch profiles; no cross-job dynamic batching",
            "prefetch": "explicit paid load plan; load time and resident memory are charged",
            "preemption": "node-level recompute after preempt; no free checkpoint/resume",
            "priority": "unchanged binary priority/normal service_class",
        },
    }
    (args.output_dir / "scheduler_extensions_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "episodes": len(episodes), "result_rows": len(rows), "aggregate": aggregate}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
