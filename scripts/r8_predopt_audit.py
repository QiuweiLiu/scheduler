#!/usr/bin/env python3
"""Paired P0 audit for legacy PredOpt and the R8 soft-priority objective.

The runner replays the same frozen episodes with ``predopt_h*`` and
``predopt_v2_h*`` and records action divergence together with the normal
summary metrics.  It deliberately does not read T_final or execution truth
for policy decisions; execution truth remains inside the simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import tracing.analysis.workload_v02_simulator as simulator
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


def action_sequence(events: list[dict[str, Any]]) -> list[tuple[str, int | None]]:
    return [
        (str(event.get("node_id")), event.get("gpu_index"))
        for event in events
        if event.get("event_type") == "node_dispatch"
    ]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completion = [float(row["mean_completion_ms"]) for row in rows if row.get("mean_completion_ms") is not None]
    deadline = [float(row.get("deadline_miss_rate") or 0.0) for row in rows]
    queue = [float(row.get("mean_job_queue_ms") or 0.0) for row in rows]
    evictions = [int(row.get("gpu_evictions") or 0) for row in rows]
    return {
        "episodes": len(rows),
        "completed_jobs": sum(int(row.get("completed_jobs") or 0) for row in rows),
        "failed_jobs": sum(int(row.get("failed_jobs") or 0) for row in rows),
        "mean_completion_ms": statistics.fmean(completion) if completion else None,
        "p95_completion_ms": quantile(completion, 0.95),
        "mean_deadline_miss_rate": statistics.fmean(deadline) if deadline else None,
        "mean_job_queue_ms": statistics.fmean(queue) if queue else None,
        "mean_gpu_evictions": statistics.fmean(evictions) if evictions else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--future-artifacts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, choices=(1, 3, 5), default=5)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--priority-weight", type=float, default=simulator.PREDOPT_V2_PRIORITY_WEIGHT)
    parser.add_argument("--future-weight", type=float, default=simulator.PREDOPT_V2_FUTURE_WEIGHT)
    parser.add_argument("--memory-weight", type=float, default=simulator.PREDOPT_V2_MEMORY_WEIGHT)
    parser.add_argument("--skip-action-audit", action="store_true", help="skip event collection for faster S_train weight calibration")
    args = parser.parse_args()

    simulator.PREDOPT_V2_PRIORITY_WEIGHT = args.priority_weight
    simulator.PREDOPT_V2_FUTURE_WEIGHT = args.future_weight
    simulator.PREDOPT_V2_MEMORY_WEIGHT = args.memory_weight

    templates = load_templates(args.templates)
    episodes = read_jsonl(args.episodes)[: args.limit]
    artifacts = load_future_artifacts(args.future_artifacts)
    train_stats = train_resource_stats(templates)
    for node_id, artifact in artifacts.items():
        for horizon in (1, 3, 5):
            artifact[f"_cost_h{horizon}"] = predicted_future_cost(node_id, artifacts, train_stats, horizon)

    legacy = f"predopt_h{args.horizon}"
    revised = f"predopt_v2_h{args.horizon}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired_path = args.output_dir / "paired_decision_audit.jsonl"
    legacy_rows: list[dict[str, Any]] = []
    revised_rows: list[dict[str, Any]] = []
    divergence_count = 0
    divergence_decisions = 0
    paired_rows: list[dict[str, Any]] = []

    for index, episode in enumerate(episodes, 1):
        summaries: dict[str, dict[str, Any]] = {}
        sequences: dict[str, list[tuple[str, int | None]]] = {}
        for policy in (legacy, revised):
            summary, events = simulate_episode(
                episode,
                templates,
                policy,
                future_artifacts=artifacts,
                train_stats=train_stats,
                collect_events=not args.skip_action_audit,
            )
            summaries[policy] = dict(summary)
            sequences[policy] = action_sequence(events)
            (legacy_rows if policy == legacy else revised_rows).append(dict(summary))

        left = sequences[legacy]
        right = sequences[revised]
        compared = max(len(left), len(right))
        diffs = sum(a != b for a, b in zip(left, right)) + abs(len(left) - len(right))
        if diffs:
            divergence_count += 1
        divergence_decisions += diffs
        paired_rows.append(
            {
                "episode_id": str(episode["episode_id"]),
                "legacy_policy": legacy,
                "revised_policy": revised,
                "legacy_summary": summaries[legacy],
                "revised_summary": summaries[revised],
                "legacy_dispatch_count": len(left),
                "revised_dispatch_count": len(right),
                "compared_dispatch_count": compared,
                "action_diff_count": diffs,
                "action_diff_rate": diffs / max(1, compared),
            }
        )
        if index % 25 == 0 or index == len(episodes):
            print(json.dumps({"progress_episodes": index, "target_episodes": len(episodes), "horizon": args.horizon}, ensure_ascii=False), flush=True)

    paired_path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in paired_rows) + "\n", encoding="utf-8")
    legacy_aggregate = aggregate(legacy_rows)
    revised_aggregate = aggregate(revised_rows)
    report = {
        "schema_version": "r8-predopt-objective-audit-v0.1",
        "status": "passed" if len(paired_rows) == len(episodes) and legacy_aggregate["failed_jobs"] == 0 and revised_aggregate["failed_jobs"] == 0 else "failed",
        "episodes": len(episodes),
        "horizon": args.horizon,
        "policies": [legacy, revised],
        "legacy": legacy_aggregate,
        "revised": revised_aggregate,
        "mean_completion_delta_revised_minus_legacy_ms": (
            revised_aggregate["mean_completion_ms"] - legacy_aggregate["mean_completion_ms"]
            if revised_aggregate["mean_completion_ms"] is not None and legacy_aggregate["mean_completion_ms"] is not None
            else None
        ),
        "episodes_with_action_differences": None if args.skip_action_audit else divergence_count,
        "episode_action_difference_rate": None if args.skip_action_audit else divergence_count / max(1, len(episodes)),
        "total_action_difference_count": None if args.skip_action_audit else divergence_decisions,
        "action_audit_collected": not args.skip_action_audit,
        "source_templates_sha256": sha256(args.templates),
        "source_episodes_sha256": sha256(args.episodes),
        "future_artifacts_manifest_sha256": sha256(args.future_artifacts / "b05_artifact_manifest.json"),
        "objective": {
            "priority_weight": args.priority_weight,
            "current_scale_ms": 100000.0,
            "future_scale_ms": 100000.0,
            "future_weight": args.future_weight,
            "memory_ratio_weight": args.memory_weight,
            "future_information": "frozen B05 finite-horizon artifact only",
        },
        "information_boundary": "deployable current ready nodes plus train-only resource estimates and frozen finite-horizon artifacts; execution truth remains engine-only",
        "paired_decision_audit": str(paired_path.name),
    }
    (args.output_dir / "predopt_audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
