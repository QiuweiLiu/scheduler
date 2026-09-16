#!/usr/bin/env python3
"""Sample full-event counterfactual rollouts from Action-Value Audit states."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import tracing.analysis.workload_v02_simulator as simulator
from scripts.r8_action_value_audit import AUDIT_SCHEMA_VERSION, POLICIES, quantile
from tracing.analysis.workload_v02_simulator import (
    build_jobs,
    load_future_artifacts,
    load_templates,
    predicted_future_cost,
    read_jsonl,
    simulate_episode,
    train_resource_stats,
)


ROLLOUT_SCHEMA_VERSION = "action-value-rollout-audit-v0.1"
TIE_EPSILON_MS = 1e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _action_key(action: Mapping[str, Any], job_indices: Mapping[str, int]) -> tuple[int, str, int]:
    job_id = str(action.get("job_instance_id") or "")
    if job_id not in job_indices:
        raise ValueError(f"unknown job in action: {job_id}")
    return (job_indices[job_id], str(action["node_id"]), int(action["gpu_index"]))


def _select_records(records: Sequence[Mapping[str, Any]], limit: int) -> list[Mapping[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in sorted(records, key=lambda row: (str(row["episode_id"]), int(row["decision_index"]))):
        reference = record["policies"].get("predopt_h5")
        if reference is None:
            reference = next(iter(record["policies"].values()))
        error = "predopt_error" if not reference["true_top1_agreement"] else "predopt_top1"
        groups[(str(record["strict_width_bucket"]), error)].append(record)
    keys = sorted(groups)
    selected: list[Mapping[str, Any]] = []
    cursor = {key: 0 for key in keys}
    while len(selected) < limit:
        advanced = False
        for key in keys:
            index = cursor[key]
            if index >= len(groups[key]):
                continue
            selected.append(groups[key][index])
            cursor[key] += 1
            advanced = True
            if len(selected) >= limit:
                break
        if not advanced:
            break
    return selected


def _summarize_policy(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    regrets = [float(row["full_rollout_regret_ms"]) for row in rows if row.get("full_rollout_regret_ms") is not None]
    return {
        "sampled_decisions": len(rows),
        "chosen_action_coverage": sum(row.get("chosen_full_rollout_ms") is not None for row in rows) / len(rows) if rows else None,
        "full_rollout_top1_agreement_rate": (
            statistics.fmean(1.0 if row["full_rollout_top1_agreement"] else 0.0 for row in rows)
            if rows
            else None
        ),
        "mean_full_rollout_regret_ms": statistics.fmean(regrets) if regrets else None,
        "p95_full_rollout_regret_ms": quantile(regrets, 0.95),
    }


def run_rollout_audit(
    records: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    templates: Mapping[str, Any],
    future_artifacts: Mapping[str, Mapping[str, Any]],
    train_stats: Mapping[str, Mapping[str, Any]],
    reference_summaries: Mapping[str, Mapping[str, Any]],
    *,
    max_samples: int = 20,
    reference_policy: str = "myopic",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episode_by_id = {str(episode["episode_id"]): episode for episode in episodes}
    selected = _select_records(records, max_samples)
    policy_names = tuple(records[0]["policies"].keys()) if records else POLICIES
    rollout_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for record in selected:
        episode_id = str(record["episode_id"])
        if episode_id not in episode_by_id:
            failures.append({"episode_id": episode_id, "decision_index": record["decision_index"], "error": "episode_missing"})
            continue
        episode = episode_by_id[episode_id]
        jobs = build_jobs(episode, templates)
        job_indices = {job.job_instance_id: index for index, job in enumerate(jobs)}
        candidate_values: dict[tuple[int, str, int], dict[str, Any]] = {}
        for candidate in record["candidates"]:
            if not candidate.get("strict_feasible"):
                continue
            action = candidate["action"]
            key = _action_key(action, job_indices)
            context = {
                "audit_forced_decision_index": int(record["decision_index"]),
                "audit_forced_action": key,
            }
            try:
                summary, _events = simulate_episode(
                    episode,
                    templates,
                    reference_policy,
                    future_artifacts=future_artifacts,
                    train_stats=train_stats,
                    collect_events=False,
                    policy_context=context,
                )
            except Exception as exc:  # noqa: BLE001 - preserve the exact sampled failure in the artifact
                failures.append(
                    {
                        "episode_id": episode_id,
                        "decision_index": record["decision_index"],
                        "action": action,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            candidate_values[key] = {
                "action": action,
                "mean_completion_ms": summary.get("mean_completion_ms"),
                "deadline_miss_rate": summary.get("deadline_miss_rate"),
                "mean_job_queue_ms": summary.get("mean_job_queue_ms"),
                "failed_jobs": summary.get("failed_jobs"),
            }

        ordered_keys = [key for key, value in candidate_values.items() if value.get("mean_completion_ms") is not None]
        if not ordered_keys:
            continue
        best_value = min(float(candidate_values[key]["mean_completion_ms"]) for key in ordered_keys)
        best_keys = {
            key
            for key in ordered_keys
            if abs(float(candidate_values[key]["mean_completion_ms"]) - best_value) <= TIE_EPSILON_MS
        }
        policies: dict[str, Any] = {}
        for policy in policy_names:
            action = record["policies"][policy]["chosen_action"]
            key = _action_key(action, job_indices)
            value = candidate_values.get(key, {}).get("mean_completion_ms")
            policies[policy] = {
                "chosen_action": action,
                "chosen_full_rollout_ms": value,
                "full_rollout_regret_ms": (float(value) - best_value) if value is not None else None,
                "full_rollout_top1_agreement": bool(key in best_keys) if value is not None else False,
            }

        reference_summary = reference_summaries.get(episode_id) or {}
        reference_action = record["policies"][reference_policy]["chosen_action"]
        reference_key = _action_key(reference_action, job_indices)
        reference_value = candidate_values.get(reference_key, {}).get("mean_completion_ms")
        reference_baseline = reference_summary.get("mean_completion_ms")
        rollout_records.append(
            {
                "schema_version": ROLLOUT_SCHEMA_VERSION,
                "episode_id": episode_id,
                "decision_index": int(record["decision_index"]),
                "state_hash": record["state_hash"],
                "strict_width_bucket": record["strict_width_bucket"],
                "strict_feasible_width": record["strict_feasible_width"],
                "candidate_rollouts": list(candidate_values.values()),
                "best_full_rollout_ms": best_value,
                "reference_baseline_mean_completion_ms": reference_baseline,
                "reference_forced_action_mean_completion_ms": reference_value,
                "reference_forced_action_consistent": (
                    reference_value is not None
                    and reference_baseline is not None
                    and abs(float(reference_value) - float(reference_baseline)) <= TIE_EPSILON_MS
                ),
                "policies": policies,
            }
        )

    policy_rows = {
        policy: [row["policies"][policy] for row in rollout_records]
        for policy in policy_names
    }
    report = {
        "schema_version": ROLLOUT_SCHEMA_VERSION,
        "status": "passed" if rollout_records and not failures and all(row["reference_forced_action_consistent"] for row in rollout_records) else "failed",
        "reference_policy": reference_policy,
        "policies": list(policy_names),
        "sampled_decisions": len(selected),
        "completed_decisions": len(rollout_records),
        "candidate_rollouts": sum(len(row["candidate_rollouts"]) for row in rollout_records),
        "failures": failures,
        "q_definition": "full episode event-simulation mean completion after forcing one action at a common state",
        "information_boundary": "true future is used only inside this audit evaluator; policies continue with reference policy after the forced action",
        "policies": {policy: _summarize_policy(rows) for policy, rows in policy_rows.items()},
        "by_strict_width": {
            width: {
                "decisions": len(group),
                "policies": {
                    policy: _summarize_policy([row["policies"][policy] for row in group])
                    for policy in policy_names
                },
            }
            for width, group in _group_by_width(rollout_records).items()
        },
    }
    return report, rollout_records


def _group_by_width(records: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["strict_width_bucket"])].append(record)
    return dict(sorted(groups.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--future-artifacts", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--reference-policy", choices=("myopic", "predopt_h5"), default="myopic")
    args = parser.parse_args()

    templates = load_templates(args.templates)
    episodes = read_jsonl(args.episodes)
    future_artifacts = load_future_artifacts(args.future_artifacts)
    train_stats = train_resource_stats(templates)
    for node_id, artifact in future_artifacts.items():
        for horizon in (1, 3, 5):
            artifact[f"_cost_h{horizon}"] = predicted_future_cost(node_id, future_artifacts, train_stats, horizon)
    audit_records = read_jsonl(args.audit_dir / "decision_records.jsonl")
    reference_summaries = {
        str(row["episode_id"]): row
        for row in read_jsonl(args.audit_dir / "reference_summaries.jsonl")
    }
    report, records = run_rollout_audit(
        audit_records,
        episodes,
        templates,
        future_artifacts,
        train_stats,
        reference_summaries,
        max_samples=args.max_samples,
        reference_policy=args.reference_policy,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    simulator.write_jsonl(args.output_dir / "rollout_decision_records.jsonl", records)
    simulator.write_json(args.output_dir / "metrics.json", report)
    simulator.write_json(
        args.output_dir / "run_manifest.json",
        {
            "schema_version": ROLLOUT_SCHEMA_VERSION,
            "templates": str(args.templates),
            "templates_sha256": sha256(args.templates),
            "episodes": str(args.episodes),
            "episodes_sha256": sha256(args.episodes),
            "audit_records": str(args.audit_dir / "decision_records.jsonl"),
            "audit_records_sha256": sha256(args.audit_dir / "decision_records.jsonl"),
            "max_samples": args.max_samples,
            "reference_policy": args.reference_policy,
        },
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
