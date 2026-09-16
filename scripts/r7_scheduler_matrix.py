#!/usr/bin/env python3
"""Summary-only R7 scheduler matrix on the predictor-unseen workload.

The event simulator remains the execution engine, but this runner discards
per-event logs and writes one compact result row per (episode, policy).  This
keeps the 1,000-episode matrix auditable without recreating a multi-gigabyte
event log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from tracing.analysis.workload_v02_simulator import (
    ALIGNED_H5_POLICIES,
    POLICIES,
    load_future_artifacts,
    load_templates,
    lookup_path_report,
    read_jsonl,
    predicted_future_cost,
    set_artifact_context,
    set_train_stats_context,
    simulate_episode,
    train_resource_stats,
)


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(len(ordered) - 1, low + 1)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_cp_rho_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-decision CP-SAT diagnostics for one episode."""

    if not events:
        return {"decisions": 0}
    solve_ms = [float(event.get("solve_ms") or 0.0) for event in events]
    gaps = [float(event["gap"]) for event in events if event.get("gap") is not None]
    statuses: dict[str, int] = {}
    for event in events:
        status = str(event.get("status") or "unknown")
        statuses[status] = statuses.get(status, 0) + 1
    at_limit = sum(1 for value in solve_ms if value >= 250.0)
    return {
        "decisions": len(events),
        "fallbacks": sum(1 for event in events if event.get("fallback")),
        "fallback_rate": sum(1 for event in events if event.get("fallback")) / len(events),
        "mean_solve_ms": statistics.fmean(solve_ms),
        "p95_solve_ms": quantile(solve_ms, 0.95),
        "max_solve_ms": max(solve_ms),
        "time_limit_hits": at_limit,
        "time_limit_hit_rate": at_limit / len(events),
        "mean_gap": statistics.fmean(gaps) if gaps else None,
        "statuses": statuses,
    }


def code_fingerprint(root: Path) -> dict[str, str]:
    """Hash the files that define experiment semantics.

    A resume that reuses an output directory produced by different code or
    different inputs would silently mix stale rows into the report; the
    fingerprint is written next to the results so the runner can refuse it.
    """

    files = (
        "src/tracing/analysis/workload_v02_simulator.py",
        "src/tracing/scheduling/event_engine.py",
        "src/tracing/scheduling/cp_rho.py",
        "scripts/r7_scheduler_matrix.py",
    )
    out: dict[str, str] = {}
    for name in files:
        path = root / name
        if path.is_file():
            out[name] = sha256(path)
    return out


def directory_digest(root: Path, pattern: str = "*.jsonl.gz") -> str | None:
    """Hash the actual artifact payloads (not just the manifest).

    Perturbation scripts rewrite the gzip files while copying the source
    manifest, so a manifest hash alone cannot tell original and perturbed
    artifact packs apart.
    """

    files = sorted(path for path in root.glob(pattern) if path.is_file())
    if not files:
        return None
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256(path).encode("utf-8"))
    return digest.hexdigest()


def build_fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    artifacts_manifest = args.future_artifacts / "b05_artifact_manifest.json"
    return {
        "schema_version": "r7-run-fingerprint-v1",
        "templates_sha256": sha256(args.templates),
        "episodes_sha256": sha256(args.episodes),
        "episode_ids_file": str(args.episode_ids_file) if args.episode_ids_file else None,
        "episode_ids_sha256": sha256(args.episode_ids_file) if args.episode_ids_file else None,
        "artifacts_manifest_sha256": sha256(artifacts_manifest) if artifacts_manifest.is_file() else None,
        "artifacts_payload_digest": directory_digest(args.future_artifacts),
        "code": code_fingerprint(Path(__file__).resolve().parents[1]),
    }


def fingerprint_matches(stored: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    diffs: list[str] = []
    for key in (
        "templates_sha256",
        "episodes_sha256",
        "episode_ids_sha256",
        "artifacts_manifest_sha256",
        "artifacts_payload_digest",
        "code",
    ):
        if stored.get(key) != current.get(key):
            diffs.append(key)
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--future-artifacts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--episode-ids-file", type=Path, default=None, help="optional newline-delimited episode id allowlist (dev/confirm split)")
    parser.add_argument(
        "--allow-fingerprint-change",
        action="store_true",
        help="allow resuming an output directory whose inputs/code fingerprint differs (records a warning)",
    )
    parser.add_argument("--policies", default=",".join(POLICIES))
    args = parser.parse_args()
    policies = tuple(value.strip() for value in args.policies.split(",") if value.strip())
    unknown = sorted(set(policies) - set(POLICIES) - set(ALIGNED_H5_POLICIES))
    if unknown:
        raise ValueError(f"unknown policies: {unknown}")
    templates = load_templates(args.templates)
    set_train_stats_context(f"{args.templates}:{len(templates)}")
    episodes = read_jsonl(args.episodes)
    if args.episode_ids_file is not None:
        wanted = {line.strip() for line in args.episode_ids_file.read_text(encoding="utf-8").splitlines() if line.strip()}
        episodes = [episode for episode in episodes if str(episode.get("episode_id")) in wanted]
        if len(episodes) != len(wanted):
            raise ValueError(f"episode allowlist mismatch: {len(episodes)} matched vs {len(wanted)} requested")
    episodes = episodes[: args.limit]
    artifacts = load_future_artifacts(args.future_artifacts)
    set_artifact_context(str(args.future_artifacts))
    train_stats = train_resource_stats(templates)
    # Materialize predicted suffix costs once.  Without this cache every
    # ready-node comparison would rescan the beam scenarios repeatedly.
    for node_id, artifact in artifacts.items():
        for horizon in (1, 3, 5):
            artifact[f"_cost_h{horizon}"] = predicted_future_cost(node_id, artifacts, train_stats, horizon)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = build_fingerprint(args)
    fingerprint_id = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    fingerprint_path = args.output_dir / "run_fingerprint.json"
    results_path = args.output_dir / "scheduler_results.jsonl"
    if fingerprint_path.is_file() and results_path.is_file():
        stored = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        diffs = fingerprint_matches(stored, fingerprint)
        if diffs and not args.allow_fingerprint_change:
            raise SystemExit(
                "refusing to resume: output directory was produced with different inputs/code "
                f"({', '.join(diffs)}); use a fresh --output-dir or pass --allow-fingerprint-change"
            )
        if diffs:
            print(json.dumps({"warning": "fingerprint mismatch allowed by flag", "diffs": diffs}))
    elif results_path.is_file():
        # Fail closed: results written before fingerprints existed cannot be
        # verified, and silently appending would mix unverifiable rows in.
        raise SystemExit(
            "refusing to resume: existing results have no run_fingerprint.json; "
            "use a fresh --output-dir or pass --allow-fingerprint-change to accept legacy rows"
        )
    fingerprint_path.write_text(
        json.dumps({**fingerprint, "fingerprint_id": fingerprint_id}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    existing: set[tuple[str, str]] = set()
    stale_rows: set[str] = set()
    if results_path.is_file():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing.add((str(row.get("episode_id")), str(row.get("policy"))))
                row_fingerprint = row.get("run_fingerprint")
                if row_fingerprint is not None and str(row_fingerprint) != fingerprint_id:
                    stale_rows.add(str(row_fingerprint))
    if stale_rows and not args.allow_fingerprint_change:
        raise SystemExit(
            "refusing to resume: existing rows carry a different run fingerprint "
            f"({sorted(stale_rows)}); use a fresh --output-dir or pass --allow-fingerprint-change"
        )
    mode = "a" if results_path.exists() else "w"
    with results_path.open(mode, encoding="utf-8") as handle:
        for episode_index, episode in enumerate(episodes):
            episode_id = str(episode["episode_id"])
            for policy in policies:
                key = (episode_id, policy)
                if key in existing:
                    continue
                policy_context = {} if policy.startswith("cp_rho_") else None
                row, _events = simulate_episode(
                    episode,
                    templates,
                    policy,
                    future_artifacts=artifacts,
                    train_stats=train_stats,
                    collect_events=False,
                    policy_context=policy_context,
                )
                if policy_context is not None:
                    row["cp_rho_diagnostics"] = summarize_cp_rho_events(policy_context.get("cp_rho_events") or [])
                row["run_fingerprint"] = fingerprint_id
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
            if (episode_index + 1) % 25 == 0 or episode_index + 1 == len(episodes):
                print(json.dumps({"progress_episodes": episode_index + 1, "target_episodes": len(episodes), "policies": len(policies)}, ensure_ascii=False), flush=True)

    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_policy[str(row["policy"])].append(row)
    aggregate: dict[str, dict[str, Any]] = {}
    capacity_violations: list[dict[str, Any]] = []
    episode_by_id = {str(row["episode_id"]): row for row in episodes}
    for policy, values in sorted(by_policy.items()):
        completion = [float(row["mean_completion_ms"]) for row in values if row.get("mean_completion_ms") is not None]
        queue = [float(row["mean_job_queue_ms"]) for row in values]
        deadline = [float(row["deadline_miss_rate"]) for row in values]
        evictions = [int(row["gpu_evictions"]) for row in values]
        utilization = [float(item) for row in values for item in row.get("gpu_utilization") or []]
        peaks = [float(item) for row in values for item in row.get("gpu_peak_memory_mb") or []]
        for row in values:
            episode = episode_by_id.get(str(row["episode_id"]), {})
            capacities = [float(item) for item in episode.get("gpu_topology_mb") or []]
            row_peaks = [float(item) for item in row.get("gpu_peak_memory_mb") or []]
            if len(capacities) != len(row_peaks) or any(peak > cap + 1e-6 for peak, cap in zip(row_peaks, capacities)):
                capacity_violations.append({"policy": policy, "episode_id": row["episode_id"], "peaks": row_peaks, "capacities": capacities})
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
    gaps: dict[str, Any] = {}
    if "oracle" in aggregate:
        oracle = aggregate["oracle"]["mean_completion_ms"]
        for policy, values in aggregate.items():
            if policy == "oracle" or oracle in (None, 0) or values["mean_completion_ms"] is None:
                continue
            gap = values["mean_completion_ms"] - oracle
            gaps[policy] = {"minus_oracle_mean_completion_ms": gap, "relative_gap": gap / oracle}
    expected_rows = len(episodes) * len(policies)
    per_policy_expected = len(episodes)
    coverage_ok = all(len(values) == per_policy_expected for values in by_policy.values()) and set(by_policy) == set(policies)
    report = {
        "schema_version": "r7-scheduler-matrix-v0.1",
        "status": "passed" if len(rows) == expected_rows and coverage_ok and not capacity_violations and all(value["failed_jobs"] == 0 for value in aggregate.values()) else "failed",
        "episodes": len(episodes),
        "policies": list(policies),
        "result_rows": len(rows),
        "expected_result_rows": expected_rows,
        "aggregate": aggregate,
        "oracle_gaps": gaps,
        "capacity_violation_count": len(capacity_violations),
        "capacity_violation_examples": capacity_violations[:5],
        "source_templates_sha256": sha256(args.templates),
        "source_episodes_sha256": sha256(args.episodes),
        "future_artifacts_manifest_sha256": sha256(args.future_artifacts / "b05_artifact_manifest.json"),
        "run_fingerprint": fingerprint,
        "run_fingerprint_id": fingerprint_id,
        "mixed_row_fingerprints": sorted({str(row.get("run_fingerprint")) for row in rows}),
        "event_logging": "disabled_in_matrix_runner; event invariants covered by sim_smoke_50",
        "lookup_path_counts": lookup_path_report(),
        "information_boundary": {
            "myopic_optimizer": "current ready nodes plus train-only resource estimates",
            "predopt_h": "B05 current-prefix artifact and train-only finite-horizon rollout",
            "trueopt_h": "actual DAG successor identities and engine truth for reference only",
            "oracle": "full suffix truth reference only",
        },
    }
    (args.output_dir / "scheduler_matrix_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "episodes": len(episodes), "result_rows": len(rows), "aggregate": aggregate}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
