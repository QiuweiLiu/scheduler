#!/usr/bin/env python3
"""Paired, scenario-cell-stratified validation statistics for R7.

The sampling unit is an episode, paired by ``episode_id`` and resampled inside
each scenario cell.  RL policies are aggregated across seeds per episode for
the primary cross-seed comparison; seed-specific RL comparisons are retained
in the report.  The matrix stores summary rows, so ``p95_completion_ms`` is
reported only as an episode-level diagnostic and is never called global
job-level P95.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MAIN_METRICS = (
    "mean_completion_ms",
    "deadline_miss_rate",
    "mean_job_queue_ms",
    "gpu_evictions",
)
DIAGNOSTIC_METRICS = ("p95_completion_ms",)
ALL_METRICS = MAIN_METRICS + DIAGNOSTIC_METRICS
PREDICTED_POLICIES = ("predopt_h1", "predopt_h3", "predopt_h5")
RL_POLICIES = ("rl_0", "rl_h5")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric(row: dict[str, Any], field: str) -> float:
    value = row.get(field)
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"non-finite {field} in episode {row.get('episode_id')}")
    return float(value)


def by_episode(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        episode_id = str(row["episode_id"])
        if episode_id in result:
            raise ValueError(f"duplicate episode_id: {episode_id}")
        result[episode_id] = row
    return result


def load_scheduler_results(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(path):
        policy = str(row["policy"])
        episode_id = str(row["episode_id"])
        if episode_id in result[policy]:
            raise ValueError(f"duplicate scheduler key: {policy}/{episode_id}")
        result[policy][episode_id] = row
    return result


def load_rl_results(root: Path, seeds: tuple[int, ...]) -> tuple[
    dict[str, dict[int, dict[str, dict[str, Any]]]], dict[str, dict[str, dict[str, Any]]]
]:
    per_seed: dict[str, dict[int, dict[str, dict[str, Any]]]] = defaultdict(dict)
    aggregate: dict[str, dict[str, dict[str, Any]]] = {}
    for policy in RL_POLICIES:
        seed_maps: list[dict[str, dict[str, Any]]] = []
        for seed in seeds:
            path = root / f"{policy}_seed{seed}" / "validation_results.jsonl"
            rows = by_episode(read_jsonl(path))
            per_seed[policy][seed] = rows
            seed_maps.append(rows)
        episode_ids = set(seed_maps[0])
        if any(set(rows) != episode_ids for rows in seed_maps[1:]):
            raise ValueError(f"RL seed episode sets differ for {policy}")
        averaged: dict[str, dict[str, Any]] = {}
        for episode_id in sorted(episode_ids):
            first = dict(seed_maps[0][episode_id])
            first["policy"] = policy
            first["seed_count"] = len(seeds)
            for field in ALL_METRICS:
                first[field] = statistics.fmean(numeric(rows[episode_id], field) for rows in seed_maps)
            first["failed_jobs"] = sum(int(rows[episode_id].get("failed_jobs") or 0) for rows in seed_maps)
            averaged[episode_id] = first
        aggregate[policy] = averaged
    return per_seed, aggregate


def cell_index(episodes: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    cells: dict[str, list[str]] = defaultdict(list)
    for episode_id, row in episodes.items():
        cells[str(row["scenario_cell"])].append(episode_id)
    return {cell: sorted(ids) for cell, ids in sorted(cells.items())}


def bootstrap_delta(
    deltas_by_cell: dict[str, np.ndarray],
    bootstrap_count: int,
    seed: int,
) -> np.ndarray:
    total = sum(len(values) for values in deltas_by_cell.values())
    samples = np.zeros(bootstrap_count, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for values in deltas_by_cell.values():
        indices = rng.integers(0, len(values), size=(bootstrap_count, len(values)))
        samples += values[indices].sum(axis=1)
    return samples / float(total)


def paired_metric(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    episodes: dict[str, dict[str, Any]],
    cells: dict[str, list[str]],
    metric: str,
    bootstrap_count: int,
    seed: int,
) -> dict[str, Any]:
    episode_ids = sorted(set(left) & set(right) & set(episodes))
    if len(episode_ids) != len(episodes) or set(left) != set(episodes) or set(right) != set(episodes):
        raise ValueError(f"unpaired episode set for metric {metric}")
    deltas = {episode_id: numeric(left[episode_id], metric) - numeric(right[episode_id], metric) for episode_id in episode_ids}
    deltas_by_cell = {cell: np.asarray([deltas[episode_id] for episode_id in ids], dtype=np.float64) for cell, ids in cells.items()}
    bootstrap = bootstrap_delta(deltas_by_cell, bootstrap_count, seed)
    cell_means = np.asarray([values.mean() for values in deltas_by_cell.values()], dtype=np.float64)
    mean_delta = float(np.mean(list(deltas.values())))
    right_mean = statistics.fmean(numeric(right[episode_id], metric) for episode_id in episode_ids)
    lower_tail = (float(np.sum(bootstrap >= 0.0)) + 1.0) / float(bootstrap_count + 1)
    upper_tail = (float(np.sum(bootstrap <= 0.0)) + 1.0) / float(bootstrap_count + 1)
    result = {
        "metric": metric,
        "metric_scope": "episode_summary" if metric != "p95_completion_ms" else "episode_p95_diagnostic_only",
        "n_episodes": len(episode_ids),
        "n_cells": len(cells),
        "mean_left": statistics.fmean(numeric(left[episode_id], metric) for episode_id in episode_ids),
        "mean_right": right_mean,
        "mean_delta_left_minus_right": mean_delta,
        "relative_improvement_left_vs_right": (-mean_delta / right_mean) if right_mean else None,
        "ci95_delta_left_minus_right": [float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))],
        "bootstrap_probability_left_better": float(np.mean(bootstrap < 0.0)),
        "bootstrap_p_value_two_sided": min(1.0, 2.0 * min(lower_tail, upper_tail)),
        "episode_win_rate_left_better": float(np.mean(np.asarray(list(deltas.values())) < 0.0)),
        "cell_win_rate_left_better": float(np.mean(cell_means < 0.0)),
    }
    return result


def comparison(
    name: str,
    left_name: str,
    right_name: str,
    policy_rows: dict[str, dict[str, dict[str, Any]]],
    episodes: dict[str, dict[str, Any]],
    cells: dict[str, list[str]],
    bootstrap_count: int,
    seed: int,
) -> dict[str, Any]:
    left = policy_rows[left_name]
    right = policy_rows[right_name]
    metrics = {}
    for index, metric in enumerate(ALL_METRICS):
        metrics[metric] = paired_metric(left, right, episodes, cells, metric, bootstrap_count, seed + index)
    return {"left": left_name, "right": right_name, "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler-results", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--rl-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-count", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260818)
    parser.add_argument("--rl-seeds", default="11,22,33")
    args = parser.parse_args()
    if args.bootstrap_count < 1000:
        parser.error("bootstrap-count must be >= 1000")
    seeds = tuple(int(value) for value in args.rl_seeds.split(",") if value.strip())
    episodes = by_episode(read_jsonl(args.episodes))
    cells = cell_index(episodes)
    scheduler = load_scheduler_results(args.scheduler_results)
    rl_per_seed, rl_aggregate = load_rl_results(args.rl_root, seeds)
    policy_rows: dict[str, dict[str, dict[str, Any]]] = {**scheduler, **rl_aggregate}
    required = ("myopic", "optimizer_0", *PREDICTED_POLICIES, *RL_POLICIES)
    missing = [policy for policy in required if policy not in policy_rows]
    if missing:
        raise ValueError(f"missing policies: {missing}")
    row_audit = {
        policy: {
            "episodes": len(policy_rows[policy]),
            "failed_jobs": sum(int(row.get("failed_jobs") or 0) for row in policy_rows[policy].values()),
        }
        for policy in required
    }
    comparisons = [
        (f"{policy}_vs_myopic", policy, "myopic") for policy in PREDICTED_POLICIES
    ] + [
        (f"{policy}_vs_optimizer_0", policy, "optimizer_0") for policy in PREDICTED_POLICIES
    ] + [
        ("rl_h5_vs_rl_0", "rl_h5", "rl_0"),
        ("rl_h5_vs_myopic", "rl_h5", "myopic"),
        ("rl_h5_vs_predopt_h5", "rl_h5", "predopt_h5"),
        ("rl_0_vs_myopic", "rl_0", "myopic"),
    ]
    comparison_results = {}
    for index, (name, left, right) in enumerate(comparisons):
        comparison_results[name] = comparison(name, left, right, policy_rows, episodes, cells, args.bootstrap_count, args.bootstrap_seed + index * 100,)

    seed_results = {}
    for seed_index, seed in enumerate(seeds):
        seed_rows = {
            "rl_0": rl_per_seed["rl_0"][seed],
            "rl_h5": rl_per_seed["rl_h5"][seed],
        }
        seed_results[str(seed)] = {
            "rl_h5_vs_rl_0": comparison(
                f"rl_h5_seed{seed}_vs_rl_0_seed{seed}",
                "rl_h5",
                "rl_0",
                seed_rows,
                episodes,
                cells,
                args.bootstrap_count,
                args.bootstrap_seed + 10000 + seed_index * 100,
            ),
            "rl_0_vs_myopic": comparison(
                f"rl_0_seed{seed}_vs_myopic",
                "rl_0",
                "myopic",
                {"rl_0": rl_per_seed["rl_0"][seed], "myopic": policy_rows["myopic"]},
                episodes,
                cells,
                args.bootstrap_count,
                args.bootstrap_seed + 11000 + seed_index * 100,
            ),
            "rl_h5_vs_myopic": comparison(
                f"rl_h5_seed{seed}_vs_myopic",
                "rl_h5",
                "myopic",
                {"rl_h5": rl_per_seed["rl_h5"][seed], "myopic": policy_rows["myopic"]},
                episodes,
                cells,
                args.bootstrap_count,
                args.bootstrap_seed + 12000 + seed_index * 100,
            ),
        }

    def holm_adjust(values: dict[str, float]) -> dict[str, float]:
        ordered = sorted(values.items(), key=lambda item: item[1])
        adjusted: dict[str, float] = {}
        running = 0.0
        count = len(ordered)
        for rank, (name, value) in enumerate(ordered):
            running = max(running, min(1.0, (count - rank) * value))
            adjusted[name] = running
        return adjusted

    multiple_comparison: dict[str, Any] = {}
    for metric in MAIN_METRICS:
        multiple_comparison[metric] = {}
        for baseline in ("myopic", "optimizer_0"):
            names = [f"{policy}_vs_{baseline}" for policy in PREDICTED_POLICIES]
            raw = {name: comparison_results[name]["metrics"][metric]["bootstrap_p_value_two_sided"] for name in names}
            multiple_comparison[metric][f"vs_{baseline}"] = {
                "method": "Holm",
                "raw_p_values": raw,
                "holm_p_values": holm_adjust(raw),
            }

    rl_seed_direction = {
        "completion_all_seeds_left_better": all(
            seed_results[str(seed)]["rl_h5_vs_rl_0"]["metrics"]["mean_completion_ms"]["mean_delta_left_minus_right"] < 0
            for seed in seeds
        ),
        "deadline_all_seeds_left_better": all(
            seed_results[str(seed)]["rl_h5_vs_rl_0"]["metrics"]["deadline_miss_rate"]["mean_delta_left_minus_right"] < 0
            for seed in seeds
        ),
    }

    report = {
        "schema_version": "r7-paired-policy-stats-v0.1",
        "status": "passed" if len(episodes) == 1000 and len(cells) == 135 and all(item["episodes"] == len(episodes) and item["failed_jobs"] == 0 for item in row_audit.values()) else "failed",
        "bootstrap": {"count": args.bootstrap_count, "seed": args.bootstrap_seed, "unit": "episode_within_scenario_cell"},
        "validation": {
            "episodes": len(episodes),
            "scenario_cells": len(cells),
            "cell_episode_count_min": min(len(ids) for ids in cells.values()),
            "cell_episode_count_max": max(len(ids) for ids in cells.values()),
            "expected_scenario_cells": 135,
            "source_episodes_sha256": sha256(args.episodes),
            "source_scheduler_results_sha256": sha256(args.scheduler_results),
        },
        "row_audit": row_audit,
        "comparisons": comparison_results,
        "rl_seed_comparisons": seed_results,
        "multiple_comparison": multiple_comparison,
        "selection_gates": {
            "predopt_h5_is_lowest_mean_completion_among_h1_h3_h5": min(
                comparison_results[f"{policy}_vs_myopic"]["metrics"]["mean_completion_ms"]["mean_left"] for policy in PREDICTED_POLICIES
            )
            == comparison_results["predopt_h5_vs_myopic"]["metrics"]["mean_completion_ms"]["mean_left"],
            "rl_h5_direction_consistent_across_seeds": rl_seed_direction,
            "rl_h5_final_freeze_allowed": all(rl_seed_direction.values()),
        },
        "p95_note": "p95_completion_ms is an episode-level summary diagnostic. A global job-level P95 is not computed because this matrix stores no job-level completion values; final T_final must collect them.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "paired_policy_stats.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "episodes": len(episodes), "scenario_cells": len(cells), "comparisons": len(comparison_results), "output": str(output)}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
