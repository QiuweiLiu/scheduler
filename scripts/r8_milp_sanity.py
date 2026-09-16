#!/usr/bin/env python3
"""Run the bounded R8-P2 MILP/CP-SAT action-agreement sanity check."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from tracing.analysis.workload_v02_simulator import (
    estimate,
    load_future_artifacts,
    load_templates,
    predicted_future_cost,
    read_jsonl,
    train_resource_stats,
)
from tracing.scheduling.cp_rho import candidate_from_scheduler_row, solve_first_action
from tracing.scheduling.milp_sanity import solve_first_action_milp


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--future-artifacts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--nodes-per-case", type=int, default=4)
    parser.add_argument("--time-limit-s", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.cases < 1 or args.nodes_per_case < 2 or args.time_limit_s <= 0.0:
        parser.error("cases/nodes-per-case must be positive and time-limit-s must be > 0")

    templates = load_templates(args.templates)
    episodes = read_jsonl(args.episodes)
    if not episodes:
        raise ValueError("episodes input is empty")
    train_stats = train_resource_stats(templates)
    artifacts = load_future_artifacts(args.future_artifacts)
    for node_id, artifact in artifacts.items():
        for horizon in (1, 3, 5):
            artifact[f"_cost_h{horizon}"] = predicted_future_cost(node_id, artifacts, train_stats, horizon)

    gpu_nodes = [
        (template_id, node)
        for template_id, template in sorted(templates.items())
        for node in template.nodes
        if node.lane == "gpu"
    ]
    if len(gpu_nodes) < args.nodes_per_case:
        raise ValueError(f"only {len(gpu_nodes)} GPU nodes available")

    results: list[dict[str, Any]] = []
    for case_index in range(args.cases):
        episode = episodes[case_index % len(episodes)]
        capacities = [float(value) for value in episode.get("gpu_topology_mb") or []]
        capacities = capacities[:2] or [32768.0, 32768.0]
        if len(capacities) == 1:
            capacities.append(capacities[0])
        candidates = []
        for node_offset in range(args.nodes_per_case):
            template_id, node = gpu_nodes[(case_index * args.nodes_per_case + node_offset) % len(gpu_nodes)]
            estimate_row = estimate(node, train_stats)
            priority = float((case_index + node_offset) % 2)
            for gpu_index, capacity in enumerate(capacities):
                cache_hit = gpu_index == (case_index + node_offset) % len(capacities)
                candidates.append(
                    candidate_from_scheduler_row(
                        index=len(candidates),
                        node_key=(case_index, f"{template_id}:{node.node_id}"),
                        gpu_index=gpu_index,
                        priority=priority,
                        ready_since_ms=float(node_offset * 10 + case_index),
                        model_id=node.model_id,
                        estimate_row=estimate_row,
                        cache_hit=cache_hit,
                        future_cost_ms=predicted_future_cost(node.node_id, artifacts, train_stats, 5),
                        capacity_mb=capacity,
                    )
                )
        cp_result = solve_first_action(
            candidates,
            horizon=5,
            time_limit_s=args.time_limit_s,
            random_seed=args.seed,
            num_workers=1,
        )
        milp_result = solve_first_action_milp(candidates, time_limit_s=args.time_limit_s)
        cp_candidate = candidates[cp_result.candidate_index]
        milp_candidate = candidates[milp_result.candidate_index]
        cp_action = [list(cp_candidate.node_key), cp_candidate.gpu_index]
        milp_action = [list(milp_candidate.node_key), milp_candidate.gpu_index]
        objective_gap = None
        if cp_result.objective is not None and milp_result.objective is not None:
            objective_gap = abs(cp_result.objective - milp_result.objective) / max(1.0, abs(cp_result.objective))
        results.append(
            {
                "case_index": case_index,
                "candidate_count": len(candidates),
                "cp_rho": cp_result.to_dict(),
                "milp": milp_result.to_dict(),
                "cp_action": cp_action,
                "milp_action": milp_action,
                "gpu_capacities_mb": capacities,
                "candidate_summary": [
                    {
                        "index": candidate.index,
                        "node_key": list(candidate.node_key),
                        "gpu_index": candidate.gpu_index,
                        "priority": candidate.priority,
                        "ready_since_ms": candidate.ready_since_ms,
                        "model_id": candidate.model_id,
                        "runtime_ms": candidate.runtime_ms,
                        "load_ms": candidate.load_ms,
                        "memory_ratio": candidate.memory_ratio,
                        "future_cost_ms": candidate.future_cost_ms,
                        "duration_ms": candidate.duration_ms,
                    }
                    for candidate in candidates
                ],
                "exact_action_agreement": cp_action == milp_action,
                "node_action_agreement": cp_action[0] == milp_action[0],
                "objective_relative_gap": objective_gap,
            }
        )

    cp_statuses = Counter(row["cp_rho"]["status"] for row in results)
    milp_statuses = Counter(row["milp"]["status"] for row in results)
    objective_gaps = [row["objective_relative_gap"] for row in results if row["objective_relative_gap"] is not None]
    objective_gap_tolerance = 1e-6
    report = {
        "schema_version": "r8-milp-sanity-v0.1",
        "status": "passed"
        if len(results) == args.cases
        and all(not row["cp_rho"]["fallback"] for row in results)
        and all(not row["milp"]["fallback"] for row in results)
        and all(row["node_action_agreement"] for row in results)
        and len(objective_gaps) == len(results)
        and all(gap <= objective_gap_tolerance for gap in objective_gaps)
        else "failed",
        "cases": len(results),
        "nodes_per_case": args.nodes_per_case,
        "exact_action_agreement": sum(row["exact_action_agreement"] for row in results) / len(results),
        "node_action_agreement": sum(row["node_action_agreement"] for row in results) / len(results),
        "mean_objective_relative_gap": statistics.fmean(objective_gaps) if objective_gaps else None,
        "max_objective_relative_gap": max(objective_gaps) if objective_gaps else None,
        "objective_gap_tolerance": objective_gap_tolerance,
        "cp_rho_status_counts": dict(sorted(cp_statuses.items())),
        "milp_status_counts": dict(sorted(milp_statuses.items())),
        "cp_rho_mean_solve_ms": statistics.fmean(row["cp_rho"]["solve_ms"] for row in results),
        "milp_mean_solve_ms": statistics.fmean(row["milp"]["solve_ms"] for row in results),
        "command": " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv]),
        "seed": args.seed,
        "time_limit_s": args.time_limit_s,
        "source_code_sha256": {
            "cp_rho.py": sha256(Path(__file__).resolve().parents[1] / "src/tracing/scheduling/cp_rho.py"),
            "milp_sanity.py": sha256(Path(__file__).resolve().parents[1] / "src/tracing/scheduling/milp_sanity.py"),
            "workload_v02_simulator.py": sha256(
                Path(__file__).resolve().parents[1] / "src/tracing/analysis/workload_v02_simulator.py"
            ),
            "r8_milp_sanity.py": sha256(Path(__file__).resolve()),
        },
        "execution_context": {
            "cwd": str(Path.cwd()),
            "pythonpath": os.environ.get("PYTHONPATH"),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "scipy": __import__("scipy").__version__,
            "ortools": __import__("ortools").__version__,
            "milp_backend": "HiGHS via scipy.optimize.milp",
        },
        "information_boundary": "same current-ready candidate estimates, no execution truth, no suffix truth",
        "source_templates_sha256": sha256(args.templates),
        "source_episodes_sha256": sha256(args.episodes),
        "future_artifacts_manifest_sha256": sha256(args.future_artifacts / "b05_artifact_manifest.json"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "milp_sanity_results.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in results) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "milp_sanity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
