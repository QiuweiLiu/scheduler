"""Pressure-grid comparison (EXP-20261010_pressure_grid_v1) — exp C.

Frozen design: docs/research/2026-10-10_exp_c_pressure_grid.md

3 arrival-scale levels (alpha 1.00 / 0.90 / 0.80, frozen stress-v2 transform
re-used by import) x 3 pinned-capacity levels (24/24, 32/24, 32/32 GB, natural
fleet topologies) x 3 arms (myopic / F0 sameshape_h5_p95 / main line
pdrs_resident).  Cells are built deterministically IN MEMORY from the frozen
natural episode file; the artifact records the per-cell aggregate hash, alpha,
capacity and the natural source sha so any drift is detectable.

Per cell: paired episode bootstrap of (arm - F0) on the SAME confirm300
episodes (ids preserved across cells).  Zero-failure hard abort.  No
cross-cell significance search: the pre-registered monotone orderings and
point-estimate heat map are the deliverables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pressure_grid_lib import (  # noqa: E402
    CAPACITY_MB,
    PRESSURE_GRID_VERSION,
    build_pressure_cell,
    cell_aggregate_sha256,
    cell_id,
    cell_ids,
)
from tracing.analysis.profile_contract import apply_real_workload_profile_contract  # noqa: E402
from tracing.analysis.workload_v02_simulator import (  # noqa: E402
    load_resource_v2_overlay,
    load_templates,
    read_jsonl,
    simulate_episode,
    train_resource_stats,
)

PROJECTION = (ROOT / "results/processed/r7_workload_v041_ontology_no_run_container"
              / "job_templates_r7_v041.jsonl")
EPISODES_FILE = (ROOT / "results/processed/r7_workload_v03_no_run_container"
                 / "episodes/workload_validation_r7_v03.jsonl")
SPLIT_MANIFEST = ROOT / "data/manifests/validation_split_dev700_confirm300.json"
FROZEN_PROJECTION_SHA = (
    "15c62dafd99701b3883a3fe47034b5c7771f8fdcc8d6226fa73ff06fdca7f03b")
EXTENSION_CONFIG = (ROOT / "experiments/EXP-20261004_real_strata_calibration_v1"
                    / "artifacts/substrate_extension_v7.json")
BASE_ARTIFACTS = ROOT / "outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts"
F0_ARTIFACTS = ROOT / "outputs/resource_v2_artifacts/f0_seed11"
FUTURE_HORIZON = 5
REFERENCE = "sameshape_h5_p95"
SEED = 11
BOOTSTRAP = 2000
ART = ROOT / "experiments/EXP-20261010_pressure_grid_v1/artifacts"
DESIGN_DOC = "docs/research/2026-10-10_exp_c_pressure_grid.md"

ARM_LABELS = ("myopic", "pdrs_resident")

METRICS = ("mean_completion_ms", "p95_completion_ms", "deadline_miss_rate", "makespan_ms")
MECHANISM_FIELDS = (
    "gpu_evictions",
    "prefetch_count",
    "prefetch_capacity_failures",
    "wasted_prefetches",
    "used_prefetches",
    "cold_loads_on_demand",
    "evicted_then_reloaded",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    import os
    import subprocess

    env_dir = os.environ.get("SCHEDULER_GIT_DIR")
    if env_dir:
        try:
            out = subprocess.run(
                ["git", "--git-dir", env_dir, "--work-tree", str(ROOT), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=30)
            value = out.stdout.strip()
            if value:
                return value
        except Exception:  # noqa: BLE001
            pass
    return "unknown"


def run_arm(policy: str, label: str, episodes: Sequence[Dict[str, Any]],
            templates: Dict[str, Any], stats: Dict[str, Any],
            extension_config: Dict[str, Any],
            future_artifacts: Dict[str, Any],
            ) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    values: Dict[str, List[float]] = {metric: [] for metric in METRICS}
    activation: Counter[str] = Counter()
    mechanism: Counter[str] = Counter()
    for index, episode in enumerate(episodes):
        policy_context: Dict[str, Any] = {"activation": {}}
        summary, _ = simulate_episode(
            episode, templates, policy, train_stats=stats,
            policy_context=policy_context, future_artifacts=future_artifacts,
            future_horizon=FUTURE_HORIZON, extension_config=extension_config,
            collect_events=False,
        )
        expected_jobs = len(episode.get("jobs") or [])
        completed = int(summary.get("completed_jobs") or 0)
        failed = int(summary.get("failed_jobs") or 0)
        if failed != 0 or completed != expected_jobs:
            raise AssertionError(
                "%s episode %s: completed %d/%d with %d failed"
                % (label, episode.get("episode_id", index), completed, expected_jobs, failed))
        for metric in METRICS:
            value = float(summary[metric])
            if not math.isfinite(value):
                raise AssertionError(f"{label} episode {index}: non-finite {metric}")
            values[metric].append(value)
        activation.update(policy_context["activation"])
        for field_name in MECHANISM_FIELDS:
            mechanism[field_name] += int(round(float(summary.get(field_name) or 0.0)))
    mechanism["episodes"] = len(episodes)
    return values, {"activation": dict(sorted(activation.items())),
                    "mechanism": dict(sorted(mechanism.items()))}


def paired_bootstrap_ci(deltas: Sequence[float], n_boot: int = BOOTSTRAP,
                        seed: int = SEED) -> Tuple[float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--out", default="pressure_grid_v1.json")
    parser.add_argument("--cells", default="", help="comma-separated cell ids (default: all)")
    parser.add_argument("--arms", default="", help="comma-separated arm labels")
    args = parser.parse_args()
    arm_labels = tuple(a for a in args.arms.split(",") if a) or ARM_LABELS
    selected_cells = tuple(c for c in args.cells.split(",") if c) or tuple(cell_ids())
    for cell in selected_cells:
        if cell not in cell_ids():
            raise AssertionError("unknown cell id: %s" % cell)

    if args.formal:
        if not (ROOT / DESIGN_DOC).exists():
            raise AssertionError("--formal requires %s" % DESIGN_DOC)
        head = git_head()
        if not head or head == "unknown" or len(head) != 40:
            raise AssertionError("--formal: git HEAD is %r" % (head,))
    assert sha256_file(PROJECTION) == FROZEN_PROJECTION_SHA, "projection hash mismatch"

    extension_config = json.loads(EXTENSION_CONFIG.read_text(encoding="utf-8"))
    templates = load_templates(PROJECTION, topology_view="causal_v3")
    split_manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    wanted = set(split_manifest["confirm"])
    natural_by_id = {str(e.get("episode_id")): e for e in read_jsonl(EPISODES_FILE)}
    natural = [natural_by_id[eid] for eid in sorted(wanted) if eid in natural_by_id]
    if args.smoke:
        natural = natural[:args.smoke]

    stats = train_resource_stats({k: v for k, v in templates.items() if v.split == "train"})
    future_artifacts, preflight = load_resource_v2_overlay(BASE_ARTIFACTS, F0_ARTIFACTS)

    alpha_for = {cell: float(cell.split("_")[0][1:]) / 100.0 for cell in cell_ids()}
    capacity_for = {cell: cell.split("_", 1)[1] for cell in cell_ids()}

    config = {
        "split": "confirm",
        "formal": bool(args.formal),
        "n_episodes_per_cell": len(natural),
        "primary_metric": "mean_completion_ms",
        "reference": REFERENCE,
        "arms": list(arm_labels),
        "cells": list(selected_cells),
        "grid_version": PRESSURE_GRID_VERSION,
        "alphas": [alpha_for[c] for c in selected_cells],
        "capacities_mb": [list(CAPACITY_MB[capacity_for[c]]) for c in selected_cells],
        "natural_episodes_file": str(EPISODES_FILE.relative_to(ROOT)),
        "natural_episodes_sha256": sha256_file(EPISODES_FILE),
        "seed": SEED,
        "bootstrap": BOOTSTRAP,
        "projection_sha256": sha256_file(PROJECTION),
        "extension_config_sha256": sha256_file(EXTENSION_CONFIG),
        "reference_pack": preflight.get("artifact_id"),
        "reference_artifact_sha256": preflight.get("artifact_sha256"),
        "future_horizon": FUTURE_HORIZON,
        "smoke_episodes": args.smoke or None,
        "git_head": git_head(),
    }
    print("== config ==")
    for key, value in config.items():
        print("   %-28s %s" % (key, value))

    started = time.time()
    cells: Dict[str, Any] = {}
    for cell in selected_cells:
        alpha = alpha_for[cell]
        capacity_key = capacity_for[cell]
        cell_episodes = build_pressure_cell(natural, alpha, capacity_key)
        cell_sha = cell_aggregate_sha256(cell_episodes)
        cell_templates, cell_episodes = apply_real_workload_profile_contract(
            templates, cell_episodes, extension_config)
        print("\n######## cell %s (alpha=%.2f, cap=%s, sha=%s) ########"
              % (cell, alpha, capacity_key, cell_sha[:16]))
        reference_values, reference_info = run_arm(
            REFERENCE, REFERENCE, cell_episodes, cell_templates, stats,
            extension_config, future_artifacts)
        print("   F0 mean %.1f | p95 %.1f" % (
            statistics.fmean(reference_values["mean_completion_ms"]),
            statistics.fmean(reference_values["p95_completion_ms"])))
        cell_results: Dict[str, Any] = {
            "alpha": alpha,
            "capacity_key": capacity_key,
            "capacity_mb": list(CAPACITY_MB[capacity_key]),
            "episode_aggregate_sha256": cell_sha,
            "reference_summary": {
                "metrics": {m: statistics.fmean(reference_values[m]) for m in METRICS},
                "episode_values": {m: reference_values[m] for m in METRICS},
                "info": reference_info,
            },
            "results": {},
        }
        for label in arm_labels:
            values, info = run_arm(label, label, cell_episodes, cell_templates, stats,
                                   extension_config, future_artifacts)
            per_metric: Dict[str, Any] = {}
            for metric in METRICS:
                deltas = [a - b for a, b in zip(values[metric], reference_values[metric])]
                lo, hi = paired_bootstrap_ci(deltas)
                per_metric[metric] = {
                    "mean": statistics.fmean(values[metric]),
                    "delta_point": statistics.fmean(deltas),
                    "delta_ci95": [lo, hi],
                    "n_worse": sum(1 for d in deltas if d > 0),
                    "episode_values": values[metric],
                }
            cell_results["results"][label] = {"metrics": per_metric, **info}
            print("   %-16s mean %10.1f  delta %+9.1f CI95 [%+.1f, %+.1f] | p95 delta %+9.1f"
                  % (label, per_metric["mean_completion_ms"]["mean"],
                     per_metric["mean_completion_ms"]["delta_point"],
                     per_metric["mean_completion_ms"]["delta_ci95"][0],
                     per_metric["mean_completion_ms"]["delta_ci95"][1],
                     per_metric["p95_completion_ms"]["delta_point"]))
            print("      activation: %s" % json.dumps(info["activation"])[:180])
        cells[cell] = cell_results

    print("\n== heat summary: delta vs same-cell F0 (negative = better) ==")
    header = "%-16s %-14s" % ("cell", "arm")
    print(header + "  mean_delta  [CI95]                      p95_delta")
    for cell, entry in cells.items():
        for label, result in entry["results"].items():
            m = result["metrics"]["mean_completion_ms"]
            p = result["metrics"]["p95_completion_ms"]
            print("%-16s %-14s %+10.1f [%+.0f,%+.0f]  %+10.1f"
                  % (cell, label, m["delta_point"], m["delta_ci95"][0], m["delta_ci95"][1],
                     p["delta_point"]))

    elapsed = time.time() - started
    artifact = {
        "gate": "pressure_grid",
        "version": "v1",
        "mode": "smoke" if args.smoke else "formal",
        "config": config,
        "cells": cells,
        "criteria": {
            "delta": "arm - sameshape_h5_p95 within the same cell (negative = better)",
            "h1_load": "main-line gain vs F0 should not shrink as alpha decreases (per capacity column)",
            "h2_capacity": "gain should grow as capacity tightens at alpha=0.80",
            "metrics": list(METRICS),
        },
        "wall_seconds": elapsed,
    }
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / args.out
    out.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\nwrote %s (%.1f s)" % (out.relative_to(ROOT), elapsed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
