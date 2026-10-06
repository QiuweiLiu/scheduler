"""PDRS comparison v1 — prefix-conditioned receding-horizon scheduling (frozen design).

Design frozen before the run (docs/research/2026-10-05_architecture_round5_pdrs.md):

  reference    sameshape_h5_p95 (F0), the frozen deployment predictor
  arms         myopic               — no future information
               pdrs_o               — ordering only (survival-weighted suffix cost W)
               pdrs_place           — ordering identical to F0; only the GPU changes (H1 affinity A)
               pdrs_p               — both
               pdrs_prior           — pdrs_p with pooled unconditional priors (matched control)
               pdrs_shuffle         — pdrs_p with another instance's belief (matched control)
               pdrs_oracle          — true next-model identity + true limited-horizon cost (ceiling)
  metric       mean_completion_ms (primary); mechanism: evictions, prefetch load ms, deadline miss
  pairing      the SAME confirm300 episodes and the SAME seed per arm
  uncertainty  paired episode bootstrap (2000, seed 11)
  zero-failure any episode with failed_jobs>0 aborts the run

No machine-response table enters any PDRS consumer (see tracing.analysis.pdrs_methods).
"""
from __future__ import annotations

import argparse
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

from tracing.analysis.pdrs_methods import (  # noqa: E402
    build_prior_artifacts,
    build_shuffled_artifacts,
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
ART = ROOT / "experiments/EXP-20261005_pdrs_comparison_v1/artifacts"

ARM_LABELS = (
    "myopic",
    "pdrs_o",
    "pdrs_place",
    "pdrs_p",
    "pdrs_prior",
    "pdrs_shuffle",
    "pdrs_oracle",
)


def sha256_file(path: Path) -> str:
    import hashlib

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
    for candidate in (ROOT / ".." / "scheduler_public_repo", ROOT):
        try:
            out = subprocess.run(["git", "-C", str(candidate), "rev-parse", "HEAD"],
                                 capture_output=True, text=True, timeout=30)
            value = out.stdout.strip()
            if value:
                return value
        except Exception:  # noqa: BLE001
            continue
    return "unknown"


METRICS = ("mean_completion_ms", "p95_completion_ms", "deadline_miss_rate", "makespan_ms")


def run_arm(policy: str, label: str, episodes: Sequence[Dict[str, Any]],
            templates: Dict[str, Any], stats: Dict[str, Any],
            extension_config: Dict[str, Any],
            future_artifacts: Dict[str, Any]) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    values: Dict[str, List[float]] = {metric: [] for metric in METRICS}
    activation: Counter[str] = Counter()
    mechanism: Counter[str] = Counter()
    for index, episode in enumerate(episodes):
        policy_context = {"activation": {}}
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
        mechanism["gpu_evictions"] += int(summary.get("gpu_evictions") or 0)
        mechanism["prefetch_load_ms"] += int(round(float(summary.get("prefetch_load_ms") or 0.0)))
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
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--out", default="pdrs_comparison_v1.json")
    parser.add_argument("--arms", default="",
                        help="comma-separated arm labels (default: all registered arms)")
    args = parser.parse_args()
    arm_labels = tuple(a for a in args.arms.split(",") if a) or ARM_LABELS

    if args.formal:
        for record in ("docs/research/2026-10-05_architecture_round5_pdrs.md",
                       "docs/research/2026-10-05_positioning_ruling.md"):
            if not (ROOT / record).exists():
                raise AssertionError("--formal requires %s" % record)
        head = git_head()
        if not head or head == "unknown" or len(head) != 40:
            raise AssertionError("--formal: git HEAD is %r" % (head,))
    assert sha256_file(PROJECTION) == FROZEN_PROJECTION_SHA, "projection hash mismatch"

    extension_config = json.loads(EXTENSION_CONFIG.read_text(encoding="utf-8"))
    templates = load_templates(PROJECTION, topology_view="causal_v3")
    split_manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    wanted = set(split_manifest["confirm"])
    by_id = {str(e.get("episode_id")): e for e in read_jsonl(EPISODES_FILE)}
    episodes = [by_id[eid] for eid in sorted(wanted) if eid in by_id]
    if args.episodes:
        episodes = episodes[:args.episodes]
    if args.smoke:
        episodes = episodes[:args.smoke]

    templates, episodes = apply_real_workload_profile_contract(
        templates, episodes, extension_config)
    stats = train_resource_stats({k: v for k, v in templates.items() if v.split == "train"})
    future_artifacts, preflight = load_resource_v2_overlay(BASE_ARTIFACTS, F0_ARTIFACTS)

    variant_artifacts = {
        "pdrs_prior": build_prior_artifacts(future_artifacts),
        "pdrs_shuffle": build_shuffled_artifacts(future_artifacts, seed=SEED),
    }

    config = {
        "split": "confirm",
        "formal": bool(args.formal),
        "n_episodes": len(episodes),
        "metric": "mean_completion_ms",
        "reference": REFERENCE,
        "arms": list(arm_labels),
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
    reference_values, reference_info = run_arm(
        REFERENCE, REFERENCE, episodes, templates, stats, extension_config, future_artifacts)
    print("\n== %s (reference) ==" % REFERENCE)
    for metric in METRICS:
        print("   %-20s %.4f" % (metric, statistics.fmean(reference_values[metric])))

    results: Dict[str, Any] = {}
    for label in arm_labels:
        if label == "pdrs_prior":
            policy, artifacts = "pdrs_p", variant_artifacts["pdrs_prior"]
        elif label == "pdrs_shuffle":
            policy, artifacts = "pdrs_p", variant_artifacts["pdrs_shuffle"]
        else:
            policy, artifacts = label, future_artifacts
        values, info = run_arm(policy, label, episodes, templates, stats,
                               extension_config, artifacts)
        per_metric: Dict[str, Any] = {}
        for metric in METRICS:
            deltas = [a - b for a, b in zip(values[metric], reference_values[metric])]
            point = statistics.fmean(deltas)
            lo, hi = paired_bootstrap_ci(deltas)
            per_metric[metric] = {
                "mean": statistics.fmean(values[metric]),
                "delta_point": point,
                "delta_ci95": [lo, hi],
                "n_worse": sum(1 for d in deltas if d > 0),
                "n_episodes": len(deltas),
            }
        results[label] = {
            "policy": policy,
            "metrics": per_metric,
            **info,
        }
        print("\n== %s ==" % label)
        for metric in METRICS:
            entry = per_metric[metric]
            print("   %-20s mean %12.4f  delta %+9.4f  CI95 [%+.4f, %+.4f]  %d/%d worse"
                  % (metric, entry["mean"], entry["delta_point"],
                     entry["delta_ci95"][0], entry["delta_ci95"][1],
                     entry["n_worse"], entry["n_episodes"]))
        print("   activation: %s" % json.dumps(info["activation"])[:160])

    elapsed = time.time() - started
    artifact = {
        "gate": "pdrs_comparison",
        "version": "v1",
        "mode": "smoke" if args.smoke else "formal",
        "config": config,
        "reference_summary": {
            "metrics": {metric: statistics.fmean(reference_values[metric]) for metric in METRICS},
            "n_episodes": len(reference_values[METRICS[0]]),
            "episode_values": {metric: reference_values[metric] for metric in METRICS},
            "info": reference_info,
        },
        "results": results,
        "criteria": {
            "delta": "candidate - %s (negative = better)" % REFERENCE,
            "metrics": "mean_completion_ms (main-table metric) + p95_completion_ms / "
                       "deadline_miss_rate / makespan_ms (tail set, declared before the run: "
                       "distributional information is tail-oriented)",
            "placement_value": "pdrs_p - pdrs_o",
            "instance_information": "pdrs_p vs pdrs_prior / pdrs_shuffle",
            "ceiling": "pdrs_oracle - pdrs_p",
        },
        "wall_seconds": elapsed,
    }
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / args.out
    out.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print("\nwrote %s (%.1f s)" % (out.relative_to(ROOT), elapsed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
