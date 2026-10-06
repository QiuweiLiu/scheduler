"""Round-6 residency comparison: frozen-F0 ordering for every arm; the belief
only feeds residency actions (prefetch demand + eviction subset).

Arms (all order by the frozen F0 ``sameshape_h5_p95`` key):
  reference  sameshape_h5_p95        no residency actions (frozen main-table F0)
  pdrs_p                             belief arm without actions (historical base)
  f0point_evict                      F0-point belief, eviction subset only
  pdrs_evict                         PDRS-distribution belief, eviction subset only
  f0point_resident                   F0-point belief, eviction + conservative prefetch
  pdrs_resident                      PDRS-distribution belief, eviction + prefetch
  pdrs_resident_shuffle              shuffled belief package (same actions)
  pdrs_resident_oracle               realized future identity (same actions)

Primary metric (pre-registered): mean completion time.  Secondary: p95
completion, deadline miss rate, makespan.  Headline statistic: the paired
difference-in-differences interaction

    d_int = (PDRS_resident - F0point_resident) - (pdrs_p - F0)

with a paired bootstrap CI; CI upper bound < 0 is the strong confirmation that
the action-space expansion (not better prediction) unlocked the belief value.
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

from tracing.analysis.pdrs_methods import build_shuffled_artifacts  # noqa: E402
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
ART = ROOT / "experiments/EXP-20261005_residency_comparison_v1/artifacts"

ARM_LABELS = (
    "pdrs_p",
    "f0point_evict",
    "pdrs_evict",
    "f0point_resident",
    "pdrs_resident",
    "pdrs_p95_evict",
    "pdrs_p95_resident",
    "pdrs_resident_shuffle",
    "pdrs_resident_oracle",
)

METRICS = ("mean_completion_ms", "p95_completion_ms", "deadline_miss_rate", "makespan_ms")
MECHANISM_FIELDS = (
    "gpu_evictions",
    "prefetch_count",
    "prefetch_load_ms",
    "wasted_prefetches",
    "used_prefetches",
    "cold_loads_on_demand",
    "evicted_then_reloaded",
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
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--out", default="residency_comparison_v1.json")
    parser.add_argument("--arms", default="",
                        help="comma-separated arm labels (default: all registered arms)")
    args = parser.parse_args()
    arm_labels = tuple(a for a in args.arms.split(",") if a) or ARM_LABELS

    if args.formal:
        for record in ("docs/research/2026-10-05_architecture_round6_residency_actions.md",):
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

    config = {
        "split": "confirm",
        "formal": bool(args.formal),
        "n_episodes": len(episodes),
        "primary_metric": "mean_completion_ms",
        "secondary_metrics": list(METRICS[1:]),
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
        policy = label
        values, info = run_arm(policy, label, episodes, templates, stats,
                               extension_config, future_artifacts)
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
                "episode_values": values[metric],
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
        print("   activation: %s" % json.dumps(info["activation"])[:200])
        print("   mechanism: %s" % json.dumps(info["mechanism"])[:200])

    # paired difference-in-differences: (PDRS_res - F0point_res) - (pdrs_p - F0)
    interactions: Dict[str, Any] = {}
    if "pdrs_resident" in results and "f0point_resident" in results and "pdrs_p" in results:
        for metric in METRICS:
            f0 = reference_values[metric]
            base = results["pdrs_p"]["metrics"][metric]["episode_values"]
            point_arm = results["f0point_resident"]["metrics"][metric]["episode_values"]
            pdrs_arm = results["pdrs_resident"]["metrics"][metric]["episode_values"]
            deltas = [(p - q) - (b - r) for p, q, b, r in zip(pdrs_arm, point_arm, base, f0)]
            lo, hi = paired_bootstrap_ci(deltas)
            interactions[metric] = {
                "delta_point": statistics.fmean(deltas),
                "delta_ci95": [lo, hi],
            }
    print("\n== interaction (PDRS_res - F0point_res) - (pdrs_p - F0) ==")
    for metric, entry in interactions.items():
        print("   %-20s %+9.4f  CI95 [%+.4f, %+.4f]"
              % (metric, entry["delta_point"], entry["delta_ci95"][0], entry["delta_ci95"][1]))

    elapsed = time.time() - started
    artifact = {
        "gate": "residency_comparison",
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
        "interaction": interactions,
        "criteria": {
            "delta": "candidate - %s (negative = better)" % REFERENCE,
            "headline": "interaction = (pdrs_resident - f0point_resident) - (pdrs_p - F0); "
                        "CI upper < 0 confirms the action-space expansion unlocked the belief",
            "mechanism": list(MECHANISM_FIELDS),
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
