"""Matched-consumer mechanism comparison (EXP-20261010_matched_consumer_v1).

Frozen design: docs/research/2026-10-10_exp_b_matched_consumer.md

Same frozen prediction pack, same GPU state, same legal action set (ordering +
minimal-SigmaV eviction + conservative prefetch) for every arm; a single
decision component is swapped to the relevant baseline's transferable rule:

  reference                 sameshape_h5_p95   no actions (frozen F0)
  pdrs_resident             main line (all own rules)
  hermes_order_mainline     ordering  <- Hermes Gittins rank on pack-derived
                            remaining-demand samples; own eviction+prefetch
  hermes_prefetch_mainline  prefetch  <- Hermes-style prewarm trigger on the
                            pack (running node's first future step, K=0.5);
                            F0 ordering + own eviction
  torpor_evict_mainline     eviction  <- Torpor swap_burden_order preference;
                            F0 ordering + own prefetch

These are mechanism controls, NOT paper reproductions; the faithful-adapted
main-table arms are untouched.  Headline: pdrs_resident minus each swapped arm
(mean + p95, paired episode bootstrap 2000 / seed 11); activation audit for
every swapped component; zero-failure hard abort.
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
ART = ROOT / "experiments/EXP-20261010_matched_consumer_v1/artifacts"
DESIGN_DOC = "docs/research/2026-10-10_exp_b_matched_consumer.md"

ARM_LABELS = (
    "hermes_order_mainline",
    "hermes_prefetch_mainline",
    "torpor_evict_mainline",
    "pdrs_resident",
)

METRICS = ("mean_completion_ms", "p95_completion_ms", "deadline_miss_rate", "makespan_ms")
MECHANISM_FIELDS = (
    "gpu_evictions",
    "prefetch_count",
    "prefetch_load_ms",
    "prefetch_capacity_failures",
    "wasted_prefetches",
    "used_prefetches",
    "cold_loads_on_demand",
    "evicted_then_reloaded",
    "preemptions",
    "preemption_resume_count",
    "preempt_recompute_ms",
    "preemption_stall_ms",
)

CONTRASTS = (
    ("pdrs_resident", "hermes_order_mainline"),
    ("pdrs_resident", "hermes_prefetch_mainline"),
    ("pdrs_resident", "torpor_evict_mainline"),
    ("pdrs_resident", REFERENCE),
    ("hermes_order_mainline", REFERENCE),
    ("hermes_prefetch_mainline", REFERENCE),
    ("torpor_evict_mainline", REFERENCE),
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
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--out", default="matched_consumer_v1.json")
    parser.add_argument("--arms", default="",
                        help="comma-separated arm labels (default: all registered arms)")
    args = parser.parse_args()
    arm_labels = tuple(a for a in args.arms.split(",") if a) or ARM_LABELS

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
        "design_doc": DESIGN_DOC,
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
        values, info = run_arm(label, label, episodes, templates, stats,
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
            "policy": label,
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
        print("   activation: %s" % json.dumps(info["activation"])[:220])
        print("   mechanism: %s" % json.dumps(info["mechanism"])[:220])

    def episode_values(label: str, metric: str) -> List[float]:
        if label == REFERENCE:
            return reference_values[metric]
        return results[label]["metrics"][metric]["episode_values"]

    contrasts: Dict[str, Any] = {}
    for candidate, baseline in CONTRASTS:
        key = f"{candidate}_minus_{baseline}"
        entry: Dict[str, Any] = {}
        for metric in METRICS:
            a = episode_values(candidate, metric)
            b = episode_values(baseline, metric)
            deltas = [x - y for x, y in zip(a, b)]
            lo, hi = paired_bootstrap_ci(deltas)
            entry[metric] = {"delta_point": statistics.fmean(deltas), "delta_ci95": [lo, hi]}
        contrasts[key] = entry
    print("\n== pre-registered contrasts (negative = candidate better) ==")
    for key, entry in contrasts.items():
        print("  %s" % key)
        for metric, values in entry.items():
            print("     %-20s %+12.4f  CI95 [%+.4f, %+.4f]"
                  % (metric, values["delta_point"], values["delta_ci95"][0], values["delta_ci95"][1]))

    elapsed = time.time() - started
    artifact = {
        "gate": "matched_consumer",
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
        "contrasts": contrasts,
        "criteria": {
            "delta": "candidate - baseline (negative = better)",
            "headline": "pdrs_resident minus each swapped-rule arm; decision-rule claim requires "
                        "the main line to beat EVERY swap arm on mean or p95 (CI_upper < 0) with "
                        "the swapped component demonstrably active",
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
