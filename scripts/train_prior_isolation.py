"""Train-set prior isolation re-run (EXP-20261010_train_prior_isolation_v1).

Frozen design: docs/research/2026-10-10_exp_f_train_prior_isolation.md

Arms (identical scheduling rules / GPU env / eviction / prefetch; ONLY the
prediction source differs):

  reference  sameshape_h5_p95, frozen instance pack, no actions
  ours       pdrs_resident (main line), frozen instance pack
  partial    pdrs_resident, pack with model + length channels replaced by the
             PREDICTOR-TRAIN-SET prior (j_train only)
  full       pdrs_resident, pack with model + length + runtime + load + lane
             replaced by the same train-set prior

Per-episode metrics: mean / p95 completion, evictions, cold loads, prefetch
count, used prefetches.  Paired episode bootstrap 95% CIs for every contrast;
effective-prefetch ratio reported per arm (totals) and paired on the
both-prefetching subset.  Isolation audit from frozen files is embedded in the
artifact.  Old experiment files are never touched.  Zero-failure hard abort.
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

from train_prior_lib import (  # noqa: E402
    J_TRAIN,
    build_prior_pack,
    compute_train_prior,
    isolation_report,
    load_j_split_rows,
    prior_sha256,
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
MAINLINE = "pdrs_resident"
SEED = 11
BOOTSTRAP = 2000
ART = ROOT / "experiments/EXP-20261010_train_prior_isolation_v1/artifacts"
DESIGN_DOC = "docs/research/2026-10-10_exp_f_train_prior_isolation.md"

METRICS = ("mean_completion_ms", "p95_completion_ms")
MECHANISM_METRICS = (
    "gpu_evictions",
    "cold_loads_on_demand",
    "prefetch_count",
    "used_prefetches",
)
ALL_METRICS = METRICS + MECHANISM_METRICS

# arm label -> (policy, pack mode or None for the frozen instance pack)
ARM_SPECS = (
    ("reference", REFERENCE, None),
    ("ours", MAINLINE, None),
    ("partial", MAINLINE, "partial"),
    ("full", MAINLINE, "full"),
)
CONTRAST_PAIRS = (("ours", "partial"), ("ours", "full"), ("partial", "reference"),
                  ("full", "reference"), ("ours", "reference"))


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
    values: Dict[str, List[float]] = {metric: [] for metric in ALL_METRICS}
    activation: Counter[str] = Counter()
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
        for metric in ALL_METRICS:
            value = float(summary.get(metric) or 0.0)
            if metric in METRICS and not math.isfinite(value):
                raise AssertionError(f"{label} episode {index}: non-finite {metric}")
            values[metric].append(value)
        activation.update(policy_context["activation"])
    return values, {"activation": dict(sorted(activation.items()))}


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
    parser.add_argument("--out", default="train_prior_v1.json")
    args = parser.parse_args()

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
    if args.smoke:
        episodes = episodes[:args.smoke]

    templates, episodes = apply_real_workload_profile_contract(
        templates, episodes, extension_config)
    stats = train_resource_stats({k: v for k, v in templates.items() if v.split == "train"})
    pack, preflight = load_resource_v2_overlay(BASE_ARTIFACTS, F0_ARTIFACTS)

    isolation = isolation_report()
    j_rows = load_j_split_rows(J_TRAIN)
    prior = compute_train_prior(j_rows)
    partial_pack = build_prior_pack(pack, prior, "partial")
    full_pack = build_prior_pack(pack, prior, "full")
    packs = {None: pack, "partial": partial_pack, "full": full_pack}

    config = {
        "split": "confirm",
        "formal": bool(args.formal),
        "n_episodes": len(episodes),
        "reference": REFERENCE,
        "mainline": MAINLINE,
        "arms": [label for label, _policy, _mode in ARM_SPECS],
        "seed": SEED,
        "bootstrap": BOOTSTRAP,
        "projection_sha256": sha256_file(PROJECTION),
        "extension_config_sha256": sha256_file(EXTENSION_CONFIG),
        "reference_pack": preflight.get("artifact_id"),
        "reference_artifact_sha256": preflight.get("artifact_sha256"),
        "j_train_file": str(J_TRAIN),
        "j_train_sha256": isolation["j_series"]["train"]["sha256"],
        "prior_sha256": prior_sha256(prior),
        "prior_version": prior["version"],
        "future_horizon": FUTURE_HORIZON,
        "smoke_episodes": args.smoke or None,
        "git_head": git_head(),
    }
    print("== config ==")
    for key, value in config.items():
        print("   %-28s %s" % (key, value))
    print("\n== isolation audit ==")
    print(json.dumps(isolation, ensure_ascii=False, indent=1)[:2400])

    started = time.time()
    results: Dict[str, Any] = {}
    for label, policy, mode in ARM_SPECS:
        values, info = run_arm(policy, label, episodes, templates, stats,
                               extension_config, packs[mode])
        results[label] = {"policy": policy, "pack_mode": mode or "instance",
                          "metrics": values, **info}
        print("\n== %s (%s, pack=%s) ==" % (label, policy, mode or "instance"))
        for metric in ALL_METRICS:
            series = values[metric]
            print("   %-22s mean %12.2f" % (metric, statistics.fmean(series)))

    def ratios(label: str) -> Dict[str, Any]:
        used = results[label]["metrics"]["used_prefetches"]
        total = results[label]["metrics"]["prefetch_count"]
        total_used = sum(used)
        total_prefetch = sum(total)
        return {
            "total_prefetch": total_prefetch,
            "total_used": total_used,
            "overall_ratio": (total_used / total_prefetch) if total_prefetch else None,
            "episodes_with_prefetch": sum(1 for value in total if value > 0),
        }

    contrasts: Dict[str, Any] = {}
    for candidate, baseline in CONTRAST_PAIRS:
        key = "%s_minus_%s" % (candidate, baseline)
        entry: Dict[str, Any] = {}
        for metric in ALL_METRICS:
            a = results[candidate]["metrics"][metric]
            b = results[baseline]["metrics"][metric]
            deltas = [x - y for x, y in zip(a, b)]
            lo, hi = paired_bootstrap_ci(deltas)
            entry[metric] = {"delta_point": statistics.fmean(deltas), "delta_ci95": [lo, hi]}
        contrasts[key] = entry
    print("\n== paired contrasts (negative = candidate better / fewer) ==")
    for key, entry in contrasts.items():
        print("  %s" % key)
        for metric in ALL_METRICS:
            values = entry[metric]
            print("     %-22s %+12.2f  CI95 [%+.2f, %+.2f]"
                  % (metric, values["delta_point"], values["delta_ci95"][0], values["delta_ci95"][1]))

    # effective-prefetch ratio: totals + paired subset (both arms prefetching)
    ratio_info = {label: ratios(label) for label, _policy, _mode in ARM_SPECS}
    ratio_pairs: Dict[str, Any] = {}
    for candidate, baseline in CONTRAST_PAIRS:
        a_used = results[candidate]["metrics"]["used_prefetches"]
        a_total = results[candidate]["metrics"]["prefetch_count"]
        b_used = results[baseline]["metrics"]["used_prefetches"]
        b_total = results[baseline]["metrics"]["prefetch_count"]
        deltas = []
        for used_a, total_a, used_b, total_b in zip(a_used, a_total, b_used, b_total):
            if total_a > 0 and total_b > 0:
                deltas.append((used_a / total_a) - (used_b / total_b))
        if deltas:
            lo, hi = paired_bootstrap_ci(deltas)
            ratio_pairs["%s_minus_%s" % (candidate, baseline)] = {
                "delta_point": statistics.fmean(deltas),
                "delta_ci95": [lo, hi],
                "n_paired_episodes": len(deltas),
            }
    print("\n== effective prefetch ratio ==")
    for label, info in ratio_info.items():
        print("   %-10s prefetch %6d | used %6d | ratio %s | episodes %d"
              % (label, info["total_prefetch"], info["total_used"],
                 ("%.3f" % info["overall_ratio"]) if info["overall_ratio"] is not None else "n/a",
                 info["episodes_with_prefetch"]))
    for key, info in ratio_pairs.items():
        print("   %s  %+.4f CI95 [%+.4f, %+.4f] (n=%d)"
              % (key, info["delta_point"], info["delta_ci95"][0], info["delta_ci95"][1],
                 info["n_paired_episodes"]))

    elapsed = time.time() - started
    artifact = {
        "gate": "train_prior_isolation",
        "version": "v1",
        "mode": "smoke" if args.smoke else "formal",
        "config": config,
        "isolation_audit": isolation,
        "prior": {k: v for k, v in prior.items()},
        "results": results,
        "contrasts": contrasts,
        "effective_prefetch_ratio": {"arms": ratio_info, "paired": ratio_pairs},
        "criteria": {
            "delta": "candidate - baseline (negative = candidate better; for counts: fewer)",
            "headline": "ours - partial and ours - full on mean/p95; mechanism gain under priors "
                        "= partial/full - reference",
            "non_significance_rule": "a CI containing 0 supports neither a difference nor an "
                                     "equivalence claim (no equivalence testing performed)",
            "metrics": list(ALL_METRICS),
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
