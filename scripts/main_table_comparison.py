"""The formal main-table comparison run (frozen design, 2026-10-04).

Everything about this run is fixed before it starts:

  workload     results/processed/r7_workload_v041_ontology_no_run_container (the frozen
               projection pinned by SchedulerTopologyContractGate v1)
  episodes     the frozen confirm300 validation set (development is tuning-only)
  substrate    the v7 measured extension config (real-strata co-location + additive
               load interference + request preemption + batching) with the
               evidence-based profile contract applied (shape = node role evidence,
               episode identity = uniform node gpu_model evidence)
  metric       mean_completion_ms, the frozen primary metric
  reference    F0 sameshape_h5_p95, the selected deployment predictor
  arms         parrot_appfifo, qlm_queue, llmsched, hermes_gittins, torpor_lifecycle
               (the frozen main-table adaptations; each brings its own front end)
  criteria     Delta = candidate - F0;
                 non-inferior        CI_upper(Delta) < +485 ms
                 statistically better CI_upper < 0
                 substantively better point(Delta) <= -485 ms AND CI_upper(Delta) < 0
  pairing      the SAME episodes and the SAME seed per arm, so every Delta is within-episode
  uncertainty  EPISODE-cluster bootstrap over paired episode deltas

Usage:
    --smoke N     small-scale pre-formal check (default off; use 10-20)
    --episodes N  cap the episode count (formal runs use the whole frozen set)
    --formal      hard-assert the freeze/provenance gates before starting
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
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
ART = ROOT / "experiments/EXP-20261004_main_table_comparison_v1/artifacts"
BASE_ARTIFACTS = ROOT / "outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts"
F0_ARTIFACTS = ROOT / "outputs/resource_v2_artifacts/f0_seed11"
FUTURE_HORIZON = 5

REFERENCE = "sameshape_h5_p95"
MAIN_ARMS = ("parrot_appfifo", "qlm_queue", "llmsched", "hermes_gittins", "torpor_lifecycle")
METRIC = "mean_completion_ms"
NI_MARGIN_MS = 485.0
SEED = 11
BOOTSTRAP = 2000

# Freeze gates: the adaptations and the substrate contract were independently reviewed
# and ACCEPTED before this run; the run may not start without them.
REVIEW_RECORDS = (
    "docs/research/2026-10-03_main_table_baselines_manifest.md",
    "docs/research/2026-10-04_main_table_baselines_review_round2.md",
    "docs/research/2026-10-04_real_strata_contract_review.md",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    """The commit this run is attributable to.

    SCHEDULER_GIT_DIR lets the local control-plane workflow point at its own
    git-dir; the historical checkout path is kept as a fallback.
    """

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


def build_context(arm: str, templates: Dict[str, Any]) -> Dict[str, Any]:
    """Each arm's own frozen front end.  No arm may borrow another's artifact."""

    if arm == "parrot_appfifo":
        return {}
    if arm == "qlm_queue":
        from tracing.analysis.qlm_methods import build_duration_bank
        return {"qlm_duration_bank": build_duration_bank(templates),
                "qlm_rng": random.Random(SEED)}
    if arm == "llmsched":
        from tracing.analysis.llmsched_bn import build_bn_profiler
        return {"llmsched_bn": build_bn_profiler(templates),
                "llmsched_rng": random.Random(SEED),
                "llmsched_epsilon": 0.1}
    if arm == "hermes_gittins":
        from tracing.analysis.hermes_methods import build_pdgraph
        return {"hermes_pdgraph": build_pdgraph(templates)}
    if arm == "torpor_lifecycle":
        return {}
    raise ValueError("unknown arm %r" % arm)


def run_arm(arm: str, context: Dict[str, Any], episodes: Sequence[Dict[str, Any]],
            templates: Dict[str, Any], stats: Dict[str, Any],
            extension_config: Dict[str, Any],
            future_artifacts: Dict[str, Any] | None = None,
            future_horizon: int = 0) -> Tuple[List[float], Dict[str, int]]:
    """Run one arm, failing closed on any episode that did not complete cleanly."""

    values: List[float] = []
    activation_totals: Counter[str] = Counter()
    for index, episode in enumerate(episodes):
        policy_context = dict(context)
        policy_context["activation"] = {}
        summary, _ = simulate_episode(episode, templates, arm, train_stats=stats,
                                      policy_context=policy_context,
                                      future_artifacts=future_artifacts,
                                      future_horizon=future_horizon,
                                      extension_config=extension_config)
        expected_jobs = len(episode.get("jobs") or [])
        completed = int(summary.get("completed_jobs") or 0)
        failed = int(summary.get("failed_jobs") or 0)
        if failed != 0 or completed != expected_jobs:
            raise AssertionError(
                "%s episode %s: completed %d/%d with %d failed; a partial run cannot "
                "produce the metric" % (arm, episode.get("episode_id", index),
                                        completed, expected_jobs, failed))
        value = float(summary[METRIC])
        if not math.isfinite(value):
            raise AssertionError("%s episode %s: %s is not finite"
                                 % (arm, episode.get("episode_id", index), METRIC))
        values.append(value)
        activation_totals.update(policy_context["activation"])
    return values, dict(sorted(activation_totals.items()))


def paired_bootstrap_ci(deltas: Sequence[float], *, n_boot: int = BOOTSTRAP,
                        seed: int = SEED) -> Tuple[float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    return lo, hi


def classify(point: float, ci_upper: float) -> str:
    if ci_upper >= NI_MARGIN_MS:
        return "inferior"
    if ci_upper < 0.0 and point <= -NI_MARGIN_MS:
        return "substantively_better"
    if ci_upper < 0.0:
        return "statistically_better"
    return "non_inferior"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("confirm", "development"), default="confirm")
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--out", default="main_table_formal_v1.json")
    args = parser.parse_args()

    if args.formal:
        for record in REVIEW_RECORDS:
            if not (ROOT / record).exists():
                raise AssertionError("--formal requires the reviewed record at %s" % record)
        if not EXTENSION_CONFIG.exists():
            raise AssertionError("--formal requires the v7 extension config at %s" % EXTENSION_CONFIG)
        head = git_head()
        if not head or head == "unknown" or len(head) != 40:
            raise AssertionError(
                "--formal: git HEAD is %r; a formal run must be attributable to a commit"
                % (head,))

    assert sha256_file(PROJECTION) == FROZEN_PROJECTION_SHA, (
        "the projection hash does not match the frozen SchedulerTopologyContractGate value")

    extension_config = json.loads(EXTENSION_CONFIG.read_text(encoding="utf-8"))
    templates = load_templates(PROJECTION, topology_view="causal_v3")

    split_manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    if len(split_manifest["development"]) != 700 or len(split_manifest["confirm"]) != 300:
        raise AssertionError(
            "the split manifest is not the frozen 700/300: development=%d confirm=%d"
            % (len(split_manifest["development"]), len(split_manifest["confirm"])))
    wanted = set(split_manifest[args.split])
    all_episodes = list(read_jsonl(EPISODES_FILE))
    by_id = {str(e.get("episode_id")): e for e in all_episodes}
    missing = wanted - set(by_id)
    if missing:
        raise AssertionError("%d %s episodes are absent from the file, e.g. %s"
                             % (len(missing), args.split, sorted(missing)[:3]))
    episodes = [by_id[eid] for eid in sorted(wanted) if eid in wanted]
    if args.episodes:
        episodes = episodes[:args.episodes]
    if args.smoke:
        episodes = episodes[:args.smoke]

    templates, episodes = apply_real_workload_profile_contract(
        templates, episodes, extension_config)
    train = {k: v for k, v in templates.items() if v.split == "train"}
    stats = train_resource_stats(train)

    config = {
        "split": args.split,
        "formal": bool(args.formal),
        "split_manifest": str(SPLIT_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "split_manifest_sha256": sha256_file(SPLIT_MANIFEST),
        "episodes_sha256": sha256_file(EPISODES_FILE),
        "projection": str(PROJECTION.relative_to(ROOT)).replace("\\", "/"),
        "projection_sha256": sha256_file(PROJECTION),
        "episodes_file": str(EPISODES_FILE.relative_to(ROOT)).replace("\\", "/"),
        "n_episodes": len(episodes),
        "metric": METRIC,
        "reference": REFERENCE,
        "arms": list(MAIN_ARMS),
        "seed": SEED,
        "non_inferiority_margin_ms": NI_MARGIN_MS,
        "bootstrap": BOOTSTRAP,
        "topology_view": "causal_v3",
        "extension_config": str(EXTENSION_CONFIG.relative_to(ROOT)).replace("\\", "/"),
        "extension_config_sha256": sha256_file(EXTENSION_CONFIG),
        "profile_contract": "shape = node role evidence; episode identity = uniform node gpu_model evidence",
        "reference_base_artifacts": str(BASE_ARTIFACTS.relative_to(ROOT)).replace(chr(92), "/"),
        "reference_overlay_artifacts": str(F0_ARTIFACTS.relative_to(ROOT)).replace(chr(92), "/"),
        "future_horizon": FUTURE_HORIZON,
        "smoke_episodes": args.smoke or None,
        "git_head": git_head(),
    }

    print("== config ==")
    for key, value in config.items():
        print("   %-28s %s" % (key, value))

    started = time.time()
    reference_artifacts, reference_preflight = load_resource_v2_overlay(
        BASE_ARTIFACTS, F0_ARTIFACTS)
    print("reference overlay preflight:", json.dumps(reference_preflight, sort_keys=True)[:220])
    reference_values, reference_activation = run_arm(
        REFERENCE, {}, episodes, templates, stats, extension_config,
        future_artifacts=reference_artifacts, future_horizon=FUTURE_HORIZON)
    print("\n== %s (reference) ==" % REFERENCE)
    print("   mean %s = %.1f" % (METRIC, statistics.fmean(reference_values)))

    results: Dict[str, Any] = {}
    for arm in MAIN_ARMS:
        context = build_context(arm, templates)
        values, activation = run_arm(arm, context, episodes, templates, stats, extension_config)
        deltas = [a - b for a, b in zip(values, reference_values)]
        point = statistics.fmean(deltas)
        lo, hi = paired_bootstrap_ci(deltas)
        verdict = classify(point, hi)
        results[arm] = {
            "mean": statistics.fmean(values),
            "delta_point": point,
            "delta_ci95": [lo, hi],
            "delta_median": statistics.median(deltas),
            "n_worse": sum(1 for d in deltas if d > 0),
            "n_episodes": len(deltas),
            "verdict": verdict,
            "activation": activation,
            "episode_values": values,
        }
        print("\n== %s ==" % arm)
        print("   mean %s = %.1f" % (METRIC, statistics.fmean(values)))
        print("   Delta vs %s: point %+.1f ms  CI95 [%+.1f, %+.1f]  %d/%d worse  -> %s"
              % (REFERENCE, point, lo, hi, results[arm]["n_worse"], len(deltas), verdict))
        print("   activation: %s" % json.dumps(activation, sort_keys=True)[:200])

    elapsed = time.time() - started
    artifact = {
        "gate": "main_table_comparison",
        "version": "v1",
        "mode": "smoke" if args.smoke else "formal",
        "config": config,
        "reference_summary": {
            "mean": statistics.fmean(reference_values),
            "n_episodes": len(reference_values),
            "activation": reference_activation,
            "episode_values": reference_values,
        },
        "results": results,
        "criteria": {
            "delta": "candidate - %s" % REFERENCE,
            "non_inferior": "CI_upper < +%g ms" % NI_MARGIN_MS,
            "statistically_better": "CI_upper < 0",
            "substantively_better": "point <= -%g ms AND CI_upper < 0" % NI_MARGIN_MS,
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
