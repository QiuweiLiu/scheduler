"""Prefetch decision-module comparison (EXP-20261010_prefetch_value_methods_v1).

Frozen design: docs/research/2026-10-10_exp_g_prefetch_value_methods.md

Five+reference arms, everything except the prefetch module frozen:
  reference      sameshape_h5_p95 (F0, no actions)
  B0_no_prefetch pdrs_evict            (same eviction, no prefetch)
  B1_u_max       pdrs_resident         (frozen max-U prefetch)
  B2_distilled   pdrs_resident_da      (Method A: distilled linear value function)
  B3_rollout8    pdrs_resident_rollout (Method B: top-2 + none, 8 scenarios)
  B5_rollout_hb  pdrs_resident_hb      (high-budget rollout reference, ALL feasible
                                        (model, device) actions, frozen scenario count)

Metrics per episode: mean, p95, cold loads, evictions, prefetch count, used and
wasted prefetches, plus decision CPU (activation counters) and the net gain
vs B0 after charging the online decision time 1:1 (ms CPU = ms completion
budget; disclosed accounting convention).  Paired bootstrap 95% CIs; the net
gain gets a one-sample bootstrap CI over episodes.  Zero-failure hard abort.
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
ART = ROOT / "experiments/EXP-20261010_prefetch_value_methods_v1/artifacts"
DESIGN_DOC = "docs/research/2026-10-10_exp_g_prefetch_value_methods.md"
DISTILLED = ART / "distilled_benefit_v1.json"

ARM_SPECS = (
    ("reference", REFERENCE),
    ("B0_no_prefetch", "pdrs_evict"),
    ("B1_u_max", "pdrs_resident"),
    ("B2_distilled", "pdrs_resident_da"),
    ("B3_rollout8", "pdrs_resident_rollout"),
    ("B5_rollout_hb", "pdrs_resident_hb"),
    ("J8_joint", "pdrs_resident_joint"),
    ("JHB_joint", "pdrs_resident_joint_hb"),
)
CONTRASTS = (
    ("B1_u_max", "B0_no_prefetch"),
    ("B2_distilled", "B1_u_max"),
    ("B3_rollout8", "B1_u_max"),
    ("B5_rollout_hb", "B1_u_max"),
    ("B3_rollout8", "B2_distilled"),
    ("B5_rollout_hb", "B2_distilled"),
    ("B5_rollout_hb", "B3_rollout8"),
    ("B2_distilled", "reference"),
)

METRICS = ("mean_completion_ms", "p95_completion_ms")
MECHANISM_METRICS = (
    "gpu_evictions",
    "cold_loads_on_demand",
    "prefetch_count",
    "used_prefetches",
    "wasted_prefetches",
)
ALL_METRICS = METRICS + MECHANISM_METRICS


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
            distilled: Dict[str, Any] | None,
            scenario_override: int = 0,
            ) -> Tuple[Dict[str, List[float]], List[float], Dict[str, Any]]:
    values: Dict[str, List[float]] = {metric: [] for metric in ALL_METRICS}
    wall_us_series: List[float] = []
    activation: Counter[str] = Counter()
    for index, episode in enumerate(episodes):
        policy_context: Dict[str, Any] = {"activation": {}}
        if distilled is not None:
            policy_context["distilled_benefit"] = distilled
        if scenario_override:
            policy_context["rollout_scenarios"] = int(scenario_override)
        elif distilled is not None:
            policy_context["rollout_scenarios"] = int(distilled.get("n_scenarios", 32))
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
        raw_wall = float(policy_context["activation"].get("prefetch_plan_wall_us", 0.0))
        wall_us_series.append(raw_wall)
        activation.update(policy_context["activation"])
    return values, wall_us_series, dict(activation)


def paired_bootstrap_ci(deltas: Sequence[float], n_boot: int = BOOTSTRAP,
                        seed: int = SEED) -> Tuple[float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "confirm"), default="confirm")
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0,
                        help="evaluate only the first N episodes of the split")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--out", default="prefetch_ab_v1.json")
    parser.add_argument("--distilled", default=str(DISTILLED))
    parser.add_argument("--only", default="", help="comma-separated arm labels to run")
    parser.add_argument("--rollout-scenarios", type=int, default=0,
                        help="override the joint/high-budget rollout scenario count")
    args = parser.parse_args()

    if args.formal:
        if not (ROOT / DESIGN_DOC).exists():
            raise AssertionError("--formal requires %s" % DESIGN_DOC)
        head = git_head()
        if not head or head == "unknown" or len(head) != 40:
            raise AssertionError("--formal: git HEAD is %r" % (head,))
    assert sha256_file(PROJECTION) == FROZEN_PROJECTION_SHA, "projection hash mismatch"

    distilled = None
    distilled_sha = None
    distilled_path = Path(args.distilled)
    if distilled_path.exists():
        distilled = json.loads(distilled_path.read_text(encoding="utf-8"))
        distilled_sha = sha256_file(distilled_path)

    extension_config = json.loads(EXTENSION_CONFIG.read_text(encoding="utf-8"))
    templates = load_templates(PROJECTION, topology_view="causal_v3")
    split_manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    wanted = set(split_manifest["confirm"] if args.split == "confirm" else split_manifest["development"])
    by_id = {str(e.get("episode_id")): e for e in read_jsonl(EPISODES_FILE)}
    episodes = [by_id[eid] for eid in sorted(wanted) if eid in by_id]
    if args.smoke:
        episodes = episodes[:args.smoke]
    if args.limit:
        episodes = episodes[:args.limit]

    templates, episodes = apply_real_workload_profile_contract(
        templates, episodes, extension_config)
    stats = train_resource_stats({k: v for k, v in templates.items() if v.split == "train"})
    future_artifacts, preflight = load_resource_v2_overlay(BASE_ARTIFACTS, F0_ARTIFACTS)
    only = tuple(a for a in args.only.split(",") if a) or None

    config = {
        "split": args.split,
        "formal": bool(args.formal),
        "n_episodes": len(episodes),
        "reference": REFERENCE,
        "arms": [label for label, _policy in ARM_SPECS if not only or label in only],
        "seed": SEED,
        "bootstrap": BOOTSTRAP,
        "distilled_file": str(distilled_path.relative_to(ROOT)) if distilled else None,
        "distilled_sha256": distilled_sha,
        "distilled_n_scenarios": (distilled or {}).get("n_scenarios"),
        "projection_sha256": sha256_file(PROJECTION),
        "extension_config_sha256": sha256_file(EXTENSION_CONFIG),
        "reference_pack": preflight.get("artifact_id"),
        "reference_artifact_sha256": preflight.get("artifact_sha256"),
        "net_gain_accounting": "net = (B0 mean - arm mean) - planner CPU ms, 1 ms CPU = 1 ms episode budget (disclosed convention)",
        "smoke_episodes": args.smoke or None,
        "limit": args.limit or None,
        "git_head": git_head(),
    }
    print("== config ==")
    for key, value in config.items():
        print("   %-28s %s" % (key, value))

    started = time.time()
    results: Dict[str, Any] = {}
    wall_series: Dict[str, List[float]] = {}
    for label, policy in ARM_SPECS:
        if only and label not in only:
            continue
        values, wall_us, activation = run_arm(
            policy, label, episodes, templates, stats, extension_config,
            future_artifacts, distilled, args.rollout_scenarios)
        wall_series[label] = wall_us
        results[label] = {"policy": policy, "metrics": values, "activation": activation}
        print("\n== %s (%s) ==" % (label, policy))
        for metric in ALL_METRICS:
            print("   %-22s mean %12.2f" % (metric, statistics.fmean(values[metric])))
        calls = activation.get("prefetch_plan_calls", 0)
        wall = activation.get("prefetch_plan_wall_us", 0)
        moments = activation.get("prefetch_candidate_moments", 0)
        diffs = activation.get("prefetch_action_differs_from_b1", 0)
        print("   calls %d | wall_us %d | us/call %.1f | candidate_moments %d | actions %d"
              % (calls, wall, (wall / calls) if calls else 0.0, moments,
                 activation.get("prefetch_action_count", 0)))
        if moments:
            print("   action_differs_from_b1: %d/%d = %.3f" % (diffs, moments, diffs / moments))

    def series(label: str, metric: str) -> List[float]:
        return results[label]["metrics"][metric]

    contrasts: Dict[str, Any] = {}
    for candidate, baseline in CONTRASTS:
        if candidate not in results or baseline not in results:
            continue
        key = "%s_minus_%s" % (candidate, baseline)
        entry: Dict[str, Any] = {}
        for metric in ALL_METRICS:
            a = series(candidate, metric)
            b = series(baseline, metric)
            deltas = [x - y for x, y in zip(a, b)]
            lo, hi = paired_bootstrap_ci(deltas)
            entry[metric] = {"delta_point": statistics.fmean(deltas), "delta_ci95": [lo, hi]}
        contrasts[key] = entry
    print("\n== paired contrasts (negative = candidate better/fewer) ==")
    for key, entry in contrasts.items():
        print("  %s" % key)
        for metric in ALL_METRICS:
            values = entry[metric]
            print("     %-22s %+12.2f  CI95 [%+.2f, %+.2f]"
                  % (metric, values["delta_point"], values["delta_ci95"][0], values["delta_ci95"][1]))

    # net gain vs B0 (charges decision CPU 1:1); one-sample bootstrap over episodes
    net: Dict[str, Any] = {}
    if "B0_no_prefetch" in results:
        b0_mean = series("B0_no_prefetch", "mean_completion_ms")
        for label in results:
            if label == "B0_no_prefetch":
                continue
            ep_net = [
                (b0 - arm) - wall / 1000.0
                for b0, arm, wall in zip(b0_mean, series(label, "mean_completion_ms"),
                                         wall_series[label])
            ]
            lo, hi = paired_bootstrap_ci(ep_net)
            net[label] = {
                "delta_point": statistics.fmean(ep_net), "delta_ci95": [lo, hi],
                "planner_ms_per_episode": statistics.fmean(w / 1000.0 for w in wall_series[label]),
                "gain_ms_per_episode": statistics.fmean(
                    b0 - arm for b0, arm in zip(b0_mean, series(label, "mean_completion_ms"))),
            }
    print("\n== net gain vs B0 (completion gain minus planner CPU ms; positive = good) ==")
    for label, entry in net.items():
        print("   %-16s net %+10.2f CI95 [%+.2f, %+.2f] | gain %+9.2f | cpu %6.2f ms/ep"
              % (label, entry["delta_point"], entry["delta_ci95"][0], entry["delta_ci95"][1],
                 entry["gain_ms_per_episode"], entry["planner_ms_per_episode"]))

    elapsed = time.time() - started
    artifact = {
        "gate": "prefetch_value_methods",
        "version": "v1",
        "mode": ("smoke-" if args.smoke else "") + args.split,
        "config": config,
        "results": results,
        "contrasts": contrasts,
        "net_gain": net,
        "criteria": {
            "B2_vs_B1": "distilled value function vs max-U; significant improvement -> U underuses future info",
            "B5_vs_B1": "high-budget rollout reference vs max-U: how much extra gain is available",
            "B2_B3_vs_B5": "whether low-compute methods retain the high-budget gains",
            "high_budget_scope": "approximate optimum under limited candidates + sampled scenarios only; NOT global optimum",
            "non_significance_rule": "CI containing 0 supports neither difference nor equivalence",
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
