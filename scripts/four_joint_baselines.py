"""The formal paired four-joint-baseline experiment.

Everything about this run is fixed before it starts, because a comparison assembled after
seeing numbers is not a comparison:

  workload     results/processed/r7_workload_v041_ontology_no_run_container (the frozen
               projection pinned by SchedulerTopologyContractGate v1)
  episodes     the frozen 300-episode validation set
  metric       mean_completion_ms, the frozen primary metric
  reference    F0 sameshape_h5_p95, the selected deployment predictor
  criteria     Delta = candidate - F0;
                   non-inferior        CI_upper(Delta) < +485 ms
                   statistically better CI_upper(Delta) < 0
                   substantively better point(Delta) <= -485 ms AND CI_upper(Delta) < 0
  pairing      the SAME episodes and the SAME seed per arm, so every Delta is within-episode
  uncertainty  video-cluster bootstrap over paired episode deltas

Each arm brings its own frozen front end; none of them may read another arm's artifact.
A baseline does NOT have to be close to F0 to count as a successful reproduction -- the
fidelity gate is what qualifies it, and this run is what measures it.

Usage:
    --smoke N     small-scale pre-formal check (default off; use N = 10-20)
    --episodes N  cap the episode count (formal runs use the whole frozen set)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

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
ART = (ROOT / "experiments/EXP-20260921_scheduler_replication_v1/artifacts")
# F0 is an OVERLAY on the frozen J3 base pack, not a standalone directory: it carries the
# repaired layer-H5 file and is spliced onto the base, whose H1/H3/layers files it does not
# duplicate.  Loading the F0 directory alone would fail on the missing legacy files.
BASE_ARTIFACTS = ROOT / "outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts"
F0_ARTIFACTS = ROOT / "outputs/resource_v2_artifacts/f0_seed11"
FUTURE_HORIZON = 5

REFERENCE = "sameshape_h5_p95"
ARMS = ("tie_current", "pythia_completion", "llmsched", "latency_aware")
METRIC = "mean_completion_ms"
NI_MARGIN_MS = 485.0
SEED = 11
BOOTSTRAP = 2000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return "unknown"


def build_context(arm: str, templates: Dict[str, Any]) -> Dict[str, Any]:
    """Each arm's own frozen front end.  No arm may borrow another's artifact."""

    if arm == "tie_current":
        from tracing.analysis.tie_methods import build_tie_bank
        return {"tie_bank": build_tie_bank(templates)}
    if arm == "pythia_completion":
        from tracing.analysis.pythia_profiler import build_pythia_profiler
        return {"pythia_profiler": build_pythia_profiler(templates)}
    if arm == "llmsched":
        from tracing.analysis.llmsched_bn import build_bn_profiler
        return {"llmsched_bn": build_bn_profiler(templates),
                "llmsched_rng": random.Random(SEED),
                "llmsched_epsilon": 0.1}
    if arm == "latency_aware":
        from tracing.analysis.latency_aware_predictor import build_latency_predictor
        return {"latency_aware_predictor": build_latency_predictor(templates)}
    raise ValueError("unknown arm %r" % arm)


def run_arm(arm: str, context: Dict[str, Any], episodes: Sequence[Dict[str, Any]],
            templates: Dict[str, Any], stats: Dict[str, Any],
            future_artifacts: Dict[str, Any] | None = None,
            future_horizon: int = 0) -> List[float]:
    values: List[float] = []
    for episode in episodes:
        summary, _ = simulate_episode(episode, templates, arm, train_stats=stats,
                                      policy_context=dict(context),
                                      future_artifacts=future_artifacts,
                                      future_horizon=future_horizon)
        values.append(float(summary[METRIC]))
    return values


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
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--out", default="four_baseline_formal_v1.json")
    args = parser.parse_args()

    templates = load_templates(PROJECTION, topology_view="causal_v3")
    train = {k: v for k, v in templates.items() if v.split == "train"}
    stats = train_resource_stats(train)

    episodes = list(read_jsonl(EPISODES_FILE))
    if args.smoke:
        episodes = episodes[:args.smoke]
    if args.episodes:
        episodes = episodes[:args.episodes]

    config = {
        "projection": str(PROJECTION.relative_to(ROOT)).replace("\\", "/"),
        "projection_sha256": sha256_file(PROJECTION),
        "episodes_file": str(EPISODES_FILE.relative_to(ROOT)).replace("\\", "/"),
        "n_episodes": len(episodes),
        "metric": METRIC,
        "reference": REFERENCE,
        "arms": list(ARMS),
        "seed": SEED,
        "non_inferiority_margin_ms": NI_MARGIN_MS,
        "bootstrap": BOOTSTRAP,
        "topology_view": "causal_v3",
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
    # The reference is F0 sameshape_h5_p95, which is a bounded-future policy: it needs
    # the frozen F0 pack and the frozen horizon.  It is not a context-free arm.
    reference_artifacts, reference_preflight = load_resource_v2_overlay(
        BASE_ARTIFACTS, F0_ARTIFACTS)
    print("reference overlay preflight:", json.dumps(reference_preflight, sort_keys=True)[:220])
    reference_values = run_arm(REFERENCE, {}, episodes, templates, stats,
                               future_artifacts=reference_artifacts,
                               future_horizon=FUTURE_HORIZON)
    print("\n== %s (reference) ==" % REFERENCE)
    print("   mean %s = %.1f" % (METRIC, statistics.fmean(reference_values)))

    results: Dict[str, Any] = {}
    for arm in ARMS:
        context = build_context(arm, templates)
        values = run_arm(arm, context, episodes, templates, stats)
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
        }
        print("\n== %s ==" % arm)
        print("   mean %s = %.1f" % (METRIC, statistics.fmean(values)))
        print("   Delta vs %s: point %+.1f ms  CI95 [%+.1f, %+.1f]  %d/%d worse  -> %s"
              % (REFERENCE, point, lo, hi, results[arm]["n_worse"], len(deltas), verdict))

    elapsed = time.time() - started
    artifact = {
        "gate": "four_joint_baselines",
        "version": "v1",
        "mode": "smoke" if args.smoke else "formal",
        "config": config,
        "reference_summary": {
            "mean": statistics.fmean(reference_values),
            "n_episodes": len(reference_values),
        },
        "results": results,
        "criteria": {
            "delta": "candidate - %s" % REFERENCE,
            "non_inferior": "CI_upper < +%g ms" % NI_MARGIN_MS,
            "statistically_better": "CI_upper < 0",
            "substantively_better": "point <= -%g ms AND CI_upper < 0" % NI_MARGIN_MS,
            "note": "a baseline does NOT need to be close to the reference to count as a "
                    "successful reproduction; the fidelity gate qualifies it and this run "
                    "measures it",
        },
        "wall_seconds": elapsed,
        "artifacts_dir": str(ART.relative_to(ROOT)).replace("\\", "/"),
    }

    out = ART / args.out
    out.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print("\nwrote %s (%.1f s)" % (out.relative_to(ROOT), elapsed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
