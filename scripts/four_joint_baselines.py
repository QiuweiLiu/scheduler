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
  uncertainty  EPISODE-cluster bootstrap over paired episode deltas (see paired_bootstrap_ci:
               it resamples EPISODES, not videos/templates, so it describes uncertainty over
               the frozen 300-episode set and does NOT by itself generalise to unseen videos)

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
import math
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
# The control plane froze the evaluation split at seed 20260914.  Selecting episodes by
# FILE ORDER is not that split: --episodes 300 of the raw file is 212 development and 88
# confirm, because the ids were shuffled.  The final number must come from confirm300.
SPLIT_MANIFEST = ROOT / "data/manifests/validation_split_dev700_confirm300.json"
FROZEN_PROJECTION_SHA = (
    "15c62dafd99701b3883a3fe47034b5c7771f8fdcc8d6226fa73ff06fdca7f03b")
ART = (ROOT / "experiments/EXP-20260921_scheduler_replication_v1/artifacts")
# F0 is an OVERLAY on the frozen J3 base pack, not a standalone directory: it carries the
# repaired layer-H5 file and is spliced onto the base, whose H1/H3/layers files it does not
# duplicate.  Loading the F0 directory alone would fail on the missing legacy files.
BASE_ARTIFACTS = ROOT / "outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts"
F0_ARTIFACTS = ROOT / "outputs/resource_v2_artifacts/f0_seed11"
FUTURE_HORIZON = 5

REFERENCE = "sameshape_h5_p95"
ARMS = ("tie_current", "pythia_completion", "llmsched", "latency_aware", "agentix")
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
    """The commit this run is attributable to.

    The public mirror is the git repository, not the working tree, so HEAD is read from
    there; reading ROOT returned an empty string because the working tree is not a repo.
    """

    for candidate in (ROOT / ".." / "scheduler_public_repo", ROOT):
        try:
            out = subprocess.run(["git", "-C", str(candidate), "rev-parse", "HEAD"],
                                 capture_output=True, text=True, timeout=30)
            value = out.stdout.strip()
            if value:
                return value
        except Exception:
            continue
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
    if arm == "agentix":
        # Formal arm is continuous PLAS, non-preemptive: no artifact; it reads only the
        # program's COMPLETED GPU calls' service from the live job state.  The
        # non-preemptive discrete sensitivity (agentix_mode='discrete') is NOT this arm.
        return {}
    raise ValueError("unknown arm %r" % arm)


def run_arm(arm: str, context: Dict[str, Any], episodes: Sequence[Dict[str, Any]],
            templates: Dict[str, Any], stats: Dict[str, Any],
            future_artifacts: Dict[str, Any] | None = None,
            future_horizon: int = 0) -> List[float]:
    """Run one arm, failing closed on any episode that did not complete cleanly.

    A failed job leaves the surviving jobs looking fast, so mean_completion_ms can be a
    plausible number for an unusable run.  The audit asked for these assertions and they
    are the difference between a measurement and a survivor mean.
    """

    values: List[float] = []
    for index, episode in enumerate(episodes):
        summary, _ = simulate_episode(episode, templates, arm, train_stats=stats,
                                      policy_context=dict(context),
                                      future_artifacts=future_artifacts,
                                      future_horizon=future_horizon)
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
    parser.add_argument("--split", choices=("confirm", "development"), default="confirm",
                        help="confirm is the frozen one-shot evaluation set; development "
                             "is for tuning only")
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--smoke", type=int, default=0,
                        help="small-scale pre-formal check on the first N confirm episodes "
                             "(0 = off; use 10-20)")
    parser.add_argument("--formal", action="store_true",
                        help="hard-assert the freeze/provenance gates before starting; the "
                             "formal measurement must never run on an unpinned tree")
    parser.add_argument("--out", default="four_baseline_formal_v1.json")
    args = parser.parse_args()

    if args.formal:
        # The formal measurement is only comparable if every frozen input is pinned.  The
        # audit asked for these to be hard assertions rather than git_head "unknown" plus a
        # recorded hash that nobody checks.
        fidelity_path = ART / "baseline_fidelity_manifest_v1.json"
        topology_path = ART / "scheduler_topology_contract_gate_v1.json"
        for label, path in (("BaselineFidelityManifest", fidelity_path),
                            ("SchedulerTopologyContractGate", topology_path)):
            if not path.exists():
                raise AssertionError("--formal requires %s at %s" % (label, path))
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("pass") is not True:
                raise AssertionError("--formal: %s.pass is not True" % label)
        head = git_head()
        if not head or head == "unknown" or len(head) != 40:
            raise AssertionError(
                "--formal: git HEAD is %r; a formal run must be attributable to a commit"
                % (head,))

    templates = load_templates(PROJECTION, topology_view="causal_v3")
    train = {k: v for k, v in templates.items() if v.split == "train"}
    stats = train_resource_stats(train)

    # Select by the FROZEN split, never by file order.
    split_manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    # The split itself is frozen: 700 development / 300 confirm at seed 20260914.  A
    # silently resized split would change what "the frozen confirm300" means.
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

    assert sha256_file(PROJECTION) == FROZEN_PROJECTION_SHA, (
        "the projection hash does not match the frozen SchedulerTopologyContractGate "
        "value; this run would not be comparable to anything")
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
