#!/usr/bin/env python
"""Six-arm paired scheduler smoke on the resource-v2 predictor artifacts.

Arms (frozen in .project/EXPERIMENT_GATE.json -> smoke_preregistration_20260920)
------------------------------------------------------------------------------
A0  J3 three-quantile pack        + legacy sum-q95 consumption   (the champion)
A1  R1b discrete head             + legacy sum-q95 consumption   (predictor effect)
A2  R3a-U adapted res_hidden      + legacy sum-q95 consumption   (diagnostic only)
B1  R1b                           + sum of conditional means     (consumption effect)
B2  R1b                           + sum of marginal step CVaR95  (consumption effect)
O   joint future truth            + same key shape               (headroom ceiling)

A2 is barred from model selection: it fails the frozen Phase-R calibration
integrity gate (cov90 0.803), so it may not become the winner even if it scores
best on the scheduler.

Stages
------
``select``    build and freeze the 300-episode list (+ SHA256)
``preflight`` validate every artifact the run will consume
``run``       simulate every selected episode under every arm
``report``    paired bootstrap contrasts against the pre-registered margin

Usage::

    python scripts/resource_v2_scheduler_smoke.py --stage select
    python scripts/resource_v2_scheduler_smoke.py --stage preflight
    python scripts/resource_v2_scheduler_smoke.py --stage run
    python scripts/resource_v2_scheduler_smoke.py --stage report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from tracing.analysis.workload_v02_simulator import (  # noqa: E402
    load_future_artifacts,
    load_resource_v2_overlay,
    read_gzip_jsonl,
    read_jsonl,
    simulate_episode,
    train_resource_stats,
    load_templates,
)

# --------------------------------------------------------------------------- #
DEFAULT_EPISODES = PROJECT_ROOT / "results/processed/r7_workload_v03_no_run_container/episodes/workload_validation_r7_v03.jsonl"
DEFAULT_TEMPLATES = PROJECT_ROOT / "results/processed/r7_workload_v03_no_run_container/job_templates_r7_v03.jsonl"
DEFAULT_BASE_PACK = PROJECT_ROOT / "outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts"
DEFAULT_ARMS_ROOT = PROJECT_ROOT / "outputs/resource_v2_artifacts"
DEFAULT_OUT = PROJECT_ROOT / "outputs/resource_v2_scheduler_smoke_v1"

EPISODE_LIST_NAME = "resource_v2_smoke_v1_episode_ids.txt"
EPISODE_LIST_VERSION = "resource_v2_smoke_v1"
SELECTION_SEED = "20260920"
BOOTSTRAP_SEED = 20260920
BOOTSTRAP_B = 2000
N_EPISODES = 300
N_BASE_PER_CELL = 2
N_EXTRA_CELLS = 30

# pre-registered margin, signed off 2026-09-20 BEFORE any result was seen
NI_MARGIN_MS = 485.0
NI_MARGIN_SOURCE = "0.1 x 4853 ms, the frozen Phase 20B dev700 benefit of p95 over E0"

ARMS: Dict[str, Dict[str, Any]] = {
    "A0": {
        "description": "J3 three-quantile pack + legacy sum-q95 consumption",
        "overlay_arm": None,
        "simulator_policy": "sameshape_h5_p95",
        "consumer_policy_id": "legacy_sum_q95_v1",
    },
    "A1": {
        "description": "R1b discrete head + legacy sum-q95 consumption",
        "overlay_arm": "r1b",
        "simulator_policy": "sameshape_h5_p95",
        "consumer_policy_id": "legacy_sum_q95_v1",
    },
    "A2": {
        "description": "R3a-U adapted res_hidden + legacy sum-q95 consumption",
        "overlay_arm": "r3a_u",
        "simulator_policy": "sameshape_h5_p95",
        "consumer_policy_id": "legacy_sum_q95_v1",
        "eligible_for_model_selection": False,
        "role": "diagnostic_only",
        "ineligible_reason": "fails the frozen Phase-R calibration integrity gate (cov90 0.803)",
    },
    "B1": {
        "description": "R1b + sum of conditional runtime means",
        "overlay_arm": "r1b",
        "simulator_policy": "sameshape_h5_condmean",
        "consumer_policy_id": "sum_conditional_runtime_mean_plus_legacy_load_v1",
    },
    "B2": {
        "description": "R1b + sum of marginal step CVaR95",
        "overlay_arm": "r1b",
        "simulator_policy": "sameshape_h5_stepcvar95",
        "consumer_policy_id": "sum_marginal_step_cvar95_plus_legacy_load_v1",
    },
    "O": {
        "description": "joint future truth, same key shape (headroom ceiling)",
        "overlay_arm": None,
        "simulator_policy": "sameshape_h5_truth",
        "consumer_policy_id": "samekey_true_future_h5_v1",
        "predictor_artifact_id": "oracle_truth_v03",
        "naming": "joint-future-truth headroom, NOT a resource-head oracle",
    },
}

CONTRASTS: Tuple[Tuple[str, str, str], ...] = (
    ("A1", "A0", "predictor effect (R1b vs J3, same consumer)"),
    ("A2", "A1", "extra unfreezing, diagnostic control"),
    ("B1", "A1", "mean vs legacy risk consumption"),
    ("B2", "A1", "step-CVaR vs legacy risk consumption"),
    ("O", "A1", "remaining joint-future-information headroom"),
)

SUMMARY_METRICS = (
    "mean_completion_ms",
    "p95_completion_ms",
    "makespan_ms",
    "mean_job_queue_ms",
    "deadline_miss_rate",
    "gpu_evictions",
    "gpu_utilization",
    "failed_jobs",
    "completed_jobs",
)
PRIMARY_METRIC = "mean_completion_ms"


# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cell_of(episode: Mapping[str, Any]) -> str:
    return str(episode["scenario_cell"])


def cell_parts(cell: str) -> Tuple[str, str, str, str]:
    arrival, load, gpu, state = cell.split("|")
    return arrival, load, gpu, state


def balanced_extra_cells(cells: Sequence[str]) -> List[str]:
    """Pick 30 cells so each factor keeps its exact pre-registered marginal.

    Marginals after the 2-per-cell base draw are arrival 90 / load 54 / gpu 90 /
    state 90.  The extras must add 10 / 6 / 10 / 10, i.e. each of the 30 chosen
    cells contributes once to a balanced design over all four factors.  The cells
    are laid out as a full factorial (3 arrivals x 5 loads x 3 gpus x 3 states),
    so the construction below is exact rather than a random draw.
    """

    arrivals = sorted({cell_parts(c)[0] for c in cells})
    loads = sorted({cell_parts(c)[1] for c in cells})
    gpus = sorted({cell_parts(c)[2] for c in cells})
    states = sorted({cell_parts(c)[3] for c in cells})
    if (len(arrivals), len(loads), len(gpus), len(states)) != (3, 5, 3, 3):
        raise SystemExit(
            "expected a 3x5x3x3 full factorial, got %dx%dx%dx%d"
            % (len(arrivals), len(loads), len(gpus), len(states))
        )
    if len(cells) != 135:
        raise SystemExit("expected 135 cells, got %d" % len(cells))

    # six (gpu, state) pairs balanced on both factors, rotated per load
    pairs = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 0)]
    chosen: List[str] = []
    for load_index, load in enumerate(loads):
        rotated = pairs[load_index % 6:] + pairs[: load_index % 6]
        for k in range(6):
            arrival = arrivals[k % 3]
            gpu = gpus[rotated[k][0]]
            state = states[rotated[k][1]]
            chosen.append("|".join((arrival, load, gpu, state)))
    if len(set(chosen)) != N_EXTRA_CELLS:
        raise SystemExit("extra-cell construction produced duplicates")
    missing = [c for c in chosen if c not in set(cells)]
    if missing:
        raise SystemExit("constructed cells absent from the workload: %s" % missing[:3])
    return chosen


def select_episodes(
    episodes: Sequence[Mapping[str, Any]],
    n_base: int = N_BASE_PER_CELL,
    n_extra: int = N_EXTRA_CELLS,
) -> Tuple[List[str], Dict[str, Any]]:
    by_cell: Dict[str, List[str]] = {}
    for episode in episodes:
        by_cell.setdefault(cell_of(episode), []).append(str(episode["episode_id"]))
    for cell in by_cell:
        by_cell[cell].sort()

    cells = sorted(by_cell)
    selected: List[str] = []
    for cell in cells:
        ids = by_cell[cell]
        if len(ids) < n_base:
            raise SystemExit("cell %s has only %d episode(s); need %d" % (cell, len(ids), n_base))
        selected.extend(ids[:n_base])

    extras = balanced_extra_cells(cells)
    for cell in extras:
        ids = by_cell[cell]
        if len(ids) < n_base + 1:
            raise SystemExit("cell %s has only %d episode(s); need %d for the extra draw"
                             % (cell, len(ids), n_base + 1))
        selected.append(ids[n_base])

    if len(selected) != N_EPISODES:
        raise SystemExit("selected %d episodes, expected %d" % (len(selected), N_EPISODES))
    if len(set(selected)) != len(selected):
        raise SystemExit("selection contains duplicate episode ids")

    lookup = {str(e["episode_id"]): e for e in episodes}
    counts: Dict[str, Counter] = {key: Counter() for key in ("arrival", "load", "gpu", "state")}
    for episode_id in selected:
        arrival, load, gpu, state = cell_parts(cell_of(lookup[episode_id]))
        counts["arrival"][arrival] += 1
        counts["load"][load] += 1
        counts["gpu"][gpu] += 1
        counts["state"][state] += 1

    expected = {"arrival": 100, "load": 60, "gpu": 100, "state": 100}
    for key, want in expected.items():
        bad = {k: v for k, v in counts[key].items() if v != want}
        if bad:
            raise SystemExit("marginal mismatch for %s: %s (expected %d each)" % (key, bad, want))

    report = {
        "episode_selection_version": EPISODE_LIST_VERSION,
        "selection_seed": SELECTION_SEED,
        "n_episodes": len(selected),
        "n_cells": len(cells),
        "n_base_per_cell": n_base,
        "n_extra_cells": n_extra,
        "marginals": {key: dict(sorted(counts[key].items())) for key in counts},
        "extra_cells_sha256": hashlib.sha256("\n".join(sorted(extras)).encode("utf-8")).hexdigest(),
    }
    return sorted(selected), report


# --------------------------------------------------------------------------- #
def build_arm_artifacts(
    arm: str, base_root: Path, arms_root: Path, base_artifacts: Mapping[str, Mapping[str, Any]]
) -> Tuple[Mapping[str, Mapping[str, Any]], Dict[str, Any]]:
    config = ARMS[arm]
    overlay_arm = config["overlay_arm"]
    if overlay_arm is None:
        entry = {"overlay": False, "predictor_artifact_id": config.get("predictor_artifact_id")}
        if arm == "A0":
            entry["predictor_artifact_id"] = "j3_seed11@%s" % sha256_file(base_root / "b05_future_h5.jsonl.gz")
        return base_artifacts, entry
    artifacts, preflight = load_resource_v2_overlay(base_root, arms_root / overlay_arm)
    entry = {
        "overlay": True,
        "overlay_arm": overlay_arm,
        "predictor_artifact_id": "%s@%s" % (preflight.get("artifact_id"), preflight.get("artifact_sha256")),
        "preflight": preflight,
    }
    return artifacts, entry


def run_arm(
    arm: str,
    episode_ids: Sequence[str],
    episodes_by_id: Mapping[str, Mapping[str, Any]],
    templates: Mapping[str, Any],
    train_stats: Mapping[str, Any],
    base_root: Path,
    arms_root: Path,
    base_artifacts: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    config = ARMS[arm]
    artifacts, provenance = build_arm_artifacts(arm, base_root, arms_root, base_artifacts)
    rows: List[Dict[str, Any]] = []
    for episode_id in episode_ids:
        episode = episodes_by_id[episode_id]
        summary, _events = simulate_episode(
            episode,
            templates,
            config["simulator_policy"],
            future_artifacts=artifacts,
            train_stats=train_stats,
            collect_events=False,
        )
        row: Dict[str, Any] = {
            "episode_id": episode_id,
            "arm_id": arm,
            "scenario_cell": cell_of(episode),
            "arrival_pattern": cell_parts(cell_of(episode))[0],
            "realized_offered_compute_load": episode.get("realized_offered_compute_load"),
            "gpu_topology_mb": episode.get("gpu_topology_mb"),
            "initial_state": episode.get("initial_state"),
            "deadline_multiplier": episode.get("deadline_multiplier"),
        }
        for key in SUMMARY_METRICS:
            if key in summary:
                row[key] = summary[key]
        if PRIMARY_METRIC not in row:
            raise SystemExit(
                "arm %s episode %s summary lacks %s; keys=%s"
                % (arm, episode_id, PRIMARY_METRIC, sorted(summary))
            )
        rows.append(row)
    return rows, provenance


# --------------------------------------------------------------------------- #
def bootstrap_ci(
    deltas: Sequence[float], b: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED
) -> Dict[str, float]:
    if not deltas:
        raise ValueError("no deltas to bootstrap")
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(b):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    point = sum(deltas) / n
    lo = means[int(0.025 * b)]
    hi = means[min(b - 1, int(0.975 * b))]
    return {
        "point": point,
        "ci_low": lo,
        "ci_high": hi,
        "prob_le_zero": sum(1 for m in means if m <= 0.0) / b,
        "n": n,
    }


def bootstrap_diff_ci(
    a: Sequence[float], b_values: Sequence[float], b: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED
) -> Dict[str, float]:
    """Two-sided bootstrap over the paired difference, for robustness checks."""

    if len(a) != len(b_values):
        raise ValueError("paired samples differ in length")
    deltas = [x - y for x, y in zip(a, b_values)]
    return bootstrap_ci(deltas, b, seed)


def report(
    per_arm: Mapping[str, Sequence[Mapping[str, Any]]], output_dir: Path, margin: float
) -> Dict[str, Any]:
    def series(arm: str, metric: str = PRIMARY_METRIC) -> Dict[str, float]:
        return {str(r["episode_id"]): float(r[metric]) for r in per_arm[arm] if metric in r}

    base_series = series("A0")
    result: Dict[str, Any] = {
        "primary_metric": PRIMARY_METRIC,
        "non_inferiority_margin_ms": margin,
        "non_inferiority_source": NI_MARGIN_SOURCE,
        "arms": {},
        "contrasts": {},
    }
    for arm in per_arm:
        values = series(arm)
        ordered = [values[k] for k in sorted(values)]
        result["arms"][arm] = {
            "description": ARMS[arm]["description"],
            "consumer_policy_id": ARMS[arm]["consumer_policy_id"],
            "simulator_policy": ARMS[arm]["simulator_policy"],
            "eligible_for_model_selection": ARMS[arm].get("eligible_for_model_selection", True),
            "role": ARMS[arm].get("role", "candidate"),
            "n_episodes": len(values),
            "mean": sum(ordered) / len(ordered),
            "median": statistics.median(ordered),
        }

    for left, right, label in CONTRASTS:
        a, b_series = series(left), series(right)
        shared = sorted(set(a) & set(b_series))
        if not shared:
            continue
        deltas = [a[k] - b_series[k] for k in shared]
        stats = bootstrap_ci(deltas)
        stats["label"] = label
        # the old rule had no way to say "worse", so a clearly harmful arm came out
        # as merely "inconclusive"
        stats["verdict"] = (
            "material_improvement" if stats["ci_high"] < 0.0 and stats["point"] <= -margin
            else "non_inferior" if stats["ci_high"] < margin
            else "materially_inferior" if stats["ci_low"] > margin
            else "statistically_worse" if stats["ci_low"] > 0.0
            else "inconclusive"
        )
        result["contrasts"]["%s-%s" % (left, right)] = stats

    # P0-5: every arm must be measured on exactly the same episode set.  The old
    # code intersected the two episode sets, so an arm that silently ran fewer
    # episodes shrank the paired sample instead of failing.
    expected = None
    for arm, rows in per_arm.items():
        ids = {str(r["episode_id"]) for r in rows}
        if expected is None:
            expected = ids
            expected_arm = arm
        elif ids != expected:
            raise SystemExit(
                "arm %s covers %d episodes but arm %s covers %d; missing=%s extra=%s"
                % (arm, len(ids), expected_arm, len(expected),
                   sorted(expected - ids)[:3], sorted(ids - expected)[:3])
            )
    if expected is None:
        raise SystemExit("no arms to compare")
    reference_count = len(expected)
    for arm, rows in per_arm.items():
        if len(rows) != reference_count:
            raise SystemExit(
                "arm %s has %d rows for %d episodes (duplicate episode_id?)"
                % (arm, len(rows), reference_count)
            )

    def paired(left: str, right: str, label: str, **extra: Any) -> Dict[str, Any]:
        a, b_series = series(left), series(right)
        if set(a) != set(b_series):
            raise SystemExit("%s and %s do not cover the same episodes" % (left, right))
        keys = sorted(a)
        deltas = [a[k] - b_series[k] for k in keys]
        for k in keys:
            if not (a[k] == a[k]) or not (b_series[k] == b_series[k]):
                raise SystemExit("non-finite primary metric on episode %s" % k)
        stats = bootstrap_ci(deltas)
        stats["label"] = label
        stats.update(extra)
        return stats

    # P0-4: this used to overwrite the pre-registered A1-A0 entry written by the
    # CONTRASTS loop above, which is why contrasts.json lost its verdict.
    for arm in ("A1", "A2", "B1", "B2"):
        if arm in per_arm and arm != "A0" and "%s-A0" % arm not in result["contrasts"]:
            result["contrasts"]["%s-A0" % arm] = paired(
                arm, "A0", "%s vs the frozen champion" % arm
            )

    if "O" in per_arm and "A1" in per_arm:
        stats = paired("A1", "O", "A1 vs joint-future-truth reference")
        stats["practically_saturated"] = stats["ci_high"] <= margin
        result["contrasts"]["A1-O"] = stats
    result["paired_episode_count"] = reference_count

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "contrasts.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("select", "preflight", "run", "report"), required=True)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATES)
    parser.add_argument("--base-pack", type=Path, default=DEFAULT_BASE_PACK)
    parser.add_argument("--arms-root", type=Path, default=DEFAULT_ARMS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--limit", type=int, default=0, help="debug: cap the episode count")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_list = args.output_dir / EPISODE_LIST_NAME

    if args.stage == "select":
        episodes = read_jsonl(args.episodes)
        selected, selection = select_episodes(episodes)
        episode_list.write_text("\n".join(selected) + "\n", encoding="utf-8")
        selection["episode_list"] = str(episode_list)
        selection["episode_list_sha256"] = sha256_file(episode_list)
        (args.output_dir / "episode_selection.json").write_text(
            json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(selection, ensure_ascii=False, indent=2))
        return 0

    if not episode_list.is_file():
        raise SystemExit("run --stage select first; %s is missing" % episode_list)
    episode_ids = [line.strip() for line in episode_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        episode_ids = episode_ids[: args.limit]

    episodes = read_jsonl(args.episodes)
    episodes_by_id = {str(e["episode_id"]): e for e in episodes}
    unknown = [e for e in episode_ids if e not in episodes_by_id]
    if unknown:
        raise SystemExit("episode list refers to unknown ids: %s" % unknown[:3])

    if args.stage == "preflight":
        failures: List[str] = []
        for arm in args.arms:
            try:
                if ARMS[arm]["overlay_arm"] is None:
                    print("[PASS] %-3s no overlay (%s)" % (arm, ARMS[arm]["simulator_policy"]))
                    continue
                _, provenance = build_arm_artifacts(arm, args.base_pack, args.arms_root, {})
                print("[PASS] %-3s %s" % (arm, provenance["predictor_artifact_id"]))
            except Exception as exc:  # noqa: BLE001
                failures.append("%s: %s" % (arm, exc))
                print("[FAIL] %-3s %s" % (arm, exc))
        print("\npreflight:", "ALL PASS" if not failures else "FAILED (%d)" % len(failures))
        return 1 if failures else 0

    if args.stage == "run":
        templates = load_templates(args.templates)
        train_stats = train_resource_stats(templates)
        base_artifacts = load_future_artifacts(args.base_pack)
        run_meta: Dict[str, Any] = {
            "experiment_id": "EXP-20260920_resource_v2_scheduler_smoke_v1",
            "workload_file": str(args.episodes),
            "workload_file_sha256": sha256_file(args.episodes),
            "template_file": str(args.templates),
            "template_file_sha256": sha256_file(args.templates),
            "episode_selection_version": EPISODE_LIST_VERSION,
            "episode_list_sha256": sha256_file(episode_list),
            "n_episodes": len(episode_ids),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_B": BOOTSTRAP_B,
            "non_inferiority_margin_ms": NI_MARGIN_MS,
            "non_inferiority_source": NI_MARGIN_SOURCE,
            "arms_provenance": {},
        }
        all_rows: List[Dict[str, Any]] = []
        for arm in args.arms:
            rows, provenance = run_arm(
                arm, episode_ids, episodes_by_id, templates, train_stats,
                args.base_pack, args.arms_root, base_artifacts,
            )
            run_meta["arms_provenance"][arm] = provenance
            all_rows.extend(rows)
            print("[done] %-3s %d episodes, mean %s = %.1f" % (
                arm, len(rows), PRIMARY_METRIC,
                sum(r[PRIMARY_METRIC] for r in rows) / len(rows),
            ))
        with (args.output_dir / "per_episode.jsonl").open("w", encoding="utf-8") as handle:
            for row in all_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        (args.output_dir / "run_meta.json").write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("saved:", args.output_dir / "per_episode.jsonl")
        return 0

    # report
    rows = read_jsonl(args.output_dir / "per_episode.jsonl")
    per_arm: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        per_arm.setdefault(str(row["arm_id"]), []).append(row)
    result = report(per_arm, args.output_dir, NI_MARGIN_MS)
    print(json.dumps(result["contrasts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
