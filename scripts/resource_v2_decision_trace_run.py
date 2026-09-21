#!/usr/bin/env python
"""Decision-level trace run: one real trajectory, eight shadow methods.

Follows the planning review's design:

* only one method actually decides (``--trajectory-policy``, default the A0
  champion ``sameshape_h5_p95``); every other method is scored on the identical
  candidate pool, so ``current``, residency, priority and GPU assignment are shared;
* a hard regression guard compares the chosen-candidate sequence with tracing OFF
  and ON - if it differs, the run is invalid and nothing else is reported;
* three reports: a decision-level wide table, a method-pair mechanism summary
  (reporting future-only and total-score agreement separately), and the flip cases
  most worth reading by hand.

Stages
------
``guard``   regression guard on a few episodes (trace ON vs OFF)
``run``     trace the full episode list
``report``  build the three reports from an existing trace
``all``     guard, then run, then report

Usage::

    python scripts/resource_v2_decision_trace_run.py --stage all --limit 300
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from tracing.analysis import decision_trace as dt  # noqa: E402
from tracing.analysis.attribute_mixture_methods import (  # noqa: E402
    AttrArgmaxStatsMethod,
    AttrMixAtomMethod,
    AttrMixHist16Method,
    StatsBankLookup,
    load_stats_bank,
)
from tracing.analysis.workload_v02_simulator import (  # noqa: E402
    _legacy_p95_load_cost,
    load_future_artifacts,
    load_resource_v2_overlay,
    load_templates,
    read_jsonl,
    set_decision_trace,
    simulate_episode,
    train_resource_stats,
)

DEFAULT_TEMPLATES = PROJECT_ROOT / "results/processed/r7_workload_v03_no_run_container/job_templates_r7_v03.jsonl"
DEFAULT_EPISODES = PROJECT_ROOT / "results/processed/r7_workload_v03_no_run_container/episodes/workload_validation_r7_v03.jsonl"
DEFAULT_BASE = PROJECT_ROOT / "outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts"
DEFAULT_R1B = PROJECT_ROOT / "outputs/resource_v2_artifacts/r1b"
DEFAULT_BANK = PROJECT_ROOT / "outputs/resource_v2_decision_trace/stats_bank.json"
DEFAULT_BINS = PROJECT_ROOT / "outputs/j_series_resource_dist_v1/bins.json"
DEFAULT_OUT = PROJECT_ROOT / "outputs/resource_v2_decision_trace"
DEFAULT_EPISODE_LIST = PROJECT_ROOT / "outputs/resource_v2_scheduler_smoke_v1/resource_v2_smoke_v1_episode_ids.txt"

SUMMARY_KEYS = ("mean_completion_ms", "p95_completion_ms", "makespan_ms", "mean_job_queue_ms",
                "deadline_miss_rate", "gpu_evictions", "completed_jobs", "failed_jobs")

OBSERVABILITY_GATES = (
    "episodes_traced", "missing_decisions", "missing_candidates", "duplicate_candidate_ids",
    "non_finite_scores", "malformed_scheduler_keys", "identical_candidate_sets",
    "timing_fields_valid",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bin_reps(path: Path) -> Tuple[List[float], List[float]]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    return [float(e) for e in spec["edges"]], [float(r) for r in spec["representatives_ms"]]


def build_methods(base_artifacts, r1b_artifacts, bank, representatives) -> List[dt.ShadowMethod]:
    lookup = StatsBankLookup(bank)
    return [
        dt.ArtifactQuantileMethod(
            "j3_q95_v1", base_artifacts, field="runtime_ms_quantiles", statistic="p95",
            consumer_policy_id="legacy_sum_q95_v1", predictor_artifact_id="j3_seed11",
            load_cost=_legacy_p95_load_cost,
        ),
        dt.ArtifactQuantileMethod(
            "r1b_q95_v1", r1b_artifacts, field="runtime_ms_quantiles", statistic="p95",
            consumer_policy_id="legacy_sum_q95_v1", predictor_artifact_id="resource_v2_r1b_seed11",
            load_cost=_legacy_p95_load_cost,
        ),
        dt.ArtifactQuantileMethod(
            "r1b_mean_v1", r1b_artifacts, field="runtime_mean_ms", statistic="runtime_mean_ms",
            consumer_policy_id="sum_conditional_runtime_mean_plus_legacy_load_v1",
            predictor_artifact_id="resource_v2_r1b_seed11", load_cost=_legacy_p95_load_cost,
        ),
        dt.ArtifactQuantileMethod(
            "r1b_stepcvar95_v1", r1b_artifacts, field="cvar95_ms", statistic="cvar95_ms",
            consumer_policy_id="sum_marginal_step_cvar95_plus_legacy_load_v1",
            predictor_artifact_id="resource_v2_r1b_seed11", load_cost=_legacy_p95_load_cost,
        ),
        # the attribute family deliberately keeps the true legacy cost semantics
        # (runtime + stats load, no artifact load); mixing in the resource-v2 load
        # would confound the argmax-vs-mixture comparison
        AttrArgmaxStatsMethod(base_artifacts, lookup),
        AttrMixAtomMethod(base_artifacts, lookup),
        AttrMixHist16Method(base_artifacts, lookup, representatives, statistic="p95"),
        dt.TruthReferenceMethod(),
    ]


# --------------------------------------------------------------------------- #
def stage_guard(args, templates, stats, base_artifacts, r1b_artifacts, methods, episodes) -> int:
    """Tracing must not change a single decision."""

    limit = args.limit or 3
    print("=== regression guard: %d episode(s), trace OFF vs ON ===" % limit)
    mismatches: List[Tuple[str, str, Any, Any]] = []
    for episode in episodes[:limit]:
        set_decision_trace(None)
        off, _ = simulate_episode(episode, templates, args.trajectory_policy,
                                  future_artifacts=base_artifacts, train_stats=stats,
                                  collect_events=False)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        writer = dt.DecisionTraceWriter(args.output_dir / "guard_trace.jsonl.gz")
        set_decision_trace(writer, methods, truth_cost=True, trajectory_method_id="j3_q95_v1")
        on, _ = simulate_episode(episode, templates, args.trajectory_policy,
                                 future_artifacts=base_artifacts, train_stats=stats,
                                 collect_events=False)
        writer.close()
        set_decision_trace(None)
        for key in SUMMARY_KEYS:
            if off.get(key) != on.get(key):
                mismatches.append((episode["episode_id"], key, off.get(key), on.get(key)))
    if mismatches:
        print("GUARD FAILED (%d mismatches):" % len(mismatches))
        for row in mismatches[:10]:
            print("   %s %s: off=%r on=%r" % row)
        return 1
    print("GUARD PASSED: %d episodes, all %d metrics identical with tracing on and off"
          % (limit, len(SUMMARY_KEYS)))
    return 0


def stage_run(args, templates, stats, base_artifacts, r1b_artifacts, methods, episodes) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "decision_trace_v1.jsonl.gz"
    writer = dt.DecisionTraceWriter(trace_path)
    set_decision_trace(writer, methods, truth_cost=True, trajectory_method_id="j3_q95_v1")

    per_episode: List[Dict[str, Any]] = []
    started = time.time()
    for index, episode in enumerate(episodes):
        summary, _events = simulate_episode(
            episode, templates, args.trajectory_policy,
            future_artifacts=base_artifacts, train_stats=stats, collect_events=False,
        )
        row = {"episode_id": str(episode["episode_id"]), "scenario_cell": episode.get("scenario_cell")}
        for key in SUMMARY_KEYS:
            row[key] = summary.get(key)
        per_episode.append(row)
        if (index + 1) % 50 == 0:
            print("  ... %d/%d episodes, %d decisions" % (index + 1, len(episodes), writer.decisions))
    report = writer.close()
    set_decision_trace(None)

    meta = {
        "experiment_id": "EXP-20260920_resource_v2_decision_trace_v1",
        "trajectory_policy": args.trajectory_policy,
        "episodes": len(episodes),
        "wall_seconds": round(time.time() - started, 1),
        "trace": report,
        "methods": [m.describe() for m in methods],
        "cost_semantics": dt.COST_SEMANTICS,
        "templates_sha256": sha256_file(args.templates),
        "episodes_sha256": sha256_file(args.episodes),
        "stats_bank_sha256": sha256_file(args.stats_bank),
        "base_pack_sha256": sha256_file(args.base_pack / "b05_future_h5.jsonl.gz"),
        "r1b_pack_sha256": sha256_file(args.r1b_root / "b05_future_h5.jsonl.gz"),
        "bins_sha256": sha256_file(args.bins),
    }
    (args.output_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "per_episode.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_episode:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print("traced %d decisions / %d candidates in %.1f s"
          % (report["decisions"], report["candidates"], meta["wall_seconds"]))
    print("chosen_sequence_sha256:", report["chosen_sequence_sha256"])
    return 0


# --------------------------------------------------------------------------- #
def iter_decisions(path: Path) -> Iterable[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def kendall_tau(pairs: Sequence[Tuple[float, float]]) -> float:
    """Kendall tau-b over paired scores (ties handled explicitly)."""

    n = len(pairs)
    if n < 2:
        return float("nan")
    concordant = discordant = ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pairs[i][0] - pairs[j][0]
            dy = pairs[i][1] - pairs[j][1]
            if dx == 0 and dy == 0:
                ties_x += 1
                ties_y += 1
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    denominator = ((concordant + discordant + ties_x) * (concordant + discordant + ties_y)) ** 0.5
    if denominator == 0:
        return float("nan")
    return (concordant - discordant) / denominator


def bootstrap_ci(values: Sequence[float], *, b: int = 2000, seed: int = 20260920) -> Dict[str, float]:
    import random as _random

    if not values:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = _random.Random(seed)
    n = len(values)
    means = []
    for _ in range(b):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return {
        "point": sum(values) / n,
        "ci_low": means[int(0.025 * b)],
        "ci_high": means[min(b - 1, int(0.975 * b))],
        "prob_le_zero": sum(1 for m in means if m <= 0.0) / b,
        "n": n,
    }


def stage_report(args) -> int:
    trace_path = args.output_dir / "decision_trace_v1.jsonl.gz"
    decisions = list(iter_decisions(trace_path))
    print("loaded %d decisions" % len(decisions))

    method_ids = sorted({m for d in decisions for c in d["candidates"] for m in c["scores"]})
    gates = {key: 0 for key in OBSERVABILITY_GATES}
    gates["episodes_traced"] = len({d["episode_id"] for d in decisions})

    # ---- real gates (the previous version only initialised them) ---------- #
    all_episodes = {d["episode_id"] for d in decisions}
    per_episode_decisions: Dict[str, int] = defaultdict(int)
    for decision in decisions:
        per_episode_decisions[decision["episode_id"]] += 1
        if not decision.get("candidates"):
            gates["missing_candidates"] += 1
        if not (decision.get("actual_choice") or {}).get("candidate_id"):
            gates["missing_decisions"] += 1
        ids = [c["candidate_id"] for c in decision["candidates"]]
        if len(set(ids)) != len(ids):
            gates["duplicate_candidate_ids"] += 1
        for cand in decision["candidates"]:
            for method_id, score in cand["scores"].items():
                total = float(score["total_cost_ms"])
                if not (total == total) or total in (float("inf"), float("-inf")):
                    gates["non_finite_scores"] += 1
                if len(score["scheduler_key"]) != 8:
                    gates["malformed_scheduler_keys"] += 1
            if len(cand["scores"]) != len(method_ids):
                gates["identical_candidate_sets"] += 1
        for method_id, timing in (decision.get("method_timings") or {}).items():
            for key in ("score_compute_wall_ns", "rank_select_wall_ns", "total_wall_ns"):
                value = timing.get(key)
                if value is None or value < 0:
                    gates["timing_fields_valid"] += 1
    gates["episodes_expected"] = len(all_episodes)

    # ---- paired delta-tau (was a difference of medians) ------------------ #
    paired: Dict[str, List[float]] = defaultdict(list)
    pair_flip_macro: Dict[str, List[float]] = defaultdict(list)
    pair_flip_micro: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    tie_transitions: Dict[str, int] = defaultdict(int)
    strict_flips: Dict[str, int] = defaultdict(int)
    top1_dis: Dict[str, List[float]] = defaultdict(list)
    margin_rows: List[Dict[str, Any]] = []
    dominance_rows: List[Dict[str, Any]] = []
    cross_lane: List[float] = []

    for decision in decisions:
        candidates = decision["candidates"]
        competitive = [c for c in candidates if c["priority"] == decision["competitive_priority"]]
        for cand in candidates:
            for method_id, score in cand["scores"].items():
                probs = score.get("runtime_probs")
                if probs and method_id.startswith("attrmix"):
                    pass
        if len(competitive) < 2:
            continue
        for left, right in ((a, b) for i, a in enumerate(method_ids) for b in method_ids[i + 1:]):
            key = "%s|%s" % (left, right)
            fp, tp = [], []
            for i in range(len(competitive)):
                for j in range(i + 1, len(competitive)):
                    a, b = competitive[i], competitive[j]
                    fa, fb = float(a["scores"][left]["future_cost_ms"]), float(b["scores"][left]["future_cost_ms"])
                    ga, gb = float(a["scores"][right]["future_cost_ms"]), float(b["scores"][right]["future_cost_ms"])
                    fp.append((fa, ga))
                    tp.append((fa + a["current_cost_ms"], ga + a["current_cost_ms"]))
                    dl = float(a["scores"][left]["total_cost_ms"]) - float(b["scores"][left]["total_cost_ms"])
                    dr = float(a["scores"][right]["total_cost_ms"]) - float(b["scores"][right]["total_cost_ms"])
                    if dl * dr < 0:
                        strict_flips[key] += 1
                    if (dl == 0) != (dr == 0):
                        tie_transitions[key] += 1
                    pair_flip_micro[key][1] += 1
                    pair_flip_micro[key][0] += int(dl * dr < 0)
            tf, tt = kendall_tau(fp), kendall_tau(tp)
            if tf == tf and tt == tt:
                paired[key].append(tf - tt)
                pair_flip_macro[key].append(strict_flips[key] / max(1, len(fp) * (len(fp) - 1) // 2) if False else 0.0)
            lw = min(competitive, key=lambda c: c["scores"][left]["scheduler_key"])["candidate_id"]
            rw = min(competitive, key=lambda c: c["scores"][right]["scheduler_key"])["candidate_id"]
            top1_dis[key].append(1.0 if lw != rw else 0.0)

    # per-decision pair flip for macro mean, recomputed cleanly
    for decision in decisions:
        competitive = [c for c in decision["candidates"] if c["priority"] == decision["competitive_priority"]]
        if len(competitive) < 2:
            continue
        for left, right in (("j3_q95_v1", "r1b_q95_v1"),):
            if left not in method_ids or right not in method_ids:
                continue
            flips = pairs = 0
            for i in range(len(competitive)):
                for j in range(i + 1, len(competitive)):
                    a, b = competitive[i], competitive[j]
                    dl = float(a["scores"][left]["total_cost_ms"]) - float(b["scores"][left]["total_cost_ms"])
                    dr = float(a["scores"][right]["total_cost_ms"]) - float(b["scores"][right]["total_cost_ms"])
                    pairs += 1
                    flips += int(dl * dr < 0)
            if pairs:
                pair_flip_macro["j3_q95_v1|r1b_q95_v1"].append(flips / pairs)

    summary_rows = [
        "pair,n_paired,delta_tau_paired_median,delta_tau_paired_mean,"
        "delta_tau_ci_low,delta_tau_ci_high,prob_le_zero,"
        "pair_flip_macro_mean,micro_flip_rate,top1_disagreement_rate,tie_transition_rate"
    ]
    paired_report = {}
    for key in sorted(paired):
        values = paired[key]
        stats = bootstrap_ci(values)
        micro = pair_flip_micro.get(key, [0, 0])
        dis = top1_dis.get(key) or [float("nan")]
        ties = tie_transitions.get(key, 0)
        summary_rows.append(
            "%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f"
            % (key, stats["n"], statistics.median(values), stats["point"],
               stats["ci_low"], stats["ci_high"], stats["prob_le_zero"],
               statistics.fmean(pair_flip_macro.get(key) or [float("nan")]),
               micro[0] / max(1, micro[1]), statistics.fmean(dis),
               ties / max(1, micro[1]))
        )
        paired_report[key] = stats
    (args.output_dir / "report2_method_pairs.csv").write_text("\n".join(summary_rows) + "\n", encoding="utf-8")

    # ---- diagnostic: decision margin deciles ----------------------------- #
    # answers "is the 22.4 percent winner disagreement boundary jitter or a
    # structural score-geometry difference?"
    for decision in decisions:
        competitive = [c for c in decision["candidates"] if c["priority"] == decision["competitive_priority"]]
        if len(competitive) < 2 or "j3_q95_v1" not in method_ids or "r1b_q95_v1" not in method_ids:
            continue
        ordered = sorted(competitive, key=lambda c: c["scores"]["j3_q95_v1"]["total_cost_ms"])
        margin = float(ordered[1]["scores"]["j3_q95_v1"]["total_cost_ms"]) - float(
            ordered[0]["scores"]["j3_q95_v1"]["total_cost_ms"])
        denom = max(1.0, abs(float(ordered[0]["scores"]["j3_q95_v1"]["total_cost_ms"])))
        j3w = ordered[0]["candidate_id"]
        r1bw = min(competitive, key=lambda c: c["scores"]["r1b_q95_v1"]["scheduler_key"])["candidate_id"]
        current_spread = max(c["current_cost_ms"] for c in competitive) - min(
            c["current_cost_ms"] for c in competitive)
        future_spread = max(c["scores"]["j3_q95_v1"]["future_cost_ms"] for c in competitive) - min(
            c["scores"]["j3_q95_v1"]["future_cost_ms"] for c in competitive)
        margin_rows.append({
            "episode_id": decision["episode_id"],
            "decision_id": decision["decision_id"],
            "margin_ms": margin,
            "relative_margin": margin / denom,
            "disagree": int(j3w != r1bw),
            "candidate_count": len(competitive),
        })
        dominance_rows.append({
            "episode_id": decision["episode_id"],
            "decision_id": decision["decision_id"],
            "current_spread_ms": current_spread,
            "future_spread_ms": future_spread,
            "dominance": current_spread / max(1e-9, future_spread),
            "disagree": int(j3w != r1bw),
        })

    def decile_report(rows: List[Dict[str, Any]], key: str, label: str) -> List[str]:
        ordered = sorted(rows, key=lambda r: r[key])
        n = len(ordered)
        out = ["%s_bucket,n,mean_value,disagreement_rate" % label]
        for d in range(10):
            lo, hi = int(d * n / 10), int((d + 1) * n / 10)
            chunk = ordered[lo:hi]
            if not chunk:
                continue
            out.append("%d,%d,%.4f,%.4f" % (
                d, len(chunk), statistics.fmean(r[key] for r in chunk),
                statistics.fmean(r["disagree"] for r in chunk)))
        return out

    if margin_rows:
        (args.output_dir / "report4_margin_deciles.csv").write_text(
            "\n".join(decile_report(margin_rows, "relative_margin", "relative_margin")) + "\n",
            encoding="utf-8")
        (args.output_dir / "report5_dominance_deciles.csv").write_text(
            "\n".join(decile_report(dominance_rows, "dominance", "current_future_dominance")) + "\n",
            encoding="utf-8")

    # ---- diagnostic: S/R counterfactual decomposition -------------------- #
    # S: keep J3's future ranking, take R1b's future value multiset
    # R: keep J3's future value multiset, take R1b's ranking
    sr_rows = []
    for decision in decisions:
        competitive = [c for c in decision["candidates"] if c["priority"] == decision["competitive_priority"]]
        if len(competitive) < 2 or "j3_q95_v1" not in method_ids or "r1b_q95_v1" not in method_ids:
            continue
        j = [float(c["scores"]["j3_q95_v1"]["future_cost_ms"]) for c in competitive]
        r = [float(c["scores"]["r1b_q95_v1"]["future_cost_ms"]) for c in competitive]
        current = [float(c["current_cost_ms"]) for c in competitive]
        j_order = sorted(range(len(j)), key=lambda i: j[i])
        r_order = sorted(range(len(r)), key=lambda i: r[i])
        j_sorted = [j[i] for i in j_order]
        r_sorted = [r[i] for i in r_order]
        f_s = [0.0] * len(j)   # J3 ranking, R1b values
        f_r = [0.0] * len(j)   # J3 values, R1b ranking
        for rank, idx in enumerate(j_order):
            f_s[idx] = r_sorted[rank]
        for rank, idx in enumerate(r_order):
            f_r[idx] = j_sorted[rank]
        j3w = min(range(len(j)), key=lambda i: (current[i] + j[i], i))
        r1bw = min(range(len(j)), key=lambda i: (current[i] + r[i], i))
        sw = min(range(len(j)), key=lambda i: (current[i] + f_s[i], i))
        rw = min(range(len(j)), key=lambda i: (current[i] + f_r[i], i))
        sr_rows.append({
            "episode_id": decision["episode_id"],
            "decision_id": decision["decision_id"],
            "j3_vs_r1b": int(j3w != r1bw),
            "s_reproduces": int(sw == r1bw) if j3w != r1bw else -1,
            "r_reproduces": int(rw == r1bw) if j3w != r1bw else -1,
        })
    if sr_rows:
        disagreed = [row for row in sr_rows if row["j3_vs_r1b"]]
        s_rate = statistics.fmean(row["s_reproduces"] for row in disagreed) if disagreed else float("nan")
        r_rate = statistics.fmean(row["r_reproduces"] for row in disagreed) if disagreed else float("nan")
        (args.output_dir / "report6_SR_counterfactual.json").write_text(json.dumps({
            "n_decisions": len(sr_rows),
            "n_disagreements": len(disagreed),
            "S_value_only_reproduces_r1b_choice_rate": s_rate,
            "R_rank_only_reproduces_r1b_choice_rate": r_rate,
            "reading": "higher S means scale/spacing dominates; higher R means rank association dominates",
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "decisions": len(decisions),
        "method_ids": method_ids,
        "observability_gates": gates,
        "paired_delta_tau": {k: v for k, v in sorted(paired_report.items())},
        "j3_vs_r1b_top1_disagreement_competitive": statistics.fmean(
            top1_dis.get("j3_q95_v1|r1b_q95_v1") or [float("nan")]),
        "note_denominators": (
            "competitive-conditioned rates use only decisions with >=2 candidates in the "
            "minimum priority tier; the all-decision rate is reported separately"
        ),
    }
    (args.output_dir / "report_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "paired_delta_tau"},
                     ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("guard", "run", "report", "all"), required=True)
    parser.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATES)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--episode-list", type=Path, default=DEFAULT_EPISODE_LIST)
    parser.add_argument("--base-pack", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--r1b-root", type=Path, default=DEFAULT_R1B)
    parser.add_argument("--stats-bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--bins", type=Path, default=DEFAULT_BINS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--trajectory-policy", default="sameshape_h5_p95")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.stage == "report":
        return stage_report(args)

    templates = load_templates(args.templates)
    stats = train_resource_stats(templates)
    base_artifacts = load_future_artifacts(args.base_pack)
    r1b_artifacts, _preflight = load_resource_v2_overlay(args.base_pack, args.r1b_root)
    bank = load_stats_bank(args.stats_bank)
    _edges, representatives = load_bin_reps(args.bins)
    methods = build_methods(base_artifacts, r1b_artifacts, bank, representatives)

    episodes = read_jsonl(args.episodes)
    by_id = {str(e["episode_id"]): e for e in episodes}
    if args.episode_list.is_file():
        ids = [line.strip() for line in args.episode_list.read_text(encoding="utf-8").splitlines() if line.strip()]
        unknown = [i for i in ids if i not in by_id]
        if unknown:
            raise SystemExit("episode list refers to unknown ids: %s" % unknown[:3])
        episodes = [by_id[i] for i in ids]
    if args.limit:
        episodes = episodes[: args.limit]
    print("episodes: %d | methods: %d" % (len(episodes), len(methods)))

    if args.stage in ("guard", "all"):
        code = stage_guard(args, templates, stats, base_artifacts, r1b_artifacts, methods, episodes)
        if code:
            return code
    if args.stage in ("run", "all"):
        code = stage_run(args, templates, stats, base_artifacts, r1b_artifacts, methods, episodes)
        if code:
            return code
    if args.stage == "all":
        return stage_report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
