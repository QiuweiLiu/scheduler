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
        AttrArgmaxStatsMethod(base_artifacts, lookup, legacy_load=_legacy_p95_load_cost),
        AttrMixAtomMethod(base_artifacts, lookup, legacy_load=_legacy_p95_load_cost),
        AttrMixHist16Method(base_artifacts, lookup, representatives, statistic="p95",
                            legacy_load=_legacy_p95_load_cost),
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
        set_decision_trace(writer, methods, truth_cost=True)
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
    set_decision_trace(writer, methods, truth_cost=True)

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


def stage_report(args) -> int:
    trace_path = args.output_dir / "decision_trace_v1.jsonl.gz"
    decisions = list(iter_decisions(trace_path))
    print("loaded %d decisions" % len(decisions))

    method_ids = sorted({m for d in decisions for c in d["candidates"] for m in c["scores"]})
    gates = {key: 0 for key in OBSERVABILITY_GATES}
    gates["episodes_traced"] = len({d["episode_id"] for d in decisions})

    # ---- Report 1: decision-level wide table (sample) -------------------- #
    wide_rows: List[str] = []
    wide_rows.append("episode,decision,candidate,current," +
                     ",".join("%s_total" % m for m in method_ids) + ",truth_total")
    for decision in decisions[:200]:
        for cand in decision["candidates"]:
            cells = ["%.1f" % cand["current_cost_ms"]]
            for method_id in method_ids:
                cells.append("%.1f" % cand["scores"][method_id]["total_cost_ms"])
            wide_rows.append("%s,%d,%s,%s" % (
                decision["episode_id"], decision["decision_id"],
                cand["candidate_id"].replace(",", "_"), ",".join(cells)))
    (args.output_dir / "report1_decision_wide_table.csv").write_text("\n".join(wide_rows) + "\n", encoding="utf-8")

    # ---- Report 2: method-pair mechanism summary ------------------------- #
    pair_stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    disagreement = Counter()
    margin_crossings = Counter()
    truth_top1 = Counter()
    for decision in decisions:
        candidates = decision["candidates"]
        ids = [c["candidate_id"] for c in candidates]
        if len(set(ids)) != len(ids):
            gates["duplicate_candidate_ids"] += 1
        for cand in candidates:
            for method_id, score in cand["scores"].items():
                total = float(score["total_cost_ms"])
                if total != total or total in (float("inf"), float("-inf")):
                    gates["non_finite_scores"] += 1
                if len(score["scheduler_key"]) != 8:
                    gates["malformed_scheduler_keys"] += 1
        competitive = [c for c in candidates if c["priority"] == decision["competitive_priority"]]
        if len(competitive) < 2:
            continue
        truth_key = "truth_h5_v1"
        if truth_key not in method_ids:
            continue
        truth_winner = min(
            competitive,
            key=lambda c: (c["scores"][truth_key]["total_cost_ms"], c["candidate_id"]),
        )["candidate_id"]
        for method_id in method_ids:
            if method_id == truth_key:
                continue
            chosen = min(
                competitive,
                key=lambda c: (c["scores"][method_id]["scheduler_key"], c["candidate_id"]),
            )["candidate_id"]
            if chosen == truth_winner:
                truth_top1[method_id] += 1
            disagreement[(method_id, truth_key)] += int(chosen != truth_winner)
        pair_disagreement = 0
        pair_total = 0
        for left, right in ((a, b) for i, a in enumerate(method_ids) for b in method_ids[i + 1:]):
            if left == right:
                continue
            # top-1 disagreement must be measured against the pair partner, not
            # always against truth (the earlier version mislabelled that column)
            left_winner = min(
                competitive,
                key=lambda c: (c["scores"][left]["scheduler_key"], c["candidate_id"]),
            )["candidate_id"]
            right_winner = min(
                competitive,
                key=lambda c: (c["scores"][right]["scheduler_key"], c["candidate_id"]),
            )["candidate_id"]
            pair_disagreement += int(left_winner != right_winner)
            pair_total += 1
            future_pairs = []
            total_pairs = []
            flip_count = 0
            pairs = 0
            for i in range(len(competitive)):
                for j in range(i + 1, len(competitive)):
                    a, b = competitive[i], competitive[j]
                    fa = float(a["scores"][left]["future_cost_ms"])
                    fb = float(b["scores"][left]["future_cost_ms"])
                    ga = float(a["scores"][right]["future_cost_ms"])
                    gb = float(b["scores"][right]["future_cost_ms"])
                    future_pairs.append((fa, ga))
                    total_pairs.append((fa + a["current_cost_ms"], ga + a["current_cost_ms"]))
                    ta = float(a["scores"][left]["total_cost_ms"])
                    tb = float(b["scores"][left]["total_cost_ms"])
                    ua = float(a["scores"][right]["total_cost_ms"])
                    ub = float(b["scores"][right]["total_cost_ms"])
                    pairs += 1
                    flip_count += int(((ta - tb) > 0) != ((ua - ub) > 0))
            key = "%s|%s" % (left, right)
            tau_f = kendall_tau(future_pairs)
            tau_t = kendall_tau(total_pairs)
            if tau_f == tau_f:
                pair_stats[key]["tau_future"].append(tau_f)
            if tau_t == tau_t:
                pair_stats[key]["tau_total"].append(tau_t)
            if pairs:
                pair_stats[key]["pair_flip_rate"].append(flip_count / pairs)
            pair_stats[key]["top1_disagreement"].append(1.0 if left_winner != right_winner else 0.0)

    summary_rows = ["pair,n,tau_future_median,tau_total_median,delta_tau_median,pair_flip_rate_median,top1_disagreement_rate"]
    for key, values in sorted(pair_stats.items()):
        left, right = key.split("|")
        n = len(values.get("tau_total") or [])
        tf = statistics.median(values["tau_future"]) if values.get("tau_future") else float("nan")
        tt = statistics.median(values["tau_total"]) if values.get("tau_total") else float("nan")
        pfr = statistics.median(values["pair_flip_rate"]) if values.get("pair_flip_rate") else float("nan")
        dis_values = values.get("top1_disagreement") or []
        dis = statistics.fmean(dis_values) if dis_values else float("nan")
        summary_rows.append("%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f"
                            % (key, n, tf, tt, tf - tt, pfr, dis))
    (args.output_dir / "report2_method_pairs.csv").write_text("\n".join(summary_rows) + "\n", encoding="utf-8")

    # ---- Report 3: flip cases worth reading ------------------------------ #
    flips: List[Dict[str, Any]] = []
    for decision in decisions:
        competitive = [c for c in decision["candidates"] if c["priority"] == decision["competitive_priority"]]
        if len(competitive) < 2:
            continue
        left, right = "j3_q95_v1", "r1b_q95_v1"
        if left not in method_ids or right not in method_ids:
            continue
        wl = min(competitive, key=lambda c: (c["scores"][left]["scheduler_key"], c["candidate_id"]))
        wr = min(competitive, key=lambda c: (c["scores"][right]["scheduler_key"], c["candidate_id"]))
        if wl["candidate_id"] == wr["candidate_id"]:
            continue
        flips.append({
            "episode_id": decision["episode_id"],
            "decision_id": decision["decision_id"],
            "j3_choice": wl["candidate_id"],
            "r1b_choice": wr["candidate_id"],
            "j3_total": float(wl["scores"][left]["total_cost_ms"]),
            "r1b_total": float(wr["scores"][right]["total_cost_ms"]),
            "r1b_total_for_j3_choice": float(wl["scores"][right]["total_cost_ms"]),
            "j3_total_for_r1b_choice": float(wr["scores"][left]["total_cost_ms"]),
            "truth_total_j3_choice": float(wl["scores"].get("truth_h5_v1", {}).get("total_cost_ms", float("nan"))),
            "truth_total_r1b_choice": float(wr["scores"].get("truth_h5_v1", {}).get("total_cost_ms", float("nan"))),
        })
    flips.sort(key=lambda row: -(row["truth_total_r1b_choice"] - row["truth_total_j3_choice"])
               if row["truth_total_r1b_choice"] == row["truth_total_r1b_choice"] else 0.0)
    with (args.output_dir / "report3_top_flips.jsonl").open("w", encoding="utf-8") as handle:
        for row in flips[:200]:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    report = {
        "decisions": len(decisions),
        "method_ids": method_ids,
        "observability_gates": gates,
        "top1_agreement_with_truth": {m: truth_top1[m] / max(1, len(decisions)) for m in method_ids},
        "j3_vs_r1b_top1_disagreements": len(flips),
        "j3_vs_r1b_disagreement_rate": len(flips) / max(1, len(decisions)),
    }
    (args.output_dir / "report_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
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
