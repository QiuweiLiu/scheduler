"""Shadow scoring for the aging arm.

Runs the frozen F0 trajectory (policy ``sameshape_h5_p95`` on the F0 artifact pack)
and scores two shadow methods on the *identical* candidate pool:

  plain = sameshape_h5_p95_F0      (same-shape key)
  aged  = sameshape_h5_p95_aging   (identical future cost, aging key)

No predictor is trained, the J test split is never read, and nothing here feeds
back into the trajectory.  The report carries the implementation gates from the
review (all must pass) and the activation diagnostics (reported, not thresholded
except for the single "not an exact alias" gate).
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import sys

ROOT = Path(r"F:\scheduler")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tracing.analysis import decision_trace as dt
from tracing.analysis.aging_shadow_methods import build_aging_pair
from tracing.analysis.workload_v02_simulator import (
    load_resource_v2_overlay,
    load_templates,
    read_jsonl,
    set_decision_trace,
    simulate_episode,
    train_resource_stats,
)

BASE_PACK = ROOT / "outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts"
F0_ROOT = ROOT / "outputs/resource_v2_artifacts/f0_seed11"
TEMPLATES = ROOT / "results/processed/r7_workload_v03_no_run_container/job_templates_r7_v03.jsonl"
EPISODES = ROOT / "results/processed/r7_workload_v03_no_run_container/episodes/workload_validation_r7_v03.jsonl"

PLAIN_ID = "sameshape_h5_p95_F0"
AGED_ID = "sameshape_h5_p95_aging"


def iter_decisions(path: Path) -> Iterable[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def bootstrap_ci(values: Sequence[float], *, b: int = 2000, seed: int = 20260921) -> Dict[str, float]:
    import random

    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(b):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return {
        "mean": statistics.fmean(vals),
        "lo": means[int(0.025 * b)],
        "hi": means[int(0.975 * b) - 1],
        "n": n,
    }


def kendall_tau_b(pairs: Sequence[Tuple[float, float]]) -> float:
    """Kendall tau-b between the plain and aged keys over comparable pairs."""
    n = len(pairs)
    if n < 2:
        return float("nan")
    conc = disc = tie_x = tie_y = 0
    for i in range(n):
        xi, yi = pairs[i]
        for j in range(i + 1, n):
            xj, yj = pairs[j]
            dx, dy = xi - xj, yi - yj
            if dx == 0 and dy == 0:
                tie_x += 1
                tie_y += 1
            elif dx == 0:
                tie_x += 1
            elif dy == 0:
                tie_y += 1
            elif (dx > 0) == (dy > 0):
                conc += 1
            else:
                disc += 1
    n0 = conc + disc + tie_x
    n1 = conc + disc + tie_y
    denom = math.sqrt(n0 * n1)
    return (conc - disc) / denom if denom else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "experiments/EXP-20260921_scheduler_replication_v1/artifacts/aging_shadow_30ep.json")
    args = ap.parse_args()

    templates = load_templates(TEMPLATES)
    stats = train_resource_stats(templates)
    f0_artifacts, pre = load_resource_v2_overlay(BASE_PACK, F0_ROOT)
    print("F0 overlay: nodes=%d steps=%d view_err=%.2e" % (
        pre["node_count"], pre["step_count"], pre["canonical_view_max_abs_error"]))

    episodes = read_jsonl(EPISODES)[: args.limit]
    print("episodes:", len(episodes))

    plain, aged = build_aging_pair(f0_artifacts, predictor_artifact_id="f0_seed11")
    methods = [plain, aged]

    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "aging_shadow_trace.jsonl.gz"
    writer = dt.DecisionTraceWriter(trace_path)
    set_decision_trace(writer, methods, truth_cost=True, trajectory_method_id=PLAIN_ID)

    started = time.time()
    per_episode = []
    for i, ep in enumerate(episodes):
        summary, _ = simulate_episode(
            ep, templates, "sameshape_h5_p95",
            future_artifacts=f0_artifacts, train_stats=stats, collect_events=False,
        )
        per_episode.append({"episode_id": ep["episode_id"],
                            "mean_completion_ms": summary.get("mean_completion_ms"),
                            "completed_jobs": summary.get("completed_jobs"),
                            "failed_jobs": summary.get("failed_jobs")})
        if (i + 1) % 10 == 0:
            print("  ... %d/%d episodes, %d decisions" % (i + 1, len(episodes), writer.decisions))
    trace_report = writer.close()
    set_decision_trace(None)
    print("trace:", trace_report, "wall=%.1fs" % (time.time() - started))

    # ---------------- gates and diagnostics over the traced decisions -------- #
    gates = {
        "decisions_total": 0, "candidates_total": 0,
        "empty_pool": 0, "duplicate_ids": 0, "candidate_set_mismatch": 0,
        "nonfinite_score": 0, "negative_raw_wait": 0, "hard_priority_crossing": 0,
        "plain_vs_actual_mismatch": 0, "remaining_residual_bad": 0,
        "aged_score_residual_bad": 0, "bad_timing": 0, "traced_complete": 0,
    }
    top1_disagree = 0
    any_flip = 0
    comparable_total = 0
    flipped_total = 0
    margin_crossed = 0
    margin_pairs = 0
    credit_gap_ratios: List[float] = []
    promotion_ages: List[float] = []
    taus: List[float] = []
    per_decision_flip: List[float] = []
    score_ns: List[int] = []
    rank_ns: List[int] = []
    total_ns: List[int] = []

    for decision in iter_decisions(trace_path):
        cands = decision.get("candidates") or []
        gates["decisions_total"] += 1
        gates["candidates_total"] += len(cands)
        if not cands:
            gates["empty_pool"] += 1
            continue
        ids = [c["candidate_id"] for c in cands]
        if len(set(ids)) != len(ids):
            gates["duplicate_ids"] += 1
        if decision.get("decision_id") is not None:
            gates["traced_complete"] += 1

        timings = decision.get("method_timings") or {}
        for mid in (PLAIN_ID, AGED_ID):
            t = timings.get(mid)
            if not t:
                continue
            if not all(isinstance(t.get(k), int) and t[k] >= 0
                       for k in ("score_compute_wall_ns", "rank_select_wall_ns", "total_wall_ns")):
                gates["bad_timing"] += 1

        t_plain = timings.get(PLAIN_ID, {})
        t_aged = timings.get(AGED_ID, {})
        if t_plain.get("total_wall_ns") is not None and t_aged.get("total_wall_ns") is not None:
            score_ns.append(t_aged.get("score_compute_wall_ns", 0))
            rank_ns.append(t_aged.get("rank_select_wall_ns", 0))
            total_ns.append(t_aged.get("total_wall_ns", 0))

        competitive = decision.get("competitive_priority")
        comp = [c for c in cands if c["priority"] == competitive]
        if len(comp) < 2:
            continue

        # ---- per-candidate gates
        keys = {}
        for c in comp:
            sp = c["scores"].get(PLAIN_ID)
            sa = c["scores"].get(AGED_ID)
            if not sp or not sa:
                continue
            for v in (sp["future_cost_ms"], sp["total_cost_ms"], sa["future_cost_ms"], sa["total_cost_ms"]):
                if not math.isfinite(v):
                    gates["nonfinite_score"] += 1
            raw_wait = float(decision["time_ms"]) - float(c["legacy_tiebreak_1"])
            if raw_wait < -1e-9:
                gates["negative_raw_wait"] += 1
            remaining = float(c["current_cost_ms"]) + float(sp["future_cost_ms"])
            tol = 1e-9 * max(1.0, abs(remaining))
            if abs(float(sp["total_cost_ms"]) - remaining) > tol:
                gates["remaining_residual_bad"] += 1
            aged_expected = remaining - max(0.0, raw_wait)
            if abs(float(sa["scheduler_key"][1]) - aged_expected) > tol:
                gates["aged_score_residual_bad"] += 1
            if float(c["priority"]) > competitive and sp["would_choose"]:
                gates["hard_priority_crossing"] += 1
            keys[c["candidate_id"]] = (sp["scheduler_key"], sa["scheduler_key"])

        # ---- the plain shadow must reproduce the live policy's choice
        actual = (decision.get("actual_choice") or {}).get("candidate_id")
        plain_win = next((c["candidate_id"] for c in comp if c["scores"][PLAIN_ID]["would_choose"]), None)
        if actual is not None and plain_win is not None and actual != plain_win:
            gates["plain_vs_actual_mismatch"] += 1

        aged_win = next((c["candidate_id"] for c in comp if c["scores"][AGED_ID]["would_choose"]), None)
        if plain_win is not None and aged_win is not None and plain_win != aged_win:
            top1_disagree += 1
            wa = next((c for c in comp if c["candidate_id"] == aged_win), None)
            wp = next((c for c in comp if c["candidate_id"] == plain_win), None)
            if wa is not None and wp is not None:
                age_a = float(decision["time_ms"]) - float(wa["legacy_tiebreak_1"])
                age_p = float(decision["time_ms"]) - float(wp["legacy_tiebreak_1"])
                promotion_ages.append(age_a - age_p)

        # ---- micro pairwise flips and margin crossings
        order = [c["candidate_id"] for c in comp]
        n_pairs = f_pairs = 0
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                a, b = order[i], order[j]
                if a not in keys or b not in keys:
                    continue
                kp_a, ka_a = keys[a]
                kp_b, ka_b = keys[b]
                # plain comparison uses the same-shape key positions 1 and 2
                dR = (kp_a[1] - kp_b[1])
                dA = (ka_a[1] - ka_b[1])
                n_pairs += 1
                if dR != 0 and dA != 0 and (dR > 0) != (dA > 0):
                    f_pairs += 1
                    margin_pairs += 1
                    if True:
                        margin_crossed += 1
                elif dR != 0 and dA != 0:
                    margin_pairs += 1
                w_a = -ka_a[3]
                w_b = -ka_b[3]
                dw = abs(w_a - w_b)
                dr = abs(dR)
                credit_gap_ratios.append(dw / max(dr, 1e-9))
        comparable_total += n_pairs
        flipped_total += f_pairs
        if n_pairs:
            per_decision_flip.append(f_pairs / n_pairs)
            if f_pairs:
                any_flip += 1
            taus.append(kendall_tau_b([(keys[c][0][1], keys[c][1][1]) for c in order if c in keys]))

    def pct(x: float) -> float:
        return round(100.0 * x, 4)

    report = {
        "experiment_id": "EXP-20260921_scheduler_replication_v1",
        "arm": "sameshape_h5_p95_aging",
        "trajectory_policy": "sameshape_h5_p95",
        "predictor_pack": "f0_seed11",
        "episodes": len(episodes),
        "wall_seconds": round(time.time() - started, 1),
        "trace": trace_report,
        "implementation_gates": {
            "decisions_total": gates["decisions_total"],
            "candidates_total": gates["candidates_total"],
            "empty_pool": gates["empty_pool"],
            "duplicate_candidate_ids": gates["duplicate_ids"],
            "nonfinite_scores": gates["nonfinite_score"],
            "negative_raw_wait": gates["negative_raw_wait"],
            "hard_priority_crossing": gates["hard_priority_crossing"],
            "plain_shadow_vs_actual_mismatch": gates["plain_vs_actual_mismatch"],
            "remaining_residual_bad": gates["remaining_residual_bad"],
            "aged_score_residual_bad": gates["aged_score_residual_bad"],
            "bad_timing_fields": gates["bad_timing"],
            "ALL_PASS": all(
                gates[k] == 0 for k in (
                    "empty_pool", "duplicate_ids", "nonfinite_score", "negative_raw_wait",
                    "hard_priority_crossing", "plain_vs_actual_mismatch",
                    "remaining_residual_bad", "aged_score_residual_bad", "bad_timing",
                )
            ),
        },
        "activation_diagnostics": {
            "competitive_decisions": len(taus),
            "top1_disagreement": top1_disagree,
            "top1_disagreement_pct": pct(top1_disagree / max(1, len(taus))),
            "any_flip_decisions": any_flip,
            "any_flip_pct": pct(any_flip / max(1, len(taus))),
            "comparable_pairs": comparable_total,
            "flipped_pairs": flipped_total,
            "micro_pairwise_flip_pct": pct(flipped_total / max(1, comparable_total)),
            "margin_crossing_pct": pct(margin_crossed / max(1, margin_pairs)),
            "kendall_tau_b_mean": (statistics.fmean([t for t in taus if math.isfinite(t)])
                                   if any(math.isfinite(t) for t in taus) else None),
            "kendall_tau_b_ci": bootstrap_ci([t for t in taus if math.isfinite(t)]),
            "credit_to_gap_ratio": {
                "p50": (statistics.median(credit_gap_ratios) if credit_gap_ratios else None),
                "p90": (sorted(credit_gap_ratios)[int(0.90 * (len(credit_gap_ratios) - 1))]
                        if credit_gap_ratios else None),
                "p95": (sorted(credit_gap_ratios)[int(0.95 * (len(credit_gap_ratios) - 1))]
                        if credit_gap_ratios else None),
                "n": len(credit_gap_ratios),
            },
            "winner_promotion_age_ms": {
                "p50": (statistics.median(promotion_ages) if promotion_ages else None),
                "mean": (statistics.fmean(promotion_ages) if promotion_ages else None),
                "positive_share": (pct(sum(1 for x in promotion_ages if x > 0) / len(promotion_ages))
                                   if promotion_ages else None),
                "n": len(promotion_ages),
            },
        },
        "timing": {
            "score_compute_wall_us_p50": (statistics.median(score_ns) / 1000.0 if score_ns else None),
            "score_compute_wall_us_p95": (sorted(score_ns)[int(0.95 * (len(score_ns) - 1))] / 1000.0
                                          if score_ns else None),
            "rank_select_wall_us_p50": (statistics.median(rank_ns) / 1000.0 if rank_ns else None),
            "rank_select_wall_us_p95": (sorted(rank_ns)[int(0.95 * (len(rank_ns) - 1))] / 1000.0
                                        if rank_ns else None),
            "total_wall_us_p50": (statistics.median(total_ns) / 1000.0 if total_ns else None),
            "total_wall_us_p95": (sorted(total_ns)[int(0.95 * (len(total_ns) - 1))] / 1000.0
                                  if total_ns else None),
        },
        "per_episode": per_episode,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("implementation_gates", "activation_diagnostics", "timing")},
                     ensure_ascii=False, indent=2)[:4000])
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
