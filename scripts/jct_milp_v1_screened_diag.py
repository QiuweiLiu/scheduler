#!/usr/bin/env python3
"""JCT-MILP-v1 screened-vs-unscreened small-scale diagnostic (advisor 2026-10-16).

On deep-queue states of the formal workload (r150_b15 dev source):
  - probe the true legal pool + PDRS first action;
  - true cost per legal action via forced rollouts (PDRS continuation);
  - unscreened MILP  (k_candidates=None, K_c=3, d=3, tail, 2s);
  - screened MILP    (k_candidates=5, K_c=2, d=3, tail, 2s, PDRS anchor kept);
  - screened MILP    (same, 0.5s online budget).
Deliverables (advisor): OracleRecall@K_a, CaptureCeiling@K_a, variable/constraint
reduction, first-feasible + full-decision wall clock, true regret on same states.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import milp_v0_tests as T  # noqa: E402
import milp_v0_performance as P  # noqa: E402
from tracing.analysis.des_rollout import build_planning_episode, run_planning  # noqa: E402
from tracing.analysis.profile_contract import apply_real_workload_profile_contract  # noqa: E402
from tracing.analysis.workload_v02_simulator import read_jsonl, simulate_episode  # noqa: E402
from tracing.scheduling.milp_v0 import _screen_anchors, solve_v1  # noqa: E402

FORMAL = ROOT / "experiments/EXP-20261014_burst64_v1/high_contention"
TAIL_TABLE = ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/tail_estimator_train_table.json"


def load_tail_table():
    data = json.loads(TAIL_TABLE.read_text(encoding="utf-8"))
    meta = {}
    for line in (ROOT / "results/processed/r7_workload_v041_ontology_no_run_container"
                 / "job_templates_r7_v041.jsonl").read_text(encoding="utf-8").strip().split("\n"):
        row = json.loads(line)
        if str(row.get("split")) != "train":
            continue
        for node in row.get("nodes") or []:
            if node.get("execution_lane") == "gpu" and node.get("model_id"):
                meta.setdefault(str(node["model_id"]), str(row.get("model_stack_id") or ""))
    return {"table": data["table"], "model_to_stack": meta}


def capture_deep_states(episodes, tm, stats, artifacts, config, fractions):
    out = []
    for ep in episodes:
        summary, _ev = simulate_episode(ep, tm, "pdrs_resident", train_stats=stats,
                                        future_artifacts=artifacts, future_horizon=5,
                                        extension_config=config,
                                        policy_context={"activation": {}}, collect_events=False)
        times = [round(f * float(summary["makespan_ms"]), 1) for f in fractions]
        capture = {"at_ms_list": list(times), "snapshots": []}
        simulate_episode(ep, tm, "pdrs_resident", train_stats=stats,
                         future_artifacts=artifacts, future_horizon=5,
                         extension_config=config,
                         policy_context={"activation": {}, "des_capture": capture}, collect_events=False)
        jobs_meta = [{"job_instance_id": j["job_instance_id"], "template_id": j["template_id"]}
                     for j in ep["jobs"]]
        for snap in capture.get("snapshots") or []:
            out.append((ep, jobs_meta, snap))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--states", type=int, default=12)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.50, 0.70, 0.90])
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--online-budget", type=float, default=0.5)
    ap.add_argument("--ka", type=int, default=5)
    ap.add_argument("--kc", type=int, default=2)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/screened_diag_r150b15.json")
    args = ap.parse_args()

    config_path = ROOT / "experiments/EXP-20261012_milp_v0/artifacts/serial_prefetch_v1_extension.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _serial, templates, _n, stats, artifacts = P.load_split("confirm", 1, config_path)
    tm = {k: replace(v, split="train") for k, v in templates.items()}
    episodes = read_jsonl(FORMAL / "workload_burst64_r150_b15_train.jsonl")[: args.episodes]
    for row in episodes:
        row["split"] = "train"
    tm, episodes = apply_real_workload_profile_contract(tm, episodes, config)
    tail_table = load_tail_table()

    states = capture_deep_states(episodes, tm, stats, artifacts, config, args.fractions)
    rows = []
    for ep, jobs_meta, snap in states:
        if len(rows) >= args.states:
            break
        plan_episode = build_planning_episode(ep, snap, include_arrivals=True)
        _s, _e, pctx = run_planning(
            plan_episode, tm, future_artifacts=artifacts, train_stats=stats,
            extension_config=config, policy="des_rollout", forced_action=None,
            collect_events=False, extra_context={"des_probe_pool": True})
        if abs(float((pctx.get("des_first_state") or {}).get("now_ms", -1)) - float(snap["now_ms"])) > 1e-6:
            continue
        pool = sorted({(str(r[1]), int(r[2])) for r in (pctx.get("des_first_pool") or [])})
        if not pool:
            continue
        baseline = tuple(pctx.get("des_first_baseline") or ())
        obs = T.obs_from_snapshot(snap, jobs_meta, tm, stats, artifacts)
        masked = replace(obs, legal_first=frozenset(pool))
        required = frozenset({baseline[0]}) if baseline else frozenset()
        screened_ids = _screen_anchors(masked, args.ka, required)

        costs = {}
        for node, gpu in pool:
            summary, _ev, _c = run_planning(
                plan_episode, tm, future_artifacts=artifacts, train_stats=stats,
                extension_config=config, policy="des_rollout",
                forced_action={"node_key": node, "gpu_index": gpu}, collect_events=False)
            costs[(node, gpu)] = float(summary["mean_completion_ms"])
        best = min(costs.values())
        pdrs_cost = costs.get(baseline)

        def run_arm(label, fn, tl, dump):
            t0 = time.perf_counter()
            res = fn(tl)
            wall = (time.perf_counter() - t0) * 1000.0
            choice = tuple(res.first_action) if (not res.fallback and res.first_action) else None
            entry = {"label": label, "status": res.status, "fallback": bool(res.fallback),
                     "action": list(choice) if choice else None, "wall_ms": round(wall, 1),
                     "n_vars": dump.get("n_vars"), "n_rows": len(dump.get("rows") or []),
                     "n_nodes": len(dump.get("nodes") or []), "has_incumbent": bool(dump.get("values"))}
            if choice and choice in costs:
                entry["true_cost"] = round(costs[choice], 1)
                entry["regret"] = round(costs[choice] - best, 1)
            return entry

    # (note: kept simple; arms below re-solve with debug dumps)

        dumps: dict = {}
        arms = {}
        arms["unscreened_2s"] = run_arm(
            "unscreened_2s",
            lambda tl: solve_v1(masked, chain_depth=3, k_chains=3, tail=True,
                                tail_table=tail_table, time_limit_s=tl, debug_dump=dumps.setdefault("unscreened_2s", {})),
            args.budget, dumps.setdefault("unscreened_2s", {}))
        arms["screened_2s"] = run_arm(
            "screened_2s",
            lambda tl: solve_v1(masked, chain_depth=3, k_chains=args.kc, tail=True,
                                tail_table=tail_table, time_limit_s=tl,
                                k_candidates=args.ka, required_node_ids=required,
                                debug_dump=dumps.setdefault("screened_2s", {})),
            args.budget, dumps.setdefault("screened_2s", {}))
        arms["screened_online"] = run_arm(
            "screened_online",
            lambda tl: solve_v1(masked, chain_depth=3, k_chains=args.kc, tail=True,
                                tail_table=tail_table, time_limit_s=tl,
                                k_candidates=args.ka, required_node_ids=required,
                                debug_dump=dumps.setdefault("screened_online", {})),
            args.online_budget, dumps.setdefault("screened_online", {}))

        # OracleRecall@Ka: is the true-best action's anchor among the screened set?
        best_actions = [a for a, c in costs.items() if abs(c - best) < 1e-9]
        recall = any(a[0] in screened_ids for a in best_actions)
        # CaptureCeiling@Ka: share of PDRS headroom reachable within the screened set
        pdrs_headroom = (pdrs_cost - best) if pdrs_cost is not None else None
        screened_best = min((c for a, c in costs.items() if a[0] in screened_ids), default=None)
        capture = (None if (pdrs_headroom is None or screened_best is None or pdrs_headroom <= 1e-9)
                   else round((pdrs_cost - screened_best) / pdrs_headroom, 3))
        rows.append({
            "episode": ep["episode_id"], "now_ms": snap["now_ms"], "pool": len(pool),
            "screened_anchors": len(screened_ids), "pdrs_action": list(baseline) if baseline else None,
            "pdrs_regret": round(pdrs_cost - best, 1) if pdrs_cost is not None else None,
            "oracle_recall_at_ka": bool(recall), "capture_ceiling_at_ka": capture,
            "true_best": round(best, 1), "arms": arms,
        })
        print(f"state {len(rows)}: pool={len(pool)} screened={len(screened_ids)} recall={recall} "
              f"capture={capture} | unscreened: {arms['unscreened_2s']['status']} "
              f"{arms['unscreened_2s']['n_vars']}v/{arms['unscreened_2s']['n_rows']}r "
              f"{arms['unscreened_2s']['wall_ms']}ms | screened: {arms['screened_2s']['status']} "
              f"{arms['screened_2s']['n_vars']}v/{arms['screened_2s']['n_rows']}r "
              f"{arms['screened_2s']['wall_ms']}ms | online: {arms['screened_online']['status']} "
              f"{arms['screened_online']['wall_ms']}ms")

    def agg(key):
        entries = [r["arms"][key] for r in rows]
        solved = [e for e in entries if e.get("regret") is not None]
        paired = [(r["arms"][key].get("regret"), r["pdrs_regret"]) for r in rows
                  if r["arms"][key].get("regret") is not None and r["pdrs_regret"] is not None]
        return {
            "states": len(entries),
            "with_action": sum(1 for e in entries if e.get("action")),
            "incumbent_frac": round(sum(1 for e in entries if e.get("has_incumbent")) / max(1, len(entries)), 3),
            "mean_regret_ms": round(statistics.fmean([e["regret"] for e in solved]), 1) if solved else None,
            "delta_vs_pdrs_ms": round(statistics.fmean([a - b for a, b in paired]), 1) if paired else None,
            "paired_states": len(paired),
            "n_vars_p50": round(statistics.median([e["n_vars"] for e in entries if e.get("n_vars")]), 0) if any(e.get("n_vars") for e in entries) else None,
            "n_rows_p50": round(statistics.median([e["n_rows"] for e in entries if e.get("n_rows")]), 0) if any(e.get("n_rows") for e in entries) else None,
            "wall_ms_p50": round(statistics.median([e["wall_ms"] for e in entries]), 1),
        }

    recalls = [r["oracle_recall_at_ka"] for r in rows]
    captures = [r["capture_ceiling_at_ka"] for r in rows if r["capture_ceiling_at_ka"] is not None]
    report = {
        "schema": "jct-milp-v1-screened-diag-v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "workload": "r150_b15 dev (train source)", "ka": args.ka, "kc": args.kc,
        "states": len(rows),
        "oracle_recall_at_ka": round(sum(1 for x in recalls if x) / max(1, len(recalls)), 3),
        "capture_ceiling_at_ka_mean": round(statistics.fmean(captures), 3) if captures else None,
        "pdrs_mean_regret_ms": round(statistics.fmean([r["pdrs_regret"] for r in rows if r["pdrs_regret"] is not None]), 1) if any(r["pdrs_regret"] is not None for r in rows) else None,
        "arms": {k: agg(k) for k in ("unscreened_2s", "screened_2s", "screened_online")},
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1, ensure_ascii=False))
    print("evidence:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
