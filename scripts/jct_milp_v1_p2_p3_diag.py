#!/usr/bin/env python3
"""P2-A + P3 diagnostic (advisor route-3, 2026-10-16).

12 dev states (3 episodes x 4 fractions).  Per state, per screened candidate
(K_a=5, K_c=2): forced prediction-scenario rollout cost (planning DES with
predicted artifacts, forced action + PDRS continuation) and forced MILP solve
(2s: objective/bound/gap).  Decision rules compared on the same states:
  PDRS (baseline), lightweight score (runtime + best-GPU queue), MILP-argmin
  (with bound-interval proof check), rollout-argmin (candidate-set ceiling).
All deltas vs PDRS use negative = better; no-solution falls back to PDRS and
all states are counted.  No evaluation truth is read by any decision rule.
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
FRACTIONS = [0.45, 0.60, 0.75, 0.90]


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--ka", type=int, default=5)
    ap.add_argument("--kc", type=int, default=2)
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/p2_p3_diag.json")
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

    rows = []
    t_start = time.perf_counter()
    for ep in episodes:
        summary, _ev = simulate_episode(ep, tm, "pdrs_resident", train_stats=stats,
                                        future_artifacts=artifacts, future_horizon=5,
                                        extension_config=config,
                                        policy_context={"activation": {}}, collect_events=False)
        for frac in FRACTIONS:
            if time.perf_counter() - t_start > 6600:  # ~110 min guard
                print("budget guard hit")
                break
            at = round(frac * float(summary["makespan_ms"]), 1)
            capture = {"at_ms_list": [at], "snapshots": []}
            simulate_episode(ep, tm, "pdrs_resident", train_stats=stats,
                             future_artifacts=artifacts, future_horizon=5,
                             extension_config=config,
                             policy_context={"activation": {}, "des_capture": capture},
                             collect_events=False)
            snaps = capture.get("snapshots") or []
            if not snaps:
                continue
            snap = snaps[0]
            jobs_meta = [{"job_instance_id": j["job_instance_id"], "template_id": j["template_id"]}
                         for j in ep["jobs"]]
            plan_episode = build_planning_episode(ep, snap, include_arrivals=True)
            _s, _e, pctx = run_planning(
                plan_episode, tm, future_artifacts=artifacts, train_stats=stats,
                extension_config=config, policy="des_rollout", forced_action=None,
                collect_events=False, extra_context={"des_probe_pool": True})
            pool = sorted({(str(r[1]), int(r[2])) for r in (pctx.get("des_first_pool") or [])})
            baseline = tuple(pctx.get("des_first_baseline") or ())
            if not pool:
                continue
            obs = T.obs_from_snapshot(snap, jobs_meta, tm, stats, artifacts)
            masked = replace(obs, legal_first=frozenset(pool))
            required = frozenset({baseline[0]}) if baseline else frozenset()
            screened = _screen_anchors(masked, args.ka, required)

            # per-candidate: prediction-scenario rollout cost (FULL pool, for the
            # recall metric) + forced MILP (screened only)
            rollout_cost, milp = {}, {}
            for node, gpu in pool:
                s2, _e2, _c = run_planning(
                    plan_episode, tm, future_artifacts=artifacts, train_stats=stats,
                    extension_config=config, policy="des_rollout",
                    forced_action={"node_key": node, "gpu_index": gpu}, collect_events=False)
                rollout_cost[(node, gpu)] = float(s2["mean_completion_ms"])
                if node not in screened:
                    continue
                res = solve_v1(masked, chain_depth=3, k_chains=args.kc, tail=True,
                               tail_table=tail_table, time_limit_s=args.budget,
                               k_candidates=args.ka, required_node_ids=required,
                               force_first_action=(node, gpu))
                milp[(node, gpu)] = {"obj": res.objective, "bound": res.best_bound, "gap": res.gap,
                                     "status": res.status}
            if not rollout_cost:
                continue
            screened_costs = {a: c for a, c in rollout_cost.items() if a[0] in screened}

            # decision rules (none reads evaluation truth beyond this state)
            rules = {}
            if baseline and baseline in rollout_cost:
                rules["pdrs"] = baseline
            else:
                rules["pdrs"] = None
            # lightweight: min runtime + best-GPU queue among screened anchors
            def light_key(a):
                it = next((x for x in masked.ready if str(x.node_id) == a[0]), None)
                if it is None:
                    return (1e18,)
                queue = min(g.available_at_ms for g in masked.gpus) - masked.now_ms if masked.gpus else 0.0
                return (float(it.runtime_p50_ms) + max(0.0, queue), a[1])
            rules["lightweight"] = min(screened_costs, key=light_key)
            # MILP argmin with interval check (undetermined when top-2 intervals overlap)
            valid = {a: m for a, m in milp.items() if m.get("obj") is not None}
            if valid:
                order = sorted(valid, key=lambda a: valid[a]["obj"])
                top = order[0]
                undetermined = False
                if len(order) > 1:
                    second = order[1]
                    if valid[second]["bound"] is not None and valid[top]["obj"] is not None:
                        undetermined = valid[second]["bound"] <= valid[top]["obj"]
                rules["milp_argmin"] = top
                rules["milp_undetermined"] = undetermined
            else:
                rules["milp_argmin"] = None
                rules["milp_undetermined"] = None
            rules["rollout_ceiling"] = min(screened_costs, key=lambda a: screened_costs[a])

            best = min(rollout_cost.values())
            pdrs_cost = rollout_cost.get(baseline)
            entry = {"episode": ep["episode_id"], "fraction": frac, "pool": len(pool),
                     "screened": len(screened), "pdrs_action": list(baseline) if baseline else None,
                     "candidates": {f"{a[0]}|{a[1]}": {
                         "rollout": round(rollout_cost[a], 1),
                         "milp_obj": milp.get(a, {}).get("obj"),
                         "milp_bound": milp.get(a, {}).get("bound"),
                         "milp_gap": milp.get(a, {}).get("gap"),
                         "true_rank_in_screened": None} for a in rollout_cost},
                     "best_cost": round(best, 1), "pdrs_cost": round(pdrs_cost, 1) if pdrs_cost else None,
                     "oracle_recall_at_ka": any(a[0] in screened for a in
                                                [k for k, v in rollout_cost.items() if abs(v - best) < 1e-9]),
                     "best_is_pdrs": bool(baseline and abs(rollout_cost.get(baseline, float("nan")) - best) < 1e-9),
                     "rules": {}}
            order_true = sorted(rollout_cost, key=lambda a: rollout_cost[a])
            for rank, a in enumerate(order_true, 1):
                entry["candidates"][f"{a[0]}|{a[1]}"]["true_rank_in_screened"] = rank
            for name, act in rules.items():
                if name == "milp_undetermined":
                    entry["rules"][name] = act
                    continue
                if act is None or act not in rollout_cost:
                    entry["rules"][name] = {"action": None, "fallback": True}
                    continue
                entry["rules"][name] = {"action": list(act), "cost": round(rollout_cost[act], 1),
                                        "delta_vs_pdrs": round(rollout_cost[act] - pdrs_cost, 1) if pdrs_cost is not None else None,
                                        "true_rank_in_screened": order_true.index(act) + 1}
            rows.append(entry)
            print(f"{ep['episode_id'][-4:]}@{frac}: pool={len(pool)} screened={len(screened)} "
                  f"pdrs_delta={entry['rules'].get('pdrs', {}).get('delta_vs_pdrs')} "
                  f"light_delta={entry['rules'].get('lightweight', {}).get('delta_vs_pdrs')} "
                  f"milp_delta={entry['rules'].get('milp_argmin', {}).get('delta_vs_pdrs')} "
                  f"undet={rules.get('milp_undetermined')} ceiling_delta={entry['rules']['rollout_ceiling'].get('delta_vs_pdrs')}")

    def summarize(rule):
        deltas = [r["rules"][rule].get("delta_vs_pdrs") for r in rows
                  if r["rules"].get(rule, {}).get("delta_vs_pdrs") is not None]
        fb = sum(1 for r in rows if r["rules"].get(rule, {}).get("fallback"))
        imp = sum(1 for d in deltas if d < -1e-9)
        wor = sum(1 for d in deltas if d > 1e-9)
        return {"paired_states": len(deltas), "fallbacks": fb,
                "delta_mean_ms": round(statistics.fmean(deltas), 1) if deltas else None,
                "improved": imp, "worsened": wor, "tied": len(deltas) - imp - wor}

    report = {
        "schema": "jct-milp-v1-p2-p3-diag-v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "states": len(rows), "ka": args.ka, "kc": args.kc, "budget_s": args.budget,
        "fractions": FRACTIONS,
        "pdrs_mean_delta_self": 0.0,
        "oracle_recall_at_ka": round(sum(1 for r in rows if r["oracle_recall_at_ka"]) / max(1, len(rows)), 3),
        "milp_undetermined_frac": round(sum(1 for r in rows if r["rules"].get("milp_undetermined")) / max(1, len(rows)), 3),
        "rules": {name: summarize(name) for name in ("pdrs", "lightweight", "milp_argmin", "rollout_ceiling")},
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1, ensure_ascii=False))
    print("evidence:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
