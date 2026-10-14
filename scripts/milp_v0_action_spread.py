#!/usr/bin/env python3
"""Legal-action distribution + REAL JCT spread of the choice.

Part 1: distribution of legal action counts per decision (1 / 2 / >2).
Part 2: for states with >= 2 legal actions, EXECUTE each legal action in the
DES (forced first action, main-line continuation) and measure the episode JCT;
spread = worst - best.  Both passes: serial_v0 (formal env) and full v7.

Read-only: planning worlds only, no episode side effects.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import milp_v0_tests as T  # noqa: E402
from tracing.analysis.des_rollout import build_planning_episode, run_planning  # noqa: E402

V7 = ROOT / "experiments/EXP-20261004_real_strata_calibration_v1/artifacts/substrate_extension_v7.json"
EPISODES = 20
CAP = 4
MAX_ROLLOUT_STATES = 45


def run_pass(name, config, episodes, templates, stats, artifacts, fixtures):
    states = T.capture_states(episodes, templates, stats, artifacts, config, cap=CAP)
    dist = {1: 0, 2: 0, "gt2": 0}
    checked = 0
    rows = []
    for ep, jobs_meta, snap in states:
        if len(rows) >= MAX_ROLLOUT_STATES:
            break
        plan_episode = build_planning_episode(ep, snap, include_arrivals=True)
        _s, _e, pctx = run_planning(
            plan_episode, templates, future_artifacts=artifacts, train_stats=stats,
            extension_config=config, policy="des_rollout", forced_action=None,
            collect_events=False, extra_context={"des_probe_pool": True})
        probe_now = float((pctx.get("des_first_state") or {}).get("now_ms", -1.0))
        if abs(probe_now - float(snap["now_ms"])) > 1e-6:
            continue
        pool = sorted({(str(r[1]), int(r[2])) for r in (pctx.get("des_first_pool") or [])})
        checked += 1
        k = len(pool)
        dist[1 if k == 1 else (2 if k == 2 else "gt2")] += 1
        if k < 2:
            continue
        jcts = []
        for node, gpu in pool:
            summary, _events, _ctx = run_planning(
                plan_episode, templates, future_artifacts=artifacts, train_stats=stats,
                extension_config=config, policy="des_rollout",
                forced_action={"node_key": node, "gpu_index": gpu}, collect_events=False)
            jcts.append(float(summary["mean_completion_ms"]))
        best = min(jcts)
        worst = max(jcts)
        rows.append({
            "episode": ep["episode_id"], "now_ms": snap["now_ms"], "actions": k,
            "jct_best": round(best, 1), "jct_worst": round(worst, 1),
            "spread": round(worst - best, 1),
        })
    spreads = [r["spread"] for r in rows]
    return {
        "environment": name,
        "states_checked": checked,
        "dist": {"1": dist[1], "2": dist[2], "gt2": dist["gt2"],
                 "frac_1": round(dist[1] / checked, 3) if checked else None,
                 "frac_2": round(dist[2] / checked, 3) if checked else None,
                 "frac_gt2": round(dist["gt2"] / checked, 3) if checked else None},
        "rollout_states": len(rows),
        "spread_mean": round(statistics.fmean(spreads), 1) if spreads else None,
        "spread_median": round(statistics.median(spreads), 1) if spreads else None,
        "spread_p90": round(sorted(spreads)[int(0.9 * (len(spreads) - 1))], 1) if spreads else None,
        "spread_zero_states": sum(1 for s in spreads if s == 0),
        "details": rows,
    }


def main() -> int:
    import milp_v0_performance as P
    from tracing.analysis.profile_contract import apply_real_workload_profile_contract
    from tracing.analysis.workload_v02_simulator import (
        load_resource_v2_overlay, load_templates, read_jsonl, train_resource_stats)

    v7 = json.loads(V7.read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "data/manifests/validation_split_dev700_confirm300.json").read_text(encoding="utf-8"))
    wanted = set(manifest["confirm"])
    base_templates = load_templates(
        ROOT / "results/processed/r7_workload_v041_ontology_no_run_container/job_templates_r7_v041.jsonl",
        topology_view="causal_v3")
    by_id = {str(e.get("episode_id")): e for e in read_jsonl(
        ROOT / "results/processed/r7_workload_v03_no_run_container/episodes/workload_validation_r7_v03.jsonl")}
    base_episodes = [by_id[eid] for eid in sorted(wanted) if eid in by_id][:EPISODES]
    artifacts, _ = load_resource_v2_overlay(
        ROOT / "outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts",
        ROOT / "outputs/resource_v2_artifacts/f0_seed11")

    serial = json.loads(T.SERIAL.read_text(encoding="utf-8"))
    t_s, e_s = apply_real_workload_profile_contract(
        {k: v for k, v in base_templates.items()}, list(base_episodes), serial)
    stats_s = train_resource_stats({k: v for k, v in t_s.items() if v.split == "train"})
    report_serial = run_pass("serial_v0", serial, e_s, t_s, stats_s, artifacts, None)

    t_v, e_v = apply_real_workload_profile_contract(
        {k: v for k, v in base_templates.items()}, list(base_episodes), v7)
    stats_v = train_resource_stats({k: v for k, v in t_v.items() if v.split == "train"})
    report_v7 = run_pass("v7_full", v7, e_v, t_v, stats_v, artifacts, None)

    report = {"schema": "legal-action-dist-and-spread-v1",
              "episodes_per_pass": EPISODES, "cap": CAP,
              "serial_v0": report_serial, "v7_full": report_v7}
    out = ROOT / "experiments/EXP-20261012_milp_v0/artifacts/action_dist_and_spread.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for r in (report_serial, report_v7):
        print(f"[{r['environment']}] states={r['states_checked']} dist 1/2/>2 = "
              f"{r['dist']['frac_1']}/{r['dist']['frac_2']}/{r['dist']['frac_gt2']}")
        print(f"   rollout states={r['rollout_states']} | spread mean={r['spread_mean']} "
              f"median={r['spread_median']} p90={r['spread_p90']} zero={r['spread_zero_states']}")
    print("evidence:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
