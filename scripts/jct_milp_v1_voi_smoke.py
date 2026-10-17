#!/usr/bin/env python3
"""Gate 1 smoke (advisor 2026-10-16): compatible-future scenario construction.

Contract v0: for a state (episode E, snapshot at t) the PREFIX (in-flight jobs,
queues, GPU states, jobs arrived <= t) is fixed; the FUTURE (jobs arriving > t)
is replaced by a donor episode's post-t_D arrivals re-anchored at t (unique ids,
relative deadlines preserved).  The snapshot's arrival_heap is rebuilt to match.
All actions are evaluated in the SAME scenario (paired); the DES is
deterministic -> exact reproducibility expected.

Smoke: 1 state, 3 scenarios (original + 2 donors), 4 actions, reproducibility
check, cost table, timing, and a demo VOI (no-info vs scenario-known).
"""

from __future__ import annotations

import copy
import json
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

FORMAL = ROOT / "experiments/EXP-20261014_burst64_v1/high_contention"
OUT = ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/voi_smoke.json"
FRAC = 0.6


def makespan_and_snapshot(ep, tm, stats, artifacts, config, frac):
    summary, _ev = simulate_episode(ep, tm, "pdrs_resident", train_stats=stats,
                                    future_artifacts=artifacts, future_horizon=5,
                                    extension_config=config,
                                    policy_context={"activation": {}}, collect_events=False)
    at = round(frac * float(summary["makespan_ms"]), 1)
    capture = {"at_ms_list": [at], "snapshots": []}
    simulate_episode(ep, tm, "pdrs_resident", train_stats=stats,
                     future_artifacts=artifacts, future_horizon=5,
                     extension_config=config,
                     policy_context={"activation": {}, "des_capture": capture},
                     collect_events=False)
    return at, (capture.get("snapshots") or [None])[0]


def build_scenario(base_ep, snap, donor_ep, donor_t, tag):
    t = float(snap["now_ms"])
    keep = [j for j in base_ep["jobs"] if float(j["arrival_ms"]) <= t]
    donor_future = [j for j in donor_ep["jobs"] if float(j["arrival_ms"]) > donor_t]
    new_jobs = []
    for k, j in enumerate(donor_future):
        nj = dict(j)
        arr = t + (float(j["arrival_ms"]) - donor_t)
        dl_rel = float(j["deadline_ms"]) - float(j["arrival_ms"])
        nj["arrival_ms"] = arr
        nj["deadline_ms"] = arr + dl_rel
        nj["job_instance_id"] = f"{base_ep['episode_id']}::sc{tag}_job_{k:03d}"
        new_jobs.append(nj)
    sc = dict(base_ep)
    sc["jobs"] = keep + new_jobs
    sc["episode_id"] = f"{base_ep['episode_id']}::sc{tag}"
    sc_snap = copy.deepcopy(snap)
    sc_snap["arrival_heap"] = [[float(nj["arrival_ms"]), len(keep) + k] for k, nj in enumerate(new_jobs)]
    # drop injected-state rows that reference removed future jobs (donor jobs are
    # new ids -> they start fresh; prefix rows are kept as-is)
    keep_ids = {str(j["job_instance_id"]) for j in sc["jobs"]}
    sc_snap["jobs"] = [row for row in sc_snap.get("jobs") or []
                       if str(row["job_instance_id"]) in keep_ids]
    return sc, sc_snap


def main() -> int:
    config_path = ROOT / "experiments/EXP-20261012_milp_v0/artifacts/serial_prefetch_v1_extension.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _serial, templates, _n, stats, artifacts = P.load_split("confirm", 1, config_path)
    tm = {k: replace(v, split="train") for k, v in templates.items()}
    eps = read_jsonl(FORMAL / "workload_burst64_r150_b15_train.jsonl")[:3]
    for r in eps:
        r["split"] = "train"
    tm, eps = apply_real_workload_profile_contract(tm, eps, config)
    base = eps[0]

    t0 = time.perf_counter()
    t_base, snap = makespan_and_snapshot(base, tm, stats, artifacts, config, FRAC)
    print(f"state: {base['episode_id']}@{FRAC} now={t_base} | prefix jobs: "
          f"{sum(1 for j in base['jobs'] if float(j['arrival_ms']) <= t_base)}/64")

    scenarios = {"orig": (base, snap)}
    for k, donor in enumerate(eps[1:], 1):
        _td, dsnap = makespan_and_snapshot(donor, tm, stats, artifacts, config, FRAC)
        donor_t = float(dsnap["now_ms"])
        scenarios[f"donor{k}"] = build_scenario(base, snap, donor, donor_t, f"d{k}")
    print("scenarios:", {k: (len(v[0]['jobs']), float(v[1]['now_ms'])) for k, v in scenarios.items()})

    # candidate actions from the probe on the ORIGINAL scenario
    pe = build_planning_episode(base, snap, include_arrivals=True)
    _s, _e, pctx = run_planning(pe, tm, future_artifacts=artifacts, train_stats=stats,
                                extension_config=config, policy="des_rollout", forced_action=None,
                                collect_events=False, extra_context={"des_probe_pool": True})
    pool = sorted({(str(r[1]), int(r[2])) for r in (pctx.get("des_first_pool") or [])})
    baseline = tuple(pctx.get("des_first_baseline") or ())
    actions = []
    if baseline:
        actions.append(baseline)
    for a in pool:
        if a not in actions:
            actions.append(a)
        if len(actions) >= 4:
            break
    print(f"pool={len(pool)} | actions used: {len(actions)}")

    table = {}
    for sname, (sc_ep, sc_snap) in scenarios.items():
        plan_ep = build_planning_episode(sc_ep, sc_snap, include_arrivals=True)
        table[sname] = {}
        for a in actions:
            s2, _ev, _c = run_planning(plan_ep, tm, future_artifacts=artifacts, train_stats=stats,
                                       extension_config=config, policy="des_rollout",
                                       forced_action={"node_key": a[0], "gpu_index": a[1]},
                                       collect_events=False)
            table[sname][f"{a[0][:26]}|{a[1]}"] = round(float(s2["mean_completion_ms"]), 1)
    # reproducibility: rerun one (scenario, action) pair
    sc_ep, sc_snap = scenarios["orig"]
    plan_ep = build_planning_episode(sc_ep, sc_snap, include_arrivals=True)
    s3, _ev, _c = run_planning(plan_ep, tm, future_artifacts=artifacts, train_stats=stats,
                               extension_config=config, policy="des_rollout",
                               forced_action={"node_key": actions[0][0], "gpu_index": actions[0][1]},
                               collect_events=False)
    rerun = round(float(s3["mean_completion_ms"]), 1)
    first = table["orig"][f"{actions[0][0][:26]}|{actions[0][1]}"]
    reproducible = abs(rerun - first) < 1e-6

    # demo VOI: I0 (no info -> ONE action for all scenarios) vs scenario-known.
    # NOTE: average over SCENARIOS per ACTION (fixed earlier axis mix-up).
    import statistics
    action_names = list(next(iter(table.values())).keys())
    means_by_action = {a: statistics.fmean(table[s][a] for s in table) for a in action_names}
    best_i0_action = min(means_by_action, key=lambda a: means_by_action[a])
    i0_value = means_by_action[best_i0_action]
    full_value = statistics.fmean([min(row.values()) for row in table.values()])
    voi_full = i0_value - full_value

    report = {
        "schema": "voi-smoke-v0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "state": f"{base['episode_id']}@{FRAC}", "now_ms": t_base,
        "prefix_jobs": sum(1 for j in base["jobs"] if float(j["arrival_ms"]) <= t_base),
        "scenarios": {k: {"n_jobs": len(v[0]["jobs"])} for k, v in scenarios.items()},
        "actions": [[a[0], a[1]] for a in actions],
        "cost_table": table,
        "reproducible": reproducible, "rerun_value": rerun, "first_value": first,
        "demo_voi": {"i0_best_action": best_i0_action, "i0_value_ms": round(i0_value, 1),
                     "full_info_value_ms": round(full_value, 1),
                     "voi_full_ms": round(voi_full, 1)},
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("reproducible:", reproducible, "| demo VOI(full):", round(voi_full, 1), "ms")
    for sname, row in table.items():
        print(f"  {sname}: " + " | ".join(f"{k.split('|')[0][:14]}={v:.0f}" for k, v in row.items()))
    print("evidence:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
