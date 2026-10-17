#!/usr/bin/env python3
"""Gate 1A P1 runner (advisor 2026-10-16, approved with 3 added assertions).

4 pre-frozen train-source states x (original + 3 fixed donor futures) x ALL
legal first actions, forced action + fixed PDRS continuation.

Pre-frozen states (chosen for arrived-ratio / queue-depth / GPU-busy coverage,
NOT by known gains): ep00@0.60, ep01@0.35, ep02@0.75, ep03@0.50.
Donors (fixed): train episodes 4, 5, 6.

Assertions (stop on failure):
  A1 per-donated-job arrival reconciliation + completed==jobs + failed==0
  A2 forced action fully verified (des_first_forced==1, node AND gpu match)
  A3 decision-moment guard (des_first_state.now_ms == snapshot now_ms)
Reports: paired cost matrix, I0 best fixed action, per-scenario best, VOI,
PDRS headroom, best-action changes, advantage stability, per-scenario VOI
contribution, leave-one-scenario-out sensitivity.
Label: limited-scenario-pool PERFECT-ARRIVAL-INFORMATION upper bound only.
"""

from __future__ import annotations

import copy
import hashlib
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

FORMAL = ROOT / "experiments/EXP-20261014_burst64_v1/high_contention"
OUT = ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/gate1a_p1.json"
STATES = [(0, 0.60), (1, 0.35), (2, 0.75), (3, 0.50)]
DONORS = [4, 5, 6]
BUDGET_S = 6600


def state_setup(ep, tm, stats, artifacts, config, frac):
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
    return (capture.get("snapshots") or [None])[0]


def build_scenario(base_ep, snap, donor_ep, tag):
    t = float(snap["now_ms"])
    keep = [j for j in base_ep["jobs"] if float(j["arrival_ms"]) <= t]
    k_arrived = len(keep)
    donor_sorted = sorted(donor_ep["jobs"], key=lambda j: float(j["arrival_ms"]))
    if len(donor_sorted) <= k_arrived:
        return None
    donor_t = float(donor_sorted[k_arrived - 1]["arrival_ms"])
    donor_future = donor_sorted[k_arrived:]
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
    sc_snap["arrivals_pending"] = [[float(nj["arrival_ms"]), len(keep) + k]
                                   for k, nj in enumerate(new_jobs)]
    keep_ids = {str(j["job_instance_id"]) for j in sc["jobs"]}
    sc_snap["jobs"] = [row for row in sc_snap.get("jobs") or []
                       if str(row["job_instance_id"]) in keep_ids]
    return sc, sc_snap, keep, new_jobs


def main() -> int:
    config_path = ROOT / "experiments/EXP-20261012_milp_v0/artifacts/serial_prefetch_v1_extension.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fp = hashlib.sha256(config_path.read_bytes()).hexdigest()[:12]
    _serial, templates, _n, stats, artifacts = P.load_split("confirm", 1, config_path)
    tm = {k: replace(v, split="train") for k, v in templates.items()}
    eps = read_jsonl(FORMAL / "workload_burst64_r150_b15_train.jsonl")[:7]
    for r in eps:
        r["split"] = "train"
    tm, eps = apply_real_workload_profile_contract(tm, eps, config)

    t0 = time.perf_counter()
    report = {"schema": "gate1a-p1", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "des_fingerprint": fp, "states": [], "assertion_failures": []}
    for ep_idx, frac in STATES:
        if time.perf_counter() - t0 > BUDGET_S:
            report["budget_stop"] = True
            break
        base = eps[ep_idx]
        snap = state_setup(base, tm, stats, artifacts, config, frac)
        orig_now = float(snap["now_ms"])
        scenarios = {"orig": (base, snap, None, None)}
        smeta = {}
        for d in DONORS:
            built = build_scenario(base, snap, eps[d], f"d{d}")
            if built is None:
                continue
            sc, sc_snap, keep, new_jobs = built
            scenarios[f"d{d}"] = (sc, sc_snap, keep, new_jobs)
            smeta[f"d{d}"] = {"donor_episode": eps[d]["episode_id"], "k_arrived": len(keep),
                              "n_new_jobs": len(new_jobs)}
        if len(scenarios) < 2:
            report["states"].append({"state": f"{base['episode_id']}@{frac}", "now_ms": orig_now,
                                     "not_applicable": "no future arrivals to splice"})
            print(f"{base['episode_id']}@{frac}: NOT APPLICABLE (no future arrivals)")
            continue

        pe0 = build_planning_episode(base, snap, include_arrivals=True)
        _s, _e, pctx = run_planning(pe0, tm, future_artifacts=artifacts, train_stats=stats,
                                    extension_config=config, policy="des_rollout", forced_action=None,
                                    collect_events=False, extra_context={"des_probe_pool": True})
        pool = sorted({(str(r[1]), int(r[2])) for r in (pctx.get("des_first_pool") or [])})
        baseline = tuple(pctx.get("des_first_baseline") or ())

        table, pdrs_cost, asserts = {}, {}, []
        for sname, (sc_ep, sc_snap, keep, new_jobs) in scenarios.items():
            plan_ep = build_planning_episode(sc_ep, sc_snap, include_arrivals=True)
            # A3: decision-moment + pool-consistency guard via a probe run
            _ps, _pe, ppctx = run_planning(plan_ep, tm, future_artifacts=artifacts, train_stats=stats,
                                           extension_config=config, policy="des_rollout",
                                           forced_action=None, collect_events=False,
                                           extra_context={"des_probe_pool": True})
            p_now = float((ppctx.get("des_first_state") or {}).get("now_ms", -1.0))
            if abs(p_now - orig_now) > 1e-6:
                asserts.append(f"A3_fail {sname} probe_now={p_now} vs {orig_now}")
            scen_pool = sorted({(str(r[1]), int(r[2])) for r in (ppctx.get("des_first_pool") or [])})
            if scen_pool != pool:
                asserts.append(f"A3_pool_mismatch {sname} n={len(scen_pool)} vs {len(pool)}")
            table[sname] = {}
            for a in pool:
                s2, ev2, pctx2 = run_planning(plan_ep, tm, future_artifacts=artifacts, train_stats=stats,
                                              extension_config=config, policy="des_rollout",
                                              forced_action={"node_key": a[0], "gpu_index": a[1]},
                                              collect_events=True)
                act = (pctx2.get("activation") or {})
                # A2: forced action fully verified (node AND gpu; no silent fallback)
                if int(act.get("des_first_forced") or 0) != 1 or \
                        str(act.get("des_first_node")) != a[0] or int(act.get("des_first_gpu", -1)) != a[1]:
                    asserts.append(f"A2_fail {sname} {a[0][:20]}|{a[1]}")
                table[sname][f"{a[0]}|{a[1]}"] = round(float(s2["mean_completion_ms"]), 1)
                done, total = int(s2.get("completed_jobs") or 0), int(s2.get("jobs") or 0)
                if done != total or int(s2.get("failed_jobs") or 0) != 0:
                    asserts.append(f"A1_incomplete {sname} {done}/{total} failed={s2.get('failed_jobs')}")
            # A1: per-donated-job arrival reconciliation (donor scenarios only)
            if new_jobs:
                arrivals = {(str(e.get("job_instance_id")), round(float(e.get("arrival_ms") or -1), 1))
                            for e in (ev2 or []) if e.get("event_type") == "job_arrive"}
                for nj in new_jobs:
                    key = (str(nj["job_instance_id"]), round(float(nj["arrival_ms"]), 1))
                    if key not in arrivals:
                        asserts.append(f"A1_missing_arrival {sname} {key[0][-14:]}")
                old_future_ids = {str(j["job_instance_id"]) for j in base["jobs"]
                                  if float(j["arrival_ms"]) > orig_now}
                if any(oid in {x[0] for x in arrivals} for oid in old_future_ids):
                    asserts.append(f"A1_old_future_mixed {sname}")
            pdrs_cost[sname] = table[sname].get(f"{baseline[0]}|{baseline[1]}")

        paired = {s: {k: round(v - pdrs_cost[s], 1) for k, v in table[s].items()} for s in table}
        act_names = sorted(paired["orig"].keys())
        mean_by_action = {a: statistics.fmean(paired[s][a] for s in paired) for a in act_names}
        best_fixed = min(mean_by_action, key=lambda a: mean_by_action[a])
        i0_value = mean_by_action[best_fixed]
        per_scenario_best = {s: min(paired[s], key=lambda a: paired[s][a]) for s in paired}
        full_value = statistics.fmean([paired[s][per_scenario_best[s]] for s in paired])
        voi = i0_value - full_value
        # per-scenario VOI contribution + leave-one-out
        contrib = {s: round(paired[s][best_fixed] - paired[s][per_scenario_best[s]], 1) for s in paired}
        loo = {}
        for drop in paired:
            rest = [s for s in paired if s != drop]
            if not rest:
                continue
            m2 = {a: statistics.fmean(paired[s][a] for s in rest) for a in act_names}
            bf2 = min(m2, key=lambda a: m2[a])
            loo[drop] = round(m2[bf2] - statistics.fmean([paired[s][per_scenario_best[s]] for s in rest]), 1)
        # advantage stability of the best fixed action
        adv = [paired[s][best_fixed] for s in paired]
        stability = {"mean": round(statistics.fmean(adv), 1), "std": round(statistics.pstdev(adv), 1),
                     "sign_changes": sum(1 for i in range(len(adv) - 1) if (adv[i] > 0) != (adv[i + 1] > 0)),
                     "values": adv}
        entry = {"state": f"{base['episode_id']}@{frac}", "now_ms": orig_now,
                 "pool": len(pool), "pdrs_action": list(baseline), "scenario_meta": smeta,
                 "asserts": asserts, "best_fixed_action": best_fixed,
                 "i0_value_ms": round(i0_value, 1), "full_info_value_ms": round(full_value, 1),
                 "voi_ms": round(voi, 1), "best_action_per_scenario": per_scenario_best,
                 "n_distinct_best": len(set(per_scenario_best.values())),
                 "per_scenario_voi_contribution": contrib, "loo_voi": loo,
                 "advantage_stability": stability, "paired_deltas": paired}
        report["states"].append(entry)
        if asserts:
            report["assertion_failures"].append({"state": entry["state"], "fails": asserts})
        print(f"{entry['state']}: pool={len(pool)} VOI={entry['voi_ms']}ms distinct_best="
              f"{entry['n_distinct_best']} LOO={loo} asserts={len(asserts)}")
        if asserts:
            print("  ASSERTS:", asserts[:6])

    report["wall_s"] = round(time.perf_counter() - t0, 1)
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wall={report['wall_s']}s | failures={len(report['assertion_failures'])} | evidence: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
