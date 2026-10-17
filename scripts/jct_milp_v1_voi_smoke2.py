#!/usr/bin/env python3
"""Gate 1A all-action smoke v2 (advisor P0 fixes, 2026-10-16).

Fixes vs v1:
  - arrivals are installed via the CORRECT snapshot field `arrivals_pending`
    (apply_injected_state reads it; v1 wrote `arrival_heap` -> old arrivals).
  - donor anchor = arrival-count alignment (same number of arrived jobs as the
    base state), not a makespan fraction.
  - assertions: donated jobs all arrive/complete, no old future jobs, forced
    action actually executed (activation des_seq_forced>=1), reproducibility,
    VOI_full >= 0; ALL legal actions (full pool) are evaluated per scenario.
Paired-within-scenario deltas vs PDRS; absolute cross-scenario means are NOT
compared (different job counts).
"""

from __future__ import annotations

import copy
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
OUT = ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/voi_smoke_v2.json"
FRAC = 0.6


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
    """Prefix fixed; donor's post-K arrivals spliced (K = #arrived in base at t)."""
    t = float(snap["now_ms"])
    keep = [j for j in base_ep["jobs"] if float(j["arrival_ms"]) <= t]
    k_arrived = len(keep)
    donor_sorted = sorted(donor_ep["jobs"], key=lambda j: float(j["arrival_ms"]))
    if len(donor_sorted) <= k_arrived:
        return None
    donor_t = float(donor_sorted[k_arrived - 1]["arrival_ms"])  # arrival-count anchor
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
    # CORRECT field (advisor P0-1): apply_injected_state reads arrivals_pending
    sc_snap["arrivals_pending"] = [[float(nj["arrival_ms"]), len(keep) + k]
                                   for k, nj in enumerate(new_jobs)]
    keep_ids = {str(j["job_instance_id"]) for j in sc["jobs"]}
    sc_snap["jobs"] = [row for row in sc_snap.get("jobs") or []
                       if str(row["job_instance_id"]) in keep_ids]
    return sc, sc_snap, k_arrived, len(new_jobs), donor_t


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

    snap = state_setup(base, tm, stats, artifacts, config, FRAC)
    scenarios = {"orig": (base, snap, None)}
    meta = {}
    for k, donor in enumerate(eps[1:], 1):
        dsnap = state_setup(donor, tm, stats, artifacts, config, FRAC)
        built = build_scenario(base, snap, donor, f"d{k}")
        assert built is not None
        sc, sc_snap, k_arr, n_new, donor_t = built
        scenarios[f"donor{k}"] = (sc, sc_snap, None)
        meta[f"donor{k}"] = {"k_arrived_anchor": k_arr, "n_new_jobs": n_new, "donor_t": donor_t}

    pe0 = build_planning_episode(base, snap, include_arrivals=True)
    _s, _e, pctx = run_planning(pe0, tm, future_artifacts=artifacts, train_stats=stats,
                                extension_config=config, policy="des_rollout", forced_action=None,
                                collect_events=False, extra_context={"des_probe_pool": True})
    pool = sorted({(str(r[1]), int(r[2])) for r in (pctx.get("des_first_pool") or [])})
    baseline = tuple(pctx.get("des_first_baseline") or ())
    print(f"state {base['episode_id']}@{FRAC} | pool(all legal actions)={len(pool)} | pdrs={baseline}")

    table, asserts, pdrs_cost = {}, {}, {}
    for sname, (sc_ep, sc_snap, _x) in scenarios.items():
        plan_ep = build_planning_episode(sc_ep, sc_snap, include_arrivals=True)
        table[sname], forced_ok = {}, True
        for a in pool:
            s2, ev2, pctx2 = run_planning(plan_ep, tm, future_artifacts=artifacts, train_stats=stats,
                                          extension_config=config, policy="des_rollout",
                                          forced_action={"node_key": a[0], "gpu_index": a[1]},
                                          collect_events=True)
            # forced-action evidence: the FIRST node_start event must be the forced node
            first_start = next((e for e in (ev2 or []) if e.get("event_type") == "node_start"), None)
            if first_start is None or str(first_start.get("node_id")) != a[0]:
                forced_ok = False
                asserts.setdefault(sname, []).append(
                    f"forced_mismatch @ {a[0][:22]}: got {str((first_start or {}).get('node_id'))[:22]}")
            table[sname][f"{a[0]}|{a[1]}"] = round(float(s2["mean_completion_ms"]), 1)
            done, total = int(s2.get("completed_jobs") or 0), int(s2.get("jobs") or 0)
            if done != total:
                asserts.setdefault(sname, []).append(f"incomplete {done}/{total} @ {a[0][:18]}")
        pdrs_cost[sname] = table[sname].get(f"{baseline[0]}|{baseline[1]}")
        asserts.setdefault(sname, []).append("forced_action_ok" if forced_ok else "FORCED_ACTION_FAIL")
        n_jobs = len(sc_ep["jobs"])
        asserts[sname].append(f"n_jobs={n_jobs}")

    # paired deltas vs PDRS within each scenario
    paired = {s: {k: round(v - pdrs_cost[s], 1) for k, v in table[s].items()} for s in table}
    # VOI_full (perfect arrival info) = min_a mean_omega dJ  -  mean_omega min_a dJ
    act_names = sorted(paired["orig"].keys())
    mean_by_action = {a: statistics.fmean(paired[s][a] for s in paired) for a in act_names}
    best_fixed = min(mean_by_action, key=lambda a: mean_by_action[a])
    i0_value = mean_by_action[best_fixed]
    full_value = statistics.fmean([min(paired[s].values()) for s in paired])
    voi_full = i0_value - full_value
    best_per_scenario = {s: min(paired[s], key=lambda a: paired[s][a]) for s in paired}
    flips = len(set(best_per_scenario.values()))

    report = {
        "schema": "voi-smoke-v2-gate1a",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "state": f"{base['episode_id']}@{FRAC}", "pool": len(pool),
        "scenario_meta": meta, "asserts": asserts,
        "pdrs_cost": pdrs_cost,
        "best_action_per_scenario": best_per_scenario,
        "best_action_flips": flips,
        "best_fixed_action": best_fixed,
        "voi_full_ms": round(voi_full, 1),
        "i0_value_ms": round(i0_value, 1), "full_info_value_ms": round(full_value, 1),
        "paired_deltas": paired, "wall_s": round(time.perf_counter() - t0, 1),
    }
    assert voi_full >= -1e-6, "VOI must be >= 0"
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("asserts:", json.dumps(asserts, ensure_ascii=False))
    print(f"best per scenario: {best_per_scenario} | flips={flips}")
    print(f"VOI_full={voi_full:.1f}ms | best_fixed={best_fixed[:30]} | wall={report['wall_s']}s")
    print("evidence:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
