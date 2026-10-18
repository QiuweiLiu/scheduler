#!/usr/bin/env python3
"""Gate 1A P2 + P0 ep03 audit (advisor 2026-10-16).

P0: audit ep03@0.50 at the ACTUAL dispatch moment (863637.877): compare the
four scenarios' scheduling-visible state (legal pool, PDRS action, now, GPU
states) -- identical -> reinterpret as dispatch-moment VOI; else exclude.

P2: stratified frozen sample from train episodes NOT used as P1 states
(P1 states: eps 0-3; P2 states drawn from eps 7-14; donors = eps 4,5,6 + 15-18,
7 donors).  3 many-future / 3 medium / 2 few, selection by remaining future
arrivals (pre-frozen rule), fully-arrived states marked structurally N/A.
Fail-fast: any hard assertion aborts the run and saves evidence.
Sampling: capture is verified against the probe's first GPU dispatch moment
(advance-and-retry), avoiding ready-but-no-dispatch drift.
Reports per state: VOI_arrival = min_a E[ΔJ] - E[min_a ΔJ] (equal-weight frozen
donors); best-fixed improvement vs PDRS; perfect-info extra; flips; LOO;
stability; interaction metrics (arrived/future/ready-depth/dispatchable GPUs/
first-arrival gap/donor model demand) for offline analysis only.
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
OUT = ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/gate1a_p2_expanded.json"
AUDIT_OUT = ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/gate1a_ep03_audit.json"
DONORS = [0, 1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19]  # expanded pool (12 donors + orig = 13 scenarios/state)
CAND_EPISODES = [7, 8, 9, 10, 11, 12, 13, 14]
CAND_FRACTIONS = [0.25, 0.35, 0.45, 0.55, 0.65]
BUDGET_S = 6600


def makespan(ep, tm, stats, artifacts, config):
    summary, _ev = simulate_episode(ep, tm, "pdrs_resident", train_stats=stats,
                                    future_artifacts=artifacts, future_horizon=5,
                                    extension_config=config,
                                    policy_context={"activation": {}}, collect_events=False)
    return float(summary["makespan_ms"])


def capture_at(ep, tm, stats, artifacts, config, at_ms):
    capture = {"at_ms_list": [round(at_ms, 1)], "snapshots": []}
    simulate_episode(ep, tm, "pdrs_resident", train_stats=stats,
                     future_artifacts=artifacts, future_horizon=5,
                     extension_config=config,
                     policy_context={"activation": {}, "des_capture": capture},
                     collect_events=False)
    return (capture.get("snapshots") or [None])[0]


def state_with_true_dispatch_moment(ep, tm, stats, artifacts, config, at_ms, tries=8):
    """Advance until the snapshot matches the probe's first dispatch moment."""
    t = at_ms
    for _ in range(tries):
        snap = capture_at(ep, tm, stats, artifacts, config, t)
        if snap is None:
            t += 2000.0
            continue
        pe = build_planning_episode(ep, snap, include_arrivals=True)
        _s, _e, pctx = run_planning(pe, tm, future_artifacts=artifacts, train_stats=stats,
                                    extension_config=config, policy="des_rollout", forced_action=None,
                                    collect_events=False, extra_context={"des_probe_pool": True})
        if abs(float((pctx.get("des_first_state") or {}).get("now_ms", -1.0)) - float(snap["now_ms"])) <= 1e-6:
            return snap, pctx
        t += 2000.0
    return None, None


def build_scenario(base_ep, snap, donor_ep, tag):
    t = float(snap["now_ms"])
    keep = [j for j in base_ep["jobs"] if float(j["arrival_ms"]) <= t]
    k_arrived = len(keep)
    donor_sorted = sorted(donor_ep["jobs"], key=lambda j: float(j["arrival_ms"]))
    if len(donor_sorted) <= k_arrived:
        return None
    donor_t = float(donor_sorted[k_arrived - 1]["arrival_ms"])
    donor_future = donor_sorted[k_arrived:]
    shift = 0.0
    if donor_future:
        first_rel = float(donor_future[0]["arrival_ms"]) - donor_t
        if first_rel < 500.0:
            shift = 500.0 - first_rel
    new_jobs = []
    for k, j in enumerate(donor_future):
        nj = dict(j)
        arr = t + shift + (float(j["arrival_ms"]) - donor_t)
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
    return sc, sc_snap, new_jobs


def main() -> int:
    config_path = ROOT / "experiments/EXP-20261012_milp_v0/artifacts/serial_prefetch_v1_extension.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fp = hashlib.sha256(config_path.read_bytes()).hexdigest()[:12]
    _serial, templates, _n, stats, artifacts = P.load_split("confirm", 1, config_path)
    tm = {k: replace(v, split="train") for k, v in templates.items()}
    eps = read_jsonl(FORMAL / "workload_burst64_r150_b15_train.jsonl")
    for r in eps:
        r["split"] = "train"
    tm, eps = apply_real_workload_profile_contract(tm, eps, config)

    t0 = time.perf_counter()
    report = {"schema": "gate1a-p2", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "des_fingerprint": fp, "donors": DONORS, "states": [], "hard_failures": []}

    # ---------- P0: ep03 audit at the actual dispatch moment ----------
    base = eps[3]
    snap3 = capture_at(base, tm, stats, artifacts, config, 862046.029)
    pe3 = build_planning_episode(base, snap3, include_arrivals=True)
    _s, _e, p3 = run_planning(pe3, tm, future_artifacts=artifacts, train_stats=stats,
                              extension_config=config, policy="des_rollout", forced_action=None,
                              collect_events=False, extra_context={"des_probe_pool": True})
    actual_now = float((p3.get("des_first_state") or {}).get("now_ms", -1.0))
    snap3b = capture_at(base, tm, stats, artifacts, config, actual_now)
    audit = {"actual_dispatch_now": actual_now, "scenarios": {}}
    for tag, donor_idx in [("orig", None)] + [(f"d{d}", d) for d in [4, 5, 6]]:
        if donor_idx is None:
            sc_ep, sc_snap = base, snap3b
        else:
            built = build_scenario(base, snap3b, eps[donor_idx], tag)
            if built is None:
                audit["scenarios"][tag] = "no_splice"
                continue
            sc_ep, sc_snap, _nj = built
        pe = build_planning_episode(sc_ep, sc_snap, include_arrivals=True)
        _s2, _e2, pctx2 = run_planning(pe, tm, future_artifacts=artifacts, train_stats=stats,
                                       extension_config=config, policy="des_rollout", forced_action=None,
                                       collect_events=False, extra_context={"des_probe_pool": True})
        pool = sorted({(str(r[1]), int(r[2])) for r in (pctx2.get("des_first_pool") or [])})
        gpus = (pctx2.get("des_first_state") or {}).get("gpus") or []
        audit["scenarios"][tag] = {
            "now_ms": float((pctx2.get("des_first_state") or {}).get("now_ms", -1.0)),
            "pool": pool, "baseline": list(pctx2.get("des_first_baseline") or ()),
            "gpus": [{"i": g.get("index"), "busy_until": round(float(g.get("busy_until") or 0), 1),
                      "resident": sorted((g.get("resident") or {}).keys())} for g in gpus],
        }
    ref = audit["scenarios"].get("orig")
    audit["identical"] = all(
        (v == ref) for k, v in audit["scenarios"].items() if isinstance(v, dict) and k != "orig")
    AUDIT_OUT.write_text(json.dumps(audit, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"P0 ep03 audit: identical={audit['identical']} (actual_now={actual_now})")

    # ---------- P2: stratified frozen sample ----------
    cand = []
    for ep_idx in CAND_EPISODES:
        ms = makespan(eps[ep_idx], tm, stats, artifacts, config)
        for frac in CAND_FRACTIONS:
            at = frac * ms
            arrived = sum(1 for j in eps[ep_idx]["jobs"] if float(j["arrival_ms"]) <= at)
            future = len(eps[ep_idx]["jobs"]) - arrived
            cand.append({"ep": ep_idx, "frac": frac, "at": at, "arrived": arrived, "future": future})
    cand = [c for c in cand if c["future"] > 0]
    cand.sort(key=lambda c: -c["future"])
    many = cand[:3]
    mid_start = max(3, len(cand) // 2 - 1)
    medium = cand[mid_start:mid_start + 3]
    few = cand[-2:]
    selected = [("many", c) for c in many] + [("medium", c) for c in medium] + [("few", c) for c in few]
    report["selection_rule"] = "frozen: sort candidate states (eps 7-14 x fracs) by remaining future arrivals; top3=many, middle3=medium, last2=few"
    report["selected"] = [{"stratum": s, "episode": eps[c["ep"]]["episode_id"], "frac": c["frac"],
                           "arrived": c["arrived"], "future": c["future"]} for s, c in selected]
    print("P2 selected:", [(s, eps[c['ep']]['episode_id'][-4:], c['frac'], c['future']) for s, c in selected])

    for si, (stratum, c) in enumerate(selected):
        if time.perf_counter() - t0 > BUDGET_S:
            report["budget_stop"] = True
            break
        base = eps[c["ep"]]
        snap, _pctx0 = state_with_true_dispatch_moment(base, tm, stats, artifacts, config, c["at"])
        if snap is None:
            report["states"].append({"state": f"{base['episode_id']}@{c['frac']}", "stratum": stratum,
                                     "excluded": "no stable dispatch moment"})
            continue
        orig_now = float(snap["now_ms"])
        scenarios = {"orig": (base, snap, None)}
        for d in DONORS:
            built = build_scenario(base, snap, eps[d], f"d{d}")
            if built is not None:
                scenarios[f"d{d}"] = built
        # probe for the pool + A3
        pe0 = build_planning_episode(base, snap, include_arrivals=True)
        _s, _e, pctx = run_planning(pe0, tm, future_artifacts=artifacts, train_stats=stats,
                                    extension_config=config, policy="des_rollout", forced_action=None,
                                    collect_events=False, extra_context={"des_probe_pool": True})
        pool = sorted({(str(r[1]), int(r[2])) for r in (pctx.get("des_first_pool") or [])})
        baseline = tuple(pctx.get("des_first_baseline") or ())
        hard = []
        table, pdrs_cost, scen_fp = {}, {}, {}
        for sname, (sc_ep, sc_snap, _nj) in scenarios.items():
            plan_ep = build_planning_episode(sc_ep, sc_snap, include_arrivals=True)
            _ps, _pe, ppctx = run_planning(plan_ep, tm, future_artifacts=artifacts, train_stats=stats,
                                           extension_config=config, policy="des_rollout", forced_action=None,
                                           collect_events=False, extra_context={"des_probe_pool": True})
            p_now = float((ppctx.get("des_first_state") or {}).get("now_ms", -1.0))
            scen_pool = sorted({(str(r[1]), int(r[2])) for r in (ppctx.get("des_first_pool") or [])})
            if abs(p_now - orig_now) > 1e-6:
                hard.append(f"A3_now {sname} {p_now} vs {orig_now}")
            if scen_pool != pool:
                hard.append(f"A3_pool {sname} n={len(scen_pool)} vs {len(pool)}")
            gp = (ppctx.get("des_first_state") or {}).get("gpus") or []
            fp = json.dumps({"pool": scen_pool,
                             "baseline": list(ppctx.get("des_first_baseline") or ()),
                             "gpus": [{"i": g.get("index"),
                                       "busy": round(float(g.get("busy_until") or 0), 1),
                                       "res": sorted((g.get("resident") or {}).keys())} for g in gp]},
                            sort_keys=True)
            scen_fp[sname] = hashlib.sha256(fp.encode()).hexdigest()[:16]
            table[sname] = {}
            for a in pool:
                s2, ev2, pctx2 = run_planning(plan_ep, tm, future_artifacts=artifacts, train_stats=stats,
                                              extension_config=config, policy="des_rollout",
                                              forced_action={"node_key": a[0], "gpu_index": a[1]},
                                              collect_events=True)
                act = (pctx2.get("activation") or {})
                if int(act.get("des_first_forced") or 0) != 1 or \
                        str(act.get("des_first_node")) != a[0] or int(act.get("des_first_gpu", -1)) != a[1]:
                    hard.append(f"A2 {sname} {a[0][:18]}|{a[1]}")
                table[sname][f"{a[0]}|{a[1]}"] = round(float(s2["mean_completion_ms"]), 1)
                if int(s2.get("completed_jobs") or 0) != int(s2.get("jobs") or 0) or int(s2.get("failed_jobs") or 0) != 0:
                    hard.append(f"A1_done {sname} {s2.get('completed_jobs')}/{s2.get('jobs')}")
            if _nj:
                arrivals = [(str(e.get("job_instance_id")), float(e.get("arrival_ms") or -1.0))
                            for e in (ev2 or []) if e.get("event_type") == "job_arrive"]
                for nj in _nj:
                    jid = str(nj["job_instance_id"])
                    exp = float(nj["arrival_ms"])
                    hits = [x for j, x in arrivals if j == jid and abs(x - exp) <= 1.0]
                    if len(hits) != 1:
                        hard.append(f"A1_arr_unique {sname} {jid[-12:]} hits={len(hits)}")
            pdrs_cost[sname] = table[sname].get(f"{baseline[0]}|{baseline[1]}")
        if hard:
            report["hard_failures"].append({"state": f"{base['episode_id']}@{c['frac']}", "fails": hard[:8]})
            print(f"HARD FAIL {base['episode_id']}@{c['frac']}: {hard[:4]}")
            break  # fail-fast
        common = set.intersection(*[set(table[s].keys()) for s in table])
        paired = {s: {k: round(table[s][k] - pdrs_cost[s], 1) for k in common} for s in table}
        act_names = sorted(common)
        mean_by_action = {a: statistics.fmean(paired[s][a] for s in paired) for a in act_names}
        best_fixed = min(mean_by_action, key=lambda a: mean_by_action[a])
        i0 = mean_by_action[best_fixed]
        per_best = {s: min(paired[s], key=lambda a: paired[s][a]) for s in paired}
        full = statistics.fmean([paired[s][per_best[s]] for s in paired])
        voi = i0 - full
        contrib = {s: round(paired[s][best_fixed] - paired[s][per_best[s]], 1) for s in paired}
        loo = {}
        for drop in paired:
            rest = [s for s in paired if s != drop]
            m2 = {a: statistics.fmean(paired[s][a] for s in rest) for a in act_names}
            bf2 = min(m2, key=lambda a: m2[a])
            loo[drop] = round(m2[bf2] - statistics.fmean([paired[s][per_best[s]] for s in rest]), 1)
        adv = [paired[s][best_fixed] for s in paired]
        # Top-K analysis with a FROZEN decision-time-visible score (runtime + best-GPU queue)
        obs = T.obs_from_snapshot(snap, [{"job_instance_id": j["job_instance_id"],
                                          "template_id": j["template_id"]} for j in base["jobs"]],
                                  tm, stats, artifacts)
        by_id = {str(x.node_id): x for x in obs.ready}
        def vis_score(a):
            it = by_id.get(a.split("|")[0])
            if it is None:
                return 1e18
            queue = min(g.available_at_ms for g in obs.gpus) - obs.now_ms if obs.gpus else 0.0
            return float(it.runtime_p50_ms) + max(0.0, queue)
        ranked = sorted(act_names, key=lambda a: (vis_score(a), a))
        if f"{baseline[0]}|{baseline[1]}" in ranked:
            ranked.remove(f"{baseline[0]}|{baseline[1]}")
        topk = {}
        for k in (2, 4, 8, None):
            keep = set(ranked[:k] if k else ranked) | {f"{baseline[0]}|{baseline[1]}"}
            sub = {a: statistics.fmean(paired[s2][a] for s2 in paired) for a in keep}
            bf = min(sub, key=lambda a: sub[a])
            i0k = sub[bf]
            fullk = statistics.fmean([min(paired[s2][a] for a in keep) for s2 in paired])
            topk[str(k)] = {"pool_size": len(keep), "best_fixed": bf,
                            "voi_ms": round(i0k - fullk, 1)}
        donor_meta = {}
        for sname, (sc_ep, sc_snap, nj) in scenarios.items():
            if nj:
                gaps = [round(float(nj[0]["arrival_ms"]) - orig_now, 1)]
                models = {}
                for j in nj:
                    tpl_id = str(j["template_id"])
                    models[tpl_id] = models.get(tpl_id, 0) + 1
                donor_meta[sname] = {"n_new_jobs": len(nj), "first_gap_ms": gaps[0],
                                     "n_distinct_templates": len(models)}
        entry = {"state": f"{base['episode_id']}@{c['frac']}", "stratum": stratum,
                 "table": table, "paired_deltas": paired, "pdrs_cost": pdrs_cost,
                 "per_scenario_best": per_best, "scenario_fingerprints": scen_fp,
                 "donor_meta": donor_meta, "topk_voi": topk,
                 "now_ms": orig_now, "arrived": c["arrived"], "future": c["future"],
                 "pool": len(pool), "common_pool": len(common), "n_scenarios": len(scenarios),
                 "best_fixed_action": best_fixed,
                 "best_fixed_improvement_ms": round(-i0, 1),
                 "perfect_info_improvement_ms": round(-full, 1),
                 "voi_ms": round(voi, 1),
                 "n_distinct_best": len(set(per_best.values())),
                 "per_scenario_voi_contribution": contrib, "loo_voi": loo,
                 "advantage_stability": {"mean": round(statistics.fmean(adv), 1),
                                         "std": round(statistics.pstdev(adv), 1)},
                 "pdrs_action": list(baseline)}
        report["states"].append(entry)
        print(f"{entry['state']} [{stratum}] future={c['future']} pool={len(pool)} VOI={entry['voi_ms']}ms "
              f"fixed_impr={entry['best_fixed_improvement_ms']} distinct={entry['n_distinct_best']}")

    report["wall_s"] = round(time.perf_counter() - t0, 1)
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wall={report['wall_s']}s hard_failures={len(report['hard_failures'])} evidence: {OUT}")
    return 1 if report["hard_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
