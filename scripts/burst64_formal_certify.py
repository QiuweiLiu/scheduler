#!/usr/bin/env python3
"""Formal-workload certification for the Burst64-v1 high-contention set.

Full-population decision metrics (action space, competing fraction, GPU
utilization, completion) on both sources + a small PDRS Headroom sample on
r150_b15 (forced-action true-future rollouts).
"""

from __future__ import annotations

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

FOLDER = ROOT / "experiments/EXP-20261014_burst64_v1/high_contention"


def main() -> int:
    config_path = ROOT / "experiments/EXP-20261012_milp_v0/artifacts/serial_prefetch_v1_extension.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    serial, templates, _n, stats, artifacts = P.load_split("confirm", 1, config_path)
    tm = {k: replace(v, split="train") for k, v in templates.items()}

    report = {"schema": "burst64-formal-certification-v1",
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "conditions": {}}
    for source in ("train", "validation"):
        eps = read_jsonl(FOLDER / f"workload_burst64_r150_b15_{source}.jsonl")[:2]
        for e in eps:
            e["split"] = "train"
        t2, e2 = apply_real_workload_profile_contract(dict(tm), list(eps), config)
        pools, competing, utils, completed = [], 0, [], []
        for ep in e2:
            s, events = simulate_episode(ep, t2, "pdrs_resident", train_stats=stats,
                                         future_artifacts=artifacts, future_horizon=5,
                                         extension_config=config,
                                         policy_context={"activation": {}}, collect_events=True)
            starts, busy = {}, 0.0
            for ev in events or []:
                et = ev.get("event_type")
                if et == "node_dispatch":
                    raw = int(ev.get("candidate_count") or 0)
                    feas = int(ev.get("feasible_candidate_count") or 0)
                    pools.append(feas if feas > 0 else raw)
                    st = ev.get("scheduler_state") or {}
                    jobs = {str(n.get("job_instance_id")) for n in (st.get("ready_nodes") or [])}
                    if len(jobs) >= 2:
                        competing += 1
                elif et == "node_start":
                    starts[str(ev.get("node_id"))] = float(ev.get("time_ms") or 0.0)
                elif et == "node_finish":
                    key = str(ev.get("node_id"))
                    if key in starts:
                        busy += float(ev.get("time_ms") or 0.0) - starts.pop(key)
            utils.append(busy / (2.0 * float(s.get("makespan_ms") or 1.0)))
            completed.append((s["completed_jobs"], s["jobs"]))
        n = max(1, len(pools))
        report["conditions"][f"r150_b15|{source}"] = {
            "episodes": len(e2), "decisions": len(pools),
            "legal_actions_mean": round(statistics.fmean(pools), 2),
            "legal_actions_p50": round(statistics.median(pools), 1),
            "legal_actions_p90": round(sorted(pools)[int(0.9 * (len(pools) - 1))], 1),
            "ge10_frac": round(sum(1 for p in pools if p >= 10) / n, 3),
            "competing_frac": round(competing / n, 3),
            "gpu_util_mean": round(statistics.fmean(utils), 3),
            "all_completed": all(c == j for c, j in completed),
        }

    # PDRS headroom sample on r150_b15 (train source)
    eps = read_jsonl(FOLDER / "workload_burst64_r150_b15_train.jsonl")[:2]
    for e in eps:
        e["split"] = "train"
    t2, e2 = apply_real_workload_profile_contract(dict(tm), list(eps), config)
    states = T.capture_states(e2, t2, stats, artifacts, config, cap=10)
    heads, checked, pools_h = [], 0, []
    for ep, jobs_meta, snap in states:
        if checked >= 12:
            break
        plan_episode = build_planning_episode(ep, snap, include_arrivals=True)
        _s, _e, pctx = run_planning(plan_episode, t2, future_artifacts=artifacts, train_stats=stats,
                                    extension_config=config, policy="des_rollout", forced_action=None,
                                    collect_events=False, extra_context={"des_probe_pool": True})
        if abs(float((pctx.get("des_first_state") or {}).get("now_ms", -1)) - float(snap["now_ms"])) > 1e-6:
            continue
        pool = sorted({(str(r[1]), int(r[2])) for r in (pctx.get("des_first_pool") or [])})
        baseline = tuple(pctx.get("des_first_baseline") or ())
        if not pool or not baseline:
            continue
        checked += 1
        pools_h.append(len(pool))
        costs = {}
        for node, gpu in pool:
            summary, _ev, _c = run_planning(
                plan_episode, t2, future_artifacts=artifacts, train_stats=stats,
                extension_config=config, policy="des_rollout",
                forced_action={"node_key": node, "gpu_index": gpu}, collect_events=False)
            costs[(node, gpu)] = float(summary["mean_completion_ms"])
        base = costs.get(baseline)
        if base is None:
            continue
        heads.append(base - min(costs.values()))
    report["headroom_r150_b15_train"] = {
        "states": len(heads), "pool_mean": round(statistics.fmean(pools_h), 1) if pools_h else None,
        "headroom_mean": round(statistics.fmean(heads), 1) if heads else None,
        "headroom_zero_frac": round(sum(1 for h in heads if h <= 1e-9) / max(1, len(heads)), 3),
        "headroom_max": round(max(heads), 1) if heads else None,
    }
    out = FOLDER / "burst64_formal_certification.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print("evidence:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
