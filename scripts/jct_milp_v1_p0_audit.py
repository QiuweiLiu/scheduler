#!/usr/bin/env python3
"""P0/P1 audit (advisor 2026-10-16): first-action semantics + cross-candidate
cost comparability.

P0: for states A (ep00@0.7) and B (ep01@0.5) export, from the FREE solved plan,
every d_ig=1 dispatch-now bin, the x assignments and planned starts of those
nodes, and the extracted first_action — does the free MILP plan several
immediate dispatches while the executor consumes only one?
P1: for the five forced-first-action solves, verify the planned node set,
variable/constraint counts and objective boundary terms are IDENTICAL across
candidates (only the pinned first action differs).
"""

from __future__ import annotations

import argparse
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
from tracing.scheduling.milp_v0 import _build_nodes_v1, _screen_anchors, solve_v1  # noqa: E402

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


def audit_state(ep, frac, tm, stats, artifacts, config, tail_table, ka, kc, budget):
    summary, _ev = simulate_episode(ep, tm, "pdrs_resident", train_stats=stats,
                                    future_artifacts=artifacts, future_horizon=5,
                                    extension_config=config,
                                    policy_context={"activation": {}}, collect_events=False)
    at = round(frac * float(summary["makespan_ms"]), 1)
    capture = {"at_ms_list": [at], "snapshots": []}
    simulate_episode(ep, tm, "pdrs_resident", train_stats=stats,
                     future_artifacts=artifacts, future_horizon=5,
                     extension_config=config,
                     policy_context={"activation": {}, "des_capture": capture}, collect_events=False)
    snap = capture["snapshots"][0]
    jobs_meta = [{"job_instance_id": j["job_instance_id"], "template_id": j["template_id"]}
                 for j in ep["jobs"]]
    plan_episode = build_planning_episode(ep, snap, include_arrivals=True)
    _s, _e, pctx = run_planning(
        plan_episode, tm, future_artifacts=artifacts, train_stats=stats,
        extension_config=config, policy="des_rollout", forced_action=None,
        collect_events=False, extra_context={"des_probe_pool": True})
    pool = sorted({(str(r[1]), int(r[2])) for r in (pctx.get("des_first_pool") or [])})
    baseline = tuple(pctx.get("des_first_baseline") or ())
    obs = T.obs_from_snapshot(snap, jobs_meta, tm, stats, artifacts)
    masked = replace(obs, legal_first=frozenset(pool))
    required = frozenset({baseline[0]}) if baseline else frozenset()
    screened_ids = _screen_anchors(masked, ka, required)

    nodes_list = _build_nodes_v1(masked, chain_depth=3, k_chains=kc, tail=True,
                                 tail_table=tail_table, k_candidates=ka,
                                 required_node_ids=required)
    gpu_nodes = [i for i, n in enumerate(nodes_list) if n.lane == "gpu"]
    gpu_pos = {idx: pos for pos, idx in enumerate(gpu_nodes)}
    ready_gpu_nodes = [i for i in gpu_nodes if nodes_list[i].is_ready]
    gpu_count = len(masked.gpus)
    d_pairs = [(i, g) for i in ready_gpu_nodes for g in range(gpu_count)]

    def extract(dump, res):
        values = dump.get("values") or []
        index = dump.get("index") or {}
        if not values:
            return {"error": "no solution values"}
        d0, x0, a0 = index.get("d0"), index.get("x0"), index.get("a0")
        active = []
        for pos, (i, g) in enumerate(d_pairs):
            if d0 is not None and float(values[d0 + pos]) >= 0.5:
                node = nodes_list[i]
                entry = {"action": [node.key, g], "start_ms": round(float(values[a0 + i]), 1)}
                entry["assigned_gpus"] = [
                    gg for gg in range(gpu_count)
                    if float(values[x0 + gpu_pos[i] * gpu_count + gg]) >= 0.5
                ]
                entry["is_ready_anchor"] = bool(node.is_ready)
                active.append(entry)
        return {
            "n_d_active": len(active),
            "d_active": active,
            "returned_first_action": list(res.first_action) if res.first_action else None,
            "returned_is_in_d_active": bool(
                res.first_action and any(a["action"] == [res.first_action[0], int(res.first_action[1])] for a in active)),
        }

    dump_free: dict = {}
    res_free = solve_v1(masked, chain_depth=3, k_chains=kc, tail=True, tail_table=tail_table,
                        time_limit_s=budget, k_candidates=ka, required_node_ids=required,
                        debug_dump=dump_free)
    free_info = extract(dump_free, res_free) if not res_free.fallback else {"fallback": True, "reason": res_free.reason}

    forced = {}
    for node, gpu in pool:
        if node not in screened_ids:
            continue
        dump_f: dict = {}
        res_f = solve_v1(masked, chain_depth=3, k_chains=kc, tail=True, tail_table=tail_table,
                         time_limit_s=budget, k_candidates=ka, required_node_ids=required,
                         force_first_action=(node, gpu), debug_dump=dump_f)
        info = extract(dump_f, res_f) if not res_f.fallback else {"fallback": True, "reason": res_f.reason}
        info.update({
            "status": res_f.status, "objective": res_f.objective, "bound": res_f.best_bound, "gap": res_f.gap,
            "n_vars": dump_f.get("n_vars"), "n_rows": len(dump_f.get("rows") or []),
            "n_nodes": len(dump_f.get("nodes") or []),
        })
        forced[f"{node}|{gpu}"] = info

    return {
        "episode": ep["episode_id"], "fraction": frac, "now_ms": snap["now_ms"],
        "pool": len(pool), "screened": sorted(screened_ids),
        "pdrs_action": list(baseline) if baseline else None,
        "free": {"status": res_free.status, "objective": res_free.objective,
                 "bound": res_free.best_bound, "gap": res_free.gap,
                 "n_vars": dump_free.get("n_vars"), "n_rows": len(dump_free.get("rows") or []),
                 "n_nodes": len(dump_free.get("nodes") or []), **free_info},
        "forced": forced,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ka", type=int, default=5)
    ap.add_argument("--kc", type=int, default=2)
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/p0_p1_audit.json")
    args = ap.parse_args()

    config_path = ROOT / "experiments/EXP-20261012_milp_v0/artifacts/serial_prefetch_v1_extension.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _serial, templates, _n, stats, artifacts = P.load_split("confirm", 1, config_path)
    tm = {k: replace(v, split="train") for k, v in templates.items()}
    episodes = read_jsonl(FORMAL / "workload_burst64_r150_b15_train.jsonl")
    for row in episodes:
        row["split"] = "train"
    tm, episodes = apply_real_workload_profile_contract(tm, episodes, config)
    tail_table = load_tail_table()

    report = {"schema": "jct-milp-v1-p0-p1-audit-v1",
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "states": []}
    for ep_idx, frac in ((0, 0.7), (1, 0.5)):
        st = audit_state(episodes[ep_idx], frac, tm, stats, artifacts, config, tail_table,
                         args.ka, args.kc, args.budget)
        report["states"].append(st)
        f = st["free"]
        print(f"\n=== {st['episode']}@{frac} pool={st['pool']} screened={len(st['screened'])} ===")
        print(f"FREE: status={f.get('status')} n_d_active={f.get('n_d_active')} "
              f"returned={f.get('returned_first_action')} in_d_active={f.get('returned_is_in_d_active')}")
        for a in f.get("d_active") or []:
            print(f"   d=1: {a['action']} start={a['start_ms']}ms assigned={a['assigned_gpus']} ready={a['is_ready_anchor']}")
        node_counts = {k: (v.get("n_nodes"), v.get("n_vars"), v.get("n_rows")) for k, v in st["forced"].items()}
        print("FORCED (candidate -> (n_nodes,n_vars,n_rows) | status | n_d_active | returned==pinned):")
        for k, v in st["forced"].items():
            pinned = k.split("|")[0]
            returned = (v.get("returned_first_action") or [None])[0]
            print(f"   {k[:34]:34s} {node_counts[k]} {v.get('status')} d={v.get('n_d_active')} "
                  f"returned_pinned={returned == pinned}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\nevidence:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
