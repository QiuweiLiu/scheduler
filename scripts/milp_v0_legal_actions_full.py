#!/usr/bin/env python3
"""Count legal first-action options per decision in the FULL v7 environment.

Same method as the serial_v0 count (DES probe candidate pool at the decision
moment) but with substrate_extension_v7.json (colocation / batching / prefetch
/ preemption enabled).  Read-only.
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
EPISODES = 10
CAP = 4


def main() -> int:
    from tracing.analysis.profile_contract import apply_real_workload_profile_contract
    from tracing.analysis.workload_v02_simulator import (
        load_resource_v2_overlay, load_templates, read_jsonl, train_resource_stats)

    v7 = json.loads(V7.read_text(encoding="utf-8"))
    templates = load_templates(
        ROOT / "results/processed/r7_workload_v041_ontology_no_run_container/job_templates_r7_v041.jsonl",
        topology_view="causal_v3")
    manifest = json.loads((ROOT / "data/manifests/validation_split_dev700_confirm300.json").read_text(encoding="utf-8"))
    wanted = set(manifest["confirm"])
    by_id = {str(e.get("episode_id")): e for e in read_jsonl(
        ROOT / "results/processed/r7_workload_v03_no_run_container/episodes/workload_validation_r7_v03.jsonl")}
    episodes = [by_id[eid] for eid in sorted(wanted) if eid in by_id][:EPISODES]
    templates, episodes = apply_real_workload_profile_contract(templates, episodes, v7)
    stats = train_resource_stats({k: v for k, v in templates.items() if v.split == "train"})
    artifacts, _ = load_resource_v2_overlay(
        ROOT / "outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts",
        ROOT / "outputs/resource_v2_artifacts/f0_seed11")

    states = T.capture_states(episodes, templates, stats, artifacts, v7, cap=CAP)
    print(f"captured states: {len(states)}")

    pool_sizes, node_counts, free_gpus, coloc_pairs = [], [], [], []
    checked = 0
    for ep, jobs_meta, snap in states:
        plan_episode = build_planning_episode(ep, snap, include_arrivals=True)
        _s, _e, pctx = run_planning(
            plan_episode, templates, future_artifacts=artifacts, train_stats=stats,
            extension_config=v7, policy="des_rollout", forced_action=None,
            collect_events=False, extra_context={"des_probe_pool": True})
        probe_now = float((pctx.get("des_first_state") or {}).get("now_ms", -1.0))
        if abs(probe_now - float(snap["now_ms"])) > 1e-6:
            continue
        pool = {(str(r[1]), int(r[2])) for r in (pctx.get("des_first_pool") or [])}
        pairs = pctx.get("des_pair_eval") or []
        checked += 1
        pool_sizes.append(len(pool))
        node_counts.append(len({n for n, _g in pool}))
        free_gpus.append(len({e["gpu"] for e in pairs if e.get("in_free_gpus")}))
        coloc_pairs.append(sum(
            1 for e in pairs
            if e.get("in_free_gpus") and e.get("can_dispatch") and e.get("has_active")
            and (str(e["node_id"]), int(e["gpu"])) in pool))

    def dist(values):
        out = {}
        for v in sorted(set(values)):
            out[str(v)] = values.count(v)
        return out

    report = {
        "environment": "v7_full",
        "states_checked": checked,
        "pool_pairs": {"mean": round(statistics.fmean(pool_sizes), 2) if pool_sizes else None,
                       "dist": dist(pool_sizes)},
        "distinct_nodes": {"mean": round(statistics.fmean(node_counts), 2) if node_counts else None,
                           "dist": dist(node_counts)},
        "free_gpus": {"mean": round(statistics.fmean(free_gpus), 2) if free_gpus else None,
                      "dist": dist(free_gpus)},
        "colocation_pairs": {"mean": round(statistics.fmean(coloc_pairs), 2) if coloc_pairs else None,
                             "dist": dist(coloc_pairs)},
    }
    print(json.dumps(report, indent=1, ensure_ascii=False))
    out = ROOT / "experiments/EXP-20261012_milp_v0/artifacts/legal_action_counts_v7.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("evidence:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
