#!/usr/bin/env python3
"""Measure the FULL action space with eviction + prefetch dimensions.

Per decision state (confirm split, serial_prefetch_v1 rules):
- dispatch: legal (node, gpu) pairs
- eviction: for pairs that need eviction, the number of evictable subsets
  (2^k - 1 over the evictable models on that GPU, node's own model excluded)
- prefetch: chain models x fitting GPUs (same filter as the MILP candidates)
- combined: dispatch x (1 + eviction options) x (1 + prefetch options)
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


def main() -> int:
    import milp_v0_performance as P

    config = json.loads((ROOT / "experiments/EXP-20261012_milp_v0/artifacts"
                         / "serial_prefetch_v1_extension.json").read_text(encoding="utf-8"))
    serial, templates, episodes, stats, artifacts = P.load_split("confirm", 20)
    states = T.capture_states(episodes, templates, stats, artifacts, config, cap=4)

    dispatch_counts, evict_opts, prefetch_opts, combined = [], [], [], []
    checked = 0
    for ep, jobs_meta, snap in states:
        plan_episode = build_planning_episode(ep, snap, include_arrivals=True)
        _s, _e, pctx = run_planning(
            plan_episode, templates, future_artifacts=artifacts, train_stats=stats,
            extension_config=config, policy="des_rollout", forced_action=None,
            collect_events=False, extra_context={"des_probe_pool": True})
        probe_now = float((pctx.get("des_first_state") or {}).get("now_ms", -1.0))
        if abs(probe_now - float(snap["now_ms"])) > 1e-6:
            continue
        pool = sorted({(str(r[1]), int(r[2])) for r in (pctx.get("des_first_pool") or [])})
        if not pool:
            continue
        checked += 1
        # eviction options per candidate pair (from the pair eval)
        pair_evict = {}
        for e in pctx.get("des_pair_eval") or []:
            key = (str(e["node_id"]), int(e["gpu"]))
            if key not in pool:
                continue
            evicted = e.get("evicted") or []
            if evicted:
                k = len(evicted)  # evictable count on that GPU for this pair
                pair_evict[key] = max(0, 2 ** k - 1)
            else:
                pair_evict[key] = 0
        # prefetch options: chain models not resident + fitting (same filter as MILP)
        obs = T.obs_from_snapshot(snap, jobs_meta, templates, stats, artifacts)
        model_load = {}
        for steps in obs.chains.values():
            for step in steps:
                if step.model_id:
                    model_load[str(step.model_id)] = max(
                        model_load.get(str(step.model_id), 0.0), float(step.load_ms))
        pf = 0
        for m, load_m in model_load.items():
            if load_m <= 0.0:
                continue
            mb = obs.resident_mb_by_model.get(m)
            if mb is None:
                continue
            for g in obs.gpus:
                if m in g.resident:
                    continue
                if sum(g.resident.values()) + float(mb) > g.capacity_mb + 1e-9:
                    continue
                pf += 1
        dispatch_counts.append(len(pool))
        ev = statistics.fmean(pair_evict.values()) if pair_evict else 0.0
        evict_opts.append(ev)
        prefetch_opts.append(pf)
        combined.append(len(pool) * (1.0 + ev) * (1.0 + pf))

    def dist(vals):
        return {str(int(v)): sum(1 for x in vals if int(x) == int(v)) for v in sorted(set(int(x) for x in vals))}

    report = {
        "schema": "full-action-space-v1",
        "environment": "serial_prefetch_v1",
        "states_checked": checked,
        "dispatch_pairs": {"mean": round(statistics.fmean(dispatch_counts), 2) if checked else None,
                           "dist": dist(dispatch_counts)},
        "eviction_subsets_per_pair_mean": round(statistics.fmean(evict_opts), 2) if checked else None,
        "prefetch_options": {"mean": round(statistics.fmean(prefetch_opts), 2) if checked else None,
                             "dist": dist(prefetch_opts)},
        "combined_actions_mean": round(statistics.fmean(combined), 1) if checked else None,
        "combined_actions_p90": round(sorted(combined)[int(0.9 * (len(combined) - 1))], 1) if checked else None,
        "combined_actions_max": round(max(combined), 1) if checked else None,
    }
    print(json.dumps(report, indent=1, ensure_ascii=False))
    out = ROOT / "experiments/EXP-20261012_milp_v0/artifacts/full_action_space.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("evidence:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
