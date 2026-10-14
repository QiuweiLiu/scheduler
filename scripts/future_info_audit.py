#!/usr/bin/env python3
"""Future-information value audit (diagnostic only).

Per same-moment decision state (confirm split, serial_v0 config):
- legal first actions = DES candidate pool at the decision moment;
- choosers on the SAME pool: PDRS/GRU (baseline), historical prior, masked MILP;
- true-future cost of each pool action = forced-action rollout with the real
  episode continuation (main line afterwards);
- PDRS Headroom = cost(PDRS choice) - min(true costs over the pool);
- per-chooser regret and capture ratio; stratified by pool size / spread /
  memory pressure / eviction need / prefetch opportunity.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import milp_v0_tests as T  # noqa: E402
from tracing.analysis.des_rollout import build_planning_episode, run_planning  # noqa: E402


def mean(vals):
    return round(statistics.fmean(vals), 1) if vals else None


def main() -> int:
    import milp_v0_performance as P
    from tracing.analysis.pdrs_methods import build_prior_artifacts
    from tracing.scheduling.milp_v0 import solve_v0a

    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--cap", type=int, default=4)
    ap.add_argument("--budget", type=float, default=0.05)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "experiments/EXP-20261012_milp_v0/artifacts/future_info_audit.json")
    args = ap.parse_args()

    serial, templates, episodes, stats, artifacts = P.load_split("confirm", args.episodes)
    prior_pack = build_prior_artifacts(artifacts, all_steps=True)
    states = T.capture_states(episodes, templates, stats, artifacts, serial, cap=args.cap)
    print(f"captured states: {len(states)}")

    rows = []
    for ep, jobs_meta, snap in states:
        plan_episode = build_planning_episode(ep, snap, include_arrivals=True)
        _s, _e, pctx = run_planning(
            plan_episode, templates, future_artifacts=artifacts, train_stats=stats,
            extension_config=serial, policy="des_rollout", forced_action=None,
            collect_events=False,
            extra_context={"des_probe_pool": True, "_residency_prior_pack": prior_pack})
        probe_now = float((pctx.get("des_first_state") or {}).get("now_ms", -1.0))
        if abs(probe_now - float(snap["now_ms"])) > 1e-6:
            continue
        pool = sorted({(str(r[1]), int(r[2])) for r in (pctx.get("des_first_pool") or [])})
        if not pool:
            continue
        baseline = tuple(pctx.get("des_first_baseline") or ())
        prior_choice = tuple(pctx.get("des_first_baseline_prior") or ())
        obs = T.obs_from_snapshot(snap, jobs_meta, templates, stats, artifacts)
        mask = frozenset(pool)
        from dataclasses import replace as _replace
        res = solve_v0a(_replace(obs, legal_first=mask), time_limit_s=args.budget)
        milp_choice = tuple(res.first_action) if (not res.fallback and res.first_action) else None

        costs = {}
        for node, gpu in pool:
            summary, _ev, _c = run_planning(
                plan_episode, templates, future_artifacts=artifacts, train_stats=stats,
                extension_config=serial, policy="des_rollout",
                forced_action={"node_key": node, "gpu_index": gpu}, collect_events=False)
            costs[(node, gpu)] = float(summary["mean_completion_ms"])
        best_cost = min(costs.values())
        base_cost = costs.get(baseline)
        row = {
            "episode": ep["episode_id"], "now_ms": snap["now_ms"],
            "pool_size": len(pool), "costs": {f"{n}|{g}": round(c, 1) for (n, g), c in costs.items()},
            "baseline_action": list(baseline) if baseline else None,
            "prior_action": list(prior_choice) if prior_choice else None,
            "milp_action": list(milp_choice) if milp_choice else None,
            "baseline_cost": round(base_cost, 1) if base_cost is not None else None,
            "best_cost": round(best_cost, 1),
            "headroom": round(base_cost - best_cost, 1) if base_cost is not None else None,
            "spread": round(max(costs.values()) - best_cost, 1),
            "baseline_is_best": (base_cost is not None and abs(base_cost - best_cost) < 1e-9),
            "mem_pressure": round(max(
                (sum(g.resident.values()) / max(1.0, g.capacity_mb)) for g in obs.gpus), 3),
        }
        for name, choice in (("prior", prior_choice), ("milp", milp_choice)):
            if choice and choice in costs:
                row[f"{name}_cost"] = round(costs[choice], 1)
                row[f"{name}_regret"] = round(costs[choice] - best_cost, 1)
                row[f"{name}_capture"] = (
                    round((base_cost - costs[choice]) / (base_cost - best_cost), 3)
                    if base_cost is not None and (base_cost - best_cost) > 1e-9 else None
                )
        # eviction need for the baseline pair
        ev = next((e for e in (pctx.get("des_pair_eval") or [])
                   if (str(e["node_id"]), int(e["gpu"])) == baseline), None)
        row["baseline_needs_eviction"] = bool(ev and ev.get("evicted"))
        rows.append(row)

    # aggregates
    heads = [r["headroom"] for r in rows if r["headroom"] is not None]
    nonzero = [h for h in heads if h > 1e-9]
    report = {
        "schema": "future-info-audit-v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": "serial_v0",
        "split": "confirm", "episodes_captured": args.episodes,
        "states": len(rows),
        "headroom": {
            "mean": mean(heads), "mean_nonzero": mean(nonzero),
            "p50": round(statistics.median(heads), 1) if heads else None,
            "p90": round(sorted(heads)[int(0.9 * (len(heads) - 1))], 1) if heads else None,
            "zero_frac": round(sum(1 for h in heads if h <= 1e-9) / len(heads), 3) if heads else None,
        },
        "baseline_is_best_frac": round(sum(1 for r in rows if r["baseline_is_best"]) / len(rows), 3),
        "chooser_regret": {},
        "strata": {},
    }
    for name in ("prior", "milp"):
        regs = [r[f"{name}_regret"] for r in rows if f"{name}_regret" in r]
        caps = [r[f"{name}_capture"] for r in rows if r.get(f"{name}_capture") is not None]
        bests = sum(1 for r in rows if r.get(f"{name}_cost") is not None
                    and abs(r[f"{name}_cost"] - r["best_cost"]) < 1e-9)
        report["chooser_regret"][name] = {
            "covered_states": len(regs), "mean_regret": mean(regs),
            "median_regret": round(statistics.median(regs), 1) if regs else None,
            "picked_best_frac": round(bests / len(regs), 3) if regs else None,
            "mean_capture": round(statistics.fmean(caps), 3) if caps else None,
        }
    # strata by pool size and eviction need
    for key in ("pool_size", "baseline_needs_eviction"):
        groups = {}
        for r in rows:
            g = str(r[key])
            groups.setdefault(g, []).append(r["headroom"])
        report["strata"][key] = {g: {"n": len(v), "headroom_mean": mean(v)} for g, v in groups.items()}
    # spread terciles
    sp = sorted(r["spread"] for r in rows)
    if sp:
        t1, t2 = sp[len(sp) // 3], sp[2 * len(sp) // 3]
        for label, lo, hi in (("low", -1, t1), ("mid", t1, t2), ("high", t2, 1e18)):
            vals = [r["headroom"] for r in rows if lo < r["spread"] <= hi]
            report["strata"].setdefault("spread_tercile", {})[label] = {
                "n": len(vals), "headroom_mean": mean(vals)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": report, "rows": rows}, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print("evidence:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
