#!/usr/bin/env python3
"""Advisor P0: diagnose the 11 formerly-illegal MILP first actions, one by one.

For each case (identified in the pre-mask batch evidence): reproduce the
unmasked choice, then report the real GPU state, the DES candidate pool, the
per-pair dispatch/admission evaluation, and the MILP-predicted availability;
classify the failure reason (device availability / predicted-time mismatch /
dispatch rule / memory admission / pool filtering).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import milp_v0_tests as T  # noqa: E402
from tracing.analysis.des_rollout import build_planning_episode, run_planning  # noqa: E402
from tracing.scheduling.milp_v0 import solve_v0a  # noqa: E402

DEFAULT_V2 = ROOT / "experiments/EXP-20261012_milp_v0/artifacts/batch_validation_v2_premask.json"
EVIDENCE = ROOT / "experiments/EXP-20261012_milp_v0/artifacts/illegal_action_diagnosis.json"
EPS = 1e-6


def classify_case(chosen, entry, obs_avail, probe_now, snap_now, moment_match, in_pool):
    if not moment_match:
        return "probe_moment_mismatch"
    if entry is None:
        return "no_pair_eval"
    if not entry["in_free_gpus"]:
        if entry["busy_until"] > probe_now + EPS:
            return "device_busy_prediction_mismatch"
        if entry["has_active"]:
            return "device_serial_rule_rejected"
        return "device_not_in_free_set_other"
    if entry.get("can_dispatch") is False:
        return "dispatch_rule_rejected"
    if entry.get("admitted") is False or entry.get("evicted"):
        return "memory_admission_rejected"
    if in_pool is False:
        return "pool_filtered"
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-evidence", type=Path, default=DEFAULT_V2)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--cap", type=int, default=4)
    parser.add_argument("--budget", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=EVIDENCE)
    args = parser.parse_args()

    v2 = json.load(open(args.v2_evidence, encoding="utf-8"))
    illegal_rows = [r for r in v2["rows"] if not r["fallback"] and r.get("first_in_des_pool") is False]
    print(f"pre-mask illegal first actions: {len(illegal_rows)}")

    serial, templates, episodes, stats, artifacts = T.load_fixtures(args.episodes)
    states = T.capture_states(episodes, templates, stats, artifacts, serial, cap=args.cap)
    index = {(ep["episode_id"], round(float(snap["now_ms"]), 6)): (ep, jobs_meta, snap)
             for ep, jobs_meta, snap in states}

    cases = []
    for row in illegal_rows:
        key = (row["episode"], round(float(row["now_ms"]), 6))
        if key not in index:
            cases.append({"episode": row["episode"], "now_ms": row["now_ms"], "reason": "state_not_found"})
            continue
        ep, jobs_meta, snap = index[key]
        snap_now = float(snap["now_ms"])
        obs = T.obs_from_snapshot(snap, jobs_meta, templates, stats, artifacts)
        res = solve_v0a(obs, time_limit_s=args.budget)
        chosen = tuple(res.first_action) if res.first_action else None
        plan_episode = build_planning_episode(ep, snap, include_arrivals=True)
        _s, _e, pctx = run_planning(
            plan_episode, templates, future_artifacts=artifacts, train_stats=stats,
            extension_config=serial, policy="des_rollout", forced_action=None,
            collect_events=False, extra_context={"des_probe_pool": True})
        probe_now = float((pctx.get("des_first_state") or {}).get("now_ms", -1.0))
        moment_match = abs(probe_now - snap_now) <= EPS
        pool = {(str(r[1]), int(r[2])) for r in (pctx.get("des_first_pool") or [])}
        in_pool = (chosen in pool) if chosen is not None else None
        entry = None
        if chosen is not None:
            entry = next((e for e in (pctx.get("des_pair_eval") or [])
                          if str(e["node_id"]) == str(chosen[0]) and int(e["gpu"]) == int(chosen[1])), None)
        obs_avail = {int(g.index): float(g.available_at_ms) for g in obs.gpus}
        gpus_real = (pctx.get("des_first_state") or {}).get("gpus") or []
        reason = classify_case(chosen, entry, obs_avail, probe_now, snap_now, moment_match, in_pool)
        cases.append({
            "episode": row["episode"], "now_ms": snap_now,
            "change_type": row.get("change_type"),
            "milp_action": list(chosen) if chosen else None,
            "reproduced_v2": (chosen is not None and list(chosen) == list(row["first_action"])),
            "baseline_action": row.get("baseline_action"),
            "probe_now": probe_now, "moment_match": moment_match,
            "in_same_moment_pool": in_pool,
            "pool_pairs": sorted(pool),
            "chosen_pair_eval": entry,
            "obs_predicted_available": obs_avail,
            "real_gpus": gpus_real,
            "reason": reason,
        })

    counts: dict = {}
    for c in cases:
        counts[c["reason"]] = counts.get(c["reason"], 0) + 1
    report = {
        "schema": "milp-v0a-illegal-diag-v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cases": len(cases), "reason_counts": counts, "details": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=1, ensure_ascii=False))
    for c in cases:
        e = c.get("chosen_pair_eval") or {}
        print(f"- {c['episode']}@{c['now_ms']:.0f} {c.get('change_type')} milp={c.get('milp_action')} "
              f"repro={c.get('reproduced_v2')} pool={len(c.get('pool_pairs') or [])} "
              f"free={e.get('in_free_gpus')} busy_until={e.get('busy_until')} "
              f"obs_avail={c.get('obs_predicted_available')} -> {c['reason']}")
    print(f"evidence: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
