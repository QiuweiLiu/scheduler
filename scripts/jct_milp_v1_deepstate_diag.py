#!/usr/bin/env python3
"""Deep-state MILP scalability diagnostic (formal workload, r150_b15 dev).

One deep-queue state (0.6 x makespan of ep00): model size, status, reason,
incumbent presence, and time-to-solution at 0.5/2/5/20s for old v0a and new
solve_v1 (K=3 and K=1).
"""

from __future__ import annotations

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
from tracing.scheduling.milp_v0 import solve_v0a, solve_v1  # noqa: E402

FORMAL = ROOT / "experiments/EXP-20261014_burst64_v1/high_contention"


def main() -> int:
    config_path = ROOT / "experiments/EXP-20261012_milp_v0/artifacts/serial_prefetch_v1_extension.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _serial, templates, _n, stats, artifacts = P.load_split("confirm", 1, config_path)
    tm = {k: replace(v, split="train") for k, v in templates.items()}
    episodes = read_jsonl(FORMAL / "workload_burst64_r150_b15_train.jsonl")[:1]
    for row in episodes:
        row["split"] = "train"
    tm, episodes = apply_real_workload_profile_contract(tm, episodes, config)

    ep = episodes[0]
    summary, _ev = simulate_episode(ep, tm, "pdrs_resident", train_stats=stats,
                                    future_artifacts=artifacts, future_horizon=5,
                                    extension_config=config,
                                    policy_context={"activation": {}}, collect_events=False)
    at = 0.6 * float(summary["makespan_ms"])
    capture = {"at_ms_list": [round(at, 1)], "snapshots": []}
    simulate_episode(ep, tm, "pdrs_resident", train_stats=stats,
                     future_artifacts=artifacts, future_horizon=5, extension_config=config,
                     policy_context={"activation": {}, "des_capture": capture}, collect_events=False)
    snap = capture["snapshots"][0]
    jobs_meta = [{"job_instance_id": j["job_instance_id"], "template_id": j["template_id"]}
                 for j in ep["jobs"]]
    plan_episode = build_planning_episode(ep, snap, include_arrivals=True)
    _s, _e, pctx = run_planning(plan_episode, tm, future_artifacts=artifacts, train_stats=stats,
                                extension_config=config, policy="des_rollout", forced_action=None,
                                collect_events=False, extra_context={"des_probe_pool": True})
    pool = sorted({(str(r[1]), int(r[2])) for r in (pctx.get("des_first_pool") or [])})
    obs = T.obs_from_snapshot(snap, jobs_meta, tm, stats, artifacts)
    masked = replace(obs, legal_first=frozenset(pool))
    print(f"state: ep={ep['episode_id']} now={round(snap['now_ms'])} pool={len(pool)}")

    out = {"state": {"episode": ep["episode_id"], "now_ms": snap["now_ms"], "pool": len(pool)}, "runs": []}
    for label, fn, tl in (
        ("old_v0a@0.5", lambda tl: solve_v0a(masked, time_limit_s=tl, debug_dump=dump), 0.5),
        ("new_k3@0.5", lambda tl: solve_v1(masked, tail=True, time_limit_s=tl, debug_dump=dump), 0.5),
        ("new_k3@2", lambda tl: solve_v1(masked, tail=True, time_limit_s=tl, debug_dump=dump), 2.0),
        ("new_k3@5", lambda tl: solve_v1(masked, tail=True, time_limit_s=tl, debug_dump=dump), 5.0),
        ("new_k3@20", lambda tl: solve_v1(masked, tail=True, time_limit_s=tl, debug_dump=dump), 20.0),
        ("new_k1@0.5", lambda tl: solve_v1(masked, k_chains=1, tail=True, time_limit_s=tl, debug_dump=dump), 0.5),
        ("new_k1@2", lambda tl: solve_v1(masked, k_chains=1, tail=True, time_limit_s=tl, debug_dump=dump), 2.0),
        ("new_k1@5", lambda tl: solve_v1(masked, k_chains=1, tail=True, time_limit_s=tl, debug_dump=dump), 5.0),
    ):
        dump: dict = {}
        t0 = time.perf_counter()
        res = fn(tl)
        wall = (time.perf_counter() - t0) * 1000.0
        entry = {"label": label, "status": res.status, "reason": res.reason,
                 "fallback": bool(res.fallback), "action": list(res.first_action) if res.first_action else None,
                 "wall_ms": round(wall, 1), "solve_ms": round(res.solve_ms, 1) if res.solve_ms else None,
                 "gap": round(res.gap, 4) if res.gap is not None else None,
                 "n_vars": dump.get("n_vars"), "n_rows": len(dump.get("rows") or []),
                 "n_nodes": len(dump.get("nodes") or []), "has_x": bool(dump.get("values"))}
        out["runs"].append(entry)
        print(json.dumps(entry, ensure_ascii=False))
    outp = ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/deepstate_scalability.json"
    outp.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("evidence:", outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
