#!/usr/bin/env python3
"""Count legal first-action options per decision on confirm-split states.

Legal = (node, gpu) pairs in the DES candidate pool at the decision moment
(same-moment check); v0a-legal = pairs admitted WITHOUT eviction.  Read-only.
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

    serial, templates, episodes, stats, artifacts = P.load_split("confirm", 10)
    states = T.capture_states(episodes, templates, stats, artifacts, serial, cap=4)
    pool_sizes, fitting_sizes, node_counts = [], [], []
    checked = 0
    for ep, jobs_meta, snap in states:
        plan_episode = build_planning_episode(ep, snap, include_arrivals=True)
        _s, _e, pctx = run_planning(
            plan_episode, templates, future_artifacts=artifacts, train_stats=stats,
            extension_config=serial, policy="des_rollout", forced_action=None,
            collect_events=False, extra_context={"des_probe_pool": True})
        probe_now = float((pctx.get("des_first_state") or {}).get("now_ms", -1.0))
        if abs(probe_now - float(snap["now_ms"])) > 1e-6:
            continue  # no dispatchable moment at the snapshot time
        pool = {(str(r[1]), int(r[2])) for r in (pctx.get("des_first_pool") or [])}
        fitting = {
            (str(e["node_id"]), int(e["gpu"]))
            for e in (pctx.get("des_pair_eval") or [])
            if e.get("in_free_gpus") and e.get("can_dispatch") and e.get("admitted") is True
            and not e.get("evicted")
        }
        checked += 1
        pool_sizes.append(len(pool))
        fitting_sizes.append(len(fitting))
        node_counts.append(len({n for n, _g in pool}))

    def dist(values):
        if not values:
            return {}
        out = {}
        for v in sorted(set(values)):
            out[str(v)] = values.count(v)
        return out

    report = {
        "states_checked": checked,
        "pool_pairs": {"mean": round(statistics.fmean(pool_sizes), 2) if pool_sizes else None,
                       "dist": dist(pool_sizes)},
        "fitting_pairs": {"mean": round(statistics.fmean(fitting_sizes), 2) if fitting_sizes else None,
                          "dist": dist(fitting_sizes)},
        "distinct_nodes_in_pool": {"mean": round(statistics.fmean(node_counts), 2) if node_counts else None,
                                   "dist": dist(node_counts)},
    }
    print(json.dumps(report, indent=1, ensure_ascii=False))
    out = ROOT / "experiments/EXP-20261012_milp_v0/artifacts/legal_action_counts.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("evidence:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
