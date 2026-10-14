#!/usr/bin/env python3
"""Profile where the MILP decision time goes (confirm-split states).

Phases: observation build / node build / reference schedule / model assembly
(rows + sparse matrix, derived) / HiGHS solve (patched scipy.optimize.milp) /
extraction+rest (derived).  Read-only instrumentation; no solver changes.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import scipy.optimize as so  # noqa: E402

import milp_v0_tests as T  # noqa: E402
import tracing.scheduling.milp_v0 as M  # noqa: E402

BUDGET = 0.05
EPISODES = 8
CAP = 4

timers = {"obs_build": 0.0, "nodes": 0.0, "reference": 0.0, "highs": 0.0}
counts = {"states": 0}

_real_milp = so.milp
_real_build_nodes = M._build_nodes
_real_reference = M.build_reference_schedule
_real_obs = M.build_milpo_observation


def timed_milp(*a, **k):
    t0 = time.perf_counter()
    r = _real_milp(*a, **k)
    timers["highs"] += time.perf_counter() - t0
    return r


def timed_build_nodes(*a, **k):
    t0 = time.perf_counter()
    r = _real_build_nodes(*a, **k)
    timers["nodes"] += time.perf_counter() - t0
    return r


def timed_reference(*a, **k):
    t0 = time.perf_counter()
    r = _real_reference(*a, **k)
    timers["reference"] += time.perf_counter() - t0
    return r


def timed_obs(*a, **k):
    t0 = time.perf_counter()
    r = _real_obs(*a, **k)
    timers["obs_build"] += time.perf_counter() - t0
    return r


so.milp = timed_milp
M._build_nodes = timed_build_nodes
M.build_reference_schedule = timed_reference
M.build_milpo_observation = timed_obs


def main() -> int:
    from tracing.analysis.workload_v02_simulator import simulate_episode

    import milp_v0_performance as P

    serial, templates, episodes, stats, artifacts = P.load_split("confirm", EPISODES)
    states = T.capture_states(episodes, templates, stats, artifacts, serial, cap=CAP)
    print(f"states: {len(states)}")

    per_state_total = []
    for ep, jobs_meta, snap in states:
        obs = T.obs_from_snapshot(snap, jobs_meta, templates, stats, artifacts)
        if not [r for r in obs.ready if r.lane == "gpu"]:
            continue
        t0 = time.perf_counter()
        res = M.solve_v0a(obs, time_limit_s=BUDGET)
        total = time.perf_counter() - t0
        per_state_total.append(total)
        counts["states"] += 1

    n = max(1, counts["states"])
    breakdown = {k: round(v / n * 1000, 2) for k, v in timers.items()}
    assembly = statistics.fmean(per_state_total) * 1000 - sum(breakdown.values())
    report = {
        "budget_s": BUDGET,
        "states": counts["states"],
        "total_ms_mean": round(statistics.fmean(per_state_total) * 1000, 2),
        "total_ms_p50": round(statistics.median(per_state_total) * 1000, 2),
        "total_ms_p95": round(sorted(per_state_total)[int(0.95 * (len(per_state_total) - 1))] * 1000, 2),
        "phase_ms_mean": {**breakdown, "assembly_extract_rest": round(assembly, 2)},
    }
    print(json.dumps(report, indent=1, ensure_ascii=False))
    out = ROOT / "experiments/EXP-20261012_milp_v0/artifacts/milp_time_profile.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("evidence:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
