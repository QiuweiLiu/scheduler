#!/usr/bin/env python3
"""Colocation Phase P2 (advisor 2026-10-16): independent semantics diagnostic.

Compares two candidate rate semantics WITHOUT touching the official v7 config or
the main simulator:

  piecewise-rate (current DES): while both tasks are active each runs at its own
      slowdown; when the peer completes the survivor is reset to rate 1.
  sticky-rate (diagnostic candidate): a task keeps the slowdown it acquired on
      co-location until it completes; a peer's exit does NOT restore rate 1.

Boundary cases: synchronous start (fits the frozen table by identity), late join
of the short task, no-overlap control, short-task-ends-early, and the
unresolved sequential-different-peers case (sticky has no measured rule there).

Output: endpoint tables for both models; no calibration claim is made for
sticky-rate — synchronous-start fitting of the same table is NOT physical
calibration.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "experiments/EXP-20261004_real_strata_calibration_v1/artifacts"
OUT = ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/colocation_p2_semantics_diag.json"


def med(xs):
    return float(st.median(xs))


def simulate(short_solo, long_solo, s_short, s_long, join_ms, mode):
    """Two-task GPU interval with the short task joining at join_ms (>=0).

    Returns (short_finish, long_finish) or None if no overlap occurs.
    """
    if join_ms >= long_solo:
        return None  # no-overlap control
    short_finish = join_ms + short_solo * s_short
    if short_finish >= long_solo:
        # overlap spans to long completion; long keeps its rate the whole way
        long_finish = join_ms + (long_solo - join_ms) * s_long
        return short_finish, long_finish
    if mode == "piecewise":
        consumed = (short_finish - join_ms) / s_long
        long_finish = short_finish + (long_solo - join_ms - consumed)
    else:  # sticky
        consumed = (short_finish - join_ms) / s_long
        long_finish = short_finish + (long_solo - join_ms - consumed) * s_long
    return short_finish, long_finish


def main() -> int:
    cal = json.loads((CAL / "real_strata_calibration.json").read_text(encoding="utf-8"))
    a1 = json.loads((ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/colocation_a1_micro.json").read_text(encoding="utf-8"))
    a1_by_id = {c["cell_id"]: c for c in a1["cells"]}

    out = {"schema": "colocation-p2-semantics-diag", "sync_start": [], "late_join_grid": [],
           "notes": {
               "sticky_identity": "sticky long-side finish = solo*slowdown = the frozen median coloc value by definition; "
                                  "zero deviation is an identity, not calibration evidence.",
               "sequential_peers": "sticky-rate has NO measured rule for a task meeting two different peers in sequence "
                                   "(A+B then A+C): the 11-cell table contains no chained co-location or start-offset data.",
           }}

    for c in cal["cells"]["A"]:
        cid = c["cell_id"]
        ma, mb = med([float(x) for x in c["solo_a_ms"]]), med([float(x) for x in c["solo_b_ms"]])
        sa, sb = c["slowdown_a"], c["slowdown_b"]
        if ma * sa <= mb * sb:
            short_solo, long_solo, s_short, s_long, short_side = ma, mb, sa, sb, "a"
        else:
            short_solo, long_solo, s_short, s_long, short_side = mb, ma, sb, sa, "b"
        pw = simulate(short_solo, long_solo, s_short, s_long, 0.0, "piecewise")
        stk = simulate(short_solo, long_solo, s_short, s_long, 0.0, "sticky")
        meas_long = med([float(x) for x in c[f"coloc_{'b' if short_side == 'a' else 'a'}_ms"]])
        rec = {"cell_id": cid, "short_side": short_side,
               "piecewise_long_finish": round(pw[1], 1), "sticky_long_finish": round(stk[1], 1),
               "measured_long_median": meas_long,
               "piecewise_dev": round(pw[1] - meas_long, 1), "sticky_dev": round(stk[1] - meas_long, 1)}
        out["sync_start"].append(rec)

    # late-join grid on two representative pairs
    for cid, grid in (("A_8B[answer_generation]|4B[planner]", [0, 50, 100, 200, 400, 800, 1400]),
                      ("A_8B[planner]|4B[planner]", [0, 200, 600, 1200, 1800, 2200])):
        c = next(x for x in cal["cells"]["A"] if x["cell_id"] == cid)
        ma, mb = med([float(x) for x in c["solo_a_ms"]]), med([float(x) for x in c["solo_b_ms"]])
        sa, sb = c["slowdown_a"], c["slowdown_b"]
        if ma * sa <= mb * sb:
            short_solo, long_solo, s_short, s_long = ma, mb, sa, sb
        else:
            short_solo, long_solo, s_short, s_long = mb, ma, sb, sa
        rows = []
        for j in grid:
            pw = simulate(short_solo, long_solo, s_short, s_long, float(j), "piecewise")
            stk = simulate(short_solo, long_solo, s_short, s_long, float(j), "sticky")
            rows.append({"join_ms": j,
                         "piecewise_long_finish": None if pw is None else round(pw[1], 1),
                         "sticky_long_finish": None if stk is None else round(stk[1], 1),
                         "divergence_ms": None if pw is None else round(stk[1] - pw[1], 1)})
        out["late_join_grid"].append({"cell_id": cid, "rows": rows})

    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for r in out["sync_start"]:
        print(f"{r['cell_id'][:44]:44s} piecewise {r['piecewise_dev']:>+7.1f} sticky {r['sticky_dev']:>+6.1f} "
              f"(long {r['short_side']}->meas {r['measured_long_median']:.0f})")
    print("late-join divergence example:",
          out["late_join_grid"][0]["rows"][0], "->", out["late_join_grid"][0]["rows"][-1])
    print("evidence:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
