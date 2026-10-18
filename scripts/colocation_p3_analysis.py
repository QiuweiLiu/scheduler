#!/usr/bin/env python3
"""Colocation P3 analysis: fit the three FROZEN dynamic models on T0 and validate
on the offset-start conditions T1/T2 (and T3 no-overlap control).

Frozen model definitions (advisor 2026-10-16 — do not alter after seeing results):
  piecewise: while both active, each task runs at its own rate; the survivor is
             reset to rate 1 when the peer completes.
  sticky:    a task keeps its acquired slowdown until completion (no reset).
  one-time:  co-location adds a fixed additive cost c to each request; rates
             never change (short = solo_short + c, long = solo_long + c).

Fitting: T0 only (sync start). Validation: T1/T2 predictions vs observed; T3 is
a no-overlap negative control (all models predict ~solo).  Reports mean / worst /
systematic-bias / spread per model per condition; the pre-registered engineering
reference max(3%, 30ms) is kept as a fixed yardstick and is NOT a pass threshold.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


def predict_piecewise(solo_short, solo_long, s_short, s_long, j):
    sf = j + solo_short * s_short
    if sf >= solo_long:
        return sf, j + (solo_long - j) * s_long
    consumed = (sf - j) / s_long
    return sf, sf + (solo_long - j - consumed)


def predict_sticky(solo_short, solo_long, s_short, s_long, j):
    sf = j + solo_short * s_short
    if sf >= solo_long:
        return sf, j + (solo_long - j) * s_long
    return sf, solo_long * s_long


def predict_onetime(solo_short, solo_long, c, j):
    return solo_short + c, solo_long + c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    recs = [json.loads(l) for l in args.logs.read_text(encoding="utf-8").splitlines() if l.strip()]
    report = {"schema": "colocation-p3-analysis", "pairs": {}, "notes": []}

    pairs = {}
    for r in recs:
        if r.get("pair"):
            pairs.setdefault(r["pair"], []).append(r)
        elif r.get("tag", "").startswith(tuple(PAIRS_KEYS := ("P-1", "P-2", "P-3"))):
            # solo/T3 records carry pair info via tag prefix
            pairs.setdefault(r["tag"].split("_")[0], []).append(r)

    for pair_id, rs in pairs.items():
        solos = {r["tag"]: r for r in rs if r.get("solo")}
        t0 = [r for r in rs if r.get("condition") == "T0"]
        t1 = [r for r in rs if r.get("condition") == "T1"]
        t2 = [r for r in rs if r.get("condition") == "T2"]
        t3 = [r for r in rs if r.get("condition") == "T3"]
        if not t0:
            continue
        solo_a = st.median([r["result"]["ms"] for r in solos.values() if r["tag"].endswith("solo_a")] or [0]) / 1000
        solo_b = st.median([r["result"]["ms"] for r in solos.values() if r["tag"].endswith("solo_b")] or [0]) / 1000
        # determine short/long sides from T0 finishes
        fins = [(r["a"]["t_gen_end_ns"] - r["a"]["t_gen_start_ns"]) / 1e9 for r in t0]
        fina = st.median(fins)
        finb = st.median([(r["b"]["t_gen_end_ns"] - r["b"]["t_gen_start_ns"]) / 1e9 for r in t0])
        short_side = "a" if fina < finb else "b"
        solo_short = solo_a if short_side == "a" else solo_b
        solo_long = solo_b if short_side == "a" else solo_a
        t0_short = min(fina, finb)
        t0_long = max(fina, finb)

        # --- fit on T0 ---
        s_short = t0_short / solo_short
        s_long_sticky = t0_long / solo_long
        # piecewise: t0_long = t_short + (solo_long - t_short / s_long)  =>  s_long = t_short / (t_short + solo_long - t0_long)
        denom = t0_short + solo_long - t0_long
        s_long_piecewise = t0_short / denom if denom > 1e-9 else float("nan")
        c_onetime = (t0_long - solo_long + t0_short - solo_short) / 2

        pair_report = {"short_side": short_side, "solo_short_s": solo_short, "solo_long_s": solo_long,
                       "fit": {"s_short": s_short, "s_long_sticky": s_long_sticky,
                               "s_long_piecewise": s_long_piecewise, "c_onetime_s": c_onetime},
                       "conditions": {}}

        for cond, rs_c in (("T1", t1), ("T2", t2)):
            rows = {"piecewise": [], "sticky": [], "onetime": []}
            for r in rs_c:
                j = r["offset_b_ms"] / 1000.0
                obs_short = (r[short_side]["t_gen_end_ns"] - r[short_side]["t_gen_start_ns"]) / 1e9
                long_side = "b" if short_side == "a" else "a"
                obs_long = (r[long_side]["t_gen_end_ns"] - r[long_side]["t_gen_start_ns"]) / 1e9
                p = predict_piecewise(solo_short, solo_long, s_short, s_long_piecewise, j)
                s = predict_sticky(solo_short, solo_long, s_short, s_long_sticky, j)
                o = predict_onetime(solo_short, solo_long, c_onetime, j)
                for name, pred in (("piecewise", p), ("sticky", s), ("onetime", o)):
                    rows[name].append({"obs_short": obs_short, "pred_short": pred[0],
                                       "obs_long": obs_long, "pred_long": pred[1],
                                       "err_long": pred[1] - obs_long,
                                       "within_ref": abs(pred[1] - obs_long) <= max(0.03 * obs_long, 0.030)})
            stats = {}
            for name, rr in rows.items():
                errs = [x["err_long"] for x in rr]
                stats[name] = {"mean_err_long_s": st.mean(errs) if errs else None,
                               "worst_err_long_s": max(errs, key=abs) if errs else None,
                               "systematic_bias_s": st.mean(errs) if errs else None,
                               "spread_s": (max(errs) - min(errs)) if len(errs) > 1 else 0.0,
                               "n_within_ref": sum(1 for x in rr if x["within_ref"]), "n": len(rr)}
            pair_report["conditions"][cond] = {"stats": stats,
                                               "rows": rows}

        if t3:
            obs_b = [(r["b"]["t_gen_end_ns"] - r["b"]["t_gen_start_ns"]) / 1e9 for r in t3]
            pair_report["conditions"]["T3"] = {"obs_s": obs_b, "solo_ref_s": solo_b if short_side == "a" else solo_a}

        report["pairs"][pair_id] = pair_report

    report["notes"].append("Engineering reference max(3%,30ms) is a fixed yardstick, NOT a pass threshold; "
                           "fit on T0 only; T1/T2/T3 never used for fitting.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for pair_id, pr in report["pairs"].items():
        print(f"{pair_id}: short={pr['short_side']} fit s_short={pr['fit']['s_short']:.3f} "
              f"s_long_pw={pr['fit']['s_long_piecewise']:.3f} s_long_stk={pr['fit']['s_long_sticky']:.3f} "
              f"c={pr['fit']['c_onetime_s']*1000:.1f}ms")
        for cond, cd in pr["conditions"].items():
            if cond == "T3":
                continue
            for name, s in cd["stats"].items():
                print(f"   {cond} {name:9s} mean_err_long={s['mean_err_long_s']:+.3f}s worst={s['worst_err_long_s']:+.3f}s "
                      f"within_ref={s['n_within_ref']}/{s['n']}")
    print("report:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
