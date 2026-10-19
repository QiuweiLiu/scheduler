#!/usr/bin/env python3
"""Colocation P3 analysis v2 — drift-robust per-round pairing.

Interleaved protocol: each round r carries its own solo_a/solo_b baselines plus
the T0/T1/T2/T3 conditions, so machine drift is shared across conditions.

Frozen model definitions (advisor 2026-10-16; fitted on T0 only):
  piecewise: while both active each runs at its own rate; survivor resets to 1.
             long finish (peer exits first) = solo_long + d_short*(1 - 1/s_long)
             where d_short = solo_short*s_short  (j-independent in that regime);
             if peer still active at long's finish: j + (solo_long - j)*s_long.
  sticky:    rate persists after peer exit:  j + (solo_long - j)*s_long.
  one-time:  fixed additive cost:  solo_long + c.

Reports per pair: per-condition drift-corrected ratios (obs/round-solo), T0-fit
parameters, T1/T2 prediction errors for each model (mean/worst, fixed yardstick
max(3%,30ms)), and the no-overlap T3 control.  Fit on T0; T1/T2/T3 never fit.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path


def predict(solo_short, solo_long, s_short, s_long_pw, s_long_stk, c, j, model):
    sf = j + solo_short * s_short
    if model == "piecewise":
        if sf >= solo_long:
            return j + (solo_long - j) * s_long_pw
        return solo_long + solo_short * s_short * (1.0 - 1.0 / s_long_pw)
    if model == "sticky":
        return j + (solo_long - j) * s_long_stk
    return solo_long + c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    recs = [json.loads(l) for l in args.logs.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = {"schema": "colocation-p3-analysis-v2", "pairs": {}}

    # group by pair and round
    rounds: dict = {}
    for r in recs:
        tag = r.get("tag", "")
        m = re.match(r"(P-\d)_(T\d)_r(\d+)$", tag)
        if m:
            rounds.setdefault(m.group(1), {}).setdefault(int(m.group(3)), {})[m.group(2)] = r
            continue
        m = re.match(r"(P-\d)_r(\d+)_solo_([ab])$", tag)
        if m:
            rounds.setdefault(m.group(1), {}).setdefault(int(m.group(2)), {})["solo_" + m.group(3)] = r
            continue
        m = re.match(r"(P-\d)_T3_r(\d+)_a_solo$", tag)
        if m:
            rounds.setdefault(m.group(1), {}).setdefault(int(m.group(2)), {})["T3_solo_a"] = r

    for pair, reps in sorted(rounds.items()):
        pr = {"n_rounds": len(reps), "rounds": {}, "fit": {}, "conditions": {}}

        def dur(rec, side):
            return (rec[side]["t_gen_end_ns"] - rec[side]["t_gen_start_ns"]) / 1e9

        # short side from T0 medians
        t0s = [d for d in reps.values() if "T0" in d and "solo_a" in d]
        if not t0s:
            continue
        med_a = st.median([dur(d["T0"], "a") for d in t0s])
        med_b = st.median([dur(d["T0"], "b") for d in t0s])
        short_side, long_side = ("b", "a") if med_b <= med_a else ("a", "b")

        # ---- per-round fit on T0 ----
        fit_rows = []
        for rep, d in sorted(reps.items()):
            if not all(k in d for k in ("T0", "solo_a", "solo_b")):
                continue
            sa = d["solo_a"]["result"]["ms"] / 1e3
            sb = d["solo_b"]["result"]["ms"] / 1e3
            solo_short = sa if short_side == "a" else sb
            solo_long = sb if short_side == "a" else sa
            t_short = dur(d["T0"], short_side)
            t_long = dur(d["T0"], long_side)
            s_short = t_short / solo_short
            s_long_stk = t_long / solo_long
            denom = t_short + solo_long - t_long
            s_long_pw = t_short / denom if denom > 1e-9 else float("nan")
            c = ((t_long - solo_long) + (t_short - solo_short)) / 2.0
            fit_rows.append({"rep": rep, "s_short": s_short, "s_long_stk": s_long_stk,
                             "s_long_pw": s_long_pw, "c": c})
        fit = {k: st.median([r[k] for r in fit_rows]) for k in ("s_short", "s_long_stk", "s_long_pw", "c")} if fit_rows else {}
        pr["fit"] = {"short_side": short_side, "rows": fit_rows, **{k: round(v, 4) for k, v in fit.items()}}

        # ---- per-condition drift-corrected ratios + model validation ----
        for cond in ("T0", "T1", "T2"):
            rows = []
            for rep, d in sorted(reps.items()):
                if cond not in d or "solo_a" not in d:
                    continue
                sa = d["solo_a"]["result"]["ms"] / 1e3
                sb = d["solo_b"]["result"]["ms"] / 1e3
                solo_short = sa if short_side == "a" else sb
                solo_long = sb if short_side == "a" else sa
                rec = d[cond]
                j = rec.get("offset_b_ms", 0.0) / 1e3
                obs_short, obs_long = dur(rec, short_side), dur(rec, long_side)
                row = {"rep": rep, "j_s": j,
                       "ratio_short": obs_short / solo_short, "ratio_long": obs_long / solo_long,
                       "obs_short": obs_short, "obs_long": obs_long,
                       "first_finisher": rec.get("first_finisher")}
                if fit:
                    for model in ("piecewise", "sticky", "onetime"):
                        pred = predict(solo_short, solo_long, fit["s_short"], fit["s_long_pw"],
                                       fit["s_long_stk"], fit["c"], j, model)
                        row[f"err_{model}"] = pred - obs_long
                        row[f"ref_{model}"] = abs(pred - obs_long) <= max(0.03 * obs_long, 0.030)
                rows.append(row)
            stats = {}
            for model in ("piecewise", "sticky", "onetime"):
                errs = [r[f"err_{model}"] for r in rows if f"err_{model}" in r]
                if errs:
                    stats[model] = {"mean_err_s": round(st.mean(errs), 3),
                                    "worst_abs_s": round(max(errs, key=abs), 3),
                                    "n_within_ref": sum(1 for r in rows if r.get(f"ref_{model}")),
                                    "n": len(errs)}
            pr["conditions"][cond] = {
                "ratio_short_median": round(st.median([r["ratio_short"] for r in rows]), 3) if rows else None,
                "ratio_long_median": round(st.median([r["ratio_long"] for r in rows]), 3) if rows else None,
                "stats": stats, "rows": rows}

        # ---- T3 control ----
        t3b = [d["T3"]["b"]["ms"] / 1e3 for d in reps.values() if "T3" in d]
        t3sa = [d["T3_solo_a"]["result"]["ms"] / 1e3 for d in reps.values() if "T3_solo_a" in d]
        t3sb = [d["solo_b"]["result"]["ms"] / 1e3 for d in reps.values() if "solo_b" in d]
        if t3b and t3sb:
            pr["T3_control"] = {"b_ratio_median": round(st.median([b / (st.median(t3sb)) for b in t3b]), 3),
                                "b_ms": [round(x * 1e3) for x in t3b],
                                "a_solo_ms": [round(x * 1e3) for x in t3sa]}
        out["pairs"][pair] = pr

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for pair, pr in out["pairs"].items():
        print(f"{pair}: short={pr['fit'].get('short_side')} fit s_short={pr['fit'].get('s_short')} "
              f"s_long_pw={pr['fit'].get('s_long_pw')} s_long_stk={pr['fit'].get('s_long_stk')} c={pr['fit'].get('c')}")
        for cond in ("T0", "T1", "T2"):
            cd = pr["conditions"].get(cond)
            if not cd:
                continue
            print(f"   {cond}: ratio_short={cd['ratio_short_median']} ratio_long={cd['ratio_long_median']}")
            for model, s in cd["stats"].items():
                print(f"      {model:9s} mean_err={s['mean_err_s']:+.3f}s worst={s['worst_abs_s']:+.3f}s within_ref={s['n_within_ref']}/{s['n']}")
        if "T3_control" in pr:
            print(f"   T3 control: b_ratio={pr['T3_control']['b_ratio_median']}")
    print("report:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
