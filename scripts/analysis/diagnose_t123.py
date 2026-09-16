"""Diagnose T1/T2/T3 on the real traces: fixed-effect residuals, cell structure, correlations."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, r"F:\scheduler\scripts")
from verify_trace_dependence import (  # noqa: E402
    load_steps,
    variance_components_closed_form,
)

templates = Path(r"F:\scheduler\results\processed\r7_workload_20260817\job_templates_r7_v02.jsonl")
frame = load_steps(templates, "model_id", "success")
print("steps:", len(frame), "runs:", frame["run_id"].nunique(), "videos:", frame["video"].nunique())
print("type counts:", frame["type"].value_counts().to_dict())

# ---- T1 on raw y (what the smoke did) and on residuals
raw = variance_components_closed_form(frame)
print("\nT1 raw y:", {k: round(v, 4) for k, v in raw.items()})

import statsmodels.formula.api as smf

ols = smf.ols("y ~ C(type) + C(stack) + C(baseline) + C(stack):C(baseline)", data=frame).fit()
print("OLS R2:", round(ols.rsquared, 3))
resid_frame = frame.copy()
resid_frame["y"] = ols.resid
t1 = variance_components_closed_form(resid_frame)
print("T1 on residuals:", {k: round(v, 4) for k, v in t1.items()})

# ---- residual variance decomposition without video level (run only)
run_means = resid_frame.groupby("run_id")["y"].mean()
run_sizes = resid_frame.groupby("run_id")["y"].size()
run_vars = resid_frame.groupby("run_id")["y"].var(ddof=1).fillna(0.0)
print("\nrun means: var=%.4f  | within-run var (mean)=%.4f" % (run_means.var(ddof=1), run_vars.mean()))
print("mean steps/run: %.2f" % run_sizes.mean())

# ---- T2 cell diagnostics
work = resid_frame.copy()
cells = (
    work.groupby(["video", "run_id", "stack", "baseline", "type"], sort=False)["y"]
    .agg(["mean", "count"])
    .reset_index()
)
print("\ncells:", len(cells), "| counts dist:", cells["count"].value_counts().head(6).to_dict())

for min_cell in (1, 2, 3):
    sub = cells[cells["count"] >= min_cell]
    types = sorted(sub["type"].unique())
    by_type = {t: sub[sub["type"] == t].set_index("run_id")["mean"] for t in types}
    print(f"\n--- min_cell={min_cell} (types={len(types)})")
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            a, b = by_type[types[i]], by_type[types[j]]
            common = a.index.intersection(b.index)
            if len(common) < 40:
                continue
            rho, p = stats.spearmanr(a.loc[common], b.loc[common])
            print(f"    {types[i][:22]:24s} x {types[j][:22]:24s} n={len(common):4d} rho={rho:+.3f} p={p:.1e}")
