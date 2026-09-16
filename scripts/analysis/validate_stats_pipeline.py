"""Validate the T1/T2/T3 pipeline on synthetic data with known ground truth."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"F:\scheduler\scripts")
from verify_trace_dependence import (  # noqa: E402
    pair_spearman,
    pooled_spearman,
    run_type_cells,
    tail_lift,
    variance_components_closed_form,
    mixed_effects_icc,
)

rng = np.random.default_rng(7)

# ---------------------------------------------------------------- synthetic data
N_VIDEO = 160
RUNS_PER_VIDEO = 4
TYPES = ["A", "B", "C", "D"]
SIGMA_VIDEO = 0.4
SIGMA_RUN = 0.6
SIGMA_EPS = 0.8

rows = []
for v in range(N_VIDEO):
    b_video = rng.normal(0, SIGMA_VIDEO)
    for r in range(RUNS_PER_VIDEO):
        b_run = rng.normal(0, SIGMA_RUN)
        stack = "stack_a" if r % 2 == 0 else "stack_b"
        baseline = "langgraph_react" if r < 2 else "star"
        for t in TYPES:
            for _ in range(3):
                y = b_video + b_run + rng.normal(0, SIGMA_EPS)
                rows.append(
                    {
                        "video": f"v{v:03d}",
                        "run_id": f"v{v:03d}_r{r}",
                        "stack": stack,
                        "baseline": baseline,
                        "type": t,
                        "y": float(y),
                    }
                )
frame = pd.DataFrame(rows)

true_total = SIGMA_VIDEO**2 + SIGMA_RUN**2 + SIGMA_EPS**2
print("ground truth: icc_video=%.3f icc_run=%.3f" % (SIGMA_VIDEO**2 / true_total, SIGMA_RUN**2 / true_total))

closed = variance_components_closed_form(frame)
print("closed form : icc_video=%.3f icc_run=%.3f" % (closed["icc_video"], closed["icc_run"]))
mixed = mixed_effects_icc(frame)
print("mixedlm     : icc_video=%.3f icc_run=%.3f (converged=%s)" % (mixed["icc_video"], mixed["icc_run"], mixed["converged"]))

cells = run_type_cells(frame, drop_video=False, min_cell=1)
pairs = pair_spearman(cells, min_common=40)
print("T2 pooled spearman on synthetic (expect strongly positive):", round(pooled_spearman(pairs), 3))
for q in (0.8, 0.9):
    res = tail_lift(cells, q, min_common=40)
    print(f"T3 q={q}: pooled lift (expect >> 1) = {res['pooled_lift']:.2f}")
