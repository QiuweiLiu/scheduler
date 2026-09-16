import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"F:\scheduler")

def load(path):
    rows = [json.loads(l) for l in (ROOT / path).read_text(encoding="utf-8").splitlines() if l.strip()]
    by = defaultdict(dict)
    for row in rows:
        by[row["policy"]][row["episode_id"]] = row
    return by

e2 = load("outputs/phase2_r7_j_1000/scheduler_results.jsonl")            # predopt_h5 (baseline consumer)
h5 = load("outputs/phase6_consumers_h5_1000/scheduler_results.jsonl")   # lam scan + q95 + cc
h10 = load("outputs/phase6b_consumers_h10_1000/scheduler_results.jsonl")# h10 consumers
h10m = load("outputs/phase6b_h10model_h5consumer_1000/scheduler_results.jsonl")  # h10 model + 5-step consumer
cache = load("outputs/phase6c_cache_res_1000/scheduler_results.jsonl")

def paired(map_a, pol_a, map_b, pol_b, metric, seed=20260911):
    eps = sorted(set(map_a[pol_a]) & set(map_b[pol_b]))
    diffs = np.asarray([map_a[pol_a][e][metric] - map_b[pol_b][e][metric] for e in eps], dtype=float)
    rng = np.random.default_rng(seed)
    boot = [float(rng.choice(diffs, size=diffs.size, replace=True).mean()) for _ in range(2000)]
    return float(diffs.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

comparisons = [
    ("q95 - E2", h5, "predopt_h5_q95", e2, "predopt_h5"),
    ("lambda100(p90) - E2", h5, "predopt_h5_lam100", e2, "predopt_h5"),
    ("lambda50 - E2", h5, "predopt_h5_lam50", e2, "predopt_h5"),
    ("lambda0(p50) - E2", h5, "predopt_h5_lam0", e2, "predopt_h5"),
    ("cc - E2", h5, "predopt_h5_cc", e2, "predopt_h5"),
    ("q95 - lambda100", h5, "predopt_h5_q95", h5, "predopt_h5_lam100"),
    ("H10 q95(10-step) - H5 q95(5-step)", h10, "predopt_h10_q95", h5, "predopt_h5_q95"),
    ("H10 lam0 - H5 lam0", h10, "predopt_h10_lam0", h5, "predopt_h5_lam0"),
    ("H10model+5step q95 - H5model+5step q95", h10m, "predopt_h5_q95", h5, "predopt_h5_q95"),
    ("cache_res - E2", cache, "aligned_predopt_h5_layer_res", e2, "predopt_h5"),
]
print("== Phase 6 comparisons (1000 episodes) ==")
for label, ma, pa, mb, pb in comparisons:
    parts = []
    for metric, name in (("mean_completion_ms", "completion"), ("mean_job_queue_ms", "queue"), ("deadline_miss_rate", "miss")):
        m, lo, hi = paired(ma, pa, mb, pb, metric)
        if metric == "deadline_miss_rate":
            parts.append(f"{name} {m*100:+.2f}pp [{lo*100:+.2f}, {hi*100:+.2f}]")
        else:
            parts.append(f"{name} {m:+.0f} [{lo:+.0f}, {hi:+.0f}]")
    print(f"{label}\n   " + " | ".join(parts))
