import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"F:\scheduler")


def load(path):
    rows = [json.loads(line) for line in (ROOT / path).read_text(encoding="utf-8").splitlines() if line.strip()]
    by = defaultdict(dict)
    for row in rows:
        by[row["policy"]][row["episode_id"]] = row
    return by


dev = load("outputs/phase7a_dev700/scheduler_results.jsonl")


def paired(map_a, pol_a, map_b, pol_b, metric, seed=20260914):
    eps = sorted(set(map_a[pol_a]) & set(map_b[pol_b]))
    diffs = np.asarray([map_a[pol_a][e][metric] - map_b[pol_b][e][metric] for e in eps], dtype=float)
    rng = np.random.default_rng(seed)
    boot = [float(rng.choice(diffs, size=diffs.size, replace=True).mean()) for _ in range(2000)]
    return float(diffs.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


comparisons = [
    ("q95 - E2", "predopt_h5_q95", "predopt_h5"),
    ("scen_k0 - E2", "predopt_h5_scen_k0", "predopt_h5"),
    ("scen_k50 - E2", "predopt_h5_scen_k50", "predopt_h5"),
    ("scen_k100 - E2", "predopt_h5_scen_k100", "predopt_h5"),
    ("q95 - scen_k50", "predopt_h5_q95", "predopt_h5_scen_k50"),
    ("scen_k100 - scen_k50", "predopt_h5_scen_k100", "predopt_h5_scen_k50"),
]
print("== Phase 7A dev700 (n=700) ==")
for label, a, b in comparisons:
    parts = []
    for metric, name in (("mean_completion_ms", "completion"), ("mean_job_queue_ms", "queue"), ("deadline_miss_rate", "miss")):
        m, lo, hi = paired(dev, a, dev, b, metric)
        if metric == "deadline_miss_rate":
            parts.append(f"{name} {m*100:+.2f}pp [{lo*100:+.2f}, {hi*100:+.2f}]")
        else:
            parts.append(f"{name} {m:+.0f} [{lo:+.0f}, {hi:+.0f}]")
    print(f"{label}\n   " + " | ".join(parts))
