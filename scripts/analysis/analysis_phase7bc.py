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


def paired(map_a, pol_a, map_b, pol_b, metric, seed=20260914):
    eps = sorted(set(map_a[pol_a]) & set(map_b[pol_b]))
    diffs = np.asarray([map_a[pol_a][e][metric] - map_b[pol_b][e][metric] for e in eps], dtype=float)
    rng = np.random.default_rng(seed)
    boot = [float(rng.choice(diffs, size=diffs.size, replace=True).mean()) for _ in range(2000)]
    return float(diffs.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


devb = load("outputs/phase7b_dev700/scheduler_results.jsonl")
devc = load("outputs/phase7c_dev700/scheduler_results.jsonl")

print("== Phase 7B dev700 ==")
for label, a, b in [
    ("adapt - E2", "predopt_h5_adapt", "predopt_h5"),
    ("adapt_q - E2", "predopt_h5_adapt_q", "predopt_h5"),
    ("q95 - adapt_q", "predopt_h5_q95", "predopt_h5_adapt_q"),
]:
    parts = []
    for metric, name in (("mean_completion_ms", "completion"), ("mean_job_queue_ms", "queue"), ("deadline_miss_rate", "miss")):
        m, lo, hi = paired(devb, a, devb, b, metric)
        if metric == "deadline_miss_rate":
            parts.append(f"{name} {m*100:+.2f}pp [{lo*100:+.2f}, {hi*100:+.2f}]")
        else:
            parts.append(f"{name} {m:+.0f} [{lo:+.0f}, {hi:+.0f}]")
    print(f"{label}\n   " + " | ".join(parts))

print()
print("== Phase 7C dev700 ==")
for label, a, b in [
    ("rt95 - q95", "predopt_h5_rt95", "predopt_h5_q95"),
    ("rt95 - E2", "predopt_h5_rt95", "predopt_h5"),
    ("ld95 - E2", "predopt_h5_ld95", "predopt_h5"),
]:
    parts = []
    for metric, name in (("mean_completion_ms", "completion"), ("mean_job_queue_ms", "queue"), ("deadline_miss_rate", "miss")):
        m, lo, hi = paired(devc, a, devc, b, metric)
        if metric == "deadline_miss_rate":
            parts.append(f"{name} {m*100:+.2f}pp [{lo*100:+.2f}, {hi*100:+.2f}]")
        else:
            parts.append(f"{name} {m:+.0f} [{lo:+.0f}, {hi:+.0f}]")
    print(f"{label}\n   " + " | ".join(parts))
