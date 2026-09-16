import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"F:\scheduler")


def load(path):
    rows = [json.loads(l) for l in (ROOT / path).read_text(encoding="utf-8").splitlines() if l.strip()]
    out = {}
    for r in rows:
        if r["policy"] == "predopt_h5_q95":
            out[r["episode_id"]] = r
    return out


base = load("outputs/phase9_baselines_1000/scheduler_results.jsonl")
arms = {
    "tail a=0.75": "phase10_run_tail_a75",
    "tail a=0.50": "phase10_run_tail_a50",
    "tail a=0.25": "phase10_run_tail_a25",
    "len eps=0.10": "phase10_run_len_e10",
    "len eps=0.25": "phase10_run_len_e25",
    "len eps=0.50": "phase10_run_len_e50",
    "content eps=0.10": "phase10_run_con_e10",
    "content eps=0.50": "phase10_run_con_e50",
}

rng = np.random.default_rng(20260915)


def paired(a, b, metric):
    eps = sorted(set(a) & set(b))
    diffs = np.asarray([a[e][metric] - b[e][metric] for e in eps], dtype=float)
    boot = [float(rng.choice(diffs, size=diffs.size, replace=True).mean()) for _ in range(2000)]
    return float(diffs.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


print("baseline q95 (unperturbed):", round(np.mean([r['mean_completion_ms'] for r in base.values()]), 1))
print()
print("== perturbation vs unperturbed q95 (paired, 1000 episodes) ==")
for label, run in arms.items():
    data = load(f"outputs/{run}/scheduler_results.jsonl")
    m, lo, hi = paired(data, base, "mean_completion_ms")
    mm, mlo, mhi = paired(data, base, "deadline_miss_rate")
    print(f"{label:18s} completion {m:+8.0f} [{lo:+8.0f},{hi:+8.0f}]   miss {mm*100:+.3f}pp [{mlo*100:+.3f},{mhi*100:+.3f}]")
