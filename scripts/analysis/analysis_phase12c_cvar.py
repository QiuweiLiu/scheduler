import json
from pathlib import Path

import numpy as np

ROOT = Path(r"F:\scheduler")


def load(run, policy):
    rows = [json.loads(l) for l in (ROOT / f"outputs/{run}/scheduler_results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["episode_id"]: r for r in rows if r["policy"] == policy}


arms = {
    "q95 (point p95 + load)": load("phase12c_q95b", "predopt_h5_q95"),
    "r95 (runtime p95 sum)": load("phase12c_r95b", "predopt_h5_r95"),
    "comon128 k100": load("phase12c_comon100b", "predopt_h5_comon128_k100"),
    "scen128 k100": load("phase12c_scen100b", "predopt_h5_scen128_k100"),
    "scen128 k50": load("phase12c_scen50b", "predopt_h5_scen128_k50"),
    "scen128 k0": load("phase12c_scen0b", "predopt_h5_scen128_k0"),
}

rng = np.random.default_rng(20260915)


def paired(a, b, metric="mean_completion_ms"):
    eps = sorted(set(a) & set(b))
    diffs = np.asarray([a[e][metric] - b[e][metric] for e in eps], dtype=float)
    boot = [float(rng.choice(diffs, size=diffs.size, replace=True).mean()) for _ in range(2000)]
    return float(diffs.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


print("== Phase 12c leaderboard (1,000 episodes, sidecar artifacts, fixed loader) ==")
for label, data in sorted(arms.items(), key=lambda kv: np.mean([r["mean_completion_ms"] for r in kv[1].values()])):
    comp = np.mean([r["mean_completion_ms"] for r in data.values()])
    miss = np.mean([r["deadline_miss_rate"] for r in data.values()])
    print(f"{label:26s} {comp:10.1f}  miss={miss*100:5.2f}%")

print()
print("== key paired comparisons ==")
for label, a, b in [
    ("r95 - scen128_k100", "r95 (runtime p95 sum)", "scen128 k100"),
    ("r95 - comon128_k100", "r95 (runtime p95 sum)", "comon128 k100"),
    ("r95 - scen128_k50", "r95 (runtime p95 sum)", "scen128 k50"),
    ("r95 - scen128_k0", "r95 (runtime p95 sum)", "scen128 k0"),
    ("comon - scen (k100)", "comon128 k100", "scen128 k100"),
    ("scen k100 - k0", "scen128 k100", "scen128 k0"),
    ("q95 - r95 (cross-check)", "q95 (point p95 + load)", "r95 (runtime p95 sum)"),
]:
    m, lo, hi = paired(arms[a], arms[b])
    mm, mlo, mhi = paired(arms[a], arms[b], "deadline_miss_rate")
    print(f"{label:26s} {m:+8.0f} [{lo:+8.0f},{hi:+8.0f}]   miss {mm*100:+.2f}pp [{mlo*100:+.2f},{mhi*100:+.2f}]")
