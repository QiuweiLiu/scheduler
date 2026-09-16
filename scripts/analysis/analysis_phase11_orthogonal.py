import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"F:\scheduler")


def load(run, policy):
    rows = [json.loads(l) for l in (ROOT / f"outputs/{run}/scheduler_results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["episode_id"]: r for r in rows if r["policy"] == policy}


arms = {
    "E2 (hierarchical lookup)": load("phase11_e2", "predopt_h5"),
    "q95 (std, with load term)": load("phase11_q95", "predopt_h5_q95"),
    "Pred50-R": load("phase11_r50", "predopt_h5_r50"),
    "Pred90-R": load("phase11_r90", "predopt_h5_r90"),
    "Pred95-R": load("phase11_r95", "predopt_h5_r95"),
    "ScaledPred50-R (k=6.23)": load("phase11_r50k", "predopt_h5_r50k"),
    "ShuffledTail + q95": load("phase11_shuf", "predopt_h5_q95"),
}

rng = np.random.default_rng(20260915)


def paired(a, b, metric="mean_completion_ms"):
    eps = sorted(set(a) & set(b))
    diffs = np.asarray([a[e][metric] - b[e][metric] for e in eps], dtype=float)
    boot = [float(rng.choice(diffs, size=diffs.size, replace=True).mean()) for _ in range(2000)]
    return float(diffs.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


print("== Phase 11 leaderboard (1,000 episodes) ==")
for label, data in sorted(arms.items(), key=lambda kv: np.mean([r["mean_completion_ms"] for r in kv[1].values()])):
    comp = np.mean([r["mean_completion_ms"] for r in data.values()])
    miss = np.mean([r["deadline_miss_rate"] for r in data.values()])
    print(f"{label:28s} {comp:10.1f}  miss={miss*100:5.2f}%")

print()
print("== key paired comparisons ==")
for label, a, b in [
    ("Pred95-R - ScaledPred50-R", "Pred95-R", "ScaledPred50-R (k=6.23)"),
    ("Pred95-R - ShuffledTail", "Pred95-R", "ShuffledTail + q95"),
    ("Pred95-R - E2", "Pred95-R", "E2 (hierarchical lookup)"),
    ("Pred95-R - q95", "Pred95-R", "q95 (std, with load term)"),
    ("Pred90-R - Pred50-R", "Pred90-R", "Pred50-R"),
    ("Pred95-R - Pred90-R", "Pred95-R", "Pred90-R"),
]:
    m, lo, hi = paired(arms[a], arms[b])
    mm, mlo, mhi = paired(arms[a], arms[b], "deadline_miss_rate")
    print(f"{label:28s} {m:+8.0f} [{lo:+8.0f},{hi:+8.0f}]   miss {mm*100:+.2f}pp [{mlo*100:+.2f},{mhi*100:+.2f}]")

print()
report = json.loads((ROOT / "outputs/phase11_e2/scheduler_matrix_report.json").read_text(encoding="utf-8"))
print("lookup_path_counts (E2 run):", report.get("lookup_path_counts"))
