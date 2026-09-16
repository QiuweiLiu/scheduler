import json
from pathlib import Path

import numpy as np

ROOT = Path(r"F:\scheduler")


def load(run, policy):
    p = ROOT / f"outputs/{run}/scheduler_results.jsonl"
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["episode_id"]: r for r in rows if r["policy"] == policy}


arms = {
    "r95 (greedy + r95 consumption)": load("phase12c_r95b", "predopt_h5_r95"),
    "cp_rho_h5 (CP-SAT + E2 consumption)": load("phase13_cprho5", "cp_rho_h5"),
    "cp_rho_h3 (CP-SAT + E2 consumption)": load("phase13_cprho3", "cp_rho_h3"),
    "pred_mpc_h5 (MPC + E2 consumption)": load("phase13_predmpc5", "pred_mpc_h5"),
}

rng = np.random.default_rng(20260915)


def paired(a, b, metric="mean_completion_ms"):
    eps = sorted(set(a) & set(b))
    diffs = np.asarray([a[e][metric] - b[e][metric] for e in eps], dtype=float)
    boot = [float(rng.choice(diffs, size=diffs.size, replace=True).mean()) for _ in range(2000)]
    return float(diffs.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


print("== Phase 13 leaderboard (1,000 episodes, same artifacts) ==")
for label, data in sorted(arms.items(), key=lambda kv: np.mean([r["mean_completion_ms"] for r in kv[1].values()])):
    comp = np.mean([r["mean_completion_ms"] for r in data.values()])
    miss = np.mean([r["deadline_miss_rate"] for r in data.values()])
    util = np.mean([float(x) for r in data.values() for x in (r.get("gpu_utilization") or [])])
    print(f"{label:38s} {comp:10.1f}  miss={miss*100:5.2f}%  util={util*100:5.2f}%")

print()
print("== paired comparisons ==")
for label, a, b in [
    ("r95 - cp_rho_h5", "r95 (greedy + r95 consumption)", "cp_rho_h5 (CP-SAT + E2 consumption)"),
    ("r95 - cp_rho_h3", "r95 (greedy + r95 consumption)", "cp_rho_h3 (CP-SAT + E2 consumption)"),
    ("r95 - pred_mpc_h5", "r95 (greedy + r95 consumption)", "pred_mpc_h5 (MPC + E2 consumption)"),
    ("cp_rho_h5 - pred_mpc_h5", "cp_rho_h5 (CP-SAT + E2 consumption)", "pred_mpc_h5 (MPC + E2 consumption)"),
]:
    m, lo, hi = paired(arms[a], arms[b])
    mm, mlo, mhi = paired(arms[a], arms[b], "deadline_miss_rate")
    print(f"{label:26s} {m:+8.0f} [{lo:+8.0f},{hi:+8.0f}]   miss {mm*100:+.2f}pp [{mlo*100:+.2f},{mhi*100:+.2f}]")
