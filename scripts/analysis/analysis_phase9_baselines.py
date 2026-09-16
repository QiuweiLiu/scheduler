import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"F:\scheduler")
rows = [json.loads(l) for l in (ROOT / "outputs/phase9_baselines_1000/scheduler_results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
by = defaultdict(dict)
for r in rows:
    by[r["policy"]][r["episode_id"]] = r


def paired(a, b, metric, seed=20260914):
    eps = sorted(set(by[a]) & set(by[b]))
    diffs = np.asarray([by[a][e][metric] - by[b][e][metric] for e in eps], dtype=float)
    rng = np.random.default_rng(seed)
    boot = [float(rng.choice(diffs, size=diffs.size, replace=True).mean()) for _ in range(2000)]
    return float(diffs.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


print("== q95 vs baselines (1,000 episodes, paired) ==")
for other in ["fcfs", "sjf_pred", "state_aware", "myopic", "predopt_h5", "oracle_topology_h5_tab", "trueopt_h5"]:
    parts = []
    for metric, name in (("mean_completion_ms", "completion"), ("mean_job_queue_ms", "queue"), ("deadline_miss_rate", "miss")):
        m, lo, hi = paired("predopt_h5_q95", other, metric)
        if metric == "deadline_miss_rate":
            parts.append(f"{name} {m*100:+.2f}pp [{lo*100:+.2f},{hi*100:+.2f}]")
        else:
            parts.append(f"{name} {m:+.0f} [{lo:+.0f},{hi:+.0f}]")
    print(f"q95 - {other:26s} " + " | ".join(parts))

print()
print("== E2 vs no-future baselines ==")
for other in ["fcfs", "myopic", "sjf_pred", "state_aware"]:
    m, lo, hi = paired("predopt_h5", other, "mean_completion_ms")
    print(f"E2 - {other:26s} completion {m:+.0f} [{lo:+.0f},{hi:+.0f}]")

print()
print("== leaderboard ==")
for policy in sorted(by, key=lambda p: np.mean([r["mean_completion_ms"] for r in by[p].values()])):
    vals = [r["mean_completion_ms"] for r in by[policy].values()]
    miss = [r["deadline_miss_rate"] for r in by[policy].values()]
    print(f"{policy:26s} {np.mean(vals):10.1f}  miss={np.mean(miss)*100:5.2f}%  n={len(vals)}")

print()
print("== survival arm degeneracy check ==")
same = all(
    abs(by["predopt_h5_surv"][e]["mean_completion_ms"] - by["predopt_h5"][e]["mean_completion_ms"]) < 1e-9
    for e in by["predopt_h5_surv"]
)
print("surv identical to E2:", same)
