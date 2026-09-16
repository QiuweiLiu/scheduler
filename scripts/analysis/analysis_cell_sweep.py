import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"F:\scheduler")

# episode metadata -> pressure cell
episodes = {}
for line in (ROOT / "results/processed/r7_workload_20260817/episodes/workload_validation_r7.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    load = float(row.get("target_offered_compute_load") or 0.0)
    deadline = float(row.get("deadline_multiplier") or 0.0)
    episodes[str(row["episode_id"])] = {
        "load": load,
        "deadline": deadline,
        "arrival": str(row.get("arrival_pattern")),
        "state": str(row.get("initial_state")),
        "gpu": ",".join(str(x) for x in (row.get("gpu_topology_mb") or [])),
    }


def load_runs(path, policy):
    out = {}
    for line in (ROOT / path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["policy"] == policy:
            out[str(row["episode_id"])] = row
    return out


e2 = load_runs("outputs/phase2_r7_j_1000/scheduler_results.jsonl", "predopt_h5")
q95 = load_runs("outputs/phase6_consumers_h5_1000/scheduler_results.jsonl", "predopt_h5_q95")
print("E2 rows:", len(e2), "q95 rows:", len(q95))

common = sorted(set(e2) & set(q95) & set(episodes))
print("common episodes:", len(common))


def bucket_load(x):
    if x <= 0.70:
        return "low(0.5-0.7)"
    if x <= 0.85:
        return "mid(0.85)"
    return "high(0.95-1.05)"


def bucket_deadline(x):
    return "tight(1.5)" if x <= 1.5 else "loose(2.0-3.0)"


rng = np.random.default_rng(20260914)


def paired_ci(diffs):
    diffs = np.asarray(diffs, dtype=float)
    boot = [float(rng.choice(diffs, size=diffs.size, replace=True).mean()) for _ in range(2000)]
    return float(diffs.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


cells = defaultdict(list)
for eid in common:
    meta = episodes[eid]
    cells[(bucket_load(meta["load"]), bucket_deadline(meta["deadline"]))].append(eid)

print()
print("== q95 - E2 by pressure cell (paired, mean completion ms) ==")
for key in sorted(cells):
    ids = cells[key]
    comp = [q95[e]["mean_completion_ms"] - e2[e]["mean_completion_ms"] for e in ids]
    miss = [q95[e]["deadline_miss_rate"] - e2[e]["deadline_miss_rate"] for e in ids]
    m, lo, hi = paired_ci(comp)
    mm, mlo, mhi = paired_ci(miss)
    print(f"{key[0]:16s} x {key[1]:12s} n={len(ids):4d}  completion {m:+8.0f} [{lo:+8.0f},{hi:+8.0f}]  miss {mm*100:+.2f}pp [{mlo*100:+.2f},{mhi*100:+.2f}]")

print()
print("== q95 - E2 by load only ==")
for key in ["low(0.5-0.7)", "mid(0.85)", "high(0.95-1.05)"]:
    ids = [e for e in common if bucket_load(episodes[e]["load"]) == key]
    comp = [q95[e]["mean_completion_ms"] - e2[e]["mean_completion_ms"] for e in ids]
    m, lo, hi = paired_ci(comp)
    print(f"{key:16s} n={len(ids):4d}  {m:+8.0f} [{lo:+8.0f},{hi:+8.0f}]")

print()
print("== q95 - E2 by deadline only ==")
for key in ["tight(1.5)", "loose(2.0-3.0)"]:
    ids = [e for e in common if bucket_deadline(episodes[e]["deadline"]) == key]
    comp = [q95[e]["mean_completion_ms"] - e2[e]["mean_completion_ms"] for e in ids]
    m, lo, hi = paired_ci(comp)
    print(f"{key:16s} n={len(ids):4d}  {m:+8.0f} [{lo:+8.0f},{hi:+8.0f}]")
