"""Phase 19 — paired analysis of the container-excluded workload (v03 vs legacy v02).

Reads the two smoke runs, pairs them by (episode_id, policy), and reports:

* the policy ranking on each workload (mean / p95 completion, miss, queue, evictions);
* paired per-episode deltas for the policy contrasts that carry the paper's claims
  (r95-E2, r95-E0, r50-E0, E2-E0, fcfs-E0), with an episode-clustered bootstrap CI;
* the difference of those deltas between v03 and v02, i.e. how much of the effect
  the container-included workload was contributing.

Clustering is by episode_id (this is a workload-correction comparison, not a
video-level generalisation claim).

Read-only: writes only into the phase19 output directory.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

CONTRASTS = (
    ("r95_vs_e2", "predopt_h5_r95", "predopt_h5"),
    ("r95_vs_e0", "predopt_h5_r95", "myopic"),
    ("r50_vs_e0", "predopt_h5_r50", "myopic"),
    ("e2_vs_e0", "predopt_h5", "myopic"),
    ("fcfs_vs_e0", "fcfs", "myopic"),
)
DEFAULT_BOOTSTRAP = 2000
DEFAULT_SEED = 20260917


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_run(path: Path) -> dict[tuple[str, str], float]:
    rows = read_jsonl(path)
    out: dict[tuple[str, str], float] = {}
    for row in rows:
        out[(str(row["episode_id"]), str(row["policy"]))] = float(row["mean_completion_ms"])
    return out


def bootstrap_ci(values: Sequence[float], clusters: Sequence[str], n_boot: int, seed: int) -> list[float] | None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(values, clusters):
        grouped[str(cluster)].append(float(value))
    keys = sorted(grouped)
    if len(keys) < 2:
        return None
    stats = {key: (sum(grouped[key]), len(grouped[key])) for key in keys}
    rng = random.Random(seed)
    means = []
    for _ in range(int(n_boot)):
        total = 0.0
        count = 0
        for _ in range(len(keys)):
            cluster_sum, cluster_count = stats[keys[rng.randrange(len(keys))]]
            total += cluster_sum
            count += cluster_count
        if count:
            means.append(total / count)
    means.sort()
    return [means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]]


def deltas(run: dict[tuple[str, str], float], left: str, right: str) -> tuple[list[float], list[str]]:
    episodes = sorted({episode for episode, policy in run if policy == left} & {episode for episode, policy in run if policy == right})
    values = [run[(episode, left)] - run[(episode, right)] for episode in episodes]
    return values, episodes


def aggregate(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy"])].append(row)
    out = {}
    for policy, entries in grouped.items():
        out[policy] = {
            "episodes": len(entries),
            "mean_completion_ms": statistics.fmean(entry["mean_completion_ms"] for entry in entries),
            "p95_completion_ms": statistics.fmean(entry["p95_completion_ms"] for entry in entries),
            "mean_deadline_miss_rate": statistics.fmean(entry["deadline_miss_rate"] for entry in entries),
            "mean_job_queue_ms": statistics.fmean(entry["mean_job_queue_ms"] for entry in entries),
            "mean_gpu_evictions": statistics.fmean(entry["gpu_evictions"] for entry in entries),
            "mean_gpu_utilization": statistics.fmean(
                statistics.fmean(entry["gpu_utilization"]) for entry in entries
            ),
        }
    return dict(sorted(out.items(), key=lambda item: item[1]["mean_completion_ms"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] / "outputs" / "phase19_container_fix_smoke_v03")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    legacy_results = args.root / "legacy_v02" / "scheduler_results.jsonl"
    corrected_results = args.root / "v03_no_run_container" / "scheduler_results.jsonl"
    for path in (legacy_results, corrected_results):
        if not path.exists():
            raise SystemExit(f"missing result file: {path}")

    legacy = load_run(legacy_results)
    corrected = load_run(corrected_results)
    report: dict[str, Any] = {
        "ranking": {
            "legacy_v02": aggregate(legacy_results),
            "v03_no_run_container": aggregate(corrected_results),
        },
        "paired_contrasts": {},
        "episodes": len({episode for episode, _policy in corrected}),
    }

    for name, left, right in CONTRASTS:
        legacy_values, episodes = deltas(legacy, left, right)
        corrected_values, corrected_episodes = deltas(corrected, left, right)
        paired = [c - l for l, c in zip(legacy_values, corrected_values)]
        report["paired_contrasts"][name] = {
            "definition": f"{left} - {right} (negative = left is better)",
            "episodes": len(episodes),
            "legacy_v02": {
                "mean": statistics.fmean(legacy_values),
                "ci": bootstrap_ci(legacy_values, episodes, args.bootstrap, args.seed),
            },
            "v03_no_run_container": {
                "mean": statistics.fmean(corrected_values),
                "ci": bootstrap_ci(corrected_values, corrected_episodes, args.bootstrap, args.seed),
            },
            "correction_effect": {
                "mean": statistics.fmean(paired),
                "ci": bootstrap_ci(paired, episodes, args.bootstrap, args.seed),
            },
        }

    legacy_order = list(report["ranking"]["legacy_v02"])
    corrected_order = list(report["ranking"]["v03_no_run_container"])
    report["verdict"] = {
        "ranking_preserved": legacy_order == corrected_order,
        "legacy_order": legacy_order,
        "v03_order": corrected_order,
        "r95_still_best": corrected_order[0] == "predopt_h5_r95",
        "e2_still_beats_e0": report["paired_contrasts"]["e2_vs_e0"]["v03_no_run_container"]["mean"] < 0,
        "r95_gain_over_e0_legacy_ms": report["paired_contrasts"]["r95_vs_e0"]["legacy_v02"]["mean"],
        "r95_gain_over_e0_v03_ms": report["paired_contrasts"]["r95_vs_e0"]["v03_no_run_container"]["mean"],
    }
    out = args.root / "paired_diff.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["verdict"], ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
