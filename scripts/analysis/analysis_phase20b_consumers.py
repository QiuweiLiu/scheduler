"""Phase 20-B — consumer matrix on the corrected workload (v03, dev700).

Arms (per review 2026-09-17):
    E0            myopic                       (no future)
    E2            predopt_h5                   (point future cost)
    p50 / p90 / p95  predopt_h5_r50 / r90 / r95 (runtime-only quantile consumption)
    scaled-p50    predopt_h5_r50k              (mean future cost matched to p95)
    tail-shuffle  predopt_h5_q95 on shuffled artifacts (dependence structure control)
    oracle-truth  sameshape_h5_truth           (upper bound on information)

E0/E2/r50/r95 are taken from the Phase 20-A v03 run (all 1000 validation episodes)
and subset to the dev700 allowlist, so every arm shares the same episodes.

Read-only: writes only into the phase20b output directory.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

DEFAULT_BOOTSTRAP = 2000
DEFAULT_SEED = 20260917


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load(path: Path, wanted: set[str] | None = None) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        episode = str(row["episode_id"])
        if wanted is not None and episode not in wanted:
            continue
        out[(episode, str(row["policy"]))] = row
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


ARM_LABELS = {
    "myopic": "E0 (no future)",
    "predopt_h5": "E2 (point future)",
    "predopt_h5_r50": "p50 (optimistic)",
    "predopt_h5_r90": "p90",
    "predopt_h5_r95": "p95 (champion)",
    "predopt_h5_r50k": "scaled-p50 (k=6.2293)",
    "predopt_h5_q95": "tail-shuffle + p95",
    "sameshape_h5_truth": "oracle-truth",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] / "outputs")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    root = args.root

    ids_path = root.parent / "data" / "manifests" / "validation_split_dev700_ids.txt"
    wanted = {line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()}

    runs = {
        "phase20a": root / "phase20a_matrix_v03" / "v03_no_run_container" / "scheduler_results.jsonl",
        "quantiles": root / "phase20b_consumers_v03" / "arms_quantiles" / "scheduler_results.jsonl",
        "shuffle": root / "phase20b_consumers_v03" / "arm_tail_shuffle" / "scheduler_results.jsonl",
        "truth": root / "phase20b_consumers_v03" / "arm_oracle_truth" / "scheduler_results.jsonl",
    }
    for name, path in runs.items():
        if not path.exists():
            raise SystemExit(f"missing run file: {path} ({name})")

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for name, path in runs.items():
        rows.update(load(path, wanted))

    by_policy: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for (episode, policy), row in rows.items():
        by_policy[policy][episode] = row

    report: dict[str, Any] = {"episodes": len({episode for episode, _ in rows}), "arms": {}}
    for policy, entries in by_policy.items():
        values = list(entries.values())
        report["arms"][policy] = {
            "label": ARM_LABELS.get(policy, policy),
            "episodes": len(values),
            "mean_completion_ms": statistics.fmean(row["mean_completion_ms"] for row in values),
            "p95_completion_ms": statistics.fmean(row["p95_completion_ms"] for row in values),
            "mean_deadline_miss_rate": statistics.fmean(row["deadline_miss_rate"] for row in values),
            "mean_job_queue_ms": statistics.fmean(row["mean_job_queue_ms"] for row in values),
            "mean_gpu_evictions": statistics.fmean(row["gpu_evictions"] for row in values),
            "mean_gpu_utilization": statistics.fmean(statistics.fmean(row["gpu_utilization"]) for row in values),
        }

    ordered = sorted(report["arms"].items(), key=lambda item: item[1]["mean_completion_ms"])
    report["ranking"] = [policy for policy, _ in ordered]

    # paired contrasts against E0 and against the champion p95
    report["contrasts"] = {}
    baselines = {policy: entries for policy, entries in by_policy.items()}
    for baseline in ("myopic", "predopt_h5_r95"):
        if baseline not in baselines:
            continue
        for policy, entries in baselines.items():
            if policy == baseline:
                continue
            shared = sorted(set(entries) & set(baselines[baseline]))
            if not shared:
                continue
            values = [entries[e]["mean_completion_ms"] - baselines[baseline][e]["mean_completion_ms"] for e in shared]
            report["contrasts"][f"{policy}_vs_{baseline}"] = {
                "definition": f"{ARM_LABELS.get(policy, policy)} - {ARM_LABELS.get(baseline, baseline)} (negative = left better)",
                "episodes": len(shared),
                "mean": statistics.fmean(values),
                "ci": bootstrap_ci(values, shared, args.bootstrap, args.seed),
            }

    out = root / "phase20b_consumers_v03" / "consumer_matrix.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ranking": report["ranking"], "episodes": report["episodes"]}, ensure_ascii=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
