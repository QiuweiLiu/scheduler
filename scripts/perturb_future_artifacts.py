#!/usr/bin/env python3
"""Create perturbed copies of scheduler future artifacts for sensitivity studies.

Inference-only post-processing: never re-fits, never reads test/holdout. Each
perturbation is deterministic given --seed and is recorded in
``perturbation.json`` next to the copied files so runs stay auditable.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict

FILES = [
    "b05_node_h1.jsonl.gz",
    "b05_future_h1.jsonl.gz",
    "b05_future_h3.jsonl.gz",
    "b05_future_h5.jsonl.gz",
    "b05_future_h5_layers.jsonl.gz",
]


def read_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_gz(path: Path, rows) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def shrink_tail(quantiles: Dict[str, Any], alpha: float) -> None:
    p50 = quantiles.get("p50")
    if not isinstance(p50, (int, float)):
        return
    for key in ("p90", "p95"):
        value = quantiles.get(key)
        if isinstance(value, (int, float)):
            quantiles[key] = float(p50) + alpha * (float(value) - float(p50))


def perturb_row(row: Dict[str, Any], kind: str, intensity: float, rng: random.Random) -> Dict[str, Any]:
    scenarios = row.get("future_h5") or []
    if kind == "tail_shrink":
        for scenario in scenarios:
            for step in scenario.get("steps") or []:
                resource = step.get("resource") or {}
                shrink_tail(resource.get("runtime_ms_quantiles") or {}, intensity)
                shrink_tail(resource.get("load_duration_ms_quantiles") or {}, intensity)
        return row
    if kind == "length_noise":
        for scenario in scenarios:
            steps = scenario.get("steps") or []
            if not steps:
                continue
            if rng.random() < intensity:
                if rng.random() < 0.5 and len(steps) > 1:
                    scenario["steps"] = steps[:-1]
                else:
                    tail = dict(steps[-1])
                    tail["step_offset"] = int(tail.get("step_offset") or len(steps)) + 1
                    scenario["steps"] = steps + [tail]
        return row
    if kind == "content_corrupt":
        for scenario in scenarios:
            for step in scenario.get("steps") or []:
                if rng.random() < intensity:
                    lane = str(step.get("execution_lane") or "gpu")
                    step["execution_lane"] = "cpu" if lane == "gpu" else "gpu"
        return row
    raise ValueError(f"unknown perturbation kind: {kind}")


def apply_tail_shuffle(rows: list, seed: int) -> dict:
    """Replace each step's p95 with p50 + a gap drawn from the global gap pool.

    The marginal distribution of (p95 - p50) is preserved exactly (the pool is
    shuffled and dealt out), but the per-node alignment is destroyed, which is
    the control for "must the tail magnitude belong to this specific future?".
    """

    pool: list[float] = []
    for row in rows:
        for scenario in row.get("future_h5") or []:
            for step in scenario.get("steps") or []:
                quantiles = (step.get("resource") or {}).get("runtime_ms_quantiles") or {}
                p50 = quantiles.get("p50")
                p95 = quantiles.get("p95")
                if isinstance(p50, (int, float)) and isinstance(p95, (int, float)) and float(p95) >= float(p50):
                    pool.append(float(p95) - float(p50))
    rng = random.Random(seed)
    rng.shuffle(pool)
    cursor = 0
    replaced = 0
    for row in rows:
        for scenario in row.get("future_h5") or []:
            for step in scenario.get("steps") or []:
                quantiles = (step.get("resource") or {}).get("runtime_ms_quantiles") or {}
                p50 = quantiles.get("p50")
                if not isinstance(p50, (int, float)):
                    continue
                if cursor >= len(pool):
                    break
                quantiles["p95"] = float(p50) + pool[cursor]
                cursor += 1
                replaced += 1
    return {"pool_size": len(pool), "replaced_steps": replaced, "dealt": cursor}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--kind", choices=("tail_shrink", "length_noise", "content_corrupt", "tail_shuffle"), required=True)
    parser.add_argument("--intensity", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()

    args.target.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    counts: dict = {"rows": 0, "perturbed_rows": 0}
    for name in FILES:
        src = args.source / name
        if not src.is_file():
            continue
        rows = list(read_gz(src))
        if args.kind == "tail_shuffle" and name == "b05_future_h5.jsonl.gz":
            counts["rows"] += len(rows)
            shuffle_stats = apply_tail_shuffle(rows, args.seed)
            counts["shuffle"] = shuffle_stats
            counts["perturbed_rows"] += shuffle_stats["replaced_steps"]
        elif args.kind == "tail_shuffle":
            counts["rows"] += len(rows)
        else:
            for index, row in enumerate(rows):
                before = json.dumps(row, sort_keys=True)
                row = perturb_row(row, args.kind, args.intensity, rng)
                after = json.dumps(row, sort_keys=True)
                counts["rows"] += 1
                if before != after:
                    counts["perturbed_rows"] += 1
                rows[index] = row
        write_gz(args.target / name, rows)
    manifest_src = args.source / "b05_artifact_manifest.json"
    if manifest_src.is_file():
        shutil.copyfile(manifest_src, args.target / "b05_artifact_manifest.json")
    meta = {
        "schema_version": "future-artifact-perturbation-v1",
        "source": str(args.source),
        "kind": args.kind,
        "intensity": args.intensity,
        "seed": args.seed,
        "counts": counts,
        "note": "inference-only post-processing applied to the frozen J3:seed11 pack",
    }
    (args.target / "perturbation.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
