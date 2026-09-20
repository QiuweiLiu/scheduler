#!/usr/bin/env python
"""Build the train-only runtime histogram bank used by the attribute mixture.

The bank answers: for a given ``(model_id, lane, sequence_index)`` lookup group,
what does the *training split* say about that group's runtime distribution?

It deliberately reuses the simulator's own three-level hierarchy
(``model|lane|seq|exact`` -> ``model|lane|*|model_lane`` -> ``*|lane|*|lane``)
instead of inventing a second lookup order, and it invents no smoothing: a group
either has training samples or the lookup falls through to the next level.

Usage::

    python scripts/build_attribute_runtime_stats_bank.py \
        --templates results/processed/r7_workload_v03_no_run_container/job_templates_r7_v03.jsonl \
        --output outputs/resource_v2_decision_trace/stats_bank.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from tracing.analysis.workload_v02_simulator import (  # noqa: E402
    load_templates,
    quantile,
    resource_keys,
)

DEFAULT_BINS = PROJECT_ROOT / "outputs/j_series_resource_dist_v1/bins.json"


def load_bin_edges(path: Path) -> List[float]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    edges = [float(e) for e in spec["edges"]]
    if len(edges) < 2:
        raise SystemExit("bin spec needs at least two edges")
    return edges


def histogram(values: Sequence[float], edges: Sequence[float]) -> List[float]:
    """Normalised occupancy of each half-open bin; the last bin is open-ended."""

    n_bins = len(edges) - 1
    counts = [0] * n_bins
    for value in values:
        placed = False
        for index in range(n_bins):
            low, high = edges[index], edges[index + 1]
            if index == n_bins - 1:
                if value >= low:
                    counts[index] += 1
                    placed = True
                    break
            elif low <= value < high:
                counts[index] += 1
                placed = True
                break
        if not placed:
            counts[0] += 1  # value below the first edge (should not happen)
    total = sum(counts)
    if total == 0:
        raise ValueError("empty histogram")
    return [c / total for c in counts]


def build_bank(templates: Mapping[str, Any], edges: Sequence[float]) -> Dict[str, Dict[str, Any]]:
    runtimes: Dict[str, List[float]] = defaultdict(list)
    loads: Dict[str, List[float]] = defaultdict(list)

    for template in templates.values():
        if template.split != "train":
            continue
        for node in template.nodes:
            for key in resource_keys(node):
                runtimes[key].append(float(node.runtime_ms))
                if node.load_ms is not None and float(node.load_ms) > 0.0:
                    loads[key].append(float(node.load_ms))

    bank: Dict[str, Dict[str, Any]] = {}
    for key, values in runtimes.items():
        entry: Dict[str, Any] = {
            "lookup_key": key,
            "n": len(values),
            "runtime_bin_probs": histogram(values, edges),
            "runtime_p50_ms": float(quantile(values, 0.50)),
            "runtime_p90_ms": float(quantile(values, 0.90)),
            "load_p50_ms": float(quantile(loads[key], 0.50)) if loads.get(key) else 0.0,
            "lookup_level": key.split("|")[3],
        }
        bank[key] = entry
    return bank


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--templates", type=Path,
                        default=PROJECT_ROOT / "results/processed/r7_workload_v03_no_run_container/job_templates_r7_v03.jsonl")
    parser.add_argument("--bins", type=Path, default=DEFAULT_BINS)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "outputs/resource_v2_decision_trace/stats_bank.json")
    args = parser.parse_args()

    edges = load_bin_edges(args.bins)
    templates = load_templates(args.templates)
    bank = build_bank(templates, edges)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bank, ensure_ascii=False, indent=2), encoding="utf-8")

    by_level: Dict[str, int] = defaultdict(int)
    for entry in bank.values():
        by_level[str(entry["lookup_level"])] += 1
    total_samples = sum(int(e["n"]) for e in bank.values())
    print("templates: %d | bin edges: %d | groups: %d | samples: %d"
          % (len(templates), len(edges), len(bank), total_samples))
    print("groups by level:", dict(by_level))
    print("saved:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
