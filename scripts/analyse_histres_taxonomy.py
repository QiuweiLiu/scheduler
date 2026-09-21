"""Report the historical-telemetry join taxonomy for the augmented dataset.

Supersedes analyse_histres_unresolved.py, whose "unresolved" category no longer
exists: Stage 0.2 split it into merged_into_composite_parent / not_resource_applicable
/ unexpected_node_table_missing.  The most important line this prints is
``unexpected_node_table_missing == 0``, which is the standing regression guard that
no historical telemetry is silently missing.
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter, defaultdict
from typing import Dict, Set
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

HISTRES = PROJECT_ROOT / "results/processed/j_series_dataset_histres_v2"
SPLITS = ("train", "validation")
OUT = PROJECT_ROOT / "experiments/EXP-20260921_histres_causal_input_v1/artifacts/join_taxonomy.json"


def read_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    report = {"schema_version": "histres-join-taxonomy-v1", "splits": {}}
    for split in SPLITS:
        path = HISTRES / ("histres_%s.jsonl.gz" % split)
        if not path.is_file():
            continue
        occurrences = Counter()
        unique_nodes: Dict[str, Set[str]] = defaultdict(set)
        runs: Dict[str, Set[str]] = defaultdict(set)
        rows = 0
        for row in read_gz(path):
            rows += 1
            run_id = str(row.get("run_id"))
            for entry in row.get("history_resource_audit") or []:
                status = str(entry.get("join_status"))
                occurrences[status] += 1
                runs[status].add(run_id)
                node_id = entry.get("joined_node_id")
                if node_id:
                    unique_nodes[status].add(str(node_id))
        report["splits"][split] = {
            "rows": rows,
            "occurrences": dict(occurrences),
            "unique_nodes": {k: len(v) for k, v in unique_nodes.items()},
            "unique_runs": {k: len(v) for k, v in runs.items()},
            "unexpected_node_table_missing_occurrences": occurrences.get(
                "unexpected_node_table_missing", 0),
        }
        print("=== %s ===" % split)
        print("  rows: %d" % rows)
        for status in sorted(occurrences, key=lambda s: -occurrences[s]):
            print("  %-30s occurrences=%7d  unique_nodes=%6d  unique_runs=%5d"
                  % (status, occurrences[status], len(unique_nodes.get(status, ())),
                     len(runs.get(status, ()))))
        guard = occurrences.get("unexpected_node_table_missing", 0)
        print("  GUARD unexpected_node_table_missing = %d %s"
              % (guard, "OK" if guard == 0 else "FAIL"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print("saved:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
