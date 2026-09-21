"""Emit the unresolved-token composition as a checkable artefact.

Reports, for every split: how many historical tokens are unresolved, what their
(event_type, node_type, raw_action) composition is, and how much of it the
existing `_resource_applicable(event)` rule can explain.

No conclusion is drawn here; the output is evidence for review.
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_p9d_topology_dataset import _is_terminal_marker, _resource_applicable  # noqa: E402

HISTRES = PROJECT_ROOT / "results/processed/j_series_dataset_histres_v2"
SPLITS = ("train", "validation")
OUT = PROJECT_ROOT / "experiments/EXP-20260921_histres_causal_input_v1/artifacts/unresolved_breakdown.json"


def read_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    report = {"schema_version": "histres-unresolved-breakdown-v1", "splits": {}}
    for split in SPLITS:
        path = HISTRES / ("histres_%s.jsonl.gz" % split)
        if not path.is_file():
            continue
        combos = Counter()
        by_node_type = Counter()
        by_event_type = Counter()
        total = 0
        explained_by_rule = 0
        rows = 0
        for row in read_gz(path):
            rows += 1
            history = row["model_input"]["history"]
            for entry in row.get("history_resource_audit") or []:
                if entry.get("join_status") != "unresolved":
                    continue
                total += 1
                token = history[entry["token_index"]]
                by_node_type[str(token.get("node_type"))] += 1
                by_event_type[str(token.get("event_type"))] += 1
                combos[(str(token.get("event_type")), str(token.get("node_type")), str(token.get("raw_action")))] += 1
                # the rule needs `action`; the token only retains `raw_action`, so the
                # reconstruction below is the closest faithful probe
                probe = {
                    "node_type": token.get("node_type"),
                    "event_type": token.get("event_type"),
                    "action": token.get("raw_action"),
                }
                if not _resource_applicable(probe):
                    explained_by_rule += 1
        report["splits"][split] = {
            "rows": rows,
            "unresolved_tokens": total,
            "explained_by_resource_applicable_rule": explained_by_rule,
            "explained_fraction": explained_by_rule / max(1, total),
            "by_node_type": dict(by_node_type),
            "by_event_type": dict(by_event_type),
            "top_combinations": [
                {"event_type": c[0], "node_type": c[1], "raw_action": c[2], "count": n,
                 "fraction": n / max(1, total)}
                for c, n in combos.most_common(10)
            ],
        }
        print("=== %s ===" % split)
        print("  rows=%d unresolved=%d explained_by_rule=%d (%.2f%%)"
              % (rows, total, explained_by_rule, 100.0 * explained_by_rule / max(1, total)))
        print("  by_node_type:", dict(by_node_type))
        for entry in report["splits"][split]["top_combinations"][:5]:
            print("    %-60s %6d (%.1f%%)"
                  % (str((entry["event_type"], entry["node_type"], entry["raw_action"]))[:60],
                     entry["count"], 100.0 * entry["fraction"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print("saved:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
