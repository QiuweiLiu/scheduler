#!/usr/bin/env python
"""Phase 21-B - paired counterfactuals on the frozen predictor's stack_context.

Both arms keep the frozen J3:seed11 checkpoint, the same 16-video pilot, the same
anchors and the same truth. Only a stack_context value changes:

  planner_unknown   planner_model_id  ->  the literal "unknown" that the in-domain
                    training data always carried (restores the training-time
                    feature-availability contract; it does NOT invent a category)
  stack_canonical   model_stack_id    ->  an explicit mapping (only with a proven
                    stack-identity justification; never on name similarity alone)

Every row keeps a machine-readable audit trail so the counterfactual is provable.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

PLANNER_FALLBACK = "unknown"


def read_rows(path: Path) -> List[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def apply_planner_unknown(row: Dict[str, Any], counter: Counter) -> None:
    stack = row["model_input"]["stack_context"]
    before = stack.get("planner_model_id")
    stack["planner_model_id"] = PLANNER_FALLBACK
    counter["changed" if str(before) != PLANNER_FALLBACK else "already"] += 1
    counter["before:" + str(before)] += 1
    row["stack_context_counterfactual"] = {
        "mode": "planner_unknown",
        "field": "planner_model_id",
        "before": before,
        "after": PLANNER_FALLBACK,
    }


def apply_stack_canonical(row: Dict[str, Any], mapping: Dict[str, str], counter: Counter) -> None:
    stack = row["model_input"]["stack_context"]
    before = str(stack.get("model_stack_id"))
    after = mapping.get(before, before)
    stack["model_stack_id"] = after
    counter["mapped" if after != before else "unmapped"] += 1
    counter["before:" + str(before)] += 1
    row["stack_context_counterfactual"] = {
        "mode": "stack_canonical",
        "field": "model_stack_id",
        "before": before,
        "after": after,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("planner_unknown", "stack_canonical"), required=True)
    parser.add_argument("--mapping", type=Path, default=None, help="JSON file of exact stack-id rewrites")
    parser.add_argument("--audit", type=Path, default=None)
    args = parser.parse_args()

    rows = read_rows(args.anchors)
    counter: Counter = Counter()
    mapping: Dict[str, str] = {}
    if args.mode == "stack_canonical":
        if args.mapping is None:
            raise SystemExit("stack_canonical requires --mapping")
        mapping = json.loads(args.mapping.read_text(encoding="utf-8"))

    for row in rows:
        if "stack_context" not in row.get("model_input", {}):
            raise SystemExit("anchor row lacks model_input.stack_context")
        if args.mode == "planner_unknown":
            apply_planner_unknown(row, counter)
        else:
            apply_stack_canonical(row, mapping, counter)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

    summary = {"mode": args.mode, "rows": len(rows), "counts": dict(counter), "mapping": mapping}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.audit:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
