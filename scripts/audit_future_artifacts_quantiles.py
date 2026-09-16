#!/usr/bin/env python3
"""Stage-0 audit for scheduler future artifacts: quantile sanity + coverage.

Checks every step of every ``b05_future_h*.jsonl.gz`` pack for
  * finite, non-negative runtime/load quantiles,
  * monotonicity p50 <= p90 <= p95,
  * load-occurrence probability in [0, 1],
and reports missing-field and monotonicity-violation rates so the
predicted-quantile consumption experiments can quote a clean Stage-0 line.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


def read_gz(path: Path) -> Iterable[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def check_quantiles(values: Mapping[str, Any], tag: str, counters: Counter) -> None:
    for key in ("p50", "p90", "p95"):
        value = values.get(key)
        if value is None:
            counters[f"{tag}_missing_{key}"] += 1
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0:
            counters[f"{tag}_bad_{key}"] += 1
    p50, p90, p95 = values.get("p50"), values.get("p90"), values.get("p95")
    if all(isinstance(v, (int, float)) for v in (p50, p90, p95)):
        if not (float(p50) <= float(p90) <= float(p95) + 1e-9):
            counters[f"{tag}_nonmonotone"] += 1
            counters[f"{tag}_p50_gt_p95"] += 1 if float(p50) > float(p95) else 0
    counters[f"{tag}_steps"] += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    counters: Counter = Counter()
    files = sorted(args.artifacts.glob("b05_future_h*.jsonl.gz"))
    for path in files:
        for row in read_gz(path):
            for key, value in row.items():
                if not (isinstance(key, str) and key.startswith("future_h")) or not isinstance(value, list):
                    continue
                for scenario in value:
                    for step in scenario.get("steps") or []:
                        resource = step.get("resource") or {}
                        check_quantiles(resource.get("runtime_ms_quantiles") or {}, "runtime", counters)
                        check_quantiles(resource.get("load_duration_ms_quantiles") or {}, "load", counters)
                        occurrence = resource.get("load_occurrence_probability")
                        if occurrence is None:
                            counters["occ_missing"] += 1
                        elif not isinstance(occurrence, (int, float)) or not (0.0 <= float(occurrence) <= 1.0):
                            counters["occ_bad"] += 1
                        else:
                            counters["occ_ok"] += 1
    report = {
        "artifacts": str(args.artifacts),
        "files": [path.name for path in files],
        "counters": dict(counters),
        "runtime_violation_rate": (
            counters["runtime_nonmonotone"] / counters["runtime_steps"] if counters["runtime_steps"] else None
        ),
        "load_violation_rate": (
            counters["load_nonmonotone"] / counters["load_steps"] if counters["load_steps"] else None
        ),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
