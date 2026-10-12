#!/usr/bin/env python3
"""Convergence analysis for the DES-aligned rollout (scenario-count stability).

Reads the K=64 driver output and, for each K' in the prefix grid, computes the
per-state action deltas (mean over the first K' scenarios) and compares:
- delta stability vs the K=64 reference (max |delta_K' - delta_64|);
- argmin (best action) agreement vs the K=64 reference.

Also reports the myopia view: H5-truncated deltas vs full-continuation deltas
when the full-action columns are present.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--grid", default="8,16,32,64")
    args = parser.parse_args()

    grid = [int(v) for v in args.grid.split(",") if v.strip()]
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    states = [row for row in rows if row.get("scenarios", 1) > 1]
    print(f"states with multi-scenario runs: {len(states)}")

    if not states:
        print("no multi-scenario rows (deterministic run): convergence table skipped")
    summary: dict[int, dict] = {k: {"max_dev": [], "agree": 0, "total": 0} for k in grid}
    signal_states = 0
    for row in states:
        actions = row["actions"]
        base_costs = actions["default"]["costs"]
        deltas64 = {
            name: statistics.fmean(data["costs"]) - statistics.fmean(base_costs)
            for name, data in actions.items()
        }
        best64 = min(deltas64, key=lambda name: deltas64[name])
        if max(abs(v) for v in deltas64.values()) > 1.0:
            signal_states += 1
        for k in grid:
            deltas_k = {
                name: statistics.fmean(data["costs"][:k]) - statistics.fmean(base_costs[:k])
                for name, data in actions.items()
            }
            max_dev = max(abs(deltas_k[name] - deltas64[name]) for name in deltas64)
            summary[k]["max_dev"].append(max_dev)
            best_k = min(deltas_k, key=lambda name: deltas_k[name])
            summary[k]["total"] += 1
            if best_k == best64:
                summary[k]["agree"] += 1
    print(f"states with action signal (|delta|>1ms at K=64): {signal_states}")
    print(f"{'K':>4} {'max_dev mean':>14} {'max_dev p95':>12} {'argmin agree':>13}")
    for k in grid:
        devs = summary[k]["max_dev"]
        agree = summary[k]["agree"] / max(1, summary[k]["total"])
        devs_sorted = sorted(devs)
        p95 = devs_sorted[min(len(devs_sorted) - 1, int(0.95 * len(devs_sorted)))] if devs else 0.0
        if devs:
            print(f"{k:>4} {statistics.fmean(devs):>14.1f} {p95:>12.1f} {agree:>12.1%}")
        else:
            print(f"{k:>4} {'-':>14} {'-':>12} {'-':>13}")

    # myopia view (if full-action columns present)
    full_rows = [row for row in rows if any("full" in name for name in row["actions"])]
    if full_rows:
        print(f"\nmyopia check on {len(full_rows)} states with full-action columns:")
        agree = 0
        for row in full_rows:
            actions = row["actions"]
            deltas_h5 = {
                name: statistics.fmean(data["costs"]) - statistics.fmean(actions["default"]["costs"])
                for name, data in actions.items() if not name.startswith("full")
            }
            deltas_full = {
                name[5:]: statistics.fmean(data["costs"]) - statistics.fmean(actions.get("full_default", actions["default"])["costs"])
                for name, data in actions.items() if name.startswith("full")
            }
            if not deltas_full:
                continue
            best_h5 = min(deltas_h5, key=lambda name: deltas_h5[name])
            best_full = min(deltas_full, key=lambda name: deltas_full[name])
            agree += int(best_h5 == best_full)
        print(f"  best-action agreement H5 vs full: {agree}/{len(full_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
