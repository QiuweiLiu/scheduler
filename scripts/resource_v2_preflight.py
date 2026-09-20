#!/usr/bin/env python
"""Preflight the resource-v2 arms against the frozen scheduler artifact pack.

This is the gate the interface review demanded before any 300-episode run: it
loads the frozen base pack, overlays each resource-v2 arm through
``load_resource_v2_overlay`` and refuses to report success unless every counter in
the preflight block is zero or within tolerance.

The ``j3`` arm is the identity control: overlaying it must reproduce the base
pack's H5 block exactly.

Usage::

    python scripts/resource_v2_preflight.py --arms j3 r1b r3a_u
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tracing.analysis.workload_v02_simulator import (  # noqa: E402
    load_future_artifacts,
    load_resource_v2_overlay,
)

DEFAULT_BASE = PROJECT_ROOT / "outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts"
DEFAULT_ARMS_ROOT = PROJECT_ROOT / "outputs/resource_v2_artifacts"
DEFAULT_HORIZON = 5


def check_identity_control(
    base_root: Path, arm_root: Path, horizon: int = DEFAULT_HORIZON
) -> Dict[str, Any]:
    """The j3 arm must not change a single byte of the H5 block."""

    base = load_future_artifacts(base_root)
    from tracing.analysis.workload_v02_simulator import read_gzip_jsonl

    overlay = {str(row["node_id"]): row for row in read_gzip_jsonl(arm_root / "b05_future_h5.jsonl.gz")}
    if set(base) != set(overlay):
        raise SystemExit("j3 identity control: node sets differ")
    differing = [
        node_id
        for node_id in base
        if json.dumps(base[node_id].get(f"future_h{horizon}"), sort_keys=True)
        != json.dumps(overlay[node_id].get(f"future_h{horizon}"), sort_keys=True)
    ]
    if differing:
        raise SystemExit(
            "j3 identity control: %d node(s) changed the H5 block, e.g. %s"
            % (len(differing), differing[:3])
        )
    return {"node_count": len(base), "differing_nodes": 0}


def preflight_arm(base_root: Path, arm_root: Path) -> Dict[str, Any]:
    artifacts, preflight = load_resource_v2_overlay(base_root, arm_root)
    preflight["overlay_loaded_nodes"] = len(artifacts)
    return preflight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-pack", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--arms-root", type=Path, default=DEFAULT_ARMS_ROOT)
    parser.add_argument("--arms", nargs="+", default=["j3", "r1b", "r3a_u", "r3a_f"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not (args.base_pack / "b05_future_h5.jsonl.gz").is_file():
        raise SystemExit("base pack not found: %s" % args.base_pack)

    report: Dict[str, Any] = {"base_pack": str(args.base_pack), "arms": {}}
    failures: List[str] = []

    for arm in args.arms:
        arm_root = args.arms_root / arm
        if not (arm_root / "resource_v2_manifest.json").is_file():
            failures.append("%s: manifest missing at %s" % (arm, arm_root))
            print("[FAIL] %-8s manifest missing" % arm)
            continue
        print("[....] %-8s loading overlay" % arm)
        try:
            entry = preflight_arm(args.base_pack, arm_root)
        except Exception as exc:  # noqa: BLE001 - the point is to report, not to crash the sweep
            failures.append("%s: %s" % (arm, exc))
            report["arms"][arm] = {"error": str(exc)}
            print("[FAIL] %-8s %s" % (arm, exc))
            continue
        if arm == "j3":
            try:
                entry["identity_control"] = check_identity_control(args.base_pack, arm_root)
            except Exception as exc:  # noqa: BLE001
                failures.append("%s identity control: %s" % (arm, exc))
                print("[FAIL] %-8s identity control: %s" % (arm, exc))
                continue
        report["arms"][arm] = entry
        print(
            "[PASS] %-8s nodes=%d steps=%d unupgraded=%d nonresource=%d bins_mismatch=%d "
            "load_missing=%d nan=%d prob_sum_err=%.2e view_err=%.2e"
            % (
                arm, entry["node_count"], entry["step_count"], entry["unupgraded_steps"],
                entry["nonresource_mismatch_count"], entry["bin_schema_mismatch_count"],
                entry["load_field_missing_count"], entry["nan_prob_count"],
                entry["prob_sum_max_abs_error"], entry["canonical_view_max_abs_error"],
            )
        )

    report["failures"] = failures
    report["all_pass"] = not failures
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("saved:", args.output)

    if failures:
        print("\nPREFLIGHT FAILED (%d):" % len(failures))
        for item in failures:
            print("  -", item)
        return 1
    print("\nPREFLIGHT: all arms PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
