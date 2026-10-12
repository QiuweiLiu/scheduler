#!/usr/bin/env python3
"""Gate 1: planning DES vs real DES event-level alignment.

For a real episode run under the frozen main line (pdrs_resident):
1. capture the full scheduler state at a mid-episode moment;
2. replay the future in a planning episode that injects that state (true future,
   real arrivals, real templates);
3. compare the event streams after the capture point field by field.

The planning DES must reproduce the real DES continuation exactly.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracing.analysis.des_rollout import build_planning_episode, run_planning
from tracing.analysis.profile_contract import apply_real_workload_profile_contract
from tracing.analysis.workload_v02_simulator import (
    load_templates,
    read_jsonl,
    load_resource_v2_overlay,
    train_resource_stats,
    simulate_episode,
)

PROJECTION = (
    ROOT
    / "results/processed/r7_workload_v041_ontology_no_run_container"
    / "job_templates_r7_v041.jsonl"
)
TRAIN_EPISODES = (
    ROOT / "results/processed/r7_workload_20260817/episodes/workload_train_r7.jsonl"
)
EXTENSION_CONFIG = (
    ROOT
    / "experiments/EXP-20261004_real_strata_calibration_v1"
    / "artifacts/substrate_extension_v7.json"
)
BASE_ARTIFACTS = ROOT / "outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts"
F0_ARTIFACTS = ROOT / "outputs/resource_v2_artifacts/f0_seed11"
POLICY = "pdrs_resident"


INERT_KEYS = {"episode_id", "decision_index"}


def _normalize(value):
    """Drop inert bookkeeping fields (episode id suffix, decision counter)."""

    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items() if k not in INERT_KEYS}
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    return value


def compare_events(real: list[dict], plan: list[dict], limit: int = 5) -> list[str]:
    problems: list[str] = []
    if len(real) != len(plan):
        problems.append(f"event count: real {len(real)} vs plan {len(plan)}")
    for index, (a, b) in enumerate(zip(real, plan)):
        a = _normalize(a)
        b = _normalize(b)
        for key in sorted(set(a) | set(b)):
            va, vb = a.get(key), b.get(key)
            if isinstance(va, float) or isinstance(vb, float):
                if not (
                    isinstance(va, (int, float))
                    and isinstance(vb, (int, float))
                    and math.isclose(float(va), float(vb), rel_tol=1e-9, abs_tol=1e-6)
                ):
                    problems.append(f"#{index} {a.get('event_type')} {key}: {str(va)[:120]!r} != {str(vb)[:120]!r}")
            elif va != vb:
                problems.append(f"#{index} {a.get('event_type')} {key}: {str(va)[:120]!r} != {str(vb)[:120]!r}")
        if len(problems) >= limit:
            break
    return problems


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--fractions", default="0.3,0.6")
    args = parser.parse_args()

    extension_config = json.loads(EXTENSION_CONFIG.read_text(encoding="utf-8"))
    templates = load_templates(PROJECTION, topology_view="causal_v3")
    episodes = read_jsonl(TRAIN_EPISODES)[: args.episodes]
    templates, episodes = apply_real_workload_profile_contract(
        templates, episodes, extension_config
    )
    train = {k: v for k, v in templates.items() if v.split == "train"}
    stats = train_resource_stats(train)
    artifacts, _preflight = load_resource_v2_overlay(BASE_ARTIFACTS, F0_ARTIFACTS)

    fractions = [float(v) for v in args.fractions.split(",") if v.strip()]
    failures = 0
    for episode in episodes:
        window = float(episode.get("episode_window_ms") or 0.0)
        for fraction in fractions:
            at_ms = fraction * window
            capture: dict = {"at_ms": at_ms}
            ctx: dict = {"activation": {}, "des_capture": capture}
            summary, events = simulate_episode(
                episode,
                templates,
                POLICY,
                train_stats=stats,
                future_artifacts=artifacts,
                future_horizon=5,
                extension_config=extension_config,
                policy_context=ctx,
                collect_events=True,
            )
            snapshot = capture.get("snapshot")
            if snapshot is None:
                print(f"FAIL {episode['episode_id']} f={fraction}: no snapshot captured")
                failures += 1
                continue
            event_index = int(capture.get("event_index", 0))
            plan_episode = build_planning_episode(
                episode, snapshot, include_arrivals=True
            )
            plan_summary, plan_events, _plan_ctx = run_planning(
                plan_episode,
                templates,
                future_artifacts=artifacts,
                train_stats=stats,
                extension_config=extension_config,
                forced_action=None,
                policy=POLICY,
            )
            real_future = events[event_index:]
            problems = compare_events(real_future, plan_events)
            status = "OK" if not problems else "FAIL"
            print(
                f"{status} {episode['episode_id']} f={fraction:.1f} "
                f"t={snapshot['now_ms']:.0f} real_future={len(real_future)} "
                f"plan={len(plan_events)} jobs_real={summary['completed_jobs']} "
                f"jobs_plan={plan_summary['completed_jobs']}"
            )
            if problems:
                failures += 1
                for line in problems:
                    print("   ", line)
    print(f"GATE1 {'PASS' if failures == 0 else 'FAIL'} ({failures} failing cases)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
