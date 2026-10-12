#!/usr/bin/env python3
"""Gate 2: isolation — replacing the true future must not change the first decision.

For a captured state:
1. baseline planning run with the TRUE future (templates unchanged);
2. planning runs with the execution truth of all not-yet-started nodes replaced
   by seeded random models/durations (the planner's observation and predictions
   stay identical);
3. the first dispatch decision (job, node, gpu) must be identical in all runs,
   while the later event stream must actually differ (otherwise the swap is
   vacuous).

This proves the planning decision is a function of (observation, predictions,
action set) only — never of the true unexecuted future.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracing.analysis.des_rollout import (
    build_planning_episode,
    build_scenario_templates,
    run_planning,
)
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


def first_dispatch(events: list[dict]) -> tuple[str, str, int] | None:
    for event in events:
        if event.get("event_type") == "node_dispatch":
            return (
                str(event.get("job_instance_id")),
                str(event.get("node_id")),
                int(event.get("gpu_index")),
            )
    return None


def observation_nodes(events: list[dict]) -> set[str]:
    """Node ids visible to the planner at the first decision (from its state view)."""

    for event in events:
        if event.get("event_type") != "node_dispatch":
            continue
        state = event.get("scheduler_state") or {}
        nodes = {str(row.get("node_id")) for row in state.get("ready_nodes") or []}
        nodes.add(str(event.get("node_id")))
        return nodes
    return set()


def random_scenario(
    snapshot: dict,
    templates: dict,
    episode: dict,
    rng: random.Random,
    *,
    truncate: bool,
    exclude: set[str],
) -> dict:
    """Randomize the execution truth of not-yet-started nodes.

    Nodes visible at the first decision (``exclude``) keep their true identity:
    the observation must stay fixed for the isolation test to be meaningful.
    """

    model_pool = sorted(
        {
            node.model_id
            for job in episode.get("jobs") or []
            for node in templates[str(job["template_id"])].nodes
            if node.lane == "gpu" and node.model_id
        }
    )
    cpu_pool = sorted(
        {
            node.model_id
            for job in episode.get("jobs") or []
            for node in templates[str(job["template_id"])].nodes
            if node.lane == "cpu" and node.model_id
        }
    )
    specs: dict[str, dict] = {}
    for row in snapshot.get("jobs") or []:
        job_id = str(row["job_instance_id"])
        job = next(j for j in episode["jobs"] if j["job_instance_id"] == job_id)
        template = templates[str(job["template_id"])]
        replace: dict[str, dict] = {}
        pending = [
            node
            for node in template.nodes
            if (row.get("node_state") or {}).get(node.node_id) == "pending"
            and node.node_id not in exclude
        ]
        for node in pending:
            pool = model_pool if node.lane == "gpu" else cpu_pool
            replace[node.node_id] = {
                "model_id": rng.choice(pool) if pool else node.model_id,
                "runtime_ms": rng.uniform(500.0, 60000.0),
                "load_ms": rng.uniform(0.0, 8000.0) if node.lane == "gpu" else None,
                "workspace_peak_mb": (
                    rng.uniform(1000.0, 24000.0) if node.lane == "gpu" else None
                ),
            }
        drop: list[str] = []
        if truncate and len(pending) > 3:
            drop = [node.node_id for node in pending[3:]]
        specs[job_id] = {
            "template_id": str(job["template_id"]),
            "replace": replace,
            "drop": drop,
        }
    return {"jobs": specs}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--fraction", type=float, default=0.3)
    parser.add_argument("--seeds", default="101,202")
    args = parser.parse_args()

    extension_config = json.loads(EXTENSION_CONFIG.read_text(encoding="utf-8"))
    templates = load_templates(PROJECTION, topology_view="causal_v3")
    episodes = read_jsonl(TRAIN_EPISODES)[: args.episodes]
    templates, episodes = apply_real_workload_profile_contract(
        templates, episodes, extension_config
    )
    stats = train_resource_stats({k: v for k, v in templates.items() if v.split == "train"})
    artifacts, _ = load_resource_v2_overlay(BASE_ARTIFACTS, F0_ARTIFACTS)

    failures = 0
    for episode in episodes:
        at_ms = args.fraction * float(episode.get("episode_window_ms") or 0.0)
        capture: dict = {"at_ms": at_ms}
        ctx: dict = {"activation": {}, "des_capture": capture}
        simulate_episode(
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
        snapshot = capture["snapshot"]
        plan_episode = build_planning_episode(episode, snapshot, include_arrivals=True)

        _, base_events, _ = run_planning(
            plan_episode,
            templates,
            future_artifacts=artifacts,
            train_stats=stats,
            extension_config=extension_config,
            policy=POLICY,
        )
        base_first = first_dispatch(base_events)
        visible = observation_nodes(base_events)

        for seed_text in args.seeds.split(","):
            seed = int(seed_text)
            for truncate in (False, True):
                rng = random.Random(seed)
                scenario = random_scenario(
                    snapshot, templates, episode, rng, truncate=truncate,
                    exclude=visible,
                )
                scenario_templates = build_scenario_templates(templates, scenario)
                _, events, _ = run_planning(
                    plan_episode,
                    scenario_templates,
                    future_artifacts=artifacts,
                    train_stats=stats,
                    extension_config=extension_config,
                    policy=POLICY,
                )
                first = first_dispatch(events)
                changed = sum(
                    1
                    for a, b in zip(base_events, events)
                    if (a.get("event_type"), a.get("time_ms"), a.get("node_id"))
                    != (b.get("event_type"), b.get("time_ms"), b.get("node_id"))
                )
                same = first == base_first
                vacuous = changed == 0
                status = "OK" if (same and not vacuous) else "FAIL"
                if not same or vacuous:
                    failures += 1
                print(
                    f"{status} {episode['episode_id']} seed={seed} truncate={int(truncate)} "
                    f"first={first} base={base_first} changed_events={changed}"
                )
    print(f"GATE2 {'PASS' if failures == 0 else 'FAIL'} ({failures} failing cases)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
