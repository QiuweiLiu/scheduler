#!/usr/bin/env python3
"""DES-aligned rollout driver: capture states, enumerate actions, run planning sims.

Per episode (one real run, multiple captures):
1. capture states at the configured fractions (frozen main line);
2. probe planning run -> first-decision candidate pool;
3. action set (ordering x prefetch x eviction, ablation subsets);
4. one planning run per action on the SAME scenario (deterministic greedy chain);
5. future arrivals: times resampled from the training pool's inter-arrival gaps
   (the episode's own arrival table is never read for the planning world);
6. reference: full continuation with the true future (real templates);
7. cost = future flow time (waiting area); per-run wall time recorded.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracing.analysis.des_rollout import (
    MarkovScenarioModel,
    apply_injected_state,
    build_planning_episode,
    build_scenario_templates,
    future_flow_cost,
    greedy_chain_scenario,
    run_planning,
)
from tracing.analysis.profile_contract import apply_real_workload_profile_contract
from tracing.analysis.workload_v02_simulator import (
    GPU,
    build_jobs,
    estimate,
    load_templates,
    read_jsonl,
    load_resource_v2_overlay,
    residency_demand_totals,
    sameshape_future_cost,
    simulate_episode,
    train_resource_stats,
)

PROJECTION = (
    ROOT / "results/processed/r7_workload_v041_ontology_no_run_container"
    / "job_templates_r7_v041.jsonl"
)
TRAIN_EPISODES = (
    ROOT / "results/processed/r7_workload_20260817/episodes/workload_train_r7.jsonl"
)
EXTENSION_CONFIG = (
    ROOT / "experiments/EXP-20261004_real_strata_calibration_v1"
    / "artifacts/substrate_extension_v7.json"
)
BASE_ARTIFACTS = ROOT / "outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts"
F0_ARTIFACTS = ROOT / "outputs/resource_v2_artifacts/f0_seed11"
POLICY = "pdrs_resident"
HORIZON = 5


def load_all(episode_limit: int):
    extension_config = json.loads(EXTENSION_CONFIG.read_text(encoding="utf-8"))
    templates = load_templates(PROJECTION, topology_view="causal_v3")
    episodes = read_jsonl(TRAIN_EPISODES)[:episode_limit]
    templates, episodes = apply_real_workload_profile_contract(
        templates, episodes, extension_config
    )
    stats = train_resource_stats({k: v for k, v in templates.items() if v.split == "train"})
    artifacts, _ = load_resource_v2_overlay(BASE_ARTIFACTS, F0_ARTIFACTS)
    return extension_config, templates, episodes, stats, artifacts


def fit_arrival_gaps(train_episodes) -> list[float]:
    """Empirical normalized inter-arrival gaps from the training pool."""

    gaps: list[float] = []
    for ep in train_episodes:
        arrivals = sorted(float(j["arrival_ms"]) for j in ep.get("jobs") or [])
        window = float(ep.get("episode_window_ms") or (arrivals[-1] if arrivals else 1.0))
        for a, b in zip(arrivals, arrivals[1:]):
            gaps.append(max(0.0, (b - a) / max(1.0, window)))
    return gaps or [0.05]


def resampled_arrivals(snapshot, episode, gaps, rng) -> list:
    """Future arrivals: same job order, times resampled from the training pool.

    The episode's own arrival table is never read for the planning world; only
    the count of pending jobs and their order are preserved (v1 limitation).
    """

    pending = list(snapshot.get("arrivals_pending") or [])
    if not pending:
        return []
    window = float(episode.get("episode_window_ms") or 1.0)
    now0 = float(snapshot["now_ms"])
    out = []
    t = now0
    for _arrival_ms, job_index in pending:
        t += max(1.0, float(rng.choice(gaps)) * window)
        out.append([t, int(job_index)])
    return out


def resident_mb_index(templates) -> dict[str, float]:
    index: dict[str, float] = {}
    for template in templates.values():
        for node in template.nodes:
            if node.resident_model_mb is not None and node.model_id not in index:
                index[str(node.model_id)] = float(node.resident_model_mb)
    return index


def f0_rank(pool, jobs, first_state, artifacts, stats):
    residents = {
        int(row["index"]): set(str(m) for m in row["resident"])
        for row in (first_state or {}).get("gpus") or []
    }
    ready_lookup = {
        (int(job_index), str(node_id)): (float(priority), float(ready_time))
        for (priority, ready_time, job_index, node_id) in (first_state or {}).get("ready") or []
    }
    ranked = []
    for job_index, node_id, gpu_index in pool:
        job = jobs[job_index]
        node = job.template.by_id[node_id]
        row = estimate(node, stats)
        load = 0.0 if node.model_id in residents.get(int(gpu_index), set()) else float(row["load_p50_ms"])
        current = float(row["runtime_p50_ms"]) + load
        future = sameshape_future_cost(node_id, artifacts, stats, HORIZON, "p95")
        priority, ready_time = ready_lookup.get((job_index, node_id), (1, 0.0))
        ranked.append((
            (priority, current + future, future, ready_time, node_id, 0, node.model_id, gpu_index),
            {"node_key": node_id, "gpu_index": int(gpu_index)},
        ))
    ranked.sort(key=lambda item: item[0])
    return [row for _key, row in ranked]


def prefetch_targets(pool, jobs, first_state, artifacts, templates):
    """Next-step demand from the pool nodes' artifacts (first decision state)."""

    nexts: dict[str, float] = {}
    for job_index, node_id, _gpu_index in pool:
        row = artifacts.get(str(node_id)) or {}
        scenarios = row.get("future_h5") or []
        steps = (scenarios[0].get("steps") or []) if scenarios else []
        if not steps:
            continue
        step = steps[0]
        model = str(step.get("model_id") or "")
        probability = float(step.get("model_probability") or 0.0)
        if model:
            nexts[model] = nexts.get(model, 0.0) + max(probability, 0.0)
    resident_index = resident_mb_index(templates)
    gpu_rows = (first_state or {}).get("gpus") or []
    targets = []
    for model, demand in sorted(nexts.items(), key=lambda kv: (-kv[1], kv[0])):
        if demand <= 0.0:
            continue
        mb = resident_index.get(model)
        if mb is None:
            continue
        for gpu in gpu_rows:
            residents = {str(m): 0.0 for m in gpu["resident"]}
            if model in residents:
                continue
            total = sum(float(v) for v in (gpu.get("resident") or {}).values())
            if total + mb <= 32760.0:
                targets.append({"prefetch": [model, int(gpu["index"])]})
                break
        if len(targets) >= 3:
            break
    return targets


def build_actions(mode, f0_rows, pf_targets):
    actions: dict[str, dict] = {"default": {}}
    order_rows = f0_rows[:3]
    if mode in ("smoke5", "ablate_order", "joint"):
        for rank, row in enumerate(order_rows[1:], start=2):
            actions[f"order{rank}"] = dict(row)
    if mode in ("smoke5", "ablate_pf", "joint"):
        for index, row in enumerate(pf_targets[:2], start=1):
            actions[f"pf{index}"] = dict(row)
    if mode in ("smoke5", "ablate_evict", "joint"):
        actions["evict_large"] = {"evict_mode": "largest_first"}
    if mode == "smoke5" and pf_targets:
        joint = dict(order_rows[0])
        joint.update(pf_targets[0])
        actions["joint_a"] = joint
    if mode == "joint":
        for rank, row in enumerate(order_rows, start=1):
            for pf_index, pf in enumerate([{}] + pf_targets[:2]):
                for evict in ("least_needed", "largest_first"):
                    combo = dict(row)
                    combo.update(pf)
                    combo["evict_mode"] = evict
                    actions[f"o{rank}_p{pf_index}_{evict[:5]}"] = combo
    return actions


def process_state(episode, snapshot, templates, stats, artifacts, extension_config,
                  gaps, args, rows_out, run_times, scenario_model=None):
    rng = random.Random(f"{episode['episode_id']}|{snapshot['now_ms']}")
    arrivals = resampled_arrivals(snapshot, episode, gaps, rng)
    plan_episode = build_planning_episode(
        episode, {**snapshot, "arrivals_pending": arrivals}, include_arrivals=True
    )
    jobs = build_jobs(plan_episode, templates)
    gpus = [GPU(i, cap) for i, cap in enumerate(episode.get("gpu_topology_mb") or [])]
    installed = apply_injected_state(snapshot, jobs, gpus)
    ready_lookup = {
        (int(job_index), str(node_id)): (float(priority), float(ready_time))
        for (priority, ready_time, job_index, node_id) in installed["ready"]
    }
    _, _probe_events, probe_ctx = run_planning(
        plan_episode, templates, future_artifacts=artifacts, train_stats=stats,
        extension_config=extension_config, policy="des_rollout",
        forced_action=None, collect_events=False,
        extra_context={"des_probe_pool": True},
    )
    pool = probe_ctx.get("des_first_pool") or []
    if not pool:
        return
    first_state = probe_ctx.get("des_first_state") or {}
    f0_rows = f0_rank(pool, jobs, first_state, artifacts, stats)
    pf_targets = prefetch_targets(pool, jobs, first_state, artifacts, templates)
    actions = build_actions(args.actions, f0_rows, pf_targets)

    scenario_list = []
    if args.generator == "greedy":
        scenario = greedy_chain_scenario(snapshot, episode, templates, artifacts, stats)
        scenario_list.append(scenario)
    else:
        for scenario_index in range(args.scenarios):
            srng = random.Random(f"{episode['episode_id']}|{snapshot['now_ms']}|{scenario_index}")
            scenario_list.append(
                scenario_model.scenario(
                    snapshot, episode, templates, artifacts, stats, srng
                )
            )
    scenario_templates_list = [
        build_scenario_templates(templates, scenario) for scenario in scenario_list
    ]

    record = {
        "episode_id": episode["episode_id"],
        "at_ms": snapshot["now_ms"],
        "pool": pool,
        "f0_top": f0_rows[:3],
        "prefetch_targets": pf_targets,
        "n_resampled_arrivals": len(arrivals),
        "generator": args.generator,
        "scenarios": len(scenario_list),
        "capped": scenario_list[0].get("capped") if scenario_list else {},
        "actions": {},
    }
    for name, action in actions.items():
        costs = []
        activations: dict = {}
        for scenario_templates in scenario_templates_list:
            started = time.perf_counter()
            _summary, events, pctx = run_planning(
                plan_episode, scenario_templates, future_artifacts=artifacts,
                train_stats=stats, extension_config=extension_config,
                policy="des_rollout", forced_action=action or None,
                collect_events=True,
            )
            elapsed = time.perf_counter() - started
            run_times.append(elapsed)
            cost = future_flow_cost(events, snapshot, episode)
            costs.append(float(cost["future_flow_ms"]))
            activations = {
                k: v for k, v in (pctx.get("activation") or {}).items()
                if k.startswith("des_") or k.startswith("joint_")
            }
        record["actions"][name] = {
            "action": action,
            "cost_ms": statistics.fmean(costs) if costs else 0.0,
            "cost_min": min(costs) if costs else 0.0,
            "cost_max": max(costs) if costs else 0.0,
            "costs": [round(c, 1) for c in costs],
            "wall_s": round(statistics.fmean(run_times[-len(scenario_list):]), 4) if scenario_list else 0.0,
            "activation": activations,
        }
    started = time.perf_counter()
    _summary, full_events, _ = run_planning(
        plan_episode, templates, future_artifacts=artifacts, train_stats=stats,
        extension_config=extension_config, policy=POLICY,
        forced_action=None, collect_events=True,
    )
    full_wall = time.perf_counter() - started
    full_cost = future_flow_cost(full_events, snapshot, episode)
    record["full_continuation"] = {
        "cost_ms": full_cost["future_flow_ms"], "wall_s": round(full_wall, 3)
    }
    if args.full_actions:
        record["actions"]["full_default"] = {
            "action": {}, "cost_ms": full_cost["future_flow_ms"],
            "costs": [full_cost["future_flow_ms"]], "wall_s": round(full_wall, 3),
        }
        for name, action in actions.items():
            if name == "default":
                continue
            started = time.perf_counter()
            _summary, events, _ = run_planning(
                plan_episode, templates, future_artifacts=artifacts,
                train_stats=stats, extension_config=extension_config,
                policy="des_rollout", forced_action=action or None,
                collect_events=True,
            )
            elapsed = time.perf_counter() - started
            cost = future_flow_cost(events, snapshot, episode)
            record["actions"][f"full_{name}"] = {
                "action": action, "cost_ms": cost["future_flow_ms"],
                "costs": [cost["future_flow_ms"]], "wall_s": round(elapsed, 3),
            }
    rows_out.append(record)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    default_cost = record["actions"]["default"]["cost_ms"]
    print(
        f"{episode['episode_id']}@{snapshot['now_ms']:.0f}: pool={len(pool)} "
        f"arr={len(arrivals)} actions={len(actions)} K={len(scenario_list)} "
        f"cost~{default_cost:.0f}ms full~{full_cost['future_flow_ms']:.0f}ms "
        f"per_run={statistics.fmean(run_times[-len(actions) * len(scenario_list):]):.2f}s"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--fractions", default="0.3,0.5")
    parser.add_argument("--actions", default="smoke5")
    parser.add_argument("--generator", default="greedy", choices=("greedy", "markov"))
    parser.add_argument("--scenarios", type=int, default=1)
    parser.add_argument("--full-actions", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    extension_config, templates, episodes, stats, artifacts = load_all(args.episodes)
    gaps = fit_arrival_gaps(episodes)
    scenario_model = MarkovScenarioModel(templates) if args.generator == "markov" else None
    rows_out: list[dict] = []
    run_times: list[float] = []
    fractions = [float(v) for v in args.fractions.split(",") if v.strip()]
    for episode in episodes:
        window = float(episode.get("episode_window_ms") or 0.0)
        capture: dict = {"at_ms_list": [f * window for f in fractions]}
        ctx: dict = {"activation": {}, "des_capture": capture}
        simulate_episode(
            episode, templates, POLICY, train_stats=stats, future_artifacts=artifacts,
            future_horizon=HORIZON, extension_config=extension_config,
            policy_context=ctx, collect_events=False,
        )
        for snapshot in capture.get("snapshots") or []:
            process_state(
                episode, snapshot, templates, stats, artifacts, extension_config,
                gaps, args, rows_out, run_times, scenario_model=scenario_model,
            )
    if args.out:
        print(f"wrote {args.out} ({len(rows_out)} states)")
    if run_times:
        print(
            f"timing: n={len(run_times)} mean={statistics.fmean(run_times):.2f}s "
            f"median={statistics.median(run_times):.2f}s max={max(run_times):.2f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
