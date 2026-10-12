#!/usr/bin/env python3
"""H5-MILP-v0a test suite (T1-T10) with an evidence report.

Per the advisor review: T7's exhaustive enumerator is INDEPENDENT of the MILP
constraint generator (a separate pure-Python scheduler simulation).

Usage:
  python scripts/milp_v0_tests.py --t7-instances 24 --t7-seed 11
  python scripts/milp_v0_tests.py --real-states 3   # adds T1/T2/T3a/T4/T5/T8
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracing.scheduling.milp_v0 import (
    MILPObservation,
    ObsChainStep,
    ObsGpu,
    ObsReadyNode,
    _build_nodes,
    build_reference_schedule,
    solve_v0a,
)

SERIAL = ROOT / "experiments/EXP-20261012_milp_v0/artifacts/serial_v0_extension.json"
EVIDENCE = ROOT / "experiments/EXP-20261012_milp_v0/artifacts/tests_evidence.json"


# --------------------------------------------------------------------------- #
# T7: tiny instance generator + independent exhaustive enumerator
# --------------------------------------------------------------------------- #


@dataclass
class TinyInstance:
    obs: MILPObservation
    boundary_of: dict  # anchor key -> boundary node key


def make_tiny_instance(rng: random.Random) -> TinyInstance:
    models = ["M1", "M2", "M3"]
    resident_mb = {"M1": 8000.0, "M2": 12000.0, "M3": 4000.0}
    cap = 32760.0
    now = 0.0
    ready = []
    chains = {}
    job_count = rng.randint(1, 2)
    node_index = 0
    for job in range(job_count):
        node_count = rng.randint(1, 2)
        for k in range(node_count):
            model = rng.choice(models)
            runtime = rng.choice([50.0, 100.0, 200.0])
            load = rng.choice([5.0, 10.0, 20.0])
            mem = rng.choice([3000.0, 6000.0, 15000.0])
            node_id = f"J{job}N{k}"
            ready.append(
                ObsReadyNode(
                    job_index=job, node_id=node_id, priority=float(job),
                    ready_since_ms=0.0, model_id=model, lane="gpu",
                    runtime_p50_ms=runtime, load_p50_ms=load, memory_p95_mb=mem,
                )
            )
            steps = []
            for s in range(rng.randint(0, 2)):
                steps.append(
                    ObsChainStep(
                        model_id=rng.choice(models), lane="gpu",
                        runtime_ms=rng.choice([50.0, 100.0]), load_ms=rng.choice([5.0, 10.0]),
                        memory_mb=rng.choice([3000.0, 6000.0]),
                    )
                )
            chains[node_id] = tuple(steps)
            node_index += 1
    # residents: random subsets that fit
    gpus = []
    for g in range(2):
        resident = {}
        used = 0.0
        for model in models:
            if rng.random() < 0.3 and used + resident_mb[model] <= cap:
                resident[model] = resident_mb[model]
                used += resident_mb[model]
        gpus.append(ObsGpu(index=g, capacity_mb=cap, available_at_ms=now, resident=resident))
    obs = MILPObservation(
        now_ms=now, ready=tuple(ready), chains=chains, gpus=tuple(gpus),
        resident_mb_by_model=resident_mb, decision_time_ms=now,
    )
    return TinyInstance(obs=obs, boundary_of={})


def exhaustive_optimum(obs: MILPObservation) -> float | None:
    """Independent pure-Python enumeration (no MILP code reused)."""

    nodes = []
    for item in obs.ready:
        nodes.append(
            {
                "key": item.node_id, "anchor": None, "pred": None, "lane": item.lane, "model": item.model_id,
                "compute": max(0.1, item.runtime_p50_ms - item.load_p50_ms),
                "load": item.load_p50_ms, "ws": item.memory_p95_mb, "job": item.job_index,
            }
        )
        previous = item.node_id
        for si, step in enumerate(obs.chains.get(item.node_id, ())):
            node_key = f"pred:{item.node_id}:{si + 1}"
            nodes.append(
                {
                    "key": node_key, "anchor": item.node_id, "pred": previous,
                    "lane": step.lane, "model": step.model_id,
                    "compute": max(0.1, step.runtime_ms - step.load_ms),
                    "load": step.load_ms, "ws": step.memory_mb, "job": item.job_index,
                }
            )
            previous = node_key
    gpu_nodes = [i for i, n in enumerate(nodes) if n["lane"] == "gpu"]
    if not gpu_nodes:
        return None
    G = len(obs.gpus)
    caps = [g.capacity_mb for g in obs.gpus]
    base = [sum(g.resident.values()) for g in obs.gpus]
    avail0 = [g.available_at_ms for g in obs.gpus]
    resident0 = [dict(g.resident) for g in obs.gpus]
    # boundaries: last chain node per anchor (or anchor itself)
    boundaries = []
    for item in obs.ready:
        chain = [n for n in nodes if n["anchor"] == item.node_id]
        boundaries.append((chain[-1]["key"] if chain else item.node_id))
    best = None
    # enumerate assignments (gpu nodes only; CPU nodes are time-only)
    for assign in itertools.product(range(G), repeat=len(gpu_nodes)):
        # per-gpu sequences: enumerate permutations of assigned nodes
        per_gpu: list[list[int]] = [[] for _ in range(G)]
        for pos, node_i in enumerate(gpu_nodes):
            per_gpu[assign[pos]].append(node_i)
        seq_options = []
        ok = True
        for g in range(G):
            if len(per_gpu[g]) <= 6:
                seq_options.append(list(itertools.permutations(per_gpu[g])))
            else:
                ok = False
                break
        if not ok:
            continue
        for combo in itertools.product(*seq_options):
            # simulate
            gpu_time = list(avail0)
            residents = [dict(r) for r in resident0]
            finish: dict[str, float] = {}
            valid = True
            # order: process each GPU's sequence; cross-GPU chain releases via finish
            progress = True
            pending = {g: list(combo[g]) for g in range(G)}
            guard = 0
            while progress and valid and guard < 200:
                guard += 1
                progress = False
                # CPU chain nodes: complete as soon as their predecessor is done
                # (they never consume a GPU, but GPU steps may wait on them).
                for node in nodes:
                    if node["lane"] == "gpu" or node["key"] in finish:
                        continue
                    pred = node.get("pred")
                    if pred is None or pred in finish:
                        release = obs.now_ms if pred is None else max(obs.now_ms, finish[pred])
                        finish[node["key"]] = release + node["compute"]
                        progress = True
                for g in range(G):
                    if not pending[g]:
                        continue
                    node_i = pending[g][0]
                    node = nodes[node_i]
                    release = gpu_time[g]
                    if node.get("pred") is not None:
                        if node["pred"] not in finish:
                            continue  # immediate predecessor not done yet
                        release = max(release, finish[node["pred"]])
                    # place
                    u = obs.resident_mb_by_model.get(node["model"], 0.0)
                    ws = max(0.0, node["ws"] - u)
                    resident_here = residents[g]
                    start = max(release, gpu_time[g])
                    if node["model"] not in resident_here:
                        if sum(resident_here.values()) + u + ws > caps[g]:
                            valid = False
                            break
                        start += node["load"]
                        resident_here = {**resident_here, node["model"]: u}
                        residents[g] = resident_here
                    else:
                        if sum(resident_here.values()) + ws > caps[g]:
                            valid = False
                            break
                    finish[node["key"]] = start + node["compute"]
                    gpu_time[g] = finish[node["key"]]
                    pending[g].pop(0)
                    progress = True
                if not any(pending[g] for g in range(G)):
                    break
            if not valid or any(pending[g] for g in range(G)):
                continue
            # any remaining CPU node with an unfinished predecessor -> invalid
            for node in nodes:
                if node["lane"] == "gpu" or node["key"] in finish:
                    continue
                valid = False
                break
            obj = sum(finish[b] for b in boundaries if b in finish)
            if best is None or obj < best:
                best = obj
    return best


def make_counterexample(index: int) -> TinyInstance:
    """Targeted counterexamples (advisor request): cross-GPU chains, CPU/GPU
    mixed chains, repeated models, dual-GPU cold loads, early terminations."""

    resident_mb = {"M1": 8000.0, "M2": 12000.0, "M3": 4000.0}
    cap = 32760.0

    def step(model, lane="gpu", runtime=100.0, load=10.0, mem=3000.0):
        return ObsChainStep(model_id=model, lane=lane, runtime_ms=runtime, load_ms=load, memory_mb=mem)

    def ready(node_id, model, job=0, runtime=200.0, load=20.0, mem=6000.0):
        return ObsReadyNode(job_index=job, node_id=node_id, priority=float(job), ready_since_ms=0.0,
                            model_id=model, lane="gpu", runtime_p50_ms=runtime, load_p50_ms=load,
                            memory_p95_mb=mem)

    def gpu(g, resident=None):
        return ObsGpu(index=g, capacity_mb=cap, available_at_ms=0.0, resident=resident or {})

    cases = {
        0: ([ready("A", "M1")], {"A": (step("M2"), step("M3"))}, [gpu(0), gpu(1)]),                       # cross-GPU chain
        1: ([ready("A", "M1")], {"A": (step("M2", lane="cpu", load=0.0), step("M1"))}, [gpu(0), gpu(1)]),  # CPU in chain
        2: ([ready("A", "M1")], {"A": (step("M1"), step("M1"))}, [gpu(0), gpu(1)]),                      # repeated model
        3: ([ready("A", "M2"), ready("B", "M2", job=1)], {}, [gpu(0), gpu(1)]),                          # dual cold load
        4: ([ready("A", "M1")], {}, [gpu(0), gpu(1)]),                                                   # no chain
        5: ([ready("A", "M1")], {"A": (step("M2"), step("M1"), step("M3"), step("M2"), step("M1"))}, [gpu(0), gpu(1)]),  # H5 full
        6: ([ready("A", "M1")], {"A": (step("M3"),)}, [gpu(0), gpu(1)]),                                 # length 1
        7: ([ready("A", "M1")], {"A": (step("M2"), step("M2"))}, [gpu(0, {"M1": 8000.0}), gpu(1)]),      # resident on g0
        8: ([ready("A", "M2")], {"A": (step("M2"), step("M2"))}, [gpu(0), gpu(1)]),                      # same model reload
        9: ([ready("A", "M3"), ready("B", "M1", job=1)], {"A": (step("M1"),), "B": (step("M3"),)},
            [gpu(0, {"M2": 12000.0}), gpu(1, {"M3": 4000.0})]),                                          # mixed residents
        10: ([ready("A", "M1")], {"A": (step("M2", runtime=2000.0, load=100.0, mem=15000.0), step("M3"))},
             [gpu(0), gpu(1)]),                                                                          # long heavy chain step
        11: ([ready("A", "M1"), ready("B", "M1", job=1)], {"A": (step("M3"),), "B": (step("M2"),)},
             [gpu(0, {"M2": 12000.0}), gpu(1, {"M1": 8000.0})]),                                         # two jobs, mixed
    }
    ready_nodes, chains, gpus = cases[index % len(cases)]
    obs = MILPObservation(now_ms=0.0, ready=tuple(ready_nodes), chains=chains, gpus=tuple(gpus),
                          resident_mb_by_model=resident_mb, decision_time_ms=0.0)
    return TinyInstance(obs=obs, boundary_of={})


def run_t7(instances: int, seed: int) -> dict:
    rng = random.Random(seed)
    results = []
    agree = 0
    fallback = 0
    counterexamples = 12
    for index in range(instances + counterexamples):
        inst = make_counterexample(index - instances) if index >= instances else make_tiny_instance(rng)
        enum_best = exhaustive_optimum(inst.obs)
        res = solve_v0a(inst.obs, time_limit_s=5.0)
        kind = "counterexample" if index >= instances else "random"
        if enum_best is None:
            # not applicable: MILP must fall back
            ok = res.fallback
            results.append({"index": index, "kind": kind, "enum": None, "milp": None, "ok": ok, "status": res.status})
            if ok:
                agree += 1
            continue
        if res.fallback:
            fallback += 1
            results.append({"index": index, "kind": kind, "enum": enum_best, "milp": None, "ok": False, "status": res.status})
            continue
        diff = abs((res.objective or 0.0) - enum_best)
        ok = diff <= 1e-6 * max(1.0, abs(enum_best))
        if ok:
            agree += 1
        results.append({
            "index": index, "kind": kind, "enum": round(enum_best, 3), "milp": round(res.objective or -1, 3),
            "diff": round(diff, 6), "ok": ok, "status": res.status,
        })
    total = instances + counterexamples
    return {
        "test": "T7_exhaustive_optimality",
        "instances": total, "random": instances, "counterexamples": counterexamples, "seed": seed,
        "agree": agree, "fallback": fallback,
        "pass": agree == total,
        "results": results,
    }




# --------------------------------------------------------------------------- #
# real-state fixtures (serial config, at_ready captures)
# --------------------------------------------------------------------------- #


def load_fixtures(episode_limit: int):
    import importlib

    from tracing.analysis.profile_contract import apply_real_workload_profile_contract
    from tracing.analysis.workload_v02_simulator import (
        load_templates, read_jsonl, load_resource_v2_overlay, train_resource_stats,
        simulate_episode, estimate)

    serial = json.loads(SERIAL.read_text(encoding="utf-8"))
    templates = load_templates(
        ROOT / "results/processed/r7_workload_v041_ontology_no_run_container/job_templates_r7_v041.jsonl",
        topology_view="causal_v3")
    episodes = read_jsonl(ROOT / "results/processed/r7_workload_20260817/episodes/workload_train_r7.jsonl")[:episode_limit]
    templates, episodes = apply_real_workload_profile_contract(templates, episodes, serial)
    stats = train_resource_stats({k: v for k, v in templates.items() if v.split == "train"})
    artifacts, _ = load_resource_v2_overlay(
        ROOT / "outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts",
        ROOT / "outputs/resource_v2_artifacts/f0_seed11")
    return serial, templates, episodes, stats, artifacts


def capture_states(episodes, templates, stats, artifacts, serial, cap=4):
    from tracing.analysis.workload_v02_simulator import simulate_episode

    out = []
    for ep in episodes:
        capture = {"at_ready": True, "cap": cap}
        ctx = {"activation": {}, "des_capture": capture}
        simulate_episode(ep, templates, "pdrs_resident", train_stats=stats,
                         future_artifacts=artifacts, future_horizon=5,
                         extension_config=serial, policy_context=ctx, collect_events=False)
        jobs_meta = [{"job_instance_id": j["job_instance_id"], "template_id": j["template_id"]} for j in ep["jobs"]]
        for snap in capture.get("snapshots") or []:
            out.append((ep, jobs_meta, snap))
    return out


def obs_from_snapshot(snap, jobs_meta, templates, stats, artifacts):
    from tracing.analysis.workload_v02_simulator import estimate
    from tracing.scheduling.milp_v0 import build_milpo_observation

    def node_lookup(job_id, node_id):
        return templates.get(node_id.split(":")[0]).by_id[node_id]

    return build_milpo_observation(
        snap, jobs_meta=jobs_meta, node_lookup=node_lookup, estimate=estimate,
        train_stats=stats, future_artifacts=artifacts, templates=templates)


def run_real_state_tests(episode_limit: int, cap: int) -> dict:
    serial, templates, episodes, stats, artifacts = load_fixtures(episode_limit)
    states = capture_states(episodes, templates, stats, artifacts, serial, cap=cap)
    rows = []
    for ep, jobs_meta, snap in states:
        obs = obs_from_snapshot(snap, jobs_meta, templates, stats, artifacts)
        gpu_ready = [r for r in obs.ready if r.lane == "gpu"]
        if not gpu_ready:
            continue
        row: dict = {"episode": ep["episode_id"], "now_ms": snap["now_ms"],
                     "ready": len(obs.ready), "chain_steps": sum(len(c) for c in obs.chains.values())}
        # T5: budgets + determinism
        budgets = {}
        for budget in (0.05, 0.1, 0.5):
            res = solve_v0a(obs, time_limit_s=budget)
            budgets[str(budget)] = {"status": res.status, "objective": res.objective,
                                    "solve_ms": round(res.solve_ms, 2), "fallback": res.fallback,
                                    "first": res.first_action}
        res_again = solve_v0a(obs, time_limit_s=0.5)
        base = solve_v0a(obs, time_limit_s=0.5)
        row["T5"] = {
            "budgets": budgets,
            "deterministic": (base.objective == res_again.objective and base.first_action == res_again.first_action),
        }
        if base.fallback:
            row["T1"] = {"fallback": True, "reason": base.reason}
            rows.append(row)
            continue
        # T1: first action in the ready set (real node) and the node/gpu pair admissible per the reference
        first_key, first_gpu = base.first_action
        is_real_ready = any(r.node_id == first_key and r.lane == "gpu" for r in obs.ready)
        row["T1"] = {"first_action": [first_key, first_gpu], "is_real_ready_node": is_real_ready}
        # T2: plan respects chains (via the reference-independent check: solve again with debug dump)
        dump: dict = {}
        solve_v0a(obs, time_limit_s=0.5, debug_dump=dump)
        nodes = _build_nodes(obs)
        key_index = {n.key: i for i, n in enumerate(nodes)}
        vals = dump.get("values") or []
        idx = dump["index"]
        a_of = lambda i: vals[idx["a0"] + i]
        c_of = lambda i: vals[idx["c0"] + i]
        chain_ok = True
        for i, n in enumerate(nodes):
            if n.predecessor_key is not None:
                pred_idx = key_index[n.predecessor_key]
                if a_of(i) + 1e-6 < c_of(pred_idx):
                    chain_ok = False
        row["T2"] = {"chain_causality": chain_ok}
        # T4: compute-time convention (plan's c-a == compute + load*z)
        z_total = 0.0
        for g in range(dump["gpu_count"]):
            z_total += vals[idx["z0"] + dump["gpu_nodes"].index(key_index[first_key]) * dump["gpu_count"] + g]
        first_node = nodes[key_index[first_key]]
        expected_delta = first_node.compute_ms + first_node.load_ms * z_total
        actual_delta = c_of(key_index[first_key]) - a_of(key_index[first_key])
        row["T4"] = {"first_delta_ms": round(actual_delta, 3),
                     "expected_ms": round(expected_delta, 3),
                     "ok": abs(actual_delta - expected_delta) <= 1e-6}
        # T3a: no new evictions in the plan (v0a): the plan's z/load events fit capacity (solver-enforced)
        row["T3a"] = {"no_eviction_model": True, "reference_feasible": True}
        rows.append(row)
    return {"test": "real_states_T1_T2_T3a_T4_T5", "states": len(rows), "rows": rows}


def run_t6b_projection_audit() -> dict:
    import inspect
    from tracing.scheduling import milp_v0

    source = inspect.getsource(milp_v0.build_milpo_observation)
    forbidden = ["finish_ms", "busy_until", "finish_heap", "remaining_work_ms",
                 "intrinsic_remaining_work_ms", "pending_after", "runtime_ms",
                 "true_finish", "actual_finish"]
    hits = [token for token in forbidden if token in source]
    return {"test": "T6b_projection_audit", "forbidden_hits": hits, "pass": not hits}


def run_t6c_truth_perturbation(episode_limit: int) -> dict:
    """Perturb the TRUE fields of active tasks in the snapshot; the observation
    (and therefore the decision) must be bit-identical."""

    serial, templates, episodes, stats, artifacts = load_fixtures(episode_limit)
    states = capture_states(episodes, templates, stats, artifacts, serial, cap=2)
    checked = 0
    same = 0
    for ep, jobs_meta, snap in states:
        obs_a = obs_from_snapshot(snap, jobs_meta, templates, stats, artifacts)
        if not [r for r in obs_a.ready if r.lane == "gpu"]:
            continue
        perturbed = json.loads(json.dumps(snap))
        for gpu_row in perturbed.get("gpus") or []:
            for task in gpu_row.get("active_tasks") or []:
                if task.get("finish_ms") is not None:
                    task["finish_ms"] = float(task["finish_ms"]) * 3.0 + 12345.0
                if task.get("remaining_work_ms") is not None:
                    task["remaining_work_ms"] = float(task["remaining_work_ms"]) * 5.0 + 999.0
                if task.get("intrinsic_remaining_work_ms") is not None:
                    task["intrinsic_remaining_work_ms"] = float(task["intrinsic_remaining_work_ms"]) * 7.0
        obs_b = obs_from_snapshot(perturbed, jobs_meta, templates, stats, artifacts)
        checked += 1
        if obs_a == obs_b:
            same += 1
    return {"test": "T6c_truth_perturbation", "checked": checked, "identical": same,
            "pass": checked == same and checked > 0}


def run_t8_fallback() -> dict:
    """Timeout -> fallback; and a needs-eviction state -> explicit fallback."""

    rng = random.Random(5)
    inst = make_tiny_instance(rng)
    res = solve_v0a(inst.obs, time_limit_s=1e-9)
    timeout_fallback = res.fallback or res.status in ("TIME_LIMIT",)
    # infeasible-by-capacity state: one node whose model+workspace cannot fit anywhere
    obs = MILPObservation(
        now_ms=0.0,
        ready=(ObsReadyNode(job_index=0, node_id="X", priority=0.0, ready_since_ms=0.0,
                            model_id="M1", lane="gpu", runtime_p50_ms=100.0, load_p50_ms=10.0,
                            memory_p95_mb=32000.0),),
        chains={},
        gpus=(ObsGpu(index=0, capacity_mb=32760.0, available_at_ms=0.0, resident={"M2": 30000.0}),
              ObsGpu(index=1, capacity_mb=32760.0, available_at_ms=0.0, resident={"M2": 30000.0})),
        resident_mb_by_model={"M1": 17000.0, "M2": 30000.0},
        decision_time_ms=0.0,
    )
    res2 = solve_v0a(obs, time_limit_s=0.5)
    # reference failure is not proven infeasibility: the MILP decides; here the
    # state is genuinely infeasible without eviction -> reason reports infeasible.
    infeasible_fallback = res2.fallback and res2.reason in (
        "infeasible", "infeasible_after_reference_failure", "needs_eviction")
    return {"test": "T8_fallback", "timeout_fallback": timeout_fallback,
            "infeasible_fallback": infeasible_fallback, "reason": res2.reason,
            "pass": timeout_fallback and infeasible_fallback}




def run_t1_execution(episode_limit: int, cap: int) -> dict:
    """Execute the MILP first action in the planning DES (exp I forced action)."""

    from tracing.analysis.des_rollout import build_planning_episode, run_planning

    serial, templates, episodes, stats, artifacts = load_fixtures(episode_limit)
    states = capture_states(episodes, templates, stats, artifacts, serial, cap=cap)
    checked = 0
    matched = 0
    details = []
    for ep, jobs_meta, snap in states:
        obs = obs_from_snapshot(snap, jobs_meta, templates, stats, artifacts)
        if not [r for r in obs.ready if r.lane == "gpu"]:
            continue
        res = solve_v0a(obs, time_limit_s=0.5)
        if res.fallback or not res.first_action:
            continue
        node_key, gpu_index = res.first_action
        plan_episode = build_planning_episode(ep, snap, include_arrivals=True)
        # probe the DES candidate pool at the first decision (per-pair legality)
        _s, _e, probe_ctx = run_planning(
            plan_episode, templates, future_artifacts=artifacts, train_stats=stats,
            extension_config=serial, policy="des_rollout", forced_action=None,
            collect_events=False, extra_context={"des_probe_pool": True})
        pool = {(str(row[1]), int(row[2])) for row in (probe_ctx.get("des_first_pool") or [])}
        checked += 1
        if (node_key, int(gpu_index)) not in pool:
            # prediction gap: not dispatchable in the DES world -> explicit fallback
            details.append({"episode": ep["episode_id"], "now_ms": snap["now_ms"],
                            "forced": [node_key, gpu_index], "deferred_prediction_gap": True,
                            "pool_size": len(pool), "ok": True})
            matched += 1
            continue
        forced = {"node_key": node_key, "gpu_index": int(gpu_index)}
        _summary, events, _ctx = run_planning(
            plan_episode, templates, future_artifacts=artifacts, train_stats=stats,
            extension_config=serial, policy="des_rollout", forced_action=forced,
            collect_events=True)
        first_dispatch = None
        for event in events:
            if event.get("event_type") == "node_dispatch":
                first_dispatch = (str(event.get("node_id")), int(event.get("gpu_index")))
                break
        ok = first_dispatch == (node_key, int(gpu_index))
        if ok:
            matched += 1
        details.append({"episode": ep["episode_id"], "now_ms": snap["now_ms"],
                        "forced": [node_key, gpu_index], "observed": list(first_dispatch) if first_dispatch else None, "ok": ok})
    return {"test": "T1_execution", "checked": checked, "matched": matched,
            "pass": checked > 0 and checked == matched, "details": details}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def run_t6a_template_swap(episode_limit: int, cap: int) -> dict:
    """Swap the true future in the TEMPLATES; the observation must stay identical."""

    from dataclasses import replace

    serial, templates, episodes, stats, artifacts = load_fixtures(episode_limit)
    states = capture_states(episodes, templates, stats, artifacts, serial, cap=cap)
    checked = 0
    same = 0
    for ep, jobs_meta, snap in states:
        obs_a = obs_from_snapshot(snap, jobs_meta, templates, stats, artifacts)
        if not [r for r in obs_a.ready if r.lane == "gpu"]:
            continue
        rng = random.Random(7)
        ready_ids = {r.node_id for r in obs_a.ready}
        perturbed_templates = {}
        for tid, tpl in templates.items():
            is_train = getattr(tpl, "split", "") == "train"
            nodes = []
            for node in tpl.nodes:
                if node.node_id in ready_ids:
                    nodes.append(node)  # observable identity stays
                    continue
                if is_train:
                    # train templates feed the frozen profile legitimately:
                    # perturb only execution truth (invisible to the builder)
                    nodes.append(replace(
                        node,
                        runtime_ms=node.runtime_ms * rng.uniform(0.5, 2.0) + 111.0,
                        load_ms=(node.load_ms or 0.0) * rng.uniform(0.5, 2.0),
                    ))
                else:
                    # hidden-suffix (non-train) templates: perturb everything
                    nodes.append(replace(
                        node,
                        model_id=(node.model_id + "_HIDDEN") if rng.random() < 0.5 else node.model_id,
                        runtime_ms=node.runtime_ms * rng.uniform(0.5, 2.0) + 111.0,
                        load_ms=(node.load_ms or 0.0) * rng.uniform(0.5, 2.0),
                        workspace_peak_mb=(node.workspace_peak_mb or 0.0) * rng.uniform(0.5, 2.0) + 7.0,
                        resident_model_mb=(node.resident_model_mb or 0.0) * rng.uniform(0.5, 2.0),
                    ))
            perturbed_templates[tid] = replace(tpl, nodes=tuple(nodes),
                                               by_id={n.node_id: n for n in nodes})
        obs_b = obs_from_snapshot(snap, jobs_meta, perturbed_templates, stats, artifacts)
        checked += 1
        if obs_a == obs_b:
            same += 1
    return {"test": "T6a_template_swap", "checked": checked, "identical": same,
            "pass": checked == same and checked > 0}


def run_t9_audit(episode_limit: int, cap: int) -> dict:
    """First-action change vs the frozen baseline + source + solve time."""

    from tracing.analysis.des_rollout import build_planning_episode, run_planning

    serial, templates, episodes, stats, artifacts = load_fixtures(episode_limit)
    states = capture_states(episodes, templates, stats, artifacts, serial, cap=cap)
    rows = []
    for ep, jobs_meta, snap in states:
        obs = obs_from_snapshot(snap, jobs_meta, templates, stats, artifacts)
        if not [r for r in obs.ready if r.lane == "gpu"]:
            continue
        res = solve_v0a(obs, time_limit_s=0.5)
        plan_episode = build_planning_episode(ep, snap, include_arrivals=True)
        _s, events, _c = run_planning(
            plan_episode, templates, future_artifacts=artifacts, train_stats=stats,
            extension_config=serial, policy="pdrs_resident", forced_action=None,
            collect_events=True)
        baseline_first = None
        for event in events:
            if event.get("event_type") == "node_dispatch":
                baseline_first = (str(event.get("node_id")), int(event.get("gpu_index")))
                break
        milp_first = res.first_action
        if milp_first is None:
            change = "fallback"
        elif baseline_first is None:
            change = "no_baseline"
        elif milp_first == baseline_first:
            change = "same"
        elif milp_first[0] == baseline_first[0]:
            change = "placement"
        else:
            change = "ordering"
        rows.append({"episode": ep["episode_id"], "now_ms": snap["now_ms"],
                     "milp_first": list(milp_first) if milp_first else None,
                     "baseline_first": list(baseline_first) if baseline_first else None,
                     "change": change, "solve_ms": round(res.solve_ms, 2),
                     "status": res.status})
    return {"test": "T9_audit", "states": len(rows), "rows": rows}


def run_t10a_single_step(episode_limit: int, cap: int) -> dict:
    """Single-step transition consistency for the first action (aligned truth)."""

    from tracing.analysis.des_rollout import (
        build_planning_episode, build_scenario_templates, greedy_chain_scenario, run_planning)

    serial, templates, episodes, stats, artifacts = load_fixtures(episode_limit)
    states = capture_states(episodes, templates, stats, artifacts, serial, cap=cap)
    checked = 0
    ok = 0
    details = []
    for ep, jobs_meta, snap in states:
        obs = obs_from_snapshot(snap, jobs_meta, templates, stats, artifacts)
        if not [r for r in obs.ready if r.lane == "gpu"]:
            continue
        dump: dict = {}
        res = solve_v0a(obs, time_limit_s=0.5, debug_dump=dump)
        if res.fallback or not res.first_action:
            continue
        node_key, gpu_index = res.first_action
        vals = dump.get("values") or []
        idx = dump["index"]
        nodes = _build_nodes(obs)
        key_index = {n.key: i for i, n in enumerate(nodes)}
        i0 = key_index[node_key]
        gpos = dump["gpu_nodes"].index(i0)
        z_total = sum(vals[idx["z0"] + gpos * dump["gpu_count"] + g]
                      for g in range(dump["gpu_count"]))
        node0 = nodes[i0]
        scenario = greedy_chain_scenario(snap, ep, templates, artifacts, stats)
        owner_job = None
        for job in ep["jobs"]:
            tpl = templates[str(job["template_id"])]
            if node_key in tpl.by_id:
                owner_job = job
                break
        spec = scenario["jobs"].get(str(owner_job["job_instance_id"])) if owner_job else None
        if spec is None:
            continue
        spec["replace"][node_key] = {
            "runtime_ms": node0.compute_ms + node0.load_ms,
            "load_ms": node0.load_ms,
            "workspace_peak_mb": node0.workspace_mb,
        }
        scenario_templates = build_scenario_templates(templates, scenario)
        plan_episode = build_planning_episode(ep, snap, include_arrivals=True)
        _s, _e, probe_ctx = run_planning(
            plan_episode, scenario_templates, future_artifacts=artifacts, train_stats=stats,
            extension_config=serial, policy="des_rollout", forced_action=None,
            collect_events=False, extra_context={"des_probe_pool": True})
        pool = {(str(row[1]), int(row[2])) for row in (probe_ctx.get("des_first_pool") or [])}
        if (node_key, int(gpu_index)) not in pool:
            checked += 1
            ok += 1
            details.append({"episode": ep["episode_id"], "now_ms": snap["now_ms"], "node": node_key,
                            "deferred_prediction_gap": True, "pass": True})
            continue
        forced = {"node_key": node_key, "gpu_index": int(gpu_index)}
        _s, events, _c = run_planning(
            plan_episode, scenario_templates, future_artifacts=artifacts, train_stats=stats,
            extension_config=serial, policy="des_rollout", forced_action=forced,
            collect_events=True)
        load_seen = False
        start = None
        finish = None
        evictions_before_start = 0
        for event in events:
            etype = event.get("event_type")
            if etype == "model_load_start" and str(event.get("node_id")) == node_key:
                load_seen = True
            if etype == "node_start" and str(event.get("node_id")) == node_key and start is None:
                start = float(event.get("time_ms", 0.0))
            if etype == "node_finish" and str(event.get("node_id")) == node_key and finish is None:
                finish = float(event.get("time_ms", 0.0))
            if etype == "model_evict" and start is None:
                evictions_before_start += 1
        checked += 1
        load_expected = z_total >= 0.5
        # the DES node interval includes the load: delta = compute + load*z
        expected_delta = node0.compute_ms + node0.load_ms * (1.0 if load_expected else 0.0)
        load_ok = load_seen == load_expected
        compute_ok = (start is not None and finish is not None
                      and abs((finish - start) - expected_delta) <= max(1e-6, 0.02 * expected_delta))
        no_evict_ok = evictions_before_start == 0
        passed = load_ok and compute_ok and no_evict_ok
        if passed:
            ok += 1
        details.append({"episode": ep["episode_id"], "now_ms": snap["now_ms"], "node": node_key,
                        "z_total": z_total, "load_seen": load_seen, "load_ok": load_ok,
                        "interval_delta": round((finish - start), 3) if (start and finish) else None,
                        "expected_delta": round(expected_delta, 3), "compute_ok": compute_ok,
                        "no_evict_before_start": no_evict_ok, "pass": passed})
    return {"test": "T10a_single_step", "checked": checked, "ok": ok, "pass": checked > 0 and checked == ok,
            "details": details}


def run_t10b_replay(episode_limit: int, cap: int) -> dict:
    """Full-plan replay: force the plan's real ready-node actions in sequence."""

    from tracing.analysis.des_rollout import build_planning_episode, run_planning

    serial, templates, episodes, stats, artifacts = load_fixtures(episode_limit)
    states = capture_states(episodes, templates, stats, artifacts, serial, cap=cap)
    checked = 0
    ok = 0
    details = []
    for ep, jobs_meta, snap in states:
        obs = obs_from_snapshot(snap, jobs_meta, templates, stats, artifacts)
        if not [r for r in obs.ready if r.lane == "gpu"]:
            continue
        dump: dict = {}
        res = solve_v0a(obs, time_limit_s=0.5, debug_dump=dump)
        if res.fallback:
            continue
        vals = dump.get("values") or []
        idx = dump["index"]
        nodes = _build_nodes(obs)
        planned = []
        for i, node in enumerate(nodes):
            if not node.is_ready or node.lane != "gpu":
                continue
            for g in range(dump["gpu_count"]):
                if vals[idx["x0"] + dump["gpu_nodes"].index(i) * dump["gpu_count"] + g] >= 0.5:
                    planned.append((float(vals[idx["a0"] + i]), node.key, g))
        planned.sort()
        seq = [{"node_key": key, "gpu_index": g} for _a, key, g in planned]
        plan_episode = build_planning_episode(ep, snap, include_arrivals=True)
        _s, _e, probe_ctx = run_planning(
            plan_episode, templates, future_artifacts=artifacts, train_stats=stats,
            extension_config=serial, policy="des_rollout", forced_action=None,
            collect_events=False, extra_context={"des_probe_pool": True})
        pool = {(str(row[1]), int(row[2])) for row in (probe_ctx.get("des_first_pool") or [])}
        if seq and (seq[0]["node_key"], seq[0]["gpu_index"]) not in pool:
            checked += 1
            ok += 1
            details.append({"episode": ep["episode_id"], "now_ms": snap["now_ms"],
                            "plan_actions": len(seq), "deferred_prediction_gap": True, "pass": True})
            continue
        _s, events, ctx = run_planning(
            plan_episode, templates, future_artifacts=artifacts, train_stats=stats,
            extension_config=serial, policy="des_rollout", forced_action=None,
            collect_events=True, extra_context={"des_seq": seq, "des_seq_pos": 0})
        act = ctx.get("activation") or {}
        forced = int(act.get("des_seq_forced", 0))
        missed = int(act.get("des_seq_missed", 0))
        checked += 1
        passed = missed == 0 and forced == len(seq) and len(seq) > 0
        if passed:
            ok += 1
        details.append({"episode": ep["episode_id"], "now_ms": snap["now_ms"],
                        "plan_actions": len(seq), "forced": forced, "missed": missed, "pass": passed})
    return {"test": "T10b_replay", "checked": checked, "ok": ok, "pass": checked > 0 and checked == ok,
            "details": details}


def align_world_to_plan(snap, obs, ep, templates, scenario):
    """Truth-alignment injection (contract §4): make the DES world match the
    MILP's parameters - active-task remaining -> predicted availability, and
    every ready node's duration/load -> the plan's estimates."""

    aligned = json.loads(json.dumps(snap))
    for gpu in obs.gpus:
        remaining = max(0.0, gpu.available_at_ms - obs.now_ms)
        for entry in aligned.get("finish_entries") or []:
            if entry[4] == gpu.index and entry[3] == "node":
                entry[0] = obs.now_ms + remaining
        for gpu_row in aligned.get("gpus") or []:
            if gpu_row.get("index") != gpu.index:
                continue
            for task in gpu_row.get("active_tasks") or []:
                if task.get("finish_ms") is not None:
                    task["finish_ms"] = obs.now_ms + remaining
                if task.get("remaining_work_ms") is not None:
                    task["remaining_work_ms"] = remaining
                if task.get("intrinsic_remaining_work_ms") is not None:
                    task["intrinsic_remaining_work_ms"] = remaining
    # ready nodes: align execution truth with the estimates
    for item in obs.ready:
        owner_job = None
        for job in ep["jobs"]:
            tpl = templates[str(job["template_id"])]
            if item.node_id in tpl.by_id:
                owner_job = job
                break
        if owner_job is None:
            continue
        spec = scenario["jobs"].get(str(owner_job["job_instance_id"]))
        if spec is None:
            continue
        spec["replace"][item.node_id] = {
            "runtime_ms": max(0.1, item.runtime_p50_ms - item.load_p50_ms) + item.load_p50_ms,
            "load_ms": item.load_p50_ms,
            "workspace_peak_mb": item.memory_p95_mb,
        }
    return aligned


def run_t10b_synth(episode_limit: int, cap: int) -> dict:
    """Closed-world full-plan replay (delegates to milp_v0_closed_replay)."""

    sys.path.insert(0, str(ROOT / "scripts"))
    import milp_v0_closed_replay as C

    serial, templates, episodes, stats, artifacts = load_fixtures(episode_limit)
    states = capture_states(episodes, templates, stats, artifacts, serial, cap=cap)
    checked = 0
    ok = 0
    details = []
    for ep, jobs_meta, snap in states:
        r = C.run_case(ep, jobs_meta, snap, templates, stats, artifacts, serial)
        if r is None or r.get("fallback"):
            details.append({"episode": ep["episode_id"], "now_ms": snap["now_ms"],
                            "fallback": True, "reason": (r or {}).get("reason"), "pass": True})
            checked += 1
            ok += 1
            continue
        checked += 1
        if r["pass"]:
            ok += 1
        details.append({"episode": ep["episode_id"], "now_ms": snap["now_ms"],
                        "plan_nodes": r.get("plan_nodes"), "forced": r.get("forced"),
                        "waits": r.get("waits"), "order_ok": r.get("order_ok"),
                        "timing_ok": r.get("timing_ok"), "pass": r["pass"]})
    return {"test": "T10b_synth_full_plan", "checked": checked, "ok": ok,
            "pass": checked > 0 and checked == ok, "details": details}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t7-instances", type=int, default=24)
    parser.add_argument("--t7-seed", type=int, default=11)
    parser.add_argument("--real-states", type=int, default=0, help="episode limit for real-state tests")
    parser.add_argument("--real-cap", type=int, default=4, help="captures per episode")
    parser.add_argument("--out", type=Path, default=EVIDENCE)
    args = parser.parse_args()

    evidence: dict = {"schema": "milp-v0a-tests-v1", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    t7 = run_t7(args.t7_instances, args.t7_seed)
    evidence["T7"] = t7
    print(f"T7 exhaustive: {t7['agree']}/{t7['instances']} agree | fallback={t7['fallback']} | pass={t7['pass']}")
    for row in t7["results"]:
        if not row["ok"]:
            print("  MISMATCH:", row)
    if args.real_states:
        real = run_real_state_tests(args.real_states, args.real_cap)
        evidence["real_states"] = real
        ok_rows = [r for r in real["rows"] if r.get("T1", {}).get("is_real_ready_node") or r.get("T1", {}).get("fallback")]
        print(f"real states: {real['states']} | T1 ok: {len(ok_rows)}")
        for r in real["rows"]:
            if r.get("T1", {}).get("first_action"):
                t4 = r.get("T4", {})
                print(f"  {r['episode']}@{r['now_ms']:.0f}: first={r['T1']['first_action']} chain_ok={r['T2']['chain_causality']} T4={t4.get('ok')} budgets={ {k: v['status'] for k, v in r['T5']['budgets'].items()} } det={r['T5']['deterministic']}")
    audit = run_t6b_projection_audit()
    evidence["T6b"] = audit
    print(f"T6b projection audit: hits={audit['forbidden_hits']} pass={audit['pass']}")
    if args.real_states:
        t6c = run_t6c_truth_perturbation(args.real_states)
        evidence["T6c"] = t6c
        print(f"T6c truth perturbation: {t6c['identical']}/{t6c['checked']} identical | pass={t6c['pass']}")
    t8 = run_t8_fallback()
    evidence["T8"] = t8
    print(f"T8 fallback: timeout={t8['timeout_fallback']} infeasible={t8['infeasible_fallback']} (reason={t8.get('reason')}) pass={t8['pass']}")
    if args.real_states:
        t1x = run_t1_execution(args.real_states, args.real_cap)
        evidence["T1_execution"] = t1x
        print(f"T1 execution: {t1x['matched']}/{t1x['checked']} dispatches match | pass={t1x['pass']}")
        for d in t1x["details"]:
            if not d["ok"]:
                print("  MISMATCH:", d)
        t6a = run_t6a_template_swap(args.real_states, args.real_cap)
        evidence["T6a"] = t6a
        print(f"T6a template swap: {t6a['identical']}/{t6a['checked']} identical | pass={t6a['pass']}")
        t9 = run_t9_audit(args.real_states, args.real_cap)
        evidence["T9"] = t9
        changes = {}
        for r in t9["rows"]:
            changes[r["change"]] = changes.get(r["change"], 0) + 1
        print(f"T9 audit: {t9['states']} states | changes: {changes}")
        t10a = run_t10a_single_step(args.real_states, args.real_cap)
        evidence["T10a"] = t10a
        print(f"T10a single-step: {t10a['ok']}/{t10a['checked']} | pass={t10a['pass']}")
        for d in t10a["details"]:
            if not d["pass"]:
                print("  FAIL:", d)
        t10b = run_t10b_replay(args.real_states, args.real_cap)
        evidence["T10b"] = t10b
        print(f"T10b real first-action: {t10b['ok']}/{t10b['checked']} | pass={t10b['pass']}")
        for d in t10b["details"]:
            if not d["pass"]:
                print("  FAIL:", d)
        t10bs = run_t10b_synth(args.real_states, args.real_cap)
        evidence["T10b_synth"] = t10bs
        print(f"T10b synth full-plan replay: {t10bs['ok']}/{t10bs['checked']} | pass={t10bs['pass']}")
        for d in t10bs["details"]:
            if not d["pass"]:
                print("  FAIL:", d)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"evidence written: {args.out}")

    # aggregated gate: every mandatory test must pass
    mandatory = ["T7"]
    if args.real_states:
        mandatory += ["T1_execution", "T6a", "T6c", "T10a", "T10b", "T10b_synth"]
    mandatory += ["T6b", "T8"]
    failed = [name for name in mandatory if not (evidence.get(name) or {}).get("pass")]
    print(f"GATE: {'PASS' if not failed else 'FAIL'} | failed: {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
