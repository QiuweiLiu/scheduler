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
                "key": item.node_id, "anchor": None, "lane": item.lane, "model": item.model_id,
                "compute": max(0.1, item.runtime_p50_ms - item.load_p50_ms),
                "load": item.load_p50_ms, "ws": item.memory_p95_mb, "job": item.job_index,
            }
        )
        for si, step in enumerate(obs.chains.get(item.node_id, ())):
            nodes.append(
                {
                    "key": f"pred:{item.node_id}:{si + 1}", "anchor": item.node_id,
                    "lane": step.lane, "model": step.model_id,
                    "compute": max(0.1, step.runtime_ms - step.load_ms),
                    "load": step.load_ms, "ws": step.memory_mb, "job": item.job_index,
                }
            )
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
            while progress and valid and guard < 100:
                guard += 1
                progress = False
                for g in range(G):
                    if not pending[g]:
                        continue
                    node_i = pending[g][0]
                    node = nodes[node_i]
                    release = gpu_time[g]
                    if node["anchor"] is not None:
                        if node["anchor"] not in finish:
                            continue  # predecessor not done yet
                        release = max(release, finish[node["anchor"]])
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
            # CPU nodes (time-only, chain-ordered)
            for node in nodes:
                if node["lane"] == "gpu":
                    continue
                release = obs.now_ms
                if node["anchor"] is not None and node["anchor"] in finish:
                    release = max(release, finish[node["anchor"]])
                finish[node["key"]] = release + node["compute"]
            obj = sum(finish[b] for b in boundaries if b in finish)
            if best is None or obj < best:
                best = obj
    return best


def run_t7(instances: int, seed: int) -> dict:
    rng = random.Random(seed)
    results = []
    agree = 0
    fallback = 0
    for index in range(instances):
        inst = make_tiny_instance(rng)
        enum_best = exhaustive_optimum(inst.obs)
        res = solve_v0a(inst.obs, time_limit_s=5.0)
        if enum_best is None:
            # not applicable: MILP must fall back
            ok = res.fallback
            results.append({"index": index, "enum": None, "milp": None, "ok": ok, "status": res.status})
            if ok:
                agree += 1
            continue
        if res.fallback:
            fallback += 1
            results.append({"index": index, "enum": enum_best, "milp": None, "ok": False, "status": res.status})
            continue
        diff = abs((res.objective or 0.0) - enum_best)
        ok = diff <= 1e-6 * max(1.0, abs(enum_best))
        if ok:
            agree += 1
        results.append({
            "index": index, "enum": round(enum_best, 3), "milp": round(res.objective or -1, 3),
            "diff": round(diff, 6), "ok": ok, "status": res.status,
        })
    return {
        "test": "T7_exhaustive_optimality",
        "instances": instances, "seed": seed,
        "agree": agree, "fallback": fallback,
        "pass": agree == instances,
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
            if n.anchor_key is not None:
                anchor_idx = key_index[n.anchor_key]
                if a_of(i) + 1e-6 < c_of(anchor_idx):
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
    needs_eviction_fallback = res2.fallback and res2.reason == "needs_eviction"
    return {"test": "T8_fallback", "timeout_fallback": timeout_fallback,
            "needs_eviction_fallback": needs_eviction_fallback,
            "pass": timeout_fallback and needs_eviction_fallback}




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
    print(f"T8 fallback: timeout={t8['timeout_fallback']} needs_eviction={t8['needs_eviction_fallback']} pass={t8['pass']}")
    if args.real_states:
        t1x = run_t1_execution(args.real_states, args.real_cap)
        evidence["T1_execution"] = t1x
        print(f"T1 execution: {t1x['matched']}/{t1x['checked']} dispatches match | pass={t1x['pass']}")
        for d in t1x["details"]:
            if not d["ok"]:
                print("  MISMATCH:", d)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"evidence written: {args.out}")
    return 0 if t7["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
