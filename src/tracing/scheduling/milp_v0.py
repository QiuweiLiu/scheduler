"""H5-MILP-v0a: serial-environment MILP with strict observation projection.

Implements the v3 formulation (experiments/EXP-20261012_milp_v0/MILP_FORMULATION.md):
- MILPObservation: the ONLY input surface (no true finish times, no busy_until,
  no finish_heap, no hidden suffixes, no real pending_after nodes);
- v0a scope: no new evictions (states needing eviction -> not applicable -> fallback);
- GPU-conditional cold-load variables z_{ig}, same-GPU precedence y_{i'ig},
  closed time definitions (a_i, s_i, c_i), event-level memory accounting;
- solver: scipy.optimize.milp (HiGHS), reuse of the milp_sanity interface pattern.

First action extraction: earliest real ready GPU node, ties by the frozen F0 key;
post-validated against the DES legal action set by the caller (T1).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from tracing.scheduling.cp_rho import CpRhoCandidate


# --------------------------------------------------------------------------- #
# MILPObservation: strict projection
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ObsReadyNode:
    job_index: int
    node_id: str
    priority: float
    ready_since_ms: float
    model_id: str
    lane: str
    runtime_p50_ms: float
    load_p50_ms: float
    memory_p95_mb: float


@dataclass(frozen=True)
class ObsChainStep:
    model_id: str
    lane: str
    runtime_ms: float
    load_ms: float
    memory_mb: float


@dataclass(frozen=True)
class ObsGpu:
    index: int
    capacity_mb: float
    available_at_ms: float
    resident: Mapping[str, float]


@dataclass(frozen=True)
class MILPObservation:
    now_ms: float
    ready: tuple[ObsReadyNode, ...]
    chains: Mapping[str, tuple[ObsChainStep, ...]]  # anchor node_id -> H5 chain
    gpus: tuple[ObsGpu, ...]
    resident_mb_by_model: Mapping[str, float]
    decision_time_ms: float
    resource_missing: tuple[str, ...] = ()  # models without a frozen train-only profile
    # 2026-10-13 (advisor P0): currently-legal (node_id, gpu_index) pairs for the
    # IMMEDIATE dispatch.  None = no mask (offline/test default); an empty set
    # means "no legal immediate action" (solve must fall back).  Only the first
    # action is masked; future H5 planning is not.
    legal_first: frozenset[tuple[str, int]] | None = None

    # Allowed sources (audited by T6b): ready heap, estimate(), artifact rows,
    # gpu.resident, active-task predicted_work_ms/work_start_ms/slowdown/kind.

    def gpu_available(self, index: int) -> float:
        return float(self.gpus[index].available_at_ms)


def _chain_steps_from_artifact(
    anchor_node_id: str,
    artifact_row: Mapping[str, Any],
    estimate_lookup,
    missing: list[str],
) -> tuple[ObsChainStep, ...]:
    """Synthetic H5 chain steps from the frozen artifact (train-only profile)."""

    scenarios = artifact_row.get("future_h5") or []
    steps = (scenarios[0].get("steps") or []) if scenarios else []
    chain: list[ObsChainStep] = []
    for step in steps[:5]:
        resource = step.get("resource") or {}
        runtime_q = resource.get("runtime_ms_quantiles") or {}
        load_q = resource.get("load_duration_ms_quantiles") or {}
        occurrence = resource.get("load_occurrence_probability")
        model_id = str(step.get("model_id") or "")
        lane = str(step.get("execution_lane") or "gpu")
        runtime = float(runtime_q.get("p50") or 0.0)
        load = (
            float(load_q.get("p50") or 0.0)
            if isinstance(occurrence, (int, float)) and float(occurrence) >= 0.5
            else 0.0
        )
        memory_value = estimate_lookup(model_id, str(step.get("role") or "other"), lane)
        if memory_value is None:
            missing.append(model_id)
            memory = 0.0
        else:
            memory = float(memory_value)
        chain.append(
            ObsChainStep(
                model_id=model_id,
                lane=lane,
                runtime_ms=runtime,
                load_ms=load,
                memory_mb=memory,
            )
        )
    return tuple(chain)


_PROFILE_CACHE: dict[tuple[int, int], tuple[Any, Any, dict[str, float], dict[tuple[str, str], float]]] = {}


def _train_profile_index(templates, train_stats, estimate):
    """Advisor P1: precompute + cache the train-only resource catalog.

    Keyed by object identity of (templates, train_stats) so repeated decisions
    in one episode/simulation reuse the same catalog instead of rescanning all
    templates.  Rebuilt only when a different experiment load is passed in.
    """

    key = (id(templates), id(train_stats))
    hit = _PROFILE_CACHE.get(key)
    if hit is not None and hit[0] is templates and hit[1] is train_stats:
        return hit[2], hit[3]
    resident_index: dict[str, float] = {}
    memory_by: dict[tuple[str, str], float] = {}
    for template in templates.values():
        if getattr(template, "split", "") != "train":
            continue
        for node in template.nodes:
            if node.resident_model_mb is not None and node.model_id not in resident_index:
                resident_index[str(node.model_id)] = float(node.resident_model_mb)
            mk = (str(node.model_id), str(node.lane))
            if mk not in memory_by:
                try:
                    memory_by[mk] = float(estimate(node, train_stats)["memory_p95_mb"] or 0.0)
                except Exception:
                    continue
    _PROFILE_CACHE[key] = (templates, train_stats, resident_index, memory_by)
    return resident_index, memory_by


def build_milpo_observation(
    snapshot: Mapping[str, Any],
    *,
    jobs_meta: Sequence[Mapping[str, Any]],
    node_lookup,
    estimate,
    train_stats: Mapping[str, Mapping[str, Any]],
    future_artifacts: Mapping[str, Mapping[str, Any]],
    templates: Mapping[str, Any],
    legal_first: frozenset[tuple[str, int]] | None = None,
) -> MILPObservation:
    """Build the strict observation from an experiment-I state snapshot.

    ``jobs_meta``: [{job_instance_id, template_id}] aligned with snapshot jobs.
    ``node_lookup(job_instance_id, node_id) -> Node`` resolves ready nodes.
    ``legal_first``: currently-legal (node_id, gpu_index) pairs for the first
    action (advisor P0 action mask); None keeps the offline/test default.
    Only the allowed fields below are read from the snapshot.
    """

    now = float(snapshot["now_ms"])
    ready: list[ObsReadyNode] = []
    for priority, ready_time, job_index, node_id in snapshot.get("ready_items") or []:
        job = jobs_meta[int(job_index)]
        node = node_lookup(str(job["job_instance_id"]), str(node_id))
        row = estimate(node, train_stats)
        ready.append(
            ObsReadyNode(
                job_index=int(job_index),
                node_id=str(node_id),
                priority=float(priority),
                ready_since_ms=float(ready_time),
                model_id=str(node.model_id),
                lane=str(node.lane),
                runtime_p50_ms=float(row["runtime_p50_ms"]),
                load_p50_ms=float(row["load_p50_ms"] or 0.0),
                memory_p95_mb=float(row["memory_p95_mb"] or 0.0),
            )
        )

    # Frozen train-only model profile (advisor P1): never complete unknown
    # models from non-train templates (hidden suffixes).  Precomputed + cached
    # per (templates, train_stats) identity (advisor P1 cost optimization).
    resident_index, memory_by = _train_profile_index(templates, train_stats, estimate)

    def estimate_lookup(model_id: str, role: str, lane: str):
        return memory_by.get((str(model_id), str(lane)))

    gpus: list[ObsGpu] = []
    for gpu_row in snapshot.get("gpus") or []:
        # Allowed: predicted_work_ms / work_start_ms / slowdown / kind only.
        remaining = 0.0
        for task in gpu_row.get("active_tasks") or []:
            predicted = task.get("predicted_work_ms")
            work_start = task.get("work_start_ms")
            slowdown = float(task.get("slowdown") or 1.0)
            if predicted is None or work_start is None:
                continue
            elapsed = max(0.0, now - float(work_start))
            remaining = max(remaining, max(0.0, float(predicted) * slowdown - elapsed))
        gpus.append(
            ObsGpu(
                index=int(gpu_row["index"]),
                capacity_mb=float(gpu_row.get("capacity_mb") or 0.0),
                available_at_ms=now + remaining,
                resident={str(m): float(mb) for m, mb in (gpu_row.get("resident") or {}).items()},
            )
        )

    chains: dict[str, tuple[ObsChainStep, ...]] = {}
    missing: list[str] = []
    for item in ready:
        row = future_artifacts.get(item.node_id) or {}
        chains[item.node_id] = _chain_steps_from_artifact(item.node_id, row, estimate_lookup, missing)

    return MILPObservation(
        now_ms=now,
        ready=tuple(ready),
        chains=chains,
        gpus=tuple(gpus),
        resident_mb_by_model=resident_index,
        decision_time_ms=now,
        resource_missing=tuple(sorted(set(missing))),
        legal_first=legal_first,
    )


# --------------------------------------------------------------------------- #
# planning nodes
# --------------------------------------------------------------------------- #


@dataclass
class PlanNode:
    key: str
    job_index: int
    lane: str
    model_id: str
    compute_ms: float
    load_ms: float
    workspace_mb: float
    anchor_key: str | None
    predecessor_key: str | None
    is_ready: bool
    f0_key: tuple = field(default_factory=tuple)


def _build_nodes(obs: MILPObservation) -> list[PlanNode]:
    nodes: list[PlanNode] = []
    for item in obs.ready:
        # GPU: load + compute (compute floor 0.1); CPU: raw duration, no load.
        compute = (max(0.1, item.runtime_p50_ms - item.load_p50_ms)
                   if item.lane == "gpu" else item.runtime_p50_ms)
        nodes.append(
            PlanNode(
                key=str(item.node_id),
                job_index=item.job_index,
                lane=item.lane,
                model_id=item.model_id,
                compute_ms=compute,
                load_ms=item.load_p50_ms,
                workspace_mb=item.memory_p95_mb,
                anchor_key=None,
                predecessor_key=None,
                is_ready=True,
                f0_key=(item.priority, item.ready_since_ms, item.node_id),
            )
        )
        previous_key = str(item.node_id)
        for step_index, step in enumerate(obs.chains.get(item.node_id, ())):
            compute = (max(0.1, step.runtime_ms - step.load_ms)
                       if step.lane == "gpu" else step.runtime_ms)
            node_key = f"pred:{item.node_id}:{step_index + 1}"
            nodes.append(
                PlanNode(
                    key=node_key,
                    job_index=item.job_index,
                    lane=step.lane,
                    model_id=step.model_id,
                    compute_ms=compute,
                    load_ms=step.load_ms,
                    workspace_mb=step.memory_mb,
                    anchor_key=str(item.node_id),
                    predecessor_key=previous_key,
                    is_ready=False,
                    f0_key=(item.priority, item.ready_since_ms, node_key),
                )
            )
            previous_key = node_key
    return nodes


# --------------------------------------------------------------------------- #
# reference schedule (v0a-legal; used for the Big-M upper bound)
# --------------------------------------------------------------------------- #


def build_reference_schedule(
    obs: MILPObservation,
    nodes: Sequence[PlanNode],
    *,
    return_plan: bool = False,
):
    """Single pass: F0 order, place where the model fits WITHOUT eviction.

    Returns the reference makespan (ms) or None when the state is not
    v0a-applicable (some node cannot be placed without eviction).  With
    ``return_plan=True`` returns (makespan, plan) where plan maps node key to
    {"gpu": int, "start": float, "finish": float, "loaded": bool}.
    """

    gpu_count = len(obs.gpus)
    avail = [obs.gpu_available(g) for g in range(gpu_count)]
    residents: list[dict[str, float]] = [
        dict(gpu.resident) for gpu in obs.gpus
    ]
    caps = [gpu.capacity_mb for gpu in obs.gpus]
    done: dict[str, float] = {}
    plan: dict[str, dict] = {}
    ordered = sorted(nodes, key=lambda n: (n.anchor_key is not None, n.f0_key))
    for node in ordered:
        release = obs.now_ms
        if node.predecessor_key is not None and node.predecessor_key in done:
            release = max(release, done[node.predecessor_key])
        if node.lane != "gpu":
            done[node.key] = release + node.compute_ms
            plan[node.key] = {"gpu": -1, "start": release, "finish": done[node.key], "loaded": False}
            continue
        best = None
        for g in range(gpu_count):
            start = max(release, avail[g])
            u = obs.resident_mb_by_model.get(node.model_id, 0.0)
            ws = max(0.0, node.workspace_mb - u)
            resident_after = residents[g]
            if node.model_id not in resident_after:
                # load event: residents + model + workspace must fit
                if sum(resident_after.values()) + u + ws > caps[g]:
                    continue
                start += node.load_ms
                resident_after = {**resident_after, node.model_id: u}
            else:
                # cache hit: workspace is still required (v3 rule)
                if sum(resident_after.values()) + ws > caps[g]:
                    continue
            finish = start + node.compute_ms
            if best is None or finish < best[0]:
                best = (finish, g, start)
        if best is None:
            return None
        finish, g, start = best
        loaded = node.model_id not in residents[g]
        avail[g] = finish
        residents[g] = {**residents[g], node.model_id: obs.resident_mb_by_model.get(node.model_id, 0.0)}
        done[node.key] = finish
        plan[node.key] = {"gpu": g, "start": start, "finish": finish, "loaded": loaded}
    makespan = max(done.values(), default=obs.now_ms)
    if return_plan:
        return makespan, plan
    return makespan


# --------------------------------------------------------------------------- #
# solver
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MilpV0Result:
    first_action: tuple[str, int] | None
    status: str
    objective: float | None
    best_bound: float | None
    gap: float | None
    solve_ms: float
    fallback: bool
    planned_nodes: int
    boundary_completion_ms: float | None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_action": list(self.first_action) if self.first_action else None,
            "status": self.status,
            "objective": self.objective,
            "best_bound": self.best_bound,
            "gap": self.gap,
            "solve_ms": self.solve_ms,
            "fallback": self.fallback,
            "planned_nodes": self.planned_nodes,
            "boundary_completion_ms": self.boundary_completion_ms,
            "reason": self.reason,
        }


def _fallback_result(started: float, reason: str) -> MilpV0Result:
    return MilpV0Result(
        first_action=None,
        status=f"NOT_APPLICABLE:{reason}",
        objective=None,
        best_bound=None,
        gap=None,
        solve_ms=(time.perf_counter() - started) * 1000.0,
        fallback=True,
        planned_nodes=0,
        boundary_completion_ms=None,
        reason=reason,
    )


def solve_v0a(
    obs: MILPObservation,
    *,
    time_limit_s: float = 0.5,
    debug_families: frozenset | None = None,
    debug_dump: dict | None = None,
) -> MilpV0Result:
    """``debug_families``: testing aid - when set, only the named constraint
    families are emitted (assign/chain/avail/closure/prec/h/z/q/cap)."""

    def fam(name: str) -> bool:
        return debug_families is None or name in debug_families
    """Solve the v0a MILP.  Falls back (first_action=None) when not applicable."""

    started = time.perf_counter()
    if obs.resource_missing:
        return _fallback_result(started, f"unknown_model_profile:{','.join(obs.resource_missing[:3])}")
    if obs.legal_first is not None and not obs.legal_first:
        # Advisor P1: no currently-legal immediate dispatch -> skip the solve.
        return _fallback_result(started, "no_legal_action_at_now")
    nodes = _build_nodes(obs)
    if not nodes:
        return _fallback_result(started, "no_ready_nodes")
    reference_makespan = build_reference_schedule(obs, nodes)
    reference_failed = reference_makespan is None
    if reference_failed:
        # A greedy reference failure is NOT proven infeasibility: use a
        # conservative bound and let the MILP decide (advisor P1).
        reference_makespan = (
            obs.now_ms
            + sum(n.compute_ms + n.load_ms for n in nodes)
            + max(n.compute_ms + n.load_ms for n in nodes)
        )
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except Exception as exc:  # pragma: no cover
        return _fallback_result(started, f"dependency:{type(exc).__name__}")

    gpu_count = len(obs.gpus)
    gpu_nodes = [i for i, n in enumerate(nodes) if n.lane == "gpu"]
    gpu_pos = {i: k for k, i in enumerate(gpu_nodes)}  # advisor P1: index memoization
    max_op = max(n.compute_ms + n.load_ms for n in nodes)
    # Advisor P1: time origin at `now` (times are modeled relative to t0; Big-M
    # scale shrinks from ~1.6e6 to the planning horizon).  Result fields are
    # converted back to absolute time before returning.
    t0 = obs.now_ms
    big_u = 2.0 * ((reference_makespan - t0) + max_op)

    # variable layout
    offset = 0
    x0 = offset; offset += len(gpu_nodes) * gpu_count            # x[i,g]
    a0 = offset; offset += len(nodes)                            # a_i
    c0 = offset; offset += len(nodes)                            # c_i
    pairs = [(i, k) for i in gpu_nodes for k in gpu_nodes if i < k]
    pair_pos = {pair: k for k, pair in enumerate(pairs)}  # advisor P1: index memoization
    y0 = offset; offset += len(pairs) * 2 * gpu_count            # y[fwd], y[bwd] per g
    h0 = offset; offset += len(gpu_nodes) * gpu_count            # h[i,g]
    z0 = offset; offset += len(gpu_nodes) * gpu_count            # z[i,g]
    q0 = offset; offset += len(pairs) * gpu_count * 2            # q[pair,g,dir]
    ready_gpu_nodes = [i for i in gpu_nodes if nodes[i].is_ready]
    # 2026-10-13 (advisor P0): immediate-dispatch bins are per (node, gpu) pair so
    # the legality mask d_ig <= A_ig^0 can be applied directly.
    d_pairs = [(i, g) for i in ready_gpu_nodes for g in range(gpu_count)]
    d0 = offset; offset += len(d_pairs)                          # dispatch-now bins
    n_vars = offset

    def X(i, g):  # gpu node index in gpu_nodes -> flat
        return x0 + gpu_pos[i] * gpu_count + g
    def A(i): return a0 + i
    def C(i): return c0 + i
    def Y(pair_idx, g, reverse):
        return y0 + (pair_idx * gpu_count + g) * 2 + (1 if reverse else 0)
    def H(i, g): return h0 + gpu_pos[i] * gpu_count + g
    def Z(i, g): return z0 + gpu_pos[i] * gpu_count + g
    def Q(pair_idx, g, reverse):
        return q0 + (pair_idx * gpu_count + g) * 2 + (1 if reverse else 0)
    def D(pos): return d0 + pos

    objective = np.zeros(n_vars, dtype=float)
    lower = np.zeros(n_vars, dtype=float)
    upper = np.full(n_vars, big_u, dtype=float)
    integrality = np.zeros(n_vars, dtype=int)

    for i in gpu_nodes:
        for g in range(gpu_count):
            for v in (X(i, g), H(i, g), Z(i, g)):
                upper[v] = 1.0
                integrality[v] = 1
    for pair_idx in range(len(pairs)):
        for g in range(gpu_count):
            for v in (Y(pair_idx, g, False), Y(pair_idx, g, True)):
                upper[v] = 1.0
                integrality[v] = 1
    for i in range(len(nodes)):
        upper[A(i)] = big_u
        upper[C(i)] = big_u
    for pair_idx in range(len(pairs)):
        for g in range(gpu_count):
            for reverse in (False, True):
                upper[Q(pair_idx, g, reverse)] = 1.0
                integrality[Q(pair_idx, g, reverse)] = 1
    mask = obs.legal_first
    for pos, (i, g) in enumerate(d_pairs):
        upper[D(pos)] = 1.0
        integrality[D(pos)] = 1
        if mask is not None and (nodes[i].key, int(g)) not in mask:
            upper[D(pos)] = 0.0  # d_ig <= A_ig^0 (illegal pair: forced 0)

    # objective: sum of boundary completions (w=1)
    boundary_nodes = []
    by_job_anchor: dict[str, list[int]] = {}
    for i, n in enumerate(nodes):
        if n.anchor_key is None:
            by_job_anchor.setdefault(n.key, [i])
        else:
            by_job_anchor.setdefault(n.anchor_key, []).append(i)
    for anchor, chain in by_job_anchor.items():
        boundary_nodes.append(chain[-1])
    for i in boundary_nodes:
        objective[C(i)] = 1.0

    rows: list[dict[int, float]] = []
    lows: list[float] = []
    highs: list[float] = []

    def add(row: dict[int, float], lo: float, hi: float) -> None:
        rows.append(row)
        lows.append(lo)
        highs.append(hi)

    resident_flag = {
        (g, m): (1.0 if m in obs.gpus[g].resident else 0.0)
        for g in range(gpu_count)
        for m in set(obs.resident_mb_by_model)
    }

    # C1: assignment + chain causality + first-action timing
    for i in gpu_nodes:
        if fam("assign"):
            add({X(i, g): 1.0 for g in range(gpu_count)}, 1.0, 1.0)
    for i, n in enumerate(nodes):
        if n.predecessor_key is not None and fam("chain"):
            pred_idx = next(j for j, m in enumerate(nodes) if m.key == n.predecessor_key)
            add({A(i): 1.0, C(pred_idx): -1.0}, 0.0, math.inf)
        if n.lane == "gpu":
            if fam("avail"):
                for g in range(gpu_count):
                    add({A(i): 1.0, X(i, g): -(obs.gpu_available(g) - t0)}, 0.0, math.inf)
        else:
            add({A(i): 1.0}, 0.0, math.inf)  # CPU release at now (relative origin)
    # time closure: c_i = a_i + load*z_i + compute
    for i, n in enumerate(nodes):
        if not fam("closure"):
            break
        if n.lane == "gpu":
            row = {C(i): 1.0, A(i): -1.0}
            for g in range(gpu_count):
                row[Z(i, g)] = -n.load_ms
            add(row, n.compute_ms, n.compute_ms)
        else:
            add({C(i): 1.0, A(i): -1.0}, n.compute_ms, n.compute_ms)

    # P0 immediate dispatch (advisor): at least one currently-legal (node, gpu)
    # pair starts at now.  d_ig <= A_ig^0 (mask) and d_ig <= x_ig (assignment);
    # an empty mask makes the model infeasible -> explicit fallback upstream.
    if fam("immediate") and d_pairs:
        for pos, (i, g) in enumerate(d_pairs):
            if upper[D(pos)] <= 0.0:
                continue  # illegal pair: d fixed to 0, timing row vacuous
            # D=1 -> a_i <= now (immediate); D=0 -> a_i <= now + U (loose)
            add({A(i): 1.0, D(pos): big_u}, -math.inf, big_u)
            add({D(pos): 1.0, X(i, g): -1.0}, -math.inf, 0.0)
        add({D(pos): 1.0 for pos in range(len(d_pairs))}, 1.0, math.inf)

    # C2: same-GPU precedence (conditional) + load-inclusive occupancy
    for pair_idx, (i, k) in enumerate(pairs):
        if not (fam("prec") or fam("h")):
            break
        same_model = nodes[i].model_id == nodes[k].model_id
        for g in range(gpu_count):
            yf = Y(pair_idx, g, False)
            yb = Y(pair_idx, g, True)
            if not fam("prec"):
                break
            add({yf: 1.0, X(i, g): -1.0}, -math.inf, 0.0)
            add({yf: 1.0, X(k, g): -1.0}, -math.inf, 0.0)
            add({yb: 1.0, X(i, g): -1.0}, -math.inf, 0.0)
            add({yb: 1.0, X(k, g): -1.0}, -math.inf, 0.0)
            add({yf: 1.0, yb: 1.0}, -math.inf, 1.0)
            add({yf: 1.0, yb: 1.0, X(i, g): -1.0, X(k, g): -1.0}, -1.0, math.inf)
            # occupancy: a_k >= c_i - U(1 - yf); a_i >= c_k - U(1 - yb)
            add({A(k): 1.0, C(i): -1.0, yf: -big_u}, -big_u, math.inf)
            add({A(i): 1.0, C(k): -1.0, yb: -big_u}, -big_u, math.inf)
            # h_{ig}: same-model predecessor used this gpu (v0a, no eviction)
            # h_i >= (k before i) = yb ; h_k >= (i before k) = yf
            if same_model and fam("h"):
                add({H(i, g): 1.0, yb: -1.0}, 0.0, math.inf)
                add({H(k, g): 1.0, yf: -1.0}, 0.0, math.inf)


    # h upper (standalone: every gpu node, not only pair-firsts):
    # h_{ig} <= sum over same-model k of y_{k before i}
    if fam("h"):
        for i in gpu_nodes:
            for g in range(gpu_count):
                row_h = {H(i, g): 1.0}
                for other in gpu_nodes:
                    if other == i or nodes[other].model_id != nodes[i].model_id:
                        continue
                    pidx = pair_pos[(min(i, other), max(i, other))]
                    row_h[Y(pidx, g, other > i)] = -1.0
                add(row_h, -math.inf, 0.0)

    # C3: GPU-conditional cold load z_ig = AND(x_ig, 1-rho, 1-h_ig)
    for i in gpu_nodes:
        if not fam("z"):
            break
        for g in range(gpu_count):
            rho = resident_flag.get((g, nodes[i].model_id), 0.0)
            add({Z(i, g): 1.0, X(i, g): -1.0}, -math.inf, 0.0)
            add({Z(i, g): 1.0}, -math.inf, 1.0 - rho)
            add({Z(i, g): 1.0, H(i, g): 1.0}, -math.inf, 1.0)
            add({Z(i, g): 1.0, X(i, g): -1.0, H(i, g): 1.0}, -rho, math.inf)

    # C4: event-level memory, conditional on assignment (v3)
    cap_max = max(gpu.capacity_mb for gpu in obs.gpus)
    for i, n in enumerate(nodes):
        if not fam("cap"):
            break
        if n.lane != "gpu":
            continue
        for g in range(gpu_count):
            const = sum(obs.gpus[g].resident.values())
            u_i = obs.resident_mb_by_model.get(n.model_id, 0.0)
            ws_i = max(0.0, n.workspace_mb - u_i)
            lhs: dict[int, float] = {Z(i, g): u_i}
            for other in gpu_nodes:
                if other == i:
                    continue
                pidx = pair_pos[(min(i, other), max(i, other))]
                reverse = other > i  # q[...] direction: "other before i"
                lhs[Q(pidx, g, reverse)] = obs.resident_mb_by_model.get(nodes[other].model_id, 0.0)
            # conditional: lhs <= limit + M*(1 - x_ig); M must also cover a
            # negative limit (initial residents may already exceed cap - ws).
            # 2026-10-13 (P0/P1 memory-alignment fix): the limit uses THIS GPU's
            # capacity, not the max across GPUs (heterogeneous capacities made
            # the solver admit placements the DES then had to evict).
            m_cap = cap_max + sum(obs.resident_mb_by_model.values()) + max(0.0, n.workspace_mb) + 1.0
            lhs[X(i, g)] = m_cap
            add(lhs, -math.inf, (obs.gpus[g].capacity_mb - const - ws_i) + m_cap)
    # q = AND(z_other, y_other->i), two directions per pair/gpu
    for pair_idx, (i, k) in enumerate(pairs):
        if not fam("q"):
            break
        for g in range(gpu_count):
            qf = Q(pair_idx, g, False)  # AND(z_i, i before k)
            qb = Q(pair_idx, g, True)   # AND(z_k, k before i)
            yf = Y(pair_idx, g, False)
            yb = Y(pair_idx, g, True)
            add({qf: 1.0, Z(i, g): -1.0}, -math.inf, 0.0)
            add({qf: 1.0, yf: -1.0}, -math.inf, 0.0)
            add({qf: 1.0, Z(i, g): -1.0, yf: -1.0}, -1.0, math.inf)
            add({qb: 1.0, Z(k, g): -1.0}, -math.inf, 0.0)
            add({qb: 1.0, yb: -1.0}, -math.inf, 0.0)
            add({qb: 1.0, Z(k, g): -1.0, yb: -1.0}, -1.0, math.inf)

    if debug_dump is not None:
        debug_dump.update(
            {
                "rows": [dict(r) for r in rows],
                "lows": list(lows),
                "highs": list(highs),
                "n_vars": n_vars,
                "nodes": [n.key for n in nodes],
                "gpu_nodes": list(gpu_nodes),
                "pairs": list(pairs),
                "gpu_count": gpu_count,
                "big_u": big_u,
                "time_origin": t0,
                "index": {
                    "x0": x0, "a0": a0, "c0": c0, "y0": y0, "h0": h0, "z0": z0, "q0": q0,
                    "d0": d0,
                },
            }
        )
    matrix = lil_matrix((len(rows), n_vars), dtype=float)
    for row_index, row in enumerate(rows):
        for column, coefficient in row.items():
            matrix[row_index, column] = coefficient
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix.tocsr(), np.asarray(lows), np.asarray(highs)),
        options={"time_limit": float(time_limit_s), "disp": False},
    )
    solve_ms = (time.perf_counter() - started) * 1000.0
    if debug_dump is not None:
        _x = getattr(result, "x", None)
        debug_dump["values"] = [float(v) for v in _x] if _x is not None else []
    status_names = {0: "OPTIMAL", 1: "TIME_LIMIT", 2: "INFEASIBLE", 3: "UNBOUNDED", 4: "OTHER"}
    status = status_names.get(int(getattr(result, "status", 4)), "OTHER")
    values = getattr(result, "x", None)
    if values is None or status in ("INFEASIBLE", "UNBOUNDED", "OTHER"):
        reason = "infeasible" if status == "INFEASIBLE" else "no_solution"
        if reference_failed and status == "INFEASIBLE":
            reason = "infeasible_after_reference_failure"
        return MilpV0Result(None, status, None, None, None, solve_ms, True, len(nodes), None, reason)
    # first action: the best currently-legal (node, gpu) pair dispatched at NOW.
    # d_ig <= A_ig^0 (mask) and d_ig <= x_ig guarantee legality by construction;
    # the mask check below is a safety net only.
    best = None
    for pos, (i, g) in enumerate(d_pairs):
        if float(values[D(pos)]) < 0.5:
            continue
        n = nodes[i]
        if obs.legal_first is not None and (n.key, int(g)) not in obs.legal_first:
            continue
        if float(values[X(i, g)]) < 0.5:
            continue
        start = float(values[A(i)])
        if best is None or (start, n.f0_key) < (best[0], best[1]):
            best = (start, n.f0_key, n.key, g)
    if best is None:
        return MilpV0Result(None, f"{status}:NO_READY_ACTION", None, None, None, solve_ms, True, len(nodes), None, "no_ready_action")
    # convert the relative-origin plan values back to absolute time
    shift = float(len(boundary_nodes)) * t0
    boundary = (max(float(values[C(i)]) for i in boundary_nodes) + t0) if boundary_nodes else None
    objective_value = (float(result.fun) + shift) if result.fun is not None else None
    dual_bound = getattr(result, "mip_dual_bound", None)
    dual_bound = (float(dual_bound) + shift) if dual_bound is not None else None
    gap = None
    if objective_value is not None and dual_bound is not None and abs(objective_value) > 1e-9:
        gap = abs(objective_value - dual_bound) / abs(objective_value)
    return MilpV0Result(
        first_action=(best[2], int(best[3])),
        status=status,
        objective=objective_value,
        best_bound=dual_bound,
        gap=gap,
        solve_ms=solve_ms,
        fallback=False,
        planned_nodes=len(nodes),
        boundary_completion_ms=boundary,
        reason=("reference_failed_milp_ok" if reference_failed else ""),
    )
