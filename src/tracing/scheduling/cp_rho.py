"""Bounded CP-SAT rolling-horizon dispatch for the R8 scheduler.

The model is intentionally limited to the currently ready GPU nodes.  It
assigns each ready node to one feasible free GPU, sequences the assignments
with optional intervals and ``NoOverlap``, and executes only the first
scheduled action.  Future information is an identity-only predictor cost
coefficient; execution truth never enters this module.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


OBJECTIVE_SCALE = 1000
# A candidate that needs the whole GPU adds ``memory_weight * MEMORY_REFERENCE_MS``
# milliseconds to its objective, i.e. the memory term is expressed in the same
# millisecond unit as the flow-time and future-cost terms.
MEMORY_REFERENCE_MS = 60000.0


@dataclass(frozen=True)
class CpRhoCandidate:
    """Scheduler-visible candidate for one node/GPU assignment."""

    index: int
    node_key: tuple[int, str]
    gpu_index: int
    priority: float
    ready_since_ms: float
    model_id: str
    runtime_ms: float
    load_ms: float
    memory_ratio: float
    future_cost_ms: float

    @property
    def duration_ms(self) -> int:
        return max(1, int(math.ceil(max(0.1, self.runtime_ms + self.load_ms))))


@dataclass(frozen=True)
class CpRhoResult:
    """First action and auditable solver metadata."""

    candidate_index: int
    status: str
    objective: float | None
    best_bound: float | None
    gap: float | None
    solve_ms: float
    fallback: bool
    planned_nodes: int
    selected_indices: tuple[int, ...] = ()
    planned_starts_ms: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "status": self.status,
            "objective": self.objective,
            "best_bound": self.best_bound,
            "gap": self.gap,
            "solve_ms": self.solve_ms,
            "fallback": self.fallback,
            "planned_nodes": self.planned_nodes,
            "selected_indices": list(self.selected_indices),
            "planned_starts_ms": list(self.planned_starts_ms),
        }


def _fallback_index(
    candidates: Sequence[CpRhoCandidate],
    *,
    priority_weight: float,
    future_weight: float,
    memory_weight: float,
) -> int:
    """Deterministic deployable fallback used only when CP-SAT has no plan.

    Uses the same weights and the same millisecond-equivalent units as the
    CP-SAT objective (the objective sequences end times, the fallback uses the
    candidate's own duration as the flow-time proxy).
    """

    def key(position: int) -> tuple[float, float, float, tuple[int, str], int]:
        candidate = candidates[position]
        flow_weight = 1.0 + priority_weight * max(0.0, 1.0 - candidate.priority)
        return (
            flow_weight * (candidate.runtime_ms + candidate.load_ms)
            + future_weight * candidate.future_cost_ms
            + memory_weight * candidate.memory_ratio * MEMORY_REFERENCE_MS,
            candidate.future_cost_ms,
            candidate.ready_since_ms,
            candidate.node_key,
            candidate.gpu_index,
        )

    return min(range(len(candidates)), key=key)


def solve_first_action(
    candidates: Sequence[CpRhoCandidate],
    *,
    horizon: int,
    time_limit_s: float = 0.25,
    random_seed: int = 0,
    num_workers: int = 1,
    priority_weight: float = 0.25,
    future_weight: float = 1.0,
    memory_weight: float = 0.05,
    future_on_executed_action_only: bool = True,
) -> CpRhoResult:
    """Solve a bounded assignment/sequence model and return its first action.

    ``horizon`` is the predictor horizon used for the future coefficient.  The
    CP-SAT schedule itself is the current ready-node window; this V1 boundary
    avoids exposing hidden suffix nodes while still making GPU assignment and
    sequencing a joint optimization problem.

    ``future_on_executed_action_only`` matters for correctness of the future
    term: the model selects exactly one GPU per ready node and the predicted
    future cost does not depend on the GPU, so charging it for every selected
    candidate makes it a constant that cannot influence the plan.  When the flag
    is set (default) the future/memory terms are charged only to the action that
    will actually be executed (the earliest scheduled start), which is the only
    decision the rolling-horizon controller commits to.
    """

    if not candidates:
        raise ValueError("CP-RHO requires at least one candidate")
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if time_limit_s <= 0.0:
        raise ValueError("time_limit_s must be positive")
    fallback = _fallback_index(
        candidates,
        priority_weight=priority_weight,
        future_weight=future_weight,
        memory_weight=memory_weight,
    )
    started = time.perf_counter()

    try:
        from ortools.sat.python import cp_model
    except Exception as exc:  # pragma: no cover - exercised in missing-dependency gate
        return CpRhoResult(
            candidate_index=fallback,
            status=f"DEPENDENCY_ERROR:{type(exc).__name__}",
            objective=None,
            best_bound=None,
            gap=None,
            solve_ms=(time.perf_counter() - started) * 1000.0,
            fallback=True,
            planned_nodes=0,
        )

    model = cp_model.CpModel()
    groups: dict[tuple[int, str], list[int]] = {}
    for position, candidate in enumerate(candidates):
        groups.setdefault(candidate.node_key, []).append(position)

    # A loose but finite upper bound for the current ready-node window.
    unique_nodes = [candidates[positions[0]] for positions in groups.values()]
    horizon_ms = max(
        1,
        sum(candidate.duration_ms for candidate in unique_nodes)
        + max(candidate.duration_ms for candidate in unique_nodes) * max(1, len(unique_nodes)),
    )
    starts = []
    ends = []
    selected = []
    intervals_by_gpu: dict[int, list[Any]] = {}
    for position, candidate in enumerate(candidates):
        start = model.new_int_var(0, horizon_ms, f"start_{position}")
        end = model.new_int_var(0, horizon_ms + candidate.duration_ms, f"end_{position}")
        select = model.new_bool_var(f"select_{position}")
        interval = model.new_optional_interval_var(
            start,
            candidate.duration_ms,
            end,
            select,
            f"interval_{position}",
        )
        starts.append(start)
        ends.append(end)
        selected.append(select)
        intervals_by_gpu.setdefault(candidate.gpu_index, []).append(interval)

    for positions in groups.values():
        model.add(sum(selected[position] for position in positions) == 1)
    for intervals in intervals_by_gpu.values():
        model.add_no_overlap(intervals)

    objective_terms = []
    executed: list[Any] = []
    if future_on_executed_action_only:
        # Exactly one candidate is the action the controller will execute; the
        # future/memory terms are charged to it alone.  Earlier revisions marked
        # every earliest-start candidate (ties across GPUs) and then returned a
        # possibly different candidate, so the charged action and the executed
        # action could diverge.
        for position in range(len(candidates)):
            executed_var = model.new_bool_var(f"executed_{position}")
            model.add(executed_var <= selected[position])
            executed.append(executed_var)
        model.add(sum(executed) == 1)
        for position in range(len(candidates)):
            for other in range(len(candidates)):
                if other == position:
                    continue
                model.add(starts[position] <= starts[other]).only_enforce_if(executed[position])
    for position, candidate in enumerate(candidates):
        # Priority is soft: priority 0 has a slightly larger flow-time weight,
        # but a low-cost future-aware assignment can still win.  All terms are
        # scaled by OBJECTIVE_SCALE and live in the same millisecond unit.
        flow_weight = 1.0 + priority_weight * max(0.0, 1.0 - candidate.priority)
        end_cost = int(round(flow_weight * OBJECTIVE_SCALE)) * ends[position]
        if future_on_executed_action_only:
            future_cost = int(round(future_weight * candidate.future_cost_ms * OBJECTIVE_SCALE)) * executed[position]
            memory_cost = (
                int(round(memory_weight * candidate.memory_ratio * MEMORY_REFERENCE_MS * OBJECTIVE_SCALE))
                * executed[position]
            )
        else:
            future_cost = int(round(future_weight * candidate.future_cost_ms * OBJECTIVE_SCALE)) * selected[position]
            memory_cost = (
                int(round(memory_weight * candidate.memory_ratio * MEMORY_REFERENCE_MS * OBJECTIVE_SCALE))
                * selected[position]
            )
        objective_terms.extend((end_cost, future_cost, memory_cost))
    model.minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.random_seed = int(random_seed)
    solver.parameters.num_search_workers = max(1, int(num_workers))
    status_code = solver.solve(model)
    status = solver.status_name(status_code)
    solve_ms = (time.perf_counter() - started) * 1000.0
    if status not in {"OPTIMAL", "FEASIBLE"}:
        return CpRhoResult(
            candidate_index=fallback,
            status=status,
            objective=None,
            best_bound=None,
            gap=None,
            solve_ms=solve_ms,
            fallback=True,
            planned_nodes=0,
        )

    chosen_positions = [position for position, value in enumerate(selected) if solver.boolean_value(value)]
    if not chosen_positions:
        return CpRhoResult(
            candidate_index=fallback,
            status=f"{status}:NO_SELECTION",
            objective=float(solver.objective_value),
            best_bound=float(solver.best_objective_bound),
            gap=None,
            solve_ms=solve_ms,
            fallback=True,
            planned_nodes=0,
        )
    if future_on_executed_action_only:
        executed_positions = [position for position, value in enumerate(executed) if solver.boolean_value(value)]
        if len(executed_positions) != 1:
            return CpRhoResult(
                candidate_index=fallback,
                status=f"{status}:EXECUTED_NOT_UNIQUE:{len(executed_positions)}",
                objective=float(solver.objective_value),
                best_bound=float(solver.best_objective_bound),
                gap=None,
                solve_ms=solve_ms,
                fallback=True,
                planned_nodes=0,
            )
        first_position = executed_positions[0]
    else:
        ordered_positions = tuple(
            sorted(
                chosen_positions,
                key=lambda position: (
                    solver.value(starts[position]),
                    candidates[position].ready_since_ms,
                    candidates[position].node_key,
                    candidates[position].gpu_index,
                ),
            )
        )
        first_position = ordered_positions[0]
    planned_positions = tuple(
        sorted(
            chosen_positions,
            key=lambda position: (
                solver.value(starts[position]),
                candidates[position].ready_since_ms,
                candidates[position].node_key,
                candidates[position].gpu_index,
            ),
        )
    )
    objective = float(solver.objective_value)
    best_bound = float(solver.best_objective_bound)
    gap = abs(objective - best_bound) / max(1.0, abs(objective))
    return CpRhoResult(
        candidate_index=candidates[first_position].index,
        status=status,
        objective=objective,
        best_bound=best_bound,
        gap=gap,
        solve_ms=solve_ms,
        fallback=False,
        planned_nodes=len(planned_positions),
        selected_indices=tuple(candidates[position].index for position in planned_positions),
        planned_starts_ms=tuple(float(solver.value(starts[position])) for position in planned_positions),
    )


def candidate_from_scheduler_row(
    *,
    index: int,
    node_key: tuple[int, str],
    gpu_index: int,
    priority: float,
    ready_since_ms: float,
    model_id: str,
    estimate_row: Mapping[str, Any],
    cache_hit: bool,
    future_cost_ms: float,
    capacity_mb: float,
) -> CpRhoCandidate:
    memory_mb = float(estimate_row.get("memory_p95_mb") or 0.0)
    return CpRhoCandidate(
        index=index,
        node_key=node_key,
        gpu_index=int(gpu_index),
        priority=float(priority),
        ready_since_ms=float(ready_since_ms),
        model_id=str(model_id),
        runtime_ms=float(estimate_row.get("runtime_p50_ms") or 0.0),
        load_ms=0.0 if cache_hit else float(estimate_row.get("load_p50_ms") or 0.0),
        memory_ratio=memory_mb / max(1.0, float(capacity_mb)),
        future_cost_ms=float(future_cost_ms),
    )
