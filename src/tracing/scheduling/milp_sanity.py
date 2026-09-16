"""Small MILP equivalent used only for the R8-P2 solver sanity check.

This is deliberately not an online scheduler.  It mirrors the bounded CP-RHO
V1 model over the current ready candidate window and returns the first action.
Only deployable candidate estimates are accepted; execution truth is absent.
SciPy/HiGHS is imported lazily because the canonical remote environment, not
the local control-plane environment, owns the MILP dependency.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Sequence

from tracing.scheduling.cp_rho import CpRhoCandidate


@dataclass(frozen=True)
class MilpResult:
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


def _fallback_index(candidates: Sequence[CpRhoCandidate]) -> int:
    return min(
        range(len(candidates)),
        key=lambda position: (
            candidates[position].future_cost_ms,
            candidates[position].ready_since_ms,
            candidates[position].node_key,
            candidates[position].gpu_index,
        ),
    )


def solve_first_action_milp(
    candidates: Sequence[CpRhoCandidate],
    *,
    time_limit_s: float = 2.0,
    priority_weight: float = 0.25,
    future_weight: float = 1.0,
    memory_weight: float = 0.05,
) -> MilpResult:
    """Solve the bounded assignment/sequence model with SciPy/HiGHS."""

    if not candidates:
        raise ValueError("MILP sanity check requires at least one candidate")
    if time_limit_s <= 0.0:
        raise ValueError("time_limit_s must be positive")
    fallback = _fallback_index(candidates)
    started = time.perf_counter()
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except Exception as exc:  # pragma: no cover - exercised on local env only
        return MilpResult(
            candidate_index=fallback,
            status=f"DEPENDENCY_ERROR:{type(exc).__name__}",
            objective=None,
            best_bound=None,
            gap=None,
            solve_ms=(time.perf_counter() - started) * 1000.0,
            fallback=True,
            planned_nodes=0,
        )

    groups: dict[tuple[int, str], list[int]] = {}
    for position, candidate in enumerate(candidates):
        groups.setdefault(candidate.node_key, []).append(position)
    unique_nodes = [candidates[positions[0]] for positions in groups.values()]
    horizon_ms = max(
        1.0,
        sum(candidate.duration_ms for candidate in unique_nodes)
        + max(candidate.duration_ms for candidate in unique_nodes) * max(1, len(unique_nodes)),
    )

    candidate_count = len(candidates)
    start_offset = candidate_count
    pair_indices: list[tuple[int, int, int, int]] = []
    next_index = 2 * candidate_count
    for left in range(candidate_count):
        for right in range(left + 1, candidate_count):
            if candidates[left].gpu_index != candidates[right].gpu_index:
                continue
            pair_indices.append((left, right, next_index, next_index + 1))
            next_index += 2
    variable_count = next_index
    objective = np.zeros(variable_count, dtype=float)
    lower = np.zeros(variable_count, dtype=float)
    upper = np.zeros(variable_count, dtype=float)
    integrality = np.zeros(variable_count, dtype=int)
    for position, candidate in enumerate(candidates):
        flow_weight = 1.0 + priority_weight * max(0.0, 1.0 - candidate.priority)
        flow_scale = round(flow_weight * 1000.0)
        objective[position] = (
            flow_scale * candidate.duration_ms
            + round(future_weight * candidate.future_cost_ms)
            + round(memory_weight * candidate.memory_ratio * 1000.0)
        )
        objective[start_offset + position] = flow_scale
        upper[position] = 1.0
        upper[start_offset + position] = max(0.0, horizon_ms)
        integrality[position] = 1
    for _, _, first_order, second_order in pair_indices:
        upper[first_order] = 1.0
        upper[second_order] = 1.0
        integrality[first_order] = 1
        integrality[second_order] = 1

    # Keep the sparse constraint matrix explicit: this is a finite audit model,
    # not a general-purpose modeling layer.
    rows: list[dict[int, float]] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    for positions in groups.values():
        rows.append({position: 1.0 for position in positions})
        lower_bounds.append(1.0)
        upper_bounds.append(1.0)
    for position, candidate in enumerate(candidates):
        start = start_offset + position
        rows.append({start: 1.0, position: -horizon_ms})
        lower_bounds.append(-math.inf)
        upper_bounds.append(0.0)
        rows.append({start: 1.0, position: float(candidate.duration_ms)})
        lower_bounds.append(-math.inf)
        upper_bounds.append(horizon_ms)

    big_m = horizon_ms + max(candidate.duration_ms for candidate in candidates)
    for left, right, left_before, right_before in pair_indices:
        rows.extend(
            [
                {left_before: 1.0, right_before: 1.0, left: -1.0},
                {left_before: 1.0, right_before: 1.0, right: -1.0},
                {left_before: 1.0, right_before: 1.0, left: -1.0, right: -1.0},
                {
                    start_offset + left: 1.0,
                    start_offset + right: -1.0,
                    left_before: big_m,
                },
                {
                    start_offset + right: 1.0,
                    start_offset + left: -1.0,
                    right_before: big_m,
                },
            ]
        )
        lower_bounds.extend((-math.inf, -math.inf, -1.0, -math.inf, -math.inf))
        upper_bounds.extend((0.0, 0.0, 0.0, big_m - candidates[left].duration_ms, big_m - candidates[right].duration_ms))

    matrix = lil_matrix((len(rows), variable_count), dtype=float)
    for row_index, row in enumerate(rows):
        for column, coefficient in row.items():
            matrix[row_index, column] = coefficient
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix.tocsr(), np.asarray(lower_bounds), np.asarray(upper_bounds)),
        options={"time_limit": float(time_limit_s), "disp": False},
    )
    solve_ms = (time.perf_counter() - started) * 1000.0
    status_code = int(getattr(result, "status", 4))
    status_names = {0: "OPTIMAL", 1: "TIME_LIMIT", 2: "INFEASIBLE", 3: "UNBOUNDED", 4: "OTHER"}
    status = status_names.get(status_code, f"STATUS_{status_code}")
    values = getattr(result, "x", None)
    if values is None:
        return MilpResult(fallback, status, None, None, None, solve_ms, True, 0)
    chosen = [position for position in range(candidate_count) if values[position] >= 0.5]
    if not chosen:
        return MilpResult(fallback, f"{status}:NO_SELECTION", None, None, None, solve_ms, True, 0)
    ordered_positions = tuple(
        sorted(
            chosen,
            key=lambda position: (
                values[start_offset + position],
                candidates[position].ready_since_ms,
                candidates[position].node_key,
                candidates[position].gpu_index,
            ),
        )
    )
    first_position = ordered_positions[0]
    objective_value = float(result.fun) if result.fun is not None else None
    best_bound_value = getattr(result, "mip_dual_bound", None)
    best_bound = float(best_bound_value) if best_bound_value is not None else objective_value
    gap_value = getattr(result, "mip_gap", None)
    gap = float(gap_value) if gap_value is not None else None
    return MilpResult(
        candidate_index=candidates[first_position].index,
        status=status,
        objective=objective_value,
        best_bound=best_bound,
        gap=gap,
        solve_ms=solve_ms,
        fallback=status != "OPTIMAL" or len(chosen) != len(groups),
        planned_nodes=len(ordered_positions),
        selected_indices=tuple(candidates[position].index for position in ordered_positions),
        planned_starts_ms=tuple(float(values[start_offset + position]) for position in ordered_positions),
    )
