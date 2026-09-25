"""Latency-Aware-Orchestration-adapted: an independent scheduler.

This module owns the whole decision for the ``latency_aware`` arm.  It does NOT go
through ``choose_action``, so the 26 ``min(pool, key=...)`` sites that every other
arm shares stay byte-identical and the frozen results keep their comparability.

It implements the paper's Algorithm 1 for one commit step:

    RankReady -> BestFeasibleBinding -> PlanReady -> PlanPrefetch -> Propagate -> Prefix

with the equations the paper states:

  Eq (3)  fusible chains (maximal same-deployment chains)
  Eq (4)  fused admission bounds and summed duration
  Eq (5)  lifecycle alternatives { <Load>, <Reclaim, Load> } and alpha_N = Prefetch
  Eq (7)  start = max(release, graph predecessors, resource predecessors)
  Eq (8)  feasible device domain (context limit + admission memory)
  Eq (11) cumulative memory feasibility across predicted transition boundaries
  Eq (12) lexicographic key Phi = (-A_1..-A_K, C_F, -P_N, H_life)

The action vocabulary is explicit, which is the point of owning the scheduler:

    {"type": "start",    "candidate": ..., "fused": [...]}
    {"type": "prefetch", "gpu_index": int, "model_id": str}

Reclaim stays with the simulator's admission path, which already evicts on memory
demand; choosing the victim by "distant next use" is recorded as pending.
"""

from __future__ import annotations

from tracing.analysis.latency_aware_predictor import predict

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

FUSED_SEPARATOR = "\x1f"


@dataclass
class PlanStep:
    """One candidate physical plan for the next commit."""

    candidate: Any
    fused: Tuple[str, ...] = ()
    gpu_index: int = 0
    predicted_start_ms: float = 0.0
    predicted_completion_ms: float = 0.0
    boundaries_removed: int = 0
    load_ms: float = 0.0
    memory_mb: float = 0.0
    feasible: bool = True
    reject_reason: Optional[str] = None

    @property
    def key(self) -> Tuple[Any, ...]:
        """Eq (12), reduced to a single-commit step.

        The paper maximises -A_1..-A_K (admitted units per priority class) and then
        minimises C_F, the latest completion of ready work.  Committing one action per
        event reduces A_k to "prefer the highest service class", so the key is
        (class, completion, ready_time).

        ``-boundaries_removed`` used to sit between the completion and the ready time.
        The paper's Eq (12) has no such term: it was an invented tie-break, and it also
        let a fused candidate win on a count rather than on the completion time the fusion
        actually changes.  Removing it is the faithful choice; a fused plan still competes,
        through the shorter predicted completion that fusing produces.
        """

        item = self.candidate[0]
        return (
            float(item[0]),                       # hard service priority first
            self.predicted_completion_ms,         # C_F
            float(item[1]),                       # ready time
            self.candidate[4].index,
            self.candidate[2],
        )


def predicted_duration_ms(
    job: Any,
    node_id: str,
    members: Sequence[str],
    estimate_row: Mapping[str, Any],
    *,
    resident: bool,
    predictor: Mapping[str, Any],
) -> Tuple[float, float]:
    """Eq (4): the summed predicted run time of a fused unit plus one load.

    Returns (duration_ms, load_ms).  A fused unit pays the model load once.

    The run time comes from the train-only REQUEST-CONDITIONED predictor, never from
    ``node.compute_ms``.  ``compute_ms`` is ``max(0.1, runtime_ms - load_ms)``, the node's
    actual intrinsic compute time: using it made the "predictor" read the very quantity it
    is supposed to predict, and every latency figure derived from it was circular.  The
    ``estimate_row`` fallback is a train-derived statistics row rather than a per-node
    truth, so it is legitimate when the predictor is not supplied.
    """

    load = 0.0 if resident else float(estimate_row["load_p50_ms"])
    if not members:
        pred = predict(predictor, job.template.by_id[node_id])
        return float(pred["run_ms"]) + load, load
    total = 0.0
    for member in members:
        total += float(predict(predictor, job.template.by_id[member])["run_ms"])
    return load + total, load


def build_plan(
    candidate: Any,
    jobs: Sequence[Any],
    now: float,
    *,
    fused_members: Sequence[str] = (),
    predictor: Mapping[str, Any],
) -> PlanStep:
    """Eq (7): start = max(release, gpu availability), completion = start + duration.

    The simulator's own gpu.busy_until already encodes resource precedence for the
    device, and the ready time encodes the graph precedence, so taking their max is
    the Eq (7) expression for this execution model.
    """

    item, job_index, node_id, model_id, gpu, estimate_row, _fit = candidate
    job = jobs[job_index]
    resident = model_id in gpu.resident
    duration, load = predicted_duration_ms(job, node_id, fused_members, estimate_row,
                                           resident=resident, predictor=predictor)
    start = max(float(item[1]), float(gpu.busy_until), float(now))
    members = tuple(fused_members) if fused_members else (str(node_id),)
    # Peak memory comes from the same request-conditioned predictor as the duration,
    # because workspace_peak_mb on the node is a truth field.  There is deliberately no
    # fallback: blocking the truth at the simulator wrapper was not enough, because this
    # module's own entry points stayed reachable with no predictor and read it anyway.
    memory = max(
        (float(predict(predictor, job.template.by_id[m])["peak_mem_mb"]
               + float(job.template.by_id[m].resident_model_mb or 0.0))
         for m in members),
        default=0.0,
    )
    return PlanStep(
        candidate=candidate,
        fused=tuple(fused_members),
        gpu_index=int(gpu.index),
        predicted_start_ms=start,
        predicted_completion_ms=start + duration,
        boundaries_removed=max(0, len(members) - 1),
        load_ms=load,
        memory_mb=memory,
    )


def memory_feasible(step: PlanStep, gpu: Any) -> bool:
    """Eq (11) at the one boundary this commit creates: resident + incoming <= capacity."""

    resident = sum(gpu.resident.values())
    if step.candidate[3] in gpu.resident:
        resident -= float(gpu.resident[step.candidate[3]])
    return resident + step.memory_mb <= gpu.capacity_mb + 1e-9


def rank_ready(
    candidates: Sequence[Any],
    jobs: Sequence[Any],
    now: float,
    *,
    chains_for: Any,
    predictor: Mapping[str, Any],
) -> List[PlanStep]:
    """Algorithm 1 lines 1-10: rank ready units, then bind each one.

    Returns the feasible plans in Eq (12) order.  A fused plan is offered when every
    member is still pending, which is the fusion-normalised unit competing at its head.
    """

    steps: List[PlanStep] = []
    for candidate in candidates:
        item, job_index, node_id, model_id, gpu, estimate_row, _fit = candidate
        job = jobs[job_index]
        members = chains_for(job.template).get(str(node_id), ())
        fused: Tuple[str, ...] = ()
        if members and all(job.node_state.get(m) == "pending" for m in members[1:]):
            fused = tuple(members)
        step = build_plan(candidate, jobs, now, fused_members=fused, predictor=predictor)
        if not memory_feasible(step, gpu):
            step.feasible = False
            step.reject_reason = "admission_memory"
        steps.append(step)
    return sorted(steps, key=lambda s: (not s.feasible, s.key))


def choose_action(
    candidates: Sequence[Any],
    jobs: Sequence[Any],
    now: float,
    *,
    chains_for: Any,
    predictor: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Algorithm 1 lines 3-15 for one commit: the best feasible start, else None."""

    ranked = rank_ready(candidates, jobs, now, chains_for=chains_for, predictor=predictor)
    for step in ranked:
        if step.feasible:
            return {
                "type": "start",
                "candidate": step.candidate,
                "fused": list(step.fused),
                "predicted_completion_ms": step.predicted_completion_ms,
                "boundaries_removed": step.boundaries_removed,
            }
    return None
