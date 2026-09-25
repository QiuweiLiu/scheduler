"""Agentix-adapted (NSDI 2026; early arXiv name Autellix): non-clairvoyant program-level
attained-service scheduling.

Paper: *Agentix: An Efficient Serving Engine for LLM Agents as General Programs*
(USENIX NSDI 2026; arXiv:2502.13965).  Agentix schedules the LLM calls of an agent
PROGRAM by how much service the program has already ATTAINED, with NO prior knowledge of
the program's execution graph:

  * PLAS  (Program-Level Attained Service, single-threaded programs): the priority of a
    call is the sum of the runtimes of its program's PREVIOUSLY COMPLETED calls.
  * ATLAS (Adaptive Thread-Level Attained Service, multi-threaded / dynamic-DAG
    programs): the priority tracks the program's critical path; the system keeps ONE
    scalar per program -- the longest observed critical path -- that each active call
    inherits and that is updated as calls complete.

Lower attained service is scheduled first, so a long program's calls stop blocking the
short programs behind them (the paper's head-of-line-blocking fix).

Adaptation recorded, NOT a claim of the paper's system:
  * the paper schedules LLM calls inside a serving engine and routes across engines for
    KV-cache locality; our execution unit is a GPU node and placement/admission stay with
    the unified substrate.  We port the PRIORITY RULE only.
  * the paper's engine-level batching / data-locality policy is not reproduced.

Non-clairvoyance is the point of this baseline and is enforced in code: only COMPLETED
calls are read, never a predicted value and never the realized future suffix of the
template.  Reading an unexecuted node's duration or identity would make this a clairvoyant
SRPT-style arm rather than Agentix; ``test_agentix_fidelity`` pins that shut.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

AGENTIX_SCHEMA = "agentix-attained-service-v1"

AGENTIX_DEVIATION = (
    "Adapted to the unified GPU simulator: Agentix's program-level attained-service priority "
    "(PLAS / ATLAS) replaces the shared min(pool, key=...) heuristic, while placement, "
    "admission memory and model residency remain with the substrate. Only the priority rule "
    "is ported; the paper's engine-level batching, cross-engine routing and KV-locality load "
    "balancing are NOT reproduced. The arm is non-clairvoyant: it reads only COMPLETED calls."
)


def completed_service_ms(job: Any, duration_of: Callable[[str], float]) -> float:
    """PLAS: the program's attained service = sum of its COMPLETED calls' runtimes.

    ``duration_of`` returns a completed node's actual service time.  Only ``job.completed``
    is read, so a running or untouched call contributes nothing -- that is exactly the
    non-clairvoyant assumption.
    """

    return float(sum(float(duration_of(node_id)) for node_id in job.completed))


def critical_path_service_ms(job: Any, template: Any, duration_of: Callable[[str], float]) -> float:
    """ATLAS: the program's longest COMPLETED critical path (one scalar per program).

    The paper keeps a single scalar per program: the longest observed critical path, which
    each active call inherits and which grows as calls complete.  Recomputing it from the
    completed set is equivalent and avoids hidden state.  A predecessor that has not
    completed contributes nothing, so a partially observed path is never extrapolated into
    the future.
    """

    order = sorted(job.completed, key=lambda node_id: int(template.by_id[node_id].sequence_index))
    path = {}
    for node_id in order:
        node = template.by_id[node_id]
        best = 0.0
        for predecessor in node.predecessors:
            if predecessor in job.completed:
                best = max(best, path.get(predecessor, 0.0))
        path[node_id] = best + float(duration_of(node_id))
    return float(max(path.values(), default=0.0))


def program_priority_ms(job: Any, template: Any, duration_of: Callable[[str], float],
                        mode: str = "plas") -> float:
    """The program-level priority the arm schedules by; lower is scheduled first.

    ``mode='plas'`` is the single-threaded algorithm; ``mode='atlas'`` is the dynamic-DAG
    generalization.  On a serial program the two coincide, because the only critical path
    IS the completed prefix.
    """

    if mode == "atlas":
        return critical_path_service_ms(job, template, duration_of)
    if mode == "plas":
        return completed_service_ms(job, duration_of)
    raise ValueError("unknown Agentix mode %r; use 'plas' or 'atlas'" % (mode,))


def intrinsic_runtime_of(template: Any) -> Callable[[str], float]:
    """Adapter from a completed node id to its ACTUAL service time.

    The value is read only for completed nodes by the callers above; an unexecuted node's
    runtime is never consulted.  A merged CPU+GPU composite uses its own ``runtime_ms``,
    which the v3.1 contract defines as ``R_pre + R_nested + R_post``.
    """

    def duration_of(node_id: str) -> float:
        node = template.by_id[node_id]
        return float(node.runtime_ms)

    return duration_of
