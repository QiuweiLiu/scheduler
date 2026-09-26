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
    "Adapted to the unified GPU simulator. The FORMAL arm is continuous PLAS, NON-PREEMPTIVE: "
    "placement, admission memory and model residency remain with the substrate. Not part of the "
    "formal arm and NOT reproduced: the paper's preemptive MLFQ scheduler (K priority queues, the "
    "decoding time quantum, in-call demotion when a quantum is exhausted) and program-level "
    "anti-starvation, all of which need request preemption / KV-cache swapping; plus the "
    "engine-level batching, cross-engine routing and KV-locality load balancing. The hard "
    "benchmark service class precedes the PLAS score. Service accounting counts completed GPU "
    "nodes' observed model-executor service (queued wait and model load excluded). "
    "Information boundary: NO prediction of future workflow structure or remaining work; the "
    "current ready node's resource estimates are inherited from the common substrate."
)


def completed_service_ms(job: Any, duration_of: Callable[[str], float]) -> float:
    """PLAS: the program's attained service = sum of its COMPLETED calls' runtimes.

    ``duration_of`` returns a completed node's actual service time.  Only ``job.completed``
    is read, so a running or untouched call contributes nothing -- that is exactly the
    non-clairvoyant assumption.
    """

    total = 0.0
    for node_id in job.completed:
        value = duration_of(node_id)
        if value is None:
            continue  # e.g. a non-LLM node that carries no model-executor service
        total += float(value)
    return float(total)


def critical_path_service_ms(job: Any, template: Any, duration_of: Callable[[str], float]) -> float:
    """ATLAS VARIANT: the program's longest COMPLETED critical path.

    This is NOT the paper's ATLAS.  Agentix Algorithm 1 keeps ONE scalar per program -- the
    longest observed critical path -- which a call INHERITS at arrival and which is updated
    on completion as ``max(scalar, inherited + own_model_time)``; it deliberately does not
    track dependencies, and that inheritance can over-count across branches (for
    ``a(1) -> {b(100), c(1) -> e(1)}`` with ``b`` first, the paper's scalar reaches 102 while
    the true completed critical path is 101).  This function instead recomputes the exact
    longest path over the completed subgraph.  On a SERIAL program the two coincide, which is
    why the arm is used with ``mode='plas'`` on serial workloads; this variant is kept for
    parallel-workload experiments only and must NOT be described as faithful ATLAS.
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


def observed_gpu_service_of(job: Any) -> Callable[[str], float]:
    """Accessor for a COMPLETED GPU node's observed model-executor SERVICE.

    Agentix's PLAS sums the cumulative execution time of completed LLM calls on the engine's
    MODEL EXECUTOR.  Three things must therefore be excluded, all of which the raw template
    ``runtime_ms`` would wrongly include:
      * non-LLM (CPU/control/API) nodes -- ``lane != 'gpu'`` returns ``None`` and is skipped;
      * the model LOADING cost -- our substrate defines ``compute_ms = runtime_ms - load_ms``;
      * queue wait -- the observation store records the intrinsic runtime, not
        ``finish - start`` (which would fold in this schedule's own queue delay).
    The value is read from the observation store written on completion when available.
    """

    observed = getattr(job, "observed_intrinsic_ms", None) or {}

    def duration_of(node_id: str):
        node = job.template.by_id.get(node_id)
        if node is None or str(getattr(node, "lane", "gpu")) != "gpu":
            return None
        runtime = float(observed[node_id]) if node_id in observed else float(node.runtime_ms)
        load = float(getattr(node, "load_ms", 0.0) or 0.0)
        return max(0.0, runtime - load)

    return duration_of


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


# --------------------------------------------------------------------------- #
# NON-PAPER SENSITIVITY VARIANT: discretised, non-preemptive queueing
# --------------------------------------------------------------------------- #
# Agentix's own preemptive scheduler (§4.2.2) discretises the continuous priority into K
# queues AND demotes a call when it exhausts its DECODING time quantum mid-call, which needs
# request preemption and KV swap.  Our execution unit is an atomic node, so neither the
# in-call quantum nor preemption is representable.  This variant keeps only what survives a
# node-atomic substrate:
#
#   * discretisation: bin the program's attained service into K queues and rank by bin;
#   * CROSS-CALL RE-BINNING: as a program's attained service grows its later calls land in a
#     lower queue.  This is NOT a substitute for the paper's within-call demotion, which the
#     paper performs when a call exhausts its decoding quantum mid-call;
#   * program-level anti-starvation: promote a call to the top queue when its program's
#     (wait / service) ratio crosses beta.
#
# It must be reported as `Agentix-Discrete-NoPreempt` -- an adaptation that answers "does
# discretising PLAS and adding anti-starvation change anything?", NOT as the paper's
# preemptive MLFQ scheduler.  It is a sensitivity mode; the formal arm stays mode='plas'.

AGENTIX_DISCRETE_SCHEMA = "agentix-discrete-nopreempt-v1"

# Pre-registered adaptation constants.  The paper gives K queues but no edges, and no beta.
DEFAULT_QUEUE_EDGES_MS = (0.0, 30000.0, 90000.0, 210000.0)   # lower edge of each of 4 queues
DEFAULT_ANTI_STARVATION_BETA = 1.0                           # promote when wait/service >= beta

AGENTIX_DISCRETE_DEVIATION = (
    "NON-PAPER sensitivity variant. Agentix discretises the program priority into K queues and "
    "demotes a call when it exhausts its decoding quantum mid-call, requiring request preemption "
    "and KV swap; our node-atomic substrate cannot represent either. This variant keeps the "
    "discretisation and the program-level anti-starvation promotion, and applies demotion ACROSS "
    "calls instead of within a call. In-call quantum demotion and preemption are NOT reproduced."
)


def queue_index(attained_service_ms: float,
                edges: Sequence[float] = DEFAULT_QUEUE_EDGES_MS) -> int:
    """The K-queue index for an attained service value; lower index = higher priority."""

    value = max(0.0, float(attained_service_ms))
    index = 0
    for position, edge in enumerate(edges):
        if value >= float(edge):
            index = position
        else:
            break
    return int(index)


def is_starving(wait_ms: float, service_ms: float,
                beta: float = DEFAULT_ANTI_STARVATION_BETA) -> bool:
    """Program-level anti-starvation: promote when ``wait / service >= beta``.

    ``service`` is clamped to a small positive floor so a program with no completed service
    is judged by its wait rather than dividing by zero.
    """

    if beta <= 0.0:
        raise ValueError("anti-starvation beta must be positive")
    return (max(0.0, float(wait_ms)) / max(1.0, float(service_ms))) >= float(beta)


def discrete_priority_index(attained_service_ms: float, wait_ms: float,
                            edges: Sequence[float] = DEFAULT_QUEUE_EDGES_MS,
                            beta: float = DEFAULT_ANTI_STARVATION_BETA) -> int:
    """The queue index actually used: 0 (top) when the program is starving, else its bin."""

    if is_starving(wait_ms, attained_service_ms, beta):
        return 0
    return queue_index(attained_service_ms, edges)
