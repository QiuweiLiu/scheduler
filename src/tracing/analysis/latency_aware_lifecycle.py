"""Latency-Aware-adapted step 3: the lifecycle half of Constructor (Eq 5).

Paper Eq (5):
    H^X_t(u) = { <alpha_X(d(u))>, <Reclaim, alpha_X(d(u))> }
with alpha_F = Load for ready work and alpha_N = Prefetch for near-ready work.

The simulator already has the whole prefetch machinery -- pending queue, paid load
cost, residency bookkeeping, wasted-prefetch accounting.  What it lacks is a POLICY
that decides what to prepare; the existing path reads a static
``extension_config["prefetch_plan"]``.

This module supplies the policy side:

  * near-ready deployment: the deployment of a successor of a RUNNING node.  The
    paper's window is "ready units and immediate near-ready successors", which is
    exactly this set.
  * a deployment becomes eligible when it is not resident, it fits the device
    together with what is already resident, and preparing it does not disturb ready
    work (we only prepare from a device that is free at the current instant).

The reclaim alternative <Reclaim, Load> is left to the simulator's existing
admission path, which already evicts when memory demands it; choosing the victim by
"distant next use" is the remaining increment and is recorded as pending.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


# States in which a node's control result has RESOLVED, i.e. the node is part of the
# paper's L_t (the resolved logical workflow portion) rather than of the realized future.
RESOLVED_STATES = ("ready", "running", "completed", "failed")


def resolved_graph_view(job: Any) -> set:
    """The RESOLVED logical window the Constructor may read.

    The paper's ``L_t`` is the workflow portion whose control result has RESOLVED, plus
    the immediate near-ready successors of RUNNING units.  It is explicitly NOT the
    complete realized graph: a READY unit has not executed, so its branch outcome (and
    therefore its successor) is not resolved and must not be read.  Reading the realized
    template instead is exactly the structure leak the audit found -- two workflows with
    the same visible prefix but different real futures would look different before the
    branch resolves.

    A RUNNING unit with a UNIQUE successor has an unambiguous continuation, so that
    successor is in the near-ready window.  A RUNNING unit with several successors has an
    unresolved branch, and none of them is exposed.
    """

    state = job.node_state
    visible = {str(node_id) for node_id, value in state.items() if value in RESOLVED_STATES}
    for node_id in list(visible):
        if state.get(node_id) != "running":
            continue
        node = job.template.by_id.get(node_id)
        if node is None or len(node.successors) != 1:
            continue
        visible.add(str(node.successors[0]))
    return visible


def near_ready_deployments(jobs: Sequence[Any]) -> List[Tuple[str, str]]:
    """Deployments of the successors of running nodes: the near-ready window.

    Only successors that are inside the RESOLVED view are eligible, so a running unit with
    an unresolved branch exposes nothing.  Returns (model_id, lane) pairs in first-seen
    order, deduplicated.
    """

    seen: Dict[Tuple[str, str], None] = {}
    for job in jobs:
        view = resolved_graph_view(job)
        for node_id in view:
            if job.node_state.get(node_id) != "running":
                continue
            running = job.template.by_id[node_id]
            for succ_id in running.successors:
                if str(succ_id) not in view:
                    continue  # the successor is not resolved yet
                succ = job.template.by_id.get(succ_id)
                if succ is None:
                    continue
                seen.setdefault((str(succ.model_id), str(succ.lane)), None)
    return list(seen.keys())


def prefetch_candidates(
    jobs: Sequence[Any],
    gpus: Sequence[Any],
    *,
    model_memory_for: Any,
    capacity_headroom: float = 0.0,
) -> List[Dict[str, Any]]:
    """Build a prefetch plan for deployments that are near-ready but not resident.

    ``model_memory_for(model_id)`` returns the measured resident memory for that
    deployment, or None when it has no measured GPU memory (then it is skipped, not
    guessed).  A candidate is emitted only for a device that can hold it alongside
    what is already resident.
    """

    plan: List[Dict[str, Any]] = []
    wanted = near_ready_deployments(jobs)
    for model_id, lane in wanted:
        if lane != "gpu":
            continue  # only GPU deployments have a measurable load and residency
        memory = model_memory_for(model_id)
        if memory is None:
            continue
        for gpu in gpus:
            if model_id in gpu.resident:
                break  # already available somewhere
            projected = sum(gpu.resident.values()) + float(memory)
            if projected <= gpu.capacity_mb + 1e-9 - capacity_headroom:
                plan.append({"gpu_index": int(gpu.index), "model_id": model_id})
                break
    return plan
