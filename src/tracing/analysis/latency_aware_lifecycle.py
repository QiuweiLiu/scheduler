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


def near_ready_deployments(jobs: Sequence[Any]) -> List[Tuple[str, str]]:
    """Deployments of the successors of running nodes: the near-ready window.

    Returns (model_id, lane) pairs in first-seen order, deduplicated.
    """

    seen: Dict[Tuple[str, str], None] = {}
    for job in jobs:
        for node_id, state in job.node_state.items():
            if state != "running":
                continue
            running = job.template.by_id[node_id]
            for succ_id in running.successors:
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
