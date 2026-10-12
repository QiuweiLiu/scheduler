"""DES-aligned rollout (planning DES) machinery.

The planning world reuses the *real* discrete-event simulator
(:mod:`tracing.analysis.workload_v02_simulator`) as its world model:

- :func:`serialize_des_state` captures a full scheduler state at a decision
  moment (called from ``simulate_episode`` when ``policy_context["des_capture"]``
  is present).
- :func:`apply_injected_state` installs that state into a fresh planning run
  (called from ``simulate_episode`` when the episode carries ``injected_state``).
- :func:`build_scenario_templates` / :func:`build_planning_episode` /
  :func:`run_planning` construct and execute a planning world in which the
  *execution truth* of the future is replaced by a scenario, while the
  planner-visible information (frozen artifacts / estimates) is unchanged.

Design contract (docs/research/2026-10-12_exp_i_des_rollout.md):
- The workflow's causal order is fixed: scenario steps reuse the real template's
  node ids and chain order; only attributes (model / duration / resource) are
  replaced and the tail beyond the scenario horizon is trimmed.
- The planner never reads the scenario; it reads the same artifacts it would
  read online.  Scenarios are common random numbers across candidate actions.
- Time base is absolute; the planning run starts at the snapshot time.
"""

from __future__ import annotations

import copy
import heapq
from dataclasses import replace
from typing import Any, Mapping, Sequence

from tracing.analysis.workload_v02_simulator import (
    GPU,
    Job,
    Node,
    Template,
    simulate_episode,
)

SCENARIO_ATTR_KEYS = (
    "model_id",
    "lane",
    "role",
    "action_family",
    "runtime_ms",
    "load_ms",
    "workspace_peak_mb",
    "resident_model_mb",
    "status",
)


# --------------------------------------------------------------------------- #
# state serialization / injection
# --------------------------------------------------------------------------- #


def _task_to_dict(task: Any) -> dict[str, Any]:
    owner = task.owner
    return {
        "start_ms": float(task.start_ms),
        "finish_ms": float(task.finish_ms),
        "kind": str(task.kind),
        "owner_job": int(owner[0]) if owner is not None else None,
        "owner_node": str(owner[1]) if owner is not None else None,
        "model_id": str(task.model_id),
        "role": str(task.role),
        "workload_shape": str(task.workload_shape),
        "workspace_mb": float(task.workspace_mb),
        "allocation_mb": float(task.allocation_mb),
        "interference_target": task.interference_target,
        "slowdown": float(task.slowdown),
        "interference_mode": str(task.interference_mode),
        "interference_model": str(task.interference_model),
        "work_start_ms": float(task.work_start_ms) if task.work_start_ms is not None else None,
        "predicted_work_ms": (
            float(task.predicted_work_ms) if task.predicted_work_ms is not None else None
        ),
        "resume_tokens_base": task.resume_tokens_base,
        "remaining_work_ms": (
            float(task.remaining_work_ms) if task.remaining_work_ms is not None else None
        ),
        "total_work_ms": (
            float(task.total_work_ms) if task.total_work_ms is not None else None
        ),
        "intrinsic_remaining_work_ms": (
            float(task.intrinsic_remaining_work_ms)
            if task.intrinsic_remaining_work_ms is not None
            else None
        ),
        "stall_remaining_ms": float(task.stall_remaining_ms),
        "last_update_ms": (
            float(task.last_update_ms) if task.last_update_ms is not None else None
        ),
    }


def serialize_des_state(
    *,
    episode_id: str,
    now: float,
    jobs: Sequence[Job],
    gpus: Sequence[GPU],
    finish_heap: Sequence[tuple[float, int, int, str, int | None, str]],
    ready: Sequence[tuple[float, int, int, str]],
    arrival_heap: Sequence[tuple[float, int]],
    active_jobs: set[int],
    wait_logged: set[tuple[int, str]],
    sequence: int,
    rr_cursor: int,
) -> dict[str, Any]:
    """Serialize the full scheduler state (in-process, JSON-able except resume state)."""

    job_rows: list[dict[str, Any]] = []
    for job in jobs:
        job_rows.append(
            {
                "job_instance_id": str(job.job_instance_id),
                "node_state": dict(job.node_state),
                "ready_since": {k: float(v) for k, v in job.ready_since.items()},
                "started": sorted(job.started),
                "completed": sorted(job.completed),
                "failed": sorted(job.failed),
                "finish_ms": float(job.finish_ms) if job.finish_ms is not None else None,
                "queue_ms": float(job.queue_ms),
                "load_ms": float(job.load_ms),
                "assigned_gpus": [int(v) for v in job.assigned_gpus],
                "preemptions": int(job.preemptions),
                "preempt_recompute_ms": float(job.preempt_recompute_ms),
                "preempt_stall_ms": float(job.preempt_stall_ms),
                "resume_count": int(job.resume_count),
                "preempted_at": {k: float(v) for k, v in job.preempted_at.items()},
                "observed_intrinsic_ms": {
                    k: float(v) for k, v in job.observed_intrinsic_ms.items()
                },
                "request_resume_state": copy.copy(job.request_resume_state),
            }
        )
    gpu_rows: list[dict[str, Any]] = []
    for gpu in gpus:
        gpu_rows.append(
            {
                "index": int(gpu.index),
                "capacity_mb": float(gpu.capacity_mb),
                "resident": {str(m): float(mb) for m, mb in gpu.resident.items()},
                "active_tasks": [
                    {"task_id": str(task_id), **_task_to_dict(task)}
                    for task_id, task in gpu.active_tasks.items()
                    if task.active
                ],
                "prefetch_pending": [list(entry) for entry in gpu.prefetch_pending],
                "prefetched_models": sorted(gpu.prefetched_models),
                "used_prefetched_models": sorted(gpu.used_prefetched_models),
                "evicted_models_seen": sorted(gpu.evicted_models_seen),
                "evictions": int(gpu.evictions),
                "prefetch_count": int(gpu.prefetch_count),
                "prefetch_load_ms": float(gpu.prefetch_load_ms),
                "wasted_prefetches": int(gpu.wasted_prefetches),
                "cold_loads_on_demand": int(gpu.cold_loads_on_demand),
                "evicted_then_reloaded": int(gpu.evicted_then_reloaded),
                "peak_memory_mb": float(gpu.peak_memory_mb),
            }
        )
    return {
        "schema_version": "des-state-v1",
        "episode_id": str(episode_id),
        "now_ms": float(now),
        "sequence": int(sequence),
        "rr_cursor": int(rr_cursor),
        "jobs": job_rows,
        "gpus": gpu_rows,
        "finish_entries": [
            [float(t), int(order), int(job_index), str(lane), gpu_index, str(node_id)]
            for (t, order, job_index, lane, gpu_index, node_id) in finish_heap
        ],
        "ready_items": [
            [float(priority), float(ready_time), int(job_index), str(node_id)]
            for (priority, ready_time, job_index, node_id) in ready
        ],
        "arrivals_pending": [
            [float(arrival), int(job_index)] for (arrival, job_index) in arrival_heap
        ],
        "active_jobs": sorted(int(v) for v in active_jobs),
        "wait_logged": [[int(a), str(b)] for (a, b) in sorted(wait_logged)],
    }


def apply_injected_state(
    injected: Mapping[str, Any],
    jobs: Sequence[Job],
    gpus: Sequence[GPU],
) -> dict[str, Any]:
    """Install a serialized state into a fresh planning run.

    Mutates ``jobs`` / ``gpus`` in place and returns the heap/clock values the
    simulator must install.
    """

    index_by_id = {str(job.job_instance_id): index for index, job in enumerate(jobs)}
    for row in injected.get("jobs") or []:
        job = jobs[index_by_id[str(row["job_instance_id"])]]
        # The planning template may have trimmed tail nodes (H5 scenario): keep
        # only state that refers to nodes that still exist, or the job can never
        # satisfy "all nodes complete".
        live = {node.node_id for node in job.template.nodes}
        job.node_state = {
            str(k): str(v) for k, v in (row.get("node_state") or {}).items() if str(k) in live
        }
        job.ready_since = {
            str(k): float(v) for k, v in (row.get("ready_since") or {}).items() if str(k) in live
        }
        job.started = set(str(v) for v in row.get("started") or [] if str(v) in live)
        job.completed = set(str(v) for v in row.get("completed") or [] if str(v) in live)
        job.failed = set(str(v) for v in row.get("failed") or [] if str(v) in live)
        job.finish_ms = row.get("finish_ms")
        if job.finish_ms is not None:
            job.finish_ms = float(job.finish_ms)
        job.queue_ms = float(row.get("queue_ms") or 0.0)
        job.load_ms = float(row.get("load_ms") or 0.0)
        job.assigned_gpus = [int(v) for v in row.get("assigned_gpus") or []]
        job.preemptions = int(row.get("preemptions") or 0)
        job.preempt_recompute_ms = float(row.get("preempt_recompute_ms") or 0.0)
        job.preempt_stall_ms = float(row.get("preempt_stall_ms") or 0.0)
        job.resume_count = int(row.get("resume_count") or 0)
        job.preempted_at = {
            str(k): float(v) for k, v in (row.get("preempted_at") or {}).items() if str(k) in live
        }
        job.observed_intrinsic_ms = {
            str(k): float(v)
            for k, v in (row.get("observed_intrinsic_ms") or {}).items()
            if str(k) in live
        }
        job.request_resume_state = copy.copy(row.get("request_resume_state") or {})

    for gpu_row in injected.get("gpus") or []:
        gpu = gpus[int(gpu_row["index"])]
        gpu.resident = {str(m): float(mb) for m, mb in (gpu_row.get("resident") or {}).items()}
        gpu.active_tasks = {}
        for task_row in gpu_row.get("active_tasks") or []:
            owner = None
            if task_row.get("owner_job") is not None:
                owner = (int(task_row["owner_job"]), str(task_row["owner_node"]))
            gpu.add_active_task(
                str(task_row["task_id"]),
                float(task_row["start_ms"]),
                float(task_row["finish_ms"]),
                kind=str(task_row.get("kind") or "node"),
                owner=owner,
                model_id=str(task_row.get("model_id") or ""),
                role=str(task_row.get("role") or ""),
                workload_shape=str(task_row.get("workload_shape") or ""),
                workspace_mb=float(task_row.get("workspace_mb") or 0.0),
                allocation_mb=float(task_row.get("allocation_mb") or 0.0),
                interference_target=task_row.get("interference_target"),
                slowdown=float(task_row.get("slowdown") or 1.0),
                interference_mode=str(task_row.get("interference_mode") or ""),
                interference_model=str(task_row.get("interference_model") or ""),
                work_start_ms=task_row.get("work_start_ms"),
                predicted_work_ms=task_row.get("predicted_work_ms"),
                resume_tokens_base=task_row.get("resume_tokens_base"),
            )
            task = gpu.active_tasks[str(task_row["task_id"])]
            if task_row.get("remaining_work_ms") is not None:
                task.remaining_work_ms = float(task_row["remaining_work_ms"])
            if task_row.get("total_work_ms") is not None:
                task.total_work_ms = float(task_row["total_work_ms"])
            if task_row.get("intrinsic_remaining_work_ms") is not None:
                task.intrinsic_remaining_work_ms = float(task_row["intrinsic_remaining_work_ms"])
            task.stall_remaining_ms = float(task_row.get("stall_remaining_ms") or 0.0)
            if task_row.get("last_update_ms") is not None:
                task.last_update_ms = float(task_row["last_update_ms"])
        gpu.prefetch_pending = [
            tuple(entry) for entry in (gpu_row.get("prefetch_pending") or [])
        ]
        gpu.prefetched_models = set(str(v) for v in gpu_row.get("prefetched_models") or [])
        gpu.used_prefetched_models = set(
            str(v) for v in gpu_row.get("used_prefetched_models") or []
        )
        gpu.evicted_models_seen = set(str(v) for v in gpu_row.get("evicted_models_seen") or [])
        gpu.evictions = int(gpu_row.get("evictions") or 0)
        gpu.prefetch_count = int(gpu_row.get("prefetch_count") or 0)
        gpu.prefetch_load_ms = float(gpu_row.get("prefetch_load_ms") or 0.0)
        gpu.wasted_prefetches = int(gpu_row.get("wasted_prefetches") or 0)
        gpu.cold_loads_on_demand = int(gpu_row.get("cold_loads_on_demand") or 0)
        gpu.evicted_then_reloaded = int(gpu_row.get("evicted_then_reloaded") or 0)
        gpu.peak_memory_mb = float(gpu_row.get("peak_memory_mb") or 0.0)

    return {
        "now": float(injected.get("now_ms") or 0.0),
        "sequence": int(injected.get("sequence") or 0),
        "rr_cursor": int(injected.get("rr_cursor") or 0),
        "finish_heap": [
            (float(t), int(order), int(job_index), str(lane), gpu_index, str(node_id))
            for (t, order, job_index, lane, gpu_index, node_id) in injected.get("finish_entries") or []
        ],
        "ready": [
            (float(priority), float(ready_time), int(job_index), str(node_id))
            for (priority, ready_time, job_index, node_id) in injected.get("ready_items") or []
        ],
        "arrival_heap": [
            (float(arrival), int(job_index))
            for (arrival, job_index) in injected.get("arrivals_pending") or []
        ],
        "active_jobs": [int(v) for v in injected.get("active_jobs") or []],
        "wait_logged": [
            (int(a), str(b)) for (a, b) in injected.get("wait_logged") or []
        ],
    }


# --------------------------------------------------------------------------- #
# scenario templates / planning episode
# --------------------------------------------------------------------------- #


def _scenario_node(attrs: Mapping[str, Any], template: Template, seq: int) -> Node:
    return Node(
        node_id=str(attrs["node_id"]),
        sequence_index=int(attrs.get("sequence_index", seq)),
        predecessors=(),
        successors=(),
        lane=str(attrs.get("lane") or "gpu"),
        model_id=str(attrs.get("model_id") or ""),
        runtime_ms=float(attrs.get("runtime_ms") or 0.0),
        load_ms=attrs.get("load_ms"),
        workspace_peak_mb=attrs.get("workspace_peak_mb"),
        resident_model_mb=attrs.get("resident_model_mb"),
        status=str(attrs.get("status") or "ok"),
        role=str(attrs.get("role") or "other"),
        action_family=str(attrs.get("action_family") or "other"),
        workload_shape=str(attrs.get("workload_shape") or ""),
    )


def _rewire_linear(nodes: Sequence[Node]) -> tuple[Node, ...]:
    out: list[Node] = []
    for index, node in enumerate(nodes):
        predecessors = (nodes[index - 1].node_id,) if index > 0 else ()
        successors = (nodes[index + 1].node_id,) if index + 1 < len(nodes) else ()
        out.append(replace(node, predecessors=predecessors, successors=successors))
    return tuple(out)


def build_scenario_templates(
    templates: Mapping[str, Template],
    scenario: Mapping[str, Any],
) -> dict[str, Template]:
    """Templates with each job's future replaced by the scenario.

    ``scenario["jobs"][job_instance_id]`` carries ``template_id``, ``drop``
    (node ids trimmed past the scenario horizon) and ``replace`` (node id ->
    attribute overrides).  The chain order is preserved; all workloads are
    linear chains (verified: zero nodes with >1 successors).
    """

    out = dict(templates)
    for spec in (scenario.get("jobs") or {}).values():
        template = out[str(spec["template_id"])]
        drop = set(str(v) for v in spec.get("drop") or [])
        replace_map = spec.get("replace") or {}
        nodes: list[Node] = []
        for node in template.nodes:
            if node.node_id in drop:
                continue
            attrs = replace_map.get(node.node_id)
            if attrs:
                updates = {
                    key: attrs[key] for key in SCENARIO_ATTR_KEYS if key in attrs
                }
                node = replace(node, **updates)
            nodes.append(node)
        nodes = list(_rewire_linear(nodes))
        out[template.template_id] = replace(
            template, nodes=tuple(nodes), by_id={n.node_id: n for n in nodes}
        )
    return out


def build_planning_episode(
    episode: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    *,
    include_arrivals: bool,
) -> dict[str, Any]:
    """Assemble the planning episode: original fields + injected state."""

    plan_episode = copy.deepcopy(dict(episode))
    plan_episode["episode_id"] = str(episode.get("episode_id")) + "::plan"
    plan_episode["initial_residency_hint"] = []
    injected = dict(snapshot)
    if not include_arrivals:
        injected = {**injected, "arrivals_pending": []}
    plan_episode["injected_state"] = injected
    return plan_episode


# --------------------------------------------------------------------------- #
# planning run / cost
# --------------------------------------------------------------------------- #


def run_planning(
    plan_episode: Mapping[str, Any],
    templates: Mapping[str, Template],
    *,
    future_artifacts: Mapping[str, Mapping[str, Any]],
    train_stats: Mapping[str, Mapping[str, Any]],
    extension_config: Mapping[str, Any] | None,
    forced_action: Mapping[str, Any] | None = None,
    policy: str = "des_rollout",
    collect_events: bool = True,
    extra_context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Run one planning world.  Returns (summary, events, policy_context)."""

    context: dict[str, Any] = {"activation": {}}
    if extra_context:
        context.update(extra_context)
    if forced_action is not None:
        context["des_action"] = dict(forced_action)
        context["des_first_pending"] = True
    summary, events = simulate_episode(
        plan_episode,
        templates,
        policy,
        train_stats=train_stats,
        future_artifacts=future_artifacts,
        future_horizon=5,
        extension_config=extension_config,
        policy_context=context,
        collect_events=collect_events,
    )
    return summary, events, context


def future_flow_cost(
    events: Sequence[Mapping[str, Any]],
    snapshot: Mapping[str, Any],
    episode: Mapping[str, Any],
) -> dict[str, Any]:
    """C_{1:H}: future flow time (waiting area) from the snapshot time.

    Uses ``job_finish`` events; jobs already complete before the snapshot are
    excluded (their contribution is <= 0).
    """

    now0 = float(snapshot["now_ms"])
    arrivals = {
        str(row["job_instance_id"]): float(row["arrival_ms"])
        for row in episode.get("jobs") or []
    }
    finishes: dict[str, float] = {}
    for event in events:
        if event.get("event_type") == "job_finish":
            finishes[str(event["job_instance_id"])] = float(event["time_ms"])
    total = 0.0
    counted = 0
    for job_id, finish in finishes.items():
        if finish <= now0 + 1e-9:
            continue
        total += finish - max(now0, arrivals.get(job_id, now0))
        counted += 1
    return {"future_flow_ms": total, "jobs_counted": counted}


# --------------------------------------------------------------------------- #
# deterministic scenario generator (greedy chain from the frozen artifacts)
# --------------------------------------------------------------------------- #


def _resident_mb_index(templates: Mapping[str, Template]) -> dict[str, float]:
    index: dict[str, float] = {}
    for template in templates.values():
        for node in template.nodes:
            if node.resident_model_mb is not None and node.model_id not in index:
                index[str(node.model_id)] = float(node.resident_model_mb)
    return index


def greedy_chain_scenario(
    snapshot: Mapping[str, Any],
    episode: Mapping[str, Any],
    templates: Mapping[str, Template],
    future_artifacts: Mapping[str, Mapping[str, Any]],
    train_stats: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Deterministic scenario: replace each pending node with the artifact's
    greedy (top-1) predicted step anchored at the job's last visible node.

    Nodes visible to the planner at the snapshot (complete / running / ready)
    keep their true identity.  Pending nodes past the H5 horizon are trimmed
    (the job is ``capped``; its tail is handled by the terminal value).
    """

    from tracing.analysis.workload_v02_simulator import estimate

    jobs_by_id = {str(j["job_instance_id"]): j for j in episode.get("jobs") or []}
    resident_index = _resident_mb_index(templates)
    specs: dict[str, dict] = {}
    capped: dict[str, bool] = {}
    for row in snapshot.get("jobs") or []:
        job_id = str(row["job_instance_id"])
        job = jobs_by_id[job_id]
        template = templates[str(job["template_id"])]
        chain = list(template.nodes)
        states = row.get("node_state") or {}
        anchor_pos = -1
        for pos, node in enumerate(chain):
            if states.get(node.node_id) != "pending":
                anchor_pos = pos
        if anchor_pos < 0:
            anchor_pos = 0  # nothing reached yet: keep the first node visible
        pending_after = [
            node for node in chain[anchor_pos + 1 :]
            if states.get(node.node_id) == "pending"
        ]
        anchor_node = chain[anchor_pos]
        artifact = future_artifacts.get(str(anchor_node.node_id)) or {}
        scenarios = artifact.get("future_h5") or []
        steps = (scenarios[0].get("steps") or []) if scenarios else []
        replace: dict[str, dict] = {}
        for step_index, node in enumerate(pending_after[:5]):
            if step_index >= len(steps):
                break
            step = steps[step_index]
            resource = step.get("resource") or {}
            runtime_q = resource.get("runtime_ms_quantiles") or {}
            load_q = resource.get("load_duration_ms_quantiles") or {}
            occurrence = resource.get("load_occurrence_probability")
            model_id = str(step.get("model_id") or node.model_id)
            role = str(step.get("role") or node.role)
            lane = str(step.get("execution_lane") or node.lane)
            attrs: dict[str, Any] = {
                "model_id": model_id,
                "role": role,
                "lane": lane,
                "action_family": str(step.get("action_family") or node.action_family),
                "runtime_ms": float(runtime_q.get("p50") or node.runtime_ms),
                "load_ms": (
                    float(load_q.get("p50") or 0.0)
                    if isinstance(occurrence, (int, float)) and float(occurrence) >= 0.5
                    else 0.0
                ),
                "resident_model_mb": resident_index.get(model_id, node.resident_model_mb),
            }
            try:
                probe = replace(node, model_id=model_id, role=role, lane=lane)
                attrs["workspace_peak_mb"] = float(estimate(probe, train_stats)["memory_p95_mb"])
            except Exception:
                attrs["workspace_peak_mb"] = node.workspace_peak_mb
            replace[node.node_id] = attrs
        drop = [node.node_id for node in pending_after[5:]]
        specs[job_id] = {
            "template_id": str(job["template_id"]),
            "replace": replace,
            "drop": drop,
        }
        capped[job_id] = bool(drop)
    return {"jobs": specs, "capped": capped}


# --------------------------------------------------------------------------- #
# Markov joint scenario generator (fit on the training pool; leave-one-out)
# --------------------------------------------------------------------------- #


class MarkovScenarioModel:
    """Joint scenario generator fit on the training pool's real runs.

    Transition counts over consecutive (role, model, lane) triples are collected
    from the training templates only; scenarios for an episode exclude that
    episode's own templates (leave-one-out) so the generator never copies the
    planned run's true future.  Step-1 attributes come from the anchor's frozen
    artifact marginals; later steps follow the empirical transition chain.
    """

    def __init__(self, templates: Mapping[str, Template]) -> None:
        self.transitions: dict[tuple[str, str, str], dict[tuple[str, str, str], int]] = {}
        self.template_triples: dict[str, list[tuple[str, str, str]]] = {}
        for template in templates.values():
            if template.split != "train":
                continue
            triples = [
                (str(node.role), str(node.model_id), str(node.lane))
                for node in template.nodes
                if node.lane in ("gpu", "cpu")
            ]
            self.template_triples[template.template_id] = triples
            for a, b in zip(triples, triples[1:]):
                self.transitions.setdefault(a, {})
                self.transitions[a][b] = self.transitions[a].get(b, 0) + 1

    def _loo_transitions(self, exclude_template_ids: set[str]):
        counts: dict[tuple[str, str, str], dict[tuple[str, str, str], int]] = {}
        for template_id, triples in self.template_triples.items():
            if template_id in exclude_template_ids:
                continue
            for a, b in zip(triples, triples[1:]):
                counts.setdefault(a, {})
                counts[a][b] = counts[a].get(b, 0) + 1
        return counts

    def scenario(
        self,
        snapshot: Mapping[str, Any],
        episode: Mapping[str, Any],
        templates: Mapping[str, Template],
        future_artifacts: Mapping[str, Mapping[str, Any]],
        train_stats: Mapping[str, Mapping[str, Any]],
        rng: Any,
    ) -> dict[str, Any]:
        """One sampled scenario spec (same shape as :func:`greedy_chain_scenario`)."""

        from tracing.analysis.workload_v02_simulator import estimate

        jobs_by_id = {str(j["job_instance_id"]): j for j in episode.get("jobs") or []}
        resident_index = _resident_mb_index(templates)
        exclude = {
            str(jobs_by_id[str(row["job_instance_id"])]["template_id"])
            for row in snapshot.get("jobs") or []
            if str(row["job_instance_id"]) in jobs_by_id
        }
        transitions = self._loo_transitions(exclude)
        specs: dict[str, dict] = {}
        capped: dict[str, bool] = {}
        for row in snapshot.get("jobs") or []:
            job_id = str(row["job_instance_id"])
            job = jobs_by_id[job_id]
            template = templates[str(job["template_id"])]
            chain = list(template.nodes)
            states = row.get("node_state") or {}
            anchor_pos = -1
            for pos, node in enumerate(chain):
                if states.get(node.node_id) != "pending":
                    anchor_pos = pos
            if anchor_pos < 0:
                anchor_pos = 0
            pending_after = [
                node for node in chain[anchor_pos + 1 :]
                if states.get(node.node_id) == "pending"
            ]
            anchor_node = chain[anchor_pos]
            artifact = future_artifacts.get(str(anchor_node.node_id)) or {}
            scenarios = artifact.get("future_h5") or []
            steps = (scenarios[0].get("steps") or []) if scenarios else []
            length_probs = artifact.get("length_probabilities") or []
            sampled_length = 5
            if length_probs:
                bucket = rng.choices(range(len(length_probs)), weights=length_probs, k=1)[0]
                sampled_length = max(1, int(bucket))
            replace: dict[str, dict] = {}
            previous = (str(anchor_node.role), str(anchor_node.model_id), str(anchor_node.lane))
            for step_index, node in enumerate(pending_after[:5]):
                if step_index >= sampled_length:
                    break
                if step_index < len(steps):
                    step = steps[step_index]
                    model_dist = step.get("model_probabilities") or {}
                    if model_dist and step_index == 0:
                        models = list(model_dist)
                        weights = [max(0.0, float(model_dist[m])) for m in models]
                        model_id = str(rng.choices(models, weights=weights, k=1)[0]) if models else str(step.get("model_id") or node.model_id)
                        role = str(step.get("role") or node.role)
                        lane = str(step.get("execution_lane") or node.lane)
                    else:
                        role, model_id, lane = self._sample_transition(
                            transitions, previous, step, node, rng,
                        )
                    runtime_q = (step.get("resource") or {}).get("runtime_ms_quantiles") or {}
                    load_q = (step.get("resource") or {}).get("load_duration_ms_quantiles") or {}
                    occurrence = (step.get("resource") or {}).get("load_occurrence_probability")
                    runtime_ms = self._sample_quantile(runtime_q, rng, node.runtime_ms)
                    load_ms = (
                        self._sample_quantile(load_q, rng, 0.0)
                        if isinstance(occurrence, (int, float)) and rng.random() < float(occurrence)
                        else 0.0
                    )
                else:
                    role, model_id, lane = self._sample_transition(
                        transitions, previous, {}, node, rng,
                    )
                    runtime_ms = node.runtime_ms
                    load_ms = float(node.load_ms or 0.0)
                attrs: dict[str, Any] = {
                    "model_id": model_id,
                    "role": role,
                    "lane": lane,
                    "runtime_ms": float(runtime_ms),
                    "load_ms": float(load_ms),
                    "resident_model_mb": resident_index.get(model_id, node.resident_model_mb),
                }
                try:
                    probe = replace(node, model_id=model_id, role=role, lane=lane)
                    attrs["workspace_peak_mb"] = float(estimate(probe, train_stats)["memory_p95_mb"])
                except Exception:
                    attrs["workspace_peak_mb"] = node.workspace_peak_mb
                replace[node.node_id] = attrs
                previous = (str(role), str(model_id), str(lane))
            drop = [node.node_id for node in pending_after[min(sampled_length, 5):]]
            if len(pending_after) > 5:
                drop = [node.node_id for node in pending_after[5:]]
            specs[job_id] = {
                "template_id": str(job["template_id"]),
                "replace": replace,
                "drop": drop,
            }
            capped[job_id] = bool(drop) or sampled_length >= 5
        return {"jobs": specs, "capped": capped}

    @staticmethod
    def _sample_quantile(quantiles: Mapping[str, Any], rng: Any, default: float) -> float:
        values = []
        for key in ("p50", "p90", "p95"):
            value = quantiles.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
        if not values:
            return float(default)
        if len(values) == 1:
            return values[0]
        weights = [0.6, 0.3, 0.1][: len(values)]
        return float(rng.choices(values, weights=weights, k=1)[0])

    @staticmethod
    def _sample_transition(transitions, previous, step, node, rng):
        options = transitions.get(previous)
        if options:
            choices = list(options)
            weights = [options[c] for c in choices]
            role, model_id, lane = rng.choices(choices, weights=weights, k=1)[0]
            return str(role), str(model_id), str(lane)
        fallback_model = str(step.get("model_id") or node.model_id)
        fallback_role = str(step.get("role") or node.role)
        fallback_lane = str(step.get("execution_lane") or node.lane)
        return fallback_role, fallback_model, fallback_lane
