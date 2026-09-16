"""Causal scheduler state and provider boundaries.

This module is deliberately a small contract layer.  It does not run a
workload by itself; the event loop owns :class:`ExecutionTruthProvider`, while
policies receive only :class:`SchedulerStateView`.  Keeping the boundary in a
standalone module lets the legacy simulator be used for compatibility replay
without making it a second formal engine.

The objects are intentionally immutable at the policy boundary.  Resource
values on a state view are estimates, and future rows contain identities only;
runtime, load, workspace, status and output remain execution truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence


FutureMode = Literal["none", "pred_h", "true_h", "full_truth"]

_FORBIDDEN_STATE_KEYS = frozenset(
    {
        "runtime_ms",
        "load_ms",
        "workspace_peak_mb",
        "resident_model_mb",
        "status",
        "output",
        "answer",
        "teacher",
        "future_events",
        "successors",
        "predecessors",
        "template_id",
        "video_id",
    }
)


def _clean_text(value: Any, default: str = "unknown") -> str:
    result = str(value or "").strip()
    return result or default


def _nonnegative(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result >= 0.0 else default


@dataclass(frozen=True)
class ResourceEstimate:
    """Scheduler-safe resource prediction for the currently ready node."""

    runtime_p50_ms: float
    runtime_p90_ms: float
    load_p50_ms: float
    peak_memory_p95_mb: float
    supported: bool = True

    def __post_init__(self) -> None:
        for name in (
            "runtime_p50_ms",
            "runtime_p90_ms",
            "load_p50_ms",
            "peak_memory_p95_mb",
        ):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"resource estimate {name} must be non-negative")
        if self.runtime_p90_ms < self.runtime_p50_ms:
            raise ValueError("runtime_p90_ms must be >= runtime_p50_ms")

    def to_scheduler_dict(self) -> dict[str, Any]:
        return {
            "runtime_p50_ms": self.runtime_p50_ms,
            "runtime_p90_ms": self.runtime_p90_ms,
            "load_p50_ms": self.load_p50_ms,
            "peak_memory_p95_mb": self.peak_memory_p95_mb,
            "supported": self.supported,
        }


@dataclass(frozen=True)
class ReadyNodeView:
    """Identity and predicted cost of one currently ready GPU node."""

    job_instance_id: str
    node_id: str
    sequence_index: int
    role: str
    action_family: str
    model_id: str
    lane: str
    ready_since_ms: float
    resource: ResourceEstimate

    def to_scheduler_dict(self) -> dict[str, Any]:
        return {
            "job_instance_id": self.job_instance_id,
            "node_id": self.node_id,
            "sequence_index": self.sequence_index,
            "role": self.role,
            "action_family": self.action_family,
            "model_id": self.model_id,
            "lane": self.lane,
            "ready_since_ms": self.ready_since_ms,
            "resource": self.resource.to_scheduler_dict(),
        }


@dataclass(frozen=True)
class GPUStateView:
    """The GPU facts visible to a policy at a decision point."""

    index: int
    capacity_mb: float
    busy: bool
    resident_models: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.index < 0 or self.capacity_mb < 0.0:
            raise ValueError("GPU index/capacity must be non-negative")

    def to_scheduler_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "capacity_mb": self.capacity_mb,
            "busy": self.busy,
            "resident_models": list(self.resident_models),
        }


@dataclass(frozen=True)
class FutureStep:
    """A future identity, never a future execution result."""

    offset: int
    role: str
    action_family: str
    model_id: str | None = None
    lane: str | None = None
    supported: bool = True

    def __post_init__(self) -> None:
        if self.offset <= 0:
            raise ValueError("future offset must be positive")

    def to_scheduler_dict(self) -> dict[str, Any]:
        return {
            "offset": self.offset,
            "role": self.role,
            "action_family": self.action_family,
            "model_id": self.model_id,
            "lane": self.lane,
            "supported": self.supported,
        }


@dataclass(frozen=True)
class FutureScenario:
    probability: float
    steps: tuple[FutureStep, ...]

    def __post_init__(self) -> None:
        if self.probability < 0.0:
            raise ValueError("future scenario probability must be non-negative")

    def to_scheduler_dict(self) -> dict[str, Any]:
        return {
            "probability": self.probability,
            "steps": [step.to_scheduler_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class FutureReveal:
    mode: FutureMode
    requested_horizon: int
    effective_horizon: int
    supported_probability_mass: float
    scenarios: tuple[FutureScenario, ...]
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if self.requested_horizon < 0 or self.effective_horizon < 0:
            raise ValueError("future horizon must be non-negative")
        if not 0.0 <= self.supported_probability_mass <= 1.0 + 1e-9:
            raise ValueError("supported probability mass must be in [0, 1]")

    def to_scheduler_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "requested_horizon": self.requested_horizon,
            "effective_horizon": self.effective_horizon,
            "supported_probability_mass": self.supported_probability_mass,
            "fallback_reason": self.fallback_reason,
            "scenarios": [scenario.to_scheduler_dict() for scenario in self.scenarios],
        }


@dataclass(frozen=True)
class ExecutionTruth:
    """Measured execution data kept behind the engine-only boundary."""

    runtime_ms: float
    load_ms: float
    workspace_peak_mb: float
    status: str
    output: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.runtime_ms < 0.0 or self.load_ms < 0.0 or self.workspace_peak_mb < 0.0:
            raise ValueError("execution truth values must be non-negative")


class ExecutionTruthProvider:
    """Engine-only lookup for measured node outcomes.

    The provider intentionally exposes no scheduler serialization method.  A
    policy can receive a state view, but cannot obtain this provider through
    that view.
    """

    def __init__(self, rows: Mapping[tuple[str, str], ExecutionTruth]) -> None:
        self._rows = dict(rows)

    def get(self, job_instance_id: str, node_id: str) -> ExecutionTruth:
        key = (str(job_instance_id), str(node_id))
        try:
            return self._rows[key]
        except KeyError as exc:
            raise KeyError(f"missing execution truth for {key}") from exc


class FutureProvider:
    """Finite-horizon provider with explicit information modes.

    ``pred_h`` accepts scenario rows produced by a frozen predictor.  ``true_h``
    and ``full_truth`` may reveal future identities for an information upper
    bound, but strip all measured resource/result fields before constructing a
    scheduler view.  ``none`` never touches either artifact mapping.
    """

    def __init__(
        self,
        mode: FutureMode = "none",
        predicted: Mapping[tuple[str, str, int], Sequence[Mapping[str, Any]]] | None = None,
        truth: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None = None,
    ) -> None:
        if mode not in {"none", "pred_h", "true_h", "full_truth"}:
            raise ValueError(f"unsupported future mode: {mode}")
        self.mode = mode
        self._predicted = dict(predicted or {})
        self._truth = dict(truth or {})

    @staticmethod
    def _step(row: Mapping[str, Any], offset: int) -> FutureStep:
        forbidden = _FORBIDDEN_STATE_KEYS.intersection(str(key) for key in row)
        if forbidden:
            raise ValueError(f"future identity contains forbidden execution fields: {sorted(forbidden)}")
        return FutureStep(
            offset=offset,
            role=_clean_text(row.get("role")),
            action_family=_clean_text(row.get("action_family")),
            model_id=(str(row["model_id"]) if row.get("model_id") is not None else None),
            lane=(str(row["lane"]) if row.get("lane") is not None else None),
            supported=bool(row.get("supported", True)),
        )

    def reveal(self, job_instance_id: str, node_id: str, horizon: int) -> FutureReveal:
        if horizon < 0:
            raise ValueError("horizon must be non-negative")
        if self.mode == "none" or horizon == 0:
            return FutureReveal(self.mode, horizon, 0, 0.0, ())
        if self.mode == "pred_h":
            rows = self._predicted.get((str(job_instance_id), str(node_id), horizon), ())
            scenarios: list[FutureScenario] = []
            supported_mass = 0.0
            for raw in rows:
                probability = _nonnegative(raw.get("scenario_probability"), 0.0)
                raw_steps = raw.get("steps") or ()
                steps = tuple(self._step(step, index) for index, step in enumerate(raw_steps, 1))
                if all(step.supported for step in steps):
                    supported_mass += probability
                    scenarios.append(FutureScenario(probability, steps))
            if supported_mass > 1.0 + 1e-6:
                raise ValueError("predicted scenario probabilities exceed one")
            if supported_mass > 0.0:
                scenarios = [
                    FutureScenario(scenario.probability / supported_mass, scenario.steps)
                    for scenario in scenarios
                ]
            return FutureReveal(
                "pred_h",
                horizon,
                horizon if scenarios else 0,
                min(1.0, supported_mass),
                tuple(scenarios),
                None if scenarios else "no_supported_scenario",
            )

        rows = list(self._truth.get((str(job_instance_id), str(node_id)), ()))
        steps = tuple(self._step(row, index) for index, row in enumerate(rows[:horizon], 1))
        if self.mode == "true_h":
            steps = steps[:horizon]
            effective = len(steps)
        else:
            effective = len(steps)
        scenario = FutureScenario(1.0, steps) if steps else None
        return FutureReveal(
            self.mode,
            horizon,
            effective,
            1.0 if steps else 0.0,
            (scenario,) if scenario else (),
            None if steps else "no_future_step",
        )


@dataclass(frozen=True)
class SchedulerStateView:
    """Complete scheduler-visible state at one decision point."""

    episode_id: str
    decision_index: int
    time_ms: float
    ready_nodes: tuple[ReadyNodeView, ...]
    completed_prefix: Mapping[str, tuple[str, ...]]
    gpus: tuple[GPUStateView, ...]
    future: Mapping[str, FutureReveal]

    def to_scheduler_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": "scheduler-state-view-v1",
            "episode_id": self.episode_id,
            "decision_index": self.decision_index,
            "time_ms": self.time_ms,
            "ready_nodes": [node.to_scheduler_dict() for node in self.ready_nodes],
            "completed_prefix": {
                str(job_id): list(node_ids) for job_id, node_ids in self.completed_prefix.items()
            },
            "gpus": [gpu.to_scheduler_dict() for gpu in self.gpus],
            "future": {
                f"{job_id}:{node_id}": reveal.to_scheduler_dict()
                for (job_id, node_id), reveal in self.future.items()
            },
        }
        validate_scheduler_payload(payload)
        return payload


def validate_scheduler_payload(payload: Mapping[str, Any]) -> None:
    """Reject obvious truth/full-DAG fields before a policy sees a payload."""

    def walk(value: Any, path: str = "state") -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key)
                if key_text in _FORBIDDEN_STATE_KEYS:
                    raise ValueError(f"forbidden scheduler field at {path}.{key_text}")
                walk(child, f"{path}.{key_text}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(payload)


def _future_key(job_id: str, node_id: str) -> tuple[str, str]:
    return (str(job_id), str(node_id))


def build_scheduler_state(
    *,
    episode_id: str,
    decision_index: int,
    time_ms: float,
    ready_nodes: Sequence[Mapping[str, Any]],
    completed_prefix: Mapping[str, Sequence[str]],
    gpus: Sequence[Mapping[str, Any]],
    resource_predictions: Mapping[tuple[str, str], ResourceEstimate],
    future_provider: FutureProvider,
    horizon: int,
) -> SchedulerStateView:
    """Construct a policy state from explicitly separated inputs.

    Callers must provide current-node identities and a scheduler-safe resource
    estimate separately from any measured trace row.  The builder never copies
    unknown fields from the input mappings.
    """

    nodes: list[ReadyNodeView] = []
    future: dict[tuple[str, str], FutureReveal] = {}
    for raw in ready_nodes:
        job_id = _clean_text(raw.get("job_instance_id"), "unknown-job")
        node_id = _clean_text(raw.get("node_id"), "unknown-node")
        key = _future_key(job_id, node_id)
        if key not in resource_predictions:
            raise KeyError(f"missing scheduler resource prediction for {key}")
        nodes.append(
            ReadyNodeView(
                job_instance_id=job_id,
                node_id=node_id,
                sequence_index=int(raw.get("sequence_index") or 0),
                role=_clean_text(raw.get("role")),
                action_family=_clean_text(raw.get("action_family")),
                model_id=_clean_text(raw.get("model_id")),
                lane=_clean_text(raw.get("lane")),
                ready_since_ms=_nonnegative(raw.get("ready_since_ms"), time_ms),
                resource=resource_predictions[key],
            )
        )
        future[key] = future_provider.reveal(job_id, node_id, horizon)

    gpu_views = tuple(
        GPUStateView(
            index=int(raw.get("index") or 0),
            capacity_mb=_nonnegative(raw.get("capacity_mb")),
            busy=bool(raw.get("busy", False)),
            resident_models=tuple(str(model) for model in (raw.get("resident_models") or ())),
        )
        for raw in gpus
    )
    state = SchedulerStateView(
        episode_id=str(episode_id),
        decision_index=int(decision_index),
        time_ms=_nonnegative(time_ms),
        ready_nodes=tuple(nodes),
        completed_prefix={str(job): tuple(str(node) for node in nodes) for job, nodes in completed_prefix.items()},
        gpus=gpu_views,
        future=future,
    )
    state.to_scheduler_dict()
    return state

