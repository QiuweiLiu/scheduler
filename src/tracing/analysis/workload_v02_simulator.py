#!/usr/bin/env python3
"""Run a node-level event simulator for workload-episode-v0.2.

This is a controlled Phase 4 simulator, not a physical multi-GPU benchmark.
GPU nodes use one active node per GPU; CPU/API nodes can overlap.  Policies
receive train-only empirical estimates while the simulator keeps measured
template runtime/memory as hidden execution truth.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import heapq
import json
import math
import random
import statistics
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tracing.scheduling.event_engine import (
    ExecutionTruth,
    ExecutionTruthProvider,
    FutureProvider,
    GPUStateView,
    ResourceEstimate,
    build_scheduler_state,
)
from tracing.scheduling.future_topology import validate_layer_scenarios


POLICIES = (
    "round_robin",
    "fcfs",
    "sjf_pred",
    "state_aware",
    "myopic",
    "risk_aware",
    "optimizer_0",
    "oracle",
    "predopt_h1",
    "predopt_h3",
    "predopt_h5",
    "predopt_h1_jres",
    "predopt_h3_jres",
    "predopt_h5_jres",
    "predopt_h1_jrt",
    "predopt_h3_jrt",
    "predopt_h5_jrt",
    "predopt_h5_oc",
    "predopt_h5_exp",
    "predopt_h5_surv",
    "oracle_topology_h5_tab",
    "predopt_v2_h1",
    "predopt_v2_h3",
    "predopt_v2_h5",
    "cp_rho_h1",
    "cp_rho_h3",
    "cp_rho_h5",
    "pred_mpc_h3",
    "pred_mpc_h5",
    "pred_mpc_h5_no_terminal",
    "trueopt_h1",
    "trueopt_h3",
    "trueopt_h5",
    "trueopt_h10",
    "trueopt_h20",
    "predopt_h5_risk",
    "predopt_h5_lam0",
    "predopt_h5_lam25",
    "predopt_h5_lam50",
    "predopt_h5_lam100",
    "predopt_h5_q95",
    "predopt_h5_cc",
    "sameshape_h5_p50",
    "sameshape_h5_p95",
    "sameshape_h5_truth",
    "sameshape_h5_condmean",
    "sameshape_h5_stepcvar95",
    "sameshape_h5_p95_aging",
    "tie_current",
    "pythia_completion",
    "llmsched",
    "latency_aware",
    "agentix",
    "predopt_h10_lam0",
    "predopt_h10_lam25",
    "predopt_h10_lam50",
    "predopt_h10_lam100",
    "predopt_h10_q95",
    "predopt_h10_cc",
    "predopt_h5_scen_k0",
    "predopt_h5_scen_k50",
    "predopt_h5_scen_k100",
    "predopt_h5_comon_k50",
    "predopt_h5_comon_k100",
    "predopt_h5_comon_k200",
    "predopt_h5_adapt",
    "predopt_h5_adapt_q",
    "predopt_h5_rt95",
    "predopt_h5_ld95",
    "predopt_h5_r50",
    "predopt_h5_r90",
    "predopt_h5_r95",
    "predopt_h5_r50k",
    "predopt_h5_scen128_k0",
    "predopt_h5_scen128_k50",
    "predopt_h5_scen128_k100",
    "predopt_h5_comon128_k50",
    "predopt_h5_comon128_k100",
    "predopt_h5_scen128_fixedl_k100",
    "predopt_h5_comon128_fixedl_k100",
    "aligned_predopt_h5_layer_res",
    "rl_0",
    "rl_h5",
    "bc_h1",
    "bc_h3",
    "bc_h5",
)
ALIGNED_H5_POLICIES = (
    "aligned_predopt_h5",
    "aligned_predopt_h5_layer",
    "aligned_trueopt_h5",
)
PREDOPT_V2_PRIORITY_WEIGHT = 0.25
PREDOPT_V2_FUTURE_WEIGHT = 1.0
PREDOPT_V2_MEMORY_WEIGHT = 0.05
RISK_AWARE_URGENCY_WEIGHT = 1.0
RISK_AWARE_RESOURCE_WEIGHT = 1.0
RISK_AWARE_AGE_WEIGHT = 1.0
EVENT_ORDER = {"finish": 0, "arrival": 1}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def load_future_artifacts(root: Path) -> dict[str, dict[str, Any]]:
    """Load R7 finite-horizon artifacts keyed by node id.

    The legacy H1/H3/H5 files remain required.  The repaired layer-H5 file is
    an optional sidecar, so existing artifact directories and policies retain
    their exact behavior when it is absent.
    """
    result: dict[str, dict[str, Any]] = {}
    # Legacy H1/H3/H5 files stay required; every other b05_future_h* file
    # (e.g. the H=10 horizon extension) is an optional sidecar whose fields are
    # carried verbatim so horizon-specific policies can consume them.
    required = ("b05_node_h1.jsonl.gz", "b05_future_h3.jsonl.gz", "b05_future_h5.jsonl.gz")
    files = sorted(root.glob("b05_future_h*.jsonl.gz"))
    files.extend(root / name for name in required if (root / name) not in files)
    for path in files:
        filename = path.name
        if not path.is_file() and filename != "b05_future_h5_layers.jsonl.gz":
            raise FileNotFoundError(path)
        if not path.is_file():
            continue
        for row in read_gzip_jsonl(path):
            node_id = str(row.get("node_id") or "")
            if not node_id:
                raise ValueError(f"future artifact row without node_id: {path}")
            # Carry every field: future_h* structures plus the metadata that
            # consumers need (predicted_future_length, length_probabilities,
            # termination_probability, schema/contract stamps).  The earlier
            # future_h*-only filter silently dropped the length distribution and
            # made length-based consumers degenerate.
            result.setdefault(node_id, {}).update(row)
    return result


# --------------------------------------------------------------------------- #
# resource-v2 artifact contract
#
# The resource-v2 arms publish only a repaired ``b05_future_h5.jsonl.gz`` plus a
# manifest; the legacy H1/H3 files stay owned by the frozen base pack.  Rather
# than copying those files (which would let the two drift apart silently), the
# overlay loader splices the new H5 onto the base root and refuses to return
# anything unless the packs agree on everything except the runtime block.
RESOURCE_V2_OWNED_KEYS: tuple[str, ...] = (
    "runtime_probs",
    "runtime_ms_quantiles",
    "runtime_mean_ms",
    "cvar95_ms",
    "resource_head_id",
    "bin_schema_id",
)
RESOURCE_V2_PRESERVED_KEYS: tuple[str, ...] = (
    "load_occurrence_probability",
    "load_duration_ms_quantiles",
)
RESOURCE_V2_VIEW_TOLERANCE = 1e-6


def _resource_v2_views(
    probs: Sequence[float], reps: Sequence[float], alpha: float
) -> dict[str, float]:
    """Re-derive every canonical view from the distribution (pure stdlib)."""

    if len(probs) != len(reps):
        raise ValueError("runtime_probs and bin representatives have different lengths")
    cdf = 0.0
    views: dict[str, float] = {}
    mean = 0.0
    tail = 0.0
    prev = 0.0
    for prob, rep in zip(probs, reps):
        mean += float(prob) * float(rep)
        cdf += float(prob)
        if cdf > alpha:
            tail += (cdf - max(prev, alpha)) * float(rep)
        prev = cdf
    views["runtime_mean_ms"] = mean
    views["cvar95_ms"] = tail / max(1e-9, 1.0 - alpha)
    running = 0.0
    for tau, key in ((0.50, "p50"), (0.90, "p90"), (0.95, "p95")):
        running = 0.0
        for prob, rep in zip(probs, reps):
            running += float(prob)
            if running >= tau - 1e-12:
                views[key] = float(rep)
                break
        else:
            views[key] = float(reps[-1])
    return views


def _strip_resource_v2_keys(record: Mapping[str, Any], horizon: int) -> list[Any]:
    """The node's horizon block with every runtime-owned key removed.

    Only the ``future_h<horizon>`` block is compared: the base root merges H1/H3/H5
    while a resource-v2 arm deliberately publishes H5 alone, so a whole-record
    comparison would flag the intentionally absent horizons as drift.
    """

    import copy as _copy

    scenarios = record.get(f"future_h{int(horizon)}") or []
    clone = _copy.deepcopy(list(scenarios))
    for scenario in clone:
        if not isinstance(scenario, dict):
            continue
        for step in scenario.get("steps") or []:
            resource = step.get("resource")
            if isinstance(resource, dict):
                for key in RESOURCE_V2_OWNED_KEYS:
                    resource.pop(key, None)
    return clone


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_resource_v2_overlay(
    base_root: Path, arm_root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Splice a resource-v2 H5 pack onto a frozen base artifact root.

    Returns ``(artifacts, preflight)``.  Every counter in ``preflight`` is either
    zero or within tolerance on success; any real disagreement raises instead of
    returning a partially repaired artifact (the review's P0-2).
    """

    manifest_path = arm_root / "resource_v2_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not str(manifest.get("schema_version", "")).startswith("resource-v2-artifact-manifest"):
        raise ValueError(f"unexpected resource-v2 manifest schema: {manifest.get('schema_version')!r}")

    horizon = int(manifest["horizon"])
    identity = bool(manifest.get("identity_copy"))
    edges = [float(e) for e in manifest["bin_edges_ms"]]
    reps = [float(r) for r in manifest["bin_representatives_ms"]]
    alpha = float(manifest["cvar_alpha"])
    bin_schema_id = str(manifest["bin_schema_id"])

    base = load_future_artifacts(base_root)
    overlay_path = arm_root / "b05_future_h5.jsonl.gz"
    overlay: dict[str, dict[str, Any]] = {}
    for row in read_gzip_jsonl(overlay_path):
        node_id = str(row.get("node_id") or "")
        if not node_id:
            raise ValueError(f"resource-v2 row without node_id: {overlay_path}")
        if node_id in overlay:
            raise ValueError(f"duplicate node_id in resource-v2 pack: {node_id}")
        overlay[node_id] = row

    preflight: dict[str, Any] = {
        "base_root": str(base_root),
        "arm_root": str(arm_root),
        "artifact_sha256": manifest.get("artifact_sha256"),
        "artifact_id": manifest.get("artifact_id"),
        "base_pack_sha256": manifest.get("base_pack_sha256"),
        "producer_checkpoint_sha256": manifest.get("producer_checkpoint_sha256"),
        "resource_head_id": manifest.get("resource_head_id"),
        "bin_schema_id": bin_schema_id,
        "identity_copy": identity,
        "node_count": len(overlay),
        "step_count": 0,
        "missing_nodes": 0,
        "duplicate_nodes": 0,
        "multi_scenario_rows": 0,
        "unupgraded_steps": 0,
        "prob_sum_max_abs_error": 0.0,
        "canonical_view_max_abs_error": 0.0,
        "nonresource_mismatch_count": 0,
        "bin_schema_mismatch_count": 0,
        "load_field_missing_count": 0,
        "nan_prob_count": 0,
        "out_of_range_prob_count": 0,
    }

    # the manifest is the artifact's identity: verify it against the bytes on disk
    # instead of trusting the numbers it carries
    declared_artifact = manifest.get("artifact_sha256")
    actual_artifact = _sha256_file(overlay_path)
    preflight["artifact_sha256_recomputed"] = actual_artifact
    preflight["artifact_sha256_match"] = (declared_artifact == actual_artifact)
    if declared_artifact and declared_artifact != actual_artifact:
        raise ValueError(
            "resource-v2 pack sha256 mismatch: manifest=%s disk=%s"
            % (declared_artifact, actual_artifact)
        )
    base_h5 = base_root / "b05_future_h5.jsonl.gz"
    if base_h5.is_file():
        declared_base = manifest.get("base_pack_sha256")
        actual_base = _sha256_file(base_h5)
        preflight["base_pack_sha256_recomputed"] = actual_base
        preflight["base_pack_sha256_match"] = (declared_base == actual_base)
        if declared_base and declared_base != actual_base:
            raise ValueError(
                "base pack sha256 mismatch: manifest=%s disk=%s" % (declared_base, actual_base)
            )

    missing = sorted(set(base) - set(overlay))
    extra = sorted(set(overlay) - set(base))
    if missing or extra:
        raise ValueError(
            "node sets differ: %d missing, %d extra (e.g. %s)"
            % (len(missing), len(extra), (missing or extra)[:3])
        )

    result: dict[str, dict[str, Any]] = {}
    for node_id, arm_record in overlay.items():
        base_record = base[node_id]
        base_scen = base_record.get(f"future_h{horizon}") or []
        arm_scen = arm_record.get(f"future_h{horizon}") or []
        if len(arm_scen) != 1:
            preflight["multi_scenario_rows"] += 1
            raise ValueError(f"node {node_id} carries {len(arm_scen)} scenarios, expected exactly 1")
        if len(base_scen) != 1:
            raise ValueError(f"base node {node_id} carries {len(base_scen)} scenarios")

        if _strip_resource_v2_keys(base_record, horizon) != _strip_resource_v2_keys(arm_record, horizon):
            preflight["nonresource_mismatch_count"] += 1

        base_steps = base_scen[0].get("steps") or []
        arm_steps = arm_scen[0].get("steps") or []
        if len(base_steps) != len(arm_steps):
            raise ValueError(
                "node %s step count changed: %d -> %d" % (node_id, len(base_steps), len(arm_steps))
            )
        if identity and base_scen != arm_scen:
            # an identity arm must reproduce the H5 block exactly, runtime keys included
            raise ValueError("identity arm changed the H5 block of node %s" % node_id)

        for base_step, step in zip(base_steps, arm_steps):
            preflight["step_count"] += 1
            resource = step.get("resource") or {}
            # only require what the base step actually carried, so a packer that never
            # emits a load field is not reported as damage
            base_resource = base_step.get("resource") or {}
            for key in RESOURCE_V2_PRESERVED_KEYS:
                if key in base_resource and key not in resource:
                    preflight["load_field_missing_count"] += 1
            if identity:
                # declared identity artifact: it publishes no distribution by design, and
                # the exact H5 comparison above is the real guarantee
                continue
            probs = resource.get("runtime_probs")
            if not isinstance(probs, list) or len(probs) != len(reps):
                preflight["unupgraded_steps"] += 1
                continue
            # bool is a subclass of int in Python, and NaN/Inf compare false against
            # every bound, so the earlier isinstance-only check let all three through.
            if any(isinstance(p, bool) or not isinstance(p, (int, float)) for p in probs):
                preflight["nan_prob_count"] += 1
                continue
            if any(not (0.0 <= float(p) <= 1.0) for p in probs):
                preflight["out_of_range_prob_count"] += 1
                continue
            total = sum(float(p) for p in probs)
            preflight["prob_sum_max_abs_error"] = max(
                preflight["prob_sum_max_abs_error"], abs(total - 1.0)
            )
            if resource.get("bin_schema_id") != bin_schema_id:
                preflight["bin_schema_mismatch_count"] += 1
            want = _resource_v2_views(probs, reps, alpha)
            quantiles = resource.get("runtime_ms_quantiles") or {}
            err = 0.0
            for key in ("p50", "p90", "p95"):
                got = optional_number(quantiles.get(key))
                if got is None:
                    err = float("inf")
                else:
                    err = max(err, abs(got - want[key]))
            for key in ("runtime_mean_ms", "cvar95_ms"):
                got = optional_number(resource.get(key))
                if got is None:
                    err = float("inf")
                else:
                    err = max(err, abs(got - want[key]))
            preflight["canonical_view_max_abs_error"] = max(
                preflight["canonical_view_max_abs_error"], err
            )

        merged = dict(base_record)
        merged[f"future_h{horizon}"] = arm_scen
        result[node_id] = merged

    if preflight["multi_scenario_rows"]:
        raise ValueError("multi-scenario rows: %d" % preflight["multi_scenario_rows"])
    if preflight["unupgraded_steps"]:
        raise ValueError("unupgraded steps: %d" % preflight["unupgraded_steps"])
    if preflight["nonresource_mismatch_count"]:
        raise ValueError(
            "resource-v2 pack disagrees with the base pack outside the runtime block: %d node(s)"
            % preflight["nonresource_mismatch_count"]
        )
    if preflight["bin_schema_mismatch_count"]:
        raise ValueError("steps carrying the wrong bin_schema_id: %d" % preflight["bin_schema_mismatch_count"])
    if preflight["load_field_missing_count"]:
        raise ValueError("steps missing a preserved load field: %d" % preflight["load_field_missing_count"])
    if preflight["nan_prob_count"]:
        raise ValueError("steps carrying non-finite probabilities: %d" % preflight["nan_prob_count"])
    if preflight["out_of_range_prob_count"]:
        raise ValueError(
            "steps carrying probabilities outside [0, 1]: %d" % preflight["out_of_range_prob_count"]
        )
    if preflight["prob_sum_max_abs_error"] > RESOURCE_V2_VIEW_TOLERANCE:
        raise ValueError("runtime_probs do not sum to 1 (max err %.3e)" % preflight["prob_sum_max_abs_error"])
    if preflight["canonical_view_max_abs_error"] > RESOURCE_V2_VIEW_TOLERANCE:
        raise ValueError(
            "canonical views disagree with runtime_probs (max err %.3e)"
            % preflight["canonical_view_max_abs_error"]
        )
    return result, preflight


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def text(value: Any, default: str = "unknown") -> str:
    result = str(value or "").strip()
    return result or default


def number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def optional_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0.0 else None


def quantile(values: Sequence[float], q: float) -> float | None:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)) and float(value) >= 0.0)
    if not clean:
        return None
    position = (len(clean) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return clean[int(lower)]
    fraction = position - lower
    return clean[int(lower)] * (1.0 - fraction) + clean[int(upper)] * fraction


@dataclass(frozen=True)
class Node:
    node_id: str
    sequence_index: int
    predecessors: tuple[str, ...]
    successors: tuple[str, ...]
    lane: str
    model_id: str
    runtime_ms: float
    load_ms: float | None
    workspace_peak_mb: float | None
    resident_model_mb: float | None
    status: str
    role: str = "other"
    action_family: str = "other"
    raw_action: str = "other"
    batch_size: int = 1
    # v3.1 section 3: run_control and the terminal answer marker are not schedulable
    # work.  The v04.1 projection already excludes them, so this defaults to True and
    # exists so the executor can REFUSE rather than silently run such a node.
    resource_applicable: bool = True
    # v3.1 composite signature: a merged summarizer node keeps the CPU lane and the
    # R_total duration, but must still record the inner GPU model so its demand is not
    # lost from GPU contention accounting
    nested_model_class: str = ""
    nested_model_mb: float | None = None
    nested_load_ms: float | None = None
    nested_runtime_ms: float | None = None
    nested_reserved_mb: float | None = None
    # the offsets that place the inner interval inside the parent window, so
    # R_total = pre + inner + post can be reconstructed exactly
    nested_pre_ms: float | None = None
    nested_post_ms: float | None = None
    nested_inner_ms: float | None = None

    @property
    def workspace_incremental_mb(self) -> float:
        if self.lane != "gpu" or self.workspace_peak_mb is None:
            return 0.0
        if self.resident_model_mb is None:
            return self.workspace_peak_mb
        return max(0.0, self.workspace_peak_mb - self.resident_model_mb)

    @property
    def compute_ms(self) -> float:
        return max(0.1, self.runtime_ms - number(self.load_ms))


@dataclass(frozen=True)
class Template:
    template_id: str
    video_id: str
    split: str
    baseline: str
    nodes: tuple[Node, ...]
    by_id: Mapping[str, Node]


@dataclass
class Job:
    job_instance_id: str
    template: Template
    arrival_ms: float
    deadline_ms: float | None
    service_class: str
    node_state: dict[str, str] = field(default_factory=dict)
    ready_since: dict[str, float] = field(default_factory=dict)
    started: set[str] = field(default_factory=set)
    completed: set[str] = field(default_factory=set)
    failed: set[str] = field(default_factory=set)
    finish_ms: float | None = None
    queue_ms: float = 0.0
    load_ms: float = 0.0
    assigned_gpus: list[int] = field(default_factory=list)
    preemptions: int = 0
    preempt_recompute_ms: float = 0.0
    # Intrinsic service durations actually OBSERVED, written at node_finish from the
    # truth provider.  LLMSched evidence reads this and nothing else, so a duration is
    # structurally unavailable before the node finishes rather than merely by
    # convention about when the template is consulted.
    observed_intrinsic_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class GPU:
    index: int
    capacity_mb: float
    busy_until: float = 0.0
    active_node: tuple[int, str] | None = None
    resident: dict[str, float] = field(default_factory=dict)
    peak_memory_mb: float = 0.0
    evictions: int = 0
    busy_time_ms: float = 0.0
    active_start_ms: float | None = None
    prefetch_pending: list[tuple[str, float, float]] = field(default_factory=list)
    prefetched_models: set[str] = field(default_factory=set)
    used_prefetched_models: set[str] = field(default_factory=set)
    # the finish time of the last QUEUED composite segment.  busy_until never falls
    # below it, so a normal completion cannot erase another segment's reservation.  It
    # is deliberately not "occupied from now": the preparation phase touches nothing.
    composite_tail: float = 0.0
    # telemetry only: which composite segment holds the device.  It is deliberately not
    # active_node, so the preemption path cannot mistake a CPU parent for a GPU victim.
    composite_owner: str = ""
    # number of composite segments that hold or are queued for this device.  While this
    # is positive the device's active node must not be preempted, otherwise busy_until
    # can be rewound past the reservation and an unrelated job can be dispatched into it.
    composite_queued: int = 0
    prefetch_count: int = 0
    prefetch_load_ms: float = 0.0
    wasted_prefetches: int = 0


TOPOLOGY_VIEWS = ("legacy", "causal_v3")

# A fused Latency-Aware execution unit closes several nodes at one finish time.  The
# member ids travel in the existing ``node_id`` slot of the finish heap, joined by
# this separator, so the heap tuple shape is unchanged.  A single-node entry splits
# back to itself and every pre-existing policy is unaffected.
FUSED_ID_SEPARATOR = "\x1f"
CAUSAL_TOPOLOGY_CONTRACT = "verified_serial_control_flow_v3_1"


def load_templates(
    path: Path,
    *,
    topology_view: str = "legacy",
) -> dict[str, Template]:
    """Load scheduler templates under an EXPLICIT topology view.

    The v03 workload was built by expanding each parent *step* into all of its
    nodes, which manufactures parallelism that the verified trace does not have.
    That mistake was possible because the executor silently read a single
    ambiguous ``predecessor_node_ids`` field.  The view is therefore explicit:

      * ``legacy``    -> ``predecessor_node_ids`` (the v03 step-expansion edges)
      * ``causal_v3`` -> ``causal_predecessor_node_ids`` (the verified serial chain)

    Fail-closed rules:
      * a file that declares a causal ``topology_contract`` may NOT be loaded as
        legacy, because the executor would silently run the old fake-parallel graph;
      * a causal load on a file without the causal field is refused rather than
        silently falling back.
    """

    if topology_view not in TOPOLOGY_VIEWS:
        raise ValueError("unknown topology_view %r (expected one of %s)"
                         % (topology_view, TOPOLOGY_VIEWS))
    result: dict[str, Template] = {}
    for row in read_jsonl(path):
        raw_nodes = row.get("nodes") or []
        nodes_by_id: dict[str, Node] = {}
        for raw in raw_nodes:
            node_id = text(raw.get("node_id"), "")
            if not node_id or node_id in nodes_by_id:
                raise ValueError(f"duplicate/empty node_id in template {row.get('template_id')}")
            # fail closed: never let a causal template be consumed as legacy
            declared = row.get("topology_contract")
            if topology_view == "legacy" and declared == CAUSAL_TOPOLOGY_CONTRACT:
                raise ValueError(
                    "template %s declares topology_contract=%s but was loaded with "
                    "topology_view='legacy'; pass topology_view='causal_v3' so the "
                    "executor cannot silently run the step-expansion edges"
                    % (row.get("template_id"), declared)
                )
            if topology_view == "causal_v3":
                if "resource_applicable" not in raw:
                    raise ValueError(
                        "topology_view='causal_v3' but node %s in template %s carries no "
                        "resource_applicable; the node ontology must be part of the contract"
                        % (node_id, row.get("template_id"))
                    )
                if "causal_predecessor_node_ids" not in raw:
                    raise ValueError(
                        "topology_view='causal_v3' but node %s in template %s carries no "
                        "causal_predecessor_node_ids" % (node_id, row.get("template_id"))
                    )
                predecessors = tuple(str(value) for value in raw["causal_predecessor_node_ids"])
            else:
                predecessors = tuple(str(value) for value in raw.get("predecessor_node_ids") or [])
            nodes_by_id[node_id] = Node(
                node_id=node_id,
                sequence_index=int(raw.get("sequence_index") or 0),
                predecessors=predecessors,
                successors=(),
                lane=text(raw.get("execution_lane")),
                model_id=text(raw.get("model_id")),
                runtime_ms=max(0.0, number(raw.get("runtime_ms"))),
                load_ms=optional_number(raw.get("load_ms")),
                workspace_peak_mb=optional_number(raw.get("workspace_peak_mb")),
                resident_model_mb=optional_number(raw.get("resident_model_mb")),
                status=text(raw.get("status")),
                role=text(raw.get("node_type"), "other"),
                action_family=text(
                    raw.get("activity") if str(raw.get("activity") or "other") != "other" else raw.get("raw_action"),
                    "other",
                ),
                raw_action=text(raw.get("raw_action"), "other"),
                resource_applicable=bool(raw.get("resource_applicable", True)),
                nested_model_class=text(raw.get("nested_model_class"), ""),
                nested_model_mb=optional_number(raw.get("nested_model_mb")),
                nested_load_ms=optional_number(raw.get("nested_load_ms")),
                nested_runtime_ms=optional_number(raw.get("nested_runtime_ms")),
                nested_reserved_mb=optional_number(raw.get("nested_reserved_mb")),
                nested_pre_ms=optional_number(raw.get("nested_pre_ms")),
                nested_post_ms=optional_number(raw.get("nested_post_ms")),
                nested_inner_ms=optional_number(raw.get("nested_inner_ms")),
                batch_size=max(1, int(raw.get("batch_size") or raw.get("yolo_batch") or 1)),
            )
        successors: dict[str, list[str]] = defaultdict(list)
        for node in nodes_by_id.values():
            for predecessor in node.predecessors:
                if predecessor not in nodes_by_id:
                    raise ValueError(f"missing predecessor {predecessor} in template {row.get('template_id')}")
                successors[predecessor].append(node.node_id)
        final_nodes = {
            node_id: Node(**{**node.__dict__, "successors": tuple(sorted(successors.get(node_id, []), key=lambda value: nodes_by_id[value].sequence_index))})
            for node_id, node in nodes_by_id.items()
        }
        roots = [node for node in final_nodes.values() if not node.predecessors]
        if not roots:
            raise ValueError(f"template has no root: {row.get('template_id')}")
        result[str(row["template_id"])] = Template(
            template_id=str(row["template_id"]),
            video_id=text(row.get("video_id")),
            split=text(row.get("split")),
            baseline=text(row.get("baseline")),
            nodes=tuple(sorted(final_nodes.values(), key=lambda node: node.sequence_index)),
            by_id=final_nodes,
        )
    if not result:
        raise ValueError(f"no templates in {path}")
    return result


def train_resource_stats(templates: Mapping[str, Template]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for template in templates.values():
        if template.split != "train":
            continue
        for node in template.nodes:
            for key in resource_keys(node):
                groups[key]["runtime"].append(node.runtime_ms)
                if node.load_ms is not None and node.load_ms > 0.0:
                    groups[key]["load"].append(node.load_ms)
                if node.workspace_peak_mb is not None:
                    groups[key]["memory"].append(node.workspace_peak_mb)
    def upper_cvar(samples: list[float], alpha: float) -> float:
        """Mean of the worst (1-alpha) tail: the paper's CVaR term."""

        if not samples:
            return 0.0
        ordered = sorted(samples)
        k = max(1, int(round((1.0 - alpha) * len(ordered))))
        tail = ordered[-k:]
        return sum(tail) / len(tail)

    return {
        key: {
            "runtime_p50_ms": quantile(values["runtime"], 0.50),
            "runtime_p90_ms": quantile(values["runtime"], 0.90),
            "runtime_mean_ms": (sum(values["runtime"]) / len(values["runtime"])) if values["runtime"] else 0.0,
            "runtime_cvar90_ms": upper_cvar(values["runtime"], 0.90),
            "load_p50_ms": quantile(values["load"], 0.50),
            "memory_p95_mb": quantile(values["memory"], 0.95),
            "count": len(values["runtime"]),
        }
        for key, values in groups.items()
    }


def resource_keys(node: Node) -> tuple[str, ...]:
    return (
        "|".join((node.model_id, node.lane, str(node.sequence_index), "exact")),
        "|".join((node.model_id, node.lane, "*", "model_lane")),
        "|".join(("*", node.lane, "*", "lane")),
    )


def estimate(node: Node, stats: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    found = None
    for key in resource_keys(node):
        candidate = stats.get(key)
        if candidate and int(candidate.get("count") or 0) >= 3:
            found = candidate
            break
    if found is None:
        # A causal estimator may use a training global only when available.
        # Missing resource estimates are a gate failure, not a zero label.
        raise ValueError(f"no train-only estimate for node {node.node_id} model={node.model_id} lane={node.lane}")
    return {
        "runtime_p50_ms": float(found["runtime_p50_ms"]),
        "runtime_p90_ms": float(found["runtime_p90_ms"]),
        "load_p50_ms": float(found["load_p50_ms"] or 0.0),
        "memory_p95_mb": float(found["memory_p95_mb"] or 0.0),
        "uncertainty": "train_only_empirical_quantile",
    }


def _batchable(node: Node) -> bool:
    """Return whether a node is a measured per-tool batch configuration.

    The existing pilot measures YOLO's multi-frame batch inside one tool
    action.  It does not establish cross-job dynamic batching, so the opt-in
    extension only exposes nodes whose trace identity is explicitly YOLO.
    """

    return node.model_id.startswith("yolo") or node.raw_action == "yolo-tracker"


def _batch_profile(
    node: Node,
    batch_size: int,
    extension_config: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if not extension_config or not _batchable(node):
        return None
    profiles = extension_config.get("batch_profiles") or {}
    model_profiles = profiles.get(node.model_id) or profiles.get("yolo11x.pt") or {}
    return model_profiles.get(str(int(batch_size))) or model_profiles.get(int(batch_size))


def _batch_options(
    node: Node,
    extension_config: Mapping[str, Any] | None,
) -> tuple[int, ...]:
    if not extension_config or not bool(extension_config.get("batch_enabled")) or not _batchable(node):
        return (int(node.batch_size),)
    configured = extension_config.get("batch_options") or [int(node.batch_size)]
    options = tuple(sorted({max(1, int(value)) for value in configured}))
    return options or (int(node.batch_size),)


def estimate_for_batch(
    node: Node,
    stats: Mapping[str, Mapping[str, Any]],
    batch_size: int,
    extension_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Overlay a measured batch profile on the train-only scheduler estimate."""

    row = dict(estimate(node, stats))
    profile = _batch_profile(node, batch_size, extension_config)
    if profile is None:
        row["batch_size"] = int(batch_size)
        return row
    if profile.get("runtime_ms") is not None:
        runtime = max(0.0, number(profile.get("runtime_ms")))
        row["runtime_p50_ms"] = runtime
        row["runtime_p90_ms"] = max(runtime, number(profile.get("runtime_p90_ms"), runtime))
    if profile.get("load_ms") is not None:
        row["load_p50_ms"] = max(0.0, number(profile.get("load_ms")))
    if profile.get("peak_memory_mb") is not None:
        row["memory_p95_mb"] = max(0.0, number(profile.get("peak_memory_mb")))
    row["batch_size"] = int(batch_size)
    row["batch_profile_source"] = profile.get("source", "experiment_measured")
    return row


def effective_batch_runtime(
    node: Node,
    row: Mapping[str, Any],
    batch_size: int,
    extension_config: Mapping[str, Any] | None,
) -> tuple[float, float]:
    """Return execution runtime and load for the selected batch.

    If no profile is supplied, the frozen template truth is used exactly as in
    the legacy simulator.  Profiled batch values are an explicit experiment
    condition and are never exposed through ``SchedulerStateView``.
    """

    profile = _batch_profile(node, batch_size, extension_config)
    if profile is None:
        return node.runtime_ms, max(number(node.load_ms), float(row.get("load_p50_ms") or 0.0))
    runtime = max(0.1, number(profile.get("runtime_ms"), node.runtime_ms))
    load = max(0.0, number(profile.get("load_ms"), number(node.load_ms)))
    return runtime, load


def _transition_profile(extension_config: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not extension_config:
        return None
    raw = extension_config.get("transition_profile")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("transition_profile must be an object")
    if raw.get("enabled", True) is False:
        return None
    return raw


def validate_transition_profile(
    extension_config: Mapping[str, Any] | None,
    gpu_capacities: Sequence[float],
) -> None:
    profile = _transition_profile(extension_config)
    if profile is None:
        return
    models = profile.get("models")
    if not isinstance(models, Mapping) or not models:
        raise ValueError("transition_profile.models must be a non-empty object")
    expected_capacity = optional_number(profile.get("gpu_capacity_mb"))
    if expected_capacity is not None and any(
        abs(float(capacity) - expected_capacity) > 1e-6 for capacity in gpu_capacities
    ):
        raise ValueError(
            "transition profile GPU capacity mismatch: "
            f"expected {expected_capacity}, got {list(gpu_capacities)}"
        )
    for model_id, raw in models.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"transition profile entry must be an object: {model_id}")
        for field_name in ("cold_load_ms", "evict_proxy_ms"):
            if optional_number(raw.get(field_name)) is None:
                raise ValueError(f"transition profile {model_id} missing {field_name}")
        if raw.get("checkpoint_supported") is not False:
            raise ValueError(
                f"transition profile {model_id} must explicitly declare checkpoint_supported=false"
            )


def _transition_entry_for_model(
    model_id: str,
    extension_config: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    profile = _transition_profile(extension_config)
    if profile is None:
        return None
    models = profile.get("models") or {}
    entry = models.get(model_id)
    if entry is None and bool(profile.get("strict")):
        raise ValueError(f"transition profile has no strict entry for model {model_id}")
    return entry if isinstance(entry, Mapping) else None


def transition_load_cost(
    node: Node,
    estimate_row: Mapping[str, Any],
    batch_size: int,
    extension_config: Mapping[str, Any] | None,
) -> tuple[float, str]:
    """Resolve a cold load cost without changing the default simulator path."""

    batch_profile = _batch_profile(node, batch_size, extension_config)
    if batch_profile is not None and batch_profile.get("load_ms") is not None:
        return max(0.0, number(batch_profile.get("load_ms"))), "batch_profile"
    entry = _transition_entry_for_model(node.model_id, extension_config)
    if entry is not None:
        return max(0.0, number(entry.get("cold_load_ms"))), "transition_profile"
    return max(number(node.load_ms), number(estimate_row.get("load_p50_ms"))), "resource_estimate"


def transition_eviction_cost(
    model_id: str,
    extension_config: Mapping[str, Any] | None,
) -> tuple[float, str]:
    entry = _transition_entry_for_model(model_id, extension_config)
    if entry is None:
        return 0.0, "none"
    return max(0.0, number(entry.get("evict_proxy_ms"))), "transition_profile"


def model_memory(node: Node, estimate_row: Mapping[str, Any]) -> float:
    value = node.resident_model_mb
    if value is None:
        # The template gate rejects unknown GPU resident memory before formal
        # generation.  This error prevents silent zero replacement.
        if node.lane == "gpu":
            raise ValueError(f"unknown resident model memory for GPU node {node.node_id}")
        return 0.0
    return max(0.0, float(value))


def fits_gpu(gpu: GPU, node: Node, estimate_row: Mapping[str, Any]) -> bool:
    model_mb = model_memory(node, estimate_row)
    workspace_mb = max(0.0, float(estimate_row["memory_p95_mb"]) - model_mb)
    # Other resident models may be evicted before this node starts.  The
    # physical feasibility test is therefore this node's model plus its
    # workspace; cache reuse is a policy cost, not a capacity exemption.
    return model_mb + workspace_mb <= gpu.capacity_mb + 1e-9


def plan_gpu_admission(
    gpu: GPU,
    node: Node,
    estimate_row: Mapping[str, Any],
) -> tuple[bool, tuple[str, ...], float, float, float]:
    """Plan cache eviction and verify the full resident+workspace ledger.

    The plan is side-effect free.  Callers apply the returned evictions only
    after the admission succeeds, so an OOM decision cannot corrupt the cache.
    """
    model_mb = model_memory(node, estimate_row)
    workspace_mb = max(0.0, float(estimate_row["memory_p95_mb"]) - model_mb)
    target_resident = node.model_id in gpu.resident
    required_model_mb = 0.0 if target_resident else model_mb
    projected = sum(gpu.resident.values()) + required_model_mb + workspace_mb
    if projected <= gpu.capacity_mb + 1e-9:
        return True, (), model_mb, workspace_mb, projected

    evicted = tuple(sorted(model_id for model_id in gpu.resident if model_id != node.model_id))
    remaining = sum(
        memory
        for model_id, memory in gpu.resident.items()
        if model_id not in evicted
    )
    projected_after_eviction = remaining + required_model_mb + workspace_mb
    if projected_after_eviction <= gpu.capacity_mb + 1e-9:
        return True, evicted, model_mb, workspace_mb, projected_after_eviction
    return False, (), model_mb, workspace_mb, projected_after_eviction


def roots(template: Template) -> list[str]:
    return [node.node_id for node in template.nodes if not node.predecessors]


def build_jobs(episode: Mapping[str, Any], templates: Mapping[str, Template]) -> list[Job]:
    jobs: list[Job] = []
    for raw in episode.get("jobs") or []:
        template_id = str(raw["template_id"])
        if template_id not in templates:
            raise ValueError(f"episode references unknown template {template_id}")
        template = templates[template_id]
        if template.split != str(episode.get("split")):
            raise ValueError(f"episode split leak: {episode.get('episode_id')} -> {template_id}")
        jobs.append(Job(
            job_instance_id=str(raw["job_instance_id"]),
            template=template,
            arrival_ms=number(raw.get("arrival_ms")),
            deadline_ms=optional_number(raw.get("deadline_ms")),
            service_class=text(raw.get("service_class"), "normal"),
            node_state={node.node_id: "pending" for node in template.nodes},
        ))
    return jobs


def ready_nodes(job: Job, now: float) -> list[str]:
    if now + 1e-9 < job.arrival_ms:
        return []
    ready: list[str] = []
    for node in job.template.nodes:
        if job.node_state.get(node.node_id) != "pending":
            continue
        if all(job.node_state.get(predecessor) == "complete" for predecessor in node.predecessors):
            ready.append(node.node_id)
    return ready


def future_truth_cost(
    job: Job,
    node_id: str,
    gpu: GPU,
    train_stats: Mapping[str, Mapping[str, Any]],
) -> float:
    pending_ids: set[str] = set()
    frontier = [node_id]
    while frontier:
        current_id = frontier.pop()
        if current_id in pending_ids:
            continue
        pending_ids.add(current_id)
        frontier.extend(job.template.by_id[current_id].successors)
    resident = dict(gpu.resident)
    total = 0.0
    ordered_nodes = sorted(
        (job.template.by_id[current_id] for current_id in pending_ids),
        key=lambda candidate: candidate.sequence_index,
    )
    for candidate in ordered_nodes:
        if candidate.lane == "gpu":
            estimate_row = estimate(candidate, train_stats)
            memory = model_memory(candidate, estimate_row)
            workspace = max(0.0, float(estimate_row["memory_p95_mb"]) - memory)
            if candidate.model_id not in resident:
                if sum(resident.values()) + memory + workspace > gpu.capacity_mb:
                    resident.clear()
                resident[candidate.model_id] = memory
                total += number(candidate.load_ms)
            total += candidate.compute_ms
        else:
            total += candidate.runtime_ms
    return total


def limited_future_table_cost(
    job: Job,
    node_id: str,
    gpu: GPU,
    train_stats: Mapping[str, Mapping[str, Any]],
    horizon: int,
) -> float:
    """Truth-topology walk with train-table resource values (gap decomposition arm).

    Identical structure to ``limited_future_truth_cost`` but each successor is
    priced by the identity lookup instead of execution truth, so comparing the
    two isolates the resource-truth contribution while topology stays oracle.
    """

    frontier = list(job.template.by_id[node_id].successors)
    seen: set[str] = set()
    resident: dict[str, float] = dict(gpu.resident)
    total = 0.0
    for _ in range(max(0, int(horizon))):
        next_frontier: list[str] = []
        for current_id in frontier:
            if current_id in seen:
                continue
            seen.add(current_id)
            candidate = job.template.by_id[current_id]
            next_frontier.extend(candidate.successors)
            estimate_row = estimate(candidate, train_stats)
            runtime = number(estimate_row.get("runtime_p50_ms"), 0.0)
            if candidate.lane == "gpu":
                load = number(estimate_row.get("load_p50_ms"), 0.0)
                memory = model_memory(candidate, estimate_row)
                workspace = max(0.0, float(estimate_row["memory_p95_mb"]) - memory)
                cached = candidate.model_id in resident
                if not cached:
                    if sum(resident.values()) + memory + workspace > gpu.capacity_mb:
                        resident.clear()
                    resident[candidate.model_id] = memory
                total += runtime - load + (0.0 if cached else load)
            else:
                total += runtime
        frontier = next_frontier
        if not frontier:
            break
    return total


def limited_future_truth_cost(
    job: Job,
    node_id: str,
    gpu: GPU,
    train_stats: Mapping[str, Mapping[str, Any]],
    horizon: int,
) -> float:
    """Cost of only the next ``horizon`` DAG layers using execution truth.

    This is the TrueOpt-H reference: it may inspect the actual suffix, but it
    never exposes that suffix to the predicted policy or its scheduler state.
    """
    frontier = list(job.template.by_id[node_id].successors)
    seen: set[str] = set()
    resident = dict(gpu.resident)
    total = 0.0
    for _ in range(max(0, int(horizon))):
        next_frontier: list[str] = []
        for current_id in frontier:
            if current_id in seen:
                continue
            seen.add(current_id)
            candidate = job.template.by_id[current_id]
            next_frontier.extend(candidate.successors)
            if candidate.lane == "gpu":
                estimate_row = estimate(candidate, train_stats)
                memory = model_memory(candidate, estimate_row)
                workspace = max(0.0, float(estimate_row["memory_p95_mb"]) - memory)
                if candidate.model_id not in resident:
                    if sum(resident.values()) + memory + workspace > gpu.capacity_mb:
                        resident.clear()
                    resident[candidate.model_id] = memory
                    total += number(candidate.load_ms)
                total += candidate.compute_ms
            else:
                total += candidate.runtime_ms
        frontier = next_frontier
        if not frontier:
            break
    return total


_ARTIFACT_CONTEXT = ""
_STATS_CONTEXT = ""
_LOOKUP_PATH_COUNTS: Counter[str] = Counter()

# Frozen on dev700 nodes (experiments/EXP-20260911_forecast_aware_scheduling/phase11_r50_scale.json:
# mean sum_p95 / mean sum_p50 = 6.2293); used by the scale-matched p50 control arm.
_RUNTIME_P50_SCALE = 6.2293


def set_artifact_context(value: str) -> None:
    """Tag the active future-artifact pack so cost caches cannot leak across packs."""
    global _ARTIFACT_CONTEXT
    _ARTIFACT_CONTEXT = str(value)


def set_train_stats_context(value: str) -> None:
    global _STATS_CONTEXT
    _STATS_CONTEXT = str(value)


# --------------------------------------------------------------------------- #
# Decision tracing (opt-in, shadow-only)
#
# When a trace is registered the same-shape branch additionally scores the whole
# candidate pool with every registered shadow method and writes one record per
# decision state.  The live decision is unchanged: the trace only observes.
_DECISION_TRACE: dict[str, Any] = {
    "writer": None, "methods": (), "truth": None, "trajectory_method_id": None,
}


def set_decision_trace(
    writer: Any = None,
    methods: Sequence[Any] = (),
    truth_cost: Any = None,
    trajectory_method_id: str | None = None,
) -> None:
    """Register (or clear) the decision tracer for the current process.

    `trajectory_method_id` names the shadow method that mirrors the live
    consumer; every decision asserts that its reconstructed winner equals the
    candidate the simulator actually chose.
    """

    _DECISION_TRACE["writer"] = writer
    _DECISION_TRACE["methods"] = tuple(methods)
    _DECISION_TRACE["truth"] = truth_cost
    _DECISION_TRACE["trajectory_method_id"] = trajectory_method_id


def decision_trace_active() -> bool:
    return _DECISION_TRACE["writer"] is not None


def lookup_path_report() -> dict[str, int]:
    return dict(_LOOKUP_PATH_COUNTS)


_STEP_COST_CACHE: dict[tuple[str, str, str], float] = {}


def _hierarchical_train_row(
    model_id: str,
    lane: str,
    train_stats: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """model_lane -> lane -> global fallback, mirroring ``estimate()``.

    The previous prefix scan mixed exact / model_lane / lane rows and took a
    median of medians; this walks one granularity at a time and records which
    path was used so the run report can expose fallback rates.
    """

    if model_id and model_id != "unknown":
        row = train_stats.get(f"{model_id}|{lane}|*|model_lane")
        if row is not None and int(row.get("count") or 0) >= 3:
            _LOOKUP_PATH_COUNTS["model_lane"] += 1
            return row
    row = train_stats.get(f"*|{lane}|*|lane")
    if row is not None and int(row.get("count") or 0) >= 3:
        _LOOKUP_PATH_COUNTS["lane"] += 1
        return row
    if train_stats:
        runtimes = [number(candidate.get("runtime_p50_ms"), 1.0) for candidate in train_stats.values()]
        loads = [number(candidate.get("load_p50_ms"), 0.0) for candidate in train_stats.values()]
        _LOOKUP_PATH_COUNTS["global"] += 1
        return {
            "runtime_p50_ms": statistics.median(runtimes),
            "load_p50_ms": statistics.median(loads),
            "count": len(train_stats),
        }
    _LOOKUP_PATH_COUNTS["missing"] += 1
    return None


def _step_estimate_cost(step: Mapping[str, Any], train_stats: Mapping[str, Mapping[str, Any]]) -> float:
    """Translate a predicted identity-only step into train-only cost."""
    model_id = text(step.get("model_id"), "unknown")
    lane = text(step.get("execution_lane"), "unknown")
    cache_key = (_STATS_CONTEXT, model_id, lane)
    cached = _STEP_COST_CACHE.get(cache_key)
    if cached is not None:
        return cached
    row = _hierarchical_train_row(model_id, lane, train_stats)
    if row is None:
        result = 1.0
    else:
        result = number(row.get("runtime_p50_ms"), 1.0) + number(row.get("load_p50_ms"), 0.0)
    _STEP_COST_CACHE[cache_key] = result
    return result


def predicted_future_cost(
    node_id: str,
    future_artifacts: Mapping[str, Mapping[str, Any]],
    train_stats: Mapping[str, Mapping[str, Any]],
    horizon: int,
) -> float:
    row = future_artifacts.get(str(node_id)) or {}
    cached = row.get(f"_cost_h{int(horizon)}")
    if cached is not None:
        return number(cached, 0.0)
    scenarios = row.get(f"future_h{int(horizon)}") or []
    if not scenarios:
        return 0.0
    return sum(
        number(scenario.get("scenario_probability"), 0.0)
        * sum(_step_estimate_cost(step, train_stats) for step in scenario.get("steps") or [])
        for scenario in scenarios
    )


_JRES_COST_CACHE: dict[tuple[str, str, str, int, str], float] = {}


def _mix_step_cost(step: Mapping[str, Any], train_stats: Mapping[str, Mapping[str, Any]], lam: float) -> float:
    """Risk-mixed step cost: (1-lam)*p50 + lam*p90 (load mixed the same way at occ>=0.5)."""

    resource = step.get("resource") or {}
    quantiles = resource.get("runtime_ms_quantiles") or {}
    p50 = quantiles.get("p50")
    p90 = quantiles.get("p90")
    if isinstance(p50, (int, float)) and isinstance(p90, (int, float)) and float(p50) > 0.0:
        runtime = (1.0 - lam) * float(p50) + lam * float(p90)
        lane = text(step.get("execution_lane"), "unknown")
        load = 0.0
        if lane == "gpu":
            occurrence = resource.get("load_occurrence_probability")
            duration = resource.get("load_duration_ms_quantiles") or {}
            d50 = duration.get("p50")
            d90 = duration.get("p90")
            if isinstance(occurrence, (int, float)) and float(occurrence) >= 0.5 and isinstance(d50, (int, float)) and isinstance(d90, (int, float)):
                load = (1.0 - lam) * float(d50) + lam * float(d90)
        return runtime + load
    return _step_estimate_cost(step, train_stats)


def _legacy_p95_load_cost(step: Mapping[str, Any]) -> float:
    """The frozen load surcharge: conditional p95 load on GPU steps at occ>=0.5.

    Extracted verbatim from the original ``_q95_step_cost`` so that every
    same-shape consumer shares one load rule.  The resource-v2 arms must not
    change the load semantics at the same time as the runtime functional, or the
    measured consumer effect would be confounded.
    """

    lane = text(step.get("execution_lane"), "unknown")
    if lane != "gpu":
        return 0.0
    resource = step.get("resource") or {}
    occurrence = resource.get("load_occurrence_probability")
    duration = (resource.get("load_duration_ms_quantiles") or {}).get("p95")
    if isinstance(occurrence, (int, float)) and isinstance(duration, (int, float)) and float(occurrence) >= 0.5:
        return max(0.0, float(duration))
    return 0.0


def _q95_step_cost(step: Mapping[str, Any], train_stats: Mapping[str, Mapping[str, Any]]) -> float:
    """Legacy risk score: predicted p95 runtime + the legacy p95 load surcharge.

    This is the frozen champion consumer, not a CVaR and not runtime-only: the
    load term is part of the definition and must stay that way.  A missing p95
    falls back to the statistics estimate, which is why the resource-v2 arms are
    deliberately fail-closed instead.
    """

    resource = step.get("resource") or {}
    runtime = (resource.get("runtime_ms_quantiles") or {}).get("p95")
    if not (isinstance(runtime, (int, float)) and float(runtime) > 0.0):
        return _step_estimate_cost(step, train_stats)
    return float(runtime) + _legacy_p95_load_cost(step)


def _required_runtime_field(step: Mapping[str, Any], key: str, stat: str) -> float:
    """Fail-closed read of a resource-v2 field.

    The legacy consumer silently falls back to ``_step_estimate_cost`` when a
    quantile is missing; an experimental arm must not do that, because a single
    missing field would quietly turn it into a different consumer.  NaN and
    negative values are errors too: they cannot be distinguished from a real
    measurement once they enter a sum.
    """

    resource = step.get("resource") or {}
    value = resource.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{stat} consumer requires resource.{key}; got {value!r}")
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{stat} consumer requires a finite resource.{key}; got {value!r}")
    if value < 0.0:
        raise ValueError(f"{stat} consumer requires a non-negative resource.{key}; got {value!r}")
    return value


def _mean_step_cost(step: Mapping[str, Any], train_stats: Mapping[str, Mapping[str, Any]]) -> float:
    """Conditional-mean step cost: E[T | active] + the legacy p95 load surcharge."""

    return _required_runtime_field(step, "runtime_mean_ms", "condmean") + _legacy_p95_load_cost(step)


def _cvar95_step_cost(step: Mapping[str, Any], train_stats: Mapping[str, Mapping[str, Any]]) -> float:
    """Marginal step CVaR95 of the discretised distribution + the legacy load.

    This is ``sum_h CVaR_0.95(T_h)``, a sum of per-step marginal risk scores.  It
    is NOT ``CVaR_0.95(sum_h T_h)``: the artifact carries per-step marginals only,
    so no joint over the horizon exists and the total-runtime CVaR is simply not
    identified.  Papers and reports must use the former phrasing.
    """

    return _required_runtime_field(step, "cvar95_ms", "stepcvar95") + _legacy_p95_load_cost(step)


# Truth-same-consumer arms: identical current cost, chain, structure and key
# order (priority, current + future, future, ...); only the future statistic
# differs. ``sameshape_h5_p95`` is by construction the same consumer as
# ``predopt_h5_q95`` and is asserted equal to it in the unit tests.
SAMESHAPE_POLICIES: tuple[str, ...] = (
    "sameshape_h5_p50",
    "sameshape_h5_p95",
    "sameshape_h5_truth",
    # resource-v2 consumption functionals; same key shape and same load rule, only
    # the runtime functional differs from sameshape_h5_p95
    "sameshape_h5_condmean",
    "sameshape_h5_stepcvar95",
)
_SAMESHAPE_PREDICTED_STATS = ("p50", "p95", "condmean", "stepcvar95")


def sameshape_future_cost(
    node_id: str,
    future_artifacts: Mapping[str, Mapping[str, Any]],
    train_stats: Mapping[str, Mapping[str, Any]],
    horizon: int,
    stat: str,
) -> float:
    """Sum one per-step statistic over the frozen predicted chain (p50 or p95).

    The ``truth`` variant of this consumer cannot use this helper: the frozen
    artifacts carry identity-only future steps, so there is no per-step ground
    truth to attach to them. It uses ``limited_future_truth_cost`` on the real
    successor walk instead, keeping the same key shape.
    """

    if stat not in _SAMESHAPE_PREDICTED_STATS:
        raise ValueError(f"unknown same-shape future statistic: {stat!r}")
    row = future_artifacts.get(str(node_id)) or {}
    scenarios = row.get(f"future_h{int(horizon)}") or []
    steps = (scenarios[0].get("steps") or []) if scenarios else []
    window = steps[: int(horizon)]
    if stat == "p50":
        return sum(_mix_step_cost(step, train_stats, 0.0) for step in window)
    if stat == "p95":
        return sum(_q95_step_cost(step, train_stats) for step in window)
    if stat == "condmean":
        return sum(_mean_step_cost(step, train_stats) for step in window)
    return sum(_cvar95_step_cost(step, train_stats) for step in window)


def _risk_step_cost(step: Mapping[str, Any], train_stats: Mapping[str, Mapping[str, Any]]) -> float:
    """Risk-sensitive step cost: predicted p90 runtime (+ conditional p90 load at occ>=0.5)."""

    resource = step.get("resource") or {}
    runtime = (resource.get("runtime_ms_quantiles") or {}).get("p90")
    if not (isinstance(runtime, (int, float)) and float(runtime) > 0.0):
        return _step_estimate_cost(step, train_stats)
    lane = text(step.get("execution_lane"), "unknown")
    load = 0.0
    if lane == "gpu":
        occurrence = resource.get("load_occurrence_probability")
        duration = (resource.get("load_duration_ms_quantiles") or {}).get("p90")
        if isinstance(occurrence, (int, float)) and isinstance(duration, (int, float)) and float(occurrence) >= 0.5:
            load = max(0.0, float(duration))
    return float(runtime) + load


def _split95_step_cost(
    step: Mapping[str, Any],
    train_stats: Mapping[str, Mapping[str, Any]],
    runtime_lam: float,
    load_lam: float,
) -> float:
    """Component split: runtime at p50+lam*(p95-p50), load at p50+lam*(p95-p50)."""

    resource = step.get("resource") or {}
    quantiles = resource.get("runtime_ms_quantiles") or {}
    p50 = quantiles.get("p50")
    p95 = quantiles.get("p95")
    if isinstance(p50, (int, float)) and isinstance(p95, (int, float)) and float(p95) > 0.0:
        runtime = float(p50) + runtime_lam * (float(p95) - float(p50))
        load = 0.0
        if text(step.get("execution_lane"), "unknown") == "gpu":
            occurrence = resource.get("load_occurrence_probability")
            duration = resource.get("load_duration_ms_quantiles") or {}
            d50 = duration.get("p50")
            d95 = duration.get("p95")
            if isinstance(occurrence, (int, float)) and float(occurrence) >= 0.5 and isinstance(d50, (int, float)) and isinstance(d95, (int, float)):
                load = float(d50) + load_lam * (float(d95) - float(d50))
        return runtime + load
    return _step_estimate_cost(step, train_stats)


def _runtime_only_step_cost(
    step: Mapping[str, Any],
    train_stats: Mapping[str, Mapping[str, Any]],
    tau_key: str = "p95",
) -> float:
    """Runtime-only step cost: predicted runtime quantile, no future load.

    The load block is intentionally excluded so the orthogonal consumption
    family isolates the runtime statistic (p50/p90/p95) without the
    table-vs-predicted load-semantics confound.
    """

    resource = step.get("resource") or {}
    value = (resource.get("runtime_ms_quantiles") or {}).get(tau_key)
    if isinstance(value, (int, float)) and float(value) > 0.0:
        return float(value)
    runtime, _load = _table_step_components(step, train_stats)
    return runtime


_SCEN2_COST_CACHE: dict[tuple[str, str, str, int, int, str, str], list[float]] = {}


def _lognormal_params(quantiles: Mapping[str, Any]) -> tuple[float, float] | None:
    """Fit a lognormal (mu, sigma) to predicted p50/p90/p95 with clamped sigma.

    sigma is chosen to match the p95 (the statistic the scheduler consumes),
    falling back to the p90 ratio when p95 is degenerate; sigma is clamped to
    [0.05, 1.6] so a degenerate row cannot produce explosive samples.  Unlike
    the earlier piecewise-linear reconstruction this imposes no artificial
    Q(0)=0 floor and no hand-tuned tail extension.
    """

    p50 = number(quantiles.get("p50"), 0.0)
    p90 = number(quantiles.get("p90"), 0.0)
    p95 = number(quantiles.get("p95"), 0.0)
    if p50 <= 0.0:
        return None
    sigma = None
    if p95 > p50:
        sigma = math.log(p95 / p50) / 1.6448536269514722
    elif p90 > p50:
        sigma = math.log(p90 / p50) / 1.2815515655446004
    if sigma is None or not math.isfinite(sigma) or sigma <= 0.0:
        return None
    return math.log(p50), min(max(sigma, 0.05), 1.6)


def _sample_lognormal(rng: random.Random, mu: float, sigma: float, u: float | None = None) -> float:
    if u is None:
        z = rng.gauss(0.0, 1.0)
    else:
        z = statistics.NormalDist().inv_cdf(min(max(u, 1e-6), 1.0 - 1e-6))
    return math.exp(mu + sigma * z)


def _scenario_v2_costs(
    node_id: str,
    future_artifacts: Mapping[str, Mapping[str, Any]],
    train_stats: Mapping[str, Mapping[str, Any]],
    horizon: int = 5,
    samples: int = 128,
    coupling: str = "independent",
    length_mode: str = "sample",
) -> list[float]:
    """Joint scenario sampling with real length draws and lognormal reconstruction.

    Draws a future length (``length_mode='sample'``: from ``length_probabilities``
    over the full emitted chain; ``length_mode='argmax'``: the fixed predicted
    length, the shape-matched control) then samples each step's runtime and
    (conditionally) load duration from lognormal fits.  ``coupling='comonotone'``
    shares one uniform draw per component across the scenario.
    """

    key = (
        _ARTIFACT_CONTEXT,
        _STATS_CONTEXT,
        str(node_id),
        int(horizon),
        int(samples),
        str(coupling),
        str(length_mode),
    )
    cached = _SCEN2_COST_CACHE.get(key)
    if cached is not None:
        return cached
    row = future_artifacts.get(str(node_id)) or {}
    scenarios = row.get(f"future_h{horizon}_full") or row.get(f"future_h{horizon}") or []
    steps = (scenarios[0].get("steps") or []) if scenarios else []
    length_p = row.get("length_probabilities") or []
    rng = random.Random(_stable_seed(str(node_id), str(horizon), str(samples), str(coupling), str(length_mode), "v2"))
    costs: list[float] = []
    for _ in range(samples):
        if length_mode == "argmax":
            sampled_length = int(number(row.get("predicted_future_length"), len(steps)))
        elif len(length_p) >= 1:
            draw = rng.random()
            acc = 0.0
            sampled_length = len(length_p) - 1
            for idx, prob in enumerate(length_p):
                acc += float(prob)
                if draw <= acc:
                    sampled_length = idx
                    break
        else:
            sampled_length = len(steps)
        length = max(0, min(int(sampled_length), len(steps)))
        comonotone = coupling == "comonotone"
        u_runtime = rng.random() if comonotone else None
        u_occurrence = rng.random() if comonotone else None
        u_load = rng.random() if comonotone else None
        total = 0.0
        for step in steps[:length]:
            resource = step.get("resource") or {}
            params = _lognormal_params(resource.get("runtime_ms_quantiles") or {})
            if params is None:
                total += _runtime_only_step_cost(step, train_stats, "p50")
            else:
                total += _sample_lognormal(rng, params[0], params[1], u_runtime)
            if text(step.get("execution_lane"), "unknown") == "gpu":
                occurrence = resource.get("load_occurrence_probability")
                if isinstance(occurrence, (int, float)):
                    if u_occurrence is not None:
                        fires = u_occurrence > 1.0 - float(occurrence)
                    else:
                        fires = rng.random() < float(occurrence)
                    if fires:
                        load_params = _lognormal_params(resource.get("load_duration_ms_quantiles") or {})
                        if load_params is not None:
                            total += _sample_lognormal(rng, load_params[0], load_params[1], u_load)
        costs.append(total)
    _SCEN2_COST_CACHE[key] = costs
    return costs


def scenario_v2_risk_cost(
    node_id: str,
    future_artifacts: Mapping[str, Mapping[str, Any]],
    train_stats: Mapping[str, Mapping[str, Any]],
    kappa: float,
    horizon: int = 5,
    samples: int = 128,
    alpha: float = 0.9,
    coupling: str = "independent",
    length_mode: str = "sample",
) -> float:
    """Convex risk blend (1-kappa) * E[C] + kappa * CVaR_alpha(C)."""

    costs = _scenario_v2_costs(
        node_id, future_artifacts, train_stats, horizon, samples, coupling, length_mode
    )
    if not costs:
        return 0.0
    mean = statistics.fmean(costs)
    if kappa <= 0.0:
        return float(mean)
    ordered = sorted(costs, reverse=True)
    tail_n = max(1, int(round((1.0 - alpha) * len(ordered))))
    cvar = statistics.fmean(ordered[:tail_n])
    return float((1.0 - kappa) * mean + kappa * cvar)


_SCEN_COST_CACHE: dict[tuple[str, str, str, int, int, str], list[float]] = {}


def _mix95_step_cost(step: Mapping[str, Any], train_stats: Mapping[str, Mapping[str, Any]], lam: float) -> float:
    """Interpolate between predicted p50 and p95: p50 + lam*(p95-p50)."""

    resource = step.get("resource") or {}
    quantiles = resource.get("runtime_ms_quantiles") or {}
    p50 = quantiles.get("p50")
    p95 = quantiles.get("p95")
    if isinstance(p50, (int, float)) and isinstance(p95, (int, float)) and float(p95) > 0.0:
        runtime = float(p50) + lam * (float(p95) - float(p50))
        lane = text(step.get("execution_lane"), "unknown")
        load = 0.0
        if lane == "gpu":
            occurrence = resource.get("load_occurrence_probability")
            duration = resource.get("load_duration_ms_quantiles") or {}
            d50 = duration.get("p50")
            d95 = duration.get("p95")
            if isinstance(occurrence, (int, float)) and float(occurrence) >= 0.5 and isinstance(d50, (int, float)) and isinstance(d95, (int, float)):
                load = float(d50) + lam * (float(d95) - float(d50))
        return runtime + load
    return _step_estimate_cost(step, train_stats)


def _stable_seed(*parts: str) -> int:
    return zlib.crc32("|".join(parts).encode("utf-8")) & 0xFFFFFFFF


def _value_at_quantile(u: float, quantiles: Mapping[str, Any]) -> float:
    """Bounded inverse CDF through predicted p50/p90/p95 with capped tail extension."""

    p50 = number(quantiles.get("p50"), 0.0)
    p90 = number(quantiles.get("p90"), 0.0)
    p95 = number(quantiles.get("p95"), 0.0)
    if p95 <= 0.0:
        return 0.0
    if p50 <= 0.0:
        p50 = p95 / 4.0
    p90 = max(p90, p50)
    p95 = max(p95, p90)
    u = min(max(u, 0.0), 1.0)
    if u <= 0.5:
        return p50 * (u / 0.5)
    if u <= 0.9:
        return p50 + (p90 - p50) * (u - 0.5) / 0.4
    if u <= 0.95:
        return p90 + (p95 - p90) * (u - 0.9) / 0.05
    return p95 + 9.0 * (p95 - p90) * (u - 0.95) / 0.05


def _sample_from_quantiles(rng: random.Random, quantiles: Mapping[str, Any]) -> float:
    return _value_at_quantile(rng.random(), quantiles)


def _scenario_future_costs(
    node_id: str,
    future_artifacts: Mapping[str, Mapping[str, Any]],
    train_stats: Mapping[str, Mapping[str, Any]],
    horizon: int = 5,
    samples: int = 32,
    coupling: str = "independent",
) -> list[float]:
    """Sample joint future-cost scenarios from predicted length/model/resource heads.

    ``independent`` draws a fresh quantile level per step (steps partially
    cancel); ``comonotone`` reuses one draw across the steps of a scenario,
    which keeps the correlation that the per-step p95 sum implicitly assumes.
    """

    key = (_ARTIFACT_CONTEXT, _STATS_CONTEXT, str(node_id), int(horizon), int(samples), str(coupling))
    cached = _SCEN_COST_CACHE.get(key)
    if cached is not None:
        return cached
    row = future_artifacts.get(str(node_id)) or {}
    scenarios = row.get(f"future_h{horizon}") or []
    steps = (scenarios[0].get("steps") or []) if scenarios else []
    length_p = row.get("length_probabilities") or []
    rng = random.Random(_stable_seed(str(node_id), str(horizon), str(samples), str(coupling)))
    costs: list[float] = []
    for _ in range(samples):
        if len(length_p) >= 2:
            draw = rng.random()
            acc = 0.0
            sampled_length = len(length_p) - 1
            for idx, prob in enumerate(length_p):
                acc += float(prob)
                if draw <= acc:
                    sampled_length = idx
                    break
        else:
            sampled_length = len(steps)
        length = min(int(sampled_length), len(steps))
        common_u = rng.random() if coupling == "comonotone" else None
        total = 0.0
        for step in steps[:length]:
            resource = step.get("resource") or {}
            quantiles = resource.get("runtime_ms_quantiles") or {}
            if number((quantiles.get("p95")), 0.0) > 0.0:
                u = common_u if common_u is not None else rng.random()
                total += _value_at_quantile(u, quantiles)
            else:
                total += _step_estimate_cost(step, train_stats)
            if text(step.get("execution_lane"), "unknown") == "gpu":
                occurrence = resource.get("load_occurrence_probability")
                duration = resource.get("load_duration_ms_quantiles") or {}
                if isinstance(occurrence, (int, float)) and rng.random() < float(occurrence):
                    if number((duration.get("p95")), 0.0) > 0.0:
                        total += _sample_from_quantiles(rng, duration)
        costs.append(total)
    _SCEN_COST_CACHE[key] = costs
    return costs


def scenario_cvar_future_cost(
    node_id: str,
    future_artifacts: Mapping[str, Mapping[str, Any]],
    train_stats: Mapping[str, Mapping[str, Any]],
    kappa: float,
    horizon: int = 5,
    samples: int = 32,
    alpha: float = 0.9,
    coupling: str = "independent",
) -> float:
    """Consume the predicted joint future distribution as E[F] + kappa * CVaR_alpha(F)."""

    costs = _scenario_future_costs(node_id, future_artifacts, train_stats, horizon, samples, coupling)
    mean = statistics.fmean(costs) if costs else 0.0
    if kappa <= 0.0 or not costs:
        return float(mean)
    ordered = sorted(costs, reverse=True)
    tail_n = max(1, int(round((1.0 - alpha) * len(ordered))))
    cvar = statistics.fmean(ordered[:tail_n])
    return float(mean + kappa * cvar)


def _table_step_components(step: Mapping[str, Any], train_stats: Mapping[str, Mapping[str, Any]]) -> tuple[float, float]:
    """Return (runtime_p50, load_p50) from the hierarchical train-only lookup."""

    model_id = text(step.get("model_id"), "unknown")
    lane = text(step.get("execution_lane"), "unknown")
    row = _hierarchical_train_row(model_id, lane, train_stats)
    if row is None:
        return 1.0, 0.0
    return number(row.get("runtime_p50_ms"), 1.0), number(row.get("load_p50_ms"), 0.0)


def _jres_step_cost(
    step: Mapping[str, Any],
    train_stats: Mapping[str, Mapping[str, Any]],
    load_threshold: float = 0.5,
) -> float:
    """Step cost using the J resource block with decision-compatible semantics.

    Runtime: predicted p50 when present.  Load: the predicted *conditional*
    duration (no occurrence discount) only for GPU steps whose predicted load
    occurrence is at least ``load_threshold``; other steps contribute no load.
    The previous expected-load form (occ * duration) systematically discounted
    rare-load steps and made the policy behave myopically.
    """

    resource = step.get("resource") or {}
    lane = text(step.get("execution_lane"), "unknown")
    runtime = (resource.get("runtime_ms_quantiles") or {}).get("p50")
    if not (isinstance(runtime, (int, float)) and float(runtime) > 0.0):
        table_runtime, table_load = _table_step_components(step, train_stats)
        return table_runtime + table_load
    runtime = float(runtime)
    load = 0.0
    if lane == "gpu":
        occurrence = resource.get("load_occurrence_probability")
        duration = (resource.get("load_duration_ms_quantiles") or {}).get("p50")
        if isinstance(occurrence, (int, float)) and isinstance(duration, (int, float)) and float(occurrence) >= load_threshold:
            load = max(0.0, float(duration))
    return runtime + load


def _jrt_step_cost(step: Mapping[str, Any], train_stats: Mapping[str, Mapping[str, Any]]) -> float:
    """Runtime-only swap: predicted runtime p50 + the table's load semantics."""

    resource = step.get("resource") or {}
    runtime = (resource.get("runtime_ms_quantiles") or {}).get("p50")
    table_runtime, table_load = _table_step_components(step, train_stats)
    if isinstance(runtime, (int, float)) and float(runtime) > 0.0:
        return float(runtime) + table_load
    return table_runtime + table_load


def predicted_future_cost_jres(
    node_id: str,
    future_artifacts: Mapping[str, Mapping[str, Any]],
    train_stats: Mapping[str, Mapping[str, Any]],
    horizon: int,
    step_cost: Any = None,
) -> float:
    """H-horizon predicted cost using the J-predictor resource block."""

    cost_fn = step_cost or _jres_step_cost
    row = future_artifacts.get(str(node_id)) or {}
    cost_label = getattr(cost_fn, "__name__", None) or type(cost_fn).__name__
    cache_key = (_ARTIFACT_CONTEXT, _STATS_CONTEXT, str(node_id), int(horizon), cost_label)
    if cache_key in _JRES_COST_CACHE:
        return _JRES_COST_CACHE[cache_key]
    scenarios = row.get(f"future_h{int(horizon)}") or []
    if not scenarios:
        return 0.0
    total = sum(
        number(scenario.get("scenario_probability"), 0.0)
        * sum(cost_fn(step, train_stats) for step in scenario.get("steps") or [])
        for scenario in scenarios
    )
    _JRES_COST_CACHE[cache_key] = total
    return total


def model_memory_by_model(templates: Iterable[Template]) -> dict[str, float]:
    """Return the frozen resident-memory value for each model identity."""

    result: dict[str, float] = {}
    for template in templates:
        for node in template.nodes:
            if node.lane != "gpu" or node.resident_model_mb is None:
                continue
            previous = result.get(node.model_id)
            memory = float(node.resident_model_mb)
            if previous is not None and abs(previous - memory) > 1e-9:
                raise ValueError(f"inconsistent resident memory for model {node.model_id}")
            result[node.model_id] = memory
    return result


def _predicted_step_components(
    step: Mapping[str, Any],
    train_stats: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str, float, float, float]:
    """Resolve scheduler-visible p50 runtime/load and p95 workspace estimates."""

    model_id = text(step.get("model_id"), "unknown")
    lane = text(step.get("execution_lane"), "unknown")
    candidates = [
        row
        for key, row in train_stats.items()
        if (model_id != "unknown" and key.startswith(f"{model_id}|{lane}|"))
        or key.startswith(f"*|{lane}|")
    ]
    if not candidates:
        candidates = list(train_stats.values())
    if not candidates:
        raise ValueError("cannot estimate predicted future step without train-only resource statistics")
    runtimes = [number(row.get("runtime_p50_ms"), 1.0) for row in candidates]
    loads = [number(row.get("load_p50_ms"), 0.0) for row in candidates]
    memories = [number(row.get("memory_p95_mb"), 0.0) for row in candidates]
    return (
        model_id,
        lane,
        statistics.median(runtimes),
        statistics.median(loads),
        statistics.median(memories),
    )


def _advance_cache(
    resident: dict[str, float],
    model_id: str,
    model_mb: float,
    workspace_mb: float,
    capacity_mb: float,
) -> bool:
    """Apply the same resident-plus-workspace admission approximation to a rollout."""

    cached = model_id in resident
    projected = sum(resident.values()) + (0.0 if cached else model_mb) + workspace_mb
    if projected > capacity_mb + 1e-9:
        keep = resident.get(model_id) if cached else None
        resident.clear()
        if keep is not None:
            resident[model_id] = keep
    if model_id not in resident:
        resident[model_id] = max(0.0, model_mb)
    return cached


def _truth_step_cost(
    node: Node,
    resident: dict[str, float],
    capacity_mb: float,
    train_stats: Mapping[str, Mapping[str, Any]],
) -> float:
    if node.lane != "gpu":
        return max(0.0, node.runtime_ms)
    estimate_row = estimate(node, train_stats)
    model_mb = model_memory(node, estimate_row)
    workspace_mb = max(0.0, float(estimate_row["memory_p95_mb"]) - model_mb)
    cached = _advance_cache(resident, node.model_id, model_mb, workspace_mb, capacity_mb)
    load = 0.0 if cached else number(node.load_ms)
    return node.compute_ms + load


def _predicted_step_cost(
    step: Mapping[str, Any],
    resident: dict[str, float],
    capacity_mb: float,
    train_stats: Mapping[str, Mapping[str, Any]],
    model_memories: Mapping[str, float],
) -> float:
    model_id, lane, runtime, load, memory = _predicted_step_components(step, train_stats)
    if lane != "gpu":
        return max(0.0, runtime)
    model_mb = float(model_memories.get(model_id, 0.0))
    workspace_mb = max(0.0, memory - model_mb)
    if model_id == "unknown" or model_mb <= 0.0:
        return max(0.0, runtime)
    cached = _advance_cache(resident, model_id, model_mb, workspace_mb, capacity_mb)
    compute = max(0.0, runtime - load)
    return compute + (0.0 if cached else load)


def _predicted_node_step_cost(
    node: Node,
    resident: dict[str, float],
    capacity_mb: float,
    train_stats: Mapping[str, Mapping[str, Any]],
    model_memories: Mapping[str, float],
) -> float:
    estimate_row = estimate(node, train_stats)
    if node.lane != "gpu":
        return max(0.0, number(estimate_row.get("runtime_p50_ms"), 0.0))
    model_id = node.model_id
    model_mb = float(model_memories.get(model_id, 0.0))
    load = number(estimate_row.get("load_p50_ms"), 0.0)
    runtime = number(estimate_row.get("runtime_p50_ms"), 0.0)
    if model_id == "unknown" or model_mb <= 0.0:
        return max(0.0, runtime)
    workspace_mb = max(0.0, number(estimate_row.get("memory_p95_mb"), 0.0) - model_mb)
    cached = _advance_cache(resident, model_id, model_mb, workspace_mb, capacity_mb)
    return max(0.0, runtime - load) + (0.0 if cached else load)


def _aligned_predicted_future_cost(
    node_id: str,
    resident_after_current: Mapping[str, float],
    capacity_mb: float,
    future_artifacts: Mapping[str, Mapping[str, Any]],
    train_stats: Mapping[str, Mapping[str, Any]],
    model_memories: Mapping[str, float],
    horizon: int,
) -> float:
    row = future_artifacts.get(str(node_id)) or {}
    scenarios = row.get(f"future_h{int(horizon)}") or []
    total = 0.0
    for scenario in scenarios:
        resident = dict(resident_after_current)
        scenario_cost = sum(
            _predicted_step_cost(
                step,
                resident,
                capacity_mb,
                train_stats,
                model_memories,
            )
            for step in (scenario.get("steps") or [])[: max(0, int(horizon))]
        )
        total += number(scenario.get("scenario_probability"), 0.0) * scenario_cost
    return total


def _aligned_predicted_layer_future_cost(
    node_id: str,
    resident_after_current: Mapping[str, float],
    capacity_mb: float,
    future_artifacts: Mapping[str, Mapping[str, Any]],
    train_stats: Mapping[str, Mapping[str, Any]],
    model_memories: Mapping[str, float],
    horizon: int,
) -> float:
    """Score predicted future layers; each layer may contain many prototypes."""

    row = future_artifacts.get(str(node_id)) or {}
    if "future_h5_layers" not in row:
        raise ValueError(f"missing future_h5_layers artifact for node {node_id}")
    scenarios = row.get("future_h5_layers") or []
    validate_layer_scenarios(scenarios, horizon)
    total = 0.0
    for scenario in scenarios:
        resident = dict(resident_after_current)
        scenario_cost = 0.0
        for layer in (scenario.get("layers") or [])[: max(0, int(horizon))]:
            for step in layer.get("nodes") or []:
                scenario_cost += _predicted_step_cost(
                    step,
                    resident,
                    capacity_mb,
                    train_stats,
                    model_memories,
                )
        total += number(scenario.get("scenario_probability"), 0.0) * scenario_cost
    return total


def _cache_res_step_cost(
    step: Mapping[str, Any],
    resident: dict[str, float],
    capacity_mb: float,
    train_stats: Mapping[str, Mapping[str, Any]],
    model_memories: Mapping[str, float],
    lam: float = 0.5,
) -> float:
    """Cache-aware predicted-resource step cost (Phase 3 consumption arm).

    Runtime/load come from the predicted quantile mix; residency is simulated
    with the shared cache-advance approximation so a cold load is charged once
    per model rather than on every step.
    """

    model_id = text(step.get("model_id"), "unknown")
    lane = text(step.get("execution_lane"), "unknown")
    if lane != "gpu":
        return _mix_step_cost(step, train_stats, lam)
    model_mb = float(model_memories.get(model_id, 0.0))
    resource = step.get("resource") or {}
    memory = number((resource.get("memory_p95_mb")), 0.0)
    if model_id == "unknown" or model_mb <= 0.0:
        return _mix_step_cost(step, train_stats, lam)
    workspace_mb = max(0.0, memory - model_mb)
    cached = _advance_cache(resident, model_id, model_mb, workspace_mb, capacity_mb)
    runtime = _mix_step_cost(step, train_stats, lam)
    if cached:
        return runtime
    occurrence = resource.get("load_occurrence_probability")
    duration = resource.get("load_duration_ms_quantiles") or {}
    d50 = duration.get("p50")
    d90 = duration.get("p90")
    load = 0.0
    if isinstance(d50, (int, float)) and isinstance(d90, (int, float)):
        load = (1.0 - lam) * float(d50) + lam * float(d90)
    elif isinstance(d50, (int, float)):
        load = float(d50)
    return runtime + load


def _aligned_predicted_layer_future_cost_res(
    node_id: str,
    resident_after_current: Mapping[str, float],
    capacity_mb: float,
    future_artifacts: Mapping[str, Mapping[str, Any]],
    train_stats: Mapping[str, Mapping[str, Any]],
    model_memories: Mapping[str, float],
    horizon: int,
) -> float:
    row = future_artifacts.get(str(node_id)) or {}
    if "future_h5_layers" not in row:
        raise ValueError(f"missing future_h5_layers artifact for node {node_id}")
    scenarios = row.get("future_h5_layers") or []
    validate_layer_scenarios(scenarios, horizon)
    total = 0.0
    for scenario in scenarios:
        resident = dict(resident_after_current)
        scenario_cost = 0.0
        for layer in (scenario.get("layers") or [])[: max(0, int(horizon))]:
            for step in layer.get("nodes") or []:
                scenario_cost += _cache_res_step_cost(step, resident, capacity_mb, train_stats, model_memories)
        total += number(scenario.get("scenario_probability"), 0.0) * scenario_cost
    return total


def aligned_h5_score_res(
    job: Job,
    node_id: str,
    gpu: GPU,
    train_stats: Mapping[str, Mapping[str, Any]],
    future_artifacts: Mapping[str, Mapping[str, Any]],
    model_memories: Mapping[str, float],
    horizon: int = 5,
) -> float:
    """Cache-aware consumption of predicted resources over the predicted layers."""

    resident = dict(gpu.resident)
    future = _aligned_predicted_layer_future_cost_res(
        node_id, resident, gpu.capacity_mb, future_artifacts, train_stats, model_memories, horizon
    )
    node = job.template.by_id[node_id]
    estimate_row = estimate(node, train_stats)
    current = max(0.0, number(estimate_row.get("runtime_p50_ms"), 0.0))
    return float(current + future)


def aligned_h5_score(
    source: str,
    job: Job,
    node_id: str,
    gpu: GPU,
    train_stats: Mapping[str, Mapping[str, Any]],
    future_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    horizon: int = 5,
    model_memories: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Score current action plus H successor layers under one H5 contract.

    ``source`` is either ``predicted`` or ``truth``.  The action scope,
    cache-advance approximation, current-action inclusion, and horizon are
    shared; only runtime/load/future identities differ.  This isolated
    contract intentionally excludes transition-profile overhead so the first
    experiment changes scoring semantics only.
    """

    if source not in {"predicted", "predicted_layer", "truth"}:
        raise ValueError(f"unsupported aligned H5 score source: {source}")
    model_memories = dict(model_memories or model_memory_by_model((job.template,)))
    resident = dict(gpu.resident)
    node = job.template.by_id[node_id]
    if source == "truth":
        current = _truth_step_cost(node, resident, gpu.capacity_mb, train_stats)
        frontier = list(node.successors)
        seen: set[str] = set()
        future = 0.0
        for _ in range(max(0, int(horizon))):
            next_frontier: list[str] = []
            for current_id in frontier:
                if current_id in seen:
                    continue
                seen.add(current_id)
                successor = job.template.by_id[current_id]
                future += _truth_step_cost(successor, resident, gpu.capacity_mb, train_stats)
                next_frontier.extend(successor.successors)
            frontier = next_frontier
            if not frontier:
                break
    else:
        if future_artifacts is None:
            raise ValueError("aligned predicted H5 score requires future artifacts")
        current = _predicted_node_step_cost(
            node,
            resident,
            gpu.capacity_mb,
            train_stats,
            model_memories,
        )
        if source == "predicted_layer":
            future = _aligned_predicted_layer_future_cost(
                node_id,
                resident,
                gpu.capacity_mb,
                future_artifacts,
                train_stats,
                model_memories,
                horizon,
            )
        else:
            future = _aligned_predicted_future_cost(
                node_id,
                resident,
                gpu.capacity_mb,
                future_artifacts,
                train_stats,
                model_memories,
                horizon,
            )
    return {
        "current_ms": float(current),
        "future_ms": float(future),
        "total_ms": float(current + future),
    }


def aligned_h5_policy_key(
    source: str,
    candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool],
    jobs: Sequence[Job],
    train_stats: Mapping[str, Mapping[str, Any]],
    future_artifacts: Mapping[str, Mapping[str, Any]] | None,
    model_memories: Mapping[str, float],
    horizon: int = 5,
) -> tuple[Any, ...]:
    item, job_index, node_id, _model_id, gpu, _estimate_row, _predicted_fit = candidate
    score = aligned_h5_score(
        source,
        jobs[job_index],
        node_id,
        gpu,
        train_stats,
        future_artifacts,
        horizon,
        model_memories,
    )
    # Priority is a shared hard business rule; cost comparison is identical
    # after priority, regardless of whether the values are predicted or true.
    return (
        float(item[0]),
        score["total_ms"],
        score["future_ms"],
        score["current_ms"],
        float(item[1]),
        int(item[2]),
        str(item[3]),
        gpu.index,
    )


def _rl_features(
    candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool],
    jobs: Sequence[Job],
    train_stats: Mapping[str, Mapping[str, Any]],
    future_artifacts: Mapping[str, Mapping[str, Any]] | None,
    horizon: int,
) -> list[float]:
    item, job_index, node_id, model_id, gpu, estimate_row, predicted_fit = candidate
    job = jobs[job_index]
    cache_hit = 1.0 if model_id in gpu.resident else 0.0
    capacity = max(1.0, float(gpu.capacity_mb))
    current_load = 0.0 if cache_hit else number(estimate_row.get("load_p50_ms"), 0.0)
    future_values = [
        predicted_future_cost(node_id, future_artifacts or {}, train_stats, future_h)
        if horizon >= future_h and future_artifacts is not None
        else 0.0
        for future_h in (1, 3, 5)
    ]
    return [
        float(item[0]),
        min(1.0, max(0.0, (float(item[1]) / 100000.0))),
        min(5.0, number(estimate_row.get("runtime_p50_ms"), 0.0) / 100000.0),
        min(5.0, current_load / 100000.0),
        min(2.0, number(estimate_row.get("memory_p95_mb"), 0.0) / capacity),
        cache_hit,
        min(2.0, sum(gpu.resident.values()) / capacity),
        min(2.0, len(gpu.resident) / 8.0),
        1.0 if predicted_fit else 0.0,
        1.0 if job.service_class == "priority" else 0.0,
        min(5.0, future_values[0] / 100000.0),
        min(5.0, future_values[1] / 100000.0),
        min(5.0, future_values[2] / 100000.0),
        1.0 if horizon > 0 else 0.0,
    ]


def _rl_choose(
    pool: Sequence[tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool]],
    jobs: Sequence[Job],
    train_stats: Mapping[str, Mapping[str, Any]],
    future_artifacts: Mapping[str, Mapping[str, Any]] | None,
    horizon: int,
    rl_context: dict[str, Any],
) -> tuple[tuple[float, int, int, str], int, dict[str, Any], int, int]:
    import torch

    model = rl_context.get("model")
    if model is None:
        raise ValueError("RL policy requires rl_context['model']")
    features = torch.tensor(
        [_rl_features(candidate, jobs, train_stats, future_artifacts, horizon) for candidate in pool],
        dtype=torch.float32,
        device=next(model.parameters()).device,
    )
    logits = model(features).reshape(-1)
    distribution = torch.distributions.Categorical(logits=logits)
    if str(rl_context.get("mode") or "train") == "eval":
        index = int(torch.argmax(logits).item())
    else:
        index = int(distribution.sample().item())
    entry = {
        "log_prob": distribution.log_prob(torch.tensor(index, device=logits.device)).detach(),
        "entropy": distribution.entropy().detach(),
        "candidate_count": len(pool),
        "action_index": index,
        "features": features.cpu().tolist(),
    }
    critic = rl_context.get("critic")
    if critic is not None:
        with torch.no_grad():
            state_feat = features.mean(dim=0, keepdim=True)
            entry["value"] = critic(state_feat).squeeze().item()
    rl_context.setdefault("trajectory", []).append(entry)
    chosen = pool[index]
    return chosen[0], chosen[4].index, chosen[5], len(pool), sum(bool(candidate[6]) for candidate in pool)


def _cp_rho_choose(
    pool: Sequence[tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool]],
    jobs: Sequence[Job],
    train_stats: Mapping[str, Mapping[str, Any]],
    future_artifacts: Mapping[str, Mapping[str, Any]] | None,
    horizon: int,
    policy_context: dict[str, Any] | None,
) -> tuple[tuple[float, int, int, str], int, dict[str, Any], int, int]:
    from tracing.scheduling.cp_rho import candidate_from_scheduler_row, solve_first_action

    rows = []
    for index, candidate in enumerate(pool):
        item, job_index, node_id, model_id, gpu, estimate_row, _predicted_fit = candidate
        future = (
            predicted_future_cost(node_id, future_artifacts or {}, train_stats, horizon)
            if future_artifacts is not None and horizon > 0
            else 0.0
        )
        rows.append(
            candidate_from_scheduler_row(
                index=index,
                node_key=(job_index, node_id),
                gpu_index=gpu.index,
                priority=float(item[0]),
                ready_since_ms=float(item[1]),
                model_id=model_id,
                estimate_row=estimate_row,
                cache_hit=model_id in gpu.resident,
                future_cost_ms=future,
                capacity_mb=gpu.capacity_mb,
            )
        )
    result = solve_first_action(
        rows,
        horizon=horizon,
        time_limit_s=float((policy_context or {}).get("cp_rho_time_limit_s", 0.25)),
        random_seed=int((policy_context or {}).get("cp_rho_seed", 0)),
        num_workers=int((policy_context or {}).get("cp_rho_num_workers", 1)),
        priority_weight=PREDOPT_V2_PRIORITY_WEIGHT,
        future_weight=PREDOPT_V2_FUTURE_WEIGHT,
        memory_weight=PREDOPT_V2_MEMORY_WEIGHT,
    )
    if policy_context is not None:
        policy_context.setdefault("cp_rho_events", []).append(result.to_dict())
        if policy_context.get("bc_collect"):
            features = [
                _rl_features(candidate, jobs, train_stats, future_artifacts, horizon)
                for candidate in pool
            ]
            policy_context.setdefault("bc_events", []).append({
                "features": features,
                "action_index": int(result.candidate_index),
                "horizon": horizon,
                "candidate_count": len(pool),
                "fallback": bool(result.fallback),
            })
    position = int(result.candidate_index)
    if position < 0 or position >= len(pool):
        raise RuntimeError(f"CP-RHO returned invalid candidate index: {position}")
    chosen = pool[position]
    return chosen[0], chosen[4].index, chosen[5], len(pool), sum(bool(candidate[6]) for candidate in pool)


def _predicted_node_duration(
    node: Node,
    gpu: GPU,
    estimate_row: Mapping[str, Any],
) -> float:
    """Return a scheduler-model duration without consulting execution truth."""

    load = 0.0 if node.model_id in gpu.resident else number(estimate_row.get("load_p50_ms"))
    return max(0.1, number(estimate_row.get("runtime_p50_ms"))) + load


def _minmax_normalize(values: Sequence[float]) -> list[float]:
    """Normalize one feature over the current candidate set."""

    if not values:
        return []
    lower = min(values)
    upper = max(values)
    span = upper - lower
    if span <= 1e-9:
        return [0.0 for _ in values]
    return [(value - lower) / span for value in values]


def _risk_aware_candidate_metrics(
    candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool],
    jobs: Sequence[Job],
    decision_time_ms: float,
    extension_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build visible urgency/resource/age features for one candidate.

    The feature calculation is intentionally limited to the current node,
    current GPU state, workload deadline/class and ready age.  It does not
    inspect successors, template suffixes or execution truth.
    """

    item, job_index, node_id, _model_id, gpu, estimate_row, _predicted_fit = candidate
    job = jobs[job_index]
    node = job.template.by_id[node_id]
    admitted, evicted, _model_mb, _workspace_mb, projected_mb = plan_gpu_admission(
        gpu,
        node,
        estimate_row,
    )
    if node.model_id in gpu.resident:
        cold_load_ms = 0.0
    else:
        cold_load_ms = transition_load_cost(node, estimate_row, 1, extension_config)[0]
    eviction_ms = sum(
        transition_eviction_cost(model_id, extension_config)[0]
        for model_id in evicted
    )
    transition_ms = cold_load_ms + eviction_ms
    predicted_duration_ms = max(0.1, number(estimate_row.get("runtime_p50_ms"))) + transition_ms

    slack_ms = math.inf
    deadline_pressure = 0.0
    if job.deadline_ms is not None:
        slack_ms = float(job.deadline_ms) - decision_time_ms - predicted_duration_ms
        deadline_window_ms = max(1.0, float(job.deadline_ms) - job.arrival_ms)
        deadline_pressure = min(
            1.0,
            max(0.0, 1.0 - max(0.0, slack_ms) / deadline_window_ms),
        )
    urgency = (1.0 if job.service_class == "priority" else 0.0) + deadline_pressure
    wait_age_ms = max(0.0, decision_time_ms - float(item[1]))
    memory_ratio = max(0.0, float(projected_mb)) / max(1.0, float(gpu.capacity_mb))
    return {
        "candidate": candidate,
        "job_index": job_index,
        "node_id": node_id,
        "gpu_index": gpu.index,
        "admitted": bool(admitted),
        "urgency": urgency,
        "slack_ms": slack_ms,
        "wait_age_ms": wait_age_ms,
        "transition_ms": transition_ms,
        "eviction_count": len(evicted),
        "memory_ratio": memory_ratio,
        "predicted_duration_ms": predicted_duration_ms,
    }


def _risk_aware_choose(
    pool: Sequence[tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool]],
    candidate_count: int,
    jobs: Sequence[Job],
    decision_time_ms: float,
    extension_config: Mapping[str, Any] | None,
    policy_context: dict[str, Any] | None,
) -> tuple[tuple[float, int, int, str], int, dict[str, Any], int, int]:
    """Choose an action using the thesis-inspired transparent linear score."""

    metrics = [
        _risk_aware_candidate_metrics(candidate, jobs, decision_time_ms, extension_config)
        for candidate in pool
    ]
    strict_metrics = [row for row in metrics if row["admitted"]]
    active_metrics = strict_metrics or metrics
    urgency_norm = _minmax_normalize([float(row["urgency"]) for row in active_metrics])
    transition_norm = _minmax_normalize([float(row["transition_ms"]) for row in active_metrics])
    memory_norm = _minmax_normalize([float(row["memory_ratio"]) for row in active_metrics])
    eviction_norm = _minmax_normalize([float(row["eviction_count"]) for row in active_metrics])
    age_norm = _minmax_normalize([float(row["wait_age_ms"]) for row in active_metrics])

    scored: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for row, urgency, transition, memory, evictions, age in zip(
        active_metrics,
        urgency_norm,
        transition_norm,
        memory_norm,
        eviction_norm,
        age_norm,
    ):
        resource_risk = (transition + memory + evictions) / 3.0
        score = (
            -RISK_AWARE_URGENCY_WEIGHT * urgency
            + RISK_AWARE_RESOURCE_WEIGHT * resource_risk
            - RISK_AWARE_AGE_WEIGHT * age
        )
        row["urgency_norm"] = urgency
        row["resource_risk_norm"] = resource_risk
        row["age_norm"] = age
        row["score"] = score
        candidate = row["candidate"]
        item = candidate[0]
        scored.append(
            (
                (
                    score,
                    -float(row["urgency"]),
                    float(row["resource_risk_norm"]),
                    -float(row["wait_age_ms"]),
                    float(row["predicted_duration_ms"]),
                    float(item[0]),
                    float(item[1]),
                    int(item[2]),
                    str(item[3]),
                    int(row["gpu_index"]),
                ),
                row,
            )
        )

    _key, chosen_metrics = min(scored, key=lambda value: value[0])
    chosen = chosen_metrics["candidate"]
    if policy_context is not None:
        policy_context.setdefault("risk_aware_events", []).append(
            {
                "candidate_count": int(candidate_count),
                "strict_feasible_count": int(len(strict_metrics)),
                "active_candidate_count": int(len(active_metrics)),
                "selected_job_index": int(chosen_metrics["job_index"]),
                "selected_node_id": str(chosen_metrics["node_id"]),
                "selected_gpu_index": int(chosen_metrics["gpu_index"]),
                "score": float(chosen_metrics["score"]),
                "urgency": float(chosen_metrics["urgency"]),
                "urgency_norm": float(chosen_metrics["urgency_norm"]),
                "resource_risk_norm": float(chosen_metrics["resource_risk_norm"]),
                "age_ms": float(chosen_metrics["wait_age_ms"]),
                "age_norm": float(chosen_metrics["age_norm"]),
                "slack_ms": (
                    None
                    if math.isinf(float(chosen_metrics["slack_ms"]))
                    else float(chosen_metrics["slack_ms"])
                ),
                "transition_ms": float(chosen_metrics["transition_ms"]),
                "eviction_count": int(chosen_metrics["eviction_count"]),
            }
        )
    return chosen[0], chosen[4].index, chosen[5], candidate_count, len(strict_metrics)


def _predicted_candidate_key(
    candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool],
    jobs: Sequence[Job],
    train_stats: Mapping[str, Mapping[str, Any]],
    future_artifacts: Mapping[str, Mapping[str, Any]],
    horizon: int,
) -> tuple[float, float, float, float, int, str, int]:
    item, job_index, node_id, _model_id, gpu, estimate_row, _predicted_fit = candidate
    node = jobs[job_index].template.by_id[node_id]
    current = _predicted_node_duration(node, gpu, estimate_row)
    future = predicted_future_cost(node_id, future_artifacts, train_stats, horizon)
    return (
        current + future,
        future,
        float(item[0]),
        float(item[1]),
        int(item[2]),
        str(item[3]),
        gpu.index,
    )


def _predicted_schedule(
    candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool],
    jobs: Sequence[Job],
    gpus: Mapping[int, GPU],
    busy_gpus: set[int],
    events: list[tuple[float, int, int, int, str]],
    sequence: list[int],
    now_ms: float,
    train_stats: Mapping[str, Mapping[str, Any]],
) -> bool:
    _item, job_index, node_id, _model_id, candidate_gpu, candidate_row, _fit = candidate
    gpu_index = candidate_gpu.index
    if gpu_index in busy_gpus:
        return False
    job = jobs[job_index]
    node = job.template.by_id[node_id]
    if job.node_state.get(node_id) != "ready":
        return False
    gpu = gpus[gpu_index]
    estimate_row = dict(candidate_row) if candidate_row else estimate(node, train_stats)
    admitted, evicted, memory, _workspace, _projected = plan_gpu_admission(gpu, node, estimate_row)
    if not admitted:
        return False
    for model_id in evicted:
        gpu.resident.pop(model_id, None)
    cold = node.model_id not in gpu.resident
    if cold:
        gpu.resident[node.model_id] = memory
    duration = max(0.1, number(estimate_row.get("runtime_p50_ms")))
    if cold:
        duration += number(estimate_row.get("load_p50_ms"))
    job.node_state[node_id] = "running"
    gpu.active_node = (job_index, node_id)
    gpu.busy_until = now_ms + duration
    busy_gpus.add(gpu_index)
    sequence[0] += 1
    heapq.heappush(events, (now_ms + duration, sequence[0], job_index, gpu_index, node_id))
    return True


def _predicted_visible_terminal_work(
    jobs: Sequence[Job],
    pool: Sequence[tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool]],
    train_stats: Mapping[str, Mapping[str, Any]],
    known_job_indices: set[int],
    future_artifacts: Mapping[str, Mapping[str, Any]],
    horizon: int,
) -> float:
    """Estimate only visible ready-node continuation, never template suffixes."""

    visible: dict[tuple[int, str], tuple[Node, dict[str, Any], GPU]] = {}
    for _item, job_index, node_id, _model_id, gpu, row, _fit in pool:
        if job_index not in known_job_indices:
            continue
        key = (job_index, node_id)
        node = jobs[job_index].template.by_id[node_id]
        previous = visible.get(key)
        if previous is None or _predicted_node_duration(node, gpu, row) < _predicted_node_duration(node, previous[2], previous[1]):
            visible[key] = (node, row, gpu)
    total = 0.0
    for (job_index, node_id), (node, row, gpu) in visible.items():
        state = jobs[job_index].node_state.get(node_id)
        if state in {"complete", "failed"}:
            continue
        if state != "running":
            total += _predicted_node_duration(node, gpu, row)
        total += predicted_future_cost(node_id, future_artifacts, train_stats, horizon)
    return total


def _predicted_rollout_score(
    first_candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool],
    pool: Sequence[tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool]],
    jobs: Sequence[Job],
    free_gpus: Sequence[GPU],
    train_stats: Mapping[str, Mapping[str, Any]],
    future_artifacts: Mapping[str, Mapping[str, Any]],
    horizon: int,
    decision_time_ms: float,
    terminal: bool,
) -> tuple[float, dict[str, Any]]:
    predicted_jobs = copy.deepcopy(list(jobs))
    known_job_indices = {
        job_index
        for job_index, job in enumerate(jobs)
        if any(state != "pending" for state in job.node_state.values())
    }
    visible_nodes_by_job: dict[int, set[str]] = defaultdict(set)
    for _item, job_index, node_id, _model_id, _gpu, _row, _fit in pool:
        visible_nodes_by_job[job_index].add(node_id)
    predicted_gpus = {gpu.index: copy.deepcopy(gpu) for gpu in free_gpus}
    busy_gpus: set[int] = set()
    events: list[tuple[float, int, int, int, str]] = []
    sequence = [0]
    finished_jobs: set[int] = set()

    def schedule(candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool]) -> bool:
        return _predicted_schedule(
            candidate,
            predicted_jobs,
            predicted_gpus,
            busy_gpus,
            events,
            sequence,
            cursor[0],
            train_stats,
        )

    cursor = [float(decision_time_ms)]
    if not schedule(first_candidate):
        return math.inf, {"area_ms": None, "terminal_ms": None, "predicted_events": 0}

    # Fill other currently free GPUs with a deterministic predicted
    # continuation.  The real event loop will re-observe after each dispatch.
    while True:
        choices = [
            candidate
            for candidate in pool
            if candidate[4].index not in busy_gpus
            and predicted_jobs[candidate[1]].node_state.get(candidate[2]) == "ready"
        ]
        if not choices:
            break
        if not schedule(min(choices, key=lambda item: _predicted_candidate_key(item, predicted_jobs, train_stats, future_artifacts, horizon))):
            break

    area_ms = 0.0
    completed_events = 0
    while events and completed_events < max(1, int(horizon)):
        next_time = events[0][0]
        active_jobs = sum(
            1
            for job_index, job in enumerate(predicted_jobs)
            if job_index in known_job_indices
            and job.arrival_ms <= cursor[0] + 1e-9
            and job_index not in finished_jobs
        )
        area_ms += max(0.0, next_time - cursor[0]) * active_jobs
        cursor[0] = next_time
        while events and events[0][0] <= cursor[0] + 1e-9:
            _finish, _order, job_index, gpu_index, node_id = heapq.heappop(events)
            job = predicted_jobs[job_index]
            job.node_state[node_id] = "complete"
            job.completed.add(node_id)
            gpu = predicted_gpus[gpu_index]
            gpu.active_node = None
            gpu.busy_until = cursor[0]
            busy_gpus.discard(gpu_index)
            completed_events += 1
            if all(
                job.node_state.get(visible_node_id) == "complete"
                for visible_node_id in visible_nodes_by_job.get(job_index, set())
            ):
                finished_jobs.add(job_index)
    terminal_ms = (
        _predicted_visible_terminal_work(
            predicted_jobs,
            pool,
            train_stats,
            known_job_indices,
            future_artifacts,
            horizon,
        )
        if terminal
        else 0.0
    )
    score = area_ms + terminal_ms
    return score, {
        "area_ms": area_ms,
        "terminal_ms": terminal_ms,
        "predicted_events": completed_events,
    }


def _pred_mpc_choose(
    pool: Sequence[tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool]],
    jobs: Sequence[Job],
    free_gpus: Sequence[GPU],
    train_stats: Mapping[str, Mapping[str, Any]],
    future_artifacts: Mapping[str, Mapping[str, Any]],
    horizon: int,
    decision_time_ms: float,
    terminal: bool,
    policy_context: dict[str, Any] | None,
) -> tuple[tuple[float, int, int, str], int, dict[str, Any], int, int]:
    started = time.perf_counter()
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for position, candidate in enumerate(pool):
        score, details = _predicted_rollout_score(
            candidate,
            pool,
            jobs,
            free_gpus,
            train_stats,
            future_artifacts,
            horizon,
            decision_time_ms,
            terminal,
        )
        scored.append((score, position, details))
    chosen_score, chosen_position, chosen_details = min(
        scored,
        key=lambda value: (
            value[0],
            _predicted_candidate_key(pool[value[1]], jobs, train_stats, future_artifacts, horizon),
        ),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if policy_context is not None:
        policy_context.setdefault("pred_mpc_events", []).append(
            {
                "horizon": int(horizon),
                "terminal": bool(terminal),
                "candidate_count": len(pool),
                "selected_position": int(chosen_position),
                "score": float(chosen_score),
                "area_ms": chosen_details.get("area_ms"),
                "terminal_ms": chosen_details.get("terminal_ms"),
                "predicted_events": chosen_details.get("predicted_events"),
                "solve_ms": elapsed_ms,
            }
        )
    chosen = pool[chosen_position]
    return chosen[0], chosen[4].index, chosen[5], len(pool), sum(bool(candidate[6]) for candidate in pool)


def srtf_aging_key(
    hard_priority: float,
    remaining_ms: float,
    wait_age_ms: float,
) -> tuple[float, float, float, float]:
    """SRTF + aging, as an exact and independently checkable ordering key.

    ``S_j = R_j - wait_age_j`` where ``R_j`` is the predicted remaining work of the
    emitted chain and one millisecond of waiting buys one millisecond of credit, so
    the two terms share a unit and a long enough wait always promotes a job.  The
    hard service priority is the first element of the tuple, and Python compares
    tuples lexicographically, so no amount of aging can ever cross it.

    Negative waits are clamped to zero: a candidate cannot earn credit for a clock
    that has not advanced.
    """

    wait = float(wait_age_ms)
    if wait < -1e-9:
        # a ready candidate can never be ready after the current decision time, so a
        # materially negative age means the simulator invariant broke.  Clamping it
        # away would hide that, so fail closed; only floating-point noise is tolerated.
        raise ValueError(f"negative ready age: {wait}")
    wait = max(0.0, wait)
    return (
        float(hard_priority),
        float(remaining_ms) - wait,
        float(remaining_ms),
        -wait,
    )


def tie_beta(queue_length: float, batch_capacity: float) -> float:
    """The paper's adaptive tail weight: clip(0.1 * L_q / B, 0.1, 0.5).

    ``L_q`` is the queue length and ``B`` the batch capacity in the paper.  Our
    adaptation maps L_q to the number of candidates competing in the top priority
    tier and B to the simulator's GPU service concurrency; that is a stated
    deviation, not a tuned parameter.
    """

    b = max(1e-9, float(batch_capacity))
    return min(0.5, max(0.1, 0.1 * float(queue_length) / b))


def tie_current_score(
    mean_ms: float,
    cvar90_ms: float,
    beta: float,
    load_ms: float,
) -> float:
    """TIE(X) = E[X] + beta * CVaR_0.9[X], plus the model-load surcharge.

    The distribution is the train-only current-node histogram, so this score uses
    no information about the future workflow.
    """

    return float(mean_ms) + float(beta) * float(cvar90_ms) + float(load_ms)


def _require_resource_applicable(node: Node, template_id: str) -> None:
    """v3.1 section 3: run_control and terminal markers are not schedulable work.

    Module level so every pool-building site can enforce it, not just choose_action.
    """

    if not getattr(node, "resource_applicable", True):
        raise ValueError(
            "node %s in template %s is not resource-applicable but reached the "
            "candidate pool" % (node.node_id, template_id)
        )


def choose_action(
    policy: str,
    ready_items: Sequence[tuple[float, int, int, str]],
    jobs: Sequence[Job],
    free_gpus: Sequence[GPU],
    train_stats: Mapping[str, Mapping[str, Any]],
    round_robin_cursor: int,
    future_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    rl_context: dict[str, Any] | None = None,
    policy_context: dict[str, Any] | None = None,
    decision_time_ms: float = 0.0,
    extension_config: Mapping[str, Any] | None = None,
) -> tuple[tuple[float, int, int, str], int, dict[str, Any], int, int]:
    if not free_gpus:
        raise RuntimeError("choose_action called without a free GPU")


    # Candidate tuples intentionally carry scheduler-visible fields only:
    # item, job index, node id, model id, GPU, prediction, predicted fit.
    candidates: list[tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool]] = []
    for item in ready_items:
        _priority, _ready_time, job_index, node_id = item
        job = jobs[job_index]
        node = job.template.by_id[node_id]
        if job.node_state.get(node_id) != "ready" or node.lane != "gpu":
            continue
        _require_resource_applicable(node, job.template.template_id)
        estimate_row = estimate(node, train_stats)
        for gpu in free_gpus:
            candidates.append(
                (
                    item,
                    job_index,
                    node_id,
                    node.model_id,
                    gpu,
                    estimate_row,
                    float(estimate_row["memory_p95_mb"]) <= gpu.capacity_mb + 1e-9,
                )
            )
    if not candidates:
        raise RuntimeError("choose_action called without a ready GPU node")
    fitting = [candidate for candidate in candidates if candidate[6]]
    pool = fitting or candidates

    if policy_context is not None:
        forced_decision = policy_context.get("audit_forced_decision_index")
        current_decision = policy_context.get("current_decision_index")
        forced_action = policy_context.get("audit_forced_action")
        if forced_decision == current_decision and forced_action is not None:
            if not isinstance(forced_action, (tuple, list)) or len(forced_action) != 3:
                raise ValueError(f"invalid audit_forced_action: {forced_action!r}")
            forced_job_index, forced_node_id, forced_gpu_index = forced_action
            for candidate in pool:
                if (
                    candidate[1] == int(forced_job_index)
                    and candidate[2] == str(forced_node_id)
                    and candidate[4].index == int(forced_gpu_index)
                ):
                    return candidate[0], candidate[4].index, candidate[5], len(candidates), len(fitting)
            raise ValueError(
                "audit_forced_action is not in the current policy candidate pool: "
                f"{forced_action!r}"
            )

    if policy in {"rl_0", "rl_h5"}:
        if rl_context is None:
            raise ValueError(f"{policy} requires rl_context")
        horizon = 0 if policy == "rl_0" else 5
        return _rl_choose(pool, jobs, train_stats, future_artifacts, horizon, rl_context)
    if policy.startswith("bc_h"):
        if rl_context is None:
            raise ValueError(f"{policy} requires rl_context with bc_model")
        horizon = int(policy.rsplit("h", 1)[1])
        # Reuse the RL scorer with a BC-trained model
        return _rl_choose(pool, jobs, train_stats, future_artifacts, horizon, rl_context)
    if policy == "risk_aware":
        strict_pool = [
            candidate
            for candidate in pool
            if plan_gpu_admission(candidate[4], jobs[candidate[1]].template.by_id[candidate[2]], candidate[5])[0]
        ]
        return _risk_aware_choose(
            strict_pool or pool,
            len(candidates),
            jobs,
            decision_time_ms,
            extension_config,
            policy_context,
        )
    if policy.startswith("cp_rho_h"):
        horizon = int(policy.rsplit("h", 1)[1])
        if future_artifacts is None:
            raise ValueError(f"{policy} requires finite-horizon artifacts")
        return _cp_rho_choose(pool, jobs, train_stats, future_artifacts, horizon, policy_context)
    if policy.startswith("pred_mpc_h"):
        if future_artifacts is None:
            raise ValueError(f"{policy} requires finite-horizon artifacts")
        no_terminal = policy.endswith("_no_terminal")
        horizon_token = policy[len("pred_mpc_h") :]
        if no_terminal:
            horizon_token = horizon_token[: -len("_no_terminal")]
        horizon = int(horizon_token)
        return _pred_mpc_choose(
            pool,
            jobs,
            free_gpus,
            train_stats,
            future_artifacts,
            horizon,
            decision_time_ms,
            not no_terminal,
            policy_context,
        )
    if policy == "aligned_predopt_h5_layer_res":
        if future_artifacts is None:
            raise ValueError(f"{policy} requires finite-horizon artifacts")
        strict_pool = [
            candidate
            for candidate in candidates
            if plan_gpu_admission(
                candidate[4],
                jobs[candidate[1]].template.by_id[candidate[2]],
                candidate[5],
            )[0]
        ]
        if not strict_pool:
            raise RuntimeError(f"{policy} has no strict-feasible GPU action")
        model_memories = model_memory_by_model(job.template for job in jobs)
        chosen = min(
            strict_pool,
            key=lambda candidate: (
                aligned_h5_score_res(
                    jobs[candidate[1]],
                    candidate[2],
                    candidate[4],
                    train_stats,
                    future_artifacts,
                    model_memories,
                    5,
                ),
                candidate[0][0],
                candidate[0][1],
                candidate[0][2],
                candidate[0][3],
                candidate[4].index,
            ),
        )
        return chosen[0], chosen[4].index, chosen[5], len(candidates), len(strict_pool)
    if policy in ALIGNED_H5_POLICIES:
        if policy in {"aligned_predopt_h5", "aligned_predopt_h5_layer"} and future_artifacts is None:
            raise ValueError(f"{policy} requires finite-horizon artifacts")
        strict_pool = [
            candidate
            for candidate in candidates
            if plan_gpu_admission(
                candidate[4],
                jobs[candidate[1]].template.by_id[candidate[2]],
                candidate[5],
            )[0]
        ]
        if not strict_pool:
            raise RuntimeError(f"{policy} has no strict-feasible GPU action")
        model_memories = model_memory_by_model(job.template for job in jobs)
        source = (
            "predicted_layer"
            if policy == "aligned_predopt_h5_layer"
            else "predicted"
            if policy == "aligned_predopt_h5"
            else "truth"
        )
        chosen = min(
            strict_pool,
            key=lambda candidate: aligned_h5_policy_key(
                source,
                candidate,
                jobs,
                train_stats,
                future_artifacts,
                model_memories,
                5,
            ),
        )
        return chosen[0], chosen[4].index, chosen[5], len(candidates), len(strict_pool)
    if policy == "round_robin":
        chosen = min(
            pool,
            key=lambda candidate: (
                candidate[0][0],
                candidate[0][1],
                candidate[0][2],
                candidate[0][3],
                (candidate[4].index - round_robin_cursor) % max(1, len(free_gpus)),
                candidate[4].index,
            ),
        )
    if policy == "fcfs":
        chosen = min(
            pool,
            key=lambda candidate: (
                candidate[0][0],
                candidate[0][1],
                candidate[0][2],
                candidate[0][3],
                candidate[4].index,
            ),
        )
    elif policy == "sjf_pred":
        if future_artifacts is None:
            raise ValueError(f"{policy} requires finite-horizon artifacts")

        def predicted_remaining_work(candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool]) -> float:
            item, _job_index, node_id, model_id, gpu, estimate_row, _predicted_fit = candidate
            load = 0.0 if model_id in gpu.resident else float(estimate_row["load_p50_ms"])
            current = float(estimate_row["runtime_p50_ms"]) + load
            row = future_artifacts.get(str(node_id)) or {}
            remaining_length = float(row.get("predicted_future_length") or 0.0)
            per_step = float(estimate_row["runtime_p50_ms"]) + float(estimate_row["load_p50_ms"])
            return current + remaining_length * per_step

        chosen = min(
            pool,
            key=lambda candidate: (
                candidate[0][0],
                predicted_remaining_work(candidate),
                candidate[0][1],
                candidate[0][2],
                candidate[0][3],
                candidate[4].index,
            ),
        )
    elif policy == "state_aware":
        def state_score(candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool]) -> tuple[Any, ...]:
            item, job_index, _node_id, model_id, gpu, estimate_row, _predicted_fit = candidate
            job = jobs[job_index]
            load = 0.0 if model_id in gpu.resident else float(estimate_row["load_p50_ms"])
            current = float(estimate_row["runtime_p50_ms"]) + load
            miss_flag = 0
            urgency = 0.0
            if job.deadline_ms is not None:
                window = max(1.0, float(job.deadline_ms) - float(job.arrival_ms))
                slack = float(job.deadline_ms) - decision_time_ms - current
                if slack < 0.0:
                    miss_flag = 1
                urgency = min(1.0, max(0.0, 1.0 - slack / window))
            return (
                item[0],
                miss_flag,
                current,
                -urgency,
                item[1],
                item[2],
                item[3],
                gpu.index,
            )

        chosen = min(pool, key=state_score)
    elif policy == "oracle":
        chosen = min(
            pool,
            key=lambda candidate: (
                future_truth_cost(
                    jobs[candidate[1]],
                    candidate[2],
                    candidate[4],
                    train_stats,
                ),
                candidate[0][0],
                candidate[0][1],
                candidate[0][2],
                candidate[0][3],
                candidate[4].index,
            ),
        )
    elif policy.startswith("trueopt_h"):
        horizon = int(policy.rsplit("h", 1)[1])
        chosen = min(
            pool,
            key=lambda candidate: (
                limited_future_truth_cost(
                    jobs[candidate[1]],
                    candidate[2],
                    candidate[4],
                    train_stats,
                    horizon,
                ),
                candidate[0][0],
                candidate[0][1],
                candidate[0][2],
                candidate[0][3],
                candidate[4].index,
            ),
        )
    elif policy.startswith("predopt_h") and ("_lam" in policy or policy.endswith("_q95") or policy.endswith("_cc")):
        if future_artifacts is None:
            raise ValueError(f"{policy} requires finite-horizon artifacts")
        horizon = int(policy[len("predopt_h"):].split("_", 1)[0])
        if "_lam" in policy:
            lam = int(policy.rsplit("_lam", 1)[1]) / 100.0
            step_cost_choice: Any = lambda step, stats: _mix_step_cost(step, stats, lam)
        elif policy.endswith("_q95"):
            step_cost_choice = _q95_step_cost
        else:
            step_cost_choice = None  # chance-constrained handled below

        def future_steps(node_id: str) -> list[Mapping[str, Any]]:
            row = future_artifacts.get(str(node_id)) or {}
            scenarios = row.get(f"future_h{horizon}") or []
            return (scenarios[0].get("steps") or []) if scenarios else []

        if step_cost_choice is None:
            def predicted_score_cc(
                candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool],
            ) -> tuple[Any, ...]:
                item, job_index, node_id, model_id, gpu, estimate_row, _predicted_fit = candidate
                load = 0.0 if model_id in gpu.resident else float(estimate_row["load_p50_ms"])
                current = float(estimate_row["runtime_p50_ms"]) + load
                steps = future_steps(node_id)[:horizon]
                future_p90 = sum(_mix_step_cost(step, train_stats, 1.0) for step in steps)
                future_mix = sum(_mix_step_cost(step, train_stats, 0.5) for step in steps)
                job = jobs[job_index]
                risk = 0
                if job.deadline_ms is not None:
                    finish_p90 = decision_time_ms + current + future_p90
                    risk = 1 if finish_p90 > float(job.deadline_ms) else 0
                return (
                    float(item[0]),
                    risk,
                    current + future_mix,
                    future_p90,
                    float(item[1]),
                    item[2],
                    0,
                    item[3],
                    gpu.index,
                )

            chosen = min(pool, key=predicted_score_cc)
        else:
            def predicted_score_lam(
                candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool],
            ) -> tuple[float, float, float, float, int, int, str, int]:
                item, _job_index, node_id, model_id, gpu, estimate_row, _predicted_fit = candidate
                load = 0.0 if model_id in gpu.resident else float(estimate_row["load_p50_ms"])
                current = float(estimate_row["runtime_p50_ms"]) + load
                future = sum(step_cost_choice(step, train_stats) for step in future_steps(node_id)[:horizon])
                return (
                    float(item[0]),
                    current + future,
                    future,
                    float(item[1]),
                    item[2],
                    0,
                    item[3],
                    gpu.index,
                )

            chosen = min(pool, key=predicted_score_lam)
    elif policy in SAMESHAPE_POLICIES:
        if future_artifacts is None:
            raise ValueError(f"{policy} requires finite-horizon artifacts")
        horizon = int(policy.split("_h", 1)[1].rsplit("_", 1)[0])
        stat = policy.rsplit("_", 1)[1]

        def sameshape_score(
            candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool],
        ) -> tuple[float, float, float, float, int, int, str, int]:
            item, job_index, node_id, model_id, gpu, estimate_row, _predicted_fit = candidate
            load = 0.0 if model_id in gpu.resident else float(estimate_row["load_p50_ms"])
            current = float(estimate_row["runtime_p50_ms"]) + load
            if stat == "truth":
                future = limited_future_truth_cost(jobs[job_index], node_id, gpu, train_stats, horizon)
            else:
                future = sameshape_future_cost(node_id, future_artifacts, train_stats, horizon, stat)
            return (
                float(item[0]),
                current + future,
                future,
                float(item[1]),
                item[2],
                0,
                item[3],
                gpu.index,
            )

        chosen = min(pool, key=sameshape_score)


        if _DECISION_TRACE["writer"] is not None:
            from tracing.analysis import decision_trace as _dt

            views = []
            owners = {}
            for pool_index, candidate in enumerate(pool):
                item, job_index, node_id, model_id, gpu, estimate_row, _fit = candidate
                resident = model_id in gpu.resident
                cid = _dt.candidate_id(
                    jobs[job_index].job_instance_id, str(node_id), int(gpu.index)
                )
                owners[cid] = (jobs[job_index], gpu)
                views.append(
                    _dt.CandidateView(
                        candidate_id=cid,
                        job_instance_id=jobs[job_index].job_instance_id,
                        node_id=str(node_id),
                        model_id=str(model_id),
                        gpu_index=int(gpu.index),
                        priority=float(item[0]),
                        model_resident=bool(resident),
                        current_runtime_p50_ms=float(estimate_row["runtime_p50_ms"]),
                        current_load_ms=0.0 if resident else float(estimate_row["load_p50_ms"]),
                        legacy_tiebreak_1=float(item[1]),
                        legacy_tiebreak_2=item[2],
                        legacy_tiebreak_3=item[3],
                        pool_index=int(pool_index),
                    )
                )

            def _truth(view: Any, _owners: dict = owners) -> float:
                job, gpu = _owners[view.candidate_id]
                return limited_future_truth_cost(job, view.node_id, gpu, train_stats, horizon)

            item, job_index, node_id, model_id, gpu, _estimate_row, _fit = chosen
            chosen_id = _dt.candidate_id(
                jobs[job_index].job_instance_id, str(node_id), int(gpu.index)
            )
            context = _dt.DecisionContext(
                episode_id=str(_DECISION_TRACE.get("episode_id") or "unknown"),
                decision_index=int(_DECISION_TRACE.get("index", 0)),
                time_ms=float(decision_time_ms),
                trajectory_policy_id=str(policy),
                candidates=views,
                truth_future_cost=(_truth if _DECISION_TRACE["truth"] is not None else None),
            )
            # strict guard, per decision: the reconstructed key must equal the live
            # key for every candidate, and the reconstructed winner must equal the
            # candidate the simulator actually committed to.  The ON/OFF summary
            # guard only proves tracing did not change behaviour; it cannot prove the
            # trace explains the real choice.
            for view in views:
                live = sameshape_score(next(c for c in pool
                                            if c[2] == view.node_id and c[4].index == view.gpu_index))
                rebuilt = _dt.build_scheduler_key(view, float(live[2]))
                if tuple(live) != tuple(rebuilt):
                    raise AssertionError(
                        "trace key mismatch on %s: live=%r rebuilt=%r"
                        % (view.candidate_id, tuple(live), tuple(rebuilt))
                    )
            scored = _dt.score_decision(context, _DECISION_TRACE["methods"])
            trajectory_method = _DECISION_TRACE.get("trajectory_method_id")
            if trajectory_method:
                winners = [c["candidate_id"] for c in scored["candidate_records"]
                           if c["scores"][trajectory_method].get("would_choose")]
                if len(winners) != 1 or winners[0] != chosen_id:
                    raise AssertionError(
                        "trace winner %r != live choice %r" % (winners, chosen_id)
                    )
            _DECISION_TRACE["index"] = int(_DECISION_TRACE.get("index", 0)) + 1
            _DECISION_TRACE["writer"].write(
                {
                    "schema_version": _dt.SCHEMA_VERSION,
                    "episode_id": context.episode_id,
                    "decision_id": context.decision_index,
                    "time_ms": context.time_ms,
                    "trajectory_policy_id": context.trajectory_policy_id,
                    "cost_semantics": _dt.COST_SEMANTICS,
                    "competitive_priority": scored["competitive_priority"],
                    "competitive_candidate_count": scored["competitive_candidate_count"],
                    "candidate_count": len(views),
                    "actual_choice": {"candidate_id": chosen_id},
                    "method_timings": scored["method_timings"],
                    "candidates": scored["candidate_records"],
                }
            )
    elif policy == "sameshape_h5_p95_aging":
        if future_artifacts is None:
            raise ValueError(f"{policy} requires finite-horizon artifacts")
        horizon = 5

        def sameshape_aging_score(
            candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool],
        ) -> tuple[Any, ...]:
            """Aging-augmented predicted remaining work over the H=5 chain.

            R_j = current + future is the predicted remaining work of the emitted chain,
            and one millisecond of waiting buys one millisecond of credit, so the units
            match and a sufficiently long wait always promotes a job.  The hard service
            priority remains the first key, so aging can never cross it.  This is a
            project baseline: there is no single source paper for an "SJF + aging"
            formula, and the simulator has no node-level preemption, so this is a
            non-preemptive SRTF / SRPT-inspired ranking rather than true SRPT.
            """

            item, job_index, node_id, model_id, gpu, estimate_row, _predicted_fit = candidate
            load = 0.0 if model_id in gpu.resident else float(estimate_row["load_p50_ms"])
            current = float(estimate_row["runtime_p50_ms"]) + load
            future = sameshape_future_cost(node_id, future_artifacts, train_stats, horizon, "p95")
            remaining = current + future
            # the heap carries the node ready time; make sure it has not drifted from
            # the Job's own bookkeeping, otherwise the aging credit is measuring the
            # wrong clock
            ready_from_job = jobs[job_index].ready_since.get(node_id)
            if ready_from_job is not None and abs(float(item[1]) - float(ready_from_job)) > 1e-9:
                raise ValueError(
                    f"ready time drift for {node_id}: heap={item[1]} job={ready_from_job}"
                )
            wait_age = float(decision_time_ms) - float(item[1])
            return srtf_aging_key(item[0], remaining, wait_age) + (
                float(item[1]),
                item[2],
                item[3],
                gpu.index,
            )

        chosen = min(pool, key=sameshape_aging_score)

    elif policy == "tie_current":
        """Empirical-TIE-adapted: E[X] + beta * CVaR_0.9[X] over a current-node-only
        empirical distribution.

        The distribution comes from the dedicated TIE bank keyed by (model_id, lane)
        only, never from the shared estimate() path and never from resource_keys()
        (which carries sequence_index and would leak workflow position).  A missing
        mean or CVaR raises instead of falling back to a percentile.

        B is a frozen system constant taken from the episode's GPU topology, not the
        number of currently free devices.  The waiting-time decay is applied as the
        paper's multiplicative factor.
        """

        from tracing.analysis.tie_methods import (
            tie_beta,
            tie_queue_length,
            tie_score_for,
            tie_wait_adjust,
        )

        ctx = policy_context if policy_context is not None else {}
        bank = ctx.get("tie_bank")
        if bank is None:
            raise ValueError("tie_current requires policy_context['tie_bank']")
        B = float(ctx.get("tie_B") or 1.0)
        gamma = float(ctx.get("tie_gamma", 0.9))
        tau_ms = float(ctx.get("tie_tau_ms", 30000.0))

        competitive_priority = min(item[0] for item, *_rest in pool)
        L_q = tie_queue_length(pool, competitive_priority)
        beta = tie_beta(L_q, B)

        def tie_score(
            candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool],
        ) -> tuple[Any, ...]:
            item, job_index, node_id, model_id, gpu, estimate_row, _predicted_fit = candidate
            node = jobs[job_index].template.by_id[node_id]
            # the whole score comes from TIE's own bank: no runtime and no load is read
            # from the shared estimate() row
            score = tie_score_for(bank, node, beta, resident=(model_id in gpu.resident))
            wait_ms = max(0.0, float(decision_time_ms) - float(item[1]))
            score = tie_wait_adjust(score, wait_ms, gamma, tau_ms)
            return (
                float(item[0]),
                score,
                float(item[1]),
                item[2],
                item[3],
                gpu.index,
            )

        chosen = min(pool, key=tie_score)

    elif policy == "pythia_completion":
        """Pythia-adapted: the FULL Algorithm 3 priority, not just its completion half.

        Pythia Algorithm 3's base priority is
            ``omega1 * S_completion + omega2 * S_unblock``
        and the local worker re-scores each window with an aging factor on accumulated
        waiting time.  This branch restores BOTH halves (see ``pythia_methods``):

          * ``S_completion = 1 / (1 + V(role))``, V = expected remaining DISTANCE IN STEPS
            over the train-only role-PFA.  No millisecond term is added -- the earlier port
            accumulated milliseconds, which was a unit error.
          * ``S_unblock`` = DownstreamIdleRisk: SUM over the PFA-reachable GPU future roles
            whose train-mapped deployment has NO visible ready/running demand of
            ``1 / E[D(current, a)]`` -- the distance from the CURRENT role to the future
            agent, so a nearer downstream agent is worth more.  The paper reads replica
            queue depth; this simulator has no replica queue, so it uses a queue-demand
            PROXY.  Future roles come from the TRAIN-ONLY PFA and a train-only role->model
            mapping, never from the template.
          * aging: ``S_eff = S_base + lambda * (wait / tau)``, dimensionless.

        The hard service priority stays first (the benchmark substrate contract); Pythia's
        priorities order programs WITHIN a class.  ``omega1``/``omega2``/``lambda``/``tau``
        are pre-registered frozen constants (the paper gives no values).
        """

        profiler = (policy_context or {}).get("pythia_profiler")
        if profiler is None:
            raise ValueError("pythia_completion requires policy_context['pythia_profiler']")

        from tracing.analysis.pythia_profiler import current_role
        from tracing.analysis.pythia_methods import (
            PYTHIA_AGING_SCALE_MS,
            PYTHIA_AGING_WEIGHT,
            PYTHIA_OMEGA1,
            PYTHIA_OMEGA2,
            pythia_base_priority,
            pythia_effective_priority,
            visible_model_demand,
        )

        ctx = policy_context if policy_context is not None else {}
        omega1 = float(ctx.get("pythia_omega1", PYTHIA_OMEGA1))
        omega2 = float(ctx.get("pythia_omega2", PYTHIA_OMEGA2))
        aging_weight = float(ctx.get("pythia_aging_weight", PYTHIA_AGING_WEIGHT))
        aging_scale_ms = float(ctx.get("pythia_aging_scale_ms", PYTHIA_AGING_SCALE_MS))

        # Scheduler-VISIBLE model demand, merged across ALL workflows (a model is not
        # starved just because this workflow is not using it).  Zero visible demand is the
        # queue-demand proxy; residency is deliberately NOT consulted (resident != busy).
        demand = visible_model_demand(jobs)

        def model_has_zero_visible_demand(model_id: str) -> bool:
            return demand.get(str(model_id), 0) == 0

        role_cache = ctx.setdefault("pythia_role", {})

        def pythia_score(
            candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool],
        ) -> tuple[Any, ...]:
            item, job_index, node_id, model_id, gpu, estimate_row, _predicted_fit = candidate
            job = jobs[job_index]
            # Indexed by the CANDIDATE's own role: V(role) is the work remaining AFTER that
            # role, so indexing by the last COMPLETED role would use the wrong state and
            # double-count the current node.
            role = role_cache.get(node_id)
            if role is None:
                role = current_role(job, node_id)
                role_cache[node_id] = role
            base = pythia_base_priority(profiler, role, model_has_zero_visible_demand,
                                        omega1=omega1, omega2=omega2)
            wait_ms = max(0.0, float(decision_time_ms) - float(item[1]))
            effective = pythia_effective_priority(base, wait_ms, aging_weight, aging_scale_ms)
            return (
                float(item[0]),          # the hard service priority stays first
                -effective,              # smaller key wins, so negate S_eff
                float(item[1]),
                item[2],
                item[3],
                gpu.index,
            )

        chosen = min(pool, key=pythia_score)

    elif policy == "llmsched":
        """LLMSched-adapted: posterior uncertainty reduction vs JCT exploitation.

        A policy wrapper, not a static key: ONE epsilon coin per decision selects the
        mode, then the chosen mode ranks the whole candidate pool.

          EXPLOIT -> smallest estimated remaining job duration (JCT/SRTF flavoured)
          EXPLORE -> largest entropy reduction about the remaining structure

        The posterior advances with the job's own completed count, i.e. completed
        stages act as evidence.  The agent's intra-stage ``sample_tasks(r)`` is
        omitted because our execution unit is already indivisible.
        """

        ctx = policy_context or {}
        profiler = ctx.get("llmsched_bn")
        if profiler is None:
            raise ValueError("llmsched requires policy_context['llmsched_bn']")
        rng = ctx.get("llmsched_rng")
        if rng is None:
            raise ValueError("llmsched requires policy_context['llmsched_rng']")
        epsilon = float(ctx.get("llmsched_epsilon", 0.1))

        # v2 front end: a joint Bayesian network over canonical stages, with exact
        # inference and evidence taken from completed stages' OBSERVED durations.
        from tracing.analysis.llmsched_bn import (
            draw_mode,
            current_service_ms as _bn_current,
            evidence_from_completed,
            expected_job_remaining_ms as _bn_job_remaining,
            job_duration_interval_ms as _bn_interval,
            non_overlapping_sets as _bn_sets,
            uncertainty_reduction as _bn_info,
        )
        from tracing.analysis.llmsched_stage import canonical_stage_map

        # ONE coin per decision, not per candidate
        mode = draw_mode(rng, epsilon)

        # Evidence and the node -> stage map are per job and only change when a node
        # completes, so they are cached on the policy context instead of being rebuilt
        # for every candidate.
        stage_maps: dict[Any, Any] = ctx.setdefault("llmsched_stage_maps", {})
        evidence_cache: dict[Any, Any] = ctx.setdefault("llmsched_evidence", {})

        def stage_map_for(job: Any) -> dict[str, str]:
            key = id(job.template)
            cached = stage_maps.get(key)
            if cached is None:
                cached = canonical_stage_map(job.template)
                stage_maps[key] = cached
            return cached

        def evidence_for(job: Any) -> dict[str, str]:
            """Evidence from COMPLETED nodes only, each at its intrinsic duration.

            The simulator is the truth provider for its own workload definition, so a
            finished node's intrinsic runtime is a historical observation.  Using
            ``finish_ms - node_start_ms`` instead would fold in the GPU queue delay
            that this schedule itself produced, which is not what the network was
            trained on.
            """

            key = job.job_instance_id
            done = frozenset(job.completed)
            entry = evidence_cache.get(key)
            if entry is not None and entry[0] == done:
                return entry[1]
            value = evidence_from_completed(
                job, profiler, observed_ms=job.observed_intrinsic_ms)
            evidence_cache[key] = (done, value)
            return value

        # The non-overlapping duration sets are a property of the whole candidate pool at
        # this decision, so they are computed once and shared by every candidate.  They
        # are built AFTER the per-job helpers above, which they depend on.
        # The interval describes the JOB's remaining work, so it is computed once per job
        # and does not depend on which candidate is being scored.  It covers every
        # model-side unresolved stage rather than one candidate's correlated descendants,
        # because Algorithm 1 is comparing jobs against each other.
        # Ready candidates are KNOWN to exist, and each job's interval must reflect that.
        # The set is per job (a job can have several ready nodes), collected here because
        # the interval is a property of the job at this decision, not of one candidate.
        _ready_stages: dict[Any, set] = {}
        for candidate in pool:
            _ji = candidate[1]
            _stage = stage_map_for(jobs[_ji])[candidate[2]]
            _ready_stages.setdefault(_ji, set()).add(_stage)
        duration_intervals: dict[Any, Any] = {}
        for _ji, _known in _ready_stages.items():
            duration_intervals[_ji] = _bn_interval(
                profiler, evidence_for(jobs[_ji]), sorted(_known))
        group_index = _bn_sets(duration_intervals)

        def _group_of(job_index: int) -> int:
            return int(group_index.get(job_index, 0))

        def llmsched_score(
            candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool],
        ) -> tuple[Any, ...]:
            item, job_index, node_id, model_id, gpu, estimate_row, _predicted_fit = candidate
            job = jobs[job_index]
            stage = stage_map_for(job)[node_id]
            evidence = evidence_for(job)
            # Y comes from the learned network plus what has actually been observed.
            # It is NOT the template's unexecuted suffix: the exact stages and
            # dependencies of a job are the structure uncertainty this baseline exists
            # to resolve, so reading them would leak the answer while still producing
            # perfectly reasonable numbers.
            if mode == "EXPLORE":
                # Algorithm 1 orders candidates by uncertainty reduction only WITHIN a
                # set of jobs whose duration intervals overlap.  Across non-overlapping
                # sets the ordering is already determined by the bounds, so a
                # high-variance job must not jump ahead of a provably shorter one.  The
                # group index is therefore the primary term and R(X) only breaks ties
                # inside a group.
                return (
                    float(item[0]),
                    float(_group_of(job_index)),
                    -_bn_info(profiler, stage, evidence),
                    float(item[1]),
                    item[2],
                    item[3],
                    gpu.index,
                )
            # No separate load term: the BN's duration states are intrinsic ``runtime_ms``,
            # which already CONTAINS the cold-load time, so adding the estimate row's
            # load here would double-count it.  (The previous version computed a load and
            # then never used it; the remaining-work term is the whole-job estimator.)
            current = _bn_current(profiler, stage, evidence)
            primary = current + _bn_job_remaining(profiler, evidence, stage)
            return (
                float(item[0]),
                primary,
                float(item[1]),
                item[2],
                item[3],
                gpu.index,
            )

        chosen = min(pool, key=llmsched_score)

    elif policy == "latency_aware":
        """Delegate to the independent Latency-Aware scheduler.

        This arm does not use the shared ``min(pool, key=...)`` idiom at all: the
        ranking, the Eq (7) timeline prediction, the Eq (11) admission-memory check
        and the Eq (12) lexicographic key all live in
        ``tracing.analysis.latency_aware_scheduler``.  The branch below only adapts
        the already-built candidate pool to that module and applies its action, so
        every other arm stays byte-identical.
        """

        from tracing.analysis.latency_aware_fusion import maximal_fusible_chains
        from tracing.analysis.latency_aware_lifecycle import resolved_graph_view
        from tracing.analysis import latency_aware_scheduler as la

        ctx = policy_context if policy_context is not None else {}
        cache = ctx.setdefault("_fuse_cache", {})

        # The duration and memory the scheduler plans with must be PREDICTIONS.  An
        # earlier version let it read node.compute_ms, which is the node's actual
        # intrinsic compute time, so the arm was reading the answer it was supposed to
        # predict and its latency figures were circular.  Failing closed here rather than
        # letting a missing predictor silently fall back to a truth field is the point.
        la_predictor = ctx.get("latency_aware_predictor")
        if la_predictor is None:
            raise ValueError(
                "latency_aware requires policy_context['latency_aware_predictor']; "
                "without it the scheduler has no legitimate duration or memory source"
            )

        def chains_for(job: Any) -> Dict[str, tuple[str, ...]]:
            # The CONTRACT reads the resolved logical window, not the realized template:
            # a chain may not cross a successor whose control result has not resolved.
            # The cache key therefore includes the job's current resolved view, because
            # the same template yields different chains at different frontier states.
            view = frozenset(resolved_graph_view(job))
            key = (str(job.template.template_id), view)
            cached = cache.get(key)
            if cached is None:
                cached = {}
                for chain in maximal_fusible_chains(job.template, visible_ids=view):
                    if chain.length > 1:
                        cached[chain.node_ids[0]] = chain.node_ids
                cache[key] = cached
            return cached

        ctx.pop("_fused_chain", None)
        action = la.choose_action(pool, jobs, decision_time_ms, chains_for=chains_for,
                                  predictor=la_predictor)
        if action is None:
            # nothing is admissible on memory grounds; the caller's prefetch hook will
            # try to prepare a near-ready deployment instead
            chosen = min(pool, key=lambda c: (float(c[0][0]), float(c[0][1]),
                                              c[0][2], c[0][3], c[4].index))
        else:
            chosen = action["candidate"]
            if action["fused"]:
                ctx["_fused_chain"] = list(action["fused"])

    elif policy == "agentix":
        """Agentix-adapted: non-clairvoyant program-level attained-service priority.

        The priority of a ready call is the service its PROGRAM has already attained -- the
        sum of its COMPLETED calls' runtimes (PLAS) or its longest completed critical path
        (ATLAS, ``policy_context['agentix_mode']``).  Lower attained service is scheduled
        first, which is the paper's head-of-line-blocking fix: a long program's later calls
        stop starving the short programs behind them.

        Non-clairvoyant by construction: only ``job.completed`` is read.  The template's
        unexecuted suffix and any predicted value are never consulted, so this arm cannot
        degenerate into a clairvoyant SRPT.  The hard service priority stays first; the
        substrate keeps placement, admission memory and residency.

        ``policy_context['agentix_mode']``:
          * ``plas`` (default, formal) / ``atlas`` -- continuous attained service;
          * ``discrete`` -- a NON-PAPER sensitivity variant: discretised, non-preemptive
            queueing with program-level anti-starvation (see ``agentix_methods``).  It is
            NOT the paper's preemptive MLFQ scheduler.
        """

        from tracing.analysis.agentix_methods import (
            DEFAULT_QUEUE_EDGES_MS,
            discrete_priority_index,
            observed_gpu_service_of,
            program_priority_ms,
        )

        ctx = policy_context if policy_context is not None else {}
        mode = str(ctx.get("agentix_mode", "plas"))

        if mode == "discrete":
            # NON-PAPER sensitivity variant: discretised, non-preemptive queueing with
            # program-level anti-starvation (see agentix_methods).  The formal arm is 'plas'.
            from tracing.analysis.agentix_methods import is_starving

            # Queue edges: train-only quantile calibration supplied by the caller.  The
            # paper publishes no boundaries, so the fallback is only for fixtures/tests.
            edges = tuple(ctx.get("agentix_queue_edges", DEFAULT_QUEUE_EDGES_MS))

            # Activation telemetry, recorded once per DECISION over the distinct ready calls
            # (the pool replicates a call per free device, so counting pool entries would
            # multiply it).  These are mechanism-aliveness diagnostics, NOT performance
            # metrics, and must never be used to tune the arm.
            queue_counts: Dict[int, int] = ctx.setdefault("agentix_queue_counts", {})
            tally = ctx.setdefault("agentix_telemetry", {"decisions": 0, "calls": 0, "promotions": 0})
            attained_cache: Dict[int, float] = {}
            queue_of: Dict[tuple, int] = {}
            seen_calls = set()
            for candidate in pool:
                item, job_index, node_id, _model_id, _gpu, _row, _fit = candidate
                key = (job_index, node_id)
                if key in seen_calls:
                    continue
                seen_calls.add(key)
                job = jobs[job_index]
                if job_index not in attained_cache:
                    attained_cache[job_index] = program_priority_ms(
                        job, job.template, observed_gpu_service_of(job), mode="plas")
                attained = attained_cache[job_index]
                wait_ms = max(0.0, float(decision_time_ms) - float(item[1]))
                starving = is_starving(wait_ms, attained)
                index = 0 if starving else discrete_priority_index(attained, wait_ms, edges)
                queue_of[key] = index
                queue_counts[index] = queue_counts.get(index, 0) + 1
                tally["calls"] += 1
                if starving:
                    tally["promotions"] += 1
            tally["decisions"] += 1

            def agentix_score(
                candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool],
            ) -> tuple[Any, ...]:
                item, job_index, node_id, model_id, gpu, estimate_row, _predicted_fit = candidate
                queue = queue_of.get((job_index, node_id), 0)
                return (
                    float(item[0]),
                    queue,                   # K-queue index (0 is top); anti-starvation promotes to 0
                    float(item[1]),
                    item[2],
                    item[3],
                    gpu.index,
                )
        else:
            def agentix_score(
                candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool],
            ) -> tuple[Any, ...]:
                item, job_index, node_id, model_id, gpu, estimate_row, _predicted_fit = candidate
                job = jobs[job_index]
                attained = program_priority_ms(
                    job, job.template, observed_gpu_service_of(job), mode=mode)
                return (
                    float(item[0]),
                    attained,
                    float(item[1]),
                    item[2],
                    item[3],
                    gpu.index,
                )

        chosen = min(pool, key=agentix_score)

    elif policy == "predopt_h5_risk":
        if future_artifacts is None:
            raise ValueError(f"{policy} requires finite-horizon artifacts")

        def predicted_score_risk(
            candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool],
        ) -> tuple[float, float, float, float, int, int, str, int]:
            item, _job_index, node_id, model_id, gpu, estimate_row, _predicted_fit = candidate
            load = 0.0 if model_id in gpu.resident else float(estimate_row["load_p50_ms"])
            current = float(estimate_row["runtime_p50_ms"]) + load
            row = future_artifacts.get(str(node_id)) or {}
            scenarios = row.get("future_h5") or []
            steps = (scenarios[0].get("steps") or []) if scenarios else []
            future = sum(_risk_step_cost(step, train_stats) for step in steps[:5])
            return (
                float(item[0]),
                current + future,
                future,
                float(item[1]),
                item[2],
                0,
                item[3],
                gpu.index,
            )

        chosen = min(pool, key=predicted_score_risk)
    elif policy == "predopt_h5_exp":
        if future_artifacts is None:
            raise ValueError(f"{policy} requires finite-horizon artifacts")

        def predicted_score_exp(
            candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool],
        ) -> tuple[float, float, float, float, int, int, str, int]:
            item, _job_index, node_id, model_id, gpu, estimate_row, _predicted_fit = candidate
            load = 0.0 if model_id in gpu.resident else float(estimate_row["load_p50_ms"])
            current = float(estimate_row["runtime_p50_ms"]) + load
            row = future_artifacts.get(str(node_id)) or {}
            scenarios = row.get("future_h5") or []
            steps = (scenarios[0].get("steps") or []) if scenarios else []
            future = 0.0
            for step in steps:
                expected = step.get("expected_cost")
                future += float(expected) if expected is not None else _step_estimate_cost(step, train_stats)
            return (
                float(item[0]),
                current + future,
                future,
                float(item[1]),
                item[2],
                0,
                item[3],
                gpu.index,
            )

        chosen = min(pool, key=predicted_score_exp)
    elif policy == "predopt_h5_surv":
        if future_artifacts is None:
            raise ValueError(f"{policy} requires finite-horizon artifacts")

        def predicted_score_surv(
            candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool],
        ) -> tuple[float, float, float, float, int, int, str, int]:
            item, _job_index, node_id, model_id, gpu, estimate_row, _predicted_fit = candidate
            load = 0.0 if model_id in gpu.resident else float(estimate_row["load_p50_ms"])
            current = float(estimate_row["runtime_p50_ms"]) + load
            row = future_artifacts.get(str(node_id)) or {}
            scenarios = row.get("future_h5") or []
            steps = (scenarios[0].get("steps") or []) if scenarios else []
            future = 0.0
            for step in steps[:5]:
                survival = float(step.get("survival_probability", 1.0))
                future += survival * _step_estimate_cost(step, train_stats)
            return (
                float(item[0]),
                current + future,
                future,
                float(item[1]),
                item[2],
                0,
                item[3],
                gpu.index,
            )

        chosen = min(pool, key=predicted_score_surv)
    elif policy == "oracle_topology_h5_tab":
        chosen = min(
            pool,
            key=lambda candidate: (
                limited_future_table_cost(
                    jobs[candidate[1]],
                    candidate[2],
                    candidate[4],
                    train_stats,
                    5,
                ),
                candidate[0][0],
                candidate[0][1],
                candidate[0][2],
                candidate[0][3],
                candidate[4].index,
            ),
        )
    elif policy == "predopt_h5_oc":
        if future_artifacts is None:
            raise ValueError(f"{policy} requires finite-horizon artifacts")

        def predicted_score_oc(
            candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool],
        ) -> tuple[float, float, float, float, int, int, str, int]:
            item, job_index, node_id, model_id, gpu, _estimate_row, _predicted_fit = candidate
            node = jobs[job_index].template.by_id[node_id]
            load = 0.0 if model_id in gpu.resident else number(node.load_ms, 0.0)
            current = number(node.runtime_ms, 0.0) + load
            future = predicted_future_cost(node_id, future_artifacts, train_stats, 5)
            return (
                float(item[0]),
                current + future,
                future,
                float(item[1]),
                item[2],
                0,
                item[3],
                gpu.index,
            )

        chosen = min(pool, key=predicted_score_oc)
    elif policy.startswith("predopt_h") and (policy.endswith("_jres") or policy.endswith("_jrt")):
        if future_artifacts is None:
            raise ValueError(f"{policy} requires finite-horizon artifacts")
        suffix = "_jres" if policy.endswith("_jres") else "_jrt"
        horizon = int(policy[len("predopt_h") : -len(suffix)])
        step_cost = _jres_step_cost if suffix == "_jres" else _jrt_step_cost

        def predicted_score_jres(
            candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool],
        ) -> tuple[float, float, float, float, int, int, str, int]:
            item, _job_index, node_id, model_id, gpu, estimate_row, _predicted_fit = candidate
            load = 0.0 if model_id in gpu.resident else float(estimate_row["load_p50_ms"])
            current = float(estimate_row["runtime_p50_ms"]) + load
            future = predicted_future_cost_jres(node_id, future_artifacts, train_stats, horizon, step_cost)
            return (
                float(item[0]),
                current + future,
                future,
                float(item[1]),
                item[2],
                0,
                item[3],
                gpu.index,
            )

        chosen = min(pool, key=predicted_score_jres)
    elif policy in {"predopt_h5_r50", "predopt_h5_r90", "predopt_h5_r95", "predopt_h5_r50k"}:
        if future_artifacts is None:
            raise ValueError(f"{policy} requires finite-horizon artifacts")
        runtime_token = policy.rsplit("_", 1)[1]
        if runtime_token == "r50k":
            tau_key = "p50"
            scale = _RUNTIME_P50_SCALE
        else:
            tau_key = {"r50": "p50", "r90": "p90", "r95": "p95"}[runtime_token]
            scale = 1.0

        def predicted_score_runtime_only(
            candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool],
        ) -> tuple[float, float, float, float, int, int, str, int]:
            item, _job_index, node_id, model_id, gpu, estimate_row, _predicted_fit = candidate
            load = 0.0 if model_id in gpu.resident else float(estimate_row["load_p50_ms"])
            current = float(estimate_row["runtime_p50_ms"]) + load
            row = future_artifacts.get(str(node_id)) or {}
            scenarios = row.get("future_h5") or []
            steps = (scenarios[0].get("steps") or []) if scenarios else []
            future = scale * sum(_runtime_only_step_cost(step, train_stats, tau_key) for step in steps[:5])
            return (
                float(item[0]),
                current + future,
                future,
                float(item[1]),
                item[2],
                0,
                item[3],
                gpu.index,
            )

        chosen = min(pool, key=predicted_score_runtime_only)
    elif policy in {"predopt_h5_rt95", "predopt_h5_ld95"}:
        if future_artifacts is None:
            raise ValueError(f"{policy} requires finite-horizon artifacts")
        runtime_lam, load_lam = (1.0, 0.0) if policy.endswith("_rt95") else (0.0, 1.0)

        def predicted_score_split(
            candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool],
        ) -> tuple[float, float, float, float, int, int, str, int]:
            item, _job_index, node_id, model_id, gpu, estimate_row, _predicted_fit = candidate
            load = 0.0 if model_id in gpu.resident else float(estimate_row["load_p50_ms"])
            current = float(estimate_row["runtime_p50_ms"]) + load
            row = future_artifacts.get(str(node_id)) or {}
            scenarios = row.get("future_h5") or []
            steps = (scenarios[0].get("steps") or []) if scenarios else []
            future = sum(
                _split95_step_cost(step, train_stats, runtime_lam, load_lam) for step in steps[:5]
            )
            return (
                float(item[0]),
                current + future,
                future,
                float(item[1]),
                item[2],
                0,
                item[3],
                gpu.index,
            )

        chosen = min(pool, key=predicted_score_split)
    elif policy in {"predopt_h5_adapt", "predopt_h5_adapt_q"}:
        if future_artifacts is None:
            raise ValueError(f"{policy} requires finite-horizon artifacts")
        use_pressure = policy.endswith("_q")

        def adapt_future_cost(node_id: str, lam: float) -> float:
            row = future_artifacts.get(str(node_id)) or {}
            scenarios = row.get("future_h5") or []
            steps = (scenarios[0].get("steps") or []) if scenarios else []
            return sum(_mix95_step_cost(step, train_stats, lam) for step in steps[:5])

        def adapt_lambda(job: Job, node_id: str, estimate_row: Mapping[str, Any]) -> float:
            lam = 0.5
            if job.deadline_ms is not None:
                window = max(1.0, float(job.deadline_ms) - float(job.arrival_ms))
                predicted = decision_time_ms + float(estimate_row["runtime_p50_ms"]) + adapt_future_cost(node_id, 0.5)
                slack_ratio = (float(job.deadline_ms) - predicted) / window
                if slack_ratio < 0.05:
                    lam = 1.0
                elif slack_ratio < 0.25:
                    lam = 0.75
                else:
                    lam = 0.4
            if use_pressure:
                pressure = len(ready_items) / max(1, len(free_gpus))
                lam = min(1.0, lam + 0.25 * min(1.0, pressure / 4.0))
            return lam

        def predicted_score_adapt(
            candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool],
        ) -> tuple[float, float, float, float, int, int, str, int]:
            item, job_index, node_id, model_id, gpu, estimate_row, _predicted_fit = candidate
            load = 0.0 if model_id in gpu.resident else float(estimate_row["load_p50_ms"])
            current = float(estimate_row["runtime_p50_ms"]) + load
            lam = adapt_lambda(jobs[job_index], node_id, estimate_row)
            future = adapt_future_cost(node_id, lam)
            return (
                float(item[0]),
                current + future,
                future,
                float(item[1]),
                item[2],
                0,
                item[3],
                gpu.index,
            )

        chosen = min(pool, key=predicted_score_adapt)
    elif policy.startswith("predopt_h") and ("_scen128_" in policy or "_comon128_" in policy):
        if future_artifacts is None:
            raise ValueError(f"{policy} requires finite-horizon artifacts")
        horizon = int(policy[len("predopt_h"):].split("_", 1)[0])
        if "_comon128_fixedl_k" in policy:
            token, coupling, fixed_length = "_comon128_fixedl_k", "comonotone", True
        elif "_comon128_k" in policy:
            token, coupling, fixed_length = "_comon128_k", "comonotone", False
        elif "_scen128_fixedl_k" in policy:
            token, coupling, fixed_length = "_scen128_fixedl_k", "independent", True
        else:
            token, coupling, fixed_length = "_scen128_k", "independent", False
        kappa = int(policy.rsplit(token, 1)[1]) / 100.0

        def predicted_score_scenario_v2(
            candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool],
        ) -> tuple[float, float, float, float, int, int, str, int]:
            item, _job_index, node_id, model_id, gpu, estimate_row, _predicted_fit = candidate
            load = 0.0 if model_id in gpu.resident else float(estimate_row["load_p50_ms"])
            current = float(estimate_row["runtime_p50_ms"]) + load
            future = scenario_v2_risk_cost(
                node_id,
                future_artifacts,
                train_stats,
                kappa,
                horizon,
                coupling=coupling,
                length_mode="argmax" if fixed_length else "sample",
            )
            return (
                float(item[0]),
                current + future,
                future,
                float(item[1]),
                item[2],
                0,
                item[3],
                gpu.index,
            )

        chosen = min(pool, key=predicted_score_scenario_v2)
    elif policy.startswith("predopt_h") and "_comon_k" in policy:
        if future_artifacts is None:
            raise ValueError(f"{policy} requires finite-horizon artifacts")
        horizon = int(policy[len("predopt_h"):].split("_", 1)[0])
        kappa = int(policy.rsplit("_comon_k", 1)[1]) / 100.0

        def predicted_score_comon(
            candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool],
        ) -> tuple[float, float, float, float, int, int, str, int]:
            item, _job_index, node_id, model_id, gpu, estimate_row, _predicted_fit = candidate
            load = 0.0 if model_id in gpu.resident else float(estimate_row["load_p50_ms"])
            current = float(estimate_row["runtime_p50_ms"]) + load
            future = scenario_cvar_future_cost(
                node_id, future_artifacts, train_stats, kappa, horizon, coupling="comonotone"
            )
            return (
                float(item[0]),
                current + future,
                future,
                float(item[1]),
                item[2],
                0,
                item[3],
                gpu.index,
            )

        chosen = min(pool, key=predicted_score_comon)
    elif policy.startswith("predopt_h") and "_scen_k" in policy:
        if future_artifacts is None:
            raise ValueError(f"{policy} requires finite-horizon artifacts")
        horizon = int(policy[len("predopt_h"):].split("_", 1)[0])
        kappa = int(policy.rsplit("_scen_k", 1)[1]) / 100.0

        def predicted_score_scen(
            candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool],
        ) -> tuple[float, float, float, float, int, int, str, int]:
            item, _job_index, node_id, model_id, gpu, estimate_row, _predicted_fit = candidate
            load = 0.0 if model_id in gpu.resident else float(estimate_row["load_p50_ms"])
            current = float(estimate_row["runtime_p50_ms"]) + load
            future = scenario_cvar_future_cost(node_id, future_artifacts, train_stats, kappa, horizon)
            return (
                float(item[0]),
                current + future,
                future,
                float(item[1]),
                item[2],
                0,
                item[3],
                gpu.index,
            )

        chosen = min(pool, key=predicted_score_scen)
    elif policy.startswith("predopt_h"):
        if future_artifacts is None:
            raise ValueError(f"{policy} requires finite-horizon artifacts")
        horizon = int(policy.rsplit("h", 1)[1])

        def predicted_score(
            candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool],
        ) -> tuple[float, float, float, float, int, int, str, int]:
            item, _job_index, node_id, model_id, gpu, estimate_row, _predicted_fit = candidate
            load = 0.0 if model_id in gpu.resident else float(estimate_row["load_p50_ms"])
            current = float(estimate_row["runtime_p50_ms"]) + load
            future = predicted_future_cost(node_id, future_artifacts, train_stats, horizon)
            return (
                float(item[0]),
                current + future,
                future,
                float(item[1]),
                item[2],
                0,
                item[3],
                gpu.index,
            )

        chosen = min(pool, key=predicted_score)
    elif policy.startswith("predopt_v2_h"):
        if future_artifacts is None:
            raise ValueError(f"{policy} requires finite-horizon artifacts")
        horizon = int(policy.rsplit("h", 1)[1])

        def predicted_score_v2(
            candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool],
        ) -> tuple[float, float, float, float, int, str, int]:
            item, _job_index, node_id, model_id, gpu, estimate_row, _predicted_fit = candidate
            load = 0.0 if model_id in gpu.resident else float(estimate_row["load_p50_ms"])
            current = float(estimate_row["runtime_p50_ms"]) + load
            future = predicted_future_cost(node_id, future_artifacts, train_stats, horizon)
            memory_ratio = float(estimate_row["memory_p95_mb"]) / max(1.0, float(gpu.capacity_mb))
            # P0 audit objective: priority is a soft penalty; current and
            # finite-horizon costs are normalized to the same 100-second
            # scale so future information can change cross-priority order.
            total = (
                PREDOPT_V2_PRIORITY_WEIGHT * float(item[0])
                + current / 100000.0
                + PREDOPT_V2_FUTURE_WEIGHT * future / 100000.0
                + PREDOPT_V2_MEMORY_WEIGHT * memory_ratio
            )
            return (
                total,
                future / 100000.0,
                current / 100000.0,
                float(item[1]),
                item[2],
                item[3],
                gpu.index,
            )

        chosen = min(pool, key=predicted_score_v2)
    elif policy == "optimizer_0":
        # Mathematical baseline: minimize current estimated completion plus a
        # soft peak-memory tie-breaker, with no future information.
        chosen = min(
            pool,
            key=lambda candidate: (
                candidate[0][0],
                float(candidate[5]["runtime_p50_ms"]) + float(candidate[5]["load_p50_ms"]),
                float(candidate[5]["memory_p95_mb"]),
                float(candidate[0][1]),
                candidate[0][2],
                candidate[3],
                candidate[4].index,
            ),
        )
    else:
        def myopic_score(
            candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool],
        ) -> tuple[float, float, float, float, int, int, str, int]:
            item, _job_index, _node_id, model_id, gpu, estimate_row, _predicted_fit = candidate
            load = 0.0 if model_id in gpu.resident else float(estimate_row["load_p50_ms"])
            reuse = 0.0 if model_id in gpu.resident else 1.0
            return (
                float(item[0]),
                float(estimate_row["runtime_p50_ms"]) + load,
                reuse,
                float(item[1]),
                item[2],
                0,
                item[3],
                gpu.index,
            )
        chosen = min(pool, key=myopic_score)

    return chosen[0], chosen[4].index, chosen[5], len(candidates), len(fitting)


def choose_action_with_batch(
    policy: str,
    ready_items: Sequence[tuple[float, int, int, str]],
    jobs: Sequence[Job],
    free_gpus: Sequence[GPU],
    train_stats: Mapping[str, Mapping[str, Any]],
    round_robin_cursor: int,
    extension_config: Mapping[str, Any],
) -> tuple[tuple[float, int, int, str], int, dict[str, Any], int, int, int]:
    """Choose a ready node, GPU, and measured per-node batch configuration.

    This is deliberately separate from the legacy action contract.  A batch
    action is ``(ready_node, free_gpu, batch_size)`` only when the opt-in
    experiment supplies an explicit measured profile; all other nodes retain
    batch size one.  It does not claim cross-job dynamic batching.
    """

    if not free_gpus:
        raise RuntimeError("choose_action_with_batch called without a free GPU")
    candidates: list[tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool, int]] = []
    for item in ready_items:
        _priority, _ready_time, job_index, node_id = item
        job = jobs[job_index]
        node = job.template.by_id[node_id]
        if job.node_state.get(node_id) != "ready" or node.lane != "gpu":
            continue
        _require_resource_applicable(node, job.template.template_id)
        for gpu in free_gpus:
            for batch_size in _batch_options(node, extension_config):
                row = estimate_for_batch(node, train_stats, batch_size, extension_config)
                predicted_fit = fits_gpu(gpu, node, row)
                candidates.append((item, job_index, node_id, node.model_id, gpu, row, predicted_fit, batch_size))
    if not candidates:
        raise RuntimeError("choose_action_with_batch called without a ready GPU node")
    fitting = [candidate for candidate in candidates if candidate[6]]
    pool = fitting or candidates
    base_policy = policy[: -len("_preempt")] if policy.endswith("_preempt") else policy

    def batch_cost(candidate: tuple[tuple[float, int, int, str], int, str, str, GPU, dict[str, Any], bool, int]) -> tuple[Any, ...]:
        item, _job_index, _node_id, model_id, gpu, row, _fit, batch_size = candidate
        return (
            float(item[0]),
            float(row["runtime_p50_ms"]) + float(row["load_p50_ms"]),
            float(row["memory_p95_mb"]),
            float(item[1]),
            item[2],
            model_id,
            gpu.index,
            batch_size,
        )

    if base_policy in {"batch_max_fit", "batch_throughput"}:
        chosen = min(
            pool,
            key=lambda candidate: (
                float(candidate[0][0]),
                -candidate[7],
                float(candidate[5]["runtime_p50_ms"]) / max(1, candidate[7]),
                float(candidate[5]["memory_p95_mb"]),
                float(candidate[0][1]),
                candidate[0][2],
                candidate[4].index,
            ),
        )
    elif base_policy == "batch_round_robin":
        chosen = min(
            pool,
            key=lambda candidate: (
                candidate[0][0],
                candidate[0][1],
                candidate[0][2],
                (candidate[4].index - round_robin_cursor) % max(1, len(free_gpus)),
                candidate[4].index,
                candidate[7],
            ),
        )
    else:
        # batch_myopic: same current-cost objective as Myopic, with batch as
        # an additional legal action and memory as a deterministic tie-break.
        chosen = min(pool, key=batch_cost)
    return chosen[0], chosen[4].index, chosen[5], len(candidates), len(fitting), chosen[7]

def simulator_view(node: Node, estimate_row: Mapping[str, Any], now: float, gpu_index: int | None) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "lane": node.lane,
        "model_id": node.model_id,
        "runtime_p50_ms": estimate_row["runtime_p50_ms"],
        "runtime_p90_ms": estimate_row["runtime_p90_ms"],
        "load_p50_ms": estimate_row["load_p50_ms"],
        "peak_memory_p95_mb": estimate_row["memory_p95_mb"],
        "batch_size": int(estimate_row.get("batch_size") or node.batch_size),
        "decision_time_ms": now,
        "gpu_index": gpu_index,
    }


def scheduler_state_at_decision(
    episode: Mapping[str, Any],
    jobs: Sequence[Job],
    gpus: Sequence[GPU],
    ready_items: Sequence[tuple[float, int, int, str]],
    train_stats: Mapping[str, Mapping[str, Any]],
    future_provider: FutureProvider,
    horizon: int,
    decision_index: int,
    now: float,
) -> dict[str, Any]:
    """Build the scheduler-only view without copying node truth fields.

    This adapter is called by the existing event loop immediately before a
    GPU dispatch.  The old policy ranking remains unchanged; future-aware
    policies will consume this same payload in R7-2.
    """

    ready_nodes: list[dict[str, Any]] = []
    resources: dict[tuple[str, str], ResourceEstimate] = {}
    for _priority, ready_since, job_index, node_id in ready_items:
        job = jobs[job_index]
        node = job.template.by_id[node_id]
        if job.node_state.get(node_id) != "ready" or node.lane != "gpu":
            continue
        _require_resource_applicable(node, job.template.template_id)
        row = estimate(node, train_stats)
        ready_nodes.append(
            {
                "job_instance_id": job.job_instance_id,
                "node_id": node.node_id,
                "sequence_index": node.sequence_index,
                "role": node.role,
                "action_family": node.action_family,
                "model_id": node.model_id,
                "lane": node.lane,
                "ready_since_ms": ready_since,
            }
        )
        resources[(job.job_instance_id, node.node_id)] = ResourceEstimate(
            runtime_p50_ms=float(row["runtime_p50_ms"]),
            runtime_p90_ms=float(row["runtime_p90_ms"]),
            load_p50_ms=float(row["load_p50_ms"]),
            peak_memory_p95_mb=float(row["memory_p95_mb"]),
        )
    gpu_rows = [
        {
            "index": gpu.index,
            "capacity_mb": gpu.capacity_mb,
            "busy": gpu.active_node is not None,
            "resident_models": tuple(sorted(gpu.resident)),
        }
        for gpu in gpus
    ]
    completed_prefix = {
        job.job_instance_id: tuple(
            node_id
            for node_id in sorted(
                job.completed,
                key=lambda value: job.template.by_id[value].sequence_index,
            )
        )
        for job in jobs
    }
    state = build_scheduler_state(
        episode_id=str(episode["episode_id"]),
        decision_index=decision_index,
        time_ms=now,
        ready_nodes=ready_nodes,
        completed_prefix=completed_prefix,
        gpus=gpu_rows,
        resource_predictions=resources,
        future_provider=future_provider,
        horizon=horizon,
    )
    return state.to_scheduler_dict()


def simulate_episode(
    episode: Mapping[str, Any],
    templates: Mapping[str, Template],
    policy: str,
    *,
    future_provider: FutureProvider | None = None,
    future_horizon: int = 0,
    future_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    train_stats: Mapping[str, Mapping[str, Any]] | None = None,
    collect_events: bool = True,
    rl_context: dict[str, Any] | None = None,
    policy_context: dict[str, Any] | None = None,
    extension_config: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # the scheduling arms may hand state back through policy_context (for example the
    # resolved fused unit), so it must always be a real dict the commit path can read
    if policy_context is None:
        policy_context = {}
    # TIE's B is a frozen system constant: the episode's GPU service concurrency,
    # not the number of currently free devices
    if policy == "tie_current" and "tie_B" not in policy_context:
        policy_context["tie_B"] = float(len(episode.get("gpu_topology_mb") or []))

    """Simulate one episode, with opt-in batch/prefetch/preemption features.

    ``extension_config`` is intentionally absent from the default path so all
    sealed R7 results retain their original semantics.  The extension is used
    only by a dated experiment runner and records every non-default action in
    the event/result schema.
    """

    extension_config = dict(extension_config or {})
    if policy_context is None and (policy.startswith("pred_mpc_h") or policy == "risk_aware"):
        policy_context = {}
    jobs = build_jobs(episode, templates)
    if _DECISION_TRACE["writer"] is not None:
        # choose_action has no access to the episode record, so publish the id here
        _DECISION_TRACE["episode_id"] = str(episode["episode_id"])
        _DECISION_TRACE["index"] = 0
    train_stats = train_stats if train_stats is not None else train_resource_stats(templates)
    future_provider = future_provider or FutureProvider("none")
    truth_rows = {
        (job.job_instance_id, node.node_id): ExecutionTruth(
            runtime_ms=node.runtime_ms,
            load_ms=number(node.load_ms),
            workspace_peak_mb=number(node.workspace_peak_mb),
            status=node.status,
        )
        for job in jobs
        for node in job.template.nodes
    }
    truth_provider = ExecutionTruthProvider(truth_rows)
    capacities = [float(value) for value in episode.get("gpu_topology_mb") or []]
    validate_transition_profile(extension_config, capacities)
    gpus = [GPU(index, capacity) for index, capacity in enumerate(capacities)]
    for gpu, resident_models in zip(gpus, episode.get("initial_residency_hint") or []):
        for model in resident_models:
            memory = None
            for template in templates.values():
                for node in template.nodes:
                    if node.model_id == model and node.resident_model_mb is not None:
                        memory = node.resident_model_mb
                        break
                if memory is not None:
                    break
            if memory is None:
                raise ValueError(f"unknown initial resident model {model}")
            if sum(gpu.resident.values()) + memory > gpu.capacity_mb:
                raise ValueError(f"initial residency exceeds GPU capacity {gpu.index}")
            gpu.resident[model] = memory
    arrival_heap = [(job.arrival_ms, index) for index, job in enumerate(jobs)]
    heapq.heapify(arrival_heap)
    finish_heap: list[tuple[float, int, int, str, int | None, str]] = []
    ready: list[tuple[float, int, int, str]] = []
    events: list[dict[str, Any]] = []
    now = 0.0
    sequence = 0
    rr_cursor = 0
    active_jobs: set[int] = set()
    wait_logged: set[tuple[int, str]] = set()
    decision_index = 0
    batch_choice_counts: Counter[str] = Counter()
    batch_action_widths: list[int] = []
    transition_stats = {
        "load_ms": 0.0,
        "evict_ms": 0.0,
        "load_hits": 0,
        "evict_hits": 0,
    }

    def log(event_type: str, job: Job, node: Node | None = None, **extra: Any) -> None:
        if not collect_events:
            return
        row = {
            "schema_version": "simulation-event-v0.2",
            "episode_id": episode["episode_id"],
            "policy": policy,
            "event_type": event_type,
            "time_ms": round(now, 3),
            "job_instance_id": job.job_instance_id,
        }
        if node is not None:
            row.update({"node_id": node.node_id, "lane": node.lane, "model_id": node.model_id})
        row.update(extra)
        events.append(row)

    def log_prefetch(event_type: str, gpu: GPU, model_id: str, **extra: Any) -> None:
        if not collect_events:
            return
        event_time_ms = extra.pop("_event_time_ms", now)
        row = {
            "schema_version": "simulation-event-v0.3-extensions",
            "episode_id": episode["episode_id"],
            "policy": policy,
            "event_type": event_type,
            "time_ms": round(event_time_ms, 3),
            "job_instance_id": f"__prefetch__gpu{gpu.index}",
            "lane": "gpu_load",
            "model_id": model_id,
            "gpu_index": gpu.index,
        }
        row.update(extra)
        events.append(row)

    def find_model_node(model_id: str) -> Node | None:
        for template in templates.values():
            for candidate in template.nodes:
                if candidate.model_id == model_id and candidate.lane == "gpu":
                    return candidate
        return None

    def _latency_aware_prefetch_plan() -> list[dict[str, Any]]:
        """Eq (5) alpha_N = Prefetch, derived from the live pool state.

        A deployment is eligible when it is required by a near-ready successor of a
        running node, is not already resident anywhere, and fits a device alongside
        what that device already holds.  Unmeasured deployments are skipped rather
        than guessed, and a device that is busy right now is not disturbed.
        """

        from tracing.analysis.latency_aware_lifecycle import prefetch_candidates

        def memory_for(model_id: str) -> float | None:
            node = find_model_node(model_id)
            if node is None or node.resident_model_mb is None:
                return None
            row = estimate(node, train_stats)
            return model_memory(node, row)

        free = [gpu for gpu in gpus if gpu.active_node is None and not gpu.prefetch_pending]
        return prefetch_candidates(jobs, free, model_memory_for=memory_for)

    def admit_nested_gpu_work(
        node: Node, job: Job, job_index: int, now: float, sequence: int
    ) -> int:
        """Composite CPU+GPU segment, as an explicit state machine.

            parent_start -> nested_ready -> nested_gpu_start -> nested_gpu_finish
                         -> parent_finish

        The review's model:
            R_total = R_pre + R_nested + R_post
            nested_ready  = t + R_pre
            nested_start  = max(nested_ready, device availability)
            parent_finish = nested_finish + R_post
        so with no contention parent_finish recovers exactly t + R_total, and with
        contention the parent is delayed by exactly as much as the inner call.

        nested_gpu_finish releases ONLY the device segment: process_finish handles the
        gpu_nested lane specially and never completes the parent node there, which is
        the P0 the previous attempt had.  The load is NOT added because the parent's
        measured R_total already contains the inner interval.

        Returns the updated tie-break counter.
        """

        model = str(getattr(node, "nested_model_class", "") or "")
        if not model:
            return sequence
        need_mb = getattr(node, "nested_model_mb", None)
        if not isinstance(need_mb, (int, float)) or need_mb <= 0:
            raise ValueError(
                "composite node %s carries nested model %r with no measured allocation"
                % (node.node_id, model)
            )
        inner = getattr(node, "nested_inner_ms", None)
        pre = getattr(node, "nested_pre_ms", None)
        post = getattr(node, "nested_post_ms", None)
        if not all(isinstance(v, (int, float)) for v in (inner, pre, post)):
            raise ValueError("composite node %s lacks pre/inner/post" % node.node_id)

        nested_ready = float(now) + float(pre)

        # Device choice and admission are deliberately deferred to nested_ready.  Between
        # parent_start and nested_ready normal GPU work may run and change residency and
        # free capacity, so deciding here would be a check-then-use hazard.  No gpu_index
        # is carried either; the ready event resolves the device from live state.
        sequence += 1
        heapq.heappush(
            finish_heap,
            (nested_ready, int(sequence), int(job_index), "gpu_nested_ready", None,
             node.node_id),
        )
        log("nested_gpu_admit", job, node, nested_model_id=model,
            nested_ready_ms=round(nested_ready, 3), nested_memory_mb=round(float(need_mb), 3))
        return sequence

    def initialize_prefetch() -> None:
        """Schedule explicit, paid model loads before the first dispatch.

        The plan comes from ``extension_config["prefetch_plan"]`` for the existing
        static arms.  The Latency-Aware arm supplies one derived from live state
        instead: the deployments of near-ready successors, which is the paper's
        alpha_N = Prefetch half of Eq (5).
        """

        plan = extension_config.get("prefetch_plan") or []
        if not plan and policy == "latency_aware":
            plan = _latency_aware_prefetch_plan()
        if not plan:
            return
        # A prefetch cannot start in the past.  The cursor was seeded from busy_until
        # alone, so an idle device produced a start time that had already elapsed.
        cursors = {gpu.index: max(float(now), float(gpu.busy_until)) for gpu in gpus}
        for raw in plan:
            gpu_index = int(raw.get("gpu_index", 0))
            model_id = str(raw.get("model_id") or "")
            if gpu_index < 0 or gpu_index >= len(gpus) or not model_id:
                raise ValueError(f"invalid prefetch plan entry: {raw}")
            gpu = gpus[gpu_index]
            if model_id in gpu.resident:
                log_prefetch("prefetch_skip", gpu, model_id, reason="already_resident")
                continue
            model_node = find_model_node(model_id)
            if model_node is None or model_node.resident_model_mb is None:
                raise ValueError(f"prefetch model has no measured GPU memory: {model_id}")
            estimate_row = estimate(model_node, train_stats)
            memory = model_memory(model_node, estimate_row)
            projected = sum(gpu.resident.values()) + memory
            if projected > gpu.capacity_mb + 1e-9:
                log_prefetch(
                    "prefetch_fail",
                    gpu,
                    model_id,
                    reason="capacity",
                    requested_memory_mb=round(memory, 3),
                    projected_memory_mb=round(projected, 3),
                    gpu_capacity_mb=gpu.capacity_mb,
                )
                continue
            load, load_source = transition_load_cost(
                model_node,
                estimate_row,
                model_node.batch_size,
                extension_config,
            )
            # the same lower bound at the point of use, so a plan built earlier in the
            # event cannot schedule itself backwards
            start = max(float(now), float(cursors[gpu_index]))
            finish = start + load
            gpu.resident[model_id] = memory
            gpu.prefetch_pending.append((model_id, start, finish))
            gpu.prefetched_models.add(model_id)
            gpu.prefetch_count += 1
            gpu.prefetch_load_ms += load
            if load_source == "transition_profile":
                transition_stats["load_ms"] += load
                transition_stats["load_hits"] += 1
            gpu.busy_until = max(gpu.busy_until, finish)
            gpu.peak_memory_mb = max(gpu.peak_memory_mb, sum(gpu.resident.values()))
            cursors[gpu_index] = finish
            log_prefetch(
                "prefetch_start",
                gpu,
                model_id,
                start_ms=round(start, 3),
                finish_ms=round(finish, 3),
                load_ms=round(load, 3),
                load_source=load_source,
                resident_memory_mb=round(sum(gpu.resident.values()), 3),
            )

    def process_prefetch_finish() -> None:
        for gpu in gpus:
            remaining: list[tuple[str, float, float]] = []
            for model_id, start, finish in gpu.prefetch_pending:
                if finish <= now + 1e-9:
                    log_prefetch(
                        "prefetch_end",
                        gpu,
                        model_id,
                        start_ms=round(start, 3),
                        finish_ms=round(finish, 3),
                        load_ms=round(finish - start, 3),
                        _event_time_ms=finish,
                    )
                else:
                    remaining.append((model_id, start, finish))
            gpu.prefetch_pending = remaining

    initialize_prefetch()

    def release_ready(job_index: int) -> None:
        job = jobs[job_index]
        for node_id in ready_nodes(job, now):
            if job.node_state[node_id] == "pending":
                job.node_state[node_id] = "ready"
                job.ready_since[node_id] = now
                node = job.template.by_id[node_id]
                priority = 0 if job.service_class == "priority" else 1
                heapq.heappush(ready, (priority, now, job_index, node_id))
                log("node_ready", job, node, predecessor_count=len(node.predecessors))

    def complete_job_if_done(job: Job) -> None:
        if job.finish_ms is None and all(state == "complete" for state in job.node_state.values()):
            job.finish_ms = now
            log("job_finish", job, deadline_met=(job.deadline_ms is None or now <= job.deadline_ms))

    def process_finish() -> None:
        nonlocal now, sequence
        while finish_heap and finish_heap[0][0] <= now + 1e-9:
            finish, _order, job_index, lane, gpu_index, node_id = heapq.heappop(finish_heap)
            job = jobs[job_index]

            if lane == "gpu_nested_ready":
                # the device is actually needed now, so residency, capacity and placement
                # are all resolved against LIVE state at this instant
                parent = job.template.by_id[node_id]
                need_mb = float(getattr(parent, "nested_model_mb", 0.0) or 0.0)
                model = str(getattr(parent, "nested_model_class", "") or "")
                inner = float(getattr(parent, "nested_inner_ms", 0.0) or 0.0)

                target = None
                for candidate_gpu in gpus:
                    if model in candidate_gpu.resident:
                        target = candidate_gpu
                        break
                if target is None:
                    for candidate_gpu in gpus:
                        if sum(candidate_gpu.resident.values()) + need_mb <= candidate_gpu.capacity_mb + 1e-9:
                            target = candidate_gpu
                            break
                if target is None:
                    job.node_state[node_id] = "failed"
                    job.failed.add(node_id)
                    log("node_fail", job, parent, reason="nested_gpu_no_capacity", model_id=model)
                    complete_job_if_done(job)
                    continue

                start = max(float(now), float(target.busy_until), float(target.composite_tail))
                inner_finish = start + inner
                if model and model not in target.resident:
                    target.resident[model] = need_mb
                    log("model_load_start", job, parent, gpu_index=target.index, load_ms=0.0,
                        load_source="nested_composite")
                target.peak_memory_mb = max(target.peak_memory_mb, sum(target.resident.values()))
                target.busy_until = max(float(target.busy_until), inner_finish)
                target.composite_tail = max(float(target.composite_tail), inner_finish)
                # deliberately NOT target.active_node: the preemption path would treat the
                # CPU parent as an ordinary GPU victim and tear down this segment
                target.composite_owner = "%d:%s" % (int(job_index), node_id)
                target.composite_queued += 1
                target.busy_time_ms += inner
                sequence += 1
                heapq.heappush(
                    finish_heap,
                    (inner_finish, int(sequence), job_index, "gpu_nested", target.index, node_id),
                )
                log("nested_gpu_start", job, parent, gpu_index=target.index,
                    nested_start_ms=round(start, 3), nested_finish_ms=round(inner_finish, 3))
                continue

            if lane == "gpu_nested":
                # the inner segment ended.  Release ONLY the composite ownership: the
                # parent is still running and must not be completed here, no successor may
                # be released, and composite_tail must never be shrunk.
                if gpu_index is not None:
                    gpu = gpus[gpu_index]
                    if getattr(gpu, "composite_owner", "") == "%d:%s" % (int(job_index), node_id):
                        gpu.composite_owner = ""
                    gpu.composite_queued = max(0, int(getattr(gpu, "composite_queued", 0)) - 1)
                parent = job.template.by_id[node_id]
                post = float(getattr(parent, "nested_post_ms", 0.0) or 0.0)
                log("nested_gpu_finish", job, parent, finish_ms=round(float(finish), 3),
                    gpu_index=gpu_index)
                sequence += 1
                heapq.heappush(
                    finish_heap,
                    (finish + post, int(sequence), job_index, parent.lane, None, node_id),
                )
                continue
            # a fused unit carries several member ids; a plain dispatch carries one
            member_ids = node_id.split(FUSED_ID_SEPARATOR)
            if gpu_index is not None:
                gpu = gpus[gpu_index]
                # queued composite segments still need the device after this completion
                gpu.busy_until = max(float(finish), float(gpu.composite_tail))
                gpu.active_node = None
                gpu.active_start_ms = None
            for member_id in member_ids:
                node = job.template.by_id[member_id]
                job.node_state[member_id] = "complete"
                job.completed.add(member_id)
                truth = truth_provider.get(job.job_instance_id, node.node_id)
                # The intrinsic duration becomes an OBSERVATION exactly here, at the
                # finish event, and is stored on the job.  LLMSched evidence reads this
                # store and nothing else, so a duration is structurally unavailable
                # before the node finishes rather than merely by convention about when
                # the template happens to be consulted.
                job.observed_intrinsic_ms[member_id] = float(truth.runtime_ms)
                log("node_finish", job, node, finish_ms=round(finish, 3), gpu_index=gpu_index, observed_status=truth.status)
            release_ready(job_index)
            complete_job_if_done(job)

    def maybe_preempt() -> bool:
        """Preempt one active normal node using explicit recompute semantics."""

        if not extension_config.get("preemption_enabled"):
            return False
        if policy not in {"myopic_preempt", "batch_myopic_preempt"}:
            return False
        max_preemptions = int(extension_config.get("max_preemptions", max(1, 2 * len(jobs))))
        if sum(job.preemptions for job in jobs) >= max_preemptions:
            return False
        candidates = [
            item
            for item in ready
            if jobs[item[2]].node_state.get(item[3]) == "ready"
            and jobs[item[2]].template.by_id[item[3]].lane == "gpu"
        ]
        if not candidates:
            return False
        target = min(candidates)
        target_priority, target_ready, target_job_index, target_node_id = target
        target_job = jobs[target_job_index]
        target_node = target_job.template.by_id[target_node_id]
        target_row = estimate_for_batch(target_node, train_stats, target_node.batch_size, extension_config)
        target_slack = math.inf
        if target_job.deadline_ms is not None:
            target_slack = target_job.deadline_ms - now - float(target_row["runtime_p50_ms"])
        age_threshold = max(0.0, number(extension_config.get("preempt_min_queue_age_ms"), 0.0))
        slack_threshold = number(extension_config.get("preempt_deadline_slack_ms"), 0.0)
        victim_rows = []
        for gpu in gpus:
            if gpu.active_node is None:
                continue
            victim_job_index, victim_node_id = gpu.active_node
            victim_job = jobs[victim_job_index]
            victim_node = victim_job.template.by_id[victim_node_id]
            if victim_job.service_class == "priority":
                continue
            if int(getattr(gpu, "composite_queued", 0)) > 0:
                # a composite segment holds or is queued for this device.  Preempting the
                # active node would rewind busy_until past the reservation, so the device
                # would look free while a queued segment still owns [now, composite_tail].
                continue
            should_preempt = (
                target_priority < (0 if victim_job.service_class == "priority" else 1)
                or now - target_ready >= age_threshold
                or target_slack <= slack_threshold
            )
            if not should_preempt:
                continue
            victim_rows.append((victim_job.service_class, -gpu.index, gpu, victim_job_index, victim_job, victim_node))
        if not victim_rows:
            return False
        _service_class, _gpu_order, gpu, victim_job_index, victim_job, victim_node = min(victim_rows, key=lambda row: (row[0], row[1]))
        matching = [
            entry
            for entry in finish_heap
            if entry[2] == victim_job_index and entry[5] == victim_node.node_id and entry[4] == gpu.index
        ]
        if not matching:
            return False
        finish, _order, victim_job_index, lane, gpu_index, node_id = matching[0]
        finish_heap.remove(matching[0])
        heapq.heapify(finish_heap)
        elapsed = max(0.0, now - float(gpu.active_start_ms if gpu.active_start_ms is not None else now))
        remaining = max(0.0, finish - now)
        # The node has no checkpoint contract.  Discard progress and pay the
        # full compute time again when it resumes (recompute semantics).
        gpu.busy_time_ms = max(0.0, gpu.busy_time_ms - remaining)
        victim_job.preemptions += 1
        victim_job.preempt_recompute_ms += elapsed
        victim_job.node_state[node_id] = "ready"
        victim_job.ready_since[node_id] = now
        victim_priority = 0 if victim_job.service_class == "priority" else 1
        heapq.heappush(ready, (victim_priority, now, victim_job_index, node_id))
        gpu.active_node = None
        gpu.active_start_ms = None
        gpu.busy_until = now
        log(
            "node_preempt",
            victim_job,
            victim_node,
            gpu_index=gpu_index,
            reason="deadline_or_queue_age",
            target_job_instance_id=target_job.job_instance_id,
            target_node_id=target_node.node_id,
            elapsed_ms=round(elapsed, 3),
            remaining_ms=round(remaining, 3),
            recompute=True,
        )
        return True

    while arrival_heap or finish_heap or ready or any(gpu.prefetch_pending for gpu in gpus):
        process_prefetch_finish()
        if not ready and not finish_heap and arrival_heap:
            next_prefetch = min(
                (gpu.busy_until for gpu in gpus if gpu.prefetch_pending and gpu.busy_until > now + 1e-9),
                default=math.inf,
            )
            now = max(now, min(arrival_heap[0][0], next_prefetch))
        while arrival_heap and arrival_heap[0][0] <= now + 1e-9:
            arrival, job_index = heapq.heappop(arrival_heap)
            job = jobs[job_index]
            active_jobs.add(job_index)
            log("job_arrive", job, arrival_ms=round(arrival, 3))
            release_ready(job_index)
        process_finish()
        process_prefetch_finish()

        made_progress = True
        while made_progress:
            made_progress = False
            # CPU/API work does not consume a GPU slot in this first model.
            deferred: list[tuple[float, int, int, str]] = []
            while ready:
                item = heapq.heappop(ready)
                _priority, ready_time, job_index, node_id = item
                job = jobs[job_index]
                node = job.template.by_id[node_id]
                if node.lane == "gpu":
                    deferred.append(item)
                    continue
                if job.node_state[node_id] != "ready":
                    continue
                row = estimate(node, train_stats)
                job.node_state[node_id] = "running"
                job.started.add(node_id)
                job.queue_ms += max(0.0, now - ready_time)
                duration = node.runtime_ms
                if getattr(node, "nested_model_class", ""):
                    # state machine: the parent does NOT finish at now + R_total here.
                    # Its inner segment is admitted, and process_finish pushes the
                    # parent completion once that segment ends (nested_finish + post).
                    sequence = admit_nested_gpu_work(node, job, job_index, now, sequence)
                else:
                    sequence += 1
                    event_key = int(sequence)
                    heapq.heappush(finish_heap,
                                   (now + duration, event_key, job_index, node.lane, None, node_id))
                log("node_start", job, node, scheduler_view=simulator_view(node, row, now, None), start_ms=round(now, 3), queue_ms=round(max(0.0, now - ready_time), 3))
                made_progress = True
            for item in deferred:
                heapq.heappush(ready, item)

            free_gpu = [gpu for gpu in gpus
                  if gpu.active_node is None
                  and gpu.busy_until <= now + 1e-9
]
            if not free_gpu and maybe_preempt():
                made_progress = True
                free_gpu = [gpu for gpu in gpus
                  if gpu.active_node is None
                  and gpu.busy_until <= now + 1e-9
]
            if free_gpu:
                gpu_ready_items: list[tuple[float, int, int, str]] = []
                deferred_items: list[tuple[float, int, int, str]] = []
                while ready:
                    item = heapq.heappop(ready)
                    _priority, _ready_time, job_index, node_id = item
                    job = jobs[job_index]
                    node = job.template.by_id[node_id]
                    if job.node_state[node_id] != "ready":
                        continue
                    if node.lane == "gpu":
                        gpu_ready_items.append(item)
                    else:
                        deferred_items.append(item)
                for item in deferred_items:
                    heapq.heappush(ready, item)

                while gpu_ready_items and free_gpu:
                    scheduler_state = scheduler_state_at_decision(
                        episode,
                        jobs,
                        gpus,
                        gpu_ready_items,
                        train_stats,
                        future_provider,
                        future_horizon,
                        decision_index,
                        now,
                    )
                    state_decision_index = decision_index
                    decision_index += 1
                    if policy_context is not None:
                        policy_context["current_decision_index"] = state_decision_index
                    if extension_config.get("batch_enabled"):
                        ready_item, gpu_index, row, candidate_count, feasible_candidate_count, batch_size = choose_action_with_batch(
                            policy,
                            gpu_ready_items,
                            jobs,
                            free_gpu,
                            train_stats,
                            rr_cursor,
                            extension_config,
                        )
                        batch_choice_counts[str(batch_size)] += 1
                        batch_action_widths.append(candidate_count)
                    else:
                        ready_item, gpu_index, row, candidate_count, feasible_candidate_count = choose_action(
                            policy,
                            gpu_ready_items,
                            jobs,
                            free_gpu,
                            train_stats,
                            rr_cursor,
                            future_artifacts,
                            rl_context,
                            policy_context,
                            now,
                            extension_config,
                        )
                        batch_size = 1
                    gpu_ready_items.remove(ready_item)
                    _priority, ready_time, job_index, node_id = ready_item
                    job = jobs[job_index]
                    node = job.template.by_id[node_id]
                    gpu = next(candidate for candidate in free_gpu if candidate.index == gpu_index)
                    log(
                        "node_dispatch",
                        job,
                        node,
                        gpu_index=gpu_index,
                        candidate_count=candidate_count,
                        feasible_candidate_count=feasible_candidate_count,
                        action_policy=policy,
                        batch_size=batch_size,
                        scheduler_state=scheduler_state,
                    )
                    admitted, evicted_plan, memory, predicted_workspace, total_memory = plan_gpu_admission(gpu, node, row)
                    if not admitted:
                        job.node_state[node_id] = "failed"
                        job.failed.add(node_id)
                        log("node_fail", job, node, reason="simulated_oom", gpu_index=gpu_index, scheduler_view=simulator_view(node, row, now, gpu_index), predicted_total_memory_mb=round(total_memory, 3), gpu_capacity_mb=gpu.capacity_mb)
                        complete_job_if_done(job)
                        made_progress = True
                        continue
                    load = 0.0
                    load_source = "resident"
                    eviction_cost = 0.0
                    eviction_sources: list[dict[str, Any]] = []
                    evicted = list(evicted_plan)
                    for model_id in evicted:
                        gpu.resident.pop(model_id, None)
                        if model_id in gpu.prefetched_models and model_id not in gpu.used_prefetched_models:
                            gpu.wasted_prefetches += 1
                            log_prefetch("prefetch_wasted", gpu, model_id, reason="evicted_before_first_use")
                        evict_cost, evict_source = transition_eviction_cost(model_id, extension_config)
                        eviction_cost += evict_cost
                        eviction_sources.append(
                            {
                                "model_id": model_id,
                                "evict_ms": round(evict_cost, 3),
                                "source": evict_source,
                            }
                        )
                    if evicted:
                        gpu.evictions += len(evicted)
                        transition_stats["evict_ms"] += eviction_cost
                        transition_stats["evict_hits"] += sum(
                            source["source"] == "transition_profile" for source in eviction_sources
                        )
                        log(
                            "model_evict",
                            job,
                            node,
                            gpu_index=gpu_index,
                            models=evicted,
                            evict_cost_ms=round(eviction_cost, 3),
                            evict_costs=eviction_sources,
                        )
                    if node.model_id in gpu.prefetched_models:
                        gpu.used_prefetched_models.add(node.model_id)
                    if node.model_id not in gpu.resident:
                        gpu.resident[node.model_id] = memory
                        _effective_runtime, effective_load = effective_batch_runtime(node, row, batch_size, extension_config)
                        load, load_source = transition_load_cost(
                            node,
                            row,
                            batch_size,
                            extension_config,
                        )
                        if load_source == "transition_profile":
                            transition_stats["load_ms"] += load
                            transition_stats["load_hits"] += 1
                        log(
                            "model_load_start",
                            job,
                            node,
                            gpu_index=gpu_index,
                            load_ms=round(load, 3),
                            load_source=load_source,
                        )
                    total_memory = sum(gpu.resident.values()) + predicted_workspace
                    gpu.peak_memory_mb = max(gpu.peak_memory_mb, total_memory)
                    fused_members = list((policy_context or {}).get("_fused_chain") or [])
                    if fused_members and fused_members[0] == node_id:
                        for member_id in fused_members:
                            job.node_state[member_id] = "running"
                            job.started.add(member_id)
                    else:
                        fused_members = []
                        job.node_state[node_id] = "running"
                        job.started.add(node_id)
                    job.queue_ms += max(0.0, now - ready_time)
                    job.load_ms += load
                    effective_runtime, effective_load = effective_batch_runtime(node, row, batch_size, extension_config)
                    if fused_members:
                        # one grant: the model loads once and the members run back to back
                        duration = eviction_cost + load + sum(
                            float(job.template.by_id[m].compute_ms) for m in fused_members
                        )
                    elif _batch_profile(node, batch_size, extension_config) is None:
                        duration = node.compute_ms + eviction_cost + load
                    else:
                        duration = max(0.1, effective_runtime - effective_load) + eviction_cost + load
                    start_time = now
                    finish = start_time + duration
                    gpu.active_node = (job_index, node_id)
                    gpu.busy_until = finish
                    gpu.active_start_ms = start_time
                    gpu.busy_time_ms += duration
                    job.assigned_gpus.append(gpu_index)
                    sequence += 1
                    event_key = int(sequence)
                    heapq.heappush(
                        finish_heap,
                        (finish, event_key, job_index, "gpu", gpu_index,
                         FUSED_ID_SEPARATOR.join(fused_members) if fused_members else node_id),
                    )
                    truth = truth_provider.get(job.job_instance_id, node.node_id)
                    log(
                        "node_start",
                        job,
                        node,
                        scheduler_view=simulator_view(node, row, now, gpu_index),
                        start_ms=round(start_time, 3),
                        queue_ms=round(max(0.0, now - ready_time), 3),
                        load_ms=round(load, 3),
                        load_source=load_source,
                        eviction_ms=round(eviction_cost, 3),
                        transition_overhead_ms=round(eviction_cost + load, 3),
                        truth_runtime_ms=round(truth.runtime_ms, 3),
                        resident_memory_mb=round(sum(gpu.resident.values()), 3),
                        workspace_memory_mb=round(predicted_workspace, 3),
                        total_memory_mb=round(total_memory, 3),
                        gpu_capacity_mb=gpu.capacity_mb,
                    )
                    rr_cursor = (gpu_index + 1) % max(1, len(gpus))
                    free_gpu = [candidate for candidate in free_gpu if candidate.index != gpu_index]
                    made_progress = True

                for item in gpu_ready_items:
                    heapq.heappush(ready, item)
                waiting_items = gpu_ready_items if not free_gpu else []
            else:
                waiting_items = [
                    item
                    for item in ready
                    if jobs[item[2]].node_state.get(item[3]) == "ready"
                    and jobs[item[2]].template.by_id[item[3]].lane == "gpu"
                ]
            for item in waiting_items:
                _priority, ready_time, job_index, node_id = item
                wait_key = (job_index, node_id)
                if wait_key not in wait_logged:
                    log(
                        "node_wait",
                        jobs[job_index],
                        jobs[job_index].template.by_id[node_id],
                        reason="no_free_gpu",
                        waited_ms=round(max(0.0, now - ready_time), 3),
                    )
                    wait_logged.add(wait_key)

        if policy == "latency_aware":
            # Eq (5) alpha_N, run AFTER ready dispatch and only on devices no ready unit
            # could use.  Running it at the TOP of the loop let a prefetch seize an idle
            # device and push gpu.busy_until past now, which removed that device from
            # ``free_gpu`` and delayed work that was already runnable.  Preparing near-ready
            # work is only faithful if it does NOT disturb ready work, so a device still
            # idle once every runnable unit has been placed is the only one eligible.
            _la_idle = [
                gpu for gpu in gpus
                if gpu.active_node is None
                and not gpu.prefetch_pending
                and gpu.busy_until <= now + 1e-9
            ]
            if _la_idle:
                _la_plan = _latency_aware_prefetch_plan()
                if _la_plan:
                    _la_saved = extension_config.get("prefetch_plan")
                    extension_config = {**(extension_config or {}), "prefetch_plan": _la_plan}
                    initialize_prefetch()
                    extension_config = {**(extension_config or {}), "prefetch_plan": _la_saved}

        process_finish()
        next_finish = finish_heap[0][0] if finish_heap else math.inf
        next_arrival = arrival_heap[0][0] if arrival_heap else math.inf
        next_gpu_ready = min(
            (gpu.busy_until for gpu in gpus if gpu.active_node is None and gpu.busy_until > now + 1e-9),
            default=math.inf,
        )
        next_time = min(next_finish, next_arrival, next_gpu_ready)
        if next_time == math.inf:
            if ready:
                raise RuntimeError("simulation deadlock")
            break
        if next_time <= now + 1e-9:
            raise RuntimeError(f"simulation made no time progress at {now}")
        now = max(now, next_time)

    completed = [job for job in jobs if job.finish_ms is not None and not job.failed]
    durations = [float(job.finish_ms) - job.arrival_ms for job in completed]
    makespan = max((job.finish_ms or 0.0 for job in jobs), default=0.0)
    summary = {
        "schema_version": "simulation-result-v0.2",
        "episode_id": episode["episode_id"],
        "policy": policy,
        "jobs": len(jobs),
        "completed_jobs": len(completed),
        "failed_jobs": sum(1 for job in jobs if job.failed),
        "nodes": sum(len(job.template.nodes) for job in jobs),
        "completed_nodes": sum(len(job.completed) for job in jobs),
        "mean_completion_ms": statistics.fmean(durations) if durations else None,
        "p95_completion_ms": quantile(durations, 0.95),
        "makespan_ms": makespan,
        "mean_job_queue_ms": statistics.fmean(job.queue_ms for job in jobs) if jobs else 0.0,
        "deadline_miss_rate": statistics.fmean(1.0 if job.finish_ms is None or (job.deadline_ms is not None and job.finish_ms > job.deadline_ms) else 0.0 for job in jobs) if jobs else 0.0,
        "gpu_evictions": sum(gpu.evictions for gpu in gpus),
        "gpu_peak_memory_mb": [gpu.peak_memory_mb for gpu in gpus],
        "gpu_utilization": [gpu.busy_time_ms / makespan if makespan > 0 else 0.0 for gpu in gpus],
        "preemptions": sum(job.preemptions for job in jobs),
        "preempt_recompute_ms": sum(job.preempt_recompute_ms for job in jobs),
        "prefetch_count": sum(gpu.prefetch_count for gpu in gpus),
        "prefetch_load_ms": sum(gpu.prefetch_load_ms for gpu in gpus),
        "wasted_prefetches": sum(gpu.wasted_prefetches for gpu in gpus),
        "transition_profile_enabled": _transition_profile(extension_config) is not None,
        "transition_profile_load_ms": round(transition_stats["load_ms"], 3),
        "transition_profile_evict_ms": round(transition_stats["evict_ms"], 3),
        "transition_profile_load_hits": int(transition_stats["load_hits"]),
        "transition_profile_evict_hits": int(transition_stats["evict_hits"]),
        "batch_action_count": sum(batch_choice_counts.values()),
        "batch_choice_counts": dict(sorted(batch_choice_counts.items())),
        "batch_action_width_p50": quantile([float(value) for value in batch_action_widths], 0.5),
    }
    if policy_context is not None and policy_context.get("cp_rho_events"):
        cp_events = list(policy_context["cp_rho_events"])
        solve_times = [number(row.get("solve_ms")) for row in cp_events]
        statuses = Counter(str(row.get("status")) for row in cp_events)
        summary.update(
            {
                "cp_rho_solver_calls": len(cp_events),
                "cp_rho_status_counts": dict(sorted(statuses.items())),
                "cp_rho_timeouts": sum(str(row.get("status")) in {"UNKNOWN", "MODEL_INVALID"} for row in cp_events),
                "cp_rho_fallbacks": sum(bool(row.get("fallback")) for row in cp_events),
                "cp_rho_mean_solve_ms": statistics.fmean(solve_times) if solve_times else 0.0,
                "cp_rho_p95_solve_ms": quantile(solve_times, 0.95),
            }
        )
    if policy_context is not None and policy_context.get("pred_mpc_events"):
        mpc_events = list(policy_context["pred_mpc_events"])
        solve_times = [number(row.get("solve_ms")) for row in mpc_events]
        summary.update(
            {
                "pred_mpc_rollout_calls": len(mpc_events),
                "pred_mpc_mean_rollout_ms": statistics.fmean(solve_times) if solve_times else 0.0,
                "pred_mpc_p95_rollout_ms": quantile(solve_times, 0.95),
                "pred_mpc_mean_predicted_area_ms": statistics.fmean(
                    number(row.get("area_ms")) for row in mpc_events
                ) if mpc_events else 0.0,
                "pred_mpc_mean_terminal_ms": statistics.fmean(
                    number(row.get("terminal_ms")) for row in mpc_events
                ) if mpc_events else 0.0,
                "pred_mpc_mean_predicted_events": statistics.fmean(
                    number(row.get("predicted_events")) for row in mpc_events
                ) if mpc_events else 0.0,
            }
        )
    if policy_context is not None and policy_context.get("risk_aware_events"):
        risk_events = list(policy_context["risk_aware_events"])
        summary.update(
            {
                "risk_aware_decision_count": len(risk_events),
                "risk_aware_mean_score": statistics.fmean(number(row.get("score")) for row in risk_events),
                "risk_aware_mean_urgency": statistics.fmean(number(row.get("urgency")) for row in risk_events),
                "risk_aware_mean_resource_risk": statistics.fmean(
                    number(row.get("resource_risk_norm")) for row in risk_events
                ),
                "risk_aware_mean_age_ms": statistics.fmean(number(row.get("age_ms")) for row in risk_events),
                "risk_aware_strict_feasible_rate": statistics.fmean(
                    number(row.get("strict_feasible_count")) / max(1.0, number(row.get("active_candidate_count")))
                    for row in risk_events
                ),
            }
        )
    return summary, events


def run(templates_path: Path, episodes_path: Path, output_dir: Path, policies: Sequence[str]) -> dict[str, Any]:
    templates = load_templates(templates_path)
    episodes = read_jsonl(episodes_path)
    summaries: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    for episode in episodes:
        for policy in policies:
            summary, events = simulate_episode(episode, templates, policy)
            summaries.append(summary)
            all_events.extend(events)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "simulation_results.jsonl", summaries)
    write_jsonl(output_dir / "simulation_events.jsonl", all_events)
    by_policy: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in summaries:
        by_policy[str(row["policy"])].append(row)
    aggregate: list[dict[str, Any]] = []
    for policy, rows in sorted(by_policy.items()):
        aggregate.append({
            "policy": policy,
            "episodes": len(rows),
            "mean_completion_ms": statistics.fmean(number(row.get("mean_completion_ms")) for row in rows),
            "p95_completion_ms": statistics.fmean(number(row.get("p95_completion_ms")) for row in rows),
            "mean_job_queue_ms": statistics.fmean(number(row.get("mean_job_queue_ms")) for row in rows),
            "deadline_miss_rate": statistics.fmean(number(row.get("deadline_miss_rate")) for row in rows),
            "mean_gpu_evictions": statistics.fmean(number(row.get("gpu_evictions")) for row in rows),
            "completed_jobs": min(number(row.get("completed_jobs")) for row in rows),
        })
    oracle = next((row for row in aggregate if row["policy"] == "oracle"), None)
    myopic = next((row for row in aggregate if row["policy"] == "myopic"), None)
    oracle_gap = None
    if oracle and myopic:
        oracle_gap = {
            "myopic_minus_oracle_mean_completion_ms": myopic["mean_completion_ms"] - oracle["mean_completion_ms"],
            "myopic_relative_gap": (myopic["mean_completion_ms"] - oracle["mean_completion_ms"]) / max(1e-9, oracle["mean_completion_ms"]),
        }
    report = {
        "schema_version": "simulation-report-v0.2",
        "templates": len(templates),
        "episodes": len(episodes),
        "policies": list(policies),
        "aggregate": aggregate,
        "oracle_gap": oracle_gap,
        "event_invariants": {
            "checked": True,
            "scheduler_truth_separated": True,
            "queue_generated_by_events": True,
            "oom_labels": "simulator_outcome_only",
            "node_dispatch_events": sum(row["event_type"] == "node_dispatch" for row in all_events),
            "node_wait_events": sum(row["event_type"] == "node_wait" for row in all_events),
            "candidate_actions_recorded": True,
        },
    }
    write_json(output_dir / "simulation_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policies", default=",".join(POLICIES))
    args = parser.parse_args()
    policies = tuple(value.strip() for value in args.policies.split(",") if value.strip())
    unknown = sorted(set(policies) - set(POLICIES))
    if unknown:
        raise ValueError(f"unknown policies: {unknown}")
    report = run(args.templates, args.episodes, args.output_dir, policies)
    print(json.dumps({"episodes": report["episodes"], "templates": report["templates"], "oracle_gap": report["oracle_gap"], "output_dir": str(args.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
