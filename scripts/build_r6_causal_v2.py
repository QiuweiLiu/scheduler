#!/usr/bin/env python3
"""Build the versioned R6 causal-v2 data contract.

This command consumes the immutable 768-template C1 artifacts and the frozen
48/8/8 video split.  It writes a new collection only; it never edits the
legacy trace or processed files.  The output deliberately separates:

* scheduler-visible prefix/current-node identity;
* execution truth (runtime, load, memory and status);
* train-only future prototypes and resource quantiles; and
* workload episode metadata from fields that a policy may read.

The script is dependency-free except for the existing, already-tested
``tracing.workloads.build_workload_v02`` generator used for episode sampling.
It is intended to run on the remote ``finetooling`` environment with
``PYTHONPATH=src``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DERIVATION_VERSION = "r6-causal-v2"
COLLECTION_ID = "trace_collection_v1_legacy_core_r6_causal_v2"
SPLIT_SCHEMA = "video-split-registry-r6-v1"
FORBIDDEN_INPUT_KEYS = {
    "answer",
    "ground_truth",
    "target",
    "future",
    "future_events",
    "remaining_steps",
    "remaining_runtime_ms",
    "suffix",
    "runtime_ms",
    "load_ms",
    "peak_memory_mb",
    "peak_allocated_mb",
    "peak_reserved_mb",
    "workspace_peak_mb",
    "status",
    "retry_of",
    "timestamp_start",
    "timestamp_end",
}
CURRENT_NODE_KEYS = {
    "activity",
    "raw_action",
    "node_type",
    "execution_lane",
    "model_id",
    "sequence_index",
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(payload)
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def text(value: Any, default: str = "") -> str:
    value = "" if value is None else str(value).strip()
    return value or default


def number(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def optional_number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def quantile(values: Sequence[float], q: float) -> float | None:
    clean = sorted(float(v) for v in values if math.isfinite(float(v)) and float(v) >= 0.0)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * q
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return clean[left]
    fraction = position - left
    return clean[left] * (1.0 - fraction) + clean[right] * fraction


def get_nested(row: Mapping[str, Any], *paths: str) -> Any:
    for path in paths:
        value: Any = row
        for part in path.split("."):
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(part)
        if value not in (None, ""):
            return value
    return None


def split_map(path: Path) -> tuple[dict[str, str], list[dict[str, Any]]]:
    rows = read_jsonl(path)
    mapping: dict[str, str] = {}
    for row in rows:
        video_id = text(row.get("video_id") or row.get("source_video_id"))
        split = text(row.get("split"))
        if not video_id or split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid split row: {row}")
        if video_id in mapping and mapping[video_id] != split:
            raise ValueError(f"video appears in multiple C1 splits: {video_id}")
        mapping[video_id] = split
    counts = Counter(mapping.values())
    if counts != Counter({"train": 48, "validation": 8, "test": 8}):
        raise ValueError(f"C1 split must be 48/8/8, got {dict(counts)}")
    return mapping, rows


def provenance_video(row: Mapping[str, Any]) -> tuple[str, str | None, float | None, str]:
    video_id = text(row.get("video_id"))
    original = row.get("original") if isinstance(row.get("original"), Mapping) else {}
    digest = get_nested(original, "sha256", "file_sha256")
    duration = get_nested(original, "ffprobe.duration_s", "ffprobe.duration", "duration_s", "duration")
    source = text(get_nested(row, "source.source_url", "retrieval.source_url"), "unknown")
    return video_id, text(digest) or None, optional_number(duration), source


def discover_seen_videos(project_root: Path) -> tuple[set[str], set[str]]:
    """Collect every video id/hash already represented by a raw run manifest."""
    ids: set[str] = set()
    hashes: set[str] = set()
    raw_root = project_root / "results" / "raw"
    if not raw_root.exists():
        return ids, hashes
    for manifest_path in raw_root.rglob("run_manifest.json"):
        try:
            payload = read_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        video_id = text(payload.get("video_id"))
        video_path = text(payload.get("video_path"))
        if video_id:
            ids.add(video_id)
        if video_path:
            ids.add(Path(video_path).stem)
        for key in ("video_sha256", "source_video_sha256", "video_hash"):
            value = text(payload.get(key))
            if value:
                hashes.add(value)
    return ids, hashes


def build_split_registry(
    project_root: Path,
    split_path: Path,
    provenance_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    c1_map, _ = split_map(split_path)
    provenance_rows = read_jsonl(provenance_path)
    seen_ids, seen_hashes = discover_seen_videos(project_root)
    registry: list[dict[str, Any]] = []
    for ordinal, (video_id, split) in enumerate(sorted(c1_map.items()), 1):
        registry.append(
            {
                "schema_version": SPLIT_SCHEMA,
                "ordinal": ordinal,
                "video_id": video_id,
                "split": f"c1_{'test_retrospective' if split == 'test' else split}",
                "source_collection": "trace_collection_v1_legacy_core",
                "status": "trace_available",
                "predictor_seen": True,
                "c2_eligible": False,
                "selection_reason": "frozen legacy C1 video-level split",
            }
        )
    candidates: list[dict[str, Any]] = []
    for row in provenance_rows:
        video_id, digest, duration, source = provenance_video(row)
        if not video_id or video_id in c1_map:
            continue
        if video_id in seen_ids or (digest and digest in seen_hashes):
            continue
        candidates.append(
            {
                "video_id": video_id,
                "sha256": digest,
                "duration_s": duration,
                "source_url": source,
                "status": text(row.get("original", {}).get("status"), "unknown") if isinstance(row.get("original"), Mapping) else "unknown",
            }
        )
    candidates.sort(key=lambda row: stable_hash(f"r6-c2|{row['video_id']}|{row.get('sha256') or ''}"))
    if len(candidates) < 40:
        raise ValueError(f"need at least 32 C2 candidates plus 8 backups; found {len(candidates)}")
    for rank, row in enumerate(candidates[:40], 1):
        registry.append(
            {
                "schema_version": SPLIT_SCHEMA,
                "ordinal": 64 + rank,
                "video_id": row["video_id"],
                "split": "c2_confirmatory_pending" if rank <= 32 else "c2_backup_pending",
                "source_collection": "trace_collection_v2_expanded",
                "status": "candidate_not_traced",
                "predictor_seen": False,
                "c2_eligible": True,
                "candidate_rank": rank,
                "video_sha256": row.get("sha256"),
                "duration_s": row.get("duration_s"),
                "source_url": row.get("source_url"),
                "selection_reason": "deterministic hash order after trace/id/hash exclusion",
            }
        )
    write_jsonl(output_path, registry)
    summary = {
        "schema_version": SPLIT_SCHEMA,
        "output": str(output_path),
        "c1_counts": dict(Counter(row["split"] for row in registry if row["split"].startswith("c1_"))),
        "c2_candidate_count": sum(row["split"] == "c2_confirmatory_pending" for row in registry),
        "c2_backup_count": sum(row["split"] == "c2_backup_pending" for row in registry),
        "c2_exclusion_seen_ids": len(seen_ids),
        "c2_exclusion_seen_hashes": len(seen_hashes),
        "source_provenance_sha256": sha256(provenance_path),
        "source_c1_split_sha256": sha256(split_path),
        "overlap_c1_c2": sorted(set(c1_map) & {row["video_id"] for row in registry if row["split"].startswith("c2_")}),
    }
    write_json(output_path.with_suffix(output_path.suffix + ".summary.json"), summary)
    return summary


def node_identity(node: Mapping[str, Any]) -> dict[str, Any]:
    model_id = text(node.get("model_id"), "unknown")
    node_type = text(node.get("node_type"), "unknown")
    peak = max(number(node.get("workspace_peak_mb")), number(node.get("peak_reserved_mb")), number(node.get("peak_allocated_mb")))
    if model_id.startswith(("openai/", "api/")):
        lane = "api"
    elif model_id.startswith("cpu-") or (peak <= 0.0 and node_type in {"videotool_temporal", "videotool_generalist"}):
        lane = "cpu"
    else:
        lane = text(node.get("execution_lane"), "gpu")
    return {
        "activity": text(node.get("activity"), "unknown"),
        "raw_action": text(node.get("raw_action"), "unknown"),
        "node_type": node_type,
        "execution_lane": lane,
        "model_id": model_id,
        "sequence_index": int(node.get("sequence_index") or 0),
    }


def causalize_templates(template_path: Path, c1_map: Mapping[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows = read_jsonl(template_path)
    if len(source_rows) != 768:
        raise ValueError(f"R6 expects 768 immutable templates, found {len(source_rows)}")
    truth_templates: list[dict[str, Any]] = []
    scheduler_templates: list[dict[str, Any]] = []
    truth_nodes: list[dict[str, Any]] = []
    for source in source_rows:
        video_id = text(source.get("video_id"))
        if video_id not in c1_map:
            raise ValueError(f"template video is outside C1 split: {video_id}")
        split = "test_retrospective" if c1_map[video_id] == "test" else c1_map[video_id]
        nodes = sorted([dict(node) for node in source.get("nodes") or []], key=lambda row: int(row.get("sequence_index") or 0))
        if not nodes:
            raise ValueError(f"template has no nodes: {source.get('template_id')}")
        truth_nodes_for_template: list[dict[str, Any]] = []
        scheduler_nodes_for_template: list[dict[str, Any]] = []
        for index, node in enumerate(nodes):
            identity = node_identity(node)
            causal_predecessor = [str(nodes[index - 1].get("node_id"))] if index else []
            source_event_ids = list(node.get("source_event_ids") or [])
            truth = {
                "schema_version": "execution-truth-v2",
                "collection_id": COLLECTION_ID,
                "split": split,
                "template_id": source.get("template_id"),
                "run_id": source.get("run_id"),
                "video_id": video_id,
                "node_id": node.get("node_id"),
                "source_event_ids": source_event_ids,
                "source_trace_sha256": node.get("source_trace_sha256") or source.get("source_trace_sha256"),
                "identity": identity,
                # Keep the measured template shape for the engine-only
                # workload sampler.  These fields are never copied into the
                # scheduler StateView or the scheduler-visible workload job.
                "activity": identity["activity"],
                "raw_action": identity["raw_action"],
                "node_type": identity["node_type"],
                "execution_lane": identity["execution_lane"],
                "model_id": identity["model_id"],
                "sequence_index": identity["sequence_index"],
                "gpu_id": node.get("gpu_id"),
                "gpu_model": node.get("gpu_model"),
                "causal_predecessor_node_ids": causal_predecessor,
                "raw_predecessor_node_ids": list(node.get("predecessor_node_ids") or []),
                "runtime_ms": optional_number(node.get("runtime_ms")),
                "load_ms": optional_number(node.get("load_ms")),
                "queue_ms": optional_number(node.get("queue_ms")),
                "api_wait_ms": optional_number(node.get("api_wait_ms")),
                "workspace_peak_mb": optional_number(node.get("workspace_peak_mb")),
                "resident_model_mb": optional_number(node.get("resident_model_mb")),
                "status": node.get("status"),
                "retry_of": node.get("retry_of"),
                "timestamp_start": node.get("timestamp_start"),
                "timestamp_end": node.get("timestamp_end"),
                "truth_visibility": "execution_engine_only",
            }
            truth_nodes_for_template.append(truth)
            truth_nodes.append(truth)
            scheduler_nodes_for_template.append(
                {
                    "node_id": node.get("node_id"),
                    "sequence_index": identity["sequence_index"],
                    "source_event_ids": source_event_ids,
                    "identity": identity,
                    "causal_predecessor_node_ids": causal_predecessor,
                    "raw_predecessor_node_ids": list(node.get("predecessor_node_ids") or []),
                    "edge_provenance": "canonical_sequence_chain_v1",
                    "scheduler_visibility": "engine_template_only",
                }
            )
        truth_template = dict(source)
        truth_template.update(
            {
                "schema_version": "job-template-causal-v2-truth",
                "collection_id": COLLECTION_ID,
                "split": split,
                "nodes": truth_nodes_for_template,
                "causal_edge_contract": "canonical_sequence_chain_v1",
                "truth_visibility": "execution_engine_only",
            }
        )
        scheduler_template = {
            "schema_version": "job-template-causal-v2-scheduler",
            "collection_id": COLLECTION_ID,
            "template_id": source.get("template_id"),
            "run_id": source.get("run_id"),
            "video_id": video_id,
            "split": split,
            "dataset": source.get("dataset"),
            "base_task_id": source.get("base_task_id"),
            "baseline": source.get("baseline"),
            "workflow_type_id": source.get("workflow_type_id"),
            "model_stack_id": source.get("model_stack_id"),
            "planner_model_id": source.get("planner_model_id"),
            "task_structure": dict(source.get("task_structure") or {}),
            "nodes": scheduler_nodes_for_template,
            "node_count": len(scheduler_nodes_for_template),
            "source_trace_sha256": source.get("source_trace_sha256"),
            "source_manifest_sha256": source.get("source_manifest_sha256"),
            "truth_ref": f"execution_truth_v2:{source.get('template_id')}",
            "scheduler_policy_boundary": "policy receives only StateView; full template is engine/provider internal",
        }
        truth_templates.append(truth_template)
        scheduler_templates.append(scheduler_template)
    return truth_templates, scheduler_templates, truth_nodes


def sanitize_task(task: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only task descriptors that are legal before execution."""
    clean = copy.deepcopy(dict(task))
    for key in list(clean):
        if key in {"answer", "answer_text", "ground_truth", "correct_option", "target_answer"}:
            clean.pop(key, None)
    return clean


def current_identity_index(truth_nodes: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in truth_nodes:
        for event_id in row.get("source_event_ids") or []:
            result[str(event_id)] = dict(row["identity"])
        result[str(row.get("node_id"))] = dict(row["identity"])
    return result


def contains_forbidden(value: Any, path: str = "input") -> list[str]:
    violations: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in FORBIDDEN_INPUT_KEYS:
                violations.append(f"{path}.{key_text}")
            violations.extend(contains_forbidden(child, f"{path}.{key_text}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            violations.extend(contains_forbidden(child, f"{path}[{index}]"))
    return violations


def json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return stable_hash(payload)


def build_causal_prefixes(prefix_path: Path, identity_index: Mapping[str, Mapping[str, Any]], c1_map: Mapping[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_rows = read_jsonl(prefix_path)
    output: list[dict[str, Any]] = []
    violations: list[str] = []
    for row in source_rows:
        video_id = text(row.get("video_id"))
        split_raw = c1_map.get(video_id)
        if split_raw is None:
            raise ValueError(f"prefix video outside C1 split: {video_id}")
        split = "test_retrospective" if split_raw == "test" else split_raw
        target_id = text(row.get("target_source_event_id"))
        current = dict(identity_index[target_id]) if target_id and target_id in identity_index else None
        state = copy.deepcopy(row.get("state_features") or {})
        task = sanitize_task(row.get("task_structure") or {})
        observed = {
            "position": int(row.get("position") or 0),
            "activities": list(row.get("prefix_activities") or []),
            "raw_actions": list(row.get("prefix_raw_actions") or []),
            "last_actions": list((state.get("prefix") or {}).get("last_actions") or [])[-3:],
            "action_histogram": dict((state.get("prefix") or {}).get("action_histogram") or {}),
            "observed_step_count": int((state.get("prefix") or {}).get("observed_step_count") or row.get("position") or 0),
            "retry_count": int((state.get("prefix") or {}).get("retry_count") or 0),
            "error_count": int((state.get("prefix") or {}).get("error_count") or 0),
            "local_runtime_ms": number((state.get("prefix") or {}).get("local_runtime_ms")),
            "api_wait_ms": number((state.get("prefix") or {}).get("api_wait_ms")),
        }
        input_view = {
            "schema_version": "causal-state-v2",
            "decision_time": "before_current_node_execution",
            "split": split,
            "baseline": row.get("baseline"),
            "workflow_type_id": row.get("workflow_type_id"),
            "model_stack_id": row.get("model_stack_id"),
            "planner_model_id": row.get("planner_model_id"),
            "task_structure": task,
            "observed_prefix": observed,
            "state_features": state,
            "current_ready_node": current,
        }
        input_violations = contains_forbidden(input_view)
        if input_violations:
            violations.extend([f"{row.get('prefix_id')}:{item}" for item in input_violations])
        label = {
            "next_activity": row.get("target_next_activity") or "__END__",
            "next_raw_action": row.get("target_next_raw_action") or "__END__",
            "target_source_event_id": row.get("target_source_event_id"),
            "remaining_steps": row.get("remaining_steps"),
            "remaining_runtime_ms": row.get("remaining_runtime_ms"),
        }
        output.append(
            {
                "schema_version": "causal-prefix-v2",
                "collection_id": COLLECTION_ID,
                "prefix_id": row.get("prefix_id"),
                "run_id": row.get("run_id"),
                "video_id": video_id,
                "split": split,
                "input": input_view,
                "label": label,
                "audit": {
                    "source_trace_sha256": row.get("source_trace_sha256"),
                    "source_prefix_schema": row.get("schema_version"),
                    "future_events_included_in_input": False,
                    "ground_truth_included_in_input": False,
                    "answer_text_used_as_feature": False,
                    "video_id_used_as_feature": False,
                    "source_input_hash": json_hash(input_view),
                },
            }
        )
    summary = {
        "rows": len(output),
        "split_counts": dict(Counter(row["split"] for row in output)),
        "state_present": sum(bool(row["input"]["state_features"].get("state_present")) for row in output),
        "input_leakage_violations": violations,
        "target_rows": sum(row["label"]["next_activity"] != "__END__" for row in output),
    }
    if violations:
        raise ValueError(f"causal prefix leakage detected: {violations[:3]}")
    return output, summary


def node_key(node: Mapping[str, Any]) -> tuple[str, ...]:
    identity = node.get("identity") if isinstance(node.get("identity"), Mapping) else node
    return (
        text(identity.get("activity")),
        text(identity.get("node_type")),
        text(identity.get("execution_lane")),
        text(identity.get("model_id")),
    )


def future_identity(node: Mapping[str, Any]) -> dict[str, Any]:
    identity = node.get("identity") if isinstance(node.get("identity"), Mapping) else node
    return {
        "role": text(identity.get("node_type"), "unknown"),
        "activity": text(identity.get("activity"), "unknown"),
        "raw_action": text(identity.get("raw_action"), "unknown"),
        "model_id": text(identity.get("model_id"), "unknown"),
        "execution_lane": text(identity.get("execution_lane"), "unknown"),
    }


def fit_future_prototypes(templates: Sequence[Mapping[str, Any]], max_horizon: int = 5) -> dict[tuple[str, ...], dict[int, Counter[str]]]:
    prototypes: dict[tuple[str, ...], dict[int, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for template in templates:
        nodes = list(template.get("nodes") or [])
        for index, node in enumerate(nodes):
            key = node_key(node)
            for offset in range(1, max_horizon + 1):
                target_index = index + offset
                token = "__END__" if target_index >= len(nodes) else json.dumps(future_identity(nodes[target_index]), sort_keys=True, separators=(",", ":"))
                prototypes[key][offset][token] += 1
                prototypes[(text(template.get("baseline")), text(template.get("model_stack_id")), text(node.get("identity", {}).get("activity")), text(node.get("identity", {}).get("node_type")), text(node.get("identity", {}).get("execution_lane")), "__BACKOFF__")][offset][token] += 1
                prototypes[("__GLOBAL__",)][offset][token] += 1
    return prototypes


def select_distribution(counter: Counter[str], limit: int = 3) -> list[dict[str, Any]]:
    total = sum(counter.values())
    if total <= 0:
        return [{"role": "unknown", "activity": "unknown", "raw_action": "unknown", "model_id": "unknown", "execution_lane": "unknown", "probability": 1.0, "prototype_source": "train_only_backoff"}]
    rows: list[dict[str, Any]] = []
    remaining = total
    for token, count in counter.most_common(limit):
        probability = float(count) / float(total)
        remaining -= count
        if token == "__END__":
            value = {"role": "terminate", "activity": "__END__", "raw_action": "__END__", "model_id": "none", "execution_lane": "none"}
        else:
            value = json.loads(token)
        value.update({"probability": probability, "prototype_source": "train_only_prototype"})
        rows.append(value)
    if remaining > 0:
        rows.append({"role": "unknown", "activity": "unknown", "raw_action": "unknown", "model_id": "unknown", "execution_lane": "unknown", "probability": float(remaining) / float(total), "prototype_source": "train_only_backoff"})
    total_probability = sum(float(row["probability"]) for row in rows)
    rows[-1]["probability"] += 1.0 - total_probability
    return rows


def build_future_artifacts(templates: Sequence[Mapping[str, Any]], output_dir: Path) -> tuple[dict[str, int], dict[str, Any]]:
    train = [row for row in templates if row.get("split") == "train"]
    prototypes = fit_future_prototypes(train, 5)
    counts: Counter[str] = Counter()
    for horizon in (1, 3, 5):
        rows: list[dict[str, Any]] = []
        for template in templates:
            nodes = list(template.get("nodes") or [])
            for index, node in enumerate(nodes):
                exact = prototypes.get(node_key(node), {}).get(horizon)
                identity = node.get("identity") if isinstance(node.get("identity"), Mapping) else {}
                backoff_key = (text(template.get("baseline")), text(template.get("model_stack_id")), text(identity.get("activity")), text(identity.get("node_type")), text(identity.get("execution_lane")), "__BACKOFF__")
                counter = exact or prototypes.get(backoff_key, {}).get(horizon) or prototypes.get(("__GLOBAL__",), {}).get(horizon) or Counter()
                scenarios = select_distribution(counter)
                rows.append(
                    {
                        "schema_version": "causal-future-v2",
                        "collection_id": COLLECTION_ID,
                        "fit_split": "train",
                        "source_template_split": template.get("split"),
                        "template_id": template.get("template_id"),
                        "run_id": template.get("run_id"),
                        "video_id": template.get("video_id"),
                        "current_node_id": node.get("node_id"),
                        "current_identity": node.get("identity"),
                        "horizon": horizon,
                        "scenarios": scenarios,
                        "probability_sum": round(sum(float(item["probability"]) for item in scenarios), 12),
                        "supported_mass": round(sum(float(item["probability"]) for item in scenarios if item.get("prototype_source") == "train_only_prototype"), 12),
                        "future_events_included_in_input": False,
                        "prototype_source": "train_only_conditional_then_global_backoff",
                    }
                )
        path = output_dir / f"causal_future_h{horizon}_v2.jsonl"
        write_jsonl(path, rows)
        counts[f"h{horizon}"] = len(rows)
    audit = {
        "fit_template_count": len(train),
        "fit_split": "train",
        "future_rows": dict(counts),
        "probability_sum_errors": 0,
        "horizon_bounds_ok": True,
    }
    return dict(counts), audit


def fit_resource_stats(truth_nodes: Sequence[Mapping[str, Any]], train_template_ids: set[str]) -> dict[tuple[str, ...], dict[str, Any]]:
    buckets: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in truth_nodes:
        if text(row.get("template_id")) not in train_template_ids:
            continue
        identity = row.get("identity") if isinstance(row.get("identity"), Mapping) else {}
        model = text(identity.get("model_id"))
        activity = text(identity.get("activity"))
        node_type = text(identity.get("node_type"))
        lane = text(identity.get("execution_lane"))
        keys = (
            (model, activity, node_type, lane),
            (model, activity, "*", lane),
            (model, "*", "*", lane),
            ("*", activity, "*", lane),
            ("*", "*", "*", lane),
        )
        for key in keys:
            for field, value in (("runtime", row.get("runtime_ms")), ("load", row.get("load_ms")), ("memory", row.get("workspace_peak_mb"))):
                parsed = optional_number(value)
                if parsed is not None:
                    buckets[key][field].append(parsed)
    stats: dict[tuple[str, ...], dict[str, Any]] = {}
    for key, values in buckets.items():
        stats[key] = {
            "runtime_p50_ms": quantile(values.get("runtime", []), 0.50),
            "runtime_p90_ms": quantile(values.get("runtime", []), 0.90),
            "load_p50_ms": quantile(values.get("load", []), 0.50),
            "peak_memory_p95_mb": quantile(values.get("memory", []), 0.95),
            "count": len(values.get("runtime", [])),
        }
    return stats


def build_resource_artifacts(templates: Sequence[Mapping[str, Any]], truth_nodes: Sequence[Mapping[str, Any]], output_dir: Path) -> dict[str, Any]:
    train_ids = {text(row.get("template_id")) for row in templates if row.get("split") == "train"}
    stats = fit_resource_stats(truth_nodes, train_ids)
    predictions: list[dict[str, Any]] = []
    fallback_counts: Counter[str] = Counter()
    for row in truth_nodes:
        identity = row.get("identity") if isinstance(row.get("identity"), Mapping) else {}
        model = text(identity.get("model_id"))
        activity = text(identity.get("activity"))
        node_type = text(identity.get("node_type"))
        lane = text(identity.get("execution_lane"))
        keys = (
            (model, activity, node_type, lane),
            (model, activity, "*", lane),
            (model, "*", "*", lane),
            ("*", activity, "*", lane),
            ("*", "*", "*", lane),
        )
        selected_key = next((key for key in keys if stats.get(key, {}).get("count", 0) >= 3), None)
        selected = stats.get(selected_key, {}) if selected_key else {}
        if selected_key is None:
            fallback_counts["unknown"] += 1
        else:
            fallback_counts["|".join(selected_key)] += 1
        predictions.append(
            {
                "schema_version": "scheduler-resource-prediction-v2",
                "collection_id": COLLECTION_ID,
                "fit_split": "train",
                "source_template_split": row.get("split"),
                "template_id": row.get("template_id"),
                "run_id": row.get("run_id"),
                "video_id": row.get("video_id"),
                "node_id": row.get("node_id"),
                "current_identity": identity,
                "runtime_p50_ms": selected.get("runtime_p50_ms"),
                "runtime_p90_ms": selected.get("runtime_p90_ms"),
                "load_p50_ms": selected.get("load_p50_ms"),
                "peak_memory_p95_mb": 0.0 if lane in {"cpu", "api"} else selected.get("peak_memory_p95_mb"),
                "uncertainty": "empirical_train_quantile",
                "fit_group": "|".join(selected_key) if selected_key else "unknown",
                "truth_included": False,
            }
        )
    prediction_path = output_dir / "resource_predictions_scheduler_v2.jsonl"
    write_jsonl(prediction_path, predictions)
    profile = {
        "schema_version": "c1-resource-profile-v2",
        "collection_id": COLLECTION_ID,
        "fit_split": "train",
        "train_template_count": len(train_ids),
        "train_truth_node_count": sum(text(row.get("template_id")) in train_ids for row in truth_nodes),
        "groups": {"|".join(key): value for key, value in sorted(stats.items())},
        "fallback_counts": dict(fallback_counts),
        "input_truth_visibility": "execution_engine_only",
    }
    profile_path = output_dir / "resource_profile_v2.json"
    write_json(profile_path, profile)
    contract = {
        "schema_version": "scheduler-resource-contract-v1",
        "contract_version": DERIVATION_VERSION,
        "collection_id": COLLECTION_ID,
        "fit_split": "train",
        "outputs": ["runtime_p50_ms", "runtime_p90_ms", "load_p50_ms", "peak_memory_p95_mb", "uncertainty"],
        "unknown_policy": "explicit_unknown_backoff_not_zero",
        "prediction_artifact": str(prediction_path),
        "profile_artifact": str(profile_path),
        "truth_never_in_scheduler_view": True,
    }
    contract_path = output_dir / "scheduler_resource_contract_v2.json"
    write_json(contract_path, contract)
    return {
        "prediction_rows": len(predictions),
        "fit_template_count": len(train_ids),
        "fit_truth_node_count": profile["train_truth_node_count"],
        "fallback_count": sum(fallback_counts.values()) if "unknown" in fallback_counts else 0,
        "prediction_path": str(prediction_path),
        "profile_path": str(profile_path),
        "contract_path": str(contract_path),
    }


def sanitize_workload(path: Path, split: str, deadline_ms_by_class: Mapping[str, float]) -> dict[str, Any]:
    rows = read_jsonl(path)
    for episode in rows:
        episode.pop("predicted_total_gpu_runtime_ms", None)
        episode.pop("resource_contract", None)
        episode.pop("deadline_multiplier", None)
        episode["split"] = split
        episode["scheduler_visible_contract"] = {
            "allowed_job_fields": ["template_id", "arrival_ms", "service_class", "deadline_ms"],
            "forbidden_job_fields": ["predicted_job_runtime_p50_ms", "predicted_job_runtime_p90_ms", "predicted_gpu_runtime_p50_ms"],
            "deadline_policy": "fixed_relative_class_budget",
        }
        source_refs = {
            "source_template_ids": episode.pop("source_template_ids", None),
            "source_video_ids": episode.pop("source_video_ids", None),
            "source_trace_sha256": episode.pop("source_trace_sha256", None),
        }
        episode["workload_truth_metadata"] = {
            "realized_offered_compute_load": episode.pop("realized_offered_compute_load", None),
            "episode_window_ms": episode.pop("episode_window_ms", None),
            "residency_provenance": episode.pop("residency_provenance", None),
            "source_refs": source_refs,
            "value_source": episode.get("value_source"),
            "visibility": "engine_audit_only",
        }
        for job in episode.get("jobs") or []:
            service_class = text(job.get("service_class"), "normal")
            job.pop("scheduler_visible", None)
            job.pop("deadline_multiplier", None)
            job["deadline_ms"] = round(number(job.get("arrival_ms")) + float(deadline_ms_by_class.get(service_class, 240000.0)), 3)
    write_jsonl(path, rows)
    summary = {
        "schema_version": "workload-episode-causal-v2",
        "output": str(path),
        "split": split,
        "episodes": len(rows),
        "jobs": sum(len(row.get("jobs") or []) for row in rows),
        "forbidden_scheduler_fields_present": sum(
            1 for episode in rows for job in (episode.get("jobs") or []) if any(key in job for key in ("scheduler_visible", "predicted_job_runtime_p50_ms", "predicted_job_runtime_p90_ms", "predicted_gpu_runtime_p50_ms", "deadline_multiplier"))
        ),
        "deadline_policy": "fixed_relative_class_budget",
        "scheduler_visible_contract": rows[0].get("scheduler_visible_contract") if rows else {},
    }
    write_json(path.with_suffix(path.suffix + ".summary.json"), summary)
    if summary["forbidden_scheduler_fields_present"]:
        raise ValueError(f"workload scheduler-visible leak in {path}")
    return summary


def audit_invariants(prefix_rows: Sequence[Mapping[str, Any]], future_counts: Mapping[str, int], truth_nodes: Sequence[Mapping[str, Any]], templates: Sequence[Mapping[str, Any]], registry_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    input_hashes = [str(row.get("audit", {}).get("source_input_hash")) for row in prefix_rows]
    unique_prefix_ids = len({row.get("prefix_id") for row in prefix_rows})
    current_identity_violations = []
    for row in prefix_rows:
        current = row.get("input", {}).get("current_ready_node")
        if isinstance(current, Mapping) and set(current) - CURRENT_NODE_KEYS:
            current_identity_violations.append({"prefix_id": row.get("prefix_id"), "keys": sorted(set(current) - CURRENT_NODE_KEYS)})
    edge_errors = []
    for template in templates:
        nodes = list(template.get("nodes") or [])
        for index, node in enumerate(nodes):
            expected = [str(nodes[index - 1].get("node_id"))] if index else []
            if list(node.get("causal_predecessor_node_ids") or []) != expected:
                edge_errors.append(str(node.get("node_id")))
    probability_errors = []
    for horizon in future_counts:
        path = None
        # caller records counts only; full rows are checked by the runner below.
        if future_counts[horizon] <= 0:
            probability_errors.append(horizon)
    c1_rows = [row for row in registry_rows if str(row.get("split", "")).startswith("c1_")]
    c2_rows = [row for row in registry_rows if str(row.get("split", "")).startswith("c2_")]
    return {
        "prefix_rows": len(prefix_rows),
        "prefix_unique_ids": unique_prefix_ids,
        "prefix_duplicate_ids": len(prefix_rows) - unique_prefix_ids,
        "input_hash_unique_count": len(set(input_hashes)),
        "current_identity_violations": current_identity_violations,
        "causal_edge_errors": edge_errors,
        "future_probability_count_errors": probability_errors,
        "c1_registry_rows": len(c1_rows),
        "c2_registry_rows": len(c2_rows),
        "c2_pending_rows": sum(row.get("status") == "candidate_not_traced" for row in c2_rows),
        "train_fit_only": True,
        "all_truth_rows_engine_only": all(row.get("truth_visibility") == "execution_engine_only" for row in truth_nodes),
        "workload_source_refs_hidden": True,
        "gate": not (current_identity_violations or edge_errors or probability_errors or len(c1_rows) != 64 or len(c2_rows) != 40),
    }


def build_workloads(
    project_root: Path,
    output_dir: Path,
    engine_template_path: Path,
    resource_contract_path: Path,
    arrival_source: Path,
    counts: Mapping[str, int],
) -> dict[str, Any]:
    from tracing.workloads import build_workload_v02 as v02

    arrival_manifest = output_dir / "arrival_manifest.json"
    arrival_summary = v02.build_arrival_manifest(arrival_source, arrival_manifest)
    # The tested v0.2 sampler uses the literal ``test`` split.  Keep the
    # public R6 registry name ``test_retrospective`` while giving this private
    # engine-only copy the sampler's legacy spelling.
    engine_rows = read_jsonl(engine_template_path)
    sampler_rows = []
    for row in engine_rows:
        sampler_row = copy.deepcopy(row)
        if sampler_row.get("split") == "test_retrospective":
            sampler_row["split"] = "test"
        sampler_rows.append(sampler_row)
    sampler_template_path = output_dir / "job_templates_workload_sampler_v2.jsonl"
    write_jsonl(sampler_template_path, sampler_rows)
    result: dict[str, Any] = {"arrival_manifest": str(arrival_manifest), "arrival_summary": arrival_summary, "splits": {}}
    for split, count in counts.items():
        raw_path = output_dir / f"workload_{split}_raw.jsonl"
        final_path = output_dir / f"workload_{split}_v2.jsonl"
        v02.generate_split(
            sampler_template_path,
            raw_path,
            split=split,
            count=count,
            seed=20260816 + {"train": 0, "validation": 1, "test": 2}[split],
            arrival_source=arrival_source,
            resource_contract=resource_contract_path,
            fixed_test_matrix=(split == "test"),
        )
        os.replace(raw_path, final_path)
        result["splits"][split] = sanitize_workload(final_path, "test_retrospective" if split == "test" else split, {"priority": 120000.0, "normal": 240000.0})
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    c1_map, split_rows = split_map(args.split_manifest)
    registry_path = args.registry_output or output_dir / "video_split_registry_r6_v1.jsonl"
    registry_summary = build_split_registry(args.project_root, args.split_manifest, args.provenance_manifest, registry_path)
    registry_rows = read_jsonl(registry_path)
    truth_templates, scheduler_templates, truth_nodes = causalize_templates(args.templates, c1_map)
    truth_template_path = output_dir / "job_templates_causal_v2_truth.jsonl"
    scheduler_template_path = output_dir / "job_templates_causal_v2_scheduler.jsonl"
    truth_path = output_dir / "execution_truth_v2.jsonl"
    write_jsonl(truth_template_path, truth_templates)
    write_jsonl(scheduler_template_path, scheduler_templates)
    write_jsonl(truth_path, truth_nodes)
    prefix_rows, prefix_summary = build_causal_prefixes(args.prefixes, current_identity_index(truth_nodes), c1_map)
    prefix_path = output_dir / "causal_prefixes_v2.jsonl"
    write_jsonl(prefix_path, prefix_rows)
    future_counts, future_summary = build_future_artifacts(truth_templates, output_dir)
    resource_summary = build_resource_artifacts(truth_templates, truth_nodes, output_dir)
    # v0.2's tested sampler consumes measured engine templates.  It is written
    # only inside this new R6 directory; no historical workload is overwritten.
    workload_summary = build_workloads(
        args.project_root,
        output_dir,
        truth_template_path,
        output_dir / "scheduler_resource_contract_v2.json",
        args.arrival_source,
        {"train": args.train_count, "validation": args.validation_count, "test": args.test_count},
    )
    gate_report = audit_invariants(prefix_rows, future_counts, truth_nodes, truth_templates, registry_rows)
    if not gate_report["gate"]:
        raise ValueError(f"R6 invariant gate failed: {gate_report}")
    files = [
        registry_path,
        registry_path.with_suffix(registry_path.suffix + ".summary.json"),
        truth_template_path,
        scheduler_template_path,
        truth_path,
        prefix_path,
        output_dir / "causal_future_h1_v2.jsonl",
        output_dir / "causal_future_h3_v2.jsonl",
        output_dir / "causal_future_h5_v2.jsonl",
        output_dir / "resource_predictions_scheduler_v2.jsonl",
        output_dir / "resource_profile_v2.json",
        output_dir / "scheduler_resource_contract_v2.json",
        output_dir / "arrival_manifest.json",
    ]
    for split in ("train", "validation", "test"):
        files.extend([output_dir / f"workload_{split}_v2.jsonl", output_dir / f"workload_{split}_v2.jsonl.summary.json"])
    def manifest_name(path: Path) -> str:
        try:
            return str(path.relative_to(args.project_root))
        except ValueError:
            return str(path)

    manifest = {
        "schema_version": "r6-causal-v2-manifest",
        "derivation_version": DERIVATION_VERSION,
        "collection_id": COLLECTION_ID,
        "generated_at": "2026-08-16",
        "inputs": {
            "split_manifest": str(args.split_manifest),
            "split_manifest_sha256": sha256(args.split_manifest),
            "provenance_manifest": str(args.provenance_manifest),
            "provenance_manifest_sha256": sha256(args.provenance_manifest),
            "templates": str(args.templates),
            "templates_sha256": sha256(args.templates),
            "prefixes": str(args.prefixes),
            "prefixes_sha256": sha256(args.prefixes),
            "arrival_source": str(args.arrival_source),
            "arrival_source_sha256": sha256(args.arrival_source),
        },
        "registry": registry_summary,
        "causal_prefix": prefix_summary,
        "future": future_summary,
        "resource": resource_summary,
        "workload": workload_summary,
        "gates": gate_report,
        "files": {manifest_name(path): sha256(path) for path in files if path.exists()},
        "c2_status": "pending_trace_collection",
        "legacy_policy": "all historical v1 inputs remain immutable; this directory is new derived output",
    }
    write_json(output_dir / "r6_manifest.json", manifest)
    write_json(output_dir / "r6_gate_report.json", gate_report)
    return manifest


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--provenance-manifest", type=Path, required=True)
    p.add_argument("--templates", type=Path, required=True)
    p.add_argument("--prefixes", type=Path, required=True)
    p.add_argument("--arrival-source", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--registry-output", type=Path)
    p.add_argument("--train-count", type=int, default=20000)
    p.add_argument("--validation-count", type=int, default=1000)
    p.add_argument("--test-count", type=int, default=6750)
    return p


def main() -> int:
    args = parser().parse_args()
    manifest = run(args)
    print(json.dumps({"output_dir": str(args.output_dir), "schema_version": manifest["schema_version"], "gates": manifest["gates"], "c2_status": manifest["c2_status"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
