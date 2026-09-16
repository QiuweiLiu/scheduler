#!/usr/bin/env python3
"""Build the predictor-aligned P9d topology dataset from causal trace prefixes.

The source of truth for future topology labels is the raw trace's
``parent_step_ids``.  Behavior role samples provide the exact anchor set and
video-level split, so this dataset can be joined to the existing behavior
dataset without using scheduler traces or future-derived features as input.

This module intentionally has no training dependency.  It emits separate
feature and label JSONL files for each split and refuses to overwrite an
existing output directory.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple


DATASET_SCHEMA_VERSION = "topology-predictor-p9d-v1"
FEATURE_SCHEMA_VERSION = "topology-predictor-features-v1"
LABEL_SCHEMA_VERSION = "topology-predictor-labels-v1"
HORIZON = 5
MAX_WIDTH_BUCKET = 4
HOLDOUT_SOURCE = "final_holdout_v1"
DEV_SPLITS = ("train", "validation", "test")
OUTPUT_SPLITS = DEV_SPLITS + ("holdout",)
SUPPORTED_EVENT_TYPES = {"run", "api_call", "action"}

# --- v3 label policy constants -------------------------------------------------
LABEL_POLICIES = ("v2", "v3")
DEFAULT_LABEL_POLICY = "v2"
V2_LABEL_POLICY = "v2_step_graph_last_event"
V3_LABEL_POLICY = "v3_verified_serial_control_flow"
V3_SUPPORTED_BASELINES = ("star", "langgraph_react", "st_fixed")
TOOL_STEP_ID_RE = re.compile(r"^compat-(?:star|react)-step-(\d+)$")
PARSED_PLANNER_STATUSES = ("json", "bare_tool_token")

ROLE_BY_NODE = {
    "run_control": "init",
    "planner": "plan",
    "answer_generation": "aggregate",
    "videotool_spatial": "execute",
    "videotool_temporal": "execute",
    "videotool_generalist": "execute",
}

RAW_TO_FAMILY = {
    "sample_seek": "select_frames",
    "frame-selector": "select_frames",
    "image-grid-selector": "select_frames",
    "spatial_qa": "visual_qa",
    "image-qa": "visual_qa",
    "image-grid-qa": "visual_qa",
    "patch-zoomer": "visual_qa",
    "temporal-qa": "temporal_ops",
    "temporal-grounding": "temporal_ops",
    "summarize": "summarize",
    "summarization-tool": "summarize",
    "object_detection": "detect",
    "yolo-tracker": "detect",
}

# These names are forbidden only inside ``model_input``.  Metadata such as
# run_id and the true future labels remain in their separate audit columns.
FORBIDDEN_MODEL_INPUT_KEYS = frozenset(
    {
        "video_id",
        "run_id",
        "event_id",
        "node_id",
        "source_event_id",
        "target_source_event_id",
        "successor_node_ids",
        "predecessor_node_ids",
        "successors",
        "predecessors",
        "future_events",
        "future_state",
        "future_layers",
        "next_role",
        "family_label",
        "runtime_ms",
        "local_runtime_ms",
        "load_ms",
        "remaining_steps",
        "remaining_runtime_ms",
        "memory",
        "memory_mb",
        "peak_allocated_mb",
        "peak_reserved_mb",
        "resource",
        "status",
        "answer",
        "answer_label",
        "gold",
        "truth",
        "video_path",
        "trace_sha256",
    }
)


def open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open_text(path, "r") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            yield value


def write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(values: Iterable[Any]) -> str:
    payload = json.dumps(list(values), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def text(value: Any, default: str = "unknown") -> str:
    if value is None or value == "":
        return default
    return str(value)


def role_for_event(event: Mapping[str, Any]) -> str:
    return ROLE_BY_NODE.get(text(event.get("node_type"), ""), "execute")


def action_for_event(event: Mapping[str, Any]) -> str:
    if text(event.get("node_type"), "") == "run_control":
        return "baseline_start"
    return text(event.get("action"), "other")


def family_for_event(event: Mapping[str, Any]) -> str:
    return RAW_TO_FAMILY.get(action_for_event(event), "other")


def load_video_splits(path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    groups = data.get("groups") or {}
    result: Dict[str, str] = {}
    for group_name, group in groups.items():
        if group_name == "P_dev":
            for split, video_ids in (group.get("subsplit") or {}).items():
                if split not in DEV_SPLITS:
                    continue
                for video_id in video_ids:
                    video_id = text(video_id, "")
                    if video_id in result:
                        raise ValueError(f"video appears in multiple registry splits: {video_id}")
                    result[video_id] = split
        elif group_name == "P_holdout_diag":
            for video_id in group.get("video_ids") or []:
                video_id = text(video_id, "")
                if video_id in result:
                    raise ValueError(f"holdout video overlaps P_dev: {video_id}")
                result[video_id] = "holdout"
    if not result:
        raise ValueError(f"no P_dev/P_holdout_diag videos found in {path}")
    canonical_by_input = {video_id: video_id for video_id in result}
    raw_to_canonical = ((data.get("normalization") or {}).get("raw_development_to_canonical") or {})
    for raw_id, canonical_id in raw_to_canonical.items():
        raw_id = text(raw_id, "")
        canonical_id = text(canonical_id, "")
        if canonical_id not in result:
            raise ValueError(f"normalization points outside P_dev/P_holdout registry: {raw_id} -> {canonical_id}")
        existing = canonical_by_input.get(raw_id)
        if existing is not None and existing != canonical_id:
            raise ValueError(f"conflicting video normalization: {raw_id} -> {existing}, {canonical_id}")
        canonical_by_input[raw_id] = canonical_id
    return result, canonical_by_input


def _anchor_key(run_id: str, event_id: str) -> str:
    return f"{run_id}::{event_id}"


def select_behavior_anchors(
    behavior_role_samples: Path,
    holdout_role_samples: Path,
    video_splits: Mapping[str, str],
    canonical_by_input: Mapping[str, str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select the exact behavior-role anchors for P_dev and holdout.

    The final holdout file deliberately has ``split=validation`` for its
    diagnostic rows, so its source marker—not that field—selects holdout.
    """

    anchors: Dict[str, Dict[str, Any]] = {}
    stats: Dict[str, Any] = {
        "source_rows": {"P_dev": 0, "P_holdout_diag": 0},
        "duplicate_anchor_rows": {"P_dev": 0, "P_holdout_diag": 0},
        "unique_anchors": {"P_dev": 0, "P_holdout_diag": 0},
    }

    def add_row(row: Dict[str, Any], scope: str, split: str) -> None:
        video_id = text(row.get("video_id"), "")
        run_id = text(row.get("run_id"), "")
        event_id = text(row.get("source_event_id"), "")
        if not video_id or not run_id or not event_id:
            raise ValueError(f"behavior anchor is missing video_id/run_id/source_event_id: {row}")
        registry_video_id = canonical_by_input.get(video_id, video_id)
        expected = video_splits.get(registry_video_id)
        if expected != split:
            raise ValueError(
                f"video split mismatch for {video_id} (registry={registry_video_id}): "
                f"expected {expected!r}, got {split!r}"
            )
        key = _anchor_key(run_id, event_id)
        existing = anchors.get(key)
        if existing is not None:
            if (existing["video_id"], existing["registry_video_id"], existing["split"]) != (video_id, registry_video_id, split):
                raise ValueError(f"anchor appears with conflicting metadata: {key}")
            stats["duplicate_anchor_rows"][scope] += 1
            return
        anchors[key] = {
            "sample_id": key,
            "video_id": video_id,
            "registry_video_id": registry_video_id,
            "run_id": run_id,
            "current_node_id": event_id,
            "split": split,
            "sample_row": row,
        }

    for row in read_jsonl(behavior_role_samples):
        stats["source_rows"]["P_dev"] += 1
        video_id = text(row.get("video_id"), "")
        if video_splits.get(canonical_by_input.get(video_id, video_id)) == "holdout":
            raise ValueError(f"P_dev behavior file contains holdout video: {video_id}")
        split = text(row.get("split"), "")
        if split not in DEV_SPLITS:
            raise ValueError(f"unexpected P_dev split {split!r} for {video_id}")
        add_row(row, "P_dev", split)

    for row in read_jsonl(holdout_role_samples):
        if text(row.get("source"), "") != HOLDOUT_SOURCE and text(row.get("source_split"), "") != HOLDOUT_SOURCE:
            continue
        stats["source_rows"]["P_holdout_diag"] += 1
        add_row(row, "P_holdout_diag", "holdout")

    stats["unique_anchors"] = {
        "P_dev": sum(1 for row in anchors.values() if row["split"] in DEV_SPLITS),
        "P_holdout_diag": sum(1 for row in anchors.values() if row["split"] == "holdout"),
    }
    ordered = sorted(anchors.values(), key=lambda row: (row["split"], row["video_id"], row["run_id"], row["current_node_id"]))
    return ordered, stats


def index_trace_paths(raw_roots: Sequence[Path], run_ids: Set[str]) -> Dict[str, List[Path]]:
    candidates: Dict[str, List[Path]] = defaultdict(list)
    for root in raw_roots:
        if root.is_file():
            paths = [root] if root.name == "trace.jsonl" else []
        else:
            paths = sorted(root.rglob("trace.jsonl")) if root.exists() else []
        for trace_path in paths:
            candidate_run_id = trace_path.parent.name
            if candidate_run_id in run_ids:
                candidates[candidate_run_id].append(trace_path)
    missing = sorted(run_ids.difference(candidates))
    if missing:
        raise RuntimeError(f"raw trace coverage failed: {len(missing)} missing run_ids, first={missing[:5]}")
    return {run_id: sorted(paths) for run_id, paths in candidates.items()}


def resolve_trace_path(candidates: Sequence[Path], expected_hashes: Set[str]) -> Tuple[Path, Dict[str, Any]]:
    if not candidates:
        raise ValueError("cannot resolve an empty trace candidate list")
    records: List[Tuple[Path, Dict[str, Any]]] = []
    for trace_path in candidates:
        manifest_path = trace_path.parent / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing run_manifest.json beside {trace_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(f"invalid run manifest: {manifest_path}")
        if not manifest.get("trace_sha256"):
            manifest = dict(manifest)
            manifest["trace_sha256"] = sha256(trace_path)
            manifest["trace_sha256_source"] = "computed_from_trace"
        records.append((trace_path, manifest))

    if expected_hashes:
        matched = [(path, manifest) for path, manifest in records if text(manifest.get("trace_sha256"), "") in expected_hashes]
        if not matched:
            raise ValueError(f"source_trace_sha256 does not match any raw candidate: {[str(p) for p, _ in records]}")
        if len(matched) > 1:
            raise ValueError(f"source_trace_sha256 matches multiple raw candidates: {[str(p) for p, _ in matched]}")
        return matched[0]
    if len(records) != 1:
        raise ValueError(f"ambiguous raw trace without source hash: {[str(p) for p, _ in records]}")
    return records[0]


def load_trace(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for row in read_jsonl(path):
        event_id = text(row.get("event_id"), "")
        event_type = text(row.get("event_type"), "")
        if not event_id or event_type not in SUPPORTED_EVENT_TYPES:
            continue
        if event_id in seen:
            raise ValueError(f"duplicate event_id in {path}: {event_id}")
        seen.add(event_id)
        events.append(row)
    if not events:
        raise ValueError(f"trace has no supported events: {path}")
    return events


def _step_key(value: Any) -> str:
    return text(value, "")


def _parent_steps(event: Mapping[str, Any]) -> List[str]:
    values = event.get("parent_step_ids") or []
    if not isinstance(values, list):
        raise ValueError(f"parent_step_ids must be a list for {event.get('event_id')}")
    return [_step_key(value) for value in values]


def _workload_scale(event: Mapping[str, Any]) -> Dict[str, Any]:
    """Workload-scale descriptor recovered from causally visible inputs.

    Only fields present in the raw event ``input`` are used; query text
    itself is excluded (only its length is kept) to avoid memorization.
    Missing values stay ``None``/``0`` instead of being imputed.
    """
    raw_input = event.get("input")
    params = (raw_input.get("parameters") if isinstance(raw_input, Mapping) else None) or {}
    clip_len: Optional[float] = None
    try:
        if "start_time" in params and "end_time" in params:
            clip_len = float(params["end_time"]) - float(params["start_time"])
    except (TypeError, ValueError):
        clip_len = None
    query = params.get("query")
    nested = raw_input.get("nested_api_call_count") if isinstance(raw_input, Mapping) else None
    return {
        "clip_len": clip_len,
        "query_char_len": len(query) if isinstance(query, str) else 0,
        "nested_api_call_count": nested if type(nested) is int else 0,
    }


def _node_label(event: Mapping[str, Any], sequence_index: int, predecessors: Sequence[str]) -> Dict[str, Any]:
    return {
        "node_id": text(event.get("event_id"), ""),
        "sequence_index": sequence_index,
        "step_id": event.get("step_id"),
        "event_type": text(event.get("event_type"), "unknown"),
        "node_type": text(event.get("node_type"), "unknown"),
        "role": role_for_event(event),
        "raw_action": action_for_event(event),
        "action_family": family_for_event(event),
        "model_id": text(event.get("model_id")),
        # The raw trace does not expose the scheduler execution lane.  Keeping
        # this explicit is safer than deriving it from runtime or model names.
        "execution_lane": "unknown",
        "execution_lane_source": "not_in_trace",
        "workload_scale": _workload_scale(event),
        "predecessor_node_ids": list(predecessors),
    }


def build_graph(events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Reconstruct event-level edges and return a forward adjacency graph.

    ``run_control`` is retained as an anchor but excluded from future target
    nodes, matching the scheduler workload node boundary.  A root anchor is
    connected to the first graph roots so the initial behavior row remains
    joinable.
    """

    event_positions = {text(event.get("event_id"), ""): index for index, event in enumerate(events)}
    all_step_to_events: Dict[str, List[str]] = defaultdict(list)
    for event in events:
        all_step_to_events[_step_key(event.get("step_id"))].append(text(event.get("event_id"), ""))

    graph_events = [
        (index, event)
        for index, event in enumerate(events)
        if text(event.get("node_type"), "") != "run_control"
    ]
    step_to_graph_ids: Dict[str, List[str]] = defaultdict(list)
    for index, event in graph_events:
        step_to_graph_ids[_step_key(event.get("step_id"))].append(text(event.get("event_id"), ""))

    graph_nodes: Dict[str, Dict[str, Any]] = {}
    predecessors: Dict[str, List[str]] = {}
    edges = 0
    explicit_parent_refs = 0
    missing_parent_refs = 0
    forward_parent_refs = 0
    same_step_fallback_edges = 0
    step_parent_last_event_expansions = 0
    step_parent_dropped_edges = 0
    previous_graph_id: Optional[str] = None

    for sequence_index, event in graph_events:
        event_id = text(event.get("event_id"), "")
        parent_steps = _parent_steps(event)
        explicit_parent_refs += len(parent_steps)
        candidate_ids: List[str] = []
        for parent_step in parent_steps:
            parent_event_ids = all_step_to_events.get(parent_step)
            if not parent_event_ids:
                missing_parent_refs += 1
                continue
            prior_parent_ids = [
                parent_id
                for parent_id in parent_event_ids
                if event_positions.get(parent_id, sequence_index) < sequence_index
            ]
            if not prior_parent_ids:
                forward_parent_refs += 1
                continue
            # A step-level parent means "after that step", i.e. after its
            # last prior graph event — not "depends on every event of it".
            # Expanding to all step events invents intra-layer edges.
            prior_graph_ids = sorted(
                {
                    candidate_id
                    for candidate_id in step_to_graph_ids.get(parent_step, [])
                    if event_positions.get(candidate_id, sequence_index) < sequence_index
                },
                key=lambda candidate_id: event_positions[candidate_id],
            )
            if prior_graph_ids:
                step_parent_last_event_expansions += 1
                step_parent_dropped_edges += max(0, len(prior_graph_ids) - 1)
                candidate_ids.append(prior_graph_ids[-1])
        candidate_ids = sorted(
            {candidate_id for candidate_id in candidate_ids if event_positions.get(candidate_id, sequence_index) < sequence_index},
            key=lambda candidate_id: event_positions[candidate_id],
        )
        if parent_steps and not candidate_ids:
            raise ValueError(f"parent steps do not map to a prior graph node: {event_id} <- {parent_steps}")
        if not parent_steps and previous_graph_id is not None:
            candidate_ids = [previous_graph_id]
            same_step_fallback_edges += 1
        predecessors[event_id] = candidate_ids
        graph_nodes[event_id] = _node_label(event, sequence_index, candidate_ids)
        edges += len(candidate_ids)
        previous_graph_id = event_id

    if missing_parent_refs or forward_parent_refs:
        raise ValueError(
            f"invalid parent references: missing={missing_parent_refs}, forward={forward_parent_refs}"
        )

    successors: Dict[str, List[str]] = defaultdict(list)
    roots: List[str] = []
    for node_id, node_predecessors in predecessors.items():
        if not node_predecessors:
            roots.append(node_id)
        for predecessor in node_predecessors:
            successors[predecessor].append(node_id)
    roots.sort(key=lambda node_id: event_positions[node_id])
    for node_id in successors:
        successors[node_id].sort(key=lambda child_id: event_positions[child_id])

    return {
        "event_positions": event_positions,
        "graph_nodes": graph_nodes,
        "predecessors": predecessors,
        "successors": dict(successors),
        "roots": roots,
        "stats": {
            "raw_events": len(events),
            "graph_nodes": len(graph_nodes),
            "edges": edges,
            "root_nodes": len(roots),
            "explicit_parent_refs": explicit_parent_refs,
            "missing_parent_refs": missing_parent_refs,
            "forward_parent_refs": forward_parent_refs,
            "same_step_fallback_edges": same_step_fallback_edges,
            "step_parent_last_event_expansions": step_parent_last_event_expansions,
            "step_parent_dropped_edges": step_parent_dropped_edges,
        },
    }


def _longest_path_depths(graph: Mapping[str, Any]) -> Dict[str, int]:
    """Longest-path depth from roots; every edge strictly increases depth.

    Event positions are a topological order (all predecessors are positionally
    prior by construction), so a single pass in position order is exact.
    """
    predecessors = graph["predecessors"]
    event_positions = graph["event_positions"]
    depths: Dict[str, int] = {}
    for node_id in sorted(predecessors, key=lambda value: event_positions[value]):
        parent_depths = [depths[parent_id] for parent_id in predecessors[node_id] if parent_id in depths]
        depths[node_id] = (max(parent_depths) + 1) if parent_depths else 0
    return depths


def future_layers(graph: Mapping[str, Any], current_node_id: str, horizon: int = HORIZON) -> List[List[str]]:
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    graph_nodes = graph["graph_nodes"]
    successors = graph["successors"]
    event_positions = graph["event_positions"]
    depths = _longest_path_depths(graph)
    if current_node_id in graph_nodes:
        base_depth = depths[current_node_id]
        reachable: Set[str] = set()
        frontier = list(successors.get(current_node_id, []))
        while frontier:
            node_id = frontier.pop()
            if node_id in reachable:
                continue
            reachable.add(node_id)
            frontier.extend(successors.get(node_id, []))
    else:
        # The only non-graph anchor currently expected is run_control.
        base_depth = -1
        reachable = set(graph_nodes)
    layers: List[List[str]] = []
    for offset in range(1, horizon + 1):
        next_ids = sorted(
            (node_id for node_id in reachable if depths[node_id] == base_depth + offset),
            key=lambda node_id: event_positions[node_id],
        )
        if not next_ids:
            break
        layers.append(next_ids)
    return layers


# --- v3 verified-serial-control-flow reconstruction ----------------------------


def _is_terminal_marker(event: Mapping[str, Any]) -> bool:
    """The final ``answer`` run event closes a run; it is not schedulable work."""
    return str(event.get("event_type")) == "run" and str(event.get("action")) == "answer"


def _resource_applicable(event: Mapping[str, Any]) -> bool:
    if str(event.get("node_type")) == "run_control":
        return False
    if _is_terminal_marker(event):
        return False
    return True


def _resource_signature(
    event: Mapping[str, Any],
    merged: bool,
    nested_event: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    raw_input = event.get("input")
    mode = text(raw_input.get("planner_mode"), "unknown") if isinstance(raw_input, Mapping) else "unknown"
    signature: Dict[str, Any] = {
        "exec_class": text(event.get("node_type")),
        "tool_family": family_for_event(event),
        "model_class": text(event.get("model_id")),
        "planner_mode": mode,
        "merged_nested_call": bool(merged),
    }
    if merged and nested_event is not None:
        # Composite signature: the outer tool wrapper plus the inner model call.
        signature["nested_model_class"] = text(nested_event.get("model_id"))
    return signature


def _nested_call_record(event: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "event_id": text(event.get("event_id"), ""),
        "event_type": text(event.get("event_type")),
        "node_type": text(event.get("node_type")),
        "action": text(event.get("action")),
        "role": role_for_event(event),
        "action_family": family_for_event(event),
        "model_id": text(event.get("model_id")),
    }


def build_chain(events: Sequence[Mapping[str, Any]], baseline: str) -> Dict[str, Any]:
    """Reconstruct the verified top-level serial control-flow chain (v3).

    Fail-closed: every seriality/signal invariant violation is collected in
    ``violations``; callers must refuse to emit labels when it is non-empty.
    """

    positions = {text(event.get("event_id"), ""): index for index, event in enumerate(events)}
    violations: List[str] = []
    stats: Counter = Counter()
    steps: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        steps[int(event.get("step_id") or 0)].append(event)

    if baseline not in V3_SUPPORTED_BASELINES:
        violations.append(f"unsupported baseline {baseline!r}")

    # Nested answer calls live inside the Summarizer tool; merge them.
    nested_map: Dict[str, str] = {}
    nested_by_parent: Dict[str, str] = {}
    for step, step_events in steps.items():
        for index, event in enumerate(step_events):
            if str(event.get("node_type")) == "answer_generation" and str(event.get("action")) == "generalist.generate":
                followers = [x for x in step_events[index + 1:] if str(x.get("action")) == "summarization-tool"]
                if followers:
                    nested_map[event["event_id"]] = followers[0]["event_id"]
    for nested_id, parent_id in nested_map.items():
        if parent_id in nested_by_parent:
            violations.append(f"multiple nested calls merged into one tool: {parent_id}")
        nested_by_parent[parent_id] = nested_id
    stats["nested_merged_calls"] = len(nested_map)

    chain_events = [
        event for event in events
        if str(event.get("node_type")) != "run_control" and event["event_id"] not in nested_map
    ]
    chain_index = {event["event_id"]: index for index, event in enumerate(chain_events)}

    # Seriality gate: one tool per iteration, step-level parents only, step order.
    for step, step_events in steps.items():
        tools = [x for x in step_events if str(x.get("event_type")) == "action"]
        if len(tools) > 1:
            violations.append(f"multi-tool step {step}: {[str(t.get('action')) for t in tools]}")
        planners = [x for x in step_events if str(x.get("node_type")) == "planner"]
        parsed_planners = [
            p for p in planners
            if text((p.get("input") or {}).get("parse_status"), "") in PARSED_PLANNER_STATUSES
        ]
        for tool in tools:
            call = (tool.get("input") or {}).get("standard_tool_call") or {}
            match = TOOL_STEP_ID_RE.match(text(call.get("id"), ""))
            if match:
                stats["tool_step_id_present"] += 1
                if int(match.group(1)) != int(tool.get("step_id") or 0):
                    violations.append(f"tool step id mismatch: {tool.get('event_id')}")
                else:
                    stats["tool_step_id_match"] += 1
            if baseline in ("star", "langgraph_react"):
                action = str(tool.get("action"))
                if parsed_planners:
                    names = {
                        text(((p.get("input") or {}).get("parsed_decision") or {}).get("tool_name"), "")
                        for p in parsed_planners
                    }
                    if action not in names:
                        violations.append(f"planner tool_name mismatch: {tool.get('event_id')} action={action} names={sorted(names)}")
                    else:
                        stats["planner_tool_name_match"] += 1
                elif not planners and action == "yolo-tracker":
                    stats["yolo_preobserve_tools"] += 1
                else:
                    violations.append(f"tool without parsed same-step planner: {tool.get('event_id')} action={action}")

    for event in events:
        parents = event.get("parent_step_ids") or []
        if parents:
            if len(parents) != 1 or int(parents[0]) != int(event.get("step_id") or 0) - 1:
                violations.append(
                    f"non-canonical parent_step_ids: {event.get('event_id')} step={event.get('step_id')} parents={list(parents)}"
                )

    for event in events:
        retry_of = event.get("retry_of")
        if not retry_of:
            continue
        stats["retry_events"] += 1
        target = str(retry_of)
        if target in chain_index and event["event_id"] in chain_index:
            if chain_index[event["event_id"]] - chain_index[target] != 1:
                violations.append(f"retry not adjacent to failed attempt: {event.get('event_id')}")
            else:
                stats["retry_adjacent"] += 1

    for index in range(1, len(chain_events)):
        previous_step = int(chain_events[index - 1].get("step_id") or 0)
        current_step = int(chain_events[index].get("step_id") or 0)
        if current_step == previous_step:
            stats["intra_step_transitions"] += 1
        elif current_step == previous_step + 1:
            stats["cross_step_transitions"] += 1
        else:
            violations.append(
                f"step jump on chain: {chain_events[index - 1].get('event_id')} -> {chain_events[index].get('event_id')}"
            )

    node_labels: Dict[str, Dict[str, Any]] = {}
    predecessors: Dict[str, List[str]] = {}
    for index, event in enumerate(chain_events):
        event_id = event["event_id"]
        prior = [chain_events[index - 1]["event_id"]] if index else []
        predecessors[event_id] = prior
        label = _node_label(event, positions[event_id], prior)
        nested_id = nested_by_parent.get(event_id)
        nested_event = None
        if nested_id is not None:
            nested_event = next(x for x in events if x["event_id"] == nested_id)
            label["merged_nested_call"] = True
            label["nested_calls"] = [_nested_call_record(nested_event)]
        else:
            label["merged_nested_call"] = False
            label["nested_calls"] = []
        label["resource_applicable"] = _resource_applicable(event)
        label["is_retry"] = bool(event.get("retry_of"))
        label["status_class"] = text(event.get("status"), "unknown")
        label["retry_of"] = text(event.get("retry_of"), "") or None
        label["resource_signature"] = _resource_signature(event, nested_id is not None, nested_event)
        label["terminal_marker"] = _is_terminal_marker(event)
        node_labels[event_id] = label
        if label["resource_applicable"]:
            stats["resource_applicable_nodes"] += 1
        else:
            stats["terminal_marker_nodes"] += 1

    return {
        "nodes": [event["event_id"] for event in chain_events],
        "node_labels": node_labels,
        "predecessors": predecessors,
        "event_positions": positions,
        "nested_map": nested_map,
        "baseline": baseline,
        "stats": stats,
        "violations": violations,
    }


def future_chain_layers(
    chain: Mapping[str, Any],
    current_node_id: str,
    horizon: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Next ``horizon`` schedulable compute nodes along the chain.

    Returns ``(layers, remaining_compute_nodes)``. The terminal ``answer``
    marker is excluded from layers (it closes the run instead of consuming a
    horizon slot).
    """

    nodes = chain["nodes"]
    labels = chain["node_labels"]
    nested_map = chain.get("nested_map") or {}
    effective_id = nested_map.get(current_node_id, current_node_id)
    if effective_id in nodes:
        start = nodes.index(effective_id) + 1
    else:
        # run_control anchors (or unknown ids) start from the chain head.
        start = 0
    remaining_ids = [node_id for node_id in nodes[start:] if labels[node_id]["resource_applicable"]]
    remaining = len(remaining_ids)
    layers: List[Dict[str, Any]] = []
    for offset, node_id in enumerate(remaining_ids[:horizon], 1):
        layers.append({"layer_offset": offset, "nodes": [dict(labels[node_id])]})
    return layers, remaining


def make_label_v3(
    anchor: Mapping[str, Any],
    chain: Mapping[str, Any],
    raw_manifest: Mapping[str, Any],
    horizon: int,
) -> Dict[str, Any]:
    layers, remaining = future_chain_layers(chain, anchor["current_node_id"], horizon)
    widths = [len(layer["nodes"]) for layer in layers]
    censored = remaining > horizon
    return {
        "schema_version": LABEL_SCHEMA_VERSION,
        "sample_id": anchor["sample_id"],
        "video_id": anchor["video_id"],
        "registry_video_id": anchor["registry_video_id"],
        "run_id": anchor["run_id"],
        "split": anchor["split"],
        "current_node_id": anchor["current_node_id"],
        "current_event_index": chain["event_positions"].get(anchor["current_node_id"], -1),
        "future_horizon": horizon,
        "future_layers": layers,
        "label_summary": {
            "future_layer_count": len(layers),
            "future_node_count": sum(widths),
            "layer_widths": widths,
            "layer_widths_capped_4plus": widths,
            "width_is_deterministic": True,
            "width_policy": "chain_width_1",
            "first_layer_width": widths[0] if widths else 0,
            "has_future": bool(layers),
            "truncated_at_horizon": censored,
            "remaining_future_compute_nodes": remaining,
            # Censored (chain continues beyond H) vs terminated (run ends within H).
            "termination_status": "censored" if censored else "terminated",
        },
        "source_trace_sha256": text(raw_manifest.get("trace_sha256"), "unknown"),
        "reconstruction_policy": V3_LABEL_POLICY,
        "label_contract": {
            "future_unit": "verified_serial_control_flow_chain",
            "edge_policy": "verified_serial_control_flow",
            "nested_call_policy": "merged_into_enclosing_tool",
            "retry_policy": "kept_as_chain_node",
            "width_policy": "deterministic_width_1_compatibility_field",
            "resource_policy": "node_total_includes_merged_nested_call",
            "termination": "terminated_vs_censored_explicit",
            "terminal_marker_excluded_from_horizon": True,
            "workload_scale": "clip_len_query_char_len_nested_api_call_count_per_node",
            "node_identity_retained_in_label": True,
            "execution_lane_source": "not_in_trace",
        },
    }


def _v2_predecessors_for_diff(events: Sequence[Mapping[str, Any]]) -> Dict[str, List[str]]:
    graph = build_graph(events)
    return {node_id: list(parents) for node_id, parents in graph["predecessors"].items()}


def _node_kind_for_diff(event: Mapping[str, Any]) -> str:
    if str(event.get("node_type")) == "planner":
        return "planner"
    if str(event.get("event_type")) == "action":
        return "tool"
    if str(event.get("node_type")) == "answer_generation" and str(event.get("action")) == "generalist.generate":
        return "generate"
    if _is_terminal_marker(event):
        return "answer"
    return "other"


def _run_baseline(raw_manifest: Mapping[str, Any], trace_path: Path) -> str:
    value = text(raw_manifest.get("baseline"), "")
    if value:
        return value
    directory = trace_path.parent.name
    for candidate in V3_SUPPORTED_BASELINES:
        if f"_{candidate}_" in directory:
            return candidate
    return "unknown"


def _task_context(sample: Mapping[str, Any]) -> Dict[str, Any]:
    task_structure = sample.get("task_structure") or {}
    result: Dict[str, Any] = {}
    for key in ("answer_type", "domain", "question_type", "required_modalities", "temporal_scope", "sub_category", "official_task_type"):
        value = sample.get(key)
        if value is None and isinstance(task_structure, Mapping):
            value = task_structure.get(key)
        if value is None:
            value = "unknown"
        if isinstance(value, list):
            result[key] = [text(item) for item in value]
        else:
            result[key] = text(value)
    return result


def _stack_context(sample: Mapping[str, Any]) -> Dict[str, str]:
    return {
        "baseline": text(sample.get("baseline")),
        "model_stack_id": text(sample.get("model_stack_id") or sample.get("model_id")),
        "planner_model_id": text(sample.get("planner_model_id")),
    }


def _history_token(event: Mapping[str, Any], position: int) -> Dict[str, Any]:
    return {
        "position": position,
        "event_type": text(event.get("event_type")),
        "node_type": text(event.get("node_type")),
        "role": role_for_event(event),
        "raw_action": action_for_event(event),
        "action_family": family_for_event(event),
        "model_id": text(event.get("model_id")),
    }


def make_feature(
    anchor: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    event_index: int,
    raw_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    sample = anchor["sample_row"]
    history = [_history_token(event, index) for index, event in enumerate(events[: event_index + 1])]
    current = dict(history[-1])
    model_input = {
        "history": history,
        "current_node": current,
        "task_context": _task_context(sample),
        "stack_context": _stack_context(sample),
    }
    leaked = audit_model_input(model_input)
    if leaked:
        raise ValueError(f"causal model input leaked forbidden keys: {leaked}")
    event_ids = [text(event.get("event_id"), "") for event in events[: event_index + 1]]
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "sample_id": anchor["sample_id"],
        "video_id": anchor["video_id"],
        "registry_video_id": anchor["registry_video_id"],
        "run_id": anchor["run_id"],
        "split": anchor["split"],
        "current_node_id": anchor["current_node_id"],
        "current_event_index": event_index,
        "prefix_hash": canonical_hash(event_ids),
        "source_trace_sha256": text(raw_manifest.get("trace_sha256"), "unknown"),
        "model_input": model_input,
        "input_contract": {
            "visible_prefix_only": True,
            "future_events_excluded": True,
            "future_edges_excluded": True,
            "remaining_steps_excluded": True,
            "execution_runtime_load_memory_status_excluded": True,
            "resource_truth_excluded": True,
            "video_id_as_model_feature": False,
            "upstream_prediction_features_included": False,
        },
    }


def make_label(
    anchor: Mapping[str, Any],
    graph: Mapping[str, Any],
    raw_manifest: Mapping[str, Any],
    horizon: int,
) -> Dict[str, Any]:
    layer_ids = future_layers(graph, anchor["current_node_id"], horizon)
    layers: List[Dict[str, Any]] = []
    for offset, node_ids in enumerate(layer_ids, 1):
        layers.append(
            {
                "layer_offset": offset,
                "nodes": [dict(graph["graph_nodes"][node_id]) for node_id in node_ids],
            }
        )
    widths = [len(layer["nodes"]) for layer in layers]
    # Width class 5 is nearly absent dataset-wide; merge 4+ into one bucket
    # instead of deleting those layers (deletion would distort topology).
    widths_capped_4plus = [min(width, MAX_WIDTH_BUCKET) for width in widths]
    truncated = bool(layers) and len(layers) == horizon and any(
        graph["successors"].get(node_id, []) for node_id in layer_ids[-1]
    )
    return {
        "schema_version": LABEL_SCHEMA_VERSION,
        "sample_id": anchor["sample_id"],
        "video_id": anchor["video_id"],
        "registry_video_id": anchor["registry_video_id"],
        "run_id": anchor["run_id"],
        "split": anchor["split"],
        "current_node_id": anchor["current_node_id"],
        "current_event_index": graph["event_positions"][anchor["current_node_id"]],
        "future_horizon": horizon,
        "future_layers": layers,
        "label_summary": {
            "future_layer_count": len(layers),
            "future_node_count": sum(widths),
            "layer_widths": widths,
            "layer_widths_capped_4plus": widths_capped_4plus,
            "width_capped_layer_count": sum(1 for width in widths if width > MAX_WIDTH_BUCKET),
            "first_layer_width": widths[0] if widths else 0,
            "has_future": bool(layers),
            "truncated_at_horizon": truncated,
            # Censored (future continues beyond H) vs terminated (truly ended).
            # Truncated rows supervise L>=H, not L==H; never treat L==H as exact.
            "termination_status": "censored" if truncated else "terminated",
        },
        "source_trace_sha256": text(raw_manifest.get("trace_sha256"), "unknown"),
        "label_contract": {
            "future_unit": "event_level_dag_longest_path_layer",
            "parent_source": "raw_trace.parent_step_ids",
            "step_parent_policy": "last_event_of_parent_step",
            "same_step_no_parent_policy": "previous_graph_node_chain",
            "termination": "terminated_vs_censored_explicit",
            "width_cap": "4plus_merged_not_deleted",
            "workload_scale": "clip_len_query_char_len_nested_api_call_count_per_node",
            "node_identity_retained_in_label": True,
            "execution_lane_source": "not_in_trace",
        },
    }


def audit_model_input(value: Any, prefix: str = "model_input") -> List[str]:
    leaked: List[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in FORBIDDEN_MODEL_INPUT_KEYS:
                leaked.append(f"{prefix}.{key_text}")
            leaked.extend(audit_model_input(child, f"{prefix}.{key_text}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            leaked.extend(audit_model_input(child, f"{prefix}[{index}]"))
    return leaked


def build_dataset(
    *,
    behavior_role_samples: Path,
    holdout_role_samples: Path,
    video_registry: Path,
    raw_roots: Sequence[Path],
    output_root: Path,
    horizon: int = HORIZON,
    label_policy: str = DEFAULT_LABEL_POLICY,
) -> Dict[str, Any]:
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if label_policy not in LABEL_POLICIES:
        raise ValueError(f"unknown label_policy {label_policy!r}; expected one of {LABEL_POLICIES}")
    if output_root.exists():
        if any(output_root.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_root}")
        raise FileExistsError(f"refusing to reuse output directory: {output_root}")

    video_splits, canonical_by_input = load_video_splits(video_registry)
    anchors, selection_stats = select_behavior_anchors(
        behavior_role_samples,
        holdout_role_samples,
        video_splits,
        canonical_by_input,
    )
    anchors_by_run: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    expected_hashes_by_run: Dict[str, Set[str]] = defaultdict(set)
    for anchor in anchors:
        anchors_by_run[anchor["run_id"]].append(anchor)
        source_hash = text(anchor["sample_row"].get("source_trace_sha256"), "")
        if source_hash:
            expected_hashes_by_run[anchor["run_id"]].add(source_hash)
    trace_index = index_trace_paths(raw_roots, set(anchors_by_run))

    output_root.parent.mkdir(parents=True, exist_ok=True)
    partial_root = output_root.parent / f".{output_root.name}.partial-{os.getpid()}"
    if partial_root.exists():
        raise FileExistsError(f"refusing to reuse partial output directory: {partial_root}")
    partial_root.mkdir(parents=False)

    handles: Dict[str, Tuple[Any, Any]] = {}
    for split in OUTPUT_SPLITS:
        handles[split] = (
            gzip.open(partial_root / f"features_{split}.jsonl.gz", "wt", encoding="utf-8", newline="\n"),
            gzip.open(partial_root / f"labels_{split}.jsonl.gz", "wt", encoding="utf-8", newline="\n"),
        )

    counts: Counter = Counter()
    video_sets: Dict[str, Set[str]] = defaultdict(set)
    registry_video_sets: Dict[str, Set[str]] = defaultdict(set)
    run_sets: Dict[str, Set[str]] = defaultdict(set)
    width_hist: Counter = Counter()
    graph_totals: Counter = Counter()
    missing_anchor_events: List[str] = []
    v3_gates: Counter = Counter()
    v3_violations: List[str] = []
    nested_by_baseline: Counter = Counter()
    anchors_on_merged_nested: Counter = Counter()
    v2_diff: Counter = Counter()
    v2_diff_examples: List[Dict[str, Any]] = []
    try:
        for run_id in sorted(anchors_by_run):
            trace_path, raw_manifest = resolve_trace_path(trace_index[run_id], expected_hashes_by_run[run_id])
            if text(raw_manifest.get("run_id"), run_id) != run_id:
                raise ValueError(f"run_manifest run_id mismatch for {trace_path}")
            if raw_manifest.get("status") not in (None, "success"):
                raise ValueError(f"selected raw trace is not successful: {trace_path}")
            events = load_trace(trace_path)
            graph = build_graph(events)
            graph_totals.update(graph["stats"])
            positions = graph["event_positions"]
            chain: Optional[Dict[str, Any]] = None
            v2_preds: Dict[str, List[str]] = {}
            event_by_id: Dict[str, Mapping[str, Any]] = {}
            if label_policy == "v3":
                baseline = _run_baseline(raw_manifest, trace_path)
                chain = build_chain(events, baseline)
                for key, value in chain["stats"].items():
                    v3_gates[key] += value
                v3_gates["runs"] += 1
                v3_violations.extend(chain["violations"])
                nested_by_baseline[baseline] += len(chain["nested_map"])
                v2_preds = _v2_predecessors_for_diff(events)
                event_by_id = {event["event_id"]: event for event in events}
                # v2 -> v3 predecessor diff is per run, not per anchor.
                for node_id in chain["nodes"]:
                    v2_parents = sorted(v2_preds.get(node_id, []))
                    v3_parents = sorted(chain["predecessors"][node_id])
                    if v2_parents == v3_parents:
                        continue
                    node_event = event_by_id[node_id]
                    node_kind = _node_kind_for_diff(node_event)

                    def kind_of(parent_id: str) -> str:
                        parent_event = event_by_id.get(parent_id)
                        return _node_kind_for_diff(parent_event) if parent_event is not None else "unknown"

                    v2_kind = "+".join(kind_of(parent_id) for parent_id in v2_parents) or "none"
                    v3_kind = "+".join(kind_of(parent_id) for parent_id in v3_parents) or "none"
                    v2_diff[f"{node_kind}:{v2_kind}->{v3_kind}"] += 1
                    if len(v2_diff_examples) < 5:
                        v2_diff_examples.append(
                            {
                                "run_id": run_id,
                                "node_id": node_id,
                                "node_kind": node_kind,
                                "v2_predecessors": v2_parents,
                                "v3_predecessors": v3_parents,
                            }
                        )
            for anchor in sorted(anchors_by_run[run_id], key=lambda row: positions.get(row["current_node_id"], 1 << 60)):
                event_index = positions.get(anchor["current_node_id"])
                if event_index is None:
                    missing_anchor_events.append(anchor["sample_id"])
                    continue
                feature = make_feature(anchor, events, event_index, raw_manifest)
                if label_policy == "v3":
                    assert chain is not None
                    if anchor["current_node_id"] in chain["nested_map"]:
                        anchors_on_merged_nested[anchor["split"]] += 1
                    label = make_label_v3(anchor, chain, raw_manifest, horizon)
                else:
                    label = make_label(anchor, graph, raw_manifest, horizon)
                feature_handle, label_handle = handles[anchor["split"]]
                feature_handle.write(json.dumps(feature, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                label_handle.write(json.dumps(label, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                counts[anchor["split"]] += 1
                video_sets[anchor["split"]].add(anchor["video_id"])
                registry_video_sets[anchor["split"]].add(anchor["registry_video_id"])
                run_sets[anchor["split"]].add(run_id)
                for width in label["label_summary"]["layer_widths"]:
                    width_hist[str(width)] += 1
    finally:
        for feature_handle, label_handle in handles.values():
            feature_handle.close()
            label_handle.close()

    if missing_anchor_events:
        raise RuntimeError(f"behavior anchor coverage failed: {len(missing_anchor_events)} missing, first={missing_anchor_events[:5]}")
    if label_policy == "v3" and v3_violations:
        raise RuntimeError(
            f"v3 seriality gate failed: {len(v3_violations)} violations, first={v3_violations[:5]}"
        )

    files = {}
    for split in OUTPUT_SPLITS:
        files[f"features_{split}"] = f"features_{split}.jsonl.gz"
        files[f"labels_{split}"] = f"labels_{split}.jsonl.gz"

    alignment = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "behavior_role_samples": str(behavior_role_samples),
        "holdout_role_samples": str(holdout_role_samples),
        "selection": selection_stats,
        "produced_rows": {split: counts[split] for split in OUTPUT_SPLITS},
        "produced_videos": {split: len(video_sets[split]) for split in OUTPUT_SPLITS},
        "produced_registry_videos": {split: len(registry_video_sets[split]) for split in OUTPUT_SPLITS},
        "produced_runs": {split: len(run_sets[split]) for split in OUTPUT_SPLITS},
        "unmatched_behavior_anchors": missing_anchor_events,
        "join_key": ["video_id", "run_id", "current_node_id"],
        "behavior_target_columns_not_copied": ["next_role", "target_source_event_id", "family_label"],
    }
    (partial_root / "alignment_report.json").write_text(
        json.dumps(alignment, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )

    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "status": "generated",
        "builder": {
            "script": "scripts/build_p9d_topology_dataset.py",
            "builder_version": f"{DATASET_SCHEMA_VERSION}+{label_policy}",
            "python_dependencies": "standard_library_only",
        },
        "horizon": horizon,
        "label_policy": label_policy,
        "source": {
            "video_registry": str(video_registry),
            "behavior_role_samples": str(behavior_role_samples),
            "holdout_role_samples": str(holdout_role_samples),
            "raw_roots": [str(path) for path in raw_roots],
            "raw_label_source": "trace.jsonl parent_step_ids",
            "scheduler_trace_groups_used_for_fit": False,
            "s_train_s_val_t_final_used": False,
        },
        "split_policy": {
            "P_dev": "behavior role rows retain registry train/validation/test",
            "P_holdout_diag": "rows selected by source/source_split=final_holdout_v1 and emitted as holdout",
            "video_level_split": True,
        },
        "counts": {
            "rows_by_split": {split: counts[split] for split in OUTPUT_SPLITS},
            "videos_by_split": {split: len(video_sets[split]) for split in OUTPUT_SPLITS},
            "registry_videos_by_split": {split: len(registry_video_sets[split]) for split in OUTPUT_SPLITS},
            "runs_by_split": {split: len(run_sets[split]) for split in OUTPUT_SPLITS},
            "width_histogram_nonempty_layers": dict(sorted(width_hist.items(), key=lambda item: int(item[0]))),
            "graph_totals": dict(graph_totals),
        },
        "files": files,
        "feature_contract": {
            "model_input_keys": ["history", "current_node", "task_context", "stack_context"],
            "future_events_excluded": True,
            "future_edges_excluded": True,
            "remaining_steps_excluded": True,
            "execution_runtime_load_memory_status_excluded": True,
            "resource_truth_excluded": True,
            "video_id_as_model_feature": False,
            "upstream_behavior_prediction_features": "not_included_in_v1",
            "forbidden_model_input_keys": sorted(FORBIDDEN_MODEL_INPUT_KEYS),
        },
        "label_contract": {
            "future_unit": "event_level_dag_longest_path_layer",
            "layer_horizon": horizon,
            "multiple_nodes_per_layer": True,
            "node_identity_retained": True,
            "predecessor_source": "raw_trace.parent_step_ids",
            "step_parent_policy": "last_event_of_parent_step",
            "same_step_no_parent_policy": "previous_graph_node_chain",
            "termination": "terminated_vs_censored_explicit",
            "width_cap": "4plus_merged_not_deleted",
            "workload_scale": "clip_len_query_char_len_nested_api_call_count_per_node",
            "execution_lane": "unknown because absent from raw trace",
        },
        "alignment_report": "alignment_report.json",
    }
    if label_policy == "v3":
        manifest["reconstruction_policy"] = V3_LABEL_POLICY
        manifest["label_contract"] = {
            "future_unit": "verified_serial_control_flow_chain",
            "layer_horizon": horizon,
            "width_policy": "deterministic_width_1_compatibility_field",
            "edge_policy": "verified_serial_control_flow",
            "nested_call_policy": "merged_into_enclosing_tool",
            "retry_policy": "kept_as_chain_node",
            "resource_policy": "node_total_includes_merged_nested_call",
            "terminal_marker": "answer_run_event_excluded_from_horizon",
            "termination": "terminated_vs_censored_explicit",
            "workload_scale": "clip_len_query_char_len_nested_api_call_count_per_node",
            "node_identity_retained": True,
            "execution_lane": "unknown because absent from raw trace",
        }
        manifest["counts"]["v3_gates"] = dict(sorted(v3_gates.items()))
        manifest["counts"]["nested_merged_by_baseline"] = dict(sorted(nested_by_baseline.items()))
        manifest["counts"]["anchors_on_merged_nested"] = dict(sorted(anchors_on_merged_nested.items()))
        manifest["v2_comparison"] = {
            "predecessor_changes_total": sum(v2_diff.values()),
            "predecessor_changes_by_class": dict(sorted(v2_diff.items())),
            "examples": v2_diff_examples,
        }
    (partial_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    partial_root.rename(output_root)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior-role-samples", type=Path, required=True)
    parser.add_argument("--holdout-role-samples", type=Path, required=True)
    parser.add_argument("--video-registry", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--label-policy", choices=LABEL_POLICIES, default=DEFAULT_LABEL_POLICY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset(
        behavior_role_samples=args.behavior_role_samples,
        holdout_role_samples=args.holdout_role_samples,
        video_registry=args.video_registry,
        raw_roots=args.raw_root,
        output_root=args.output_root,
        horizon=args.horizon,
        label_policy=args.label_policy,
    )
    print(json.dumps({"status": manifest["status"], "label_policy": manifest["label_policy"], "counts": manifest["counts"], "output_root": str(args.output_root)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
