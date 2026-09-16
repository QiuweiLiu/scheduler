#!/usr/bin/env python3
"""Build finite-horizon B05 artifacts for the predictor-unseen R7 traces.

The older ``scheduling_future_b05_artifacts.py`` is intentionally frozen to
the legacy 768-template input.  This adapter keeps that model/checkpoint
implementation but constructs causal role/tool samples from the R7 raw
traces, then reuses the same scenario expansion and artifact schema.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/root/autodl-tmp/scheduler")
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from scheduling_future_b05_artifacts import (  # noqa: E402
    B05Runner,
    FAMILY_LABELS,
    FAMILY_TO_RAW,
    RAW_TO_FAMILY,
    ROLE_LABELS,
    actual_history_index,
    canonical_hash,
    expand_scenarios,
    node_family,
    node_role,
    prototype_map,
    read_jsonl,
    top_probabilities,
    write_gzip_jsonl,
)
from tracing.scheduling.future_topology import (  # noqa: E402
    LAYER_H5_SCHEMA_VERSION,
    project_event_scenarios_to_unary_layers,
)


ROLE_BY_NODE = {
    "run_control": "init",
    "planner": "plan",
    "answer_generation": "aggregate",
    "videotool_spatial": "execute",
    "videotool_temporal": "execute",
    "videotool_generalist": "execute",
}
FAMILY_BY_ACTION = {
    **RAW_TO_FAMILY,
    "planner.generate": "other",
    "generalist.generate": "other",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def role_for_event(event: dict[str, Any]) -> str:
    return ROLE_BY_NODE.get(str(event.get("node_type") or ""), "execute")


def action_for_event(event: dict[str, Any]) -> str:
    if str(event.get("node_type") or "") == "run_control":
        return "baseline_start"
    return str(event.get("action") or "other")


def family_for_event(event: dict[str, Any]) -> str:
    action = str(event.get("action") or "")
    return FAMILY_BY_ACTION.get(action, "other")


def task_key(task_id: str) -> list[str]:
    values = [task_id]
    values.append(re.sub(r"_r\d+$", "", task_id))
    return list(dict.fromkeys(values))


def resource_number(event: dict[str, Any], key: str, default: float = 0.0) -> float:
    resource = event.get("resource") or {}
    value = resource.get(key)
    if value is None and key == "runtime_ms":
        value = resource.get("local_runtime_ms")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if value == value and value not in (float("inf"), float("-inf")) else default


def base_task_fields(task: dict[str, Any]) -> dict[str, Any]:
    required = task.get("required_modalities") or ["scene"]
    return {
        "answer_type": "multiple_choice",
        "domain": task.get("domain") or "unknown",
        "question_type": task.get("question_type") or "unknown",
        "question_type_source": "r7_manifest_or_default",
        "required_modalities": required,
        "temporal_scope": task.get("temporal_scope") or "unspecified",
        "sub_category": task.get("sub_category") or "unknown",
        "official_task_type": task.get("official_task_type") or "unknown",
    }


def build_samples(
    trace_root: Path,
    task_manifest_path: Path,
    selected_run_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    tasks: dict[str, dict[str, Any]] = {}
    for task in load_jsonl(task_manifest_path):
        for key in task_key(str(task.get("task_id") or "")):
            tasks[key] = task

    role_rows: list[dict[str, Any]] = []
    tool_rows: list[dict[str, Any]] = []
    events_by_run: dict[str, list[dict[str, Any]]] = {}
    for trace_path in sorted(trace_root.glob("*/trace.jsonl")):
        run_dir = trace_path.parent
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = str(manifest.get("run_id") or run_dir.name)
        if selected_run_ids and run_id not in selected_run_ids:
            continue
        task_id = str(manifest.get("task_id") or "")
        task = tasks.get(task_id) or tasks.get(re.sub(r"_r\d+$", "", task_id)) or {}
        events = load_jsonl(trace_path)
        events = [event for event in events if event.get("event_id") and event.get("event_type") in {"run", "api_call", "action"}]
        if not events:
            continue
        events_by_run[run_id] = events
        trace_hash = str(manifest.get("trace_sha256") or sha256(trace_path))
        task_fields = base_task_fields(task)
        split = str((task.get("group") or "train")).lower()
        if split not in {"train", "validation", "test"}:
            split = "train"
        common = {
            **task_fields,
            "baseline": manifest.get("baseline"),
            "model_id": manifest.get("model_stack_id"),
            "model_stack_id": manifest.get("model_stack_id"),
            "planner_model_id": manifest.get("planner_model_id"),
            "video_id": task.get("video_id") or manifest.get("video_id"),
            "task_id": task_id,
            "run_id": run_id,
            "source_trace_sha256": trace_hash,
            "source_split": split,
            "split": "train",
        }
        roles = [role_for_event(event) for event in events]
        for index, event in enumerate(events):
            next_role = roles[index + 1] if index + 1 < len(roles) else "terminate"
            resource = event.get("resource") or {}
            row = {
                **common,
                "event_index": index,
                "cutoff_event_index": index,
                "current_role": roles[index],
                "current_raw_action": action_for_event(event),
                "next_role": next_role,
                "source_event_id": event["event_id"],
                "target_source_event_id": events[index + 1]["event_id"] if index + 1 < len(events) else None,
                "target_source_event_id_reason": "r7_next_trace_event",
                "is_terminal_event": index + 1 == len(events),
                "status": event.get("status") or "unknown",
                "runtime_ms": resource_number(event, "runtime_ms"),
                "peak_allocated_mb": resource.get("peak_allocated_mb"),
                "load_ms": resource.get("load_ms"),
                "api_wait_ms": resource.get("api_wait_ms", 0.0),
                "retry_of": event.get("retry_of"),
                "error_count": 1 if event.get("status") == "error" else 0,
                "retry_count": 1 if event.get("retry_of") else 0,
                "oom_count": 1 if str(event.get("error") or "").lower().find("oom") >= 0 else 0,
                "task_structure": task_fields,
            }
            role_rows.append(row)

        # The tool head predicts the action immediately after the current
        # prefix.  Its features are derived only from events at or before i.
        prefix_by_target = {str(row.get("target_source_event_id")): row for row in load_jsonl(
            run_dir / "prefix_samples_v0_1.jsonl"
        )} if (run_dir / "prefix_samples_v0_1.jsonl").is_file() else {}
        for index in range(len(events) - 1):
            current = events[index]
            target = events[index + 1]
            prefix = prefix_by_target.get(str(target["event_id"]), {})
            prefix_state = prefix.get("state_features") or {}
            evidence = prefix_state.get("evidence") or {}
            prefix_stats = prefix_state.get("prefix") or {}
            current_resource = current.get("resource") or {}
            prefix_events = events[: index + 1]
            actions = [action_for_event(item) for item in prefix_events]
            families = [family_for_event(item) for item in prefix_events]
            categorical = {
                "answer_type": task_fields["answer_type"],
                "compute_last_event_type": current.get("event_type") or "unknown",
                "compute_last_model_id": current.get("model_id") or "unknown",
                "compute_last_node_type": current.get("node_type") or "unknown",
                "domain": task_fields["domain"],
                "question_type": task_fields["question_type"],
                "raw_last_action": action_for_event(current),
                "raw_prefix_tail2": "|".join(actions[-2:]) if actions else "__START__",
                "raw_prefix_tail3": "|".join(actions[-3:]) if actions else "__START__",
                "required_modalities": "|".join(sorted(task_fields["required_modalities"])),
                "sub_category": task_fields["sub_category"],
                "temporal_scope": task_fields["temporal_scope"],
            }
            numeric = {
                "compute_event_count": float(len(prefix_events)),
                "compute_api_call_count": float(sum(item.get("event_type") == "api_call" for item in prefix_events)),
                "compute_action_count": float(sum(item.get("event_type") == "action" for item in prefix_events)),
                "compute_runtime_ms": sum(resource_number(item, "runtime_ms") for item in prefix_events),
                "compute_load_ms": sum(resource_number(item, "load_ms") for item in prefix_events),
                "compute_peak_allocated_mb": max((resource_number(item, "peak_allocated_mb") for item in prefix_events), default=0.0),
                "compute_retry_count": float(sum(bool(item.get("retry_of")) for item in prefix_events)),
                "compute_oom_count": float(sum("oom" in str(item.get("error") or "").lower() for item in prefix_events)),
                "compute_success_rate": sum(item.get("status") == "success" for item in prefix_events) / max(1, len(prefix_events)),
                "compute_model_switches": float(sum(prefix_events[j].get("model_id") != prefix_events[j - 1].get("model_id") for j in range(1, len(prefix_events)))),
                "raw_action_count": float(len(actions)),
                "raw_action_unique_count": float(len(set(actions))),
                "option_count": float(len(task.get("options") or [])),
                "question_chars": float(len(str(task.get("question") or ""))),
                "question_tokens": float(len(str(task.get("question") or "").split())),
                "option_chars_mean": float(sum(len(str(value)) for value in (task.get("options") or [])) / max(1, len(task.get("options") or []))),
                "task_text_chars_total": float(len(str(task.get("question") or "")) + sum(len(str(value)) for value in (task.get("options") or []))),
                "compute_yolo_batch": float(task.get("yolo_batch") or 0),
                "duration_s": float((prefix_state.get("video") or {}).get("duration_s") or 0.0),
                "fps": float((prefix_state.get("video") or {}).get("fps") or 0.0),
            }
            target_role = role_for_event(target)
            target_family = family_for_event(target) if target_role == "execute" else "other"
            tool_rows.append({
                **common,
                "prefix_id": f"{run_id}:semantic:{index}",
                "position": index,
                "cutoff_event_index": index,
                "target_event_index": index + 1,
                "current_role": roles[index],
                "target_role": target_role,
                "target_raw_action": action_for_event(target),
                "family_label": target_family,
                "target_source_event_id": target["event_id"],
                "vision_available": bool(prefix.get("state_present", False)),
                "missing_reason": None,
                "features": {
                    "categorical": categorical,
                    "numeric": numeric,
                    "evidence": evidence,
                    "text": {"answer_excluded": True},
                    "compute_prefix": {"cutoff_source_event_index": index + 1, "included_event_count": len(prefix_events)},
                },
                "input_contract": {
                    "answer_text_used_as_feature": False,
                    "future_events_excluded": True,
                    "ground_truth_excluded": True,
                    "remaining_runtime_excluded": True,
                    "remaining_steps_excluded": True,
                    "video_id_used_as_feature": False,
                },
                "visual_context": {
                    "frame_indices": [],
                    "source_event_ids": [],
                    "prefix_event_count": index,
                    "target_event_excluded": True,
                    "future_events_excluded": True,
                },
                "prefix_activities": [family_for_event(item) for item in prefix_events],
                "prefix_raw_actions": actions,
                "prefix_source_event_ids": [str(item["event_id"]) for item in prefix_events],
            })
    return role_rows, tool_rows, events_by_run


def markov_tables(runner: B05Runner) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    """Create causal transition priors from the frozen P_dev train rows only."""
    role_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    role_fallback: dict[str, Counter[str]] = defaultdict(Counter)
    for row in runner.role_train:
        current = str(row.get("current_role") or "init")
        raw = str(row.get("current_raw_action") or "other")
        target = str(row.get("next_role") or "terminate")
        role_counts[(current, raw)][target] += 1
        role_fallback[current][target] += 1
    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in runner.tool_train:
        family_counts[str(row.get("current_role") or "plan")][str(row.get("family_label") or "other")] += 1

    def normalize(counter: Counter[str], labels: tuple[str, ...]) -> list[dict[str, Any]]:
        total = float(sum(counter.values())) or 1.0
        return [{"label": label, "probability": float(counter.get(label, 0)) / total} for label in labels if counter.get(label, 0)]

    role_table = {(current, raw): normalize(counts, ROLE_LABELS) for (current, raw), counts in role_counts.items()}
    for current, counts in role_fallback.items():
        role_table[(current, "__fallback__")] = normalize(counts, ROLE_LABELS)
    family_table = {current: normalize(counts, FAMILY_LABELS) for current, counts in family_counts.items()}
    return role_table, family_table


def fast_markov_scenarios(
    base: dict[str, Any],
    prototypes: dict[tuple[str, str | None], dict[str, Any]],
    role_table: dict[tuple[str, str], list[dict[str, Any]]],
    family_table: dict[str, list[dict[str, Any]]],
    horizon: int,
    beam_size: int,
    initial_role_prob: Any,
    initial_family_prob: Any,
) -> dict[int, list[dict[str, Any]]]:
    """Fast causal rollout: B05 is used at the observed prefix, then train-only Markov priors."""
    beams: list[dict[str, Any]] = [{"probability": 1.0, "steps": [], "last_role": str(base.get("current_role") or "init"), "last_raw_action": str(base.get("current_raw_action") or "other")}]
    snapshots: dict[int, list[dict[str, Any]]] = {}

    def normalized(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
        total = sum(float(item["probability"]) for item in values) or 1.0
        return [{"scenario_id": f"s{index:02d}", "scenario_probability": float(item["probability"]) / total, "steps": item["steps"], "synthetic_rollout": True} for index, item in enumerate(values)]

    for offset in range(1, horizon + 1):
        expanded: list[dict[str, Any]] = []
        for beam in beams:
            if offset == 1:
                role_items = top_probabilities(initial_role_prob, ROLE_LABELS, min(beam_size, len(ROLE_LABELS)))
            else:
                role_items = role_table.get((str(beam["last_role"]), str(beam["last_raw_action"]))) or role_table.get((str(beam["last_role"]), "__fallback__"), [])
                role_items = role_items[:beam_size]
            for role_item in role_items:
                role = str(role_item["label"])
                if role == "execute":
                    if offset == 1 and initial_family_prob is not None:
                        family_items = top_probabilities(initial_family_prob, FAMILY_LABELS, min(beam_size, len(FAMILY_LABELS)))
                    else:
                        family_items = family_table.get(str(beam["last_role"]), family_table.get("plan", []))[:beam_size]
                else:
                    family_items = [{"label": None, "probability": 1.0}]
                for family_item in family_items:
                    family = family_item.get("label")
                    probability = float(beam["probability"]) * float(role_item.get("probability", 0.0)) * float(family_item.get("probability", 1.0))
                    proto = prototypes.get((role, family if role == "execute" else None)) or prototypes.get((role, None)) or {"model_id": "unknown", "execution_lane": "unknown", "raw_action": "other", "prototype_source": "unsupported"}
                    raw_action = "planner.generate" if role == "plan" else "generalist.generate" if role == "aggregate" else FAMILY_TO_RAW.get(str(family or "other"), "other") if role == "execute" else "end"
                    expanded.append({
                        "probability": probability,
                        "steps": beam["steps"] + [{"step_offset": offset, "role": role, "action_family": family, "raw_action": raw_action, "model_id": proto.get("model_id"), "execution_lane": proto.get("execution_lane"), "prototype_source": "r7_template_mode", "role_probability": float(role_item.get("probability", 0.0)), "family_probability": float(family_item.get("probability", 1.0))}],
                        "last_role": role,
                        "last_raw_action": raw_action,
                    })
        if not expanded:
            break
        expanded.sort(key=lambda item: (-float(item["probability"]), json.dumps(item["steps"], sort_keys=True)))
        beams = expanded[:beam_size]
        if offset in (1, 3, 5):
            snapshots[offset] = normalized(beams)
    final = normalized(beams)
    for requested in (1, 3, 5):
        snapshots.setdefault(requested, final)
    return snapshots


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, default=ROOT / "results/raw/r7_trace_full_20260817")
    parser.add_argument("--task-manifest", type=Path, default=ROOT / "data/manifests/r7_trace_manifest_v1.jsonl")
    parser.add_argument("--templates", type=Path, default=ROOT / "results/processed/r7_workload_20260817/job_templates_r7_v02.jsonl")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/processed/r7_scheduling_future_20260817")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "results/processed/behavior_nn_v1_r2/final_holdout/dataset")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "results/processed/behavior_nn_v1_r2/final_holdout/evaluation/B05")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--limit-nodes", type=int, default=0)
    parser.add_argument("--rollout-mode", choices=("fast_markov", "recursive_b05"), default="fast_markov")
    parser.add_argument(
        "--emit-layer-h5",
        action="store_true",
        help="also emit the opt-in layer-H5 sidecar; current adapter is explicitly unary projection",
    )
    args = parser.parse_args()

    templates = read_jsonl(args.templates)
    selected = {str(row.get("run_id")) for row in templates}
    role_rows, tool_rows, events_by_run = build_samples(args.trace_root, args.task_manifest, selected)
    if not role_rows or not tool_rows:
        raise RuntimeError("R7 causal sample construction produced no rows")
    candidate_dir = args.output_root / "candidate_samples"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "role_event_samples.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in role_rows) + "\n", encoding="utf-8"
    )
    (candidate_dir / "semantic_tool_samples.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in tool_rows) + "\n", encoding="utf-8"
    )

    checkpoints = [args.checkpoint_dir / f"seed_{seed}" / "checkpoint.pt" for seed in (11, 22, 33)]
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    runner = B05Runner(args.dataset_dir, checkpoints, args.device)
    role_index = actual_history_index(role_rows)
    role_by_source = {str(row["source_event_id"]): row for row in role_rows}
    tool_by_run_cutoff = {(str(row["run_id"]), int(row["cutoff_event_index"])): row for row in tool_rows}
    tool_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tool_rows:
        tool_by_run[str(row["run_id"])].append(row)
    role_probabilities = runner.predict_rows(role_rows, role_index, "role")
    family_probabilities = runner.predict_rows(tool_rows, role_index, "family")
    role_prob_by_source = {str(row["source_event_id"]): role_probabilities[i] for i, row in enumerate(role_rows)}
    family_prob_by_prefix = {str(row["prefix_id"]): family_probabilities[i] for i, row in enumerate(tool_rows)}
    prototypes = prototype_map(templates)
    role_table, family_table = markov_tables(runner)

    node_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for template in templates:
        proto = prototypes[str(template["template_id"])]
        for node in template.get("nodes") or []:
            if args.limit_nodes and len(node_rows) >= args.limit_nodes:
                break
            source_id = str((node.get("source_event_ids") or [""])[0])
            role_row = role_by_source.get(source_id)
            if role_row is None:
                missing.append(source_id)
                continue
            run_id = str(role_row["run_id"])
            actual_history = role_index.get(run_id, [])
            cutoff = int(role_row.get("event_index", 0))
            tool_sample = tool_by_run_cutoff.get((run_id, cutoff))
            if tool_sample is None:
                tool_sample = tool_by_run.get(run_id, [None])[0]
            role_prob = role_prob_by_source[source_id]
            family_prob = family_prob_by_prefix.get(str(tool_sample.get("prefix_id"))) if tool_sample else None
            if args.rollout_mode == "recursive_b05":
                horizons = expand_scenarios(
                    runner,
                    role_row,
                    actual_history,
                    {int(row["cutoff_event_index"]): row for row in tool_by_run.get(run_id, [])},
                    proto,
                    5,
                    args.beam_size,
                    collect_horizons=(1, 3, 5),
                    initial_role_prob=role_prob,
                    initial_family_prob=family_prob,
                )
            else:
                horizons = fast_markov_scenarios(
                    role_row,
                    proto,
                    role_table,
                    family_table,
                    5,
                    args.beam_size,
                    role_prob,
                    family_prob,
                )
            prefix_ids = [str(row.get("source_event_id")) for row in actual_history if int(row.get("event_index", 0)) < cutoff]
            node_rows.append({
                "schema_version": "scheduling-future-b05-node-v1",
                "template_id": template["template_id"],
                "video_id": template.get("video_id"),
                "baseline": template.get("baseline"),
                "model_stack_id": template.get("model_stack_id"),
                "node_id": node.get("node_id"),
                "source_event_id": source_id,
                "sequence_index": node.get("sequence_index"),
                "prefix_hash": canonical_hash(prefix_ids),
                "current_role": role_row.get("current_role"),
                "current_raw_action": role_row.get("current_raw_action"),
                "role_top": top_probabilities(role_prob, ROLE_LABELS, len(ROLE_LABELS)),
                "family_top": top_probabilities(family_prob, FAMILY_LABELS, len(FAMILY_LABELS)) if family_prob is not None else [],
                "family_source": "r7_observed_prefix" if family_prob is not None else "unavailable",
                "future_h1": horizons[1],
                "future_h3": horizons[3],
                "future_h5": horizons[5],
                "input_contract": {
                    "fit_split": "P_dev_train_only",
                    "future_events_excluded": True,
                    "target_labels_excluded": True,
                    "observed_resource_targets_excluded": True,
                    "synthetic_rollout": True,
                },
            })
    if missing:
        raise RuntimeError(f"R7 source-event coverage failed: {len(missing)} missing, first={missing[:5]}")
    expected = args.limit_nodes or sum(len(row.get("nodes") or []) for row in templates)
    if len(node_rows) != expected:
        raise RuntimeError(f"expected {expected} node rows, got {len(node_rows)}")

    artifact_dir = args.output_root / "prediction_artifacts"
    audit_dir = args.output_root / "leakage_audit"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "h1": {"path": str(artifact_dir / "b05_node_h1.jsonl.gz"), "rows": write_gzip_jsonl(artifact_dir / "b05_node_h1.jsonl.gz", [{**row, "future_h3": None, "future_h5": None} for row in node_rows])},
        "h3": {"path": str(artifact_dir / "b05_future_h3.jsonl.gz"), "rows": write_gzip_jsonl(artifact_dir / "b05_future_h3.jsonl.gz", [{**row, "role_top": None, "family_top": None, "future_h1": None, "future_h5": None} for row in node_rows])},
        "h5": {"path": str(artifact_dir / "b05_future_h5.jsonl.gz"), "rows": write_gzip_jsonl(artifact_dir / "b05_future_h5.jsonl.gz", [{**row, "role_top": None, "family_top": None, "future_h1": None, "future_h3": None} for row in node_rows])},
    }
    if args.emit_layer_h5:
        layer_rows = [
            {
                "schema_version": LAYER_H5_SCHEMA_VERSION,
                "template_id": row["template_id"],
                "node_id": row["node_id"],
                "future_h5_layers": project_event_scenarios_to_unary_layers(row["future_h5"], 5),
                "input_contract": {
                    **row["input_contract"],
                    "topology_source": "legacy_event_to_unary_layer_projection",
                    "predicted_parallel_width": "fixed_one_not_a_topology_predictor",
                },
            }
            for row in node_rows
        ]
        outputs["h5_layers"] = {
            "path": str(artifact_dir / "b05_future_h5_layers.jsonl.gz"),
            "rows": write_gzip_jsonl(artifact_dir / "b05_future_h5_layers.jsonl.gz", layer_rows),
        }
    manifest = {
        "schema_version": "scheduling-future-b05-r7-manifest-v1",
        "model_id": "B05",
        "seeds": [11, 22, 33],
        "beam_size": args.beam_size,
        "horizons": [1, 3, 5],
        "device": args.device,
        "rollout_mode": args.rollout_mode,
        "candidate_role_rows": len(role_rows),
        "candidate_tool_rows": len(tool_rows),
        "templates": len(templates),
        "nodes": len(node_rows),
        "source_event_coverage": len(node_rows) / max(1, expected),
        "selected_run_count": len(events_by_run),
        "checkpoints": [{"path": str(path), "sha256": sha256(path)} for path in checkpoints],
        "source_templates_sha256": sha256(args.templates),
        "outputs": outputs,
        "layer_h5_contract": {
            "enabled": bool(args.emit_layer_h5),
            "schema_version": LAYER_H5_SCHEMA_VERSION,
            "topology_source": "legacy_event_to_unary_layer_projection" if args.emit_layer_h5 else None,
            "formal_topology_predictor": False,
        },
        "input_contract": {
            "fit_split": "P_dev_train_only",
            "target_labels_used_as_features": False,
            "future_events_used_as_features": False,
            "video_id_used_as_feature": False,
            "resource_truth_used_as_feature": False,
        },
    }
    (artifact_dir / "b05_artifact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (audit_dir / "r7_sample_manifest.json").write_text(json.dumps({"role_rows": len(role_rows), "tool_rows": len(tool_rows), "runs": len(events_by_run), "templates": len(templates), "nodes": len(node_rows)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"role_rows": len(role_rows), "tool_rows": len(tool_rows), "runs": len(events_by_run), "nodes": len(node_rows), "outputs": outputs}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
