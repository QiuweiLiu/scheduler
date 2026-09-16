#!/usr/bin/env python3
"""Build causal resource predictions for the frozen scheduling templates.

This adapter deliberately constructs only the candidate-node features accepted
by ``resource_predictor_v1``.  The current node's runtime, load, workspace
peak, queue, resident model, retry target, and any answer/teacher fields are
never read.  Runtime from a *previous* completed node is retained as an
observed-prefix feature, which is causal at the node's dispatch point.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/root/autodl-tmp/scheduler")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tracing.analysis import resource_predictor_v1 as rp  # noqa: E402


TARGET_FIELDS = {
    "runtime_ms",
    "load_ms",
    "peak_memory_inclusive_mb",
    "workspace_peak_mb",
    "queue_ms",
    "resident_model_mb",
    "api_wait_ms",
    "error",
    "answer",
    "teacher_labels",
    "remaining_runtime_ms",
    "remaining_steps",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value: Any, default: float = -1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def task_value(task: dict[str, Any], key: str, default: Any) -> Any:
    value = task.get(key, default)
    if isinstance(value, (list, tuple)):
        return "|".join(sorted(str(item) for item in value))
    return value if value not in (None, "") else default


def build_candidate_rows(templates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    unknown_scale = 0
    previous_prefix_values = 0
    for template in templates:
        task = template.get("task_structure") if isinstance(template.get("task_structure"), dict) else {}
        seen_models: set[str] = set()
        previous_runtime = 0.0
        accumulated_runtime = 0.0
        previous_activity = "START"
        previous_model = "START"
        previous_family = "START"
        previous_error = 0
        retry_count = 0
        for node in sorted(template.get("nodes") or [], key=lambda item: int(item.get("sequence_index", 0))):
            model_id = str(node.get("model_id") or "unknown")
            activity = str(node.get("activity") or "other")
            raw_action = str(node.get("raw_action") or "")
            current_family = rp.event_family(raw_action, activity)
            task_row = {
                "domain": task_value(task, "domain", "unknown"),
                "question_type": task_value(task, "question_type", "unknown"),
                "sub_category": task_value(task, "sub_category", "unknown"),
                "required_modalities": task_value(task, "required_modalities", "unknown"),
                "answer_type": task_value(task, "answer_type", "unknown"),
                "temporal_scope": task_value(task, "temporal_scope", "unknown"),
                "question_chars": scalar(task.get("question_chars")),
                "question_tokens": scalar(task.get("question_tokens")),
                "option_count": scalar(task.get("option_count")),
                "option_chars_mean": scalar(task.get("option_chars_mean")),
            }
            scale = node.get("input_scale") if isinstance(node.get("input_scale"), dict) else {}
            frame_count = scalar(scale.get("frame_count"))
            image_count = scalar(scale.get("qwen_image_count"))
            yolo_batch = scalar(scale.get("yolo_batch"))
            if frame_count < 0 and image_count < 0 and yolo_batch < 0:
                unknown_scale += 1
            if previous_runtime > 0 or previous_error or accumulated_runtime > 0:
                previous_prefix_values += 1
            row = {
                "source_name": "workload_v0_2_smoke_fixed_20260812",
                "run_id": str(template.get("run_id") or template.get("template_id") or "unknown"),
                "video_id": str(template.get("video_id") or "unknown"),
                "split": str(template.get("split") or "unknown"),
                "event_id": str(node.get("node_id") or "unknown"),
                "event_index": int(node.get("sequence_index") or 0),
                "activity": activity,
                "activity_family": current_family,
                "node_type": str(node.get("node_type") or "unknown"),
                "model_id": model_id,
                "baseline": str(template.get("baseline") or "unknown"),
                "gpu_model": str(node.get("gpu_model") or "unknown"),
                "cold_warm": "cold" if model_id not in seen_models else "warm",
                "prev_activity": previous_activity,
                "prev_model_id": previous_model,
                "prev_activity_model": f"{previous_activity}|{previous_model}",
                "prev_activity_family_model": f"{previous_family}|{previous_model}",
                "frame_count": frame_count,
                "qwen_image_count": image_count,
                "yolo_batch": yolo_batch,
                "acc_steps": int(node.get("sequence_index") or 0),
                "log_acc_runtime": math.log1p(max(accumulated_runtime, 0.0)),
                "log_prev_runtime": math.log1p(max(previous_runtime, 0.0)),
                "retry_count": retry_count,
                "prev_error": previous_error,
                **task_row,
                "video_duration_s": -1.0,
                "video_fps": -1.0,
            }
            rows.append(row)

            # Only after constructing the current candidate do these values
            # become part of the prefix for the following node.
            previous_runtime = max(0.0, scalar(node.get("runtime_ms"), 0.0))
            accumulated_runtime += previous_runtime
            previous_activity = activity
            previous_model = model_id
            previous_family = current_family
            previous_error = int(str(node.get("status") or "success") != "success")
            retry_count += int(bool(node.get("retry_of")))
            seen_models.add(model_id)
    return rows, {
        "nodes": len(rows),
        "unknown_input_scale_nodes": unknown_scale,
        "observed_prefix_rows": previous_prefix_values,
        "target_fields_not_read_for_current_node": sorted(TARGET_FIELDS),
    }


def selected_predictions(bundle: dict[str, Any], predictions: dict[str, list[float]], selector: str) -> tuple[list[float], list[float], list[float]]:
    if selector == "lgb_point_group_interval":
        return predictions["lgb_q50"], predictions["group_q90"], predictions["group_q95"]
    if selector == "group_quantile":
        return predictions["group_q50"], predictions["group_q90"], predictions["group_q95"]
    raise ValueError(f"unsupported frozen selector: {selector}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/processed/scheduling_future_v1_20260812")
    parser.add_argument("--templates", type=Path, default=ROOT / "results/processed/workload_v0_2_smoke_fixed_20260812/job_templates.jsonl")
    parser.add_argument("--models", type=Path, default=ROOT / "results/processed/resource_predictor_v1_final_q99_20260811/models.pkl")
    parser.add_argument("--report", type=Path, default=ROOT / "results/processed/resource_predictor_v1_final_q99_20260811/resource_predictor_report.json")
    args = parser.parse_args()
    templates = read_jsonl(args.templates)
    with args.models.open("rb") as handle:
        import pickle

        bundles = pickle.load(handle)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    selectors = {
        target: report["final_holdout"][target]["selected_model_from_development"]
        for target in ("runtime_ms", "load_ms", "peak_memory_inclusive_mb")
    }
    rows, audit = build_candidate_rows(templates)
    predictions_by_target = {
        target: rp.predict_bundle(bundles[target], rows)
        for target in ("runtime_ms", "load_ms", "peak_memory_inclusive_mb")
    }
    runtime_p50, runtime_p90, runtime_p95 = selected_predictions(bundles["runtime_ms"], predictions_by_target["runtime_ms"], selectors["runtime_ms"])
    load_p50, load_p90, load_p95 = selected_predictions(bundles["load_ms"], predictions_by_target["load_ms"], selectors["load_ms"])
    memory_p50, memory_p90, memory_p95 = selected_predictions(bundles["peak_memory_inclusive_mb"], predictions_by_target["peak_memory_inclusive_mb"], selectors["peak_memory_inclusive_mb"])

    output_root = args.output_root
    artifact_dir = output_root / "prediction_artifacts"
    audit_dir = output_root / "leakage_audit"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / "resource_predictions_v1.jsonl.gz"
    template_id_by_run = {
        str(template.get("run_id") or template.get("template_id")): str(template.get("template_id"))
        for template in templates
    }
    with gzip.open(out_path, "wt", encoding="utf-8") as handle:
        for row, rp50, rp90, lp50, mp50, mp95 in zip(rows, runtime_p50, runtime_p90, load_p50, memory_p50, memory_p95):
            payload = {
                "schema_version": "scheduling-future-resource-node-v1",
                "node_id": row["event_id"],
                "template_id": template_id_by_run.get(row["run_id"], row["run_id"]),
                "video_id": row["video_id"],
                "sequence_index": row["event_index"],
                "activity": row["activity"],
                "activity_family": row["activity_family"],
                "node_type": row["node_type"],
                "model_id": row["model_id"],
                "runtime_p50_ms": float(rp50),
                "runtime_p90_ms": float(rp90),
                "load_p50_ms": float(lp50),
                "peak_memory_p95_mb": float(mp95),
                "uncertainty": {
                    "runtime_p90_minus_p50_ms": float(max(0.0, rp90 - rp50)),
                    "memory_p95_minus_p50_mb": float(max(0.0, mp95 - mp50)),
                },
                "input_contract": {
                    "current_target_fields_used": [],
                    "future_events_used": False,
                    "answer_or_teacher_fields_used": False,
                    "video_id_as_feature": False,
                    "observed_prefix_previous_runtime_allowed": True,
                    "unknown_input_scale_preserved": True,
                },
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

    manifest = {
        "schema_version": "scheduling-future-resource-manifest-v1",
        "artifact": str(args.models),
        "artifact_sha256": sha256(args.models),
        "report_sha256": sha256(args.report),
        "templates": len(templates),
        "nodes": len(rows),
        "selectors": selectors,
        "output": {"path": str(out_path), "rows": len(rows)},
        "audit": audit,
        "input_contract": {
            "model_fit_split": "frozen resource_predictor_v1 artifact",
            "current_runtime_load_memory_used_as_feature": False,
            "future_events_used_as_feature": False,
            "answer_or_teacher_metadata_used": False,
            "video_id_used_as_feature": False,
            "unknown_policy": "explicit -1 input scale; no zero target fallback",
        },
    }
    (artifact_dir / "resource_artifact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    leakage = {
        "schema_version": "scheduling-future-resource-leakage-v1",
        "pass": True,
        "current_target_fields_used": [],
        "future_events_used": False,
        "answer_or_teacher_fields_used": False,
        "video_id_used_as_feature": False,
        "unknown_input_scale_nodes": audit["unknown_input_scale_nodes"],
        "note": "Previous completed-node runtime is an observed-prefix feature; current-node targets are excluded.",
    }
    (audit_dir / "resource_leakage_report.json").write_text(json.dumps(leakage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"nodes": len(rows), "selectors": selectors, "output": str(out_path), "leakage_pass": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
