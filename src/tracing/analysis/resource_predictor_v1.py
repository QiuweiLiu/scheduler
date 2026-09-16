#!/usr/bin/env python3
"""Train and audit the v1 resource predictor on the remote trace corpus.

The predictor is intentionally event-level and causal. It estimates service
runtime, load cost, and inclusive peak GPU memory for a candidate event using
only the candidate metadata and the observed prefix of the same run. Queue
time and strict OOM are reported as missing/unsupported labels rather than
invented values.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.preprocessing import OneHotEncoder


# When this file is launched as a script, make the stable import path visible
# before custom classes are serialized.  This keeps models.pkl loadable by a
# scheduler process that imports `tracing.analysis.resource_predictor_v1`.
if __name__ == "__main__":
    sys.modules.setdefault("tracing.analysis.resource_predictor_v1", sys.modules[__name__])


CATEGORICAL = [
    "activity",
    "activity_family",
    "node_type",
    "model_id",
    "baseline",
    "gpu_model",
    "cold_warm",
    "prev_activity",
    "prev_model_id",
    "prev_activity_model",
    "prev_activity_family_model",
    "domain",
    "question_type",
    "sub_category",
    "required_modalities",
    "answer_type",
    "temporal_scope",
]
NUMERICAL = [
    "event_index",
    "frame_count",
    "qwen_image_count",
    "yolo_batch",
    "acc_steps",
    "log_acc_runtime",
    "log_prev_runtime",
    "retry_count",
    "prev_error",
    "scale_missing",
    "video_duration_s",
    "video_fps",
    "question_chars",
    "question_tokens",
    "option_count",
    "option_chars_mean",
]
TARGETS = {
    "runtime_ms": "runtime_ms",
    "load_ms": "load_ms",
    "peak_memory_inclusive_mb": "peak_memory_inclusive_mb",
}
SCHEDULER_Q95_COVERAGE_FLOOR = 0.93

# Raw actions are execution metadata, not labels.  The family is deliberately
# coarse so that a new baseline can share resource statistics without using a
# future event or an answer-derived field.
FAMILY = {
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
    "planner.generate": "planner",
    "answer.generate": "answer",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_split_map(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        out: dict[str, str] = {}
        for row in read_jsonl(path):
            if row.get("video_id") and row.get("split"):
                out[str(row["video_id"])] = str(row["split"])
        return out
    value = json.loads(text)
    if isinstance(value, dict) and isinstance(value.get("split"), dict):
        return {str(k): str(v) for k, v in value["split"].items()}
    if isinstance(value, dict) and value.get("video_id") and value.get("split"):
        return {str(value["video_id"]): str(value["split"])}
    raise ValueError(f"unsupported split manifest: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text(value: Any, default: str = "unknown") -> str:
    result = str(value or "").strip()
    return result or default


def number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def scale_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def event_family(raw_action: Any, activity: Any) -> str:
    raw = text(raw_action, "")
    act = text(activity, "other")
    return FAMILY.get(raw, FAMILY.get(act, act))


def metadata_value(row: Mapping[str, Any], key: str) -> Any:
    """Read only static task/video metadata from a prefix row.

    The function intentionally does not read `features.numeric.compute_*`,
    remaining runtime/steps, teacher labels, or answer fields.  Those values
    are either future-derived or target-adjacent and must not enter a causal
    resource feature.
    """
    task_structure = row.get("task_structure")
    if not isinstance(task_structure, Mapping):
        task_structure = {}
    state = row.get("state_features")
    if not isinstance(state, Mapping):
        state = {}
    task = state.get("task")
    if not isinstance(task, Mapping):
        task = {}
    video = state.get("video")
    if not isinstance(video, Mapping):
        video = {}
    features = row.get("features")
    if not isinstance(features, Mapping):
        features = {}
    feature_cat = features.get("categorical")
    if not isinstance(feature_cat, Mapping):
        feature_cat = {}
    feature_num = features.get("numeric")
    if not isinstance(feature_num, Mapping):
        feature_num = {}

    categorical_aliases = {
        "domain": (task_structure.get("domain"), task.get("domain"), feature_cat.get("domain")),
        "question_type": (task_structure.get("question_type"), task.get("question_type"), feature_cat.get("question_type")),
        "sub_category": (task_structure.get("sub_category"), task.get("sub_category"), feature_cat.get("sub_category")),
        "required_modalities": (task_structure.get("required_modalities"), task.get("required_modalities"), feature_cat.get("required_modalities")),
        "answer_type": (task_structure.get("answer_type"), task.get("answer_type"), feature_cat.get("answer_type")),
        "temporal_scope": (task_structure.get("temporal_scope"), task.get("temporal_scope"), feature_cat.get("temporal_scope")),
    }
    numeric_aliases = {
        "video_duration_s": (video.get("duration_s"), feature_num.get("duration_s")),
        "video_fps": (video.get("fps"), feature_num.get("fps")),
        "question_chars": (task_structure.get("question_chars"), task.get("question_chars"), feature_num.get("question_chars")),
        "question_tokens": (task_structure.get("question_tokens"), task.get("question_tokens"), feature_num.get("question_tokens")),
        "option_count": (task_structure.get("option_count"), task.get("option_count"), feature_num.get("option_count")),
        "option_chars_mean": (task_structure.get("option_chars_mean"), task.get("option_chars_mean"), feature_num.get("option_chars_mean")),
    }
    if key in categorical_aliases:
        for value in categorical_aliases[key]:
            if isinstance(value, (list, tuple)):
                return "|".join(sorted(text(item) for item in value))
            if value not in (None, "", "unknown"):
                return text(value)
        return "unknown"
    if key in numeric_aliases:
        for value in numeric_aliases[key]:
            parsed = number(value)
            if parsed is not None:
                return parsed
        return -1.0
    raise KeyError(key)


def read_metadata(path: Path | None) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Index safe static metadata by run and video without retaining answers."""
    by_run: dict[str, dict[str, Any]] = {}
    by_video: dict[str, dict[str, Any]] = {}
    if path is None:
        return by_run, by_video
    for raw in read_jsonl(path):
        run_id = text(raw.get("run_id"), "")
        video_id = text(raw.get("video_id"), "")
        safe = {key: metadata_value(raw, key) for key in (*CATEGORICAL[-6:], *NUMERICAL[-6:])}
        if run_id and run_id not in by_run:
            by_run[run_id] = safe
        if video_id and video_id not in by_video:
            by_video[video_id] = safe
    return by_run, by_video


def build_rows(
    compute_path: Path,
    split_path: Path,
    source_name: str,
    metadata_path: Path | None = None,
) -> list[dict[str, Any]]:
    split_map = read_split_map(split_path)
    raw = read_jsonl(compute_path)
    metadata_by_run, metadata_by_video = read_metadata(metadata_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, original in enumerate(raw):
        row = dict(original)
        row["_source_index"] = index
        grouped[text(row.get("run_id"))].append(row)

    output: list[dict[str, Any]] = []
    for run_id, events in grouped.items():
        events.sort(key=lambda item: (int(item.get("event_index") or 0), int(item["_source_index"])))
        seen_models: set[str] = set()
        previous_runtime = 0.0
        previous_activity = "START"
        previous_model = "START"
        previous_family = "START"
        previous_error = 0
        accumulated_runtime = 0.0
        retry_count = 0
        for event in events:
            video_id = text(event.get("video_id"))
            split = split_map.get(video_id, "unassigned")
            scale = scale_dict(event.get("input_scale"))
            model_id = text(event.get("model_id"))
            activity = text(event.get("activity"))
            node_type = text(event.get("node_type"))
            current_family = event_family(event.get("raw_action"), activity)
            metadata = metadata_by_run.get(run_id) or metadata_by_video.get(video_id) or {}
            frame_count = number(scale.get("frame_count"))
            image_count = number(scale.get("qwen_image_count"))
            yolo_batch = number(scale.get("yolo_batch"))
            runtime = number(event.get("runtime_ms"))
            load = number(event.get("load_ms"))
            peak_allocated = number(event.get("peak_allocated_mb"))
            peak_reserved = number(event.get("peak_reserved_mb"))
            status = text(event.get("status"))
            error_text = text(event.get("error"), "")
            is_error = int(status != "success")
            is_retry = int(bool(event.get("retry_of")))
            retry_count += is_retry
            row = {
                "source_name": source_name,
                "run_id": run_id,
                "video_id": video_id,
                "split": split,
                "event_id": text(event.get("derived_event_id"), f"{run_id}:{event.get('event_index', 0)}"),
                "event_index": int(event.get("event_index") or 0),
                "activity": activity,
                "activity_family": current_family,
                "node_type": node_type,
                "model_id": model_id,
                "baseline": text(event.get("baseline")),
                "gpu_model": text(event.get("gpu_model")),
                "cold_warm": "cold" if model_id not in seen_models else "warm",
                "prev_activity": previous_activity,
                "prev_model_id": previous_model,
                "prev_activity_model": f"{previous_activity}|{previous_model}",
                "prev_activity_family_model": f"{previous_family}|{previous_model}",
                "frame_count": frame_count if frame_count is not None else -1.0,
                "qwen_image_count": image_count if image_count is not None else -1.0,
                "yolo_batch": yolo_batch if yolo_batch is not None else -1.0,
                "scale_missing": int(frame_count is None and image_count is None and yolo_batch is None),
                "acc_steps": int(event.get("event_index") or 0),
                "log_acc_runtime": math.log1p(max(accumulated_runtime, 0.0)),
                "log_prev_runtime": math.log1p(max(previous_runtime, 0.0)),
                "retry_count": retry_count,
                "prev_error": previous_error,
                "raw_action": text(event.get("raw_action"), ""),
                "domain": metadata.get("domain", "unknown"),
                "question_type": metadata.get("question_type", "unknown"),
                "sub_category": metadata.get("sub_category", "unknown"),
                "required_modalities": metadata.get("required_modalities", "unknown"),
                "answer_type": metadata.get("answer_type", "unknown"),
                "temporal_scope": metadata.get("temporal_scope", "unknown"),
                "video_duration_s": metadata.get("video_duration_s", -1.0),
                "video_fps": metadata.get("video_fps", -1.0),
                "question_chars": metadata.get("question_chars", -1.0),
                "question_tokens": metadata.get("question_tokens", -1.0),
                "option_count": metadata.get("option_count", -1.0),
                "option_chars_mean": metadata.get("option_chars_mean", -1.0),
                "runtime_ms": runtime,
                "load_ms": load,
                "peak_memory_inclusive_mb": (
                    max(value for value in (peak_allocated, peak_reserved) if value is not None)
                    if peak_allocated is not None or peak_reserved is not None
                    else None
                ),
                "status": status,
                "is_error": is_error,
                "is_retry": is_retry,
                "error_text": error_text,
                "queue_ms_observed": number(event.get("queue_ms")),
                "model_resident_before_observed": event.get("model_resident_before"),
                "value_source": text(event.get("value_source")),
            }
            output.append(row)
            seen_models.add(model_id)
            accumulated_runtime += runtime or 0.0
            previous_runtime = runtime or 0.0
            previous_activity = activity
            previous_model = model_id
            previous_family = current_family
            previous_error = is_error
    return output


def eligible(rows: Iterable[Mapping[str, Any]], target: str) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if row.get("status") == "success"
        and isinstance(row.get(target), (int, float))
        and float(row[target]) > 0
    ]


class FeatureEncoder:
    def __init__(self) -> None:
        try:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        self.categories: list[list[str]] = []

    def fit(self, rows: list[Mapping[str, Any]]) -> "FeatureEncoder":
        values = [[text(row.get(column)) for column in CATEGORICAL] for row in rows]
        self.encoder.fit(np.asarray(values, dtype=object))
        self.categories = [[str(value) for value in values] for values in self.encoder.categories_]
        return self

    def transform(self, rows: list[Mapping[str, Any]]) -> np.ndarray:
        values = [[text(row.get(column)) for column in CATEGORICAL] for row in rows]
        categorical = self.encoder.transform(np.asarray(values, dtype=object)).astype(np.float32)
        numerical = np.asarray(
            [[float(row.get(column) or 0.0) for column in NUMERICAL] for row in rows],
            dtype=np.float32,
        )
        return np.concatenate([categorical, numerical], axis=1)

    def schema(self) -> dict[str, Any]:
        return {
            "categorical": CATEGORICAL,
            "numerical": NUMERICAL,
            "categories": self.categories,
        }


class HierarchicalQuantile:
    levels = (
        ("structured", ("model_id", "activity", "node_type", "cold_warm", "frame_count", "qwen_image_count", "yolo_batch")),
        ("model_activity", ("model_id", "activity")),
        ("activity", ("activity",)),
        ("global", tuple()),
    )

    def __init__(self, minimum_count: int = 5) -> None:
        self.minimum_count = minimum_count
        self.values: dict[str, dict[tuple[Any, ...], list[float]]] = {
            name: defaultdict(list) for name, _ in self.levels
        }

    def fit(self, rows: list[Mapping[str, Any]], target: str) -> "HierarchicalQuantile":
        for row in rows:
            value = float(row[target])
            for name, columns in self.levels:
                self.values[name][tuple(row.get(column) for column in columns)].append(value)
        return self

    def predict(self, row: Mapping[str, Any], quantile: float) -> float:
        for name, columns in self.levels:
            values = self.values[name].get(tuple(row.get(column) for column in columns), [])
            if values and (name == "global" or len(values) >= self.minimum_count):
                return float(np.quantile(values, quantile))
        return 0.0


FeatureEncoder.__module__ = "tracing.analysis.resource_predictor_v1"
HierarchicalQuantile.__module__ = "tracing.analysis.resource_predictor_v1"


def quantile_loss(y_true: list[float], y_pred: list[float], quantile: float) -> float:
    if not y_true:
        return float("nan")
    return float(
        np.mean(
            [
                quantile * max(true - pred, 0.0) + (1.0 - quantile) * max(pred - true, 0.0)
                for true, pred in zip(y_true, y_pred)
            ]
        )
    )


def metric_bundle(y_true: list[float], p50: list[float], p90: list[float], p95: list[float]) -> dict[str, Any]:
    if not y_true:
        return {"n": 0}
    errors = [pred - true for pred, true in zip(p50, y_true)]
    abs_errors = [abs(error) for error in errors]
    mean_true = float(np.mean(y_true))
    return {
        "n": len(y_true),
        "mae_p50": float(np.mean(abs_errors)),
        "rmse_p50": float(math.sqrt(np.mean([error * error for error in errors]))),
        "p95_absolute_error_p50": float(np.quantile(abs_errors, 0.95)),
        "relative_mae_p50": float(np.mean(abs_errors) / mean_true) if mean_true else None,
        "pinball_q90": quantile_loss(y_true, p90, 0.90),
        "pinball_q95": quantile_loss(y_true, p95, 0.95),
        "coverage_q90": float(np.mean([true <= pred for true, pred in zip(y_true, p90)])),
        "coverage_q95": float(np.mean([true <= pred for true, pred in zip(y_true, p95)])),
        "underprediction_rate_q90": float(np.mean([true > pred for true, pred in zip(y_true, p90)])),
        "underprediction_rate_q95": float(np.mean([true > pred for true, pred in zip(y_true, p95)])),
        "bucket_accuracy_p50": float(
            np.mean(
                [
                    (0 if true < 1000 else 1 if true < 10000 else 2)
                    == (0 if pred < 1000 else 1 if pred < 10000 else 2)
                    for true, pred in zip(y_true, p50)
                ]
            )
        ),
    }


def fit_bundle(rows: list[dict[str, Any]], target: str, seed: int) -> dict[str, Any]:
    train_rows = eligible(rows, target)
    if len(train_rows) < 30:
        raise ValueError(f"not enough positive successful rows for {target}: {len(train_rows)}")
    group = HierarchicalQuantile().fit(train_rows, target)
    encoder = FeatureEncoder().fit(train_rows)
    x_train = encoder.transform(train_rows)
    y_train = np.log1p(np.asarray([float(row[target]) for row in train_rows], dtype=np.float32))

    import lightgbm as lgb
    import xgboost as xgb

    lgb_models: dict[str, Any] = {}
    for name, alpha in (("q50", 0.50), ("q90", 0.90), ("q95", 0.95)):
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=400,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            verbosity=-1,
        )
        model.fit(x_train, y_train)
        lgb_models[name] = model
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=400,
        learning_rate=0.04,
        max_depth=6,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=4,
        tree_method="hist",
    )
    xgb_model.fit(x_train, y_train)
    return {
        "target": target,
        "group": group,
        "encoder": encoder,
        "lgb_models": lgb_models,
        "xgb_model": xgb_model,
        "train_rows": len(train_rows),
    }


def predict_bundle(bundle: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    group: HierarchicalQuantile = bundle["group"]
    encoder: FeatureEncoder = bundle["encoder"]
    x = encoder.transform(rows)
    lgb_models = bundle["lgb_models"]
    lgb_predictions = {
        name: np.maximum(0.0, np.expm1(model.predict(x))).astype(float).tolist()
        for name, model in lgb_models.items()
    }
    xgb_prediction = np.maximum(0.0, np.expm1(bundle["xgb_model"].predict(x))).astype(float).tolist()
    group_predictions = {
        name: [group.predict(row, quantile) for row in rows]
        for name, quantile in (("q50", 0.50), ("q90", 0.90), ("q95", 0.95), ("q99", 0.99))
    }
    return {
        "lgb_q50": lgb_predictions["q50"],
        "lgb_q90": lgb_predictions["q90"],
        "lgb_q95": lgb_predictions["q95"],
        "xgb_p50": xgb_prediction,
        "group_q50": group_predictions["q50"],
        "group_q90": group_predictions["q90"],
        "group_q95": group_predictions["q95"],
        "group_q99": group_predictions["q99"],
    }


def evaluate_candidates(
    rows: list[dict[str, Any]],
    target: str,
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    valid = eligible(rows, target)
    predictions = predict_bundle(bundle, valid)
    y_true = [float(row[target]) for row in valid]
    candidates = {
        "lgb_quantile": (predictions["lgb_q50"], predictions["lgb_q90"], predictions["lgb_q95"]),
        "xgb_point_lgb_interval": (predictions["xgb_p50"], predictions["lgb_q90"], predictions["lgb_q95"]),
        "lgb_point_group_interval": (predictions["lgb_q50"], predictions["group_q90"], predictions["group_q95"]),
        "lgb_point_group_q99_interval": (predictions["lgb_q50"], predictions["group_q90"], predictions["group_q99"]),
        "group_quantile": (predictions["group_q50"], predictions["group_q90"], predictions["group_q95"]),
    }
    reports: dict[str, Any] = {}
    for name, (p50, p90, p95) in candidates.items():
        reports[name] = metric_bundle(y_true, p50, p90, p95)
    best = min(reports, key=lambda name: reports[name].get("mae_p50", float("inf")))
    return {"n": len(valid), "candidates": reports, "selected_by_validation_mae": best}


def select_scheduler_candidate(report: Mapping[str, Any]) -> str:
    candidates = report.get("candidates", {})
    safe = [
        name
        for name, metrics in candidates.items()
        if float(metrics.get("coverage_q95", 0.0)) >= SCHEDULER_Q95_COVERAGE_FLOOR
    ]
    if safe:
        return min(safe, key=lambda name: float(candidates[name].get("mae_p50", float("inf"))))
    return max(
        candidates,
        key=lambda name: (
            float(candidates[name].get("coverage_q95", 0.0)),
            -float(candidates[name].get("mae_p50", float("inf"))),
        ),
    )


def runtime_stall_threshold(train_rows: list[dict[str, Any]]) -> float:
    values = [float(row["runtime_ms"]) for row in eligible(train_rows, "runtime_ms")]
    if not values:
        return 300_000.0
    return max(300_000.0, 5.0 * float(np.quantile(values, 0.99)))


def target_coverage(rows: list[dict[str, Any]], target: str) -> dict[str, int]:
    return {
        "all_rows": len(rows),
        "positive_measured": sum(
            isinstance(row.get(target), (int, float)) and float(row[target]) > 0 for row in rows
        ),
        "successful_positive": len(eligible(rows, target)),
        "error_rows": sum(row.get("status") != "success" for row in rows),
    }


def audit_rows(dev_rows: list[dict[str, Any]], holdout_rows: list[dict[str, Any]]) -> dict[str, Any]:
    dev_videos = {row["video_id"] for row in dev_rows}
    holdout_videos = {row["video_id"] for row in holdout_rows}
    all_rows = dev_rows + holdout_rows
    strict_oom = sum(
        any(token in text(row.get("error_text"), "").lower() for token in ("out of memory", "cuda allocation"))
        for row in all_rows
    )
    return {
        "development": {
            "rows": len(dev_rows),
            "videos": len(dev_videos),
            "split_counts": dict(Counter(row["split"] for row in dev_rows)),
            "train_rows": sum(row["split"] == "train" for row in dev_rows),
            "validation_rows": sum(row["split"] == "validation" for row in dev_rows),
            "excluded_old_test_rows": sum(row["split"] == "test" for row in dev_rows),
            "target_coverage": {name: target_coverage(dev_rows, target) for name, target in TARGETS.items()},
        },
        "final_holdout": {
            "rows": len(holdout_rows),
            "videos": len(holdout_videos),
            "split_counts": dict(Counter(row["split"] for row in holdout_rows)),
            "target_coverage": {name: target_coverage(holdout_rows, target) for name, target in TARGETS.items()},
        },
        "overlap_videos": sorted(dev_videos & holdout_videos),
        "queue_nonnull": sum(row.get("queue_ms_observed") is not None for row in all_rows),
        "resident_before_nonnull": sum(row.get("model_resident_before_observed") is not None for row in all_rows),
        "strict_oom_rows": strict_oom,
        "leakage_contract": {
            "video_id_feature": False,
            "future_events_feature": False,
            "target_runtime_feature": False,
            "target_peak_feature": False,
            "target_load_feature": False,
            "previous_event_features_only": True,
            "fit_statistics_train_only": True,
            "static_task_metadata_only": True,
            "answer_or_teacher_metadata_feature": False,
        },
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-compute", type=Path, required=True)
    parser.add_argument("--dev-split", type=Path, required=True)
    parser.add_argument("--holdout-compute", type=Path, required=True)
    parser.add_argument("--holdout-split", type=Path, required=True)
    parser.add_argument("--dev-metadata", type=Path)
    parser.add_argument("--holdout-metadata", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dev_all = build_rows(args.dev_compute, args.dev_split, "development", args.dev_metadata)
    holdout_rows = build_rows(args.holdout_compute, args.holdout_split, "final_holdout", args.holdout_metadata)
    dev_train = [row for row in dev_all if row["split"] == "train"]
    dev_validation = [row for row in dev_all if row["split"] == "validation"]
    if not dev_train or not dev_validation:
        raise ValueError("development train/validation split is empty")

    audit = audit_rows(dev_all, holdout_rows)
    if audit["overlap_videos"]:
        raise ValueError(f"development/holdout video overlap: {audit['overlap_videos']}")

    validation_reports: dict[str, Any] = {}
    selected_models: dict[str, str] = {}
    for name, target in TARGETS.items():
        bundle = fit_bundle(dev_train, target, args.seed)
        report = evaluate_candidates(dev_validation, target, bundle)
        report["selected_for_scheduler"] = select_scheduler_candidate(report)
        validation_reports[name] = report
        selected_models[name] = report["selected_for_scheduler"]

    fit_rows = dev_train + dev_validation
    stall_threshold_ms = runtime_stall_threshold(dev_train)
    holdout_stalls = [
        row for row in holdout_rows
        if row.get("status") == "success"
        and isinstance(row.get("runtime_ms"), (int, float))
        and float(row["runtime_ms"]) > stall_threshold_ms
    ]
    holdout_normal_runtime = [row for row in holdout_rows if row not in holdout_stalls]
    final_reports: dict[str, Any] = {}
    final_predictions: list[dict[str, Any]] = [
        {
            "event_id": row["event_id"],
            "run_id": row["run_id"],
            "video_id": row["video_id"],
            "event_index": row["event_index"],
            "activity": row["activity"],
            "node_type": row["node_type"],
            "model_id": row["model_id"],
            "status": row["status"],
            "is_retry": row["is_retry"],
        }
        for row in holdout_rows
    ]
    bundles: dict[str, Any] = {}
    for name, target in TARGETS.items():
        bundle = fit_bundle(fit_rows, target, args.seed)
        bundles[name] = bundle
        final_report = evaluate_candidates(holdout_rows, target, bundle)
        final_report["selected_model_from_development"] = selected_models[name]
        if target == "runtime_ms":
            normal_report = evaluate_candidates(holdout_normal_runtime, target, bundle)
            final_report["normal_service_only"] = {
                "n": normal_report["n"],
                "candidates": normal_report["candidates"],
            }
        final_reports[name] = final_report
        predictions = predict_bundle(bundle, holdout_rows)
        selected = selected_models[name]
        if selected == "xgb_point_lgb_interval":
            p50_key = "xgb_p50"
        elif selected == "group_quantile":
            p50_key = "group_q50"
        else:
            p50_key = "lgb_q50"
        if selected == "lgb_point_group_q99_interval":
            p90_key, p95_key = "group_q90", "group_q99"
        elif selected in {"group_quantile", "lgb_point_group_interval"}:
            p90_key, p95_key = "group_q90", "group_q95"
        else:
            p90_key, p95_key = "lgb_q90", "lgb_q95"
        for index, row in enumerate(final_predictions):
            final_predictions[index][f"{name}_p50_ms"] = predictions[p50_key][index]
            final_predictions[index][f"{name}_p90_ms"] = predictions[p90_key][index]
            final_predictions[index][f"{name}_p95_ms"] = predictions[p95_key][index]
            if row.get("status") == "success" and isinstance(holdout_rows[index].get(target), (int, float)):
                final_predictions[index][f"observed_{name}_ms"] = holdout_rows[index][target]

    audit["input_hashes"] = {
        "dev_compute": sha256_file(args.dev_compute),
        "dev_split": sha256_file(args.dev_split),
        "holdout_compute": sha256_file(args.holdout_compute),
        "holdout_split": sha256_file(args.holdout_split),
    }
    if args.dev_metadata:
        audit["input_hashes"]["dev_metadata"] = sha256_file(args.dev_metadata)
    if args.holdout_metadata:
        audit["input_hashes"]["holdout_metadata"] = sha256_file(args.holdout_metadata)
    report = {
        "schema_version": "resource-predictor-v1.0",
        "seed": args.seed,
        "fit_splits": ["development_train", "development_validation"],
        "evaluation_splits": ["development_validation", "final_holdout_v1"],
        "primary_targets": list(TARGETS),
        "selection_policy": "select the lowest-MAE candidate among models with validation q95 coverage >= 0.93; otherwise select highest validation q95 coverage; fit final artifacts on development train+validation; evaluate final holdout once",
        "scheduler_q95_coverage_floor": SCHEDULER_Q95_COVERAGE_FLOOR,
        "metadata_contract": {
            "development": str(args.dev_metadata) if args.dev_metadata else None,
            "final_holdout": str(args.holdout_metadata) if args.holdout_metadata else None,
            "allowed": [
                "domain", "question_type", "sub_category", "required_modalities",
                "answer_type", "temporal_scope", "video_duration_s", "video_fps",
                "question_chars", "question_tokens", "option_count", "option_chars_mean",
            ],
            "excluded": ["answer", "teacher_labels", "remaining_runtime_ms", "remaining_steps", "video_id"],
        },
        "models": {
            "group_quantile": "hierarchical P50/P90/P95 with train-only backoff",
            "lightgbm_quantile": "log1p target, quantiles 0.50/0.90/0.95",
            "xgb_point": "log1p target point regression control",
            "lgb_point_group_interval": "LightGBM P50 with hierarchical train-only P90/P95 safety interval",
            "lgb_point_group_q99_interval": "LightGBM P50 with hierarchical train-only P90 and conservative q99 upper interval",
        },
        "audit": audit,
        "validation": validation_reports,
        "final_holdout": final_reports,
        "runtime_stall_diagnostic": {
            "threshold_ms": stall_threshold_ms,
            "threshold_source": "max(300000, 5 * development-train runtime p99)",
            "count": len(holdout_stalls),
            "fraction_of_successful_runtime_rows": len(holdout_stalls) / max(len(eligible(holdout_rows, "runtime_ms")), 1),
            "event_ids": [row["event_id"] for row in holdout_stalls],
            "video_ids": sorted({row["video_id"] for row in holdout_stalls}),
            "interpretation": "preserved as measured long-tail events; not silently clipped or removed from all-event metrics; no train-positive stall examples exist for a causal stall classifier",
        },
        "unsupported_yet": {
            "queue_prediction": audit["queue_nonnull"] == 0,
            "strict_oom_prediction": audit["strict_oom_rows"] == 0,
            "resident_weight_separation": audit["resident_before_nonnull"] == 0,
            "stall_risk_prediction": True,
        },
    }
    write_json(args.output_dir / "resource_predictor_report.json", report)
    write_json(args.output_dir / "feature_schema.json", {
        "schema_version": "resource-predictor-features-v1.1",
        "categorical": CATEGORICAL,
        "numerical": NUMERICAL,
        "target_definitions": {
            "runtime_ms": "measured positive runtime on successful compute events",
            "load_ms": "measured positive load time on successful compute events",
            "peak_memory_inclusive_mb": "max(peak_reserved_mb, peak_allocated_mb), inclusive resident/workspace label",
        },
    })
    with (args.output_dir / "holdout_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in final_predictions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (args.output_dir / "models.pkl").open("wb") as handle:
        pickle.dump(bundles, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(args.output_dir / "scheduler_resource_contract.json", {
        "schema_version": "scheduler-resource-contract-v1",
        "inputs": ["candidate activity/node/model", "observed prefix", "GPU capacity", "cold/warm", "input scale"],
        "outputs": ["runtime_p50_ms", "runtime_p90_ms", "load_p50_ms", "peak_memory_p95_mb", "uncertainty"],
        "runtime_semantics": "normal service runtime; measured long-tail stalls are reported separately",
        "queue_source": "scheduler/workload simulator, not this predictor",
        "oom_source": "unsupported until controlled OOM calibration exists",
        "stall_source": "unsupported until controlled stall/timeout calibration exists",
        "unknown_policy": "retain unknown and use explicit fallback; never replace missing resource labels with zero",
        "artifact": str(args.output_dir / "models.pkl"),
    })
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "development_train_rows": len(dev_train),
        "development_validation_rows": len(dev_validation),
        "final_holdout_rows": len(holdout_rows),
        "selected_models": selected_models,
        "queue_nonnull": audit["queue_nonnull"],
        "strict_oom_rows": audit["strict_oom_rows"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
