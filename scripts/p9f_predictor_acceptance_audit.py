#!/usr/bin/env python3
"""Audit the frozen P9e predictor before any scheduler integration.

This experiment does not retrain a model.  It loads the frozen P9e ``B``
checkpoints, generates logits on the fixed P9d splits, fits scalar
temperature calibrators on P_dev/train only, and reports calibration and
condition-stratified errors on validation, test, and the frozen holdout.

The resource part is deliberately conservative.  The reduced future-node
contract emits only ``role`` and ``action_family``; it does not emit
``model_id``, ``node_type`` or input scale.  Therefore this script evaluates
only a train-only role/family runtime profile as a coarse future-cost proxy.
It records model-aware load/memory as not identifiable rather than silently
inventing those fields.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import p9d_future_role_family_multitask as p9e  # noqa: E402
from scripts.p9d_shared_causal_gru import (  # noqa: E402
    FAMILY_LABELS,
    HORIZON,
    InputEncoder,
    MAX_WIDTH,
    ROLE_LABELS,
    _read_split,
    join_behavior_targets,
)


SCHEMA_VERSION = "p9f-predictor-acceptance-audit-v1"
SPLITS = ("train", "validation", "test", "holdout")
EVALUATION_SPLITS = ("validation", "test", "holdout")
DEFAULT_TEMPERATURE_GRID = tuple(round(0.25 + 0.05 * index, 2) for index in range(76))
STRATUM_FIELDS = (
    "current_role",
    "current_node_type",
    "current_action_family",
    "model_stack_id",
    "true_layer_count",
    "true_first_layer_width",
    "true_future_node_count_bin",
    "truncated_at_horizon",
)


def open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open_text(path, "r") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def stable_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


def classification_metrics(
    logits: np.ndarray, targets: np.ndarray, temperature: float = 1.0, bins: int = 10
) -> Dict[str, Any]:
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if logits.ndim != 2 or targets.ndim != 1 or len(logits) != len(targets):
        raise ValueError("categorical logits/target shapes do not align")
    if len(targets) == 0:
        return {
            "sample_count": 0,
            "accuracy": None,
            "nll": None,
            "brier": None,
            "ece": None,
        }
    probabilities = stable_softmax(logits, temperature)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == targets
    target_probability = probabilities[np.arange(len(targets)), targets]
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(targets)), targets] = 1.0
    confidence = probabilities.max(axis=1)
    ece = 0.0
    reliability: List[Dict[str, Any]] = []
    for index in range(bins):
        lower = index / float(bins)
        upper = (index + 1) / float(bins)
        mask = (confidence >= lower) & (
            confidence < upper if index + 1 < bins else confidence <= upper
        )
        count = int(mask.sum())
        if count:
            bucket_accuracy = float(correct[mask].mean())
            bucket_confidence = float(confidence[mask].mean())
            ece += count / len(targets) * abs(bucket_accuracy - bucket_confidence)
        else:
            bucket_accuracy = None
            bucket_confidence = None
        reliability.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "accuracy": bucket_accuracy,
                "confidence": bucket_confidence,
            }
        )
    return {
        "sample_count": int(len(targets)),
        "accuracy": float(correct.mean()),
        "nll": float(-np.log(np.clip(target_probability, 1e-12, 1.0)).mean()),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece": float(ece),
        "reliability": reliability,
    }


def fit_temperature(
    logits: np.ndarray, targets: np.ndarray, grid: Sequence[float]
) -> Dict[str, Any]:
    candidates = []
    for temperature in grid:
        value = classification_metrics(logits, targets, temperature)
        candidates.append((float(value["nll"]) if value["nll"] is not None else math.inf, float(temperature)))
    if not candidates or not math.isfinite(candidates[0][0]):
        return {"temperature": 1.0, "fit_nll": None, "grid": list(grid)}
    best_nll, best_temperature = min(candidates, key=lambda item: (item[0], item[1]))
    return {
        "temperature": best_temperature,
        "fit_nll": best_nll,
        "grid": [float(value) for value in grid],
    }


def tensors_to_numpy(outputs: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "role_logits": outputs["role_logits"].numpy().astype(np.float64),
        "family_logits": outputs["family_logits"].numpy().astype(np.float64),
        "layer_logits": outputs["layer_logits"].numpy().astype(np.float64),
        "width_logits": outputs["width_logits"].numpy().astype(np.float64),
        "content_logits": [value.numpy().astype(np.float64) for value in outputs["content_logits"]],
    }


def content_observations(
    outputs: Mapping[str, Any], bundle: Mapping[str, np.ndarray]
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Collect oracle-structure matched content logits for calibration.

    Matching uses both content fields jointly and is performed once per active
    layer.  This is a calibration diagnostic; it does not change the
    predicted-structure endpoint used by P9e.
    """

    logits_by_field = [value.numpy().astype(np.float64) for value in outputs["content_logits"]]
    targets = bundle["content_targets"]
    widths = bundle["width_targets"]
    collected: Dict[str, List[np.ndarray]] = {field: [] for field in p9e.CONTENT_FIELDS}
    target_values: Dict[str, List[int]] = {field: [] for field in p9e.CONTENT_FIELDS}
    for row_index in range(len(widths)):
        for layer_index in range(HORIZON):
            width = int(widths[row_index, layer_index])
            if width <= 0:
                continue
            cost = np.zeros((width, width), dtype=np.float64)
            for field_index in range(len(p9e.CONTENT_FIELDS)):
                probabilities = stable_softmax(logits_by_field[field_index][row_index, layer_index, :width])
                target = targets[row_index, layer_index, :width, field_index]
                cost += -np.log(np.clip(probabilities[:, target].T, 1e-12, 1.0))
            assignment = p9e._best_assignment(cost)
            for true_index, predicted_index in assignment:
                for field_index, field in enumerate(p9e.CONTENT_FIELDS):
                    collected[field].append(
                        logits_by_field[field_index][row_index, layer_index, predicted_index]
                    )
                    target_values[field].append(int(targets[row_index, layer_index, true_index, field_index]))
    return {
        field: (
            np.stack(collected[field], axis=0)
            if collected[field]
            else np.zeros((0, logits_by_field[index].shape[-1]), dtype=np.float64),
            np.asarray(target_values[field], dtype=np.int64),
        )
        for index, field in enumerate(p9e.CONTENT_FIELDS)
    }


def fit_calibrators(
    outputs: Mapping[str, Any], bundle: Mapping[str, np.ndarray], grid: Sequence[float]
) -> Dict[str, Any]:
    arrays = tensors_to_numpy(outputs)
    active_width = bundle["width_targets"] > 0
    width_logits = arrays["width_logits"][active_width]
    width_targets = bundle["width_targets"][active_width] - 1
    family_mask = bundle["behavior_family_mask"].astype(bool)
    calibrators: Dict[str, Any] = {
        "layer_count": fit_temperature(arrays["layer_logits"], bundle["layer_targets"], grid),
        "width": fit_temperature(width_logits, width_targets, grid),
        "next_role": fit_temperature(arrays["role_logits"], bundle["behavior_role_targets"], grid),
        "next_family": fit_temperature(
            arrays["family_logits"][family_mask], bundle["behavior_family_targets"][family_mask], grid
        ),
        "future_content": {},
    }
    observations = content_observations(outputs, bundle)
    for field, (field_logits, field_targets) in observations.items():
        calibrators["future_content"][field] = fit_temperature(field_logits, field_targets, grid)
    return calibrators


def calibration_report(
    outputs: Mapping[str, Any], bundle: Mapping[str, np.ndarray], calibrators: Mapping[str, Any]
) -> Dict[str, Any]:
    arrays = tensors_to_numpy(outputs)
    active_width = bundle["width_targets"] > 0
    family_mask = bundle["behavior_family_mask"].astype(bool)
    entries: Dict[str, Any] = {
        "layer_count": {
            "raw": classification_metrics(arrays["layer_logits"], bundle["layer_targets"]),
            "calibrated": classification_metrics(
                arrays["layer_logits"],
                bundle["layer_targets"],
                float(calibrators["layer_count"]["temperature"]),
            ),
        },
        "width": {
            "raw": classification_metrics(
                arrays["width_logits"][active_width], bundle["width_targets"][active_width] - 1
            ),
            "calibrated": classification_metrics(
                arrays["width_logits"][active_width],
                bundle["width_targets"][active_width] - 1,
                float(calibrators["width"]["temperature"]),
            ),
        },
        "next_role": {
            "raw": classification_metrics(arrays["role_logits"], bundle["behavior_role_targets"]),
            "calibrated": classification_metrics(
                arrays["role_logits"],
                bundle["behavior_role_targets"],
                float(calibrators["next_role"]["temperature"]),
            ),
        },
        "next_family": {
            "raw": classification_metrics(
                arrays["family_logits"][family_mask], bundle["behavior_family_targets"][family_mask]
            ),
            "calibrated": classification_metrics(
                arrays["family_logits"][family_mask],
                bundle["behavior_family_targets"][family_mask],
                float(calibrators["next_family"]["temperature"]),
            ),
        },
        "future_content_oracle_structure": {},
    }
    observations = content_observations(outputs, bundle)
    for field, (field_logits, field_targets) in observations.items():
        temperature = float(calibrators["future_content"][field]["temperature"])
        entries["future_content_oracle_structure"][field] = {
            "raw": classification_metrics(field_logits, field_targets),
            "calibrated": classification_metrics(field_logits, field_targets, temperature),
        }
    return {"temperatures": calibrators, "heads": entries}


def future_node_count_bin(value: int) -> str:
    if value == 0:
        return "0"
    if value <= 2:
        return "1-2"
    if value <= 5:
        return "3-5"
    return "6+"


def stratum_values(feature: Mapping[str, Any], label: Mapping[str, Any]) -> Dict[str, str]:
    model_input = feature.get("model_input") or {}
    current = model_input.get("current_node") or {}
    stack = model_input.get("stack_context") or {}
    layers = label.get("future_layers") or []
    widths = [len(layer.get("nodes") or []) for layer in layers]
    node_count = sum(widths)
    return {
        "current_role": str(current.get("role") or "unknown"),
        "current_node_type": str(current.get("node_type") or "unknown"),
        "current_action_family": str(current.get("action_family") or "unknown"),
        "model_stack_id": str(stack.get("model_stack_id") or "unknown"),
        "true_layer_count": str(len(layers)),
        "true_first_layer_width": str(widths[0] if widths else 0),
        "true_future_node_count_bin": future_node_count_bin(node_count),
        "truncated_at_horizon": str(bool((label.get("label_summary") or {}).get("truncated_at_horizon"))),
    }


def macro_f1(targets: Sequence[int], predictions: Sequence[int]) -> Optional[float]:
    labels = sorted(set(int(value) for value in targets) | set(int(value) for value in predictions))
    if not labels:
        return None
    scores = []
    for label in labels:
        tp = sum(int(true == label and predicted == label) for true, predicted in zip(targets, predictions))
        fp = sum(int(true != label and predicted == label) for true, predicted in zip(targets, predictions))
        fn = sum(int(true == label and predicted != label) for true, predicted in zip(targets, predictions))
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(scores))


def structure_and_behavior_metrics(
    arrays: Mapping[str, np.ndarray], bundle: Mapping[str, np.ndarray], indices: np.ndarray
) -> Dict[str, Any]:
    layer_prediction = arrays["layer_logits"].argmax(axis=-1)
    width_prediction = arrays["width_logits"].argmax(axis=-1) + 1
    true_layers = bundle["layer_targets"]
    true_widths = bundle["width_targets"]
    selected_widths = np.asarray(
        [
            [
                int(width_prediction[row, layer]) if layer < int(layer_prediction[row]) else 0
                for layer in range(HORIZON)
            ]
            for row in range(len(layer_prediction))
        ],
        dtype=np.int64,
    )
    subset = np.asarray(indices, dtype=np.int64)
    true_nodes = true_widths.sum(axis=1)
    predicted_nodes = selected_widths.sum(axis=1)
    role_prediction = arrays["role_logits"].argmax(axis=-1)
    role_true = bundle["behavior_role_targets"]
    family_prediction = arrays["family_logits"].argmax(axis=-1)
    family_true = bundle["behavior_family_targets"]
    family_mask = bundle["behavior_family_mask"].astype(bool)
    role_values = role_true[subset]
    role_predicted = role_prediction[subset]
    family_subset = subset[family_mask[subset]]
    family_values = family_true[family_subset]
    family_predicted = family_prediction[family_subset]
    return {
        "sample_count": int(len(subset)),
        "layer_count_mae": float(np.abs(layer_prediction[subset] - true_layers[subset]).mean()),
        "layer_count_bias": float((layer_prediction[subset] - true_layers[subset]).mean()),
        "node_count_mae": float(np.abs(predicted_nodes[subset] - true_nodes[subset]).mean()),
        "node_count_bias": float((predicted_nodes[subset] - true_nodes[subset]).mean()),
        "width_vector_mae": float(np.abs(selected_widths[subset] - true_widths[subset]).mean()),
        "structure_exact_coverage": float(
            np.mean(
                (layer_prediction[subset] == true_layers[subset])
                & np.all(selected_widths[subset] == true_widths[subset], axis=1)
            )
        ),
        "future_exists_accuracy": float(
            np.mean((layer_prediction[subset] > 0) == (true_layers[subset] > 0))
        ),
        "behavior_role_accuracy": float((role_prediction[subset] == role_true[subset]).mean()),
        "behavior_role_macro_f1": macro_f1(role_values.tolist(), role_predicted.tolist()),
        "behavior_family_accuracy": float((family_predicted == family_values).mean())
        if len(family_values)
        else None,
        "behavior_family_macro_f1": macro_f1(family_values.tolist(), family_predicted.tolist()),
        "behavior_family_coverage": float(family_mask[subset].mean()),
    }


def stratified_report(
    pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]],
    arrays: Mapping[str, np.ndarray],
    bundle: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for field in STRATUM_FIELDS:
        groups: Dict[str, List[int]] = defaultdict(list)
        for index, (feature, label) in enumerate(pairs):
            groups[stratum_values(feature, label)[field]].append(index)
        result[field] = {
            value: structure_and_behavior_metrics(arrays, bundle, np.asarray(indices, dtype=np.int64))
            for value, indices in sorted(groups.items())
        }
    return result


def load_resource_map(path: Path) -> Tuple[Dict[Tuple[str, int], Dict[str, Any]], Dict[str, Any]]:
    rows = read_jsonl(path)
    result: Dict[Tuple[str, int], Dict[str, Any]] = {}
    duplicate_conflicts = 0
    for row in rows:
        run_id = str(row.get("run_id") or "")
        event_index = row.get("event_index")
        try:
            event_index = int(event_index)
        except (TypeError, ValueError):
            continue
        key = (run_id, event_index)
        compact = {
            "runtime_ms": finite(row.get("runtime_ms")),
            "load_ms": finite(row.get("load_ms")),
            "peak_allocated_mb": finite(row.get("peak_allocated_mb")),
            "status": str(row.get("status") or "unknown"),
        }
        previous = result.get(key)
        if previous is not None and previous != compact:
            duplicate_conflicts += 1
        result[key] = compact
    return result, {
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        "source_rows": len(rows),
        "unique_run_event_keys": len(result),
        "duplicate_conflicts": duplicate_conflicts,
    }


def train_resource_profiles(
    train_pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]],
    resource_map: Mapping[Tuple[str, int], Mapping[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    values: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    missing = 0
    excluded_non_success = 0
    for _feature, label in train_pairs:
        run_id = str(label.get("run_id") or "")
        for layer in label.get("future_layers") or []:
            for node in layer.get("nodes") or []:
                try:
                    sequence_index = int(node.get("sequence_index"))
                except (TypeError, ValueError):
                    missing += 1
                    continue
                row = resource_map.get((run_id, sequence_index))
                if row is None or row.get("runtime_ms") is None:
                    missing += 1
                    continue
                if row.get("status") != "success":
                    excluded_non_success += 1
                    continue
                values[(str(node.get("role") or "unknown"), str(node.get("action_family") or "other"))].append(
                    float(row["runtime_ms"])
                )

    def summarize(items: Sequence[float]) -> Dict[str, Any]:
        ordered = np.asarray(items, dtype=np.float64)
        return {
            "count": int(len(ordered)),
            "p50_ms": float(np.quantile(ordered, 0.50)),
            "p90_ms": float(np.quantile(ordered, 0.90)),
            "mean_ms": float(ordered.mean()),
        }

    profiles = {f"{role}|{family}": summarize(items) for (role, family), items in sorted(values.items()) if items}
    return profiles, {
        "target": "runtime_ms",
        "fit_source": "P_dev/train future labels joined to P_dev compute rows",
        "groups": len(profiles),
        "matched_successful_future_nodes": int(sum(len(items) for items in values.values())),
        "missing_future_resource_rows": missing,
        "excluded_non_success_rows": excluded_non_success,
        "fallback_order": ["role|action_family", "role|*", "*|action_family", "*|*"],
    }


def profile_lookup(
    profiles: Mapping[str, Mapping[str, Any]], role: str, family: str
) -> Tuple[Optional[float], str]:
    keys = [
        (f"{role}|{family}", "role_family"),
        (f"{role}|*", "role"),
        (f"*|{family}", "family"),
        ("*|*", "global"),
    ]
    for key, source in keys:
        value = profiles.get(key)
        if value is not None:
            return finite(value.get("p50_ms")), source
    return None, "missing"


def true_future_runtime(
    label: Mapping[str, Any], resource_map: Mapping[Tuple[str, int], Mapping[str, Any]]
) -> Tuple[Optional[float], int]:
    total = 0.0
    missing = 0
    run_id = str(label.get("run_id") or "")
    for layer in label.get("future_layers") or []:
        for node in layer.get("nodes") or []:
            try:
                sequence_index = int(node.get("sequence_index"))
            except (TypeError, ValueError):
                missing += 1
                continue
            row = resource_map.get((run_id, sequence_index))
            runtime = row.get("runtime_ms") if row else None
            if runtime is None or (row and row.get("status") != "success"):
                missing += 1
                continue
            total += float(runtime)
    return (total if missing == 0 else None), missing


def predicted_scenario_costs(
    outputs: Mapping[str, Any], row_index: int, codec: p9e.ContentCodec, profiles: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    decoded = p9e._shape_scenarios(outputs, row_index)
    total_probability = sum(float(item[0]) for item in decoded) or 1.0
    content_predictions = [
        value[row_index].argmax(dim=-1).numpy() for value in outputs["content_logits"]
    ]
    scenarios = []
    for rank, (probability, layer_count, widths) in enumerate(decoded):
        cost = 0.0
        missing_profiles = 0
        fallback_counts: Counter[str] = Counter()
        predicted_node_count = 0
        for layer_index in range(int(layer_count)):
            width = int(widths[layer_index])
            for node_index in range(width):
                role = codec.decode("role", int(content_predictions[0][layer_index, node_index]))
                family = codec.decode("action_family", int(content_predictions[1][layer_index, node_index]))
                value, source = profile_lookup(profiles, role, family)
                if value is None:
                    missing_profiles += 1
                else:
                    cost += value
                fallback_counts[source] += 1
                predicted_node_count += 1
        scenarios.append(
            {
                "rank": rank,
                "probability": float(probability) / total_probability,
                "future_runtime_p50_ms": float(cost) if missing_profiles == 0 else None,
                "predicted_node_count": predicted_node_count,
                "missing_profile_nodes": missing_profiles,
                "profile_sources": dict(sorted(fallback_counts.items())),
            }
        )
    if not scenarios:
        scenarios = [
            {
                "rank": 0,
                "probability": 1.0,
                "future_runtime_p50_ms": 0.0,
                "predicted_node_count": 0,
                "missing_profile_nodes": 0,
                "profile_sources": {},
            }
        ]
    expected = [item for item in scenarios if item["future_runtime_p50_ms"] is not None]
    expected_cost = (
        sum(float(item["probability"]) * float(item["future_runtime_p50_ms"]) for item in expected)
        if len(expected) == len(scenarios)
        else None
    )
    return {
        "top1": scenarios[0],
        "expected": expected_cost,
        "scenarios": scenarios,
    }


def cost_error_metrics(values: Sequence[Tuple[float, float]], unit: str = "ms") -> Dict[str, Any]:
    suffix = f"_{unit}"
    if not values:
        return {
            "sample_count": 0,
            f"mae{suffix}": None,
            f"bias{suffix}": None,
            "relative_mae": None,
            "underprediction_rate": None,
        }
    truth = np.asarray([item[0] for item in values], dtype=np.float64)
    predicted = np.asarray([item[1] for item in values], dtype=np.float64)
    error = predicted - truth
    return {
        "sample_count": int(len(values)),
        f"mae{suffix}": float(np.abs(error).mean()),
        f"bias{suffix}": float(error.mean()),
        "relative_mae": float((np.abs(error) / np.maximum(np.abs(truth), 1.0)).mean()),
        "underprediction_rate": float((predicted < truth).mean()),
        f"p95_absolute_error{suffix}": float(np.quantile(np.abs(error), 0.95)),
    }


def evaluate_cost_split(
    pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]],
    outputs: Mapping[str, Any],
    codec: p9e.ContentCodec,
    profiles: Mapping[str, Mapping[str, Any]],
    resource_map: Mapping[Tuple[str, int], Mapping[str, Any]],
) -> Dict[str, Any]:
    top1_values: List[Tuple[float, float]] = []
    expected_values: List[Tuple[float, float]] = []
    node_count_values: List[Tuple[float, float]] = []
    missing_truth_nodes = 0
    missing_profile_nodes = 0
    scenario_rows = 0
    for row_index, (_feature, label) in enumerate(pairs):
        prediction = predicted_scenario_costs(outputs, row_index, codec, profiles)
        scenario_rows += len(prediction["scenarios"])
        missing_profile_nodes += sum(item["missing_profile_nodes"] for item in prediction["scenarios"])
        truth, missing = true_future_runtime(label, resource_map)
        missing_truth_nodes += missing
        if truth is None:
            continue
        top1 = prediction["top1"]
        if top1["future_runtime_p50_ms"] is not None:
            top1_values.append((truth, float(top1["future_runtime_p50_ms"])))
        if prediction["expected"] is not None:
            expected_values.append((truth, float(prediction["expected"])))
        node_count_values.append(
            (
                float(sum(len(layer.get("nodes") or []) for layer in label.get("future_layers") or [])),
                float(top1["predicted_node_count"]),
            )
        )
    return {
        "true_runtime_coverage": float(len(top1_values) / len(pairs)) if pairs else 0.0,
        "missing_truth_nodes": int(missing_truth_nodes),
        "missing_profile_nodes_across_top_scenarios": int(missing_profile_nodes),
        "scenario_rows": scenario_rows,
        "top1_runtime_p50": cost_error_metrics(top1_values),
        "expected_runtime_p50": cost_error_metrics(expected_values),
        "top1_node_count": cost_error_metrics(node_count_values, unit="count"),
        "not_evaluated": {
            "load_ms": "future node contract lacks model_id/node_type/input_scale/cold_warm",
            "peak_memory": "future node contract lacks model_id/node_type/input_scale/cold_warm",
        },
    }


def load_model(config: Mapping[str, Any], checkpoint: Path, encoder: InputEncoder, codec: p9e.ContentCodec, device: Any) -> Any:
    if p9e.torch is None or p9e.FutureRoleFamilyGRU is None:
        raise RuntimeError(f"PyTorch is required for P9f: {p9e.TORCH_IMPORT_ERROR}")
    state = p9e.torch.load(checkpoint, map_location=device, weights_only=False)
    if state.get("variant") != "B":
        raise ValueError(f"P9f only audits frozen P9e B checkpoints, got {state.get('variant')!r}")
    if tuple(state.get("content_fields") or ()) != p9e.CONTENT_FIELDS:
        raise ValueError("checkpoint content fields do not match P9e reduced contract")
    history_vocab_sizes, context_vocab_sizes = encoder.vocab_sizes()
    if list(state.get("history_vocab_sizes") or []) != list(history_vocab_sizes):
        raise ValueError("checkpoint history vocabulary does not match train-only encoder")
    if list(state.get("context_vocab_sizes") or []) != list(context_vocab_sizes):
        raise ValueError("checkpoint context vocabulary does not match train-only encoder")
    if list(state.get("content_field_sizes") or []) != list(codec.field_sizes()):
        raise ValueError("checkpoint content vocabulary does not match train-only codec")
    model = p9e.FutureRoleFamilyGRU(
        history_vocab_sizes,
        context_vocab_sizes,
        codec.field_sizes(),
        hidden=int(config["hidden"]),
        history_embedding_dim=int(config["history_embedding_dim"]),
        context_embedding_dim=int(config["context_embedding_dim"]),
        slot_embedding_dim=int(config["slot_embedding_dim"]),
        dropout=float(config["dropout"]),
        use_soft_topology_conditioning=bool(state.get("soft_topology_conditioning", False)),
    ).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    return model


def aggregate_seed_metrics(per_seed: Mapping[str, Mapping[str, Any]], split: str, path: Sequence[str]) -> Dict[str, Any]:
    values = []
    for seed_result in per_seed.values():
        current: Any = seed_result.get(split)
        for key in path:
            if not isinstance(current, Mapping):
                current = None
                break
            current = current.get(key)
        if isinstance(current, (int, float)):
            values.append(float(current))
    if not values:
        return {"mean": None, "std": None, "values": []}
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "values": values}


def run(config_path: Path) -> Dict[str, Any]:
    if p9e.torch is None:
        raise RuntimeError(f"PyTorch is required for P9f: {p9e.TORCH_IMPORT_ERROR}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected P9f config schema")
    if tuple(config.get("seeds") or ()) != (11, 22, 33):
        raise ValueError("P9f must audit P9e B seeds 11/22/33")
    if config.get("data_boundary", {}).get("t_final_read") is not False:
        raise ValueError("T_final must remain unread")
    if config.get("data_boundary", {}).get("scheduler_groups_used") is not False:
        raise ValueError("scheduler groups must remain excluded")
    dataset_root = Path(config["dataset_root"])
    experiment_root = Path(config["experiment_root"])
    experiment_root.mkdir(parents=True, exist_ok=True)
    stage0 = p9e.stage0_audit(dataset_root, str(config.get("dataset_manifest_sha256") or ""))
    if not stage0["passed"]:
        raise ValueError("P9f Stage 0 failed: " + "; ".join(stage0["errors"][:8]))
    write_json(experiment_root / "stage0_audit.json", stage0)

    pairs = {split: _read_split(dataset_root, split) for split in SPLITS}
    behavior, behavior_audit = join_behavior_targets(
        pairs,
        Path(config["p_dev_role_samples"]),
        Path(config["p_holdout_role_samples"]),
        Path(config["p_dev_family_samples"]),
        Path(config["p_holdout_family_samples"]),
    )
    encoder = InputEncoder(int(config["max_history"])).fit(pairs["train"])
    codec = p9e.ContentCodec().fit(pairs["train"])
    bundles: Dict[str, Dict[str, np.ndarray]] = {}
    for split in SPLITS:
        encoded = encoder.transform(pairs[split])
        encoded.update(p9e.build_target_arrays(pairs[split], behavior[split], codec))
        bundles[split] = encoded

    resource_dev, resource_dev_audit = load_resource_map(Path(config["resource_dev_compute_path"]))
    resource_holdout, resource_holdout_audit = load_resource_map(Path(config["resource_holdout_compute_path"]))
    profiles, profile_audit = train_resource_profiles(pairs["train"], resource_dev)
    device = p9e.torch.device(str(config.get("device", "cuda:0")))
    if device.type == "cuda" and not p9e.torch.cuda.is_available():
        raise RuntimeError("configured CUDA device is unavailable")

    per_seed: Dict[str, Any] = {}
    started = time.time()
    grid = tuple(float(value) for value in config.get("temperature_grid") or DEFAULT_TEMPERATURE_GRID)
    for seed in config["seeds"]:
        checkpoint = Path(config["checkpoint_paths"][str(seed)])
        model = load_model(config, checkpoint, encoder, codec, device)
        outputs = {
            split: p9e._predict_outputs(model, bundles[split], int(config["batch_size"]), device)
            for split in SPLITS
        }
        calibrators = fit_calibrators(outputs["train"], bundles["train"], grid)
        calibration = {
            split: calibration_report(outputs[split], bundles[split], calibrators)
            for split in SPLITS
        }
        array_outputs = {split: tensors_to_numpy(outputs[split]) for split in SPLITS}
        split_results: Dict[str, Any] = {}
        for split in EVALUATION_SPLITS:
            resource_map = resource_holdout if split == "holdout" else resource_dev
            split_results[split] = {
                "core_metrics": structure_and_behavior_metrics(
                    array_outputs[split], bundles[split], np.arange(len(pairs[split]))
                ),
                "calibration": calibration[split],
                "stratified": stratified_report(pairs[split], array_outputs[split], bundles[split]),
                "future_cost": evaluate_cost_split(
                    pairs[split], outputs[split], codec, profiles, resource_map
                ),
            }
        per_seed[str(seed)] = {
            "seed": int(seed),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "calibration_fit": calibrators,
            "train_calibration": calibration["train"],
            "validation": split_results["validation"],
            "test": split_results["test"],
            "holdout": split_results["holdout"],
        }
        del model, outputs
        if device.type == "cuda":
            p9e.torch.cuda.empty_cache()

    baseline_path = Path(config["empirical_baseline_metrics_path"])
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_holdout = baseline.get("selected_metrics", {}).get("holdout", {})
    summary_paths = {
        "holdout_layer_count_mae": ("core_metrics", "layer_count_mae"),
        "holdout_width_vector_mae": ("core_metrics", "width_vector_mae"),
        "holdout_node_count_mae": ("core_metrics", "node_count_mae"),
        "holdout_structure_exact": ("core_metrics", "structure_exact_coverage"),
        "holdout_next_role_f1": ("core_metrics", "behavior_role_macro_f1"),
        "holdout_next_family_f1": ("core_metrics", "behavior_family_macro_f1"),
        "holdout_top1_cost_mae_ms": ("future_cost", "top1_runtime_p50", "mae_ms"),
        "holdout_expected_cost_mae_ms": ("future_cost", "expected_runtime_p50", "mae_ms"),
    }
    aggregate_summary = {
        name: aggregate_seed_metrics(per_seed, "holdout", path) for name, path in summary_paths.items()
    }
    learned_holdout = {
        "layer_count_mae": aggregate_summary["holdout_layer_count_mae"]["mean"],
        "width_vector_mae": aggregate_summary["holdout_width_vector_mae"]["mean"],
        "node_count_mae": aggregate_summary["holdout_node_count_mae"]["mean"],
        "structure_exact_coverage": aggregate_summary["holdout_structure_exact"]["mean"],
    }
    structure_improvements = {
        "layer_count_mae_lower": learned_holdout["layer_count_mae"] < baseline_holdout.get("layer_count_mae", math.inf),
        "width_vector_mae_lower": learned_holdout["width_vector_mae"] < baseline_holdout.get("width_vector_mae", math.inf),
        "node_count_mae_lower": learned_holdout["node_count_mae"] < baseline_holdout.get("node_count_mae", math.inf),
    }
    calibration_head_results: Dict[str, Any] = {}
    for head in ("layer_count", "width", "next_role", "next_family"):
        raw = []
        calibrated = []
        for result in per_seed.values():
            raw_value = result["holdout"]["calibration"]["heads"][head]["raw"]
            calibrated_value = result["holdout"]["calibration"]["heads"][head]["calibrated"]
            raw.append(raw_value)
            calibrated.append(calibrated_value)
        calibration_head_results[head] = {
            "raw_mean": {key: float(np.mean([item[key] for item in raw])) for key in ("nll", "brier", "ece")},
            "calibrated_mean": {key: float(np.mean([item[key] for item in calibrated])) for key in ("nll", "brier", "ece")},
            "nll_not_worse": float(np.mean([item["nll"] for item in calibrated]))
            <= float(np.mean([item["nll"] for item in raw])) + 1e-12,
            "ece_not_worse": float(np.mean([item["ece"] for item in calibrated]))
            <= float(np.mean([item["ece"] for item in raw])) + 1e-12,
        }

    metrics = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "status": "completed_diagnostic",
        "stage0": stage0,
        "fit_contract": {
            "model_fit": "none; frozen P9e B checkpoints only",
            "calibration_fit": "P_dev/train logits only",
            "selection_split": "none; P9e epoch/variant already frozen",
            "diagnostic_splits": ["P_dev/validation", "P_dev/test"],
            "frozen_holdout": "P_holdout_diag/holdout",
            "seeds": list(config["seeds"]),
            "temperature_grid": list(grid),
        },
        "data_contract": {
            "dataset_manifest_sha256": sha256_file(dataset_root / "dataset_manifest.json"),
            "split_counts": stage0["split_counts"],
            "behavior_join": behavior_audit,
            "scheduler_groups_used": False,
            "t_final_read": False,
            "future_events_or_edges_used_as_model_input": False,
            "resource_truth_used_as_model_input": False,
        },
        "resource_contract": {
            "development_compute": resource_dev_audit,
            "holdout_compute": resource_holdout_audit,
            "train_profile": profile_audit,
            "profile_sha256": hashlib.sha256(
                json.dumps(profiles, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "evaluated_proxy": "role|action_family -> train P50 runtime_ms",
            "model_aware_load_memory": "not_identifiable_from_reduced_future_node_contract",
        },
        "per_seed": per_seed,
        "aggregate_summary": aggregate_summary,
        "comparison_to_empirical_holdout": {
            "empirical_baseline_metrics_path": str(baseline_path),
            "empirical_baseline_metrics_sha256": sha256_file(baseline_path),
            "empirical_selected_holdout": baseline_holdout,
            "learned_B_holdout_mean": learned_holdout,
            "structure_improvements": structure_improvements,
        },
        "calibration_summary": calibration_head_results,
        "acceptance_gate": {
            "structure_better_than_empirical_on_all_three_mae": all(structure_improvements.values()),
            "calibration_reported_for_all_heads": all(
                head in calibration_head_results for head in ("layer_count", "width", "next_role", "next_family")
            ),
            "resource_proxy_has_complete_holdout_runtime_coverage": all(
                result["holdout"]["future_cost"]["true_runtime_coverage"] >= 0.99
                for result in per_seed.values()
            ),
            "model_aware_resource_gate": False,
            "scheduler_integration_allowed": False,
            "reason": "P9e reduced future-node outputs do not identify model_id/node_type/input_scale/cold_warm for load or memory.",
        },
        "boundary": {
            "scheduler_integration_started": False,
            "scheduler_groups_used_for_fit": False,
            "t_final_read": False,
            "raw_traces_modified": False,
        },
        "reproducibility": {
            "command": [sys.executable] + sys.argv,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": p9e.torch.__version__,
            "cuda": p9e.torch.version.cuda,
            "device": str(device),
            "gpu_name": p9e.torch.cuda.get_device_name(0) if p9e.torch.cuda.is_available() else None,
            "elapsed_seconds": time.time() - started,
        },
    }
    write_json(experiment_root / "metrics.json", metrics)
    write_json(experiment_root / "cost_profiles.json", profiles)
    write_json(experiment_root / "run_manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "status": metrics["status"],
        "source_sha256": {
            "scripts/p9f_predictor_acceptance_audit.py": sha256_file(Path(__file__)),
            "scripts/p9d_future_role_family_multitask.py": sha256_file(PROJECT_ROOT / "scripts/p9d_future_role_family_multitask.py"),
            "scripts/p9d_shared_causal_gru.py": sha256_file(PROJECT_ROOT / "scripts/p9d_shared_causal_gru.py"),
            "experiments/config.json": sha256_file(config_path),
        },
        "inputs": {
            "dataset_manifest_sha256": sha256_file(dataset_root / "dataset_manifest.json"),
            "resource_dev_compute_sha256": resource_dev_audit["source_sha256"],
            "resource_holdout_compute_sha256": resource_holdout_audit["source_sha256"],
            "empirical_baseline_metrics_sha256": sha256_file(baseline_path),
            "checkpoint_sha256": {str(seed): per_seed[str(seed)]["checkpoint_sha256"] for seed in config["seeds"]},
        },
        "outputs_sha256": {
            "stage0_audit.json": sha256_file(experiment_root / "stage0_audit.json"),
            "cost_profiles.json": sha256_file(experiment_root / "cost_profiles.json"),
            "metrics.json": sha256_file(experiment_root / "metrics.json"),
        },
        "verification": {
            "stage0_passed": True,
            "frozen_checkpoints_only": True,
            "calibration_fit_split": "P_dev/train",
            "scheduler_groups_used": False,
            "t_final_read": False,
            "raw_traces_modified": False,
            "scheduler_integration_started": False,
        },
    })
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps({
        "experiment_id": result["experiment_id"],
        "status": result["status"],
        "acceptance_gate": result["acceptance_gate"],
        "aggregate_summary": result["aggregate_summary"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
