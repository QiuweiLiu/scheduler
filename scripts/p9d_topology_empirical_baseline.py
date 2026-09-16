#!/usr/bin/env python3
"""Run the P9d train-only conditional empirical topology baseline.

The input is the predictor-aligned P9d feature/label dataset.  Only the
training split is used to fit conditional frequency tables.  Validation is
used to select one pre-registered condition schema; test is diagnostic and
holdout is evaluated only after that selection is frozen.

The target signature is identity-free: each future DAG layer is represented
by a sorted multiset of scheduler-visible node prototypes.  Node ids, edges,
runtime, memory, status and other execution truth are never used in a model
key or emitted prediction.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


HORIZON = 5
MAX_SCENARIOS = 3
SCHEMA_VERSION = "p9d-topology-empirical-baseline-v1"
PROTOTYPE_FIELDS = (
    "node_type",
    "raw_action",
    "model_id",
    "execution_lane",
    "action_family",
)
CURRENT_FULL_FIELDS = (
    "event_type",
    "node_type",
    "role",
    "raw_action",
    "action_family",
    "model_id",
)
CURRENT_STRUCTURAL_FIELDS = (
    "event_type",
    "node_type",
    "role",
    "action_family",
)
TASK_FIELDS = (
    "answer_type",
    "domain",
    "question_type",
    "required_modalities",
    "temporal_scope",
    "sub_category",
    "official_task_type",
)
STACK_FIELDS = ("baseline", "model_stack_id", "planner_model_id")
HISTORY_FIELDS = (
    "event_type",
    "node_type",
    "role",
    "raw_action",
    "action_family",
    "model_id",
)
SCHEME_NAMES = ("current_full", "current_structural", "history2_full")

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

Signature = Tuple[Tuple[Tuple[str, ...], ...], ...]
ConditionKey = Tuple[str, ...]
Pair = Tuple[Mapping[str, Any], Mapping[str, Any]]


def _open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with _open_text(path, "r") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if value is None:
        return "unknown"
    text = str(value).strip()
    return text or "unknown"


def _audit_model_input(value: Any, path: str = "model_input") -> List[str]:
    leaked: List[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in FORBIDDEN_MODEL_INPUT_KEYS:
                leaked.append(f"{path}.{key_text}")
            leaked.extend(_audit_model_input(child, f"{path}.{key_text}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            leaked.extend(_audit_model_input(child, f"{path}[{index}]"))
    return leaked


def _field_values(mapping: Mapping[str, Any], fields: Sequence[str], prefix: str) -> List[str]:
    result: List[str] = []
    for field in fields:
        result.extend((f"{prefix}.{field}", _canonical(mapping.get(field))))
    return result


def condition_key(model_input: Mapping[str, Any], scheme: str) -> ConditionKey:
    """Build one of the pre-registered causal condition keys."""

    if scheme not in SCHEME_NAMES:
        raise ValueError(f"unknown condition scheme: {scheme}")
    leaked = _audit_model_input(model_input)
    if leaked:
        raise ValueError(f"model input contains forbidden keys: {leaked}")
    stack = model_input.get("stack_context") or {}
    task = model_input.get("task_context") or {}
    current = model_input.get("current_node") or {}
    if not isinstance(stack, Mapping) or not isinstance(task, Mapping) or not isinstance(current, Mapping):
        raise ValueError("stack_context, task_context and current_node must be objects")

    values: List[str] = [f"scheme={scheme}"]
    values.extend(_field_values(stack, STACK_FIELDS, "stack"))
    values.extend(_field_values(task, TASK_FIELDS, "task"))
    current_fields = CURRENT_FULL_FIELDS if scheme != "current_structural" else CURRENT_STRUCTURAL_FIELDS
    values.extend(_field_values(current, current_fields, "current"))

    if scheme == "history2_full":
        history = model_input.get("history") or []
        if not isinstance(history, list) or not history:
            raise ValueError("history2_full requires a non-empty history")
        previous = history[:-1][-2:]
        padded: List[Mapping[str, Any]] = ([{"_pad": "<BOS>"}] * (2 - len(previous))) + previous
        for index, token in enumerate(padded):
            if not isinstance(token, Mapping):
                raise ValueError("history entries must be objects")
            if "_pad" in token:
                values.extend((f"history{index + 1}._pad", _canonical(token.get("_pad"))))
            else:
                values.extend(_field_values(token, HISTORY_FIELDS, f"history{index + 1}"))
    return tuple(values)


def _prototype(node: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(_canonical(node.get(field)) for field in PROTOTYPE_FIELDS)


def topology_signature(label: Mapping[str, Any], horizon: int = HORIZON) -> Signature:
    """Extract a canonical identity-free layer signature from one label."""

    if int(label.get("future_horizon", horizon)) != horizon:
        raise ValueError(f"unexpected label horizon: {label.get('future_horizon')}")
    layers = label.get("future_layers") or []
    if not isinstance(layers, list):
        raise ValueError("future_layers must be a list")
    if len(layers) > horizon:
        raise ValueError("future_layers exceeds configured horizon")
    result: List[Tuple[Tuple[str, ...], ...]] = []
    for expected_offset, layer in enumerate(layers, 1):
        if not isinstance(layer, Mapping):
            raise ValueError("future layer must be an object")
        if int(layer.get("layer_offset", -1)) != expected_offset:
            raise ValueError("future layer offsets must be contiguous")
        nodes = layer.get("nodes") or []
        if not isinstance(nodes, list) or not nodes:
            raise ValueError("future layers must not be empty")
        prototypes: List[Tuple[str, ...]] = []
        for node in nodes:
            if not isinstance(node, Mapping):
                raise ValueError("future node must be an object")
            prototypes.append(_prototype(node))
        result.append(tuple(sorted(prototypes)))
    return tuple(result)


def signature_hash(signature: Signature) -> str:
    payload = json.dumps(signature, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_split(dataset_root: Path, split: str) -> List[Pair]:
    features = _read_jsonl(dataset_root / f"features_{split}.jsonl.gz")
    labels = _read_jsonl(dataset_root / f"labels_{split}.jsonl.gz")
    if len(features) != len(labels):
        raise ValueError(f"{split} feature/label count mismatch: {len(features)} != {len(labels)}")
    pairs: List[Pair] = []
    seen: Set[str] = set()
    for index, (feature, label) in enumerate(zip(features, labels)):
        feature_id = str(feature.get("sample_id"))
        label_id = str(label.get("sample_id"))
        if not feature_id or feature_id != label_id:
            raise ValueError(f"{split}:{index} feature/label sample_id mismatch")
        if feature_id in seen:
            raise ValueError(f"{split}:{index} duplicate sample_id {feature_id}")
        seen.add(feature_id)
        if str(feature.get("split")) != split or str(label.get("split")) != split:
            raise ValueError(f"{split}:{index} split field mismatch")
        model_input = feature.get("model_input")
        if not isinstance(model_input, Mapping):
            raise ValueError(f"{split}:{index} missing model_input")
        leaked = _audit_model_input(model_input)
        if leaked:
            raise ValueError(f"{split}:{index} causal input leak: {leaked}")
        topology_signature(label)
        pairs.append((feature, label))
    if not pairs:
        raise ValueError(f"empty split: {split}")
    return pairs


class EmpiricalModel:
    def __init__(self) -> None:
        self.by_key: Dict[ConditionKey, Counter[Signature]] = defaultdict(Counter)
        self.global_counter: Counter[Signature] = Counter()
        self._ranked_by_key: Dict[ConditionKey, List[Tuple[Signature, int]]] = {}
        self._ranked_global: Optional[List[Tuple[Signature, int]]] = None

    def add(self, key: ConditionKey, signature: Signature) -> None:
        self.by_key[key][signature] += 1
        self.global_counter[signature] += 1

    def ranked_for(self, key: ConditionKey) -> Tuple[Counter[Signature], bool, List[Tuple[Signature, int]]]:
        conditional = self.by_key.get(key)
        if conditional is None:
            if self._ranked_global is None:
                self._ranked_global = _ranked(self.global_counter)
            return self.global_counter, False, self._ranked_global
        if key not in self._ranked_by_key:
            self._ranked_by_key[key] = _ranked(conditional)
        return conditional, True, self._ranked_by_key[key]


def fit_model(pairs: Sequence[Pair], scheme: str) -> EmpiricalModel:
    model = EmpiricalModel()
    for feature, label in pairs:
        model.add(condition_key(feature["model_input"], scheme), topology_signature(label))
    if not model.global_counter:
        raise ValueError(f"no training targets for scheme {scheme}")
    return model


def _ranked(counter: Counter[Signature]) -> List[Tuple[Signature, int]]:
    return sorted(
        counter.items(),
        key=lambda item: (-int(item[1]), json.dumps(item[0], ensure_ascii=False, separators=(",", ":"))),
    )


def _layers_json(signature: Signature, source: str) -> List[Dict[str, Any]]:
    layers: List[Dict[str, Any]] = []
    for layer_offset, layer in enumerate(signature, 1):
        nodes: List[Dict[str, Any]] = []
        for node_index, values in enumerate(layer):
            node = dict(zip(PROTOTYPE_FIELDS, values))
            node.update(
                {
                    "layer_offset": layer_offset,
                    "predicted_node_index": node_index,
                    "prototype_source": source,
                }
            )
            nodes.append(node)
        layers.append({"layer_offset": layer_offset, "nodes": nodes})
    return layers


def predict_scenarios(
    model: EmpiricalModel,
    key: ConditionKey,
    max_scenarios: int = MAX_SCENARIOS,
) -> Tuple[List[Dict[str, Any]], bool, int, float]:
    if max_scenarios < 1:
        raise ValueError("max_scenarios must be positive")
    counter, seen_key, ranked = model.ranked_for(key)
    if not ranked:
        raise ValueError("cannot predict from an empty empirical model")
    selected = ranked[:max_scenarios]
    selected_total = sum(count for _signature, count in selected)
    full_total = sum(counter.values())
    scenarios: List[Dict[str, Any]] = []
    for rank, (signature, count) in enumerate(selected):
        scenarios.append(
            {
                "scenario_id": f"empirical_top{rank:02d}",
                "scenario_probability": float(count) / float(selected_total),
                "layers": _layers_json(signature, "conditional_empirical_train"),
                "synthetic_rollout": True,
                "topology_source": "conditional_empirical_train",
                "support_count": int(count),
            }
        )
    return scenarios, seen_key, len(ranked), float(selected_total) / float(full_total)


def evaluate(pairs: Sequence[Pair], model: EmpiricalModel, scheme: str, nll_floor: float) -> Dict[str, Any]:
    if nll_floor <= 0.0 or nll_floor >= 1.0:
        raise ValueError("nll_floor must be in (0, 1)")
    count = len(pairs)
    nll_sum = 0.0
    top1_hits = 0
    top3_hits = 0
    future_hits = 0
    layer_exact_hits = 0
    layer_abs = 0.0
    layer_bias = 0.0
    node_abs = 0.0
    node_bias = 0.0
    first_width_abs = 0.0
    first_width_bias = 0.0
    width_vector_abs = 0.0
    seen_keys = 0
    support_size = 0.0
    top3_mass = 0.0
    for feature, label in pairs:
        true_signature = topology_signature(label)
        key = condition_key(feature["model_input"], scheme)
        scenarios, seen_key, support, mass = predict_scenarios(model, key)
        conditional, _seen_key, ranked = model.ranked_for(key)
        true_probability = float(conditional.get(true_signature, 0)) / float(sum(conditional.values()))
        nll_sum += -math.log(max(true_probability, nll_floor))
        top_signatures = [item[0] for item in ranked[:MAX_SCENARIOS]]
        predicted = top_signatures[0]
        top1_hits += int(predicted == true_signature)
        top3_hits += int(true_signature in top_signatures)
        future_hits += int(bool(predicted) == bool(true_signature))
        layer_count = len(predicted)
        true_layer_count = len(true_signature)
        node_count = sum(len(layer) for layer in predicted)
        true_node_count = sum(len(layer) for layer in true_signature)
        first_width = len(predicted[0]) if predicted else 0
        true_first_width = len(true_signature[0]) if true_signature else 0
        layer_abs += abs(layer_count - true_layer_count)
        layer_bias += layer_count - true_layer_count
        node_abs += abs(node_count - true_node_count)
        node_bias += node_count - true_node_count
        first_width_abs += abs(first_width - true_first_width)
        first_width_bias += first_width - true_first_width
        for index in range(HORIZON):
            width = len(predicted[index]) if index < len(predicted) else 0
            true_width = len(true_signature[index]) if index < len(true_signature) else 0
            width_vector_abs += abs(width - true_width)
        layer_exact_hits += int(layer_count == true_layer_count)
        seen_keys += int(seen_key)
        support_size += support
        top3_mass += mass
    return {
        "sample_count": count,
        "nll_mean": nll_sum / count,
        "nll_floor": nll_floor,
        "top1_exact_signature_coverage": float(top1_hits) / count,
        "top3_exact_signature_coverage": float(top3_hits) / count,
        "future_exists_accuracy": float(future_hits) / count,
        "layer_count_exact_accuracy": float(layer_exact_hits) / count,
        "layer_count_mae": layer_abs / count,
        "layer_count_bias": layer_bias / count,
        "node_count_mae": node_abs / count,
        "node_count_bias": node_bias / count,
        "first_layer_width_mae": first_width_abs / count,
        "first_layer_width_bias": first_width_bias / count,
        "width_vector_mae": width_vector_abs / float(count * HORIZON),
        "seen_condition_key_rate": float(seen_keys) / count,
        "mean_condition_support_size": support_size / count,
        "mean_top3_probability_mass": top3_mass / count,
    }


def prediction_rows(
    pairs: Sequence[Pair],
    model: EmpiricalModel,
    scheme: str,
    split: str,
) -> Iterable[Mapping[str, Any]]:
    for feature, label in pairs:
        true_signature = topology_signature(label)
        key = condition_key(feature["model_input"], scheme)
        scenarios, seen_key, support, mass = predict_scenarios(model, key)
        yield {
            "sample_id": feature["sample_id"],
            "split": split,
            "condition_key_scheme": scheme,
            "seen_condition_key": seen_key,
            "condition_support_size": support,
            "top3_probability_mass": mass,
            "true_signature_hash": signature_hash(true_signature),
            "true_summary": {
                "future_layer_count": len(true_signature),
                "future_node_count": sum(len(layer) for layer in true_signature),
                "layer_widths": [len(layer) for layer in true_signature],
                "first_layer_width": len(true_signature[0]) if true_signature else 0,
                "has_future": bool(true_signature),
            },
            "top_scenarios": scenarios,
            "prediction_contract": {
                "identity_free": True,
                "future_edges_emitted": False,
                "successor_ids_emitted": False,
                "execution_truth_emitted": False,
                "scenario_probability_normalization": "top3_selected_support",
            },
        }


def _write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _unique_count(pairs: Sequence[Pair], field: str) -> int:
    return len({str(feature.get(field)) for feature, _label in pairs})


def _model_summary(model: EmpiricalModel, pairs: Sequence[Pair]) -> Dict[str, Any]:
    supports = [sum(counter.values()) for counter in model.by_key.values()]
    global_ranked = _ranked(model.global_counter)[:5]
    return {
        "train_sample_count": len(pairs),
        "train_video_count": _unique_count(pairs, "video_id"),
        "condition_key_count": len(model.by_key),
        "signature_support_count": len(model.global_counter),
        "condition_support_min": min(supports),
        "condition_support_max": max(supports),
        "global_top5": [
            {"signature_hash": signature_hash(signature), "count": int(count)}
            for signature, count in global_ranked
        ],
    }


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected config schema: {config.get('schema_version')}")
    if int(config.get("horizon", -1)) != HORIZON:
        raise ValueError("P9d empirical baseline is fixed at H=5")
    if int(config.get("max_scenarios", -1)) != MAX_SCENARIOS:
        raise ValueError("P9d empirical baseline is fixed at top-3 scenarios")
    schemes = tuple(config.get("candidate_schemes") or ())
    if schemes != SCHEME_NAMES:
        raise ValueError(f"candidate schemes must be exactly {SCHEME_NAMES}, got {schemes}")
    selection = config.get("selection") or {}
    if selection.get("primary_metric") != "nll_mean":
        raise ValueError("selection must use validation nll_mean")
    if selection.get("split") != "validation":
        raise ValueError("selection must use validation only")


def run(config_path: Path) -> Dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    _validate_config(config)
    experiment_root = Path(str(config["experiment_root"]))
    dataset_root = Path(str(config["dataset_root"]))
    artifact_root = experiment_root / "artifacts"
    metrics_path = experiment_root / "metrics.json"
    if artifact_root.exists() and any(artifact_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty artifact directory: {artifact_root}")
    if metrics_path.exists():
        raise FileExistsError(f"refusing to overwrite existing metrics: {metrics_path}")
    manifest_path = dataset_root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "topology-predictor-p9d-v1":
        raise ValueError("dataset schema is not topology-predictor-p9d-v1")
    source = manifest.get("source") or {}
    if source.get("scheduler_trace_groups_used_for_fit") is not False:
        raise ValueError("dataset manifest does not prove scheduler traces were excluded from fit")
    if source.get("s_train_s_val_t_final_used") is not False:
        raise ValueError("dataset manifest does not prove S/T groups were excluded")
    expected_manifest_hash = config.get("dataset_manifest_sha256")
    actual_manifest_hash = _sha256(manifest_path)
    if expected_manifest_hash and expected_manifest_hash != actual_manifest_hash:
        raise ValueError(f"dataset manifest hash mismatch: {actual_manifest_hash}")

    splits: Dict[str, List[Pair]] = {}
    for split in ("train", "validation", "test", "holdout"):
        splits[split] = read_split(dataset_root, split)

    models: Dict[str, EmpiricalModel] = {
        scheme: fit_model(splits["train"], scheme) for scheme in SCHEME_NAMES
    }
    candidate_validation_metrics = {
        scheme: evaluate(splits["validation"], models[scheme], scheme, float(config["nll_floor"]))
        for scheme in SCHEME_NAMES
    }
    selected_scheme = min(
        SCHEME_NAMES,
        key=lambda scheme: (
            candidate_validation_metrics[scheme]["nll_mean"],
            -candidate_validation_metrics[scheme]["top3_exact_signature_coverage"],
            candidate_validation_metrics[scheme]["layer_count_mae"],
            scheme,
        ),
    )
    selected_model = models[selected_scheme]
    selected_metrics = {
        split: evaluate(splits[split], selected_model, selected_scheme, float(config["nll_floor"]))
        for split in ("validation", "test", "holdout")
    }

    artifact_root.mkdir(parents=True, exist_ok=True)
    prediction_paths: Dict[str, str] = {}
    prediction_counts: Dict[str, int] = {}
    for split in ("validation", "test", "holdout"):
        path = artifact_root / f"predictions_{split}.jsonl.gz"
        prediction_counts[split] = _write_jsonl_gz(
            path, prediction_rows(splits[split], selected_model, selected_scheme, split)
        )
        prediction_paths[split] = str(path)

    dataset_files = {
        name: _sha256(dataset_root / name)
        for name in (
            "features_train.jsonl.gz",
            "labels_train.jsonl.gz",
            "features_validation.jsonl.gz",
            "labels_validation.jsonl.gz",
            "features_test.jsonl.gz",
            "labels_test.jsonl.gz",
            "features_holdout.jsonl.gz",
            "labels_holdout.jsonl.gz",
            "dataset_manifest.json",
            "alignment_report.json",
        )
    }
    metrics: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "status": "passed_empirical_baseline",
        "fit_contract": {
            "fit_split": "train",
            "selection_split": "validation",
            "diagnostic_split": "test",
            "frozen_holdout_split": "holdout",
            "fitted_schemes": list(SCHEME_NAMES),
            "selected_scheme": selected_scheme,
            "nll_floor": float(config["nll_floor"]),
            "max_scenarios": MAX_SCENARIOS,
            "horizon": HORIZON,
            "random_seed": int(config.get("seed", 0)),
            "randomness": "none_frequency_counts_only",
        },
        "candidate_validation_metrics": candidate_validation_metrics,
        "selected_metrics": selected_metrics,
        "model_summary": {scheme: _model_summary(models[scheme], splits["train"]) for scheme in SCHEME_NAMES},
        "split_counts": {
            split: {
                "samples": len(splits[split]),
                "videos": _unique_count(splits[split], "video_id"),
                "runs": _unique_count(splits[split], "run_id"),
            }
            for split in ("train", "validation", "test", "holdout")
        },
        "prediction_artifacts": {
            "paths": prediction_paths,
            "rows": prediction_counts,
            "identity_free": True,
            "true_node_ids_emitted": False,
            "true_edges_emitted": False,
        },
        "inputs": {
            "dataset_root": str(dataset_root),
            "dataset_manifest_sha256": actual_manifest_hash,
            "dataset_files_sha256": dataset_files,
            "config_sha256": _sha256(config_path),
        },
        "reproducibility": {
            "command": [sys.executable] + sys.argv,
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "git_commit": config.get("git_commit", "unavailable_no_git_metadata"),
            "remote_host_identity": config.get("remote_host_identity", "not_recorded"),
        },
        "boundary": {
            "scheduler_groups_used_for_fit": False,
            "t_final_read": False,
            "raw_traces_modified": False,
            "behavior_or_resource_targets_used_as_features": False,
            "future_events_or_edges_used_as_features": False,
            "resource_truth_used_as_features": False,
            "model_type": "conditional_empirical_frequency_baseline",
        },
    }
    _write_json(metrics_path, metrics)
    _write_json(artifact_root / "model_summary.json", metrics["model_summary"])
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    metrics = run(parse_args().config)
    print(
        json.dumps(
            {
                "status": metrics["status"],
                "experiment_id": metrics["experiment_id"],
                "selected_scheme": metrics["fit_contract"]["selected_scheme"],
                "selected_metrics": metrics["selected_metrics"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
