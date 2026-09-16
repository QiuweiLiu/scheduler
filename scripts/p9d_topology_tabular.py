#!/usr/bin/env python3
"""Run the P9d tabular topology comparator.

LightGBM predicts the causal structural targets (future layer count and the
width at each of five layers).  Node prototypes are decoded from a separate
train-only empirical table so that this first tabular comparator can be
evaluated on the complete identity-free layer contract without inventing
resource or future-edge features.

Two pre-registered LightGBM configurations are fit on P_dev/train.  The
validation split selects one configuration; test is diagnostic and holdout
is evaluated only after selection is frozen.
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

import numpy as np


HORIZON = 5
MAX_WIDTH = 5
MAX_SCENARIOS = 3
SCHEMA_VERSION = "p9d-topology-tabular-v1"
PROTOTYPE_FIELDS = (
    "node_type",
    "raw_action",
    "model_id",
    "execution_lane",
    "action_family",
)
STACK_FIELDS = ("baseline", "model_stack_id", "planner_model_id")
TASK_FIELDS = (
    "answer_type",
    "domain",
    "question_type",
    "required_modalities",
    "temporal_scope",
    "sub_category",
    "official_task_type",
)
CURRENT_FIELDS = (
    "event_type",
    "node_type",
    "role",
    "raw_action",
    "action_family",
    "model_id",
)
HISTORY_FIELDS = (
    "event_type",
    "node_type",
    "role",
    "raw_action",
    "action_family",
    "model_id",
)
FEATURE_FIELDS = (
    tuple("stack." + field for field in STACK_FIELDS)
    + tuple("task." + field for field in TASK_FIELDS)
    + tuple("current." + field for field in CURRENT_FIELDS)
    + tuple("history1." + field for field in HISTORY_FIELDS)
    + tuple("history2." + field for field in HISTORY_FIELDS)
)
MODEL_CONFIGS = ("lgbm_small", "lgbm_wide")
PAD = "__PAD__"
NLL_FLOOR = 1e-12
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
Pair = Tuple[Mapping[str, Any], Mapping[str, Any]]
ContextKey = Tuple[Tuple[str, str], ...]


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


def _field_values(mapping: Mapping[str, Any], fields: Sequence[str], prefix: str) -> List[Tuple[str, str]]:
    return [(f"{prefix}.{field}", _canonical(mapping.get(field))) for field in fields]


def context_values(model_input: Mapping[str, Any]) -> Dict[str, str]:
    leaked = _audit_model_input(model_input)
    if leaked:
        raise ValueError(f"causal model input leak: {leaked}")
    stack = model_input.get("stack_context") or {}
    task = model_input.get("task_context") or {}
    current = model_input.get("current_node") or {}
    history = model_input.get("history") or []
    if not isinstance(stack, Mapping) or not isinstance(task, Mapping) or not isinstance(current, Mapping):
        raise ValueError("stack_context, task_context and current_node must be objects")
    if not isinstance(history, list) or not history:
        raise ValueError("history must be a non-empty list")
    previous = history[:-1][-2:]
    padded: List[Mapping[str, Any]] = ([{"_pad": PAD}] * (2 - len(previous))) + previous
    values: Dict[str, str] = {}
    values.update(dict(_field_values(stack, STACK_FIELDS, "stack")))
    values.update(dict(_field_values(task, TASK_FIELDS, "task")))
    values.update(dict(_field_values(current, CURRENT_FIELDS, "current")))
    for index, token in enumerate(padded, 1):
        if not isinstance(token, Mapping):
            raise ValueError("history entries must be objects")
        if "_pad" in token:
            values[f"history{index}._pad"] = _canonical(token.get("_pad"))
        else:
            values.update(dict(_field_values(token, HISTORY_FIELDS, f"history{index}")))
    values["history_length"] = str(len(history))
    return values


def context_key(model_input: Mapping[str, Any]) -> ContextKey:
    values = context_values(model_input)
    return tuple(sorted(values.items()))


class FeatureEncoder:
    """Train-only categorical one-hot encoder for causal model_input fields."""

    def __init__(self) -> None:
        self.categories: Dict[str, List[str]] = {}
        self.feature_names: List[str] = []
        self.history_scale = 32.0

    def fit(self, pairs: Sequence[Pair]) -> "FeatureEncoder":
        observed: Dict[str, Set[str]] = {field: set() for field in FEATURE_FIELDS}
        for feature, _label in pairs:
            values = context_values(feature["model_input"])
            for field in FEATURE_FIELDS:
                observed[field].add(values.get(field, "unknown"))
        self.categories = {field: sorted(values) for field, values in observed.items()}
        self.feature_names = [
            f"{field}={value}" for field in FEATURE_FIELDS for value in self.categories[field]
        ] + ["history_length_scaled"]
        return self

    def transform_one(self, feature: Mapping[str, Any]) -> np.ndarray:
        values = context_values(feature["model_input"])
        output: List[float] = []
        for field in FEATURE_FIELDS:
            choices = self.categories[field]
            current = values.get(field, "unknown")
            output.extend(1.0 if current == choice else 0.0 for choice in choices)
        history_length = float(values["history_length"])
        output.append(history_length / self.history_scale)
        return np.asarray(output, dtype=np.float32)

    def transform(self, pairs: Sequence[Pair]) -> np.ndarray:
        return np.stack([self.transform_one(feature) for feature, _label in pairs])


def topology_signature(label: Mapping[str, Any], horizon: int = HORIZON) -> Signature:
    if int(label.get("future_horizon", horizon)) != horizon:
        raise ValueError(f"unexpected label horizon: {label.get('future_horizon')}")
    layers = label.get("future_layers") or []
    if not isinstance(layers, list) or len(layers) > horizon:
        raise ValueError("invalid future_layers")
    result: List[Tuple[Tuple[str, ...], ...]] = []
    for offset, layer in enumerate(layers, 1):
        if not isinstance(layer, Mapping) or int(layer.get("layer_offset", -1)) != offset:
            raise ValueError("future layer offsets must be contiguous")
        nodes = layer.get("nodes") or []
        if not isinstance(nodes, list) or not nodes:
            raise ValueError("future layers must not be empty")
        if len(nodes) > MAX_WIDTH:
            raise ValueError(f"future layer width exceeds configured maximum {MAX_WIDTH}")
        prototypes = [_prototype(node) for node in nodes if isinstance(node, Mapping)]
        if len(prototypes) != len(nodes):
            raise ValueError("future node must be an object")
        result.append(tuple(sorted(prototypes)))
    return tuple(result)


def _prototype(node: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(_canonical(node.get(field)) for field in PROTOTYPE_FIELDS)


def _shape_targets(signature: Signature) -> Tuple[int, Tuple[int, ...]]:
    return len(signature), tuple(len(signature[index]) if index < len(signature) else 0 for index in range(HORIZON))


def _prototype_value(signature: Signature, layer_index: int, node_index: int, field_index: int) -> str:
    if layer_index < len(signature) and node_index < len(signature[layer_index]):
        return signature[layer_index][node_index][field_index]
    return PAD


class PrototypeDecoder:
    """Train-only empirical decoder for node prototype fields."""

    def __init__(self) -> None:
        self.by_context: Dict[ContextKey, Dict[Tuple[int, int, int], Counter[str]]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        self.global_counts: Dict[Tuple[int, int, int], Counter[str]] = defaultdict(Counter)
        self._top_cache: Dict[Tuple[ContextKey, Tuple[int, int, int]], str] = {}

    def fit(self, pairs: Sequence[Pair]) -> "PrototypeDecoder":
        for feature, label in pairs:
            signature = topology_signature(label)
            key = context_key(feature["model_input"])
            for layer_index in range(HORIZON):
                for node_index in range(MAX_WIDTH):
                    for field_index in range(len(PROTOTYPE_FIELDS)):
                        slot = (layer_index, node_index, field_index)
                        value = _prototype_value(signature, layer_index, node_index, field_index)
                        self.by_context[key][slot][value] += 1
                        self.global_counts[slot][value] += 1
        return self

    def _counter(self, key: ContextKey, slot: Tuple[int, int, int]) -> Counter[str]:
        conditional = self.by_context.get(key, {}).get(slot)
        if conditional:
            return conditional
        return self.global_counts[slot]

    def probability(self, feature: Mapping[str, Any], slot: Tuple[int, int, int], value: str) -> float:
        counter = self._counter(context_key(feature["model_input"]), slot)
        total = sum(counter.values())
        if not total:
            return 0.0
        return float(counter.get(value, 0)) / float(total)

    def top_value(
        self,
        feature: Mapping[str, Any],
        slot: Tuple[int, int, int],
        key: Optional[ContextKey] = None,
    ) -> str:
        if key is None:
            key = context_key(feature["model_input"])
        cache_key = (key, slot)
        if cache_key in self._top_cache:
            return self._top_cache[cache_key]
        counter = self._counter(key, slot)
        candidates = [(value, count) for value, count in counter.items() if value != PAD]
        if not candidates:
            result = "unknown"
        else:
            result = min(candidates, key=lambda item: (-int(item[1]), item[0]))[0]
        self._top_cache[cache_key] = result
        return result


class LightGBMHead:
    def __init__(self, params: Mapping[str, Any], seed: int) -> None:
        self.params = dict(params)
        self.seed = seed
        self.classes: List[int] = []
        self.model: Any = None

    def fit(self, x_train: np.ndarray, values: Sequence[int]) -> "LightGBMHead":
        import lightgbm as lgb

        self.classes = sorted(set(int(value) for value in values))
        class_to_index = {value: index for index, value in enumerate(self.classes)}
        encoded = np.asarray([class_to_index[int(value)] for value in values], dtype=np.int32)
        self.model = lgb.LGBMClassifier(
            objective="multiclass",
            n_estimators=int(self.params["n_estimators"]),
            learning_rate=float(self.params["learning_rate"]),
            num_leaves=int(self.params["num_leaves"]),
            min_child_samples=int(self.params["min_child_samples"]),
            max_depth=int(self.params["max_depth"]),
            random_state=self.seed,
            n_jobs=1,
            verbosity=-1,
        )
        self.model.fit(x_train, encoded)
        return self

    def probabilities(self, x: np.ndarray) -> Dict[int, float]:
        raw = np.asarray(self.model.predict_proba(x.reshape(1, -1))[0], dtype=np.float64)
        return {value: float(raw[index]) for index, value in enumerate(self.classes)}

    def probabilities_batch(self, x: np.ndarray) -> List[Dict[int, float]]:
        raw = np.asarray(self.model.predict_proba(x), dtype=np.float64)
        return [
            {value: float(row[index]) for index, value in enumerate(self.classes)}
            for row in raw
        ]


class TabularModel:
    def __init__(self, params: Mapping[str, Any], seed: int) -> None:
        self.params = dict(params)
        self.seed = seed
        self.encoder = FeatureEncoder()
        self.prototype_decoder = PrototypeDecoder()
        self.heads: Dict[str, LightGBMHead] = {}

    def fit(self, pairs: Sequence[Pair]) -> "TabularModel":
        self.encoder.fit(pairs)
        x_train = self.encoder.transform(pairs)
        targets: Dict[str, List[int]] = {"layer_count": [], **{f"width_{i + 1}": [] for i in range(HORIZON)}}
        for _feature, label in pairs:
            layer_count, widths = _shape_targets(topology_signature(label))
            targets["layer_count"].append(layer_count)
            for index, width in enumerate(widths):
                targets[f"width_{index + 1}"].append(width)
        for name, values in targets.items():
            self.heads[name] = LightGBMHead(self.params, self.seed).fit(x_train, values)
        self.prototype_decoder.fit(pairs)
        return self

    def distributions(self, feature: Mapping[str, Any]) -> Dict[str, Dict[int, float]]:
        x = self.encoder.transform_one(feature)
        return {name: head.probabilities(x) for name, head in self.heads.items()}

    def distributions_batch(self, pairs: Sequence[Pair]) -> List[Dict[str, Dict[int, float]]]:
        x = self.encoder.transform(pairs)
        by_head: Dict[str, List[Dict[int, float]]] = {}
        for name, head in self.heads.items():
            if hasattr(head, "probabilities_batch"):
                by_head[name] = head.probabilities_batch(x)
            else:
                by_head[name] = [head.probabilities(row) for row in x]
        return [
            {name: by_head[name][index] for name in self.heads}
            for index in range(len(pairs))
        ]


def _top_choices(distribution: Mapping[int, float], positive_only: bool = False, limit: int = 3) -> List[Tuple[int, float]]:
    values = [
        (int(value), max(float(probability), NLL_FLOOR))
        for value, probability in distribution.items()
        if not positive_only or int(value) > 0
    ]
    if not values:
        raise ValueError("no valid tabular width choices")
    return sorted(values, key=lambda item: (-item[1], item[0]))[:limit]


def _decode_shape_scenarios(
    model: TabularModel,
    feature: Mapping[str, Any],
    distributions: Optional[Mapping[str, Mapping[int, float]]] = None,
    max_scenarios: int = MAX_SCENARIOS,
) -> List[Tuple[float, int, Tuple[int, ...]]]:
    if distributions is None:
        distributions = model.distributions(feature)
    layer_choices = _top_choices(distributions["layer_count"], limit=3)
    beams: List[Tuple[float, int, Tuple[int, ...]]] = []
    for layer_count, layer_probability in layer_choices:
        if layer_count < 0 or layer_count > HORIZON:
            continue
        beams.append((layer_probability, layer_count, tuple()))
    for layer_index in range(HORIZON):
        expanded: List[Tuple[float, int, Tuple[int, ...]]] = []
        for probability, layer_count, widths in beams:
            if layer_index >= layer_count:
                expanded.append((probability, layer_count, widths + (0,)))
                continue
            choices = _top_choices(distributions[f"width_{layer_index + 1}"], positive_only=True)
            for width, width_probability in choices:
                expanded.append((probability * width_probability, layer_count, widths + (width,)))
        beams = sorted(expanded, key=lambda item: (-item[0], item[1], item[2]))[:12]
    unique: Dict[Tuple[int, Tuple[int, ...]], float] = {}
    for probability, layer_count, widths in beams:
        unique[(layer_count, widths)] = max(unique.get((layer_count, widths), 0.0), probability)
    return [
        (probability, layer_count, widths)
        for (layer_count, widths), probability in sorted(
            unique.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        )[:max_scenarios]
    ]


def _scenario_json(
    model: TabularModel,
    feature: Mapping[str, Any],
    layer_count: int,
    widths: Sequence[int],
    probability: float,
    rank: int,
) -> Dict[str, Any]:
    layers: List[Dict[str, Any]] = []
    key = context_key(feature["model_input"])
    for layer_index in range(layer_count):
        nodes: List[Dict[str, Any]] = []
        for node_index in range(int(widths[layer_index])):
            values = [
                model.prototype_decoder.top_value(
                    feature, (layer_index, node_index, field_index), key
                )
                for field_index in range(len(PROTOTYPE_FIELDS))
            ]
            node = dict(zip(PROTOTYPE_FIELDS, values))
            node.update(
                {
                    "layer_offset": layer_index + 1,
                    "predicted_node_index": node_index,
                    "prototype_source": "train_empirical_tabular_decoder",
                }
            )
            nodes.append(node)
        layers.append({"layer_offset": layer_index + 1, "nodes": nodes})
    return {
        "scenario_id": f"tabular_top{rank:02d}",
        "scenario_probability": probability,
        "layers": layers,
        "synthetic_rollout": True,
        "topology_source": "tabular_lgbm_structure_train_empirical_prototypes",
        "prototype_context_key_hash": hashlib.sha256(
            json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20],
    }


def _predicted_signature(scenario: Mapping[str, Any]) -> Signature:
    result: List[Tuple[Tuple[str, ...], ...]] = []
    for layer in scenario.get("layers") or []:
        nodes = layer.get("nodes") or []
        result.append(tuple(sorted(_prototype(node) for node in nodes)))
    return tuple(result)


def read_split(dataset_root: Path, split: str) -> List[Pair]:
    features = _read_jsonl(dataset_root / f"features_{split}.jsonl.gz")
    labels = _read_jsonl(dataset_root / f"labels_{split}.jsonl.gz")
    if len(features) != len(labels):
        raise ValueError(f"{split} feature/label mismatch")
    pairs: List[Pair] = []
    seen: Set[str] = set()
    for index, (feature, label) in enumerate(zip(features, labels)):
        sample_id = str(feature.get("sample_id"))
        if not sample_id or sample_id != str(label.get("sample_id")) or sample_id in seen:
            raise ValueError(f"{split}:{index} invalid sample_id alignment")
        if str(feature.get("split")) != split or str(label.get("split")) != split:
            raise ValueError(f"{split}:{index} split mismatch")
        model_input = feature.get("model_input")
        if not isinstance(model_input, Mapping) or _audit_model_input(model_input):
            raise ValueError(f"{split}:{index} invalid causal model_input")
        topology_signature(label)
        seen.add(sample_id)
        pairs.append((feature, label))
    if not pairs:
        raise ValueError(f"empty split {split}")
    return pairs


def evaluate(pairs: Sequence[Pair], model: TabularModel) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    count = len(pairs)
    nll_sum = 0.0
    top1_exact = 0
    top3_exact = 0
    shape_top1 = 0
    future_exists = 0
    layer_exact = 0
    layer_abs = 0.0
    layer_bias = 0.0
    node_abs = 0.0
    node_bias = 0.0
    first_abs = 0.0
    first_bias = 0.0
    width_abs = 0.0
    prototype_correct = 0
    prototype_total = 0
    output_rows: List[Dict[str, Any]] = []
    batch_distributions = model.distributions_batch(pairs)
    for pair_index, (feature, label) in enumerate(pairs):
        true_signature = topology_signature(label)
        true_layer_count, true_widths = _shape_targets(true_signature)
        distributions = batch_distributions[pair_index]
        true_layer_probability = distributions["layer_count"].get(true_layer_count, 0.0)
        shape_probability = max(float(true_layer_probability), NLL_FLOOR)
        for index, width in enumerate(true_widths):
            if index >= true_layer_count:
                break
            shape_probability *= max(
                float(distributions[f"width_{index + 1}"].get(width, 0.0)), NLL_FLOOR
            )
        nll_sum += -math.log(shape_probability)
        decoded = _decode_shape_scenarios(model, feature, distributions)
        raw_total = sum(item[0] for item in decoded)
        scenarios = [
            _scenario_json(model, feature, layer_count, widths, probability / raw_total, rank)
            for rank, (probability, layer_count, widths) in enumerate(decoded)
        ]
        predicted_signature = _predicted_signature(scenarios[0])
        top_signatures = [_predicted_signature(scenario) for scenario in scenarios]
        top1_exact += int(predicted_signature == true_signature)
        top3_exact += int(true_signature in top_signatures)
        predicted_shape = (len(predicted_signature), tuple(len(layer) for layer in predicted_signature))
        shape_top1 += int(predicted_shape == (true_layer_count, true_widths[:true_layer_count]))
        future_exists += int(bool(predicted_signature) == bool(true_signature))
        predicted_layer_count = len(predicted_signature)
        predicted_widths = tuple(len(layer) for layer in predicted_signature)
        predicted_node_count = sum(predicted_widths)
        true_node_count = sum(true_widths)
        predicted_first = predicted_widths[0] if predicted_widths else 0
        true_first = true_widths[0] if true_widths else 0
        layer_exact += int(predicted_layer_count == true_layer_count)
        layer_abs += abs(predicted_layer_count - true_layer_count)
        layer_bias += predicted_layer_count - true_layer_count
        node_abs += abs(predicted_node_count - true_node_count)
        node_bias += predicted_node_count - true_node_count
        first_abs += abs(predicted_first - true_first)
        first_bias += predicted_first - true_first
        for index in range(HORIZON):
            predicted_width = predicted_widths[index] if index < len(predicted_widths) else 0
            true_width = true_widths[index]
            width_abs += abs(predicted_width - true_width)
        for layer_index, layer in enumerate(true_signature):
            for node_index, true_node in enumerate(layer):
                for field_index, true_value in enumerate(true_node):
                    if layer_index < len(predicted_signature) and node_index < len(predicted_signature[layer_index]):
                        prototype_total += 1
                        prototype_correct += int(
                            predicted_signature[layer_index][node_index][field_index] == true_value
                        )
        output_rows.append(
            {
                "sample_id": feature["sample_id"],
                "split": feature["split"],
                "true_signature_hash": hashlib.sha256(
                    json.dumps(true_signature, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "true_summary": {
                    "future_layer_count": true_layer_count,
                    "future_node_count": true_node_count,
                    "layer_widths": list(true_widths[:true_layer_count]),
                    "first_layer_width": true_first,
                    "has_future": bool(true_signature),
                },
                "top_scenarios": scenarios,
                "prediction_contract": {
                    "identity_free": True,
                    "true_node_ids_emitted": False,
                    "true_edges_emitted": False,
                    "resource_truth_emitted": False,
                    "structure_probability": "factorized_layer_count_and_active_widths",
                },
            }
        )
    return (
        {
            "sample_count": count,
            "structure_nll_mean": nll_sum / count,
            "layer_count_mae": layer_abs / count,
            "layer_count_bias": layer_bias / count,
            "layer_count_exact_accuracy": layer_exact / count,
            "node_count_mae": node_abs / count,
            "node_count_bias": node_bias / count,
            "first_layer_width_mae": first_abs / count,
            "first_layer_width_bias": first_bias / count,
            "width_vector_mae": width_abs / float(count * HORIZON),
            "future_exists_accuracy": future_exists / count,
            "top1_exact_signature_coverage": top1_exact / count,
            "top3_exact_signature_coverage": top3_exact / count,
            "top1_exact_shape_coverage": shape_top1 / count,
            "prototype_field_accuracy": prototype_correct / prototype_total if prototype_total else 0.0,
            "prototype_field_observations": prototype_total,
        },
        output_rows,
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def _params(config_name: str) -> Dict[str, Any]:
    if config_name == "lgbm_small":
        return {
            "n_estimators": 120,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "max_depth": -1,
        }
    if config_name == "lgbm_wide":
        return {
            "n_estimators": 200,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "min_child_samples": 10,
            "max_depth": -1,
        }
    raise ValueError(f"unknown config {config_name}")


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected tabular config schema")
    if int(config.get("horizon", -1)) != HORIZON or int(config.get("max_width", -1)) != MAX_WIDTH:
        raise ValueError("tabular structural dimensions are fixed at H=5 and width=5")
    candidates = tuple(config.get("candidate_models") or ())
    if candidates != MODEL_CONFIGS:
        raise ValueError(f"candidate models must be {MODEL_CONFIGS}")
    selection = config.get("selection") or {}
    if selection.get("split") != "validation" or selection.get("primary_metric") != "layer_count_mae":
        raise ValueError("validation layer_count_mae must be the primary selection metric")


def run(config_path: Path) -> Dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config must be an object")
    _validate_config(config)
    dataset_root = Path(str(config["dataset_root"]))
    experiment_root = Path(str(config["experiment_root"]))
    artifact_root = experiment_root / "artifacts"
    metrics_path = experiment_root / "metrics.json"
    if artifact_root.exists() and any(artifact_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty artifact directory: {artifact_root}")
    if metrics_path.exists():
        raise FileExistsError(f"refusing to overwrite metrics: {metrics_path}")
    manifest_path = dataset_root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "topology-predictor-p9d-v1":
        raise ValueError("wrong P9d dataset schema")
    source = manifest.get("source") or {}
    if source.get("scheduler_trace_groups_used_for_fit") is not False or source.get("s_train_s_val_t_final_used") is not False:
        raise ValueError("dataset manifest does not prove scheduler groups were excluded")
    actual_manifest_hash = _sha256(manifest_path)
    expected_manifest_hash = config.get("dataset_manifest_sha256")
    if expected_manifest_hash and expected_manifest_hash != actual_manifest_hash:
        raise ValueError("dataset manifest hash mismatch")
    splits = {split: read_split(dataset_root, split) for split in ("train", "validation", "test", "holdout")}
    candidates: Dict[str, Dict[str, Any]] = {}
    for config_name in MODEL_CONFIGS:
        model = TabularModel(_params(config_name), int(config.get("seed", 0))).fit(splits["train"])
        validation_metrics, _validation_rows = evaluate(splits["validation"], model)
        candidates[config_name] = {"model": model, "validation": validation_metrics}
    selected_name = min(
        MODEL_CONFIGS,
        key=lambda name: (
            candidates[name]["validation"]["layer_count_mae"],
            candidates[name]["validation"]["width_vector_mae"],
            candidates[name]["validation"]["node_count_mae"],
            -candidates[name]["validation"]["top3_exact_signature_coverage"],
            name,
        ),
    )
    selected_model = candidates[selected_name]["model"]
    selected_metrics: Dict[str, Dict[str, Any]] = {}
    prediction_rows: Dict[str, List[Dict[str, Any]]] = {}
    for split in ("validation", "test", "holdout"):
        selected_metrics[split], prediction_rows[split] = evaluate(splits[split], selected_model)
    artifact_root.mkdir(parents=True, exist_ok=True)
    prediction_paths: Dict[str, str] = {}
    prediction_counts: Dict[str, int] = {}
    for split in ("validation", "test", "holdout"):
        path = artifact_root / f"predictions_{split}.jsonl.gz"
        prediction_counts[split] = _write_jsonl_gz(path, prediction_rows[split])
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
        "status": "passed_tabular_comparator",
        "fit_contract": {
            "fit_split": "train",
            "selection_split": "validation",
            "diagnostic_split": "test",
            "frozen_holdout_split": "holdout",
            "candidate_models": list(MODEL_CONFIGS),
            "selected_model": selected_name,
            "seed": int(config.get("seed", 0)),
            "horizon": HORIZON,
            "max_width": MAX_WIDTH,
            "max_scenarios": MAX_SCENARIOS,
        },
        "candidate_validation_metrics": {
            name: candidates[name]["validation"] for name in MODEL_CONFIGS
        },
        "selected_metrics": selected_metrics,
        "split_counts": {
            split: {
                "samples": len(splits[split]),
                "videos": len({str(feature.get("video_id")) for feature, _label in splits[split]}),
                "runs": len({str(feature.get("run_id")) for feature, _label in splits[split]}),
            }
            for split in ("train", "validation", "test", "holdout")
        },
        "feature_contract": {
            "source": "P9d model_input only",
            "one_hot_fields": list(FEATURE_FIELDS),
            "numeric_fields": ["history_length_scaled"],
            "future_events_or_edges_used": False,
            "resource_truth_used": False,
            "video_id_used_as_feature": False,
            "behavior_or_resource_targets_used_as_feature": False,
        },
        "target_contract": {
            "structure_heads": ["layer_count"] + [f"width_{i + 1}" for i in range(HORIZON)],
            "prototype_decoder": "train_only_empirical_conditional",
            "identity_free_output": True,
            "node_ids_or_edges_emitted": False,
            "execution_lane": "unknown_from_dataset",
        },
        "prediction_artifacts": {
            "paths": prediction_paths,
            "rows": prediction_counts,
            "identity_free": True,
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
            "git_commit": config.get("git_commit", "unavailable_uncommitted_source_mirror"),
            "remote_host_identity": config.get("remote_host_identity", "not_recorded"),
            "lightgbm_version": __import__("lightgbm").__version__,
        },
        "boundary": {
            "scheduler_groups_used_for_fit": False,
            "t_final_read": False,
            "raw_traces_modified": False,
            "scheduler_integration_started": False,
        },
    }
    _write_json(metrics_path, metrics)
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
                "selected_model": metrics["fit_contract"]["selected_model"],
                "selected_metrics": metrics["selected_metrics"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
