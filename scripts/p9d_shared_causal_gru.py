#!/usr/bin/env python3
"""Train the predictor-aligned shared causal GRU experiment.

The experiment uses one causal history encoder and separate behavior/topology
heads.  Behavior targets are joined from the existing train-only role/tool
views; topology targets come only from the P9d labels.  The output topology is
identity-free H=5 DAG-layer shape plus trainable node prototypes.

Four pre-registered variants are compared:
``behavior_only``, ``topology_only``, ``shared_multitask`` and
``shared_first_layer_consistency``.  Every fit uses P_dev/train, validation
selects the best epoch, test is diagnostic, and holdout is evaluated only
after each run is frozen.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

try:  # The local macOS environment has no torch; the formal run is remote.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - exercised by local contract tests
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = str(exc)
else:
    TORCH_IMPORT_ERROR = None


SCHEMA_VERSION = "p9d-shared-causal-gru-v1"
DATASET_SCHEMA_VERSION = "topology-predictor-p9d-v1"
HORIZON = 5
MAX_WIDTH = 5
MAX_SCENARIOS = 3
MAX_HISTORY = 64
PAD = "__PAD__"
UNK = "__UNK__"
NLL_FLOOR = 1e-12

ROLE_LABELS = ("aggregate", "execute", "init", "plan", "terminate")
FAMILY_LABELS = ("detect", "other", "select_frames", "summarize", "temporal_ops", "visual_qa")
PROTOTYPE_FIELDS = ("node_type", "raw_action", "model_id", "execution_lane", "action_family")
AUXILIARY_TOPOLOGY_FIELDS = ("role",)
ALL_TOPOLOGY_FIELDS = PROTOTYPE_FIELDS + AUXILIARY_TOPOLOGY_FIELDS
HISTORY_FIELDS = ("event_type", "node_type", "role", "raw_action", "action_family", "model_id")
CONTEXT_FIELDS = (
    "stack.baseline",
    "stack.model_stack_id",
    "stack.planner_model_id",
    "task.answer_type",
    "task.domain",
    "task.question_type",
    "task.required_modalities",
    "task.temporal_scope",
    "task.sub_category",
    "task.official_task_type",
)
ROLE_INDEX = {value: index for index, value in enumerate(ROLE_LABELS)}
FAMILY_INDEX = {value: index for index, value in enumerate(FAMILY_LABELS)}
TERMINATE_INDEX = ROLE_INDEX["terminate"]

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


def _open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with _open_text(path, "r") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if value is None:
        return UNK
    result = str(value).strip()
    return result or UNK


def audit_model_input(value: Any, path: str = "model_input") -> List[str]:
    violations: List[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in FORBIDDEN_MODEL_INPUT_KEYS:
                violations.append(f"{path}.{key_text}")
            violations.extend(audit_model_input(child, f"{path}.{key_text}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            violations.extend(audit_model_input(child, f"{path}[{index}]"))
    return violations


def _context_values(model_input: Mapping[str, Any]) -> Dict[str, str]:
    violations = audit_model_input(model_input)
    if violations:
        raise ValueError(f"causal model input leak: {violations}")
    stack = model_input.get("stack_context") or {}
    task = model_input.get("task_context") or {}
    current = model_input.get("current_node") or {}
    history = model_input.get("history") or []
    if not isinstance(stack, Mapping) or not isinstance(task, Mapping) or not isinstance(current, Mapping):
        raise ValueError("stack_context, task_context and current_node must be objects")
    if not isinstance(history, list):
        raise ValueError("history must be a list")
    values: Dict[str, str] = {
        "stack.baseline": _canonical(stack.get("baseline")),
        "stack.model_stack_id": _canonical(stack.get("model_stack_id")),
        "stack.planner_model_id": _canonical(stack.get("planner_model_id")),
        "task.answer_type": _canonical(task.get("answer_type")),
        "task.domain": _canonical(task.get("domain")),
        "task.question_type": _canonical(task.get("question_type")),
        "task.required_modalities": _canonical(task.get("required_modalities")),
        "task.temporal_scope": _canonical(task.get("temporal_scope")),
        "task.sub_category": _canonical(task.get("sub_category")),
        "task.official_task_type": _canonical(task.get("official_task_type")),
    }
    return values


def topology_signature(label: Mapping[str, Any]) -> Signature:
    if int(label.get("future_horizon", -1)) != HORIZON:
        raise ValueError(f"unexpected future horizon: {label.get('future_horizon')}")
    layers = label.get("future_layers") or []
    if not isinstance(layers, list) or len(layers) > HORIZON:
        raise ValueError("future_layers must contain at most H=5 layers")
    result: List[Tuple[Tuple[str, ...], ...]] = []
    for offset, layer in enumerate(layers, 1):
        if not isinstance(layer, Mapping) or int(layer.get("layer_offset", -1)) != offset:
            raise ValueError("future layer offsets must be contiguous")
        nodes = layer.get("nodes") or []
        if not isinstance(nodes, list) or not nodes or len(nodes) > MAX_WIDTH:
            raise ValueError("future layer width is outside the configured range")
        prototypes = []
        for node in nodes:
            if not isinstance(node, Mapping):
                raise ValueError("future node must be an object")
            prototypes.append(tuple(_canonical(node.get(field)) for field in PROTOTYPE_FIELDS))
        result.append(tuple(sorted(prototypes)))
    return tuple(result)


def shape_targets(signature: Signature) -> Tuple[int, Tuple[int, ...]]:
    return len(signature), tuple(
        len(signature[index]) if index < len(signature) else 0 for index in range(HORIZON)
    )


def _role_layers(label: Mapping[str, Any]) -> Tuple[Tuple[str, ...], ...]:
    layers = label.get("future_layers") or []
    result: List[Tuple[str, ...]] = []
    for layer in layers:
        result.append(tuple(_canonical(node.get("role")) for node in layer.get("nodes") or []))
    return tuple(result)


def _read_split(dataset_root: Path, split: str) -> List[Pair]:
    features = _read_jsonl(dataset_root / f"features_{split}.jsonl.gz")
    labels = _read_jsonl(dataset_root / f"labels_{split}.jsonl.gz")
    if len(features) != len(labels):
        raise ValueError(f"{split} feature/label count mismatch")
    pairs: List[Pair] = []
    seen: Set[str] = set()
    for index, (feature, label) in enumerate(zip(features, labels)):
        sample_id = str(feature.get("sample_id"))
        if not sample_id or sample_id in seen or sample_id != str(label.get("sample_id")):
            raise ValueError(f"{split}:{index} invalid sample_id alignment")
        if feature.get("split") != split or label.get("split") != split:
            raise ValueError(f"{split}:{index} split mismatch")
        model_input = feature.get("model_input")
        if not isinstance(model_input, Mapping) or audit_model_input(model_input):
            raise ValueError(f"{split}:{index} invalid causal model_input")
        if feature.get("video_id") is None or feature.get("run_id") is None:
            raise ValueError(f"{split}:{index} missing metadata")
        topology_signature(label)
        seen.add(sample_id)
        pairs.append((feature, label))
    if not pairs:
        raise ValueError(f"empty {split} split")
    return pairs


def _key(run_id: Any, event_id: Any) -> Tuple[str, str]:
    return str(run_id), str(event_id)


def _unique_map(rows: Sequence[Mapping[str, Any]], keys: Sequence[str], value_field: str, name: str) -> Dict[Tuple[str, str], str]:
    result: Dict[Tuple[str, str], str] = {}
    for row in rows:
        key = _key(row.get(keys[0]), row.get(keys[1]))
        value = str(row.get(value_field))
        if not key[0] or not key[1] or value in ("None", ""):
            continue
        previous = result.get(key)
        if previous is not None and previous != value:
            raise ValueError(f"conflicting {name} labels for {key}: {previous!r} vs {value!r}")
        result[key] = value
    return result


def join_behavior_targets(
    pairs_by_split: Mapping[str, Sequence[Pair]],
    role_path: Path,
    holdout_role_path: Path,
    family_path: Path,
    holdout_family_path: Path,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Join role labels by current anchor and execute-family labels by target event."""

    role_rows = _read_jsonl(role_path)
    holdout_role_rows = _read_jsonl(holdout_role_path)
    family_rows = _read_jsonl(family_path)
    holdout_family_rows = _read_jsonl(holdout_family_path)

    def role_index(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], Mapping[str, Any]]:
        result: Dict[Tuple[str, str], Mapping[str, Any]] = {}
        for row in rows:
            key = _key(row.get("run_id"), row.get("source_event_id"))
            if not key[0] or not key[1]:
                continue
            previous = result.get(key)
            if previous is not None:
                if (previous.get("next_role"), previous.get("target_source_event_id")) != (
                    row.get("next_role"), row.get("target_source_event_id")
                ):
                    raise ValueError(f"conflicting role labels for {key}")
                continue
            result[key] = row
        return result

    role_maps = {"train": role_index(role_rows), "validation": role_index(role_rows), "test": role_index(role_rows), "holdout": role_index(holdout_role_rows)}
    family_maps = {
        "train": _unique_map(family_rows, ("run_id", "target_source_event_id"), "family_label", "family"),
        "validation": _unique_map(family_rows, ("run_id", "target_source_event_id"), "family_label", "family"),
        "test": _unique_map(family_rows, ("run_id", "target_source_event_id"), "family_label", "family"),
        "holdout": _unique_map(holdout_family_rows, ("run_id", "target_source_event_id"), "family_label", "family"),
    }
    joined: Dict[str, List[Dict[str, Any]]] = {}
    coverage: Dict[str, Any] = {}
    for split, pairs in pairs_by_split.items():
        rows: List[Dict[str, Any]] = []
        role_map = role_maps[split]
        family_map = family_maps[split]
        missing_role = 0
        family_count = 0
        for feature, _label in pairs:
            anchor = _key(feature.get("run_id"), feature.get("current_node_id"))
            role_row = role_map.get(anchor)
            if role_row is None:
                missing_role += 1
                continue
            next_role = str(role_row.get("next_role"))
            if next_role not in ROLE_INDEX:
                raise ValueError(f"unknown next_role {next_role!r} for {anchor}")
            target_key = _key(role_row.get("run_id"), role_row.get("target_source_event_id"))
            family = family_map.get(target_key)
            if family is not None:
                if next_role != "execute":
                    raise ValueError(f"family target is not execute-gated for {anchor}")
                if family not in FAMILY_INDEX:
                    raise ValueError(f"unknown family label {family!r} for {anchor}")
                family_count += 1
            rows.append(
                {
                    "role": next_role,
                    "family": family,
                    "family_mask": family is not None,
                }
            )
        if missing_role:
            raise ValueError(f"{split} role anchor join missing {missing_role} rows")
        if len(rows) != len(pairs):
            raise ValueError(f"{split} behavior join changed row count")
        joined[split] = rows
        coverage[split] = {
            "p9d_rows": len(pairs),
            "role_joined": len(rows),
            "family_joined": family_count,
            "family_coverage": family_count / len(rows),
        }
    return joined, {
        "source_rows": {
            "p_dev_role": len(role_rows),
            "p_holdout_role": len(holdout_role_rows),
            "p_dev_family": len(family_rows),
            "p_holdout_family": len(holdout_family_rows),
        },
        "coverage": coverage,
        "family_target_semantics": "execute_gated semantic_tool target_source_event_id join",
    }


class InputEncoder:
    """Train-only categorical encoding for P9d model_input."""

    def __init__(self, max_history: int = MAX_HISTORY) -> None:
        self.max_history = max_history
        self.history_maps: Dict[str, Dict[str, int]] = {}
        self.context_maps: Dict[str, Dict[str, int]] = {}

    @staticmethod
    def _map(values: Iterable[str]) -> Dict[str, int]:
        result = {PAD: 0, UNK: 1}
        for index, value in enumerate(sorted(set(values)), 2):
            if value not in result:
                result[value] = index
        return result

    def fit(self, pairs: Sequence[Pair]) -> "InputEncoder":
        history_values = {field: [] for field in HISTORY_FIELDS}
        context_values = {field: [] for field in CONTEXT_FIELDS}
        for feature, _label in pairs:
            model_input = feature["model_input"]
            for token in model_input.get("history") or []:
                for field in HISTORY_FIELDS:
                    history_values[field].append(_canonical(token.get(field)))
            values = _context_values(model_input)
            for field in CONTEXT_FIELDS:
                context_values[field].append(values[field])
        self.history_maps = {field: self._map(values) for field, values in history_values.items()}
        self.context_maps = {field: self._map(values) for field, values in context_values.items()}
        return self

    def transform(self, pairs: Sequence[Pair]) -> Dict[str, np.ndarray]:
        if not self.history_maps or not self.context_maps:
            raise ValueError("InputEncoder must be fit on train pairs first")
        tokens = np.zeros((len(pairs), self.max_history, len(HISTORY_FIELDS)), dtype=np.int64)
        lengths = np.zeros(len(pairs), dtype=np.int64)
        context = np.zeros((len(pairs), len(CONTEXT_FIELDS)), dtype=np.int64)
        for row_index, (feature, _label) in enumerate(pairs):
            model_input = feature["model_input"]
            history = list(model_input.get("history") or [])[-self.max_history :]
            lengths[row_index] = len(history)
            for time_index, token in enumerate(history):
                for field_index, field in enumerate(HISTORY_FIELDS):
                    tokens[row_index, time_index, field_index] = self.history_maps[field].get(
                        _canonical(token.get(field)), 1
                    )
            values = _context_values(model_input)
            for field_index, field in enumerate(CONTEXT_FIELDS):
                context[row_index, field_index] = self.context_maps[field].get(values[field], 1)
        return {"tokens": tokens, "lengths": lengths, "context": context}

    def vocab_sizes(self) -> Tuple[List[int], List[int]]:
        return (
            [len(self.history_maps[field]) for field in HISTORY_FIELDS],
            [len(self.context_maps[field]) for field in CONTEXT_FIELDS],
        )


class TargetCodec:
    """Train-only target vocabulary and dense target arrays."""

    def __init__(self) -> None:
        self.field_maps: Dict[str, Dict[str, int]] = {}

    def fit(self, pairs: Sequence[Pair]) -> "TargetCodec":
        values = {field: [] for field in ALL_TOPOLOGY_FIELDS}
        for _feature, label in pairs:
            layers = label.get("future_layers") or []
            for layer in layers:
                for node in layer.get("nodes") or []:
                    for field in PROTOTYPE_FIELDS:
                        values[field].append(_canonical(node.get(field)))
                    values["role"].append(_canonical(node.get("role")))
        self.field_maps = {field: InputEncoder._map(items) for field, items in values.items()}
        return self

    def transform(self, pairs: Sequence[Pair]) -> Dict[str, np.ndarray]:
        if not self.field_maps:
            raise ValueError("TargetCodec must be fit on train pairs first")
        layer_targets = np.zeros(len(pairs), dtype=np.int64)
        width_targets = np.zeros((len(pairs), HORIZON), dtype=np.int64)
        prototype_targets = np.zeros(
            (len(pairs), HORIZON, MAX_WIDTH, len(PROTOTYPE_FIELDS)), dtype=np.int64
        )
        role_targets = np.full((len(pairs), HORIZON, MAX_WIDTH), -1, dtype=np.int64)
        family_targets = np.full((len(pairs), HORIZON, MAX_WIDTH), -1, dtype=np.int64)
        for row_index, (_feature, label) in enumerate(pairs):
            layers = label.get("future_layers") or []
            layer_targets[row_index] = len(layers)
            for layer_index, layer in enumerate(layers):
                nodes = layer.get("nodes") or []
                width_targets[row_index, layer_index] = len(nodes)
                for node_index, node in enumerate(nodes):
                    for field_index, field in enumerate(PROTOTYPE_FIELDS):
                        prototype_targets[row_index, layer_index, node_index, field_index] = self.field_maps[field].get(
                            _canonical(node.get(field)), 1
                        )
                    role = _canonical(node.get("role"))
                    if role not in ROLE_INDEX:
                        raise ValueError(f"unknown topology node role {role!r}")
                    role_targets[row_index, layer_index, node_index] = ROLE_INDEX[role]
                    family = _canonical(node.get("action_family"))
                    if family not in FAMILY_INDEX:
                        raise ValueError(f"unknown topology node family {family!r}")
                    family_targets[row_index, layer_index, node_index] = FAMILY_INDEX[family]
        return {
            "layer_targets": layer_targets,
            "width_targets": width_targets,
            "prototype_targets": prototype_targets,
            "topology_role_targets": role_targets,
            "topology_family_targets": family_targets,
        }

    def field_sizes(self) -> List[int]:
        return [len(self.field_maps[field]) for field in PROTOTYPE_FIELDS]

    def role_size(self) -> int:
        return len(ROLE_LABELS)

    def decode(self, field: str, index: int) -> str:
        choices = self.field_maps[field]
        reverse = {value: key for key, value in choices.items()}
        value = reverse.get(int(index), UNK)
        return "unknown" if value in (PAD, UNK) else value


def build_target_arrays(
    pairs: Sequence[Pair], behavior_targets: Sequence[Mapping[str, Any]], codec: TargetCodec
) -> Dict[str, np.ndarray]:
    if len(pairs) != len(behavior_targets):
        raise ValueError("behavior target count mismatch")
    result = codec.transform(pairs)
    result["behavior_role_targets"] = np.asarray(
        [ROLE_INDEX[str(row["role"])] for row in behavior_targets], dtype=np.int64
    )
    result["behavior_family_targets"] = np.asarray(
        [FAMILY_INDEX.get(str(row["family"]), -1) if row["family"] is not None else -1 for row in behavior_targets],
        dtype=np.int64,
    )
    result["behavior_family_mask"] = np.asarray(
        [bool(row["family_mask"]) for row in behavior_targets], dtype=bool
    )
    return result


if torch is not None:

    class SharedCausalGRU(nn.Module):
        def __init__(
            self,
            history_vocab_sizes: Sequence[int],
            context_vocab_sizes: Sequence[int],
            prototype_field_sizes: Sequence[int],
            topology_role_size: int,
            hidden: int = 128,
            history_embedding_dim: int = 16,
            context_embedding_dim: int = 12,
            slot_embedding_dim: int = 16,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.history_embeddings = nn.ModuleList(
                [nn.Embedding(size, history_embedding_dim, padding_idx=0) for size in history_vocab_sizes]
            )
            self.context_embeddings = nn.ModuleList(
                [nn.Embedding(size, context_embedding_dim, padding_idx=0) for size in context_vocab_sizes]
            )
            history_input_dim = len(history_vocab_sizes) * history_embedding_dim
            self.gru = nn.GRU(history_input_dim, hidden, batch_first=True)
            context_dim = len(context_vocab_sizes) * context_embedding_dim
            self.representation_dim = hidden + context_dim + 1
            self.behavior_trunk = nn.Sequential(
                nn.Linear(self.representation_dim, hidden), nn.ReLU(), nn.Dropout(dropout)
            )
            self.role_head = nn.Linear(hidden, len(ROLE_LABELS))
            self.family_head = nn.Linear(hidden, len(FAMILY_LABELS))
            self.structure_trunk = nn.Sequential(
                nn.Linear(self.representation_dim, hidden), nn.ReLU(), nn.Dropout(dropout)
            )
            self.layer_head = nn.Linear(hidden, HORIZON + 1)
            self.width_heads = nn.ModuleList([nn.Linear(hidden, MAX_WIDTH + 1) for _ in range(HORIZON)])
            self.slot_embedding = nn.Embedding(HORIZON * MAX_WIDTH, slot_embedding_dim)
            self.prototype_trunk = nn.Sequential(
                nn.Linear(self.representation_dim + slot_embedding_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.prototype_heads = nn.ModuleList([nn.Linear(hidden, size) for size in prototype_field_sizes])
            self.topology_role_head = nn.Linear(hidden, topology_role_size)
            self.topology_family_head = nn.Linear(hidden, len(FAMILY_LABELS))

        def encode(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
            tokens = batch["tokens"]
            lengths = batch["lengths"]
            history_channels = [embedding(tokens[..., i]) for i, embedding in enumerate(self.history_embeddings)]
            encoded_history = torch.cat(history_channels, dim=-1)
            valid = lengths > 0
            safe_lengths = lengths.clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                encoded_history,
                safe_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _packed_out, hidden = self.gru(packed)
            last = hidden[-1]
            last = torch.where(valid.unsqueeze(1), last, torch.zeros_like(last))
            context_channels = [embedding(batch["context"][:, i]) for i, embedding in enumerate(self.context_embeddings)]
            context = torch.cat(context_channels, dim=-1)
            length_feature = lengths.float().unsqueeze(1) / float(MAX_HISTORY)
            return torch.cat([last, context, length_feature], dim=1)

        def forward(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            representation = self.encode(batch)
            behavior_hidden = self.behavior_trunk(representation)
            structure_hidden = self.structure_trunk(representation)
            batch_size = representation.shape[0]
            slots = self.slot_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
            repeated = representation.unsqueeze(1).expand(-1, HORIZON * MAX_WIDTH, -1)
            prototype_hidden = self.prototype_trunk(torch.cat([repeated, slots], dim=-1))
            prototype_logits = [
                head(prototype_hidden).view(batch_size, HORIZON, MAX_WIDTH, -1)
                for head in self.prototype_heads
            ]
            return {
                "role_logits": self.role_head(behavior_hidden),
                "family_logits": self.family_head(behavior_hidden),
                "layer_logits": self.layer_head(structure_hidden),
                "width_logits": torch.stack([head(structure_hidden) for head in self.width_heads], dim=1),
                "prototype_logits": prototype_logits,
                "topology_role_logits": self.topology_role_head(prototype_hidden).view(
                    batch_size, HORIZON, MAX_WIDTH, -1
                ),
                "topology_family_logits": self.topology_family_head(prototype_hidden).view(
                    batch_size, HORIZON, MAX_WIDTH, -1
                ),
            }

else:
    SharedCausalGRU = None  # type: ignore[assignment,misc]


def _batch_from_tuple(values: Sequence[Any], device: Any) -> Dict[str, Any]:
    names = (
        "tokens",
        "lengths",
        "context",
        "behavior_role_targets",
        "behavior_family_targets",
        "behavior_family_mask",
        "layer_targets",
        "width_targets",
        "prototype_targets",
        "topology_role_targets",
        "topology_family_targets",
    )
    return {name: value.to(device) for name, value in zip(names, values)}


def _symmetric_kl(left: Any, right: Any) -> Any:
    left = left.clamp_min(NLL_FLOOR)
    right = right.clamp_min(NLL_FLOOR)
    return 0.5 * (
        (left * (left.log() - right.log())).sum(dim=-1)
        + (right * (right.log() - left.log())).sum(dim=-1)
    )


def _first_layer_marginal(
    outputs: Mapping[str, Any], field_index: int = -1,
    topology_role: bool = False, topology_family: bool = False,
) -> Any:
    width_probability = torch.softmax(outputs["width_logits"][:, 0], dim=-1)
    active = torch.stack(
        [width_probability[:, index + 1 :].sum(dim=-1) for index in range(MAX_WIDTH)], dim=1
    )
    if topology_role:
        logits = outputs["topology_role_logits"][:, 0]
    elif topology_family:
        logits = outputs["topology_family_logits"][:, 0]
    else:
        logits = outputs["prototype_logits"][field_index][:, 0]
    probabilities = torch.softmax(logits, dim=-1)
    denominator = active.sum(dim=1, keepdim=True).clamp_min(NLL_FLOOR)
    return (active.unsqueeze(-1) * probabilities).sum(dim=1) / denominator


def _role_consistency(outputs: Mapping[str, Any]) -> Any:
    behavior = torch.softmax(outputs["role_logits"], dim=-1)
    layer_probability = torch.softmax(outputs["layer_logits"], dim=-1)
    future_probability = layer_probability[:, 1:].sum(dim=-1)
    first_layer = _first_layer_marginal(outputs, -1, topology_role=True)
    topology = first_layer * future_probability.unsqueeze(1)
    topology = topology.clone()
    topology[:, TERMINATE_INDEX] = 1.0 - future_probability
    return _symmetric_kl(behavior, topology).mean()


def _family_consistency(outputs: Mapping[str, Any], mask: Any) -> Any:
    if not bool(mask.any()):
        return outputs["family_logits"].sum() * 0.0
    behavior = torch.softmax(outputs["family_logits"][mask], dim=-1)
    topology = _first_layer_marginal(outputs, topology_family=True)[mask]
    return _symmetric_kl(behavior, topology).mean()


def loss_components(outputs: Mapping[str, Any], batch: Mapping[str, Any], weights: Mapping[str, float]) -> Dict[str, Any]:
    role_loss = F.cross_entropy(outputs["role_logits"], batch["behavior_role_targets"])
    family_mask = batch["behavior_family_mask"].bool()
    if bool(family_mask.any()):
        family_loss = F.cross_entropy(
            outputs["family_logits"][family_mask], batch["behavior_family_targets"][family_mask]
        )
    else:
        family_loss = outputs["family_logits"].sum() * 0.0
    layer_loss = F.cross_entropy(outputs["layer_logits"], batch["layer_targets"])
    width_loss = torch.stack(
        [F.cross_entropy(outputs["width_logits"][:, index], batch["width_targets"][:, index]) for index in range(HORIZON)]
    ).mean()
    active_mask = (
        torch.arange(MAX_WIDTH, device=batch["width_targets"].device).view(1, 1, MAX_WIDTH)
        < batch["width_targets"].unsqueeze(-1)
    )
    prototype_losses = []
    for index, logits in enumerate(outputs["prototype_logits"]):
        if bool(active_mask.any()):
            prototype_losses.append(
                F.cross_entropy(
                    logits[active_mask], batch["prototype_targets"][..., index][active_mask]
                )
            )
        else:
            prototype_losses.append(logits.sum() * 0.0)
    prototype_loss = torch.stack(prototype_losses).mean()
    topology_role_mask = batch["topology_role_targets"] >= 0
    if bool(topology_role_mask.any()):
        topology_role_loss = F.cross_entropy(
            outputs["topology_role_logits"][topology_role_mask],
            batch["topology_role_targets"][topology_role_mask],
        )
    else:
        topology_role_loss = outputs["topology_role_logits"].sum() * 0.0
    topology_family_mask = batch["topology_family_targets"] >= 0
    if bool(topology_family_mask.any()):
        topology_family_loss = F.cross_entropy(
            outputs["topology_family_logits"][topology_family_mask],
            batch["topology_family_targets"][topology_family_mask],
        )
    else:
        topology_family_loss = outputs["topology_family_logits"].sum() * 0.0
    consistency_loss = _role_consistency(outputs) + _family_consistency(outputs, family_mask)
    total = (
        float(weights["role"]) * role_loss
        + float(weights["family"]) * family_loss
        + float(weights["layer"]) * layer_loss
        + float(weights["width"]) * width_loss
        + float(weights["prototype"]) * prototype_loss
        + float(weights.get("topology_role", 0.25)) * topology_role_loss
        + float(weights.get("topology_family", 0.25)) * topology_family_loss
        + float(weights["consistency"]) * consistency_loss
    )
    return {
        "total": total,
        "role": role_loss,
        "family": family_loss,
        "layer": layer_loss,
        "width": width_loss,
        "prototype": prototype_loss,
        "topology_role": topology_role_loss,
        "topology_family": topology_family_loss,
        "consistency": consistency_loss,
    }


def _bundle_to_loader(bundle: Mapping[str, np.ndarray], batch_size: int, shuffle: bool, seed: int, device: Any):
    tensors = [
        torch.from_numpy(bundle[name])
        for name in (
            "tokens",
            "lengths",
            "context",
            "behavior_role_targets",
            "behavior_family_targets",
            "behavior_family_mask",
            "layer_targets",
            "width_targets",
            "prototype_targets",
            "topology_role_targets",
            "topology_family_targets",
        )
    ]
    dataset = torch.utils.data.TensorDataset(*tensors)
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator
    )


def _predict_outputs(model: Any, bundle: Mapping[str, np.ndarray], batch_size: int, device: Any) -> Dict[str, Any]:
    model.eval()
    tensors = {
        name: torch.from_numpy(bundle[name]).to(device)
        for name in ("tokens", "lengths", "context")
    }
    chunks: Dict[str, List[Any]] = {}
    with torch.no_grad():
        for start in range(0, len(bundle["tokens"]), batch_size):
            batch = {name: value[start : start + batch_size] for name, value in tensors.items()}
            output = model(batch)
            for name, value in output.items():
                if isinstance(value, list):
                    for index, tensor in enumerate(value):
                        chunks.setdefault(f"{name}:{index}", []).append(tensor.detach().cpu())
                else:
                    chunks.setdefault(name, []).append(value.detach().cpu())
    result: Dict[str, Any] = {}
    for name, values in chunks.items():
        result[name] = torch.cat(values, dim=0)
    result["prototype_logits"] = [result.pop(f"prototype_logits:{index}") for index in range(len(PROTOTYPE_FIELDS))]
    return result


def _compact_metrics(outputs: Mapping[str, Any], bundle: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    layer_prediction = outputs["layer_logits"].argmax(dim=-1).numpy()
    width_prediction = outputs["width_logits"].argmax(dim=-1).numpy()
    true_layers = bundle["layer_targets"]
    true_widths = bundle["width_targets"]
    selected_widths = np.asarray(
        [[width_prediction[row, index] if index < layer_prediction[row] else 0 for index in range(HORIZON)] for row in range(len(layer_prediction))],
        dtype=np.int64,
    )
    true_nodes = true_widths.sum(axis=1)
    predicted_nodes = selected_widths.sum(axis=1)
    true_exists = true_layers > 0
    predicted_exists = layer_prediction > 0
    role_prediction = outputs["role_logits"].argmax(dim=-1).numpy()
    role_true = bundle["behavior_role_targets"]
    family_mask = bundle["behavior_family_mask"].astype(bool)
    family_prediction = outputs["family_logits"].argmax(dim=-1).numpy()
    family_true = bundle["behavior_family_targets"]
    role_correct = role_prediction == role_true
    family_correct = np.zeros(len(family_mask), dtype=bool)
    family_correct[family_mask] = family_prediction[family_mask] == family_true[family_mask]
    joint_correct = role_correct & (~family_mask | family_correct)
    return {
        "sample_count": int(len(layer_prediction)),
        "layer_count_mae": float(np.abs(layer_prediction - true_layers).mean()),
        "layer_count_bias": float((layer_prediction - true_layers).mean()),
        "node_count_mae": float(np.abs(predicted_nodes - true_nodes).mean()),
        "node_count_bias": float((predicted_nodes - true_nodes).mean()),
        "width_vector_mae": float(np.abs(selected_widths - true_widths).mean()),
        "top1_exact_shape_coverage": float(
            np.mean(
                (layer_prediction == true_layers)
                & np.all(selected_widths == true_widths, axis=1)
            )
        ),
        "future_exists_accuracy": float(np.mean(predicted_exists == true_exists)),
        "behavior_role_accuracy": float(role_correct.mean()),
        "behavior_family_accuracy": float(family_correct[family_mask].mean()) if bool(family_mask.any()) else 0.0,
        "behavior_family_coverage": float(family_mask.mean()),
        "behavior_joint_accuracy": float(joint_correct.mean()),
    }


def _top_choices(probabilities: np.ndarray, positive_only: bool = False, limit: int = 3) -> List[Tuple[int, float]]:
    choices = [
        (index, max(float(probability), NLL_FLOOR))
        for index, probability in enumerate(probabilities)
        if not positive_only or index > 0
    ]
    return sorted(choices, key=lambda item: (-item[1], item[0]))[:limit]


def _shape_scenarios(outputs: Mapping[str, Any], row_index: int, codec: TargetCodec) -> List[Tuple[float, int, Tuple[int, ...]]]:
    layer_probability = torch.softmax(outputs["layer_logits"][row_index], dim=-1).numpy()
    width_probability = torch.softmax(outputs["width_logits"][row_index], dim=-1).numpy()
    beams: List[Tuple[float, int, Tuple[int, ...]]] = [
        (probability, layer_count, tuple())
        for layer_count, probability in _top_choices(layer_probability)
        if layer_count <= HORIZON
    ]
    for layer_index in range(HORIZON):
        expanded: List[Tuple[float, int, Tuple[int, ...]]] = []
        for probability, layer_count, widths in beams:
            if layer_index >= layer_count:
                expanded.append((probability, layer_count, widths + (0,)))
            else:
                for width, width_prob in _top_choices(width_probability[layer_index], positive_only=True):
                    expanded.append((probability * width_prob, layer_count, widths + (width,)))
        beams = sorted(expanded, key=lambda item: (-item[0], item[1], item[2]))[:12]
    unique: Dict[Tuple[int, Tuple[int, ...]], float] = {}
    for probability, layer_count, widths in beams:
        unique[(layer_count, widths)] = max(unique.get((layer_count, widths), 0.0), probability)
    return [
        (probability, layer_count, widths)
        for (layer_count, widths), probability in sorted(
            unique.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        )[:MAX_SCENARIOS]
    ]


def _scenario_json(
    outputs: Mapping[str, Any], row_index: int, codec: TargetCodec,
    layer_count: int, widths: Sequence[int], probability: float, rank: int,
) -> Dict[str, Any]:
    layers: List[Dict[str, Any]] = []
    prototype_indices = [logits[row_index].argmax(dim=-1).numpy() for logits in outputs["prototype_logits"]]
    for layer_index in range(layer_count):
        nodes: List[Dict[str, Any]] = []
        for node_index in range(int(widths[layer_index])):
            node = {
                field: codec.decode(field, int(prototype_indices[field_index][layer_index, node_index]))
                for field_index, field in enumerate(PROTOTYPE_FIELDS)
            }
            node.update({"layer_offset": layer_index + 1, "predicted_node_index": node_index, "prototype_source": "shared_causal_gru"})
            nodes.append(node)
        layers.append({"layer_offset": layer_index + 1, "nodes": nodes})
    return {
        "scenario_id": f"shared_gru_top{rank:02d}",
        "scenario_probability": float(probability),
        "layers": layers,
        "synthetic_rollout": True,
        "topology_source": "shared_causal_gru_structure_and_prototype_heads",
    }


def _predicted_signature(scenario: Mapping[str, Any]) -> Signature:
    return tuple(
        tuple(
            sorted(tuple(_canonical(node.get(field)) for field in PROTOTYPE_FIELDS) for node in layer.get("nodes") or [])
        )
        for layer in scenario.get("layers") or []
    )


def evaluate_split(
    model: Any,
    pairs: Sequence[Pair],
    bundle: Mapping[str, np.ndarray],
    codec: TargetCodec,
    batch_size: int,
    device: Any,
    include_topology: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    outputs = _predict_outputs(model, bundle, batch_size, device)
    metrics = _compact_metrics(outputs, bundle)
    if not include_topology:
        metrics.update(
            {
                "topology_metrics_valid": False,
                "top1_exact_signature_coverage": None,
                "top3_exact_signature_coverage": None,
                "prototype_field_accuracy": None,
            }
        )
        return metrics, []
    top1_signature = 0
    top3_signature = 0
    prototype_correct = 0
    prototype_total = 0
    rows: List[Dict[str, Any]] = []
    for row_index, (feature, label) in enumerate(pairs):
        true_signature = topology_signature(label)
        decoded = _shape_scenarios(outputs, row_index, codec)
        raw_total = sum(item[0] for item in decoded)
        scenarios = [
            _scenario_json(outputs, row_index, codec, layer_count, widths, probability / raw_total, rank)
            for rank, (probability, layer_count, widths) in enumerate(decoded)
        ]
        predicted_signature = _predicted_signature(scenarios[0])
        signatures = [_predicted_signature(scenario) for scenario in scenarios]
        top1_signature += int(predicted_signature == true_signature)
        top3_signature += int(true_signature in signatures)
        for layer_index, layer in enumerate(true_signature):
            for node_index, true_node in enumerate(layer):
                if layer_index < len(predicted_signature) and node_index < len(predicted_signature[layer_index]):
                    prototype_total += len(PROTOTYPE_FIELDS)
                    prototype_correct += sum(
                        predicted_signature[layer_index][node_index][field_index] == value
                        for field_index, value in enumerate(true_node)
                    )
        rows.append(
            {
                "sample_id": feature["sample_id"],
                "split": feature["split"],
                "top_scenarios": scenarios,
                "true_summary": {
                    "future_layer_count": len(true_signature),
                    "future_node_count": sum(len(layer) for layer in true_signature),
                    "layer_widths": [len(layer) for layer in true_signature],
                    "first_layer_width": len(true_signature[0]) if true_signature else 0,
                    "has_future": bool(true_signature),
                },
                "true_signature_hash": hashlib.sha256(
                    json.dumps(true_signature, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "prediction_contract": {
                    "identity_free": True,
                    "true_node_ids_emitted": False,
                    "true_edges_emitted": False,
                    "resource_truth_emitted": False,
                    "structure_probability": "factorized_layer_count_and_active_widths",
                },
            }
        )
    metrics.update(
        {
            "topology_metrics_valid": True,
            "top1_exact_signature_coverage": top1_signature / len(pairs),
            "top3_exact_signature_coverage": top3_signature / len(pairs),
            "prototype_field_accuracy": prototype_correct / prototype_total if prototype_total else 0.0,
            "prototype_field_observations": prototype_total,
        }
    )
    return metrics, rows


def _cpu_state_dict(model: Any) -> Dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _train_one(
    variant: Mapping[str, Any],
    seed: int,
    encoder: InputEncoder,
    codec: TargetCodec,
    train_bundle: Mapping[str, np.ndarray],
    validation_bundle: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    run_root: Path,
) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError(f"PyTorch is required for formal training: {TORCH_IMPORT_ERROR}")
    device = torch.device(str(config.get("device", "cuda:0")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configured CUDA device is unavailable")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    history_vocab_sizes, context_vocab_sizes = encoder.vocab_sizes()
    model = SharedCausalGRU(
        history_vocab_sizes,
        context_vocab_sizes,
        codec.field_sizes(),
        codec.role_size(),
        hidden=int(config["hidden"]),
        history_embedding_dim=int(config["history_embedding_dim"]),
        context_embedding_dim=int(config["context_embedding_dim"]),
        slot_embedding_dim=int(config["slot_embedding_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    loader = _bundle_to_loader(
        train_bundle, int(config["batch_size"]), True, seed + 1000, device
    )
    best_score = math.inf
    best_epoch = 0
    best_state: Optional[Dict[str, Any]] = None
    best_train_losses: Optional[Dict[str, float]] = None
    history: List[Dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        sums = Counter()
        rows_seen = 0
        for values in loader:
            batch = _batch_from_tuple(values, device)
            optimizer.zero_grad(set_to_none=True)
            losses = loss_components(outputs := model(batch), batch, variant["loss_weights"])
            losses["total"].backward()
            optimizer.step()
            count = int(batch["tokens"].shape[0])
            rows_seen += count
            for key, value in losses.items():
                sums[key] += float(value.detach().cpu()) * count
        validation_outputs = _predict_outputs(model, validation_bundle, int(config["batch_size"]), device)
        validation_metrics = _compact_metrics(validation_outputs, validation_bundle)
        if variant["name"] == "behavior_only":
            score = -(
                validation_metrics["behavior_role_accuracy"]
                + validation_metrics["behavior_family_accuracy"]
            )
        else:
            score = validation_metrics["layer_count_mae"] + validation_metrics["width_vector_mae"]
        record = {
            "epoch": epoch,
            "selection_score": float(score),
            "validation": validation_metrics,
            "train_loss": {key: value / rows_seen for key, value in sums.items()},
        }
        history.append(record)
        if score < best_score - 1e-12:
            best_score = score
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
            best_train_losses = record["train_loss"]
        if epoch - best_epoch >= int(config["patience"]):
            break
    if best_state is None or best_train_losses is None:
        raise RuntimeError("no validation checkpoint was produced")
    model.load_state_dict(best_state)
    run_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_root / "checkpoint.pt"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "variant": variant["name"],
            "seed": seed,
            "model_state_dict": best_state,
            "history_vocab_sizes": history_vocab_sizes,
            "context_vocab_sizes": context_vocab_sizes,
            "prototype_field_sizes": codec.field_sizes(),
            "topology_role_size": codec.role_size(),
        },
        checkpoint_path,
    )
    validation_metrics, validation_rows = evaluate_split(
        model, config["pairs"]["validation"], validation_bundle, codec,
        int(config["batch_size"]), device, include_topology=variant["name"] != "behavior_only"
    )
    test_metrics, test_rows = evaluate_split(
        model, config["pairs"]["test"], config["bundles"]["test"], codec,
        int(config["batch_size"]), device, include_topology=variant["name"] != "behavior_only"
    )
    holdout_metrics, holdout_rows = evaluate_split(
        model, config["pairs"]["holdout"], config["bundles"]["holdout"], codec,
        int(config["batch_size"]), device, include_topology=variant["name"] != "behavior_only"
    )
    paths = {
        "validation": run_root / "predictions_validation.jsonl.gz",
        "test": run_root / "predictions_test.jsonl.gz",
        "holdout": run_root / "predictions_holdout.jsonl.gz",
    }
    counts = {
        split: _write_jsonl_gz(paths[split], rows)
        for split, rows in (("validation", validation_rows), ("test", test_rows), ("holdout", holdout_rows))
    }
    run_metrics = {
        "variant": variant["name"],
        "seed": seed,
        "best_epoch": best_epoch,
        "best_selection_score": float(best_score),
        "best_train_losses": best_train_losses,
        "validation": validation_metrics,
        "test": test_metrics,
        "holdout": holdout_metrics,
        "prediction_rows": counts,
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(checkpoint_path),
    }
    _write_json(run_root / "metrics.json", run_metrics)
    _write_json(run_root / "epoch_history.json", {"epochs": history})
    return run_metrics


def _aggregate(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"seed_count": len(runs), "seeds": [int(run["seed"]) for run in runs]}
    for split in ("validation", "test", "holdout"):
        metrics = [run[split] for run in runs]
        keys = sorted({key for metric in metrics for key, value in metric.items() if isinstance(value, (int, float))})
        result[split] = {}
        for key in keys:
            values = np.asarray([float(metric[key]) for metric in metrics if metric.get(key) is not None], dtype=float)
            if len(values):
                result[split][key] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    return result


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected shared GRU config schema")
    if int(config.get("horizon", -1)) != HORIZON or int(config.get("max_width", -1)) != MAX_WIDTH:
        raise ValueError("shared GRU topology dimensions must be H=5 and width=5")
    if tuple(config.get("variants") or ()) != tuple(
        ["behavior_only", "topology_only", "shared_multitask", "shared_first_layer_consistency"]
    ):
        raise ValueError("variants must remain in the pre-registered order")
    if tuple(config.get("seeds") or ()) != (11, 22, 33):
        raise ValueError("formal shared GRU seeds are locked to 11/22/33")
    boundary = config.get("data_boundary") or {}
    if boundary.get("fit") != "P_dev/train" or boundary.get("selection") != "P_dev/validation":
        raise ValueError("fit/selection split contract mismatch")
    if boundary.get("diagnostic") != "P_dev/test" or boundary.get("frozen_acceptance") != "P_holdout_diag/holdout":
        raise ValueError("diagnostic/holdout split contract mismatch")
    if boundary.get("scheduler_groups_used") or boundary.get("t_final_read"):
        raise ValueError("scheduler groups and T_final must remain excluded")
    if boundary.get("t_final_sealed") is not True:
        raise ValueError("T_final must remain sealed")


def run(config_path: Path) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError(f"PyTorch is required for the formal run: {TORCH_IMPORT_ERROR}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    dataset_root = Path(str(config["dataset_root"]))
    experiment_root = Path(str(config["experiment_root"]))
    artifact_root = experiment_root / "artifacts"
    if artifact_root.exists() and any(artifact_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty artifact directory: {artifact_root}")
    if (experiment_root / "metrics.json").exists():
        raise FileExistsError(f"refusing to overwrite metrics: {experiment_root / 'metrics.json'}")
    manifest_path = dataset_root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("wrong P9d dataset schema")
    if (manifest.get("source") or {}).get("scheduler_trace_groups_used_for_fit") is not False:
        raise ValueError("dataset scheduler fit exclusion is not proven")
    if (manifest.get("source") or {}).get("s_train_s_val_t_final_used") is not False:
        raise ValueError("dataset scheduler/T_final exclusion is not proven")
    expected_dataset_hash = config.get("dataset_manifest_sha256")
    actual_dataset_hash = sha256_file(manifest_path)
    if expected_dataset_hash and expected_dataset_hash != actual_dataset_hash:
        raise ValueError("P9d dataset manifest hash mismatch")
    pairs = {split: _read_split(dataset_root, split) for split in ("train", "validation", "test", "holdout")}
    behavior, behavior_audit = join_behavior_targets(
        pairs,
        Path(config["p_dev_role_samples"]),
        Path(config["p_holdout_role_samples"]),
        Path(config["p_dev_family_samples"]),
        Path(config["p_holdout_family_samples"]),
    )
    encoder = InputEncoder(int(config["max_history"])).fit(pairs["train"])
    codec = TargetCodec().fit(pairs["train"])
    bundles: Dict[str, Dict[str, np.ndarray]] = {}
    for split in pairs:
        encoded = encoder.transform(pairs[split])
        encoded.update(build_target_arrays(pairs[split], behavior[split], codec))
        bundles[split] = encoded
    run_config = dict(config)
    run_config["pairs"] = pairs
    run_config["bundles"] = bundles
    variant_configs = config["variant_configs"]
    variant_runs: Dict[str, List[Dict[str, Any]]] = {}
    started = time.time()
    for variant_name in config["variants"]:
        variant = dict(variant_configs[variant_name])
        variant["name"] = variant_name
        variant_runs[variant_name] = []
        for seed in config["seeds"]:
            variant_runs[variant_name].append(
                _train_one(
                    variant,
                    int(seed),
                    encoder,
                    codec,
                    bundles["train"],
                    bundles["validation"],
                    run_config,
                    artifact_root / variant_name / f"seed_{seed}",
                )
            )
    aggregates = {name: _aggregate(runs) for name, runs in variant_runs.items()}
    topology_candidates = [name for name in config["variants"] if name != "behavior_only"]
    selected_variant = min(
        topology_candidates,
        key=lambda name: (
            aggregates[name]["validation"]["layer_count_mae"]["mean"]
            + aggregates[name]["validation"]["width_vector_mae"]["mean"],
            name,
        ),
    )
    output_hashes: Dict[str, str] = {}
    for path in artifact_root.rglob("*"):
        if path.is_file():
            output_hashes[str(path.relative_to(experiment_root))] = sha256_file(path)
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "status": "passed_shared_gru_experiment",
        "fit_contract": {
            "fit_split": "P_dev/train",
            "selection_split": "P_dev/validation",
            "diagnostic_split": "P_dev/test",
            "frozen_holdout_split": "P_holdout_diag/holdout",
            "seeds": list(config["seeds"]),
            "variants": list(config["variants"]),
            "selected_variant_by_validation_structure_score": selected_variant,
        },
        "data_contract": {
            "dataset_manifest_sha256": actual_dataset_hash,
            "split_counts": manifest.get("counts", {}),
            "behavior_join": behavior_audit,
            "upstream_behavior_or_resource_predictions_used_as_feature": False,
            "future_events_or_edges_used_as_feature": False,
            "resource_truth_used_as_feature": False,
            "video_id_used_as_feature": False,
        },
        "model_contract": {
            "encoder": "single_unidirectional_causal_GRU",
            "behavior_heads": ["next_role", "execute_gated_family"],
            "topology_structure_heads": ["layer_count"] + [f"width_{i + 1}" for i in range(HORIZON)],
            "topology_prototype_heads": list(PROTOTYPE_FIELDS),
            "topology_auxiliary_consistency_role_head": True,
            "output": "H=5 top-3 identity-free DAG layers with multi-node widths",
            "node_ids_or_edges_emitted": False,
        },
        "variant_runs": variant_runs,
        "aggregate": aggregates,
        "boundary": {
            "scheduler_groups_used_for_fit": False,
            "scheduler_integration_started": False,
            "t_final_read": False,
            "raw_traces_modified": False,
        },
        "reproducibility": {
            "command": [sys.executable] + sys.argv,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(config["device"]),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "elapsed_seconds": time.time() - started,
        },
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "dataset_manifest_sha256": actual_dataset_hash,
            "p_dev_role_samples_sha256": sha256_file(Path(config["p_dev_role_samples"])),
            "p_holdout_role_samples_sha256": sha256_file(Path(config["p_holdout_role_samples"])),
            "p_dev_family_samples_sha256": sha256_file(Path(config["p_dev_family_samples"])),
            "p_holdout_family_samples_sha256": sha256_file(Path(config["p_holdout_family_samples"])),
        },
    }
    experiment_root.mkdir(parents=True, exist_ok=True)
    _write_json(experiment_root / "metrics.json", metrics)
    manifest_output = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "status": metrics["status"],
        "source_sha256": {
            "scripts/p9d_shared_causal_gru.py": sha256_file(Path(__file__)),
            "tests/test_p9d_shared_causal_gru.py": sha256_file(Path(config["test_path"])),
            "experiments/config.json": sha256_file(config_path),
            "experiments/run.sh": sha256_file(Path(config["run_script_path"])),
        },
        "inputs": metrics["inputs"],
        "outputs_sha256": {
            **output_hashes,
            "metrics.json": sha256_file(experiment_root / "metrics.json"),
        },
        "verification": {
            "scheduler_groups_used_for_fit": False,
            "t_final_read": False,
            "raw_traces_modified": False,
            "prediction_contract": "identity_free_topology_only",
        },
    }
    _write_json(experiment_root / "run_manifest.json", manifest_output)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps({"status": result["status"], "selected_variant": result["fit_contract"]["selected_variant_by_validation_structure_score"]}, sort_keys=True))


if __name__ == "__main__":
    main()
