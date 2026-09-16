#!/usr/bin/env python3
"""Train the P9d future-node content multi-task predictor.

This is the first implementation of the approved ``N0/A/B/D`` experiment:

* ``N0`` is the next-step behavior-only reference.
* ``A`` predicts the future layer profile and all future-node content.
* ``B`` adds the next-step behavior auxiliary loss.
* ``D`` adds one soft topology-probability conditioning path to the content
  decoder.

The model consumes only the visible causal prefix and task/stack context.  It
predicts an identity-free leveled DAG profile plus unordered per-layer node
content.  Each active layer uses fixed slots and one joint bipartite matching
for all content fields.  Edges, node IDs, execution truth, resource truth and
scheduler labels are never model inputs.

The formal protocol is fixed-split P9d for this run: P_dev/train is fit,
P_dev/validation selects the epoch, P_dev/test is diagnostic and
P_holdout_diag/holdout is opened only after each run is frozen.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import platform
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.p9d_shared_causal_gru import (  # noqa: E402
    CONTEXT_FIELDS,
    DATASET_SCHEMA_VERSION,
    FAMILY_INDEX,
    FAMILY_LABELS,
    HISTORY_FIELDS,
    InputEncoder,
    MAX_HISTORY,
    MAX_WIDTH,
    NLL_FLOOR,
    ROLE_INDEX,
    ROLE_LABELS,
    HORIZON,
    _canonical,
    _read_split,
    audit_model_input,
    join_behavior_targets,
    sha256_file,
)

try:  # Local macOS has no torch; the formal fit is remote.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - exercised by local tests
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = str(exc)
else:
    TORCH_IMPORT_ERROR = None


SCHEMA_VERSION = "p9d-future-content-multitask-v1"
CONTENT_FIELDS = ("node_type", "raw_action", "model_id", "action_family")
VARIANTS = ("N0", "A", "B", "D")
MISSING_NODE = -1

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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def _content_signature(label: Mapping[str, Any]) -> Signature:
    layers = label.get("future_layers") or []
    result: List[Tuple[Tuple[str, ...], ...]] = []
    for offset, layer in enumerate(layers, 1):
        if int(layer.get("layer_offset", -1)) != offset:
            raise ValueError("future layer offsets must be contiguous")
        nodes = layer.get("nodes") or []
        result.append(
            tuple(
                sorted(
                    tuple(_canonical(node.get(field)) for field in CONTENT_FIELDS)
                    for node in nodes
                )
            )
        )
    return tuple(result)


def _direct_identifier_overlap(value: str, literals: Iterable[str]) -> bool:
    candidate = str(value)
    if not candidate or candidate == "unknown":
        return False
    return any(literal and (candidate == literal or literal in candidate) for literal in literals)


def stage0_audit(dataset_root: Path, expected_manifest_sha256: Optional[str] = None) -> Dict[str, Any]:
    """Run non-training target, split and leakage hard gates."""

    errors: List[str] = []
    manifest_path = dataset_root / "dataset_manifest.json"
    if not manifest_path.exists():
        return {"status": "failed", "passed": False, "errors": [f"missing {manifest_path}"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_manifest_sha256 = sha256_file(manifest_path)
    if expected_manifest_sha256 and actual_manifest_sha256 != expected_manifest_sha256:
        errors.append("dataset manifest SHA-256 mismatch")
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        errors.append("unexpected P9d dataset schema")
    source = manifest.get("source") or {}
    if source.get("scheduler_trace_groups_used_for_fit") is not False:
        errors.append("scheduler trace-group exclusion is not proven")
    if source.get("s_train_s_val_t_final_used") is not False:
        errors.append("S_train/S_val/T_final exclusion is not proven")

    split_rows: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {}
    video_sets: Dict[str, Set[str]] = {}
    run_sets: Dict[str, Set[str]] = {}
    source_trace_sets: Dict[str, Set[str]] = {}
    field_values: Dict[str, Set[str]] = {field: set() for field in CONTENT_FIELDS}
    layer_counts: Counter = Counter()
    width_counts: Counter = Counter()
    truncation_counts: Counter = Counter()
    duplicate_content_layers = 0
    duplicate_content_nodes = 0
    direct_identifier_overlaps = 0
    missing_content_fields = 0
    invalid_layer_offsets = 0
    duplicate_node_ids = 0
    input_violations = 0
    sample_id_violations = 0

    for split in ("train", "validation", "test", "holdout"):
        feature_path = dataset_root / f"features_{split}.jsonl.gz"
        label_path = dataset_root / f"labels_{split}.jsonl.gz"
        if not feature_path.exists() or not label_path.exists():
            errors.append(f"missing feature/label file for {split}")
            continue
        features = _read_jsonl(feature_path)
        labels = _read_jsonl(label_path)
        split_rows[split] = (features, labels)
        if len(features) != len(labels):
            errors.append(f"feature/label count mismatch for {split}")
            continue
        seen_samples: Set[str] = set()
        video_sets[split] = set()
        run_sets[split] = set()
        source_trace_sets[split] = set()
        for feature, label in zip(features, labels):
            sample_id = str(feature.get("sample_id"))
            if (
                not sample_id
                or sample_id in seen_samples
                or sample_id != str(label.get("sample_id"))
                or feature.get("split") != split
                or label.get("split") != split
            ):
                sample_id_violations += 1
            seen_samples.add(sample_id)
            video_id = str(feature.get("video_id"))
            run_id = str(feature.get("run_id"))
            video_sets[split].add(video_id)
            run_sets[split].add(run_id)
            trace_hash = str(label.get("source_trace_sha256") or "")
            if trace_hash:
                source_trace_sets[split].add(trace_hash)
            model_input = feature.get("model_input")
            if not isinstance(model_input, Mapping) or audit_model_input(model_input):
                input_violations += 1

            layers = label.get("future_layers")
            if not isinstance(layers, list) or len(layers) > HORIZON:
                invalid_layer_offsets += 1
                continue
            summary = label.get("label_summary") or {}
            if not isinstance(summary.get("truncated_at_horizon"), bool):
                errors.append("termination/truncation flag missing or non-boolean")
            else:
                truncation_counts[str(bool(summary["truncated_at_horizon"]))] += 1
            if int(label.get("future_horizon", -1)) != HORIZON:
                errors.append("future horizon is not H=5")
            if (label.get("label_contract") or {}).get("future_unit") != "event_level_dag_bfs_layer":
                errors.append("future layer unit is not canonical event-level BFS layer")
            if (label.get("label_contract") or {}).get("parent_source") != "raw_trace.parent_step_ids":
                errors.append("future parent source is not raw trace parent_step_ids")
            expected_offsets = list(range(1, len(layers) + 1))
            actual_offsets = [int(layer.get("layer_offset", -1)) for layer in layers]
            if actual_offsets != expected_offsets:
                invalid_layer_offsets += 1
            layer_counts[len(layers)] += 1
            node_ids: Set[str] = set()
            literals = [video_id, run_id, str(feature.get("current_node_id") or ""), trace_hash]
            for layer in layers:
                nodes = layer.get("nodes") or []
                width_counts[len(nodes)] += 1
                tuples: List[Tuple[str, ...]] = []
                for node in nodes:
                    node_id = str(node.get("node_id") or "")
                    if node_id and node_id in node_ids:
                        duplicate_node_ids += 1
                    node_ids.add(node_id)
                    tuples.append(tuple(_canonical(node.get(field)) for field in CONTENT_FIELDS))
                    for field in CONTENT_FIELDS:
                        if field not in node or node.get(field) in (None, ""):
                            missing_content_fields += 1
                        field_values[field].add(_canonical(node.get(field)))
                    raw_action = _canonical(node.get("raw_action"))
                    if _direct_identifier_overlap(raw_action, literals):
                        direct_identifier_overlaps += 1
                duplicate_counts = Counter(tuples)
                repeated = sum(max(count - 1, 0) for count in duplicate_counts.values())
                if repeated:
                    duplicate_content_layers += 1
                    duplicate_content_nodes += repeated

    for left, right in itertools.combinations(video_sets, 2):
        overlap = sorted(video_sets[left] & video_sets[right])
        if overlap:
            errors.append(f"video split overlap {left}/{right}: {overlap[:5]}")
    for left, right in itertools.combinations(source_trace_sets, 2):
        overlap = sorted(source_trace_sets[left] & source_trace_sets[right])
        if overlap:
            errors.append(f"source-trace split overlap {left}/{right}: {overlap[:3]}")
    if sample_id_violations:
        errors.append(f"sample-id/split violations: {sample_id_violations}")
    if input_violations:
        errors.append(f"causal model-input violations: {input_violations}")
    if missing_content_fields:
        errors.append(f"missing future-node content fields: {missing_content_fields}")
    if invalid_layer_offsets:
        errors.append(f"invalid/non-contiguous layer labels: {invalid_layer_offsets}")
    if duplicate_node_ids:
        errors.append(f"duplicate node IDs inside labels: {duplicate_node_ids}")
    if direct_identifier_overlaps:
        errors.append(f"raw_action direct identifier overlaps: {direct_identifier_overlaps}")
    if not all(split in split_rows for split in ("train", "validation", "test", "holdout")):
        errors.append("not all four fixed P9d splits are available")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "errors": errors,
        "dataset_manifest_sha256": actual_manifest_sha256,
        "split_counts": {
            split: {
                "rows": len(rows[0]),
                "videos": len(video_sets.get(split, set())),
                "runs": len(run_sets.get(split, set())),
            }
            for split, rows in split_rows.items()
        },
        "split_video_intersections": {
            f"{left}:{right}": len(video_sets.get(left, set()) & video_sets.get(right, set()))
            for left, right in itertools.combinations(("train", "validation", "test", "holdout"), 2)
        },
        "split_source_trace_intersections": {
            f"{left}:{right}": len(source_trace_sets.get(left, set()) & source_trace_sets.get(right, set()))
            for left, right in itertools.combinations(("train", "validation", "test", "holdout"), 2)
        },
        "layerization": {
            "future_unit": "event_level_dag_bfs_layer",
            "parent_source": "raw_trace.parent_step_ids",
            "layer_count_histogram": dict(sorted(layer_counts.items())),
            "width_histogram": dict(sorted(width_counts.items())),
            "invalid_or_noncontiguous_labels": invalid_layer_offsets,
            "duplicate_node_ids": duplicate_node_ids,
        },
        "termination_truncation": {
            "flag_field": "label_summary.truncated_at_horizon",
            "counts": dict(truncation_counts),
            "explicit_flag_for_all_rows": not any(
                "termination/truncation flag missing or non-boolean" in error for error in errors
            ),
        },
        "raw_action_ontology": {
            "unique_values_by_field": {field: len(values) for field, values in field_values.items()},
            "raw_action_values": sorted(field_values["raw_action"]),
            "direct_identifier_overlaps": direct_identifier_overlaps,
            "video_or_run_literal_leakage_gate": direct_identifier_overlaps == 0,
        },
        "content_collision_audit": {
            "duplicate_content_layers": duplicate_content_layers,
            "duplicate_content_nodes_after_first": duplicate_content_nodes,
            "matching_policy": "one_joint_layer_assignment_reused_across_all_content_fields",
            "collision_is_not_silently_deduplicated": True,
        },
        "feature_contract": {
            "forbidden_model_input_violations": input_violations,
            "future_events_edges_and_resource_truth_excluded": input_violations == 0,
        },
        "policy": {
            "scheduler_groups_used_for_fit": False,
            "t_final_read": False,
            "raw_traces_modified": False,
        },
    }


class ContentCodec:
    """Train-only vocabularies and dense future-node content targets."""

    def __init__(self, fields: Sequence[str] = CONTENT_FIELDS) -> None:
        self.fields = tuple(fields)
        self.field_maps: Dict[str, Dict[str, int]] = {}

    def fit(self, pairs: Sequence[Pair]) -> "ContentCodec":
        values = {field: [] for field in self.fields}
        for _feature, label in pairs:
            for layer in label.get("future_layers") or []:
                for node in layer.get("nodes") or []:
                    for field in self.fields:
                        values[field].append(_canonical(node.get(field)))
        self.field_maps = {field: InputEncoder._map(items) for field, items in values.items()}
        return self

    def transform(self, pairs: Sequence[Pair]) -> Dict[str, np.ndarray]:
        if not self.field_maps:
            raise ValueError("ContentCodec must be fit on train pairs first")
        layer_targets = np.zeros(len(pairs), dtype=np.int64)
        width_targets = np.zeros((len(pairs), HORIZON), dtype=np.int64)
        content_targets = np.full(
            (len(pairs), HORIZON, MAX_WIDTH, len(self.fields)), -1, dtype=np.int64
        )
        for row_index, (_feature, label) in enumerate(pairs):
            layers = label.get("future_layers") or []
            layer_targets[row_index] = len(layers)
            for layer_index, layer in enumerate(layers):
                nodes = layer.get("nodes") or []
                width_targets[row_index, layer_index] = len(nodes)
                for node_index, node in enumerate(nodes):
                    for field_index, field in enumerate(self.fields):
                        content_targets[row_index, layer_index, node_index, field_index] = self.field_maps[
                            field
                        ].get(_canonical(node.get(field)), 1)
        return {
            "layer_targets": layer_targets,
            "width_targets": width_targets,
            "content_targets": content_targets,
        }

    def field_sizes(self) -> List[int]:
        return [len(self.field_maps[field]) for field in self.fields]

    def decode(self, field: str, index: int) -> str:
        mapping = self.field_maps[field]
        inverse = {value: key for key, value in mapping.items()}
        return inverse.get(int(index), "__UNK__")


def build_target_arrays(
    pairs: Sequence[Pair], behavior_targets: Sequence[Mapping[str, Any]], codec: ContentCodec
) -> Dict[str, np.ndarray]:
    if len(pairs) != len(behavior_targets):
        raise ValueError("behavior target count mismatch")
    result = codec.transform(pairs)
    result["behavior_role_targets"] = np.asarray(
        [ROLE_INDEX[str(row["role"])] for row in behavior_targets], dtype=np.int64
    )
    result["behavior_family_targets"] = np.asarray(
        [
            FAMILY_INDEX.get(str(row["family"]), -1) if row["family"] is not None else -1
            for row in behavior_targets
        ],
        dtype=np.int64,
    )
    result["behavior_family_mask"] = np.asarray(
        [bool(row["family_mask"]) for row in behavior_targets], dtype=bool
    )
    return result


if torch is not None:

    class FutureContentGRU(nn.Module):
        def __init__(
            self,
            history_vocab_sizes: Sequence[int],
            context_vocab_sizes: Sequence[int],
            content_field_sizes: Sequence[int],
            *,
            hidden: int = 128,
            history_embedding_dim: int = 16,
            context_embedding_dim: int = 12,
            slot_embedding_dim: int = 16,
            dropout: float = 0.1,
            use_soft_topology_conditioning: bool = False,
        ) -> None:
            super().__init__()
            self.use_soft_topology_conditioning = bool(use_soft_topology_conditioning)
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
            # Width classes are 1..MAX_WIDTH; inactive layers are masked by L.
            self.width_heads = nn.ModuleList([nn.Linear(hidden, MAX_WIDTH) for _ in range(HORIZON)])

            self.slot_embedding = nn.Embedding(HORIZON * MAX_WIDTH, slot_embedding_dim)
            condition_dim = 32 if self.use_soft_topology_conditioning else 0
            self.condition_dim = condition_dim
            if self.use_soft_topology_conditioning:
                self.topology_condition_trunk = nn.Sequential(
                    nn.Linear((HORIZON + 1) + HORIZON * MAX_WIDTH, condition_dim),
                    nn.ReLU(),
                )
            self.content_trunk = nn.Sequential(
                nn.Linear(self.representation_dim + slot_embedding_dim + condition_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.content_heads = nn.ModuleList(
                [nn.Linear(hidden, size) for size in content_field_sizes]
            )

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
            context_channels = [
                embedding(batch["context"][:, i]) for i, embedding in enumerate(self.context_embeddings)
            ]
            context = torch.cat(context_channels, dim=-1)
            length_feature = lengths.float().unsqueeze(1) / float(MAX_HISTORY)
            return torch.cat([last, context, length_feature], dim=1)

        def forward(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
            representation = self.encode(batch)
            behavior_hidden = self.behavior_trunk(representation)
            structure_hidden = self.structure_trunk(representation)
            layer_logits = self.layer_head(structure_hidden)
            width_logits = torch.stack(
                [head(structure_hidden) for head in self.width_heads], dim=1
            )

            batch_size = representation.shape[0]
            slots = self.slot_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
            repeated = representation.unsqueeze(1).expand(-1, HORIZON * MAX_WIDTH, -1)
            if self.use_soft_topology_conditioning:
                layer_probability = torch.softmax(layer_logits, dim=-1)
                width_probability = torch.softmax(width_logits, dim=-1)
                topology_features = torch.cat(
                    [layer_probability, width_probability.reshape(batch_size, -1)], dim=1
                )
                condition = self.topology_condition_trunk(topology_features)
                condition = condition.unsqueeze(1).expand(-1, HORIZON * MAX_WIDTH, -1)
            else:
                condition = representation.new_zeros((batch_size, HORIZON * MAX_WIDTH, 0))
            content_hidden = self.content_trunk(torch.cat([repeated, slots, condition], dim=-1))
            content_logits = [
                head(content_hidden).view(batch_size, HORIZON, MAX_WIDTH, -1)
                for head in self.content_heads
            ]
            return {
                "role_logits": self.role_head(behavior_hidden),
                "family_logits": self.family_head(behavior_hidden),
                "layer_logits": layer_logits,
                "width_logits": width_logits,
                "content_logits": content_logits,
            }

else:
    FutureContentGRU = None  # type: ignore[assignment,misc]


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
        "content_targets",
    )
    return {name: value.to(device) for name, value in zip(names, values)}


def _bundle_to_loader(
    bundle: Mapping[str, np.ndarray], batch_size: int, shuffle: bool, seed: int
):
    names = (
        "tokens",
        "lengths",
        "context",
        "behavior_role_targets",
        "behavior_family_targets",
        "behavior_family_mask",
        "layer_targets",
        "width_targets",
        "content_targets",
    )
    tensors = [torch.from_numpy(bundle[name]) for name in names]
    dataset = torch.utils.data.TensorDataset(*tensors)
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator
    )


def _permutations_for(width: int) -> np.ndarray:
    return np.asarray(list(itertools.permutations(range(width))), dtype=np.int64)


_PERMUTATIONS = {width: _permutations_for(width) for width in range(1, MAX_WIDTH + 1)}


def _joint_content_loss(outputs: Mapping[str, Any], batch: Mapping[str, Any]) -> Any:
    """Joint per-layer set loss; assignment is reused across all content fields."""

    content_logits = outputs["content_logits"]
    targets = batch["content_targets"]
    widths = batch["width_targets"]
    total_loss = content_logits[0].sum() * 0.0
    matched_nodes = 0
    for layer_index in range(HORIZON):
        for width in range(1, MAX_WIDTH + 1):
            row_indices = torch.nonzero(widths[:, layer_index] == width, as_tuple=False).flatten()
            if row_indices.numel() == 0:
                continue
            cost: Optional[Any] = None
            field_logits: List[Any] = []
            for field_index, logits in enumerate(content_logits):
                selected = logits[row_indices, layer_index, :width, :]
                target = targets[row_indices, layer_index, :width, field_index]
                log_probability = F.log_softmax(selected, dim=-1)
                field_cost = -log_probability.gather(
                    2, target.unsqueeze(1).expand(-1, width, -1)
                ).transpose(1, 2)
                cost = field_cost if cost is None else cost + field_cost
                field_logits.append(selected)
            permutations = torch.as_tensor(
                _PERMUTATIONS[width], dtype=torch.long, device=cost.device
            )
            count = int(row_indices.numel())
            permutation_count = int(permutations.shape[0])
            cost_expanded = cost.detach().unsqueeze(1).expand(-1, permutation_count, -1, -1)
            indices = permutations.unsqueeze(0).expand(count, -1, -1).unsqueeze(-1)
            selected_cost = cost_expanded.gather(-1, indices).squeeze(-1).sum(dim=-1)
            best = selected_cost.argmin(dim=1)
            assignment = permutations[best]
            for field_index, selected in enumerate(field_logits):
                classes = selected.shape[-1]
                chosen = selected.gather(
                    1, assignment.unsqueeze(-1).expand(-1, -1, classes)
                )
                target = targets[row_indices, layer_index, :width, field_index]
                total_loss = total_loss + F.cross_entropy(
                    chosen.reshape(-1, classes), target.reshape(-1), reduction="sum"
                )
            matched_nodes += count * width
    if matched_nodes == 0:
        return total_loss
    return total_loss / float(matched_nodes * len(content_logits))


def loss_components(
    outputs: Mapping[str, Any], batch: Mapping[str, Any], weights: Mapping[str, float]
) -> Dict[str, Any]:
    zero = outputs["role_logits"].sum() * 0.0
    if float(weights.get("next", 0.0)) > 0:
        role_loss = F.cross_entropy(outputs["role_logits"], batch["behavior_role_targets"])
        family_mask = batch["behavior_family_mask"].bool()
        if bool(family_mask.any()):
            family_loss = F.cross_entropy(
                outputs["family_logits"][family_mask], batch["behavior_family_targets"][family_mask]
            )
        else:
            family_loss = zero
        next_loss = 0.5 * role_loss + 0.5 * family_loss
    else:
        role_loss = zero
        family_loss = zero
        next_loss = zero

    if float(weights.get("structure", 0.0)) > 0:
        layer_loss = F.cross_entropy(outputs["layer_logits"], batch["layer_targets"])
        active = batch["width_targets"] > 0
        if bool(active.any()):
            width_logits = outputs["width_logits"][active]
            width_targets = batch["width_targets"][active] - 1
            width_loss = F.cross_entropy(width_logits, width_targets)
        else:
            width_loss = zero
        structure_loss = 0.5 * layer_loss + 0.5 * width_loss
    else:
        layer_loss = zero
        width_loss = zero
        structure_loss = zero

    if float(weights.get("content", 0.0)) > 0:
        content_loss = _joint_content_loss(outputs, batch)
    else:
        content_loss = zero

    total = (
        float(weights.get("next", 0.0)) * next_loss
        + float(weights.get("structure", 0.0)) * structure_loss
        + float(weights.get("content", 0.0)) * content_loss
    )
    return {
        "total": total,
        "next": next_loss,
        "role": role_loss,
        "family": family_loss,
        "structure": structure_loss,
        "layer": layer_loss,
        "width": width_loss,
        "content": content_loss,
    }


def _predict_outputs(
    model: Any, bundle: Mapping[str, np.ndarray], batch_size: int, device: Any
) -> Dict[str, Any]:
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
    result["content_logits"] = [
        result.pop(f"content_logits:{index}") for index in range(len(CONTENT_FIELDS))
    ]
    return result


def _macro_f1(true_values: Sequence[int], predicted_values: Sequence[int]) -> float:
    labels = sorted(
        set(int(value) for value in true_values if int(value) >= 0)
        | set(int(value) for value in predicted_values if int(value) >= 0)
    )
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = sum(int(true == label and predicted == label) for true, predicted in zip(true_values, predicted_values))
        fp = sum(int(true != label and predicted == label) for true, predicted in zip(true_values, predicted_values))
        fn = sum(int(true == label and predicted != label) for true, predicted in zip(true_values, predicted_values))
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(scores))


def _compact_metrics(outputs: Mapping[str, Any], bundle: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    layer_prediction = outputs["layer_logits"].argmax(dim=-1).numpy()
    width_prediction = outputs["width_logits"].argmax(dim=-1).numpy() + 1
    true_layers = bundle["layer_targets"]
    true_widths = bundle["width_targets"]
    selected_widths = np.asarray(
        [
            [
                int(width_prediction[row, index]) if index < layer_prediction[row] else 0
                for index in range(HORIZON)
            ]
            for row in range(len(layer_prediction))
        ],
        dtype=np.int64,
    )
    true_nodes = true_widths.sum(axis=1)
    predicted_nodes = selected_widths.sum(axis=1)
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
        "structure_exact_coverage": float(
            np.mean(
                (layer_prediction == true_layers)
                & np.all(selected_widths == true_widths, axis=1)
            )
        ),
        "future_exists_accuracy": float(
            np.mean((layer_prediction > 0) == (true_layers > 0))
        ),
        "behavior_role_accuracy": float(role_correct.mean()),
        "behavior_role_macro_f1": _macro_f1(role_true.tolist(), role_prediction.tolist()),
        "behavior_family_accuracy": float(family_correct[family_mask].mean())
        if bool(family_mask.any())
        else 0.0,
        "behavior_family_macro_f1": _macro_f1(
            family_true[family_mask].tolist(), family_prediction[family_mask].tolist()
        )
        if bool(family_mask.any())
        else 0.0,
        "behavior_family_coverage": float(family_mask.mean()),
        "behavior_joint_accuracy": float(joint_correct.mean()),
    }


def _best_assignment(cost: np.ndarray) -> List[Tuple[int, int]]:
    """Return minimum-cost (true-index, predicted-index) pairs for <=5 nodes."""

    rows, columns = cost.shape
    if rows == 0 or columns == 0:
        return []
    best_value = math.inf
    best_pairs: List[Tuple[int, int]] = []
    if rows <= columns:
        for permutation in itertools.permutations(range(columns), rows):
            value = sum(float(cost[row, permutation[row]]) for row in range(rows))
            if value < best_value:
                best_value = value
                best_pairs = [(row, permutation[row]) for row in range(rows)]
    else:
        for permutation in itertools.permutations(range(rows), columns):
            value = sum(float(cost[permutation[column], column]) for column in range(columns))
            if value < best_value:
                best_value = value
                best_pairs = [(permutation[column], column) for column in range(columns)]
    return best_pairs


def _content_path_values(
    outputs: Mapping[str, Any],
    bundle: Mapping[str, np.ndarray],
    codec: ContentCodec,
    use_predicted_structure: bool,
) -> Dict[str, Dict[str, List[int]]]:
    values = {
        field: {"true": [], "predicted": []} for field in CONTENT_FIELDS
    }
    layer_prediction = outputs["layer_logits"].argmax(dim=-1).numpy()
    width_prediction = outputs["width_logits"].argmax(dim=-1).numpy() + 1
    content_probabilities = [
        torch.softmax(logits, dim=-1).numpy() for logits in outputs["content_logits"]
    ]
    content_classes = [
        np.asarray(logits.argmax(dim=-1).numpy(), dtype=np.int64)
        for logits in outputs["content_logits"]
    ]
    targets = bundle["content_targets"]
    true_widths = bundle["width_targets"]
    true_layers = bundle["layer_targets"]
    for row_index in range(len(true_layers)):
        for layer_index in range(HORIZON):
            true_width = int(true_widths[row_index, layer_index])
            predicted_width = (
                int(width_prediction[row_index, layer_index])
                if layer_index < int(layer_prediction[row_index])
                else 0
            )
            if not use_predicted_structure:
                predicted_width = true_width if layer_index < int(true_layers[row_index]) else 0
            if true_width == 0 and predicted_width == 0:
                continue
            cost = np.zeros((true_width, predicted_width), dtype=np.float64)
            if true_width and predicted_width:
                for field_index, field in enumerate(CONTENT_FIELDS):
                    probabilities = content_probabilities[field_index][
                        row_index, layer_index, :predicted_width
                    ]
                    target = targets[row_index, layer_index, :true_width, field_index]
                    cost += -np.log(
                        np.clip(probabilities[:, target].T, NLL_FLOOR, 1.0)
                    )
            pairs = _best_assignment(cost)
            paired_true = {true_index for true_index, _predicted_index in pairs}
            paired_predicted = {predicted_index for _true_index, predicted_index in pairs}
            for field_index, field in enumerate(CONTENT_FIELDS):
                target_values = targets[row_index, layer_index, :true_width, field_index]
                predicted_values = content_classes[field_index][
                    row_index, layer_index, :predicted_width
                ]
                for true_index, predicted_index in pairs:
                    values[field]["true"].append(int(target_values[true_index]))
                    values[field]["predicted"].append(int(predicted_values[predicted_index]))
                for true_index in range(true_width):
                    if true_index not in paired_true:
                        values[field]["true"].append(int(target_values[true_index]))
                        values[field]["predicted"].append(MISSING_NODE)
                for predicted_index in range(predicted_width):
                    if predicted_index not in paired_predicted:
                        values[field]["true"].append(MISSING_NODE)
                        values[field]["predicted"].append(int(predicted_values[predicted_index]))
    return values


def _content_metrics(
    outputs: Mapping[str, Any], bundle: Mapping[str, np.ndarray], codec: ContentCodec
) -> Dict[str, Any]:
    oracle_values = _content_path_values(outputs, bundle, codec, use_predicted_structure=False)
    predicted_values = _content_path_values(outputs, bundle, codec, use_predicted_structure=True)
    oracle_by_field = {
        field: _macro_f1(values["true"], values["predicted"])
        for field, values in oracle_values.items()
    }
    predicted_by_field = {
        field: _macro_f1(values["true"], values["predicted"])
        for field, values in predicted_values.items()
    }
    return {
        "oracle_structure_attr_macro_f1": float(np.mean(list(oracle_by_field.values()))),
        "predicted_structure_attr_macro_f1": float(np.mean(list(predicted_by_field.values()))),
        "structure_error_amplification_gap": float(
            np.mean(list(oracle_by_field.values())) - np.mean(list(predicted_by_field.values()))
        ),
        "oracle_structure_attr_macro_f1_by_field": oracle_by_field,
        "predicted_structure_attr_macro_f1_by_field": predicted_by_field,
        "content_observations_predicted_structure": {
            field: len(values["true"]) for field, values in predicted_values.items()
        },
    }


def _top_choices(probabilities: np.ndarray, limit: int = 3) -> List[Tuple[int, float]]:
    choices = [
        (index, max(float(probability), NLL_FLOOR))
        for index, probability in enumerate(probabilities)
    ]
    return sorted(choices, key=lambda item: (-item[1], item[0]))[:limit]


def _shape_scenarios(outputs: Mapping[str, Any], row_index: int) -> List[Tuple[float, int, Tuple[int, ...]]]:
    layer_probability = torch.softmax(outputs["layer_logits"][row_index], dim=-1).numpy()
    width_probabilities = torch.softmax(outputs["width_logits"][row_index], dim=-1).numpy()
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
                for width_class, width_prob in _top_choices(width_probabilities[layer_index]):
                    expanded.append(
                        (probability * width_prob, layer_count, widths + (width_class + 1,))
                    )
        beams = sorted(expanded, key=lambda item: (-item[0], item[1], item[2]))[:12]
    unique: Dict[Tuple[int, Tuple[int, ...]], float] = {}
    for probability, layer_count, widths in beams:
        unique[(layer_count, widths)] = max(unique.get((layer_count, widths), 0.0), probability)
    return [
        (probability, layer_count, widths)
        for (layer_count, widths), probability in sorted(
            unique.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        )[:3]
    ]


def _scenario_json(
    outputs: Mapping[str, Any],
    row_index: int,
    layer_count: int,
    widths: Sequence[int],
    probability: float,
    rank: int,
    codec: ContentCodec,
) -> Dict[str, Any]:
    predictions = [
        logits[row_index].argmax(dim=-1).numpy() for logits in outputs["content_logits"]
    ]
    layers: List[Dict[str, Any]] = []
    for layer_index in range(layer_count):
        nodes: List[Dict[str, Any]] = []
        for node_index in range(int(widths[layer_index])):
            node = {
                field: codec.decode(field, int(predictions[field_index][layer_index, node_index]))
                for field_index, field in enumerate(CONTENT_FIELDS)
            }
            node.update(
                {
                    "layer_offset": layer_index + 1,
                    "predicted_node_index": node_index,
                    "prototype_source": "fixed_layerwise_set_decoder",
                }
            )
            nodes.append(node)
        layers.append({"layer_offset": layer_index + 1, "nodes": nodes})
    return {
        "scenario_id": f"future_content_top{rank:02d}",
        "scenario_probability": float(probability),
        "layers": layers,
        "synthetic_rollout": True,
        "topology_source": "structure_head_layer_and_width_probabilities",
        "content_source": "jointly_matched_fixed_layerwise_slots",
        "edges_emitted": False,
    }


def _behavior_json(outputs: Mapping[str, Any], row_index: int) -> Dict[str, Any]:
    role_probability = torch.softmax(outputs["role_logits"][row_index], dim=-1).numpy()
    family_probability = torch.softmax(outputs["family_logits"][row_index], dim=-1).numpy()
    return {
        "next_role": ROLE_LABELS[int(role_probability.argmax())],
        "next_role_probability": float(role_probability.max()),
        "next_family_if_execute": FAMILY_LABELS[int(family_probability.argmax())],
        "next_family_probability_if_execute": float(family_probability.max()),
    }


def evaluate_split(
    model: Any,
    pairs: Sequence[Pair],
    bundle: Mapping[str, np.ndarray],
    codec: ContentCodec,
    batch_size: int,
    device: Any,
    include_future: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    outputs = _predict_outputs(model, bundle, batch_size, device)
    metrics = _compact_metrics(outputs, bundle)
    rows: List[Dict[str, Any]] = []
    top1_signature = 0
    top3_signature = 0
    for row_index, (feature, label) in enumerate(pairs):
        true_signature = _content_signature(label)
        if include_future:
            decoded = _shape_scenarios(outputs, row_index)
            raw_total = sum(item[0] for item in decoded) or 1.0
            scenarios = [
                _scenario_json(
                    outputs,
                    row_index,
                    layer_count,
                    widths,
                    probability / raw_total,
                    rank,
                    codec,
                )
                for rank, (probability, layer_count, widths) in enumerate(decoded)
            ]
            predicted_signatures = [_content_signature(scenario) for scenario in scenarios]
            top1_signature += int(predicted_signatures[0] == true_signature)
            top3_signature += int(true_signature in predicted_signatures)
        else:
            scenarios = []
        rows.append(
            {
                "sample_id": feature["sample_id"],
                "split": feature["split"],
                "next_step_behavior": _behavior_json(outputs, row_index),
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
                    "node_mask_source": "predicted_layer_count_and_width_prefix",
                    "content_matching": "one_joint_assignment_per_active_layer",
                },
            }
        )
    if include_future:
        metrics.update(_content_metrics(outputs, bundle, codec))
        metrics.update(
            {
                "topology_metrics_valid": True,
                "top1_exact_content_signature_coverage": top1_signature / len(pairs),
                "top3_exact_content_signature_coverage": top3_signature / len(pairs),
            }
        )
    else:
        metrics.update(
            {
                "topology_metrics_valid": False,
                "oracle_structure_attr_macro_f1": None,
                "predicted_structure_attr_macro_f1": None,
                "structure_error_amplification_gap": None,
                "oracle_structure_attr_macro_f1_by_field": None,
                "predicted_structure_attr_macro_f1_by_field": None,
                "top1_exact_content_signature_coverage": None,
                "top3_exact_content_signature_coverage": None,
            }
        )
    return metrics, rows


def _cpu_state_dict(model: Any) -> Dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _train_one(
    variant: Mapping[str, Any],
    seed: int,
    encoder: InputEncoder,
    codec: ContentCodec,
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
    model = FutureContentGRU(
        history_vocab_sizes,
        context_vocab_sizes,
        codec.field_sizes(),
        hidden=int(config["hidden"]),
        history_embedding_dim=int(config["history_embedding_dim"]),
        context_embedding_dim=int(config["context_embedding_dim"]),
        slot_embedding_dim=int(config["slot_embedding_dim"]),
        dropout=float(config["dropout"]),
        use_soft_topology_conditioning=bool(variant.get("use_soft_topology_conditioning", False)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    loader = _bundle_to_loader(train_bundle, int(config["batch_size"]), True, seed + 1000)
    best_score = math.inf
    best_epoch = 0
    best_state: Optional[Dict[str, Any]] = None
    best_train_losses: Optional[Dict[str, float]] = None
    history: List[Dict[str, Any]] = []
    started = time.time()
    weights = variant["loss_weights"]
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        sums = Counter()
        rows_seen = 0
        for values in loader:
            batch = _batch_from_tuple(values, device)
            optimizer.zero_grad(set_to_none=True)
            losses = loss_components(model(batch), batch, weights)
            losses["total"].backward()
            optimizer.step()
            count = int(batch["tokens"].shape[0])
            rows_seen += count
            for key, value in losses.items():
                sums[key] += float(value.detach().cpu()) * count
        validation_outputs = _predict_outputs(model, validation_bundle, int(config["batch_size"]), device)
        validation_metrics = _compact_metrics(validation_outputs, validation_bundle)
        if variant["name"] == "N0":
            score = -(
                validation_metrics["behavior_role_macro_f1"]
                + validation_metrics["behavior_family_macro_f1"]
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
            "content_field_sizes": codec.field_sizes(),
            "content_fields": list(CONTENT_FIELDS),
            "soft_topology_conditioning": bool(variant.get("use_soft_topology_conditioning", False)),
        },
        checkpoint_path,
    )
    validation_metrics, validation_rows = evaluate_split(
        model,
        config["pairs"]["validation"],
        validation_bundle,
        codec,
        int(config["batch_size"]),
        device,
        include_future=variant["name"] != "N0",
    )
    test_metrics, test_rows = evaluate_split(
        model,
        config["pairs"]["test"],
        config["bundles"]["test"],
        codec,
        int(config["batch_size"]),
        device,
        include_future=variant["name"] != "N0",
    )
    holdout_metrics, holdout_rows = evaluate_split(
        model,
        config["pairs"]["holdout"],
        config["bundles"]["holdout"],
        codec,
        int(config["batch_size"]),
        device,
        include_future=variant["name"] != "N0",
    )
    paths = {
        "validation": run_root / "predictions_validation.jsonl.gz",
        "test": run_root / "predictions_test.jsonl.gz",
        "holdout": run_root / "predictions_holdout.jsonl.gz",
    }
    counts = {
        split: _write_jsonl_gz(paths[split], rows)
        for split, rows in (
            ("validation", validation_rows),
            ("test", test_rows),
            ("holdout", holdout_rows),
        )
    }
    run_metrics = {
        "variant": variant["name"],
        "seed": seed,
        "soft_topology_conditioning": bool(variant.get("use_soft_topology_conditioning", False)),
        "conditioning_gradient": "joint" if variant.get("use_soft_topology_conditioning") else None,
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
    result: Dict[str, Any] = {
        "seed_count": len(runs),
        "seeds": [int(run["seed"]) for run in runs],
    }
    for split in ("validation", "test", "holdout"):
        metrics = [run[split] for run in runs]
        keys = sorted(
            {
                key
                for metric in metrics
                for key, value in metric.items()
                if isinstance(value, (int, float))
            }
        )
        result[split] = {}
        for key in keys:
            values = np.asarray(
                [float(metric[key]) for metric in metrics if metric.get(key) is not None], dtype=float
            )
            if len(values):
                result[split][key] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                }
    return result


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected future-content config schema")
    if int(config.get("horizon", -1)) != HORIZON or int(config.get("max_width", -1)) != MAX_WIDTH:
        raise ValueError("future-content dimensions must be H=5 and width=5")
    if tuple(config.get("variants") or ()) != VARIANTS:
        raise ValueError(f"variants must remain {VARIANTS}")
    if tuple(config.get("seeds") or ()) != (11, 22, 33):
        raise ValueError("formal P1 seeds are locked to 11/22/33")
    if tuple(config.get("target", {}).get("content_fields") or ()) != CONTENT_FIELDS:
        raise ValueError("content fields must remain the approved four fields")
    boundary = config.get("data_boundary") or {}
    if boundary.get("fit") != "P_dev/train" or boundary.get("selection") != "P_dev/validation":
        raise ValueError("fit/selection split contract mismatch")
    if boundary.get("diagnostic") != "P_dev/test" or boundary.get("frozen_acceptance") != "P_holdout_diag/holdout":
        raise ValueError("diagnostic/holdout split contract mismatch")
    if boundary.get("scheduler_groups_used") or boundary.get("t_final_read"):
        raise ValueError("scheduler groups and T_final must remain excluded")
    if boundary.get("t_final_sealed") is not True:
        raise ValueError("T_final must remain sealed")


def run(config_path: Path, resume: bool = False) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError(f"PyTorch is required for the formal run: {TORCH_IMPORT_ERROR}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    dataset_root = Path(str(config["dataset_root"]))
    experiment_root = Path(str(config["experiment_root"]))
    artifact_root = experiment_root / "artifacts"
    if artifact_root.exists() and any(artifact_root.iterdir()) and not resume:
        raise FileExistsError(f"refusing to overwrite non-empty artifact directory: {artifact_root}")
    if (experiment_root / "metrics.json").exists():
        raise FileExistsError(f"refusing to overwrite metrics: {experiment_root / 'metrics.json'}")
    stage0 = stage0_audit(dataset_root, str(config.get("dataset_manifest_sha256") or ""))
    if not stage0["passed"]:
        raise ValueError("Stage 0 hard gate failed: " + "; ".join(stage0["errors"][:8]))
    experiment_root.mkdir(parents=True, exist_ok=True)
    _write_json(experiment_root / "stage0_audit.json", stage0)
    manifest_path = dataset_root / "dataset_manifest.json"
    actual_dataset_hash = sha256_file(manifest_path)
    pairs = {
        split: _read_split(dataset_root, split)
        for split in ("train", "validation", "test", "holdout")
    }
    behavior, behavior_audit = join_behavior_targets(
        pairs,
        Path(config["p_dev_role_samples"]),
        Path(config["p_holdout_role_samples"]),
        Path(config["p_dev_family_samples"]),
        Path(config["p_holdout_family_samples"]),
    )
    encoder = InputEncoder(int(config["max_history"])).fit(pairs["train"])
    codec = ContentCodec().fit(pairs["train"])
    bundles: Dict[str, Dict[str, np.ndarray]] = {}
    for split in pairs:
        encoded = encoder.transform(pairs[split])
        encoded.update(build_target_arrays(pairs[split], behavior[split], codec))
        bundles[split] = encoded
    run_config = dict(config)
    run_config["pairs"] = pairs
    run_config["bundles"] = bundles
    variant_runs: Dict[str, List[Dict[str, Any]]] = {}
    started = time.time()
    for variant_name in config["variants"]:
        variant = dict(config["variant_configs"][variant_name])
        variant["name"] = variant_name
        variant_runs[variant_name] = []
        for seed in config["seeds"]:
            run_root = artifact_root / variant_name / f"seed_{seed}"
            existing_metrics = run_root / "metrics.json"
            if resume and existing_metrics.exists():
                variant_runs[variant_name].append(
                    json.loads(existing_metrics.read_text(encoding="utf-8"))
                )
                continue
            variant_runs[variant_name].append(
                _train_one(
                    variant,
                    int(seed),
                    encoder,
                    codec,
                    bundles["train"],
                    bundles["validation"],
                    run_config,
                    run_root,
                )
            )
    aggregates = {name: _aggregate(runs) for name, runs in variant_runs.items()}
    topology_candidates = [name for name in config["variants"] if name != "N0"]
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
        "status": "passed_p1_future_content_experiment",
        "stage0": stage0,
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
            "split_counts": stage0["split_counts"],
            "behavior_join": behavior_audit,
            "upstream_behavior_or_resource_predictions_used_as_feature": False,
            "future_events_or_edges_used_as_feature": False,
            "resource_truth_used_as_feature": False,
            "video_id_used_as_feature": False,
        },
        "model_contract": {
            "encoder": "single_unidirectional_causal_GRU",
            "variants": {
                "N0": "next_step_behavior_only_reference",
                "A": "future_structure_plus_all_future_node_content",
                "B": "A_plus_next_step_behavior_auxiliary",
                "D": "B_plus_soft_topology_probability_content_conditioning",
            },
            "structure_heads": ["layer_count"] + [f"width_{index + 1}" for index in range(HORIZON)],
            "content_fields": list(CONTENT_FIELDS),
            "content_decoder": "fixed_layerwise_unordered_slots_one_joint_assignment_per_layer",
            "node_mask": "derived_from_layer_count_and_width_prefix",
            "next_step_heads": ["next_role", "execute_gated_family"],
            "soft_conditioning_gradient": "joint_for_D",
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
            "stage0_audit_sha256": sha256_file(experiment_root / "stage0_audit.json"),
        },
    }
    _write_json(experiment_root / "metrics.json", metrics)
    manifest_output = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "status": metrics["status"],
        "source_sha256": {
            "scripts/p9d_future_content_multitask.py": sha256_file(Path(__file__)),
            "scripts/p9d_shared_causal_gru.py": sha256_file(PROJECT_ROOT / "scripts/p9d_shared_causal_gru.py"),
            "tests/test_p9d_future_content_multitask.py": sha256_file(
                Path(config["test_path"])
            ),
            "experiments/config.json": sha256_file(config_path),
            "experiments/run.sh": sha256_file(Path(config["run_script_path"])),
        },
        "inputs": metrics["inputs"],
        "outputs_sha256": {
            **output_hashes,
            "stage0_audit.json": sha256_file(experiment_root / "stage0_audit.json"),
            "metrics.json": sha256_file(experiment_root / "metrics.json"),
        },
        "verification": {
            "stage0_passed": True,
            "scheduler_groups_used_for_fit": False,
            "t_final_read": False,
            "raw_traces_modified": False,
            "prediction_contract": "identity_free_future_node_content_with_derived_structure_mask",
        },
    }
    _write_json(experiment_root / "run_manifest.json", manifest_output)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--stage0-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stage0_only:
        if args.dataset_root is None:
            parser.error("--dataset-root is required with --stage0-only")
        result = stage0_audit(args.dataset_root)
        if args.output:
            _write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return
    if args.config is None:
        parser.error("--config is required for training")
    result = run(args.config, resume=args.resume)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_variant": result["fit_contract"]["selected_variant_by_validation_structure_score"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
