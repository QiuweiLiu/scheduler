#!/usr/bin/env python3
"""J-series shared components: encoding, causal GRU model, metrics, bootstrap.

Scope (frozen design `docs/p9d_j_series_design.md`, v3.1):
- model input is execution-time only: `stack/baseline`, task context, history;
  no future-derived information (`prefix_model_reuse` removed);
- one shared interface of structure/content/behavior heads for every variant;
- resource targets (runtime primary, load hurdle, memory descriptive) live only
  under ``targets`` and enter training through the resource heads;
- validation/test scoring is per row, aggregated by source video for bootstrap.

This module is imported by ``scripts/j_series_train_eval.py`` only.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - exercised only when torch is missing
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = str(exc)
else:  # pragma: no cover
    TORCH_IMPORT_ERROR = None

PAD = "__PAD__"
UNK = "__UNK__"
MAX_HISTORY = 64
POSITION_BUCKETS = 64

HISTORY_FIELDS = ("event_type", "node_type", "role", "raw_action", "action_family", "model_id")

# ---- causal historical telemetry channel (Stage 0 contract) --------------- #
# Frozen order.  j_series_dataset_histres_v2 writes these keys inside
# model_input.history[*].history_resource; the model reads ONLY this block and
# never history_resource_audit.  A merged nested child keeps mask=0 by design
# (its runtime is already inside the composite parent), so mask=0 never means
# "forgot to fill it in".
HISTRES_NUMERIC_FIELDS = (
    "runtime_z",
    "runtime_present",
    "load_positive_z",
    "load_present",
    "load_nonzero",
    "peak_alloc_z",
    "peak_alloc_present",
    "peak_reserved_z",
    "peak_reserved_present",
)
HISTRES_OBSERVED_SOURCES = (
    "runtime_present",
    "load_present",
    "peak_alloc_present",
    "peak_reserved_present",
)
STATUS_VALUES = ("UNOBSERVED", "success", "failed", "timeout", "unknown")
STATUS_INDEX = {value: index for index, value in enumerate(STATUS_VALUES)}
CONTEXT_FIELDS = (
    "answer_type",
    "domain",
    "official_task_type",
    "question_type",
    "sub_category",
    "temporal_scope",
    "baseline",
    "model_stack_id",
    "planner_model_id",
)
SLOT_CAT_FIELDS = ("node_type", "role", "action_family", "model_class")
NESTED_FIELD = "nested_model_class"
ATTRIBUTE_HEADS = ("node_type", "role", "action_family", "model_class", "merged_nested_call", "is_retry", "nested_model_class")
TARGET_TAUS = (0.50, 0.90, 0.95)
MISSING_INT = -1
MISSING_FLOAT = -1.0


# --------------------------------------------------------------------------- #
# generic io
# --------------------------------------------------------------------------- #
def read_jsonl_gz(path: Path) -> Iterable[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- #
# vocabularies and encoding
# --------------------------------------------------------------------------- #
class Vocab:
    def __init__(self, values: Iterable[str]):
        self.values = sorted({str(v) for v in values if v is not None and str(v) != ""})
        self.index = {PAD: 0, UNK: 1}
        for value in self.values:
            self.index[value] = len(self.index)

    def __len__(self) -> int:
        return len(self.index)

    def encode(self, value: Any) -> int:
        if value is None:
            return self.index[UNK]
        return self.index.get(str(value), self.index[UNK])


class VocabCollection:
    def __init__(self, history: Mapping[str, Vocab], context: Mapping[str, Vocab], modalities: Vocab, slot: Mapping[str, Vocab], nested: Vocab):
        self.history = dict(history)
        self.context = dict(context)
        self.modalities = modalities
        self.slot = dict(slot)
        self.nested = nested

    def to_json(self) -> Dict[str, Any]:
        def dump(vocab: Vocab) -> Dict[str, Any]:
            return {"size": len(vocab), "values": vocab.values}

        return {
            "history": {field: dump(vocab) for field, vocab in self.history.items()},
            "context": {field: dump(vocab) for field, vocab in self.context.items()},
            "modalities": dump(self.modalities),
            "slot": {field: dump(vocab) for field, vocab in self.slot.items()},
            "nested": dump(self.nested),
        }


def build_vocabs(rows: Sequence[Mapping[str, Any]]) -> VocabCollection:
    history_values: Dict[str, List[str]] = {field: [] for field in HISTORY_FIELDS}
    context_values: Dict[str, List[str]] = {field: [] for field in CONTEXT_FIELDS}
    modality_values: List[str] = []
    slot_values: Dict[str, List[str]] = {field: [] for field in SLOT_CAT_FIELDS}
    nested_values: List[str] = []
    for row in rows:
        model_input = row["model_input"]
        for step in model_input["history"]:
            for field in HISTORY_FIELDS:
                history_values[field].append(str(step.get(field)))
        for field in CONTEXT_FIELDS:
            context_values[field].append(str(model_input["task_context"].get(field)) if field in model_input["task_context"] else str(model_input["stack_context"].get(field)))
        for value in model_input["task_context"].get("required_modalities") or []:
            modality_values.append(str(value))
        for slot in row["future"]:
            for field in SLOT_CAT_FIELDS:
                slot_values[field].append(str(slot.get(field)))
            nested_values.append(str(slot.get(NESTED_FIELD) or "__NONE__"))
    return VocabCollection(
        history={field: Vocab(values) for field, values in history_values.items()},
        context={field: Vocab(values) for field, values in context_values.items()},
        modalities=Vocab(modality_values),
        slot={field: Vocab(values) for field, values in slot_values.items()},
        nested=Vocab(nested_values),
    )


def encode_rows(rows: Sequence[Mapping[str, Any]], vocabs: VocabCollection, horizon: int) -> Dict[str, np.ndarray]:
    n = len(rows)
    hist_ids = {field: np.zeros((n, MAX_HISTORY), dtype=np.int64) for field in HISTORY_FIELDS}
    hist_len = np.zeros(n, dtype=np.int64)
    ctx_ids = np.zeros((n, len(CONTEXT_FIELDS)), dtype=np.int64)
    mod_vec = np.zeros((n, len(vocabs.modalities)), dtype=np.float32)
    slot_cat = {field: np.full((n, horizon), MISSING_INT, dtype=np.int64) for field in SLOT_CAT_FIELDS}
    slot_nested = np.full((n, horizon), MISSING_INT, dtype=np.int64)
    slot_merged = np.full((n, horizon), MISSING_INT, dtype=np.int64)
    slot_retry = np.full((n, horizon), MISSING_INT, dtype=np.int64)
    slot_present = np.zeros((n, horizon), dtype=np.float32)
    hist_res_num = np.zeros((n, MAX_HISTORY, len(HISTRES_NUMERIC_FIELDS)), dtype=np.float32)
    hist_res_mask = np.zeros((n, MAX_HISTORY), dtype=np.float32)
    hist_status = np.zeros((n, MAX_HISTORY), dtype=np.int64)
    runtime_ms = np.full((n, horizon), MISSING_FLOAT, dtype=np.float64)
    load_ms = np.full((n, horizon), MISSING_FLOAT, dtype=np.float64)
    load_occ = np.full((n, horizon), MISSING_FLOAT, dtype=np.float64)
    memory_mb = np.full((n, horizon), MISSING_FLOAT, dtype=np.float64)

    for i, row in enumerate(rows):
        model_input = row["model_input"]
        history = model_input["history"][-MAX_HISTORY:]
        hist_len[i] = len(history)
        for j, step in enumerate(history):
            for field in HISTORY_FIELDS:
                hist_ids[field][i, j] = vocabs.history[field].encode(step.get(field))
            # causal telemetry: read only the model-visible block
            channel = step.get("history_resource")
            if not isinstance(channel, dict):
                continue
            observed = False
            for k, key in enumerate(HISTRES_NUMERIC_FIELDS):
                value = channel.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                hist_res_num[i, j, k] = float(value)
                if key in HISTRES_OBSERVED_SOURCES and float(value) > 0.0:
                    observed = True
            hist_res_mask[i, j] = 1.0 if observed else 0.0
            hist_status[i, j] = STATUS_INDEX.get(str(channel.get("status_class")), STATUS_INDEX["unknown"])
        for field_idx, field in enumerate(CONTEXT_FIELDS):
            if field in model_input["task_context"]:
                value = model_input["task_context"].get(field)
            else:
                value = model_input["stack_context"].get(field)
            ctx_ids[i, field_idx] = vocabs.context[field].encode(value)
        for value in model_input["task_context"].get("required_modalities") or []:
            mod_vec[i, vocabs.modalities.encode(value)] = 1.0
        for t, slot in enumerate(row["future"][:horizon]):
            slot_present[i, t] = 1.0
            for field in SLOT_CAT_FIELDS:
                slot_cat[field][i, t] = vocabs.slot[field].encode(slot.get(field))
            slot_merged[i, t] = 1 if slot.get("merged_nested_call") else 0
            slot_retry[i, t] = 1 if slot.get("is_retry") else 0
            if slot.get("merged_nested_call"):
                slot_nested[i, t] = vocabs.nested.encode(slot.get(NESTED_FIELD))
            targets = slot["targets"]
            if targets.get("runtime_ms") is not None:
                runtime_ms[i, t] = float(targets["runtime_ms"])
            if targets.get("load_ms") is not None:
                load_ms[i, t] = float(targets["load_ms"])
                load_occ[i, t] = 1.0 if float(targets["load_ms"]) > 0.0 else 0.0
            if targets.get("peak_memory_inclusive_mb") is not None:
                memory_mb[i, t] = float(targets["peak_memory_inclusive_mb"])

    fields = {
        "hist_len": hist_len,
        "hist_res_num": hist_res_num,
        "hist_res_mask": hist_res_mask,
        "hist_status": hist_status,
        "ctx_ids": ctx_ids,
        "mod_vec": mod_vec,
        "slot_present": slot_present,
        "slot_nested": slot_nested,
        "slot_merged": slot_merged,
        "slot_retry": slot_retry,
        "runtime_ms": runtime_ms,
        "load_ms": load_ms,
        "load_occ": load_occ,
        "memory_mb": memory_mb,
    }
    for field in HISTORY_FIELDS:
        fields[f"hist_{field}"] = hist_ids[field]
    for field, array in slot_cat.items():
        fields[f"slot_{field}"] = array

    length = np.asarray([int(row["bounded_future_length"]) for row in rows], dtype=np.int64)
    termination = np.asarray([int(row["termination"]) for row in rows], dtype=np.int64)
    nxt_role = np.full(n, MISSING_INT, dtype=np.int64)
    nxt_family = np.full(n, MISSING_INT, dtype=np.int64)
    for i, row in enumerate(rows):
        if row["future"]:
            nxt_role[i] = vocabs.slot["role"].encode(row["future"][0].get("role"))
            nxt_family[i] = vocabs.slot["action_family"].encode(row["future"][0].get("action_family"))
    fields["length"] = length
    fields["termination"] = termination
    fields["next_role"] = nxt_role
    fields["next_family"] = nxt_family
    return fields


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
if nn is not None:

    class JSeriesModel(nn.Module):
        """Causal GRU backbone with structure/content/behavior/resource heads."""

        def __init__(self, vocabs: VocabCollection, model_cfg: Mapping[str, Any], horizon: int = 5, attr_dim_pad: int = 0, duration_mode: str = "shared"):
            super().__init__()
            self.horizon = int(horizon)
            self.hidden = int(model_cfg.get("hidden", 128))
            self.hist_dim = int(model_cfg.get("history_embedding_dim", 16))
            self.ctx_dim = int(model_cfg.get("context_embedding_dim", 12))
            self.slot_dim = int(model_cfg.get("slot_embedding_dim", 16))
            self.dropout = float(model_cfg.get("dropout", 0.1))
            self.duration_mode = str(duration_mode)
            self.duration_branch_hidden = int(model_cfg.get("duration_branch_hidden", 64))
            # categorical cardinalities
            self.card = {
                "node_type": len(vocabs.slot["node_type"]),
                "role": len(vocabs.slot["role"]),
                "action_family": len(vocabs.slot["action_family"]),
                "model_class": len(vocabs.slot["model_class"]),
                "nested_model_class": len(vocabs.nested),
            }
            self.attr_dim = (
                self.card["node_type"]
                + self.card["role"]
                + self.card["action_family"]
                + self.card["model_class"]
                + 2
                + self.card["nested_model_class"]
            ) + int(attr_dim_pad)

            # history embeddings: one per field, summed per step
            self.hist_emb = nn.ModuleDict({field: nn.Embedding(len(vocabs.history[field]), self.hist_dim, padding_idx=0) for field in HISTORY_FIELDS})
            self.pos_emb = nn.Embedding(POSITION_BUCKETS, self.hist_dim)
            self.gru = nn.GRU(self.hist_dim, self.hidden, batch_first=True)
            # causal historical-telemetry branch.  F0 keeps the mask at zero; F1
            # feeds the real observed mask.  The projection is multiplied by the
            # mask so an all-zero input cannot leak a bias-only pseudo-signal.
            self.hist_res_dim = int(model_cfg.get("history_resource_dim", self.hist_dim))
            self.hist_res_proj = nn.Sequential(
                nn.Linear(len(HISTRES_NUMERIC_FIELDS), self.hist_res_dim),
                nn.SiLU(),
                nn.Linear(self.hist_res_dim, self.hist_dim),
            )
            # padding_idx=0 pins UNOBSERVED to the zero vector, so a token with no
            # observed outcome contributes exactly nothing - including padding and the
            # current node - which is what makes the telemetry-off arm equal to the
            # old categorical encoder.
            self.hist_status_emb = nn.Embedding(len(STATUS_VALUES), self.hist_dim, padding_idx=0)
            # single switch for the whole telemetry branch; F0 sets it False
            self.use_history_telemetry = bool(model_cfg.get("use_history_telemetry", True))
            # context embeddings
            self.ctx_emb = nn.ModuleDict({field: nn.Embedding(len(vocabs.context[field]), self.ctx_dim) for field in CONTEXT_FIELDS})
            self.mod_proj = nn.Linear(len(vocabs.modalities), self.ctx_dim)
            self.repr = nn.Linear(self.hidden, self.hidden)
            self.repr_ctx = nn.Linear(self.ctx_dim, self.hidden)
            # slot attribute heads (shared across slots, slot conditioned)
            self.slot_emb = nn.Embedding(self.horizon + 1, self.slot_dim)
            self.attr_hidden = nn.Linear(self.hidden + self.slot_dim, self.hidden)
            self.head_node_type = nn.Linear(self.hidden, self.card["node_type"])
            self.head_role = nn.Linear(self.hidden, self.card["role"])
            self.head_action_family = nn.Linear(self.hidden, self.card["action_family"])
            self.head_model_class = nn.Linear(self.hidden, self.card["model_class"])
            self.head_merged = nn.Linear(self.hidden, 1)
            self.head_retry = nn.Linear(self.hidden, 1)
            self.head_nested = nn.Linear(self.hidden, self.card["nested_model_class"])
            # structure / behavior heads
            self.head_length = nn.Linear(self.hidden, self.horizon + 1)
            self.head_termination = nn.Linear(self.hidden, 1)
            self.head_next_role = nn.Linear(self.hidden, len(vocabs.slot["role"]))
            self.head_next_family = nn.Linear(self.hidden, len(vocabs.slot["action_family"]))
            # resource heads: condition = repr + slot + attribute feature
            self.res_hidden = nn.Linear(self.hidden + self.slot_dim + self.attr_dim, self.hidden)
            self.head_runtime = nn.Linear(self.hidden, len(TARGET_TAUS))
            self.head_load_occ = nn.Linear(self.hidden, 1)
            self.head_load_dur = nn.Linear(self.hidden if self.duration_mode == "shared" else self.duration_branch_hidden, len(TARGET_TAUS))
            self.head_memory = nn.Linear(self.hidden, 1)
            if self.duration_mode != "shared":
                self.dur_adapter = nn.Linear(self.hidden + self.slot_dim + self.attr_dim, self.duration_branch_hidden)

        def encode(self, batch: Mapping[str, Any], *, use_history_telemetry: bool | None = None) -> Any:
            if use_history_telemetry is None:
                use_history_telemetry = self.use_history_telemetry
            device = next(self.parameters()).device
            hist_ids = {field: batch[f"hist_{field}"] for field in HISTORY_FIELDS}
            lengths = batch["hist_len"]
            steps = hist_ids[HISTORY_FIELDS[0]].shape[1]
            emb = None
            for field in HISTORY_FIELDS:
                value = self.hist_emb[field](hist_ids[field])
                emb = value if emb is None else emb + value
            positions = torch.arange(steps, device=device).unsqueeze(0).clamp(max=POSITION_BUCKETS - 1)
            emb = emb + self.pos_emb(positions)
            # the whole telemetry branch (numeric magnitudes AND outcome status) is
            # gated by one switch: with it off, F0 is the categorical encoder plus
            # nothing, which is the arm the F1 - F0 contrast needs
            if use_history_telemetry and "hist_res_num" in batch:
                observed = batch["hist_res_mask"].unsqueeze(-1)
                emb = emb + observed * self.hist_res_proj(batch["hist_res_num"])
                emb = emb + self.hist_status_emb(batch["hist_status"])
            packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
            _, hidden = self.gru(packed)
            h = hidden[-1]
            ctx = None
            for idx, field in enumerate(CONTEXT_FIELDS):
                value = self.ctx_emb[field](batch["ctx_ids"][:, idx])
                ctx = value if ctx is None else ctx + value
            ctx = ctx + self.mod_proj(batch["mod_vec"])
            repr_vec = torch.tanh(self.repr(h) + self.repr_ctx(ctx))
            return repr_vec

        def structure(self, repr_vec: Any) -> Dict[str, Any]:
            return {"length_logits": self.head_length(repr_vec), "termination_logit": self.head_termination(repr_vec).squeeze(-1)}

        def behavior(self, repr_vec: Any) -> Dict[str, Any]:
            return {"next_role_logits": self.head_next_role(repr_vec), "next_family_logits": self.head_next_family(repr_vec)}

        def attribute_logits(self, repr_vec: Any) -> Dict[str, Any]:
            batch_size = repr_vec.shape[0]
            slot_ids = torch.arange(1, self.horizon + 1, device=repr_vec.device)
            slot_vec = self.slot_emb(slot_ids).unsqueeze(0).expand(batch_size, -1, -1)
            z = torch.tanh(self.attr_hidden(torch.cat([repr_vec.unsqueeze(1).expand(-1, self.horizon, -1), slot_vec], dim=-1)))
            z = F.dropout(z, p=self.dropout, training=self.training)
            return {
                "node_type": self.head_node_type(z),
                "role": self.head_role(z),
                "action_family": self.head_action_family(z),
                "model_class": self.head_model_class(z),
                "merged": self.head_merged(z).squeeze(-1),
                "retry": self.head_retry(z).squeeze(-1),
                "nested": self.head_nested(z),
            }

        def attribute_distribution(self, logits: Mapping[str, Any]) -> Any:
            probs = {
                "node_type": F.softmax(logits["node_type"], dim=-1),
                "role": F.softmax(logits["role"], dim=-1),
                "action_family": F.softmax(logits["action_family"], dim=-1),
                "model_class": F.softmax(logits["model_class"], dim=-1),
                "merged": torch.sigmoid(logits["merged"]),
                "retry": torch.sigmoid(logits["retry"]),
                "nested": F.softmax(logits["nested"], dim=-1),
            }
            feature = torch.cat(
                [
                    probs["node_type"],
                    probs["role"],
                    probs["action_family"],
                    probs["model_class"],
                    probs["merged"].unsqueeze(-1),
                    probs["retry"].unsqueeze(-1),
                    probs["nested"],
                ],
                dim=-1,
            )
            return feature, probs

        def resource(self, repr_vec: Any, attr_feature: Any) -> Dict[str, Any]:
            batch_size = repr_vec.shape[0]
            if attr_feature is None:
                attr_feature = torch.zeros(batch_size, self.horizon, self.attr_dim, device=repr_vec.device)
            slot_ids = torch.arange(1, self.horizon + 1, device=repr_vec.device)
            slot_vec = self.slot_emb(slot_ids).unsqueeze(0).expand(batch_size, -1, -1)
            x = torch.cat([repr_vec.unsqueeze(1).expand(-1, self.horizon, -1), slot_vec, attr_feature], dim=-1)
            z = F.dropout(torch.tanh(self.res_hidden(x)), p=self.dropout, training=self.training)
            if self.duration_mode == "shared":
                duration = self.head_load_dur(z)
            elif self.duration_mode == "decoupled":
                duration = self.head_load_dur(F.dropout(torch.tanh(self.dur_adapter(x)), p=self.dropout, training=self.training))
            elif self.duration_mode == "decoupled_shared_frozen":
                dur_input = torch.cat([z.detach(), slot_vec, attr_feature], dim=-1)
                duration = self.head_load_dur(F.dropout(torch.tanh(self.dur_adapter(dur_input)), p=self.dropout, training=self.training))
            else:
                raise ValueError(f"unknown duration_mode: {self.duration_mode}")
            return {
                "runtime_log_quantiles": self.head_runtime(z),
                "load_occ_logit": self.head_load_occ(z).squeeze(-1),
                "load_dur_log_quantiles": duration,
                "memory_log": self.head_memory(z).squeeze(-1),
            }


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
def pinball_loss(prediction: Any, target: Any, tau: float) -> Any:
    diff = target - prediction
    return torch.maximum(tau * diff, (tau - 1.0) * diff)


def masked_mean(values: Any, mask: Any, eps: float = 1e-9) -> Any:
    mask_f = mask.to(values.dtype)
    return (values * mask_f).sum() / mask_f.sum().clamp(min=eps)


def structure_loss(outputs: Mapping[str, Any], batch: Mapping[str, Any]) -> Any:
    length_loss = F.cross_entropy(outputs["length_logits"], batch["length"])
    termination_loss = F.binary_cross_entropy_with_logits(outputs["termination_logit"], batch["termination"].to(outputs["termination_logit"].dtype))
    return length_loss + termination_loss


def content_loss(logits: Mapping[str, Any], batch: Mapping[str, Any], slot_mask: Any) -> Any:
    parts: List[Any] = []
    for field in SLOT_CAT_FIELDS:
        target = batch[f"slot_{field}"]
        ce = F.cross_entropy(logits[field].transpose(1, 2), target.clamp(min=0), reduction="none")
        parts.append(masked_mean(ce, slot_mask))
    merged_target = batch["slot_merged"].clamp(min=0).to(logits["merged"].dtype)
    parts.append(masked_mean(F.binary_cross_entropy_with_logits(logits["merged"], merged_target, reduction="none"), slot_mask))
    retry_target = batch["slot_retry"].clamp(min=0).to(logits["retry"].dtype)
    parts.append(masked_mean(F.binary_cross_entropy_with_logits(logits["retry"], retry_target, reduction="none"), slot_mask))
    nested_mask = slot_mask * (batch["slot_merged"] == 1).to(slot_mask.dtype)
    nested_target = batch["slot_nested"].clamp(min=0)
    nested_ce = F.cross_entropy(logits["nested"].transpose(1, 2), nested_target, reduction="none")
    parts.append(masked_mean(nested_ce, nested_mask))
    return torch.stack(parts).mean()


def behavior_loss(outputs: Mapping[str, Any], batch: Mapping[str, Any]) -> Any:
    mask = (batch["next_role"] >= 0).to(outputs["next_role_logits"].dtype)
    role_loss = masked_mean(F.cross_entropy(outputs["next_role_logits"], batch["next_role"].clamp(min=0), reduction="none"), mask)
    family_loss = masked_mean(F.cross_entropy(outputs["next_family_logits"], batch["next_family"].clamp(min=0), reduction="none"), mask)
    return role_loss + family_loss


def resource_loss(outputs: Mapping[str, Any], batch: Mapping[str, Any], slot_mask: Any) -> Any:
    runtime_target = torch.log1p(batch["runtime_ms"].clamp(min=0.0))
    runtime_valid = slot_mask * (batch["runtime_ms"] > 0).to(slot_mask.dtype)
    runtime_parts = [masked_mean(pinball_loss(outputs["runtime_log_quantiles"][:, :, k], runtime_target, tau), runtime_valid) for k, tau in enumerate(TARGET_TAUS)]
    runtime_part = torch.stack(runtime_parts).mean()

    occ_valid = slot_mask * (batch["load_occ"] >= 0).to(slot_mask.dtype)
    occ_target = batch["load_occ"].clamp(min=0).to(outputs["load_occ_logit"].dtype)
    load_occ = masked_mean(F.binary_cross_entropy_with_logits(outputs["load_occ_logit"], occ_target, reduction="none"), occ_valid)

    dur_valid = occ_valid * (batch["load_occ"] == 1).to(slot_mask.dtype)
    dur_target = torch.log1p(batch["load_ms"].clamp(min=0.0))
    dur_parts = [masked_mean(pinball_loss(outputs["load_dur_log_quantiles"][:, :, k], dur_target, tau), dur_valid) for k, tau in enumerate(TARGET_TAUS)]
    load_dur = torch.stack(dur_parts).mean()
    return runtime_part + load_occ + load_dur


def nanmean_stack(stack: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(stack, axis=0)


# --------------------------------------------------------------------------- #
# per-row metrics (numpy, for validation/test selection and bootstrap)
# --------------------------------------------------------------------------- #
def per_row_metrics(outputs: Mapping[str, np.ndarray], batch: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Return per-row metric arrays; NaN marks rows excluded from that endpoint."""
    horizon = batch["length"].shape[0]
    metrics: Dict[str, np.ndarray] = {}
    slot_valid = (np.arange(outputs["runtime_ms"].shape[1])[None, :] < batch["length"][:, None]).astype(np.float64)

    # runtime quantile pinball on raw ms (primary)
    tau_scores = []
    for k, tau in enumerate(TARGET_TAUS):
        pred = outputs["runtime_ms"][:, :, k]
        diff = batch["runtime_ms"] - pred
        pb = np.maximum(tau * diff, (tau - 1.0) * diff) * slot_valid
        tau_scores.append(np.divide(pb.sum(axis=1), slot_valid.sum(axis=1), out=np.full(horizon, np.nan), where=slot_valid.sum(axis=1) > 0))
    metrics["runtime_qscore"] = nanmean_stack(np.stack(tau_scores))

    # structure
    length_pred = outputs["length_logits"].argmax(axis=1)
    metrics["length_mae"] = np.abs(length_pred - batch["length"]).astype(np.float64)
    term_p = 1.0 / (1.0 + np.exp(-outputs["termination_logit"]))
    term_p = np.clip(term_p, 1e-12, 1 - 1e-12)
    metrics["termination_bce"] = -(batch["termination"] * np.log(term_p) + (1 - batch["termination"]) * np.log(1 - term_p))

    # content accuracy: mean over fields/valid slots
    field_acc = []
    for field in SLOT_CAT_FIELDS:
        pred = outputs[f"attr_{field}"].argmax(axis=-1)
        correct = ((pred == batch[f"slot_{field}"]) & (batch[f"slot_{field}"] >= 0)).astype(np.float64) * slot_valid
        field_acc.append(np.divide(correct.sum(axis=1), slot_valid.sum(axis=1), out=np.full(horizon, np.nan), where=slot_valid.sum(axis=1) > 0))
    merged_pred = (outputs["attr_merged"] > 0).astype(np.int64)
    merged_correct = ((merged_pred == batch["slot_merged"]) & (batch["slot_merged"] >= 0)).astype(np.float64) * slot_valid
    field_acc.append(np.divide(merged_correct.sum(axis=1), slot_valid.sum(axis=1), out=np.full(horizon, np.nan), where=slot_valid.sum(axis=1) > 0))
    retry_pred = (outputs["attr_retry"] > 0).astype(np.int64)
    retry_correct = ((retry_pred == batch["slot_retry"]) & (batch["slot_retry"] >= 0)).astype(np.float64) * slot_valid
    field_acc.append(np.divide(retry_correct.sum(axis=1), slot_valid.sum(axis=1), out=np.full(horizon, np.nan), where=slot_valid.sum(axis=1) > 0))
    nested_valid = slot_valid * (batch["slot_merged"] == 1).astype(np.float64)
    nested_pred = outputs["attr_nested"].argmax(axis=-1)
    nested_correct = ((nested_pred == batch["slot_nested"]) & (batch["slot_nested"] >= 0)).astype(np.float64) * nested_valid
    field_acc.append(np.divide(nested_correct.sum(axis=1), nested_valid.sum(axis=1), out=np.full(horizon, np.nan), where=nested_valid.sum(axis=1) > 0))
    metrics["content_accuracy"] = nanmean_stack(np.stack(field_acc))

    # behavior
    metrics["next_role_acc"] = (outputs["next_role_logits"].argmax(axis=1) == batch["next_role"]).astype(np.float64)
    metrics["next_role_acc"][batch["next_role"] < 0] = np.nan
    metrics["next_family_acc"] = (outputs["next_family_logits"].argmax(axis=1) == batch["next_family"]).astype(np.float64)
    metrics["next_family_acc"][batch["next_family"] < 0] = np.nan

    # load hurdle
    occ_valid = (batch["load_occ"] >= 0).astype(np.float64) * slot_valid
    occ_p = 1.0 / (1.0 + np.exp(-outputs["load_occ_logit"]))
    brier = ((occ_p - batch["load_occ"].clip(min=0)) ** 2) * occ_valid
    metrics["load_brier"] = np.divide(brier.sum(axis=1), occ_valid.sum(axis=1), out=np.full(horizon, np.nan), where=occ_valid.sum(axis=1) > 0)
    dur_valid = occ_valid * (batch["load_occ"] == 1).astype(np.float64)
    dur_scores = []
    for k, tau in enumerate(TARGET_TAUS):
        pred = outputs["load_dur_ms"][:, :, k]
        diff = batch["load_ms"] - pred
        pb = np.maximum(tau * diff, (tau - 1.0) * diff) * dur_valid
        dur_scores.append(np.divide(pb.sum(axis=1), dur_valid.sum(axis=1), out=np.full(horizon, np.nan), where=dur_valid.sum(axis=1) > 0))
    metrics["load_dur_qscore"] = nanmean_stack(np.stack(dur_scores))
    return metrics


# --------------------------------------------------------------------------- #
# bootstrap
# --------------------------------------------------------------------------- #
def make_bootstrap_indices(video_codes: np.ndarray, replicates: int, seed: int) -> np.ndarray:
    """Video-cluster bootstrap: [B, V] matrix of sampled video positions."""
    rng = np.random.default_rng(seed)
    videos = np.unique(video_codes)
    if videos.size == 0:
        raise ValueError("no videos to bootstrap")
    draws = rng.integers(0, videos.size, size=(replicates, videos.size), dtype=np.int64)
    return draws


def bootstrap_delta_ci(delta: np.ndarray, video_codes: np.ndarray, draws: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """Paired video bootstrap CI for mean(delta); NaN rows are dropped pairwise."""
    videos = np.unique(video_codes)
    index = {int(v): i for i, v in enumerate(videos)}
    codes = np.asarray([index[int(v)] for v in video_codes], dtype=np.int64)
    sums = np.zeros(videos.size, dtype=np.float64)
    counts = np.zeros(videos.size, dtype=np.int64)
    valid = ~np.isnan(delta)
    np.add.at(sums, codes[valid], delta[valid])
    np.add.at(counts, codes[valid], 1)
    flat = draws.ravel()
    sums_flat = sums[flat]
    counts_flat = counts[flat]
    sums_rep = sums_flat.reshape(draws.shape).sum(axis=1)
    counts_rep = counts_flat.reshape(draws.shape).sum(axis=1)
    means = np.full(draws.shape[0], np.nan, dtype=np.float64)
    ok = counts_rep > 0
    means[ok] = sums_rep[ok] / counts_rep[ok]
    valid_ratio = float(ok.mean())
    finite = means[np.isfinite(means)]
    result = {
        "delta_mean": float(np.nanmean(delta)) if np.isfinite(delta).any() else float("nan"),
        "ci_lower": float(np.percentile(finite, 100 * alpha / 2)) if finite.size else float("nan"),
        "ci_upper": float(np.percentile(finite, 100 * (1 - alpha / 2))) if finite.size else float("nan"),
        "valid_ratio": valid_ratio,
        "n_rows": int(valid.sum()),
    }
    result["reliable"] = valid_ratio >= 0.95
    return result


def ni_feasible(record: Mapping[str, Any], manifest: Sequence[Mapping[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
    details: Dict[str, Any] = {}
    feasible = True
    for spec in manifest:
        endpoint = spec["endpoint"]
        entry = record.get(endpoint) or {}
        ci_upper = entry.get("ci_upper", float("nan"))
        ci_lower = entry.get("ci_lower", float("nan"))
        reliable = bool(entry.get("reliable", False))
        if spec["direction"] == "lower_is_better":
            ok = bool(np.isfinite(ci_upper) and ci_upper < spec["delta"])
        else:
            ok = bool(np.isfinite(ci_lower) and ci_lower > -spec["delta"])
        ok = ok and reliable
        details[endpoint] = {"ok": ok, "ci_lower": ci_lower, "ci_upper": ci_upper, "delta": spec["delta"], "reliable": reliable}
        feasible = feasible and ok
    return feasible, details
