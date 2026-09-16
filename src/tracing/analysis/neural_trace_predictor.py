#!/usr/bin/env python3
"""Leakage-audited structured neural predictor for Video Agent traces.

Modes:
  build: convert canonical prefix samples into trainable horizon labels.
  train: fit/evaluate a multi-horizon GRU with a train-only transition graph.

This file intentionally stays dependency-light: torch plus the Python standard
library. It does not load a VLM. Qwen feature extraction is a later ablation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

HORIZONS = (1, 2, 3, 4, 5)
TOKEN_PAD = "__PAD__"
TOKEN_START = "__START__"
TOKEN_UNK = "__UNK__"
CAT_KEYS = (
    "baseline",
    "planner_model_id",
    "model_stack_id",
    "dataset",
    "question_type",
    "temporal_scope",
    "answer_type",
    "domain",
    "sub_category",
    "required_modalities",
    "last_activity",
    "position_bucket",
    "prefix_tail2",
    "prefix_tail3",
    "planner_position",
    "baseline_position",
    "task_signature",
)
NUM_KEYS = (
    "position",
    "coverage_ratio",
    "frames_seen",
    "observed_interval_count",
    "ocr_chars",
    "temporal_relation_count",
    "object_count",
    "modality_count",
    "question_chars",
    "question_tokens",
    "option_count",
    "option_chars_mean",
    "retry_count",
    "error_count",
    "api_wait_ms",
    "local_runtime_ms",
    "duration_s",
    "fps",
)


def text(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    value = str(value).strip()
    return value if value else default


def number(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass
    return default


def integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def state_features(raw: dict[str, Any]) -> dict[str, Any]:
    value = raw.get("state_features")
    return value if isinstance(value, dict) else {}


def build_features(raw: dict[str, Any], prefix: list[str]) -> dict[str, Any]:
    state = state_features(raw)
    task = state.get("task") if isinstance(state.get("task"), dict) else {}
    if not task:
        task = raw.get("task_structure") if isinstance(raw.get("task_structure"), dict) else {}
    evidence = state.get("evidence") if isinstance(state.get("evidence"), dict) else {}
    prefix_state = state.get("prefix") if isinstance(state.get("prefix"), dict) else {}
    video = state.get("video") if isinstance(state.get("video"), dict) else {}
    modalities = sorted(text(x) for x in (task.get("required_modalities") or []))
    modality_counts = evidence.get("modality_counts")
    modality_count = sum(number(v) for v in modality_counts.values()) if isinstance(modality_counts, dict) else 0.0
    object_counts = evidence.get("object_counts")
    object_count = sum(number(v) for v in object_counts.values()) if isinstance(object_counts, dict) else 0.0
    categorical = {
        "baseline": text(raw.get("baseline")),
        "planner_model_id": text(raw.get("planner_model_id")),
        "model_stack_id": text(raw.get("model_stack_id")),
        "dataset": text(raw.get("dataset")),
        "question_type": text(task.get("question_type")),
        "temporal_scope": text(task.get("temporal_scope")),
        "answer_type": text(task.get("answer_type")),
        "domain": text(task.get("domain")),
        "sub_category": text(task.get("sub_category")),
        "required_modalities": "|".join(modalities) or "none",
        "last_activity": prefix[-1] if prefix else TOKEN_START,
        "position_bucket": f"p{integer(raw.get('position'))}",
        "prefix_tail2": "|".join(prefix[-2:]) if prefix else TOKEN_START,
        "prefix_tail3": "|".join(prefix[-3:]) if prefix else TOKEN_START,
        "planner_position": f"{text(raw.get('planner_model_id'))}|p{integer(raw.get('position'))}",
        "baseline_position": f"{text(raw.get('baseline'))}|p{integer(raw.get('position'))}",
        "task_signature": "|".join([
            text(task.get("question_type")),
            text(task.get("temporal_scope")),
            text(task.get("answer_type")),
            text(task.get("domain")),
        ]),
    }
    numeric = {
        "position": number(raw.get("position")),
        "coverage_ratio": number(evidence.get("coverage_ratio")),
        "frames_seen": number(evidence.get("frames_seen")),
        "observed_interval_count": number(evidence.get("observed_interval_count")),
        "ocr_chars": number(evidence.get("ocr_chars")),
        "temporal_relation_count": number(evidence.get("temporal_relation_count")),
        "object_count": object_count,
        "modality_count": modality_count,
        "question_chars": number(task.get("question_chars")),
        "question_tokens": number(task.get("question_tokens")),
        "option_count": number(task.get("option_count")),
        "option_chars_mean": number(task.get("option_chars_mean")),
        "retry_count": number(prefix_state.get("retry_count")),
        "error_count": number(prefix_state.get("error_count")),
        "api_wait_ms": number(prefix_state.get("api_wait_ms")),
        "local_runtime_ms": number(prefix_state.get("local_runtime_ms")),
        "duration_s": number(video.get("duration_s")),
        "fps": number(video.get("fps")),
    }
    return {"categorical": categorical, "numeric": numeric}


def validate_prefix(raw: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if raw.get("future_events_included_in_input") is not False:
        errors.append("future_events_included_in_input")
    if raw.get("ground_truth_included_in_input") is not False:
        errors.append("ground_truth_included_in_input")
    state = state_features(raw)
    if state.get("future_events_excluded") is not True:
        errors.append("state.future_events_excluded")
    if state.get("ground_truth_excluded") is not True:
        errors.append("state.ground_truth_excluded")
    guard = state.get("leakage_guard") if isinstance(state.get("leakage_guard"), dict) else {}
    if guard.get("answer_text_used_as_feature") is True:
        errors.append("answer_text_used_as_feature")
    if guard.get("video_id_used_as_feature") is True:
        errors.append("video_id_used_as_feature")
    if not text(raw.get("target_next_activity"), ""):
        errors.append("missing_target")
    return errors


def build_dataset(args: argparse.Namespace) -> None:
    source = Path(args.input)
    destination = Path(args.output)
    metadata_path = Path(args.metadata)
    raw_rows = read_jsonl(source)
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    leakage: Counter[str] = Counter()
    run_splits: dict[str, set[str]] = defaultdict(set)
    video_splits: dict[str, set[str]] = defaultdict(set)
    for raw in raw_rows:
        errors = validate_prefix(raw)
        if errors:
            leakage.update(errors)
            continue
        run_id = text(raw.get("run_id"), "")
        split = text(raw.get("split"), "unknown")
        video_id = text(raw.get("video_id"), "unknown")
        if not run_id:
            leakage["missing_run_id"] += 1
            continue
        by_run[run_id].append(raw)
        run_splits[run_id].add(split)
        video_splits[video_id].add(split)
    output_rows: list[dict[str, Any]] = []
    prefix_mismatches = 0
    for run_id, group in by_run.items():
        group.sort(key=lambda row: (integer(row.get("position")), text(row.get("prefix_id"))))
        targets = [text(row.get("target_next_activity"), "") for row in group]
        if any(not value for value in targets):
            leakage["empty_target"] += 1
            continue
        for index, raw in enumerate(group):
            prefix = [text(value, "unknown") for value in (raw.get("prefix_activities") or [])]
            expected_prefix = targets[:index]
            if prefix != expected_prefix:
                prefix_mismatches += 1
            features = build_features(raw, prefix)
            item: dict[str, Any] = {
                "schema_version": "neural-prefix-v0.1",
                "run_id": run_id,
                "video_id": text(raw.get("video_id")),
                "split": text(raw.get("split"), "unknown"),
                "position": integer(raw.get("position")),
                "prefix_activities": prefix,
                "features": features,
                "source_trace_sha256": text(raw.get("source_trace_sha256"), ""),
                "prefix_id": text(raw.get("prefix_id"), ""),
                "target_next_raw_action": text(raw.get("target_next_raw_action"), ""),
            }
            for horizon in HORIZONS:
                future_index = index + horizon - 1
                item[f"target_h{horizon}"] = targets[future_index] if future_index < len(targets) else None
            output_rows.append(item)
    output_rows.sort(key=lambda row: (row["split"], row["run_id"], row["position"]))
    count = write_jsonl(destination, output_rows)
    split_counts = Counter(row["split"] for row in output_rows)
    target_counts = {str(h): Counter(row[f"target_h{h}"] for row in output_rows if row[f"target_h{h}"]) for h in HORIZONS}
    metadata = {
        "schema_version": "neural-prefix-meta-v0.1",
        "source": str(source),
        "output": str(destination),
        "rows": count,
        "runs": len(by_run),
        "videos": len(video_splits),
        "split_counts": dict(split_counts),
        "run_split_conflicts": {k: sorted(v) for k, v in run_splits.items() if len(v) > 1},
        "video_split_conflicts": {k: sorted(v) for k, v in video_splits.items() if len(v) > 1},
        "prefix_mismatches": prefix_mismatches,
        "leakage_rejected": dict(leakage),
        "target_counts": {h: dict(counter) for h, counter in target_counts.items()},
        "input_contract": {
            "video_id_stored_only_as_metadata": True,
            "remaining_steps_excluded": True,
            "remaining_runtime_ms_excluded": True,
            "future_events_excluded": True,
            "ground_truth_excluded": True,
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if metadata["run_split_conflicts"] or metadata["video_split_conflicts"] or leakage:
        raise RuntimeError(f"leakage/quality gate failed: {json.dumps(metadata, ensure_ascii=False)}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_vocab(rows: list[dict[str, Any]]) -> dict[str, Any]:
    train = [row for row in rows if row["split"] == "train"]
    token_values = {TOKEN_START}
    label_values: set[str] = set()
    cat_values: dict[str, set[str]] = {key: set() for key in CAT_KEYS}
    for row in train:
        token_values.update(row["prefix_activities"])
        for horizon in HORIZONS:
            target = row.get(f"target_h{horizon}")
            if target:
                token_values.add(target)
                label_values.add(target)
        categorical = row["features"]["categorical"]
        for key in CAT_KEYS:
            cat_values[key].add(text(categorical.get(key)))
    token_vocab = {TOKEN_PAD: 0, TOKEN_START: 1, TOKEN_UNK: 2}
    for value in sorted(token_values):
        if value not in token_vocab:
            token_vocab[value] = len(token_vocab)
    label_vocab = {value: index for index, value in enumerate(sorted(label_values))}
    cat_vocabs: dict[str, dict[str, int]] = {}
    for key in CAT_KEYS:
        values = {"__UNK__"} | cat_values[key]
        cat_vocabs[key] = {"__UNK__": 0}
        for value in sorted(values):
            if value not in cat_vocabs[key]:
                cat_vocabs[key][value] = len(cat_vocabs[key])
    numeric_stats: dict[str, dict[str, float]] = {}
    for key in NUM_KEYS:
        values = [number(row["features"]["numeric"].get(key)) for row in train]
        mean = sum(values) / max(len(values), 1)
        variance = sum((value - mean) ** 2 for value in values) / max(len(values), 1)
        numeric_stats[key] = {"mean": mean, "std": math.sqrt(variance) or 1.0}
    teacher_dim = 0
    for row in train:
        value = row.get("teacher_probs")
        if isinstance(value, list):
            teacher_dim = len(value)
            break
    return {
        "token_vocab": token_vocab,
        "label_vocab": label_vocab,
        "cat_vocabs": cat_vocabs,
        "numeric_stats": numeric_stats,
        "teacher_dim": teacher_dim,
    }


class TraceDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]], artifacts: dict[str, Any]) -> None:
        self.rows = rows
        self.artifacts = artifacts
        self.token_vocab = artifacts["token_vocab"]
        self.label_vocab = artifacts["label_vocab"]
        self.cat_vocabs = artifacts["cat_vocabs"]
        self.numeric_stats = artifacts["numeric_stats"]

    def __len__(self) -> int:
        return len(self.rows)

    def _token(self, value: str) -> int:
        return int(self.token_vocab.get(value, self.token_vocab[TOKEN_UNK]))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        prefix = row["prefix_activities"]
        sequence = [TOKEN_START] + prefix
        token_ids = [self._token(value) for value in sequence]
        categorical = row["features"]["categorical"]
        cat_ids = [self.cat_vocabs[key].get(text(categorical.get(key)), 0) for key in CAT_KEYS]
        numeric = []
        for key in NUM_KEYS:
            stats = self.numeric_stats[key]
            value = number(row["features"]["numeric"].get(key))
            numeric.append((value - stats["mean"]) / stats["std"])
        targets = []
        for horizon in HORIZONS:
            value = row.get(f"target_h{horizon}")
            targets.append(self.label_vocab.get(value, -1) if value else -1)
        teacher_values = row.get("teacher_probs")
        if not isinstance(teacher_values, list) or len(teacher_values) != int(self.artifacts.get("teacher_dim", 0)):
            teacher_values = [0.0] * int(self.artifacts.get("teacher_dim", 0))
        return {
            "index": index,
            "tokens": torch.tensor(token_ids, dtype=torch.long),
            "cats": torch.tensor(cat_ids, dtype=torch.long),
            "numeric": torch.tensor(numeric, dtype=torch.float32),
            "teacher": torch.tensor([number(value) for value in teacher_values], dtype=torch.float32),
            "targets": torch.tensor(targets, dtype=torch.long),
            "length": len(token_ids),
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_length = max(item["length"] for item in batch)
    tokens = torch.zeros((len(batch), max_length), dtype=torch.long)
    lengths = []
    for index, item in enumerate(batch):
        tokens[index, : item["length"]] = item["tokens"]
        lengths.append(item["length"])
    return {
        "index": torch.tensor([item["index"] for item in batch], dtype=torch.long),
        "tokens": tokens,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "cats": torch.stack([item["cats"] for item in batch]),
        "numeric": torch.stack([item["numeric"] for item in batch]),
        "teacher": torch.stack([item["teacher"] for item in batch]),
        "targets": torch.stack([item["targets"] for item in batch]),
    }


class MultiHorizonGRU(nn.Module):
    def __init__(self, token_count: int, cat_sizes: list[int], numeric_dim: int, label_count: int, teacher_dim: int = 0, teacher_prior_scale: float = 0.0) -> None:
        super().__init__()
        self.teacher_dim = teacher_dim
        self.teacher_prior_scale = float(teacher_prior_scale)
        self.token_embedding = nn.Embedding(token_count, 96, padding_idx=0)
        self.gru = nn.GRU(96, 192, num_layers=2, dropout=0.1, batch_first=True)
        self.cat_embeddings = nn.ModuleList(nn.Embedding(size, 16) for size in cat_sizes)
        static_dim = 16 * len(cat_sizes) + numeric_dim + teacher_dim
        self.static = nn.Sequential(nn.Linear(static_dim, 128), nn.GELU(), nn.Dropout(0.1))
        self.fusion = nn.Sequential(nn.Linear(192 + 128, 256), nn.GELU(), nn.Dropout(0.1))
        self.heads = nn.ModuleDict({str(h): nn.Linear(256, label_count) for h in HORIZONS})
        if self.teacher_dim and self.teacher_prior_scale:
            for head in self.heads.values():
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def forward(self, tokens: Tensor, lengths: Tensor, cats: Tensor, numeric: Tensor, teacher: Tensor | None = None) -> dict[str, Tensor]:
        embedded = self.token_embedding(tokens)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        sequence_state = hidden[-1]
        categorical_state = torch.cat([embedding(cats[:, index]) for index, embedding in enumerate(self.cat_embeddings)], dim=-1)
        if self.teacher_dim:
            if teacher is None:
                teacher = torch.zeros((tokens.shape[0], self.teacher_dim), dtype=numeric.dtype, device=numeric.device)
            static_input = torch.cat([categorical_state, numeric, teacher], dim=-1)
        else:
            static_input = torch.cat([categorical_state, numeric], dim=-1)
        static_state = self.static(static_input)
        fused = self.fusion(torch.cat([sequence_state, static_state], dim=-1))
        outputs = {str(h): self.heads[str(h)](fused) for h in HORIZONS}
        if self.teacher_dim and self.teacher_prior_scale:
            prior = torch.log(teacher.clamp_min(1e-4))
            outputs["1"] = outputs["1"] + self.teacher_prior_scale * prior
        return outputs


def class_weights(rows: list[dict[str, Any]], label_vocab: dict[str, int]) -> Tensor:
    counts = Counter(row.get("target_h1") for row in rows if row.get("target_h1") in label_vocab)
    total = sum(counts.values())
    values = []
    for label, _ in sorted(label_vocab.items(), key=lambda item: item[1]):
        count = max(counts.get(label, 1), 1)
        weight = math.sqrt(total / max(len(label_vocab) * count, 1))
        values.append(min(max(weight, 0.5), 3.0))
    return torch.tensor(values, dtype=torch.float32)


def loss_for_batch(model: MultiHorizonGRU, batch: dict[str, Tensor], weights: Tensor, label_smoothing: float = 0.05) -> Tensor:
    outputs = model(batch["tokens"], batch["lengths"], batch["cats"], batch["numeric"], batch["teacher"])
    losses = []
    horizon_weights = {1: 1.0, 2: 0.8, 3: 0.7, 4: 0.5, 5: 0.4}
    for position, horizon in enumerate(HORIZONS):
        target = batch["targets"][:, position]
        mask = target >= 0
        if bool(mask.any()):
            losses.append(horizon_weights[horizon] * F.cross_entropy(outputs[str(horizon)][mask], target[mask], weight=weights, label_smoothing=label_smoothing))
    return sum(losses) / max(len(losses), 1)


def transition_graph(rows: list[dict[str, Any]]) -> tuple[dict[str, Counter[str]], Counter[str]]:
    edges: dict[str, Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    for row in rows:
        previous = row["prefix_activities"][-1] if row["prefix_activities"] else TOKEN_START
        target = row.get("target_h1")
        if target:
            edges[previous][target] += 1
            global_counts[target] += 1
    return dict(edges), global_counts


def path_decode(start: str, log_probs: dict[int, Tensor], edges: dict[str, Counter[str]], labels: list[str]) -> list[str]:
    paths: dict[str, tuple[float, list[str]]] = {start: (0.0, [])}
    label_to_index = {label: index for index, label in enumerate(labels)}
    for horizon in HORIZONS:
        next_paths: dict[str, tuple[float, list[str]]] = {}
        for previous, (score, path) in paths.items():
            allowed = list(edges.get(previous, Counter()).keys())
            if not allowed:
                allowed = labels
            for candidate in allowed:
                if candidate not in label_to_index:
                    continue
                index = label_to_index[candidate]
                candidate_score = score + float(log_probs[horizon][index])
                old = next_paths.get(candidate)
                if old is None or candidate_score > old[0]:
                    next_paths[candidate] = (candidate_score, path + [candidate])
        if not next_paths:
            break
        paths = dict(sorted(next_paths.items(), key=lambda item: item[1][0], reverse=True)[:64])
    if not paths:
        return []
    return max(paths.values(), key=lambda value: value[0])[1]


def baseline_path(start: str, edges: dict[str, Counter[str]], global_counts: Counter[str]) -> list[str]:
    path: list[str] = []
    previous = start
    for _ in HORIZONS:
        counts = edges.get(previous) or global_counts
        if not counts:
            break
        candidate = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        path.append(candidate)
        previous = candidate
    return path


def metric_block(values: list[tuple[float, bool]], nll_values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "top1": 0.0, "top3": 0.0, "mrr": 0.0, "nll": 0.0}
    ranks = [rank for rank, _ in values]
    return {
        "n": len(values),
        "top1": sum(rank <= 1 for rank in ranks) / len(ranks),
        "top3": sum(rank <= 3 for rank in ranks) / len(ranks),
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
        "nll": sum(nll_values) / max(len(nll_values), 1),
    }


def evaluate_rows(rows: list[dict[str, Any]], logits_by_index: dict[int, dict[int, Tensor]], edges: dict[str, Counter[str]], global_counts: Counter[str], label_vocab: dict[str, int]) -> dict[str, Any]:
    labels = [label for label, _ in sorted(label_vocab.items(), key=lambda item: item[1])]
    direct: dict[str, list[tuple[float, bool]]] = {str(h): [] for h in HORIZONS}
    nlls: dict[str, list[float]] = {str(h): [] for h in HORIZONS}
    ece_data: list[tuple[float, bool]] = []
    path_hits: Counter[str] = Counter()
    path_counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        outputs = logits_by_index.get(index)
        if outputs is None:
            continue
        log_probs = {h: torch.log_softmax(outputs[h], dim=-1) for h in HORIZONS}
        for horizon in HORIZONS:
            target = row.get(f"target_h{horizon}")
            target_index = label_vocab.get(target, -1) if target else -1
            if target_index < 0:
                continue
            values, indices = torch.sort(log_probs[horizon], descending=True)
            rank_positions = (indices == target_index).nonzero(as_tuple=False)
            if rank_positions.numel() == 0:
                continue
            rank = int(rank_positions[0].item()) + 1
            direct[str(horizon)].append((rank, rank <= 1))
            nlls[str(horizon)].append(float(-log_probs[horizon][target_index]))
            if horizon == 1:
                confidence = float(torch.exp(values[0]))
                ece_data.append((confidence, rank == 1))
        start = row["prefix_activities"][-1] if row["prefix_activities"] else TOKEN_START
        predicted = path_decode(start, log_probs, edges, labels)
        for k in (1, 3, 5):
            targets = [row.get(f"target_h{h}") for h in range(1, k + 1)]
            if all(target in label_vocab for target in targets):
                path_counts[str(k)] += 1
                if predicted[:k] == targets:
                    path_hits[str(k)] += 1
    ece = 0.0
    if ece_data:
        for lower in [x / 10 for x in range(10)]:
            upper = lower + 0.1
            bucket = [(confidence, correct) for confidence, correct in ece_data if lower <= confidence < upper or (upper >= 1.0 and confidence <= upper)]
            if bucket:
                ece += len(bucket) / len(ece_data) * abs(sum(confidence for confidence, _ in bucket) / len(bucket) - sum(correct for _, correct in bucket) / len(bucket))
    return {
        "classification": {key: metric_block(value, nlls[key]) for key, value in direct.items()},
        "path": {f"prefix_hit@{key}": path_hits[key] / max(path_counts[key], 1) for key in ("1", "3", "5")},
        "path_n": dict(path_counts),
        "ece_h1": ece,
    }


def evaluate_baseline(rows: list[dict[str, Any]], edges: dict[str, Counter[str]], global_counts: Counter[str], label_vocab: dict[str, int]) -> dict[str, Any]:
    labels = set(label_vocab)
    hits: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for row in rows:
        start = row["prefix_activities"][-1] if row["prefix_activities"] else TOKEN_START
        predicted = baseline_path(start, edges, global_counts)
        for k in (1, 3, 5):
            targets = [row.get(f"target_h{h}") for h in range(1, k + 1)]
            if all(target in labels for target in targets):
                counts[str(k)] += 1
                if predicted[:k] == targets:
                    hits[str(k)] += 1
    return {
        "path": {f"prefix_hit@{key}": hits[key] / max(counts[key], 1) for key in ("1", "3", "5")},
        "path_n": dict(counts),
    }


def collect_logits(model: MultiHorizonGRU, rows: list[dict[str, Any]], artifacts: dict[str, Any], device: torch.device, batch_size: int) -> dict[int, dict[int, Tensor]]:
    dataset = TraceDataset(rows, artifacts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    model.eval()
    result: dict[int, dict[int, Tensor]] = {}
    with torch.no_grad():
        for batch in loader:
            tensors = {key: value.to(device) for key, value in batch.items() if key in {"tokens", "lengths", "cats", "numeric", "teacher"}}
            outputs = model(**tensors)
            for offset, index in enumerate(batch["index"].tolist()):
                result[index] = {h: outputs[str(h)][offset].detach().cpu() for h in HORIZONS}
    return result


def autoregressive_path(model: MultiHorizonGRU, row: dict[str, Any], artifacts: dict[str, Any], edges: dict[str, Counter[str]], device: torch.device) -> list[str]:
    labels = list(artifacts["label_vocab"])
    working = {
        **row,
        "prefix_activities": list(row["prefix_activities"]),
        "features": {
            "categorical": dict(row["features"]["categorical"]),
            "numeric": dict(row["features"]["numeric"]),
        },
    }
    predicted: list[str] = []
    model.eval()
    for _ in HORIZONS:
        item = TraceDataset([working], artifacts)[0]
        batch = collate([item])
        tensors = {key: value.to(device) for key, value in batch.items() if key in {"tokens", "lengths", "cats", "numeric", "teacher"}}
        with torch.no_grad():
            output = model(**tensors)["1"][0]
        log_probs = torch.log_softmax(output, dim=-1)
        previous = working["prefix_activities"][-1] if working["prefix_activities"] else TOKEN_START
        allowed = list(edges.get(previous, Counter()).keys()) or labels
        allowed = [candidate for candidate in allowed if candidate in artifacts["label_vocab"]]
        if not allowed:
            allowed = labels
        choice = max(allowed, key=lambda candidate: float(log_probs[artifacts["label_vocab"][candidate]]))
        predicted.append(choice)
        working["prefix_activities"].append(choice)
        working["features"]["categorical"]["last_activity"] = choice
        working["features"]["numeric"]["position"] = number(working["features"]["numeric"].get("position")) + 1.0
        if choice == "__END__":
            break
    return predicted


def evaluate_autoregressive_path(model: MultiHorizonGRU, rows: list[dict[str, Any]], artifacts: dict[str, Any], edges: dict[str, Counter[str]], device: torch.device) -> dict[str, Any]:
    hits: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for row in rows:
        predicted = autoregressive_path(model, row, artifacts, edges, device)
        for k in (1, 3, 5):
            targets = [row.get(f"target_h{h}") for h in range(1, k + 1)]
            if all(target in artifacts["label_vocab"] for target in targets):
                counts[str(k)] += 1
                if predicted[:k] == targets:
                    hits[str(k)] += 1
    return {
        "path": {f"prefix_hit@{key}": hits[key] / max(counts[key], 1) for key in ("1", "3", "5")},
        "path_n": dict(counts),
    }


def evaluate_model(model: MultiHorizonGRU, rows: list[dict[str, Any]], artifacts: dict[str, Any], edges: dict[str, Counter[str]], global_counts: Counter[str], device: torch.device, batch_size: int) -> dict[str, Any]:
    logits = collect_logits(model, rows, artifacts, device, batch_size)
    metrics = evaluate_rows(rows, logits, edges, global_counts, artifacts["label_vocab"])
    metrics["direct_graph_path"] = {"path": metrics["path"], "path_n": metrics["path_n"]}
    metrics["autoregressive_path"] = evaluate_autoregressive_path(model, rows, artifacts, edges, device)
    metrics["path"] = metrics["autoregressive_path"]["path"]
    metrics["path_n"] = metrics["autoregressive_path"]["path_n"]
    return metrics



def load_pretrained_gru(model: MultiHorizonGRU, checkpoint_path: Path) -> dict[str, Any]:
    """Load only shape-compatible recurrent encoder weights from public pretraining."""
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"invalid public checkpoint: {checkpoint_path}")
    source = payload.get("encoder_state_dict")
    if not isinstance(source, dict):
        source = payload.get("state_dict")
    if not isinstance(source, dict):
        raise ValueError(f"checkpoint has no encoder_state_dict/state_dict: {checkpoint_path}")
    current = model.state_dict()
    loaded: list[str] = []
    skipped: dict[str, str] = {}
    for key, value in source.items():
        if key not in current:
            skipped[key] = "missing_in_own_model"
            continue
        if not isinstance(value, Tensor) or tuple(value.shape) != tuple(current[key].shape):
            skipped[key] = "shape_mismatch"
            continue
        current[key] = value
        loaded.append(key)
    model.load_state_dict(current)
    return {
        "initialized": bool(loaded),
        "path": str(checkpoint_path),
        "loaded_keys": sorted(loaded),
        "skipped": skipped,
        "source_schema_version": payload.get("schema_version"),
        "note": "public labels and public token embedding are not reused; only recurrent GRU weights are shape-compatible.",
    }


def train_seed(seed: int, rows: list[dict[str, Any]], artifacts: dict[str, Any], output_dir: Path, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    seed_everything(seed)
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    test_rows = [row for row in rows if row["split"] == "test"]
    edges, global_counts = transition_graph(train_rows)
    cat_sizes = [len(artifacts["cat_vocabs"][key]) for key in CAT_KEYS]
    model = MultiHorizonGRU(
        len(artifacts["token_vocab"]),
        cat_sizes,
        len(NUM_KEYS),
        len(artifacts["label_vocab"]),
        int(artifacts.get("teacher_dim", 0)),
        float(args.teacher_prior_scale),
    ).to(device)
    transfer_info = {"initialized": False, "path": None, "loaded_keys": [], "skipped": {}, "note": "no public checkpoint supplied"}
    if args.pretrained_gru:
        transfer_info = load_pretrained_gru(model, Path(args.pretrained_gru))
    train_loader = DataLoader(TraceDataset(train_rows, artifacts), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    weights = (torch.ones(len(artifacts["label_vocab"]), dtype=torch.float32) if args.no_class_weights else class_weights(train_rows, artifacts["label_vocab"])).to(device)
    best_score = -float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for_batch(model, batch, weights, args.label_smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach())
        val_metrics = evaluate_model(model, validation_rows, artifacts, edges, global_counts, device, args.batch_size)
        path = val_metrics["path"]
        if args.selection == "h1":
            score = val_metrics["classification"]["1"]["top1"] - 0.02 * val_metrics["classification"]["1"]["nll"]
        elif args.selection == "joint":
            score = 0.40 * val_metrics["classification"]["1"]["top1"] + 0.25 * path["prefix_hit@3"] + 0.35 * path["prefix_hit@5"]
        else:
            score = 0.45 * path["prefix_hit@3"] + 0.55 * path["prefix_hit@5"]
        record = {"epoch": epoch, "loss": epoch_loss / max(len(train_loader), 1), "validation": val_metrics, "selection_score": score}
        history.append(record)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch - best_epoch >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state)
    validation_metrics = evaluate_model(model, validation_rows, artifacts, edges, global_counts, device, args.batch_size)
    test_metrics = evaluate_model(model, test_rows, artifacts, edges, global_counts, device, args.batch_size)
    baseline = {
        "validation": evaluate_baseline(validation_rows, edges, global_counts, artifacts["label_vocab"]),
        "test": evaluate_baseline(test_rows, edges, global_counts, artifacts["label_vocab"]),
    }
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "artifacts": artifacts, "seed": seed}, seed_dir / "checkpoint_best.pt")
    report = {
        "schema_version": "neural-trace-predictor-v0.1",
        "seed": seed,
        "device": str(device),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_validation_selection_score": best_score,
        "rows": {"train": len(train_rows), "validation": len(validation_rows), "test": len(test_rows)},
        "transfer": transfer_info,
        "baseline": baseline,
        "neural": {"validation": validation_metrics, "test": test_metrics},
        "history": history,
    }
    (seed_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def train(args: argparse.Namespace) -> None:
    source = Path(args.input)
    output_dir = Path(args.output_dir)
    rows = read_jsonl(source)
    artifacts = build_vocab(rows)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    reports = [train_seed(seed, rows, artifacts, output_dir, args, device) for seed in seeds]
    summary = {
        "schema_version": "neural-trace-predictor-summary-v0.1",
        "input": str(source),
        "output_dir": str(output_dir),
        "device": str(device),
        "seeds": seeds,
        "label_count": len(artifacts["label_vocab"]),
        "rows": {split: sum(row["split"] == split for row in rows) for split in ("train", "validation", "test")},
        "reports": [
            {
                "seed": report["seed"],
                "best_epoch": report["best_epoch"],
                "test_neural": report["neural"]["test"],
                "test_baseline": report["baseline"]["test"],
                "transfer": report.get("transfer"),
            }
            for report in reports
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "artifacts.json").write_text(json.dumps(artifacts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("build", "train"), required=True)
    parser.add_argument("--input", default="/root/autodl-tmp/scheduler/results/processed/trace_enrichment_core_fixed_canonical_v02_20260804/prefix_samples_v0_2.jsonl")
    parser.add_argument("--output", default="/root/autodl-tmp/scheduler/results/processed/neural_prefix_dataset_v0_1/prefixes.jsonl")
    parser.add_argument("--metadata", default="/root/autodl-tmp/scheduler/results/processed/neural_prefix_dataset_v0_1/metadata.json")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/scheduler/results/processed/neural_trace_predictor_structured_v0_1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pretrained-gru", default="", help="public-event GRU checkpoint; only shape-compatible gru.* weights are loaded")
    parser.add_argument("--no-class-weights", action="store_true", help="use unweighted cross-entropy for top-1 optimization")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--teacher-prior-scale", type=float, default=0.0, help="add train-only teacher log-probability to H1 logits")
    parser.add_argument("--selection", choices=("path", "h1", "joint"), default="path")
    args = parser.parse_args()
    if args.mode == "build":
        build_dataset(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
