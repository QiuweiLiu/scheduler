
#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from tracing.analysis import neural_trace_predictor as n

HORIZONS = n.HORIZONS
VISUAL_KEYS = ("coverage_ratio", "frames_seen", "observed_interval_count", "ocr_chars", "temporal_relation_count", "object_count", "modality_count")
TASK_KEYS = ("question_chars", "question_tokens", "option_count", "option_chars_mean", "duration_s", "fps")
RESOURCE_KEYS = ("retry_count", "error_count", "api_wait_ms", "local_runtime_ms")

def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MultimodalDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]], artifacts: dict[str, Any], max_length: int) -> None:
        self.rows = rows
        self.base = n.TraceDataset(rows, artifacts)
        self.max_length = max_length
    def __len__(self) -> int:
        return len(self.rows)
    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[index]
        row = self.rows[index]
        item["remaining_length"] = min(max_length_for_row(row), self.max_length)
        return item

def max_length_for_row(row: dict[str, Any]) -> int:
    value = 0
    for h in HORIZONS:
        if row.get(f"target_h{h}"):
            value = h
    return max(value, 1)

def collate(batch: list[dict[str, Any]]) -> dict[str, Tensor]:
    out = n.collate(batch)
    out["remaining_length"] = torch.tensor([x["remaining_length"] for x in batch], dtype=torch.long)
    return out

def build_graph(rows: list[dict[str, Any]], artifacts: dict[str, Any]) -> Tensor:
    size = len(artifacts["token_vocab"])
    adj = torch.zeros((size, size), dtype=torch.float32)
    for row in rows:
        previous = row["prefix_activities"][-1] if row["prefix_activities"] else n.TOKEN_START
        target = row.get("target_h1")
        if not target:
            continue
        i = artifacts["token_vocab"].get(previous, artifacts["token_vocab"][n.TOKEN_UNK])
        j = artifacts["token_vocab"].get(target, artifacts["token_vocab"][n.TOKEN_UNK])
        adj[i, j] += 1.0
    adj += torch.eye(size)
    adj = adj / adj.sum(dim=1, keepdim=True).clamp_min(1.0)
    return adj

class MultimodalPredictor(nn.Module):
    def __init__(self, artifacts: dict[str, Any], graph_adj: Tensor, model_kind: str) -> None:
        super().__init__()
        if model_kind not in {"gru", "lstm", "gnn"}:
            raise ValueError(model_kind)
        self.model_kind = model_kind
        self.teacher_dim = int(artifacts.get("teacher_dim", 0))
        self.numeric_index = {key: i for i, key in enumerate(n.NUM_KEYS)}
        token_count = len(artifacts["token_vocab"])
        self.token_embedding = nn.Embedding(token_count, 96, padding_idx=0)
        if model_kind == "gru":
            self.sequence = nn.GRU(96, 192, num_layers=2, dropout=0.1, batch_first=True)
        elif model_kind == "lstm":
            self.sequence = nn.LSTM(96, 192, num_layers=2, dropout=0.1, batch_first=True)
        else:
            self.node_embedding = nn.Embedding(token_count, 192)
            self.graph_self = nn.Linear(192, 192)
            self.graph_neigh = nn.Linear(192, 192)
            self.register_buffer("graph_adj", graph_adj)
        self.cat_embeddings = nn.ModuleList(nn.Embedding(len(artifacts["cat_vocabs"][key]), 16) for key in n.CAT_KEYS)
        cat_dim = 16 * len(n.CAT_KEYS)
        self.visual_branch = nn.Sequential(nn.Linear(len(VISUAL_KEYS), 64), nn.GELU(), nn.Dropout(0.1))
        self.task_branch = nn.Sequential(nn.Linear(len(TASK_KEYS), 64), nn.GELU(), nn.Dropout(0.1))
        self.resource_branch = nn.Sequential(nn.Linear(len(RESOURCE_KEYS), 64), nn.GELU(), nn.Dropout(0.1))
        self.static_branch = nn.Sequential(nn.Linear(cat_dim + 192, 128), nn.GELU(), nn.Dropout(0.1))
        self.fusion = nn.Sequential(nn.Linear(192 + 64 + 64 + 64 + 128, 256), nn.GELU(), nn.Dropout(0.1))
        label_count = len(artifacts["label_vocab"])
        self.heads = nn.ModuleDict({str(h): nn.Linear(256, label_count) for h in HORIZONS})
        self.length_head = nn.Linear(256, 9)

    def encode(self, tokens: Tensor, lengths: Tensor, cats: Tensor, numeric: Tensor, teacher: Tensor | None) -> Tensor:
        if self.model_kind in {"gru", "lstm"}:
            embedded = self.token_embedding(tokens)
            packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, hidden = self.sequence(packed)
            if self.model_kind == "lstm":
                hidden = hidden[0]
            sequence_state = hidden[-1]
        else:
            node = self.node_embedding.weight
            for _ in range(2):
                neighbor = self.graph_adj @ node
                node = torch.tanh(self.graph_self(node) + self.graph_neigh(neighbor))
            ids = tokens.clamp_min(0).clamp_max(node.shape[0] - 1)
            states = node[ids]
            mask = (torch.arange(tokens.shape[1], device=tokens.device)[None, :] < lengths[:, None]).float().unsqueeze(-1)
            sequence_state = (states * mask).sum(dim=1) / lengths.float().clamp_min(1).unsqueeze(-1)
        categorical = torch.cat([embedding(cats[:, i]) for i, embedding in enumerate(self.cat_embeddings)], dim=-1)
        visual = numeric[:, [self.numeric_index[k] for k in VISUAL_KEYS]]
        task = numeric[:, [self.numeric_index[k] for k in TASK_KEYS]]
        resource = numeric[:, [self.numeric_index[k] for k in RESOURCE_KEYS]]
        static = self.static_branch(torch.cat([categorical, sequence_state], dim=-1))
        return self.fusion(torch.cat([sequence_state, self.visual_branch(visual), self.task_branch(task), self.resource_branch(resource), static], dim=-1))

    def forward(self, tokens: Tensor, lengths: Tensor, cats: Tensor, numeric: Tensor, teacher: Tensor | None = None) -> dict[str, Tensor]:
        state = self.encode(tokens, lengths, cats, numeric, teacher)
        output = {str(h): self.heads[str(h)](state) for h in HORIZONS}
        output["length"] = self.length_head(state)
        return output

def evaluate(model: MultimodalPredictor, rows: list[dict[str, Any]], artifacts: dict[str, Any], edges: dict[str, Counter[str]], device: torch.device, batch_size: int, write_details: Path | None = None) -> dict[str, Any]:
    loader = DataLoader(MultimodalDataset(rows, artifacts, 8), batch_size=batch_size, shuffle=False, collate_fn=collate)
    labels = [label for label, _ in sorted(artifacts["label_vocab"].items(), key=lambda x: x[1])]
    hits = Counter()
    counts = Counter()
    direct_hits = Counter()
    direct_counts = Counter()
    length_hits = 0
    length_count = 0
    details: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            tensors = {key: batch[key].to(device) for key in ("tokens", "lengths", "cats", "numeric", "teacher")}
            output = model(**tensors)
            for offset, index in enumerate(batch["index"].tolist()):
                row = rows[int(index)]
                log_probs = {h: torch.log_softmax(output[str(h)][offset].detach().cpu(), dim=-1) for h in HORIZONS}
                start = row["prefix_activities"][-1] if row["prefix_activities"] else n.TOKEN_START
                predicted_path = n.path_decode(start, log_probs, edges, labels)
                for h in (1, 3, 5):
                    targets = [row.get(f"target_h{i}") for i in range(1, h + 1)]
                    if all(target in artifacts["label_vocab"] for target in targets):
                        counts[str(h)] += 1
                        hits[str(h)] += int(predicted_path[:h] == targets)
                for h in HORIZONS:
                    target = row.get(f"target_h{h}")
                    if target in artifacts["label_vocab"]:
                        direct_counts[str(h)] += 1
                        direct_hits[str(h)] += int(labels[int(output[str(h)][offset].argmax())] == target)
                predicted_len = int(output["length"][offset].argmax().item())
                true_len = max_length_for_row(row)
                length_count += 1
                length_hits += int(predicted_len == true_len)
                if write_details is not None:
                    top: dict[str, Any] = {}
                    for h in HORIZONS:
                        probs = torch.softmax(output[str(h)][offset].detach().cpu(), dim=-1)
                        vals, inds = probs.topk(min(3, probs.numel()))
                        top[str(h)] = {
                            "candidates": [{"activity": labels[int(i)], "probability": float(v)} for v, i in zip(vals, inds)],
                            "entropy": float(-(probs * probs.clamp_min(1e-8).log()).sum()),
                        }
                    len_probs = torch.softmax(output["length"][offset].detach().cpu(), dim=-1)
                    details.append({
                        "run_id": row["run_id"], "position": row["position"], "prefix_activities": row["prefix_activities"],
                        "predicted_path": predicted_path, "predicted_remaining_steps": predicted_len,
                        "true_remaining_steps_for_audit": true_len, "horizon_predictions": top,
                        "path_confidence": float(torch.exp(sum(log_probs[h][artifacts["label_vocab"][predicted_path[h-1]]] for h in (1, 3, 5) if len(predicted_path) >= h) / max(len(predicted_path), 1))),
                        "length_entropy": float(-(len_probs * len_probs.clamp_min(1e-8).log()).sum()),
                    })
    metrics = {
        "path": {f"prefix_hit@{h}": hits[str(h)] / max(counts[str(h)], 1) for h in (1, 3, 5)},
        "path_n": dict(counts),
        "direct_top1": {str(h): direct_hits[str(h)] / max(direct_counts[str(h)], 1) for h in HORIZONS},
        "length_exact": length_hits / max(length_count, 1),
        "length_n": length_count,
    }
    if write_details is not None:
        write_details.parent.mkdir(parents=True, exist_ok=True)
        write_details.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in details), encoding="utf-8")
    return metrics

def loss_for_batch(model: MultimodalPredictor, batch: dict[str, Tensor], device: torch.device, class_weight: Tensor) -> Tensor:
    tensors = {key: batch[key].to(device) for key in ("tokens", "lengths", "cats", "numeric", "teacher")}
    output = model(**tensors)
    losses = []
    for h in HORIZONS:
        target = batch["targets"][:, h - 1].to(device)
        valid = target >= 0
        if bool(valid.any()):
            weight = 1.0 if h == 1 else (0.8 if h == 2 else (0.7 if h == 3 else 0.6))
            losses.append(weight * F.cross_entropy(output[str(h)][valid], target[valid], weight=class_weight, label_smoothing=0.03))
    length_target = batch["remaining_length"].to(device)
    losses.append(0.15 * F.cross_entropy(output["length"], length_target.clamp(0, 8)))
    return sum(losses) / max(len(losses), 1)

def train_one(args: argparse.Namespace, model_kind: str) -> dict[str, Any]:
    rows = n.read_jsonl(Path(args.input))
    artifacts = n.build_vocab(rows)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validation"]
    test_rows = [r for r in rows if r["split"] == "test"]
    artifacts["graph_adj"] = build_graph(train_rows, artifacts).tolist()
    edges, _ = n.transition_graph(train_rows)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    seed_everything(args.seed)
    graph_adj = torch.tensor(artifacts["graph_adj"], dtype=torch.float32)
    model = MultimodalPredictor(artifacts, graph_adj, model_kind).to(device)
    loader = DataLoader(MultimodalDataset(train_rows, artifacts, 8), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    weights = n.class_weights(train_rows, artifacts["label_vocab"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    history = []
    output_dir = Path(args.output_dir) / model_kind
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for_batch(model, batch, device, weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
        val = evaluate(model, val_rows, artifacts, edges, device, args.batch_size)
        score = 0.45 * val["path"]["prefix_hit@3"] + 0.55 * val["path"]["prefix_hit@5"]
        record = {"epoch": epoch, "loss": total / max(len(loader), 1), "validation": val, "selection_score": score}
        history.append(record)
        print(json.dumps({"model": model_kind, "seed": args.seed, **record}, ensure_ascii=False), flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch - best_epoch >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state)
    val = evaluate(model, val_rows, artifacts, edges, device, args.batch_size)
    test = evaluate(model, test_rows, artifacts, edges, device, args.batch_size, output_dir / "test_predictions.jsonl")
    report = {"schema_version": "multimodal-predictor-v0.1", "model": model_kind, "seed": args.seed, "input": args.input, "best_epoch": best_epoch, "best_validation_selection_score": best_score, "rows": {"train": len(train_rows), "validation": len(val_rows), "test": len(test_rows)}, "validation": val, "test": test, "feature_branches": {"visual": list(VISUAL_KEYS), "task": list(TASK_KEYS), "resource": list(RESOURCE_KEYS)}, "leakage_contract": {"future_events_excluded": True, "ground_truth_excluded": True, "video_id_used_as_feature": False, "remaining_length_used_as_input": False}, "history": history}
    torch.save({"schema_version": "multimodal-predictor-checkpoint-v0.1", "state_dict": best_state, "artifacts": artifacts, "model": model_kind, "seed": args.seed}, output_dir / "checkpoint_best.pt")
    (output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"model": model_kind, "final": {"validation": val, "test": test}}, ensure_ascii=False), flush=True)
    return report

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/root/autodl-tmp/scheduler/results/processed/neural_prefix_dataset_v0_1/prefixes.jsonl")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/scheduler/results/processed/multimodal_predictor_v0_1")
    parser.add_argument("--model", choices=("gru", "lstm", "gnn", "all"), default="all")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    kinds = ("gru", "lstm", "gnn") if args.model == "all" else (args.model,)
    reports = [train_one(args, kind) for kind in kinds]
    summary = {"schema_version": "multimodal-predictor-summary-v0.1", "reports": reports, "config": vars(args)}
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
