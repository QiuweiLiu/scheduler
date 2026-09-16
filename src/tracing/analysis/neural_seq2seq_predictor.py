#!/usr/bin/env python3
"""Causal seq2seq neural predictor trained for exact multi-step paths.

Stage N1 public GRU weights are transferred into the encoder. Stage N2 uses
own Video Agent prefix rows and a teacher-forced future decoder. The graph used
for beam decoding is fit on own train rows only.
"""
from __future__ import annotations
import argparse, copy, json, math, os, random, sys
from collections import Counter
from pathlib import Path
from typing import Any
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "/root/autodl-tmp/scheduler")
from tracing.analysis import neural_trace_predictor as n

HORIZONS = n.HORIZONS
BOS_OFFSET = 0

def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Seq2SeqGRU(nn.Module):
    def __init__(self, token_count: int, cat_sizes: list[int], numeric_dim: int, label_count: int, teacher_dim: int) -> None:
        super().__init__()
        self.label_count = label_count
        self.teacher_dim = teacher_dim
        self.bos_id = label_count
        self.token_embedding = nn.Embedding(token_count, 96, padding_idx=0)
        # Keep this name/shape equal to the public event pretraining encoder.
        self.gru = nn.GRU(96, 192, num_layers=2, dropout=0.1, batch_first=True)
        self.cat_embeddings = nn.ModuleList(nn.Embedding(size, 16) for size in cat_sizes)
        static_dim = 16 * len(cat_sizes) + numeric_dim + teacher_dim
        self.static = nn.Sequential(nn.Linear(static_dim, 128), nn.GELU(), nn.Dropout(0.1))
        self.fusion = nn.Sequential(nn.Linear(192 + 128, 256), nn.GELU(), nn.Dropout(0.1))
        self.init_proj = nn.Sequential(nn.Linear(256, 384), nn.Tanh())
        self.decoder_embedding = nn.Embedding(label_count + 1, 96)
        self.decoder = nn.GRU(96, 192, num_layers=2, dropout=0.1, batch_first=True)
        self.head = nn.Linear(192, label_count)

    def encode(self, tokens: Tensor, lengths: Tensor, cats: Tensor, numeric: Tensor, teacher: Tensor) -> Tensor:
        packed = n.pack_padded_sequence(self.token_embedding(tokens), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        sequence_state = hidden[-1]
        categorical_state = torch.cat([embedding(cats[:, i]) for i, embedding in enumerate(self.cat_embeddings)], dim=-1)
        if self.teacher_dim:
            static_input = torch.cat([categorical_state, numeric, teacher], dim=-1)
        else:
            static_input = torch.cat([categorical_state, numeric], dim=-1)
        fused = self.fusion(torch.cat([sequence_state, self.static(static_input)], dim=-1))
        return self.init_proj(fused).view(-1, 2, 192).transpose(0, 1).contiguous()

    def step(self, previous: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        embedded = self.decoder_embedding(previous).unsqueeze(1)
        output, hidden = self.decoder(embedded, hidden)
        return self.head(output[:, 0]), hidden

    def teacher_forward(self, tokens: Tensor, lengths: Tensor, cats: Tensor, numeric: Tensor, teacher: Tensor, targets: Tensor, forcing_ratio: float) -> Tensor:
        hidden = self.encode(tokens, lengths, cats, numeric, teacher)
        previous = torch.full((tokens.shape[0],), self.bos_id, dtype=torch.long, device=tokens.device)
        outputs = []
        for position, _horizon in enumerate(HORIZONS):
            logits, hidden = self.step(previous, hidden)
            outputs.append(logits)
            gold = targets[:, position]
            valid = gold >= 0
            use_gold = (torch.rand(tokens.shape[0], device=tokens.device) < forcing_ratio) & valid
            prediction = logits.argmax(dim=-1)
            previous = torch.where(use_gold, gold, prediction)
        return torch.stack(outputs, dim=1)

def load_public_encoder(model: Seq2SeqGRU, checkpoint: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu")
    source = payload.get("encoder_state_dict") if isinstance(payload, dict) else None
    if not isinstance(source, dict):
        raise ValueError(f"public checkpoint has no encoder_state_dict: {checkpoint}")
    current = model.state_dict()
    loaded, skipped = [], {}
    for key, value in source.items():
        if key not in current:
            skipped[key] = "missing"
        elif tuple(current[key].shape) != tuple(value.shape):
            skipped[key] = "shape_mismatch"
        else:
            current[key] = value
            loaded.append(key)
    model.load_state_dict(current)
    return {"initialized": bool(loaded), "path": str(checkpoint), "loaded_keys": sorted(loaded), "skipped": skipped, "source_schema_version": payload.get("schema_version")}

def batch_inputs(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items() if key in {"tokens", "lengths", "cats", "numeric", "teacher"}}

def loss_batch(model: Seq2SeqGRU, batch: dict[str, Tensor], device: torch.device, forcing_ratio: float) -> Tensor:
    inputs = batch_inputs(batch, device)
    targets = batch["targets"].to(device)
    logits = model.teacher_forward(**inputs, targets=targets, forcing_ratio=forcing_ratio)
    losses = []
    horizon_weights = {1: 0.8, 2: 0.9, 3: 1.0, 4: 1.1, 5: 1.2}
    for position, horizon in enumerate(HORIZONS):
        target = targets[:, position]
        valid = target >= 0
        if bool(valid.any()):
            losses.append(horizon_weights[horizon] * F.cross_entropy(logits[valid, position], target[valid], label_smoothing=0.02))
    return sum(losses) / max(len(losses), 1)

def beam_decode(model: Seq2SeqGRU, row: dict[str, Any], artifacts: dict[str, Any], edges: dict[str, Counter[str]], labels: list[str], device: torch.device, beam_width: int) -> list[str]:
    item = n.TraceDataset([row], artifacts)[0]
    batch = n.collate([item])
    tensors = batch_inputs(batch, device)
    model.eval()
    with torch.no_grad():
        hidden = model.encode(tensors["tokens"], tensors["lengths"], tensors["cats"], tensors["numeric"], tensors["teacher"])
    beams: list[tuple[float, list[str], int, Tensor]] = [(0.0, [], model.bos_id, hidden)]
    label_to_index = {label: index for index, label in enumerate(labels)}
    for _step in HORIZONS:
        candidates: list[tuple[float, list[str], int, Tensor]] = []
        for score, path, previous_token, beam_hidden in beams:
            previous_activity = path[-1] if path else (row["prefix_activities"][-1] if row["prefix_activities"] else n.TOKEN_START)
            allowed = list(edges.get(previous_activity, Counter()).keys()) or labels
            allowed = [value for value in allowed if value in label_to_index] or labels
            previous = torch.tensor([previous_token], dtype=torch.long, device=device)
            with torch.no_grad():
                logits, next_hidden = model.step(previous, beam_hidden)
                log_probs = torch.log_softmax(logits[0], dim=-1)
            ranked = sorted(allowed, key=lambda label: float(log_probs[label_to_index[label]]), reverse=True)[: max(beam_width, 1)]
            for label in ranked:
                index = label_to_index[label]
                candidates.append((score + float(log_probs[index]), path + [label], index, next_hidden.detach()))
        if not candidates:
            break
        candidates.sort(key=lambda item: (-item[0], item[1]))
        beams = candidates[: max(beam_width, 1)]
    return beams[0][1] if beams else []

def path_metrics(model: Seq2SeqGRU, rows: list[dict[str, Any]], artifacts: dict[str, Any], edges: dict[str, Counter[str]], labels: list[str], device: torch.device, beam_width: int) -> dict[str, Any]:
    hits, counts = Counter(), Counter()
    for row in rows:
        predicted = beam_decode(model, row, artifacts, edges, labels, device, beam_width)
        for k in (1, 3, 5):
            targets = [row.get(f"target_h{h}") for h in range(1, k + 1)]
            if all(target in artifacts["label_vocab"] for target in targets):
                counts[str(k)] += 1
                hits[str(k)] += predicted[:k] == targets
    return {"path": {f"prefix_hit@{k}": hits[str(k)] / max(counts[str(k)], 1) for k in (1, 3, 5)}, "path_n": dict(counts)}

def train_seed(seed: int, rows: list[dict[str, Any]], artifacts: dict[str, Any], output_dir: Path, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    seed_everything(seed)
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    test_rows = [row for row in rows if row["split"] == "test"]
    edges, _global = n.transition_graph(train_rows)
    labels = [label for label, _ in sorted(artifacts["label_vocab"].items(), key=lambda pair: pair[1])]
    cat_sizes = [len(artifacts["cat_vocabs"][key]) for key in n.CAT_KEYS]
    model = Seq2SeqGRU(len(artifacts["token_vocab"]), cat_sizes, len(n.NUM_KEYS), len(labels), int(artifacts.get("teacher_dim", 0))).to(device)
    transfer = load_public_encoder(model, Path(args.pretrained_gru)) if args.pretrained_gru else {"initialized": False, "path": None, "loaded_keys": []}
    loader = DataLoader(n.TraceDataset(train_rows, artifacts), batch_size=args.batch_size, shuffle=True, collate_fn=n.collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_score, best_epoch, best_state = -float("inf"), 0, None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        forcing = max(args.min_teacher_forcing, args.teacher_forcing - (epoch - 1) * args.teacher_forcing_decay)
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_batch(model, batch, device, forcing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach())
        validation = path_metrics(model, validation_rows, artifacts, edges, labels, device, args.beam_width)
        path = validation["path"]
        score = args.h3_weight * path["prefix_hit@3"] + args.h5_weight * path["prefix_hit@5"]
        history.append({"epoch": epoch, "loss": total_loss / max(len(loader), 1), "teacher_forcing": forcing, "validation": validation, "selection_score": score})
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch - best_epoch >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("no seq2seq checkpoint selected")
    model.load_state_dict(best_state)
    validation = path_metrics(model, validation_rows, artifacts, edges, labels, device, args.beam_width)
    test = path_metrics(model, test_rows, artifacts, edges, labels, device, args.beam_width)
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "neural-seq2seq-checkpoint-v0.1", "state_dict": best_state, "artifacts": artifacts, "seed": seed, "transfer": transfer}, seed_dir / "checkpoint_best.pt")
    report = {"schema_version": "neural-seq2seq-report-v0.1", "seed": seed, "device": str(device), "best_epoch": best_epoch, "best_validation_selection_score": best_score, "rows": {"train": len(train_rows), "validation": len(validation_rows), "test": len(test_rows)}, "transfer": transfer, "validation": validation, "test": test, "history": history}
    (seed_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

def train(args: argparse.Namespace) -> None:
    rows = n.read_jsonl(Path(args.input))
    artifacts = n.build_vocab(rows)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    device = torch.device(device_name)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    reports = [train_seed(seed, rows, artifacts, output_dir, args, device) for seed in seeds]
    summary = {"schema_version": "neural-seq2seq-summary-v0.1", "input": args.input, "output_dir": args.output_dir, "device": str(device), "seeds": seeds, "rows": {split: sum(row["split"] == split for row in rows) for split in ("train", "validation", "test")}, "reports": [{"seed": r["seed"], "best_epoch": r["best_epoch"], "validation": r["validation"], "test": r["test"], "transfer": r["transfer"]} for r in reports], "config": vars(args)}
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "artifacts.json").write_text(json.dumps(artifacts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/root/autodl-tmp/scheduler/results/processed/neural_prefix_dataset_v0_2_teacher/prefixes.jsonl")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/scheduler/results/processed/neural_seq2seq_public_transfer_v0_1")
    parser.add_argument("--pretrained-gru", default="/root/autodl-tmp/scheduler/results/processed/public_event_gru_pretrain_v0_1/public_gru_checkpoint.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--teacher-forcing", type=float, default=0.90)
    parser.add_argument("--teacher-forcing-decay", type=float, default=0.03)
    parser.add_argument("--min-teacher-forcing", type=float, default=0.55)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--h3-weight", type=float, default=0.45)
    parser.add_argument("--h5-weight", type=float, default=0.55)
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--device", default="auto")
    train(parser.parse_args())

if __name__ == "__main__":
    main()
