#!/usr/bin/env python3
"""Pretrain a causal GRU event encoder on public process-log prefix samples."""
from __future__ import annotations
import argparse, csv, hashlib, json, random, time
from collections import Counter
from pathlib import Path
from typing import Any
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

PAD, START, UNK = "__PAD__", "__START__", "__UNK__"

def clean_event(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    marker = " - Values: Values:"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip()
    return text or UNK

def parse_prefix(value: Any) -> list[str]:
    return [event for event in (clean_event(part) for part in str(value or "").split(",")) if event != UNK]

def read_rows(raw_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(raw_dir.glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "prefix" not in reader.fieldnames or "prediction" not in reader.fieldnames:
                continue
            dataset = path.stem.replace("test_set_", "")
            for row_index, raw in enumerate(reader):
                prefix, target = parse_prefix(raw.get("prefix")), clean_event(raw.get("prediction"))
                if target != UNK:
                    rows.append({"dataset": dataset, "row_index": row_index, "prefix": prefix, "target": target})
    if not rows:
        raise RuntimeError(f"no public rows under {raw_dir}")
    return rows

def assign_split(row: dict[str, Any]) -> str:
    key = row["dataset"] + "|" + ",".join(row["prefix"]) + "|" + row["target"]
    return "validation" if int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 10 < 2 else "train"

def build_vocab(rows):
    tokens, labels = {PAD, START, UNK}, set()
    for row in rows:
        if row["split"] == "train":
            tokens.update(row["prefix"]); labels.add(row["target"])
    tv = {PAD: 0, START: 1, UNK: 2}
    for value in sorted(tokens):
        if value not in tv: tv[value] = len(tv)
    lv = {value: index for index, value in enumerate(sorted(labels))}
    return {"token_vocab": tv, "label_vocab": lv}

class PublicDataset(Dataset):
    def __init__(self, rows, artifacts):
        self.rows, self.tv, self.lv = rows, artifacts["token_vocab"], artifacts["label_vocab"]
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        tokens = [self.tv.get(x, self.tv[UNK]) for x in [START] + row["prefix"]]
        return {"index": index, "tokens": torch.tensor(tokens), "length": len(tokens), "target": self.lv.get(row["target"], -1)}

def collate(batch):
    m = max(x["length"] for x in batch)
    tokens = torch.zeros((len(batch), m), dtype=torch.long)
    for i, item in enumerate(batch): tokens[i, :item["length"]] = item["tokens"]
    return {"index": torch.tensor([x["index"] for x in batch]), "tokens": tokens, "lengths": torch.tensor([x["length"] for x in batch]), "targets": torch.tensor([x["target"] for x in batch])}

class PublicGRU(nn.Module):
    def __init__(self, token_count, label_count):
        super().__init__()
        self.token_embedding = nn.Embedding(token_count, 96, padding_idx=0)
        self.gru = nn.GRU(96, 192, num_layers=2, dropout=0.1, batch_first=True)
        self.head = nn.Linear(192, label_count)
    def forward(self, tokens, lengths):
        packed = pack_padded_sequence(self.token_embedding(tokens), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.head(hidden[-1])

def seed_everything(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def evaluate(model, rows, artifacts, device, batch_size):
    loader = DataLoader(PublicDataset(rows, artifacts), batch_size=batch_size, shuffle=False, collate_fn=collate)
    model.eval(); n = top1 = top3 = 0; nll = 0.0
    with torch.no_grad():
        for batch in loader:
            targets = batch["targets"].to(device); valid = targets >= 0
            if not bool(valid.any()): continue
            logits = model(batch["tokens"].to(device), batch["lengths"].to(device))[valid]
            t = targets[valid]; lp = torch.log_softmax(logits, -1)
            _, inds = torch.sort(lp, descending=True)
            ranks = (inds == t.unsqueeze(1)).nonzero(as_tuple=False)[:, 1] + 1
            top1 += int((ranks <= 1).sum()); top3 += int((ranks <= 3).sum()); n += int(t.numel())
            nll += float((-lp[torch.arange(t.numel(), device=device), t]).sum())
    return {"n": n, "top1": top1 / max(n, 1), "top3": top3 / max(n, 1), "nll": nll / max(n, 1)}

def train(args):
    seed_everything(args.seed)
    raw_dir, output_dir = Path(args.raw_dir), Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(raw_dir)
    for row in rows: row["split"] = assign_split(row)
    artifacts = build_vocab(rows)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    model = PublicGRU(len(artifacts["token_vocab"]), len(artifacts["label_vocab"])).to(device)
    train_rows = [r for r in rows if r["split"] == "train"]; val_rows = [r for r in rows if r["split"] == "validation"]
    loader = DataLoader(PublicDataset(train_rows, artifacts), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_score, best_epoch, best_state = -float("inf"), 0, None; history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for batch in loader:
            targets = batch["targets"].to(device); valid = targets >= 0
            if not bool(valid.any()): continue
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["tokens"].to(device), batch["lengths"].to(device))
            loss = nn.functional.cross_entropy(logits[valid], targets[valid], label_smoothing=0.05)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss.detach())
        metrics = evaluate(model, val_rows, artifacts, device, args.batch_size)
        score = 0.6 * metrics["top1"] + 0.4 * metrics["top3"]
        history.append({"epoch": epoch, "loss": total / max(len(loader), 1), "validation": metrics, "selection_score": score})
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= args.patience: break
    if best_state is None: raise RuntimeError("no public checkpoint selected")
    encoder = {k: v for k, v in best_state.items() if k.startswith("gru.")}
    ckpt = {"schema_version": "public-event-gru-checkpoint-v0.1", "encoder_state_dict": encoder, "full_state_dict": best_state, "public_token_vocab": artifacts["token_vocab"], "public_label_vocab": artifacts["label_vocab"], "seed": args.seed, "best_epoch": best_epoch, "rows": {"all": len(rows), "train": len(train_rows), "validation": len(val_rows)}, "source_dir": str(raw_dir), "note": "Only shape-compatible GRU recurrent weights transfer; public labels and token embedding are not own labels/features."}
    ckpt_path = output_dir / "public_gru_checkpoint.pt"; torch.save(ckpt, ckpt_path)
    metrics = evaluate(model, val_rows, artifacts, device, args.batch_size)
    report = {"schema_version": "public-event-gru-pretrain-v0.1", "created_at_epoch": time.time(), "device": str(device), "seed": args.seed, "source_dir": str(raw_dir), "rows": {"all": len(rows), "train": len(train_rows), "validation": len(val_rows)}, "datasets": dict(Counter(r["dataset"] for r in rows)), "token_count": len(artifacts["token_vocab"]), "label_count": len(artifacts["label_vocab"]), "best_epoch": best_epoch, "validation": metrics, "history": history, "checkpoint": str(ckpt_path), "transfer_contract": {"public_samples_are_training_samples": True, "own_test_rows_used_during_pretraining": False, "transferable_parameters": sorted(encoder), "own_label_head_reinitialized": True, "own_token_embedding_reinitialized": True}}
    (output_dir / "pretrain_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "pretrain_rows.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="/root/autodl-tmp/scheduler/data/public/raw")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/scheduler/results/processed/public_event_gru_pretrain_v0_1")
    parser.add_argument("--epochs", type=int, default=30); parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=20260804); parser.add_argument("--device", default="auto")
    train(parser.parse_args())
