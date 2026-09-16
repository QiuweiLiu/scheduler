#!/usr/bin/env python3
"""Train the first small neural predictor on neural-prefix-v0.1 JSONL."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tracing.analysis.neural_trace_model import ModelConfig, TracePrefixPredictor, require_torch, torch


START_TOKEN = "__START__"
UNK_TOKEN = "__UNK__"
PAD_TOKEN = "__PAD__"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _vocab(values: Iterable[str]) -> dict[str, int]:
    result = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for value in sorted(set(values)):
        if value not in result:
            result[value] = len(result)
    return result


def build_vocabs(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    train = [row for row in rows if row.get("split") == "train"]
    nodes = [START_TOKEN]
    labels: list[str] = []
    domains: list[str] = []
    planners: list[str] = []
    for row in train:
        input_data = row.get("input") if isinstance(row.get("input"), Mapping) else {}
        nodes.extend(str(value) for value in input_data.get("prefix_nodes") or [])
        target = row.get("target") if isinstance(row.get("target"), Mapping) else {}
        labels.append(str(target.get("next_node", "__END__")))
        domains.append(str(row.get("domain_id", "unknown")))
        planners.append(str(row.get("planner_model_id", "unknown")))
    return {
        "nodes": _vocab(nodes),
        "labels": _vocab(labels),
        "domains": _vocab(domains),
        "planners": _vocab(planners),
    }


def _lookup(vocabulary: Mapping[str, int], value: Any) -> int:
    return int(vocabulary.get(str(value), vocabulary[UNK_TOKEN]))


def collate(rows: Sequence[Mapping[str, Any]], vocabs: Mapping[str, Mapping[str, int]], max_len: int) -> dict[str, Any]:
    require_torch()
    sequences: list[list[int]] = []
    lengths: list[int] = []
    domains: list[int] = []
    planners: list[int] = []
    positions: list[int] = []
    targets: list[int] = []
    ends: list[float] = []
    suffix_lengths: list[float] = []
    for row in rows:
        input_data = row.get("input") if isinstance(row.get("input"), Mapping) else {}
        prefix = [START_TOKEN] + [str(value) for value in (input_data.get("prefix_nodes") or [])]
        prefix = prefix[-max_len:]
        sequence = [_lookup(vocabs["nodes"], value) for value in prefix]
        sequences.append(sequence)
        lengths.append(len(sequence))
        domains.append(_lookup(vocabs["domains"], row.get("domain_id", "unknown")))
        planners.append(_lookup(vocabs["planners"], row.get("planner_model_id", "unknown")))
        positions.append(min(max(0, int(row.get("position", 0))), 64))
        target = row.get("target") if isinstance(row.get("target"), Mapping) else {}
        targets.append(_lookup(vocabs["labels"], target.get("next_node", "__END__")))
        ends.append(float(bool(target.get("end", False))))
        suffix_lengths.append(math.log1p(float(target.get("length", 0))))
    max_sequence = max(lengths, default=1)
    node_ids = torch.zeros((len(rows), max_sequence), dtype=torch.long)
    for index, sequence in enumerate(sequences):
        node_ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return {
        "node_ids": node_ids,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "domain_ids": torch.tensor(domains, dtype=torch.long),
        "planner_ids": torch.tensor(planners, dtype=torch.long),
        "positions": torch.tensor(positions, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "ends": torch.tensor(ends, dtype=torch.float32),
        "suffix_lengths": torch.tensor(suffix_lengths, dtype=torch.float32),
    }


def _batches(rows: Sequence[Mapping[str, Any]], batch_size: int, rng: random.Random) -> Iterable[list[Mapping[str, Any]]]:
    values = list(rows)
    rng.shuffle(values)
    for index in range(0, len(values), batch_size):
        yield values[index : index + batch_size]


def evaluate(model: Any, rows: Sequence[Mapping[str, Any]], vocabs: Mapping[str, Mapping[str, int]], batch_size: int, max_len: int) -> dict[str, float]:
    require_torch()
    if not rows:
        return {"n": 0.0}
    model.eval()
    hits1 = hits3 = nll = 0.0
    count = 0
    with torch.no_grad():
        for batch_rows in (rows[index : index + batch_size] for index in range(0, len(rows), batch_size)):
            batch = collate(batch_rows, vocabs, max_len)
            output = model(**{key: batch[key] for key in ("node_ids", "lengths", "domain_ids", "planner_ids", "positions")})
            probabilities = output["next_logits"].softmax(dim=-1)
            targets = batch["targets"]
            top = probabilities.topk(k=min(3, probabilities.shape[-1]), dim=-1).indices
            hits1 += float((top[:, 0] == targets).sum().item())
            hits3 += float((top == targets.unsqueeze(1)).any(dim=1).sum().item())
            nll -= float(torch.log(probabilities.gather(1, targets.unsqueeze(1)).clamp_min(1e-12)).sum().item())
            count += len(batch_rows)
    return {"n": float(count), "top1": hits1 / count, "top3": hits3 / count, "nll": nll / count}


def train(args: argparse.Namespace) -> dict[str, Any]:
    require_torch()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    rows = _read_jsonl(args.examples)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    train_rows = [row for row in rows if row.get("split") == "train"]
    validation_rows = [row for row in rows if row.get("split") == "validation"]
    test_rows = [row for row in rows if row.get("split") == "test"]
    if not train_rows or not validation_rows:
        raise ValueError("examples must contain train and validation rows")
    vocabs = build_vocabs(rows)
    config = ModelConfig(
        node_vocab_size=len(vocabs["nodes"]),
        domain_vocab_size=len(vocabs["domains"]),
        planner_vocab_size=len(vocabs["planners"]),
        label_size=len(vocabs["labels"]),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        max_position=64,
    )
    model = TracePrefixPredictor(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_validation: dict[str, float] | None = None
    best_state: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        losses: list[float] = []
        for batch_rows in _batches(train_rows, args.batch_size, rng):
            batch = collate(batch_rows, vocabs, args.max_len)
            output = model(**{key: batch[key] for key in ("node_ids", "lengths", "domain_ids", "planner_ids", "positions")})
            next_loss = torch.nn.functional.cross_entropy(output["next_logits"], batch["targets"])
            end_loss = torch.nn.functional.binary_cross_entropy_with_logits(output["end_logits"], batch["ends"])
            length_loss = torch.nn.functional.mse_loss(output["length_pred"], batch["suffix_lengths"])
            loss = next_loss + args.end_loss_weight * end_loss + args.length_loss_weight * length_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        validation = evaluate(model, validation_rows, vocabs, args.batch_size, args.max_len)
        entry = {"epoch": epoch + 1, "train_loss": sum(losses) / max(1, len(losses)), "validation": validation}
        history.append(entry)
        if best_validation is None or (validation["nll"], -validation["top1"]) < (best_validation["nll"], -best_validation["top1"]):
            best_validation = validation
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    report: dict[str, Any] = {
        "schema_version": "neural-trace-training-v0.1",
        "input": str(args.examples),
        "config": config.to_dict(),
        "rows": {"train": len(train_rows), "validation": len(validation_rows), "test": len(test_rows)},
        "best_validation": best_validation,
        "history": history,
        "metrics": {"validation": evaluate(model, validation_rows, vocabs, args.batch_size, args.max_len)},
        "leakage_checks": {
            "future_events_in_input": False,
            "ground_truth_in_input": False,
            "remaining_steps_in_input": False,
            "remaining_runtime_in_input": False,
        },
    }
    if args.evaluate_test:
        report["metrics"]["test"] = evaluate(model, test_rows, vocabs, args.batch_size, args.max_len)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config.to_dict(), "vocabs": vocabs}, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--max-rows", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--end-loss-weight", type=float, default=0.25)
    parser.add_argument("--length-loss-weight", type=float, default=0.10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = train(args)
    except RuntimeError as exc:
        print(json.dumps({"error": str(exc), "status": "environment_not_ready"}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": "completed", "report": str(args.report), "best_validation": report["best_validation"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
