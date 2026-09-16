#!/usr/bin/env python3
"""Train a behavior-cloned policy from CP-RHO expert demonstrations.

Loads the BC dataset collected by r8_bc_collect.py, trains a BcActor
with supervised CrossEntropy loss (padding + masking for variable-length
candidate pools), and evaluates on S_val episodes against PredOpt-v2
and CP-RHO baselines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any

import torch
from torch import nn

from tracing.analysis.workload_v02_simulator import (
    load_future_artifacts,
    load_templates,
    read_jsonl,
    simulate_episode,
    train_resource_stats,
)


FEATURE_DIM = 14


class BcActor(nn.Module):
    """Same architecture as CandidateActor; scores each candidate independently."""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bc_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def evaluate(
    model: BcActor,
    templates_path: Path,
    episodes_path: Path,
    future_artifacts_path: Path,
    limit: int,
    horizons: tuple[int, ...],
    device: torch.device,
) -> dict[str, Any]:
    """Run BC evaluation on validation episodes (S_val) with torch.no_grad()."""
    templates = load_templates(templates_path)
    episodes = read_jsonl(episodes_path)[:limit]
    artifacts = load_future_artifacts(future_artifacts_path)
    train_stats = train_resource_stats(templates)

    results: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for episode_index, episode in enumerate(episodes, 1):
            for horizon in horizons:
                policy = f"bc_h{horizon}"
                context: dict[str, Any] = {
                    "model": model,
                    "mode": "eval",
                }
                summary, _events = simulate_episode(
                    episode,
                    templates,
                    policy,
                    future_artifacts=artifacts,
                    train_stats=train_stats,
                    collect_events=False,
                    rl_context=context,
                )
                results.append(summary)
            if episode_index % 10 == 0 or episode_index == len(episodes):
                print(json.dumps({
                    "eval_progress": episode_index,
                    "target": len(episodes),
                }, ensure_ascii=False), flush=True)

    by_policy: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        by_policy.setdefault(str(row["policy"]), []).append(row)

    aggregate: dict[str, dict[str, float | None]] = {}
    for policy, rows in sorted(by_policy.items()):
        completions = [float(r["mean_completion_ms"]) for r in rows if r.get("mean_completion_ms") is not None]
        aggregate[policy] = {
            "episodes": len(rows),
            "failed_jobs": sum(int(r.get("failed_jobs") or 0) for r in rows),
            "mean_completion_ms": statistics.fmean(completions) if completions else None,
        }
    return aggregate


def _pad_batch(
    features: list[torch.Tensor],
    actions: list[int],
    max_c: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length candidate pools to max_c; return mask of valid positions."""
    b = len(features)
    padded = torch.zeros(b, max_c, FEATURE_DIM, device=device)
    mask = torch.zeros(b, max_c, dtype=torch.bool, device=device)
    targets = torch.tensor(actions, dtype=torch.long, device=device)
    for i, feat in enumerate(features):
        c = min(feat.shape[0], max_c)
        padded[i, :c] = feat[:c].to(device)
        mask[i, :c] = True
    return padded, targets, mask


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--val-split", type=float, default=0.1)
    # Evaluation paths (optional)
    parser.add_argument("--eval-templates", type=Path)
    parser.add_argument("--eval-episodes", type=Path)
    parser.add_argument("--eval-future-artifacts", type=Path)
    parser.add_argument("--eval-limit", type=int, default=100)
    parser.add_argument("--eval-horizons", default="1,3,5")
    # Baseline comparison paths
    parser.add_argument("--baseline-templates", type=Path)
    parser.add_argument("--baseline-episodes", type=Path)
    parser.add_argument("--baseline-future-artifacts", type=Path)
    parser.add_argument("--baseline-limit", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_horizons = tuple(int(v) for v in args.eval_horizons.split(",") if v.strip())

    # --- Load dataset ---
    raw = load_bc_dataset(args.dataset)
    if not raw:
        parser.error("empty BC dataset")

    all_features: list[torch.Tensor] = []
    all_actions: list[int] = []
    for row in raw:
        all_features.append(torch.tensor(row["features"], dtype=torch.float32))
        all_actions.append(int(row["action_index"]))
    n = len(all_features)

    # Train/val split
    indices = list(range(n))
    random.shuffle(indices)
    val_count = max(1, int(n * args.val_split))
    val_indices = set(indices[:val_count])
    train_indices = indices[val_count:]

    # Compute action agreement baseline: how often does the majority class win?
    from collections import Counter
    action_counts = Counter(all_actions[i] for i in val_indices)
    majority_agreement = max(action_counts.values()) / max(1, len(val_indices))

    print(json.dumps({
        "dataset_rows": n,
        "train_decisions": len(train_indices),
        "val_decisions": len(val_indices),
        "majority_baseline_agreement": round(majority_agreement, 4),
    }, ensure_ascii=False), flush=True)

    # --- Model ---
    model = BcActor(hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    losses: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_indices)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(train_indices), args.batch_size):
            batch_idx = train_indices[start:start + args.batch_size]
            batch_feat = [all_features[i] for i in batch_idx]
            batch_act = [all_actions[i] for i in batch_idx]

            max_c = max(f.shape[0] for f in batch_feat)
            padded, targets, mask = _pad_batch(batch_feat, batch_act, max_c, device)

            logits = model(padded.reshape(-1, FEATURE_DIM)).reshape(len(batch_idx), max_c)
            logits = logits.masked_fill(~mask, float("-inf"))

            loss = nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for i in val_indices:
                feat = all_features[i].to(device)
                c = feat.shape[0]
                logits = model(feat).reshape(-1)
                act = all_actions[i]
                val_loss += nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([act], device=device)).item()
                if int(torch.argmax(logits).item()) == act:
                    val_correct += 1
                val_total += 1

        mean_train = epoch_loss / max(1, batches)
        mean_val = val_loss / max(1, val_total)
        val_acc = val_correct / max(1, val_total)
        losses.append({"epoch": epoch, "train_loss": mean_train, "val_loss": mean_val, "val_accuracy": val_acc})
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            best_epoch = epoch
            torch.save(model.state_dict(), args.output_dir / "bc_model_best.pt")
        if epoch % 10 == 0 or epoch == args.epochs:
            print(json.dumps({
                "epoch": epoch,
                "train_loss": round(mean_train, 4),
                "val_loss": round(mean_val, 4),
                "val_accuracy": round(val_acc, 4),
            }, ensure_ascii=False), flush=True)

    # Load best checkpoint
    model.load_state_dict(torch.load(args.output_dir / "bc_model_best.pt", weights_only=True))

    # --- Evaluation on S_val ---
    eval_results: dict[str, Any] = {}
    bc_results: dict[str, Any] = {}
    if args.eval_templates and args.eval_episodes and args.eval_future_artifacts:
        bc_results = evaluate(model, args.eval_templates, args.eval_episodes,
                              args.eval_future_artifacts, args.eval_limit,
                              eval_horizons, device)
        eval_results["bc"] = bc_results

    # --- Baseline comparison (PredOpt-v2, CP-RHO) ---
    baseline_results: dict[str, Any] = {}
    if args.baseline_templates and args.baseline_episodes and args.baseline_future_artifacts:
        for tag, policies in [("predopt_v2", ["predopt_v2_h5"]), ("cp_rho", ["cp_rho_h5"])]:
            templates = load_templates(args.baseline_templates)
            episodes = read_jsonl(args.baseline_episodes)[:args.baseline_limit]
            artifacts = load_future_artifacts(args.baseline_future_artifacts)
            train_stats = train_resource_stats(templates)
            rows: list[dict[str, Any]] = []
            for ep in episodes:
                for policy in policies:
                    summary, _ = simulate_episode(ep, templates, policy,
                                                  future_artifacts=artifacts,
                                                  train_stats=train_stats,
                                                  collect_events=False)
                    rows.append(summary)
            by_p = {}
            for r in rows:
                by_p.setdefault(str(r["policy"]), []).append(r)
            agg = {}
            for p, rs in sorted(by_p.items()):
                comps = [float(r["mean_completion_ms"]) for r in rs if r.get("mean_completion_ms") is not None]
                agg[p] = {
                    "episodes": len(rs),
                    "failed_jobs": sum(int(r.get("failed_jobs") or 0) for r in rs),
                    "mean_completion_ms": statistics.fmean(comps) if comps else None,
                }
            baseline_results[tag] = agg
        eval_results["baselines"] = baseline_results

    # --- Gate check ---
    gate = {"passed": False, "reasons": []}
    bc_best = None
    if bc_results:
        completions = [v["mean_completion_ms"] for v in bc_results.values() if v.get("mean_completion_ms") is not None]
        if completions:
            bc_best = min(completions)
    predopt_baseline = None
    if baseline_results.get("predopt_v2"):
        for v in baseline_results["predopt_v2"].values():
            if v.get("mean_completion_ms") is not None:
                predopt_baseline = v["mean_completion_ms"]
    if bc_best is not None and predopt_baseline is not None:
        bc_relative = (bc_best - predopt_baseline) / max(1.0, predopt_baseline)
        gate["bc_best_completion_ms"] = bc_best
        gate["predopt_v2_completion_ms"] = predopt_baseline
        gate["bc_relative_gap"] = bc_relative
        if bc_relative > 0.10:
            gate["reasons"].append(f"BC completion {bc_relative*100:.1f}% worse than PredOpt-v2 (>10% threshold)")
        if val_acc < 0.6:
            gate["reasons"].append(f"Validation action agreement {val_acc:.3f} < 0.6 threshold")
        if not gate["reasons"]:
            gate["passed"] = True
    else:
        gate["reasons"].append("Insufficient data for gate check")

    # --- Save report ---
    report = {
        "schema_version": "r8-bc-train-v0.1",
        "status": "passed" if gate["passed"] else "gate_blocked",
        "dataset": str(args.dataset),
        "dataset_sha256": sha256(args.dataset),
        "model_architecture": "BcActor(14→128→128→1)",
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": val_acc,
        "majority_baseline_agreement": majority_agreement,
        "train_decisions": len(train_indices),
        "val_decisions": len(val_indices),
        "gate": gate,
        "loss_curve": losses,
        "eval_results": eval_results,
    }
    (args.output_dir / "bc_train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "loss_curve.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in losses) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": round(val_acc, 4),
        "gate_passed": gate["passed"],
    }, ensure_ascii=False), flush=True)
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())