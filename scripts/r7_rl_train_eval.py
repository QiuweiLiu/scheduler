#!/usr/bin/env python3
"""Small PyTorch-only masked policy-gradient RL-0/RL-H5 runner for R7.

The environment is the same node-level event engine used by the optimizer
matrix.  It deliberately writes compact episode summaries only; raw event
logs remain covered by the 50-episode simulator gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from tracing.analysis.workload_v02_simulator import (
    load_future_artifacts,
    load_templates,
    read_jsonl,
    predicted_future_cost,
    simulate_episode,
    train_resource_stats,
)


FEATURE_DIM = 14


class CandidateActor(nn.Module):
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


def read_jsonl_limit(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) >= limit:
                    break
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reward(summary: dict[str, Any]) -> float:
    completion = float(summary.get("mean_completion_ms") or 0.0) / 100000.0
    deadline = float(summary.get("deadline_miss_rate") or 0.0)
    evictions = float(summary.get("gpu_evictions") or 0.0) / 100.0
    return -(completion + 0.5 * deadline + 0.01 * evictions)


def evaluate(
    model: CandidateActor,
    policy: str,
    episodes: list[dict[str, Any]],
    templates: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    train_stats: dict[str, dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for episode in episodes:
            context = {"model": model, "mode": "eval", "trajectory": []}
            summary, _events = simulate_episode(
                episode,
                templates,
                policy,
                future_artifacts=artifacts,
                train_stats=train_stats,
                collect_events=False,
                rl_context=context,
            )
            summary = dict(summary)
            summary.update({"seed": seed, "policy": policy, "decision_count": len(context["trajectory"])})
            rows.append(summary)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("rl_0", "rl_h5"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--train-episodes", type=Path, required=True)
    parser.add_argument("--validation-episodes", type=Path, required=True)
    parser.add_argument("--future-artifacts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--eval-limit", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=int(os.environ.get("R7_TORCH_THREADS", "1")),
        help="intra-op PyTorch threads for this process",
    )
    parser.add_argument(
        "--torch-interop-threads",
        type=int,
        default=int(os.environ.get("R7_TORCH_INTEROP_THREADS", "1")),
        help="inter-op PyTorch threads for this process",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="print a flushed heartbeat every N training episodes",
    )
    args = parser.parse_args()
    if args.torch_threads < 1 or args.torch_interop_threads < 1 or args.progress_every < 1:
        parser.error("--torch-threads, --torch-interop-threads and --progress-every must be >= 1")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    templates = load_templates(args.templates)
    train_episodes = read_jsonl_limit(args.train_episodes, args.train_limit)
    validation_episodes = read_jsonl_limit(args.validation_episodes, args.eval_limit)
    artifacts = load_future_artifacts(args.future_artifacts)
    train_stats = train_resource_stats(templates)
    for node_id, artifact in artifacts.items():
        for horizon in (1, 3, 5):
            artifact[f"_cost_h{horizon}"] = predicted_future_cost(node_id, artifacts, train_stats, horizon)
    model = CandidateActor()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    baseline = 0.0
    rewards: list[float] = []
    decisions = 0
    model.train()
    train_start = time.monotonic()
    for index, episode in enumerate(train_episodes, 1):
        context = {"model": model, "mode": "train", "trajectory": []}
        summary, _events = simulate_episode(
            episode,
            templates,
            args.policy,
            future_artifacts=artifacts,
            train_stats=train_stats,
            collect_events=False,
            rl_context=context,
        )
        episode_reward = reward(summary)
        rewards.append(episode_reward)
        decisions += len(context["trajectory"])
        if context["trajectory"]:
            baseline = 0.95 * baseline + 0.05 * episode_reward
            advantage = episode_reward - baseline
            log_prob = torch.stack([item["log_prob"] for item in context["trajectory"]]).sum()
            entropy = torch.stack([item["entropy"] for item in context["trajectory"]]).mean()
            loss = -float(advantage) * log_prob - args.entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if index % args.progress_every == 0 or index == len(train_episodes):
            print(
                json.dumps(
                    {
                        "progress_train_episodes": index,
                        "target_train_episodes": len(train_episodes),
                        "decisions": decisions,
                        "episode_decisions": len(context["trajectory"]),
                        "mean_reward": statistics.fmean(rewards[-250:]),
                        "elapsed_s": round(time.monotonic() - train_start, 3),
                        "torch_threads": args.torch_threads,
                        "torch_interop_threads": args.torch_interop_threads,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "checkpoint.pt"
    torch.save({"policy": args.policy, "seed": args.seed, "feature_dim": FEATURE_DIM, "model_state_dict": model.state_dict(), "train_episodes": len(train_episodes), "train_decisions": decisions}, checkpoint)
    validation_rows = evaluate(model, args.policy, validation_episodes, templates, artifacts, train_stats, args.seed)
    (args.output_dir / "validation_results.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in validation_rows) + "\n", encoding="utf-8")
    completion = [float(row["mean_completion_ms"]) for row in validation_rows if row.get("mean_completion_ms") is not None]
    report = {
        "schema_version": "r7-masked-policy-gradient-v0.1",
        "status": "passed" if len(validation_rows) == len(validation_episodes) and all(int(row.get("failed_jobs") or 0) == 0 for row in validation_rows) else "failed",
        "policy": args.policy,
        "seed": args.seed,
        "feature_dim": FEATURE_DIM,
        "train_episodes": len(train_episodes),
        "train_decisions": decisions,
        "validation_episodes": len(validation_episodes),
        "validation_mean_completion_ms": statistics.fmean(completion) if completion else None,
        "validation_p95_completion_ms": sorted(completion)[max(0, int(len(completion) * 0.95) - 1)] if completion else None,
        "validation_failed_jobs": sum(int(row.get("failed_jobs") or 0) for row in validation_rows),
        "train_elapsed_s": round(time.monotonic() - train_start, 3),
        "torch_threads": args.torch_threads,
        "torch_interop_threads": args.torch_interop_threads,
        "progress_every": args.progress_every,
        "checkpoint_sha256": sha256(checkpoint),
        "optimizer": "Adam",
        "reward": "-(mean_completion_ms/100000 + 0.5*deadline_miss_rate + 0.01*gpu_evictions/100)",
        "future_mode": "zeroed future slots" if args.policy == "rl_0" else "B05 H5 predicted future slots",
        "action_mask": "engine candidate pool; non-fitting candidates excluded when any fitting candidate exists",
    }
    (args.output_dir / "rl_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
