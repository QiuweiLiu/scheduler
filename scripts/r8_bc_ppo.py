#!/usr/bin/env python3
"""BC-initialized PPO + GAE training for R8-P4.

Initializes actor with BC weights, adds a critic network, and trains with
clipped PPO objective and Generalized Advantage Estimation.  Runs multiple
seeds for seed-consistency verification (invoke separately per seed).
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
    predicted_future_cost,
    read_jsonl,
    simulate_episode,
    train_resource_stats,
)


FEATURE_DIM = 14
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
VALUE_COEF = 0.5
MAX_GRAD_NORM = 1.0


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


class Critic(nn.Module):
    """Value function: aggregates candidate features via mean, then MLP."""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        return self.net(state_features)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl_limit(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) >= limit:
                    break
    return rows


def episode_reward(summary: dict[str, Any]) -> float:
    completion = float(summary.get("mean_completion_ms") or 0.0) / 100000.0
    deadline = float(summary.get("deadline_miss_rate") or 0.0)
    evictions = float(summary.get("gpu_evictions") or 0.0) / 100.0
    failed = float(summary.get("failed_jobs") or 0)
    return -(completion + 0.5 * deadline + 0.01 * evictions + 5.0 * failed)


def compute_gae(rewards: list[float], values: list[float], gamma: float, lam: float) -> tuple[list[float], list[float]]:
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_val = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def evaluate(
    actor: BcActor,
    critic: Critic | None,
    episodes: list[dict[str, Any]],
    templates: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    train_stats: dict[str, dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    actor.eval()
    with torch.no_grad():
        for episode in episodes:
            context: dict[str, Any] = {"model": actor, "mode": "eval", "trajectory": []}
            if critic is not None:
                context["critic"] = critic
            summary, _events = simulate_episode(
                episode, templates, "bc_h5", future_artifacts=artifacts,
                train_stats=train_stats, collect_events=False, rl_context=context,
            )
            summary = dict(summary)
            summary.update({"seed": seed, "policy": "ppo_h5", "decision_count": len(context["trajectory"])})
            rows.append(summary)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--train-episodes", type=Path, required=True)
    parser.add_argument("--validation-episodes", type=Path, required=True)
    parser.add_argument("--future-artifacts", type=Path, required=True)
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--eval-limit", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    templates = load_templates(args.templates)
    train_episodes = read_jsonl_limit(args.train_episodes, args.train_limit)
    validation_episodes = read_jsonl_limit(args.validation_episodes, args.eval_limit)
    artifacts = load_future_artifacts(args.future_artifacts)
    train_stats = train_resource_stats(templates)
    for node_id, artifact in artifacts.items():
        for horizon in (1, 3, 5):
            artifact[f"_cost_h{horizon}"] = predicted_future_cost(node_id, artifacts, train_stats, horizon)

    # --- Model ---
    actor = BcActor(hidden=args.hidden).to(device)
    bc_state = torch.load(args.bc_checkpoint, weights_only=True, map_location=device)
    missing, unexpected = actor.load_state_dict(bc_state, strict=False)
    if missing:
        print(json.dumps({"warning": "BC checkpoint missing keys", "keys": missing}, ensure_ascii=False), flush=True)
    if unexpected:
        print(json.dumps({"warning": "BC checkpoint unexpected keys", "keys": unexpected}, ensure_ascii=False), flush=True)
    print(json.dumps({"bc_weights_loaded": True, "seed": args.seed, "missing_keys": missing, "unexpected_keys": unexpected}, ensure_ascii=False), flush=True)

    critic = Critic(hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=args.lr,
    )

    # --- Training ---
    train_start = time.monotonic()
    total_decisions = 0
    actor.train()
    critic.train()

    for episode_idx, episode in enumerate(train_episodes, 1):
        context: dict[str, Any] = {"model": actor, "mode": "train", "trajectory": [], "critic": critic}
        summary, _events = simulate_episode(
            episode, templates, "bc_h5",
            future_artifacts=artifacts, train_stats=train_stats,
            collect_events=False, rl_context=context,
        )

        traj = context["trajectory"]
        if not traj:
            continue
        total_decisions += len(traj)

        ep_reward = episode_reward(summary)

        # Per-step rewards: terminal reward only at last step
        rewards = [0.0] * (len(traj) - 1) + [ep_reward]
        values = [t["value"] for t in traj] + [0.0]

        # GAE
        advantages, returns = compute_gae(rewards, values, GAMMA, LAMBDA)
        adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
        if len(traj) > 1:
            adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std(unbiased=False) + 1e-8)
        ret_tensor = torch.tensor(returns, dtype=torch.float32, device=device)

        # Detach old log_probs from the rollout graph
        old_log_probs = torch.stack([t["log_prob"] for t in traj]).to(device).detach()

        # PPO: multiple epochs on the same trajectory
        for _ in range(args.ppo_epochs):
            new_log_probs_list = []
            new_values_list = []
            entropies = []
            for step in traj:
                feat = torch.tensor(step["features"], dtype=torch.float32, device=device)
                action_idx = int(step["action_index"])
                if action_idx < 0 or action_idx >= feat.shape[0]:
                    raise RuntimeError(
                        f"Invalid action_index {action_idx} for {feat.shape[0]} candidates"
                    )
                # Actor forward
                new_logits = actor(feat).reshape(-1)
                new_log_probs_list.append(
                    -nn.functional.cross_entropy(
                        new_logits.unsqueeze(0),
                        torch.tensor([action_idx], device=device),
                        reduction="sum",
                    )
                )
                # Entropy from current policy
                dist = torch.distributions.Categorical(logits=new_logits)
                entropies.append(dist.entropy())
                # Critic forward
                state_feat = feat.mean(dim=0, keepdim=True)
                new_values_list.append(critic(state_feat).squeeze())

            new_log_probs = torch.stack(new_log_probs_list)
            new_values = torch.stack(new_values_list)
            entropy = torch.stack(entropies).mean()

            # Clipped PPO objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * adv_tensor
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_tensor
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.functional.mse_loss(new_values, ret_tensor)

            loss = policy_loss + VALUE_COEF * value_loss - args.entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()), MAX_GRAD_NORM,
            )
            optimizer.step()

        if episode_idx % args.progress_every == 0 or episode_idx == len(train_episodes):
            print(json.dumps({
                "progress_train_episodes": episode_idx,
                "target": len(train_episodes),
                "decisions": total_decisions,
                "episode_decisions": len(traj),
                "mean_reward": round(ep_reward, 4),
                "policy_loss": round(policy_loss.item(), 4),
                "value_loss": round(value_loss.item(), 4),
                "entropy": round(entropy.item(), 4),
                "elapsed_s": round(time.monotonic() - train_start, 3),
            }, ensure_ascii=False), flush=True)

    # --- Save checkpoint ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "checkpoint.pt"
    torch.save({
        "seed": args.seed,
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "train_episodes": len(train_episodes),
        "train_decisions": total_decisions,
    }, checkpoint)

    # --- Evaluate ---
    eval_rows = evaluate(actor, critic, validation_episodes,
                         templates, artifacts, train_stats, args.seed)
    (args.output_dir / "validation_results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in eval_rows) + "\n",
        encoding="utf-8",
    )
    completions = [float(r["mean_completion_ms"]) for r in eval_rows if r.get("mean_completion_ms") is not None]
    failed_jobs = sum(int(r.get("failed_jobs") or 0) for r in eval_rows)

    # --- Source hash ---
    source_hashes = {
        "templates": sha256(args.templates),
        "train_episodes": sha256(args.train_episodes),
        "validation_episodes": sha256(args.validation_episodes),
        "bc_checkpoint": sha256(args.bc_checkpoint),
    }

    report = {
        "schema_version": "r8-bc-ppo-v0.1",
        "status": "passed" if failed_jobs == 0 else "failed",
        "seed": args.seed,
        "train_episodes": len(train_episodes),
        "train_decisions": total_decisions,
        "validation_episodes": len(validation_episodes),
        "validation_mean_completion_ms": statistics.fmean(completions) if completions else None,
        "validation_p95_completion_ms": sorted(completions)[max(0, int(len(completions) * 0.95) - 1)] if completions else None,
        "validation_failed_jobs": failed_jobs,
        "train_elapsed_s": round(time.monotonic() - train_start, 3),
        "bc_checkpoint": str(args.bc_checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "source_hashes": source_hashes,
        "ppo_epochs": args.ppo_epochs,
        "gamma": GAMMA,
        "lambda": LAMBDA,
        "clip_eps": CLIP_EPS,
        "reward_function": "-(completion/100000 + 0.5*deadline + 0.01*evictions + 5*failed_jobs)",
        "future_mode": "B05 H5 predicted future slots",
        "action_mask": "engine candidate pool; non-fitting excluded when any fitting candidate exists",
    }
    (args.output_dir / "ppo_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())