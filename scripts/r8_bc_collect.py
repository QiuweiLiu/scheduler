#!/usr/bin/env python3
"""Collect per-decision (state, action) pairs from CP-RHO for BC training.

Runs CP-RHO H1/H3/H5 on S_train episodes and saves each decision point's
candidate features and the selected action index.  The output is a JSONL
dataset where each row is one decision with keys:
  - features: list of 14-dim feature vectors (one per candidate)
  - action_index: which candidate CP-RHO chose
  - horizon: H value
  - episode_id: source episode
  - decision_index: sequential decision number within the episode
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tracing.analysis.workload_v02_simulator import (
    load_future_artifacts,
    load_templates,
    read_jsonl,
    simulate_episode,
    train_resource_stats,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--future-artifacts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--horizons", default="1,3,5")
    parser.add_argument("--time-limit-s", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    horizons = tuple(int(v) for v in args.horizons.split(",") if v.strip())
    if not horizons or any(v not in (1, 3, 5) for v in horizons):
        parser.error("--horizons must contain only 1, 3 or 5")

    templates = load_templates(args.templates)
    episodes = read_jsonl(args.episodes)[: args.limit]
    artifacts = load_future_artifacts(args.future_artifacts)
    train_stats = train_resource_stats(templates)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bc_rows: list[dict[str, Any]] = []
    episode_ids: set[str] = set()
    total_decisions = 0

    for episode_index, episode in enumerate(episodes, 1):
        episode_id = str(episode.get("episode_id", f"ep_{episode_index}"))
        episode_ids.add(episode_id)
        for horizon in horizons:
            policy = f"cp_rho_h{horizon}"
            context: dict[str, Any] = {
                "cp_rho_time_limit_s": args.time_limit_s,
                "cp_rho_seed": args.seed,
                "cp_rho_num_workers": 1,
                "bc_collect": True,
            }
            summary, _events = simulate_episode(
                episode,
                templates,
                policy,
                future_artifacts=artifacts,
                train_stats=train_stats,
                collect_events=False,
                policy_context=context,
            )
            bc_events = context.get("bc_events", [])
            for decision_index, event in enumerate(bc_events):
                bc_rows.append({
                    "episode_id": episode_id,
                    "policy": policy,
                    "decision_index": decision_index,
                    "features": event["features"],
                    "action_index": event["action_index"],
                    "horizon": event["horizon"],
                    "candidate_count": event["candidate_count"],
                    "fallback": event["fallback"],
                })
            total_decisions += len(bc_events)
        if episode_index % 10 == 0 or episode_index == len(episodes):
            print(json.dumps({
                "progress_episodes": episode_index,
                "target_episodes": len(episodes),
                "decisions_collected": total_decisions,
            }, ensure_ascii=False), flush=True)

    dataset_path = args.output_dir / "bc_dataset.jsonl"
    dataset_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in bc_rows) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": "r8-bc-dataset-v0.1",
        "status": "passed",
        "episodes": len(episodes),
        "episode_ids": len(episode_ids),
        "horizons": list(horizons),
        "decisions_collected": total_decisions,
        "source_templates_sha256": sha256(args.templates),
        "source_episodes_sha256": sha256(args.episodes),
    }
    (args.output_dir / "bc_dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())