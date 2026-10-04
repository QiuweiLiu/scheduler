#!/usr/bin/env python3
"""Main-table baseline activation audit (pre-run evidence; local, no GPU).

Runs the four new baseline arms on a development subset of the frozen workload
and aggregates each mechanism's activation counters.  This is an AUDIT, not a
performance measurement: no metric comparison, no confirm300 episodes, no
tuning.  The acceptance question is only whether each mechanism genuinely
entered the decision path; real activation rates are reported as-is.

Profile-dependent mechanisms (Hermes prewarm, Torpor covered-load ranking) need
the workload shape / GPU-identity contract that the current templates do not yet
carry; their fail-closed reasons are reported here and their per-mechanism
activation is covered by the fidelity tests until the contract lands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracing.analysis.hermes_methods import build_pdgraph  # noqa: E402
from tracing.analysis.qlm_methods import build_duration_bank  # noqa: E402
from tracing.analysis.workload_v02_simulator import (  # noqa: E402
    load_templates,
    read_jsonl,
    simulate_episode,
    train_resource_stats,
)

ARMS = ("parrot_appfifo", "qlm_queue", "hermes_gittins", "torpor_lifecycle")


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
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--topology-view", default="causal_v3",
        help="explicit topology view; causal projections must not load as legacy",
    )
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    templates = load_templates(args.templates, topology_view=args.topology_view)
    episodes_all = read_jsonl(args.episodes)
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    development = list(split["development"])
    by_id = {str(episode["episode_id"]): episode for episode in episodes_all}
    episodes = [by_id[eid] for eid in development if eid in by_id][: args.limit]
    if not episodes:
        raise SystemExit("no development episodes found")
    train_stats = train_resource_stats(templates)
    if not train_stats:
        raise SystemExit("train_resource_stats is empty; templates lack the train split")

    bank = build_duration_bank(templates)
    graph = build_pdgraph(templates)
    contexts = {
        "parrot_appfifo": {},
        "qlm_queue": {"qlm_duration_bank": bank, "qlm_rng": random.Random(args.seed)},
        "hermes_gittins": {"hermes_pdgraph": graph},
        "torpor_lifecycle": {},
    }

    report = {
        "schema": "main-table-activation-audit-v1",
        "seed": args.seed,
        "templates": str(args.templates),
        "templates_sha256": sha256(args.templates),
        "episodes": str(args.episodes),
        "episode_count": len(episodes),
        "split_manifest": str(args.split_manifest),
        "split_usage": "development subset only (confirm300 untouched)",
        "arms": {},
    }
    for arm in ARMS:
        counters: Counter[str] = Counter()
        unsupported: Counter[str] = Counter()
        evictions = 0
        failed = 0
        for episode in episodes:
            context = dict(contexts[arm])
            context["activation"] = {}
            summary, events = simulate_episode(
                episode,
                templates,
                arm,
                train_stats=train_stats,
                policy_context=context,
                collect_events=True,
            )
            counters.update(context["activation"])
            if int(summary.get("failed_jobs") or 0):
                failed += 1
            evictions += sum(1 for event in events if event.get("event_type") == "model_evict")
            unsupported.update(summary.get("prefetch_interference_unsupported") or {})
        decisions = int(counters.get("decisions", 0))
        entry = {
            "decisions": decisions,
            "episodes": len(episodes),
            "episodes_with_failed_jobs": failed,
            "counters": dict(sorted(counters.items())),
            "eviction_events": evictions,
            "fail_closed_reasons": dict(sorted(unsupported.items())),
        }
        if decisions:
            entry["rates_per_decision"] = {
                key: round(value / decisions, 4)
                for key, value in sorted(counters.items())
                if key != "decisions"
            }
        report["arms"][arm] = entry

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "activation_report_v1.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("WROTE", out)
    for arm, entry in report["arms"].items():
        print(f"[{arm}] decisions={entry['decisions']} counters={entry['counters']} "
              f"evictions={entry['eviction_events']} failed_episodes={entry['episodes_with_failed_jobs']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
