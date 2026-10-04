#!/usr/bin/env python3
"""Main-table baseline activation audit v2 (pre-run evidence; local, no GPU).

Two phases:

* **Phase A** — the real development subset (30 episodes, causal_v3, NO
  extension config): the ordering mechanisms of all four arms with corrected
  counters.  The profile-dependent mechanisms cannot be configured here because
  the frozen episodes/templates do not yet carry workload_shape / GPU identity;
  that gap is reported, not papered over.
* **Phase B** — a DECLARED synthetic smoke set carrying the same real models,
  shapes, GPU identity and the real extension profile artifacts, used only to
  exercise the profile-dependent mechanisms (Hermes online prewarm under a
  covered F4 cell, Torpor covered-load ranking, QLM load counterfactual).  No
  performance claims come from phase B.

The acceptance question is only whether each mechanism genuinely entered the
decision path; real activation rates are reported as-is (no ratio threshold).
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
    Node,
    Template,
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


def _node(node_id, model, role, runtime, shape, predecessors=()):
    return Node(
        node_id=node_id, sequence_index=0, predecessors=predecessors, successors=(),
        lane="gpu", model_id=model, runtime_ms=runtime, load_ms=0.0,
        workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
        workload_shape=shape, role=role,
    )


def synthetic_profile_smoke(extension_config, train_stats):
    """One declared synthetic episode exercising Hermes prewarm under the real
    F4 cell (8B|medium <- 3B) with the real PDGraph state (13, videotool_spatial).
    Filler nodes use only models that are resident or already loaded so the 3B
    target stays cold until the prewarm fires."""

    nodes = []
    previous = ()
    for index in range(12):
        node = _node(f"pre:n{index}", "Qwen3-4B", "videotool_temporal", 120.0, "medium", previous)
        nodes.append(node)
        previous = (node.node_id,)
    runner = _node("pre:n12", "Qwen3-VL-8B-Instruct", "videotool_spatial", 4200.0, "medium", previous)
    answer = _node("pre:n13", "Qwen2.5-VL-3B-Instruct", "answer_generation", 500.0, "short", (runner.node_id,))
    nodes.extend([runner, answer])
    template = Template(
        "pre_smoke", "v", "validation", "b", tuple(nodes), {n.node_id: n for n in nodes},
        workflow_type_id="videomme.langgraph_react",
    )
    episode = {
        "episode_id": "synthetic-prewarm-smoke",
        "split": "validation",
        "gpu_topology_mb": [32760.0],
        "gpu_identity": "NVIDIA GeForce RTX 4080 SUPER",
        "initial_residency_hint": [["Qwen3-VL-8B-Instruct"]],
        "jobs": [
            {"job_instance_id": "j", "template_id": "pre_smoke", "arrival_ms": 0.0,
             "deadline_ms": 600000.0, "service_class": "normal"},
        ],
    }
    return episode, {"pre_smoke": template}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--extension-config", type=Path,
                        help="real profile bundle for phase B (optional; phase B skipped without it)")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    templates = load_templates(args.templates, topology_view="causal_v3")
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
    # Parrot disclosure: the serial-chain projection has no task-group parallelism.
    successor_counts = Counter(len(node.successors) for template in templates.values() for node in template.nodes)
    max_successors = max(successor_counts) if successor_counts else 0
    workflow_distribution = Counter(
        str(template.workflow_type_id or "") for template in templates.values()
    )

    bank = build_duration_bank(templates)
    graph = build_pdgraph(templates)
    contexts = {
        "parrot_appfifo": {},
        "qlm_queue": {"qlm_duration_bank": bank, "qlm_rng": random.Random(args.seed)},
        "hermes_gittins": {"hermes_pdgraph": graph},
        "torpor_lifecycle": {},
    }

    report = {
        "schema": "main-table-activation-audit-v2",
        "seed": args.seed,
        "templates": str(args.templates),
        "templates_sha256": sha256(args.templates),
        "episodes": str(args.episodes),
        "episode_count": len(episodes),
        "split_manifest": str(args.split_manifest),
        "split_usage": "development subset only (confirm300 untouched)",
        "workflow_type_distribution": {
            "counts": {key: value for key, value in sorted(workflow_distribution.items())},
            "note": "reviewer reproducibility: the Hermes PDGraph conditioning needs more than one "
                    "real workflow family; this distribution plus the templates sha pins it",
        },
        "task_group_size_disclosure": {
            "successor_count_distribution": {str(k): v for k, v in sorted(successor_counts.items())},
            "max_successors": max_successors,
            "note": "serial-chain projection: task_group_size == 1 for every node, so Parrot's "
                    "parallel task-group scheduling has no room and is disclosed as omitted",
        },
        "phase_a": {},
        "phase_b": None,
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
                episode, templates, arm, train_stats=train_stats,
                policy_context=context, collect_events=True,
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
            "profile_attached": False,
        }
        if decisions:
            entry["rates_per_decision"] = {
                key: round(value / decisions, 4)
                for key, value in sorted(counters.items())
                if key != "decisions"
            }
        report["phase_a"][arm] = entry

    if args.extension_config is not None:
        extension_config = json.loads(args.extension_config.read_text(encoding="utf-8"))
        # The Hermes prewarm path is gated by the prefetch feature flag; the
        # profile builder does not emit it, so the runner (and this audit) must.
        extension_config = {**extension_config, "prefetch_overlap": True}
        episode, smoke_templates = synthetic_profile_smoke(extension_config, train_stats)
        phase_b = {}
        for arm in ARMS:
            context = dict(contexts[arm])
            context["activation"] = {}
            summary, events = simulate_episode(
                episode, smoke_templates, arm, train_stats=train_stats,
                policy_context=context, extension_config=extension_config, collect_events=True,
            )
            phase_b[arm] = {
                "counters": dict(sorted(context["activation"].items())),
                "failed_jobs": int(summary.get("failed_jobs") or 0),
                "prefetch_starts": [
                    {"model_id": event.get("model_id"), "time_ms": event.get("time_ms")}
                    for event in events
                    if event.get("event_type") == "prefetch_start"
                ],
                "interference_events": [
                    {"extra_ms": event.get("extra_ms")}
                    for event in events
                    if event.get("event_type") == "node_interference_start"
                ],
                "fail_closed_reasons": dict(sorted((summary.get("prefetch_interference_unsupported") or {}).items())),
            }
        report["phase_b"] = {
            "declared": "synthetic smoke; real models/shapes/identity and the real extension artifacts; "
                        "no performance claims",
            "episode": episode["episode_id"],
            "arms": phase_b,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "activation_report_v2.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("WROTE", out)
    for arm, entry in report["phase_a"].items():
        print(f"[A:{arm}] decisions={entry['decisions']} counters={entry['counters']} "
              f"evictions={entry['eviction_events']}")
    if report["phase_b"]:
        for arm, entry in report["phase_b"]["arms"].items():
            print(f"[B:{arm}] counters={entry['counters']} prewarm={entry['prefetch_starts']} "
                  f"interference={entry['interference_events']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
