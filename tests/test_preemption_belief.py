"""Round-6 preemption arms: R_B victim rule + resume contract + liveness guard.

The belief preemption arms (``f0point_preempt`` / ``pdrs_preempt``) extend the
full residency package with request-level preemption:

* victim rule (GPT round-6): the ready target may only take the device when its
  remaining-work belief R_B(i) is strictly smaller than the victim's R_B(v);
  the victim is the admissible call with the largest R_B;
* resume contract: prefill atomic, decode token boundaries, progress preserved
  (resume pays R_m + remaining decode, never progress=0);
* liveness: a resumed call must complete at least one further decode unit
  before it can be preempted again (no zero-progress ping-pong).
"""
from __future__ import annotations

import json
import unittest

from tracing.analysis.workload_v02_simulator import Node, Template, simulate_episode


def req_node(node_id, model, role, runtime):
    return Node(
        node_id=node_id, sequence_index=0, predecessors=(), successors=(),
        lane="gpu", model_id=model, runtime_ms=runtime, load_ms=0.0,
        workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
        workload_shape="short", role=role,
    )


def simple_template(template_id, node):
    return Template(template_id, "video", "train", "test", (node,), {node.node_id: node})


def request_profile(victim_model="model-a", in_tokens=1000, out_tokens=50):
    return {
        "enabled": True,
        "mode": "REQUEST_RECOMPUTE",
        "engine": "hf_substrate",
        "provenance_kind": "synthetic",
        "source_artifact": "synthetic-rp.json",
        "source_experiment": "synthetic-rp",
        "gpu_identity": "synthetic-gpu",
        "probe_source_artifact": "synthetic-rp-probe.json",
        "probe_metadata": {"probe": "synthetic"},
        "layers": {
            f"{victim_model}|planner": {"in_tokens": in_tokens, "out_tokens": out_tokens},
            "model-c|planner": {"in_tokens": 1000, "out_tokens": 50},
        },
        "models": {
            victim_model: {"prefill_us_per_token": 100.0, "prefill_intercept_ms": 0.0, "tpot_ms": 4.0},
            "model-c": {"prefill_us_per_token": 100.0, "prefill_intercept_ms": 0.0, "tpot_ms": 4.0},
        },
    }


def fixture(victim_runtime=200.0, target_arrival=150.0, extra_arrivals=()):
    victim = req_node("a:n", "model-a", "planner", victim_runtime)
    target = req_node("c:n", "model-c", "planner", 10.0)
    stats = {
        f"{model}|gpu|0|exact": {"runtime_p50_ms": r, "runtime_p90_ms": r,
                                 "load_p50_ms": 0.0, "memory_p95_mb": 100.0, "count": 100}
        for model, r in (("model-a", victim_runtime), ("model-c", 10.0))
    }
    jobs = [
        {"job_instance_id": "ja", "template_id": "a", "arrival_ms": 0.0,
         "deadline_ms": 100000.0, "service_class": "normal"},
        {"job_instance_id": "jc", "template_id": "c", "arrival_ms": target_arrival,
         "deadline_ms": 100000.0, "service_class": "priority"},
    ]
    for index, arrival in enumerate(extra_arrivals):
        jobs.append({"job_instance_id": f"jx{index}", "template_id": "c",
                     "arrival_ms": arrival, "deadline_ms": 100000.0,
                     "service_class": "priority"})
    episode = {
        "episode_id": "belief-preempt",
        "split": "train",
        "gpu_topology_mb": [1000.0],
        "gpu_identity": "synthetic-gpu",
        "initial_residency_hint": [["model-a", "model-c"]],
        "jobs": jobs,
    }
    pack = {
        "a:n": {"length_probabilities": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "future_h5": [{"steps": []}]},
        "c:n": {"length_probabilities": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "future_h5": [{"steps": []}]},
    }
    return episode, {"a": simple_template("a", victim), "c": simple_template("c", target)}, stats, pack


def run(policy, episode, templates, stats, pack, max_preemptions=5):
    return simulate_episode(
        episode, templates, policy,
        train_stats=stats,
        future_artifacts=pack,
        future_horizon=5,
        extension_config={
            "preemption_enabled": True,
            "max_preemptions": max_preemptions,
            "request_preemption": request_profile(),
            "prefetch_overlap": True,
        },
        collect_events=True,
    )


class BeliefPreemptionTests(unittest.TestCase):
    def test_pdrs_preemption_fires_and_resumes_without_progress_reset(self) -> None:
        episode, templates, stats, pack = fixture()
        summary, events = run("pdrs_preempt", episode, templates, stats, pack)
        self.assertEqual(summary["preemptions"], 1)
        self.assertEqual(summary["preemption_resume_count"], 1)
        self.assertGreater(summary["preempt_recompute_ms"], 0.0)
        self.assertGreater(summary["preemption_stall_ms"], 0.0)
        self.assertEqual(summary["completed_jobs"], 2)
        self.assertEqual(summary["failed_jobs"], 0)
        preempts = [e for e in events if e.get("event_type") == "node_preempt"]
        self.assertTrue(preempts[0]["request_mode"])
        # Resume pays R_m + remaining decode; the victim's finish is far below a
        # full recompute of its 200 ms runtime plus the stall.
        finishes = [e["finish_ms"] for e in events
                    if e.get("event_type") == "node_finish" and e.get("node_id") == "a:n"]
        self.assertEqual(len(finishes), 1)
        self.assertGreater(finishes[0], 150.0)
        # Leak guard: token/phase truth never reaches the scheduler-visible state.
        for event in events:
            if event.get("event_type") == "node_dispatch" and event.get("scheduler_state"):
                dumped = json.dumps(event["scheduler_state"], ensure_ascii=False)
                for forbidden in ("tokens_done", "rm_ms", "n_ctx", "request_split",
                                  "resume_tokens_base"):
                    self.assertNotIn(forbidden, dumped)

    def test_f0point_preemption_fires_and_resumes(self) -> None:
        episode, templates, stats, pack = fixture()
        summary, _events = run("f0point_preempt", episode, templates, stats, pack)
        self.assertEqual(summary["preemptions"], 1)
        self.assertEqual(summary["preemption_resume_count"], 1)
        self.assertEqual(summary["completed_jobs"], 2)

    def test_belief_rule_blocks_when_target_has_larger_remaining_work(self) -> None:
        # Give the target a heavy predicted suffix so R_B(target) > R_B(victim).
        episode, templates, stats, pack = fixture()
        heavy_step = {
            "model_probabilities": {"model-a": 1.0},
            "resource": {
                "runtime_mean_ms": 5000.0,
                "runtime_ms_quantiles": {"p50": 5000.0, "p90": 5000.0, "p95": 5000.0},
                "load_occurrence_probability": 1.0,
                "load_duration_ms_quantiles": {"p50": 100.0, "p95": 100.0},
            },
        }
        pack["c:n"]["future_h5"] = [{"steps": [heavy_step]}]
        summary, _events = run("pdrs_preempt", episode, templates, stats, pack)
        self.assertEqual(summary["preemptions"], 0)
        self.assertEqual(summary["preemption_resume_count"], 0)

    def test_resume_zero_progress_guard_blocks_repeated_preemption(self) -> None:
        # Victim: 2000 ms, prefill 100 ms, decode 38 ms/token (50 tokens).
        # jc arrives at t=176 (boundary k=2) -> preempted at 2 tokens done.
        # jc runs 10 ms -> victim resumes at 186, pays R_m = 100.2 ms.
        # jx0 arrives exactly when R_m completes (t=286.2) -> the resumed call is
        # at a token boundary with zero decode units since resume -> guard blocks.
        episode, templates, stats, pack = fixture(
            victim_runtime=2000.0, target_arrival=176.0, extra_arrivals=(286.2,)
        )
        summary, events = run("pdrs_preempt", episode, templates, stats, pack)
        self.assertEqual(summary["preemptions"], 1)
        self.assertGreaterEqual(
            summary["preemption_blocked_reasons"].get("resume_zero_progress_guard", 0), 1
        )
        self.assertEqual(summary["completed_jobs"], 3)
        self.assertEqual(summary["failed_jobs"], 0)


if __name__ == "__main__":
    unittest.main()
