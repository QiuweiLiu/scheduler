from __future__ import annotations

import unittest

from scripts.r8_action_value_audit import audit_dispatch_event, run_audit
from tracing.analysis.workload_v02_simulator import Node, Template, simulate_episode, train_resource_stats


def make_templates() -> dict[str, Template]:
    root_a = Node(
        node_id="a0",
        sequence_index=0,
        predecessors=(),
        successors=("a1",),
        lane="gpu",
        model_id="model-a",
        runtime_ms=10.0,
        load_ms=0.0,
        workspace_peak_mb=2100.0,
        resident_model_mb=2000.0,
        status="success",
        role="plan",
        action_family="planner",
    )
    tail_a = Node(
        node_id="a1",
        sequence_index=1,
        predecessors=("a0",),
        successors=(),
        lane="gpu",
        model_id="model-a",
        runtime_ms=20.0,
        load_ms=0.0,
        workspace_peak_mb=2100.0,
        resident_model_mb=2000.0,
        status="success",
        role="answer",
        action_family="answer",
    )
    root_b = Node(
        node_id="b0",
        sequence_index=0,
        predecessors=(),
        successors=("b1",),
        lane="gpu",
        model_id="model-b",
        runtime_ms=10.0,
        load_ms=0.0,
        workspace_peak_mb=2100.0,
        resident_model_mb=2000.0,
        status="success",
        role="plan",
        action_family="planner",
    )
    tail_b = Node(
        node_id="b1",
        sequence_index=1,
        predecessors=("b0",),
        successors=(),
        lane="gpu",
        model_id="model-b",
        runtime_ms=80.0,
        load_ms=0.0,
        workspace_peak_mb=2100.0,
        resident_model_mb=2000.0,
        status="success",
        role="answer",
        action_family="answer",
    )
    return {
        "template-a": Template("template-a", "video-a", "train", "test", (root_a, tail_a), {"a0": root_a, "a1": tail_a}),
        "template-b": Template("template-b", "video-b", "train", "test", (root_b, tail_b), {"b0": root_b, "b1": tail_b}),
    }


def make_episode() -> dict[str, object]:
    return {
        "episode_id": "audit-smoke-0",
        "split": "train",
        "gpu_topology_mb": [24000.0, 24000.0],
        "initial_residency_hint": [[], []],
        "jobs": [
            {"job_instance_id": "job-a", "template_id": "template-a", "arrival_ms": 0.0, "service_class": "normal"},
            {"job_instance_id": "job-b", "template_id": "template-b", "arrival_ms": 0.0, "service_class": "normal"},
        ],
    }


class ActionValueAuditTests(unittest.TestCase):
    def test_audit_reconstructs_common_state_without_mutating_simulation(self) -> None:
        templates = make_templates()
        episode = make_episode()
        train_stats = train_resource_stats(templates)
        future_artifacts = {node_id: {"future_h5": []} for node_id in ("a0", "a1", "b0", "b1")}
        first_summary, first_events = simulate_episode(
            episode, templates, "myopic", future_artifacts=future_artifacts, train_stats=train_stats, collect_events=True
        )
        dispatch = next(event for event in first_events if event["event_type"] == "node_dispatch")
        record = audit_dispatch_event(episode, templates, dispatch, train_stats, future_artifacts)
        self.assertEqual(record["schema_version"], "action-value-audit-v0.1")
        self.assertGreater(record["raw_candidate_width"], 0)
        self.assertGreater(record["strict_feasible_width"], 0)
        self.assertEqual(record["raw_candidate_width"], len(record["candidates"]))
        self.assertTrue(all(row["strict_feasible"] for row in record["candidates"] if row["strict_feasible"]))
        self.assertTrue(record["state_hash"])

        job_index = next(index for index, job in enumerate(("job-a", "job-b")) if job == dispatch["job_instance_id"])
        forced_context = {
            "audit_forced_decision_index": dispatch["scheduler_state"]["decision_index"],
            "audit_forced_action": (job_index, dispatch["node_id"], dispatch["gpu_index"]),
        }
        forced_summary, _forced_events = simulate_episode(
            episode,
            templates,
            "myopic",
            future_artifacts=future_artifacts,
            train_stats=train_stats,
            collect_events=True,
            policy_context=forced_context,
        )
        self.assertEqual(first_summary, forced_summary)

        second_summary, second_events = simulate_episode(
            episode, templates, "myopic", future_artifacts=future_artifacts, train_stats=train_stats, collect_events=True
        )
        self.assertEqual(first_summary, second_summary)
        self.assertEqual(first_events, second_events)

    def test_run_audit_reports_smoke_metrics(self) -> None:
        templates = make_templates()
        episode = make_episode()
        train_stats = train_resource_stats(templates)
        future_artifacts = {node_id: {"future_h5": []} for node_id in ("a0", "a1", "b0", "b1")}
        report, records, summaries = run_audit(
            [episode], templates, future_artifacts, train_stats, limit=1, reference_policy="myopic", horizon=5
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["episodes"], 1)
        self.assertEqual(len(summaries), 1)
        self.assertGreater(len(records), 0)
        self.assertIn("predopt_h5", report["metrics"]["policies"])

    def test_run_audit_supports_opt_in_aligned_contract(self) -> None:
        templates = make_templates()
        episode = make_episode()
        train_stats = train_resource_stats(templates)
        future_artifacts = {node_id: {"future_h5": []} for node_id in ("a0", "a1", "b0", "b1")}
        report, records, _summaries = run_audit(
            [episode],
            templates,
            future_artifacts,
            train_stats,
            limit=1,
            reference_policy="myopic",
            horizon=5,
            policies=("myopic", "aligned_predopt_h5", "aligned_trueopt_h5"),
            truth_contract="aligned_h5",
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["truth_contract"], "aligned_h5")
        self.assertEqual(
            report["policies"],
            ["myopic", "aligned_predopt_h5", "aligned_trueopt_h5"],
        )
        self.assertGreater(len(records), 0)
        self.assertIn("aligned_predopt_h5", report["metrics"]["policies"])
        self.assertIn("aligned_trueopt_h5", report["metrics"]["policies"])


if __name__ == "__main__":
    unittest.main()
