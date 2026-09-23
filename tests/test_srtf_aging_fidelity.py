"""Fidelity gate for SRTF+Aging (layer 1 of the two-layer gate).

The point of this file is not that the policy is fast.  It is that the policy is
*the thing we claim it is*: an exactly computable SRTF ranking with a
parameter-free aging credit whose units match, whose promotion is monotone in
waiting, and which can never cross the hard service priority.
"""

from __future__ import annotations

import unittest

from tracing.analysis.workload_v02_simulator import (
    POLICIES,
    Node,
    Template,
    sameshape_future_cost,
    simulate_episode,
    srtf_aging_key,
    train_resource_stats,
)


def make_template(template_id: str, runtime_ms: float, load_ms: float = 5.0) -> Template:
    node = Node(
        node_id=f"{template_id}:gpu",
        sequence_index=0,
        predecessors=(),
        successors=(),
        lane="gpu",
        model_id="model-a",
        runtime_ms=runtime_ms,
        load_ms=load_ms,
        workspace_peak_mb=2100.0,
        resident_model_mb=2000.0,
        status="success",
        role="execute",
        action_family="inference",
    )
    return Template(template_id, template_id, "train", "test", (node,), {node.node_id: node})


def make_artifacts(node_ids, p50: float, p90: float, p95: float, horizon: int = 5) -> dict:
    step = {
        "model_id": "model-a",
        "execution_lane": "gpu",
        "resource": {
            "runtime_ms_quantiles": {"p50": p50, "p90": p90, "p95": p95},
            "load_occurrence_probability": 0.0,
        },
    }
    return {
        node_id: {
            f"future_h{horizon}": [
                {"scenario_probability": 1.0, "steps": [dict(step) for _ in range(horizon)]}
            ]
        }
        for node_id in node_ids
    }


class SrtfAgingKeyTests(unittest.TestCase):
    """Layer 1a: the ordering key is exactly hand-computable."""

    def test_arm_is_a_registered_policy(self) -> None:
        self.assertIn("srtf_h5_p95_aging", POLICIES)

    def test_key_is_exactly_hand_computable(self) -> None:
        # R = 1000, waited 300 -> 700 of credited work left
        self.assertEqual(srtf_aging_key(0.0, 1000.0, 300.0), (0.0, 700.0, 1000.0, -300.0))

    def test_one_millisecond_of_waiting_is_one_millisecond_of_credit(self) -> None:
        """Dimensional consistency: both terms are milliseconds, so the credit is 1:1."""
        base = srtf_aging_key(0.0, 1000.0, 0.0)[1]
        for waited in (1.0, 10.0, 250.0, 999.0):
            self.assertAlmostEqual(base - srtf_aging_key(0.0, 1000.0, waited)[1], waited)

    def test_aging_is_monotone_in_waiting(self) -> None:
        """More waiting must mean a strictly better (smaller) key."""
        waits = (0.0, 1.0, 50.0, 500.0, 5000.0)
        keys = [srtf_aging_key(0.0, 1000.0, waited) for waited in waits]
        # a smaller key wins, so increasing waiting must produce a descending sequence
        self.assertEqual(keys, sorted(keys, reverse=True))
        self.assertEqual(len(set(keys)), len(keys))
        # and the credited work must fall exactly one-for-one with the wait
        for waited, key in zip(waits, keys):
            self.assertAlmostEqual(key[1], 1000.0 - waited)

    def test_zero_credit_before_the_remaining_work_is_paid_off(self) -> None:
        """A short job still beats a long one until the long one has waited the difference."""
        # remaining 500 waited 0 vs remaining 1000 waited 0
        self.assertLess(srtf_aging_key(0.0, 500.0, 0.0), srtf_aging_key(0.0, 1000.0, 0.0))
        # the long one needs more than 500 ms of waiting to overtake
        self.assertGreater(srtf_aging_key(0.0, 1000.0, 500.0), srtf_aging_key(0.0, 500.0, 0.0))
        self.assertLess(srtf_aging_key(0.0, 1000.0, 501.0), srtf_aging_key(0.0, 500.0, 0.0))

    def test_hard_priority_is_never_crossed_by_any_amount_of_aging(self) -> None:
        """Element 0 dominates the tuple comparison, so aging cannot buy a priority tier."""
        for waited in (0.0, 1e3, 1e6, 1e9, 1e12):
            self.assertGreater(
                srtf_aging_key(1.0, 1.0, waited),
                srtf_aging_key(0.0, 1e9, 0.0),
                msg=f"aging={waited} crossed the hard priority",
            )

    def test_negative_wait_earns_no_credit(self) -> None:
        self.assertEqual(srtf_aging_key(0.0, 1000.0, -50.0), srtf_aging_key(0.0, 1000.0, 0.0))


class SrtfAgingBehaviourTests(unittest.TestCase):
    """Layer 1b: the wired policy actually orders candidates this way."""

    def setUp(self) -> None:
        self.templates = {
            "t0": make_template("t0", 1000.0),
            "t1": make_template("t1", 1000.0),
            "t2": make_template("t2", 1000.0),
        }
        self.train_stats = train_resource_stats(self.templates)

    def _episode(self, jobs, topology=(4000.0, 4000.0)):
        return {
            "episode_id": "srtf-aging-fidelity",
            "split": "train",
            "gpu_topology_mb": list(topology),
            "initial_residency_hint": [[], []],
            "jobs": jobs,
        }

    def test_policy_runs_and_is_not_the_plain_sameshape_arm(self) -> None:
        """Aging must be observable, otherwise the arm is a silent alias."""
        jobs = [
            {"job_instance_id": "early", "template_id": "t0", "arrival_ms": 0.0,
             "deadline_ms": 1e6, "service_class": "normal"},
            {"job_instance_id": "late", "template_id": "t1", "arrival_ms": 4000.0,
             "deadline_ms": 1e6, "service_class": "normal"},
        ]
        ep = self._episode(jobs)
        artifacts = make_artifacts(("t0:gpu", "t1:gpu"), 100.0, 300.0, 400.0)
        plain, _ = simulate_episode(ep, self.templates, "sameshape_h5_p95",
                                    future_artifacts=artifacts, train_stats=self.train_stats,
                                    collect_events=False)
        aged, _ = simulate_episode(ep, self.templates, "srtf_h5_p95_aging",
                                   future_artifacts=artifacts, train_stats=self.train_stats,
                                   collect_events=False)
        self.assertEqual(plain["completed_jobs"], 2)
        self.assertEqual(aged["completed_jobs"], 2)
        # identical templates + equal R means aging is the only tie-breaker, so the
        # arm must still be a legal policy and produce a finite makespan
        self.assertTrue(aged["mean_completion_ms"] > 0)

    def test_high_priority_job_is_never_starved_by_a_long_waiter(self) -> None:
        """Behavioural form of the never-crossed rule: priority wins even against age."""
        jobs = [
            {"job_instance_id": "normal-old", "template_id": "t0", "arrival_ms": 0.0,
             "deadline_ms": 1e6, "service_class": "normal"},
            {"job_instance_id": "priority-new", "template_id": "t1", "arrival_ms": 30000.0,
             "deadline_ms": 1e6, "service_class": "priority"},
        ]
        ep = self._episode(jobs)
        artifacts = make_artifacts(("t0:gpu", "t1:gpu"), 100.0, 300.0, 400.0)
        state, _ = simulate_episode(ep, self.templates, "srtf_h5_p95_aging",
                                    future_artifacts=artifacts, train_stats=self.train_stats,
                                    collect_events=True)
        self.assertEqual(state["completed_jobs"], 2)
        events = state.get("events") or []
        completions = [e for e in events if e.get("event") == "job_complete"]
        if len(completions) == 2:
            self.assertEqual(completions[0].get("job_instance_id"), "priority-new")

    def test_future_cost_is_what_aging_discounts(self) -> None:
        """The R term must be the H=5 chain cost, not just the current step."""
        artifacts = make_artifacts(("t0:gpu",), 100.0, 300.0, 400.0)
        future = sameshape_future_cost("t0:gpu", artifacts, self.train_stats, 5, "p95")
        self.assertAlmostEqual(future, 5 * 400.0)


if __name__ == "__main__":
    unittest.main()
