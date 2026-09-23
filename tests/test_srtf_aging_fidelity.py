"""Fidelity gate for the aging-augmented remaining-work arm (layer 1).

The point of this file is not that the policy is fast.  It is that the policy is
*the thing we claim it is*: an exactly computable predicted-remaining-work
ranking with a parameter-free node-level aging credit whose units match, whose
promotion is monotone in waiting, which can never cross the hard service
priority, and which is not a silent alias of the plain same-shape arm.

Review-driven corrections (all three were real defects in the first draft):
  * negative ready ages now fail closed instead of being silently clamped;
  * the priority test used the wrong event name (job_complete vs job_finish) and
    its two jobs never competed in the same decision state, so it asserted
    nothing.  It now uses a blocker so both candidates are ready at once;
  * the "not a silent alias" test compared identical candidates, where the plain
    arm's ready-time tie-break already produces the same winner.  It now
    constructs a case where the two arms provably pick different winners.
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


def make_artifacts(p95_by_node, horizon: int = 5) -> dict:
    out = {}
    for node_id, p95 in p95_by_node.items():
        step = {
            "model_id": "model-a",
            "execution_lane": "gpu",
            "resource": {
                "runtime_ms_quantiles": {"p50": p95, "p90": p95, "p95": p95},
                "load_occurrence_probability": 0.0,
            },
        }
        out[node_id] = {
            f"future_h{horizon}": [
                {"scenario_probability": 1.0, "steps": [dict(step) for _ in range(horizon)]}
            ]
        }
    return out


def one_gpu_episode(jobs):
    return {
        "episode_id": "aging-fidelity",
        "split": "train",
        "gpu_topology_mb": [4000.0],
        "initial_residency_hint": [[]],
        "jobs": jobs,
    }


def job(job_id, template_id, arrival_ms, service_class="normal"):
    return {
        "job_instance_id": job_id,
        "template_id": template_id,
        "arrival_ms": float(arrival_ms),
        "deadline_ms": 1e9,
        "service_class": service_class,
    }


def starts_of(events):
    """simulate_episode returns (summary, events); the event log is the second value."""
    return [e for e in (events or []) if e.get("event_type") == "node_start"]


class SrtfAgingKeyTests(unittest.TestCase):
    """Layer 1a: the ordering key is exactly hand-computable."""

    def test_arm_is_a_registered_policy(self) -> None:
        self.assertIn("sameshape_h5_p95_aging", POLICIES)

    def test_key_is_exactly_hand_computable(self) -> None:
        self.assertEqual(srtf_aging_key(0.0, 1000.0, 300.0), (0.0, 700.0, 1000.0, -300.0))

    def test_one_millisecond_of_waiting_is_one_millisecond_of_credit(self) -> None:
        """Dimensional consistency: both terms are milliseconds, so the credit is 1:1."""
        base = srtf_aging_key(0.0, 1000.0, 0.0)[1]
        for waited in (1.0, 10.0, 250.0, 999.0):
            self.assertAlmostEqual(base - srtf_aging_key(0.0, 1000.0, waited)[1], waited)

    def test_aging_is_monotone_in_waiting(self) -> None:
        """More waiting must mean a strictly better (smaller) key."""
        waits = (0.0, 1.0, 50.0, 500.0, 5000.0)
        keys = [srtf_aging_key(0.0, 1000.0, w) for w in waits]
        self.assertEqual(keys, sorted(keys, reverse=True))
        self.assertEqual(len(set(keys)), len(keys))
        for waited, key in zip(waits, keys):
            self.assertAlmostEqual(key[1], 1000.0 - waited)

    def test_only_the_ready_time_matters_not_the_absolute_wait(self) -> None:
        """S_j = R_j + r_j - t, so at a fixed decision time only r_j separates candidates.

        This is the property the review flagged: "is aging too weak" must be judged
        on |dr| versus |dR|, never on wait/R.  At one decision time t, a candidate
        that has been ready for 9000 ms and one ready for 1000 ms have the same R
        here, so the older-ready one must win.
        """
        ready_9000_ago = srtf_aging_key(0.0, 5000.0, 9000.0)[1]
        ready_1000_ago = srtf_aging_key(0.0, 5000.0, 1000.0)[1]
        self.assertLess(ready_9000_ago, ready_1000_ago)
        self.assertAlmostEqual(ready_9000_ago, 5000.0 - 9000.0)
        self.assertAlmostEqual(ready_1000_ago, 5000.0 - 1000.0)
        # the credit is exactly the ready-time gap, which is the quantity that matters
        self.assertAlmostEqual(ready_1000_ago - ready_9000_ago, 9000.0 - 1000.0)

    def test_zero_credit_before_the_remaining_work_is_paid_off(self) -> None:
        self.assertLess(srtf_aging_key(0.0, 500.0, 0.0), srtf_aging_key(0.0, 1000.0, 0.0))
        self.assertGreater(srtf_aging_key(0.0, 1000.0, 500.0), srtf_aging_key(0.0, 500.0, 0.0))
        self.assertLess(srtf_aging_key(0.0, 1000.0, 501.0), srtf_aging_key(0.0, 500.0, 0.0))

    def test_hard_priority_is_never_crossed_by_any_amount_of_aging(self) -> None:
        for waited in (0.0, 1e3, 1e6, 1e9, 1e12):
            self.assertGreater(
                srtf_aging_key(1.0, 1.0, waited),
                srtf_aging_key(0.0, 1e9, 0.0),
                msg=f"aging={waited} crossed the hard priority",
            )

    def test_materially_negative_wait_fails_closed(self) -> None:
        """A broken simulator invariant must not be hidden by a clamp."""
        with self.assertRaises(ValueError):
            srtf_aging_key(0.0, 1000.0, -50.0)
        with self.assertRaises(ValueError):
            srtf_aging_key(0.0, 1000.0, -1.0)

    def test_floating_point_noise_is_still_clamped(self) -> None:
        for noise in (-1e-12, -1e-15, 0.0):
            self.assertEqual(
                srtf_aging_key(0.0, 1000.0, noise),
                srtf_aging_key(0.0, 1000.0, 0.0),
                msg=f"noise={noise} should clamp to zero",
            )


class SrtfAgingBehaviourTests(unittest.TestCase):
    """Layer 1b: the wired policy actually orders candidates this way."""

    def setUp(self) -> None:
        self.templates = {
            "blocker": make_template("blocker", 5000.0),
            "old": make_template("old", 0.0),
            "new": make_template("new", 0.0),
        }
        self.train_stats = train_resource_stats(self.templates)

    def test_priority_wins_against_a_much_older_normal_candidate(self) -> None:
        """Both candidates must actually compete, and priority must still win.

        A blocker occupies the single GPU until t=5000.  The normal job has been
        ready since t=0 (age 5000 ms) while the priority job only arrives at
        t=4999, so this is the strongest form of the never-crossed claim.  The
        event name is node_start, which is the scheduler action itself, and the
        count assertion is fail-closed.
        """
        ep = one_gpu_episode([
            job("blocker", "blocker", 0, "priority"),
            job("normal-old", "old", 0, "normal"),
            job("priority-new", "new", 4999, "priority"),
        ])
        artifacts = make_artifacts({"blocker:gpu": 0.0, "old:gpu": 1000.0, "new:gpu": 1000.0})
        _state, events = simulate_episode(ep, self.templates, "sameshape_h5_p95_aging",
                                          future_artifacts=artifacts, train_stats=self.train_stats,
                                          collect_events=True)
        starts = starts_of(events)
        by_job = [e.get("job_instance_id") for e in starts]
        # fail closed: if the event vocabulary changes this test must break loudly
        self.assertIn("priority-new", by_job, f"priority job never started; events={by_job}")
        self.assertIn("normal-old", by_job, f"normal job never started; events={by_job}")
        self.assertLess(by_job.index("priority-new"), by_job.index("normal-old"),
                        "aging let a 5000 ms-old normal candidate overtake priority")

    def test_aging_is_not_a_silent_alias_of_the_plain_sameshape_arm(self) -> None:
        """The two arms must provably pick different winners on some state.

        Construction: both candidates share the same current cost, so R is driven
        by the H=5 future term.  old has R = c + 5000 and new has R = c + 500.  At
        t=5000 the old candidate has accumulated 5000 ms of credit, so
        plain picks new (c+500 < c+5000) while aging picks old (c < c+500).
        """
        ep = one_gpu_episode([
            job("blocker", "blocker", 0, "priority"),
            job("old", "old", 0, "normal"),
            job("new", "new", 5000, "normal"),
        ])
        artifacts = make_artifacts({"blocker:gpu": 0.0, "old:gpu": 1000.0, "new:gpu": 100.0})
        runs = {}
        for policy in ("sameshape_h5_p95", "sameshape_h5_p95_aging"):
            _state, events = simulate_episode(ep, self.templates, policy,
                                              future_artifacts=artifacts, train_stats=self.train_stats,
                                              collect_events=True)
            runs[policy] = [e.get("job_instance_id") for e in starts_of(events)]

        plain, aged = runs["sameshape_h5_p95"], runs["sameshape_h5_p95_aging"]
        self.assertEqual(plain[0], "blocker")
        self.assertEqual(aged[0], "blocker")
        # the second start is the contested decision
        self.assertGreaterEqual(len(plain), 2, f"plain only started {plain}")
        self.assertGreaterEqual(len(aged), 2, f"aging only started {aged}")
        self.assertEqual(plain[1], "new", f"plain should prefer the short candidate: {plain}")
        self.assertEqual(aged[1], "old", f"aging should promote the old candidate: {aged}")
        self.assertNotEqual(plain[1], aged[1], "aging is a silent alias of the plain arm")

    def test_future_cost_is_the_h5_chain_cost(self) -> None:
        artifacts = make_artifacts({"old:gpu": 400.0})
        future = sameshape_future_cost("old:gpu", artifacts, self.train_stats, 5, "p95")
        self.assertAlmostEqual(future, 5 * 400.0)


if __name__ == "__main__":
    unittest.main()
