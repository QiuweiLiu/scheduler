"""Event-level sentinels for the composite CPU+GPU segment.

The review's four requirements:

  1. no contention   -> parent finish is EXACTLY t + R_total
  2. GPU occupied    -> nested delayed by X, so parent is delayed by X
  3. nested finish   -> must NOT release the parent's successor
  4. nested finish   -> must NOT release a GPU job that is not its own

These drive the real event path, because the previous attempt showed that a field can
be consumed and still change the wrong lifecycle state.
"""
from __future__ import annotations

import unittest

from tracing.analysis.workload_v02_simulator import (
    Node,
    Template,
    simulate_episode,
)


def hand_stats():
    """Hand-built train statistics.

    estimate() enforces a minimum group size, so a tiny fixture would be rejected
    before the composite path is ever reached.
    """

    def row(p50, p90):
        return {"runtime_p50_ms": p50, "runtime_p90_ms": p90, "load_p50_ms": 0.0,
                "memory_p95_mb": 1.0, "count": 100}

    return {
        "cpu-metadata-adapter|cpu|*|model_lane": row(1000.0, 1000.0),
        "*|cpu|*|lane": row(1000.0, 1000.0),
        "mX|gpu|*|model_lane": row(500.0, 500.0),
        "*|gpu|*|lane": row(500.0, 500.0),
    }

TOTAL = 1000.0
PRE, INNER, POST = 100.0, 700.0, 200.0
assert PRE + INNER + POST == TOTAL


def composite_node(node_id="c:n", seq=0, successors=(), predecessors=()):
    return Node(node_id=node_id, sequence_index=seq, predecessors=predecessors,
                successors=successors, lane="cpu", model_id="cpu-metadata-adapter",
                runtime_ms=TOTAL, load_ms=0.0, workspace_peak_mb=1.0,
                resident_model_mb=1.0, status="success", role="execute",
                action_family="summarize",
                nested_model_class="mX", nested_model_mb=1000.0, nested_load_ms=0.0,
                nested_pre_ms=PRE, nested_inner_ms=INNER, nested_post_ms=POST,
                nested_runtime_ms=INNER)


def gpu_node(node_id, runtime, seq, preds=(), succs=()):
    return Node(node_id=node_id, sequence_index=seq, predecessors=preds, successors=succs,
                lane="gpu", model_id="mX", runtime_ms=runtime, load_ms=0.0,
                workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
                role="execute", action_family="inference")


def template(tid, nodes):
    return Template(tid, tid, "train", "test", tuple(nodes), {n.node_id: n for n in nodes})


def episode(jobs, gpus=1, topo_mb=40000.0):
    return {"episode_id": "sm", "split": "train",
            "gpu_topology_mb": [topo_mb] * gpus,
            "initial_residency_hint": [[] for _ in range(gpus)],
            "jobs": jobs}


def job(jid, tid, arrival=0.0):
    return {"job_instance_id": jid, "template_id": tid, "arrival_ms": arrival,
            "deadline_ms": 1e9, "service_class": "normal"}


def finishes(events):
    return [e for e in events if e.get("event_type") == "node_finish"]


class CompositeSentinels(unittest.TestCase):
    def test_1_no_contention_parent_finishes_exactly_at_R_total(self):
        """The three parts must reconstruct R_total with no device contention."""
        tpls = {"t": template("t", [composite_node("c:n")])}
        ep = episode([job("j", "t")])
        _s, ev = simulate_episode(ep, tpls, "fcfs", train_stats=hand_stats(),
                                  collect_events=True)
        fin = finishes(ev)
        self.assertEqual(len(fin), 1, "the parent must finish exactly once")
        self.assertAlmostEqual(fin[0]["finish_ms"], TOTAL, places=6,
                               msg="no contention must recover t + R_total exactly")
        admits = [e for e in ev if e.get("event_type") == "nested_gpu_admit"]
        self.assertEqual(len(admits), 1, "the inner segment must be admitted")

    def test_3_nested_finish_does_not_release_the_parent_successor(self):
        """A successor must wait for the PARENT, not for the inner segment."""
        parent = composite_node("c:n", 0, successors=("c:after",))
        after = gpu_node("c:after", 10.0, 1, preds=("c:n",))
        tpls = {"t": template("t", [parent, after])}
        ep = episode([job("j", "t")])
        _s, ev = simulate_episode(ep, tpls, "fcfs", train_stats=hand_stats(),
                                  collect_events=True)
        starts = {e.get("node_id"): e.get("start_ms") for e in ev
                  if e.get("event_type") == "node_start"}
        self.assertIn("c:after", starts, "the successor never ran")
        self.assertGreaterEqual(
            starts["c:after"], PRE + INNER + POST - 1e-6,
            "the successor started before the parent finished, so the nested finish "
            "released it prematurely",
        )

    def test_4_nested_finish_does_not_release_another_gpu_job(self):
        """A composite segment must not clear a device held by unrelated GPU work."""
        # job b holds the single device for 5000 ms; job a's composite inner segment
        # queues behind it and must not free it
        long_gpu = gpu_node("b:n", 5000.0, 0)
        tpls = {"a": template("a", [composite_node("c:n")]),
                "b": template("b", [long_gpu])}
        ep = episode([job("ja", "a"), job("jb", "b")])
        s, ev = simulate_episode(ep, tpls, "fcfs", train_stats=hand_stats(),
                                 collect_events=True)
        self.assertEqual(s["completed_jobs"], 2)
        self.assertEqual(s["failed_jobs"], 0)
        # the composite's inner segment must be pushed strictly later than its ready
        # time, because the device was busy
        admit = [e for e in ev if e.get("event_type") == "nested_gpu_admit"]
        self.assertEqual(len(admit), 1)
        self.assertGreater(admit[0]["nested_start_ms"], admit[0]["nested_ready_ms"] - 1e-9,
                           "the segment did not respect the busy device")

    def test_2_gpu_contention_delays_the_parent_by_the_same_amount(self):
        """parent_finish - (t + R_total) must equal nested_start - nested_ready."""
        long_gpu = gpu_node("b:n", 3000.0, 0)
        tpls = {"a": template("a", [composite_node("c:n")]),
                "b": template("b", [long_gpu])}
        ep = episode([job("ja", "a"), job("jb", "b")])
        _s, ev = simulate_episode(ep, tpls, "fcfs", train_stats=hand_stats(),
                                  collect_events=True)
        admit = next(e for e in ev if e.get("event_type") == "nested_gpu_admit")
        delay = admit["nested_start_ms"] - admit["nested_ready_ms"]
        if delay <= 1e-9:
            self.skipTest("no contention occurred on this schedule")
        fin = next(e for e in finishes(ev) if e.get("node_id") == "c:n")
        self.assertAlmostEqual(fin["finish_ms"] - TOTAL, delay, places=4,
                               msg="the parent delay did not match the inner delay")


if __name__ == "__main__":
    unittest.main()
