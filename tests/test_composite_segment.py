"""Event-level sentinels for the composite CPU+GPU segment.

The six the review requires:

  1. no contention      -> parent finish is EXACTLY t + R_total
  2. forced contention  -> the parent is delayed by exactly the inner delay
  3. successor isolation-> a successor waits for the PARENT, not the inner segment
  4. foreign isolation  -> an inner segment never releases another job's device
  5. two composites     -> they serialise, they never overlap
  6. GPU stays usable during R_pre  -> a short normal GPU job can finish before
                                       nested_ready, which is what the old
                                       "reserve from t=0" model got wrong
"""
from __future__ import annotations

import unittest

from tracing.analysis.workload_v02_simulator import Node, Template, simulate_episode


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


def composite_node(node_id, seq, pre, inner, post, succs=()):
    return Node(node_id=node_id, sequence_index=seq, predecessors=(), successors=succs,
                lane="cpu", model_id="cpu-metadata-adapter",
                runtime_ms=pre + inner + post, load_ms=0.0, workspace_peak_mb=1.0,
                resident_model_mb=1.0, status="success", role="execute",
                action_family="summarize",
                nested_model_class="mX", nested_model_mb=1000.0, nested_load_ms=0.0,
                nested_pre_ms=pre, nested_inner_ms=inner, nested_post_ms=post,
                nested_runtime_ms=inner)


def gpu_node(node_id, seq, runtime, preds=(), succs=()):
    return Node(node_id=node_id, sequence_index=seq, predecessors=preds, successors=succs,
                lane="gpu", model_id="mX", runtime_ms=runtime, load_ms=0.0,
                workspace_peak_mb=100.0, resident_model_mb=100.0, status="success",
                role="execute", action_family="inference")


def template(tid, nodes):
    return Template(tid, tid, "train", "test", tuple(nodes), {n.node_id: n for n in nodes})


def episode(jobs, gpus=1):
    return {"episode_id": "sm", "split": "train",
            "gpu_topology_mb": [40000.0] * gpus,
            "initial_residency_hint": [[] for _ in range(gpus)],
            "jobs": jobs}


def job(jid, tid, arrival=0.0):
    return {"job_instance_id": jid, "template_id": tid, "arrival_ms": arrival,
            "deadline_ms": 1e9, "service_class": "normal"}


def run(tpls, jobs, gpus=1):
    _s, ev = simulate_episode(episode(jobs, gpus), tpls, "fcfs",
                              train_stats=hand_stats(), collect_events=True)
    return _s, ev


def by_type(ev, kind, node_id=None):
    return [e for e in ev if e.get("event_type") == kind
            and (node_id is None or e.get("node_id") == node_id)]


class CompositeSentinels(unittest.TestCase):
    def test_1_no_contention_parent_finishes_exactly_at_R_total(self):
        pre, inner, post = 100.0, 700.0, 200.0
        tpls = {"t": template("t", [composite_node("c:n", 0, pre, inner, post)])}
        _s, ev = run(tpls, [job("j", "t")])
        fin = by_type(ev, "node_finish", "c:n")
        self.assertEqual(len(fin), 1, "the parent must finish exactly once")
        self.assertAlmostEqual(fin[0]["finish_ms"], pre + inner + post, places=6,
                               msg="no contention must recover t + R_total exactly")

    def test_2_forced_contention_delays_the_parent_by_the_inner_delay(self):
        """Deterministic: a 3000 ms GPU job holds the only device.

        composite arrival 10, pre 100 -> nested_ready 110, and the device is busy until
        3000, so nested_start must be exactly 3000 and the parent must absorb the delay.
        """
        pre, inner, post = 100.0, 700.0, 200.0
        tpls = {"a": template("a", [composite_node("c:n", 0, pre, inner, post)]),
                "b": template("b", [gpu_node("b:n", 0, 3000.0)])}
        _s, ev = run(tpls, [job("jb", "b", 0.0), job("ja", "a", 10.0)])
        start = by_type(ev, "nested_gpu_start", "c:n")
        self.assertEqual(len(start), 1, "the inner segment never started")
        self.assertAlmostEqual(start[0]["nested_start_ms"], 3000.0, places=4,
                               msg="the segment did not queue behind the busy device")
        admit = by_type(ev, "nested_gpu_admit", "c:n")[0]
        delay = start[0]["nested_start_ms"] - admit["nested_ready_ms"]
        self.assertAlmostEqual(delay, 3000.0 - (10.0 + pre), places=4)
        fin = by_type(ev, "node_finish", "c:n")[0]
        self.assertAlmostEqual(fin["finish_ms"] - (10.0 + pre + inner + post), delay,
                               places=4,
                               msg="the parent delay must equal the inner delay")

    def test_3_nested_finish_does_not_release_the_parent_successor(self):
        pre, inner, post = 100.0, 700.0, 200.0
        parent = composite_node("c:n", 0, pre, inner, post, succs=("c:after",))
        after = gpu_node("c:after", 1, 10.0, preds=("c:n",))
        tpls = {"t": template("t", [parent, after])}
        _s, ev = run(tpls, [job("j", "t")])
        starts = {e.get("node_id"): e.get("start_ms") for e in by_type(ev, "node_start")}
        self.assertIn("c:after", starts, "the successor never ran")
        self.assertGreaterEqual(starts["c:after"], pre + inner + post - 1e-6,
                                "the successor started before the parent finished")

    def test_4_nested_finish_does_not_release_another_job_device(self):
        """The composite queues; the long GPU job's window is untouched."""
        pre, inner, post = 100.0, 700.0, 200.0
        tpls = {"a": template("a", [composite_node("c:n", 0, pre, inner, post)]),
                "b": template("b", [gpu_node("b:n", 0, 3000.0)])}
        s, ev = run(tpls, [job("jb", "b", 0.0), job("ja", "a", 10.0)])
        self.assertEqual(s["completed_jobs"], 2)
        self.assertEqual(s["failed_jobs"], 0)
        b_fin = by_type(ev, "node_finish", "b:n")[0]
        self.assertAlmostEqual(b_fin["finish_ms"], 3000.0, places=4,
                               msg="the unrelated GPU job's window was disturbed")
        start = by_type(ev, "nested_gpu_start", "c:n")[0]
        self.assertGreaterEqual(start["nested_start_ms"], 3000.0 - 1e-9)

    def test_5_two_composites_serialise_and_never_overlap(self):
        """Both become ready at 100; the second must queue behind the first."""
        tpls = {"a": template("a", [composite_node("ca:n", 0, 100.0, 700.0, 100.0)]),
                "b": template("b", [composite_node("cb:n", 1, 100.0, 800.0, 100.0)])}
        _s, ev = run(tpls, [job("ja", "a"), job("jb", "b")])
        segs = sorted((e["nested_start_ms"], e["nested_finish_ms"])
                      for e in by_type(ev, "nested_gpu_start"))
        self.assertEqual(len(segs), 2, "both inner segments must start")
        self.assertGreaterEqual(segs[1][0], segs[0][1] - 1e-9,
                                "the two composite segments overlap")

    def test_6_gpu_stays_usable_during_the_preparation_phase(self):
        """A short normal GPU job must fit entirely inside the composite's R_pre."""
        pre, inner, post = 5000.0, 700.0, 200.0
        tpls = {"a": template("a", [composite_node("c:n", 0, pre, inner, post)]),
                "b": template("b", [gpu_node("short:n", 0, 500.0)])}
        _s, ev = run(tpls, [job("ja", "a", 0.0), job("jb", "b", 100.0)])
        admit = by_type(ev, "nested_gpu_admit", "c:n")[0]
        self.assertAlmostEqual(admit["nested_ready_ms"], 5000.0, places=4)
        short = by_type(ev, "node_finish", "short:n")
        self.assertEqual(len(short), 1, "the short GPU job never ran")
        self.assertLess(short[0]["finish_ms"], admit["nested_ready_ms"],
                        "the preparation phase blocked the device, so the short job "
                        "could not finish before nested_ready")


if __name__ == "__main__":
    unittest.main()
