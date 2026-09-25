"""Fidelity gate for the independent Latency-Aware scheduler (Algorithm 1).

The review's list for Latency-Aware: fusion preserves semantics; lifecycle state
transitions are correct; every step respects memory feasibility; the physical plan
can change with live state.

This file covers the scheduler module itself: Eq (4) fused duration, Eq (7) start
time, Eq (11) admission memory, Eq (12) key ordering, and that it is genuinely
independent of the shared min(pool, key=...) idiom.
"""
from __future__ import annotations

import unittest
from collections import Counter

from tracing.analysis.latency_aware_predictor import (
    build_latency_predictor,
    predict,
)
from tracing.analysis.latency_aware_scheduler import (
    FUSED_SEPARATOR,
    PlanStep,
    build_plan,
    choose_action,
    memory_feasible,
    predicted_duration_ms,
    rank_ready,
)
from tracing.analysis.workload_v02_simulator import (
    POLICIES,
    Node,
    Template,
    simulate_episode,
    train_resource_stats,
)


def chain(tid, specs, family="fam"):
    nodes = []
    for i, (rt, mdl, ln, ws) in enumerate(specs):
        nodes.append(Node(
            node_id=f"{tid}:n{i}", sequence_index=i,
            predecessors=(f"{tid}:n{i-1}",) if i else (),
            successors=(f"{tid}:n{i+1}",) if i + 1 < len(specs) else (),
            lane=ln, model_id=mdl, runtime_ms=float(rt), load_ms=5.0,
            workspace_peak_mb=float(ws), resident_model_mb=90.0, status="success",
            role="execute", action_family="inference",
        ))
    return Template(tid, tid, "train", family, tuple(nodes), {n.node_id: n for n in nodes})




def _tiny_predictor(template):
    """A one-template, train-only predictor whose table is exactly that template."""

    return build_latency_predictor({str(template.template_id): template}, min_support=1)

def _predictor_for(_episode, templates):
    """A train-only predictor over exactly these templates."""

    from tracing.analysis.latency_aware_predictor import build_latency_predictor
    table = (templates if isinstance(templates, dict)
             else {str(t.template_id): t for t in templates})
    return build_latency_predictor(table, min_support=1)

class _Gpu:
    def __init__(self, index=0, capacity=40000.0, resident=None, busy_until=0.0):
        self.index = index
        self.capacity_mb = capacity
        self.resident = dict(resident or {})
        self.busy_until = busy_until
        self.active_node = None


class Eq4FusedDurationTests(unittest.TestCase):
    def test_fused_duration_comes_from_the_predictor_not_from_truth(self):
        """A fused unit sums the PREDICTED member run times and pays one load.

        The old form summed ``compute_ms``, which is ``runtime_ms - load_ms`` -- the
        members' ACTUAL intrinsic compute time.  Reading truth is what made the whole arm
        circular, so the sum must now follow the train-only predictor.
        """

        tpl = chain("t", [(100, "m1", "gpu", 5.0), (200, "m1", "gpu", 5.0),
                          (300, "m1", "gpu", 5.0)])
        row = {"runtime_p50_ms": 100.0, "load_p50_ms": 7.0}

        class _J:
            template = tpl

        predictor = _tiny_predictor(tpl)
        dur, load = predicted_duration_ms(_J(), "t:n0", ("t:n0", "t:n1", "t:n2"), row,
                                          resident=False, predictor=predictor)
        self.assertAlmostEqual(load, 7.0, msg="a fused unit pays the load once")
        predicted_sum = sum(
            predict(predictor, tpl.by_id[m])["run_ms"] for m in ("t:n0", "t:n1", "t:n2"))
        self.assertAlmostEqual(dur, 7.0 + predicted_sum)
        # and it is NOT the truth sum: the predictor returns the train mean, which for a
        # single train node equals runtime_ms, so the distinguishing check is that the
        # truth-only path is never consulted
        truth_sum = sum(tpl.by_id[m].compute_ms for m in ("t:n0", "t:n1", "t:n2"))
        self.assertNotAlmostEqual(dur, 7.0 + truth_sum, places=6)

    def test_resident_unit_pays_no_load(self):
        tpl = chain("t", [(100, "m1", "gpu", 10.0)])

        class _J:
            template = tpl

        dur, load = predicted_duration_ms(_J(), "t:n0", (), {"runtime_p50_ms": 100.0,
                                                            "load_p50_ms": 7.0},
                                          resident=True)
        self.assertAlmostEqual(load, 0.0)
        self.assertAlmostEqual(dur, 100.0)


class Eq7StartTimeTests(unittest.TestCase):
    def test_start_is_the_max_of_release_and_device_availability(self):
        tpl = chain("t", [(100, "m1", "gpu", 10.0)])
        row = {"runtime_p50_ms": 100.0, "load_p50_ms": 0.0}
        gpu = _Gpu(busy_until=500.0)
        cand = ((0.0, 300.0, 0, "t:n0"), 0, "t:n0", "m1", gpu, row, True)

        class _J:
            template = tpl
            node_state = {"t:n0": "ready"}

        step = build_plan(cand, [_J()], 400.0)
        self.assertAlmostEqual(step.predicted_start_ms, 500.0,
                               msg="the busy device must dominate the release time")
        self.assertAlmostEqual(step.predicted_completion_ms, 600.0)

    def test_release_time_dominates_when_the_device_is_idle(self):
        tpl = chain("t", [(100, "m1", "gpu", 10.0)])
        row = {"runtime_p50_ms": 100.0, "load_p50_ms": 0.0}
        gpu = _Gpu(busy_until=0.0)
        cand = ((0.0, 900.0, 0, "t:n0"), 0, "t:n0", "m1", gpu, row, True)

        class _J:
            template = tpl
            node_state = {"t:n0": "ready"}

        step = build_plan(cand, [_J()], 100.0)
        self.assertAlmostEqual(step.predicted_start_ms, 900.0)
        self.assertAlmostEqual(step.predicted_completion_ms, 1000.0)


class Eq11MemoryTests(unittest.TestCase):
    def test_admission_memory_blocks_an_otherwise_valid_plan(self):
        tpl = chain("t", [(100, "m1", "gpu", 10.0)])
        row = {"runtime_p50_ms": 100.0, "load_p50_ms": 0.0}
        gpu = _Gpu(capacity=1000.0, resident={"other": 950.0})
        cand = ((0.0, 0.0, 0, "t:n0"), 0, "t:n0", "m1", gpu, row, True)

        class _J:
            template = tpl
            node_state = {"t:n0": "ready"}

        step = build_plan(cand, [_J()], 0.0)
        step.memory_mb = 100.0
        self.assertFalse(memory_feasible(step, gpu))

    def test_an_already_resident_model_does_not_double_count(self):
        tpl = chain("t", [(100, "m1", "gpu", 10.0)])
        row = {"runtime_p50_ms": 100.0, "load_p50_ms": 0.0}
        gpu = _Gpu(capacity=1000.0, resident={"m1": 900.0})
        cand = ((0.0, 0.0, 0, "t:n0"), 0, "t:n0", "m1", gpu, row, True)

        class _J:
            template = tpl
            node_state = {"t:n0": "ready"}

        step = build_plan(cand, [_J()], 0.0)
        step.memory_mb = 50.0
        self.assertTrue(memory_feasible(step, gpu))


class Eq12KeyTests(unittest.TestCase):
    def _step(self, priority, completion, boundaries, ready, gpu_index=0):
        gpu = _Gpu(index=gpu_index)
        cand = ((priority, ready, 0, "t:n0"), 0, "t:n0", "m1", gpu,
                {"runtime_p50_ms": 0.0, "load_p50_ms": 0.0}, True)
        s = PlanStep(candidate=cand)
        s.predicted_completion_ms = completion
        s.boundaries_removed = boundaries
        return s

    def test_hard_priority_is_the_first_key(self):
        high = self._step(0.0, 99999.0, 0, 0.0)
        low = self._step(1.0, 0.0, 0, 0.0)
        self.assertLess(high.key, low.key, "service class must dominate completion")

    def test_completion_breaks_ties_before_the_fusion_bonus(self):
        sooner = self._step(0.0, 100.0, 0, 0.0)
        later_fused = self._step(0.0, 200.0, 5, 0.0)
        self.assertLess(sooner.key, later_fused.key,
                        "C_F precedes the boundary-removal term")

    def test_the_key_carries_no_invented_boundary_term(self):
        """Eq (12) has no ``-boundaries_removed`` term; it was an invented tie-break.

        The old test asserted that a fused plan wins at EQUAL completion because of the
        admission count.  That is not the paper's criterion, and it also let a candidate
        win on a count rather than on the completion time fusion actually changes.
        """

        plain = self._step(0.0, 100.0, 0, 0.0)
        fused = self._step(0.0, 100.0, 3, 0.0)
        self.assertEqual(plain.key, fused.key,
                         "at equal completion and equal ready time the key must not "
                         "distinguish two plans by an admission count")

    def test_fusion_wins_through_the_completion_it_changes(self):
        """A fused plan competes on C_F, which is the honest mechanism."""

        plain = self._step(0.0, 300.0, 0, 0.0)
        fused = self._step(0.0, 120.0, 3, 0.0)
        self.assertLess(fused.key, plain.key,
                        "the shorter predicted completion must win")


class IndependenceTests(unittest.TestCase):
    def test_scheduler_is_a_separate_module_with_its_own_action_vocabulary(self):
        import tracing.analysis.latency_aware_scheduler as la
        self.assertTrue(callable(la.choose_action))
        self.assertTrue(hasattr(la, "PlanStep"))
        self.assertEqual(la.FUSED_SEPARATOR, "\x1f")

    def test_returns_none_when_nothing_is_memory_feasible(self):
        tpl = chain("t", [(100, "m1", "gpu", 9000.0)])
        row = {"runtime_p50_ms": 100.0, "load_p50_ms": 0.0}
        gpu = _Gpu(capacity=1000.0)
        cand = ((0.0, 0.0, 0, "t:n0"), 0, "t:n0", "m1", gpu, row, True)

        class _J:
            template = tpl
            node_state = {"t:n0": "ready"}

        act = choose_action([cand], [_J()], 0.0, chains_for=lambda t: {})
        self.assertIsNone(act, "an infeasible plan must not be committed")

    def test_end_to_end_run_still_completes_every_job(self):
        tpls = {"t": chain("t", [(100, "m1", "gpu", 10.0)] * 3)}
        stats = train_resource_stats(tpls)
        ep = {"episode_id": "la-sched", "split": "train",
              "gpu_topology_mb": [40000.0, 40000.0], "initial_residency_hint": [[], []],
              "jobs": [{"job_instance_id": "j0", "template_id": "t", "arrival_ms": 0.0,
                        "deadline_ms": 1e9, "service_class": "normal"}]}
        s, ev = simulate_episode(ep, tpls, "latency_aware", train_stats=stats,
                                  policy_context={"latency_aware_predictor": _predictor_for(ep, tpls)},
                                 collect_events=True)
        self.assertEqual(s["completed_jobs"], 1)
        self.assertEqual(s["failed_jobs"], 0)
        self.assertEqual(len([e for e in ev if e.get("event_type") == "node_finish"]), 3)


if __name__ == "__main__":
    unittest.main()
