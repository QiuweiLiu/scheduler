"""Fidelity gate for Agentix-adapted (NSDI 2026; formerly Autellix).

Agentix is the NON-CLAIRVOYANT reference: its priority is the service a program has
ALREADY ATTAINED (PLAS = sum of completed calls; ATLAS = longest completed critical
path), with no knowledge of the program's execution graph.  The gate therefore checks
the two things that separate it from a clairvoyant SRPT arm:

  * only COMPLETED calls enter the priority -- changing an unexecuted node's runtime
    must not move the score;
  * PLAS and ATLAS coincide on a serial program and differ on a parallel one.

It also runs the arm end to end so the simulator branch is actually exercised.
"""
from __future__ import annotations

import unittest

from tracing.analysis.agentix_methods import (
    AGENTIX_SCHEMA,
    critical_path_service_ms,
    completed_service_ms,
    intrinsic_runtime_of,
    program_priority_ms,
)
from tracing.analysis.workload_v02_simulator import (
    POLICIES,
    Node,
    Template,
    simulate_episode,
    train_resource_stats,
)


class _Job:
    def __init__(self, template, completed):
        self.template = template
        self.completed = set(completed)


def _node(nid, seq, preds, runtime, model="m1", lane="gpu"):
    return Node(node_id=nid, sequence_index=seq, predecessors=tuple(preds), successors=(),
                lane=lane, model_id=model, runtime_ms=float(runtime), load_ms=0.0,
                workspace_peak_mb=10.0, resident_model_mb=10.0, status="success",
                role="execute", action_family="inference")


def _template(tid, nodes):
    return Template(tid, tid, "train", "fam", tuple(nodes), {n.node_id: n for n in nodes})


def _chain(tid, runtimes):
    nodes = []
    for i, rt in enumerate(runtimes):
        nodes.append(_node(f"{tid}:n{i}", i, [f"{tid}:n{i-1}"] if i else [], rt))
    return _template(tid, nodes)


class RegistrationTests(unittest.TestCase):
    def test_arm_is_registered(self):
        self.assertIn("agentix", POLICIES)

    def test_schema_is_pinned(self):
        self.assertEqual(AGENTIX_SCHEMA, "agentix-attained-service-v1")


class NonClairvoyanceTests(unittest.TestCase):
    """Only completed calls may move the priority."""

    def test_plas_sums_only_completed_calls(self):
        tpl = _chain("t", [10.0, 20.0, 40.0])
        job = _Job(tpl, {"t:n0", "t:n1"})
        self.assertAlmostEqual(completed_service_ms(job, intrinsic_runtime_of(tpl)), 30.0)

    def test_unexecuted_future_does_not_move_the_score(self):
        """Replacing an unexecuted node's runtime must not change PLAS or ATLAS."""

        tpl = _chain("t", [10.0, 20.0, 40.0])
        job = _Job(tpl, {"t:n0"})
        before = program_priority_ms(job, tpl, intrinsic_runtime_of(tpl))
        before_atlas = program_priority_ms(job, tpl, intrinsic_runtime_of(tpl), mode="atlas")
        # Node is frozen, so swap in a differently-timed future node (unaltered semantics)
        tpl.by_id["t:n1"] = _node("t:n1", 1, ["t:n0"], 999999.0)
        tpl.by_id["t:n2"] = _node("t:n2", 2, ["t:n1"], 999999.0)
        after = program_priority_ms(job, tpl, intrinsic_runtime_of(tpl))
        after_atlas = program_priority_ms(job, tpl, intrinsic_runtime_of(tpl), mode="atlas")
        self.assertAlmostEqual(before, after)
        self.assertAlmostEqual(before_atlas, after_atlas)

    def test_completed_call_does_move_the_score(self):
        tpl = _chain("t", [10.0, 20.0, 40.0])
        job = _Job(tpl, {"t:n0"})
        before = completed_service_ms(job, intrinsic_runtime_of(tpl))
        tpl.by_id["t:n0"] = _node("t:n0", 0, [], 7.0)  # the call HAS completed
        after = completed_service_ms(job, intrinsic_runtime_of(tpl))
        self.assertNotAlmostEqual(before, after)

    def test_unknown_mode_fails_closed(self):
        tpl = _chain("t", [1.0])
        with self.assertRaises(ValueError):
            program_priority_ms(_Job(tpl, set()), tpl, intrinsic_runtime_of(tpl), mode="oracle")


class PlasAtlasTests(unittest.TestCase):
    def test_they_coincide_on_a_serial_program(self):
        tpl = _chain("t", [10.0, 20.0, 40.0])
        job = _Job(tpl, {"t:n0", "t:n1"})
        self.assertAlmostEqual(
            completed_service_ms(job, intrinsic_runtime_of(tpl)),
            critical_path_service_ms(job, tpl, intrinsic_runtime_of(tpl)))

    def test_atlas_takes_the_critical_path_on_a_parallel_program(self):
        # a -> {b, c} -> d ; completed {a, b, c}, d still pending
        a = _node("t:a", 0, [], 10.0)
        b = _node("t:b", 1, ["t:a"], 30.0)
        c = _node("t:c", 2, ["t:a"], 5.0)
        d = _node("t:d", 3, ["t:b", "t:c"], 1.0)
        tpl = _template("t", [a, b, c, d])
        job = _Job(tpl, {"t:a", "t:b", "t:c"})
        plas = completed_service_ms(job, intrinsic_runtime_of(tpl))
        atlas = critical_path_service_ms(job, tpl, intrinsic_runtime_of(tpl))
        self.assertAlmostEqual(plas, 45.0)     # 10 + 30 + 5
        self.assertAlmostEqual(atlas, 40.0)    # a + b
        self.assertGreater(plas, atlas, "ATLAS must not sum parallel branches")


class EndToEndTests(unittest.TestCase):
    def _episode(self, n_jobs):
        return {
            "episode_id": "agentix-fidelity", "split": "train",
            "gpu_topology_mb": [4000.0, 4000.0], "initial_residency_hint": [[], []],
            "jobs": [{"job_instance_id": f"j{i}", "template_id": f"t{i}", "arrival_ms": 0.0,
                      "deadline_ms": 1e9, "service_class": "normal"} for i in range(n_jobs)],
        }

    def test_runs_and_completes_every_node(self):
        tpls = {f"t{i}": _chain(f"t{i}", [100.0, 120.0, 140.0]) for i in range(3)}
        stats = train_resource_stats(tpls)
        summary, events = simulate_episode(
            self._episode(3), tpls, "agentix", train_stats=stats, collect_events=True)
        self.assertEqual(summary["completed_jobs"], 3)
        self.assertEqual(summary["failed_jobs"], 0)
        self.assertEqual(len([e for e in events if e.get("event_type") == "node_finish"]),
                         9)

    def test_atlas_mode_runs(self):
        tpls = {"t0": _chain("t0", [100.0, 120.0, 140.0])}
        stats = train_resource_stats(tpls)
        summary, _ = simulate_episode(
            self._episode(1), tpls, "agentix", train_stats=stats,
            policy_context={"agentix_mode": "atlas"})
        self.assertEqual(summary["completed_jobs"], 1)
        self.assertEqual(summary["failed_jobs"], 0)


if __name__ == "__main__":
    unittest.main()
