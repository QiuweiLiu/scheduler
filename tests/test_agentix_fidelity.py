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
    AGENTIX_DEVIATION,
    AGENTIX_SCHEMA,
    calibrate_queue_edges,
    DEFAULT_ANTI_STARVATION_BETA,
    DEFAULT_QUEUE_EDGES_MS,
    critical_path_service_ms,
    completed_service_ms,
    discrete_priority_index,
    observed_gpu_service_of,
    intrinsic_runtime_of,
    is_starving,
    program_priority_ms,
    queue_index,
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


class ServiceAccountingTests(unittest.TestCase):
    """PLAS service = completed GPU nodes' observed model-executor service, nothing else."""

    @staticmethod
    def _job(nodes, completed, observed=None):
        tpl = _template("t", nodes)

        class _J:
            pass

        j = _J()
        j.template = tpl
        j.completed = set(completed)
        j.observed_intrinsic_ms = dict(observed or {})
        return j

    def test_non_gpu_nodes_are_excluded(self):
        g1 = _node("t:g", 0, [], 100.0)
        c1 = _node("t:c", 1, [], 50.0, lane="cpu")
        job = self._job([g1, c1], {"t:g", "t:c"})
        self.assertAlmostEqual(completed_service_ms(job, observed_gpu_service_of(job)), 100.0)

    def test_model_load_is_excluded(self):
        # runtime 100 with load 30 -> service 70
        g1 = Node(node_id="t:g", sequence_index=0, predecessors=(), successors=(),
                  lane="gpu", model_id="m1", runtime_ms=100.0, load_ms=30.0,
                  workspace_peak_mb=10.0, resident_model_mb=10.0, status="success",
                  role="execute", action_family="inference")
        job = self._job([g1], {"t:g"})
        self.assertAlmostEqual(completed_service_ms(job, observed_gpu_service_of(job)), 70.0)

    def test_observation_store_is_used(self):
        g1 = _node("t:g", 0, [], 100.0)
        job = self._job([g1], {"t:g"}, observed={"t:g": 120.0})
        # observed runtime 120 (load 0) -> service 120
        self.assertAlmostEqual(completed_service_ms(job, observed_gpu_service_of(job)), 120.0)

    def test_deviation_discloses_the_formal_arm_boundary(self):
        for token in ("preemptive", "MLFQ", "quantum", "anti-starvation", "prediction"):
            self.assertIn(token, AGENTIX_DEVIATION)


class EdgeCalibrationTests(unittest.TestCase):
    """Train-only quantile calibration of the K-queue boundaries (paper publishes none)."""

    @staticmethod
    def _train(tid, runtimes, split="train"):
        tpl = _chain(tid, runtimes)
        return Template(tid, tid, split, "fam", tpl.nodes, tpl.by_id)

    def test_edges_are_monotone_and_capped_at_k(self):
        tpls = {f"t{i}": self._train(f"t{i}", [100.0, 200.0, 300.0, 400.0, 500.0])
                for i in range(5)}
        edges = calibrate_queue_edges(tpls, queues=4)
        self.assertEqual(edges[0], 0.0)
        self.assertEqual(edges, tuple(sorted(edges)))
        self.assertLessEqual(len(edges), 4)

    def test_validation_templates_are_excluded(self):
        train = {"t": self._train("t", [1000.0, 1000.0, 1000.0, 1000.0])}
        val = {"v": self._train("v", [1.0, 1.0, 1.0, 1.0], split="validation")}
        only = calibrate_queue_edges(train, queues=4)
        both = calibrate_queue_edges({**train, **val}, queues=4)
        self.assertEqual(only, both)

    def test_duplicate_boundaries_are_merged_never_jittered(self):
        # every program has ONE gpu call, so every admission sees attained service 0
        tpls = {f"t{i}": self._train(f"t{i}", [100.0]) for i in range(6)}
        edges = calibrate_queue_edges(tpls, queues=4)
        self.assertEqual(edges, (0.0,), "identical quantiles must collapse to one bin")

    def test_more_data_produces_more_bins(self):
        # varying program lengths give a spread of attained-service values
        tpls = {f"t{i}": self._train(f"t{i}", [1000.0] * (i + 2)) for i in range(8)}
        edges = calibrate_queue_edges(tpls, queues=4)
        self.assertGreater(len(edges), 1)


class QueueDiscretizationTests(unittest.TestCase):
    """NON-PAPER sensitivity variant: discretised, non-preemptive queueing."""

    def test_queue_index_is_monotone_in_service(self):
        edges = DEFAULT_QUEUE_EDGES_MS
        idx = [queue_index(v, edges) for v in (0.0, 45000.0, 150000.0, 999999.0)]
        self.assertEqual(idx, sorted(idx))
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[-1], len(edges) - 1)

    def test_queue_index_boundaries(self):
        edges = (0.0, 10.0, 20.0)
        self.assertEqual(queue_index(9.999, edges), 0)
        self.assertEqual(queue_index(10.0, edges), 1)
        self.assertEqual(queue_index(25.0, edges), 2)

    def test_anti_starvation_ratio(self):
        self.assertTrue(is_starving(100.0, 50.0, beta=1.0))    # 2.0 >= 1.0
        self.assertFalse(is_starving(10.0, 50.0, beta=1.0))    # 0.2 < 1.0
        with self.assertRaises(ValueError):
            is_starving(1.0, 1.0, beta=0.0)

    def test_starving_program_is_promoted_to_the_top_queue(self):
        edges = (0.0, 10.0, 20.0)
        # high attained service would normally bin to the last queue
        self.assertEqual(queue_index(100.0, edges), 2)
        # but a starving program is promoted to queue 0
        self.assertEqual(discrete_priority_index(100.0, wait_ms=1000.0, edges=edges, beta=1.0), 0)
        # a non-starving program keeps its bin
        self.assertEqual(discrete_priority_index(100.0, wait_ms=1.0, edges=edges, beta=1.0), 2)

    def test_demotion_happens_across_calls(self):
        """A program's LATER calls land in a lower queue as its attained service grows."""

        edges = DEFAULT_QUEUE_EDGES_MS
        early = queue_index(0.0, edges)          # first call: top queue
        late = queue_index(300000.0, edges)      # after much service: bottom queue
        self.assertLess(early, late)

    def test_queueing_does_not_read_the_future(self):
        tpl = _chain("t", [10.0, 20.0, 40.0])
        job = _Job(tpl, {"t:n0"})
        before = queue_index(completed_service_ms(job, intrinsic_runtime_of(tpl)))
        tpl.by_id["t:n1"] = _node("t:n1", 1, ["t:n0"], 999999.0)
        tpl.by_id["t:n2"] = _node("t:n2", 2, ["t:n1"], 999999.0)
        after = queue_index(completed_service_ms(job, intrinsic_runtime_of(tpl)))
        self.assertEqual(before, after)


class DiscreteModeEndToEndTests(unittest.TestCase):
    def test_discrete_mode_runs_and_completes(self):
        tpls = {f"t{i}": _chain(f"t{i}", [100.0, 120.0, 140.0]) for i in range(3)}
        stats = train_resource_stats(tpls)
        ep = {"episode_id": "agentix-discrete", "split": "train",
              "gpu_topology_mb": [4000.0, 4000.0], "initial_residency_hint": [[], []],
              "jobs": [{"job_instance_id": f"j{i}", "template_id": f"t{i}", "arrival_ms": 0.0,
                        "deadline_ms": 1e9, "service_class": "normal"} for i in range(3)]}
        summary, events = simulate_episode(ep, tpls, "agentix", train_stats=stats,
                                           policy_context={"agentix_mode": "discrete"},
                                           collect_events=True)
        self.assertEqual(summary["completed_jobs"], 3)
        self.assertEqual(summary["failed_jobs"], 0)
        self.assertEqual(len([e for e in events if e.get("event_type") == "node_finish"]), 9)

    def test_unknown_mode_still_fails_closed(self):
        tpls = {"t0": _chain("t0", [100.0, 120.0, 140.0])}
        stats = train_resource_stats(tpls)
        ep = {"episode_id": "x", "split": "train", "gpu_topology_mb": [4000.0],
              "initial_residency_hint": [[]],
              "jobs": [{"job_instance_id": "j", "template_id": "t0", "arrival_ms": 0.0,
                        "deadline_ms": 1e9, "service_class": "normal"}]}
        with self.assertRaises(ValueError):
            simulate_episode(ep, tpls, "agentix", train_stats=stats,
                             policy_context={"agentix_mode": "oracle"})


if __name__ == "__main__":
    unittest.main()
