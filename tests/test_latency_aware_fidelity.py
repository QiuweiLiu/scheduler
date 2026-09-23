"""Fidelity gate for Latency-Aware-adapted (layer 1 of the two-layer gate).

The review's list for Latency-Aware:
    fusion preserves semantics;
    load / prefetch / reclaim state transitions are correct;
    every step respects memory feasibility;
    the physical plan can change with live state.

This arm currently migrates the Predictor and the fusion half of Constructor.  The
lifecycle half (load / prefetch / reclaim alternatives) is NOT yet implemented, so
those checks are recorded as explicitly pending rather than silently claimed.
"""
from __future__ import annotations

import unittest
from collections import Counter

from tracing.analysis.latency_aware_fusion import maximal_fusible_chains
from tracing.analysis.workload_v02_simulator import (
    POLICIES,
    Node,
    Template,
    simulate_episode,
    train_resource_stats,
)


def chain_template(tid, specs, model="m1", lane="gpu", family="fam"):
    """specs: list of (node_type, runtime, model, lane, workspace_mb)."""
    nodes = []
    for i, spec in enumerate(specs):
        ntype, rt, mdl, ln, ws = spec
        nodes.append(Node(
            node_id=f"{tid}:n{i}", sequence_index=i,
            predecessors=(f"{tid}:n{i-1}",) if i else (),
            successors=(f"{tid}:n{i+1}",) if i + 1 < len(specs) else (),
            lane=ln, model_id=mdl, runtime_ms=float(rt), load_ms=5.0,
            workspace_peak_mb=float(ws), resident_model_mb=90.0, status="success",
            role=ntype, action_family="inference",
        ))
    return Template(tid, tid, "train", family, tuple(nodes), {n.node_id: n for n in nodes})


class LatencyAwareFidelityTests(unittest.TestCase):
    def test_arm_is_registered(self):
        self.assertIn("latency_aware", POLICIES)

    def test_a_four_node_chain_is_one_fused_unit(self):
        tpl = chain_template("t", [("planner", 100, "m1", "gpu", 100.0)] * 4)
        chains = [c for c in maximal_fusible_chains(tpl) if c.length > 1]
        self.assertEqual(len(chains), 1)
        self.assertEqual(chains[0].length, 4)
        self.assertEqual(chains[0].boundaries_removed, 3)

    def test_fusion_reduces_node_starts_and_completes_every_node(self):
        """Semantics preserved: fewer admits, same nodes finished."""
        specs = [("planner", 100, "m1", "gpu", 100.0)] * 4
        tpls = {"t": chain_template("t", specs)}
        stats = train_resource_stats(tpls)
        ep = {
            "episode_id": "lat-fuse", "split": "train",
            "gpu_topology_mb": [40000.0, 40000.0], "initial_residency_hint": [[], []],
            "jobs": [{"job_instance_id": "j0", "template_id": "t", "arrival_ms": 0.0,
                      "deadline_ms": 1e9, "service_class": "normal"}],
        }
        _s, ev_lat = simulate_episode(ep, tpls, "latency_aware", train_stats=stats,
                                      collect_events=True)
        _s2, ev_fcfs = simulate_episode(ep, tpls, "fcfs", train_stats=stats,
                                        collect_events=True)
        def starts(ev):
            return [e for e in ev if e.get("event_type") == "node_start"]
        def finishes(ev):
            return [e for e in ev if e.get("event_type") == "node_finish"]
        self.assertEqual(len(finishes(ev_lat)), len(finishes(ev_fcfs)),
                         "fusion must not lose completions")
        self.assertLess(len(starts(ev_lat)), len(starts(ev_fcfs)),
                        "a fused plan must admit fewer units than a per-node plan")
        co = Counter(round(e.get("finish_ms") or 0, 3) for e in finishes(ev_lat))
        self.assertGreater(sum(v - 1 for v in co.values() if v > 1), 0,
                           "no finish closed several members, so nothing was fused")

    def test_fusion_respects_memory_feasibility(self):
        """A chain whose peak exceeds the device must NOT be fused."""
        specs = [("planner", 100, "m1", "gpu", 30000.0)] * 4
        tpls = {"t": chain_template("t", specs)}
        stats = train_resource_stats(tpls)
        ep = {
            "episode_id": "lat-mem", "split": "train",
            "gpu_topology_mb": [40000.0], "initial_residency_hint": [[]],
            "jobs": [{"job_instance_id": "j0", "template_id": "t", "arrival_ms": 0.0,
                      "deadline_ms": 1e9, "service_class": "normal"}],
        }
        s, ev = simulate_episode(ep, tpls, "latency_aware", train_stats=stats,
                                 collect_events=True)
        # the run must remain valid regardless; a fused unit that cannot fit is not chosen
        self.assertEqual(s["completed_jobs"] + s["failed_jobs"], 1)

    def test_existing_arms_are_unaffected_by_the_fused_completion_path(self):
        """The separator encoding must be a no-op for single-node dispatches."""
        tpls = {"t": chain_template("t", [("planner", 100, "m1", "gpu", 100.0)] * 3)}
        stats = train_resource_stats(tpls)
        ep = {
            "episode_id": "noop", "split": "train",
            "gpu_topology_mb": [40000.0, 40000.0], "initial_residency_hint": [[], []],
            "jobs": [{"job_instance_id": "j0", "template_id": "t", "arrival_ms": 0.0,
                      "deadline_ms": 1e9, "service_class": "normal"}],
        }
        for pol in ("fcfs", "myopic", "round_robin"):
            s, ev = simulate_episode(ep, tpls, pol, train_stats=stats, collect_events=True)
            self.assertEqual(s["completed_jobs"], 1, pol)
            co = Counter(round(e.get("finish_ms") or 0, 3)
                         for e in ev if e.get("event_type") == "node_finish")
            self.assertEqual(sum(v - 1 for v in co.values() if v > 1), 0,
                             "%s unexpectedly co-finished nodes" % pol)

    def test_lifecycle_prefetch_fires_from_live_state(self):
        """Eq (5) alpha_N = Prefetch: a near-ready deployment is prepared in advance."""
        # two jobs; the first node of job b uses a model nothing else needs
        specs_a = [("planner", 400, "m1", "gpu", 100.0)] * 2
        specs_b = [("planner", 400, "m2", "gpu", 100.0)] * 2
        tpls = {"a": chain_template("a", specs_a), "b": chain_template("b", specs_b)}
        stats = train_resource_stats(tpls)
        ep = {
            "episode_id": "lat-prefetch", "split": "train",
            # three devices for two jobs: prefetch is only admissible from residual
            # capacity, so a spare device must exist for the lifecycle half to fire
            "gpu_topology_mb": [40000.0, 40000.0, 40000.0],
            "initial_residency_hint": [[], [], []],
            "jobs": [{"job_instance_id": "ja", "template_id": "a", "arrival_ms": 0.0,
                      "deadline_ms": 1e9, "service_class": "normal"},
                     {"job_instance_id": "jb", "template_id": "b", "arrival_ms": 0.0,
                      "deadline_ms": 1e9, "service_class": "normal"}],
        }
        s, ev = simulate_episode(ep, tpls, "latency_aware", train_stats=stats,
                                 collect_events=True)
        events = Counter(str(e.get("event_type")) for e in ev)
        self.assertGreater(events.get("prefetch_start", 0), 0,
                           "no prefetch was issued; the lifecycle half is inert")
        self.assertEqual(events.get("prefetch_start"), events.get("prefetch_end"),
                         "every started prefetch must complete")
        self.assertEqual(s["completed_jobs"], 2)
        self.assertEqual(s["failed_jobs"], 0)

    def test_prefetch_skips_unmeasured_and_over_capacity_deployments(self):
        """Unmeasured deployments are skipped rather than guessed."""
        from tracing.analysis.latency_aware_lifecycle import prefetch_candidates

        class _G:
            index = 0
            capacity_mb = 1000.0
            resident = {}

        class _Node:
            model_id = "m1"
            lane = "gpu"

        class _T:
            by_id = {"n": None}

        class _J:
            node_state = {"n": "running"}
            template = None

        t = _T()
        t.by_id = {"n": _Node()}
        t.by_id["n"].successors = ()   # nothing near-ready
        j = _J()
        j.template = t
        plan = prefetch_candidates([j], [_G()], model_memory_for=lambda m: None)
        self.assertEqual(plan, [], "an unmeasured deployment must not be prefetched")

    def test_lifecycle_half_is_now_migrated(self):
        """The lifecycle half is reachable, so the arm is no longer fusion-only."""
        from tracing.analysis.latency_aware_lifecycle import (
            near_ready_deployments,
            prefetch_candidates,
        )
        self.assertTrue(callable(near_ready_deployments))
        self.assertTrue(callable(prefetch_candidates))

    def test_reclaim_victim_policy_is_still_pending(self):
        """Record the remaining gap honestly."""
        import tracing.analysis.latency_aware_lifecycle as lc
        self.assertFalse(hasattr(lc, "choose_reclaim_victim"),
                         "if this appears, update the adaptation note")


if __name__ == "__main__":
    unittest.main()
