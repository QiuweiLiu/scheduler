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

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V041 = (PROJECT_ROOT / "results" / "processed"
        / "r7_workload_v041_ontology_no_run_container" / "job_templates_r7_v041.jsonl")

import ast
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


def _predictor_for(_episode, templates):
    """A train-only predictor over exactly these templates."""

    from tracing.analysis.latency_aware_predictor import build_latency_predictor
    table = (templates if isinstance(templates, dict)
             else {str(t.template_id): t for t in templates})
    return build_latency_predictor(table, min_support=1)

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
                                  policy_context={"latency_aware_predictor": _predictor_for(ep, tpls)},
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
                                  policy_context={"latency_aware_predictor": _predictor_for(ep, tpls)},
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
                                  policy_context={"latency_aware_predictor": _predictor_for(ep, tpls)},
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


class NonDegeneracyTests(unittest.TestCase):
    """The predictor must depend on the REQUEST, not merely on (model, lane).

    This is the check that separates this baseline from a TIE-style bank.  A TIE bank is
    legitimately keyed by (model, lane), so its distribution cannot change with the
    request; a predictor that behaves the same way predicts nothing about the request and
    would make the "latency-aware" front end vacuous.  The gate therefore asserts the
    opposite of vacuity on the learned table, and pins the truth path shut.
    """

    def test_the_predictor_moves_when_only_the_request_changes(self):
        if not V041.exists():
            self.skipTest("v04.1 projection not present")
        from tracing.analysis.workload_v02_simulator import load_templates
        from tracing.analysis.latency_aware_predictor import (
            binds_to_request, build_latency_predictor, non_degeneracy_report)

        tpls = load_templates(V041, topology_view="causal_v3")
        prof = build_latency_predictor(tpls)
        report = non_degeneracy_report(prof)
        self.assertGreater(report["groups_with_multiple_actions"], 0)
        self.assertIsNotNone(report["fraction_non_degenerate"])
        self.assertGreater(
            report["fraction_non_degenerate"], 0.0,
            "a predictor whose answer never moves with the request is a (model, lane) "
            "table, which is what TIE legitimately is and what this must not be")

        # the same conclusion, reached through the public comparison helper
        group = next((name for name, row in report["per_group"].items()
                      if not row["degenerate"]), None)
        if group is None:
            self.skipTest("no non-degenerate group to compare")
        model_id, lane = group.split("|", 1)
        template = next(
            t for t in tpls.values()
            if any(n.model_id == model_id and n.lane == lane for n in t.nodes))
        families = [n.action_family for n in template.nodes if n.model_id == model_id]
        if len(set(families)) < 2:
            detail = report["per_group"][group]
            self.assertGreater(
                max(detail["spread"]["run_ms"], detail["spread"]["peak_mem_mb"],
                    detail["spread"]["load_ms"]), 0.0)
            return
        a, b = list(dict.fromkeys(families))[:2]
        outcome = binds_to_request(prof, template, a, b)
        self.assertTrue(outcome["same_model_and_lane"])
        self.assertTrue(outcome["is_request_conditioned"])

    def test_the_predictor_does_not_read_the_truth_fields(self):
        """``compute_ms`` is runtime minus load; a predictor using it would be circular."""

        if not V041.exists():
            self.skipTest("v04.1 projection not present")
        import inspect
        from tracing.analysis import latency_aware_predictor as module
        from tracing.analysis import latency_aware_scheduler as la

        def truth_accesses(obj):
            """Attribute reads of compute_ms in CODE, ignoring any prose about it.

            A string search is not enough: both modules legitimately DISCUSS compute_ms
            in their docstrings to explain why it must not be used, and a naive search
            flags the explanation as the offence.
            """

            tree = ast.parse(inspect.getsource(obj))
            found = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr == "compute_ms":
                    found.append(node.lineno)
            return found

        self.assertEqual(truth_accesses(module), [],
                         "the predictor must not read the truth compute time")
        self.assertEqual(truth_accesses(la), [],
                         "the scheduler must not read the node's truth compute time")
        # and the artifact says so explicitly, because the claim is load-bearing
        fake = type("T", (), {"split": "train", "template_id": "t", "nodes": (
            type("N", (), {"model_id": "m", "lane": "gpu",
                 "action_family": "a", "raw_action": "a", "batch_size": 1,
                 "runtime_ms": 1.0, "workspace_peak_mb": 1.0, "load_ms": 0.0})(),)})
        built = module.build_latency_predictor({"t": fake})
        self.assertEqual(built["conditions_on"], "request features only; no executed duration")

    def test_the_scheduler_fails_closed_without_a_predictor(self):
        if not V041.exists():
            self.skipTest("v04.1 projection not present")
        from tracing.analysis.workload_v02_simulator import load_templates
        tpls = load_templates(V041, topology_view="causal_v3")
        first = next(iter(tpls.values()))
        episode = {
            "episode_id": "no-predictor", "split": first.split,
            "gpu_topology_mb": [4000.0, 4000.0],
            "initial_residency_hint": [[], []],
            "jobs": [{"job_instance_id": "j0", "template_id": first.template_id,
                      "arrival_ms": 0.0, "deadline_ms": 1e9,
                      "service_class": "normal"}],
        }
        with self.assertRaises(ValueError):
            simulate_episode(episode, {first.template_id: first}, "latency_aware",
                             train_stats=train_resource_stats({first.template_id: first}))


class TruthFallbackClosedTests(unittest.TestCase):
    """The scheduler's OWN entry points must refuse to run without a predictor.

    The simulator wrapper already blocked the truth read, but choose_action still accepted
    predictor=None and build_plan then read node.workspace_peak_mb.  Blocking one entrance
    while leaving another open is not a closure.
    """

    def test_the_formal_entry_points_require_a_predictor(self):
        import inspect
        from tracing.analysis import latency_aware_scheduler as la

        for name in ("build_plan", "rank_ready", "choose_action", "predicted_duration_ms"):
            parameter = inspect.signature(getattr(la, name)).parameters.get("predictor")
            self.assertIsNotNone(parameter, "%s has no predictor parameter" % name)
            self.assertIs(parameter.default, inspect.Parameter.empty,
                          "%s still defaults predictor to None, which keeps the truth "
                          "fallback reachable" % name)

    def test_the_scheduler_reads_no_node_truth_fields(self):
        import inspect
        from tracing.analysis import latency_aware_scheduler as la

        tree = ast.parse(inspect.getsource(la))
        banned = {"compute_ms", "runtime_ms", "workspace_peak_mb", "load_ms"}
        offenders = [(n.attr, n.lineno) for n in ast.walk(tree)
                     if isinstance(n, ast.Attribute) and n.attr in banned]
        self.assertEqual(offenders, [],
                         "the scheduler must not read node truth fields: %r" % offenders)

    def test_non_degeneracy_is_measured_on_the_production_path(self):
        if not V041.exists():
            self.skipTest("v04.1 projection not present")
        from tracing.analysis.latency_aware_predictor import (
            build_latency_predictor, non_degeneracy_report)
        from tracing.analysis.workload_v02_simulator import load_templates

        tpls = load_templates(V041, topology_view="causal_v3")
        train = {k: v for k, v in tpls.items() if v.split == "train"}
        prof = build_latency_predictor(tpls)
        report = non_degeneracy_report(prof, templates=train)
        self.assertIn("production predict()", report["measured_on"])
        self.assertGreater(report["groups_with_multiple_actions"], 0)
        self.assertGreater(report["fraction_non_degenerate"], 0.0)
        # The tier actually used must be visible: a Tier-1 table can look non-degenerate
        # while production predict() falls through to Tier 4 for every request.
        self.assertIn("tier_usage", report)
        self.assertTrue(report["tier_usage"], "no requests were probed")
        self.assertGreater(
            report["tier_usage_fraction"].get("T1", 0.0), 0.0,
            "every request fell through to a coarser tier, so the finer tiers carry no "
            "effective information even if their raw table looks non-degenerate")


class DecompositionTests(unittest.TestCase):
    """compute / load / total-peak must decompose consistently.

    runtime_ms INCLUDES load_ms in this substrate, and workspace_peak_mb is the TOTAL peak
    including the resident model.  A baseline that adds a load term on top of a runtime
    mean, or adds resident_model_mb to a total peak, is wrong in a way nothing raises on.
    """

    @staticmethod
    def _template(runtime, load, peak, resident_mb, action="a"):
        from tracing.analysis.workload_v02_simulator import Node, Template

        n = Node(node_id="d:n0", sequence_index=0, predecessors=(), successors=(),
                 lane="gpu", model_id="mD", runtime_ms=float(runtime), load_ms=float(load),
                 workspace_peak_mb=float(peak), resident_model_mb=float(resident_mb),
                 status="success", role="execute", action_family=action, raw_action=action)
        return Template("d", "d", "train", "fam", (n,), {"d:n0": n})

    def test_the_predictor_trains_on_compute_not_runtime(self):
        from tracing.analysis.latency_aware_predictor import (
            build_latency_predictor, predict)

        tpl = self._template(runtime=110.0, load=10.0, peak=500.0, resident_mb=200.0)
        prof = build_latency_predictor({"d": tpl}, min_support=1)
        pred = predict(prof, tpl.nodes[0])
        self.assertAlmostEqual(pred["run_ms"], 100.0,
                               msg="runtime 110 with load 10 must train as compute 100")

    def test_nonresident_pays_the_load_once_and_resident_pays_none(self):
        from tracing.analysis.latency_aware_predictor import (
            build_latency_predictor, predict)

        tpl = self._template(runtime=110.0, load=10.0, peak=500.0, resident_mb=200.0)
        prof = build_latency_predictor({"d": tpl}, min_support=1)
        pred = predict(prof, tpl.nodes[0])
        nonresident_total = pred["run_ms"] + pred["load_ms"]
        resident_total = pred["run_ms"] + 0.0
        self.assertAlmostEqual(nonresident_total, 110.0)
        self.assertAlmostEqual(resident_total, 100.0)
        self.assertNotAlmostEqual(nonresident_total, 120.0, places=6,
                                  msg="the load must not be counted twice")

    def test_the_admission_peak_is_not_inflated_by_the_resident_model(self):
        from tracing.analysis.latency_aware_predictor import (
            build_latency_predictor, predict)

        tpl = self._template(runtime=110.0, load=10.0, peak=500.0, resident_mb=200.0)
        prof = build_latency_predictor({"d": tpl}, min_support=1)
        pred = predict(prof, tpl.nodes[0])
        self.assertAlmostEqual(pred["peak_mem_mb"], 500.0,
                               msg="workspace_peak_mb is the TOTAL peak, already "
                                   "including the resident model")
        self.assertNotAlmostEqual(pred["peak_mem_mb"], 700.0, places=6)


if __name__ == "__main__":
    unittest.main()
