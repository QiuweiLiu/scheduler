"""Fidelity tests for Eq (3): maximal fusible chains."""
from __future__ import annotations

import unittest

from tracing.analysis.latency_aware_fusion import (
    FusedChain,
    config_compatible,
    deployment_identity,
    is_fusible_edge,
    maximal_fusible_chains,
    summary,
)
from tracing.analysis.workload_v02_simulator import Node, Template


def node(nid, sequence, preds, succs, model="m1", lane="gpu", runtime=100.0, batch=1):
    return Node(
        node_id=nid, sequence_index=sequence, predecessors=tuple(preds), successors=tuple(succs),
        lane=lane, model_id=model, runtime_ms=runtime, load_ms=10.0, workspace_peak_mb=100.0,
        resident_model_mb=90.0, status="success", role="execute", action_family="inference",
        batch_size=batch,
    )


def template(nodes, tid="t0"):
    return Template(tid, tid, "train", "test", tuple(nodes), {n.node_id: n for n in nodes})


class Eq3LegalityTests(unittest.TestCase):
    def test_same_model_chain_is_fusible(self):
        a = node("a", 0, [], ["b"])
        b = node("b", 1, ["a"], [])
        self.assertTrue(is_fusible_edge(a, b))

    def test_different_model_is_not_fusible(self):
        a = node("a", 0, [], ["b"], model="m1")
        b = node("b", 1, ["a"], [], model="m2")
        self.assertFalse(is_fusible_edge(a, b))

    def test_branch_stops_the_chain(self):
        """succ(u) = {v} must hold: a fan-out keeps the boundary."""
        a = node("a", 0, [], ["b", "c"])
        b = node("b", 1, ["a"], [])
        self.assertFalse(is_fusible_edge(a, b), "fan-out must not fuse")

    def test_join_stops_the_chain(self):
        """pred(v) = {u} must hold: a fan-in keeps the boundary."""
        a = node("a", 0, [], ["c"])
        c = node("c", 2, ["a", "b"], [])
        self.assertFalse(is_fusible_edge(a, c), "fan-in must not fuse")

    def test_different_lane_is_not_fusible(self):
        a = node("a", 0, [], ["b"], lane="gpu")
        b = node("b", 1, ["a"], [], lane="cpu")
        self.assertFalse(is_fusible_edge(a, b))

    def test_incompatible_batch_is_not_fusible(self):
        a = node("a", 0, [], ["b"], batch=1)
        b = node("b", 1, ["a"], [], batch=4)
        self.assertFalse(is_fusible_edge(a, b))
        self.assertFalse(config_compatible(a, b))

    def test_deployment_identity_is_model_plus_lane(self):
        a = node("a", 0, [], [], model="qwen", lane="gpu")
        self.assertEqual(deployment_identity(a), ("qwen", "gpu"))


class Eq3ContractionTests(unittest.TestCase):
    def test_three_node_chain_fuses_to_one_unit(self):
        """The paper's QMSum example: three consecutive same-deployment nodes."""
        a = node("a", 0, [], ["b"], runtime=100.0)
        b = node("b", 1, ["a"], ["c"], runtime=200.0)
        c = node("c", 2, ["b"], [], runtime=300.0)
        chains = maximal_fusible_chains(template([a, b, c]))
        self.assertEqual(len(chains), 1)
        ch = chains[0]
        self.assertEqual(ch.node_ids, ("a", "b", "c"))
        self.assertEqual(ch.length, 3)
        self.assertEqual(ch.boundaries_removed, 2, "h-1 boundaries removed")
        # Eq (4)'s fused duration is computed from the PREDICTOR at plan time; the chain
        # object must NOT carry a realized-runtime truth field (audit P2).
        self.assertFalse(hasattr(ch, "summed_runtime_ms"),
                         "the chain must not expose a summed realized runtime")

    def test_visible_ids_restrict_the_chain_to_the_resolved_window(self):
        """A chain may not cross a successor whose control result has not resolved."""

        a = node("a", 0, [], ["b"])
        b = node("b", 1, ["a"], ["c"])
        c = node("c", 2, ["b"], [])
        tpl = template([a, b, c])
        self.assertEqual([ch.node_ids for ch in maximal_fusible_chains(tpl)],
                         [("a", "b", "c")])
        restricted = maximal_fusible_chains(tpl, visible_ids={"a"})
        self.assertEqual([ch.node_ids for ch in restricted], [("a",)],
                         "only the resolved node may be in the window")

    def test_branch_splits_into_separate_units(self):
        a = node("a", 0, [], ["b", "c"])
        b = node("b", 1, ["a"], [])
        c = node("c", 2, ["a"], [])
        chains = maximal_fusible_chains(template([a, b, c]))
        # a cannot fuse (fan-out), so all three are singletons
        self.assertTrue(all(ch.length == 1 for ch in chains))

    def test_join_splits_into_separate_units(self):
        a = node("a", 0, [], ["c"])
        b = node("b", 1, [], ["c"])
        c = node("c", 2, ["a", "b"], [])
        chains = maximal_fusible_chains(template([a, b, c]))
        self.assertTrue(all(ch.length == 1 for ch in chains))

    def test_chain_breaks_when_the_model_changes(self):
        a = node("a", 0, [], ["b"], model="m1")
        b = node("b", 1, ["a"], ["c"], model="m2")
        c = node("c", 2, ["b"], [], model="m2")
        chains = {tuple(ch.node_ids) for ch in maximal_fusible_chains(template([a, b, c]))}
        self.assertIn(("a",), chains)
        self.assertIn(("b", "c"), chains)

    def test_every_node_belongs_to_exactly_one_unit(self):
        a = node("a", 0, [], ["b"])
        b = node("b", 1, ["a"], ["c", "d"])
        c = node("c", 2, ["b"], [])
        d = node("d", 3, ["b"], [])
        chains = maximal_fusible_chains(template([a, b, c, d]))
        seen = [nid for ch in chains for nid in ch.node_ids]
        self.assertEqual(sorted(seen), ["a", "b", "c", "d"])
        self.assertEqual(len(seen), len(set(seen)), "no node may appear twice")

    def test_max_workspace_is_the_chain_max(self):
        a = node("a", 0, [], ["b"])
        b = node("b", 1, ["a"], [])
        a = Node(**{**a.__dict__, "workspace_peak_mb": 100.0})
        b = Node(**{**b.__dict__, "workspace_peak_mb": 250.0})
        ch = maximal_fusible_chains(template([a, b]))[0]
        self.assertAlmostEqual(ch.max_workspace_peak_mb, 250.0)

    def test_summary_counts(self):
        a = node("a", 0, [], ["b"])
        b = node("b", 1, ["a"], ["c"])
        c = node("c", 2, ["b"], [])
        s = summary(maximal_fusible_chains(template([a, b, c])))
        self.assertEqual(s["chains_total"], 1)
        self.assertEqual(s["chains_multi"], 1)
        self.assertEqual(s["fused_units"], 3)
        self.assertEqual(s["boundaries_removed"], 2)


if __name__ == "__main__":
    unittest.main()
