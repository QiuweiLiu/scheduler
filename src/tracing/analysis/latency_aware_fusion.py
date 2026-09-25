"""Eq (3) of the Latency-Aware-Orchestration paper: maximal fusible chains.

The paper contracts a chain when

    u, v in V_A,  succ_{L_t}(u) = {v},  pred_{L_t}(v) = {u},
    d(u) = d(v),  theta(u) ~_cfg theta(v)

i.e. a *one-to-one* successor/predecessor relation (which keeps branches and joins
at chain boundaries), the same deployment identity, and configuration-compatible
request shapes.  The fused unit then runs under one grant and one replica lease,
while every activation keeps its own request configuration and hands its output to
the next.

This module implements the legality rule exactly and measures how much material the
real v03 workload offers.  It is deliberately read-only and dependency-free so the
rule can be unit-tested against hand-built graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class FusedChain:
    """One maximal fusible chain: activations that share a single grant."""

    unit_id: str
    node_ids: Tuple[str, ...]
    deployment: Tuple[str, str]
    # There is deliberately NO summed_runtime_ms here.  It used to carry
    # ``sum(m.runtime_ms)``, i.e. the SUM OF REALIZED RUN TIMES of the chain members --
    # a truth field that the scheduler is supposed to PREDICT.  It was unused by the
    # scoring path, so it was never an active leak, but keeping a truth quantity on the
    # object that a fused plan is built from invites exactly the substitution this arm
    # already had to remove elsewhere.  The fused duration is computed from the
    # request-conditioned predictor in ``latency_aware_scheduler.predicted_duration_ms``.
    max_workspace_peak_mb: float

    @property
    def length(self) -> int:
        return len(self.node_ids)

    @property
    def boundaries_removed(self) -> int:
        """A length-h chain removes h-1 intermediate scheduling boundaries."""

        return max(0, self.length - 1)


def deployment_identity(node: Any) -> Tuple[str, str]:
    """The paper's deployment identity d: model + version + serving configuration.

    Our workload exposes model_id and execution lane; the serving configuration is
    not separately recorded per node, so the identity is (model_id, lane).  Stated
    as an adaptation deviation: two nodes on the same model but different lanes are
    different deployments, which is the conservative reading.
    """

    return (str(node.model_id), str(node.lane))


def config_compatible(a: Any, b: Any) -> bool:
    """theta(a) ~_cfg theta(b): configuration compatibility up to output limits.

    We do not record per-node output-length limits, so the rule reduces to "same
    lane and same batch size", which is the part of the serving configuration our
    data actually carries.  NOT MIGRATED: the output-length ceiling comparison.
    """

    return str(a.lane) == str(b.lane) and int(getattr(a, "batch_size", 1)) == int(
        getattr(b, "batch_size", 1)
    )


def is_fusible_edge(u: Any, v: Any) -> bool:
    """Eq (3), evaluated on one directed edge."""

    if not u.successors or not v.predecessors:
        return False
    # one-to-one: keeps branches and joins at chain boundaries
    if tuple(u.successors) != (v.node_id,):
        return False
    if tuple(v.predecessors) != (u.node_id,):
        return False
    if deployment_identity(u) != deployment_identity(v):
        return False
    return config_compatible(u, v)


def maximal_fusible_chains(template: Any, visible_ids: Any = None) -> List[FusedChain]:
    """Contract every maximal eligible chain of one workflow (one template).

    ``visible_ids`` restricts the contraction to the RESOLVED logical window: when it is
    given, only nodes whose control result has resolved (and, on an edge, only a successor
    that is itself resolved) may enter a chain.  The paper's ``L_t`` is the resolved
    workflow portion, not the complete realized graph, so contracting a chain across a
    not-yet-resolved successor would read structure the scheduler is not entitled to.
    ``None`` keeps the full-template behaviour for offline analysis and its own unit tests.
    """

    allowed = None if visible_ids is None else {str(v) for v in visible_ids}
    nodes = [n for n in template.nodes if allowed is None or n.node_id in allowed]
    by_id = {n.node_id: n for n in nodes}

    # a node may have at most one fusible successor and one fusible predecessor,
    # because Eq (3) is a one-to-one relation
    next_node: Dict[str, str] = {}
    prev_node: Dict[str, str] = {}
    for u in nodes:
        for v_id in u.successors:
            v = by_id.get(v_id)
            if v is None:
                continue
            if is_fusible_edge(u, v):
                if u.node_id in next_node or v.node_id in prev_node:
                    raise AssertionError("Eq (3) is one-to-one but the graph branched")
                next_node[u.node_id] = v.node_id
                prev_node[v.node_id] = u.node_id

    chains: List[FusedChain] = []
    seen: set[str] = set()
    for n in nodes:
        if n.node_id in prev_node or n.node_id in seen:
            continue  # not a chain head
        ids = [n.node_id]
        cur = n.node_id
        while cur in next_node:
            cur = next_node[cur]
            ids.append(cur)
        seen.update(ids)
        members = [by_id[i] for i in ids]
        chains.append(
            FusedChain(
                unit_id="%s::fuse%d" % (template.template_id, len(chains)),
                node_ids=tuple(ids),
                deployment=deployment_identity(members[0]),
                max_workspace_peak_mb=max(
                    (float(m.workspace_peak_mb or 0.0) for m in members), default=0.0
                ),
            )
        )
    return chains


def summary(chains: Sequence[FusedChain]) -> Dict[str, Any]:
    multi = [c for c in chains if c.length > 1]
    return {
        "chains_total": len(chains),
        "chains_multi": len(multi),
        "fused_units": sum(c.length for c in multi),
        "boundaries_removed": sum(c.boundaries_removed for c in multi),
        "length_histogram": {
            str(k): sum(1 for c in chains if c.length == k)
            for k in sorted({c.length for c in chains})
        },
        "longest": max((c.length for c in chains), default=0),
    }
