"""Pythia-adapted: a history-derived workflow profiler + completion-aware priority.

Paper: Pythia: Exploiting Workflow Predictability for Efficient Agent-Native LLM
Serving (arXiv:2604.25899).  Its front end is NOT a neural prefix predictor: it
builds a workflow profiler / probabilistic finite automaton from historical agent
execution traces over a role alphabet, prunes low-probability paths, and derives an
expected remaining distance.  Its back end turns that into a completion-aware
priority.

Migrated faithfully (the Pythia-core chain the review named):
    train-only workflow traces -> profiler -> bounded probable future
        -> E[remaining distance] -> completion-aware priority

NOT migrated, and stated as such:
  * cache routing, prefix caching, model-replica idleness and autoscaling are
    outside our GPU execution abstraction;
  * the downstream-idle term S_unblock is omitted because this simulator has no
    model-server queue abstraction to attach it to, and the review explicitly
    preferred an honest omission over inventing one.

So priority = omega1 * S_completion with omega1 = 1 and omega2 = 0.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping

PROFILER_SCHEMA = "pythia-workflow-profiler-v1"


def build_pythia_profiler(templates: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the profiler from TRAIN-ONLY templates.

    The role alphabet is abstracted to the workflow family (``baseline``), which is
    the coarse identity a history-derived profiler can legitimately know.  For each
    family we record, for every consumed count ``k``, the mean runtime of the nodes
    that remain after position ``k`` -- the expected remaining distance.
    """

    by_family: Dict[str, List[List[float]]] = defaultdict(list)
    for tpl in templates.values():
        if getattr(tpl, "split", None) != "train":
            continue
        order = sorted(tpl.nodes, key=lambda n: n.sequence_index)
        by_family[str(getattr(tpl, "baseline", "") or "unknown")].append(
            [float(n.runtime_ms) for n in order]
        )

    profiler: Dict[str, Any] = {"schema": PROFILER_SCHEMA, "families": {}}
    for family, runs in by_family.items():
        max_len = max(len(r) for r in runs)
        # remaining_mean[k] = mean over runs of sum(run[k:]), i.e. the expected
        # runtime still ahead once k nodes have been consumed
        remaining_mean: List[float] = []
        for k in range(max_len + 1):
            vals = [sum(r[k:]) for r in runs if len(r) >= k]
            remaining_mean.append(sum(vals) / len(vals) if vals else 0.0)
        profiler["families"][family] = {
            "n_runs": len(runs),
            "max_length": max_len,
            "remaining_mean_by_consumed": remaining_mean,
            "length_histogram": {
                str(n): sum(1 for r in runs if len(r) == n) for n in sorted({len(r) for r in runs})
            },
        }
    return profiler


def expected_remaining_ms(profiler: Mapping[str, Any], family: str, consumed: int) -> float:
    """E[D_remaining after the current node] for a job that has consumed k nodes."""

    fam = (profiler.get("families") or {}).get(str(family))
    if fam is None:
        raise KeyError("profiler has no family %r" % family)
    table = fam["remaining_mean_by_consumed"]
    idx = max(0, min(int(consumed) + 1, len(table) - 1))
    return float(table[idx])


def s_completion(d_remaining_ms: float) -> float:
    """S_completion = 1 / E[D_remaining]; larger is better, so the key uses -S."""

    return 1.0 / max(1.0, float(d_remaining_ms))
