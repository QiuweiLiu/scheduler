"""Scheduler projection gate, rebuilt on provenance fields only.

The review's core criticism of the previous gate: it validated a transform using
fields that transform had just produced, which is a self-certifying loop.

    builder forces sequence_index = 0..n-1
    gate checks sequence_index increments by 1
    -> always passes, proves nothing

So every semantic check here reads a field that the projection does NOT synthesise:

    source_step_ids      the raw step identity
    parent_step_ids      the raw step-level dependency
    retry_of             the raw retry link
    node_type / raw_action / execution_lane   raw event vocabulary
    nested_calls         the merge record (checked for survival, not used as proof)

sequence_index is still checked, but only as a projection-index-contiguity gate and
explicitly NOT as a seriality proof.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

REGISTERED_BASELINES = {"langgraph_react", "star", "st_fixed"}
TOOL_PREFIX = "videotool_"
EXPECTED_TEMPLATES = 640
EXPECTED_NESTED_MERGED = 169


def _step_of(node: Mapping[str, Any]) -> int | None:
    """The raw step identity.  A node belongs to exactly one source step here."""

    steps = [int(s) for s in (node.get("source_step_ids") or []) if str(s).lstrip("-").isdigit()]
    if len(steps) != 1:
        return None
    return steps[0]


def _parent_steps(node: Mapping[str, Any]) -> List[int]:
    return [int(p) for p in (node.get("parent_step_ids") or []) if str(p).lstrip("-").isdigit()]


@dataclass
class GateResult:
    template_id: str
    baseline: str
    nodes: int
    violations: Dict[str, int] = field(default_factory=dict)
    examples: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.violations


def check_template(template: Mapping[str, Any]) -> GateResult:
    tid = str(template.get("template_id"))
    baseline = str(template.get("baseline") or "")
    nodes: Sequence[Mapping[str, Any]] = template.get("nodes") or []
    res = GateResult(template_id=tid, baseline=baseline, nodes=len(nodes))

    def flag(key: str, example: Any = None) -> None:
        res.violations[key] = res.violations.get(key, 0) + 1
        if key not in res.examples and example is not None:
            res.examples[key] = example

    if baseline not in REGISTERED_BASELINES:
        flag("unknown_baseline", baseline)

    # ---- provenance gate 1: parent_step_ids is () or [N-1] ----------------- #
    for n in nodes:
        ps = _parent_steps(n)
        step = _step_of(n)
        if not ps:
            continue
        if len(ps) != 1:
            flag("multi_parent_step", {"node": n.get("node_id"), "parent_step_ids": ps})
        elif step is None:
            flag("missing_source_step", {"node": n.get("node_id")})
        elif ps[0] != step - 1:
            flag("non_canonical_parent_step",
                 {"node": n.get("node_id"), "step": step, "parent_step_ids": ps})

    # ---- provenance gate 2: consecutive source steps are same or +1 -------- #
    steps_seq = [_step_of(n) for n in nodes]
    if any(s is None for s in steps_seq):
        flag("missing_source_step", {"n": sum(1 for s in steps_seq if s is None)})
    else:
        for a, b in zip(steps_seq, steps_seq[1:]):
            if b - a not in (0, 1):
                flag("source_step_jump", {"from": a, "to": b})

    # ---- provenance gate 3: at most one tool per source step -------------- #
    tools_by_step: Dict[int, List[str]] = defaultdict(list)
    for n in nodes:
        if str(n.get("node_type") or "").startswith(TOOL_PREFIX):
            s = _step_of(n)
            if s is not None:
                tools_by_step[s].append(str(n.get("node_id")))
    for s, ids in tools_by_step.items():
        if len(ids) > 1:
            flag("multi_tool_per_step", {"step": s, "tools": ids[:3]})

    # ---- provenance gate 4: retry adjacency -------------------------------- #
    index_of = {str(n.get("node_id")): i for i, n in enumerate(nodes)}
    for n in nodes:
        target = n.get("retry_of")
        if not target:
            continue
        src = str(n.get("node_id"))
        if str(target) in index_of and src in index_of:
            if index_of[src] - index_of[str(target)] != 1:
                flag("retry_not_adjacent", {"node": src, "retry_of": target})

    # ---- structural gates over the causal chain ---------------------------- #
    ids = {str(n.get("node_id")) for n in nodes}
    indeg = {i: 0 for i in ids}
    outdeg = {i: 0 for i in ids}
    edges: Dict[str, List[str]] = defaultdict(list)
    for n in nodes:
        src = str(n.get("node_id"))
        for p in (n.get("causal_predecessor_node_ids") or []):
            if str(p) not in ids:
                flag("dangling_causal_predecessor", {"node": src, "pred": p})
                continue
            indeg[src] += 1
            outdeg[str(p)] += 1
            edges[str(p)].append(src)

    roots = [i for i, d in indeg.items() if d == 0]
    if len(roots) != 1:
        flag("not_exactly_one_root", {"roots": len(roots), "sample": roots[:3]})

    for i in ids:
        if indeg[i] > 1:
            flag("causal_indegree_gt1", {"node": i, "indegree": indeg[i]})

    if len(nodes) > 1:
        for i in ids:
            expected = 0 if outdeg[i] == 0 and indeg[i] >= 0 else None
        tails = [i for i in ids if outdeg[i] == 0]
        if len(tails) != 1:
            flag("not_exactly_one_tail", {"tails": len(tails), "sample": tails[:3]})

    # cycle detection by Kahn's algorithm over the causal edges.  A colour DFS is easy
    # to get wrong (the previous version cleared the node before its children finished),
    # whereas an indegree sweep is exact and independent of the other checks.
    indeg_work = dict(indeg)
    queue = [i for i in ids if indeg_work[i] == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for nxt in edges.get(node, []):
            indeg_work[nxt] -= 1
            if indeg_work[nxt] == 0:
                queue.append(nxt)
    if visited != len(ids):
        flag("causal_cycle", {"visited": visited, "nodes": len(ids)})

    # chain covers every node exactly once
    if len(roots) == 1:
        walk = []
        cur = roots[0]
        seen = set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            walk.append(cur)
            nxt = edges.get(cur, [])
            cur = nxt[0] if len(nxt) == 1 else None
        if len(walk) != len(nodes):
            flag("chain_does_not_cover_all_nodes", {"walked": len(walk), "nodes": len(nodes)})

    # ---- ontology gates ---------------------------------------------------- #
    for n in nodes:
        if not n.get("resource_applicable", False):
            flag("resource_inapplicable_node_in_projection", {"node": n.get("node_id")})
        # provenance assertions, so a transform that mistakes a non-schedulable node for
        # a schedulable one cannot pass on the strength of the flag alone
        if str(n.get("node_type")) == "run_control":
            flag("run_control_in_projection", {"node": n.get("node_id")})
        if str(n.get("event_type")) == "run" and str(n.get("raw_action")) == "answer":
            flag("terminal_marker_in_projection", {"node": n.get("node_id")})
        if n.get("nested_calls"):
            pass  # a parent carrying a merge record is expected
    merged_ids = {m for n in nodes for m in (n.get("nested_calls") or [])}
    for m in merged_ids:
        if m in ids:
            flag("nested_child_survived", {"node": m})

    # ---- projection index contiguity (NOT a seriality proof) --------------- #
    idx = [int(n.get("sequence_index") or 0) for n in nodes]
    if idx != list(range(len(idx))):
        flag("projection_index_not_contiguous")

    return res


def main() -> int:
    root = Path(r"F:\scheduler")
    P = root / "results/processed/r7_workload_v041_ontology_no_run_container/job_templates_r7_v041.jsonl"
    with open(P, "rt", encoding="utf-8") as h:
        rows = [json.loads(l) for l in h if l.strip()]

    results = [check_template(r) for r in rows]
    failed = [r for r in results if not r.passed]
    agg: Counter = Counter()
    for r in results:
        for k, v in r.violations.items():
            agg[k] += v

    nested_merged = sum(len(n.get("nested_calls") or []) for r in rows for n in r["nodes"])
    # these are gate failures, not informational: a short file whose templates all happen
    # to be internally legal must not pass
    if len(results) != EXPECTED_TEMPLATES:
        agg["template_count_mismatch"] = abs(len(results) - EXPECTED_TEMPLATES)
    if nested_merged != EXPECTED_NESTED_MERGED:
        agg["nested_merged_count_mismatch"] = abs(nested_merged - EXPECTED_NESTED_MERGED)

    print("=" * 92)
    print("Scheduler projection gate (provenance fields only)")
    print("=" * 92)
    print("  templates            : %d (expected %d)  %s"
          % (len(results), EXPECTED_TEMPLATES, "OK" if len(results) == EXPECTED_TEMPLATES else "MISMATCH"))
    print("  nested merged        : %d (expected %d)  %s"
          % (nested_merged, EXPECTED_NESTED_MERGED,
             "OK" if nested_merged == EXPECTED_NESTED_MERGED else "MISMATCH"))
    print("  PASS                 : %d" % (len(results) - len(failed)))
    print("  FAIL                 : %d" % len(failed))
    print()
    print("  violation counts:")
    for k, v in agg.most_common():
        print("     %-40s %d" % (k, v))
    if not agg:
        print("     (none)")
    for r in failed[:5]:
        print("     %-52s %s" % (r.template_id[:50], r.violations))

    out = root / "experiments/EXP-20260921_scheduler_replication_v1/artifacts"
    out.mkdir(parents=True, exist_ok=True)
    (out / "scheduler_projection_gate_v041.json").write_text(json.dumps({
        "gate": "provenance-only scheduler projection gate",
        "templates": len(results),
        "expected_templates": EXPECTED_TEMPLATES,
        "nested_merged": nested_merged,
        "expected_nested_merged": EXPECTED_NESTED_MERGED,
        "pass": len(results) - len(failed),
        "fail": len(failed),
        "violations": dict(agg),
        "note": "semantic checks read source_step_ids / parent_step_ids / retry_of / raw node "
                "vocabulary; sequence_index is checked only for projection-index contiguity and "
                "is explicitly not a seriality proof",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote scheduler_projection_gate_v041.json")
    return 0 if not agg else 1


if __name__ == "__main__":
    raise SystemExit(main())
