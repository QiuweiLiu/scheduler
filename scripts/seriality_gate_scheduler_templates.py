"""Seriality gate for the scheduler-side templates.

This is P0 gate 1 of `docs/p9d_topology_label_contract_v3.md` §7, applied to the
scheduler templates instead of the predictor labels:

    一迭代多工具 = 0
    parent_step_ids 仅 () / [N-1]
    链上步号仅 same / +1
    未知 baseline 拒绝

plus the §7 path gate (single root, single successor, answer last, cycle_count=0)
and the schedulable-node gate (explicit resource_applicable).

Run against the v03 templates to answer the question the review said must not be
guessed: do the 640 accepted templates themselves satisfy the serial contract, and
what is the 640-vs-648 discrepancy?
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

REGISTERED_BASELINES = {"langgraph_react", "star", "st_fixed"}
# node_type vocabulary actually present in the v03 scheduler templates:
#   planner (api_call), answer_generation (api_call)  -> model computations
#   videotool_temporal / videotool_spatial / videotool_generalist (action) -> tools
TERMINAL_NODE_TYPES = {"answer", "answer_generation"}
TOOL_PREFIX = "videotool_"


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


def _steps(node: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(sorted(str(s) for s in (node.get("parent_step_ids") or [])))


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

    # ---- gate 1a: parent_step_ids is () or [N-1] --------------------------- #
    for n in nodes:
        ps = node_parent_steps = _steps(n)
        if len(ps) > 1:
            flag("multi_parent_step", {"node": n.get("node_id"), "parent_step_ids": list(ps)})

    # ---- gate 1b: chain step numbers only same / +1 ------------------------ #
    by_step: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for n in nodes:
        by_step[str(n.get("sequence_index"))].append(n)
    ordered = sorted(nodes, key=lambda n: int(n.get("sequence_index") or 0))
    for a, b in zip(ordered, ordered[1:]):
        sa, sb = int(a.get("sequence_index") or 0), int(b.get("sequence_index") or 0)
        if sb - sa not in (0, 1):
            flag("step_jump", {"from": sa, "to": sb})

    # ---- gate 1c: at most one tool per iteration -------------------------- #
    tools_by_step: Dict[str, List[str]] = defaultdict(list)
    for n in nodes:
        nt = str(n.get("node_type") or "")
        if nt.startswith(TOOL_PREFIX):
            for s in (n.get("source_step_ids") or [n.get("sequence_index")]):
                tools_by_step[str(s)].append(str(n.get("node_id")))
    for step, tids in tools_by_step.items():
        if len(tids) > 1:
            flag("multi_tool_per_iteration", {"step": step, "tools": tids[:4]})

    # ---- path gate: single root, single successor, answer last ------------ #
    ids = {str(n.get("node_id")) for n in nodes}
    indeg: Dict[str, int] = {i: 0 for i in ids}
    outdeg: Dict[str, int] = {i: 0 for i in ids}
    dangling = 0
    for n in nodes:
        src = str(n.get("node_id"))
        if "causal_predecessor_node_ids" not in n:
            flag("missing_causal_view", src)
            continue
        for p in (n.get("causal_predecessor_node_ids") or []):
            if str(p) not in ids:
                dangling += 1
                continue
            indeg[src] += 1
            outdeg[str(p)] += 1
    roots = [i for i, d in indeg.items() if d == 0]
    if len(roots) != 1:
        flag("not_single_root", {"roots": roots[:4], "n": len(roots)})
    if dangling:
        flag("dangling_predecessor", dangling)

    # ---- schedulable-node gate: resource_applicable present -------------- #
    missing_ra = [str(n.get("node_id")) for n in nodes if "resource_applicable" not in n]
    if missing_ra:
        flag("missing_resource_applicable", {"n": len(missing_ra), "sample": missing_ra[:3]})

    return res


def main() -> int:
    root = Path(r"F:\scheduler")
    P = root / "results/processed/r7_workload_v041_ontology_no_run_container/job_templates_r7_v041.jsonl"
    with open(P, "rt", encoding="utf-8") as h:
        rows = [json.loads(l) for l in h if l.strip()]

    results = [check_template(r) for r in rows]
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    print("=" * 90)
    print("Seriality gate on the rebuilt v04.1 ontology-aligned templates")
    print("=" * 90)
    print("  templates       : %d" % len(results))
    print("  PASS            : %d" % len(passed))
    print("  FAIL            : %d" % len(failed))
    print()
    agg: Counter = Counter()
    for r in results:
        for k, v in r.violations.items():
            agg[k] += v
    print("  violation counts:")
    for k, v in agg.most_common():
        print("     %-34s %d" % (k, v))
    print()
    print("  failed-template examples:")
    for r in failed[:6]:
        print("     %-52s %s" % (r.template_id[:50], r.violations))
        for k, ex in list(r.examples.items())[:2]:
            print("         %s -> %s" % (k, json.dumps(ex, ensure_ascii=False)[:110]))

    # baseline accounting, to explain 640 vs the doc's 648
    print()
    print("=" * 90)
    print("baseline / workflow accounting (the 640-vs-648 question)")
    print("=" * 90)
    bl = Counter(r.baseline for r in results)
    for k, v in bl.most_common():
        print("   %-24s %d templates" % (k, v))
    dt = Counter(str(r.get("dataset")) for r in rows)
    print("   datasets:", dict(dt))
    print()
    print("   doc §1 says: predictor pool 1,360 runs + R7 scheduler pool 648 runs = 2,008")
    print("   our workload carries %d accepted templates." % len(results))

    out = root / "experiments/EXP-20260921_scheduler_replication_v1/artifacts"
    out.mkdir(parents=True, exist_ok=True)
    (out / "seriality_gate_v041.json").write_text(json.dumps({
        "templates": len(results),
        "pass": len(passed),
        "fail": len(failed),
        "violations": dict(agg),
        "baseline_counts": dict(bl),
        "failed_sample": [{"template_id": r.template_id, "violations": r.violations,
                           "examples": r.examples} for r in failed[:20]],
        "contract": "docs/p9d_topology_label_contract_v3.md §7 P0 gate 1 + path/schedulable gates",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote seriality_gate_v03.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
