"""v04.1: align the scheduler template with the v3.1 node ontology.

The v04 rebuild fixed only the EDGES.  This one fixes the NODE SET, which the review
showed was still unmigrated:

  * nested merge (v3.1 section 4 rule 3): an ``answer_generation`` whose action is
    ``generalist.generate`` and which is immediately followed, inside the same step,
    by the ``summarization-tool`` is a call nested INSIDE that tool.  It must be
    merged into its parent and removed from the schedulable node set, because v3.1
    fixes ``R_node = R_total(parent summarizer)`` and forbids counting the nested
    resource again.  Leaving it in double-counts both the work and the duration.
  * run_control exclusion (v3.1 section 3): not a topology node.
  * terminal marker (v3.1 section 3): ``event_type == "run" and action == "answer"``,
    resource-inapplicable.  The v03 workload already dropped every ``run`` container,
    so this projection carries no terminal marker at all.  The contract is therefore
    named a PROJECTION rather than claimed identical to the predictor-side v3.1.

Everything else (causal edges, raw edges as an audit field) is inherited from v04.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

CONTRACT = "scheduler_projection_of_verified_serial_control_flow_v3_1"
NESTED_ACTION = "generalist.generate"
SUMMARIZER_ACTION = "summarization-tool"
NESTED_NODE_TYPE = "answer_generation"


def is_nested_call(node: Mapping[str, Any]) -> bool:
    """v3.1 section 4 rule 3: an answer model invoked inside the summarizer tool."""

    return (str(node.get("node_type")) == NESTED_NODE_TYPE
            and str(node.get("raw_action")) == NESTED_ACTION)


def is_summarizer_tool(node: Mapping[str, Any]) -> bool:
    return str(node.get("raw_action")) == SUMMARIZER_ACTION


def is_terminal_marker(node: Mapping[str, Any]) -> bool:
    """v3.1 section 3: the final ``answer`` run event, not schedulable work."""

    return str(node.get("event_type")) == "run" and str(node.get("raw_action")) == "answer"


def is_run_control(node: Mapping[str, Any]) -> bool:
    return str(node.get("node_type")) == "run_control"


def resource_applicable(node: Mapping[str, Any]) -> bool:
    if is_run_control(node):
        return False
    if is_terminal_marker(node):
        return False
    return True


def merge_nested(nodes: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """Apply the nested merge inside each source step; return (kept, merged_into)."""

    by_step: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for n in nodes:
        by_step[tuple(sorted(str(s) for s in (n.get("source_step_ids") or ["?"])))].append(n)

    drop: set[str] = set()
    merged_into: Dict[str, List[str]] = defaultdict(list)
    for step, members in by_step.items():
        ordered = sorted(members, key=lambda x: int(x.get("sequence_index") or 0))
        for i, n in enumerate(ordered):
            if not is_nested_call(n):
                continue
            followers = [x for x in ordered[i + 1:] if is_summarizer_tool(x)]
            if not followers:
                continue
            parent = str(followers[0]["node_id"])
            drop.add(str(n["node_id"]))
            merged_into[parent].append(str(n["node_id"]))

    kept: List[Dict[str, Any]] = []
    for n in nodes:
        nid = str(n.get("node_id"))
        if nid in drop or is_run_control(n):
            continue
        entry = dict(n)
        entry["resource_applicable"] = bool(resource_applicable(n))
        entry["merged_nested_call"] = nid in merged_into
        entry["nested_calls"] = list(merged_into.get(nid, []))
        entry["topology_contract"] = CONTRACT
        kept.append(entry)
    return kept, dict(merged_into)


def rebuild(template: Mapping[str, Any]) -> Dict[str, Any]:
    nodes = list(template.get("nodes") or [])
    kept, merged_into = merge_nested(nodes)

    # causal chain over the SURVIVING nodes, in source order (v3.1 section 2: file
    # order is the serialisation of the verified sequence; it is not re-sorted here)
    out: List[Dict[str, Any]] = []
    for i, n in enumerate(kept):
        prior = [str(kept[i - 1]["node_id"])] if i else []
        out.append({
            **n,
            # renumber: removing a merged node leaves a hole in the original positions,
            # and the chain position must stay contiguous
            "sequence_index": i,
            "causal_predecessor_node_ids": prior,
            "raw_predecessor_node_ids": [str(p) for p in (n.get("raw_predecessor_node_ids")
                                                          or n.get("predecessor_node_ids") or [])],
        })

    rebuilt = dict(template)
    rebuilt.pop("predecessor_node_ids", None)
    rebuilt["nodes"] = out
    rebuilt["node_count"] = len(out)
    rebuilt["topology_contract"] = CONTRACT
    rebuilt["node_ontology_version"] = CONTRACT
    rebuilt["edge_policy"] = "verified_serial_control_flow_chain"
    rebuilt["nested_merged_count"] = sum(len(v) for v in merged_into.values())
    rebuilt["terminal_marker_policy"] = (
        "run_control and terminal markers are resource-inapplicable; this projection "
        "already dropped every `run` container, so it carries no terminal marker"
    )
    return rebuilt


def main() -> int:
    root = Path(r"F:\scheduler")
    src = root / "results/processed/r7_workload_v04_causal_v31_no_run_container/job_templates_r7_v04.jsonl"
    out_dir = root / "results/processed/r7_workload_v041_ontology_no_run_container"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "job_templates_r7_v041.jsonl"

    with open(src, "rt", encoding="utf-8") as h:
        rows = [json.loads(l) for l in h if l.strip()]
    rebuilt = [rebuild(r) for r in rows]

    merged_ids = {m for r in rebuilt for n in r["nodes"] for m in n["nested_calls"]}
    before = sum(len(r.get("nodes") or []) for r in rows)
    after = sum(len(r["nodes"]) for r in rebuilt)
    nested = sum(r["nested_merged_count"] for r in rebuilt)
    ra_false = sum(1 for r in rebuilt for n in r["nodes"] if not n["resource_applicable"])

    # post-conditions (fail closed)
    for r in rebuilt:
        for i, n in enumerate(r["nodes"]):
            assert n["causal_predecessor_node_ids"] == ([r["nodes"][i - 1]["node_id"]] if i else [])
        for n in r["nodes"]:
            # only the MERGED nested ids must be gone; a generalist.generate with no
            # summarizer follower is a legitimate post-loop answer computation
            assert str(n["node_id"]) not in merged_ids, "a merged nested call survived"
            assert not is_run_control(n), "a run_control survived in %s" % r["template_id"]

    print("templates            : %d" % len(rebuilt))
    print("nodes before         : %d" % before)
    print("nodes after          : %d" % after)
    print("nested merged        : %d   (the review counted 169)" % nested)
    print("resource_applicable=false : %d" % ra_false)
    print("contract             : %s" % CONTRACT)

    with open(out, "w", encoding="utf-8") as h:
        for r in rebuilt:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "job_templates_r7_v041.summary.json").write_text(json.dumps({
        "schema_version": "job-template-v0.4.1",
        "source": str(src),
        "output": str(out),
        "derivation": "v3.1 node ontology: nested summarizer calls merged into their parent, "
                      "run_control excluded, explicit resource_applicable, causal chain over the "
                      "surviving nodes in source order",
        "topology_contract": CONTRACT,
        "templates": len(rebuilt),
        "nodes_before": before,
        "nodes_after": after,
        "nested_merged": nested,
        "resource_applicable_false": ra_false,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
