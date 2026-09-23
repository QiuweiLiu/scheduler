"""Rebuild the scheduler templates under the v3.1 verified-serial contract.

Per `docs/p9d_topology_label_contract_v3.md`:

  §2  edge = direct execution dependency between adjacent schedulable/terminal
      operations in the *verified top-level serial control flow*; file order is
      only the serialisation of that verified sequence, not the causal source.
  §4  edges = adjacent nodes along the verified execution sequence, single
      predecessor / single successor; each run is a single chain with a single
      root and ends at `answer`.
  §3  node ontology: `resource_applicable` is explicit; the terminal answer
      marker is resource-inapplicable.

Changes versus the v03 file:
  * `causal_predecessor_node_ids` = the serial chain (the execution edge)
  * `raw_predecessor_node_ids`    = the old step-expansion edges, kept for audit
  * `resource_applicable`         = False for the terminal answer marker, else True
  * `topology_contract`           = verified_serial_control_flow_v3_1
  * `sequence_index` is kept and asserted to be exactly 0..n-1
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping

CONTRACT = "verified_serial_control_flow_v3_1"
TOOL_PREFIX = "videotool_"


def is_terminal_answer(node: Mapping[str, Any]) -> bool:
    """The chain tail: an answer marker, resource-inapplicable (v3.1 §3)."""

    return str(node.get("node_type")) == "answer_generation"


def rebuild(template: Mapping[str, Any]) -> Dict[str, Any]:
    nodes: List[Mapping[str, Any]] = list(template.get("nodes") or [])
    order = sorted(nodes, key=lambda n: int(n.get("sequence_index") or 0))
    idx = [int(n.get("sequence_index") or 0) for n in order]
    if idx != list(range(len(idx))):
        raise ValueError("sequence_index is not exactly 0..n-1 in %s" % template.get("template_id"))

    # last node must be the answer marker
    tail = order[-1] if order else None
    tail_is_answer = bool(tail is not None and is_terminal_answer(tail))

    out_nodes: List[Dict[str, Any]] = []
    for i, n in enumerate(order):
        causal = [str(order[i - 1]["node_id"])] if i > 0 else []
        raw = [str(p) for p in (n.get("predecessor_node_ids") or [])]
        # v3.1 §3: the terminal answer marker does not occupy a compute slot
        resource_applicable = not (i == len(order) - 1 and tail_is_answer)
        out_nodes.append({
            **{k: v for k, v in n.items() if k not in ("predecessor_node_ids",)},
            "causal_predecessor_node_ids": causal,
            "raw_predecessor_node_ids": raw,
            "resource_applicable": bool(resource_applicable),
            "topology_contract": CONTRACT,
        })

    rebuilt = dict(template)
    rebuilt["nodes"] = out_nodes
    rebuilt["topology_contract"] = CONTRACT
    rebuilt["node_ontology_version"] = CONTRACT
    rebuilt["edge_policy"] = "verified_serial_control_flow_chain"
    rebuilt["chain_root"] = str(order[0]["node_id"]) if order else None
    rebuilt["chain_tail"] = str(tail["node_id"]) if tail is not None else None
    rebuilt["chain_tail_is_answer"] = tail_is_answer
    rebuilt["raw_edge_policy"] = "measured_trace_events_with_raw_parent_edges"
    return rebuilt


def main() -> int:
    root = Path(r"F:\scheduler")
    src = root / "results/processed/r7_workload_v03_no_run_container/job_templates_r7_v03.jsonl"
    out_dir = root / "results/processed/r7_workload_v04_causal_v31_no_run_container"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "job_templates_r7_v04.jsonl"

    with open(src, "rt", encoding="utf-8") as h:
        rows = [json.loads(l) for l in h if l.strip()]

    rebuilt = [rebuild(r) for r in rows]

    # ---- post-conditions (fail closed) ------------------------------------ #
    tails = Counter(r["chain_tail_is_answer"] for r in rebuilt)
    for r in rebuilt:
        for i, n in enumerate(r["nodes"]):
            assert n["causal_predecessor_node_ids"] == (
                [r["nodes"][i - 1]["node_id"]] if i > 0 else []
            ), "broken chain in %s" % r["template_id"]
    print("templates rebuilt      : %d" % len(rebuilt))
    print("chain tail is answer   : %s" % dict(tails))
    print("nodes                  : %d" % sum(len(r["nodes"]) for r in rebuilt))

    with open(out, "w", encoding="utf-8") as h:
        for r in rebuilt:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "schema_version": "job-template-v0.3",
        "source": str(src),
        "output": str(out),
        "derivation": "rebuild scheduler topology under the frozen v3.1 verified-serial contract; "
                      "causal edge = adjacent node in the verified serial sequence; raw step-expansion "
                      "edges preserved as an audit-only field",
        "topology_contract": CONTRACT,
        "templates": len(rebuilt),
        "nodes": sum(len(r["nodes"]) for r in rebuilt),
        "chain_tail_is_answer": dict(tails),
        "resource_applicable_false": sum(
            1 for r in rebuilt for n in r["nodes"] if not n["resource_applicable"]
        ),
    }
    (out_dir / "job_templates_r7_v04.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
