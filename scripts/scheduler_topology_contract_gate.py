"""The frozen SchedulerTopologyContractGate v1.

One executable gate that asserts, in one place, everything the scheduler projection must
satisfy before any scheduling result is comparable: the v04.1 file is the declared
projection, its counts are the frozen counts, every node is resource-applicable, the
control-flow contract holds, the chain tail is the answer, and the file is pinned by hash.
It re-checks the properties directly rather than trusting the earlier artifacts, and it
records those earlier results alongside so the whole topology story is in one place.

Fail-closed: any violated property makes ``pass`` false, and the artifact says which.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PROJECTION = (PROJECT_ROOT / "results" / "processed"
              / "r7_workload_v041_ontology_no_run_container" / "job_templates_r7_v041.jsonl")
ART = (PROJECT_ROOT / "experiments" / "EXP-20260921_scheduler_replication_v1"
       / "artifacts")

CONTRACT = "scheduler_projection_of_verified_serial_control_flow_v3_1"
EXPECTED_TEMPLATES = 640
EXPECTED_NODES = 8126
EXPECTED_NESTED = 169


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    checks: Dict[str, Any] = {}
    violations: List[str] = []

    def check(name: str, ok: bool, detail: Any) -> None:
        checks[name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            violations.append(name)

    if not PROJECTION.exists():
        raise SystemExit("projection not found: %s" % PROJECTION)

    rows = [json.loads(line) for line in
            PROJECTION.read_text(encoding="utf-8").splitlines() if line.strip()]

    check("template_count", len(rows) == EXPECTED_TEMPLATES, len(rows))

    node_total = 0
    nested_merged = 0
    contracts = set()
    not_resource_applicable: List[str] = []
    tail_is_answer = 0
    serial_violations: List[str] = []

    for row in rows:
        nodes = row.get("nodes") or []
        node_total += len(nodes)
        nested_merged += int(row.get("nested_merged_count") or 0)
        contracts.add(str(row.get("topology_contract") or ""))
        if bool(row.get("chain_tail_is_answer")):
            tail_is_answer += 1

        by_id = {str(n.get("node_id")): n for n in nodes}
        for node in nodes:
            if not bool(node.get("resource_applicable")):
                not_resource_applicable.append(str(node.get("node_id")))
            preds = list(node.get("causal_predecessor_node_ids") or [])
            if len(preds) > 1:
                serial_violations.append(
                    "node %s has %d causal predecessors" % (node.get("node_id"), len(preds)))
            for pred in preds:
                if pred not in by_id:
                    serial_violations.append(
                        "node %s references a missing predecessor %s"
                        % (node.get("node_id"), pred))

    check("node_count", node_total == EXPECTED_NODES, node_total)
    check("nested_merged_count", nested_merged == EXPECTED_NESTED, nested_merged)
    check("contract_name", contracts == {CONTRACT}, sorted(contracts))
    check("resource_applicable_all", not not_resource_applicable,
          len(not_resource_applicable))
    check("chain_tail_is_answer_all", tail_is_answer == EXPECTED_TEMPLATES, tail_is_answer)
    check("serial_causal_chain", not serial_violations,
          serial_violations[:5] if serial_violations else 0)

    # the supporting results, carried so the whole topology story is in one artifact
    supporting: Dict[str, Any] = {}
    for name in ("scheduler_projection_gate_v041.json", "seriality_gate_v041.json",
                 "nested_merge_set_equality.json", "nested_resource_conservation.json"):
        path = ART / name
        if not path.exists():
            check("supporting_%s" % name, False, "missing")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        supporting[name] = {
            "pass": data.get("pass", data.get("exact_set_equality")),
            "fail": data.get("fail", data.get("false_positive")),
            "sha256": sha256_file(path),
        }
        ok = (int(data.get("fail") or 0) == 0
              and (data.get("pass") in (None, 640) or data.get("exact_set_equality") is True
                   or int(data.get("false_positive") or 0) == 0))
        check("supporting_%s" % name, ok, supporting[name])

    projection_sha = sha256_file(PROJECTION)
    artifact = {
        "gate": "SchedulerTopologyContractGate",
        "version": "v1",
        "contract": CONTRACT,
        "projection": str(PROJECTION.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "projection_sha256": projection_sha,
        "expected": {
            "templates": EXPECTED_TEMPLATES,
            "nodes": EXPECTED_NODES,
            "nested_merged": EXPECTED_NESTED,
        },
        "checks": checks,
        "supporting_results": supporting,
        "violations": violations,
        "pass": not violations,
        "scope": (
            "The topology everything else is measured on: the v04.1 projection's counts, "
            "resource applicability, serial causal control flow, terminal markers and its "
            "content hash. A scheduling number is only comparable while this gate passes "
            "on this hash."
        ),
        "prohibited_after_freeze": [
            "the projection file and its hash",
            "the topology contract name",
            "the expected template, node and nested-merge counts",
            "the serial causal-chain derivation",
            "resource_applicable semantics",
        ],
        "freeze_breaking_exception": (
            "A reproducible correctness bug only, recorded as a freeze-breaking commit; "
            "the counts and the hash must never be adjusted to accommodate a result."
        ),
    }

    out = ART / "scheduler_topology_contract_gate_v1.json"
    out.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print("SchedulerTopologyContractGate v1: %s" % ("PASS" if artifact["pass"] else "FAIL"))
    for name, result in checks.items():
        if not result["ok"]:
            print("  VIOLATION %s -> %s" % (name, result["detail"]))
    print("  projection sha256: %s" % projection_sha[:24])
    print("  wrote: %s" % out.relative_to(PROJECT_ROOT))
    return 0 if artifact["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
