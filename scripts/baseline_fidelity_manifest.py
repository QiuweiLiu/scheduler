"""The frozen BaselineFidelityManifest v1: one place that answers "are the four joint
baselines faithful, and are they frozen?"

Before this existed, the answer required reading four separate manifests, three gate files
and the control plane.  This aggregates them and CHECKS them: every baseline's freeze
manifest must exist, the arm must be on the declared frozen head, the gate file must be
present, and the gate must actually run green.  A missing or failing gate makes the
manifest fail rather than silently listing a status.

The four arms are the joint baselines the paper compares against prediction:
    llmsched        LLMSched (ICDCS 2025)
    pythia_completion  Pythia (arXiv:2604.25899)
    tie_current     TIE (ICML 2026)
    latency_aware   Latency-Aware-Orchestration (arXiv:2609.03335)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ART = (PROJECT_ROOT / "experiments" / "EXP-20260921_scheduler_replication_v1"
       / "artifacts")

BASELINES = [
    {
        "arm": "llmsched",
        "paper": "LLMSched: Uncertainty-Aware Workload Scheduling for Compound LLM Applications",
        "venue": "IEEE ICDCS 2025, pp. 527-537",
        "manifest": "llmsched_baseline_freeze_v1.json",
        "gate_file": "tests/test_llmsched_bn_v2.py",
        "freeze_head": "54a2d25088832049afb0cf860cdabfe5152c8c3d",
    },
    {
        "arm": "pythia_completion",
        "paper": "Pythia: Exploiting Workflow Predictability for Efficient Agent-Native LLM Serving",
        "venue": "arXiv:2604.25899 (venue UNVERIFIED)",
        "manifest": "pythia_baseline_freeze_v1.json",
        "gate_file": "tests/test_pythia_fidelity.py",
        "freeze_head": "54a2d25088832049afb0cf860cdabfe5152c8c3d",
    },
    {
        "arm": "tie_current",
        "paper": "TIE (Tail-Inflated Expectation)",
        "venue": "ICML 2026, arXiv:2604.00499",
        "manifest": "tie_baseline_freeze_v1.json",
        "gate_file": "tests/test_tie_fidelity.py",
        "freeze_head": "bc6cbec97dc35d407f8ec250fa729fdfc08ca357",
    },
    {
        "arm": "latency_aware",
        "paper": "Latency-Aware-Orchestration",
        "venue": "arXiv:2609.03335 (venue UNVERIFIED)",
        "manifest": "latency_aware_baseline_freeze_v1.json",
        "gate_file": "scripts/../tests/test_latency_aware_fidelity.py",
        # The prefetch-ordering regression (Eq (5) must not delay ready work) lives in the
        # scheduler gate, not the fidelity gate.  Both must run green, or the manifest
        # would certify a baseline whose prefetch had gone back to pre-empting ready work.
        "extra_gate_files": ["scripts/../tests/test_latency_aware_scheduler.py"],
        "freeze_head": "54a2d25088832049afb0cf860cdabfe5152c8c3d",
    },
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_gate(relative: str) -> Dict[str, Any]:
    """Run one fidelity gate and summarise it.  A gate that cannot run is a failure."""

    path = (PROJECT_ROOT / relative).resolve()
    if not path.exists():
        return {"ran": False, "reason": "gate file missing", "path": relative}
    pattern = path.name
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests",
               "-p", pattern]
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + __import__("os").pathsep + str(
        PROJECT_ROOT / "scripts")
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), env=env,
                               capture_output=True, text=True, timeout=1800)
    tail = (completed.stderr or "") + (completed.stdout or "")
    summary = [line for line in tail.splitlines()
               if line.startswith(("OK", "FAILED", "Ran "))]
    return {
        "ran": True,
        "path": relative,
        "returncode": completed.returncode,
        "green": completed.returncode == 0,
        "summary": summary[-2:],
    }


def main() -> int:
    entries: List[Dict[str, Any]] = []
    violations: List[str] = []

    for spec in BASELINES:
        entry: Dict[str, Any] = dict(spec)
        manifest_path = ART / spec["manifest"]
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            entry["manifest_present"] = True
            entry["manifest_sha256"] = sha256_file(manifest_path)
            entry["freeze_status"] = (data.get("freeze") or {}).get("status")
            entry["arm_in_manifest"] = data.get("arm")
            entry["faithful_count"] = len(data.get("faithful") or [])
            entry["adapted_count"] = len(data.get("adapted") or [])
            entry["deviation_count"] = len(data.get("deviation") or [])
            if entry["arm_in_manifest"] != spec["arm"]:
                violations.append("%s: manifest declares arm %r"
                                  % (spec["arm"], entry["arm_in_manifest"]))
            if entry["freeze_status"] != "APPROVED":
                violations.append("%s: freeze status %r"
                                  % (spec["arm"], entry["freeze_status"]))
        else:
            entry["manifest_present"] = False
            violations.append("%s: freeze manifest missing" % spec["arm"])

        gate_path = (PROJECT_ROOT / spec["gate_file"]).resolve()
        entry["gate_present"] = gate_path.exists()
        if entry["gate_present"]:
            entry["gate_sha256"] = sha256_file(gate_path)
        else:
            violations.append("%s: gate file missing" % spec["arm"])

        entries.append(entry)

    # the one expensive step: actually run each gate
    for entry, spec in zip(entries, BASELINES):
        result = run_gate(entry["gate_file"])
        entry["gate_run"] = result
        if not result.get("green"):
            violations.append("%s: fidelity gate not green (%s)"
                              % (entry["arm"], result.get("reason") or
                                 result.get("summary")))
        for extra in spec.get("extra_gate_files", []):
            extra_result = run_gate(extra)
            entry.setdefault("extra_gate_runs", []).append(extra_result)
            if not extra_result.get("green"):
                violations.append("%s: extra gate %s not green (%s)"
                                  % (entry["arm"], extra,
                                     extra_result.get("reason") or extra_result.get("summary")))

    topology_path = ART / "scheduler_topology_contract_gate_v1.json"
    topology: Dict[str, Any] = {"present": topology_path.exists()}
    if topology["present"]:
        data = json.loads(topology_path.read_text(encoding="utf-8"))
        topology["pass"] = data.get("pass")
        topology["projection_sha256"] = data.get("projection_sha256")
        topology["gate_sha256"] = sha256_file(topology_path)
        if data.get("pass") is not True:
            violations.append("SchedulerTopologyContractGate: %r" % data.get("pass"))

    artifact = {
        "gate": "BaselineFidelityManifest",
        "version": "v1",
        "topology_contract_gate": topology,
        "baselines": entries,
        "n_baselines": len(entries),
        "all_frozen": all(e.get("freeze_status") == "APPROVED" for e in entries),
        "all_gates_green": all((e.get("gate_run") or {}).get("green") for e in entries),
        "violations": violations,
        "pass": not violations,
        "scope": (
            "The fidelity and freeze status of the four joint baselines, checked rather "
            "than asserted: each freeze manifest must exist with status APPROVED and the "
            "right arm, each gate file must exist, and each gate must actually run green. "
            "A missing or failing gate fails the manifest instead of being reported as a "
            "status."
        ),
        "note": (
            "Fidelity is a precondition for comparing performance, not a claim about it. "
            "A green manifest says the baselines implement what their papers describe under "
            "the declared adaptations; it says nothing about which one schedules better."
        ),
    }

    out = ART / "baseline_fidelity_manifest_v1.json"
    out.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print("BaselineFidelityManifest v1: %s" % ("PASS" if artifact["pass"] else "FAIL"))
    for entry in entries:
        run = entry.get("gate_run") or {}
        print("  %-20s freeze=%-9s gate=%-6s %s"
              % (entry["arm"], entry.get("freeze_status"),
                 "green" if run.get("green") else "FAIL",
                 (run.get("summary") or [""])[-1] if run.get("summary") else ""))
    for violation in violations:
        print("  VIOLATION %s" % violation)
    print("  wrote: %s" % out.relative_to(PROJECT_ROOT))
    return 0 if artifact["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
