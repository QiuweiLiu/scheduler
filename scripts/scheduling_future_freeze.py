#!/usr/bin/env python3
"""Freeze the inputs and environment for the future-information experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_summary(path: Path) -> dict[str, Any]:
    rows = 0
    first_keys: list[str] | None = None
    for line_number, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        rows += 1
        if first_keys is None:
            first_keys = sorted(value)
    return {"rows": rows, "first_row_keys": first_keys or []}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_text(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError:
        return None
    value = result.stdout.strip()
    return value or None


def package_version(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    module = __import__(name)
    return str(getattr(module, "__version__", "present"))


def file_record(path: Path, include_jsonl: bool = False) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    record: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if include_jsonl:
        record["jsonl"] = jsonl_summary(path)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/processed/scheduling_future_v1_20260812",
    )
    args = parser.parse_args()
    output_root: Path = args.output_root
    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite existing output root: {output_root}")

    templates = ROOT / "results/processed/workload_v0_2_smoke_fixed_20260812/job_templates.jsonl"
    workload_root = ROOT / "results/processed/workload_v0_2_formal_20260812"
    candidate_root = ROOT / "results/processed/behavior_nn_v1_r1v5_candidate/data"
    r2_root = ROOT / "results/processed/behavior_nn_v1_r2/final_holdout"
    resource_root = ROOT / "results/processed/resource_predictor_v1_final_q99_20260811"

    input_paths: dict[str, tuple[Path, bool]] = {
        "templates": (templates, True),
        "workload_train": (workload_root / "train.jsonl", True),
        "workload_validation": (workload_root / "validation.jsonl", True),
        "workload_test": (workload_root / "test.jsonl", True),
        "workload_test_summary": (workload_root / "test.jsonl.summary.json", False),
        "workload_generation_manifest": (workload_root / "generation_manifest.json", False),
        "workload_audit": (workload_root / "audit_summary.json", False),
        "candidate_role_samples": (candidate_root / "role_event_samples.jsonl", True),
        "candidate_tool_samples": (candidate_root / "semantic_tool_samples.jsonl", True),
        "candidate_split_manifest": (candidate_root / "split_manifest.json", False),
        "r2_dataset_report": (r2_root / "dataset/dataset_report.json", False),
        "r2_role_samples": (r2_root / "dataset/role_event_samples.jsonl", True),
        "r2_tool_samples": (r2_root / "dataset/semantic_tool_samples.jsonl", True),
        "resource_contract": (resource_root / "scheduler_resource_contract.json", False),
        "resource_models": (resource_root / "models.pkl", False),
    }
    inputs = {name: file_record(path, jsonl) for name, (path, jsonl) in input_paths.items()}

    code_paths = [
        ROOT / "src/tracing/analysis/workload_v02_simulator.py",
        ROOT / "src/tracing/analysis/resource_predictor_v1.py",
        ROOT / "scripts_behavior_r2_v5/r2/preprocessing.py",
        ROOT / "scripts_behavior_r2_v5/r2_v5/r2b/models_seq.py",
        ROOT / "scripts_behavior_r2_v5/r2_v5/r2b/shared.py",
        ROOT / "scripts_behavior_r2_v5/r2_v5/r2b/final_holdout_runner.py",
    ]
    code_snapshot = {
        str(path): {"sha256": sha256(path), "size_bytes": path.stat().st_size}
        for path in code_paths
    }

    template_summary = inputs["templates"]["jsonl"]
    workload_manifest = load_json(workload_root / "generation_manifest.json")
    workload_audit = load_json(workload_root / "audit_summary.json")
    workload_test_summary = load_json(workload_root / "test.jsonl.summary.json")
    resource_contract = load_json(resource_root / "scheduler_resource_contract.json")

    if template_summary["rows"] != 768:
        raise RuntimeError(f"expected 768 templates, got {template_summary['rows']}")
    if workload_manifest["outputs"]["train"]["episodes"] != 20000:
        raise RuntimeError("unexpected workload train episode count")
    if workload_manifest["outputs"]["validation"]["episodes"] != 2000:
        raise RuntimeError("unexpected workload validation episode count")
    if workload_manifest["outputs"]["test"]["episodes"] != 6750:
        raise RuntimeError("unexpected workload test episode count")
    if workload_test_summary.get("scenario_matrix_expected_cells") != 135 or not workload_audit.get("gate"):
        raise RuntimeError("workload audit gate/cell count is not frozen as expected")
    if resource_contract.get("schema_version") != "scheduler-resource-contract-v1":
        raise RuntimeError("unexpected resource contract schema")

    output_root.mkdir(parents=True)
    for name in ("configs", "manifests", "prediction_artifacts", "smoke", "validation", "locked_test", "leakage_audit", "overhead"):
        (output_root / name).mkdir()

    environment: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": package_version("torch"),
        "numpy": package_version("numpy"),
        "scipy": package_version("scipy"),
        "git_commit": run_text(["git", "rev-parse", "HEAD"]),
        "git_worktree": run_text(["git", "rev-parse", "--show-toplevel"]),
        "nvidia_smi": run_text(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "solver_availability": {
            "scipy_milp": bool(package_version("scipy")),
            "ortools": package_version("ortools"),
            "gurobipy": package_version("gurobipy"),
        },
    }
    manifest = {
        "schema_version": "scheduling-future-input-freeze-v1",
        "experiment_id": "scheduling_future_v1_20260812",
        "project_root": str(ROOT),
        "git_commit": environment["git_commit"],
        "git_worktree": environment["git_worktree"],
        "code_snapshot": code_snapshot,
        "inputs": inputs,
        "expected_counts": {
            "templates": 768,
            "nodes": 9366,
            "workload_train_episodes": 20000,
            "workload_validation_episodes": 2000,
            "workload_test_episodes": 6750,
            "locked_test_cells": 135,
        },
        "source_policy": {
            "raw_traces_immutable": True,
            "locked_test_not_read_for_parameter_selection": True,
            "predicted_and_truth_artifacts_separate": True,
            "unknown_resource_not_zeroed": True,
        },
        "environment": environment,
    }
    (output_root / "configs/e0_freeze.json").write_text(
        json.dumps({"experiment_id": manifest["experiment_id"], "output_root": str(output_root), "source_policy": manifest["source_policy"]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "manifests/input_freeze.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_root": str(output_root), "templates": 768, "nodes": 9366, "workload_test_episodes": 6750, "locked_test_cells": 135}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
