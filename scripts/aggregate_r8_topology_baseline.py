#!/usr/bin/env python3
"""Aggregate the reproducible evidence for the R8 topology baseline experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


EXPERIMENT_ID = "EXP-20260901_topology_predictor_baseline"
POLICIES = (
    "myopic",
    "predopt_h5",
    "aligned_predopt_h5",
    "aligned_predopt_h5_layer",
    "aligned_trueopt_h5",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def require_status(data: Mapping[str, Any], expected: str, path: Path) -> None:
    if data.get("status") != expected:
        raise ValueError(f"{path} has status {data.get('status')!r}, expected {expected!r}")


def select_policy_metrics(data: Mapping[str, Any], path: Path) -> dict[str, dict[str, Any]]:
    policies = ((data.get("metrics") or {}).get("policies") or {})
    if not isinstance(policies, Mapping):
        raise ValueError(f"{path} has no metrics.policies object")
    missing = [policy for policy in POLICIES if policy not in policies]
    if missing:
        raise ValueError(f"{path} is missing policies: {missing}")
    fields = (
        "chosen_in_strict_feasible_rate",
        "predicted_score_top1_agreement_rate",
        "true_top1_agreement_rate",
        "mean_true_regret_ms",
        "p95_true_regret_ms",
        "mean_spearman_predicted_vs_true",
        "priority_violation_rate",
    )
    return {
        policy: {field: policies[policy].get(field) for field in fields}
        for policy in POLICIES
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--base-artifact-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.experiment_root
    baseline_path = root / "artifacts/prediction_artifacts/baseline_metrics.json"
    layer_path = root / "artifacts/layer_topology_audit.json"
    action_path = root / "artifacts/action_value_audit/metrics.json"
    action_manifest_path = root / "artifacts/action_value_audit/run_manifest.json"
    smoke_path = root / "artifacts/smoke_10/metrics.json"
    baseline = load_json(baseline_path)
    layer = load_json(layer_path)
    action = load_json(action_path)
    action_manifest = load_json(action_manifest_path)
    smoke = load_json(smoke_path)
    templates_sha256 = sha256(args.templates)
    episodes_sha256 = sha256(args.episodes)
    require_status(baseline, "generated", baseline_path)
    require_status(layer, "passed", layer_path)
    require_status(action, "passed", action_path)
    require_status(smoke, "passed", smoke_path)

    if baseline["input"].get("templates_sha256") != templates_sha256:
        raise ValueError("baseline input template hash does not match --templates")
    if action_manifest.get("templates_sha256") != templates_sha256 or action_manifest.get("episodes_sha256") != episodes_sha256:
        raise ValueError("action audit manifest hashes do not match the aggregation inputs")
    if smoke.get("source_templates_sha256") != templates_sha256 or smoke.get("source_episodes_sha256") != episodes_sha256:
        raise ValueError("scheduler smoke input hashes do not match the aggregation inputs")
    base_manifest = args.base_artifact_root / "b05_artifact_manifest.json"
    if baseline["input"].get("base_artifact_manifest_sha256") != sha256(base_manifest):
        raise ValueError("baseline base-artifact manifest hash does not match --base-artifact-root")

    smoke_aggregate = smoke.get("aggregate") or {}
    if sorted(smoke_aggregate) != sorted(POLICIES):
        raise ValueError(f"{smoke_path} policy set does not match {POLICIES}")

    metrics = {
        "schema_version": "r8-topology-predictor-baseline-experiment-v0.1",
        "experiment_id": EXPERIMENT_ID,
        "status": "passed_diagnostic_empirical_topology_baseline",
        "formal_topology_predictor": False,
        "execution_environment": "local_only",
        "remote_used": False,
        "t_final_read": False,
        "inputs": {
            "templates": str(args.templates),
            "templates_sha256": templates_sha256,
            "episodes": str(args.episodes),
            "episodes_sha256": episodes_sha256,
            "base_artifact_root": str(args.base_artifact_root),
            "base_artifact_manifest_sha256": baseline["input"].get("base_artifact_manifest_sha256"),
        },
        "baseline": {
            "status": baseline["status"],
            "predictor_type": baseline["predictor_type"],
            "horizon": baseline["horizon"],
            "max_scenarios": baseline["max_scenarios"],
            "fit": baseline["fit"],
            "prediction": baseline["prediction"],
            "output_contract": baseline["output_contract"],
        },
        "topology_audit": {
            "status": layer["status"],
            "counts": layer["counts"],
            "predicted_topology_sources": layer["predicted_topology_sources"],
            "predicted_layer_width_max": layer["predicted_layer_width_max"],
            "predicted_layer_unit": layer["predicted_layer_unit"],
            "truth_dag_layer_unit": layer["truth_dag_layer_unit"],
            "interpretation": layer["interpretation"],
        },
        "action_value_audit": {
            "status": action["status"],
            "episodes": action["episodes"],
            "decisions": action["decisions"],
            "failed_jobs": action["failed_jobs"],
            "future_horizon": action["future_horizon"],
            "truth_contract": action["truth_contract"],
            "policies": select_policy_metrics(action, action_path),
        },
        "smoke_10": {
            "status": smoke["status"],
            "episodes": smoke["episodes"],
            "result_rows": smoke["result_rows"],
            "expected_result_rows": smoke["expected_result_rows"],
            "aggregate": smoke_aggregate,
        },
        "evidence_sha256": {
            "baseline_metrics": sha256(baseline_path),
            "layer_topology_audit": sha256(layer_path),
            "action_value_audit_metrics": sha256(action_path),
            "smoke_10_metrics": sha256(smoke_path),
        },
        "conclusion": {
            "topology_contract": "Pred and True use the aligned H5 DAG-layer unit through the opt-in layer scorer",
            "baseline_boundary": "conditional empirical S_train DAG-layer labels; no checkpoint, future event, target label, successor identity, or resource truth feature",
            "policy_selection": "do_not_promote",
            "next_scope": "only consider a learned train-only topology predictor after this diagnostic baseline is reviewed; keep formal matrix and T_final sealed",
        },
    }
    output = root / "metrics.json"
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": metrics["status"], "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
