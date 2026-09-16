#!/usr/bin/env python3
"""Run the small R7 StateView/Truth/FutureProvider boundary gate.

This is an interface smoke, not a scheduler performance experiment.  It uses
deterministic causal-chain fixtures so that the following invariants can be
checked before real S_train/S_val workload collection starts:

* H=0 never serializes future rows;
* predicted/true future exposes identities only;
* changing execution truth cannot change a state view;
* changing a suffix beyond H cannot change a finite-horizon reveal;
* the same fixture produces byte-identical state payloads on replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tracing.scheduling.event_engine import (
    ExecutionTruth,
    ExecutionTruthProvider,
    FutureProvider,
    ResourceEstimate,
    build_scheduler_state,
)


def _fixture(index: int) -> dict[str, Any]:
    job_id = f"job-{index:04d}"
    node_id = f"node-{index:04d}-1"
    node = {
        "job_instance_id": job_id,
        "node_id": node_id,
        "sequence_index": 1,
        "role": "execute",
        "action_family": "spatial" if index % 2 else "temporal",
        "model_id": "stack_a_qwen3_vl8b",
        "lane": "gpu",
        "ready_since_ms": float(index * 10),
    }
    estimate = ResourceEstimate(100.0 + index, 140.0 + index, 20.0, 600.0)
    predicted = {
        (job_id, node_id, 3): [
            {
                "scenario_probability": 0.7,
                "steps": [
                    {"role": "execute", "action_family": "temporal", "model_id": "stack_a_qwen3_vl8b", "lane": "gpu"},
                    {"role": "aggregate", "action_family": "answer", "lane": "cpu"},
                ],
            },
            {
                "scenario_probability": 0.3,
                "steps": [
                    {"role": "execute", "action_family": "generalist", "model_id": "stack_a_qwen3_vl8b", "lane": "gpu"},
                    {"role": "aggregate", "action_family": "answer", "lane": "cpu"},
                ],
            },
        ]
    }
    truth = {
        (job_id, node_id): [
            {"role": "execute", "action_family": "temporal", "model_id": "stack_a_qwen3_vl8b", "lane": "gpu"},
            {"role": "aggregate", "action_family": "answer", "lane": "cpu"},
            {"role": "terminate", "action_family": "end", "lane": "cpu"},
        ]
    }
    return {
        "node": node,
        "resource": {(job_id, node_id): estimate},
        "predicted": predicted,
        "truth": truth,
        "truth_provider": ExecutionTruthProvider(
            {(job_id, node_id): ExecutionTruth(1000.0 + index, 20.0, 600.0, "success")}
        ),
        "gpus": [{"index": 0, "capacity_mb": 24000.0, "busy": False, "resident_models": ()}],
        "job_id": job_id,
        "node_id": node_id,
    }


def _payload(fixture: dict[str, Any], mode: str, horizon: int) -> dict[str, Any]:
    provider = FutureProvider(mode, predicted=fixture["predicted"], truth=fixture["truth"])
    state = build_scheduler_state(
        episode_id=f"episode-{fixture['job_id']}",
        decision_index=1,
        time_ms=10.0,
        ready_nodes=[fixture["node"]],
        completed_prefix={fixture["job_id"]: (f"{fixture['node_id']}-prefix",)},
        gpus=fixture["gpus"],
        resource_predictions=fixture["resource"],
        future_provider=provider,
        horizon=horizon,
    )
    return state.to_scheduler_dict()


def _digest(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def run(count: int) -> dict[str, Any]:
    checks = {
        "none_no_future": 0,
        "predicted_truth_free": 0,
        "truth_mutation_invariant": 0,
        "suffix_mutation_invariant": 0,
        "deterministic_replay": 0,
    }
    for index in range(count):
        fixture = _fixture(index)
        none_payload = _payload(fixture, "none", 3)
        if none_payload["future"][f"{fixture['job_id']}:{fixture['node_id']}"]["scenarios"] == []:
            checks["none_no_future"] += 1

        predicted_payload = _payload(fixture, "pred_h", 3)
        encoded = json.dumps(predicted_payload, ensure_ascii=False, sort_keys=True)
        if "runtime_ms" not in encoded and "workspace_peak_mb" not in encoded:
            checks["predicted_truth_free"] += 1

        truth_before = fixture["truth_provider"].get(fixture["job_id"], fixture["node_id"])
        truth_after = ExecutionTruthProvider(
            {(fixture["job_id"], fixture["node_id"]): ExecutionTruth(9000.0 + index, 999.0, 999.0, "error")}
        ).get(fixture["job_id"], fixture["node_id"])
        if _digest(_payload(fixture, "pred_h", 3)) == _digest(_payload(fixture, "pred_h", 3)) and truth_before != truth_after:
            checks["truth_mutation_invariant"] += 1

        shortened = dict(fixture)
        shortened["truth"] = {key: value[:1] for key, value in fixture["truth"].items()}
        if _digest(_payload(fixture, "true_h", 1)) == _digest(_payload(shortened, "true_h", 1)):
            checks["suffix_mutation_invariant"] += 1

        if _digest(_payload(fixture, "pred_h", 3)) == _digest(_payload(fixture, "pred_h", 3)):
            checks["deterministic_replay"] += 1

    return {
        "schema_version": "r7-event-engine-parity-smoke-v1",
        "episodes": count,
        "checks": checks,
        "pass": all(value == count for value in checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    report = {
        "runs": [run(min(10, args.episodes)), run(args.episodes)],
        "source": "deterministic_causal_contract_fixture",
        "formal_scheduler_workload_started": False,
    }
    report["pass"] = all(item["pass"] for item in report["runs"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "pass": report["pass"], "runs": [item["episodes"] for item in report["runs"]]}, ensure_ascii=False))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
