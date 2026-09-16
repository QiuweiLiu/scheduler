#!/usr/bin/env python3
"""Audit whether predicted H5 and truth H5 use the same future topology unit."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


SCHEMA_VERSION = "h5-topology-audit-v0.1"
HORIZON = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_write(handle: Any, row: Mapping[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    handle.write("\n")


def read_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError("non-object JSON at %s:%d" % (path, line_number))
            yield row


def action_key(candidate: Mapping[str, Any]) -> str:
    action = candidate.get("action") or {}
    return "%s|%s|%s" % (
        action.get("job_instance_id"),
        action.get("node_id"),
        action.get("gpu_index"),
    )


def node_id(candidate: Mapping[str, Any]) -> str:
    return str((candidate.get("action") or {}).get("node_id"))


def future_step_count(artifact: Mapping[str, Any]) -> List[int]:
    scenarios = artifact.get("future_h5") or []
    return [len((scenario.get("steps") or [])[:HORIZON]) for scenario in scenarios]


def truth_layers(successors: Mapping[str, Sequence[str]], current: str) -> List[List[str]]:
    frontier = list(successors.get(current, ()))
    seen = set()
    layers: List[List[str]] = []
    for _ in range(HORIZON):
        layer: List[str] = []
        for child in frontier:
            if child in seen:
                continue
            seen.add(child)
            layer.append(child)
        layers.append(layer)
        next_frontier: List[str] = []
        for child in layer:
            next_frontier.extend(successors.get(child, ()))
        frontier = next_frontier
        if not frontier:
            break
    return layers


def distribution(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-records", type=Path, required=True)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--future-artifacts", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--node-output", type=Path, required=True)
    args = parser.parse_args(argv)

    node_info: Dict[str, Mapping[str, Any]] = {}
    successors: Dict[str, List[str]] = defaultdict(list)
    template_for: Dict[str, str] = {}
    for template in read_jsonl(args.templates):
        template_id = str(template.get("template_id"))
        for node in template.get("nodes") or []:
            current = str(node["node_id"])
            if current in node_info and node_info[current] != node:
                raise ValueError("duplicate node id with different contents: %s" % current)
            node_info[current] = node
            template_for[current] = template_id
            for predecessor in node.get("predecessor_node_ids") or []:
                predecessor_id = str(predecessor)
                if current not in successors[predecessor_id]:
                    successors[predecessor_id].append(current)

    artifacts: Dict[str, Mapping[str, Any]] = {}
    with gzip.open(args.future_artifacts, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            current = str(row["node_id"])
            if current in artifacts and artifacts[current] != row:
                raise ValueError("duplicate future artifact with different contents: %s" % current)
            artifacts[current] = row

    occurrences: Counter[str] = Counter()
    models: Dict[str, Counter[str]] = defaultdict(Counter)
    for record in read_jsonl(args.audit_records):
        for candidate in record.get("candidates") or []:
            if not bool(candidate.get("strict_feasible")):
                continue
            current = node_id(candidate)
            occurrences[current] += 1
            models[current][str(candidate.get("model_id"))] += 1

    if not occurrences:
        raise ValueError("no strict-feasible candidate nodes found")

    node_rows: List[Dict[str, Any]] = []
    missing_template = []
    missing_artifact = []
    for current, occurrence_count in sorted(occurrences.items()):
        if current not in node_info:
            missing_template.append(current)
            continue
        artifact = artifacts.get(current)
        if artifact is None:
            missing_artifact.append(current)
            continue
        predicted_counts = future_step_count(artifact)
        predicted_mean = statistics.fmean(predicted_counts) if predicted_counts else 0.0
        layers = truth_layers(successors, current)
        truth_node_count = sum(len(layer) for layer in layers)
        truth_layer_count = len(layers)
        node_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "node_id": current,
                "template_id": template_for.get(current),
                "occurrences": occurrence_count,
                "current_model_ids": dict(sorted(models[current].items())),
                "predicted_future_scenario_count": len(predicted_counts),
                "predicted_future_step_counts": predicted_counts,
                "predicted_future_step_count_mean": predicted_mean,
                "predicted_future_synthetic_rollout": all(
                    bool(scenario.get("synthetic_rollout"))
                    for scenario in (artifact.get("future_h5") or [])
                ),
                "truth_next_h5_layer_count": truth_layer_count,
                "truth_next_h5_dag_node_count": truth_node_count,
                "truth_next_h5_layer_sizes": [len(layer) for layer in layers],
                "truth_dag_nodes_gt_predicted_events": truth_node_count > predicted_mean,
                "truth_dag_nodes_eq_predicted_events": truth_node_count == predicted_mean,
                "truth_dag_nodes_lt_predicted_events": truth_node_count < predicted_mean,
            }
        )

    if missing_template or missing_artifact:
        raise ValueError(
            "missing topology inputs: templates=%d artifacts=%d"
            % (len(missing_template), len(missing_artifact))
        )

    total_occurrences = sum(row["occurrences"] for row in node_rows)
    predicted_step_values = [
        float(row["predicted_future_step_count_mean"]) for row in node_rows
    ]
    truth_node_values = [float(row["truth_next_h5_dag_node_count"]) for row in node_rows]
    truth_layer_values = [float(row["truth_next_h5_layer_count"]) for row in node_rows]
    weighted_predicted = sum(
        row["occurrences"] * row["predicted_future_step_count_mean"] for row in node_rows
    ) / total_occurrences
    weighted_truth_nodes = sum(
        row["occurrences"] * row["truth_next_h5_dag_node_count"] for row in node_rows
    ) / total_occurrences
    weighted_truth_layers = sum(
        row["occurrences"] * row["truth_next_h5_layer_count"] for row in node_rows
    ) / total_occurrences

    by_model: Dict[str, Dict[str, Any]] = {}
    for model_id in sorted({model for row in node_rows for model in row["current_model_ids"]}):
        subset = [row for row in node_rows if model_id in row["current_model_ids"]]
        model_occurrences = sum(row["current_model_ids"].get(model_id, 0) for row in subset)
        undercovered_unique = sum(
            1 for row in subset if row["truth_dag_nodes_gt_predicted_events"]
        )
        undercovered_occurrences = sum(
            row["current_model_ids"].get(model_id, 0)
            for row in subset
            if row["truth_dag_nodes_gt_predicted_events"]
        )
        by_model[model_id] = {
            "unique_candidate_nodes": len(subset),
            "candidate_occurrences": model_occurrences,
            "weighted_mean_truth_next_h5_dag_node_count": sum(
                row["current_model_ids"].get(model_id, 0)
                * row["truth_next_h5_dag_node_count"]
                for row in subset
            )
            / model_occurrences,
            "weighted_mean_predicted_future_event_count": sum(
                row["current_model_ids"].get(model_id, 0)
                * row["predicted_future_step_count_mean"]
                for row in subset
            )
            / model_occurrences,
            "truth_dag_nodes_gt_predicted_events_unique": undercovered_unique,
            "truth_dag_nodes_gt_predicted_events_unique_rate": undercovered_unique / len(subset),
            "truth_dag_nodes_gt_predicted_events_occurrences": undercovered_occurrences,
            "truth_dag_nodes_gt_predicted_events_occurrence_rate": undercovered_occurrences
            / model_occurrences,
        }

    metrics = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "horizon": HORIZON,
        "input": {
            "audit_records": str(args.audit_records),
            "audit_records_sha256": sha256(args.audit_records),
            "templates": str(args.templates),
            "templates_sha256": sha256(args.templates),
            "future_artifacts": str(args.future_artifacts),
            "future_artifacts_sha256": sha256(args.future_artifacts),
        },
        "counts": {
            "unique_candidate_nodes": len(node_rows),
            "candidate_occurrences": total_occurrences,
            "future_artifact_nodes": len(artifacts),
            "template_nodes": len(node_info),
        },
        "predicted_future_event_unit": {
            "scenario_count_distribution": dict(
                sorted(Counter(row["predicted_future_scenario_count"] for row in node_rows).items())
            ),
            "step_count_distribution": dict(
                sorted(Counter(row["predicted_future_step_count_mean"] for row in node_rows).items())
            ),
            "all_scenarios_synthetic_rollout": all(
                row["predicted_future_synthetic_rollout"] for row in node_rows
            ),
            "unique_node_step_count": distribution(predicted_step_values),
            "weighted_mean_step_count": weighted_predicted,
        },
        "truth_future_layer_unit": {
            "layer_count_distribution": dict(
                sorted(Counter(row["truth_next_h5_layer_count"] for row in node_rows).items())
            ),
            "dag_node_count_distribution": dict(
                sorted(Counter(row["truth_next_h5_dag_node_count"] for row in node_rows).items())
            ),
            "unique_node_layer_count": distribution(truth_layer_values),
            "unique_node_dag_node_count": distribution(truth_node_values),
            "weighted_mean_layer_count": weighted_truth_layers,
            "weighted_mean_dag_node_count": weighted_truth_nodes,
        },
        "topology_unit_mismatch": {
            "truth_dag_nodes_gt_predicted_events_unique": sum(
                1 for row in node_rows if row["truth_dag_nodes_gt_predicted_events"]
            ),
            "truth_dag_nodes_gt_predicted_events_unique_rate": sum(
                1 for row in node_rows if row["truth_dag_nodes_gt_predicted_events"]
            )
            / len(node_rows),
            "truth_dag_nodes_gt_predicted_events_occurrences": sum(
                row["occurrences"]
                for row in node_rows
                if row["truth_dag_nodes_gt_predicted_events"]
            ),
            "truth_dag_nodes_gt_predicted_events_occurrence_rate": sum(
                row["occurrences"]
                for row in node_rows
                if row["truth_dag_nodes_gt_predicted_events"]
            )
            / total_occurrences,
            "truth_dag_nodes_eq_predicted_events_unique": sum(
                1 for row in node_rows if row["truth_dag_nodes_eq_predicted_events"]
            ),
            "truth_dag_nodes_lt_predicted_events_unique": sum(
                1 for row in node_rows if row["truth_dag_nodes_lt_predicted_events"]
            ),
            "interpretation": (
                "Predicted future_h5 contains a short synthetic event sequence, while aligned truth "
                "expands the next five DAG layers and can include multiple nodes per layer."
            ),
        },
        "by_current_model": by_model,
        "node_summary_file": str(args.node_output),
    }
    args.node_output.parent.mkdir(parents=True, exist_ok=True)
    with args.node_output.open("w", encoding="utf-8") as handle:
        for row in node_rows:
            jsonl_write(handle, row)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "passed", "counts": metrics["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
