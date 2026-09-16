#!/usr/bin/env python3
"""Audit predicted layer widths against the privileged DAG-layer reference."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tracing.scheduling.future_topology import layer_node_count, validate_layer_scenarios


HORIZON = 5
SCHEMA_VERSION = "h5-layer-contract-audit-v0.1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}:{line_number} is not an object")
            yield row


def read_gzip_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}:{line_number} is not an object")
            yield row


def truth_layers(successors: Mapping[str, Sequence[str]], current: str) -> list[list[str]]:
    frontier = list(successors.get(current, ()))
    seen: set[str] = set()
    layers: list[list[str]] = []
    for _ in range(HORIZON):
        layer: list[str] = []
        for child in frontier:
            if child in seen:
                continue
            seen.add(child)
            layer.append(child)
        if not layer:
            break
        layers.append(layer)
        next_frontier: list[str] = []
        for child in layer:
            next_frontier.extend(successors.get(child, ()))
        frontier = next_frontier
    return layers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-records", type=Path, required=True)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--future-artifacts", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    successors: dict[str, list[str]] = {}
    for template in read_jsonl(args.templates):
        local_successors: dict[str, list[str]] = {str(node["node_id"]): [] for node in template.get("nodes") or []}
        for node in template.get("nodes") or []:
            current = str(node["node_id"])
            for predecessor in node.get("predecessor_node_ids") or []:
                local_successors.setdefault(str(predecessor), []).append(current)
        successors.update(local_successors)

    artifacts: dict[str, Mapping[str, Any]] = {}
    path = args.future_artifacts / "b05_future_h5_layers.jsonl.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    for row in read_gzip_jsonl(path):
        node_id = str(row.get("node_id") or "")
        if not node_id:
            raise ValueError(f"layer artifact row without node_id: {path}")
        artifacts[node_id] = row

    topology_sources = sorted(
        {
            str(
                (row.get("input_contract") or {}).get("topology_source")
                or next(
                    (
                        scenario.get("topology_source")
                        for scenario in row.get("future_h5_layers") or []
                        if scenario.get("topology_source")
                    ),
                    "unknown",
                )
            )
            for row in artifacts.values()
        }
    )

    occurrences: Counter[str] = Counter()
    for record in read_jsonl(args.audit_records):
        for candidate in record.get("candidates") or []:
            if bool(candidate.get("strict_feasible")):
                node_id = str((candidate.get("action") or {}).get("node_id") or "")
                if node_id:
                    occurrences[node_id] += 1
    if not occurrences:
        raise ValueError("no strict-feasible candidate occurrences")

    rows: list[dict[str, Any]] = []
    for node_id, count in sorted(occurrences.items()):
        artifact = artifacts.get(node_id)
        if artifact is None:
            raise ValueError(f"missing layer artifact for {node_id}")
        scenarios = artifact.get("future_h5_layers") or []
        validate_layer_scenarios(scenarios, HORIZON)
        predicted_layer_counts = [len(scenario.get("layers") or []) for scenario in scenarios]
        predicted_node_counts = layer_node_count(scenarios, HORIZON)
        truth = truth_layers(successors, node_id)
        truth_widths = [len(layer) for layer in truth]
        rows.append(
            {
                "node_id": node_id,
                "occurrences": count,
                "predicted_layer_count_mean": statistics.fmean(predicted_layer_counts) if predicted_layer_counts else 0.0,
                "predicted_node_count_mean": statistics.fmean(predicted_node_counts) if predicted_node_counts else 0.0,
                "predicted_layer_widths": sorted({
                    len(layer.get("nodes") or [])
                    for scenario in scenarios
                    for layer in scenario.get("layers") or []
                }),
                "truth_layer_count": len(truth),
                "truth_node_count": sum(truth_widths),
                "truth_layer_widths": truth_widths,
            }
        )

    total_occurrences = sum(row["occurrences"] for row in rows)
    weighted = lambda key: sum(row["occurrences"] * float(row[key]) for row in rows) / total_occurrences
    predicted_width_max = max(
        (max(row["predicted_layer_widths"]) for row in rows if row["predicted_layer_widths"]),
        default=0,
    )
    if topology_sources == ["legacy_event_to_unary_layer_projection"] and predicted_width_max <= 1:
        interpretation = (
            "The schema supports multi-node layers, but this artifact is a unary projection. "
            "A nonzero predicted-versus-truth width gap measures missing topology prediction, "
            "not a completed topology-model result."
        )
    elif topology_sources == ["conditional_empirical_s_train_dag_layers"]:
        interpretation = (
            "This artifact is a conditional empirical multi-node layer baseline fit on the "
            "r7_s_train DAG labels. The width gap is diagnostic prediction error for this baseline; "
            "it is not evidence of a learned topology model or formal policy improvement."
        )
    else:
        interpretation = (
            "The layer schema supports multi-node layers. This audit reports the topology source "
            "and predicted-versus-truth width gap as diagnostic evidence; it does not by itself "
            "establish a learned topology model or formal policy improvement."
        )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "horizon": HORIZON,
        "audit_only": True,
        "input": {
            "audit_records": str(args.audit_records),
            "audit_records_sha256": sha256(args.audit_records),
            "templates": str(args.templates),
            "templates_sha256": sha256(args.templates),
            "future_layer_artifacts": str(path),
            "future_layer_artifacts_sha256": sha256(path),
        },
        "counts": {
            "unique_candidate_nodes": len(rows),
            "candidate_occurrences": total_occurrences,
            "layer_artifact_nodes": len(artifacts),
        },
        "predicted_topology_sources": topology_sources,
        "predicted_layer_width_max": predicted_width_max,
        "predicted_layer_unit": {
            "layer_count_distribution": dict(sorted(Counter(row["predicted_layer_count_mean"] for row in rows).items())),
            "node_count_distribution": dict(sorted(Counter(row["predicted_node_count_mean"] for row in rows).items())),
            "weighted_mean_layer_count": weighted("predicted_layer_count_mean"),
            "weighted_mean_node_count": weighted("predicted_node_count_mean"),
        },
        "truth_dag_layer_unit": {
            "layer_count_distribution": dict(sorted(Counter(row["truth_layer_count"] for row in rows).items())),
            "node_count_distribution": dict(sorted(Counter(row["truth_node_count"] for row in rows).items())),
            "weighted_mean_layer_count": weighted("truth_layer_count"),
            "weighted_mean_node_count": weighted("truth_node_count"),
        },
        "interpretation": interpretation,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": metrics["status"], "counts": metrics["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
