#!/usr/bin/env python3
"""Build a local opt-in layer-H5 sidecar from frozen legacy artifacts.

The adapter is intentionally limited to a unary projection.  It changes the
representation from five event positions to five ordered layers, but it does
not claim to predict parallel DAG width.  A later topology predictor can
replace the sidecar generation while keeping the same schema.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from tracing.analysis.workload_v02_simulator import read_gzip_jsonl
from tracing.scheduling.future_topology import (
    LAYER_H5_SCHEMA_VERSION,
    layer_node_count,
    project_event_scenarios_to_unary_layers,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_gzip_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def build(input_root: Path, output_root: Path, limit: int = 0) -> dict[str, Any]:
    required = ("b05_node_h1.jsonl.gz", "b05_future_h3.jsonl.gz", "b05_future_h5.jsonl.gz")
    for filename in required:
        path = input_root / filename
        if not path.is_file():
            raise FileNotFoundError(path)

    source_rows = read_gzip_jsonl(input_root / "b05_future_h5.jsonl.gz")
    if limit:
        source_rows = source_rows[:limit]
    if not source_rows:
        raise ValueError("legacy future_h5 artifact is empty")

    output_root.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for filename in required:
        source = input_root / filename
        target = output_root / filename
        shutil.copy2(source, target)
        copied[filename] = sha256(target)

    layer_rows: list[dict[str, Any]] = []
    node_counts: list[int] = []
    for row in source_rows:
        scenarios = project_event_scenarios_to_unary_layers(row.get("future_h5") or [], 5)
        node_counts.extend(layer_node_count(scenarios, 5))
        layer_rows.append(
            {
                "schema_version": LAYER_H5_SCHEMA_VERSION,
                "template_id": row.get("template_id"),
                "node_id": row.get("node_id"),
                "future_h5_layers": scenarios,
                "input_contract": {
                    "fit_split": "P_dev_train_only",
                    "future_events_excluded": True,
                    "target_labels_excluded": True,
                    "observed_resource_targets_excluded": True,
                    "synthetic_rollout": True,
                    "topology_source": "legacy_event_to_unary_layer_projection",
                    "predicted_parallel_width": "fixed_one_not_a_topology_predictor",
                },
            }
        )
    layer_path = output_root / "b05_future_h5_layers.jsonl.gz"
    layer_rows_written = write_gzip_jsonl(layer_path, layer_rows)
    source_manifest = input_root / "b05_artifact_manifest.json"
    artifact_manifest = {
        "schema_version": "scheduling-future-b05-layer-repair-manifest-v1",
        "base_artifact_manifest_sha256": sha256(source_manifest) if source_manifest.is_file() else None,
        "legacy_files_copied_unchanged": True,
        "layer_file": str(layer_path),
        "layer_schema_version": LAYER_H5_SCHEMA_VERSION,
        "topology_source": "legacy_event_to_unary_layer_projection",
        "formal_topology_predictor": False,
        "source_h5_sha256": sha256(input_root / "b05_future_h5.jsonl.gz"),
    }
    (output_root / "b05_artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "h5-layer-contract-repair-manifest-v0.1",
        "layer_schema_version": LAYER_H5_SCHEMA_VERSION,
        "topology_source": "legacy_event_to_unary_layer_projection",
        "formal_topology_predictor": False,
        "source_root": str(input_root),
        "source_files": {
            filename: {
                "path": str(input_root / filename),
                "sha256": sha256(input_root / filename),
            }
            for filename in required
        },
        "output_files": {
            **copied,
            "b05_future_h5_layers.jsonl.gz": sha256(layer_path),
            "b05_artifact_manifest.json": sha256(output_root / "b05_artifact_manifest.json"),
        },
        "source_rows": len(source_rows),
        "layer_rows": layer_rows_written,
        "scenario_count": sum(len(row["future_h5_layers"]) for row in layer_rows),
        "predicted_node_count": {
            "mean": sum(node_counts) / len(node_counts) if node_counts else 0.0,
            "min": min(node_counts) if node_counts else 0,
            "max": max(node_counts) if node_counts else 0,
        },
        "input_contract": {
            "future_events_used_as_features": False,
            "target_template_suffix_used": False,
            "target_node_ids_copied": False,
            "resource_truth_used_as_feature": False,
        },
    }
    (output_root / "h5_layer_contract_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    manifest = build(args.input_root, args.output_root, args.limit)
    print(json.dumps({"rows": manifest["layer_rows"], "output_root": str(args.output_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
