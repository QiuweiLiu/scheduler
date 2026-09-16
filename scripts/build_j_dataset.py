#!/usr/bin/env python3
"""Build the J-series dataset (v1): v3.1 future-slot supervision + measured resources.

Reads the frozen v3 predictor dataset (features + labels) and the R0 measured node
table, then emits one supervision record per anchor with its future H=5 compute
slots and measured resource targets.

Contract highlights (frozen design v3.1):
- composite join key (run_id, node_id); unmatched resource fails closed;
- bounded_future_length = min(len(future_layers), 5) in {0..5};
- termination from label_summary.termination_status (terminated within horizon);
- holdout is excluded by default (explicit flag only, non-confirmatory path);
- model inputs are copied verbatim from the frozen v3 features (only the four
  audited model_input keys) and resources live exclusively under ``targets``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_MODEL_INPUT_KEYS = {"history", "current_node", "task_context", "stack_context"}
SPLITS_DEFAULT = ("train", "validation", "test")


def read_jsonl_gz(path: Path) -> Iterable[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_resource_table(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    table: Dict[Tuple[str, str], Dict[str, Any]] = {}
    duplicates = 0
    for row in read_jsonl_gz(path):
        key = (str(row["run_id"]), str(row["node_id"]))
        if key in table:
            duplicates += 1
            continue
        table[key] = row
    if duplicates:
        raise ValueError(f"resource table composite key is not unique: {duplicates} duplicates")
    return table


def node_slot(label_node: Mapping[str, Any]) -> Dict[str, Any]:
    nested_calls = label_node.get("nested_calls") or []
    nested_model = None
    if nested_calls:
        nested_model = str(nested_calls[0].get("model_id") or "") or None
    return {
        "node_id": str(label_node.get("node_id") or ""),
        "node_type": str(label_node.get("node_type") or "unknown"),
        "raw_action": str(label_node.get("raw_action") or "unknown"),
        "action_family": str(label_node.get("action_family") or "other"),
        "model_class": str(label_node.get("model_id") or "unknown"),
        "role": str(label_node.get("role") or "unknown"),
        "is_retry": bool(label_node.get("is_retry")),
        "merged_nested_call": bool(label_node.get("merged_nested_call")),
        "nested_model_class": nested_model,
        "resource_applicable": bool(label_node.get("resource_applicable")),
        "terminal_marker": bool(label_node.get("terminal_marker")),
    }


def build_split(
    split: str,
    features_path: Path,
    labels_path: Path,
    resources: Mapping[Tuple[str, str], Mapping[str, Any]],
    expected_rows: int,
    audit: Dict[str, Any],
    horizon: int = 5,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen_sample_ids: set = set()
    for feature, label in zip(read_jsonl_gz(features_path), read_jsonl_gz(labels_path)):
        sample_id = str(label.get("sample_id") or "")
        if sample_id != str(feature.get("sample_id") or ""):
            raise ValueError(f"feature/label sample_id mismatch in {split}: {sample_id}")
        if sample_id in seen_sample_ids:
            raise ValueError(f"duplicate sample_id in {split}: {sample_id}")
        seen_sample_ids.add(sample_id)

        run_id = str(label.get("run_id") or "")
        model_input = feature.get("model_input") or {}
        extra_keys = set(model_input) - ALLOWED_MODEL_INPUT_KEYS
        if extra_keys:
            raise ValueError(f"unexpected model_input keys for {sample_id}: {sorted(extra_keys)}")

        future: List[Dict[str, Any]] = []
        for layer in label.get("future_layers") or []:
            for label_node in layer.get("nodes") or []:
                slot = node_slot(label_node)
                key = (run_id, slot["node_id"])
                resource = resources.get(key)
                if resource is None:
                    raise ValueError(f"unmatched resource node (fail-closed): {key}")
                if resource.get("runtime_ms") is None:
                    raise ValueError(f"resource node missing runtime_ms: {key}")
                slot["targets"] = {
                    "runtime_ms": resource.get("runtime_ms"),
                    "load_ms": resource.get("load_ms"),
                    "peak_memory_inclusive_mb": resource.get("peak_memory_inclusive_mb"),
                }
                future.append(slot)

        summary = label.get("label_summary") or {}
        bounded_length = len(future)
        termination = 1 if str(summary.get("termination_status")) == "terminated" else 0
        if not 0 <= bounded_length <= int(horizon):
            raise ValueError(f"bounded_future_length out of range for {sample_id}: {bounded_length}")

        row = {
            "schema_version": "j-series-dataset-v1",
            "sample_id": sample_id,
            "split": split,
            "run_id": run_id,
            "video_id": str(label.get("video_id") or ""),
            "registry_video_id": str(label.get("registry_video_id") or ""),
            "current_node_id": str(label.get("current_node_id") or ""),
            "prefix_hash": str(feature.get("prefix_hash") or ""),
            "source_trace_sha256": str(feature.get("source_trace_sha256") or ""),
            "model_input": model_input,
            "future": future,
            "bounded_future_length": bounded_length,
            "termination": termination,
        }
        rows.append(row)

        audit["slots_by_split"][split] += len(future)
        audit["length_histogram"][str(bounded_length)] += 1
        audit["termination_by_split"][split] += termination
        for slot in future:
            audit["resource_nodes_referenced"][(run_id, slot["node_id"])] = True
            load = slot["targets"]["load_ms"]
            if load is None:
                audit["load_missing"] += 1
            elif float(load) == 0.0:
                audit["load_zero"] += 1
            else:
                audit["load_positive"] += 1
            if slot["targets"]["peak_memory_inclusive_mb"] is not None:
                audit["memory_available"] += 1
            else:
                audit["memory_missing"] += 1

    if len(rows) != expected_rows:
        raise ValueError(f"{split} rows {len(rows)} != expected {expected_rows}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--include-nonconfirmatory-holdout", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    dataset_root = Path(config["dataset_root"])
    output_root = Path(config["output_root"])
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    splits: List[str] = list(SPLITS_DEFAULT)
    nonconfirmatory_holdout = False
    if args.include_nonconfirmatory_holdout:
        splits.append("holdout")
        nonconfirmatory_holdout = True

    resources = load_resource_table(Path(config["resource_table"]))
    expected = config.get("expected", {})
    horizon = int(config.get("horizon", 5))
    audit: Dict[str, Any] = {
        "resource_table_unique_keys": len(resources),
        "expected_unique_resource_nodes": expected.get("r0_unique_resource_nodes"),
        "rows_by_split": Counter(),
        "slots_by_split": Counter(),
        "termination_by_split": Counter(),
        "length_histogram": Counter(),
        "resource_nodes_referenced": {},
        "load_zero": 0,
        "load_positive": 0,
        "load_missing": 0,
        "memory_available": 0,
        "memory_missing": 0,
        "nonconfirmatory_holdout_included": nonconfirmatory_holdout,
        "splits_built": splits,
    }

    started = time.time()
    outputs: Dict[str, str] = {}
    for split in splits:
        features_path = dataset_root / f"features_{split}.jsonl.gz"
        labels_path = dataset_root / f"labels_{split}.jsonl.gz"
        if not features_path.is_file() or not labels_path.is_file():
            raise FileNotFoundError(f"missing v3 files for split {split}")
        expected_rows = int(expected.get("rows_by_split", {}).get(split, -1))
        if expected_rows < 0:
            expected_rows = sum(1 for _ in read_jsonl_gz(labels_path))
        rows = build_split(split, features_path, labels_path, resources, expected_rows, audit, horizon)
        audit["rows_by_split"][split] = len(rows)
        out_path = f"j_{split}.jsonl.gz"
        write_jsonl_gz(output_root / out_path, rows)
        outputs[split] = out_path

    referenced = len(audit["resource_nodes_referenced"])
    audit["unique_resource_nodes_referenced"] = referenced
    audit["supervision_instances_total"] = sum(audit["slots_by_split"].values())
    audit["length_histogram"] = dict(sorted(audit["length_histogram"].items(), key=lambda item: int(item[0])))
    audit_summary = dict(audit)
    audit_summary.pop("resource_nodes_referenced", None)

    if not 0 <= audit["load_zero"] + audit["load_positive"] + audit["load_missing"]:
        raise ValueError("load accounting impossible")
    if audit["load_zero"] + audit["load_positive"] + audit["load_missing"] != audit["supervision_instances_total"]:
        raise ValueError("load accounting does not cover all supervision slots")

    manifest = {
        "schema_version": "j-series-dataset-v1",
        "status": "generated",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "experiment_config": str(args.config),
        "dataset_root": str(dataset_root),
        "resource_table": str(config["resource_table"]),
        "resource_table_sha256": sha256_file(Path(config["resource_table"])),
        "outputs": outputs,
        "rows_by_split": {split: audit["rows_by_split"][split] for split in splits},
        "slots_by_split": {split: audit["slots_by_split"][split] for split in splits},
        "unique_resource_nodes_referenced": referenced,
        "supervision_instances_total": audit["supervision_instances_total"],
        "length_histogram": audit["length_histogram"],
        "termination_by_split": {split: audit["termination_by_split"][split] for split in splits},
        "load_accounting": {"zero": audit["load_zero"], "positive": audit["load_positive"], "missing": audit["load_missing"]},
        "memory_accounting": {"available": audit["memory_available"], "missing": audit["memory_missing"]},
        "nonconfirmatory_holdout_included": nonconfirmatory_holdout,
        "runtime_seconds": round(time.time() - started, 2),
    }

    write_json(output_root / "stage0_audit.json", audit_summary)
    write_json(output_root / "dataset_manifest.json", manifest)
    print(json.dumps({
        "ok": True,
        "rows_by_split": manifest["rows_by_split"],
        "slots_by_split": manifest["slots_by_split"],
        "unique_resource_nodes_referenced": referenced,
        "length_histogram": manifest["length_histogram"],
        "load_accounting": manifest["load_accounting"],
        "output_root": str(output_root),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
