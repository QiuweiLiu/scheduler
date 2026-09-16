#!/usr/bin/env python3
"""Build an identity-free empirical H5 DAG-layer predictor.

The predictor fits a conditional distribution of successor-layer prototypes
from the R7 ``r7_s_train`` templates and applies it to ``r7_s_val`` templates.
It deliberately uses no model checkpoint and does not copy target successor
identities, edges, runtime, load, or memory into the predicted artifact.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tracing.scheduling.future_topology import (
    LAYER_H5_SCHEMA_VERSION,
    layer_node_count,
    validate_layer_scenarios,
)


HORIZON = 5
MAX_SCENARIOS = 3
FIT_TASK_PREFIX = "r7_s_train_"
PREDICT_TASK_PREFIX = "r7_s_val_"
KEY_FIELDS = (
    "model_stack_id",
    "baseline",
    "node_type",
    "raw_action",
    "model_id",
    "execution_lane",
)
PROTOTYPE_FIELDS = (
    "node_type",
    "raw_action",
    "model_id",
    "execution_lane",
    "action_family",
)
FORBIDDEN_PREDICTED_FIELDS = {
    "node_id",
    "successors",
    "predecessors",
    "successor_node_ids",
    "predecessor_node_ids",
    "source_event_ids",
    "source_step_ids",
    "runtime_ms",
    "load_ms",
    "workspace_peak_mb",
    "resident_model_mb",
    "gpu_id",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: Any, default: str = "unknown") -> str:
    text = str(value or "").strip()
    return text or default


def template_split(row: Mapping[str, Any], fit_prefix: str, predict_prefix: str) -> str:
    task_id = _text(row.get("task_id"), "")
    if task_id.startswith(fit_prefix):
        return "r7_s_train"
    if task_id.startswith(predict_prefix):
        return "r7_s_val"
    return "other"


def _node_index(row: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for node in row.get("nodes") or []:
        if not isinstance(node, Mapping):
            raise ValueError(f"template {row.get('template_id')} contains a non-object node")
        node_id = _text(node.get("node_id"), "")
        if not node_id or node_id in result:
            raise ValueError(f"template {row.get('template_id')} has duplicate/empty node_id")
        result[node_id] = node
    if not result:
        raise ValueError(f"template {row.get('template_id')} has no nodes")
    return result


def _successors(row: Mapping[str, Any], nodes: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    successors: dict[str, list[str]] = defaultdict(list)
    for node_id in nodes:
        successors.setdefault(node_id, [])
    for node in nodes.values():
        node_id = _text(node.get("node_id"), "")
        for predecessor in node.get("predecessor_node_ids") or []:
            predecessor_id = _text(predecessor, "")
            if predecessor_id not in nodes:
                raise ValueError(
                    f"template {row.get('template_id')} references missing predecessor {predecessor_id}"
                )
            successors[predecessor_id].append(node_id)
    for node_id, children in successors.items():
        children.sort(key=lambda value: (int(nodes[value].get("sequence_index") or 0), value))
    return dict(successors)


def future_layers(
    row: Mapping[str, Any],
    node_id: str,
    horizon: int = HORIZON,
) -> list[list[Mapping[str, Any]]]:
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    nodes = _node_index(row)
    if node_id not in nodes:
        raise ValueError(f"template {row.get('template_id')} has no node {node_id}")
    successors = _successors(row, nodes)
    frontier = list(successors.get(node_id, ()))
    seen: set[str] = set()
    layers: list[list[Mapping[str, Any]]] = []
    for _ in range(horizon):
        layer: list[Mapping[str, Any]] = []
        for child in frontier:
            if child in seen:
                continue
            seen.add(child)
            layer.append(nodes[child])
        if not layer:
            break
        layer.sort(key=lambda node: (int(node.get("sequence_index") or 0), _text(node.get("node_id"), "")))
        layers.append(layer)
        frontier = [
            grandchild
            for child in layer
            for grandchild in successors.get(_text(child.get("node_id"), ""), ())
        ]
    return layers


def action_family(node: Mapping[str, Any]) -> str:
    activity = _text(node.get("activity"), "other")
    if activity != "other":
        return activity
    return _text(node.get("raw_action"), "other")


def condition_key(row: Mapping[str, Any], node: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for field in KEY_FIELDS:
        source = row.get(field) if field in {"model_stack_id", "baseline"} else node.get(field)
        values.append(_text(source))
    return tuple(values)


def prototype_tuple(node: Mapping[str, Any]) -> tuple[str, ...]:
    values = {
        "node_type": _text(node.get("node_type")),
        "raw_action": _text(node.get("raw_action")),
        "model_id": _text(node.get("model_id")),
        "execution_lane": _text(node.get("execution_lane")),
        "action_family": action_family(node),
    }
    return tuple(values[field] for field in PROTOTYPE_FIELDS)


Signature = tuple[tuple[tuple[str, ...], ...], ...]
ConditionKey = tuple[str, ...]


def topology_signature(row: Mapping[str, Any], node_id: str, horizon: int = HORIZON) -> Signature:
    return tuple(
        tuple(prototype_tuple(node) for node in layer)
        for layer in future_layers(row, node_id, horizon)
    )


def split_templates(
    rows: Sequence[Mapping[str, Any]],
    fit_prefix: str = FIT_TASK_PREFIX,
    predict_prefix: str = PREDICT_TASK_PREFIX,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    fit: list[Mapping[str, Any]] = []
    predict: list[Mapping[str, Any]] = []
    other: list[str] = []
    for row in rows:
        split = template_split(row, fit_prefix, predict_prefix)
        if split == "r7_s_train":
            fit.append(row)
        elif split == "r7_s_val":
            predict.append(row)
        else:
            other.append(_text(row.get("task_id"), "<missing-task-id>"))
    if other:
        raise ValueError(f"unclassified template task_id values: {other[:5]!r}")
    if not fit or not predict:
        raise ValueError("both r7_s_train and r7_s_val template partitions are required")
    return fit, predict


def fit_empirical_models(
    fit_rows: Sequence[Mapping[str, Any]],
    horizon: int = HORIZON,
) -> dict[ConditionKey, Counter[Signature]]:
    models: dict[ConditionKey, Counter[Signature]] = defaultdict(Counter)
    for row in fit_rows:
        nodes = _node_index(row)
        for node_id, node in nodes.items():
            models[condition_key(row, node)][topology_signature(row, node_id, horizon)] += 1
    return dict(models)


def _signature_sort_key(item: tuple[Signature, int]) -> tuple[int, str]:
    signature, count = item
    return (-count, json.dumps(signature, ensure_ascii=False, separators=(",", ":")))


def signature_layers(signature: Signature, source: str) -> list[dict[str, Any]]:
    layers: list[dict[str, Any]] = []
    for layer_offset, layer in enumerate(signature, 1):
        nodes: list[dict[str, Any]] = []
        for predicted_node_index, values in enumerate(layer):
            prototype = dict(zip(PROTOTYPE_FIELDS, values))
            prototype.update(
                {
                    "layer_offset": layer_offset,
                    "predicted_node_index": predicted_node_index,
                    "prototype_source": source,
                }
            )
            nodes.append(prototype)
        layers.append({"layer_offset": layer_offset, "nodes": nodes})
    return layers


def predict_scenarios(
    key: ConditionKey,
    models: Mapping[ConditionKey, Counter[Signature]],
    max_scenarios: int = MAX_SCENARIOS,
) -> list[dict[str, Any]]:
    if max_scenarios < 1:
        raise ValueError("max_scenarios must be positive")
    counter = models.get(key)
    if not counter:
        raise ValueError(f"no empirical topology model for condition key {key!r}")
    selected = sorted(counter.items(), key=_signature_sort_key)[:max_scenarios]
    denominator = sum(count for _signature, count in selected)
    scenarios: list[dict[str, Any]] = []
    for rank, (signature, count) in enumerate(selected):
        scenarios.append(
            {
                "scenario_id": f"empirical_top{rank:02d}",
                "scenario_probability": float(count) / float(denominator),
                "layers": signature_layers(signature, "conditional_empirical_s_train_dag_layers"),
                "synthetic_rollout": True,
                "topology_source": "conditional_empirical_s_train_dag_layers",
                "support_count": int(count),
            }
        )
    validate_layer_scenarios(scenarios, HORIZON)
    return scenarios


def _assert_identity_free(scenarios: Sequence[Mapping[str, Any]]) -> None:
    for scenario in scenarios:
        for layer in scenario.get("layers") or []:
            for node in layer.get("nodes") or []:
                leaked = sorted(FORBIDDEN_PREDICTED_FIELDS.intersection(node))
                if leaked:
                    raise ValueError(f"empirical topology prediction leaks fields: {leaked}")


def _write_gzip_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def build(
    templates_path: Path,
    base_artifact_root: Path,
    output_root: Path,
    horizon: int = HORIZON,
    max_scenarios: int = MAX_SCENARIOS,
    fit_prefix: str = FIT_TASK_PREFIX,
    predict_prefix: str = PREDICT_TASK_PREFIX,
    expected_fit_templates: int | None = 480,
    expected_predict_templates: int | None = 160,
) -> dict[str, Any]:
    if horizon != HORIZON:
        raise ValueError(f"this baseline is defined for H={HORIZON}, got {horizon}")
    rows = read_jsonl(templates_path)
    fit_rows, predict_rows = split_templates(rows, fit_prefix, predict_prefix)
    if expected_fit_templates is not None and len(fit_rows) != expected_fit_templates:
        raise ValueError(f"expected {expected_fit_templates} fit templates, got {len(fit_rows)}")
    if expected_predict_templates is not None and len(predict_rows) != expected_predict_templates:
        raise ValueError(f"expected {expected_predict_templates} prediction templates, got {len(predict_rows)}")

    models = fit_empirical_models(fit_rows, horizon)
    output_root.mkdir(parents=True, exist_ok=True)
    required = ("b05_node_h1.jsonl.gz", "b05_future_h3.jsonl.gz", "b05_future_h5.jsonl.gz")
    copied: dict[str, str] = {}
    for filename in required:
        source = base_artifact_root / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        target = output_root / filename
        shutil.copy2(source, target)
        copied[filename] = sha256(target)

    layer_rows: list[dict[str, Any]] = []
    predicted_layer_counts: list[int] = []
    predicted_node_counts: list[int] = []
    unseen_keys: list[ConditionKey] = []
    for row in predict_rows:
        nodes = _node_index(row)
        for node_id, node in nodes.items():
            key = condition_key(row, node)
            if key not in models:
                unseen_keys.append(key)
                continue
            scenarios = predict_scenarios(key, models, max_scenarios)
            _assert_identity_free(scenarios)
            predicted_layer_counts.extend(len(scenario.get("layers") or []) for scenario in scenarios)
            predicted_node_counts.extend(layer_node_count(scenarios, horizon))
            layer_rows.append(
                {
                    "schema_version": LAYER_H5_SCHEMA_VERSION,
                    "template_id": row.get("template_id"),
                    "node_id": node_id,
                    "conditioning_key": dict(zip(KEY_FIELDS, key)),
                    "future_h5_layers": scenarios,
                    "input_contract": {
                        "fit_split": "r7_s_train_template_dag_labels",
                        "predict_split": "r7_s_val_templates",
                        "future_events_used_as_features": False,
                        "target_labels_used_as_features": False,
                        "target_successor_ids_copied": False,
                        "target_successor_edges_copied": False,
                        "resource_truth_used_as_feature": False,
                        "model_checkpoint_used": False,
                        "topology_source": "conditional_empirical_s_train_dag_layers",
                    },
                }
            )
    if unseen_keys:
        raise ValueError(f"prediction partition contains unseen condition keys: {unseen_keys[:5]!r}")

    layer_path = output_root / "b05_future_h5_layers.jsonl.gz"
    layer_rows_written = _write_gzip_jsonl(layer_path, layer_rows)
    base_manifest = base_artifact_root / "b05_artifact_manifest.json"
    overlay_manifest = {
        "schema_version": "scheduling-future-empirical-topology-overlay-v1",
        "base_artifact_manifest_sha256": sha256(base_manifest) if base_manifest.is_file() else None,
        "legacy_files_copied_unchanged": True,
        "layer_schema_version": LAYER_H5_SCHEMA_VERSION,
        "topology_source": "conditional_empirical_s_train_dag_layers",
        "predictor_type": "conditional_empirical_baseline",
        "formal_topology_predictor": False,
        "fit_task_prefix": fit_prefix,
        "predict_task_prefix": predict_prefix,
        "fit_templates": len(fit_rows),
        "predict_templates": len(predict_rows),
        "fit_condition_keys": len(models),
        "prediction_nodes": len(layer_rows),
        "source_templates_sha256": sha256(templates_path),
    }
    (output_root / "b05_artifact_manifest.json").write_text(
        json.dumps(overlay_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fit_node_count = sum(len(_node_index(row)) for row in fit_rows)
    predict_video_count = len({_text(row.get("video_id"), "") for row in predict_rows})
    fit_video_count = len({_text(row.get("video_id"), "") for row in fit_rows})
    support_counts = [sum(counter.values()) for counter in models.values()]
    metrics = {
        "schema_version": "r8-topology-predictor-baseline-v0.1",
        "status": "generated",
        "formal_topology_predictor": False,
        "predictor_type": "conditional_empirical_baseline",
        "horizon": horizon,
        "max_scenarios": max_scenarios,
        "input": {
            "templates": str(templates_path),
            "templates_sha256": sha256(templates_path),
            "base_artifact_root": str(base_artifact_root),
            "base_artifact_manifest_sha256": sha256(base_manifest) if base_manifest.is_file() else None,
            "model_checkpoints_used": False,
            "future_events_used_as_features": False,
            "target_successor_ids_copied": False,
            "resource_truth_used_as_feature": False,
        },
        "fit": {
            "task_prefix": fit_prefix,
            "template_count": len(fit_rows),
            "video_count": fit_video_count,
            "node_count": fit_node_count,
            "condition_key_fields": list(KEY_FIELDS),
            "condition_key_count": len(models),
            "support_count_min": min(support_counts),
            "support_count_max": max(support_counts),
        },
        "prediction": {
            "task_prefix": predict_prefix,
            "template_count": len(predict_rows),
            "video_count": predict_video_count,
            "node_count": len(layer_rows),
            "unseen_condition_key_count": len(unseen_keys),
            "layer_sidecar": str(layer_path),
            "layer_sidecar_sha256": sha256(layer_path),
            "predicted_layer_count_mean": sum(predicted_layer_counts) / len(predicted_layer_counts)
            if predicted_layer_counts
            else 0.0,
            "predicted_node_count_mean": sum(predicted_node_counts) / len(predicted_node_counts)
            if predicted_node_counts
            else 0.0,
        },
        "outputs": {
            "legacy_files": copied,
            "layer_rows": layer_rows_written,
            "overlay_manifest": str(output_root / "b05_artifact_manifest.json"),
        },
        "training_target": "successor DAG layers reconstructed from r7_s_train predecessor_node_ids",
        "output_contract": "future_h5_layers with identity-free scheduler-visible node prototypes",
    }
    (output_root / "baseline_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--base-artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--max-scenarios", type=int, default=MAX_SCENARIOS)
    parser.add_argument("--fit-task-prefix", default=FIT_TASK_PREFIX)
    parser.add_argument("--predict-task-prefix", default=PREDICT_TASK_PREFIX)
    parser.add_argument("--expected-fit-templates", type=int, default=480)
    parser.add_argument("--expected-predict-templates", type=int, default=160)
    args = parser.parse_args()
    metrics = build(
        templates_path=args.templates,
        base_artifact_root=args.base_artifact_root,
        output_root=args.output_root,
        horizon=args.horizon,
        max_scenarios=args.max_scenarios,
        fit_prefix=args.fit_task_prefix,
        predict_prefix=args.predict_task_prefix,
        expected_fit_templates=args.expected_fit_templates,
        expected_predict_templates=args.expected_predict_templates,
    )
    print(
        json.dumps(
            {
                "status": metrics["status"],
                "fit_templates": metrics["fit"]["template_count"],
                "predict_templates": metrics["prediction"]["template_count"],
                "prediction_nodes": metrics["prediction"]["node_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
