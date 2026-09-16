#!/usr/bin/env python3
"""R0: oracle-signature empirical resource ceiling (frozen design v3).

Stage 0 builds the node-level join table on the v3.1 resource-applicable node
ontology and runs the six hard assertions. The main stage fits the frozen
baselines / LightGBM quantile models / occurrence model, evaluates the frozen
metrics, and writes metrics.json + run_manifest.json.

Read-only with respect to datasets and raw traces; writes only inside the
experiment directory named by the config.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_p9d_topology_dataset import build_chain  # noqa: E402

SUPPORTED_EVENT_TYPES = {"run", "api_call", "action"}
TAUS = (0.5, 0.9, 0.95)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run_meta(config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}

    def add(row: Mapping[str, Any], split_override: Optional[str] = None) -> None:
        run_id = str(row.get("run_id") or "")
        if not run_id or run_id in meta:
            return
        meta[run_id] = {
            "split": split_override or str(row.get("split") or "unknown"),
            "video_id": str(row.get("video_id") or ""),
            "model_stack_id": str(row.get("model_stack_id") or row.get("model_id") or "unknown"),
            "baseline": str(row.get("baseline") or "unknown"),
        }

    for row in read_jsonl(Path(config["behavior_role_samples"])):
        add(row)
    for row in read_jsonl(Path(config["holdout_role_samples"])):
        if str(row.get("source") or row.get("source_split") or "") == "final_holdout_v1":
            add(row, split_override="holdout")
    return meta


def run_baseline(manifest: Mapping[str, Any], trace_path: Path) -> str:
    value = str(manifest.get("baseline") or "")
    if value:
        return value
    name = trace_path.parent.name
    for candidate in ("star", "langgraph_react", "st_fixed"):
        if f"_{candidate}_" in name:
            return candidate
    return "unknown"


def node_kind(event: Mapping[str, Any]) -> str:
    if str(event.get("node_type")) == "planner":
        return "planner"
    if str(event.get("event_type")) == "action":
        return "tool"
    if str(event.get("node_type")) == "answer_generation" and str(event.get("action")) == "generalist.generate":
        return "post_loop_generation"
    return "other"


def load_events(trace_path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    seen = set()
    for row in read_jsonl(trace_path):
        event_id = str(row.get("event_id") or "")
        if not event_id or row.get("event_type") not in SUPPORTED_EVENT_TYPES or event_id in seen:
            continue
        seen.add(event_id)
        events.append(row)
    return events


def index_traces(raw_roots: Sequence[str], run_ids: set) -> Dict[str, Path]:
    traces: Dict[str, Path] = {}
    for root in raw_roots:
        base = Path(root)
        if not base.exists():
            continue
        for path in base.rglob("trace.jsonl"):
            run_id = path.parent.name
            if run_id in run_ids and run_id not in traces:
                traces[run_id] = path
    return traces


def build_node_table(config: Mapping[str, Any], smoke_runs: Optional[set] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    meta = load_run_meta(config)
    run_ids = set(meta)
    if smoke_runs is not None:
        run_ids &= smoke_runs
    traces = index_traces(config["raw_roots"], run_ids)
    missing = sorted(run_ids - set(traces))
    if missing:
        raise RuntimeError(f"missing traces for {len(missing)} runs, first={missing[:5]}")

    rows: List[Dict[str, Any]] = []
    audit: Counter = Counter()
    for run_id in sorted(traces):
        trace_path = traces[run_id]
        events = load_events(trace_path)
        manifest_path = trace_path.parent / "run_manifest.json"
        manifest: Dict[str, Any] = {}
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        baseline = run_baseline(manifest, trace_path)
        chain = build_chain(events, baseline)
        audit["chain_nodes"] += len(chain["nodes"])
        audit["nested_merged"] += len(chain["nested_map"])
        audit["violations"] += len(chain["violations"])
        seen_models: set = set()
        events_by_id = {event["event_id"]: event for event in events}
        for position, node_id in enumerate(chain["nodes"]):
            label = chain["node_labels"][node_id]
            if not label["resource_applicable"]:
                continue
            event = events_by_id[node_id]
            resource = event.get("resource") or {}
            peak_alloc = resource.get("peak_allocated_mb")
            peak_res = resource.get("peak_reserved_mb")
            peaks = [value for value in (peak_alloc, peak_res) if value is not None]
            workload = label.get("workload_scale") or {}
            nested_calls = label.get("nested_calls") or []
            model_class = str(event.get("model_id") or "unknown")
            signature = label.get("resource_signature") or {}
            row = {
                "run_id": run_id,
                "split": meta[run_id]["split"],
                "video_id": meta[run_id]["video_id"] or run_id.split("_")[0],
                "node_id": node_id,
                "chain_position": position,
                "step_id": event.get("step_id"),
                "node_kind": node_kind(event),
                "exec_class": f"{event.get('node_type')}/{event.get('action')}",
                "role": label.get("role"),
                "action_family": label.get("action_family"),
                "model_class": model_class,
                "planner_mode": signature.get("planner_mode") or "unknown",
                "is_retry": bool(label.get("is_retry")),
                "merged_nested_call": bool(label.get("merged_nested_call")),
                "nested_model_class": nested_calls[0]["model_id"] if nested_calls else None,
                "baseline": baseline,
                "model_stack_id": meta[run_id]["model_stack_id"],
                "clip_len": workload.get("clip_len"),
                "query_char_len": workload.get("query_char_len"),
                "nested_api_call_count": workload.get("nested_api_call_count"),
                "clip_len_missing": workload.get("clip_len") is None,
                "query_char_len_missing": workload.get("query_char_len") is None,
                "prefix_model_reuse": model_class in seen_models,
                "runtime_ms": resource.get("runtime_ms"),
                "load_ms": resource.get("load_ms"),
                "peak_allocated_mb": peak_alloc,
                "peak_reserved_mb": peak_res,
                "peak_memory_inclusive_mb": max(peaks) if peaks else None,
                "status_class": str(event.get("status") or "unknown"),
            }
            seen_models.add(model_class)
            rows.append(row)
    return rows, {"counts": dict(audit), "missing_runs": missing}


def compute_thresholds(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> Dict[str, Any]:
    train = [row for row in rows if row["split"] == "train" and row["runtime_ms"] is not None]
    per_class: Dict[str, float] = {}
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in train:
        grouped[row["exec_class"]].append(float(row["runtime_ms"]))
    for key, values in grouped.items():
        per_class[key] = float(np.median(values))
    runtime_values = np.asarray([float(row["runtime_ms"]) for row in train], dtype=float)
    tail_q90 = float(np.quantile(runtime_values, config["tail_rule"]["quantile"]))
    return {"per_exec_class_train_median": per_class, "tail_q90_ms": tail_q90}


def is_stall(row: Mapping[str, Any], thresholds: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    runtime = row.get("runtime_ms")
    if runtime is None:
        return False
    median = thresholds["per_exec_class_train_median"].get(row["exec_class"])
    if median is not None and float(runtime) > config["stall_rule"]["multiple"] * median:
        return True
    return float(runtime) > config["stall_rule"]["absolute_ms"]


def stage0_assertions(
    rows: List[Dict[str, Any]],
    config: Mapping[str, Any],
    thresholds: Dict[str, Any],
    worker_hashes: Mapping[str, str],
    smoke: bool = False,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    expected = config["expected_node_counts"]

    total = len(rows)
    kind_counts = Counter(row["node_kind"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    if not smoke:
        if total != expected["total"]:
            errors.append(f"total nodes {total} != expected {expected['total']}")
        for kind in ("planner", "tool", "post_loop_generation"):
            if kind_counts[kind] != expected[kind]:
                errors.append(f"{kind} nodes {kind_counts[kind]} != expected {expected[kind]}")
        for split, count in expected["by_split"].items():
            if split_counts[split] != count:
                errors.append(f"{split} nodes {split_counts[split]} != expected {count}")
    else:
        warnings.append(f"smoke mode: full-pool count equality skipped (nodes={total})")

    expected_load = config["expected_load_accounting"]
    load_table: Dict[str, Dict[str, int]] = {}
    for kind in ("planner", "tool", "post_loop_generation"):
        table = Counter()
        for row in rows:
            if row["node_kind"] != kind:
                continue
            value = row.get("load_ms")
            if value is None:
                table["missing"] += 1
            elif float(value) == 0.0:
                table["zero"] += 1
            else:
                table["positive"] += 1
        table["total"] = table["missing"] + table["zero"] + table["positive"]
        load_table[kind] = dict(table)
        if table["total"] != kind_counts[kind]:
            errors.append(f"load table {kind} total {table['total']} != class nodes {kind_counts[kind]}")
        if not smoke:
            if table["total"] != expected_load[kind]["total"]:
                errors.append(f"load table {kind} total {table['total']} != {expected_load[kind]['total']}")
            for field in ("missing", "zero", "positive"):
                if table[field] != expected_load[kind][field]:
                    errors.append(f"load table {kind}.{field} {table[field]} != {expected_load[kind][field]}")

    roles = config["feature_roles"]
    feature_cols = set(config["c_ablation"]["C4"])
    target_only = set(roles["target_only"])
    if feature_cols & target_only:
        errors.append(f"target-only columns in feature set: {sorted(feature_cols & target_only)}")
    if "load_ms" in feature_cols:
        errors.append("load_ms must never be a feature")
    overlap = set(roles["pre_execution"]) & set(roles["oracle_label"])
    if overlap:
        errors.append(f"feature role overlap: {sorted(overlap)}")

    memory_missing = sum(1 for row in rows if row["peak_memory_inclusive_mb"] is None)
    memory_by_exec = Counter()
    memory_total_by_exec = Counter()
    memory_by_model = Counter()
    memory_total_by_model = Counter()
    for row in rows:
        memory_total_by_exec[row["exec_class"]] += 1
        memory_total_by_model[row["model_class"]] += 1
        if row["peak_memory_inclusive_mb"] is not None:
            memory_by_exec[row["exec_class"]] += 1
            memory_by_model[row["model_class"]] += 1
    memory_ok = sum(
        1
        for row in rows
        if row["peak_memory_inclusive_mb"] is not None
        and row["peak_memory_inclusive_mb"]
        == max(
            [v for v in (row["peak_allocated_mb"], row["peak_reserved_mb"]) if v is not None]
        )
    )
    if memory_missing + memory_ok != total:
        errors.append("peak_memory_inclusive_mb formula violated")
    warnings.append(f"memory coverage {total - memory_missing}/{total}")

    hierarchy = config["fallback_hierarchy"]
    if not hierarchy or hierarchy[-1] != []:
        errors.append("fallback hierarchy must end with the global level ([])")

    scale_report: Dict[str, Dict[str, int]] = {}
    for field in ("clip_len", "query_char_len", "nested_api_call_count"):
        values = [row.get(field) for row in rows]
        scale_report[field] = {
            "non_null": sum(1 for value in values if value is not None),
            "distinct_non_null": len({value for value in values if value is not None}),
            "total": len(values),
        }
    if all(entry["distinct_non_null"] <= 1 for entry in scale_report.values()):
        warnings.append(
            "workload_scale descriptors are constant in this pool: " + json.dumps(scale_report, sort_keys=True)
        )

    # Thresholds are only computed on train; record evidence.
    thresholds_evidence = {
        "tail_q90_ms": thresholds["tail_q90_ms"],
        "per_exec_class_train_median": thresholds["per_exec_class_train_median"],
    }

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "counts": {"total": total, "by_kind": dict(kind_counts), "by_split": dict(split_counts)},
        "load_accounting": load_table,
        "memory": {"missing": memory_missing, "available": memory_ok},
        "memory_coverage_by_exec_class": {
            key: {"available": memory_by_exec[key], "total": memory_total_by_exec[key]}
            for key in sorted(memory_total_by_exec)
        },
        "memory_coverage_by_model_class": {
            key: {"available": memory_by_model[key], "total": memory_total_by_model[key]}
            for key in sorted(memory_total_by_model)
        },
        "feature_roles": roles,
        "feature_columns_C4": sorted(feature_cols),
        "workload_scale_availability": scale_report,
        "thresholds": thresholds_evidence,
        "worker_source_sha256": dict(worker_hashes),
    }


def bucket_value(value: Any, edges: Sequence[int]) -> int:
    if value is None:
        return -1
    numeric = int(value)
    bucket = 0
    for edge in edges:
        if numeric >= edge:
            bucket += 1
    return bucket


def hierarchy_key(row: Mapping[str, Any], level: Sequence[str], config: Mapping[str, Any]) -> Tuple:
    parts: List[Any] = []
    for name in level:
        if name == "load_bucket":
            parts.append(bucket_value(row.get("nested_api_call_count"), config["load_bucket_edges"]))
        else:
            value = row.get(name)
            parts.append(value if not isinstance(value, bool) else int(value))
    return tuple(parts)


def fit_quantile_tables(
    train_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    target: str,
) -> Dict[int, Dict[Tuple, Dict[str, float]]]:
    taus = [t for t in TAUS]
    tables: Dict[int, Dict[Tuple, Dict[str, float]]] = {}
    for level_index, level in enumerate(config["fallback_hierarchy"]):
        grouped: Dict[Tuple, List[float]] = defaultdict(list)
        for row in train_rows:
            value = row.get(target)
            if value is None or float(value) <= 0.0:
                continue
            grouped[hierarchy_key(row, level, config)].append(float(value))
        table: Dict[Tuple, Dict[str, float]] = {}
        for key, values in grouped.items():
            if len(values) < config["min_support"]:
                continue
            arr = np.asarray(values, dtype=float)
            table[key] = {str(tau): float(np.quantile(arr, tau)) for tau in taus}
        tables[level_index] = table
    return tables


def apply_quantile_tables(
    tables: Mapping[int, Mapping[Tuple, Mapping[str, float]]],
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Tuple[np.ndarray, Dict[str, int]]:
    predictions = np.full((len(rows), len(TAUS)), np.nan, dtype=float)
    level_usage: Counter = Counter()
    for row_index, row in enumerate(rows):
        for index, level in enumerate(config["fallback_hierarchy"]):
            entry = tables[index].get(hierarchy_key(row, level, config))
            if entry is not None:
                level_usage[f"L{index}"] += 1
                for tau_index, tau in enumerate(TAUS):
                    predictions[row_index, tau_index] = entry[str(tau)]
                break
        else:
            level_usage["unmatched"] += 1
    return predictions, dict(level_usage)


def evaluate_predictions(
    rows: Sequence[Mapping[str, Any]],
    predictions: np.ndarray,
    target: str,
    thresholds: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    y = np.asarray([float(row[target]) for row in rows], dtype=float)
    preds = np.asarray(predictions, dtype=float)
    crossing_rate = float(np.mean(~np.all(np.diff(preds, axis=1) >= 0, axis=1))) if preds.size else float("nan")
    sorted_preds = np.sort(preds, axis=1)
    result: Dict[str, Any] = {"n": int(len(rows)), "crossing_rate_raw": crossing_rate}
    for label, mat in (("raw", preds), ("rearranged", sorted_preds)):
        per_tau = {}
        for index, tau in enumerate(TAUS):
            diff = y - mat[:, index]
            pinball = float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))
            per_tau[str(tau)] = {
                "pinball": pinball,
                "calibration": float(np.mean(y <= mat[:, index])),
            }
        result[label] = {
            "per_tau": per_tau,
            "PB_primary": float(np.mean([per_tau[str(tau)]["pinball"] for tau in TAUS])),
            "mae_q50": float(np.mean(np.abs(y - mat[:, 0]))),
            "median_ae_q50": float(np.median(np.abs(y - mat[:, 0]))),
            "max_abs_calibration_error": float(max(abs(per_tau[str(tau)]["calibration"] - tau) for tau in TAUS)),
        }
    result["PB_primary"] = result["rearranged"]["PB_primary"]
    return result


def group_macro(
    rows: Sequence[Mapping[str, Any]],
    predictions: np.ndarray,
    target: str,
    group_key: str,
) -> float:
    groups: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row.get(group_key))].append(index)
    values = []
    y = np.asarray([float(row[target]) for row in rows], dtype=float)
    sorted_preds = np.sort(np.asarray(predictions, dtype=float), axis=1)
    for indices in groups.values():
        idx = np.asarray(indices)
        diff = y[idx] - sorted_preds[idx, 0]
        pinballs = []
        for tau_index, tau in enumerate(TAUS):
            d = y[idx] - sorted_preds[idx, tau_index]
            pinballs.append(float(np.mean(np.maximum(tau * d, (tau - 1.0) * d))))
        values.append(float(np.mean(pinballs)))
    return float(np.mean(values)) if values else float("nan")


def bootstrap_improvement(
    rows: Sequence[Mapping[str, Any]],
    base_preds: np.ndarray,
    model_preds: np.ndarray,
    target: str,
    config: Mapping[str, Any],
    n_resamples: int,
) -> Dict[str, Any]:
    y = np.asarray([float(row[target]) for row in rows], dtype=float)
    base = np.sort(np.asarray(base_preds, dtype=float), axis=1)
    model = np.sort(np.asarray(model_preds, dtype=float), axis=1)
    groups: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row.get("video_id"))].append(index)
    group_keys = sorted(groups)
    rng = np.random.default_rng(config["bootstrap"]["seed"])
    improvements = []
    for _ in range(n_resamples):
        sampled = rng.choice(group_keys, size=len(group_keys), replace=True)
        indices = np.concatenate([groups[key] for key in sampled])
        diffs_base = []
        diffs_model = []
        for tau_index, tau in enumerate(TAUS):
            d = y[indices] - base[indices, tau_index]
            diffs_base.append(float(np.mean(np.maximum(tau * d, (tau - 1.0) * d))))
            d = y[indices] - model[indices, tau_index]
            diffs_model.append(float(np.mean(np.maximum(tau * d, (tau - 1.0) * d))))
        pb_base = float(np.mean(diffs_base))
        pb_model = float(np.mean(diffs_model))
        improvements.append(1.0 - pb_model / pb_base if pb_base > 0 else float("nan"))
    arr = np.asarray([value for value in improvements if math.isfinite(value)], dtype=float)
    return {
        "n_resamples": n_resamples,
        "mean": float(np.mean(arr)),
        "ci95_low": float(np.quantile(arr, 0.025)),
        "ci95_high": float(np.quantile(arr, 0.975)),
    }


def encode_features(
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    config: Mapping[str, Any],
    categories: Optional[Dict[str, List[str]]] = None,
) -> Tuple[np.ndarray, List[int], Dict[str, List[str]]]:
    categorical = set(config["categorical_features"])
    if categories is None:
        categories = {}
        for name in feature_names:
            if name in categorical:
                values = sorted({str(row.get(name) if row.get(name) is not None else "<none>") for row in rows})
                categories[name] = values
    matrix = np.full((len(rows), len(feature_names)), np.nan, dtype=float)
    cat_indices: List[int] = []
    for column, name in enumerate(feature_names):
        if name in categorical:
            cat_indices.append(column)
            mapping = {value: index for index, value in enumerate(categories[name])}
            for row_index, row in enumerate(rows):
                value = str(row.get(name) if row.get(name) is not None else "<none>")
                matrix[row_index, column] = float(mapping.get(value, -1))
        else:
            for row_index, row in enumerate(rows):
                value = row.get(name)
                if value is None:
                    matrix[row_index, column] = np.nan
                elif isinstance(value, bool):
                    matrix[row_index, column] = 1.0 if value else 0.0
                else:
                    matrix[row_index, column] = float(value)
    return matrix, cat_indices, categories


def fit_predict_quantile(
    train_rows: Sequence[Mapping[str, Any]],
    predict_sets: Mapping[str, Sequence[Mapping[str, Any]]],
    feature_names: Sequence[str],
    target: str,
    params: Mapping[str, Any],
    seed: int,
    config: Mapping[str, Any],
    categories: Optional[Dict[str, List[str]]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    train_rows = [row for row in train_rows if row.get(target) is not None and float(row[target]) > 0.0]
    x_train, cat_indices, categories = encode_features(train_rows, feature_names, config, categories)
    log_target = np.log1p([float(row[target]) for row in train_rows])
    outputs: Dict[str, np.ndarray] = {}
    for tau in TAUS:
        lgb_params = {
            "objective": "quantile",
            "alpha": tau,
            "learning_rate": config["lgbm_quantile"]["learning_rate"],
            "num_leaves": params["num_leaves"],
            "min_data_in_leaf": params["min_data_in_leaf"],
            "subsample": config["lgbm_quantile"]["subsample"],
            "colsample_bytree": config["lgbm_quantile"]["colsample_bytree"],
            "verbosity": -1,
            "num_threads": 4,
            "seed": seed,
            "deterministic": True,
            "force_row_wise": True,
        }
        dataset = lgb.Dataset(x_train, label=log_target, categorical_feature=cat_indices, free_raw_data=False)
        booster = lgb.train(lgb_params, dataset, num_boost_round=config["lgbm_quantile"]["n_estimators"])
        predictions = []
        for name, rows in predict_sets.items():
            x_pred, _, _ = encode_features(rows, feature_names, config, categories)
            raw = booster.predict(x_pred)
            predictions.append(np.expm1(raw))
        outputs[str(tau)] = predictions
    stacked = {name: np.column_stack([outputs[str(tau)][index] for tau in TAUS]) for index, name in enumerate(predict_sets)}
    return stacked, categories


def lgb_selection(
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    target: str,
    config: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, List[str]]]]:
    candidates = []
    categories: Optional[Dict[str, List[str]]] = None
    thresholds = compute_thresholds(train_rows, config)
    for params in config["lgbm_quantile"]["grid"]:
        seed_predictions = []
        per_seed_pb = []
        for seed in config["lgbm_quantile"]["seeds"]:
            stacked, categories = fit_predict_quantile(
                train_rows, {"validation": val_rows}, feature_names, target, params, seed, config, categories
            )
            seed_prediction = stacked["validation"]
            seed_predictions.append(seed_prediction)
            per_seed_pb.append(evaluate_predictions(val_rows, seed_prediction, target, thresholds, config)["PB_primary"])
        mean_pred = np.mean(np.stack(seed_predictions, axis=0), axis=0)
        mean_metrics = evaluate_predictions(val_rows, mean_pred, target, thresholds, config)
        candidates.append(
            {
                "params": params,
                "mean_pb": mean_metrics["PB_primary"],
                "worst_pb": float(max(per_seed_pb)),
                "mean_pred": mean_pred,
                "seed_predictions": seed_predictions,
            }
        )
    candidates.sort(key=lambda item: (item["mean_pb"], item["params"]["num_leaves"], item["params"]["min_data_in_leaf"]))
    best = candidates[0]
    selection = {
        "chosen": best["params"],
        "candidates": [
            {"params": item["params"], "validation_PB_primary_mean": item["mean_pb"], "validation_PB_primary_worst_seed": item["worst_pb"]}
            for item in candidates
        ],
    }
    return selection, {"best": best}, categories


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def encode_dense(
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    config: Mapping[str, Any],
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Dense one-hot + standardized-numeric encoding for the logistic reference."""
    categorical = [name for name in feature_names if name in set(config["categorical_features"])]
    numeric = [name for name in feature_names if name not in set(config["categorical_features"])]
    if state is None:
        state = {"categories": {}, "median": {}, "mean": {}, "std": {}}
        for name in categorical:
            state["categories"][name] = sorted({str(row.get(name) if row.get(name) is not None else "<none>") for row in rows})
        for name in numeric:
            values = np.asarray([float(row[name]) if row.get(name) is not None else np.nan for row in rows], dtype=float)
            median = float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0
            filled = np.where(np.isfinite(values), values, median)
            state["median"][name] = median
            state["mean"][name] = float(np.mean(filled))
            std = float(np.std(filled))
            state["std"][name] = std if std > 0 else 1.0
    blocks = []
    for name in categorical:
        categories = state["categories"][name]
        mapping = {value: index for index, value in enumerate(categories)}
        block = np.zeros((len(rows), len(categories)), dtype=float)
        for row_index, row in enumerate(rows):
            value = str(row.get(name) if row.get(name) is not None else "<none>")
            index = mapping.get(value)
            if index is not None:
                block[row_index, index] = 1.0
        blocks.append(block)
    for name in numeric:
        values = np.asarray([float(row[name]) if row.get(name) is not None else np.nan for row in rows], dtype=float)
        filled = np.where(np.isfinite(values), values, state["median"][name])
        blocks.append(((filled - state["mean"][name]) / state["std"][name]).reshape(-1, 1))
    return np.hstack(blocks) if blocks else np.zeros((len(rows), 0)), state


def fit_logistic_reference(
    x_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
) -> np.ndarray:
    from scipy.optimize import minimize

    sample_count, dimension = x_train.shape
    design = np.hstack([np.ones((sample_count, 1)), x_train])

    def objective(weights: np.ndarray) -> float:
        z = design @ weights
        loss = float(np.mean(np.logaddexp(0.0, z) - y_train * z))
        return loss + 0.5 * float(np.sum(weights[1:] ** 2)) / C

    def gradient(weights: np.ndarray) -> np.ndarray:
        z = design @ weights
        probabilities = _sigmoid(z)
        grad = design.T @ (probabilities - y_train) / sample_count
        grad[1:] += weights[1:] / C
        return grad

    result = minimize(objective, np.zeros(design.shape[1]), jac=gradient, method="L-BFGS-B", options={"maxiter": 500})
    return result.x


def predict_logistic(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    design = np.hstack([np.ones((len(x), 1)), x])
    return _sigmoid(design @ weights)


def _occurrence_report(rows: Sequence[Mapping[str, Any]], probabilities: np.ndarray, prefix: str = "") -> Dict[str, Any]:
    y = np.asarray([1.0 if float(row["load_ms"]) > 0.0 else 0.0 for row in rows], dtype=float)
    bins = np.linspace(0.0, 1.0, 11)
    calibration = []
    for index in range(10):
        mask = (probabilities >= bins[index]) & (
            (probabilities < bins[index + 1]) if index < 9 else (probabilities <= bins[index + 1])
        )
        if mask.sum() > 0:
            calibration.append(
                {
                    "bin": [float(bins[index]), float(bins[index + 1])],
                    "n": int(mask.sum()),
                    "mean_pred": float(np.mean(probabilities[mask])),
                    "mean_true": float(np.mean(y[mask])),
                }
            )
    return {
        "n": int(len(rows)),
        "base_rate": float(np.mean(y)),
        "brier": float(np.mean((probabilities - y) ** 2)),
        "calibration": calibration,
        "prefix": prefix,
    }


def fit_predict_occurrence(
    train_rows: Sequence[Mapping[str, Any]],
    predict_sets: Mapping[str, Sequence[Mapping[str, Any]]],
    feature_names: Sequence[str],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    train_rows = [row for row in train_rows if row.get("load_ms") is not None]
    labels = np.asarray([1.0 if float(row["load_ms"]) > 0.0 else 0.0 for row in train_rows], dtype=float)
    x_train, cat_indices, categories = encode_features(train_rows, feature_names, config)
    params = config["lgbm_occurrence"]
    report: Dict[str, Any] = {"train_base_rate": float(np.mean(labels)), "splits": {}}
    seed_probabilities: Dict[str, List[np.ndarray]] = defaultdict(list)
    for seed in params["seeds"]:
        lgb_params = {
            "objective": "binary",
            "learning_rate": params["learning_rate"],
            "num_leaves": params["num_leaves"],
            "min_data_in_leaf": params["min_data_in_leaf"],
            "subsample": params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
            "verbosity": -1,
            "num_threads": 4,
            "seed": seed,
            "deterministic": True,
            "force_row_wise": True,
        }
        dataset = lgb.Dataset(x_train, label=labels, categorical_feature=cat_indices, free_raw_data=False)
        booster = lgb.train(lgb_params, dataset, num_boost_round=params["n_estimators"])
        for name, rows in predict_sets.items():
            rows = [row for row in rows if row.get("load_ms") is not None]
            x_pred, _, _ = encode_features(rows, feature_names, config, categories)
            seed_probabilities[name].append(booster.predict(x_pred))
    for name in predict_sets:
        rows = [row for row in predict_sets[name] if row.get("load_ms") is not None]
        if not rows:
            continue
        stacked = np.stack(seed_probabilities[name], axis=0)
        mean_prob = np.mean(stacked, axis=0)
        per_seed_brier = [
            float(np.mean((stacked[index] - np.asarray([1.0 if float(row["load_ms"]) > 0.0 else 0.0 for row in rows])) ** 2))
            for index in range(stacked.shape[0])
        ]
        entry = _occurrence_report(rows, mean_prob)
        entry["worst_seed_brier"] = float(max(per_seed_brier))
        report["splits"][name] = entry

    # Fixed L2 logistic reference (no tuning).
    one_hot_state = None
    x_train_dense, one_hot_state = encode_dense(train_rows, feature_names, config)
    weights = fit_logistic_reference(x_train_dense, labels, C=1.0)
    report["logistic_reference"] = {"splits": {}}
    for name, rows in predict_sets.items():
        rows = [row for row in rows if row.get("load_ms") is not None]
        if not rows:
            continue
        x_pred, _ = encode_dense(rows, feature_names, config, one_hot_state)
        probabilities = predict_logistic(weights, x_pred)
        report["logistic_reference"]["splits"][name] = _occurrence_report(rows, probabilities)
    return report


def fit_predict_duration(
    train_rows: Sequence[Mapping[str, Any]],
    predict_sets: Mapping[str, Sequence[Mapping[str, Any]]],
    feature_names: Sequence[str],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    train_positive = [row for row in train_rows if row.get("load_ms") is not None and float(row["load_ms"]) > 0.0]
    if len(train_positive) < config["min_support"]:
        return {"status": "insufficient_positive_rows", "n_train": len(train_positive)}
    params = config["lgbm_quantile"]["grid"][0]
    stacked, _ = fit_predict_quantile(train_positive, predict_sets, feature_names, "load_ms", params, config["lgbm_quantile"]["seeds"][0], config)
    report: Dict[str, Any] = {"n_train_positive": len(train_positive), "splits": {}}
    for name, rows in predict_sets.items():
        positive_rows = [row for row in rows if row.get("load_ms") is not None and float(row["load_ms"]) > 0.0]
        if not positive_rows:
            continue
        indices = [index for index, row in enumerate(rows) if row.get("load_ms") is not None and float(row["load_ms"]) > 0.0]
        predictions = stacked[name][indices]
        metrics = evaluate_predictions(positive_rows, predictions, "load_ms", {}, config)
        report["splits"][name] = {"n": len(positive_rows), "PB_primary": metrics["PB_primary"], "per_tau": metrics["rearranged"]["per_tau"]}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("stage0", "run"), default="run")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if lgb is None:
        print(json.dumps({"ok": False, "error": "lightgbm is required"}, ensure_ascii=False))
        return 2

    config = json.loads(args.config.read_text(encoding="utf-8"))
    experiment_root = args.config.resolve().parent
    artifacts_root = experiment_root / "artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    started = time.time()

    smoke_runs = None
    if args.smoke:
        meta = load_run_meta(config)
        by_split: Dict[str, List[str]] = defaultdict(list)
        for run_id, info in meta.items():
            by_split[info["split"]].append(run_id)
        smoke_runs = set()
        for split, run_ids in by_split.items():
            smoke_runs.update(sorted(run_ids)[: config["smoke"]["runs_per_split"]])
        print(f"smoke runs: {len(smoke_runs)}")

    rows, build_audit = build_node_table(config, smoke_runs)
    print(f"node rows: {len(rows)}")
    thresholds = compute_thresholds(rows, config)
    worker_hashes = {}
    for name in ("qwen3_vl_worker.py", "qwen_text_worker.py"):
        path = PROJECT_ROOT / "src" / "tracing" / "collectors" / name
        if path.is_file():
            worker_hashes[name] = sha256_file(path)
    audit = stage0_assertions(rows, config, thresholds, worker_hashes, smoke=bool(args.smoke))
    audit["build"] = build_audit
    audit["smoke"] = bool(args.smoke)
    stage0_path = experiment_root / f"stage0_audit{suffix}.json"
    stage0_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    if not audit["ok"]:
        print(json.dumps({"ok": False, "stage0_errors": audit["errors"]}, ensure_ascii=False, indent=2))
        return 1
    print("stage0: ok")

    table_path = artifacts_root / f"node_table{suffix}.jsonl.gz"
    with gzip.open(table_path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

    if args.stage == "stage0":
        print(json.dumps({"ok": True, "stage": "stage0", "nodes": len(rows), "audit": str(stage0_path)}, ensure_ascii=False))
        return 0

    splits: Dict[str, List[Dict[str, Any]]] = {name: [] for name in config["splits_order"]}
    for row in rows:
        splits.setdefault(row["split"], []).append(row)
    train_rows = splits["train"]
    predict_sets = {name: splits[name] for name in config["splits_order"] if splits[name]}

    selection, best, categories = lgb_selection(train_rows, splits["validation"], config["c_ablation"]["C4"], "runtime_ms", config)
    best_params = best["best"]["params"]
    print(f"selected lgbm: {best_params}")

    runtime_models: Dict[str, Any] = {}
    for name, rows_set in predict_sets.items():
        seed_predictions = []
        per_seed_pb = []
        for seed in config["lgbm_quantile"]["seeds"]:
            stacked, _ = fit_predict_quantile(train_rows, {name: rows_set}, config["c_ablation"]["C4"], "runtime_ms", best_params, seed, config, categories)
            seed_prediction = stacked[name]
            seed_predictions.append(seed_prediction)
            per_seed_pb.append(evaluate_predictions(rows_set, seed_prediction, "runtime_ms", thresholds, config)["PB_primary"])
        mean_pred = np.mean(np.stack(seed_predictions, axis=0), axis=0)
        metrics = evaluate_predictions(rows_set, mean_pred, "runtime_ms", thresholds, config)
        runtime_models[name] = {
            "mean_seed_metrics": metrics,
            "worst_seed_PB_primary": float(max(per_seed_pb)),
            "per_run_macro_PB_primary": group_macro(rows_set, mean_pred, "runtime_ms", "run_id"),
            "per_video_macro_PB_primary": group_macro(rows_set, mean_pred, "runtime_ms", "video_id"),
            "seed_predictions": seed_predictions,
            "mean_prediction": mean_pred,
        }

    baseline_tables = fit_quantile_tables(train_rows, config, "runtime_ms")
    baselines: Dict[str, Any] = {}
    for name, rows_set in predict_sets.items():
        table_predictions, usage = apply_quantile_tables(baseline_tables, rows_set, config)
        median_column = table_predictions[:, 0]
        core_pred = np.column_stack([median_column, median_column, median_column])
        metrics_core = evaluate_predictions(rows_set, core_pred, "runtime_ms", thresholds, config)
        secondary_metrics = evaluate_predictions(rows_set, table_predictions, "runtime_ms", thresholds, config)
        baselines[name] = {
            "core_median": {"metrics": metrics_core, "level_usage": usage},
            "core_prediction": core_pred,
            "secondary_quantile_metrics": secondary_metrics,
            "table_predictions": table_predictions,
        }

    ablation: Dict[str, Any] = {}
    for level, feature_names in config["c_ablation"].items():
        seed_predictions = []
        for seed in config["lgbm_quantile"]["seeds"]:
            stacked, _ = fit_predict_quantile(train_rows, {"validation": splits["validation"], "test": splits["test"]}, feature_names, "runtime_ms", best_params, seed, config, categories)
            seed_predictions.append(stacked)
        mean_val = np.mean(np.stack([p["validation"] for p in seed_predictions], axis=0), axis=0)
        mean_test = np.mean(np.stack([p["test"] for p in seed_predictions], axis=0), axis=0)
        ablation[level] = {
            "validation_PB_primary": evaluate_predictions(splits["validation"], mean_val, "runtime_ms", thresholds, config)["PB_primary"],
            "test_PB_primary": evaluate_predictions(splits["test"], mean_test, "runtime_ms", thresholds, config)["PB_primary"],
        }

    exact_vs_coarse: Dict[str, Any] = {}
    for arm in ("exact", "coarse"):
        if arm == "coarse":
            mapping = config["model_class_coarse_map"]
            coarse_rows = []
            for row in rows:
                new_row = dict(row)
                key = str(row["model_class"]).strip().lower()
                new_row["model_class"] = mapping.get(key, "other")
                coarse_rows.append(new_row)
            split_rows = {name: [] for name in config["splits_order"]}
            for row in coarse_rows:
                split_rows.setdefault(row["split"], []).append(row)
        else:
            split_rows = splits
        seed_predictions = []
        for seed in config["lgbm_quantile"]["seeds"]:
            stacked, _ = fit_predict_quantile(split_rows["train"], {"validation": split_rows["validation"]}, config["c_ablation"]["C4"], "runtime_ms", best_params, seed, config)
            seed_predictions.append(stacked["validation"])
        mean_pred = np.mean(np.stack(seed_predictions, axis=0), axis=0)
        exact_vs_coarse[arm] = {
            "validation_PB_primary": evaluate_predictions(split_rows["validation"], mean_pred, "runtime_ms", thresholds, config)["PB_primary"]
        }

    bootstrap = {
        name: bootstrap_improvement(
            rows_set,
            baselines[name]["core_prediction"],
            runtime_models[name]["mean_prediction"],
            "runtime_ms",
            config,
            config["smoke"]["bootstrap_resamples"] if args.smoke else config["bootstrap"]["n_resamples"],
        )
        for name, rows_set in predict_sets.items()
    }

    tail_report: Dict[str, Any] = {}
    stall_report: Dict[str, Any] = {}
    for name, rows_set in predict_sets.items():
        tail_idx = [
            index
            for index, row in enumerate(rows_set)
            if row["runtime_ms"] is not None and float(row["runtime_ms"]) > thresholds["tail_q90_ms"]
        ]
        if tail_idx:
            tail_report[name] = evaluate_predictions(
                [rows_set[index] for index in tail_idx],
                runtime_models[name]["mean_prediction"][tail_idx],
                "runtime_ms",
                thresholds,
                config,
            )
        stall_idx = [index for index, row in enumerate(rows_set) if is_stall(row, thresholds, config)]
        stall_set = set(stall_idx)
        normal_idx = [index for index in range(len(rows_set)) if index not in stall_set]
        if stall_idx:
            stall_report[name] = {
                "n_stall": len(stall_idx),
                "n_normal": len(normal_idx),
                "stall_PB_primary": evaluate_predictions([rows_set[i] for i in stall_idx], runtime_models[name]["mean_prediction"][stall_idx], "runtime_ms", thresholds, config)["PB_primary"],
                "normal_PB_primary": evaluate_predictions([rows_set[i] for i in normal_idx], runtime_models[name]["mean_prediction"][normal_idx], "runtime_ms", thresholds, config)["PB_primary"],
            }

    strata: Dict[str, Any] = {}
    per_class: Dict[str, Any] = {}
    for name, rows_set in predict_sets.items():
        grouped: Dict[str, List[int]] = defaultdict(list)
        for index, row in enumerate(rows_set):
            grouped[row["exec_class"]].append(index)
        entries = {}
        class_entries = {}
        for key, indices in sorted(grouped.items()):
            subset = [rows_set[i] for i in indices]
            metrics = evaluate_predictions(subset, runtime_models[name]["mean_prediction"][np.asarray(indices)], "runtime_ms", thresholds, config)
            class_entries[key] = {
                "n": len(indices),
                "PB_primary": metrics["PB_primary"],
                "max_abs_calibration_error": metrics["rearranged"]["max_abs_calibration_error"],
            }
            rule = config["metrics"]["major_strata_rule"]
            videos = len({rows_set[i]["video_id"] for i in indices})
            if len(indices) >= rule["min_nodes"] and videos >= rule["min_videos"]:
                entries[key] = dict(class_entries[key], videos=videos)
        strata[name] = entries
        per_class[name] = class_entries

    occurrence = fit_predict_occurrence(train_rows, predict_sets, config["c_ablation"]["C4"], config)
    duration = fit_predict_duration(train_rows, predict_sets, config["c_ablation"]["C4"], config)

    prediction_paths = {}
    for name, rows_set in predict_sets.items():
        path = artifacts_root / f"predictions_{name}{suffix}.jsonl.gz"
        mean_prediction = runtime_models[name]["mean_prediction"]
        core_prediction = baselines[name]["core_prediction"]
        with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
            for index, row in enumerate(rows_set):
                record = {
                    "node_id": row["node_id"],
                    "run_id": row["run_id"],
                    "video_id": row["video_id"],
                    "exec_class": row["exec_class"],
                    "model_class": row["model_class"],
                    "baseline": row["baseline"],
                    "runtime_ms": row["runtime_ms"],
                    "q50_ms": float(mean_prediction[index, 0]),
                    "q90_ms": float(mean_prediction[index, 1]),
                    "q95_ms": float(mean_prediction[index, 2]),
                    "core_median_ms": float(core_prediction[index, 0]),
                    "tail": bool(row["runtime_ms"] is not None and float(row["runtime_ms"]) > thresholds["tail_q90_ms"]),
                    "stall": is_stall(row, thresholds, config),
                }
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        prediction_paths[name] = path

    memory_report: Dict[str, Any] = {}
    memory_rows_train = [row for row in train_rows if row["peak_memory_inclusive_mb"] is not None]
    if len(memory_rows_train) >= config["min_support"]:
        seed_predictions = []
        for seed in config["lgbm_quantile"]["seeds"]:
            stacked, _ = fit_predict_quantile(memory_rows_train, {name: [row for row in predict_sets[name] if row["peak_memory_inclusive_mb"] is not None] for name in predict_sets}, config["c_ablation"]["C4"], "peak_memory_inclusive_mb", best_params, seed, config)
            seed_predictions.append(stacked)
        for name in predict_sets:
            subset = [row for row in predict_sets[name] if row["peak_memory_inclusive_mb"] is not None]
            if not subset:
                continue
            mean_pred = np.mean(np.stack([p[name] for p in seed_predictions], axis=0), axis=0)
            memory_report[name] = evaluate_predictions(subset, mean_pred, "peak_memory_inclusive_mb", thresholds, config)

    metrics = {
        "experiment_id": config["experiment_id"],
        "smoke": bool(args.smoke),
        "dataset_root": config["dataset_root"],
        "config_sha256": sha256_file(args.config),
        "stage0": audit,
        "thresholds": {"tail_q90_ms": thresholds["tail_q90_ms"], "stall_per_exec_class_median": thresholds["per_exec_class_train_median"]},
        "lgbm_selection": selection,
        "runtime": {
            "baselines": {
                name: {
                    "core_median": {
                        "PB_primary": baselines[name]["core_median"]["metrics"]["PB_primary"],
                        "level_usage": baselines[name]["core_median"]["level_usage"],
                    },
                    "secondary_quantile_diagnostic": {
                        "PB_primary": baselines[name]["secondary_quantile_metrics"]["PB_primary"],
                        "max_abs_calibration_error": baselines[name]["secondary_quantile_metrics"]["rearranged"]["max_abs_calibration_error"],
                    },
                }
                for name in baselines
            },
            "models": {name: {"mean_seed_metrics": runtime_models[name]["mean_seed_metrics"], "worst_seed_PB_primary": runtime_models[name]["worst_seed_PB_primary"], "per_run_macro_PB_primary": runtime_models[name]["per_run_macro_PB_primary"], "per_video_macro_PB_primary": runtime_models[name]["per_video_macro_PB_primary"]} for name in runtime_models},
            "improvement_over_core": {
                name: {
                    "improvement": 1.0 - runtime_models[name]["mean_seed_metrics"]["PB_primary"] / baselines[name]["core_median"]["metrics"]["PB_primary"],
                    "bootstrap": bootstrap[name],
                }
                for name in runtime_models
            },
            "ablation": ablation,
            "exact_vs_coarse": exact_vs_coarse,
            "tail": tail_report,
            "stall_vs_normal": stall_report,
            "strata": strata,
            "per_exec_class": per_class,
        },
        "load": {"occurrence": occurrence, "duration": duration},
        "memory": memory_report,
        "run_seconds": round(time.time() - started, 2),
    }
    metrics_path = experiment_root / f"metrics{suffix}.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    manifest = {
        "experiment_id": config["experiment_id"],
        "smoke": bool(args.smoke),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "lightgbm_version": getattr(lgb, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "dataset_root": config["dataset_root"],
        "dataset_registry_sha256": sha256_file(Path(config["dataset_registry"])) if Path(config["dataset_registry"]).is_file() else None,
        "config_sha256": sha256_file(args.config),
        "stage0_sha256": sha256_file(stage0_path),
        "node_table_sha256": sha256_file(table_path),
        "metrics_sha256": sha256_file(metrics_path),
        "nodes": len(rows),
        "splits": {name: len(splits[name]) for name in config["splits_order"]},
        "selected_lgbm": best_params,
        "prediction_files": {name: sha256_file(path) for name, path in prediction_paths.items()},
        "run_seconds": metrics["run_seconds"],
    }
    (experiment_root / f"run_manifest{suffix}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({"ok": True, "nodes": len(rows), "metrics": str(metrics_path), "run_seconds": metrics["run_seconds"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
