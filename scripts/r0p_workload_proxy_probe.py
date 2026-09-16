#!/usr/bin/env python3
"""R0.1 workload proxy probe.

Reuses the frozen R0 node ontology and estimator protocol, adds workload-size
proxies that already exist in the traces (tool request length, frame counts,
task size, video metadata), and compares feature sets F0/F1/F2 on
train/validation/test. The frozen holdout is deliberately NOT used.

Read-only with respect to existing datasets; writes only inside this
experiment directory.
"""

from __future__ import annotations

import argparse
import gzip
import json
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

from scripts.r0_oracle_signature_ceiling import (  # noqa: E402
    TAUS,
    bootstrap_improvement,
    encode_features,
    evaluate_predictions,
    fit_predict_quantile,
    read_jsonl,
    sha256_file,
)

SUPPORTED_EVENT_TYPES = {"run", "api_call", "action"}


def load_node_table(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_trace_proxies(raw_roots: Sequence[str], node_ids: set, run_ids: set) -> Dict[str, Dict[str, Any]]:
    proxies: Dict[str, Dict[str, Any]] = {}
    for root in raw_roots:
        base = Path(root)
        if not base.exists():
            continue
        for path in base.rglob("trace.jsonl"):
            if path.parent.name not in run_ids:
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                event_id = str(event.get("event_id") or "")
                if event_id not in node_ids:
                    continue
                inp = event.get("input") if isinstance(event.get("input"), dict) else {}
                frame_indices = inp.get("frame_indices")
                proxies[event_id] = {
                    "tool_input_len": len(inp["tool_input"]) if isinstance(inp.get("tool_input"), str) else None,
                    "frame_count_before": float(inp["frame_count_before"]) if isinstance(inp.get("frame_count_before"), (int, float)) else None,
                    "frame_count": float(len(frame_indices)) if isinstance(frame_indices, list) else None,
                }
    return proxies


def load_run_task_fields(config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    fields = ("question_chars", "question_tokens", "option_chars_mean", "option_count")
    runs: Dict[str, Dict[str, Any]] = {}
    for path in (Path(config["behavior_role_samples"]), Path(config["holdout_role_samples"])):
        for row in read_jsonl(path):
            run_id = str(row.get("run_id") or "")
            if not run_id or run_id in runs:
                continue
            structure = row.get("task_structure") if isinstance(row.get("task_structure"), dict) else {}
            values: Dict[str, Any] = {}
            for field in fields:
                value = structure.get(field)
                if value is None:
                    value = row.get(field)
                values[field] = float(value) if isinstance(value, (int, float)) else None
            runs[run_id] = values
    return runs


def load_video_metadata(config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}
    path = Path(config["provenance"])
    if not path.is_file():
        return meta
    for row in read_jsonl(path):
        video_id = str(row.get("video_id") or "")
        original = row.get("original") if isinstance(row.get("original"), dict) else {}
        if not video_id:
            continue
        meta[video_id] = {
            "video_duration_s": float(original["duration_s"]) if isinstance(original.get("duration_s"), (int, float)) else None,
            "video_width": float(original["width"]) if isinstance(original.get("width"), (int, float)) else None,
            "video_height": float(original["height"]) if isinstance(original.get("height"), (int, float)) else None,
        }
    return meta


def merge_rows(
    rows: Sequence[Mapping[str, Any]],
    proxies: Mapping[str, Mapping[str, Any]],
    task_fields: Mapping[str, Mapping[str, Any]],
    video_meta: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    merged = []
    coverage: Counter = Counter()
    for row in rows:
        new_row = dict(row)
        proxy = proxies.get(row["node_id"], {})
        for key in ("tool_input_len", "frame_count_before", "frame_count"):
            value = proxy.get(key)
            new_row[key] = value
            if value is not None:
                coverage[f"proxy_{key}"] += 1
        task = task_fields.get(row["run_id"], {})
        for key, value in task.items():
            new_row[key] = value
            if value is not None:
                coverage[f"task_{key}"] += 1
        video = video_meta.get(row["video_id"], {})
        for key, value in video.items():
            new_row[key] = value
            if value is not None:
                coverage[f"video_{key}"] += 1
        merged.append(new_row)
    return merged, dict(coverage)


def select_and_evaluate(
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    thresholds: Dict[str, Any] = {}
    candidates = []
    for params in config["lgbm_quantile"]["grid"]:
        seed_predictions = []
        per_seed_pb = []
        for seed in config["lgbm_quantile"]["seeds"]:
            stacked, _ = fit_predict_quantile(train_rows, {"validation": val_rows}, feature_names, "runtime_ms", params, seed, config)
            prediction = stacked["validation"]
            seed_predictions.append(prediction)
            per_seed_pb.append(evaluate_predictions(val_rows, prediction, "runtime_ms", thresholds, config)["PB_primary"])
        mean_pred = np.mean(np.stack(seed_predictions, axis=0), axis=0)
        candidates.append(
            {
                "params": params,
                "mean_pb": evaluate_predictions(val_rows, mean_pred, "runtime_ms", thresholds, config)["PB_primary"],
                "worst_pb": float(max(per_seed_pb)),
            }
        )
    candidates.sort(key=lambda item: (item["mean_pb"], item["params"]["num_leaves"], item["params"]["min_data_in_leaf"]))
    chosen = candidates[0]["params"]
    result: Dict[str, Any] = {
        "selected": chosen,
        "selection_candidates": candidates,
        "splits": {},
    }
    for name, rows_set in (("train", train_rows), ("validation", val_rows), ("test", test_rows)):
        seed_predictions = []
        for seed in config["lgbm_quantile"]["seeds"]:
            stacked, _ = fit_predict_quantile(train_rows, {name: rows_set}, feature_names, "runtime_ms", chosen, seed, config)
            seed_predictions.append(stacked[name])
        mean_pred = np.mean(np.stack(seed_predictions, axis=0), axis=0)
        metrics = evaluate_predictions(rows_set, mean_pred, "runtime_ms", thresholds, config)
        result["splits"][name] = {
            "PB_primary": metrics["PB_primary"],
            "mae_q50": metrics["rearranged"]["mae_q50"],
            "max_abs_calibration_error": metrics["rearranged"]["max_abs_calibration_error"],
            "per_tau": metrics["rearranged"]["per_tau"],
            "prediction": mean_pred,
            "rows": rows_set,
        }
    return result


def feature_importance(
    train_rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    config: Mapping[str, Any],
    seed: int,
) -> Dict[str, float]:
    rows = [row for row in train_rows if row.get("runtime_ms") is not None and float(row["runtime_ms"]) > 0.0]
    matrix, cat_indices, _ = encode_features(rows, feature_names, config)
    target = np.log1p([float(row["runtime_ms"]) for row in rows])
    params = {
        "objective": "quantile",
        "alpha": 0.5,
        "learning_rate": config["lgbm_quantile"]["learning_rate"],
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "verbosity": -1,
        "num_threads": 4,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
    }
    booster = lgb.train(params, lgb.Dataset(matrix, label=target, categorical_feature=cat_indices), num_boost_round=config["lgbm_quantile"]["n_estimators"])
    gains = booster.feature_importance(importance_type="gain")
    return {name: float(gain) for name, gain in zip(feature_names, gains)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if lgb is None:
        print(json.dumps({"ok": False, "error": "lightgbm is required"}, ensure_ascii=False))
        return 2

    config = json.loads(args.config.read_text(encoding="utf-8"))
    experiment_root = args.config.resolve().parent
    artifacts_root = experiment_root / "artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    started = time.time()

    base_rows = load_node_table(Path(config["r0_node_table"]))
    node_ids = {row["node_id"] for row in base_rows}
    run_ids = {row["run_id"] for row in base_rows}
    proxies = extract_trace_proxies(config["raw_roots"], node_ids, run_ids)
    task_fields = load_run_task_fields(config)
    video_meta = load_video_metadata(config)
    rows, coverage = merge_rows(base_rows, proxies, task_fields, video_meta)
    print(f"rows={len(rows)} coverage={json.dumps(coverage, sort_keys=True)}")

    split_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        split_rows[row["split"]].append(row)
    train_rows = split_rows["train"]
    val_rows = split_rows["validation"]
    test_rows = split_rows["test"]

    # Stage 0 checks
    errors = []
    if len(rows) != 15481:
        errors.append(f"rows {len(rows)} != 15481")
    if "runtime_ms" in set().union(*[set(config["feature_sets"][name]) for name in config["feature_sets"]]):
        errors.append("runtime_ms leaked into a feature set")
    audit = {
        "ok": not errors,
        "errors": errors,
        "rows": len(rows),
        "splits": {name: len(split_rows[name]) for name in ("train", "validation", "test", "holdout")},
        "coverage": coverage,
        "holdout_used": False,
        "provenance_rows": len(video_meta),
    }
    (experiment_root / "stage0_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False))
        return 1

    joined_path = artifacts_root / "joined_table.jsonl.gz"
    with gzip.open(joined_path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

    results: Dict[str, Any] = {}
    for name, feature_names in config["feature_sets"].items():
        output = select_and_evaluate(train_rows, val_rows, test_rows, feature_names, config)
        results[name] = output
        print(f"{name}: selected={output['selected']} val_PB={output['splits']['validation']['PB_primary']:.1f} test_PB={output['splits']['test']['PB_primary']:.1f}")

    f0 = results["F0_r0_c4"]
    comparisons = {}
    for name in results:
        comparisons[name] = {}
        for split in ("validation", "test"):
            base_pred = f0["splits"][split]["prediction"]
            model_pred = results[name]["splits"][split]["prediction"]
            positive = np.full_like(base_pred, 1.0)
            improvement = 1.0 - results[name]["splits"][split]["PB_primary"] / f0["splits"][split]["PB_primary"]
            comparisons[name][split] = {"improvement_vs_F0": float(improvement)}

    importance = feature_importance(train_rows, config["feature_sets"]["F2_full_proxies"], config, config["lgbm_quantile"]["seeds"][0])

    metrics = {
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(args.config),
        "r0_node_table": config["r0_node_table"],
        "r0_metrics": config["r0_metrics"],
        "holdout_used": False,
        "stage0": audit,
        "feature_sets": {name: list(names) for name, names in config["feature_sets"].items()},
        "results": {
            name: {
                "selected": output["selected"],
                "splits": {
                    split: {
                        "PB_primary": output["splits"][split]["PB_primary"],
                        "mae_q50": output["splits"][split]["mae_q50"],
                        "max_abs_calibration_error": output["splits"][split]["max_abs_calibration_error"],
                    }
                    for split in ("train", "validation", "test")
                },
            }
            for name, output in results.items()
        },
        "comparisons_vs_F0": comparisons,
        "feature_importance_gain_F2": importance,
        "run_seconds": round(time.time() - started, 2),
    }
    metrics_path = experiment_root / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    manifest = {
        "experiment_id": config["experiment_id"],
        "command": " ".join(sys.argv),
        "lightgbm_version": getattr(lgb, "__version__", "unknown"),
        "config_sha256": sha256_file(args.config),
        "joined_table_sha256": sha256_file(joined_path),
        "metrics_sha256": sha256_file(metrics_path),
        "rows": len(rows),
        "run_seconds": metrics["run_seconds"],
    }
    (experiment_root / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({"ok": True, "metrics": str(metrics_path), "run_seconds": metrics["run_seconds"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
