"""R2 B08 Group-OOF XGB + logistic meta(stacking,seed 0,确定性)。

meta 输入 = GroupKFold OOF 概率(不含自身 video,已缓存);fit 于 train OOF 概率,
评估 validation 用全量 XGB 概率(缓存中的 full log_prob 转回概率)。
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from r2 import constants, evaluation, provenance, run_artifacts
from r2b import oof_cache, shared

MODEL_ID = "B08"
DEFAULT_HYPERPARAMETERS = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
}

SOURCE_PATHS = {
    "constants.py": constants.__file__,
    "data.py": __import__("r2.data", fromlist=["x"]).__file__,
    "preprocessing.py": __import__("r2.preprocessing", fromlist=["x"]).__file__,
    "evaluation.py": evaluation.__file__,
    "oof_cache.py": oof_cache.__file__,
    "provenance.py": provenance.__file__,
    "run_artifacts.py": run_artifacts.__file__,
    "shared.py": shared.__file__,
    "models_meta.py": __file__,
}


def build_meta_config(experiment_id, mode="formal", hyperparameter_overrides=None):
    parameters = dict(DEFAULT_HYPERPARAMETERS)
    if hyperparameter_overrides is not None:
        if not isinstance(hyperparameter_overrides, dict):
            raise shared.ModelError("hyperparameter_overrides must be a dict")
        unknown = set(hyperparameter_overrides) - set(parameters)
        if unknown:
            raise shared.ModelError(f"unknown override(s): {sorted(unknown)}")
        parameters.update(hyperparameter_overrides)
    if mode == "formal" and parameters != DEFAULT_HYPERPARAMETERS:
        raise shared.ModelError("formal B08 requires the locked default hyperparameters")
    config = {
        "experiment_id": experiment_id,
        "model_id": MODEL_ID,
        "mode": mode,
        "seed": 0,
        "selection_split": "validation",
        "selection_metric": "joint_pipeline_accuracy",
        "allow_test_selection": False,
        "feature_contract_version": "v5.1",
        "feature_mode": "transferable_main",
        "feature_flags": {
            "task_text": False, "visual": False, "provenance": False, "resource_history": False,
        },
        "hyperparameters": parameters,
    }
    run_artifacts.validate_run_config(config)
    return config


def run_validation_meta(artifact_root, config):
    if config["hyperparameters"].get("solver") != "lbfgs":
        raise shared.ModelError("B08 solver must be 'lbfgs'")
    runtime_start = provenance.start_runtime()
    (role_train, role_validation, family_train, family_validation, _) = (
        shared.load_train_validation_only()
    )
    cache = oof_cache.load_cache(artifact_root)

    role_meta = _fit_meta(
        cache, "role", role_train, constants.ROLES, config["hyperparameters"]
    )
    family_meta = _fit_meta(
        cache, "family", family_train, constants.FAMILIES, config["hyperparameters"]
    )
    role_prob = _meta_predict(role_meta, cache["role_val_log_prob"], constants.ROLES)
    family_prob = _meta_predict(family_meta, cache["family_val_log_prob"], constants.FAMILIES)
    role_rows = shared.prediction_rows(role_validation, "role", role_prob)
    family_rows = shared.prediction_rows(family_validation, "family", family_prob)
    metrics = evaluation.evaluate(role_rows, family_rows)
    run_id = run_artifacts.derive_run_id(config)
    metrics["run"] = {
        "run_id": run_id, "model_id": MODEL_ID, "mode": config["mode"],
        "hyperparameters": config["hyperparameters"], "fit_splits": ["train"],
        "evaluation_splits": ["validation"],
        "selection_metric": config["selection_metric"], "best_epoch": None,
        "oof_cache": {"manifest_sha256": cache["manifest"].get("manifest_sha256"),
                      "npz_sha256": cache["manifest"].get("npz_sha256"),
                      "npz_path": "baselines/xgb_oof_logits.npz"},
        "status": "success",
    }
    environment = provenance.build_environment(
        {"numpy": np.__version__, "sklearn": __import__("sklearn").__version__},
        SOURCE_PATHS, runtime_start,
    )
    staging = run_artifacts.begin_run(artifact_root, config)
    shared.write_json(staging / "environment.json", environment)
    shared.write_json(
        staging / "dataset_manifest.json",
        shared.dataset_manifest(
            role_train, role_validation, family_train, family_validation,
            {"role": {"dimension": len(constants.ROLES)}, "family": {"dimension": len(constants.FAMILIES)}},
        ),
    )
    shared.write_json(staging / "metrics.json", metrics)
    shared.write_predictions(staging / "predictions.jsonl", role_rows, family_rows)
    shared.write_json(
        staging / "confusion_matrix.json",
        {
            "labels": metrics["labels"],
            "role": metrics["role"]["confusion_matrix"],
            "family_oracle_gate": metrics["family_oracle_gate"]["confusion_matrix"],
        },
    )
    torch.save(
        {
            "config": config,
            "role_meta": role_meta,
            "family_meta": family_meta,
        },
        staging / "checkpoint.pt",
    )
    (staging / "run.log").write_text(
        "\n".join(
            (
                f"{MODEL_ID} OOF-meta validation-only {config['mode']}",
                f"run_id={run_id}", "seed=0",
                f"role_train={len(role_train)}", f"role_validation={len(role_validation)}",
                f"family_train={len(family_train)}", f"family_validation={len(family_validation)}",
                "test_targets_used=False", "test_rows_predicted=0",
                f"elapsed_seconds={environment['runtime']['elapsed_seconds']}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return run_artifacts.finalize_run(artifact_root, config, staging)


def _fit_meta(cache, view, train_records, labels, hyperparameters):
    index = cache[f"{view}_index"]
    arr = cache[f"{view}_oof_log_prob"]
    X = np.zeros((len(train_records), arr.shape[1]), dtype=np.float64)
    for i, r in enumerate(train_records):
        key = (r["run_id"], r["target_source_event_id"])
        if key not in index:
            raise shared.ModelError(f"OOF cache missing key for {view} row {i}")
        X[i] = arr[index[key]]
    target_field = "next_role" if view == "role" else "family_label"
    label_index = {label: i for i, label in enumerate(labels)}
    y = np.asarray([label_index[r[target_field]] for r in train_records], dtype=np.int64)
    meta = LogisticRegression(
        C=hyperparameters["C"], max_iter=hyperparameters["max_iter"],
        solver=hyperparameters["solver"], tol=1e-4,
    )
    meta.fit(X, y)
    if not set(meta.classes_) <= set(range(len(labels))):
        raise shared.ModelError("meta classes out of fixed label range")
    return meta


def _meta_predict(meta, log_prob, fixed_labels):
    """meta 在 log 空间训练(标签为整数 0..K-1),预测同样用 log 输入;
    输出按整数 class index 对齐 fixed_labels,缺类补零。"""
    raw = meta.predict_proba(log_prob)
    out = np.zeros((raw.shape[0], len(fixed_labels)), dtype=np.float64)
    classes_int = [int(c) for c in meta.classes_]
    for i in range(len(fixed_labels)):
        if i in classes_int:
            out[:, i] = raw[:, classes_int.index(i)]
    row_sums = out.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise shared.ModelError("meta output probabilities do not sum to 1")
    return out
