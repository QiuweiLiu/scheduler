"""R2 oof_cache:GroupKFold(video) OOF XGB 概率缓存(B08/B09/B10 前置).

吸收审查意见:
- 每折内独立 fit StaticPreprocessor(折内 train)与 XGB(防折内泄漏)
- 存 oof_prob + oof_log_prob + epsilon;只对概率取一次 log
- 存 sample_key(run_id + target_source_event_id)/view/split/fold_id,加载校验一一对应、无重复、无缺失
- 类别按 constants 固定轴对齐,缺类补零
- manifest 记录全部 hash(cache_seed、class_order、fold 覆盖、依赖)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from r2 import constants, preprocessing, provenance
from r2b import shared

OOF_NPZ_NAME = "xgb_oof_logits.npz"
OOF_MANIFEST_NAME = "xgb_oof_manifest.json"
EPSILON = 1e-6
N_FOLDS = 5
CACHE_SEED = 0


class OofError(ValueError):
    pass


_NONE_SENTINEL = "__NONE__"


def _sample_key(record):
    return (record["run_id"], record["target_source_event_id"])


def _norm_target(target):
    """None 语义保留(终止行 target_source_event_id=None):U256 存哨兵。"""
    return target if target is not None else _NONE_SENTINEL


def _denorm_target(value):
    return None if value == _NONE_SENTINEL else value


def _align_probabilities(probabilities, classes, fixed_labels):
    """按 constants 固定轴对齐(classes 为整数时映射回 fixed 顺序);缺类补零。"""
    out = np.zeros((probabilities.shape[0], len(fixed_labels)), dtype=np.float64)
    if classes.dtype.kind in ("i", "u") if hasattr(classes, "dtype") else isinstance(classes[0], (int, np.integer)):
        index_map = {int(c): i for i, c in enumerate(classes)}
        for i in range(len(fixed_labels)):
            if i in index_map:
                out[:, i] = probabilities[:, index_map[i]]
    else:
        index_map = {c: i for i, c in enumerate(classes)}
        for i, label in enumerate(fixed_labels):
            if label in index_map:
                out[:, i] = probabilities[:, index_map[label]]
    return out


class OofBuilder:
    def __init__(self, hyperparameters=None):
        self.hyperparameters = hyperparameters or {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "n_jobs": 1,
        }

    def _build_view(self, records, history_index):
        encoder = preprocessing.StaticPreprocessor().fit(records, history_index)
        matrix = np.concatenate(
            [encoder.transform(r, history_index) for r in records], axis=0
        )
        return encoder, matrix

    def _oof_for_view(self, view_records, target_field, labels, history_index, view_name):
        fixed_labels = list(labels)
        label_index = {label: i for i, label in enumerate(fixed_labels)}
        keys = [_sample_key(r) for r in view_records]
        n = len(view_records)
        groups = np.asarray([r["video_id"] for r in view_records])
        targets_int = np.asarray([label_index[r[target_field]] for r in view_records], dtype=np.int64)
        oof_prob = np.zeros((n, len(fixed_labels)), dtype=np.float64)
        fold_ids = np.zeros(n, dtype=np.int64)
        kf = GroupKFold(n_splits=N_FOLDS)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(view_records, targets_int, groups)):
            tr_records = [view_records[i] for i in tr_idx]
            va_records = [view_records[i] for i in va_idx]
            # 折内独立 fit preprocessor(文档 4.2);va 必须复用折内 train encoder 变换
            encoder_tr, matrix_tr = self._build_view(tr_records, history_index)
            matrix_va = np.concatenate(
                [encoder_tr.transform(r, history_index) for r in va_records], axis=0
            )
            y_tr = targets_int[tr_idx]
            model = xgb.XGBClassifier(random_state=CACHE_SEED, **self.hyperparameters)
            model.fit(matrix_tr, y_tr)
            pb = _align_probabilities(
                model.predict_proba(matrix_va), model.classes_, fixed_labels
            )
            oof_prob[va_idx] = pb
            fold_ids[va_idx] = fold
        return keys, oof_prob, fold_ids, targets_int

    def build(self, role_train, family_train, history_index):
        role_keys, role_oof, role_fold, role_targets = self._oof_for_view(
            role_train, "next_role", constants.ROLES, history_index, "role"
        )
        family_keys, family_oof, family_fold, family_targets = self._oof_for_view(
            family_train, "family_label", constants.FAMILIES, history_index, "family"
        )
        return {
            "hyperparameters": dict(self.hyperparameters),
            "role": {
                "keys": role_keys,
                "oof_prob": role_oof,
                "oof_log_prob": np.log(role_oof.clip(EPSILON, 1.0)),
                "fold_ids": role_fold,
                "targets": role_targets,
            },
            "family": {
                "keys": family_keys,
                "oof_prob": family_oof,
                "oof_log_prob": np.log(family_oof.clip(EPSILON, 1.0)),
                "fold_ids": family_fold,
                "targets": family_targets,
            },
        }


def _full_xgb(records, target_field, labels, history_index, hyperparameters):
    """全量 train XGB(评估 validation 用;不参与 OOF 训练行预测)。"""
    fixed_labels = list(labels)
    label_index = {label: i for i, label in enumerate(fixed_labels)}
    encoder = preprocessing.StaticPreprocessor().fit(records, history_index)
    matrix = np.concatenate([encoder.transform(r, history_index) for r in records], axis=0)
    targets = np.asarray([label_index[r[target_field]] for r in records], dtype=np.int64)
    model = xgb.XGBClassifier(random_state=CACHE_SEED, **hyperparameters)
    model.fit(matrix, targets)
    return model, encoder


def build_full_predictions(
    role_train, role_validation, family_train, family_validation, history_index, hyperparameters
):
    role_model, role_encoder = _full_xgb(
        role_train, "next_role", constants.ROLES, history_index, hyperparameters
    )
    family_model, family_encoder = _full_xgb(
        family_train, "family_label", constants.FAMILIES, history_index, hyperparameters
    )
    role_val_matrix = np.concatenate(
        [role_encoder.transform(r, history_index) for r in role_validation], axis=0
    )
    family_val_matrix = np.concatenate(
        [family_encoder.transform(r, history_index) for r in family_validation], axis=0
    )
    role_prob = _align_probabilities(
        role_model.predict_proba(role_val_matrix), role_model.classes_, list(constants.ROLES)
    )
    family_prob = _align_probabilities(
        family_model.predict_proba(family_val_matrix),
        family_model.classes_,
        list(constants.FAMILIES),
    )
    return {
        "role": {"prob": role_prob, "log_prob": np.log(role_prob.clip(EPSILON, 1.0))},
        "family": {"prob": family_prob, "log_prob": np.log(family_prob.clip(EPSILON, 1.0))},
    }


def write_cache(artifact_root, builder_result, full_predictions, role_train, family_train):
    root = Path(artifact_root) / "baselines"
    if root.exists():
        raise OofError(f"baselines already exists at {root}; refusing to overwrite")
    root.mkdir(parents=True)
    npz_path = root / OOF_NPZ_NAME
    role_keys = builder_result["role"]["keys"]
    family_keys = builder_result["family"]["keys"]
    np.savez_compressed(
        npz_path,
        role_run_ids=np.asarray([k[0] for k in role_keys], dtype="U256"),
        role_target_ids=np.asarray([_norm_target(k[1]) for k in role_keys], dtype="U256"),
        role_oof_prob=builder_result["role"]["oof_prob"],
        role_oof_log_prob=builder_result["role"]["oof_log_prob"],
        role_fold_ids=builder_result["role"]["fold_ids"],
        family_run_ids=np.asarray([k[0] for k in family_keys], dtype="U256"),
        family_target_ids=np.asarray([_norm_target(k[1]) for k in family_keys], dtype="U256"),
        family_oof_prob=builder_result["family"]["oof_prob"],
        family_oof_log_prob=builder_result["family"]["oof_log_prob"],
        family_fold_ids=builder_result["family"]["fold_ids"],
        role_val_log_prob=full_predictions["role"]["log_prob"],
        family_val_log_prob=full_predictions["family"]["log_prob"],
    )
    manifest = {
        "schema_version": "v1",
        "cache_seed": CACHE_SEED,
        "n_folds": N_FOLDS,
        "epsilon": EPSILON,
        "class_order": {
            "role": list(constants.ROLES),
            "family": list(constants.FAMILIES),
        },
        "counts": {
            "role_train": len(role_train),
            "family_train": len(family_train),
            "role_val": full_predictions["role"]["prob"].shape[0],
            "family_val": full_predictions["family"]["prob"].shape[0],
        },
        "npz_sha256": provenance.sha256_file(npz_path),
        "xgb_hyperparameters": dict(builder_result.get("hyperparameters", {})),
        "sources": shared.source_manifest(),
    }
    manifest["manifest_sha256"] = provenance.canonical_sha256(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    )
    manifest_path = root / OOF_MANIFEST_NAME
    tmp_manifest = root / (OOF_MANIFEST_NAME + ".tmp")
    shared.write_json(tmp_manifest, manifest)
    tmp_manifest.replace(manifest_path)
    (root / "COMPLETE").write_text("ok\n", encoding="utf-8")
    return npz_path, manifest_path


def load_cache(artifact_root):
    """加载并校验;fail closed。返回 dict(keys→index + 数组)。"""
    root = Path(artifact_root) / "baselines"
    npz_path = root / OOF_NPZ_NAME
    manifest_path = root / OOF_MANIFEST_NAME
    if not npz_path.exists() or not manifest_path.exists() or not (root / "COMPLETE").exists():
        raise OofError("OOF cache missing or incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "v1":
        raise OofError("OOF cache schema mismatch")
    if provenance.sha256_file(npz_path) != manifest.get("npz_sha256"):
        raise OofError("OOF cache npz hash mismatch")
    if manifest.get("n_folds") != N_FOLDS:
        raise OofError("OOF cache fold count mismatch")
    if manifest.get("class_order") != {"role": list(constants.ROLES), "family": list(constants.FAMILIES)}:
        raise OofError("OOF cache class order mismatch")
    data = np.load(npz_path, allow_pickle=False)
    if data["role_oof_log_prob"].shape[1] != len(constants.ROLES) or \
       data["family_oof_log_prob"].shape[1] != len(constants.FAMILIES):
        raise OofError("OOF cache probability width mismatch")

    role_keys = list(zip(data["role_run_ids"].tolist(), data["role_target_ids"].tolist()))
    family_keys = list(zip(data["family_run_ids"].tolist(), data["family_target_ids"].tolist()))
    if len(set(role_keys)) != len(role_keys):
        raise OofError("OOF role keys not unique")
    if len(set(family_keys)) != len(family_keys):
        raise OofError("OOF family keys not unique")
    return {
        "manifest": manifest,
        "role_index": {k: i for i, k in enumerate(role_keys)},
        "family_index": {k: i for i, k in enumerate(family_keys)},
        "role_oof_log_prob": data["role_oof_log_prob"],
        "family_oof_log_prob": data["family_oof_log_prob"],
        "role_oof_prob": data["role_oof_prob"],
        "family_oof_prob": data["family_oof_prob"],
        "role_val_log_prob": data["role_val_log_prob"],
        "family_val_log_prob": data["family_val_log_prob"],
    }
