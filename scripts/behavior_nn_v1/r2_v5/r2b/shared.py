"""R2 B04-B10 shared scaffolding(复用 B03 模板,抽取共享生命周期).

吸收审查意见:
- 统一 prediction contract(row keys/固定类别轴/概率有限性)
- runs 独立命名空间(artifact_root 由执行方传入 behavior_nn_v1_r2/)
- config 契约复用 run_artifacts.validate_run_config,超参扩展在各模型模块内精确校验
- formal 必须 CUDA;smoke 可 CPU
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

from r2 import constants, data, evaluation, preprocessing, provenance, run_artifacts

HEAD_SPECS = {
    "role": ("next_role", constants.ROLES),
    "family": ("family_label", constants.FAMILIES),
}


class ModelError(ValueError):
    """Raised when a B04-B10 contract is violated."""


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelError(f"{name} must be a positive integer")
    return value


def _positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelError(f"{name} must be a finite positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ModelError(f"{name} must be a finite positive number")
    return number


def _dropout(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelError("dropout must be a finite number in [0, 1)")
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number >= 1.0:
        raise ModelError("dropout must be a finite number in [0, 1)")
    return number


def _validate_hyperparameters(model_id, hyperparameters, mode, defaults):
    if not isinstance(hyperparameters, dict) or set(hyperparameters) != set(defaults):
        raise ModelError(
            f"{model_id} hyperparameters must have exactly keys {sorted(defaults)}"
        )
    for name, value in hyperparameters.items():
        if name == "dropout":
            _dropout(value)
        elif name in ("max_epochs", "patience", "batch_size"):
            _positive_int(value, name)
        else:
            _positive_number(value, name)
    if mode == "formal" and hyperparameters != defaults:
        raise ModelError(f"formal {model_id} requires the locked default hyperparameters")
    return hyperparameters


def build_nn_config(experiment_id, model_id, seed, mode, defaults, hyperparameter_overrides=None):
    parameters = dict(defaults)
    if hyperparameter_overrides is not None:
        if not isinstance(hyperparameter_overrides, dict):
            raise ModelError("hyperparameter_overrides must be a dict")
        unknown = set(hyperparameter_overrides) - set(parameters)
        if unknown:
            raise ModelError(f"unknown hyperparameter override(s): {sorted(unknown)}")
        parameters.update(hyperparameter_overrides)
    config = {
        "experiment_id": experiment_id,
        "model_id": model_id,
        "mode": mode,
        "seed": seed,
        "selection_split": "validation",
        "selection_metric": "joint_pipeline_accuracy",
        "allow_test_selection": False,
        "feature_contract_version": "v5.1",
        "feature_mode": "transferable_main",
        "feature_flags": {
            "task_text": False,
            "visual": False,
            "provenance": False,
            "resource_history": False,
        },
        "hyperparameters": parameters,
    }
    run_artifacts.validate_run_config(config)
    _validate_hyperparameters(model_id, config["hyperparameters"], mode, defaults)
    return config


def require_split(records, expected, name):
    if not records:
        raise ModelError(f"{name} is empty")
    if any(record.get("split") != expected for record in records):
        raise ModelError(f"{name} must contain only split={expected!r}")
    return records


def load_train_validation_only():
    all_role = data.load_role_samples()
    all_family = data.load_tool_samples()
    role_train = [r for r in all_role if r["split"] == "train"]
    role_validation = [r for r in all_role if r["split"] == "validation"]
    family_train = [r for r in all_family if r["split"] == "train"]
    family_validation = [r for r in all_family if r["split"] == "validation"]
    history_index = preprocessing.build_history_index(role_train + role_validation)
    return role_train, role_validation, family_train, family_validation, history_index


def resolve_device(mode, requested_device):
    if requested_device is None:
        requested_device = "cuda:0"
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ModelError("requested CUDA but CUDA is unavailable")
    if mode == "formal" and device.type != "cuda":
        raise ModelError("formal mode requires an explicit CUDA device")
    return device


def seed_everything(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def device_facts(device):
    facts = {"type": device.type, "index": device.index, "torch_cuda_version": torch.version.cuda}
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        facts.update(
            {
                "name": torch.cuda.get_device_name(index),
                "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
            }
        )
    return facts


def prediction_rows(records, head, probabilities):
    require_split(records, "validation", f"{head} validation records")
    target_field, labels = HEAD_SPECS[head]
    if len(records) != len(probabilities):
        raise ModelError(f"{head} prediction count does not match records")
    if probabilities.shape[1] != len(labels):
        raise ModelError(
            f"{head} probability width {probabilities.shape[1]} != fixed labels {len(labels)}"
        )
    rows = []
    for record, probability in zip(records, probabilities.tolist()):
        if not all(math.isfinite(float(v)) for v in probability):
            raise ModelError(f"{head} prediction contains non-finite probability")
        row = {
            "head": head,
            "run_id": record["run_id"],
            "task_id": record["task_id"],
            "split": record["split"],
            "target_source_event_id": record["target_source_event_id"],
            "y_true": record[target_field],
            "probabilities": probability,
        }
        if head == "role":
            row["source_event_id"] = record["source_event_id"]
        else:
            row["prefix_id"] = record["prefix_id"]
        rows.append(row)
    return rows


def feature_info(encoder):
    return {
        "dimension": encoder.dimension,
        "feature_names_sha256": provenance.canonical_sha256(encoder.feature_names),
    }


def source_manifest():
    sources = {
        "role_event_samples": {
            "path": constants.ROLE_SAMPLES_PATH,
            "sha256": provenance.sha256_file(constants.ROLE_SAMPLES_PATH),
        },
        "semantic_tool_samples": {
            "path": constants.TOOL_SAMPLES_PATH,
            "sha256": provenance.sha256_file(constants.TOOL_SAMPLES_PATH),
        },
    }
    split_manifest_path = Path(constants.CANDIDATE_DATA_DIR) / "split_manifest.json"
    return {
        "dataset_sha256": provenance.canonical_sha256(sources),
        "split_manifest": {
            "path": str(split_manifest_path),
            "sha256": provenance.sha256_file(split_manifest_path),
        },
        "feature_contract": {
            "path": preprocessing.CONTRACT_PATH,
            "sha256": provenance.sha256_file(preprocessing.CONTRACT_PATH),
            "version": "v5.1",
        },
        "sources": sources,
    }


def dataset_manifest(role_train, role_validation, family_train, family_validation, info):
    manifest = source_manifest()
    manifest.update(
        {
            "fit_splits": ["train"],
            "evaluation_splits": ["validation"],
            "test_targets_used": False,
            "test_rows_predicted": 0,
            "history_index_role_rows": len(role_train) + len(role_validation),
            "counts": {
                "role": dict(constants.EXPECTED_ROLE_SPLITS),
                "family": dict(constants.EXPECTED_TOOL_SPLITS),
            },
            "actual_counts": {
                "role_train": len(role_train),
                "role_validation": len(role_validation),
                "family_train": len(family_train),
                "family_validation": len(family_validation),
            },
            "static_features": info,
        }
    )
    return manifest


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_predictions(path, role_rows, family_rows):
    lines = [
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
        for row in role_rows + family_rows
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def cpu_state_dict(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_epoch(model, optimizer, loader, device):
    model.train()
    total_loss = 0.0
    total_rows = 0
    for values, targets in loader:
        values = values.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = functional.cross_entropy(model(values), targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(targets)
        total_rows += len(targets)
    return total_loss / total_rows


def probabilities(model, matrix, batch_size, device):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(matrix), batch_size):
            values = torch.from_numpy(matrix[start : start + batch_size]).to(device)
            chunks.append(torch.softmax(model(values), dim=1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def make_loader(matrix, targets, batch_size, seed, shuffle=True):
    return DataLoader(
        TensorDataset(torch.from_numpy(matrix), torch.from_numpy(targets)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
