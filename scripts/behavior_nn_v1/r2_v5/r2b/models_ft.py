"""R2 B04 FT-Transformer v2(静态输入)。

标量×token + 数值 bias + 2 attention block + CLS 池化(接近论文结构,非逐字复刻)。
role/family 独立模型;seed 11/22/33。
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from r2 import constants, evaluation, provenance, run_artifacts
from r2b import shared

MODEL_ID = "B04"
DEFAULT_HYPERPARAMETERS = {
    "d_model": 128,
    "dropout": 0.1,
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "batch_size": 64,
    "max_epochs": 100,
    "patience": 15,
}
NUMERIC_COUNT = 12
N_LAYERS = 2
N_HEADS = 4

SOURCE_PATHS = {
    "constants.py": constants.__file__,
    "data.py": __import__("r2.data", fromlist=["x"]).__file__,
    "evaluation.py": evaluation.__file__,
    "preprocessing.py": __import__("r2.preprocessing", fromlist=["x"]).__file__,
    "provenance.py": provenance.__file__,
    "run_artifacts.py": run_artifacts.__file__,
    "shared.py": shared.__file__,
    "models_ft.py": __file__,
}


def build_ft_config(experiment_id, seed, mode="formal", hyperparameter_overrides=None):
    return shared.build_nn_config(
        experiment_id, MODEL_ID, seed, mode, DEFAULT_HYPERPARAMETERS, hyperparameter_overrides
    )


class FTTransformer(nn.Module):
    def __init__(self, input_dimension, out_dimension, d_model, dropout):
        super().__init__()
        self.d_model = d_model
        self.tok = nn.Parameter(torch.randn(input_dimension, d_model) * 0.02)
        self.num_bias = nn.Parameter(torch.randn(NUMERIC_COUNT, d_model) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=N_HEADS, dim_feedforward=2 * d_model,
            dropout=dropout, batch_first=True, activation="relu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.head = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, out_dimension),
        )

    def forward(self, values):
        t = values.unsqueeze(2) * self.tok.unsqueeze(0)
        t[:, :NUMERIC_COUNT, :] = t[:, :NUMERIC_COUNT, :] + self.num_bias.unsqueeze(0)
        cls = self.cls.expand(values.shape[0], -1, -1)
        t = torch.cat([cls, t], dim=1)
        out = self.encoder(t)
        return self.head(out[:, 0])


def run_validation_ft(artifact_root, config, device=None):
    params = shared._validate_hyperparameters(
        MODEL_ID, config["hyperparameters"], config["mode"], DEFAULT_HYPERPARAMETERS
    )
    mode = config["mode"]
    resolved_device = shared.resolve_device(mode, device)
    shared.seed_everything(config["seed"], resolved_device)
    if resolved_device.type == "cuda":
        torch.empty(0, device=resolved_device)
        torch.cuda.reset_peak_memory_stats(
            resolved_device.index if resolved_device.index is not None
            else torch.cuda.current_device()
        )
    runtime_start = provenance.start_runtime()
    (role_train, role_validation, family_train, family_validation, history_index) = (
        shared.load_train_validation_only()
    )
    shared.require_split(role_train, "train", "role training records")
    shared.require_split(family_train, "train", "family training records")

    from r2.preprocessing import StaticPreprocessor

    role_encoder = StaticPreprocessor().fit(role_train, history_index)
    family_encoder = StaticPreprocessor().fit(family_train, history_index)
    role_tr = _matrix(role_train, role_encoder, history_index)
    role_va = _matrix(role_validation, role_encoder, history_index)
    fam_tr = _matrix(family_train, family_encoder, history_index)
    fam_va = _matrix(family_validation, family_encoder, history_index)
    role_targets = _targets(role_train, "next_role", constants.ROLES)
    family_targets = _targets(family_train, "family_label", constants.FAMILIES)

    role_model = FTTransformer(
        role_tr.shape[1], len(constants.ROLES), params["d_model"], params["dropout"]
    ).to(resolved_device)
    family_model = FTTransformer(
        fam_tr.shape[1], len(constants.FAMILIES), params["d_model"], params["dropout"]
    ).to(resolved_device)
    role_optimizer = torch.optim.AdamW(
        role_model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]
    )
    family_optimizer = torch.optim.AdamW(
        family_model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]
    )
    role_loader = shared.make_loader(role_tr, role_targets, params["batch_size"], config["seed"] + 1)
    family_loader = shared.make_loader(fam_tr, family_targets, params["batch_size"], config["seed"] + 2)

    best_joint = -1.0
    best_epoch = 0
    best_role_state = None
    best_family_state = None
    best_losses = None
    for epoch in range(1, params["max_epochs"] + 1):
        role_loss = shared.train_epoch(role_model, role_optimizer, role_loader, resolved_device)
        family_loss = shared.train_epoch(family_model, family_optimizer, family_loader, resolved_device)
        role_rows = shared.prediction_rows(
            role_validation, "role",
            shared.probabilities(role_model, role_va, params["batch_size"], resolved_device),
        )
        family_rows = shared.prediction_rows(
            family_validation, "family",
            shared.probabilities(family_model, fam_va, params["batch_size"], resolved_device),
        )
        joint = evaluation.evaluate(role_rows, family_rows)["joint"]["joint_accuracy"]
        if joint > best_joint:
            best_joint = joint
            best_epoch = epoch
            best_role_state = shared.cpu_state_dict(role_model)
            best_family_state = shared.cpu_state_dict(family_model)
            best_losses = {"role": role_loss, "family": family_loss}
        if epoch - best_epoch >= params["patience"]:
            break

    if best_role_state is None or best_family_state is None:
        raise shared.ModelError("no validation checkpoint produced")
    role_model.load_state_dict(best_role_state)
    family_model.load_state_dict(best_family_state)
    role_rows = shared.prediction_rows(
        role_validation, "role",
        shared.probabilities(role_model, role_va, params["batch_size"], resolved_device),
    )
    family_rows = shared.prediction_rows(
        family_validation, "family",
        shared.probabilities(family_model, fam_va, params["batch_size"], resolved_device),
    )
    metrics = evaluation.evaluate(role_rows, family_rows)
    run_id = run_artifacts.derive_run_id(config)
    metrics["run"] = {
        "run_id": run_id, "model_id": MODEL_ID, "mode": mode,
        "hyperparameters": params, "fit_splits": ["train"],
        "evaluation_splits": ["validation"],
        "selection_metric": config["selection_metric"], "best_epoch": best_epoch,
        "best_validation_joint_accuracy": best_joint,
        "best_epoch_train_loss": best_losses, "status": "success",
    }
    environment = provenance.build_environment(
        {"numpy": np.__version__, "torch": torch.__version__}, SOURCE_PATHS, runtime_start
    )
    environment["device"] = shared.device_facts(resolved_device)
    staging = run_artifacts.begin_run(artifact_root, config)
    shared.write_json(staging / "environment.json", environment)
    shared.write_json(
        staging / "dataset_manifest.json",
        shared.dataset_manifest(
            role_train, role_validation, family_train, family_validation,
            {"role": shared.feature_info(role_encoder), "family": shared.feature_info(family_encoder)},
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
            "model_id": MODEL_ID,
            "role_state_dict": best_role_state,
            "family_state_dict": best_family_state,
            "role_feature_dimension": role_tr.shape[1],
            "family_feature_dimension": fam_tr.shape[1],
        },
        staging / "checkpoint.pt",
    )
    (staging / "run.log").write_text(
        "\n".join(
            (
                f"{MODEL_ID} FT-Transformer validation-only {mode}",
                f"run_id={run_id}", f"seed={config['seed']}", f"device={resolved_device}",
                f"role_train={len(role_train)}", f"role_validation={len(role_validation)}",
                f"family_train={len(family_train)}", f"family_validation={len(family_validation)}",
                "test_targets_used=False", "test_rows_predicted=0",
                f"best_epoch={best_epoch}",
                f"best_validation_joint_accuracy={best_joint}",
                f"elapsed_seconds={environment['runtime']['elapsed_seconds']}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return run_artifacts.finalize_run(artifact_root, config, staging)


def _matrix(records, encoder, history_index):
    return np.concatenate([encoder.transform(r, history_index) for r in records], axis=0)


def _targets(records, target_field, labels):
    index = {label: i for i, label in enumerate(labels)}
    return np.asarray([index[r[target_field]] for r in records], dtype=np.int64)
