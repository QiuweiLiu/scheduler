"""R2 B09 XGB+MLP residual / B10 XGB+GRU residual.

最终 logits = base_log_prob(XGB)+ residual logits;训练时 base 来自 OOF(GroupKFold,
不含自身 video,防泄漏),评估 validation 用全量 XGB log_prob(不参与训练,无泄漏)。
梯度只走 NN(base 冻结)。seed 11/22/33。
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from r2 import constants, evaluation, provenance, run_artifacts
from r2b import oof_cache, shared

MODEL_IDS = ("B09", "B10")
DEFAULT_HYPERPARAMETERS = {
    "hidden": 128,
    "dropout": 0.1,
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "batch_size": 64,
    "max_epochs": 100,
    "patience": 15,
}
EMB_DIM = 32

SOURCE_PATHS = {
    "constants.py": constants.__file__,
    "evaluation.py": evaluation.__file__,
    "preprocessing.py": __import__("r2.preprocessing", fromlist=["x"]).__file__,
    "oof_cache.py": oof_cache.__file__,
    "models_seq.py": __import__("r2b.models_seq", fromlist=["x"]).__file__,
    "provenance.py": provenance.__file__,
    "run_artifacts.py": run_artifacts.__file__,
    "shared.py": shared.__file__,
    "models_residual.py": __file__,
}


def build_residual_config(experiment_id, model_id, seed, mode="formal", hyperparameter_overrides=None):
    if model_id not in MODEL_IDS:
        raise shared.ModelError(f"model_id must be one of {MODEL_IDS}")
    return shared.build_nn_config(
        experiment_id, model_id, seed, mode, DEFAULT_HYPERPARAMETERS, hyperparameter_overrides
    )


class ResidualMLP(nn.Module):
    def __init__(self, static_dim, out_dim, hidden, dropout):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(static_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, batch):
        return self.head(batch["static"])


class ResidualGRU(nn.Module):
    def __init__(self, vocab_sizes, static_dim, out_dim, hidden, dropout):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(v, EMB_DIM, padding_idx=0) for v in vocab_sizes])
        self.rnn = nn.GRU(EMB_DIM * 4, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + static_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, batch):
        tokens = batch["tokens"]
        lengths = batch["lengths"]
        channels = [emb(tokens[..., c]) for c, emb in enumerate(self.embs)]
        e = torch.cat(channels, dim=-1)
        valid = lengths > 0
        l_safe = lengths.clamp(min=1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            e, l_safe.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.rnn(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        idx = (l_safe - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, out.shape[2])
        last = out.gather(1, idx).squeeze(1)
        last = torch.where(valid.unsqueeze(1), last, torch.zeros_like(last))
        return self.head(torch.cat([last, batch["static"]], dim=1))


def _dataset(X, base_log_prob, targets):
    if "tokens" in X:
        return torch.utils.data.TensorDataset(
            torch.from_numpy(X["static"]),
            torch.from_numpy(X["tokens"]),
            torch.from_numpy(X["lengths"]),
            torch.from_numpy(X["mask"]),
            torch.from_numpy(base_log_prob),
            torch.from_numpy(targets),
        )
    return torch.utils.data.TensorDataset(
        torch.from_numpy(X["static"]),
        torch.from_numpy(base_log_prob),
        torch.from_numpy(targets),
    )


def _train_epoch(model, optimizer, loader, device, base_field="base"):
    model.train()
    total_loss = 0.0
    total_rows = 0
    for batch in loader:
        static = batch[0].to(device)
        base = batch[-2].to(device)
        targets = batch[-1].to(device)
        optimizer.zero_grad(set_to_none=True)
        if len(batch) == 6:  # 序列版
            tokens, lengths, mask = batch[1].to(device), batch[2].to(device), batch[3].to(device)
            out = model({"static": static, "tokens": tokens, "lengths": lengths, "mask": mask})
        else:
            out = model({"static": static})
        loss = functional.cross_entropy(base + out, targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(targets)
        total_rows += len(targets)
    return total_loss / total_rows


def _predict(model, X, base_log_prob, batch_size, device):
    model.eval()
    chunks = []
    n = X["static"].shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            sl = slice(start, start + batch_size)
            static = torch.from_numpy(X["static"][sl]).to(device)
            base = torch.from_numpy(base_log_prob[sl]).to(device)
            if "tokens" in X:
                batch = {
                    "static": static,
                    "tokens": torch.from_numpy(X["tokens"][sl]).to(device),
                    "lengths": torch.from_numpy(X["lengths"][sl]).to(device),
                    "mask": torch.from_numpy(X["mask"][sl]).to(device),
                }
            else:
                batch = {"static": static}
            chunks.append(torch.softmax(base + model(batch), dim=1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def run_validation_residual(artifact_root, config, model_id, device=None):
    params = shared._validate_hyperparameters(
        model_id, config["hyperparameters"], config["mode"], DEFAULT_HYPERPARAMETERS
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
    cache = oof_cache.load_cache(artifact_root)

    from r2.preprocessing import StaticPreprocessor

    role_static = StaticPreprocessor().fit(role_train, history_index)
    family_static = StaticPreprocessor().fit(family_train, history_index)
    role_tr = _static_matrix(role_train, role_static, history_index)
    role_va = _static_matrix(role_validation, role_static, history_index)
    fam_tr = _static_matrix(family_train, family_static, history_index)
    fam_va = _static_matrix(family_validation, family_static, history_index)

    # OOF log_prob 按 sample_key 对齐 train 行;validation 用全量 XGB log_prob
    role_tr_base = _gather_log_prob(cache, "role", role_train)
    fam_tr_base = _gather_log_prob(cache, "family", family_train)
    role_va_base = cache["role_val_log_prob"]
    fam_va_base = cache["family_val_log_prob"]

    role_targets = _targets(role_train, "next_role", constants.ROLES)
    family_targets = _targets(family_train, "family_label", constants.FAMILIES)

    if model_id == "B09":
        role_model = ResidualMLP(
            role_tr.shape[1], len(constants.ROLES), params["hidden"], params["dropout"]
        ).to(resolved_device)
        family_model = ResidualMLP(
            fam_tr.shape[1], len(constants.FAMILIES), params["hidden"], params["dropout"]
        ).to(resolved_device)
        role_enc = {"static": role_tr}
        fam_enc = {"static": fam_tr}
        role_enc_va = {"static": role_va}
        fam_enc_va = {"static": fam_va}
    else:
        from r2b.models_seq import shared_seq_prep, seq_vocab_sizes

        role_seq = shared_seq_prep(role_train, history_index)
        family_seq = shared_seq_prep(family_train, history_index)
        role_vocab = seq_vocab_sizes(role_seq, role_train, history_index)
        family_vocab = seq_vocab_sizes(family_seq, family_train, history_index)
        role_model = ResidualGRU(
            role_vocab, role_tr.shape[1], len(constants.ROLES), params["hidden"], params["dropout"]
        ).to(resolved_device)
        family_model = ResidualGRU(
            family_vocab, fam_tr.shape[1], len(constants.FAMILIES), params["hidden"], params["dropout"]
        ).to(resolved_device)
        role_enc = _seq_view(role_train, role_seq, history_index, role_static)
        fam_enc = _seq_view(family_train, family_seq, history_index, family_static)
        role_enc_va = _seq_view(role_validation, role_seq, history_index, role_static)
        fam_enc_va = _seq_view(family_validation, family_seq, history_index, family_static)

    role_optimizer = torch.optim.AdamW(
        role_model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]
    )
    family_optimizer = torch.optim.AdamW(
        family_model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]
    )
    role_loader = torch.utils.data.DataLoader(
        _dataset(role_enc, role_tr_base, role_targets),
        batch_size=params["batch_size"], shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(config["seed"] + 1),
    )
    family_loader = torch.utils.data.DataLoader(
        _dataset(fam_enc, fam_tr_base, family_targets),
        batch_size=params["batch_size"], shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(config["seed"] + 2),
    )

    best_joint = -1.0
    best_epoch = 0
    best_role_state = None
    best_family_state = None
    best_losses = None
    for epoch in range(1, params["max_epochs"] + 1):
        role_loss = _train_epoch(role_model, role_optimizer, role_loader, resolved_device)
        family_loss = _train_epoch(family_model, family_optimizer, family_loader, resolved_device)
        role_rows = shared.prediction_rows(
            role_validation, "role",
            _predict(role_model, role_enc_va, role_va_base, params["batch_size"], resolved_device),
        )
        family_rows = shared.prediction_rows(
            family_validation, "family",
            _predict(family_model, fam_enc_va, fam_va_base, params["batch_size"], resolved_device),
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
        _predict(role_model, role_enc_va, role_va_base, params["batch_size"], resolved_device),
    )
    family_rows = shared.prediction_rows(
        family_validation, "family",
        _predict(family_model, fam_enc_va, fam_va_base, params["batch_size"], resolved_device),
    )
    metrics = evaluation.evaluate(role_rows, family_rows)
    run_id = run_artifacts.derive_run_id(config)
    metrics["run"] = {
        "run_id": run_id, "model_id": model_id, "mode": mode,
        "hyperparameters": params, "fit_splits": ["train"],
        "evaluation_splits": ["validation"],
        "selection_metric": config["selection_metric"], "best_epoch": best_epoch,
        "best_validation_joint_accuracy": best_joint,
        "best_epoch_train_loss": best_losses,
        "oof_cache": {"manifest_sha256": cache["manifest"].get("manifest_sha256"),
                      "npz_sha256": cache["manifest"].get("npz_sha256"),
                      "npz_path": "baselines/xgb_oof_logits.npz"},
        "status": "success",
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
            {"role": {"dimension": role_tr.shape[1]}, "family": {"dimension": fam_tr.shape[1]}},
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
            "config": config, "model_id": model_id,
            "role_state_dict": best_role_state, "family_state_dict": best_family_state,
            "role_static_dim": role_tr.shape[1],
            "family_static_dim": fam_tr.shape[1],
        },
        staging / "checkpoint.pt",
    )
    (staging / "run.log").write_text(
        "\n".join(
            (
                f"{model_id} residual validation-only {mode}",
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


def _static_matrix(records, encoder, history_index):
    return np.concatenate([encoder.transform(r, history_index) for r in records], axis=0)


def _seq_view(records, seq_prep, history_index, static_encoder):
    static = np.concatenate([static_encoder.transform(r, history_index) for r in records], axis=0)
    tokens = np.zeros((len(records), seq_prep.max_len, 4), dtype=np.int64)
    lengths = np.zeros(len(records), dtype=np.int64)
    mask = np.zeros((len(records), seq_prep.max_len), dtype=bool)
    for i, r in enumerate(records):
        t, m, l, _ = seq_prep.transform(r, history_index)
        tokens[i] = t[0]
        lengths[i] = int(l[0])
        mask[i] = m[0]
    return {"static": static, "tokens": tokens, "lengths": lengths, "mask": mask}


def _gather_log_prob(cache, view, records):
    index = cache[f"{view}_index"]
    arr = cache[f"{view}_oof_log_prob"]
    out = np.zeros((len(records), arr.shape[1]), dtype=np.float64)
    for i, r in enumerate(records):
        key = (r["run_id"], r["target_source_event_id"])
        if key not in index:
            raise shared.ModelError(f"OOF cache missing key for {view} row {i}")
        out[i] = arr[index[key]]
    return out


def _targets(records, target_field, labels):
    index = {label: i for i, label in enumerate(labels)}
    return np.asarray([index[r[target_field]] for r in records], dtype=np.int64)
