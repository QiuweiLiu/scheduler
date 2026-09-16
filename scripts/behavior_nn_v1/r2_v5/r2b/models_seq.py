"""R2 B05 Masked GRU / B06 Masked LSTM / B07 Causal Transformer.

序列输入来自 SequencePreprocessor:tokens [B,T,4](4 通道 role/family/raw_action/status)+ mask + lengths。
- 每通道独立 embedding → concat → [B,T,d](审查 P0-01)
- RNN 用 pack_padded + gather 最后有效步;len=0 行输出零状态(审查 P0-02)
- Transformer 用 padding mask + 因果 mask;全 PAD 行短路(零状态,梯度有限)
- role/family 独立模型(与 B03 同构公平对比)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from r2 import constants, evaluation, provenance, run_artifacts
from r2b import shared

MODEL_IDS = ("B05", "B06", "B07")
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
NUMERIC_COUNT = 12  # StaticPreprocessor 数值字段数(数值 bias 用)

SOURCE_PATHS = {
    "constants.py": constants.__file__,
    "data.py": __import__("r2.data", fromlist=["x"]).__file__,
    "evaluation.py": evaluation.__file__,
    "preprocessing.py": __import__("r2.preprocessing", fromlist=["x"]).__file__,
    "provenance.py": provenance.__file__,
    "run_artifacts.py": run_artifacts.__file__,
    "shared.py": shared.__file__,
    "models_seq.py": __file__,
}


def build_seq_config(experiment_id, model_id, seed, mode="formal", hyperparameter_overrides=None):
    if model_id not in MODEL_IDS:
        raise shared.ModelError(f"model_id must be one of {MODEL_IDS}")
    return shared.build_nn_config(
        experiment_id, model_id, seed, mode, DEFAULT_HYPERPARAMETERS, hyperparameter_overrides
    )


# ---------------- encodings ----------------
def build_encodings(role_train, role_validation, family_train, family_validation, history_index):
    """返回每视图 dict:static [N,D]、tokens [N,T,4]、lengths [N]、mask [N,T]。
    encoder 只 fit 于 train;validation 复用同一 encoder transform(P0-02 修复)。"""

    def enc(records, static_encoder, seq_prep):
        static_enc = np.concatenate(
            [static_encoder.transform(r, history_index) for r in records], axis=0
        )
        tokens = np.zeros((len(records), seq_prep.max_len, 4), dtype=np.int64)
        lengths = np.zeros(len(records), dtype=np.int64)
        mask = np.zeros((len(records), seq_prep.max_len), dtype=bool)
        for i, r in enumerate(records):
            t, m, l, _ = seq_prep.transform(r, history_index)
            tokens[i] = t[0]
            lengths[i] = int(l[0])
            mask[i] = m[0]
        return {"static": static_enc, "tokens": tokens, "lengths": lengths, "mask": mask}

    from r2.preprocessing import StaticPreprocessor

    role_static = StaticPreprocessor().fit(role_train, history_index)
    family_static = StaticPreprocessor().fit(family_train, history_index)
    role_seq = shared_seq_prep(role_train, history_index)
    family_seq = shared_seq_prep(family_train, history_index)
    role_tr = enc(role_train, role_static, role_seq)
    role_va = enc(role_validation, role_static, role_seq)
    fam_tr = enc(family_train, family_static, family_seq)
    fam_va = enc(family_validation, family_static, family_seq)
    return role_tr, role_va, fam_tr, fam_va


def shared_feature_static(records, history_index):
    from r2.preprocessing import StaticPreprocessor

    encoder = StaticPreprocessor().fit(records, history_index)
    return np.concatenate([encoder.transform(r, history_index) for r in records], axis=0)


def shared_seq_prep(records, history_index):
    from r2.preprocessing import SequencePreprocessor

    return SequencePreprocessor().fit(records, history_index)


def seq_vocab_sizes(seq_prep, records, history_index):
    sizes = []
    for channel in ("history_roles", "history_family_labels", "history_raw_actions", "history_statuses"):
        sizes.append(len(seq_prep.channel_vocab_[channel]) + 2)  # PAD=0, UNK=1, learned 2..
    return sizes


# ---------------- models ----------------
class SequenceRNN(nn.Module):
    """GRU(单向)或 LSTM(双向=False 单向,与 B03 同规模;文档允许 1-2 层)。"""

    def __init__(self, vocab_sizes, static_dim, hidden, dropout, out_dim, cell="gru"):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(v, EMB_DIM, padding_idx=0) for v in vocab_sizes])
        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn_cls(EMB_DIM * 4, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + static_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )

    def forward(self, batch):
        tokens = batch["tokens"]
        lengths = batch["lengths"]
        static = batch["static"]
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
        return self.head(torch.cat([last, static], dim=1))


class CausalTransformer(nn.Module):
    def __init__(self, vocab_sizes, static_dim, hidden, dropout, out_dim, n_layers=2, n_heads=4):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(v, EMB_DIM, padding_idx=0) for v in vocab_sizes])
        d_model = EMB_DIM * 4
        self.pos = nn.Parameter(torch.zeros(1, 64, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=256,
            dropout=dropout, batch_first=True, activation="relu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model + static_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def _fwd(self, tokens, mask, static):
        B, T = tokens.shape[0], tokens.shape[1]
        channels = [emb(tokens[..., c]) for c, emb in enumerate(self.embs)]
        if T > self.pos.shape[1]:
            raise RuntimeError(f"sequence length {T} exceeds position encoding {self.pos.shape[1]}")
        e = torch.cat(channels, dim=-1) + self.pos[:, :T]
        causal = torch.triu(torch.ones(T, T, device=tokens.device, dtype=torch.bool), diagonal=1)
        out = self.encoder(e, mask=causal, src_key_padding_mask=~mask)
        lengths = mask.sum(dim=1).clamp(min=1)
        idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, out.shape[2])
        last = out.gather(1, idx).squeeze(1)
        return self.head(torch.cat([last, static], dim=1))

    def forward(self, batch):
        tokens = batch["tokens"]
        mask = batch["mask"]
        static = batch["static"]
        valid = mask.any(dim=1)
        if bool(valid.all()):
            return self._fwd(tokens, mask, static)
        out = self.head(torch.cat([
            torch.zeros(tokens.shape[0], EMB_DIM * 4, device=tokens.device), static
        ], dim=1))
        if bool(valid.any()):
            out[valid] = self._fwd(tokens[valid], mask[valid], static[valid])
        return out


# ---------------- runners ----------------
def _make_dataset(X, targets):
    return torch.utils.data.TensorDataset(
        torch.from_numpy(X["static"]),
        torch.from_numpy(X["tokens"]),
        torch.from_numpy(X["lengths"]),
        torch.from_numpy(X["mask"]),
        torch.from_numpy(targets),
    )


def _predict(model, X, batch_size, device):
    model.eval()
    chunks = []
    with torch.no_grad():
        n = X["static"].shape[0]
        for start in range(0, n, batch_size):
            batch = {
                "static": torch.from_numpy(X["static"][start : start + batch_size]).to(device),
                "tokens": torch.from_numpy(X["tokens"][start : start + batch_size]).to(device),
                "lengths": torch.from_numpy(X["lengths"][start : start + batch_size]).to(device),
                "mask": torch.from_numpy(X["mask"][start : start + batch_size]).to(device),
            }
            chunks.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def run_validation_seq(artifact_root, config, model_id, device=None):
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

    role_tr, role_va, fam_tr, fam_va = build_encodings(
        role_train, role_validation, family_train, family_validation, history_index
    )
    role_vocab = seq_vocab_sizes(
        shared_seq_prep(role_train, history_index), role_train, history_index
    )
    family_vocab = seq_vocab_sizes(
        shared_seq_prep(family_train, history_index), family_train, history_index
    )
    role_out = len(constants.ROLES)
    family_out = len(constants.FAMILIES)
    if model_id == "B07":
        role_model = CausalTransformer(
            role_vocab, role_tr["static"].shape[1], params["hidden"],
            params["dropout"], role_out,
        ).to(resolved_device)
        family_model = CausalTransformer(
            family_vocab, fam_tr["static"].shape[1], params["hidden"],
            params["dropout"], family_out,
        ).to(resolved_device)
    else:
        cell = "gru" if model_id == "B05" else "lstm"
        role_model = SequenceRNN(
            role_vocab, role_tr["static"].shape[1], params["hidden"],
            params["dropout"], role_out, cell=cell,
        ).to(resolved_device)
        family_model = SequenceRNN(
            family_vocab, fam_tr["static"].shape[1], params["hidden"],
            params["dropout"], family_out, cell=cell,
        ).to(resolved_device)

    role_optimizer = torch.optim.AdamW(
        role_model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]
    )
    family_optimizer = torch.optim.AdamW(
        family_model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]
    )
    role_loader = torch.utils.data.DataLoader(
        _make_dataset(role_tr, _targets(role_train, "next_role", constants.ROLES)),
        batch_size=params["batch_size"], shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(config["seed"] + 1),
    )
    family_loader = torch.utils.data.DataLoader(
        _make_dataset(fam_tr, _targets(family_train, "family_label", constants.FAMILIES)),
        batch_size=params["batch_size"], shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(config["seed"] + 2),
    )

    best_joint = -1.0
    best_epoch = 0
    best_role_state = None
    best_family_state = None
    best_losses = None
    for epoch in range(1, params["max_epochs"] + 1):
        role_loss = _train_seq_epoch(role_model, role_optimizer, role_loader, resolved_device)
        family_loss = _train_seq_epoch(family_model, family_optimizer, family_loader, resolved_device)
        role_rows = shared.prediction_rows(
            role_validation, "role", _predict(role_model, role_va, params["batch_size"], resolved_device)
        )
        family_rows = shared.prediction_rows(
            family_validation, "family",
            _predict(family_model, fam_va, params["batch_size"], resolved_device),
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
        role_validation, "role", _predict(role_model, role_va, params["batch_size"], resolved_device)
    )
    family_rows = shared.prediction_rows(
        family_validation, "family",
        _predict(family_model, fam_va, params["batch_size"], resolved_device),
    )
    metrics = evaluation.evaluate(role_rows, family_rows)
    run_id = run_artifacts.derive_run_id(config)
    metrics["run"] = {
        "run_id": run_id, "model_id": model_id, "mode": mode,
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
            {"role": {"dimension": role_tr["static"].shape[1]},
             "family": {"dimension": fam_tr["static"].shape[1]}},
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
            "model_id": model_id,
            "role_state_dict": best_role_state,
            "family_state_dict": best_family_state,
            "role_vocab_sizes": role_vocab,
            "family_vocab_sizes": family_vocab,
        },
        staging / "checkpoint.pt",
    )
    (staging / "run.log").write_text(
        "\n".join(
            (
                f"{model_id} sequence validation-only {mode}",
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


def _targets(records, target_field, labels):
    index = {label: i for i, label in enumerate(labels)}
    return np.asarray([index[r[target_field]] for r in records], dtype=np.int64)


def _train_seq_epoch(model, optimizer, loader, device):
    model.train()
    total_loss = 0.0
    total_rows = 0
    for static, tokens, lengths, mask, targets in loader:
        batch = {
            "static": static.to(device),
            "tokens": tokens.to(device),
            "lengths": lengths.to(device),
            "mask": mask.to(device),
        }
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = functional.cross_entropy(model(batch), targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(targets)
        total_rows += len(targets)
    return total_loss / total_rows
