#!/usr/bin/env python3
"""J-series training and evaluation entry point (new; legacy P9d/P9e scripts untouched).

Stages (frozen design `docs/p9d_j_series_design.md`, v3.1):
- stage0: dataset/registry audit + encoding sanity + gradient-routing assertions;
- smoke: 2-epoch subset run (backbone -> B1 -> J2) in a separate run root;
- backbone: train the attribute-only backbone per seed, select by validation
  objective (structure+content+behavior), save the frozen predictor state;
- variants: J0/B1 (resource head only) and J1/J2/J3 (joint training from the
  backbone state) per seed, with NI-feasible -> argmin RuntimeQScore selection;
- test: freeze-time evaluation on P_dev/test with paired video bootstrap;
  refuses to overwrite an existing test_eval.json unless --force is given.

Variants (design section 2):
  J0 frozen encoder, no attribute interface, resource head only
  B1 frozen backbone as deployable two-stage: sg(q(A)) -> resource head
  J1 trainable encoder, no attribute interface, L_total = LS+LC+LB+LR
  J2 trainable, sg(q(A)) interface (resource gradient blocked from attr heads)
  J3 trainable, q(A) interface (resource gradient reaches attr heads)
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import j_series_common as common  # noqa: E402

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"torch is required for J-series training: {exc}")

SCHEMA_VERSION = "j-series-run-v1"
RESOURCE_HEADS = ("res_hidden", "head_runtime", "head_load_occ", "head_load_dur", "head_memory", "dur_adapter")
ENCODER_PREFIXES = ("hist_emb", "pos_emb", "gru", "ctx_emb", "mod_proj", "repr", "repr_ctx", "slot_emb")
ATTRIBUTE_PREFIXES = ("attr_hidden", "head_node_type", "head_role", "head_action_family", "head_model_class", "head_merged", "head_retry", "head_nested")
DEFAULT_VARIANTS = {
    "backbone": {"encoder": "trainable", "use_attribute_distribution": False, "attribute_gradient": "none", "duration_mode": "shared"},
    "J0": {"encoder": "frozen", "use_attribute_distribution": False, "attribute_gradient": "none", "duration_mode": "shared"},
    "B1": {"encoder": "frozen", "use_attribute_distribution": True, "attribute_gradient": "none", "duration_mode": "shared"},
    "J1": {"encoder": "trainable", "use_attribute_distribution": False, "attribute_gradient": "none", "duration_mode": "shared"},
    "J2": {"encoder": "trainable", "use_attribute_distribution": True, "attribute_gradient": "stopgrad", "duration_mode": "shared"},
    "J3": {"encoder": "trainable", "use_attribute_distribution": True, "attribute_gradient": "full", "duration_mode": "shared"},
    "J4a": {"encoder": "trainable", "use_attribute_distribution": True, "attribute_gradient": "stopgrad", "duration_mode": "decoupled", "duration_gate": True},
    "J4b": {"encoder": "trainable", "use_attribute_distribution": True, "attribute_gradient": "stopgrad", "duration_mode": "decoupled_shared_frozen", "duration_gate": True},
}
INTERFACE_ENDPOINTS = {
    "future_length": "length_mae",
    "termination": "termination_bce",
    "future_content": "content_accuracy",
    "next_role": "next_role_acc",
    "next_family": "next_family_acc",
}
FLOAT_KEYS = {"mod_vec", "runtime_ms", "load_ms", "load_occ", "memory_mb",
              "hist_res_num", "hist_res_mask"}
VARIANT_ORDER = ("J0", "B1", "J1", "J2", "J3")
JOINT_VARIANTS = ("J1", "J2", "J3")
BACKBONE = "backbone"

TRAIN_LOSS_KEYS = ("structure", "content", "behavior", "resource")


# --------------------------------------------------------------------------- #
# context
# --------------------------------------------------------------------------- #
@dataclass
class Ctx:
    config: Dict[str, Any]
    run_root: Path
    exp_dir: Path
    device: torch.device
    horizon: int
    batch_size: int
    vocabs: common.VocabCollection
    train: Dict[str, np.ndarray]
    validation: Dict[str, np.ndarray]
    test: Dict[str, np.ndarray]
    val_draws: np.ndarray
    test_draws: np.ndarray
    copy_summaries: bool = True
    manifest: Dict[str, Any] = field(default_factory=dict)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_video_code(value: str) -> int:
    return zlib.crc32(value.encode("utf-8")) & 0x7FFFFFFF


def to_torch_batch(arrays: Mapping[str, np.ndarray], indices: np.ndarray, device: torch.device) -> Dict[str, Any]:
    batch: Dict[str, Any] = {}
    for key, array in arrays.items():
        if key == "video_code":
            continue
        value = array[indices]
        if key in FLOAT_KEYS:
            batch[key] = torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)).to(device)
        else:
            batch[key] = torch.from_numpy(np.ascontiguousarray(value, dtype=np.int64)).to(device)
    return batch


def split_indices(n: int, batch_size: int) -> List[np.ndarray]:
    return [np.arange(start, min(start + batch_size, n), dtype=np.int64) for start in range(0, n, batch_size)]


def horizon_mask(batch: Mapping[str, Any], horizon: int) -> Any:
    slots = torch.arange(horizon, device=batch["length"].device).unsqueeze(0)
    return (slots < batch["length"].unsqueeze(1)).to(torch.float32)


def variant_config(ctx: Ctx, variant: str) -> Mapping[str, Any]:
    if variant in ctx.config["variants"]:
        return ctx.config["variants"][variant]
    if variant in DEFAULT_VARIANTS:
        return DEFAULT_VARIANTS[variant]
    raise KeyError(f"unknown variant: {variant}")


def duration_mode(ctx: Ctx, variant: str) -> str:
    return str(variant_config(ctx, variant).get("duration_mode", "shared"))


# --------------------------------------------------------------------------- #
# model forward / trainability
# --------------------------------------------------------------------------- #
def forward(model: common.JSeriesModel, batch: Mapping[str, Any], variant: str, ctx: Ctx) -> Dict[str, Any]:
    cfg = variant_config(ctx, variant)
    repr_vec = model.encode(batch)
    structure = model.structure(repr_vec)
    behavior = model.behavior(repr_vec)
    attribute_logits = model.attribute_logits(repr_vec)
    attr_feature = None
    if cfg["use_attribute_distribution"]:
        if variant in ("B1", "J2"):
            with torch.no_grad():
                feature, _ = model.attribute_distribution(attribute_logits)
            attr_feature = feature.detach()
        else:  # J3 keeps the graph through q(A)
            feature, _ = model.attribute_distribution(attribute_logits)
            attr_feature = feature
    resource = model.resource(repr_vec, attr_feature)
    return {"structure": structure, "behavior": behavior, "attributes": attribute_logits, "resource": resource}


def loss_terms(outputs: Mapping[str, Any], batch: Mapping[str, Any], horizon: int) -> Dict[str, Any]:
    mask = horizon_mask(batch, horizon)
    return {
        "structure": common.structure_loss(outputs["structure"], batch),
        "content": common.content_loss(outputs["attributes"], batch, mask),
        "behavior": common.behavior_loss(outputs["behavior"], batch),
        "resource": common.resource_loss(outputs["resource"], batch, mask),
    }


def total_loss(variant: str, terms: Mapping[str, Any]) -> Any:
    if variant in ("J0", "B1"):
        return terms["resource"]
    if variant == BACKBONE:
        return terms["structure"] + terms["content"] + terms["behavior"]
    return terms["structure"] + terms["content"] + terms["behavior"] + terms["resource"]


def configure_trainable(model: common.JSeriesModel, variant: str) -> List[str]:
    if variant in ("J0", "B1"):
        prefixes: Optional[Tuple[str, ...]] = RESOURCE_HEADS
    else:
        prefixes = None
    for name, parameter in model.named_parameters():
        if prefixes is None:
            parameter.requires_grad_(True)
        else:
            parameter.requires_grad_(any(name.startswith(prefix) for prefix in prefixes))
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def make_model(ctx: Ctx, variant: Optional[str] = None) -> common.JSeriesModel:
    mode = duration_mode(ctx, variant) if variant else "shared"
    return common.JSeriesModel(ctx.vocabs, ctx.config["model"], horizon=ctx.horizon, duration_mode=mode).to(ctx.device)


def load_compatible(model: common.JSeriesModel, state: Mapping[str, Any]) -> Dict[str, Any]:
    """Copy shape-matching tensors only (used when initializing J4 branches from backbone)."""
    own = model.state_dict()
    filtered = {key: value for key, value in state.items() if key in own and own[key].shape == value.shape}
    skipped = sorted(key for key in state if key not in filtered)
    model.load_state_dict(filtered, strict=False)
    return {"loaded": len(filtered), "skipped": skipped}


def cpu_state(model: common.JSeriesModel) -> Dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def load_checkpoint(path: Path, ctx: Ctx, variant: Optional[str] = None) -> Tuple[common.JSeriesModel, Dict[str, Any]]:
    payload = torch.load(path, map_location=ctx.device, weights_only=False)
    model = make_model(ctx, variant or payload.get("variant"))
    model.load_state_dict(payload["model_state"])
    return model, payload


def grad_norm(model: common.JSeriesModel, prefixes: Sequence[str]) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes) and parameter.grad is not None:
            total += float(parameter.grad.norm() ** 2)
    return float(total ** 0.5)


# --------------------------------------------------------------------------- #
# evaluation helpers
# --------------------------------------------------------------------------- #
def per_row_metrics(model: common.JSeriesModel, arrays: Mapping[str, np.ndarray], variant: str, ctx: Ctx, max_rows: Optional[int] = None) -> Dict[str, np.ndarray]:
    model.eval()
    n = len(arrays["length"]) if max_rows is None else min(max_rows, len(arrays["length"]))
    collected: Dict[str, List[np.ndarray]] = {}
    with torch.no_grad():
        for indices in split_indices(n, ctx.batch_size):
            batch = to_torch_batch(arrays, indices, ctx.device)
            outputs = forward(model, batch, variant, ctx)
            runtime_ms = torch.expm1(outputs["resource"]["runtime_log_quantiles"]).clamp(min=0.0).cpu().numpy()
            load_dur_ms = torch.expm1(outputs["resource"]["load_dur_log_quantiles"]).clamp(min=0.0).cpu().numpy()
            attrs = outputs["attributes"]
            numpy_outputs = {
                "runtime_ms": runtime_ms,
                "load_dur_ms": load_dur_ms,
                "length_logits": outputs["structure"]["length_logits"].cpu().numpy(),
                "termination_logit": outputs["structure"]["termination_logit"].cpu().numpy(),
                "next_role_logits": outputs["behavior"]["next_role_logits"].cpu().numpy(),
                "next_family_logits": outputs["behavior"]["next_family_logits"].cpu().numpy(),
                "load_occ_logit": outputs["resource"]["load_occ_logit"].cpu().numpy(),
            }
            for field in common.SLOT_CAT_FIELDS + ("merged", "retry", "nested"):
                numpy_outputs[f"attr_{field}"] = attrs[field].cpu().numpy()
            numpy_batch = {key: value[indices] for key, value in arrays.items() if isinstance(value, np.ndarray) and key != "video_code"}
            metrics = common.per_row_metrics(numpy_outputs, numpy_batch)
            for key, value in metrics.items():
                collected.setdefault(key, []).append(value)
    return {key: np.concatenate(values) for key, values in collected.items()}


def point_metrics(metrics: Mapping[str, np.ndarray]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in metrics.items():
        finite = value[np.isfinite(value)]
        out[key] = float(finite.mean()) if finite.size else float("nan")
    return out


def eval_loss_means(model: common.JSeriesModel, arrays: Mapping[str, np.ndarray], variant: str, ctx: Ctx, max_rows: Optional[int] = None) -> Dict[str, float]:
    model.eval()
    n = len(arrays["length"]) if max_rows is None else min(max_rows, len(arrays["length"]))
    sums = {key: 0.0 for key in TRAIN_LOSS_KEYS}
    count = 0
    with torch.no_grad():
        for indices in split_indices(n, ctx.batch_size):
            batch = to_torch_batch(arrays, indices, ctx.device)
            outputs = forward(model, batch, variant, ctx)
            terms = loss_terms(outputs, batch, ctx.horizon)
            for key in TRAIN_LOSS_KEYS:
                sums[key] += float(terms[key]) * len(indices)
            count += len(indices)
    return {key: sums[key] / max(1, count) for key in TRAIN_LOSS_KEYS}


def ni_record(metrics: Mapping[str, np.ndarray], reference: Mapping[str, np.ndarray], ctx: Ctx, max_rows: Optional[int] = None) -> Tuple[Dict[str, Any], bool]:
    record: Dict[str, Any] = {}
    if max_rows is None:
        video_codes, draws = ctx.validation["video_code"], ctx.val_draws
    else:
        video_codes = ctx.validation["video_code"][:max_rows]
        draws = common.make_bootstrap_indices(video_codes, int(ctx.config["bootstrap"]["B"]), int(ctx.config["bootstrap"]["seed"]))
    for endpoint, metric_key in INTERFACE_ENDPOINTS.items():
        delta = metrics[metric_key] - reference[metric_key]
        record[endpoint] = common.bootstrap_delta_ci(delta, video_codes, draws)
    feasible, _ = common.ni_feasible(record, ctx.config["ni_manifest"])
    return record, feasible


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def copy_small(ctx: Ctx, *names: str) -> None:
    for name in names:
        source = ctx.run_root / name
        if source.is_file():
            (ctx.exp_dir / name).write_bytes(source.read_bytes())


def copy_summary(ctx: Ctx, variant: str, seed: int) -> None:
    if not ctx.copy_summaries:
        return
    source = ctx.run_root / "runs" / variant / f"seed{seed}" / "run.json"
    if not source.is_file():
        return
    target_dir = ctx.exp_dir / "summaries"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / f"{variant}_seed{seed}.json").write_bytes(source.read_bytes())


# --------------------------------------------------------------------------- #
# gradient probe + routing checks
# --------------------------------------------------------------------------- #
def gradient_probe(model: common.JSeriesModel, batch: Mapping[str, Any], variant: str, ctx: Ctx) -> Dict[str, Any]:
    model.train()
    outputs = forward(model, batch, variant, ctx)
    terms = loss_terms(outputs, batch, ctx.horizon)
    names = ["structure", "content", "behavior", "resource"]
    parameters = [p for p in model.parameters() if p.requires_grad]
    vectors: Dict[str, np.ndarray] = {}
    for name in names:
        model.zero_grad(set_to_none=True)
        terms[name].backward(retain_graph=True)
        pieces = [
            parameter.grad.detach().flatten().cpu().numpy() if parameter.grad is not None else np.zeros(parameter.numel(), dtype=np.float32)
            for parameter in parameters
        ]
        vectors[name] = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
    model.zero_grad(set_to_none=True)
    result: Dict[str, Any] = {"norms": {name: float(np.linalg.norm(vector)) for name, vector in vectors.items()}}
    cosines: Dict[str, float] = {}
    for left in names:
        for right in names:
            if left >= right:
                continue
            denom = float(np.linalg.norm(vectors[left]) * np.linalg.norm(vectors[right]))
            cosines[f"{left}_vs_{right}"] = float(np.dot(vectors[left], vectors[right]) / denom) if denom > 0 else float("nan")
    result["cosines"] = cosines
    return result


def routing_checks(ctx: Ctx) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    indices = np.arange(min(32, len(ctx.train["length"])), dtype=np.int64)
    batch = to_torch_batch(ctx.train, indices, ctx.device)
    base_model = make_model(ctx)
    base_state = cpu_state(base_model)
    for variant in VARIANT_ORDER:
        set_seed(11)
        model = make_model(ctx)
        model.load_state_dict(base_state)
        configure_trainable(model, variant)
        outputs = forward(model, batch, variant, ctx)
        terms = loss_terms(outputs, batch, ctx.horizon)
        model.zero_grad(set_to_none=True)
        terms["resource"].backward()
        checks[variant] = {
            "encoder_resource_grad": grad_norm(model, ENCODER_PREFIXES) > 0.0,
            "attribute_head_resource_grad": grad_norm(model, ATTRIBUTE_PREFIXES),
            "resource_head_grad": grad_norm(model, RESOURCE_HEADS) > 0.0,
        }
    set_seed(11)
    model = make_model(ctx)
    model.load_state_dict(base_state)
    configure_trainable(model, BACKBONE)
    outputs = forward(model, batch, BACKBONE, ctx)
    terms = loss_terms(outputs, batch, ctx.horizon)
    model.zero_grad(set_to_none=True)
    (terms["structure"] + terms["content"] + terms["behavior"]).backward()
    checks[BACKBONE] = {
        "resource_head_untouched": grad_norm(model, RESOURCE_HEADS) == 0.0,
        "encoder_grad_present": grad_norm(model, ENCODER_PREFIXES) > 0.0,
    }
    for variant in ("J4a", "J4b", "J3"):
        set_seed(11)
        model = make_model(ctx, variant)
        load_compatible(model, base_state)
        configure_trainable(model, variant)
        outputs = forward(model, batch, variant, ctx)
        mask = horizon_mask(batch, ctx.horizon)
        dur_valid = mask * (batch["load_occ"] == 1).float()
        load_log = torch.log1p(batch["load_ms"].clamp(min=0.0))
        dur_pred = outputs["resource"]["load_dur_log_quantiles"]
        dur_loss = torch.stack([
            common.masked_mean(common.pinball_loss(dur_pred[:, :, k], load_log, tau), dur_valid)
            for k, tau in enumerate(common.TARGET_TAUS)
        ]).mean()
        model.zero_grad(set_to_none=True)
        dur_loss.backward()
        checks[variant] = {
            "duration_only_shared_hidden_grad": grad_norm(model, ("res_hidden",)),
            "duration_only_adapter_grad": grad_norm(model, ("dur_adapter",)) if any(name.startswith("dur_adapter") for name, _ in model.named_parameters()) else None,
            "duration_head_grad": grad_norm(model, ("head_load_dur",)) > 0.0,
        }
    return checks


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def j3_regression_check(ctx: Ctx) -> Optional[Dict[str, Any]]:
    """Verify the J4 code extension leaves the frozen J-series behavior unchanged.

    Loads the frozen J3 checkpoints from the reference run root, re-evaluates
    validation metrics with the current code and compares them against the
    frozen run.json values. Requires ``regression_check`` in config.
    """
    spec = ctx.config.get("regression_check")
    if not spec:
        return None
    ref_root = Path(spec.get("run_root", ctx.config.get("reference_run_root", ctx.run_root)))
    variant = str(spec.get("variant", "J3"))
    metrics_keys = ("runtime_qscore", "load_dur_qscore", "load_brier", "content_accuracy", "length_mae", "next_role_acc", "next_family_acc")
    tolerance = float(spec.get("tolerance", 1e-6))
    report: Dict[str, Any] = {"variant": variant, "tolerance": tolerance, "seeds": {}}
    all_ok = True
    for seed in spec.get("seeds", [11, 22, 33]):
        checkpoint_path = ref_root / "runs" / variant / f"seed{seed}" / "checkpoint.pt"
        run_path = ref_root / "runs" / variant / f"seed{seed}" / "run.json"
        if not checkpoint_path.is_file() or not run_path.is_file():
            report["seeds"][str(seed)] = {"status": "missing_reference"}
            all_ok = False
            continue
        frozen = json.loads(run_path.read_text(encoding="utf-8"))
        model, payload = load_checkpoint(checkpoint_path, ctx, variant)
        if int(payload["epoch"]) != int(frozen["selected_epoch"]):
            report["seeds"][str(seed)] = {"status": "epoch_mismatch", "checkpoint_epoch": int(payload["epoch"]), "recorded": int(frozen["selected_epoch"])}
            all_ok = False
            continue
        metrics = per_row_metrics(model, ctx.validation, variant, ctx)
        point = point_metrics(metrics)
        frozen_point = frozen["selected_record"]["val"]
        deltas = {key: float(point[key] - frozen_point[key]) for key in metrics_keys if key in frozen_point}
        seed_ok = all(abs(value) <= tolerance for value in deltas.values())
        report["seeds"][str(seed)] = {"status": "ok" if seed_ok else "mismatch", "max_abs_delta": max((abs(v) for v in deltas.values()), default=0.0), "deltas": deltas}
        all_ok = all_ok and seed_ok
    report["status"] = "ok" if all_ok else "failed"
    return report


def stage0(ctx: Ctx) -> Dict[str, Any]:
    registry_path = PROJECT_ROOT / "data" / "manifests" / "j_series_dataset_v1.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.is_file() else {}
    file_hashes: Dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        path = Path(ctx.config["output_root"]) / f"j_{split}.jsonl.gz"
        digest = common.sha256_file(path)
        expected = (registry.get("files_sha256") or {}).get(f"j_{split}.jsonl.gz")
        file_hashes[split] = {"sha256": digest, "registry_match": expected == digest}
    rows = {
        "train": len(ctx.train["length"]),
        "validation": len(ctx.validation["length"]),
        "test": len(ctx.test["length"]),
    }
    lengths = ctx.train["length"]
    histogram = {str(value): int((lengths == value).sum()) for value in range(ctx.horizon + 1)}
    termination = {str(value): int((ctx.train["termination"] == value).sum()) for value in (0, 1)}
    coverage = {
        "runtime_slots": int((ctx.train["runtime_ms"] > 0).sum() + (ctx.validation["runtime_ms"] > 0).sum() + (ctx.test["runtime_ms"] > 0).sum()),
        "load_occ_slots": int((ctx.train["load_occ"] >= 0).sum() + (ctx.validation["load_occ"] >= 0).sum() + (ctx.test["load_occ"] >= 0).sum()),
        "memory_slots": int((ctx.train["memory_mb"] > 0).sum() + (ctx.validation["memory_mb"] > 0).sum() + (ctx.test["memory_mb"] > 0).sum()),
    }
    model = make_model(ctx)
    audit = {
        "stage": "stage0",
        "rows": rows,
        "expected_rows": ctx.config["expected"]["rows_by_split"],
        "rows_match": rows == ctx.config["expected"]["rows_by_split"],
        "file_hashes": file_hashes,
        "histogram_train": histogram,
        "termination_train": termination,
        "coverage": coverage,
        "vocab_sizes": {
            "history": {field: len(ctx.vocabs.history[field]) for field in common.HISTORY_FIELDS},
            "context": {field: len(ctx.vocabs.context[field]) for field in common.CONTEXT_FIELDS},
            "modalities": len(ctx.vocabs.modalities),
            "slot": {field: len(ctx.vocabs.slot[field]) for field in common.SLOT_CAT_FIELDS},
            "nested": len(ctx.vocabs.nested),
        },
        "model_params": int(sum(p.numel() for p in model.parameters())),
        "attr_dim": int(model.attr_dim),
        "video_counts": {
            "train": int(np.unique(ctx.train["video_code"]).size),
            "validation": int(np.unique(ctx.validation["video_code"]).size),
            "test": int(np.unique(ctx.test["video_code"]).size),
        },
        "bootstrap": {"B": int(ctx.config["bootstrap"]["B"]), "seed": int(ctx.config["bootstrap"]["seed"]), "draws_shape": list(ctx.val_draws.shape)},
        "routing_checks": routing_checks(ctx),
        "j3_regression_check": j3_regression_check(ctx),
    }
    common.write_json(ctx.run_root / "stage0_audit.json", audit)
    common.write_json(ctx.run_root / "vocab.json", ctx.vocabs.to_json())
    np.savez_compressed(ctx.run_root / "bootstrap_indices_val.npz", draws=ctx.val_draws)
    np.savez_compressed(ctx.run_root / "bootstrap_indices_test.npz", draws=ctx.test_draws)
    copy_small(ctx, "stage0_audit.json")
    return audit


def train_epoch(model: common.JSeriesModel, arrays: Mapping[str, np.ndarray], variant: str, seed: int, epoch: int, ctx: Ctx, optimizer: Any) -> Dict[str, float]:
    model.train()
    order = torch.randperm(len(arrays["length"]), generator=torch.Generator().manual_seed(seed * 1000 + epoch)).numpy()
    sums = {key: 0.0 for key in TRAIN_LOSS_KEYS}
    count = 0
    for indices in [order[start:start + ctx.batch_size] for start in range(0, len(order), ctx.batch_size)]:
        batch = to_torch_batch(arrays, indices, ctx.device)
        outputs = forward(model, batch, variant, ctx)
        terms = loss_terms(outputs, batch, ctx.horizon)
        loss = total_loss(variant, terms)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
        optimizer.step()
        for key in TRAIN_LOSS_KEYS:
            sums[key] += float(terms[key]) * len(indices)
        count += len(indices)
    return {key: sums[key] / max(1, count) for key in TRAIN_LOSS_KEYS}


def run_training(ctx: Ctx, variant: str, seed: int, epochs: int, max_train_rows: Optional[int] = None, max_val_rows: Optional[int] = None) -> Dict[str, Any]:
    set_seed(seed)
    init_root = Path(ctx.config.get("init_run_root", ctx.run_root))
    reference_root = Path(ctx.config.get("reference_run_root", init_root))
    model = make_model(ctx, variant)
    init = "scratch"
    if variant != BACKBONE:
        backbone_path = init_root / "runs" / BACKBONE / f"seed{seed}" / "checkpoint.pt"
        if not backbone_path.is_file():
            raise FileNotFoundError(f"backbone checkpoint missing: {backbone_path}")
        payload = torch.load(backbone_path, map_location=ctx.device, weights_only=False)
        info = load_compatible(model, payload["model_state"])
        init = f"backbone:seed{seed}:epoch{payload['epoch']} ({info['loaded']} tensors, skipped {len(info['skipped'])})"
    configure_trainable(model, variant)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(ctx.config["training"]["learning_rate"]),
        weight_decay=float(ctx.config["training"]["weight_decay"]),
    )

    train_arrays = ctx.train
    if max_train_rows is not None:
        train_arrays = {key: value[:max_train_rows] for key, value in ctx.train.items()}
    reference_metrics = None
    duration_gate_limit = None
    duration_gate_base = None
    cfg = variant_config(ctx, variant)
    if variant != BACKBONE:
        backbone_model, _ = load_checkpoint(init_root / "runs" / BACKBONE / f"seed{seed}" / "checkpoint.pt", ctx, BACKBONE)
        reference_metrics = per_row_metrics(backbone_model, ctx.validation, BACKBONE, ctx, max_val_rows)
        if cfg.get("duration_gate"):
            b1_model, _ = load_checkpoint(reference_root / "runs" / "B1" / f"seed{seed}" / "checkpoint.pt", ctx, "B1")
            b1_metrics = per_row_metrics(b1_model, ctx.validation, "B1", ctx, max_val_rows)
            duration_gate_base = point_metrics(b1_metrics)["load_dur_qscore"]
            duration_gate_limit = duration_gate_base * 1.05
    duration_was_gated = duration_gate_limit is not None

    run_dir = ctx.run_root / "runs" / variant / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {"score": float("inf"), "epoch": -1, "state": None, "record": None}
    probe: Optional[Dict[str, Any]] = None
    started = time.time()
    for epoch in range(1, epochs + 1):
        train_means = train_epoch(model, train_arrays, variant, seed, epoch, ctx, optimizer)
        metrics = per_row_metrics(model, ctx.validation, variant, ctx, max_val_rows)
        point = point_metrics(metrics)
        record: Dict[str, Any] = {
            "epoch": epoch,
            "train": train_means,
            "val": {key: point[key] for key in ("runtime_qscore", "length_mae", "termination_bce", "content_accuracy", "next_role_acc", "next_family_acc", "load_brier", "load_dur_qscore")},
        }
        if variant == BACKBONE:
            val_loss = eval_loss_means(model, ctx.validation, BACKBONE, ctx, max_val_rows)
            record["val_loss"] = val_loss
            selection = val_loss["structure"] + val_loss["content"] + val_loss["behavior"]
            record["selection_score"] = selection
            feasible = True
            record["ni_feasible"] = True
        else:
            record_ni, feasible = ni_record(metrics, reference_metrics, ctx, max_val_rows)  # type: ignore[arg-type]
            record["ni"] = record_ni
            if duration_was_gated:
                duration_ok = bool(point["load_dur_qscore"] <= duration_gate_limit)
                record["duration_gate"] = {
                    "value": point["load_dur_qscore"],
                    "base_B1": duration_gate_base,
                    "limit": duration_gate_limit,
                    "ok": duration_ok,
                }
                feasible = bool(feasible and duration_ok)
            record["ni_feasible"] = feasible
            selection = float(point["runtime_qscore"])
            record["selection_score"] = selection
            if epoch == 1 and variant in JOINT_VARIANTS:
                probe = gradient_probe(model, to_torch_batch(train_arrays, np.arange(min(64, len(train_arrays["length"])), dtype=np.int64), ctx.device), variant, ctx)
        history.append(record)
        if feasible and selection < best["score"]:
            best = {"score": selection, "epoch": epoch, "state": cpu_state(model), "record": record}
        print(f"[{variant} seed={seed}] epoch {epoch}/{epochs} train={train_means} selection={selection:.6f} feasible={feasible} best_epoch={best['epoch']}", flush=True)
    duration = time.time() - started
    summary: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "variant": variant,
        "seed": seed,
        "status": "ok" if best["epoch"] > 0 else "failed_no_feasible_epoch",
        "epochs": epochs,
        "init": init,
        "selected_epoch": best["epoch"],
        "selected_score": best["score"],
        "selected_record": best["record"],
        "history": history,
        "duration_seconds": round(duration, 2),
        "device": str(ctx.device),
        "gradient_probe": probe,
        "dataset_manifest": ctx.manifest,
    }
    common.write_json(run_dir / "run.json", json_safe(summary))
    if best["state"] is not None:
        torch.save({"schema_version": SCHEMA_VERSION, "variant": variant, "seed": seed, "epoch": best["epoch"], "model_state": best["state"]}, run_dir / "checkpoint.pt")
    copy_summary(ctx, variant, seed)
    return summary


def stage_smoke(ctx: Ctx) -> Dict[str, Any]:
    smoke_ctx = dataclasses.replace(ctx, run_root=ctx.run_root / "smoke", copy_summaries=False)
    smoke_ctx.run_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results: Dict[str, Any] = {"stage": "smoke", "epochs": 2, "max_train_rows": 512, "max_val_rows": 512, "runs": {}}
    smoke_variants = ctx.config.get("smoke_variants", ["backbone", "B1", "J2"])
    for variant in smoke_variants:
        seed = 11
        summary = run_training(smoke_ctx, variant, seed, epochs=2, max_train_rows=512, max_val_rows=512)
        results["runs"][f"{variant}:{seed}"] = {
            "status": summary["status"],
            "selected_epoch": summary["selected_epoch"],
            "train_loss_first": summary["history"][0]["train"],
            "train_loss_last": summary["history"][-1]["train"],
            "val": summary["history"][-1]["val"],
            "selection_score": summary["selected_score"],
            "duration_seconds": summary["duration_seconds"],
        }
    results["duration_seconds"] = round(time.time() - started, 2)
    common.write_json(smoke_ctx.run_root / "smoke.json", results)
    results["run_root"] = str(smoke_ctx.run_root)
    return results


def stage_backbone(ctx: Ctx, seeds: Sequence[int], epochs: int) -> Dict[str, Any]:
    summaries = {}
    for seed in seeds:
        summary = run_training(ctx, BACKBONE, seed, epochs)
        summaries[f"{seed}"] = {"status": summary["status"], "selected_epoch": summary["selected_epoch"], "selected_score": summary["selected_score"]}
    return summaries


def stage_variants(ctx: Ctx, variants: Sequence[str], seeds: Sequence[int], epochs: int) -> Dict[str, Any]:
    summaries: Dict[str, Any] = {}
    for variant in variants:
        for seed in seeds:
            summary = run_training(ctx, variant, seed, epochs)
            summaries[f"{variant}:{seed}"] = {"status": summary["status"], "selected_epoch": summary["selected_epoch"], "selected_score": summary["selected_score"]}
    return summaries


def test_variant(ctx: Ctx, variant: str, seed: int, reference_metrics: Mapping[str, np.ndarray], extra_reference: Optional[Mapping[str, np.ndarray]] = None, extra_label: str = "J3") -> Dict[str, Any]:
    path = ctx.run_root / "runs" / variant / f"seed{seed}" / "checkpoint.pt"
    if not path.is_file():
        return {"variant": variant, "seed": seed, "status": "missing_checkpoint"}
    model, payload = load_checkpoint(path, ctx, variant)
    metrics = per_row_metrics(model, ctx.test, variant, ctx)
    point = point_metrics(metrics)
    endpoints: Dict[str, Any] = {}
    for endpoint, metric_key in list(INTERFACE_ENDPOINTS.items()) + [("load_brier", "load_brier"), ("load_dur_qscore", "load_dur_qscore")]:
        delta = metrics[metric_key] - reference_metrics[metric_key]
        endpoints[endpoint] = common.bootstrap_delta_ci(delta, ctx.test["video_code"], ctx.test_draws)
    runtime = common.bootstrap_delta_ci(metrics["runtime_qscore"] - reference_metrics["runtime_qscore"], ctx.test["video_code"], ctx.test_draws)
    ref_dur = point_metrics(reference_metrics)["load_dur_qscore"]
    thresholds = ctx.config["core_success"]["load_noninferiority"]
    brier_ok = bool(np.isfinite(endpoints["load_brier"]["ci_upper"]) and endpoints["load_brier"]["ci_upper"] < 0.005 and endpoints["load_brier"]["reliable"])
    dur_ok = bool(np.isfinite(endpoints["load_dur_qscore"]["ci_upper"]) and endpoints["load_dur_qscore"]["ci_upper"] < 0.05 * ref_dur and endpoints["load_dur_qscore"]["reliable"])
    core_ok = bool(np.isfinite(runtime["ci_upper"]) and runtime["ci_upper"] < 0.0 and runtime["reliable"] and brier_ok and dur_ok)
    entry = {
        "variant": variant,
        "seed": seed,
        "status": "ok",
        "selected_epoch": payload["epoch"],
        "model_sha256": common.sha256_file(path),
        "point": point,
        "runtime_delta": runtime,
        "endpoints": endpoints,
        "load_noninferiority": {"brier_ok": brier_ok, "duration_ok": dur_ok, "thresholds": thresholds, "reference_load_dur": ref_dur},
        "core_success_seed": core_ok,
    }
    if extra_reference is not None:
        entry[f"vs_{extra_label}"] = {
            "runtime_delta": common.bootstrap_delta_ci(metrics["runtime_qscore"] - extra_reference["runtime_qscore"], ctx.test["video_code"], ctx.test_draws),
            "load_dur_delta": common.bootstrap_delta_ci(metrics["load_dur_qscore"] - extra_reference["load_dur_qscore"], ctx.test["video_code"], ctx.test_draws),
            "load_brier_delta": common.bootstrap_delta_ci(metrics["load_brier"] - extra_reference["load_brier"], ctx.test["video_code"], ctx.test_draws),
        }
    return entry


def stage_test(ctx: Ctx, seeds: Sequence[int], force: bool = False) -> Dict[str, Any]:
    output_path = ctx.run_root / "test_eval.json"
    if output_path.exists() and not force:
        raise FileExistsError(f"test_eval.json already exists (frozen determination); use --force to overwrite: {output_path}")
    results: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "test",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "per_seed": {},
        "per_variant": {},
    }
    per_variant: Dict[str, List[Dict[str, Any]]] = {variant: [] for variant in VARIANT_ORDER}
    for seed in seeds:
        b1_path = ctx.run_root / "runs" / "B1" / f"seed{seed}" / "checkpoint.pt"
        if not b1_path.is_file():
            raise FileNotFoundError(f"B1 checkpoint missing for seed {seed}: {b1_path}")
        b1_model, _ = load_checkpoint(b1_path, ctx, "B1")
        reference = per_row_metrics(b1_model, ctx.test, "B1", ctx)
        seed_entry: Dict[str, Any] = {"B1": {"status": "reference", "point": point_metrics(reference)}}
        for variant in VARIANT_ORDER:
            entry = test_variant(ctx, variant, seed, reference)
            seed_entry[variant] = entry
            if entry.get("status") == "ok":
                per_variant[variant].append(entry)
        results["per_seed"][str(seed)] = seed_entry
    for variant, entries in per_variant.items():
        go_seeds = [entry["seed"] for entry in entries if entry["core_success_seed"]]
        deltas = [entry["runtime_delta"]["delta_mean"] for entry in entries]
        results["per_variant"][variant] = {
            "seeds": [entry["seed"] for entry in entries],
            "core_success_seeds": go_seeds,
            "mean_runtime_delta": float(np.nanmean(deltas)) if deltas else float("nan"),
            "worst_runtime_delta": float(np.nanmax(deltas)) if deltas else float("nan"),
            "verdict": "GO" if len(go_seeds) >= 2 else ("NO-GO" if entries else "INCOMPLETE"),
        }
    common.write_json(output_path, json_safe(results))
    copy_small(ctx, "test_eval.json")
    artifacts = ctx.exp_dir / "artifacts" / "checkpoints"
    artifacts.mkdir(parents=True, exist_ok=True)
    for variant in VARIANT_ORDER:
        for seed in seeds:
            source = ctx.run_root / "runs" / variant / f"seed{seed}" / "checkpoint.pt"
            if source.is_file():
                (artifacts / f"{variant}_seed{seed}.pt").write_bytes(source.read_bytes())
    return results


def stage_test_exploratory(ctx: Ctx, seeds: Sequence[int], force: bool = False) -> Dict[str, Any]:
    """Exploratory second look on the previously used P_dev/test (no confirmatory claim)."""
    output_path = ctx.run_root / "test_exploratory.json"
    if output_path.exists() and not force:
        raise FileExistsError(f"test_exploratory.json already exists; use --force to overwrite: {output_path}")
    reference_root = Path(ctx.config.get("reference_run_root", ctx.run_root))
    variants = ctx.config.get("test_variants", ["J4a", "J4b"])
    results: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "test_exploratory",
        "status": "exploratory_second_look_on_previously_used_P_dev_test",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "per_seed": {},
        "per_variant": {},
    }
    per_variant: Dict[str, List[Dict[str, Any]]] = {variant: [] for variant in variants}
    for seed in seeds:
        b1_path = reference_root / "runs" / "B1" / f"seed{seed}" / "checkpoint.pt"
        j3_path = reference_root / "runs" / "J3" / f"seed{seed}" / "checkpoint.pt"
        if not b1_path.is_file() or not j3_path.is_file():
            raise FileNotFoundError(f"reference checkpoints missing for seed {seed}: B1={b1_path.is_file()} J3={j3_path.is_file()}")
        b1_model, _ = load_checkpoint(b1_path, ctx, "B1")
        b1_metrics = per_row_metrics(b1_model, ctx.test, "B1", ctx)
        j3_model, _ = load_checkpoint(j3_path, ctx, "J3")
        j3_metrics = per_row_metrics(j3_model, ctx.test, "J3", ctx)
        seed_entry: Dict[str, Any] = {
            "B1_reference": {"status": "inherited_frozen", "point": point_metrics(b1_metrics)},
            "J3_reference": {"status": "inherited_frozen", "point": point_metrics(j3_metrics)},
        }
        for variant in variants:
            entry = test_variant(ctx, variant, seed, b1_metrics, extra_reference=j3_metrics, extra_label="J3")
            seed_entry[variant] = entry
            if entry.get("status") == "ok":
                per_variant[variant].append(entry)
        results["per_seed"][str(seed)] = seed_entry
    for variant, entries in per_variant.items():
        results["per_variant"][variant] = {
            "seeds": [entry["seed"] for entry in entries],
            "note": "exploratory only; validation layer is the official determination tier",
            "mean_runtime_delta_vs_B1": float(np.nanmean([entry["runtime_delta"]["delta_mean"] for entry in entries])) if entries else float("nan"),
            "mean_load_dur_delta_vs_J3": float(np.nanmean([entry["vs_J3"]["load_dur_delta"]["delta_mean"] for entry in entries])) if entries else float("nan"),
        }
    common.write_json(output_path, json_safe(results))
    copy_small(ctx, "test_exploratory.json")
    return results


# --------------------------------------------------------------------------- #
# manifest / main
# --------------------------------------------------------------------------- #
def build_manifest(ctx: Ctx, config_path: Path) -> Dict[str, Any]:
    vocab_hash = hashlib.sha256(json.dumps(ctx.vocabs.to_json(), sort_keys=True).encode("utf-8")).hexdigest()
    files = {}
    for split in ("train", "validation", "test"):
        path = Path(ctx.config["output_root"]) / f"j_{split}.jsonl.gz"
        files[f"j_{split}.jsonl.gz"] = {"sha256": common.sha256_file(path), "bytes": path.stat().st_size}
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": ctx.config["experiment_id"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config_sha256": common.sha256_file(config_path),
        "dataset_files": files,
        "vocab_sha256": vocab_hash,
        "bootstrap": ctx.config["bootstrap"],
        "device": str(ctx.device),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "seeds": ctx.config["training"]["seeds"],
        "variants": ctx.config["variants"],
        "batch_size": ctx.batch_size,
        "max_epochs": ctx.config["training"]["max_epochs"],
    }


def make_ctx(config_path: Path, run_root: Optional[Path]) -> Ctx:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    exp_dir = PROJECT_ROOT / "experiments" / config["experiment_id"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    root = Path(run_root) if run_root else PROJECT_ROOT / "outputs" / str(config.get("run_root_name", "j_series_joint_resource"))
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device(config["environment"].get("device", "cuda")) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cpu":
        print("WARNING: CUDA not available; formal consistency requires a single device", flush=True)
    train_rows = list(common.read_jsonl_gz(Path(config["output_root"]) / "j_train.jsonl.gz"))
    vocabs = common.build_vocabs(train_rows)
    horizon = int(config["horizon"])
    train = common.encode_rows(train_rows, vocabs, horizon)
    train["video_code"] = np.asarray([stable_video_code(row["video_id"]) for row in train_rows], dtype=np.int64)
    validation_rows = list(common.read_jsonl_gz(Path(config["output_root"]) / "j_validation.jsonl.gz"))
    validation = common.encode_rows(validation_rows, vocabs, horizon)
    validation["video_code"] = np.asarray([stable_video_code(row["video_id"]) for row in validation_rows], dtype=np.int64)
    test_rows = list(common.read_jsonl_gz(Path(config["output_root"]) / "j_test.jsonl.gz"))
    test = common.encode_rows(test_rows, vocabs, horizon)
    test["video_code"] = np.asarray([stable_video_code(row["video_id"]) for row in test_rows], dtype=np.int64)
    replicates = int(config["bootstrap"]["B"])
    base_seed = int(config["bootstrap"]["seed"])
    ctx = Ctx(
        config=config,
        run_root=root,
        exp_dir=exp_dir,
        device=device,
        horizon=horizon,
        batch_size=int(config["training"]["batch_size"]),
        vocabs=vocabs,
        train=train,
        validation=validation,
        test=test,
        val_draws=common.make_bootstrap_indices(validation["video_code"], replicates, base_seed),
        test_draws=common.make_bootstrap_indices(test["video_code"], replicates, base_seed + 1),
    )
    ctx.manifest = build_manifest(ctx, config_path)
    common.write_json(root / "run_manifest.json", ctx.manifest)
    common.write_json(exp_dir / "run_manifest.json", ctx.manifest)
    return ctx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("stage0", "smoke", "backbone", "variants", "test", "test_exploratory"), required=True)
    parser.add_argument("--variant", action="append", default=None)
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ctx = make_ctx(args.config, args.run_root)
    seeds = args.seed or [int(seed) for seed in ctx.config["training"]["seeds"]]
    epochs = int(args.epochs or ctx.config["training"]["max_epochs"])
    if args.stage == "stage0":
        audit = stage0(ctx)
        print(json.dumps({"stage0": {key: audit[key] for key in ("rows", "rows_match", "file_hashes", "model_params", "vocab_sizes", "routing_checks", "j3_regression_check")}}, ensure_ascii=False))
    elif args.stage == "smoke":
        print(json.dumps(stage_smoke(ctx), ensure_ascii=False, indent=2))
    elif args.stage == "backbone":
        print(json.dumps(stage_backbone(ctx, seeds, epochs), ensure_ascii=False, indent=2))
    elif args.stage == "variants":
        variants = args.variant or ctx.config.get("run_variants") or list(VARIANT_ORDER)
        print(json.dumps(stage_variants(ctx, variants, seeds, epochs), ensure_ascii=False, indent=2))
    elif args.stage == "test":
        result = stage_test(ctx, seeds, force=args.force)
        print(json.dumps(result["per_variant"], ensure_ascii=False, indent=2))
    elif args.stage == "test_exploratory":
        result = stage_test_exploratory(ctx, seeds, force=args.force)
        print(json.dumps(result["per_variant"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
