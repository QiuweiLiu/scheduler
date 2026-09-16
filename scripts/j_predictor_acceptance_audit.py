#!/usr/bin/env python3
"""J predictor acceptance audit (freeze candidate + P9f-style audit on the v3.1 contract).

Official tier: validation. Test numbers are inherited from the frozen J determination.
Audits: structure/content/behavior quality, resource calibration (coverage/Brier/ECE),
and missing-variable identifiability (oracle residency proxy, workload proxies).
Read-only w.r.t. the frozen J experiments; fits/selects nothing.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import j_series_common as common  # noqa: E402
from scripts import j_series_train_eval as jte  # noqa: E402

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"torch is required: {exc}")

TAUS = (0.50, 0.90, 0.95)


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    diff = y - q
    return np.maximum(tau * diff, (tau - 1.0) * diff)


def collect(model, variant: str, ctx: jte.Ctx) -> Dict[str, np.ndarray]:
    """Collect predictions on the validation split (GT arrays passed through)."""
    model.eval()
    preds: Dict[str, List[np.ndarray]] = {}
    with torch.no_grad():
        for indices in jte.split_indices(len(ctx.validation["length"]), ctx.batch_size):
            batch = jte.to_torch_batch(ctx.validation, indices, ctx.device)
            o = jte.forward(model, batch, variant, ctx)
            payload = {
                "length_logits": o["structure"]["length_logits"].cpu().numpy(),
                "termination_logit": o["structure"]["termination_logit"].cpu().numpy(),
                "next_role_logits": o["behavior"]["next_role_logits"].cpu().numpy(),
                "next_family_logits": o["behavior"]["next_family_logits"].cpu().numpy(),
                "rt_pred": torch.expm1(o["resource"]["runtime_log_quantiles"]).clamp(min=0.0).cpu().numpy(),
                "dur_pred": torch.expm1(o["resource"]["load_dur_log_quantiles"]).clamp(min=0.0).cpu().numpy(),
                "occ_prob": torch.sigmoid(o["resource"]["load_occ_logit"]).cpu().numpy(),
            }
            for field in common.SLOT_CAT_FIELDS + ("merged", "retry", "nested"):
                payload[f"attr_{field}"] = o["attributes"][field].cpu().numpy()
            for key, value in payload.items():
                preds.setdefault(key, []).append(value)
    arrays = {key: np.concatenate(value) for key, value in preds.items()}
    for key in (
        "length", "termination", "next_role", "next_family",
        "runtime_ms", "load_ms", "load_occ",
        "slot_node_type", "slot_role", "slot_action_family", "slot_model_class",
        "slot_merged", "slot_retry", "slot_nested",
    ):
        arrays[key] = ctx.validation[key]
    return arrays


def audit_metrics(arrays: Mapping[str, np.ndarray], horizon: int, class_names: List[str] | None = None) -> Dict[str, Any]:
    n = arrays["length"].shape[0]
    slots = np.arange(horizon)[None, :] < arrays["length"][:, None]
    slot_f = slots.astype(np.float64)

    # --- structure ---
    length_pred = arrays["length_logits"].argmax(axis=1)
    length_mae = np.abs(length_pred - arrays["length"]).astype(np.float64)
    term_p = 1.0 / (1.0 + np.exp(-arrays["termination_logit"]))
    term_p = np.clip(term_p, 1e-12, 1 - 1e-12)
    term_bce = -(arrays["termination"] * np.log(term_p) + (1 - arrays["termination"]) * np.log(1 - term_p))
    by_length = {}
    for value in range(horizon + 1):
        mask = arrays["length"] == value
        if mask.any():
            by_length[str(value)] = {"n": int(mask.sum()), "length_mae": float(length_mae[mask].mean())}

    # --- content ---
    field_acc: Dict[str, float] = {}
    for field in common.SLOT_CAT_FIELDS:
        gt = arrays[f"slot_{field}"]
        pred = arrays[f"attr_{field}"].argmax(axis=-1)
        correct = ((pred == gt) & (gt >= 0)).astype(np.float64) * slot_f
        field_acc[field] = float(correct.sum() / max(1.0, slot_f.sum()))
    merged_pred = (arrays["attr_merged"] > 0).astype(np.int64)
    field_acc["merged"] = float((((merged_pred == arrays["slot_merged"]) & (arrays["slot_merged"] >= 0)).astype(np.float64) * slot_f).sum() / max(1.0, slot_f.sum()))
    retry_pred = (arrays["attr_retry"] > 0).astype(np.int64)
    field_acc["retry"] = float((((retry_pred == arrays["slot_retry"]) & (arrays["slot_retry"] >= 0)).astype(np.float64) * slot_f).sum() / max(1.0, slot_f.sum()))
    nested_valid = slot_f * (arrays["slot_merged"] == 1).astype(np.float64)
    nested_pred = arrays["attr_nested"].argmax(axis=-1)
    nested_correct = ((nested_pred == arrays["slot_nested"]) & (arrays["slot_nested"] >= 0)).astype(np.float64) * nested_valid
    field_acc["nested"] = float(nested_correct.sum() / max(1.0, nested_valid.sum()))

    # --- behavior ---
    behavior_mask = arrays["next_role"] >= 0
    next_role_acc = float(np.mean((arrays["next_role_logits"].argmax(axis=1) == arrays["next_role"])[behavior_mask]))
    next_family_acc = float(np.mean((arrays["next_family_logits"].argmax(axis=1) == arrays["next_family"])[behavior_mask]))

    # --- resource: runtime ---
    rt_valid = slot_f * (arrays["runtime_ms"] > 0).astype(np.float64)
    rt_pb, rt_cover = [], []
    for k, tau in enumerate(TAUS):
        pb = pinball(arrays["runtime_ms"], arrays["rt_pred"][:, :, k], tau) * rt_valid
        rt_pb.append(float(pb.sum() / max(1.0, rt_valid.sum())))
        cover = ((arrays["runtime_ms"] <= arrays["rt_pred"][:, :, k]).astype(np.float64) * rt_valid).sum() / max(1.0, rt_valid.sum())
        rt_cover.append(float(cover))

    # --- load occurrence ---
    occ_valid = slot_f * (arrays["load_occ"] >= 0).astype(np.float64)
    occ_gt = arrays["load_occ"].clip(min=0)
    brier = float((((arrays["occ_prob"] - occ_gt) ** 2) * occ_valid).sum() / max(1.0, occ_valid.sum()))
    bins = np.clip((arrays["occ_prob"] * 10).astype(int), 0, 9)
    ece_num, ece_den = 0.0, 0.0
    for b in range(10):
        sel = (bins == b) & (occ_valid > 0)
        cnt = int(sel.sum())
        if cnt:
            gap = abs(float(arrays["occ_prob"][sel].mean()) - float(occ_gt[sel].mean()))
            ece_num += gap * cnt
            ece_den += cnt
    ece = float(ece_num / max(1.0, ece_den))

    # --- load duration ---
    dur_valid = slot_f * (arrays["load_occ"] == 1).astype(np.float64)
    dur_pb, dur_cover = [], []
    for k, tau in enumerate(TAUS):
        pb = pinball(arrays["load_ms"], arrays["dur_pred"][:, :, k], tau) * dur_valid
        dur_pb.append(float(pb.sum() / max(1.0, dur_valid.sum())))
        cover = ((arrays["load_ms"] <= arrays["dur_pred"][:, :, k]).astype(np.float64) * dur_valid).sum() / max(1.0, dur_valid.sum())
        dur_cover.append(float(cover))

    # --- identifiability: runtime error by GT model class ---
    by_class: Dict[str, Any] = {}
    class_names = list(class_names or [])
    for idx in sorted(set(arrays["slot_model_class"][slot_f > 0].tolist())):
        sel = slot_f * (arrays["slot_model_class"] == idx).astype(np.float64)
        if sel.sum() < 20:
            continue
        per_tau = [pinball(arrays["runtime_ms"], arrays["rt_pred"][:, :, k], tau) * sel for k, tau in enumerate(TAUS)]
        pb = np.stack(per_tau, axis=0).mean(axis=0)
        label = class_names[idx - 2] if 0 <= idx - 2 < len(class_names) else str(int(idx))
        by_class[label] = {"n": int(sel.sum()), "runtime_pb": float(pb.sum() / max(1.0, sel.sum()))}

    return {
        "rows": n,
        "structure": {
            "length_mae": float(length_mae.mean()),
            "length_mae_by_gt_length": by_length,
            "termination_bce": float(term_bce.mean()),
            "termination_acc": float(((term_p > 0.5).astype(int) == arrays["termination"]).mean()),
        },
        "content": {"per_field_accuracy": field_acc, "mean_of_fields": float(np.mean(list(field_acc.values())))},
        "behavior": {"next_role_acc": next_role_acc, "next_family_acc": next_family_acc},
        "resource": {
            "runtime_pinball_by_tau": rt_pb,
            "runtime_coverage_by_tau": rt_cover,
            "runtime_pinball_mean": float(np.mean(rt_pb)),
            "load_occurrence": {"brier": brier, "ece": ece, "base_rate": float(occ_gt[occ_valid > 0].mean())},
            "load_duration_pinball_by_tau": dur_pb,
            "load_duration_coverage_by_tau": dur_cover,
            "load_duration_pinball_mean": float(np.mean(dur_pb)),
            "memory": "head not trained (absent from L_R); do not expose; availability 37,553/60,925 slots",
        },
        "identifiability": {"runtime_pinball_by_model_class": by_class},
    }


def residency_identifiability(ctx: jte.Ctx, config: Mapping[str, Any]) -> Dict[str, Any]:
    table_path = Path(config["resource_table"])
    keys = set()
    with gzip.open(Path(config["output_root"]) / "j_validation.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for slot in row["future"]:
                keys.add((str(row["run_id"]), str(slot["node_id"])))
    warm_loads: List[float] = []
    cold_loads: List[float] = []
    warm_n = cold_n = 0
    clip_missing = clip_total = q_missing = q_total = 0
    with gzip.open(table_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            node = json.loads(line)
            if str(node.get("split")) != "validation":
                continue
            clip_total += 1
            q_total += 1
            if node.get("clip_len") is None:
                clip_missing += 1
            if node.get("query_char_len") is None:
                q_missing += 1
            key = (str(node["run_id"]), str(node["node_id"]))
            if key not in keys:
                continue
            load = node.get("load_ms")
            if load is None or float(load) <= 0:
                continue
            if bool(node.get("prefix_model_reuse")):
                warm_n += 1
                warm_loads.append(float(load))
            else:
                cold_n += 1
                cold_loads.append(float(load))
    return {
        "note": "oracle proxy (prefix_model_reuse) is not a deployable feature; used only to quantify the missing variable",
        "warm_load_events": warm_n,
        "cold_load_events": cold_n,
        "warm_load_ms_mean": float(np.mean(warm_loads)) if warm_loads else None,
        "warm_load_ms_median": float(np.median(warm_loads)) if warm_loads else None,
        "cold_load_ms_mean": float(np.mean(cold_loads)) if cold_loads else None,
        "cold_load_ms_median": float(np.median(cold_loads)) if cold_loads else None,
        "workload_proxy_missing_rate": {"clip_len": clip_missing / max(1, clip_total), "query_char_len": q_missing / max(1, q_total)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    ctx = jte.make_ctx(args.config, None)
    candidate = config["candidate"]
    source_root = Path(candidate["source_run_root"])
    art_dir = ctx.exp_dir / "artifacts" / "predictor"
    art_dir.mkdir(parents=True, exist_ok=True)

    specs = [("primary", candidate["primary"])] + [("robustness", item) for item in candidate["robustness"]] + [("fallback", candidate["fallback"])]
    audit: Dict[str, Any] = {"experiment_id": config["experiment_id"], "tier": "validation (official)", "models": {}}
    hashes = {}
    for role, spec in specs:
        variant, seed = spec["variant"], spec["seed"]
        path = source_root / "runs" / variant / f"seed{seed}" / "checkpoint.pt"
        model, payload = jte.load_checkpoint(path, ctx, variant)
        arrays = collect(model, variant, ctx)
        key = f"{variant}:seed{seed}"
        audit["models"][key] = {
            "role": role,
            "selected_epoch": int(payload["epoch"]),
            "checkpoint": str(path),
            "checkpoint_sha256": common.sha256_file(path),
            "metrics": audit_metrics(arrays, ctx.horizon, list(ctx.vocabs.slot["model_class"].values)),
        }
        target = art_dir / f"{variant}_seed{seed}.pt"
        target.write_bytes(path.read_bytes())
        hashes[key] = common.sha256_file(target)
        print(f"audited {key} ({role})")

    audit["identifiability"] = {"residency_oracle_proxy": residency_identifiability(ctx, config)}
    audit["test_headline_inherited"] = {
        "source": "outputs/j_series_joint_resource/test_eval.json (frozen; not recomputed)",
        "J3": "runtime delta vs B1: -78.1/-77.1/-64.0 ms (all seeds CI upper < 0); load duration +6.7%/+4.7%/+4.3%",
        "J2": "runtime delta vs B1: -49.0/-65.4/-29.9 ms (all seeds CI upper < 0); load duration +7.3%/+3.0%/+5.4%",
        "verdict": "no Core GO (load-duration NI); test consumed once",
    }
    common.write_json(ctx.run_root / "acceptance_audit.json", jte.json_safe(audit))
    (ctx.exp_dir / "acceptance_audit.json").write_bytes((ctx.run_root / "acceptance_audit.json").read_bytes())
    frozen = {
        "selection_rule": candidate["selection_rule"],
        "primary": "J3:seed11",
        "robustness": ["J3:seed22", "J3:seed33"],
        "fallback": "B1:seed11",
        "artifact_sha256": hashes,
        "tier": candidate["tier"],
    }
    common.write_json(ctx.exp_dir / "frozen_candidate.json", jte.json_safe(frozen))
    print(json.dumps({"artifacts": hashes}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
