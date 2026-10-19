#!/usr/bin/env python3
"""Histres (latest predictor line): run F0/F1 checkpoints on the TRAIN and
VALIDATION splits of j_series_dataset_histres_v2.

Saves per-row predictions (JSONL) + summary metrics per split/arm/seed.
The sealed J test is NEVER read (build_ctx loads train+validation only).

Usage:
  python scripts/histres_predict_splits.py --out <dir> [--arms F0 F1] [--seeds 11 22 33]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import histres_train_f0f1 as H  # noqa: E402
import j_series_common as common  # noqa: E402
import j_series_train_eval as te  # noqa: E402


def run_split(model, arrays, variant, ctx, split: str, out_jsonl: Path) -> dict:
    model.eval()
    n = len(arrays["length"])
    collected: dict = {}
    rows_written = 0
    with torch.no_grad(), open(out_jsonl, "w", encoding="utf-8") as f:
        for indices in te.split_indices(n, ctx.batch_size):
            batch = te.to_torch_batch(arrays, indices, ctx.device)
            outputs = te.forward(model, batch, variant, ctx)
            resource = outputs["resource"]
            behavior = outputs["behavior"]
            structure = outputs["structure"]
            attrs = outputs["attributes"]
            role_logits = behavior["next_role_logits"]
            fam_logits = behavior["next_family_logits"]
            role_prob, role_pred = torch.softmax(role_logits, dim=-1).max(dim=-1)
            fam_prob, fam_pred = torch.softmax(fam_logits, dim=-1).max(dim=-1)
            if "runtime_logits" in resource:
                import j_series_resource_dist as _dist

                bins = model.runtime_bins
                probs = torch.softmax(resource["runtime_logits"], dim=-1)
                reps = torch.as_tensor(bins["representatives_ms"], device=probs.device, dtype=probs.dtype)
                views = _dist.derive_from_probs(probs, reps.cpu().numpy())
                def _np(x):
                    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
                runtime_ms = np.stack([_np(views["q50_ms"]), _np(views["q90_ms"]), _np(views["q95_ms"])], axis=-1)
            else:
                runtime_ms = torch.expm1(resource["runtime_log_quantiles"]).clamp(min=0.0).cpu().numpy()
            load_dur = torch.expm1(resource["load_dur_log_quantiles"]).clamp(min=0.0).cpu().numpy()
            load_occ = torch.sigmoid(resource["load_occ_logit"]).cpu().numpy()
            term = torch.sigmoid(structure["termination_logit"]).cpu().numpy()
            attr_preds = {}
            for field in list(common.SLOT_CAT_FIELDS) + ["merged", "retry", "nested"]:
                arr = np.asarray(attrs[field].argmax(dim=-1).cpu().numpy())
                if arr.ndim == 2:
                    arr = arr[:, 0]
                attr_preds[field] = arr
            numpy_outputs = {
                "runtime_ms": runtime_ms if isinstance(runtime_ms, np.ndarray) else runtime_ms.cpu().numpy(),
                "load_dur_ms": load_dur,
                "length_logits": structure["length_logits"].cpu().numpy(),
                "termination_logit": structure["termination_logit"].cpu().numpy(),
                "next_role_logits": role_logits.cpu().numpy(),
                "next_family_logits": fam_logits.cpu().numpy(),
                "load_occ_logit": resource["load_occ_logit"].cpu().numpy(),
            }
            for field in list(common.SLOT_CAT_FIELDS) + ["merged", "retry", "nested"]:
                numpy_outputs[f"attr_{field}"] = attrs[field].cpu().numpy()
            numpy_batch = {key: value[indices] for key, value in arrays.items()
                           if isinstance(value, np.ndarray) and key != "video_code"}
            metrics = common.per_row_metrics(numpy_outputs, numpy_batch)
            for key, value in metrics.items():
                collected.setdefault(key, []).append(np.asarray(value))
            rp = role_pred.cpu().numpy(); rpr = role_prob.cpu().numpy()
            fp = fam_pred.cpu().numpy(); fpr = fam_prob.cpu().numpy()
            rm = np.asarray(runtime_ms if isinstance(runtime_ms, np.ndarray) else runtime_ms.cpu().numpy())
            if rm.ndim == 2:
                rm = rm[:, None, :]
            ld = np.asarray(load_dur)
            if ld.ndim == 2:
                ld = ld[:, None, :]
            lo = np.asarray(load_occ)
            if lo.ndim == 1:
                lo = lo[:, None]
            tm = np.asarray(term)
            if tm.ndim == 2:
                tm = tm[:, 0]
            for j, row_idx in enumerate(indices.tolist()):
                rec = {"row": int(row_idx), "split": split, "arm": variant,
                       "next_role": int(rp[j]), "next_role_prob": round(float(rpr[j]), 4),
                       "next_family": int(fp[j]), "next_family_prob": round(float(fpr[j]), 4),
                       "runtime_q50_ms": round(float(rm[j, 0, 0]), 2),
                       "runtime_q90_ms": round(float(rm[j, 0, 1]), 2),
                       "runtime_q95_ms": round(float(rm[j, 0, 2]), 2),
                       "load_dur_q50_ms": round(float(ld[j, 0, 0]), 2),
                       "load_occ_prob": round(float(lo[j, 0]), 4),
                       "termination_prob": round(float(tm[j]), 4)}
                for field, arr in attr_preds.items():
                    rec[f"attr_{field}"] = int(arr[j])
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                rows_written += 1
    def _flat_mean(values):
        parts = [np.asarray(x, dtype=np.float64).reshape(-1) for x in values]
        flat = np.concatenate(parts) if parts else np.array([], dtype=np.float64)
        return float(np.nanmean(flat)) if flat.size else float("nan")
    point = {key: _flat_mean(v) for key, v in collected.items()}
    return {"n_rows": rows_written, "metrics": point}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=["F0", "F1"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    ap.add_argument("--splits", nargs="+", default=["train", "validation"])
    ap.add_argument("--allow-test", action="store_true",
                    help="explicitly open the sealed J test split for data production")
    args = ap.parse_args()

    config = json.loads(H.CONFIG.read_text(encoding="utf-8"))
    config = json.loads(json.dumps(config))
    config["variants"].update({"F0": dict(te.DEFAULT_VARIANTS["F0"]), "F1": dict(te.DEFAULT_VARIANTS["F1"])})
    config["model"]["runtime_distribution_head"] = True
    config["model"]["runtime_head_hidden"] = 64
    config["training"]["max_epochs"] = 30
    config["training"]["seeds"] = [11, 22, 33]
    ctx = H.build_ctx(config)
    if "test" in args.splits:
        if not args.allow_test:
            print("refusing: --splits includes test but --allow-test was not passed (sealed split)")
            return 2
        test_rows = list(common.read_jsonl_gz(H.HISTRES / "histres_test.jsonl.gz"))
        test = common.encode_rows(test_rows, ctx.vocabs, ctx.horizon)
        test["video_code"] = np.asarray([te.stable_video_code(r["video_id"]) for r in test_rows], dtype=np.int64)
        ctx.test = test
    print("device:", ctx.device, "| train", len(ctx.train["length"]), "| validation", len(ctx.validation["length"]),
          "| test", len(ctx.test["length"]) if ctx.test else 0, flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    report = {"schema": "histres-split-predictions", "arms": args.arms, "seeds": args.seeds,
              "per_seed": {}, "vocab": ctx.vocabs.to_json()}
    for seed in args.seeds:
        entry = {}
        for arm in args.arms:
            ckpt = H.OUT_ROOT / "runs" / arm / f"seed{seed}" / "checkpoint.pt"
            if not ckpt.is_file():
                entry[arm] = {"status": "missing", "path": str(ckpt)}
                continue
            model, _meta = te.load_checkpoint(ckpt, ctx, arm)
            sub = {}
            for split in args.splits:
                arrays = getattr(ctx, split)
                out_jsonl = args.out / f"predictions_{arm}_seed{seed}_{split}.jsonl"
                sub[split] = run_split(model, arrays, arm, ctx, split, out_jsonl)
            entry[arm] = sub
        report["per_seed"][str(seed)] = entry
        print(f"seed {seed} done", flush=True)
    (args.out / "predict_report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for seed, entry in report["per_seed"].items():
        for arm, sub in entry.items():
            if isinstance(sub, dict) and "train" in sub:
                t = sub["train"]["metrics"]; v = sub["validation"]["metrics"]
                keys = [k for k in t if "acc" in k][:5]
                print(f"seed{seed} {arm}: " + " | ".join(
                    f"{k} train={t[k]:.3f} val={v[k]:.3f}" for k in keys))
    print("report:", args.out / "predict_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
