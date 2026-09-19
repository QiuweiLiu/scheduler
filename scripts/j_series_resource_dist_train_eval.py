#!/usr/bin/env python
"""Resource-v2 training/eval CLI (Phase R line).

Stages
------
``bins``    build train-only bin edges from the J training split and save them.
``cache``   run the FROZEN J3:seed11 model once per split with a forward hook on
            ``head_runtime`` and cache the exact resource-head input feature
            ``z`` (+ runtime labels, slot mask, video codes).  Everything else is
            frozen, so this cache is all the new head ever needs.
``r0``      no training: recompute the J3 baseline report from the cache with the
            SAME metric code that will score R1, and fail loudly unless it
            reproduces the recorded acceptance numbers (coverage 0.5555 / 0.9166
            / 0.9680 and RuntimeQScore 845.0 ms on the validation split).
``smoke``   3-epoch run on a small subset.
``train``   train the discrete head (``--variant R1_dist``), single seed.
``eval``    four-layer report + pre-registered gates against the r0 baseline.
``extend``  R2/R3 placeholders; they refuse to run until the prerequisites exist.

Design constraints honoured here: the frozen checkpoint, the frozen vocabulary and
every existing head are untouched; the new head consumes only cached features, so
R1 is an exact "swap only the resource head" experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import j_series_common as common  # noqa: E402
import j_series_resource_dist as dist  # noqa: E402
import j_series_train_eval as base  # noqa: E402

J3_CHECKPOINT = PROJECT_ROOT / "experiments/EXP-20260911_p9d_j_predictor_acceptance/artifacts/predictor/J3_seed11.pt"
# recorded J3:seed11 acceptance numbers, used only as the r0 sanity reference
J3_ACCEPTANCE = {"coverage": (0.5555, 0.9166, 0.9680), "runtime_qscore_ms": 845.0}
FLOAT_TOL = {"coverage": 0.002, "runtime_qscore_ms": 5.0}


# --------------------------------------------------------------------------- #
def edges_path(ctx: base.Ctx) -> Path:
    return ctx.run_root / "bins.json"


def load_edges(path: Path) -> np.ndarray:
    payload = dist.read_json(path)
    finite = np.asarray(payload["edges_finite"], dtype=np.float64)
    return np.append(finite, math.inf)


def split_arrays(ctx: base.Ctx, split: str) -> Mapping[str, np.ndarray]:
    return {"train": ctx.train, "validation": ctx.validation, "test": ctx.test}[split]


def masked_values(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float64)[mask.astype(bool)]


# --------------------------------------------------------------------------- #
def stage_bins(ctx: base.Ctx, args: argparse.Namespace) -> Dict[str, Any]:
    runtimes = masked_values(ctx.train["runtime_ms"], ctx.train["runtime_ms"] > 0)
    cfg = ctx.config["resource_dist"]
    edges = dist.build_bin_edges(
        runtimes,
        n_bins=int(cfg["bins"]),
        near_zero_edge_ms=float(cfg["near_zero_edge_ms"]),
        upper_quantile=float(cfg["upper_quantile"]),
    )
    payload = {
        "schema_version": dist.SCHEMA_VERSION,
        "n_bins": int(cfg["bins"]),
        "near_zero_edge_ms": float(cfg["near_zero_edge_ms"]),
        "upper_quantile": float(cfg["upper_quantile"]),
        "edges_finite": [float(e) for e in edges[:-1]],
        "overflow": True,
        "train_positive_slots": int(len(runtimes)),
        "fill_counts": np.bincount(dist.bin_index(runtimes, edges), minlength=len(edges) - 1).tolist(),
    }
    dist.write_json(edges_path(ctx), payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "fill_counts"}, indent=1))
    print("bin fill counts:", payload["fill_counts"])
    return payload


# --------------------------------------------------------------------------- #
def stage_cache(ctx: base.Ctx, args: argparse.Namespace) -> Dict[str, Any]:
    model, meta = base.load_checkpoint(J3_CHECKPOINT, ctx, "J3")
    model.eval()
    captured: List[Any] = []

    def pre_hook(_module: Any, inputs: Tuple[Any, ...]) -> None:
        captured.append(inputs[0].detach())

    handle = model.head_runtime.register_forward_pre_hook(pre_hook)
    summary: Dict[str, Any] = {"checkpoint": str(J3_CHECKPOINT), "splits": {}}
    try:
        for split in ("train", "validation", "test"):
            arrays = split_arrays(ctx, split)
            n = arrays["runtime_ms"].shape[0]
            feats, labels, masks, videos = [], [], [], []
            with torch.no_grad():
                for indices in base.split_indices(n, ctx.batch_size):
                    batch = base.to_torch_batch(arrays, indices, ctx.device)
                    captured.clear()
                    base.forward(model, batch, "J3", ctx)
                    if not captured:
                        raise RuntimeError("forward hook captured nothing; head_runtime was not called")
                    feats.append(captured[-1].to(torch.float32).cpu().numpy())
                    labels.append(batch["runtime_ms"].cpu().numpy())
                    masks.append(base.horizon_mask(batch, ctx.horizon).cpu().numpy())
                    videos.append(np.asarray(arrays["video_code"])[indices])
            payload = {
                "features": np.concatenate(feats, axis=0),
                "runtime_ms": np.concatenate(labels, axis=0),
                "mask": np.concatenate(masks, axis=0),
                "video_code": np.concatenate(videos, axis=0),
            }
            out = ctx.run_root / ("cache_%s.npz" % split)
            np.savez_compressed(out, **payload)
            summary["splits"][split] = {
                "rows": int(n),
                "feature_shape": list(payload["features"].shape),
                "file": out.name,
            }
            print("cached %-11s rows=%-6d feature_shape=%s -> %s" % (split, n, payload["features"].shape, out.name))
    finally:
        handle.remove()
    summary["checkpoint_meta"] = {k: meta.get(k) for k in ("sha256", "variant", "seed") if k in meta}
    dist.write_json(ctx.run_root / "cache_summary.json", summary)
    return summary


# --------------------------------------------------------------------------- #
def load_cache(ctx: base.Ctx, split: str) -> Dict[str, np.ndarray]:
    path = ctx.run_root / ("cache_%s.npz" % split)
    if not path.is_file():
        raise SystemExit("missing cache %s; run --stage cache first" % path)
    with np.load(path) as handle:
        return {key: handle[key] for key in handle.files}


def j3_quantiles_from_cache(ctx: base.Ctx, split: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Old head applied to the cached features -> raw-ms quantiles + labels."""

    model, _ = base.load_checkpoint(J3_CHECKPOINT, ctx, "J3")
    model.eval()
    cache = load_cache(ctx, split)
    feats = torch.from_numpy(cache["features"]).to(ctx.device)
    with torch.no_grad():
        log_q = model.head_runtime(feats)
    raw = torch.expm1(log_q).clamp(min=0.0).cpu().numpy()
    preds = {"q50": raw[:, :, 0], "q90": raw[:, :, 1], "q95": raw[:, :, 2]}
    return preds, cache


def stage_r0(ctx: base.Ctx, args: argparse.Namespace) -> Dict[str, Any]:
    report: Dict[str, Any] = {"stage": "r0", "checkpoint": str(J3_CHECKPOINT)}
    for split in ("validation", "test"):
        preds, cache = j3_quantiles_from_cache(ctx, split)
        report[split] = dist.evaluate_quantiles(
            preds["q50"], preds["q90"], preds["q95"], cache["runtime_ms"], cache["mask"], cache["video_code"]
        )
        dist.write_json(ctx.run_root / ("j3_baseline_%s.json" % split), report[split])

    val = report["validation"]
    layer_a, layer_b = val["A_distribution_calibration"], val["B_distribution_accuracy"]
    checks = {
        "coverage_q50": (layer_a["q50"]["coverage"], J3_ACCEPTANCE["coverage"][0], FLOAT_TOL["coverage"]),
        "coverage_q90": (layer_a["q90"]["coverage"], J3_ACCEPTANCE["coverage"][1], FLOAT_TOL["coverage"]),
        "coverage_q95": (layer_a["q95"]["coverage"], J3_ACCEPTANCE["coverage"][2], FLOAT_TOL["coverage"]),
        "runtime_qscore_ms": (layer_b["runtime_qscore_ms"], J3_ACCEPTANCE["runtime_qscore_ms"], FLOAT_TOL["runtime_qscore_ms"]),
    }
    report["reproduction"] = {
        name: {"computed": float(value), "recorded": float(target), "tolerance": float(tol), "pass": abs(value - target) <= tol}
        for name, (value, target, tol) in checks.items()
    }
    report["reproduction_pass"] = all(item["pass"] for item in report["reproduction"].values())
    print(json.dumps(report["reproduction"], indent=1))
    if not report["reproduction_pass"]:
        print("R0 FAILED: the new metric code does not reproduce the recorded J3 acceptance numbers", flush=True)
    else:
        print("R0 PASSED: metric code reproduces the recorded J3:seed11 acceptance numbers")
    dist.write_json(ctx.run_root / "r0_report.json", report)
    return report


# --------------------------------------------------------------------------- #
def flat_training_tensors(ctx: base.Ctx, split: str) -> Tuple[Any, Any, Any]:
    cache = load_cache(ctx, split)
    mask = cache["mask"].astype(np.float32)
    runtime = cache["runtime_ms"].astype(np.float64)
    valid = (mask > 0) & (runtime > 0.0)
    features = cache["features"][valid.astype(bool)]
    labels = runtime[valid.astype(bool)]
    band = dist.band_index(labels)
    return features, labels, band


def train_head(
    ctx: base.Ctx,
    seed: int,
    epochs: int,
    max_rows: Optional[int] = None,
    hidden: Optional[int] = None,
) -> Dict[str, Any]:
    cfg = ctx.config["resource_dist"]
    weights = cfg["loss_weights"]
    edges = load_edges(edges_path(ctx))
    base.set_seed(seed)

    features, labels, band = flat_training_tensors(ctx, "train")
    if max_rows:
        features, labels, band = features[:max_rows], labels[:max_rows], band[:max_rows]
    target_idx = dist.bin_index(labels, edges)
    scale = float(np.median(labels))
    in_dim = int(features.shape[1])
    n_bins = len(edges) - 1

    head = dist.DiscreteRuntimeHead(in_dim, n_bins, hidden=hidden).to(ctx.device)
    optimiser = torch.optim.AdamW(head.parameters(), lr=float(cfg["learning_rate"]))
    feat_t = torch.from_numpy(features.astype(np.float32)).to(ctx.device)
    tgt_t = torch.from_numpy(target_idx.astype(np.int64)).to(ctx.device)
    band_t = torch.from_numpy(band.astype(np.int64)).to(ctx.device)
    ms_t = torch.from_numpy(labels.astype(np.float32)).to(ctx.device)
    ones = torch.ones_like(ms_t)

    history: List[Dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        head.train()
        order = torch.randperm(feat_t.shape[0], device=ctx.device)
        totals = {"nll": 0.0, "rps": 0.0, "point": 0.0, "band": 0.0, "steps": 0}
        for start in range(0, feat_t.shape[0], ctx.batch_size):
            idx = order[start : start + ctx.batch_size]
            logits = head(feat_t[idx])
            derived = dist.derive_from_probs(F.softmax(logits, dim=-1), edges)
            nll = dist.nll_loss(logits, tgt_t[idx], ones[idx])
            rps = dist.rps_loss(logits, tgt_t[idx], ones[idx], edges)
            point = dist.pseudo_huber_point(derived["mean_ms"], ms_t[idx], scale, float(cfg["point_delta"]), ones[idx])
            band_ce = dist.coarse_band_ce(logits, band_t[idx], ones[idx], edges)
            loss = (
                float(weights["nll"]) * nll
                + float(weights["ranked_probability"]) * rps
                + float(weights["point_pseudohuber"]) * point
                + float(weights["coarse_band_ce"]) * band_ce
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            totals["nll"] += float(nll.detach()) * len(idx)
            totals["rps"] += float(rps.detach()) * len(idx)
            totals["point"] += float(point.detach()) * len(idx)
            totals["band"] += float(band_ce.detach()) * len(idx)
            totals["steps"] += len(idx)
        rows = max(1, totals["steps"])
        epoch_row = {
            "epoch": epoch,
            "loss_nll": totals["nll"] / rows,
            "loss_rps": totals["rps"] / rows,
            "loss_point": totals["point"] / rows,
            "loss_band": totals["band"] / rows,
        }
        val_metrics = evaluate_head(ctx, head, edges, "validation")
        epoch_row["val_calibration_error"] = val_metrics["A_distribution_calibration"]["mean_abs_calibration_error"]
        epoch_row["val_mae_log"] = val_metrics["C_node_discrimination"]["mae_log"]
        epoch_row["val_spearman"] = val_metrics["C_node_discrimination"]["spearman_point"]
        history.append(epoch_row)
        print(json.dumps(epoch_row), flush=True)

    variant_dir = ctx.run_root / "R1_dist"
    variant_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "edges": edges.tolist(), "in_dim": in_dim, "hidden": hidden}, variant_dir / ("seed%d.pt" % seed))
    dist.write_json(variant_dir / ("seed%d_history.json" % seed), {"seed": seed, "epochs": epochs, "history": history, "scale": scale})
    return {"seed": seed, "epochs": epochs, "history": history}


def evaluate_head(ctx: base.Ctx, head: Any, edges: np.ndarray, split: str, max_rows: Optional[int] = None) -> Dict[str, Any]:
    cache = load_cache(ctx, split)
    features = cache["features"]
    runtime = cache["runtime_ms"]
    mask = cache["mask"].astype(bool)
    if max_rows:
        features, runtime, mask = features[:max_rows], runtime[:max_rows], mask[:max_rows]
    head.eval()
    feats = torch.from_numpy(features.reshape(-1, features.shape[-1]).astype(np.float32)).to(ctx.device)
    probs_chunks = []
    with torch.no_grad():
        for start in range(0, feats.shape[0], 8192):
            probs_chunks.append(F.softmax(head(feats[start : start + 8192]), dim=-1).cpu().numpy())
    probs = np.concatenate(probs_chunks, axis=0).reshape(runtime.shape[0], runtime.shape[1], -1)
    flat_mask = mask.reshape(-1)
    flat_runtime = runtime.reshape(-1)
    flat_probs = probs.reshape(-1, probs.shape[-1])
    video = np.repeat(cache["video_code"], runtime.shape[1])
    keep = flat_mask & (flat_runtime > 0)
    return dist.evaluate_distribution(flat_probs[keep], flat_runtime[keep], edges, video[keep])


# --------------------------------------------------------------------------- #
def stage_eval(ctx: base.Ctx, args: argparse.Namespace) -> Dict[str, Any]:
    variant = args.variant or "R1_dist"
    seed = (args.seed or [11])[0]
    checkpoint = ctx.run_root / variant / ("seed%d.pt" % seed)
    if not checkpoint.is_file():
        raise SystemExit("missing trained head %s; run --stage train first" % checkpoint)
    payload = torch.load(checkpoint, map_location=ctx.device)
    edges = np.asarray(payload["edges"], dtype=np.float64)
    head = dist.DiscreteRuntimeHead(int(payload["in_dim"]), len(edges) - 1, hidden=payload.get("hidden")).to(ctx.device)
    head.load_state_dict(payload["state_dict"])

    baseline_path = ctx.run_root / "j3_baseline_validation.json"
    if not baseline_path.is_file():
        raise SystemExit("missing %s; run --stage r0 first" % baseline_path)
    baseline = dist.read_json(baseline_path)

    out: Dict[str, Any] = {"variant": variant, "seed": seed, "splits": {}}
    for split in ("validation", "test"):
        report = evaluate_head(ctx, head, edges, split)
        out["splits"][split] = report
        dist.write_json(ctx.run_root / variant / ("seed%d_%s.json" % (seed, split)), report)
    gates = dist.evaluate_gates(out["splits"]["validation"], baseline)
    out["gates"] = gates
    dist.write_json(ctx.run_root / variant / ("seed%d_gates.json" % seed), gates)
    print(json.dumps(gates, indent=1, default=str))
    print("R1 GATES:", "PASS" if gates["headline"]["pass"] else "FAIL")
    return out


# --------------------------------------------------------------------------- #
def stage_extend(ctx: base.Ctx, args: argparse.Namespace) -> Dict[str, Any]:
    raise SystemExit(
        "R2/R3 are intentionally not enabled yet.\n"
        "- R2 (future activity head) needs a dataset v2 that carries a canonical per-slot\n"
        "  ``slot_activity`` label; j_series_dataset_v1 does not have it.  Adding it means\n"
        "  re-deriving the future slots from the traces with the activity mapping, which is a\n"
        "  data-side change and must not silently alter v1.\n"
        "- R3 (joint fine-tune) must only start after R2 passes its gates.\n"
        "See PHASE_R_PLAN.md for the prerequisite chain."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("bins", "cache", "r0", "smoke", "train", "eval", "extend"), required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ctx = base.make_ctx(args.config, args.run_root)
    if args.stage == "bins":
        stage_bins(ctx, args)
    elif args.stage == "cache":
        stage_cache(ctx, args)
    elif args.stage == "r0":
        report = stage_r0(ctx, args)
        return 0 if report["reproduction_pass"] else 1
    elif args.stage == "smoke":
        train_head(ctx, (args.seed or [11])[0], args.epochs or 3, max_rows=args.max_rows or 512, hidden=args.hidden)
        print("smoke finished")
    elif args.stage == "train":
        train_head(ctx, (args.seed or [11])[0], args.epochs or int(ctx.config["resource_dist"]["epochs"]), hidden=args.hidden)
    elif args.stage == "eval":
        stage_eval(ctx, args)
    else:
        stage_extend(ctx, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
