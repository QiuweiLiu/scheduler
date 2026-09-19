#!/usr/bin/env python
"""Resource-v2 training/eval CLI (Phase R line).

Stages
------
``bins``        build the train-only bin spec (edges, empirical representatives,
                bin->band map) and refuse to continue if the overflow mass exceeds
                the construction budget.
``cache``       run the FROZEN J3:seed11 model once per split with a forward pre-hook
                on ``head_runtime``, cache the exact resource-head input feature, and
                **verify element-wise that ``head(cached_z)`` equals the normal forward
                output** before anything downstream is allowed to use the cache.
``r0``          no training: recompute the J3 baseline with the SAME metric code that
                will score R1, assert the recorded acceptance numbers, and optionally
                cross-check a packer artifact row by row.
``smoke``       short run on a subset.
``train``       train the discrete head; saves ``best`` (pre-registered epoch rule) and
                ``last`` checkpoints.
``eval-val``    validation-only report + gates.  This is the only stage R1/R2 model
                selection may look at.
``test-final``  the single frozen-test look for a locked winner; refuses to overwrite.
``extend``      R2/R3 placeholders that refuse to run until their prerequisites exist.

Frozen-checkpoint guarantee: the v1 dataset, the vocabulary, the checkpoint and every
existing head are untouched; the new head consumes only cached features, so R1 is an
exact "swap only the resource head" experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
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
J3_SHA256 = "0ee8ded4f92553853026ee24a3c320f21d524f9d2c60de841091430d14949c77"
J3_ACCEPTANCE = {"coverage": (0.5555, 0.9166, 0.9680), "runtime_qscore_ms": 845.0}
FLOAT_TOL = {"coverage": 0.002, "runtime_qscore_ms": 5.0}
FEATURE_EQUIVALENCE_TOL = 1e-6


# --------------------------------------------------------------------------- #
def spec_path(ctx: base.Ctx) -> Path:
    return ctx.run_root / "bins.json"


def load_spec(ctx: base.Ctx) -> Dict[str, Any]:
    path = spec_path(ctx)
    if not path.is_file():
        raise SystemExit("missing %s; run --stage bins first" % path)
    spec = dist.read_json(path)
    # fail closed: a spec built by an older, buggy construction must not be reused
    if spec.get("schema_version") != dist.SCHEMA_VERSION:
        raise SystemExit("bin spec schema %r != %r; rebuild with --stage bins" % (spec.get("schema_version"), dist.SCHEMA_VERSION))
    if float(spec.get("overflow_fraction", 1.0)) > dist.MAX_OVERFLOW_FRACTION:
        raise SystemExit(
            "bin spec overflow fraction %.4f exceeds the %.4f budget; rebuild"
            % (float(spec["overflow_fraction"]), dist.MAX_OVERFLOW_FRACTION)
        )
    return spec


def split_arrays(ctx: base.Ctx, split: str) -> Mapping[str, np.ndarray]:
    return {"train": ctx.train, "validation": ctx.validation, "test": ctx.test}[split]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_guard() -> str:
    digest = sha256_file(J3_CHECKPOINT)
    if digest != J3_SHA256:
        raise SystemExit("frozen checkpoint hash changed: %s != %s" % (digest, J3_SHA256))
    return digest


# --------------------------------------------------------------------------- #
def stage_bins(ctx: base.Ctx, args: argparse.Namespace) -> Dict[str, Any]:
    cfg = ctx.config["resource_dist"]
    raw = ctx.train["runtime_ms"].astype(np.float64)
    # slot validity comes from the bounded future length, not from a stored mask
    lengths = ctx.train["length"].astype(np.int64)
    slots = np.arange(raw.shape[1])[None, :] < lengths[:, None]
    if np.any((raw > 0) & ~slots):
        raise SystemExit("dataset inconsistency: positive runtime outside the valid future slots")
    runtimes = raw[raw > 0]
    spec = dist.build_bin_spec(
        runtimes,
        n_bins=int(cfg["bins"]),
        near_zero_edge_ms=float(cfg["near_zero_edge_ms"]),
        upper_quantile=float(cfg["upper_quantile"]),
    )
    spec["train_positive_slots"] = int(len(runtimes))
    spec["checkpoint_sha256"] = checkpoint_guard()
    dist.write_json(spec_path(ctx), spec)
    print("n_bins=%d  overflow_fraction=%.5f (budget %.4f)  last_finite_edge=%.1f ms"
          % (spec["n_bins"], spec["overflow_fraction"], dist.MAX_OVERFLOW_FRACTION, spec["edges"][-2]))
    print("first bin representative = %.4f ms   band_of_bin = %s" % (spec["representatives_ms"][0], spec["band_of_bin"]))
    if spec["overflow_fraction"] > dist.MAX_OVERFLOW_FRACTION:
        raise SystemExit("overflow fraction %.4f exceeds budget %.4f" % (spec["overflow_fraction"], dist.MAX_OVERFLOW_FRACTION))
    return spec


def stage_cache(ctx: base.Ctx, args: argparse.Namespace) -> Dict[str, Any]:
    sha = checkpoint_guard()
    model, meta = base.load_checkpoint(J3_CHECKPOINT, ctx, "J3")
    model.eval()
    captured: List[Any] = []

    def pre_hook(_module: Any, inputs: Tuple[Any, ...]) -> None:
        captured.append(inputs[0].detach())

    handle = model.head_runtime.register_forward_pre_hook(pre_hook)
    dataset_files = {s: Path(ctx.config["output_root"]) / ("j_%s.jsonl.gz" % s) for s in ("train", "validation", "test")}
    summary: Dict[str, Any] = {
        "checkpoint": str(J3_CHECKPOINT),
        "checkpoint_sha256": sha,
        "config_sha256": sha256_file(Path(args.config)),
        "dataset_file_sha256": {s: sha256_file(p) for s, p in dataset_files.items()},
        "feature_equivalence_tolerance": FEATURE_EQUIVALENCE_TOL,
        "splits": {},
    }
    try:
        for split in ("train", "validation", "test"):
            arrays = split_arrays(ctx, split)
            n = arrays["runtime_ms"].shape[0]
            feats, labels, masks, videos, equiv = [], [], [], [], []
            with torch.no_grad():
                for indices in base.split_indices(n, ctx.batch_size):
                    batch = base.to_torch_batch(arrays, indices, ctx.device)
                    captured.clear()
                    outputs = base.forward(model, batch, "J3", ctx)
                    if not captured:
                        raise RuntimeError("forward hook captured nothing; head_runtime was not called")
                    direct = outputs["resource"]["runtime_log_quantiles"]
                    replay = model.head_runtime(captured[-1])
                    equiv.append(float((direct - replay).abs().max().item()))
                    feats.append(captured[-1].to(torch.float32).cpu().numpy())
                    labels.append(batch["runtime_ms"].cpu().numpy())
                    masks.append(base.horizon_mask(batch, ctx.horizon).cpu().numpy())
                    videos.append(np.asarray(arrays["video_code"])[indices])
            worst = max(equiv) if equiv else float("nan")
            if not (worst <= FEATURE_EQUIVALENCE_TOL):
                raise SystemExit(
                    "cache equivalence failed on %s: max|head(cached_z) - forward| = %g > %g"
                    % (split, worst, FEATURE_EQUIVALENCE_TOL)
                )
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
                "label_shape": list(payload["runtime_ms"].shape),
                "mask_positive_count": int(payload["mask"].astype(bool).sum()),
                "positive_runtime_count": int((payload["runtime_ms"] > 0).sum()),
                "max_abs_feature_equivalence": worst,
                "file": out.name,
                "file_sha256": sha256_file(out),
            }
            print("cached %-11s rows=%-6d max|head(cached_z)-forward|=%.2e -> %s"
                  % (split, n, worst, out.name))
    finally:
        handle.remove()
    summary["checkpoint_meta"] = {k: meta.get(k) for k in ("sha256", "variant", "seed") if k in meta}
    dist.write_json(ctx.run_root / "cache_summary.json", summary)
    return summary


# --------------------------------------------------------------------------- #
def load_cache(ctx: base.Ctx, split: str) -> Dict[str, np.ndarray]:
    path = ctx.run_root / ("cache_%s.npz" % split)
    summary_path = ctx.run_root / "cache_summary.json"
    if not path.is_file() or not summary_path.is_file():
        raise SystemExit("missing cache or cache summary; run --stage cache first")
    summary = dist.read_json(summary_path)
    if summary.get("checkpoint_sha256") != J3_SHA256:
        raise SystemExit("cache was built from a different checkpoint; rebuild with --stage cache")
    # fail closed on a stale config or dataset, not only on a stale cache file
    config_path = PROJECT_ROOT / "experiments" / str(ctx.config["experiment_id"]) / "config.json"
    if config_path.is_file() and sha256_file(config_path) != summary.get("config_sha256"):
        raise SystemExit("config changed since the cache was built; rebuild with --stage cache")
    for name, expected in (summary.get("dataset_file_sha256") or {}).items():
        candidate = Path(ctx.config["output_root"]) / ("j_%s.jsonl.gz" % name)
        if candidate.is_file() and sha256_file(candidate) != expected:
            raise SystemExit("dataset %s changed since the cache was built; rebuild with --stage cache" % name)
    recorded = summary["splits"][split]
    if sha256_file(path) != recorded["file_sha256"]:
        raise SystemExit("cache %s changed on disk since it was written; rebuild with --stage cache" % path.name)
    with np.load(path) as handle:
        return {key: handle[key] for key in handle.files}


def j3_quantiles_from_cache(ctx: base.Ctx, split: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    model, _ = base.load_checkpoint(J3_CHECKPOINT, ctx, "J3")
    model.eval()
    cache = load_cache(ctx, split)
    feats = torch.from_numpy(cache["features"]).to(ctx.device)
    with torch.no_grad():
        log_q = model.head_runtime(feats)
    raw = torch.expm1(log_q).clamp(min=0.0).cpu().numpy()
    return {"q50": raw[:, :, 0], "q90": raw[:, :, 1], "q95": raw[:, :, 2]}, cache


def stage_r0(ctx: base.Ctx, args: argparse.Namespace) -> Dict[str, Any]:
    spec = load_spec(ctx)
    preds, cache = j3_quantiles_from_cache(ctx, "validation")
    report: Dict[str, Any] = {"stage": "r0", "checkpoint": str(J3_CHECKPOINT), "checkpoint_sha256": checkpoint_guard()}
    report["validation"] = dist.evaluate_quantiles(
        preds["q50"], preds["q90"], preds["q95"], cache["runtime_ms"], cache["mask"], cache["video_code"]
    )
    dist.write_json(ctx.run_root / "j3_baseline_validation.json", report["validation"])

    a, b = report["validation"]["A_distribution_calibration"], report["validation"]["B_distribution_accuracy"]
    checks = {
        "coverage_q50": (a["q50"]["coverage"], J3_ACCEPTANCE["coverage"][0], FLOAT_TOL["coverage"]),
        "coverage_q90": (a["q90"]["coverage"], J3_ACCEPTANCE["coverage"][1], FLOAT_TOL["coverage"]),
        "coverage_q95": (a["q95"]["coverage"], J3_ACCEPTANCE["coverage"][2], FLOAT_TOL["coverage"]),
        "runtime_qscore_ms": (b["runtime_qscore_ms"], J3_ACCEPTANCE["runtime_qscore_ms"], FLOAT_TOL["runtime_qscore_ms"]),
    }
    report["reproduction"] = {
        name: {"computed": float(v), "recorded": float(t), "tolerance": float(tol), "pass": abs(v - t) <= tol}
        for name, (v, t, tol) in checks.items()
    }
    # P1-7: reconcile the crossing rate with the previously reported 6.55 %
    evidence = ctx.run_root / "crossing_evidence.json"
    report["crossing_evidence"] = {
        "cache_path_crossing_rate": a["quantile_crossing_rate"],
        "previously_reported_rate": 0.0655,
        "note": "if these disagree, compute the rate from the packer artifact before trusting either",
        "reconciliation": str(evidence) if evidence.is_file() else "not attempted; pass --packer-artifacts to --stage r0",
    }
    if args.packer_artifacts:
        report["crossing_evidence"].update(packer_crosscheck(Path(args.packer_artifacts), cache, preds))
    report["reproduction_pass"] = all(item["pass"] for item in report["reproduction"].values())
    print(json.dumps(report["reproduction"], indent=1))
    print("crossing: cache=%.5f  previously_reported=0.06550" % a["quantile_crossing_rate"])
    print("R0 %s" % ("PASSED" if report["reproduction_pass"] else "FAILED"))
    dist.write_json(ctx.run_root / "r0_report.json", report)
    return report


def packer_crosscheck(pack_dir: Path, cache: Mapping[str, np.ndarray], preds: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    """Compare the packer artifact with the cache replay row by row (P1-7 / Q4)."""

    import gzip

    path = pack_dir / "j_future_h5.jsonl.gz" if pack_dir.is_dir() else pack_dir
    if not path.is_file():
        return {"packer": str(path), "error": "artifact not found"}
    rows: Dict[str, Dict[str, float]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            steps = (payload.get("future_h5") or [{}])[0].get("steps") or []
            flat = {}
            for index, step in enumerate(steps):
                quantiles = (step.get("resource") or {}).get("runtime_ms_quantiles") or {}
                for name in ("p50", "p90", "p95"):
                    value = quantiles.get(name)
                    if isinstance(value, (int, float)):
                        flat["%d:%s" % (index, name)] = float(value)
            rows[str(payload.get("node_id"))] = flat
    crossing = 0
    compared = 0
    for flat in rows.values():
        for index in range(len(flat) // 3):
            p50 = flat.get("%d:p50" % index)
            p90 = flat.get("%d:p90" % index)
            p95 = flat.get("%d:p95" % index)
            if p50 is None or p90 is None or p95 is None:
                continue
            compared += 1
            if not (p50 <= p90 <= p95):
                crossing += 1
    return {
        "packer": str(path),
        "packer_rows": len(rows),
        "packer_crossing_pairs": compared,
        "packer_crossing_rate": crossing / compared if compared else None,
        "cache_crossing_rate": float(np.mean((preds["q50"] > preds["q90"]) | (preds["q90"] > preds["q95"]))),
        "cache_rows": int(cache["runtime_ms"].shape[0]),
    }


# --------------------------------------------------------------------------- #
def training_tensors(ctx: base.Ctx, spec: Mapping[str, Any], split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = dist.edges_from_spec(spec)
    cache = load_cache(ctx, split)
    mask = cache["mask"].astype(bool)
    runtime = cache["runtime_ms"].astype(np.float64)
    valid = mask & (runtime > 0.0)
    features = cache["features"][valid]
    labels = runtime[valid]
    return features, labels, dist.bin_index(labels, edges)


def resolve_hidden(args: argparse.Namespace, ctx: base.Ctx) -> Optional[int]:
    if args.hidden is not None:
        return int(args.hidden)
    value = ctx.config["resource_dist"].get("hidden")
    return int(value) if value else None


def train_head(ctx: base.Ctx, args: argparse.Namespace, variant_dir_name: str = "R1_dist") -> Dict[str, Any]:
    cfg = ctx.config["resource_dist"]
    weights = cfg["loss_weights"]
    spec = load_spec(ctx)
    edges = dist.edges_from_spec(spec)
    reps = dist.reps_from_spec(spec)
    band_of_bin = dist.band_of_bin_from_spec(spec)
    seed = (args.seed or [11])[0]
    epochs = args.epochs or int(cfg["epochs"])
    hidden = resolve_hidden(args, ctx)
    weight_decay = float(cfg.get("weight_decay", ctx.config["training"]["weight_decay"]))
    base.set_seed(seed)

    features, labels, target_idx = training_tensors(ctx, spec, "train")
    if args.max_rows:
        features, labels, target_idx = features[: args.max_rows], labels[: args.max_rows], target_idx[: args.max_rows]
    scale = float(np.median(labels))
    head = dist.DiscreteRuntimeHead(int(features.shape[1]), spec["n_bins"], hidden=hidden).to(ctx.device)
    optimiser = torch.optim.AdamW(head.parameters(), lr=float(cfg["learning_rate"]), weight_decay=weight_decay)
    feat_t = torch.from_numpy(features.astype(np.float32)).to(ctx.device)
    tgt_t = torch.from_numpy(target_idx.astype(np.int64)).to(ctx.device)
    band_t = torch.from_numpy(dist.band_index(labels).astype(np.int64)).to(ctx.device)
    ms_t = torch.from_numpy(labels.astype(np.float32)).to(ctx.device)
    ones = torch.ones_like(ms_t)

    baseline_path = ctx.run_root / "j3_baseline_validation.json"
    if not baseline_path.is_file():
        raise SystemExit("missing %s; run --stage r0 first (the epoch rule needs it)" % baseline_path)
    baseline = dist.read_json(baseline_path)
    calibration_limit = baseline["A_distribution_calibration"]["mean_abs_calibration_error"] + float(
        ctx.config["gates"]["calibration_not_worse_by"]
    )

    history: List[Dict[str, Any]] = []
    best: Optional[Tuple[Tuple[float, float], int, Dict[str, Any]]] = None
    for epoch in range(1, epochs + 1):
        head.train()
        order = torch.randperm(feat_t.shape[0], device=ctx.device)
        totals = {"nll": 0.0, "rps": 0.0, "point": 0.0, "band": 0.0, "rows": 0}
        for start in range(0, feat_t.shape[0], ctx.batch_size):
            idx = order[start : start + ctx.batch_size]
            logits = head(feat_t[idx])
            derived = dist.derive_from_probs(F.softmax(logits, dim=-1), reps)
            nll = dist.nll_loss(logits, tgt_t[idx], ones[idx])
            rps = dist.rps_loss(logits, tgt_t[idx], ones[idx])
            point = dist.pseudo_huber_point(derived["mean_ms"], ms_t[idx], scale, float(cfg["point_delta"]), ones[idx])
            band_ce = dist.coarse_band_ce(logits, band_t[idx], ones[idx], band_of_bin)
            loss = (
                float(weights["nll"]) * nll
                + float(weights["ranked_probability"]) * rps
                + float(weights["point_pseudohuber"]) * point
                + float(weights["coarse_band_ce"]) * band_ce
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            for name, term in (("nll", nll), ("rps", rps), ("point", point), ("band", band_ce)):
                totals[name] += float(term.detach()) * len(idx)
            totals["rows"] += len(idx)
        rows = max(1, totals["rows"])
        val = evaluate_head(ctx, head, spec, "validation")
        row = {
            "epoch": epoch,
            "loss_nll": totals["nll"] / rows,
            "loss_rps": totals["rps"] / rows,
            "loss_point": totals["point"] / rows,
            "loss_band": totals["band"] / rows,
            "val_calibration_error": val["A_distribution_calibration"]["mean_abs_calibration_error"],
            "val_mae_log": val["C_node_discrimination"]["mae_log"],
            "val_spearman": val["C_node_discrimination"]["spearman_point"],
            "calibration_eligible": val["A_distribution_calibration"]["mean_abs_calibration_error"] <= calibration_limit,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["calibration_eligible"]:
            key = (row["val_mae_log"], -row["val_spearman"])
            if best is None or key < best[0]:
                # state_dict() returns references to the live tensors, so the snapshot
                # MUST be cloned: otherwise later epochs mutate the "best" weights while
                # the metadata keeps claiming the earlier epoch.
                snapshot = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                best = (key, epoch, snapshot)

    variant_dir = ctx.run_root / variant_dir_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    if variant_dir_name != "R1_dist_smoke" and not args.force:
        existing = [
            name
            for name in ("seed%d_best.pt" % seed, "seed%d_last.pt" % seed, "seed%d_history.json" % seed)
            if (variant_dir / name).is_file()
        ]
        if existing:
            raise SystemExit("formal run would overwrite %s; move it away or pass --force" % ", ".join(existing))
    payload = {
        "spec": spec,
        "in_dim": int(features.shape[1]),
        "hidden": hidden,
        "weight_decay": weight_decay,
        "learning_rate": float(cfg["learning_rate"]),
        "epochs": epochs,
        "seed": seed,
        "checkpoint_sha256": checkpoint_guard(),
        "selection_rule": "calibration-eligible epochs -> argmin val log-MAE -> tie-break higher Spearman",
    }
    torch.save({**payload, "state_dict": head.state_dict(), "selected_epoch": epochs, "which": "last"},
               variant_dir / ("seed%d_last.pt" % seed))
    selected_epoch = epochs
    if best is not None:
        selected_epoch = best[1]
        torch.save({**payload, "state_dict": best[2], "selected_epoch": selected_epoch, "which": "best"},
                   variant_dir / ("seed%d_best.pt" % seed))
        print("best epoch = %d (val log-MAE %.4f, val Spearman %.4f)" % (selected_epoch, best[0][0], -best[0][1]))
    else:
        print("WARNING: no epoch satisfied the calibration guard; only the last checkpoint is trustworthy")
    dist.write_json(
        variant_dir / ("seed%d_history.json" % seed),
        {"seed": seed, "epochs": epochs, "history": history, "scale": scale,
         "calibration_limit": calibration_limit, "selected_epoch": selected_epoch},
    )
    return {"seed": seed, "epochs": epochs, "selected_epoch": selected_epoch, "history": history}


def evaluate_head(ctx: base.Ctx, head: Any, spec: Mapping[str, Any], split: str, max_rows: Optional[int] = None) -> Dict[str, Any]:
    edges = dist.edges_from_spec(spec)
    reps = dist.reps_from_spec(spec)
    cache = load_cache(ctx, split)
    features, runtime, mask = cache["features"], cache["runtime_ms"], cache["mask"].astype(bool)
    if max_rows:
        features, runtime, mask = features[:max_rows], runtime[:max_rows], mask[:max_rows]
    head.eval()
    flat = torch.from_numpy(features.reshape(-1, features.shape[-1]).astype(np.float32)).to(ctx.device)
    chunks = []
    with torch.no_grad():
        for start in range(0, flat.shape[0], 8192):
            chunks.append(F.softmax(head(flat[start : start + 8192]), dim=-1).cpu().numpy())
    probs = np.concatenate(chunks, axis=0).reshape(runtime.shape[0], runtime.shape[1], -1)
    keep = mask.reshape(-1) & (runtime.reshape(-1) > 0)
    video = np.repeat(cache["video_code"], runtime.shape[1])[keep]
    return dist.evaluate_distribution(probs.reshape(-1, probs.shape[-1])[keep], runtime.reshape(-1)[keep], reps, edges, video)


def load_trained_head(ctx: base.Ctx, variant: str, seed: int, which: str) -> Tuple[Any, Dict[str, Any]]:
    path = ctx.run_root / variant / ("seed%d_%s.pt" % (seed, which))
    if not path.is_file():
        raise SystemExit("missing trained head %s" % path)
    payload = torch.load(path, map_location=ctx.device, weights_only=False)
    if payload.get("checkpoint_sha256") != J3_SHA256:
        raise SystemExit("trained head was built against a different checkpoint")
    spec = load_spec(ctx)
    if payload["spec"]["edges"] != spec["edges"]:
        raise SystemExit("trained head used a different bin spec; rebuild")
    head = dist.DiscreteRuntimeHead(int(payload["in_dim"]), len(spec["edges"]) - 1, hidden=payload.get("hidden")).to(ctx.device)
    head.load_state_dict(payload["state_dict"])
    return head, payload


def stage_eval_val(ctx: base.Ctx, args: argparse.Namespace) -> Dict[str, Any]:
    spec = load_spec(ctx)
    variant = args.variant or "R1_dist"
    seed = (args.seed or [11])[0]
    head, payload = load_trained_head(ctx, variant, seed, args.which)
    baseline = dist.read_json(ctx.run_root / "j3_baseline_validation.json")
    report = evaluate_head(ctx, head, spec, "validation")
    gates = dist.evaluate_gates(report, baseline, ctx.config["gates"])
    out = {"variant": variant, "seed": seed, "which": args.which, "selected_epoch": payload.get("selected_epoch"),
           "validation": report, "gates": gates}
    dist.write_json(ctx.run_root / variant / ("seed%d_val_gates.json" % seed), out)
    print(json.dumps(gates["headline"], indent=1))
    print("viability:", "PASS" if gates["headline"]["viability_pass"] else "FAIL",
          "| strong:", "PASS" if gates["headline"]["strong_pass"] else "FAIL")
    return out


def stage_test_final(ctx: base.Ctx, args: argparse.Namespace) -> Dict[str, Any]:
    variant = args.variant or "R1_dist"
    seed = (args.seed or [11])[0]
    target = ctx.run_root / variant / ("seed%d_test.json" % seed)
    if target.is_file() and not args.force:
        raise SystemExit("test result already exists at %s; refusing to overwrite without --force" % target)
    spec = load_spec(ctx)
    head, payload = load_trained_head(ctx, variant, seed, args.which)
    report = evaluate_head(ctx, head, spec, "test")
    dist.write_json(target, {"variant": variant, "seed": seed, "selected_epoch": payload.get("selected_epoch"), "test": report})
    print(json.dumps(report["A_distribution_calibration"], indent=1))
    return {"test": report}


def stage_extend(ctx: base.Ctx, args: argparse.Namespace) -> Dict[str, Any]:
    raise SystemExit(
        "R2/R3 are intentionally not enabled yet.\n"
        "- R2 (future activity head) needs a dataset v2 carrying a canonical per-slot\n"
        "  ``slot_activity`` label; j_series_dataset_v1 does not have it, and adding it is a\n"
        "  data-side change that must not silently alter v1.\n"
        "- R3 (joint fine-tune) starts only after R2 passes its gates.\n"
        "See PHASE_R_PLAN.md."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("bins", "cache", "r0", "smoke", "train", "eval-val", "test-final", "extend"),
        required=True,
    )
    parser.add_argument("--variant", default=None)
    parser.add_argument("--which", choices=("best", "last"), default="best")
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--packer-artifacts", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
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
        args.epochs = args.epochs or 3
        args.max_rows = args.max_rows or 512
        train_head(ctx, args, variant_dir_name="R1_dist_smoke")
        print("smoke finished (written to %s)" % (ctx.run_root / "R1_dist_smoke"))
    elif args.stage == "train":
        train_head(ctx, args)
    elif args.stage == "eval-val":
        stage_eval_val(ctx, args)
    elif args.stage == "test-final":
        stage_test_final(ctx, args)
    else:
        stage_extend(ctx, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
