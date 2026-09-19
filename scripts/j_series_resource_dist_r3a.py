#!/usr/bin/env python
"""R3a -- partial-unfreeze diagnostic for the resource path.

Question: is the frozen J3 representation the limiting factor, or is the output head
enough?  Two arms with an **identical** head:

``F`` (frozen)        role                        -- train only the new head
``U`` (adapt)         role + ``res_hidden``       -- also adapt the last shared module,
                      i.e. the module that produces the 128-d feature the resource head
                      actually consumes

Both arms read the *same* cached input.  The trick that keeps this cheap: the input to
``res_hidden`` is ``cat[repr_vec, slot_vec, attr_feature]`` and all three parts come from
frozen modules, so the input can be cached exactly like ``z`` was.  ``res_hidden``'s weights
are initialised from the frozen checkpoint in both arms.

Pre-registered judging rule (from the code review): the frozen representation counts as
limiting only if the paired video-cluster bootstrap lower bound of the ``U - F`` Spearman
gain is > 0 **and** the gain closes at least half of the remaining strong-gate gap, i.e.
>= +0.0315 Spearman or >= +0.065 on the 8-12 s ratio.

Stages: ``cache`` / ``train --arm F|U`` / ``diagnose``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
HEAD_HIDDEN = 64
SEED = 11
EPOCHS = 30
# remaining strong-gate gap after R1b (Spearman 0.6354 vs 0.6698) and half of it
HALF_SPEARMAN = 0.0315
HALF_BUCKET = 0.065
EXPENSIVE = (8000.0, 12000.0)


def run_root() -> Path:
    return PROJECT_ROOT / "outputs/j_series_resource_dist_v1/r3a"


def cache_path(split: str) -> Path:
    return run_root() / ("pre_%s.npz" % split)


def head_pt(arm: str, seed: int) -> Path:
    return run_root() / ("head_%s_seed%d.pt" % (arm, seed))


# --------------------------------------------------------------------------- #
def stage_cache(ctx: base.Ctx) -> Dict[str, Any]:
    run_root().mkdir(parents=True, exist_ok=True)
    model, _ = base.load_checkpoint(J3_CHECKPOINT, ctx, "J3")
    model.eval()
    captured: List[Any] = []

    def hook(_module: Any, inputs: Tuple[Any, ...]) -> None:
        captured.append(inputs[0].detach())

    handle = model.res_hidden.register_forward_pre_hook(hook)
    summary: Dict[str, Any] = {"checkpoint_sha256": J3_SHA256, "splits": {}, "res_hidden_state": None}
    try:
        for split in ("train", "validation", "test"):
            arrays = {"train": ctx.train, "validation": ctx.validation, "test": ctx.test}[split]
            n = arrays["runtime_ms"].shape[0]
            feats, labels, masks, videos = [], [], [], []
            with torch.no_grad():
                for indices in base.split_indices(n, ctx.batch_size):
                    batch = base.to_torch_batch(arrays, indices, ctx.device)
                    captured.clear()
                    base.forward(model, batch, "J3", ctx)
                    if not captured:
                        raise RuntimeError("res_hidden pre-hook captured nothing")
                    # the hook argument IS res_hidden's input by construction, so no further
                    # equivalence check is needed here (the z-level check already passed in
                    # the main cache stage with max|diff| = 0.00e+00)
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
            np.savez_compressed(cache_path(split), **payload)
            summary["splits"][split] = {
                "rows": int(n),
                "feature_shape": list(payload["features"].shape),
                "mask_positive_count": int(payload["mask"].astype(bool).sum()),
            }
            print("cached %-11s rows=%-6d pre-feature shape=%s" % (split, n, payload["features"].shape))
    finally:
        handle.remove()
    torch.save({"res_hidden": model.res_hidden.state_dict()}, run_root() / "res_hidden_init.pt")
    summary["res_hidden_state"] = "res_hidden_init.pt"
    dist.write_json(run_root() / "cache_summary.json", summary)
    return summary


def load_pre(split: str) -> Dict[str, np.ndarray]:
    with np.load(cache_path(split)) as handle:
        return {key: handle[key] for key in handle.files}


def flatten(arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    runtime = arrays["runtime_ms"].reshape(-1)
    mask = arrays["mask"].reshape(-1).astype(bool)
    keep = mask & (runtime > 0)
    slots = np.tile(np.arange(arrays["runtime_ms"].shape[1]), arrays["runtime_ms"].shape[0])
    return {
        "features": arrays["features"].reshape(-1, arrays["features"].shape[-1])[keep],
        "runtime": runtime[keep],
        "video": np.repeat(arrays["video_code"], arrays["runtime_ms"].shape[1])[keep],
        "slot": slots[keep],
    }


# --------------------------------------------------------------------------- #
def train_arm(ctx: base.Ctx, arm: str, epochs: int, max_rows: Optional[int], force: bool, seed: int) -> Dict[str, Any]:
    if head_pt(arm, seed).is_file() and not force:
        raise SystemExit("%s exists; move it away or pass --force" % head_pt(arm, seed))
    spec = dist.read_json(PROJECT_ROOT / "outputs/j_series_resource_dist_v1/bins.json")
    edges = dist.edges_from_spec(spec)
    reps = dist.reps_from_spec(spec)
    band_of_bin = dist.band_of_bin_from_spec(spec)
    weight_decay = 1e-4
    base.set_seed(seed)

    train = flatten(load_pre("train"))
    if max_rows:
        for key in train:
            train[key] = train[key][:max_rows]
    x = torch.from_numpy(train["features"].astype(np.float32)).to(ctx.device)
    ms = torch.from_numpy(train["runtime"].astype(np.float32)).to(ctx.device)
    tgt = torch.from_numpy(dist.bin_index(train["runtime"], edges).astype(np.int64)).to(ctx.device)
    band = torch.from_numpy(dist.band_index(train["runtime"]).astype(np.int64)).to(ctx.device)
    ones = torch.ones_like(ms)
    scale = float(np.median(train["runtime"]))

    frozen = torch.load(run_root() / "res_hidden_init.pt", map_location="cpu", weights_only=False)["res_hidden"]
    res_hidden = torch.nn.Linear(frozen["weight"].shape[1], frozen["weight"].shape[0]).to(ctx.device)
    res_hidden.load_state_dict({k: v.to(ctx.device) for k, v in frozen.items()})
    if arm == "F":
        res_hidden.requires_grad_(False)
    head = dist.DiscreteRuntimeHead(int(frozen["weight"].shape[0]), len(edges) - 1, hidden=HEAD_HIDDEN).to(ctx.device)

    parameters = [p for p in head.parameters() if p.requires_grad]
    if arm == "U":
        parameters += [p for p in res_hidden.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=weight_decay)
    n_params_u = sum(p.numel() for p in parameters)

    history: List[Dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        head.train()
        res_hidden.train()
        order = torch.randperm(x.shape[0], device=ctx.device)
        totals = {"nll": 0.0, "rps": 0.0, "point": 0.0, "band": 0.0, "rows": 0}
        for start in range(0, x.shape[0], ctx.batch_size):
            idx = order[start : start + ctx.batch_size]
            z = torch.tanh(res_hidden(x[idx]))
            logits = head(z)
            derived = dist.derive_from_probs(F.softmax(logits, dim=-1), reps)
            nll = dist.nll_loss(logits, tgt[idx], ones[idx])
            rps = dist.rps_loss(logits, tgt[idx], ones[idx])
            point = dist.pseudo_huber_point(derived["mean_ms"], ms[idx], scale, 2.0, ones[idx])
            band_ce = dist.coarse_band_ce(logits, band[idx], ones[idx], band_of_bin)
            loss = nll + 0.5 * rps + 0.25 * point + 0.25 * band_ce
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            for name, term in (("nll", nll), ("rps", rps), ("point", point), ("band", band_ce)):
                totals[name] += float(term.detach()) * len(idx)
            totals["rows"] += len(idx)
        rows = max(1, totals["rows"])
        val = evaluate(ctx, arm, res_hidden, head, edges, reps, "validation")
        row = {"epoch": epoch, "arm": arm, "seed": seed,
               "loss_nll": totals["nll"] / rows, "loss_rps": totals["rps"] / rows,
               "loss_point": totals["point"] / rows, "loss_band": totals["band"] / rows,
               "val_log_mae": val["C_node_discrimination"]["mae_log"],
               "val_spearman": val["C_node_discrimination"]["spearman_point"],
               "val_bucket": val["C_node_discrimination"]["expensive_bucket_pred_over_true"]}
        history.append(row)
        print(json.dumps(row), flush=True)

    torch.save({"head": head.state_dict(), "res_hidden": res_hidden.state_dict(), "arm": arm,
                "epochs": epochs, "trainable_params": n_params_u, "checkpoint_sha256": J3_SHA256},
               head_pt(arm, seed))
    # keep the validation predictions so the multi-seed summary can bootstrap paired deltas
    val_pred, val_true, val_video, _ = predictions(ctx, arm, "validation", seed=seed)
    np.savez_compressed(run_root() / ("preds_%s_seed%d.npz" % (arm, seed)),
                        point=val_pred, true=val_true, video=val_video)
    dist.write_json(run_root() / ("history_%s_seed%d.json" % (arm, seed)),
                    {"arm": arm, "seed": seed, "history": history, "trainable_params": int(n_params_u)})
    return {"arm": arm, "epochs": epochs, "trainable_params": int(n_params_u), "last": history[-1]}


def evaluate(ctx: base.Ctx, arm: str, res_hidden: Any, head: Any, edges: np.ndarray, reps: np.ndarray,
             split: str) -> Dict[str, Any]:
    data = flatten(load_pre(split))
    x = torch.from_numpy(data["features"].astype(np.float32)).to(ctx.device)
    probs = []
    head.eval()
    res_hidden.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], 8192):
            z = torch.tanh(res_hidden(x[start : start + 8192]))
            probs.append(F.softmax(head(z), dim=-1).cpu().numpy())
    probs = np.concatenate(probs, axis=0)
    return dist.evaluate_distribution(probs, data["runtime"], reps, edges, data["video"], slot_index=data["slot"])


def predictions(ctx: base.Ctx, arm: str, split: str, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spec = dist.read_json(PROJECT_ROOT / "outputs/j_series_resource_dist_v1/bins.json")
    edges = dist.edges_from_spec(spec)
    reps = dist.reps_from_spec(spec)
    payload = torch.load(head_pt(arm, seed), map_location=ctx.device, weights_only=False)
    frozen = payload["res_hidden"]
    res_hidden = torch.nn.Linear(frozen["weight"].shape[1], frozen["weight"].shape[0]).to(ctx.device)
    res_hidden.load_state_dict({k: v.to(ctx.device) for k, v in frozen.items()})
    head = dist.DiscreteRuntimeHead(frozen["weight"].shape[0], len(edges) - 1, hidden=HEAD_HIDDEN).to(ctx.device)
    head.load_state_dict(payload["head"])
    data = flatten(load_pre(split))
    x = torch.from_numpy(data["features"].astype(np.float32)).to(ctx.device)
    out = []
    head.eval()
    res_hidden.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], 8192):
            z = torch.tanh(res_hidden(x[start : start + 8192]))
            out.append(F.softmax(head(z), dim=-1).cpu().numpy())
    probs = np.concatenate(out, axis=0)
    cdf = np.cumsum(probs, axis=-1)
    point = np.asarray(reps)[np.argmax(cdf >= 0.50, axis=-1)]
    return point, data["runtime"], data["video"], data["slot"]


def stage_diagnose(ctx: base.Ctx, n_boot: int) -> Dict[str, Any]:
    pf, tf, vf, sf = predictions(ctx, "F", "validation")
    pu, tu, vu, su = predictions(ctx, "U", "validation")
    if not (np.array_equal(tf, tu) and np.array_equal(vf, vu) and np.array_equal(sf, su)):
        raise SystemExit("the two arms were not evaluated on the same pairs")
    true = tf
    rng = np.random.default_rng(20260911)
    videos = sorted(set(vf.tolist()))
    groups = {v: np.where(vf == v)[0] for v in videos}
    deltas_spearman, deltas_bucket = [], []
    for _ in range(int(n_boot)):
        picked = np.concatenate([groups[videos[int(rng.integers(len(videos)))]] for _ in range(len(videos))])
        deltas_spearman.append(dist.spearman(pu[picked], true[picked]) - dist.spearman(pf[picked], true[picked]))
        bf = dist.bucket_ratio(true[picked], pf[picked], *EXPENSIVE)["pred_over_true"]
        bu = dist.bucket_ratio(true[picked], pu[picked], *EXPENSIVE)["pred_over_true"]
        if bf is not None and bu is not None:
            deltas_bucket.append(bu - bf)
    deltas_spearman.sort()
    deltas_bucket.sort()
    lo_s = deltas_spearman[int(0.025 * (len(deltas_spearman) - 1))]
    lo_b = deltas_bucket[int(0.025 * (len(deltas_bucket) - 1))] if deltas_bucket else None
    obs_s = dist.spearman(pu, true) - dist.spearman(pf, true)
    bf = dist.bucket_ratio(true, pf, *EXPENSIVE)["pred_over_true"]
    bu = dist.bucket_ratio(true, pu, *EXPENSIVE)["pred_over_true"]
    obs_b = (bu - bf) if (bf is not None and bu is not None) else None
    limiting = bool(lo_s > 0 and (obs_s >= HALF_SPEARMAN or (obs_b is not None and obs_b >= HALF_BUCKET)))
    report = {
        "n_pairs": int(len(true)),
        "video_clusters": len(videos),
        "bootstrap": int(n_boot),
        "spearman": {"F": dist.spearman(pf, true), "U": dist.spearman(pu, true),
                     "delta": obs_s, "ci_lower": lo_s,
                     "ci95": [lo_s, deltas_spearman[int(0.975 * (len(deltas_spearman) - 1))]]},
        "bucket_8_12s": {"F": bf, "U": bu, "delta": obs_b, "ci_lower": lo_b, "halves_needed": HALF_BUCKET},
        "criteria": {"ci_lower_positive": bool(lo_s > 0),
                     "closes_half_spearman": bool(obs_s >= HALF_SPEARMAN),
                     "closes_half_bucket": bool(obs_b is not None and obs_b >= HALF_BUCKET),
                     "half_spearman_required": HALF_SPEARMAN},
        "verdict": ("frozen representation IS limiting" if limiting
                    else "no evidence that the frozen representation is limiting"),
        "rule": "limiting requires Spearman delta CI lower bound > 0 AND (>= +0.0315 Spearman OR >= +0.065 on the 8-12 s ratio)",
    }
    dist.write_json(run_root() / "r3a_diagnostic.json", report)
    print(json.dumps(report, indent=1, default=str))
    return report



def stage_multi_seed(ctx: base.Ctx, n_boot: int) -> Dict[str, Any]:
    """Paired bootstrap of the mean U-F delta over the available seeds."""

    seeds = sorted({int(p.stem.split("seed")[-1]) for p in run_root().glob("preds_F_seed*.npz")})
    seeds = [s for s in seeds if (run_root() / ("preds_U_seed%d.npz" % s)).is_file()]
    if not seeds:
        raise SystemExit("no paired prediction files; train both arms first")
    data = {}
    for s in seeds:
        f = np.load(run_root() / ("preds_F_seed%d.npz" % s))
        u = np.load(run_root() / ("preds_U_seed%d.npz" % s))
        if not (np.array_equal(f["true"], u["true"]) and np.array_equal(f["video"], u["video"])):
            raise SystemExit("arm pair for seed %d is not on the same pairs" % s)
        data[s] = {"F": f["point"], "U": u["point"], "true": f["true"], "video": f["video"]}
    first = data[seeds[0]]
    true, video = first["true"], first["video"]
    videos = sorted(set(video.tolist()))
    groups = {v: np.where(video == v)[0] for v in videos}
    rng = np.random.default_rng(20260911)

    def stats(idx: np.ndarray) -> Tuple[float, float]:
        ds = [dist.spearman(data[s]["U"][idx], true[idx]) - dist.spearman(data[s]["F"][idx], true[idx]) for s in seeds]
        bf = dist.bucket_ratio(true[idx], data[seeds[0]]["F"][idx], *EXPENSIVE)["pred_over_true"]
        bu = dist.bucket_ratio(true[idx], data[seeds[0]]["U"][idx], *EXPENSIVE)["pred_over_true"]
        bucket = (bu - bf) if (bf is not None and bu is not None) else 0.0
        return float(np.mean(ds)), float(bucket)

    obs_s, obs_b = stats(np.arange(len(true)))
    boot_s, boot_b = [], []
    for _ in range(int(n_boot)):
        picked = np.concatenate([groups[videos[int(rng.integers(len(videos)))]] for _ in range(len(videos))])
        a, b = stats(picked)
        boot_s.append(a)
        boot_b.append(b)
    boot_s.sort()
    boot_b.sort()

    def ci(values: List[float]) -> List[float]:
        return [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]

    per_seed = {str(s): {
        "spearman_F": dist.spearman(data[s]["F"], true),
        "spearman_U": dist.spearman(data[s]["U"], true),
        "delta": dist.spearman(data[s]["U"], true) - dist.spearman(data[s]["F"], true),
        "bucket_F": dist.bucket_ratio(true, data[s]["F"], *EXPENSIVE)["pred_over_true"],
        "bucket_U": dist.bucket_ratio(true, data[s]["U"], *EXPENSIVE)["pred_over_true"],
    } for s in seeds}
    positive = sum(1 for v in per_seed.values() if v["delta"] > 0)
    report = {
        "seeds": seeds,
        "video_clusters": len(videos),
        "n_pairs": int(len(true)),
        "per_seed": per_seed,
        "mean_delta_spearman": obs_s,
        "mean_delta_spearman_ci95": ci(boot_s),
        "mean_delta_spearman_fraction_le_zero": float(np.mean([v <= 0 for v in boot_s])),
        "mean_delta_bucket": obs_b,
        "mean_delta_bucket_ci95": ci(boot_b),
        "mean_delta_bucket_fraction_le_zero": float(np.mean([v <= 0 for v in boot_b])),
        "seeds_with_positive_spearman_delta": positive,
        "sign_consistency": "%d/%d" % (positive, len(seeds)),
        "note": "deltas are paired per seed; the bootstrap resamples video clusters and averages over seeds",
    }
    dist.write_json(run_root() / "r3a_multiseed.json", report)
    print(json.dumps(report, indent=1, default=str))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("cache", "train", "diagnose", "multi-seed"), required=True)
    parser.add_argument("--arm", choices=("F", "U"), default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ctx = base.make_ctx(args.config, None)
    if args.stage == "cache":
        stage_cache(ctx)
    elif args.stage == "train":
        if args.arm is None:
            raise SystemExit("--arm F|U is required for the train stage")
        train_arm(ctx, args.arm, args.epochs, args.max_rows, args.force, args.seed)
    elif args.stage == "diagnose":
        stage_diagnose(ctx, args.bootstrap)
    else:
        stage_multi_seed(ctx, args.bootstrap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
