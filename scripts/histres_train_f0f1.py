"""Formal F0/F1 training: 3 seeds x 2 arms x 30 epochs, train + validation only.

Registers the three pre-registered contrasts and the routing guard.  J test is never
opened, and J3 is never modified: it is loaded read-only as the frozen baseline.

Run:
    python scripts/histres_train_f0f1.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import j_series_common as common  # noqa: E402
import j_series_train_eval as te  # noqa: E402

CONFIG = PROJECT_ROOT / "experiments/EXP-20260911_p9d_j_predictor_acceptance/config.json"
HISTRES = PROJECT_ROOT / "results/processed/j_series_dataset_histres_v2"
OUT_ROOT = PROJECT_ROOT / "outputs/j_series_histres_f0f1_v1"
EXP_DIR = PROJECT_ROOT / "experiments/EXP-20260921_histres_causal_input_v1"
ARMS = ("F0", "F1")
SEEDS = (11, 22, 33)
EPOCHS = 30


def build_ctx(config: dict):
    """Train + validation only; J test is never read."""

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_rows = list(common.read_jsonl_gz(HISTRES / "histres_train.jsonl.gz"))
    vocabs = common.build_vocabs(train_rows)
    horizon = int(config["horizon"])
    train = common.encode_rows(train_rows, vocabs, horizon)
    train["video_code"] = np.asarray([te.stable_video_code(r["video_id"]) for r in train_rows], dtype=np.int64)
    validation_rows = list(common.read_jsonl_gz(HISTRES / "histres_validation.jsonl.gz"))
    validation = common.encode_rows(validation_rows, vocabs, horizon)
    validation["video_code"] = np.asarray(
        [te.stable_video_code(r["video_id"]) for r in validation_rows], dtype=np.int64
    )
    replicates = int(config["bootstrap"]["B"])
    base_seed = int(config["bootstrap"]["seed"])
    return te.Ctx(
        config=config,
        run_root=OUT_ROOT,
        exp_dir=EXP_DIR,
        device=device,
        horizon=horizon,
        batch_size=int(config["training"]["batch_size"]),
        vocabs=vocabs,
        train=train,
        validation=validation,
        test={},
        val_draws=common.make_bootstrap_indices(validation["video_code"], replicates, base_seed),
        test_draws=np.zeros((1, 1), dtype=np.int64),
    )


def routing_check(ctx, seed: int) -> dict:
    """F0 must receive no telemetry gradient, F1 must receive some."""

    out = {}
    init_path, init_sha, payload = te.build_shared_init_state(ctx, seed)
    out["shared_init_sha256"] = init_sha
    for arm in ARMS:
        model = te.make_model(ctx, arm)
        model.load_state_dict(payload["model_state"])
        model.train()
        arrays = {k: v[:64] for k, v in ctx.train.items() if isinstance(v, np.ndarray)}
        batch = te.to_torch_batch(arrays, np.arange(min(32, len(arrays["length"])), dtype=np.int64), ctx.device)
        enabled = arm == "F1"
        encoded = model.encode(batch, use_history_telemetry=enabled)
        outputs = {
            "structure": model.structure(encoded),
            "attributes": model.attribute_logits(encoded),
            "behavior": model.behavior(encoded),
            "resource": model.resource(encoded, None),
        }
        terms = te.loss_terms(outputs, batch, ctx.horizon, model=model)
        model.zero_grad(set_to_none=True)
        terms["resource"].backward()
        report = te.telemetry_grad_report(model)
        te.assert_telemetry_routing(arm, report)
        out["%s_telemetry_grad" % arm] = report["telemetry_grad_norm"]
    return out


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config = json.loads(json.dumps(config))
    config["variants"].update({"F0": dict(te.DEFAULT_VARIANTS["F0"]), "F1": dict(te.DEFAULT_VARIANTS["F1"])})
    config["model"]["runtime_distribution_head"] = True
    config["model"]["runtime_head_hidden"] = 64
    config["training"]["max_epochs"] = EPOCHS
    config["training"]["seeds"] = list(SEEDS)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ctx = build_ctx(config)
    print("device:", ctx.device, "| train", len(ctx.train["length"]), "| validation", len(ctx.validation["length"]),
          "| j test loaded:", bool(ctx.test), flush=True)

    results: dict = {
        "experiment_id": "EXP-20260921_histres_causal_input_v1",
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "epochs": EPOCHS,
        "preregistered_contrasts": te.PREREGISTERED_CONTRASTS,
        "disclosure": te.PREREGISTERED_DISCLOSURE,
        "j_test_loaded": bool(ctx.test),
        "runs": {},
        "routing": {},
    }
    started = time.time()
    for seed in SEEDS:
        results["routing"]["seed%d" % seed] = routing_check(ctx, seed)
        print("routing seed %d: %s" % (seed, json.dumps(results["routing"]["seed%d" % seed])), flush=True)
        for arm in ARMS:
            t0 = time.time()
            record = te.run_training(ctx, arm, seed, EPOCHS)
            results["runs"]["%s_seed%d" % (arm, seed)] = {
                "wall_seconds": round(time.time() - t0, 1),
                "record": record,
            }
            print("[done] %s seed %d in %.1f s" % (arm, seed, time.time() - t0), flush=True)

    results["total_wall_seconds"] = round(time.time() - started, 1)
    (OUT_ROOT / "f0f1_report.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (EXP_DIR / "artifacts").mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "artifacts" / "f0f1_report.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print("total wall seconds:", results["total_wall_seconds"])
    print("saved:", OUT_ROOT / "f0f1_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
