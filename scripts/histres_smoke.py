"""Two-epoch implementation smoke for the F0/F1 telemetry arms.

This is deliberately not a result: it checks that the pipeline is complete and
internally consistent, exactly the list the review asked for.  Only train and
validation are read; J test is never opened.

Run:
    python scripts/histres_smoke.py
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
OUT_ROOT = PROJECT_ROOT / "outputs/j_series_histres_smoke_v1"
SEED = 11
EPOCHS = 2
ARM = "F0"


def build_ctx(config: dict):
    """Ctx over the augmented dataset, train + validation only."""

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
    ctx = te.Ctx(
        config=config,
        run_root=OUT_ROOT,
        exp_dir=PROJECT_ROOT / "experiments/EXP-20260921_histres_causal_input_v1",
        device=device,
        horizon=horizon,
        batch_size=int(config["training"]["batch_size"]),
        vocabs=vocabs,
        train=train,
        validation=validation,
        test={},                      # J test is deliberately not loaded
        val_draws=common.make_bootstrap_indices(validation["video_code"], replicates, base_seed),
        test_draws=np.zeros((1, 1), dtype=np.int64),
    )
    return ctx, train_rows, validation_rows


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config = json.loads(json.dumps(config))
    config["variants"].update({
        "F0": dict(te.DEFAULT_VARIANTS["F0"]),
        "F1": dict(te.DEFAULT_VARIANTS["F1"]),
    })
    config["model"]["runtime_distribution_head"] = True
    config["model"]["runtime_head_hidden"] = 64
    config["training"]["max_epochs"] = EPOCHS
    config["training"]["seeds"] = [SEED]

    print("device:", "cuda" if torch.cuda.is_available() else "cpu")
    ctx, train_rows, validation_rows = build_ctx(config)
    print("train rows %d | validation rows %d | j test loaded: %s"
          % (len(train_rows), len(validation_rows), bool(ctx.test)))
    print("runtime head:", "DiscreteRuntimeHead" if config["model"].get("runtime_distribution_head") else "Linear")

    report: dict = {"arms": {}}
    for arm in ("F0", "F1"):
        started = time.time()
        result = te.run_training(ctx, arm, SEED, EPOCHS)
        elapsed = time.time() - started
        report["arms"][arm] = {
            "elapsed_s": round(elapsed, 1),
            "epochs": result.get("epochs") if isinstance(result, dict) else None,
        }
        print("[done] %s in %.1f s" % (arm, elapsed))

    # ---- the review's implementation-integrity list ---------------------- #
    checks: dict = {}
    init_path = OUT_ROOT / "init" / ("F0F1_seed%d.pt" % SEED)
    checks["shared_init_state_exists"] = init_path.is_file()
    if init_path.is_file():
        checks["shared_init_state_sha256"] = common.sha256_file(init_path)

    model = te.make_model(ctx, "F1")
    payload = torch.load(init_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    arrays = {k: v[:64] for k, v in ctx.train.items() if isinstance(v, np.ndarray)}
    batch = te.to_torch_batch(arrays, np.arange(min(32, len(arrays["length"])), dtype=np.int64), ctx.device)

    model.train()
    mask = te.horizon_mask(batch, ctx.horizon)
    outputs = {
        "structure": model.structure(model.encode(batch)),
        "attributes": model.attribute_logits(model.encode(batch)),
        "behavior": model.behavior(model.encode(batch)),
        "resource": model.resource(model.encode(batch), None),
    }
    checks["runtime_logits_shape"] = list(outputs["resource"]["runtime_logits"].shape)
    terms = te.loss_terms(outputs, batch, ctx.horizon, model=model)
    checks["resource_loss_finite"] = bool(torch.isfinite(terms["resource"]))
    terms["resource"].backward()
    checks["f1_telemetry_grad"] = te.telemetry_grad_report(model)["telemetry_grad_norm"]

    model.zero_grad(set_to_none=True)
    model.encode(batch, use_history_telemetry=False)
    outputs_f0 = model.resource(model.encode(batch, use_history_telemetry=False), None)
    te.loss_terms(
        {"structure": model.structure(model.encode(batch, use_history_telemetry=False)),
         "attributes": model.attribute_logits(model.encode(batch, use_history_telemetry=False)),
         "behavior": model.behavior(model.encode(batch, use_history_telemetry=False)),
         "resource": outputs_f0}, batch, ctx.horizon, model=model,
    )["resource"].backward()
    checks["f0_telemetry_grad"] = te.telemetry_grad_report(model)["telemetry_grad_norm"]

    report["checks"] = checks
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(json.dumps(checks, ensure_ascii=False, indent=2))

    failures = []
    if checks.get("f0_telemetry_grad", 1.0) != 0.0:
        failures.append("F0 received a telemetry gradient")
    if checks.get("f1_telemetry_grad", 0.0) <= 0.0:
        failures.append("F1 received no telemetry gradient")
    if not checks.get("resource_loss_finite"):
        failures.append("resource loss was not finite")
    print()
    print("SMOKE:", "PASS" if not failures else "FAIL " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
