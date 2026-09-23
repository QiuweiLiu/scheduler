#!/usr/bin/env python
"""Pack the F0 model's runtime predictions into the resource-v2 artifact format.

F0 is a ***whole retrained model***, not a head on frozen features, so the existing
resource-v2 packer cannot produce its artifact: that one hooks ``res_hidden`` on the
frozen J3 backbone and applies a separately trained head.  Here the full F0 forward
runs and its discretised runtime head supplies the distribution.

Arm conventions
---------------
``F0``  whole model retrained from frozen J3, telemetry branch DISABLED
``F1``  same but telemetry enabled (kept for completeness)

The telemetry flag is passed explicitly, so an F0 artifact cannot silently pick up the
F1 behaviour.

Usage::

    python scripts/pack_f0_artifacts.py --arm F0 --seed 11 \
        --output-root outputs/resource_v2_artifacts/f0_seed11
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import j_series_common as common  # noqa: E402
import j_series_resource_dist as dist  # noqa: E402
import j_series_train_eval as te  # noqa: E402

from pack_resource_v2_artifacts import (  # noqa: E402
    SCHEMA_VERSION,
    sha256_file,
    write_gzip_lines,
)

RUN_ROOT = PROJECT_ROOT / "outputs/j_series_histres_f0f1_v1/runs"
HISTRES = PROJECT_ROOT / "results/processed/j_series_dataset_histres_v2"
DEFAULT_ANCHORS = PROJECT_ROOT / "outputs/sstar_predictor_anchors/features_sstar.jsonl.gz"
DEFAULT_BASE_PACK = PROJECT_ROOT / "outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts"
DEFAULT_OUT = PROJECT_ROOT / "outputs/resource_v2_artifacts/f0_seed11"


def build_model(vocabs) -> common.JSeriesModel:
    cfg = {
        "hidden": 128,
        "history_embedding_dim": 16,
        "context_embedding_dim": 12,
        "slot_embedding_dim": 16,
        "dropout": 0.1,
        "runtime_bins": te.load_runtime_bins(),
        "runtime_head_hidden": 64,
    }
    return common.JSeriesModel(vocabs, cfg, horizon=5, duration_mode="shared")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=("F0", "F1"), default="F0")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--anchors-file", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--base-pack", type=Path, default=DEFAULT_BASE_PACK)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    checkpoint = RUN_ROOT / args.arm / ("seed%d" % args.seed) / "checkpoint.pt"
    if not checkpoint.is_file():
        raise SystemExit("checkpoint missing: %s" % checkpoint)
    target = args.output_root / "b05_future_h5.jsonl.gz"
    if target.is_file() and not args.force:
        raise SystemExit("%s exists; pass --force" % target)

    use_telemetry = args.arm == "F1"
    bins = te.load_runtime_bins()
    reps = np.asarray(bins["representatives_ms"], dtype=np.float64)

    # vocabularies must come from the rows the model was trained on
    vocabs = common.build_vocabs(list(common.read_jsonl_gz(HISTRES / "histres_train.jsonl.gz")))
    model = build_model(vocabs)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    model.eval()

    rows = list(common.read_jsonl_gz(args.anchors_file))
    rows = [{**r, "future": [], "bounded_future_length": 0, "termination": 0} for r in rows]
    print("anchors: %d | arm %s seed %d | telemetry=%s" % (len(rows), args.arm, args.seed, use_telemetry))

    arrays = common.encode_rows(rows, vocabs, 5)
    probs_out: List[np.ndarray] = []
    device = torch.device("cpu")
    with torch.no_grad():
        for start in range(0, len(rows), 256):
            idx = np.arange(start, min(start + 256, len(rows)), dtype=np.int64)
            batch = te.to_torch_batch(arrays, idx, device)
            encoded = model.encode(batch, use_history_telemetry=use_telemetry)
            logits = model.resource(encoded, None)["runtime_logits"]
            probs_out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    probs = np.concatenate(probs_out, axis=0)
    if not np.isfinite(probs).all():
        raise SystemExit("non-finite probabilities from the F0 head")
    print("probs shape:", probs.shape)

    base_pack = {}
    with __import__("gzip").open(args.base_pack / "b05_future_h5.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                base_pack[str(record["node_id"])] = record

    missing = [str(r["current_node_id"]) for r in rows if str(r["current_node_id"]) not in base_pack]
    if missing:
        raise SystemExit("anchors absent from the base pack: %d (e.g. %s)" % (len(missing), missing[:3]))

    records: List[Dict[str, Any]] = []
    upgraded = 0
    for i, row in enumerate(rows):
        record = base_pack[str(row["current_node_id"])]
        scenarios = record.get("future_h5") or []
        if len(scenarios) != 1:
            raise SystemExit("node %s carries %d scenarios" % (row["current_node_id"], len(scenarios)))
        steps = scenarios[0].get("steps") or []
        if len(steps) > probs.shape[1]:
            raise SystemExit("node %s has %d steps but only %d slots" % (row["current_node_id"], len(steps), probs.shape[1]))
        for t, step in enumerate(steps):
            resource = step.setdefault("resource", {})
            # float64: derive_from_probs casts the representatives to the probability
            # dtype, and the overlay loader re-derives the views in float64, so a
            # float32 round trip shows up as a ~1e-3 disagreement on a 20000 ms scale
            views = dist.derive_from_probs(
                torch.as_tensor(probs[i, t], dtype=torch.float64)[None, :], reps
            )
            resource["runtime_ms_quantiles"] = {
                "p50": float(np.asarray(views["q50_ms"]).ravel()[0]),
                "p90": float(np.asarray(views["q90_ms"]).ravel()[0]),
                "p95": float(np.asarray(views["q95_ms"]).ravel()[0]),
            }
            resource["runtime_mean_ms"] = float(np.asarray(views["mean_ms"]).ravel()[0])
            # the overlay loader re-derives every canonical view, so all of them must be
            # written per step; the discrete upper-tail CVaR uses the same bin
            # representatives as the quantile inversion
            probs_row = probs[i, t].astype(np.float64)
            cdf = np.cumsum(probs_row)
            alpha = 0.95
            acc = 0.0
            prev = 0.0
            for k in range(len(probs_row)):
                hi = float(cdf[k])
                if hi > alpha:
                    acc += (hi - max(prev, alpha)) * float(reps[k])
                prev = hi
            resource["cvar95_ms"] = acc / max(1e-9, 1.0 - alpha)
            resource["runtime_probs"] = [float(x) for x in probs[i, t]]
            resource["bin_schema_id"] = bins["bin_schema_id"] if "bin_schema_id" in bins else "%s_%d" % (bins["mode"], bins["n_bins"])
            resource["resource_head_id"] = "%s_seed%d_f16" % (args.arm, args.seed)
            upgraded += 1
        records.append(record)

    if upgraded == 0:
        raise SystemExit("no step was upgraded")
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_gzip_lines(target, records)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": "resource_v2_%s_seed%d" % (args.arm.lower(), args.seed),
        "arm": args.arm,
        "seed": args.seed,
        "artifact_sha256": sha256_file(target),
        "producer": "scripts/pack_f0_artifacts.py",
        "producer_checkpoint_sha256": sha256_file(checkpoint),
        "producer_checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "frozen_j3_sha256": te.J3_FROZEN_SHA256,
        "resource_head_id": "%s_seed%d_f16" % (args.arm, args.seed),
        "base_pack_sha256": sha256_file(args.base_pack / "b05_future_h5.jsonl.gz"),
        "bin_spec_sha256": sha256_file(te.RUNTIME_BIN_SPEC),
        "bin_schema_id": "%s_%d" % (bins["mode"], bins["n_bins"]),
        "bin_edges_ms": list(bins["edges"]),
        "bin_representatives_ms": list(bins["representatives_ms"]),
        "runtime_semantics": "conditional_on_active (per-step marginals; no joint over the horizon)",
        "quantile_rule": "cdf_inversion_bin_representative",
        "cvar_alpha": 0.95,
        "use_history_telemetry": use_telemetry,
        "horizon": 5,
        "nodes": len(records),
        "steps": upgraded,
        "canonical_views": ["p50", "p90", "p95", "runtime_mean_ms", "cvar95_ms"],
    }
    (args.output_root / "resource_v2_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
