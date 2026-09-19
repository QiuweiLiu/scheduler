#!/usr/bin/env python
"""Repack the scheduler-facing future artifacts with a resource-v2 head.

Two-layer resource schema (per the interface review):

* **source of truth** - ``runtime_probs[H, K]`` plus the bin metadata
  (``bin_schema_id`` / ``bin_edges_ms`` / ``bin_representatives_ms``), so later
  consumers can derive CVaR, deadline-exceedance, scenarios, ... without re-running
  the predictor;
* **canonical views** - ``q50/q90/q95`` and ``runtime_mean_ms`` frozen into the pack,
  plus ``cvar95_ms`` once its definition is fixed, so two scheduler versions cannot
  derive different q95 from the same artifact.

Everything outside the ``resource`` block is copied verbatim from a base pack produced
by the frozen packer, so two arms differ only in the resource block.

Arms
----
``j3``     the frozen three-quantile head (identity check against the base pack)
``r1b``    the trained discrete head on frozen features (mass-balanced 16 bins)
``r3a_u``  the same head plus the adapted ``res_hidden`` (R3a arm U)
``r3a_f``  the same head with ``res_hidden`` frozen (R3a arm F, control)

Usage::

    python scripts/pack_resource_v2_artifacts.py \
        --anchors-file outputs/sstar_predictor_anchors/features_sstar.jsonl.gz \
        --base-pack outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts \
        --arm r1b --output-root outputs/resource_v2_artifacts/r1b
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402

import j_series_common as common  # noqa: E402
import j_series_resource_dist as dist  # noqa: E402
import j_series_train_eval as base  # noqa: E402

J3_CHECKPOINT = PROJECT_ROOT / "experiments/EXP-20260911_p9d_j_predictor_acceptance/artifacts/predictor/J3_seed11.pt"
J3_SHA256 = "0ee8ded4f92553853026ee24a3c320f21d524f9d2c60de841091430d14949c77"
BIN_SPEC = PROJECT_ROOT / "outputs/j_series_resource_dist_v1/bins.json"
RUN_ROOT = PROJECT_ROOT / "outputs/j_series_resource_dist_v1"
HEAD_FILES = {
    "r1b": RUN_ROOT / "R1b_mlp/seed11_best.pt",
    "r3a_u": RUN_ROOT / "r3a/head_U_seed11.pt",
    "r3a_f": RUN_ROOT / "r3a/head_F_seed11.pt",
}
HEAD_HIDDEN = 64
ALPHA_CVAR = 0.95
CVAR_RULE = "discrete_upper_tail_partial_weight: CVaR_a = (1/(1-a)) * sum_k max(0, cdf_k - max(cdf_{k-1}, a)) * m_k"


# --------------------------------------------------------------------------- #
def read_anchors(path: Path) -> List[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_pack(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                out[str(row["node_id"])] = row
    return out


def cvar_from_probs(probs: np.ndarray, reps: np.ndarray, alpha: float = ALPHA_CVAR) -> float:
    """Discrete CVaR of the upper tail, with partial weight on the crossing bin."""

    cdf = np.cumsum(probs)
    acc = 0.0
    prev = 0.0
    for k in range(len(probs)):
        hi = float(cdf[k])
        if hi > alpha:
            weight = hi - max(prev, alpha)
            acc += weight * float(reps[k])
        prev = hi
    return acc / max(1e-9, 1.0 - alpha)


def derive_views(probs: np.ndarray, reps: np.ndarray) -> Dict[str, float]:
    cdf = np.cumsum(probs)
    views = {"runtime_mean_ms": float(probs @ reps), "cvar95_ms": cvar_from_probs(probs, reps)}
    for tau in (0.50, 0.90, 0.95):
        idx = int(np.argmax(cdf >= tau))
        views["p%02d" % round(tau * 100)] = float(reps[idx])
    return views


# --------------------------------------------------------------------------- #
def head_outputs(ctx: base.Ctx, rows: Sequence[Mapping[str, Any]], arm: str) -> Dict[str, np.ndarray]:
    """Per (row, slot) resource quantities for the requested arm."""

    spec = dist.read_json(BIN_SPEC)
    edges = dist.edges_from_spec(spec)
    reps = dist.reps_from_spec(spec)
    arrays = common.encode_rows(rows, ctx.vocabs, ctx.horizon)
    model, _ = base.load_checkpoint(J3_CHECKPOINT, ctx, "J3")
    model.eval()

    res_hidden = None
    new_head = None
    if arm != "j3":
        payload = torch.load(HEAD_FILES[arm], map_location=ctx.device, weights_only=False)
        if payload.get("checkpoint_sha256") != J3_SHA256:
            raise SystemExit("%s was trained against a different checkpoint" % HEAD_FILES[arm])
        saved_spec = payload.get("spec") or {}
        if saved_spec and saved_spec.get("edges") != spec["edges"]:
            raise SystemExit("%s used a different bin spec" % HEAD_FILES[arm])
        if arm == "r1b":
            # trained by j_series_resource_dist_train_eval.py on the cached z = tanh(res_hidden(x)),
            # so res_hidden is the frozen one from the J3 checkpoint and the head reads 128-d z
            res_hidden = model.res_hidden
            head_state = payload["state_dict"]
            in_dim = int(payload["in_dim"])
        else:
            # trained by j_series_resource_dist_r3a.py, which saved both the head and the
            # (frozen or adapted) res_hidden
            frozen = payload["res_hidden"]
            res_hidden = torch.nn.Linear(frozen["weight"].shape[1], frozen["weight"].shape[0]).to(ctx.device)
            res_hidden.load_state_dict({k: v.to(ctx.device) for k, v in frozen.items()})
            head_state = payload["head"]
            in_dim = int(frozen["weight"].shape[0])
        new_head = dist.DiscreteRuntimeHead(in_dim, len(edges) - 1, hidden=payload.get("hidden") or HEAD_HIDDEN).to(ctx.device)
        new_head.load_state_dict(head_state)
        res_hidden.eval()
        new_head.eval()

    probs_out: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(rows), ctx.batch_size):
            indices = np.arange(start, min(start + ctx.batch_size, len(rows)), dtype=np.int64)
            batch = base.to_torch_batch(arrays, indices, ctx.device)
            captured: List[Any] = []

            def hook(_module: Any, inputs: Tuple[Any, ...], store: List[Any] = captured) -> None:
                store.append(inputs[0].detach())

            handle = model.res_hidden.register_forward_pre_hook(hook)
            try:
                base.forward(model, batch, "J3", ctx)
            finally:
                handle.remove()
            if not captured:
                raise RuntimeError("res_hidden pre-hook captured nothing")
            x = captured[-1]
            if arm == "j3":
                # reproduce the frozen quantiles as a degenerate distribution is not
                # possible; instead keep the J3 quantities verbatim and mark them
                probs_out.append(np.full((x.shape[0], x.shape[1], len(edges) - 1), np.nan, dtype=np.float64))
            else:
                logits = new_head(torch.tanh(res_hidden(x)))
                probs_out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return {"probs": np.concatenate(probs_out, axis=0), "edges": edges, "reps": reps}


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--anchors-file", type=Path, required=True)
    parser.add_argument("--base-pack", type=Path, required=True, help="directory holding b05_future_h5.jsonl.gz")
    parser.add_argument("--arm", choices=("j3", "r1b", "r3a_u", "r3a_f"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = PROJECT_ROOT / "experiments/EXP-20260911_p9d_j_predictor_acceptance/config.json"
    ctx = base.make_ctx(config, None)
    rows = read_anchors(args.anchors_file)
    # inference-only anchors carry no supervision block; the frozen packer fills the
    # same defaults, so encode_rows can run unchanged
    rows = [{**row, "future": [], "bounded_future_length": 0, "termination": 0} for row in rows]
    print("anchors:", len(rows))

    base_pack = read_pack(args.base_pack / "b05_future_h5.jsonl.gz")
    print("base pack nodes:", len(base_pack))

    out = head_outputs(ctx, rows, args.arm)
    probs, edges, reps = out["probs"], out["edges"], out["reps"]
    spec = dist.read_json(BIN_SPEC)
    bin_schema_id = "%s_%d" % (spec["mode"], len(edges) - 1)

    args.output_root.mkdir(parents=True, exist_ok=True)
    target = args.output_root / "b05_future_h5.jsonl.gz"
    if target.is_file() and not args.force:
        raise SystemExit("%s exists; pass --force" % target)

    written = 0
    steps_written = 0
    with gzip.open(target, "wt", encoding="utf-8", newline="\n") as handle:
        for i, row in enumerate(rows):
            node_id = str(row["current_node_id"])
            record = base_pack.get(node_id)
            if record is None:
                continue
            scenarios = record.get("future_h5") or []
            if not scenarios:
                continue
            steps = scenarios[0].get("steps") or []
            for t, step in enumerate(steps):
                if t >= probs.shape[1]:
                    break
                resource = step.setdefault("resource", {})
                if args.arm == "j3":
                    resource["resource_head_id"] = "J3_seed11_three_quantile"
                    continue
                p = probs[i, t]
                views = derive_views(p, reps)
                resource["runtime_ms_quantiles"] = {
                    "p50": views["p50"], "p90": views["p90"], "p95": views["p95"],
                }
                resource["runtime_mean_ms"] = views["runtime_mean_ms"]
                resource["cvar95_ms"] = views["cvar95_ms"]
                resource["runtime_probs"] = [float(x) for x in p]
                resource["bin_schema_id"] = bin_schema_id
                resource["bin_edges_ms"] = [float(e) for e in edges]
                resource["bin_representatives_ms"] = [float(r) for r in reps]
                resource["resource_head_id"] = ("%s_seed11" % args.arm)
                resource["runtime_semantics"] = "conditional_on_active"
                resource["quantile_rule"] = "cdf_inversion_bin_representative"
                resource["overflow_rule"] = "last_bin_representative_is_train_tail_mean"
                resource["calibration_version"] = "none"
                steps_written += 1
            written += 1
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

    manifest = {
        "schema_version": "resource-v2-artifact-manifest-v1",
        "arm": args.arm,
        "nodes": written,
        "steps": steps_written,
        "base_pack": str(args.base_pack),
        "anchors_file": str(args.anchors_file),
        "checkpoint_sha256": J3_SHA256,
        "bin_spec": str(BIN_SPEC),
        "bin_schema_id": bin_schema_id,
        "head_file": str(HEAD_FILES.get(args.arm)) if args.arm != "j3" else "frozen J3 head_runtime",
        "cvar_rule": CVAR_RULE,
        "runtime_semantics": "conditional_on_active (per-step marginals; no joint over the horizon)",
        "canonical_views": ["p50", "p90", "p95", "runtime_mean_ms", "cvar95_ms"],
        "source_of_truth": "runtime_probs + bin_edges_ms + bin_representatives_ms",
        "producer": "scripts/pack_resource_v2_artifacts.py",
    }
    (args.output_root / "resource_v2_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
