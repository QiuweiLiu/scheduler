#!/usr/bin/env python
"""Repack the scheduler-facing future artifacts with a resource-v2 head.

Two-layer resource schema (per the interface review):

* **source of truth** - ``runtime_probs[H, K]`` per step plus the bin metadata
  (``bin_edges_ms`` / ``bin_representatives_ms`` / ``cvar_alpha``) carried **once** at
  the artifact level in ``resource_v2_manifest.json``, so later consumers can derive
  CVaR, deadline-exceedance, scenarios, ... without re-running the predictor;
* **canonical views** - ``q50/q90/q95`` and ``runtime_mean_ms`` frozen into every step,
  plus ``cvar95_ms`` once its definition is fixed, so two scheduler versions cannot
  derive different q95 from the same artifact.

Everything outside the runtime part of the ``resource`` block is copied verbatim from a
base pack produced by the frozen packer, so two arms differ only in the resource block.
The load fields (``load_occurrence_probability`` / ``load_duration_ms_quantiles``) are
never touched: the review flagged that changing the runtime rule *and* the load rule in
one arm would confound the consumer effect.

Arms
----
``j3``     the frozen three-quantile head: a **pure identity copy** of the base pack,
           asserted byte-for-byte, so arm A0 is provably the frozen artifact
``r1b``    the trained discrete head on frozen features (mass-balanced 16 bins)
``r3a_u``  the same head plus the adapted ``res_hidden`` (R3a arm U)
``r3a_f``  the same head with ``res_hidden`` frozen (R3a arm F, control)

Fail-closed
-----------
The review found that the first version could silently drop nodes and steps. This
version has no ``continue``/``break`` on the data path: a missing node, a missing
scenario, a multi-scenario row, an un-upgradable step or a duplicate ``node_id`` all
raise, and the run ends with hard post-conditions plus a post-pack validator.

Usage::

    python scripts/pack_resource_v2_artifacts.py \
        --anchors-file outputs/sstar_predictor_anchors/features_sstar.jsonl.gz \
        --base-pack outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts \
        --arm r1b --output-root outputs/resource_v2_artifacts/r1b
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
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
SCHEMA_VERSION = "resource-v2-artifact-manifest-v2"
CVAR_RULE = (
    "discrete_upper_tail_partial_weight: "
    "CVaR_a = (1/(1-a)) * sum_k max(0, cdf_k - max(cdf_{k-1}, a)) * m_k"
)
# the runtime keys this tool owns; everything else in `resource` must survive untouched
OWNED_RUNTIME_KEYS = (
    "runtime_probs",
    "runtime_ms_quantiles",
    "runtime_mean_ms",
    "cvar95_ms",
    "resource_head_id",
    "bin_schema_id",
)
PRESERVED_RESOURCE_KEYS = ("load_occurrence_probability", "load_duration_ms_quantiles")
STEP_VIEW_TOLERANCE = 1e-9


# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(cwd: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(cwd), capture_output=True, text=True, timeout=20,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def rel(path: Path) -> str:
    """Project-relative POSIX path so the manifest is portable across machines."""

    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_anchors(path: Path) -> List[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_pack(path: Path) -> Dict[str, Dict[str, Any]]:
    """Read a pack keyed by node_id. Duplicate node ids are a hard error."""

    out: Dict[str, Dict[str, Any]] = {}
    duplicates: List[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            node_id = str(row["node_id"])
            if node_id in out:
                duplicates.append(node_id)
            out[node_id] = row
    if duplicates:
        raise SystemExit(
            "base pack has %d duplicate node_id(s): %s" % (len(duplicates), duplicates[:5])
        )
    return out


def write_gzip_lines(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Write JSONL, refusing NaN/Infinity so the output is strict RFC JSON."""

    # mtime=0 keeps the byte stream (and therefore the artifact sha256) reproducible
    with open(path, "wb") as binary:
        with gzip.GzipFile(fileobj=binary, mode="wb", mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="\n") as text:
                for record in records:
                    text.write(json.dumps(record, ensure_ascii=False, sort_keys=True,
                                          separators=(",", ":"), allow_nan=False) + "\n")


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


def resource_without_runtime(step: Mapping[str, Any]) -> Dict[str, Any]:
    """The step's resource block minus everything this tool is allowed to rewrite."""

    resource = step.get("resource") or {}
    return {k: v for k, v in resource.items() if k not in OWNED_RUNTIME_KEYS}


def record_without_runtime(record: Mapping[str, Any], horizon: int) -> Dict[str, Any]:
    """A node record with all runtime-owned keys stripped, for equality checks."""

    clone = json.loads(json.dumps(record, ensure_ascii=False))
    for scenario in clone.get("future_h%d" % horizon) or []:
        for step in scenario.get("steps") or []:
            resource = step.get("resource")
            if isinstance(resource, dict):
                for key in OWNED_RUNTIME_KEYS:
                    resource.pop(key, None)
    return clone


# --------------------------------------------------------------------------- #
def head_outputs(ctx: base.Ctx, rows: Sequence[Mapping[str, Any]], arm: str) -> Dict[str, np.ndarray]:
    """Per (row, slot) resource probabilities for the requested arm."""

    spec = dist.read_json(BIN_SPEC)
    edges = dist.edges_from_spec(spec)
    reps = dist.reps_from_spec(spec)
    if arm == "j3":
        return {"probs": np.empty((len(rows), ctx.horizon, len(edges) - 1)), "edges": edges, "reps": reps}

    arrays = common.encode_rows(rows, ctx.vocabs, ctx.horizon)
    model, _ = base.load_checkpoint(J3_CHECKPOINT, ctx, "J3")
    model.eval()

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
    else:
        # trained by j_series_resource_dist_r3a.py, which saved both the head and the
        # (frozen or adapted) res_hidden. That file carries no bin spec, so the dimension
        # contract below is derived from the weights themselves -- a stronger check than
        # trusting metadata.
        frozen = payload["res_hidden"]
        res_hidden = torch.nn.Linear(frozen["weight"].shape[1], frozen["weight"].shape[0]).to(ctx.device)
        res_hidden.load_state_dict({k: v.to(ctx.device) for k, v in frozen.items()})
        head_state = payload["head"]

    w_in = head_state["net.0.weight"]
    w_out = head_state["net.2.weight"]
    in_dim = int(w_in.shape[1])
    hidden = int(w_in.shape[0])
    n_bins = int(w_out.shape[0])
    if n_bins != len(edges) - 1:
        raise SystemExit("%s emits %d bins but the spec has %d" % (HEAD_FILES[arm], n_bins, len(edges) - 1))
    if int(w_out.shape[1]) != hidden:
        raise SystemExit("%s hidden dims disagree between layers" % HEAD_FILES[arm])
    if int(res_hidden.weight.shape[0]) != in_dim:
        raise SystemExit("%s res_hidden output (%d) does not match the head input (%d)"
                         % (HEAD_FILES[arm], int(res_hidden.weight.shape[0]), in_dim))
    if arm == "r1b":
        if in_dim != int(payload["in_dim"]) or hidden != int(payload.get("hidden") or HEAD_HIDDEN):
            raise SystemExit("R1b metadata disagrees with the stored weights")

    # the training scripts assert this exact replay: z = tanh(res_hidden(x)) -> head(z)
    new_head = dist.DiscreteRuntimeHead(in_dim, n_bins, hidden=hidden).to(ctx.device)
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
            if len(captured) != 1:
                raise RuntimeError("res_hidden pre-hook captured %d tensors" % len(captured))
            x = captured[0]
            logits = new_head(torch.tanh(res_hidden(x)))
            # float64 from here on: the CVaR crossing-bin weight is a near-cancellation of two
            # cdf values, so float32 rounding blows up past the view-invariance tolerance
            probs_out.append(torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64))
    probs = np.concatenate(probs_out, axis=0)
    if not np.isfinite(probs).all():
        raise SystemExit("head produced non-finite probabilities")
    return {"probs": probs, "edges": edges, "reps": reps}


# --------------------------------------------------------------------------- #
def head_provenance(arm: str) -> Dict[str, Any]:
    """Training metadata of the selected head, recorded in the manifest for audit."""

    path = HEAD_FILES.get(arm)
    if path is None or not path.is_file():
        return {"kind": "frozen J3 three-quantile head (no separate file)"}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "file": rel(path),
        "trainer": ("j_series_resource_dist_train_eval.py" if arm == "r1b" else "j_series_resource_dist_r3a.py"),
        "arm": payload.get("arm"),
        "epochs": payload.get("epochs"),
        "selected_epoch": payload.get("selected_epoch"),
        "trainable_params": payload.get("trainable_params"),
        "in_dim": payload.get("in_dim"),
        "hidden": payload.get("hidden"),
        "seed": payload.get("seed"),
        "checkpoint_sha256": payload.get("checkpoint_sha256"),
    }


def _view_error(got: Any, want: float, label: str) -> float:
    """Absolute error of a canonical view, or inf when it is missing or non-finite.

    ``max(0.0, nan)`` silently returns 0.0 in Python, so the previous inline
    ``abs(float(...) - want)`` could pass an artifact whose view was absent
    entirely.  The review found exactly that divergence: validate_post_pack would
    PASS a pack that load_resource_v2_overlay then rejects.  Returning inf keeps
    the two validators agreeing on what "missing" means.
    """

    if got is None:
        return float("inf")
    if isinstance(got, bool) or not isinstance(got, (int, float)):
        return float("inf")
    value = float(got)
    if value != value or value in (float("inf"), float("-inf")):
        return float("inf")
    return abs(value - float(want))


def validate_post_pack(
    base_pack_path: Path,
    new_pack_path: Path,
    manifest: Mapping[str, Any],
    edges: np.ndarray,
    reps: np.ndarray,
) -> Dict[str, Any]:
    """Post-pack invariance audit required by the review (P0-3)."""

    base_pack = read_pack(base_pack_path)
    new_pack = read_pack(new_pack_path)
    horizon = int(manifest["horizon"])
    arm = str(manifest["arm"])

    report: Dict[str, Any] = {
        "arm": arm,
        "node_count": len(new_pack),
        "step_count": 0,
        "missing_nodes": sorted(set(base_pack) - set(new_pack)),
        "extra_nodes": sorted(set(new_pack) - set(base_pack)),
        "duplicate_nodes": 0,  # read_pack raises on duplicates, so reaching here means 0
        "multi_scenario_rows": 0,
        "unupgraded_steps": 0,
        "prob_sum_max_abs_error": 0.0,
        "canonical_view_max_abs_error": 0.0,
        "nonresource_mismatch_count": 0,
        "bin_schema_mismatch_count": 0,
        "load_field_missing_count": 0,
        "nan_prob_count": 0,
    }
    if report["missing_nodes"] or report["extra_nodes"]:
        raise SystemExit("node sets differ: missing=%d extra=%d" % (
            len(report["missing_nodes"]), len(report["extra_nodes"])))

    probs_sum_err = 0.0
    view_err = 0.0
    for node_id, base_record in base_pack.items():
        new_record = new_pack[node_id]
        base_scen = base_record.get("future_h%d" % horizon) or []
        new_scen = new_record.get("future_h%d" % horizon) or []
        if len(new_scen) != 1:
            report["multi_scenario_rows"] += 1
            raise SystemExit("node %s has %d scenarios (expected exactly 1)" % (node_id, len(new_scen)))
        if len(base_scen) != 1:
            raise SystemExit("base node %s has %d scenarios" % (node_id, len(base_scen)))

        if record_without_runtime(base_record, horizon) != record_without_runtime(new_record, horizon):
            report["nonresource_mismatch_count"] += 1

        base_steps = base_scen[0].get("steps") or []
        new_steps = new_scen[0].get("steps") or []
        if len(new_steps) != len(base_steps):
            raise SystemExit("node %s step count changed: %d -> %d" % (node_id, len(base_steps), len(new_steps)))

        for t, step in enumerate(new_steps):
            report["step_count"] += 1
            resource = step.get("resource") or {}
            for key in PRESERVED_RESOURCE_KEYS:
                if key not in resource:
                    report["load_field_missing_count"] += 1

            if arm == "j3":
                continue  # identity arm: the byte comparison below is the real check

            probs = resource.get("runtime_probs")
            if not isinstance(probs, list) or len(probs) != len(reps):
                report["unupgraded_steps"] += 1
                continue
            if not all(isinstance(p, (int, float)) for p in probs):
                report["nan_prob_count"] += 1
                continue
            arr = np.asarray(probs, dtype=np.float64)
            if not np.isfinite(arr).all():
                report["nan_prob_count"] += 1
                continue
            probs_sum_err = max(probs_sum_err, abs(float(arr.sum()) - 1.0))

            views = derive_views(arr, np.asarray(reps, dtype=np.float64))
            quantiles = resource.get("runtime_ms_quantiles") or {}
            for key, want in (
                ("p50", views["p50"]), ("p90", views["p90"]), ("p95", views["p95"]),
            ):
                view_err = max(view_err, _view_error(quantiles.get(key), want, "p" + key[1:]))
            view_err = max(view_err, _view_error(resource.get("runtime_mean_ms"), views["runtime_mean_ms"], "runtime_mean_ms"))
            view_err = max(view_err, _view_error(resource.get("cvar95_ms"), views["cvar95_ms"], "cvar95_ms"))
            if resource.get("bin_schema_id") != manifest["bin_schema_id"]:
                report["bin_schema_mismatch_count"] += 1

    report["prob_sum_max_abs_error"] = probs_sum_err
    report["canonical_view_max_abs_error"] = view_err

    if arm != "j3":
        if report["unupgraded_steps"]:
            raise SystemExit("%d steps were never upgraded" % report["unupgraded_steps"])
        if probs_sum_err > 1e-6:
            raise SystemExit("runtime_probs do not sum to 1 (max err %.3e)" % probs_sum_err)
        if view_err > STEP_VIEW_TOLERANCE:
            raise SystemExit("canonical views disagree with runtime_probs (max err %.3e)" % view_err)
    if report["nonresource_mismatch_count"]:
        raise SystemExit("%d nodes changed outside the runtime block" % report["nonresource_mismatch_count"])
    if report["bin_schema_mismatch_count"]:
        raise SystemExit("%d steps carry the wrong bin_schema_id" % report["bin_schema_mismatch_count"])
    if report["load_field_missing_count"]:
        raise SystemExit("%d steps lost a load field" % report["load_field_missing_count"])
    if report["nan_prob_count"]:
        raise SystemExit("%d steps carry non-finite probabilities" % report["nan_prob_count"])
    return report


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--anchors-file", type=Path, required=True)
    parser.add_argument("--base-pack", type=Path, required=True, help="directory holding b05_future_h5.jsonl.gz")
    parser.add_argument("--arm", choices=("j3", "r1b", "r3a_u", "r3a_f"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--producer-commit", default=None,
                        help="commit recorded in the manifest (defaults to the local public clone)")
    args = parser.parse_args()

    config = PROJECT_ROOT / "experiments/EXP-20260911_p9d_j_predictor_acceptance/config.json"
    ctx = base.make_ctx(config, None)
    horizon = int(ctx.horizon)
    rows = read_anchors(args.anchors_file)
    # inference-only anchors carry no supervision block; the frozen packer fills the
    # same defaults, so encode_rows can run unchanged
    rows = [{**row, "future": [], "bounded_future_length": 0, "termination": 0} for row in rows]
    print("anchors:", len(rows))

    base_pack_path = args.base_pack / "b05_future_h5.jsonl.gz"
    base_pack = read_pack(base_pack_path)
    print("base pack nodes:", len(base_pack))

    anchor_ids = [str(row["current_node_id"]) for row in rows]
    if len(set(anchor_ids)) != len(anchor_ids):
        raise SystemExit("anchors file has duplicate current_node_id values")
    # P0-1: the first version silently skipped anchors that were absent from the pack
    missing = sorted(set(anchor_ids) - set(base_pack))
    if missing:
        raise SystemExit("anchors file has %d node(s) absent from the base pack: %s" % (len(missing), missing[:5]))
    if len(anchor_ids) != len(base_pack):
        raise SystemExit("anchors (%d) and base pack (%d) disagree in size" % (len(anchor_ids), len(base_pack)))

    spec = dist.read_json(BIN_SPEC)
    edges = dist.edges_from_spec(spec)
    reps = dist.reps_from_spec(spec)
    bin_schema_id = "%s_%d" % (spec["mode"], len(edges) - 1)

    args.output_root.mkdir(parents=True, exist_ok=True)
    target = args.output_root / "b05_future_h5.jsonl.gz"
    if target.is_file() and not args.force:
        raise SystemExit("%s exists; pass --force" % target)

    out = head_outputs(ctx, rows, args.arm)
    probs = out["probs"]

    records: List[Dict[str, Any]] = []
    base_step_count = 0
    steps_upgraded = 0
    for i, row in enumerate(rows):
        node_id = str(row["current_node_id"])
        record = base_pack[node_id]
        scenarios = record.get("future_h%d" % horizon) or []
        if len(scenarios) != 1:
            raise SystemExit("node %s carries %d scenarios; refusing to guess" % (node_id, len(scenarios)))
        steps = scenarios[0].get("steps") or []
        if len(steps) > probs.shape[1]:
            raise SystemExit("node %s has %d steps but only %d slots were predicted"
                             % (node_id, len(steps), probs.shape[1]))
        base_step_count += len(steps)
        for t, step in enumerate(steps):
            resource = step.setdefault("resource", {})
            if args.arm == "j3":
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
            resource["resource_head_id"] = "%s_seed11" % args.arm
            steps_upgraded += 1
        records.append(record)

    write_gzip_lines(target, records)

    # P0-1 post-conditions: hard failures, not log lines
    if len(records) != len(rows) or len(records) != len(base_pack):
        raise SystemExit("wrote %d records for %d anchors / %d base nodes" % (len(records), len(rows), len(base_pack)))
    if args.arm != "j3" and steps_upgraded != base_step_count:
        raise SystemExit("upgraded %d of %d steps" % (steps_upgraded, base_step_count))

    head_path = HEAD_FILES.get(args.arm)
    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": "resource_v2_%s_seed11" % args.arm,
        "arm": args.arm,
        "artifact_sha256": sha256_file(target),
        "producer": "scripts/pack_resource_v2_artifacts.py",
        "producer_commit_sha": args.producer_commit or git_commit(PROJECT_ROOT.parent / "scheduler_public_repo"),
        "producer_checkpoint_sha256": J3_SHA256,
        "producer_checkpoint": "J3_seed11 backbone (frozen)",
        "resource_head_id": ("J3_seed11_three_quantile" if args.arm == "j3" else "%s_seed11" % args.arm),
        "resource_head_file": rel(head_path) if head_path else None,
        "resource_head_sha256": sha256_file(head_path) if head_path and head_path.is_file() else None,
        "resource_head_training": head_provenance(args.arm),
        "base_pack": rel(base_pack_path),
        "base_pack_sha256": sha256_file(base_pack_path),
        "bin_spec": rel(BIN_SPEC),
        "bin_spec_sha256": sha256_file(BIN_SPEC),
        "bin_schema_id": bin_schema_id,
        "bin_edges_ms": [float(e) for e in edges],
        "bin_representatives_ms": [float(r) for r in reps],
        "bin_upper_edge_overflow": True,
        "bin_upper_edge_note": (
            "the last finite edge is the train p99.5 boundary; the final bin is open-ended and its "
            "representative is the train tail mean, so tail moments outside the training support are "
            "not identified (bins.json still serialises Infinity for backward compatibility)"
        ),
        "runtime_semantics": "conditional_on_active (per-step marginals; no joint over the horizon)",
        "quantile_rule": "cdf_inversion_bin_representative",
        "overflow_rule": "last_bin_representative_is_train_tail_mean",
        "calibration_version": "none",
        "cvar_alpha": ALPHA_CVAR,
        "cvar_rule": CVAR_RULE,
        "cvar_semantics": (
            "CVaR of the discretized representative-valued predictive distribution, NOT the exact CVaR "
            "of the true continuous runtime distribution; the open-ended last bin bounds the worst-case "
            "tail bias only below the training support"
        ),
        "consumer_compat": {
            "legacy_sum_q95_v1": "reads runtime_ms_quantiles.{p50,p90,p95} only",
            "sum_conditional_runtime_mean_plus_legacy_load_v1": "reads runtime_mean_ms",
            "sum_marginal_step_cvar95_plus_legacy_load_v1": "reads cvar95_ms",
            "requires_manifest_for": ["bin_edges_ms", "bin_representatives_ms", "cvar_alpha"],
        },
        "horizon": horizon,
        "nodes": len(records),
        "steps": steps_upgraded if args.arm != "j3" else base_step_count,
        "canonical_views": ["p50", "p90", "p95", "runtime_mean_ms", "cvar95_ms"],
        "source_of_truth": "runtime_probs per step + bin_edges_ms/bin_representatives_ms at artifact level",
        "anchor_source": rel(args.anchors_file),
        "anchors_sha256": sha256_file(args.anchors_file),
        "identity_copy": args.arm == "j3",
    }

    if args.arm == "j3":
        # a byte-identical copy is impossible (gzip mtime and JSON key order differ), so the
        # guarantee we can actually assert is semantic: every parsed record deep-equals the base
        reparsed = read_pack(target)
        differing = [n for n in base_pack if base_pack[n] != reparsed[n]]
        if differing:
            raise SystemExit(
                "j3 arm is not a semantic identity copy: %d node(s) differ, e.g. %s"
                % (len(differing), differing[:3])
            )
        manifest["identity_verified"] = (
            "semantic identity: every parsed record deep-equals the frozen base pack; "
            "the byte streams differ only in gzip mtime and JSON key ordering"
        )
        manifest["artifact_sha256_of_base_serialisation"] = manifest["base_pack_sha256"]

    validation = validate_post_pack(base_pack_path, target, manifest, edges, reps)
    manifest["validation"] = validation
    (args.output_root / "resource_v2_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_root / "resource_v2_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "consumer_compat"},
                     ensure_ascii=False, indent=2))
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
