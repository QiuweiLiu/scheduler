#!/usr/bin/env python3
"""Bridge: run the latest predictor (histres F0/F1) on the SCHEDULER workload
(r7_workload_v041 templates, the Burst64 lineage) and produce per-node
predictions + a quality report against the templates' actual next nodes.

The workload and the predictor dataset come from different processing lines
(node IDs never intersect), so samples are BUILT here from the workload's own
node sequences:
  - attributes map 1:1 (model_id/node_type/event_type/raw_action direct;
    activity -> action_family via ACTIVITY_TO_FAMILY; role derived from
    node_type exactly as the training data does);
  - history_resource z-scores use the frozen train-fitted numeric_scaler.json
    (same formula as the dataset builder);
  - vocabularies come from the histres TRAIN rows (unknown values -> UNK).

Usage:
  python scripts/workload_histres_predict.py --variant F1 --seed 11 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
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

TEMPLATES = PROJECT_ROOT / "results/processed/r7_workload_v041_ontology_no_run_container/job_templates_r7_v041.jsonl"
SCALER = PROJECT_ROOT / "results/processed/j_series_dataset_histres_v2/numeric_scaler.json"

ACTIVITY_TO_FAMILY = {
    "other": "other",
    "sample_seek": "select_frames",
    "spatial_qa": "visual_qa",
    "temporal_qa": "temporal_ops",
    "summarize": "summarize",
    "object_detection": "detect",
}
NODE_TYPE_TO_ROLE = {
    "planner": "plan",
    "videotool_temporal": "execute",
    "videotool_spatial": "execute",
    "videotool_generalist": "execute",
    "answer_generation": "aggregate",
    "run_control": "init",
}


def _z(scaler_field: dict, value: float, zero_inflated: bool) -> tuple[float, int, int]:
    """Return (z, present, nonzero) using the frozen scaler formula."""
    if value is None or not math.isfinite(float(value)):
        return 0.0, 0, 0
    v = max(0.0, float(value))
    present = 1
    if zero_inflated:
        if v <= 0.0:
            return 0.0, present, 0
        z = (math.log1p(v) - scaler_field["log1p_median"]) / scaler_field["scale"]
        return max(-6.0, min(6.0, z)), present, 1
    z = (math.log1p(v) - scaler_field["log1p_median"]) / scaler_field["scale"]
    return max(-6.0, min(6.0, z)), present, 0


def history_resource(node: dict, scaler: dict) -> dict:
    f = scaler["fields"]
    rz, rp, _ = _z(f["runtime_ms"], node.get("runtime_ms"), False)
    lz, lp, ln = _z(f["load_ms"], node.get("load_ms"), True)
    az, ap, _ = _z(f["peak_allocated_mb"], node.get("workspace_peak_mb"), False)
    sz, sp, _ = _z(f["peak_reserved_mb"], node.get("resident_model_mb"), False)
    return {"status_class": str(node.get("status") or "UNOBSERVED"),
            "runtime_z": round(rz, 6), "runtime_present": rp,
            "load_positive_z": round(lz, 6), "load_present": lp, "load_nonzero": ln,
            "peak_alloc_z": round(az, 6), "peak_alloc_present": ap,
            "peak_reserved_z": round(sz, 6), "peak_reserved_present": sp}


def node_attrs(node: dict, position: int) -> dict:
    activity = str(node.get("activity") or "other")
    node_type = str(node.get("node_type") or "")
    return {"event_type": str(node.get("event_type") or ""),
            "node_type": node_type,
            "role": NODE_TYPE_TO_ROLE.get(node_type, "execute"),
            "raw_action": str(node.get("raw_action") or activity),
            "action_family": ACTIVITY_TO_FAMILY.get(activity, "other"),
            "model_id": str(node.get("model_id") or ""),
            "position": position}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="F1")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all templates")
    args = ap.parse_args()

    scaler = json.loads(SCALER.read_text(encoding="utf-8"))

    # vocabularies from the histres TRAIN rows (same as training)
    import gzip
    train_rows = []
    with gzip.open(PROJECT_ROOT / "results/processed/j_series_dataset_histres_v2/histres_train.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            train_rows.append(json.loads(line))
    vocabs = common.build_vocabs(train_rows)

    # build workload rows
    rows = []
    meta = []
    with open(TEMPLATES, encoding="utf-8") as f:
        for ti, line in enumerate(f):
            if args.limit and ti >= args.limit:
                break
            t = json.loads(line)
            nodes = t.get("nodes") or []
            video_id = str(t.get("base_task_id") or t.get("run_id") or f"tpl{ti}")
            task_ctx = {"baseline": str(t.get("baseline") or ""),
                        "model_stack_id": str(t.get("model_stack_id") or ""),
                        "planner_model_id": str(t.get("planner_model_id") or "")}
            unobserved = {"status_class": "UNOBSERVED", "runtime_z": 0.0, "runtime_present": 0,
                          "load_positive_z": 0.0, "load_present": 0, "load_nonzero": 0,
                          "peak_alloc_z": 0.0, "peak_alloc_present": 0,
                          "peak_reserved_z": 0.0, "peak_reserved_present": 0}
            for i, node in enumerate(nodes):
                # training convention: history[-1] == current_node; the current
                # node's resource channel is unobserved (decision-time visibility)
                hist = [dict(node_attrs(nodes[j], j),
                             history_resource=(history_resource(nodes[j], scaler) if j < i else dict(unobserved)))
                        for j in range(i + 1)]
                mi = {"current_node": node_attrs(node, i), "history": hist,
                      "task_context": dict(task_ctx), "stack_context": dict(task_ctx)}
                rows.append({"video_id": video_id, "current_node_id": str(node.get("node_id") or ""),
                             "model_input": mi, "future": [], "split": "workload",
                             "bounded_future_length": 0, "termination": 0})
                nxt = nodes[i + 1] if i + 1 < len(nodes) else None
                meta.append({"video_id": video_id, "position": i,
                             "node_id": str(node.get("node_id") or ""),
                             "actual_next": ({"node_type": str(nxt.get("node_type") or ""),
                                              "role": NODE_TYPE_TO_ROLE.get(str(nxt.get("node_type") or ""), "execute"),
                                              "action_family": ACTIVITY_TO_FAMILY.get(str(nxt.get("activity") or "other"), "other"),
                                              "model_class": str(nxt.get("model_id") or ""),
                                              "runtime_ms": nxt.get("runtime_ms"), "load_ms": nxt.get("load_ms")}
                                             if nxt else None)})
    print(f"built {len(rows)} workload anchor samples from {len(set(r['video_id'] for r in rows))} templates", flush=True)

    horizon = 5
    arrays = common.encode_rows(rows, vocabs, horizon)

    # model
    config = json.loads(H.CONFIG.read_text(encoding="utf-8"))
    config = json.loads(json.dumps(config))
    config["variants"].update({"F0": dict(te.DEFAULT_VARIANTS["F0"]), "F1": dict(te.DEFAULT_VARIANTS["F1"])})
    config["model"]["runtime_distribution_head"] = True
    config["model"]["runtime_head_hidden"] = 64
    ctx = H.build_ctx(config)  # loads histres train/validation; gives vocabs+variant configs
    ctx.vocabs = vocabs
    ctx.device = torch.device("cpu")
    ckpt = H.OUT_ROOT / "runs" / args.variant / f"seed{args.seed}" / "checkpoint.pt"
    model, _m = te.load_checkpoint(ckpt, ctx, args.variant)
    model.eval()
    print(f"loaded {args.variant} seed{args.seed} on {ctx.device}", flush=True)

    # forward in batches; extract slot predictions
    n = len(rows)
    out_records = []
    with torch.no_grad():
        for indices in te.split_indices(n, 256):
            batch = te.to_torch_batch(arrays, indices, ctx.device)
            outputs = te.forward(model, batch, args.variant, ctx)
            behavior = outputs["behavior"]
            structure = outputs["structure"]
            resource = outputs["resource"]
            attrs = outputs["attributes"]
            role = behavior["next_role_logits"].argmax(dim=-1).cpu().numpy()
            fam = behavior["next_family_logits"].argmax(dim=-1).cpu().numpy()
            slot_nt = attrs["node_type"].argmax(dim=-1).cpu().numpy()
            slot_role = attrs["role"].argmax(dim=-1).cpu().numpy()
            slot_fam = attrs["action_family"].argmax(dim=-1).cpu().numpy()
            slot_model = attrs["model_class"].argmax(dim=-1).cpu().numpy()
            term = torch.sigmoid(structure["termination_logit"]).cpu().numpy()
            if "runtime_logits" in resource:
                import j_series_resource_dist as _dist
                probs = torch.softmax(resource["runtime_logits"], dim=-1)
                reps = torch.as_tensor(model.runtime_bins["representatives_ms"], dtype=probs.dtype)
                views = _dist.derive_from_probs(probs, reps.numpy())
                q50 = views["q50_ms"].cpu().numpy(); q90 = views["q90_ms"].cpu().numpy()
            else:
                q = torch.expm1(resource["runtime_log_quantiles"]).clamp(min=0.0).cpu().numpy()
                q50 = q[..., 0]; q90 = q[..., 1]
            if term.ndim == 2:
                term = term[:, 0]
            def _s0(x):
                a = np.asarray(x)
                return a[:, 0] if a.ndim == 2 else a
            role0, fam0 = _s0(role), _s0(fam)
            nt0, sr0, sf0, sm0 = _s0(slot_nt), _s0(slot_role), _s0(slot_fam), _s0(slot_model)
            r50, r90 = _s0(q50), _s0(q90)
            for j, idx in enumerate(indices.tolist()):
                m = meta[idx]
                an = m.get("actual_next")
                rec = {"video_id": m["video_id"], "position": m["position"], "node_id": m["node_id"],
                       "pred_next_role_idx": int(role0[j]), "pred_next_family_idx": int(fam0[j]),
                       "pred_slot0_node_type_idx": int(nt0[j]), "pred_slot0_role_idx": int(sr0[j]),
                       "pred_slot0_family_idx": int(sf0[j]), "pred_slot0_model_idx": int(sm0[j]),
                       "pred_runtime_q50_ms": round(float(r50[j]), 1), "pred_runtime_q90_ms": round(float(r90[j]), 1),
                       "termination_prob": round(float(term[j]), 4)}
                if an:
                    rec["actual_next_node_type"] = an["node_type"]
                    rec["actual_next_model_class"] = an["model_class"]
                    rec["actual_next_runtime_ms"] = an["runtime_ms"]
                out_records.append(rec)

    # decode label indices -> strings via vocabs for the report
    def dec(vocab, i):
        inv = {v: k for k, v in vocab.index.items()}
        return inv.get(int(i), "?")
    nt_vocab, role_vocab, fam_vocab, model_vocab = vocabs.slot["node_type"], vocabs.slot["role"], vocabs.slot["action_family"], vocabs.slot["model_class"]
    nxt_nt = nxt_mc = 0
    tot = 0
    per_type = {}
    for rec in out_records:
        if "actual_next_node_type" not in rec:
            continue
        tot += 1
        pnt = dec(nt_vocab, rec["pred_slot0_node_type_idx"])
        pmc = dec(model_vocab, rec["pred_slot0_model_idx"])
        ok_nt = (pnt == rec["actual_next_node_type"])
        ok_mc = (pmc == rec["actual_next_model_class"])
        nxt_nt += ok_nt; nxt_mc += ok_mc
        key = rec["actual_next_node_type"]
        d = per_type.setdefault(key, [0, 0, 0])
        d[0] += 1; d[1] += ok_nt; d[2] += ok_mc
    report = {"schema": "workload-histres-predict", "variant": args.variant, "seed": args.seed,
              "n_anchors": len(out_records), "n_with_actual_next": tot,
              "slot0_node_type_acc": round(nxt_nt / max(1, tot), 4),
              "slot0_model_class_acc": round(nxt_mc / max(1, tot), 4),
              "per_actual_type": {k: {"n": v[0], "nt_acc": round(v[1] / max(1, v[0]), 3), "mc_acc": round(v[2] / max(1, v[0]), 3)}
                                  for k, v in sorted(per_type.items())}}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"workload_predictions_{args.variant}_seed{args.seed}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out_records) + "\n", encoding="utf-8")
    (args.out / f"workload_predict_report_{args.variant}_seed{args.seed}.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
