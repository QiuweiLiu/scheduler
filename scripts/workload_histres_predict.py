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
import os
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
STATES_ROOT = PROJECT_ROOT / "results/raw/r7_trace_full_20260817"
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


def load_task_context(states_root: Path, run_id: str) -> dict:
    p = states_root / run_id / "states" / "state_0000.json"
    if not p.is_file():
        return {}
    try:
        st = json.loads(p.read_text(encoding="utf-8"))
        task = st.get("task") or {}
        return {"answer_type": task.get("answer_type"), "domain": task.get("domain"),
                "question_type": task.get("question_type"), "sub_category": task.get("sub_category"),
                "temporal_scope": task.get("temporal_scope"),
                "required_modalities": task.get("required_modalities") or []}
    except Exception:
        return {}


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
            if os.environ.get("NO_CTX") != "1":
                task_ctx.update(load_task_context(STATES_ROOT, str(t.get("run_id") or "")))
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
                fut = []
                for k in range(1, 6):
                    if i + k < len(nodes):
                        nn = nodes[i + k]
                        fut.append({"node_type": str(nn.get("node_type") or ""),
                                    "role": NODE_TYPE_TO_ROLE.get(str(nn.get("node_type") or ""), "execute"),
                                    "action_family": ACTIVITY_TO_FAMILY.get(str(nn.get("activity") or "other"), "other"),
                                    "model_class": str(nn.get("model_id") or ""),
                                    "runtime_ms": nn.get("runtime_ms"), "load_ms": nn.get("load_ms")})
                meta.append({"video_id": video_id, "position": i,
                             "node_id": str(node.get("node_id") or ""),
                             "actual_next": fut[0] if fut else None, "actual_future": fut})
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

    # forward in batches; extract per-slot predictions (horizon slots 0..4)
    n = len(rows)
    HORIZON = 5
    pred_nt = np.zeros((n, HORIZON), dtype=np.int64)
    pred_role = np.zeros((n, HORIZON), dtype=np.int64)
    pred_fam = np.zeros((n, HORIZON), dtype=np.int64)
    pred_mc = np.zeros((n, HORIZON), dtype=np.int64)
    pred_r50 = np.zeros((n, HORIZON))
    pred_r90 = np.zeros((n, HORIZON))
    term_p = np.zeros(n)
    with torch.no_grad():
        for indices in te.split_indices(n, 256):
            batch = te.to_torch_batch(arrays, indices, ctx.device)
            outputs = te.forward(model, batch, args.variant, ctx)
            attrs = outputs["attributes"]
            resource = outputs["resource"]
            structure = outputs["structure"]
            nt = np.asarray(attrs["node_type"].argmax(dim=-1).cpu().numpy())
            rl = np.asarray(attrs["role"].argmax(dim=-1).cpu().numpy())
            fa = np.asarray(attrs["action_family"].argmax(dim=-1).cpu().numpy())
            mc = np.asarray(attrs["model_class"].argmax(dim=-1).cpu().numpy())
            if "runtime_logits" in resource:
                import j_series_resource_dist as _dist
                probs = torch.softmax(resource["runtime_logits"], dim=-1)
                reps = torch.as_tensor(model.runtime_bins["representatives_ms"], dtype=probs.dtype)
                views = _dist.derive_from_probs(probs, reps.numpy())
                q50 = np.asarray(views["q50_ms"].cpu().numpy())
                q90 = np.asarray(views["q90_ms"].cpu().numpy())
            else:
                q = torch.expm1(resource["runtime_log_quantiles"]).clamp(min=0.0).cpu().numpy()
                q50 = np.asarray(q[..., 0]); q90 = np.asarray(q[..., 1])
            tp = np.asarray(torch.sigmoid(structure["termination_logit"]).cpu().numpy())
            if tp.ndim == 2:
                tp = tp[:, 0]
            pred_nt[indices] = nt
            pred_role[indices] = rl
            pred_fam[indices] = fa
            pred_mc[indices] = mc
            pred_r50[indices] = q50
            pred_r90[indices] = q90
            term_p[indices] = tp

    nt_v, role_v, fam_v, mc_v = vocabs.slot["node_type"], vocabs.slot["role"], vocabs.slot["action_family"], vocabs.slot["model_class"]

    def dec(vocab, i):
        inv = {v: k for k, v in vocab.index.items()}
        return inv.get(int(i), "?")

    slot_stats = {k: {"n": 0, "nt": 0, "role": 0, "fam": 0, "mc": 0} for k in range(1, HORIZON + 1)}
    joint = {"n": 0, "all_mc": 0, "set_mc": 0}
    records = []
    for i, m in enumerate(meta):
        fut = m.get("actual_future") or []
        L = len(fut)
        slots = []
        for k in range(HORIZON):
            slots.append({"node_type": dec(nt_v, pred_nt[i, k]), "role": dec(role_v, pred_role[i, k]),
                          "action_family": dec(fam_v, pred_fam[i, k]), "model_class": dec(mc_v, pred_mc[i, k]),
                          "runtime_q50_ms": round(float(pred_r50[i, k]), 1),
                          "runtime_q90_ms": round(float(pred_r90[i, k]), 1)})
        if L:
            ok_all = True
            pred_set, true_set = set(), set()
            for k in range(L):
                st = slot_stats[k + 1]
                st["n"] += 1
                st["nt"] += (slots[k]["node_type"] == fut[k]["node_type"])
                st["role"] += (slots[k]["role"] == fut[k]["role"])
                st["fam"] += (slots[k]["action_family"] == fut[k]["action_family"])
                st["mc"] += (slots[k]["model_class"] == fut[k]["model_class"])
                ok_all = ok_all and (slots[k]["model_class"] == fut[k]["model_class"])
                pred_set.add(slots[k]["model_class"]); true_set.add(fut[k]["model_class"])
            joint["n"] += 1
            joint["all_mc"] += ok_all
            joint["set_mc"] += (pred_set == true_set)
        records.append({"video_id": m["video_id"], "position": m["position"], "node_id": m["node_id"],
                        "slots": slots, "termination_prob": round(float(term_p[i]), 4),
                        "actual_future": [{"node_type": f["node_type"], "model_class": f["model_class"],
                                           "runtime_ms": f.get("runtime_ms")} for f in fut]})

    report = {"schema": "workload-histres-predict-v2", "variant": args.variant, "seed": args.seed,
              "n_anchors": len(records),
              "slot_stats": {f"depth{k}": {"n": v["n"],
                                           "nt_acc": round(v["nt"] / max(1, v["n"]), 4),
                                           "role_acc": round(v["role"] / max(1, v["n"]), 4),
                                           "family_acc": round(v["fam"] / max(1, v["n"]), 4),
                                           "model_class_acc": round(v["mc"] / max(1, v["n"]), 4)}
                             for k, v in slot_stats.items()},
              "joint": {"n": joint["n"],
                        "all_depths_model_correct": round(joint["all_mc"] / max(1, joint["n"]), 4),
                        "model_set_match": round(joint["set_mc"] / max(1, joint["n"]), 4)}}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"workload_predictions_{args.variant}_seed{args.seed}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    (args.out / f"workload_predict_report_{args.variant}_seed{args.seed}.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
