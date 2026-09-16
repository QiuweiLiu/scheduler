#!/usr/bin/env python3
"""资源侧树模型增强:domain + 上一事件族×模型交叉(防泄漏)加入 XGBoost/LightGBM

对比:v06 基准(22.7% XGBoost)
"""
import json
import math
import statistics as st
from collections import defaultdict
import numpy as np

FAMILY = {
    "sample_seek": "select_frames", "frame-selector": "select_frames",
    "image-grid-selector": "select_frames",
    "spatial_qa": "visual_qa", "image-qa": "visual_qa",
    "image-grid-qa": "visual_qa", "patch-zoomer": "visual_qa",
    "temporal-qa": "temporal_ops", "temporal-grounding": "temporal_ops",
    "summarize": "summarize", "summarization-tool": "summarize",
    "object_detection": "detect", "yolo-tracker": "detect",
}
FAMS = ["select_frames", "visual_qa", "temporal_ops", "summarize", "detect", "other"]


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def fam_of(raw):
    return FAMILY.get(raw, "other")


def metrics(preds, ys):
    errs = [abs(p - y) for p, y in zip(preds, ys)]
    return round(sum(errs) / len(errs), 1)


def bucket_acc(preds, ys):
    def b(v):
        return 0 if v < 1000 else (1 if v < 10000 else 2)
    return round(sum(1 for p, y in zip(preds, ys) if b(p) == b(y)) / len(ys), 4)


def main():
    rows = load_jsonl("results/processed/role_dataset_v0_3.jsonl")
    prefixes = load_jsonl("results/processed/neural_prefix_dataset_v0_9_options_sourcefix_20260805/prefixes_visual_context.jsonl")
    for r in rows:
        r["_fam"] = fam_of(r.get("raw_action"))

    vid_meta = {}
    for p in prefixes:
        ts = p.get("task_structure") or {}
        vid_meta[p["video_id"]] = {
            "domain": (ts.get("domain") if isinstance(ts, dict) else None) or "unknown",
            "q_chars": (ts.get("question_chars") if isinstance(ts, dict) else None) or 0,
            "q_tokens": (ts.get("question_tokens") if isinstance(ts, dict) else None) or 0,
            "opt_count": (ts.get("option_count") if isinstance(ts, dict) else None) or 0,
            "duration": ((p.get("state_features") or {}).get("video") or {}).get("duration_s"),
            "fps": ((p.get("state_features") or {}).get("video") or {}).get("fps"),
        }
    for r in rows:
        m = vid_meta.get(r["video_id"], {})
        r["_dom"] = m.get("domain", "unknown")
        r["_q_chars"] = m.get("q_chars", 0)
        r["_q_tokens"] = m.get("q_tokens", 0)
        r["_opt_count"] = m.get("opt_count", 0)
        r["_duration"] = float(m.get("duration") or 0)
        r["_fps"] = float(m.get("fps") or 0)

    by_run = defaultdict(list)
    for i, r in enumerate(rows):
        by_run[r["run_id"]].append(r)
    for rid, rs in by_run.items():
        rs.sort(key=lambda x: x["event_index"])
        acc_steps = acc_rt = prev_rt = 0.0
        prev_fam = None
        for r in rs:
            r["_acc_steps"] = acc_steps
            r["_acc_runtime"] = acc_rt
            r["_prev_runtime"] = prev_rt
            r["_prev_fam"] = prev_fam
            acc_steps += 1
            acc_rt += float(r.get("runtime_ms") or 0)
            prev_rt = float(r.get("runtime_ms") or 0)
            prev_fam = r["_fam"]

    max_ev = max(r["event_index"] for r in rows)
    fam_idx = {f: i for i, f in enumerate(FAMS)}
    models = sorted(set(r["model_id"] for r in rows))
    model_idx = {m: i for i, m in enumerate(models)}
    bases = sorted(set(r["baseline"] for r in rows))
    base_idx = {b: i for i, b in enumerate(bases)}
    qts = sorted(set(r["question_type"] for r in rows))
    qt_idx = {q: i for i, q in enumerate(qts)}
    doms = sorted(set(r["_dom"] for r in rows))
    dom_idx = {d: i for i, d in enumerate(doms)}

    rt = [r for r in rows if (r.get("runtime_ms") or 0) > 0]
    tr = [r for r in rt if r["split"] == "train"]
    te = [r for r in rt if r["split"] == "test"]

    # 交叉统计:上一事件族 × 模型(仅 train,防泄漏)
    gmed = defaultdict(list)
    for r in tr:
        gmed[(r["_prev_fam"], r["model_id"])].append(r["runtime_ms"])
    cross_med = {k: st.median(v) for k, v in gmed.items()}
    g = st.median([r["runtime_ms"] for r in tr])

    def feat(r):
        return np.array([fam_idx[r["_fam"]], model_idx[r["model_id"]], base_idx[r["baseline"]],
                         r["event_index"] / max_ev,
                         1.0 if r.get("model_resident_before") else 0.0,
                         1.0 if r.get("yolo_batch") else 0.0,
                         qt_idx[r["question_type"]], float(r.get("frame_count") or 0),
                         r["_q_chars"], r["_q_tokens"], r["_opt_count"],
                         r["_duration"], r["_fps"],
                         r["_acc_steps"], r["_acc_runtime"], r["_prev_runtime"],
                         dom_idx[r["_dom"]],
                         cross_med.get((r["_prev_fam"], r["model_id"]), g)],
                        dtype=np.float32)

    X = np.stack([feat(r) for r in rt])
    itr = [rt.index(r) for r in tr]
    ite = [rt.index(r) for r in te]
    ytr = np.array([math.log1p(r["runtime_ms"]) for r in tr], dtype=np.float32)
    ys = [r["runtime_ms"] for r in te]

    def run(name, model):
        model.fit(X[itr], ytr)
        preds = [math.expm1(p) for p in model.predict(X[ite])]
        mae = metrics(preds, ys)
        rel = mae / (sum(ys) / len(ys))
        print(f"  {name:24s} MAE={mae:8.0f} rel={rel:.1%} bucket={bucket_acc(preds, ys):.1%}")
        return {"mae": mae, "rel": round(rel, 4), "bucket": bucket_acc(preds, ys)}

    import xgboost as xgb
    import lightgbm as lgb
    results = {}
    results["XGBoost+增强"] = run("XGBoost+增强", xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42))
    results["LightGBM+增强"] = run("LightGBM+增强", lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=42))

    with open("results/processed/benchmark_resource_treeenh_20260806.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("写入 results/processed/benchmark_resource_treeenh_20260806.json")


if __name__ == "__main__":
    main()
