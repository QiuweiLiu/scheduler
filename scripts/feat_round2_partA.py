#!/usr/bin/env python3
"""Part A:候选特征收敛(资源侧 XGBoost 13.7% 基线逐个加,记录增量)

候选:prev2_runtime / 三阶交叉(上一族×模型×位置段) / 生命周期(is_first+run_len_proxy) / 同视频历史均值
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
MODEL_ATTR = {
    "Qwen3-VL-8B-Instruct": (8.1, 19.0, 1.0),
    "Qwen3-4B": (3.9, 8.0, 0.55),
    "Qwen2.5-VL-3B-Instruct": (3.8, 7.3, 0.5),
    "cpu-metadata-adapter-v1": (0.0, 0.0, 0.9),
    "yolo11x.pt": (0.02, 0.4, 0.8),
    "finish_argument": (0.0, 0.0, 0.9),
    "stack_a_qwen3_vl8b": (8.1, 19.0, 1.0),
    "stack_b_qwen3_4b_qwen25vl3b_yolo26n": (3.9, 8.0, 0.55),
}
D = "/root/autodl-tmp/scheduler/"


def load_jsonl(p):
    return [json.loads(l) for l in open(D + p)]


def fam_of(raw):
    return FAMILY.get(raw, "other")


def metrics(preds, ys):
    errs = [abs(p - y) for p, y in zip(preds, ys)]
    return round(sum(errs) / len(errs), 1), round(sum(errs) / sum(ys), 4)


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
        acc_steps = acc_rt = prev_rt = prev2_rt = 0.0
        prev_fam = None
        for k, r in enumerate(rs):
            r["_acc_steps"] = acc_steps
            r["_acc_runtime"] = acc_rt
            r["_prev_runtime"] = prev_rt
            r["_prev2_runtime"] = prev2_rt
            r["_prev_fam"] = prev_fam
            r["_is_first"] = 1.0 if k == 0 else 0.0
            acc_steps += 1
            acc_rt += float(r.get("runtime_ms") or 0)
            prev2_rt = prev_rt
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

    # 统计特征(train 拟合,防泄漏)
    gmed2 = defaultdict(list)
    gmed3 = defaultdict(list)
    for r in tr:
        gmed2[(r["_prev_fam"], r["model_id"])].append(r["runtime_ms"])
        gmed3[(r["_prev_fam"], r["model_id"], min(r["event_index"], 9) // 3)].append(r["runtime_ms"])
    cross2 = {k: st.median(v) for k, v in gmed2.items()}
    cross3 = {k: st.median(v) for k, v in gmed3.items()}
    g = st.median([r["runtime_ms"] for r in tr])
    # 同视频+baseline 历史 run 平均长度(代理)+ 同视频历史平均 runtime
    run_len = defaultdict(list)
    vid_rt = defaultdict(list)
    for rid, rs in by_run.items():
        if rs[0]["split"] != "train":
            continue
        run_len[(rs[0]["video_id"], rs[0]["baseline"])].append(len(rs))
        vid_rt[rs[0]["video_id"]].append(st.median([x["runtime_ms"] for x in rs if (x.get("runtime_ms") or 0) > 0] or [0]))
    len_proxy = {k: st.median(v) for k, v in run_len.items()}
    vid_med = {k: st.median(v) for k, v in vid_rt.items()}
    glen = st.median([x for v in run_len.values() for x in v])
    gvid = st.median([x for v in vid_rt.values() for x in v])
    max_rt = max((float(r.get("runtime_ms") or 0) for r in rows), default=1.0)

    def feat(r, extras):
        attr = MODEL_ATTR.get(r["model_id"], (0, 0, 0))
        x = [fam_idx[r["_fam"]], model_idx[r["model_id"]], base_idx[r["baseline"]],
             r["event_index"] / max_ev,
             1.0 if r.get("model_resident_before") else 0.0,
             1.0 if r.get("yolo_batch") else 0.0,
             qt_idx[r["question_type"]], float(r.get("frame_count") or 0),
             r["_q_chars"], r["_q_tokens"], r["_opt_count"],
             r["_duration"], r["_fps"],
             r["_acc_steps"], r["_acc_runtime"], r["_prev_runtime"],
             dom_idx[r["_dom"]], cross2.get((r["_prev_fam"], r["model_id"]), g)]
        if "prev2" in extras:
            x.append(r["_prev2_runtime"] / max_rt)
        if "cross3" in extras:
            x.append(cross3.get((r["_prev_fam"], r["model_id"], min(r["event_index"], 9) // 3), g))
        if "life" in extras:
            x.append(r["_is_first"])
            x.append(len_proxy.get((r["video_id"], r["baseline"]), glen))
        if "vidhist" in extras:
            x.append(vid_med.get(r["video_id"], gvid))
        return np.array(x, dtype=np.float32)

    itr = [rt.index(r) for r in tr]
    ite = [rt.index(r) for r in te]
    ytr = np.array([math.log1p(r["runtime_ms"]) for r in tr], dtype=np.float32)
    ys = [r["runtime_ms"] for r in te]

    import xgboost as xgb
    combos = [
        ("基线(13.7% 参照)", ()),
        ("+prev2", ("prev2",)),
        ("+cross3", ("cross3",)),
        ("+life", ("life",)),
        ("+vidhist", ("vidhist",)),
        ("+prev2+cross3", ("prev2", "cross3")),
        ("+prev2+cross3+life", ("prev2", "cross3", "life")),
        ("全加", ("prev2", "cross3", "life", "vidhist")),
    ]
    results = {}
    for tag, extras in combos:
        X = np.stack([feat(r, extras) for r in rt])
        m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                             random_state=42).fit(X[itr], ytr)
        preds = [math.expm1(p) for p in m.predict(X[ite])]
        mae, rel = metrics(preds, ys)
        results[tag] = {"mae": mae, "rel": rel, "bucket": bucket_acc(preds, ys),
                        "n_feat": int(X.shape[1])}
        print(f"  {tag:24s} MAE={mae:8.0f} rel={rel:.1%} bucket={bucket_acc(preds, ys):.1%} n_feat={X.shape[1]}")

    with open(D + "results/processed/feat_round2_partA_20260806.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("写入 results/processed/feat_round2_partA_20260806.json")


if __name__ == "__main__":
    main()
