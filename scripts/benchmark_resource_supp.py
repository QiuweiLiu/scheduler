#!/usr/bin/env python3
"""资源预测补充:KNN 复核(排除同 run 邻居)+ 剩余时长聚合原型

KNN 复核:验证 2,041ms 是否被"同 run 内事件相似"虚高
剩余时长聚合:给定 run 当前位置,剩余时长 = Σ(后续事件单步 P50);
参照真实剩余时长,报 MAE;剩余步数用真实值(行为侧将来预测)
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


def main():
    rows = load_jsonl("results/processed/role_dataset_v0_3.jsonl")
    for r in rows:
        r["_fam"] = fam_of(r.get("raw_action"))
    fam_idx = {f: i for i, f in enumerate(FAMS)}
    models = sorted(set(r["model_id"] for r in rows))
    model_idx = {m: i for i, m in enumerate(models)}
    bases = sorted(set(r["baseline"] for r in rows))
    base_idx = {b: i for i, b in enumerate(bases)}
    qts = sorted(set(r["question_type"] for r in rows))
    qt_idx = {q: i for i, q in enumerate(qts)}
    max_ev = max(r["event_index"] for r in rows)

    def feat(r):
        return np.array([fam_idx[r["_fam"]], model_idx[r["model_id"]], base_idx[r["baseline"]],
                         r["event_index"] / max_ev,
                         1.0 if r.get("model_resident_before") else 0.0,
                         1.0 if r.get("yolo_batch") else 0.0,
                         qt_idx[r["question_type"]], float(r.get("frame_count") or 0)],
                        dtype=np.float32)

    rt = [r for r in rows if (r.get("runtime_ms") or 0) > 0]
    tr = [r for r in rt if r["split"] == "train"]
    te = [r for r in rt if r["split"] == "test"]

    # ===== KNN 复核:排除同 run 邻居 =====
    tf = np.stack([feat(r) for r in tr])
    ef = np.stack([feat(r) for r in te])
    mu, sd = tf.mean(0), tf.std(0).clip(min=1e-6)
    tn, en = (tf - mu) / sd, (ef - mu) / sd
    tr_rt = np.array([r["runtime_ms"] for r in tr], dtype=np.float32)

    preds_no_run = []
    for i, e in enumerate(en):
        te_run = te[i]["run_id"]
        d = np.abs(tn - e).sum(1)
        d[te_run == np.array([r["run_id"] for r in tr])] = 1e9  # 排除同 run
        idx = np.argsort(d)[:20]
        preds_no_run.append(float(np.median(tr_rt[idx])))
    ys = [r["runtime_ms"] for r in te]
    errs = [abs(p - y) for p, y in zip(preds_no_run, ys)]
    print(f"KNN 复核(排除同run): MAE={sum(errs)/len(errs):.1f}ms "
          f"(原版 {sum(abs(p-y) for p,y in zip([float(np.median(tr_rt[np.argsort(np.abs(tn-e).sum(1))[:20]])) for e in en], ys))/len(ys):.1f})")
    print(f"  对照: LightGBM MAE=3033.9ms")

    # ===== 剩余时长聚合原型 =====
    # 每 run 事件序
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run_id"]].append(r)
    for rs in by_run.values():
        rs.sort(key=lambda x: x["event_index"])
    # 单步 P50 分位表(族×模型)
    gmed = defaultdict(list)
    for r in tr:
        gmed[(r["_fam"], r["model_id"])].append(r["runtime_ms"])
    med = {k: st.median(v) for k, v in gmed.items()}
    g = st.median([r["runtime_ms"] for r in tr])

    # 对 test run 的每个位置:剩余时长预测 = Σ 后续事件 P50;真实 = Σ 后续实际 runtime
    all_pred, all_true = [], []
    for rid, rs in by_run.items():
        if rs[0]["split"] != "test":
            continue
        for k, r in enumerate(rs):
            if not (r.get("runtime_ms") or 0) > 0:
                continue
            nxt = rs[k + 1:]
            if not nxt:
                continue
            pred = sum(med.get((x["_fam"], x["model_id"]), g) for x in nxt if (x.get("runtime_ms") or 0) > 0)
            true = sum(x["runtime_ms"] for x in nxt if (x.get("runtime_ms") or 0) > 0)
            all_pred.append(pred)
            all_true.append(true)
    errs2 = [abs(p - t) for p, t in zip(all_pred, all_true)]
    rel = sum(errs2) / sum(all_true)
    print(f"剩余时长聚合(真实剩余步数): n={len(all_pred)} MAE={sum(errs2)/len(errs2):.0f}ms rel={rel:.1%}")

    out = {
        "knn_no_run_mae": round(sum(errs) / len(errs), 1),
        "remaining_duration": {"n": len(all_pred),
                               "mae": round(sum(errs2) / len(errs2), 1),
                               "rel_mae": round(rel, 4)},
    }
    with open("results/processed/resource_supplement_20260806.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("写入 results/processed/resource_supplement_20260806.json")


if __name__ == "__main__":
    main()
