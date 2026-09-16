#!/usr/bin/env python3
"""资源侧 NN 全关系特征增强:模型属性表 + domain + 族×模型交叉统计

在 nnfeat(one-hot+std)基础上加:
- 模型属性表:参数量(B)/显存基表(GB)/相对速度(已知常数)
- domain(视频级 join prefix)
- 族×模型 train 中位交叉统计(训练集算,防泄漏)
对比:MLP / ProD / KNN vs nnfeat 版(18.3%/65.7%/68.1%)
"""
import json
import math
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
# 模型属性表(参数量 B / 显存基表 GB / 相对速度 0-1)
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


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def fam_of(raw):
    return FAMILY.get(raw, "other")


def metrics(preds, ys):
    errs = [abs(p - y) for p, y in zip(preds, ys)]
    return round(sum(errs) / len(errs), 1), round(math.sqrt(sum(e * e for e in errs) / len(errs)), 1)


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
            "q_chars": (ts.get("question_chars") if isinstance(ts, dict) else None) or 0,
            "q_tokens": (ts.get("question_tokens") if isinstance(ts, dict) else None) or 0,
            "opt_count": (ts.get("option_count") if isinstance(ts, dict) else None) or 0,
            "domain": (ts.get("domain") if isinstance(ts, dict) else None) or "unknown",
            "sub_category": (ts.get("sub_category") if isinstance(ts, dict) else None) or "unknown",
            "duration": ((p.get("state_features") or {}).get("video") or {}).get("duration_s"),
            "fps": ((p.get("state_features") or {}).get("video") or {}).get("fps"),
        }
    for r in rows:
        m = vid_meta.get(r["video_id"], {})
        r["_q_chars"] = m.get("q_chars", 0)
        r["_q_tokens"] = m.get("q_tokens", 0)
        r["_opt_count"] = m.get("opt_count", 0)
        r["_domain"] = m.get("domain", "unknown")
        r["_sub"] = m.get("sub_category", "unknown")
        r["_duration"] = float(m.get("duration") or 0)
        r["_fps"] = float(m.get("fps") or 0)

    by_run = defaultdict(list)
    for i, r in enumerate(rows):
        by_run[r["run_id"]].append(r)
    for rid, rs in by_run.items():
        rs.sort(key=lambda x: x["event_index"])
        acc_steps = acc_rt = prev_rt = 0.0
        for r in rs:
            r["_acc_steps"] = acc_steps
            r["_acc_runtime"] = acc_rt
            r["_prev_runtime"] = prev_rt
            acc_steps += 1
            acc_rt += float(r.get("runtime_ms") or 0)
            prev_rt = float(r.get("runtime_ms") or 0)

    max_ev = max(r["event_index"] for r in rows)
    fams = sorted(set(r["_fam"] for r in rows))
    fam_idx = {f: i for i, f in enumerate(fams)}
    models = sorted(set(r["model_id"] for r in rows))
    model_idx = {m: i for i, m in enumerate(models)}
    bases = sorted(set(r["baseline"] for r in rows))
    base_idx = {b: i for i, b in enumerate(bases)}
    qts = sorted(set(r["question_type"] for r in rows))
    qt_idx = {q: i for i, q in enumerate(qts)}
    doms = sorted(set(r["_domain"] for r in rows))
    dom_idx = {d: i for i, d in enumerate(doms)}

    def onehot(vals, size):
        v = np.zeros(size, dtype=np.float32)
        v[vals] = 1.0
        return v

    def feat(r, cross_med):
        attr = MODEL_ATTR.get(r["model_id"], (0, 0, 0))
        return np.concatenate([
            onehot([fam_idx[r["_fam"]]], len(fams)),
            onehot([model_idx[r["model_id"]]], len(models)),
            onehot([base_idx[r["baseline"]]], len(bases)),
            onehot([qt_idx[r["question_type"]]], len(qts)),
            onehot([dom_idx[r["_domain"]]], len(doms)),
            np.array([r["event_index"] / max_ev,
                      1.0 if r.get("model_resident_before") else 0.0,
                      1.0 if r.get("yolo_batch") else 0.0,
                      float(r.get("frame_count") or 0),
                      r["_q_chars"], r["_q_tokens"], r["_opt_count"],
                      r["_duration"], r["_fps"],
                      r["_acc_steps"], r["_acc_runtime"], r["_prev_runtime"],
                      float(attr[0]), float(attr[1]), float(attr[2]),
                      cross_med], dtype=np.float32),
        ])

    rt = [r for r in rows if (r.get("runtime_ms") or 0) > 0]
    tr = [r for r in rt if r["split"] == "train"]
    te = [r for r in rt if r["split"] == "test"]

    # 族×模型交叉统计(train 中位,防泄漏)
    gmed = defaultdict(list)
    for r in tr:
        gmed[(r["_fam"], r["model_id"])].append(r["runtime_ms"])
    import statistics as st
    cross_med = {k: st.median(v) for k, v in gmed.items()}
    g = st.median([r["runtime_ms"] for r in tr])

    def feat_all(r):
        return feat(r, cross_med.get((r["_fam"], r["model_id"]), g))

    Xtr = np.stack([feat_all(r) for r in tr])
    Xte = np.stack([feat_all(r) for r in te])
    num_start = len(fams) + len(models) + len(bases) + len(qts) + len(doms)
    mu = Xtr[:, num_start:].mean(0)
    sd = Xtr[:, num_start:].std(0).clip(min=1e-6)
    Xtr[:, num_start:] = (Xtr[:, num_start:] - mu) / sd
    Xte[:, num_start:] = (Xte[:, num_start:] - mu) / sd
    ytr = np.array([math.log1p(r["runtime_ms"]) for r in tr], dtype=np.float32)
    ys = [r["runtime_ms"] for r in te]
    print(f"增强特征维度: {Xtr.shape[1]}")

    results = {}

    # KNN
    tr_rt = np.array([r["runtime_ms"] for r in tr], dtype=np.float32)
    preds = []
    for i, e in enumerate(Xte):
        d = np.abs(Xtr - e).sum(1)
        d[[j for j, r in enumerate(tr) if r["run_id"] == te[i]["run_id"]]] = 1e9
        idx = np.argsort(d)[:20]
        preds.append(float(np.median(tr_rt[idx])))
    mae, rmse = metrics(preds, ys)
    results["KNN_enh"] = {"mae": mae, "rel": round(mae / (sum(ys) / len(ys)), 4),
                          "bucket": bucket_acc(preds, ys)}
    print(f"KNN(增强): MAE={mae} rel={mae/(sum(ys)/len(ys)):.1%} bucket={bucket_acc(preds, ys):.1%}")

    import torch
    import torch.nn as nn
    torch.manual_seed(42)
    xt = torch.from_numpy(Xtr)
    xte = torch.from_numpy(Xte)
    yt = torch.from_numpy(ytr).unsqueeze(1)

    def train_mlp(model, epochs=120):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for ep in range(epochs):
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xt), yt)
            loss.backward(); opt.step()
        model.eval()
        return model

    mlp = train_mlp(nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)))
    with torch.no_grad():
        pl = mlp(xte).squeeze(1).numpy()
    preds = [math.expm1(p) for p in pl]
    mae, rmse = metrics(preds, ys)
    results["MLP_enh"] = {"mae": mae, "rel": round(mae / (sum(ys) / len(ys)), 4),
                          "bucket": bucket_acc(preds, ys)}
    print(f"MLP(增强): MAE={mae} rel={mae/(sum(ys)/len(ys)):.1%} bucket={bucket_acc(preds, ys):.1%}")

    n_bins = 32
    max_log = float(ytr.max())
    edges = np.linspace(0, max_log + 0.1, n_bins + 1)
    ybin = np.clip(np.searchsorted(edges, ytr, side="right") - 1, 0, n_bins - 1)
    dm = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(), nn.Linear(128, n_bins))
    opt2 = torch.optim.Adam(dm.parameters(), lr=1e-3)
    yb = torch.from_numpy(ybin).long()
    for ep in range(120):
        opt2.zero_grad()
        loss = nn.functional.cross_entropy(dm(xt), yb)
        loss.backward(); opt2.step()
    dm.eval()
    with torch.no_grad():
        probs = torch.softmax(dm(xte), dim=1).numpy()
    centers = (edges[:-1] + edges[1:]) / 2
    preds = [math.expm1(float(np.dot(p, centers))) for p in probs]
    mae, rmse = metrics(preds, ys)
    results["ProD_enh"] = {"mae": mae, "rel": round(mae / (sum(ys) / len(ys)), 4),
                           "bucket": bucket_acc(preds, ys)}
    print(f"ProD(增强): MAE={mae} rel={mae/(sum(ys)/len(ys)):.1%} bucket={bucket_acc(preds, ys):.1%}")

    with open("results/processed/benchmark_resource_nnenh_20260806.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("写入 results/processed/benchmark_resource_nnenh_20260806.json")


if __name__ == "__main__":
    main()
