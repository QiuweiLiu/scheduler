#!/usr/bin/env python3
"""特征数值化对比实验:类别 one-hot + 数值 z-score vs 标签编码

验证假设:NN 表现差是因为标签编码(虚假顺序)+ 数值未规范化。
对比:MLP / ProD / KNN(one-hot+标准化)vs 旧版(标签编码)vs 树模型(XGBoost 22.7% 基准)
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


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def fam_of(raw):
    return FAMILY.get(raw, "other")


def metrics(preds, ys):
    errs = [abs(p - y) for p, y in zip(preds, ys)]
    mae = sum(errs) / len(errs)
    rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
    return round(mae, 1), round(rmse, 1)


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
        sf = p.get("state_features") or {}
        vd = sf.get("video") if isinstance(sf, dict) else None
        vid_meta[p["video_id"]] = {
            "q_chars": (ts.get("question_chars") if isinstance(ts, dict) else None) or 0,
            "q_tokens": (ts.get("question_tokens") if isinstance(ts, dict) else None) or 0,
            "opt_count": (ts.get("option_count") if isinstance(ts, dict) else None) or 0,
            "duration": vd.get("duration_s") if isinstance(vd, dict) else None,
            "fps": vd.get("fps") if isinstance(vd, dict) else None,
        }
    for r in rows:
        m = vid_meta.get(r["video_id"], {})
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

    # one-hot + 数值(未标准化,后面 z-score)
    def onehot(r):
        x = []
        v = np.zeros(len(fams)); v[fam_idx[r["_fam"]]] = 1; x.extend(v)
        v = np.zeros(len(models)); v[model_idx[r["model_id"]]] = 1; x.extend(v)
        v = np.zeros(len(bases)); v[base_idx[r["baseline"]]] = 1; x.extend(v)
        v = np.zeros(len(qts)); v[qt_idx[r["question_type"]]] = 1; x.extend(v)
        x += [r["event_index"] / max_ev,
              1.0 if r.get("model_resident_before") else 0.0,
              1.0 if r.get("yolo_batch") else 0.0,
              float(r.get("frame_count") or 0),
              r["_q_chars"], r["_q_tokens"], r["_opt_count"],
              r["_duration"], r["_fps"],
              r["_acc_steps"], r["_acc_runtime"], r["_prev_runtime"]]
        return np.array(x, dtype=np.float32)

    rt = [r for r in rows if (r.get("runtime_ms") or 0) > 0]
    tr = [r for r in rt if r["split"] == "train"]
    te = [r for r in rt if r["split"] == "test"]

    Xtr = np.stack([onehot(r) for r in tr])
    Xte = np.stack([onehot(r) for r in te])
    # z-score(仅数值列,one-hot 不标准化)
    cat_dims = len(fams) + len(models) + len(bases) + len(qts)
    mu = Xtr[:, cat_dims:].mean(0)
    sd = Xtr[:, cat_dims:].std(0).clip(min=1e-6)
    Xtr[:, cat_dims:] = (Xtr[:, cat_dims:] - mu) / sd
    Xte[:, cat_dims:] = (Xte[:, cat_dims:] - mu) / sd
    ytr = np.array([math.log1p(r["runtime_ms"]) for r in tr], dtype=np.float32)
    ys = [r["runtime_ms"] for r in te]
    print(f"one-hot+标准化特征维度: {Xtr.shape[1]}(类别 {cat_dims} + 数值 {Xtr.shape[1]-cat_dims})")
    print(f"train={len(tr)} test={len(te)}")

    results = {}

    # KNN(one-hot+标准化)
    tr_rt = np.array([r["runtime_ms"] for r in tr], dtype=np.float32)
    preds = []
    for i, e in enumerate(Xte):
        d = np.abs(Xtr - e).sum(1)
        d[[j for j, r in enumerate(tr) if r["run_id"] == te[i]["run_id"]]] = 1e9
        idx = np.argsort(d)[:20]
        preds.append(float(np.median(tr_rt[idx])))
    mae, rmse = metrics(preds, ys)
    results["KNN_onehot"] = {"mae": mae, "rmse": rmse, "rel": round(mae / (sum(ys) / len(ys)), 4),
                             "bucket": bucket_acc(preds, ys)}
    print(f"KNN(onehot+std): MAE={mae} rel={mae/(sum(ys)/len(ys)):.1%} bucket={bucket_acc(preds, ys):.1%}")

    import torch
    import torch.nn as nn
    torch.manual_seed(42)
    xt = torch.from_numpy(Xtr)
    xte = torch.from_numpy(Xte)
    yt = torch.from_numpy(ytr).unsqueeze(1)

    # MLP(one-hot+标准化)
    mlp = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    for ep in range(120):
        opt.zero_grad()
        loss = nn.functional.mse_loss(mlp(xt), yt)
        loss.backward(); opt.step()
    mlp.eval()
    with torch.no_grad():
        pl = mlp(xte).squeeze(1).numpy()
    preds = [math.expm1(p) for p in pl]
    mae, rmse = metrics(preds, ys)
    results["MLP_onehot"] = {"mae": mae, "rmse": rmse, "rel": round(mae / (sum(ys) / len(ys)), 4),
                             "bucket": bucket_acc(preds, ys)}
    print(f"MLP(onehot+std):  MAE={mae} rel={mae/(sum(ys)/len(ys)):.1%} bucket={bucket_acc(preds, ys):.1%}")

    # ProD 分布(one-hot+标准化)
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
    results["ProD_onehot"] = {"mae": mae, "rmse": rmse, "rel": round(mae / (sum(ys) / len(ys)), 4),
                              "bucket": bucket_acc(preds, ys)}
    print(f"ProD(onehot+std): MAE={mae} rel={mae/(sum(ys)/len(ys)):.1%} bucket={bucket_acc(preds, ys):.1%}")

    with open("results/processed/benchmark_resource_nnfeat_20260806.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("写入 results/processed/benchmark_resource_nnfeat_20260806.json")


if __name__ == "__main__":
    main()
