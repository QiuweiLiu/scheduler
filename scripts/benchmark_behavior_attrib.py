#!/usr/bin/env python3
"""行为侧从属层特征验证:执行者(3)+ 模型栈(3)加入动作族预测

对比:nnfeat(one-hot+std,无从属层)vs 本脚本(+执行者/栈)
NN 族 + 树模型对照
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
LLM_SET = {"Qwen3-VL-8B-Instruct", "Qwen2.5-VL-3B-Instruct"}
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


def executor_of(mid):
    if mid in LLM_SET:
        return "LLM"
    if "yolo" in str(mid):
        return "YOLO"
    return "TOOL"


def stack_of(mid):
    s = str(mid)
    if "stack_b" in s or "Qwen3-4B" in s or "Qwen2.5-VL-3B" in s:
        return "stack_b"
    if "stack_a" in s or "Qwen3-VL-8B" in s:
        return "stack_a"
    return "tool"


def evaluate(y_true, probs, labels):
    idx = {l: i for i, l in enumerate(labels)}
    top1 = top3 = 0
    nll = 0.0
    for t, p in zip(y_true, probs):
        order = sorted(range(len(p)), key=lambda i: -p[i])
        if labels[order[0]] == t:
            top1 += 1
        if t in {labels[i] for i in order[:3]}:
            top3 += 1
        nll += -math.log(max(p[idx.get(t, 0)], 1e-9))
    n = len(y_true)
    return {"top1": round(top1 / n, 4), "top3": round(top3 / n, 4), "nll": round(nll / n, 4)}


def main():
    rows = load_jsonl("results/processed/role_dataset_v0_3.jsonl")
    rows = [r for r in rows if r["role"] == "execute"]
    for r in rows:
        r["_fam"] = fam_of(r.get("raw_action"))
        r["_exe"] = executor_of(r["model_id"])
        r["_stack"] = stack_of(r["model_id"])
    print(f"execute 事件: {len(rows)}")

    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "validation"]
    te = [r for r in rows if r["split"] == "test"]
    fam_idx = {f: i for i, f in enumerate(FAMS)}
    labels = FAMS
    models = sorted(set(r["model_id"] for r in rows))
    bases = sorted(set(r["baseline"] for r in rows))
    qts = sorted(set(r["question_type"] for r in rows))
    exes = sorted(set(r["_exe"] for r in rows))
    stacks = sorted(set(r["_stack"] for r in rows))
    max_ev = max(r["event_index"] for r in rows)

    # domain + 族×模型交叉统计(train 中位,防泄漏)
    prefixes = load_jsonl("results/processed/neural_prefix_dataset_v0_9_options_sourcefix_20260805/prefixes_visual_context.jsonl")
    vid_dom = {}
    for p in prefixes:
        ts = p.get("task_structure") or {}
        vid_dom[p["video_id"]] = ((ts.get("domain") if isinstance(ts, dict) else None) or "unknown")
    for r in rows:
        r["_dom"] = vid_dom.get(r["video_id"], "unknown")
    doms = sorted(set(r["_dom"] for r in rows))
    dom_idx = {d: i for i, d in enumerate(doms)}
    import statistics as st
    gmed = defaultdict(list)
    for r in tr:
        gmed[(r["_fam"], r["model_id"])].append(0)  # 占位,行为侧交叉用族频率替代
    # 行为侧交叉:族×模型组合的样本频率(train 比例)作为关系强度
    gcnt = defaultdict(int)
    for r in tr:
        gcnt[(r["_fam"], r["model_id"])] += 1
    n_tr = len(tr)
    cross_ratio = {k: v / n_tr for k, v in gcnt.items()}

    def onehot(vals, size):
        v = np.zeros(size, dtype=np.float32)
        v[vals] = 1.0
        return v

    def feat(r, last_fam):
        attr = MODEL_ATTR.get(r["model_id"], (0, 0, 0))
        cross = cross_ratio.get((last_fam, r["model_id"]), 0.0) if last_fam else 0.0
        return np.concatenate([
            onehot([models.index(r["model_id"])], len(models)),
            onehot([bases.index(r["baseline"])], len(bases)),
            onehot([qts.index(r["question_type"])], len(qts)),
            onehot([exes.index(r["_exe"])], len(exes)),
            onehot([stacks.index(r["_stack"])], len(stacks)),
            onehot([dom_idx[r["_dom"]]], len(doms)),
            np.array([r["event_index"] / max_ev,
                      1.0 if r.get("model_resident_before") else 0.0,
                      1.0 if r.get("yolo_batch") else 0.0,
                      float(attr[0]), float(attr[1]), float(attr[2]),
                      cross], dtype=np.float32),
            onehot([fam_idx[last_fam]], 6) if last_fam else np.zeros(6, dtype=np.float32),
        ])

    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run_id"]].append(r)
    for rs in by_run.values():
        rs.sort(key=lambda x: x["event_index"])
    seq_of = {rid: [r["_fam"] for r in rs] for rid, rs in by_run.items()}

    W = 5
    feats = []
    for r in rows:
        s = seq_of[r["run_id"]]
        pos = s.index(r["_fam"]) if r["_fam"] in s else len(s) - 1
        last = s[pos - 1] if pos >= 1 else None
        feats.append(feat(r, last))
    X = np.stack(feats)
    num_start = len(models) + len(bases) + len(qts) + len(exes) + len(stacks) + len(doms)
    num_end = num_start + 7
    mu = X[0:len(tr), num_start:num_end].mean(0)
    sd = X[0:len(tr), num_start:num_end].std(0).clip(min=1e-6)
    X[:, num_start:num_end] = (X[:, num_start:num_end] - mu) / sd
    y = np.array([fam_idx[r["_fam"]] for r in rows], dtype=np.int64)
    print(f"特征维度(含从属层): {X.shape[1]}")

    itr, iva, ite = [rows.index(r) for r in tr], [rows.index(r) for r in va], [rows.index(r) for r in te]
    ytr, yva, yte = y[itr], y[iva], y[ite]
    yte_names = [r["_fam"] for r in te]

    def seq_win(r):
        s = seq_of[r["run_id"]]
        pos = s.index(r["_fam"]) if r["_fam"] in s else len(s) - 1
        win = [fam_idx[x] for x in s[max(0, pos + 1 - W):pos]]
        return win + [6] * (W - len(win))

    S = np.array([seq_win(r) for r in rows], dtype=np.int64)

    results = {"n_feat": int(X.shape[1]), "models": {}}

    # 树模型(标签编码版 + 全部关系:执行者/栈/domain/模型属性/交叉/最近族,公平对比)
    from collections import Counter
    import lightgbm as lgb
    import xgboost as xgb
    exe_idx = {e: i for i, e in enumerate(exes)}
    stk_idx = {s: i for i, s in enumerate(stacks)}
    base_idx = {b: i for i, b in enumerate(bases)}
    qt_idx = {q: i for i, q in enumerate(qts)}
    mid_idx = {m: i for i, m in enumerate(models)}
    Xt = np.array([[mid_idx[r["model_id"]], base_idx[r["baseline"]], qt_idx[r["question_type"]],
                    exe_idx[r["_exe"]], stk_idx[r["_stack"]], dom_idx[r["_dom"]],
                    r["event_index"] / max_ev,
                    1.0 if r.get("model_resident_before") else 0.0,
                    MODEL_ATTR.get(r["model_id"], (0, 0, 0))[0],
                    MODEL_ATTR.get(r["model_id"], (0, 0, 0))[1],
                    MODEL_ATTR.get(r["model_id"], (0, 0, 0))[2],
                    cross_ratio.get((s[pos - 1], r["model_id"]), 0.0) if
                    (pos := (seq_of[r["run_id"]].index(r["_fam"]) if r["_fam"] in seq_of[r["run_id"]]
                             else len(seq_of[r["run_id"]]) - 1)) >= 1 else 0.0]
                   for r in rows], dtype=np.float32)
    # 加最近族 one-hot(与 NN 同信息)
    last_fam_feat = np.zeros((len(rows), 6), dtype=np.float32)
    for i, r in enumerate(rows):
        s = seq_of[r["run_id"]]
        pos = s.index(r["_fam"]) if r["_fam"] in s else len(s) - 1
        if pos >= 1:
            last_fam_feat[i, fam_idx[s[pos - 1]]] = 1.0
    Xt = np.hstack([Xt, last_fam_feat])
    xgb_m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                              random_state=42).fit(Xt[itr], ytr)
    pb = xgb_m.predict_proba(Xt[ite])
    results["models"]["XGBoost+全关系"] = evaluate(yte_names, pb.tolist(), labels)
    print(f"  XGBoost(全关系) Top1={results['models']['XGBoost+全关系']['top1']:.4f}")

    # NN 族
    import torch
    import torch.nn as nn
    torch.manual_seed(42)
    Xtr, Xva, Xte = torch.from_numpy(X[itr]), torch.from_numpy(X[iva]), torch.from_numpy(X[ite])
    Str, Sva, Ste = (torch.from_numpy(S[itr]), torch.from_numpy(S[iva]), torch.from_numpy(S[ite]))
    yt, yv = torch.from_numpy(ytr), torch.from_numpy(yva)
    D = X.shape[1]

    def train_nn(model, epochs=60):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        best, best_state = -1, None
        for ep in range(epochs):
            model.train(); opt.zero_grad()
            loss = nn.functional.cross_entropy(model(Str, Xtr), yt)
            loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                acc = (model(Sva, Xva).argmax(1) == yv).float().mean().item()
            if acc > best:
                best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state); model.eval()
        return model

    def run(name, model):
        with torch.no_grad():
            pb = torch.softmax(model(Ste, Xte), dim=1).numpy()
        m = evaluate(yte_names, pb.tolist(), labels)
        results["models"][name] = m
        print(f"  {name:20s} Top1={m['top1']:.4f} Top3={m['top3']:.4f}")

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(D, 128), nn.ReLU(), nn.Dropout(0.2),
                                     nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 6))
        def forward(self, s, x):
            return self.net(x)

    class GRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(7, 16)
            self.rnn = nn.GRU(16, 64, batch_first=True)
            self.head = nn.Sequential(nn.Linear(64 + D, 64), nn.ReLU(), nn.Linear(64, 6))
        def forward(self, s, x):
            _, h = self.rnn(self.emb(s))
            return self.head(torch.cat([h[-1], x], dim=1))

    class BiLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(7, 16)
            self.rnn = nn.LSTM(16, 64, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(nn.Linear(128 + D, 64), nn.ReLU(), nn.Linear(64, 6))
        def forward(self, s, x):
            _, h = self.rnn(self.emb(s))
            hh = torch.cat([h[0][-2], h[0][-1]], dim=1)
            return self.head(torch.cat([hh, x], dim=1))

    for name, cls in [("MLP+从属层", MLP), ("GRU+从属层", GRU), ("BiLSTM+从属层", BiLSTM)]:
        run(name, train_nn(cls()))

    with open("results/processed/benchmark_behavior_attrib_20260806.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("写入 results/processed/benchmark_behavior_attrib_20260806.json")


if __name__ == "__main__":
    main()
