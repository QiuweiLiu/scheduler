#!/usr/bin/env python3
"""动作族预测器统一评估(对应远端 docs/action_family_predictor_20260806.md)

14 个预测器:全局先验/条件计数/n-gram(1/2/3)/逻辑回归/朴素贝叶斯/随机森林/
LightGBM/XGBoost(或GradientBoosting)/MLP/CNN-1D/GRU(one-hot vs embedding)/
BiLSTM/Transformer-lite/GNN/KNN
数据:role_dataset_v0_2 execute 子集(含 expansion,240/30/30)
评估:Top-1 / Top-3 / 执行者路由正确率
"""
import argparse
import json
import math
from collections import defaultdict, Counter
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
EXECUTOR = {"select_frames": "TOOL", "visual_qa": "LLM", "temporal_ops": "TOOL",
            "summarize": "LLM", "detect": "YOLO", "other": "?"}
LLM_SET = {"Qwen3-VL-8B-Instruct", "Qwen2.5-VL-3B-Instruct"}


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def real_executor(model_id):
    if model_id in LLM_SET:
        return "LLM"
    if "yolo" in str(model_id):
        return "YOLO"
    return "TOOL"


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


def executor_route_acc(y_true, probs, labels):
    hit = 0
    for t, p in zip(y_true, probs):
        pred = labels[int(np.argmax(p))]
        if EXECUTOR.get(pred) == EXECUTOR.get(t):
            hit += 1
    return round(hit / len(y_true), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [r for r in load_jsonl(args.dataset) if r["role"] == "execute"]
    for r in rows:
        r["_fam"] = FAMILY.get(r.get("raw_action"), "other")
    print(f"execute 事件: {len(rows)}")
    print("动作族分布:", dict(Counter(r["_fam"] for r in rows)))

    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "validation"]
    te = [r for r in rows if r["split"] == "test"]
    print(f"split: train={len(tr)} val={len(va)} test={len(te)}")
    labels = FAMS

    # 序列:每 run 内 execute 事件按序的动作族
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run_id"]].append(r)
    for rid, rs in by_run.items():
        rs.sort(key=lambda x: x["event_index"])
    seq_of = {}
    for rid, rs in by_run.items():
        seq_of[rid] = [r["_fam"] for r in rs]

    fam_idx = {f: i for i, f in enumerate(FAMS)}
    models = sorted(set(r["model_id"] for r in rows))
    model_idx = {m: i for i, m in enumerate(models)}
    bases = sorted(set(r["baseline"] for r in rows))
    base_idx = {b: i for i, b in enumerate(bases)}
    qts = sorted(set(r["question_type"] for r in rows))
    qt_idx = {q: i for i, q in enumerate(qts)}
    max_ev = max(r["event_index"] for r in rows)
    W = 5  # 序列窗口

    def flat(r, seq):
        x = [model_idx[r["model_id"]], base_idx[r["baseline"]],
             r["event_index"] / max_ev,
             1.0 if r.get("model_resident_before") else 0.0,
             1.0 if r.get("yolo_batch") else 0.0,
             qt_idx[r["question_type"]]]
        # 最近动作族 one-hot(序列最后一步)
        last = seq[-2] if len(seq) >= 2 else None
        for f in FAMS:
            x.append(1.0 if last == f else 0.0)
        return np.array(x, dtype=np.float32)

    Xtr = np.stack([flat(r, seq_of[r["run_id"]]) for r in tr])
    Xva = np.stack([flat(r, seq_of[r["run_id"]]) for r in va])
    Xte = np.stack([flat(r, seq_of[r["run_id"]]) for r in te])
    ytr = np.array([fam_idx[r["_fam"]] for r in tr])
    yva = np.array([fam_idx[r["_fam"]] for r in va])
    yte = np.array([fam_idx[r["_fam"]] for r in te])
    yte_names = [r["_fam"] for r in te]

    def seq_win(r, n):
        s = seq_of[r["run_id"]]
        idx = r["event_index"]
        # execute 子集内位置
        pos = [i for i, x in enumerate(s) if x == r["_fam"]]
        # 简化:用 execute 子集顺序(按 run 内 execute 事件序)
        order = [rr for rr in by_run[r["run_id"]]]
        cur_i = order.index(r) if r in order else len(order) - 1
        win = [order[k]["_fam"] for k in range(max(0, cur_i - n), cur_i)]
        return [fam_idx[f] for f in win]

    def pad_win(win_ids, n):
        return win_ids + [len(FAMS)] * (n - len(win_ids))

    def run_eval(name, probs):
        m = evaluate(yte_names, probs.tolist(), labels)
        m["exec_route"] = executor_route_acc(yte_names, probs, labels)
        print(f"  {name:34s} Top1={m['top1']:.4f} Top3={m['top3']:.4f} exec_route={m['exec_route']:.4f}")
        return m

    results = {"dataset": args.dataset, "n": len(rows), "models": {}}

    # 1. 全局先验
    prior = Counter(r["_fam"] for r in tr)
    total = sum(prior.values())
    p0 = np.array([prior.get(f, 0) / total for f in FAMS])
    results["models"]["01_全局先验"] = run_eval("01_全局先验", np.tile(p0, (len(te), 1)))

    # 2. 条件计数(位置+baseline+model)
    ctx = defaultdict(Counter)
    for r in tr:
        ctx[(r["baseline"], min(r["event_index"], 9) // 3, r["model_id"])][r["_fam"]] += 1
    probs = []
    for r in te:
        c = ctx.get((r["baseline"], min(r["event_index"], 9) // 3, r["model_id"]), Counter())
        t = sum(c.values())
        probs.append([c.get(f, 0) / t if t else 1 / 6 for f in FAMS])
    results["models"]["02_条件计数"] = run_eval("02_条件计数", np.array(probs))

    # 3. n-gram(窗口 1/2/3)
    for n in (1, 2, 3):
        ng = defaultdict(Counter)
        for r in tr:
            w = tuple(pad_win(seq_win(r, n), n))
            ng[w][r["_fam"]] += 1
        probs = []
        for r in te:
            w = tuple(pad_win(seq_win(r, n), n))
            c = ng.get(w, Counter())
            t = sum(c.values())
            probs.append([c.get(f, 0) / t if t else 1 / 6 for f in FAMS])
        results["models"][f"03_n-gram_w{n}"] = run_eval(f"03_n-gram_w{n}", np.array(probs))

    # 4-8. ML 族
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier

    lr = LogisticRegression(max_iter=500).fit(Xtr, ytr)
    results["models"]["04_逻辑回归"] = run_eval("04_逻辑回归", lr.predict_proba(Xte))

    nb = GaussianNB().fit(Xtr, ytr)
    results["models"]["05_朴素贝叶斯"] = run_eval("05_朴素贝叶斯", nb.predict_proba(Xte))

    rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xtr, ytr)
    results["models"]["06_随机森林"] = run_eval("06_随机森林", rf.predict_proba(Xte))

    import lightgbm as lgb
    lm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                            verbose=-1, random_state=42).fit(Xtr, ytr)
    results["models"]["07_LightGBM"] = run_eval("07_LightGBM", lm.predict_proba(Xte))

    try:
        import xgboost as xgb
        xm = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                               random_state=42).fit(Xtr, ytr)
        results["models"]["08_XGBoost"] = run_eval("08_XGBoost", xm.predict_proba(Xte))
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier
        gm = GradientBoostingClassifier(n_estimators=200, random_state=42).fit(Xtr, ytr)
        results["models"]["08_GradientBoosting"] = run_eval("08_GradientBoosting(fallback)", gm.predict_proba(Xte))

    # 9-13. NN 族
    import torch
    import torch.nn as nn
    torch.manual_seed(42)

    def train_nn(model, x_tr, s_tr, y_tr, x_va, s_va, y_va, epochs=60):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        best_acc, best_state = -1, None
        for ep in range(epochs):
            model.train(); opt.zero_grad()
            loss = nn.functional.cross_entropy(model(s_tr, x_tr), y_tr)
            loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                acc = (model(s_va, x_va).argmax(1) == y_va).float().mean().item()
            if acc > best_acc:
                best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        return model

    def seq_tensor(rows_, n=W):
        arr = np.zeros((len(rows_), n), dtype=np.int64)
        for i, r in enumerate(rows_):
            w = pad_win(seq_win(r, n), n)
            arr[i] = w
        return torch.from_numpy(arr)

    Str, Sva, Ste = seq_tensor(tr), seq_tensor(va), seq_tensor(te)
    xt, xv, xte = (torch.from_numpy(Xtr), torch.from_numpy(Xva), torch.from_numpy(Xte))
    yt, yv = torch.from_numpy(ytr), torch.from_numpy(yva)

    # 9. MLP
    class MLPWrap(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
        def forward(self, s, x):
            return self.net(x)
    mlp = MLPWrap(nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(), nn.Dropout(0.2),
                                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 6)))
    mlp = train_nn(mlp, xt, Str, yt, xv, Sva, yv)
    with torch.no_grad():
        pb = torch.softmax(mlp(Ste, xte), dim=1).numpy()
    results["models"]["09_MLP"] = run_eval("09_MLP", pb)

    # 10. CNN-1D
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(7, 16)
            self.conv = nn.Conv1d(16, 64, 3, padding=1)
            self.head = nn.Sequential(nn.Linear(64 + Xtr.shape[1], 64), nn.ReLU(), nn.Linear(64, 6))
        def forward(self, s, x):
            e = self.emb(s).transpose(1, 2)
            c = torch.relu(self.conv(e)).mean(dim=2)
            return self.head(torch.cat([c, x], dim=1))
    cnn = train_nn(CNN(), xt, Str, yt, xv, Sva, yv)
    with torch.no_grad():
        pb = torch.softmax(cnn(Ste, xte), dim=1).numpy()
    results["models"]["10_CNN-1D"] = run_eval("10_CNN-1D", pb)

    # 11. GRU(one-hot vs embedding)
    for enc in ("embedding", "onehot"):
        class GRU(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(7, 16) if enc == "embedding" else nn.Embedding(7, 6)
                self.rnn = nn.GRU(16 if enc == "embedding" else 6, 64, batch_first=True)
                self.head = nn.Sequential(nn.Linear(64 + Xtr.shape[1], 64), nn.ReLU(), nn.Linear(64, 6))
            def forward(self, s, x):
                if enc == "onehot":
                    mask = (s != 6).unsqueeze(2).float()
                    e = torch.zeros(*s.shape, 6).scatter_(2, s.clamp(0, 5).unsqueeze(2), 1.0) * mask
                else:
                    e = self.emb(s)
                _, h = self.rnn(e)
                return self.head(torch.cat([h[-1], x], dim=1))
        gru = train_nn(GRU(), xt, Str, yt, xv, Sva, yv)
        with torch.no_grad():
            pb = torch.softmax(gru(Ste, xte), dim=1).numpy()
        results["models"][f"11_GRU_{enc}"] = run_eval(f"11_GRU_{enc}", pb)

    # 12. BiLSTM
    class BiLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(7, 16)
            self.rnn = nn.LSTM(16, 64, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(nn.Linear(128 + Xtr.shape[1], 64), nn.ReLU(), nn.Linear(64, 6))
        def forward(self, s, x):
            e = self.emb(s)
            _, h = self.rnn(e)
            hh = torch.cat([h[0][-2], h[0][-1]], dim=1)
            return self.head(torch.cat([hh, x], dim=1))
    bl = train_nn(BiLSTM(), xt, Str, yt, xv, Sva, yv)
    with torch.no_grad():
        pb = torch.softmax(bl(Ste, xte), dim=1).numpy()
    results["models"]["12_BiLSTM"] = run_eval("12_BiLSTM", pb)

    # 13. Transformer-lite
    class TF(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(7, 32)
            self.pos = nn.Parameter(torch.randn(1, W, 32) * 0.1)
            self.attn = nn.MultiheadAttention(32, 4, batch_first=True)
            self.head = nn.Sequential(nn.Linear(32 + Xtr.shape[1], 64), nn.ReLU(), nn.Linear(64, 6))
        def forward(self, s, x):
            e = self.emb(s) + self.pos
            a, _ = self.attn(e, e, e)
            h = a.mean(dim=1)
            return self.head(torch.cat([h, x], dim=1))
    tf = train_nn(TF(), xt, Str, yt, xv, Sva, yv)
    with torch.no_grad():
        pb = torch.softmax(tf(Ste, xte), dim=1).numpy()
    results["models"]["13_Transformer-lite"] = run_eval("13_Transformer-lite", pb)

    # 14. GNN(转移邻接)
    adj = np.zeros((6, 6))
    for r in tr:
        w = seq_win(r, 2)
        if len(w) >= 2:
            adj[w[-2], w[-1]] += 1
    adj_norm = adj / adj.sum(axis=1, keepdims=True).clip(min=1)
    class GNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.node_emb = nn.Parameter(torch.randn(6, 16) * 0.1)
            self.p1, self.p2 = nn.Linear(16, 16), nn.Linear(16, 16)
            self.head = nn.Sequential(nn.Linear(16 + Xtr.shape[1], 64), nn.ReLU(), nn.Linear(64, 6))
        def forward(self, s, x):
            emb = self.node_emb
            A = torch.from_numpy(adj_norm).float()
            for _ in range(2):
                emb = torch.tanh(self.p1(emb) + self.p2(A @ emb))
            cur = s[:, -1].clamp(0, 5)
            ns = emb[cur]
            return self.head(torch.cat([ns, x], dim=1))
    gn = train_nn(GNN(), xt, Str, yt, xv, Sva, yv)
    with torch.no_grad():
        pb = torch.softmax(gn(Ste, xte), dim=1).numpy()
    results["models"]["14_GNN"] = run_eval("14_GNN", pb)

    # 15. KNN 相似 trace(LCS)
    from difflib import SequenceMatcher
    tr_wins = [(r, pad_win(seq_win(r, W), W)) for r in tr]
    probs = []
    for r in te:
        w = pad_win(seq_win(r, W), W)
        sims = []
        for tr_r, tw in tr_wins:
            sm = SequenceMatcher(None, w, tw).ratio()
            sims.append((sm, tr_r["_fam"]))
        sims.sort(key=lambda x: -x[0])
        top5 = [f for _, f in sims[:5]]
        c = Counter(top5)
        t = len(top5)
        probs.append([c.get(f, 0) / t for f in FAMS])
    results["models"]["15_KNN"] = run_eval("15_KNN", np.array(probs))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已写入: {args.out}")


if __name__ == "__main__":
    main()
