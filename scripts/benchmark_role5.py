#!/usr/bin/env python3
"""行为预测器对比(角色 5 类,基于 role_dataset_v0_1)

模型:B1 条件计数 / B2 LightGBM / B3 MLP / B4 GRU / B4b LSTM / B5 GNN
特征:role、model、baseline、position(归一化 event_index)、resident(冷热)、yolo_batch
split:role_dataset 自带 train/validation/test(视频级 240/30/30,seed=42)
指标:Top-1 / Top-3 / NLL / ECE
"""
import argparse
import json
import math
from collections import defaultdict, Counter
import numpy as np

ROLES = ["init", "plan", "execute", "aggregate", "terminate"]
TARGET_ROLES = ["plan", "execute", "aggregate", "terminate"]  # init 永不为目标


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def evaluate(y_true, probs, labels):
    idx = {l: i for i, l in enumerate(labels)}
    top1 = top3 = 0
    nll = 0.0
    confs, hits = [], []
    for t, p in zip(y_true, probs):
        order = sorted(range(len(p)), key=lambda i: -p[i])
        if labels[order[0]] == t:
            top1 += 1
        if t in {labels[i] for i in order[:3]}:
            top3 += 1
        nll += -math.log(max(p[idx.get(t, 0)], 1e-9))
        confs.append(max(p))
        hits.append(1 if labels[order[0]] == t else 0)
    n = len(y_true)
    # ECE(10 bins)
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        ms = [(c, h) for c, h in zip(confs, hits) if lo <= c < hi]
        if ms:
            ece += abs(sum(c for c, _ in ms) / len(ms) - sum(h for _, h in ms) / len(ms)) * len(ms) / n
    return {"top1": round(top1 / n, 4), "top3": round(top3 / n, 4),
            "nll": round(nll / n, 4), "ece": round(ece, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_jsonl(args.dataset)
    print(f"数据集行数: {len(rows)}")
    split_idx = {"train": [], "validation": [], "test": []}
    for i, r in enumerate(rows):
        split_idx[r["split"]].append(i)
    print(f"split: train={len(split_idx['train'])} val={len(split_idx['validation'])} test={len(split_idx['test'])}")

    results = {"dataset": args.dataset, "roles": ROLES, "models": {}}

    # 编码
    role_idx = {r: i for i, r in enumerate(ROLES)}
    tgt_idx = {r: i for i, r in enumerate(TARGET_ROLES)}
    models = sorted(set(r["model_id"] for r in rows))
    model_idx = {m: i for i, m in enumerate(models)}
    bases = sorted(set(r["baseline"] for r in rows))
    base_idx = {b: i for i, b in enumerate(bases)}
    max_ev = max(r["event_index"] for r in rows)

    X = np.array([[role_idx[r["role"]], model_idx[r["model_id"]], base_idx[r["baseline"]],
                   r["event_index"] / max_ev,
                   1.0 if r.get("model_resident_before") else 0.0,
                   1.0 if r.get("yolo_batch") else 0.0]
                  for r in rows], dtype=np.float32)
    y = np.array([tgt_idx[r["next_role"]] for r in rows], dtype=np.int64)

    Xtr, Xva, Xte = X[split_idx["train"]], X[split_idx["validation"]], X[split_idx["test"]]
    ytr, yva, yte = y[split_idx["train"]], y[split_idx["validation"]], y[split_idx["test"]]

    def metric(y_ev, probs, labels=None):
        labs = labels or TARGET_ROLES
        return evaluate([labs[int(v)] for v in y_ev], probs.tolist(), labs)

    # B1: 条件计数(role+model+baseline)
    ctx_counts = defaultdict(Counter)
    for i in split_idx["train"]:
        ctx = (int(X[i, 0]), int(X[i, 1]), int(X[i, 2]))
        ctx_counts[ctx][int(y[i])] += 1
    ctx_prior = {c: {l: v / sum(vals.values()) for l, v in vals.items()}
                 for c, vals in ctx_counts.items()}
    probs = []
    for i in split_idx["test"]:
        p = ctx_prior.get((int(X[i, 0]), int(X[i, 1]), int(X[i, 2])), {})
        probs.append([p.get(l, 1e-6) for l in range(4)])
    results["models"]["B1 条件计数"] = metric(yte, np.array(probs))
    print(f"B1 条件计数: {results['models']['B1 条件计数']}")

    # B2: LightGBM
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                           verbose=-1, random_state=42)
    m.fit(Xtr, ytr)
    b2_labels = [TARGET_ROLES[v] for v in m.classes_]
    results["models"]["B2 LightGBM"] = metric(yte, m.predict_proba(Xte), b2_labels)
    print(f"B2 LightGBM: {results['models']['B2 LightGBM']}")

    # B3: MLP
    import torch
    import torch.nn as nn
    torch.manual_seed(42)
    net = nn.Sequential(nn.Linear(6, 128), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 4))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr)
    xv = torch.from_numpy(Xva); yv = torch.from_numpy(yva)
    best_acc, best_state = -1, None
    for ep in range(80):
        net.train(); opt.zero_grad()
        loss = nn.functional.cross_entropy(net(xt), yt)
        loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            acc = (net(xv).argmax(1) == yv).float().mean().item()
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        pb = torch.softmax(net(torch.from_numpy(Xte)), dim=1).numpy()
    results["models"]["B3 MLP"] = metric(yte, pb)
    print(f"B3 MLP: {results['models']['B3 MLP']} (val_acc={best_acc:.3f})")

    # B4/B4b: GRU/LSTM(角色序列)
    seqs = [[] for _ in rows]
    for i, r in enumerate(rows):
        seqs[i] = r["role"]
    # 每 run 的完整角色序列
    by_run = defaultdict(list)
    for i, r in enumerate(rows):
        by_run[r["run_id"]].append(i)
    run_seqs = {}
    for rid, idxs in by_run.items():
        idxs_sorted = sorted(idxs, key=lambda i: rows[i]["event_index"])
        run_seqs[rid] = [rows[i]["role"] for i in idxs_sorted]
    # 每行:该行之前的角色序列(含当前)
    seq_feats = np.zeros((len(rows), 10), dtype=np.int64)
    for i, r in enumerate(rows):
        s = run_seqs[r["run_id"]]
        pos = r["event_index"]
        window = s[max(0, pos - 9): pos + 1]
        seq_feats[i, -len(window):] = [role_idx[x] for x in window]
    S = seq_feats
    Str, Sva, Ste = S[split_idx["train"]], S[split_idx["validation"]], S[split_idx["test"]]

    def run_rnn(kind):
        torch.manual_seed(42)
        class RNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(5, 16)
                self.rnn = nn.GRU(16, 64, batch_first=True) if kind == "gru" else nn.LSTM(16, 64, batch_first=True)
                self.head = nn.Sequential(nn.Linear(64 + 6, 64), nn.ReLU(), nn.Linear(64, 4))
            def forward(self, s, x):
                e = self.emb(s)
                _, h = self.rnn(e)
                if isinstance(h, tuple):
                    h = h[0]
                return self.head(torch.cat([h[-1], x], dim=1))
        model = RNN()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        st = torch.from_numpy(Str); xv2 = torch.from_numpy(Xva)
        sv = torch.from_numpy(Sva)
        best_acc, best_state = -1, None
        for ep in range(40):
            model.train(); opt.zero_grad()
            loss = nn.functional.cross_entropy(model(st, xt), yt)
            loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                acc = (model(sv, xv2).argmax(1) == yv).float().mean().item()
            if acc > best_acc:
                best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            pb = torch.softmax(model(torch.from_numpy(Ste), torch.from_numpy(Xte)), dim=1).numpy()
        return metric(yte, pb), best_acc

    for kind in ("gru", "lstm"):
        mtr, vacc = run_rnn(kind)
        results["models"][f"B4 {kind.upper()}"] = mtr
        print(f"B4 {kind.upper()}: {mtr} (val_acc={vacc:.3f})")

    # B5: GNN(角色转移邻接传播)
    adj = np.zeros((5, 5))
    for i in split_idx["train"]:
        adj[int(X[i, 0]), int(y[i])] += 1
    adj_norm = adj / adj.sum(axis=1, keepdims=True).clip(min=1)
    torch.manual_seed(42)
    class GNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.node_emb = nn.Parameter(torch.randn(5, 16) * 0.1)
            self.prop_self = nn.Linear(16, 16)
            self.prop_neigh = nn.Linear(16, 16)
            self.head = nn.Sequential(nn.Linear(16 + 6, 64), nn.ReLU(), nn.Linear(64, 4))
        def forward(self, x):
            emb = self.node_emb
            A = torch.from_numpy(adj_norm).float()
            for _ in range(2):
                emb = torch.tanh(self.prop_self(emb) + self.prop_neigh(A @ emb))
            cur = x[:, 0].long()
            node_state = emb[cur]
            return self.head(torch.cat([node_state, x], dim=1))
    gnn = GNN()
    opt = torch.optim.Adam(gnn.parameters(), lr=1e-3)
    best_acc, best_state = -1, None
    for ep in range(40):
        gnn.train(); opt.zero_grad()
        loss = nn.functional.cross_entropy(gnn(xt), yt)
        loss.backward(); opt.step()
        gnn.eval()
        with torch.no_grad():
            acc = (gnn(xv).argmax(1) == yv).float().mean().item()
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in gnn.state_dict().items()}
    gnn.load_state_dict(best_state); gnn.eval()
    with torch.no_grad():
        pb = torch.softmax(gnn(torch.from_numpy(Xte)), dim=1).numpy()
    results["models"]["B5 GNN"] = metric(yte, pb)
    print(f"B5 GNN: {results['models']['B5 GNN']} (val_acc={best_acc:.3f})")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已写入: {args.out}")


if __name__ == "__main__":
    main()
