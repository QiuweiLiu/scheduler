#!/usr/bin/env python3
"""行为侧 NN 特征数值化对比:one-hot + z-score vs 标签编码(动作族预测)

口径与 benchmark_action_family 一致(execute 子集,当前行动作族标签)
验证:NN 在 one-hot+标准化特征下是否显著提升
"""
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


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def fam_of(raw):
    return FAMILY.get(raw, "other")


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
    print(f"execute 事件: {len(rows)}")

    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "validation"]
    te = [r for r in rows if r["split"] == "test"]

    fam_idx = {f: i for i, f in enumerate(FAMS)}
    labels = FAMS
    models = sorted(set(r["model_id"] for r in rows))
    bases = sorted(set(r["baseline"] for r in rows))
    qts = sorted(set(r["question_type"] for r in rows))
    max_ev = max(r["event_index"] for r in rows)

    # one-hot(模型/基线/question_type)+ 数值(位置/冷热/yolo,后 z-score)+ 最近族 one-hot
    def onehot(vals, size):
        v = np.zeros(size, dtype=np.float32)
        v[vals] = 1.0
        return v

    def feat(r, last_fam):
        x = np.concatenate([
            onehot([models.index(r["model_id"])], len(models)),
            onehot([bases.index(r["baseline"])], len(bases)),
            onehot([qts.index(r["question_type"])], len(qts)),
            np.array([r["event_index"] / max_ev,
                      1.0 if r.get("model_resident_before") else 0.0,
                      1.0 if r.get("yolo_batch") else 0.0], dtype=np.float32),
            onehot([fam_idx[last_fam]], 6) if last_fam else np.zeros(6, dtype=np.float32),
        ])
        return x

    # 序列(前 5 步 execute 族,含当前)
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run_id"]].append(r)
    for rs in by_run.values():
        rs.sort(key=lambda x: x["event_index"])
    seq_of = {}
    for rid, rs in by_run.items():
        seq_of[rid] = [r["_fam"] for r in rs]

    W = 5
    feats = []
    for r in rows:
        s = seq_of[r["run_id"]]
        pos = s.index(r["_fam"]) if r["_fam"] in s else len(s) - 1
        last = s[pos - 1] if pos >= 1 else None
        feats.append(feat(r, last))
    X = np.stack(feats)
    # z-score(数值列:位置/冷热/yolo 索引 = len(m)+len(b)+len(q) 起 3 个)
    num_start = len(models) + len(bases) + len(qts)
    num_end = num_start + 3
    mu = X[0:len(tr), num_start:num_end].mean(0)
    sd = X[0:len(tr), num_start:num_end].std(0).clip(min=1e-6)
    X[:, num_start:num_end] = (X[:, num_start:num_end] - mu) / sd

    y = np.array([fam_idx[r["_fam"]] for r in rows], dtype=np.int64)
    print(f"one-hot+std 特征维度: {X.shape[1]}")
    itr, iva, ite = [rows.index(r) for r in tr], [rows.index(r) for r in va], [rows.index(r) for r in te]
    ytr, yva, yte = y[itr], y[iva], y[ite]
    yte_names = [r["_fam"] for r in te]

    def seq_win(r):
        s = seq_of[r["run_id"]]
        pos = s.index(r["_fam"]) if r["_fam"] in s else len(s) - 1
        win = [fam_idx[x] for x in s[max(0, pos + 1 - W):pos]]
        return win + [6] * (W - len(win))

    S = np.array([seq_win(r) for r in rows], dtype=np.int64)

    import torch
    import torch.nn as nn
    torch.manual_seed(42)
    Xtr, Xva, Xte = torch.from_numpy(X[itr]), torch.from_numpy(X[iva]), torch.from_numpy(X[ite])
    Str, Sva, Ste = (torch.from_numpy(S[itr]), torch.from_numpy(S[iva]), torch.from_numpy(S[ite]))
    yt, yv = torch.from_numpy(ytr), torch.from_numpy(yva)

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
        print(f"  {name:24s} Top1={m['top1']:.4f} Top3={m['top3']:.4f}")
        return m

    results = {}
    D = X.shape[1]

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(D, 128), nn.ReLU(), nn.Dropout(0.2),
                                     nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 6))
        def forward(self, s, x):
            return self.net(x)

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(7, 16)
            self.conv = nn.Conv1d(16, 64, 3, padding=1)
            self.head = nn.Sequential(nn.Linear(64 + D, 64), nn.ReLU(), nn.Linear(64, 6))
        def forward(self, s, x):
            e = self.emb(s).transpose(1, 2)
            c = torch.relu(self.conv(e)).mean(2)
            return self.head(torch.cat([c, x], dim=1))

    class GRU(nn.Module):
        def __init__(self, onehot_enc):
            super().__init__()
            self.oh = onehot_enc
            self.emb = nn.Embedding(7, 16)
            self.rnn = nn.GRU(16 if not onehot_enc else 6, 64, batch_first=True)
            self.head = nn.Sequential(nn.Linear(64 + D, 64), nn.ReLU(), nn.Linear(64, 6))
        def forward(self, s, x):
            if self.oh:
                mask = (s != 6).unsqueeze(2).float()
                e = torch.zeros(*s.shape, 6).scatter_(2, s.clamp(0, 5).unsqueeze(2), 1.0) * mask
            else:
                e = self.emb(s)
            _, h = self.rnn(e)
            return self.head(torch.cat([h[-1], x], dim=1))

    class BiLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(7, 16)
            self.rnn = nn.LSTM(16, 64, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(nn.Linear(128 + D, 64), nn.ReLU(), nn.Linear(64, 6))
        def forward(self, s, x):
            e = self.emb(s)
            _, h = self.rnn(e)
            hh = torch.cat([h[0][-2], h[0][-1]], dim=1)
            return self.head(torch.cat([hh, x], dim=1))

    class TF(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(7, 32)
            self.pos = nn.Parameter(torch.randn(1, W, 32) * 0.1)
            self.attn = nn.MultiheadAttention(32, 4, batch_first=True)
            self.head = nn.Sequential(nn.Linear(32 + D, 64), nn.ReLU(), nn.Linear(64, 6))
        def forward(self, s, x):
            e = self.emb(s) + self.pos
            a, _ = self.attn(e, e, e)
            return self.head(torch.cat([a.mean(1), x], dim=1))

    class GNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.node = nn.Parameter(torch.randn(6, 16) * 0.1)
            self.p1, self.p2 = nn.Linear(16, 16), nn.Linear(16, 16)
            self.head = nn.Sequential(nn.Linear(16 + D, 64), nn.ReLU(), nn.Linear(64, 6))
        def forward(self, s, x):
            emb = self.node
            A = torch.from_numpy(np.eye(6) * 0.5 + 0.05).float()
            for _ in range(2):
                emb = torch.tanh(self.p1(emb) + self.p2(A @ emb))
            return self.head(torch.cat([emb[s[:, -1].clamp(0, 5)], x], dim=1))

    for name, cls in [("MLP", MLP), ("CNN", CNN), ("GRU_emb", lambda: GRU(False)),
                      ("GRU_onehot", lambda: GRU(True)), ("BiLSTM", BiLSTM),
                      ("Transformer", TF), ("GNN", GNN)]:
        m = train_nn(cls())
        results[name] = run(name, m)

    with open("results/processed/benchmark_behavior_nnfeat_20260806.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("写入 results/processed/benchmark_behavior_nnfeat_20260806.json")


if __name__ == "__main__":
    main()
