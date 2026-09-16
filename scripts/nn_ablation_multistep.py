#!/usr/bin/env python3
"""execute 轴族多步 K=3:NN 容量消融 + 模型族对比(2026-08-07)

- 统一数据管线(与 multistep_experiment.py 一致):execute 轴,hist 3×7 one-hot + 8 维数值
- 模型族:GRU(h16/32/64/128)、BiLSTM、Transformer、CNN、MLP(无序列对照)
- 评估:per-step k=1..3 Top-1 + W=3 全命中(test 261)
- 对照:直接 XGB k=1..3(0.7560/0.7557/0.7356)、GRU64(0.7280/0.7433/0.7778)
输出:results/processed/nn_ablation_multistep_20260807.json
"""
import json
import os
import statistics as st
import sys
from collections import defaultdict
import numpy as np

D = "/root/autodl-tmp/scheduler/"

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
LLM_SET = {"Qwen3-VL-8B-Instruct", "Qwen2.5-VL-3B-Instruct", "Qwen3-4B"}
MODEL_ATTR = {
    "Qwen3-VL-8B-Instruct": (8.1, 19.0, 1.0),
    "Qwen3-4B": (3.9, 8.0, 0.55),
    "Qwen2.5-VL-3B-Instruct": (3.8, 7.3, 0.5),
    "cpu-metadata-adapter-v1": (0.0, 0.0, 0.9),
    "yolo11x.pt": (0.02, 0.4, 0.8),
    "finish_argument": (0.0, 0.0, 0.9),
    "stack_a_qwen3_vl8b": (8.1, 19.0, 1.0),
    "stack_b_qwen3_4b_qwen25vl3b_yolo11x": (3.9, 8.0, 0.55),
}
K = 3


def load_jsonl(p):
    return [json.loads(l) for l in open(D + p)]


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


def onehot(vals, size):
    v = np.zeros(size, dtype=np.float32)
    v[vals] = 1.0
    return v


def build_data():
    rows = load_jsonl("results/processed/role_dataset_v0_3.jsonl")
    prefixes = load_jsonl("results/processed/neural_prefix_dataset_v0_9_options_sourcefix_20260805/prefixes_visual_context.jsonl")
    for r in rows:
        r["_exe"] = executor_of(r["model_id"])
        r["_stack"] = stack_of(r["model_id"])
    vid_meta = {}
    for p in prefixes:
        ts = p.get("task_structure") or {}
        vid_meta[p["video_id"]] = {"domain": (ts.get("domain") if isinstance(ts, dict) else None) or "unknown"}
    for r in rows:
        r["_dom"] = vid_meta.get(r["video_id"], {}).get("domain", "unknown")

    tr = [r for r in rows if r["split"] == "train"]
    fam_idx = {f: i for i, f in enumerate(FAMS)}
    models = sorted(set(r["model_id"] for r in rows))
    model_idx = {m: i for i, m in enumerate(models)}
    bases = sorted(set(r["baseline"] for r in rows))
    base_idx = {b: i for i, b in enumerate(bases)}
    qts = sorted(set(r["question_type"] for r in rows))
    qt_idx = {q: i for i, q in enumerate(qts)}
    exes = sorted(set(r["_exe"] for r in rows))
    exe_idx = {e: i for i, e in enumerate(exes)}
    stacks = sorted(set(r["_stack"] for r in rows))
    stk_idx = {s: i for i, s in enumerate(stacks)}
    doms = sorted(set(r["_dom"] for r in rows))
    dom_idx = {d: i for i, d in enumerate(doms)}

    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run_id"]].append(r)
    for rs in by_run.values():
        rs.sort(key=lambda x: x["event_index"])

    max_rt = max((float(r.get("runtime_ms") or 0) for r in rows), default=1.0)
    max_pk = max((float(r.get("peak_allocated_mb") or 0) for r in rows), default=1.0)
    max_ld = max((float(r.get("load_ms") or 0) for r in rows), default=1.0)

    gmed2 = defaultdict(list)
    for rid, rs in by_run.items():
        last_fam = None
        for r in rs:
            if r["split"] != "train":
                last_fam = None
                continue
            if (r.get("runtime_ms") or 0) > 0 and last_fam:
                gmed2[(last_fam, r["model_id"])].append(r["runtime_ms"])
            if r["role"] == "execute":
                last_fam = fam_of(r["raw_action"])
    cross2 = {k: st.median(v) for k, v in gmed2.items()}
    g = st.median([r["runtime_ms"] for r in tr if (r.get("runtime_ms") or 0) > 0] or [0])

    def ctx_parts(r, last_fam):
        attr = MODEL_ATTR.get(r["model_id"], (0, 0, 0))
        cross = cross2.get((last_fam, r["model_id"]), g) if last_fam else g
        oh = np.concatenate([
            onehot([model_idx[r["model_id"]]], len(models)),
            onehot([base_idx[r["baseline"]]], len(bases)),
            onehot([qt_idx[r["question_type"]]], len(qts)),
            onehot([exe_idx[r["_exe"]]], len(exes)),
            onehot([stk_idx[r["_stack"]]], len(stacks)),
            onehot([dom_idx[r["_dom"]]], len(doms)),
        ])
        num = np.array([float(attr[0]), float(attr[1]), float(attr[2]), cross],
                       dtype=np.float32)
        return oh, num

    ex_rows, ex_one, ex_num = [], [], []
    for rid, rs in by_run.items():
        exes_seq = [r for r in rs if r["role"] == "execute"]
        for q, r in enumerate(exes_seq):
            oh, num = ctx_parts(r, fam_of(exes_seq[q - 1]["raw_action"]) if q >= 1 else None)
            hist = [fam_idx[fam_of(x["raw_action"])] for x in exes_seq[max(0, q - 3):q]]
            hist = (hist + [6] * 3)[-3:]
            prev_rt = float(exes_seq[q - 1].get("runtime_ms") or 0) if q >= 1 else 0.0
            prev_pk = float(exes_seq[q - 1].get("peak_allocated_mb") or 0) if q >= 1 else 0.0
            prev_ld = float(exes_seq[q - 1].get("load_ms") or 0) if q >= 1 else 0.0
            hist_oh = np.concatenate([
                onehot([h], 7) if h != 6 else np.zeros(7, dtype=np.float32)
                for h in hist])
            ex_one.append(np.concatenate([oh, hist_oh]))
            ex_num.append(np.concatenate([
                num,
                np.array([q / max(1, len(exes_seq)), prev_rt / max_rt,
                          prev_pk / max_pk, prev_ld / max_ld], dtype=np.float32)]))
            ex_rows.append(r)
    ex_one = np.stack(ex_one)
    ex_num = np.stack(ex_num)
    ex_f = np.hstack([ex_one, ex_num])
    one_end = ex_one.shape[1]
    tr_ex = [i for i, r in enumerate(ex_rows) if r["split"] == "train"]
    mu = ex_f[tr_ex, one_end:].mean(0)
    sd = ex_f[tr_ex, one_end:].std(0).clip(min=1e-6)
    ex_f[:, one_end:] = (ex_f[:, one_end:] - mu) / sd

    fam_k = np.full((len(ex_rows), K + 1), -1, dtype=np.int64)
    ex_idx_map = {id(r): i for i, r in enumerate(ex_rows)}
    for rid, rs in by_run.items():
        exes_seq = [r for r in rs if r["role"] == "execute"]
        for q, r in enumerate(exes_seq):
            i = ex_idx_map[id(r)]
            for k in range(1, K + 1):
                if q + k < len(exes_seq):
                    fam_k[i, k] = fam_idx[fam_of(exes_seq[q + k]["raw_action"])]
    split_of = np.array([r["split"] for r in ex_rows])
    return ex_f, fam_k, split_of, one_end


def main():
    ex_f, fam_k, split_of, one_end = build_data()
    itrx = np.where(split_of == "train")[0]
    itev = np.where(split_of == "validation")[0]
    itex = np.where(split_of == "test")[0]
    g_tr = itrx[fam_k[itrx, K] >= 0]
    g_va = itev[fam_k[itev, K] >= 0]
    g_te = itex[fam_k[itex, K] >= 0]
    DIM = ex_f.shape[1] - one_end

    import torch
    import torch.nn as nn
    torch.manual_seed(42)
    torch.set_num_threads(8)

    x_tr = torch.from_numpy(ex_f[g_tr][:, one_end:]); h_tr = torch.from_numpy(ex_f[g_tr][:, one_end - 21:one_end])
    y_tr = torch.from_numpy(fam_k[g_tr, 1:4])
    x_va = torch.from_numpy(ex_f[g_va][:, one_end:]); h_va = torch.from_numpy(ex_f[g_va][:, one_end - 21:one_end])
    y_va = torch.from_numpy(fam_k[g_va, 1:4])
    x_te = torch.from_numpy(ex_f[g_te][:, one_end:]); h_te = torch.from_numpy(ex_f[g_te][:, one_end - 21:one_end])

    results = {"g_tr": int(len(g_tr)), "g_te": int(len(g_te)), "models": {}}
    OUT = D + "results/processed/nn_ablation_multistep_20260807.json"
    if os.path.exists(OUT):
        try:
            with open(OUT) as f:
                results = json.load(f)
        except Exception:
            pass

    def train_eval(name, net, epochs=150, lr=1e-3):
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        best, best_state = -1, None
        wts = None
        if weighted:
            cnt = np.bincount(y_tr.numpy().reshape(-1), minlength=6)
            cnt = cnt.astype(np.float64) + 1.0
            wts = torch.from_numpy((cnt.sum() / cnt)).float()
        crit = lambda o, y: nn.functional.cross_entropy(
            o.reshape(-1, 6), y.reshape(-1), weight=wts)
        for ep in range(epochs):
            net.train(); opt.zero_grad()
            loss = crit(net(h_tr, x_tr), y_tr)
            loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                acc = (net(h_va, x_va).argmax(2) == y_va).float().mean().item()
            if acc > best:
                best, best_state = acc, {k_: v.clone() for k_, v in net.state_dict().items()}
        net.load_state_dict(best_state); net.eval()
        with torch.no_grad():
            out = net(h_te, x_te).numpy()
        ms = {"val_acc": round(best, 4),
              "params": sum(p.numel() for p in net.parameters())}
        for k in range(1, K + 1):
            t1 = round(float((out[:, k - 1].argmax(1) == fam_k[g_te, k]).mean()), 4)
            ms[f"k{k}"] = t1
            print(f"  {name:22s} k={k}: {t1:.4f}", end="  ", flush=True)
        hits = sum(out[:, k - 1].argmax(1) == fam_k[g_te, k] for k in (1, 2, 3))
        ms["win3"] = round(float((hits == 3).mean()), 4)
        print(f"| W3={ms['win3']:.4f} val={best:.4f} params={ms['params']}", flush=True)
        results["models"][name + ("_w" if weighted else "")] = ms
        with open(OUT, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    class GRU(nn.Module):
        def __init__(self, hid):
            super().__init__()
            self.emb = nn.Linear(7, 16)
            self.rnn = nn.GRU(16, hid, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hid + DIM, 128), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU())
            self.outs = nn.ModuleList([nn.Linear(64, 6) for _ in range(3)])

        def forward(self, h, x):
            e = self.emb(h.reshape(-1, 3, 7))
            _, hh = self.rnn(e)
            z = self.head(torch.cat([hh[-1], x], dim=1))
            return torch.stack([o(z) for o in self.outs], 1)

    class BiLSTM(nn.Module):
        def __init__(self, hid=64):
            super().__init__()
            self.emb = nn.Linear(7, 16)
            self.rnn = nn.LSTM(16, hid, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(nn.Linear(2 * hid + DIM, 128), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU())
            self.outs = nn.ModuleList([nn.Linear(64, 6) for _ in range(3)])

        def forward(self, h, x):
            e = self.emb(h.reshape(-1, 3, 7))
            _, (hh, _) = self.rnn(e)
            z = self.head(torch.cat([hh[-2], hh[-1], x], dim=1))
            return torch.stack([o(z) for o in self.outs], 1)

    class Transformer(nn.Module):
        def __init__(self, d=32, heads=4, layers=2):
            super().__init__()
            self.proj = nn.Linear(7, d)
            self.pos = nn.Parameter(torch.zeros(1, 3, d))
            enc = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=128,
                                             dropout=0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(enc, num_layers=layers)
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            self.head = nn.Sequential(nn.Linear(d + DIM, 128), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU())
            self.outs = nn.ModuleList([nn.Linear(64, 6) for _ in range(3)])

        def forward(self, h, x):
            e = self.proj(h.reshape(-1, 3, 7)) + self.pos
            b = e.shape[0]
            c = self.cls.expand(b, -1, -1)
            z = self.enc(torch.cat([c, e], 1))[:, 0]
            z = self.head(torch.cat([z, x], dim=1))
            return torch.stack([o(z) for o in self.outs], 1)

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Linear(7, 16)
            self.conv = nn.Sequential(nn.Conv1d(16, 32, 2), nn.ReLU(),
                                      nn.Conv1d(32, 64, 2), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(64 + DIM, 128), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU())
            self.outs = nn.ModuleList([nn.Linear(64, 6) for _ in range(3)])

        def forward(self, h, x):
            e = self.emb(h.reshape(-1, 3, 7)).transpose(1, 2)
            z = self.conv(e).mean(2)
            z = self.head(torch.cat([z, x], dim=1))
            return torch.stack([o(z) for o in self.outs], 1)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Sequential(nn.Linear(21 + DIM, 128), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU())
            self.outs = nn.ModuleList([nn.Linear(64, 6) for _ in range(3)])

        def forward(self, h, x):
            z = self.head(torch.cat([h, x], dim=1))
            return torch.stack([o(z) for o in self.outs], 1)

    print(f"样本: train={len(g_tr)} val={len(g_va)} test={len(g_te)}", flush=True)
    want = sys.argv[1].split(",") if len(sys.argv) > 1 else ["all"]
    weighted = len(sys.argv) > 2 and sys.argv[2] == "w"
    if weighted:
        print("启用类权重(逆频率)", flush=True)
    todo = {
        "GRU_h16": GRU(16), "GRU_h32": GRU(32), "GRU_h64": GRU(64), "GRU_h128": GRU(128),
        "BiLSTM_h64": BiLSTM(64), "Transformer_d32": Transformer(32),
        "CNN": CNN(), "MLP(无序列)": MLP(),
    }
    for name, net in todo.items():
        if "all" in want or name in want:
            train_eval(name, net)
    with open(D + "results/processed/nn_ablation_multistep_20260807.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("已写入 results/processed/nn_ablation_multistep_20260807.json", flush=True)


if __name__ == "__main__":
    main()
