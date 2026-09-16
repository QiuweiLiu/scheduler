#!/usr/bin/env python3
"""NN 管线严格验证(2026-08-07):确定性 / 初始化敏感性 / 训练轨迹
只读诊断,不修改任何数据文件
"""
import sys
sys.path.insert(0, '/root/autodl-tmp/scheduler')
import json
import numpy as np
import torch
import torch.nn as nn
from nn_ablation_multistep import build_data, K

OUT = '/root/autodl-tmp/scheduler/results/processed/diag_pipeline_20260807.json'
log = []


def rec(*args):
    s = " ".join(str(a) for a in args)
    log.append(s)
    print(s, flush=True)
    with open(OUT, "w") as f:
        json.dump({"log": log}, f, ensure_ascii=False, indent=2)

ex_f, fam_k, split_of, one_end = build_data()
itev = np.where(split_of == "validation")[0]
itex = np.where(split_of == "test")[0]
g_va = itev[fam_k[itev, K] >= 0]
g_te = itex[fam_k[itex, K] >= 0]
itr_all = np.where(split_of == 'train')[0]
g_tr = itr_all[fam_k[itr_all, K] >= 0]
DIM = ex_f.shape[1] - one_end
x_va = torch.from_numpy(ex_f[g_va][:, one_end:]); h_va = torch.from_numpy(ex_f[g_va][:, one_end-21:one_end])
y_va = torch.from_numpy(fam_k[g_va, 1:4])
x_te = torch.from_numpy(ex_f[g_te][:, one_end:]); h_te = torch.from_numpy(ex_f[g_te][:, one_end-21:one_end])
y_te = torch.from_numpy(fam_k[g_te, 1:4])
x_tr = torch.from_numpy(ex_f[g_tr][:, one_end:]); h_tr = torch.from_numpy(ex_f[g_tr][:, one_end-21:one_end])
y_tr = torch.from_numpy(fam_k[g_tr, 1:4])

rec(f"数据: train={len(g_tr)} val={len(g_va)} test={len(g_te)} DIM={DIM} one_end={one_end}")


class GRU(nn.Module):
    def __init__(self, hid=16):
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


def train_once(seed, trace=False):
    torch.manual_seed(seed)
    net = GRU(16)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best, best_state, best_ep = -1, None, -1
    curve = []
    for ep in range(150):
        net.train(); opt.zero_grad()
        loss = nn.functional.cross_entropy(net(h_tr, x_tr).reshape(-1, 6), y_tr.reshape(-1))
        loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            acc = (net(h_va, x_va).argmax(2) == y_va).float().mean().item()
        curve.append((ep, round(float(loss.item()), 4), round(acc, 4)))
        if acc > best:
            best, best_state, best_ep = acc, {k: v.clone() for k, v in net.state_dict().items()}, ep
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        out = net(h_te, x_te)
        a = out.argmax(2).numpy()
    if trace:
        for i in range(0, 150, 15):
            rec(f"    ep={curve[i][0]:3d} loss={curve[i][1]:.4f} val={curve[i][2]:.4f}")
        rec(f"    best_ep={best_ep} best_val={best:.4f}")
    return best, a, curve


# 1) 确定性:seed 42 跑两次
rec("\n[1] 确定性检验(seed 42 × 2):")
b1, a1, _ = train_once(42)
b2, a2, _ = train_once(42)
rec(f"  val1={b1:.4f} val2={b2:.4f} | argmax 逐位一致: {(a1 == a2).all()}")
rec(f"  pred_dist1: {[np.bincount(a1[:, k-1], minlength=6).tolist() for k in (1,2,3)]}")

# 2) 初始化敏感性:seed 42/43/44/45
rec("\n[2] 初始化敏感性(seeds 42-45):")
bests, arrs = [], []
for s in (42, 43, 44, 45):
    b, a, _ = train_once(s)
    bests.append(round(b, 4))
    arrs.append(a)
    rec(f"  seed={s}: val={b:.4f} pred_dist={[np.bincount(a[:, k-1], minlength=6).tolist() for k in (1,2,3)]}")
base = arrs[0]
for j in range(1, 4):
    rec(f"  seed42 vs seed{[43,44,45][j-1]}: 差异 {int((base != arrs[j]).sum())} / {base.size}")
rec("  val 序列:", bests)

# 3) 训练轨迹(seed 42 详细)
rec("\n[3] 训练轨迹(seed 42):")
_, _, curve = train_once(42, trace=True)

with open(OUT, "w") as f:
    json.dump({"log": log}, f, ensure_ascii=False, indent=2)
print("已写入", OUT, flush=True)
