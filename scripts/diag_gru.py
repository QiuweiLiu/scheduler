import sys
sys.path.insert(0, '/root/autodl-tmp/scheduler')
import json
import numpy as np
import torch
import torch.nn as nn
from nn_ablation_multistep import build_data, K

ex_f, fam_k, split_of, one_end = build_data()
itev = np.where(split_of == "validation")[0]
itex = np.where(split_of == "test")[0]
g_va = itev[fam_k[itev, K] >= 0]
g_te = itex[fam_k[itex, K] >= 0]
DIM = ex_f.shape[1] - one_end
x_va = torch.from_numpy(ex_f[g_va][:, one_end:])
h_va = torch.from_numpy(ex_f[g_va][:, one_end - 21:one_end])
y_va = torch.from_numpy(fam_k[g_va, 1:4])
x_te = torch.from_numpy(ex_f[g_te][:, one_end:])
h_te = torch.from_numpy(ex_f[g_te][:, one_end - 21:one_end])
y_te = torch.from_numpy(fam_k[g_te, 1:4])

torch.manual_seed(42)

class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Linear(7, 16)
        self.rnn = nn.GRU(16, 64, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64 + DIM, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU())
        self.outs = nn.ModuleList([nn.Linear(64, 6) for _ in range(3)])
    def forward(self, h, x):
        e = self.emb(h.reshape(-1, 3, 7))
        _, hh = self.rnn(e)
        z = self.head(torch.cat([hh[-1], x], dim=1))
        return torch.stack([o(z) for o in self.outs], 1)

net = GRU()
with torch.no_grad():
    o0 = net(h_te, x_te)
    acc0 = (o0.argmax(2) == y_te).float().mean().item()
print("初始化 test acc:", round(acc0, 4))
print("初始化 test 预测分布:", [np.bincount(o0[:, k-1].argmax(1).numpy(), minlength=6).tolist() for k in (1,2,3)])
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
itr_all = np.where(split_of == 'train')[0]
g_tr = itr_all[fam_k[itr_all, K] >= 0]
x_tr = torch.from_numpy(ex_f[g_tr][:, one_end:])
h_tr = torch.from_numpy(ex_f[g_tr][:, one_end - 21:one_end])
y_tr = torch.from_numpy(fam_k[g_tr, 1:4])
for ep in range(30):
    net.train(); opt.zero_grad()
    loss = nn.functional.cross_entropy(net(h_tr, x_tr).reshape(-1, 6), y_tr.reshape(-1))
    loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        va = (net(h_va, x_va).argmax(2) == y_va).float().mean().item()
        te = (net(h_te, x_te).argmax(2) == y_te).float().mean().item()
    if ep % 5 == 0 or ep < 3:
        print(f"ep={ep} loss={loss.item():.4f} val={va:.4f} te={te:.4f}")
