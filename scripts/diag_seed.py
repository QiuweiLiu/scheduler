import sys
sys.path.insert(0, '/root/autodl-tmp/scheduler')
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
itr_all = np.where(split_of == 'train')[0]
g_tr = itr_all[fam_k[itr_all, K] >= 0]
x_va = torch.from_numpy(ex_f[g_va][:, one_end:]); h_va = torch.from_numpy(ex_f[g_va][:, one_end - 21:one_end])
y_va = torch.from_numpy(fam_k[g_va, 1:4])
x_te = torch.from_numpy(ex_f[g_te][:, one_end:]); h_te = torch.from_numpy(ex_f[g_te][:, one_end - 21:one_end])
y_te = torch.from_numpy(fam_k[g_te, 1:4])
x_tr = torch.from_numpy(ex_f[g_tr][:, one_end:]); h_tr = torch.from_numpy(ex_f[g_tr][:, one_end - 21:one_end])
y_tr = torch.from_numpy(fam_k[g_tr, 1:4])

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

def run(seed, epochs=150):
    torch.manual_seed(seed)
    net = GRU()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best, best_state = -1, None
    for ep in range(epochs):
        net.train(); opt.zero_grad()
        loss = nn.functional.cross_entropy(net(h_tr, x_tr).reshape(-1, 6), y_tr.reshape(-1))
        loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            acc = (net(h_va, x_va).argmax(2) == y_va).float().mean().item()
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        out = net(h_te, x_te)
    return best, out.argmax(2).numpy(), out

b42, a42, o42 = run(42)
b43, a43, o43 = run(43)
print("seed42 val:", round(b42,4), "| seed43 val:", round(b43,4))
print("argmax 相同比例:", float((a42 == a43).mean()))
print("k1 argmax(42):", a42[:15,0].tolist())
print("k1 argmax(43):", a43[:15,0].tolist())
print("softmax 差(42 vs 43)前3行 k1:", (o42[:,0,:3].numpy() - o43[:,0,:3].numpy())[:3].round(4).tolist())
