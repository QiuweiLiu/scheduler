#!/usr/bin/env python3
"""行为预测架构对比(对应远端 docs/behavior_architecture_comparison_20260806.md)

A: 两级分开(角色 LightGBM + 动作族 XGBoost,独立)
B1: 级联特征(动作族 XGBoost 吃角色概率特征)
B2: 端到端 NN(MLP/GRU,角色头 + 动作族头联合 loss)
C: 一级直接预测 7 类路由标签(全模型族)
统一评估:联合准确率 / 执行者路由 / 资源分层 / 训练时间
"""
import argparse
import json
import math
import time
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
ROLES = ["init", "plan", "execute", "aggregate", "terminate"]
TGT_ROLES = ["plan", "execute", "aggregate", "terminate"]
EXECUTOR = {"select_frames": "TOOL", "visual_qa": "LLM", "temporal_ops": "TOOL",
            "summarize": "LLM", "detect": "YOLO", "other": "?"}
ROUTE_LABELS = ["non_execute"] + FAMS  # 7 类
LLM_SET = {"Qwen3-VL-8B-Instruct", "Qwen2.5-VL-3B-Instruct"}


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def fam_of(raw):
    return FAMILY.get(raw, "other")


def real_executor(model_id):
    if model_id in LLM_SET:
        return "LLM"
    if "yolo" in str(model_id):
        return "YOLO"
    return "TOOL"


def build_data(rows):
    by_run = defaultdict(list)
    for i, r in enumerate(rows):
        by_run[r["run_id"]].append((i, r))
    for rid, rs in by_run.items():
        rs.sort(key=lambda x: x[1]["event_index"])
    seq_of = {}
    for rid, rs in by_run.items():
        seq_of[rid] = [fam_of(r["raw_action"]) if r["role"] == "execute" else "non_execute"
                       for _, r in rs]
        for k, (i, r) in enumerate(rs):
            nxt = rs[k + 1][1] if k + 1 < len(rs) else None
            r["_next_model"] = nxt["model_id"] if nxt else None
    return seq_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_jsonl(args.dataset)
    for r in rows:
        r["_fam"] = fam_of(r.get("raw_action"))
        r["_role"] = r["role"]
        r["_is_exec"] = 1 if r["role"] == "execute" else 0
    print(f"全量行: {len(rows)}")

    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "validation"]
    te = [r for r in rows if r["split"] == "test"]
    seq_of = build_data(rows)

    # 预测目标 = 下一事件:next_role 的角色、next 事件的族
    for r in rows:
        seq = seq_of[r["run_id"]]
        idx = r["event_index"]
        nxt_fam = None
        if idx + 1 < len(seq):
            a = seq[idx + 1]
            if a != "non_execute":
                nxt_fam = a
        r["_next_fam"] = nxt_fam

    exe_tr = [r for r in tr if r.get("_next_fam") is not None]
    exe_te = [r for r in te if r.get("_next_fam") is not None]

    # 特征编码
    fam_idx = {f: i for i, f in enumerate(FAMS)}
    role_idx = {r: i for i, r in enumerate(TGT_ROLES)}
    models = sorted(set(r["model_id"] for r in rows))
    model_idx = {m: i for i, m in enumerate(models)}
    bases = sorted(set(r["baseline"] for r in rows))
    base_idx = {b: i for i, b in enumerate(bases)}
    qts = sorted(set(r["question_type"] for r in rows))
    qt_idx = {q: i for i, q in enumerate(qts)}
    max_ev = max(r["event_index"] for r in rows)

    def flat(r):
        last = None
        for a in seq_of[r["run_id"]]:
            if a != "non_execute":
                last = a
        x = [model_idx[r["model_id"]], base_idx[r["baseline"]],
             r["event_index"] / max_ev,
             1.0 if r.get("model_resident_before") else 0.0,
             1.0 if r.get("yolo_batch") else 0.0,
             qt_idx[r["question_type"]]]
        for f in FAMS:
            x.append(1.0 if last == f else 0.0)
        return np.array(x, dtype=np.float32)

    X = np.stack([flat(r) for r in rows])
    D = X.shape[1]
    y_role = np.array([role_idx[r["next_role"]] for r in rows], dtype=np.int64)  # 4 类,无 -1
    y_fam = np.array([fam_idx[r["_next_fam"]] if r["_next_fam"] else -1 for r in rows], dtype=np.int64)
    y_route = np.array([fam_idx[r["_next_fam"]] + 1 if r["_next_fam"] else 0 for r in rows], dtype=np.int64)

    def ids(rs):
        return [rows.index(r) for r in rs]

    itr, iva, ite = ids(tr), ids(va), ids(te)
    iexe_tr, iexe_te = ids(exe_tr), ids(exe_te)

    results = {"dataset": args.dataset, "models": {}}

    def eval_pipeline(name, p_role_arr, p_fam_arr, train_s):
        # p_role: 4 类 argmax;p_fam: execute 行的 6 类 argmax 或 None
        n = len(te)
        join = route = layer = 0
        for i, j in enumerate(ite):
            tr_, tf_ = y_role[j], y_fam[j]
            pr, pf = p_role_arr[i], p_fam_arr[i]
            if tf_ >= 0:
                if pr == role_idx["execute"] and pf == tf_:
                    join += 1
            else:
                if pr == tr_:
                    join += 1
            # 执行者路由
            ex_pred = EXECUTOR.get(FAMS[pf], None) if (pf is not None and pf >= 0) else None
            if tf_ >= 0:
                ex_true = real_executor(rows[j].get("_next_model"))
                if ex_pred == ex_true:
                    route += 1
            else:
                if ex_pred is None:
                    route += 1
            # 资源分层
            layer_true = (FAMS[tf_] if tf_ >= 0 else None, ex_true if tf_ >= 0 else None)
            layer_pred = (FAMS[pf] if (pf is not None and pf >= 0) else None, ex_pred)
            if layer_true == layer_pred:
                layer += 1
        m = {"join_acc": round(join / n, 4), "exec_route": round(route / n, 4),
             "layer_acc": round(layer / n, 4), "train_s": round(train_s, 2)}
        results["models"][name] = m
        print(f"  {name:26s} join={m['join_acc']:.4f} route={m['exec_route']:.4f} layer={m['layer_acc']:.4f} ({m['train_s']}s)")

    import lightgbm as lgb
    import xgboost as xgb

    # ===== A: 角色 LightGBM + 动作族 XGBoost(独立) =====
    t0 = time.time()
    l1 = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                            verbose=-1, random_state=42).fit(X[itr], y_role[itr])
    l2 = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                           random_state=42).fit(X[iexe_tr], y_fam[iexe_tr])
    tA = time.time() - t0
    prA = l1.predict(X[ite])
    pfA = []
    for i, j in enumerate(ite):
        if prA[i] == role_idx["execute"]:
            pp = l2.predict_proba(X[j:j + 1])[0]
            if pp.max() >= 0.5:
                pfA.append(int(np.argmax(pp)))
            else:
                pfA.append(None)
        else:
            pfA.append(None)
    eval_pipeline("A_两级分开", prA, pfA, tA)

    # ===== B1: 级联特征(角色概率进 L2) =====
    t0 = time.time()
    l1b = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                             verbose=-1, random_state=42).fit(X[itr], y_role[itr])
    rp_tr = l1b.predict_proba(X[itr])
    rp_exe_tr = l1b.predict_proba(X[iexe_tr])
    Xb1 = np.hstack([X, np.zeros((len(rows), 4), dtype=np.float32)])
    Xb1[itr] = np.hstack([X[itr], rp_tr])
    Xb1[iexe_tr] = np.hstack([X[iexe_tr], rp_exe_tr])
    # 推理时 test 行也要角色概率
    Xb1[ite] = np.hstack([X[ite], l1b.predict_proba(X[ite])])
    l2b = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                            random_state=42).fit(Xb1[iexe_tr], y_fam[iexe_tr])
    tB1 = time.time() - t0
    prB1 = l1b.predict(X[ite])
    pfB1 = []
    for i, j in enumerate(ite):
        if prB1[i] == role_idx["execute"]:
            pp = l2b.predict_proba(Xb1[j:j + 1])[0]
            pfB1.append(int(np.argmax(pp)) if pp.max() >= 0.5 else None)
        else:
            pfB1.append(None)
    eval_pipeline("B1_级联特征", prB1, pfB1, tB1)

    # ===== B2: 端到端 NN(MLP / GRU) =====
    import torch
    import torch.nn as nn
    torch.manual_seed(42)
    W = 5

    def seq_win(r):
        s = seq_of[r["run_id"]]
        idx = r["event_index"]
        win = []
        for a in s[:idx + 1]:
            if a != "non_execute":
                win.append(fam_idx[a])
        win = win[-W:]
        return win + [6] * (W - len(win))

    S = np.array([seq_win(r) for r in rows], dtype=np.int64)
    Str, Sva, Ste = (torch.from_numpy(S[itr]), torch.from_numpy(S[iva]),
                     torch.from_numpy(S[ite]))
    xt, xv, xte = torch.from_numpy(X[itr]), torch.from_numpy(X[iva]), torch.from_numpy(X[ite])
    yt_r = torch.from_numpy(y_role[itr])
    yv_r = torch.from_numpy(y_role[iva])
    mask_tr = torch.from_numpy((y_fam[itr] >= 0).astype(np.float32))
    yt_f = torch.from_numpy(np.clip(y_fam[itr], 0, 5))
    mask_va = torch.from_numpy((y_fam[iva] >= 0).astype(np.float32))
    yv_f = torch.from_numpy(np.clip(y_fam[iva], 0, 5))

    def train_e2e(model, epochs=60):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        best, best_state = -1, None
        for ep in range(epochs):
            model.train(); opt.zero_grad()
            rp, fl = model(xt, Str)
            loss = (nn.functional.cross_entropy(rp, yt_r)
                    + (nn.functional.cross_entropy(fl, yt_f, reduction="none") * mask_tr).mean())
            loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                rp_v, _ = model(xv, Sva)
                acc = (rp_v.argmax(1) == yv_r).float().mean().item()
            if acc > best:
                best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        return model

    class E2E_MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Dropout(0.1))
            self.role_head = nn.Linear(64, 4)
            self.fam_head = nn.Sequential(nn.Linear(64 + 4, 64), nn.ReLU(), nn.Linear(64, 6))
        def forward(self, x, s):
            h = self.backbone(x)
            rp = torch.softmax(self.role_head(h), dim=1)
            return rp, self.fam_head(torch.cat([h, rp], dim=1))

    class E2E_GRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(7, 16)
            self.gru = nn.GRU(16, 48, batch_first=True)
            self.fuse = nn.Linear(48 + D, 64)
            self.role_head = nn.Linear(64, 4)
            self.fam_head = nn.Sequential(nn.Linear(64 + 4, 64), nn.ReLU(), nn.Linear(64, 6))
        def forward(self, x, s):
            e = self.emb(s)
            _, h = self.gru(e)
            hh = h[-1]
            hh = torch.relu(self.fuse(torch.cat([hh, x], dim=1)))
            rp = torch.softmax(self.role_head(hh), dim=1)
            return rp, self.fam_head(torch.cat([hh, rp], dim=1))

    for name, cls in (("B2_端到端MLP", E2E_MLP), ("B2_端到端GRU", E2E_GRU)):
        t0 = time.time()
        m = train_e2e(cls())
        with torch.no_grad():
            rp, fl = m(xte, Ste)
        pr = rp.argmax(1).numpy()
        fl_p = torch.softmax(fl, dim=1).numpy()
        pf = []
        for i, j in enumerate(ite):
            if pr[i] == role_idx["execute"]:
                pf.append(int(np.argmax(fl_p[i])) if fl_p[i].max() >= 0.5 else None)
            else:
                pf.append(None)
        eval_pipeline(name, pr, pf, time.time() - t0)

    # ===== C: 一级直接预测 7 类路由标签(全模型族) =====
    def run_eval_c(name, probs):
        preds = np.argmax(probs, axis=1)
        n = len(te)
        route = layer = 0
        for i, j in enumerate(ite):
            tr_route = y_route[j]
            pr = preds[i]
            ex_pred = EXECUTOR.get(ROUTE_LABELS[pr], None) if pr > 0 else None
            if tr_route > 0:
                ex_true = real_executor(rows[j].get("_next_model"))
                if ex_pred == ex_true:
                    route += 1
            else:
                if ex_pred is None:
                    route += 1
            l_true = (FAMS[tr_route - 1] if tr_route > 0 else None,
                      real_executor(rows[j].get("_next_model")) if tr_route > 0 else None)
            l_pred = (FAMS[pr - 1] if pr > 0 else None, ex_pred)
            if l_true == l_pred:
                layer += 1
        acc = (preds == y_route[ite]).mean()
        m = {"route_label_acc": round(float(acc), 4), "exec_route": round(route / n, 4),
             "layer_acc": round(layer / n, 4)}
        results["models"][name] = m
        print(f"  {name:26s} label={m['route_label_acc']:.4f} route={m['exec_route']:.4f} layer={m['layer_acc']:.4f}")

    y_route_tr = y_route[itr]
    y_route_va = y_route[iva]
    y_route_te = y_route[ite]

    # 全局先验
    prior = Counter(y_route_tr)
    tot = sum(prior.values())
    p0 = np.array([prior.get(i, 0) / tot for i in range(7)])
    run_eval_c("C_全局先验", np.tile(p0, (len(te), 1)))

    # 条件计数
    ctx = defaultdict(Counter)
    for i, j in enumerate(itr):
        ctx[(rows[j]["baseline"], min(rows[j]["event_index"], 9) // 3)][int(y_route[j])] += 1
    probs = []
    for i, j in enumerate(ite):
        c = ctx.get((rows[j]["baseline"], min(rows[j]["event_index"], 9) // 3), Counter())
        t = sum(c.values())
        probs.append([c.get(k, 0) / t if t else 1 / 7 for k in range(7)])
    run_eval_c("C_条件计数", np.array(probs))

    # n-gram(窗口 2)
    ng = defaultdict(Counter)
    # 简化 n-gram:用路由标签序列(每 run)
    def route_seq_of(r):
        s = seq_of[r["run_id"]]
        out = []
        for a in s:
            out.append(fam_idx[a] + 1 if a != "non_execute" else 0)
        return out
    for i, j in enumerate(itr):
        s = route_seq_of(rows[j])
        idx = rows[j]["event_index"]
        w = tuple(s[max(0, idx - 1): idx]) if idx >= 1 else ()
        ng[w][int(y_route[j])] += 1
    probs = []
    for i, j in enumerate(ite):
        s = route_seq_of(rows[j])
        idx = rows[j]["event_index"]
        w = tuple(s[max(0, idx - 1): idx]) if idx >= 1 else ()
        c = ng.get(w, Counter())
        t = sum(c.values())
        probs.append([c.get(k, 0) / t if t else 1 / 7 for k in range(7)])
    run_eval_c("C_n-gram_w2", np.array(probs))

    # 树族
    Xc_tr, Xc_va, Xc_te = X[itr], X[iva], X[ite]
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    lr = LogisticRegression(max_iter=500).fit(Xc_tr, y_route_tr)
    run_eval_c("C_逻辑回归", lr.predict_proba(Xc_te))
    rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xc_tr, y_route_tr)
    run_eval_c("C_随机森林", rf.predict_proba(Xc_te))
    lm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                            verbose=-1, random_state=42).fit(Xc_tr, y_route_tr)
    run_eval_c("C_LightGBM", lm.predict_proba(Xc_te))
    xm = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                           random_state=42).fit(Xc_tr, y_route_tr)
    run_eval_c("C_XGBoost", xm.predict_proba(Xc_te))

    # NN 族(复用紧凑实现)
    def train_nn_c(model, x_tr, s_tr, y_tr, x_va, s_va, y_va, epochs=50):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        best, best_state = -1, None
        yt = torch.from_numpy(y_tr); yv = torch.from_numpy(y_va)
        for ep in range(epochs):
            model.train(); opt.zero_grad()
            loss = nn.functional.cross_entropy(model(s_tr, x_tr), yt)
            loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                acc = (model(s_va, x_va).argmax(1) == yv).float().mean().item()
            if acc > best:
                best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state); model.eval()
        return model

    class CNet(nn.Module):
        def __init__(self, kind):
            super().__init__()
            self.kind = kind
            self.emb = nn.Embedding(7, 16)
            if kind == "gru":
                self.rnn = nn.GRU(16, 64, batch_first=True)
                self.head = nn.Sequential(nn.Linear(64 + D, 64), nn.ReLU(), nn.Linear(64, 7))
            elif kind == "mlp":
                self.head = nn.Sequential(nn.Linear(D, 128), nn.ReLU(), nn.Dropout(0.2),
                                          nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 7))
            elif kind == "transformer":
                self.pos = nn.Parameter(torch.randn(1, W, 16) * 0.1)
                self.attn = nn.MultiheadAttention(16, 4, batch_first=True)
                self.head = nn.Sequential(nn.Linear(16 + D, 64), nn.ReLU(), nn.Linear(64, 7))
            elif kind == "gnn":
                self.node = nn.Parameter(torch.randn(7, 16) * 0.1)
                self.p1 = nn.Linear(16, 16)
                self.p2 = nn.Linear(16, 16)
                self.head = nn.Sequential(nn.Linear(16 + D, 64), nn.ReLU(), nn.Linear(64, 7))
        def forward(self, s, x):
            if self.kind == "gru":
                _, h = self.rnn(self.emb(s))
                return self.head(torch.cat([h[-1], x], dim=1))
            if self.kind == "mlp":
                return self.head(x)
            if self.kind == "transformer":
                e = self.emb(s) + self.pos
                a, _ = self.attn(e, e, e)
                return self.head(torch.cat([a.mean(1), x], dim=1))
            emb = self.node
            A = torch.from_numpy(np.eye(7) * 0.5 + 0.05).float()
            for _ in range(2):
                emb = torch.tanh(self.p1(emb) + self.p2(A @ emb))
            return self.head(torch.cat([emb[s[:, -1].clamp(0, 6)], x], dim=1))

    for kind in ("mlp", "gru", "transformer", "gnn"):
        m = train_nn_c(CNet(kind), xt, Str, y_route_tr, xv, Sva, y_route_va)
        with torch.no_grad():
            pb = torch.softmax(m(Ste, xte), dim=1).numpy()
        run_eval_c(f"C_{kind}", pb)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已写入: {args.out}")


if __name__ == "__main__":
    main()
