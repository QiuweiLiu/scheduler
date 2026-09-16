#!/usr/bin/env python3
"""多步预测实验(2026-08-07) —— 行为 + 资源,全部远程执行

口径:
- 事件轴:每行 j 预测未来第 k 个事件的角色;单步 XGB 复刻锚点(next 语义)
- execute 轴:每个 execute 位置 q 预测未来第 k 个 execute 的族(直接 / rollout / GRU)
- 资源轴:各量(runtime/peak/load)>0 的事件序列,预测未来第 k 个资源事件的值
- 泄漏铁律:特征只用 <=当前 信息;rollout 用预测回填
- rel = sum|p-y|/sum(y);bucket = 3 量级分桶率
输出:results/processed/multistep_experiment_20260807.json
"""
import json
import statistics as st
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
TGT_ROLES = ["plan", "execute", "aggregate", "terminate"]
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
K_MAX = 5


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


def bucket(v):
    return 0 if v < 1000 else (1 if v < 10000 else 2)


def metrics(preds, ys):
    errs = [abs(p - y) for p, y in zip(preds, ys)]
    mae = sum(errs) / max(1, len(errs))
    rel = sum(errs) / max(1e-9, sum(ys))
    bc = sum(1 for p, y in zip(preds, ys) if bucket(p) == bucket(y)) / max(1, len(ys))
    return round(mae, 1), round(rel, 4), round(bc, 4)


def top1(y_true, pb):
    return round(float((np.argmax(pb, 1) == y_true).mean()), 4)


def onehot(vals, size):
    v = np.zeros(size, dtype=np.float32)
    v[vals] = 1.0
    return v


def zscore_cols(X, one_end, num_end, tr_slice):
    mu = X[tr_slice, one_end:num_end].mean(0)
    sd = X[tr_slice, one_end:num_end].std(0).clip(min=1e-6)
    X[:, one_end:num_end] = (X[:, one_end:num_end] - mu) / sd


def main():
    rows = load_jsonl("results/processed/role_dataset_v0_3.jsonl")
    prefixes = load_jsonl("results/processed/neural_prefix_dataset_v0_9_options_sourcefix_20260805/prefixes_visual_context.jsonl")
    for r in rows:
        r["_fam"] = fam_of(r.get("raw_action"))
        r["_exe"] = executor_of(r["model_id"])
        r["_stack"] = stack_of(r["model_id"])
    vid_meta = {}
    for p in prefixes:
        ts = p.get("task_structure") or {}
        vid_meta[p["video_id"]] = {
            "domain": (ts.get("domain") if isinstance(ts, dict) else None) or "unknown",
        }
    for r in rows:
        m = vid_meta.get(r["video_id"], {})
        r["_dom"] = m.get("domain", "unknown")

    tr = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] == "test"]

    fam_idx = {f: i for i, f in enumerate(FAMS)}
    role_idx = {r: i for i, r in enumerate(TGT_ROLES)}
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

    max_ev = max(r["event_index"] for r in rows)
    max_rt = max((float(r.get("runtime_ms") or 0) for r in rows), default=1.0)
    max_pk = max((float(r.get("peak_allocated_mb") or 0) for r in rows), default=1.0)
    max_ld = max((float(r.get("load_ms") or 0) for r in rows), default=1.0)

    # cross 统计(prev_fam x model,仅 train)
    gmed2 = defaultdict(list)
    for rid, rs in by_run.items():
        last_fam = None
        for r in rs:
            if (r.get("runtime_ms") or 0) > 0 and last_fam:
                gmed2[(last_fam, r["model_id"])].append(r["runtime_ms"])
            if r["split"] == "train" and r["role"] == "execute":
                pass
            if r["role"] == "execute":
                last_fam = fam_of(r["raw_action"])
            if r["split"] != "train":
                last_fam = None
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

    ctx_one = len(models) + len(bases) + len(qts) + len(exes) + len(stacks) + len(doms)
    split_of = {id(r): r["split"] for r in rows}

    # ============ 事件轴 ============
    ev_rows, ev_one, ev_num = [], [], []
    for rid, rs in by_run.items():
        last_fam = None
        for j, r in enumerate(rs):
            oh, num = ctx_parts(r, last_fam)
            prev_rt = float(rs[j - 1].get("runtime_ms") or 0) if j >= 1 else 0.0
            prev2_rt = float(rs[j - 2].get("runtime_ms") or 0) if j >= 2 else 0.0
            prev_role = role_idx[rs[j - 1]["role"]] if j >= 1 and rs[j - 1]["role"] in role_idx else -1
            cur_role = role_idx[r["role"]] if r["role"] in role_idx else -1
            ev_one.append(np.concatenate([
                oh,
                onehot([prev_role], 4) if prev_role >= 0 else np.zeros(4, dtype=np.float32),
                onehot([cur_role], 4) if cur_role >= 0 else np.zeros(4, dtype=np.float32),
            ]))
            ev_num.append(np.concatenate([
                num, np.array([j / max_ev, prev_rt / max_rt, prev2_rt / max_rt],
                              dtype=np.float32)]))
            ev_rows.append(r)
            if r["role"] == "execute":
                last_fam = fam_of(r["raw_action"])
    ev_one = np.stack(ev_one)
    ev_num = np.stack(ev_num)
    ev_f = np.hstack([ev_one, ev_num])
    ev_one_end = ev_one.shape[1]
    ev_num_end = ev_f.shape[1]
    tr_idx = [i for i, r in enumerate(ev_rows) if r["split"] == "train"]
    zscore_cols(ev_f, ev_one_end, ev_num_end, tr_idx)

    ev_idx_map = {id(r): i for i, r in enumerate(ev_rows)}
    role_k = np.full((len(ev_rows), K_MAX + 1), -1, dtype=np.int64)
    for rid, rs in by_run.items():
        for j, r in enumerate(rs):
            i = ev_idx_map[id(r)]
            for k in range(1, K_MAX + 1):
                if j + k < len(rs) and rs[j + k]["role"] in role_idx:
                    role_k[i, k] = role_idx[rs[j + k]["role"]]
    next_fam_arr = np.full(len(ev_rows), -1, dtype=np.int64)
    next_model_arr = np.array(["" for _ in ev_rows], dtype=object)
    for rid, rs in by_run.items():
        for j, r in enumerate(rs):
            i = ev_idx_map[id(r)]
            if j + 1 < len(rs) and rs[j + 1]["role"] == "execute":
                next_fam_arr[i] = fam_idx[fam_of(rs[j + 1]["raw_action"])]
                next_model_arr[i] = rs[j + 1]["model_id"]
    role_next = np.array([role_idx[r["next_role"]] if r["next_role"] in role_idx else 0
                          for r in ev_rows], dtype=np.int64)

    # ============ execute 轴 ============
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
    ex_one_end = ex_one.shape[1]
    ex_num_end = ex_f.shape[1]
    tr_ex = [i for i, r in enumerate(ex_rows) if r["split"] == "train"]
    zscore_cols(ex_f, ex_one_end, ex_num_end, tr_ex)

    ex_idx_map = {id(r): i for i, r in enumerate(ex_rows)}
    fam_k = np.full((len(ex_rows), K_MAX + 1), -1, dtype=np.int64)
    rt_k = np.full((len(ex_rows), K_MAX + 1), -1.0)
    for rid, rs in by_run.items():
        exes_seq = [r for r in rs if r["role"] == "execute"]
        for q, r in enumerate(exes_seq):
            i = ex_idx_map[id(r)]
            for k in range(1, K_MAX + 1):
                if q + k < len(exes_seq):
                    fam_k[i, k] = fam_idx[fam_of(exes_seq[q + k]["raw_action"])]
                    rt_k[i, k] = float(exes_seq[q + k].get("runtime_ms") or 0)

    # ============ 资源轴 ============
    res_data = {}
    for key, col in (("runtime", "runtime_ms"), ("peak", "peak_allocated_mb"), ("load", "load_ms")):
        r_events = []
        for rid, rs in by_run.items():
            evs = [r for r in rs if (r.get(col) or 0) > 0]
            r_events.append(evs)
        targets = np.full((sum(len(e) for e in r_events), K_MAX + 1), -1.0)
        ones, nums = [], []
        ptr = 0
        for evs in r_events:
            for q, r in enumerate(evs):
                oh, num = ctx_parts(r, fam_of(evs[q - 1]["raw_action"]) if q >= 1 else None)
                ones.append(oh)
                nums.append(np.concatenate([
                    num,
                    np.array([q / max(1, len(evs)),
                              float(evs[q - 1].get(col) or 0) if q >= 1 else 0.0,
                              float(r.get("event_index") or 0) / max_ev],
                             dtype=np.float32)]))
                for k in range(1, K_MAX + 1):
                    if q + k < len(evs):
                        targets[ptr, k] = float(evs[q + k].get(col) or 0)
                ptr += 1
        ones = np.stack(ones)
        nums = np.stack(nums)
        Xf = np.hstack([ones, nums])
        one_end = ones.shape[1]
        num_end = Xf.shape[1]
        spl = np.array([split_of[id(r)] for evs in r_events for r in evs])
        tr_r = np.where(spl == "train")[0]
        zscore_cols(Xf, one_end, num_end, tr_r)
        res_data[key] = {"X": Xf, "y": targets, "split": spl}

    results = {"models": {}}
    import xgboost as xgb

    # ========== ① 事件轴单步锚点(next 语义,复刻 script1) ==========
    itr = np.array(tr_idx)
    ite = np.array([i for i, r in enumerate(ev_rows) if r["split"] == "test"])
    l_role = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                               random_state=42).fit(ev_f[itr], role_next[itr])
    pr_role = l_role.predict_proba(ev_f[ite])
    nf_tr = itr[next_fam_arr[itr] >= 0]
    nf_te = ite[next_fam_arr[ite] >= 0]
    l_fam = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                              random_state=42).fit(ev_f[nf_tr], next_fam_arr[nf_tr])
    p_fam = l_fam.predict_proba(ev_f[nf_te])
    r_role = top1(role_next[ite], pr_role)
    r_fam = top1(next_fam_arr[nf_te], p_fam)
    print(f"[锚点] next语义 XGB: 角色 Top1={r_role:.4f} | 动作族 Top1={r_fam:.4f} (n={len(nf_te)})")
    results["models"]["锚点_next语义_XGB"] = {
        "role_top1": r_role, "fam_top1": r_fam, "fam_test_n": int(len(nf_te))}

    # ========== ② 事件轴角色多步(直接) ==========
    role_ms = {"per_step": {}, "win3": {}}
    for k in range(1, K_MAX + 1):
        mk_tr = itr[role_k[itr, k] >= 0]
        mk_te = ite[role_k[ite, k] >= 0]
        m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                              random_state=42).fit(ev_f[mk_tr], role_k[mk_tr, k])
        t1 = top1(role_k[mk_te, k], m.predict_proba(ev_f[mk_te]))
        role_ms["per_step"][k] = {"top1": t1, "n": int(len(mk_te))}
        print(f"[事件轴] 角色 k={k}: Top1={t1:.4f} (n={len(mk_te)})")
    w3 = ite[role_k[ite, 3] >= 0]
    hits = 0
    for k in (1, 2, 3):
        mk_tr = itr[role_k[itr, k] >= 0]
        m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                              random_state=42).fit(ev_f[mk_tr], role_k[mk_tr, k])
        hits += (m.predict(ev_f[w3]) == role_k[w3, k])
    win3 = round(float((hits == 3).mean()), 4)
    role_ms["win3"] = {"all_hit": win3, "n": int(len(w3))}
    print(f"[事件轴] 角色 W=3 全命中: {win3:.4f} (n={len(w3)})")
    results["models"]["事件轴_角色多步"] = role_ms

    # ========== ③ execute 轴族多步 ==========
    itrx = np.array(tr_ex)
    itex = np.array([i for i, r in enumerate(ex_rows) if r["split"] == "test"])
    fam_direct = {"per_step": {}, "win3": {}}
    fam_models = {}
    for k in range(1, K_MAX + 1):
        mk_tr = itrx[fam_k[itrx, k] >= 0]
        mk_te = itex[fam_k[itex, k] >= 0]
        m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                              random_state=42).fit(ex_f[mk_tr], fam_k[mk_tr, k])
        fam_models[k] = m
        t1 = top1(fam_k[mk_te, k], m.predict_proba(ex_f[mk_te]))
        fam_direct["per_step"][k] = {"top1": t1, "n": int(len(mk_te))}
        print(f"[execute轴] 直接 k={k}: Top1={t1:.4f} (n={len(mk_te)})")
    w3x = itex[fam_k[itex, 3] >= 0]
    hits = 0
    for k in (1, 2, 3):
        hits += (fam_models[k].predict(ex_f[w3x]) == fam_k[w3x, k])
    win3 = round(float((hits == 3).mean()), 4)
    fam_direct["win3"] = {"all_hit": win3, "n": int(len(w3x))}
    print(f"[execute轴] 直接 W=3 全命中: {win3:.4f} (n={len(w3x)})")

    # rollout(k=1 模型自回归)
    m1 = fam_models[1]
    roll = {"per_step": {}, "win3": {}}
    preds_seq = {}
    Xr = ex_f[itex].copy()
    te_hist_slice = slice(ex_one_end - 21, ex_one_end)
    for q in range(K_MAX):
        pred = m1.predict(Xr)
        preds_seq[q + 1] = pred
        hist = Xr[:, te_hist_slice].reshape(-1, 3, 7)
        hist = np.roll(hist, -1, axis=1)
        for i in range(hist.shape[0]):
            hist[i, 2, :] = 0.0
            hist[i, 2, pred[i]] = 1.0
        Xr[:, te_hist_slice] = hist.reshape(-1, 21)
    for k in range(1, K_MAX + 1):
        mk = fam_k[itex, k] >= 0
        t1 = round(float((preds_seq[k][mk] == fam_k[itex[mk], k]).mean()), 4)
        roll["per_step"][k] = {"top1": t1, "n": int(mk.sum())}
        print(f"[execute轴] rollout k={k}: Top1={t1:.4f} (n={int(mk.sum())})")
    mask3 = fam_k[itex, 3] >= 0
    hits = sum(preds_seq[k][mask3] == fam_k[itex[mask3], k] for k in (1, 2, 3))
    roll["win3"] = {"all_hit": round(float((hits == 3).mean()), 4), "n": int(mask3.sum())}
    print(f"[execute轴] rollout W=3 全命中: {roll['win3']['all_hit']:.4f}")
    results["models"]["execute轴_族直接"] = fam_direct
    results["models"]["execute轴_族rollout"] = roll

    # GRU 序列(K=3)
    try:
        import torch
        import torch.nn as nn
        torch.manual_seed(42)
        g_tr = itrx[fam_k[itrx, 3] >= 0]
        g_va = np.array([i for i, r in enumerate(ex_rows) if r["split"] == "validation"])[fam_k[
            np.array([i for i, r in enumerate(ex_rows) if r["split"] == "validation"]), 3] >= 0]
        g_te = itex[fam_k[itex, 3] >= 0]
        DIM = ex_num_end - ex_one_end

        class GRU(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Linear(7, 16)
                self.rnn = nn.GRU(16, 64, batch_first=True)
                self.head = nn.Sequential(nn.Linear(64 + DIM, 128), nn.ReLU(),
                                          nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU())
                self.outs = nn.ModuleList([nn.Linear(64, 6) for _ in range(3)])

            def forward(self, h, x):
                e = self.emb(h.reshape(-1, 3, 7))
                _, hh = self.rnn(e)
                z = self.head(torch.cat([hh[-1], x], dim=1))
                return torch.stack([o(z) for o in self.outs], 1)

        h_idx = (ex_one_end - 21, ex_one_end)
        net = GRU()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        x_tr = torch.from_numpy(ex_f[g_tr][:, ex_one_end:]); h_tr = torch.from_numpy(ex_f[g_tr][:, h_idx[0]:h_idx[1]])
        y_tr = torch.from_numpy(fam_k[g_tr, 1:4])
        x_va = torch.from_numpy(ex_f[g_va][:, ex_one_end:]); h_va = torch.from_numpy(ex_f[g_va][:, h_idx[0]:h_idx[1]])
        y_va = torch.from_numpy(fam_k[g_va, 1:4])
        best, best_state = -1, None
        for ep in range(150):
            net.train(); opt.zero_grad()
            loss = nn.functional.cross_entropy(
                net(h_tr, x_tr).reshape(-1, 6), y_tr.reshape(-1))
            loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                acc = (net(h_va, x_va).argmax(2) == y_va).float().mean().item()
            if acc > best:
                best, best_state = acc, {k_: v.clone() for k_, v in net.state_dict().items()}
        net.load_state_dict(best_state); net.eval()
        with torch.no_grad():
            out = net(torch.from_numpy(ex_f[g_te][:, h_idx[0]:h_idx[1]]),
                      torch.from_numpy(ex_f[g_te][:, ex_one_end:])).numpy()
        gru_ms = {"per_step": {}, "win3": {}}
        for k in range(1, 4):
            t1 = top1(fam_k[g_te, k], np.eye(6)[out[:, k - 1].argmax(1)])
            gru_ms["per_step"][k] = {"top1": t1, "n": int(len(g_te))}
            print(f"[execute轴] GRU k={k}: Top1={t1:.4f}")
        hits = sum(out[:, k - 1].argmax(1) == fam_k[g_te, k] for k in (1, 2, 3))
        gru_ms["win3"] = {"all_hit": round(float((hits == 3).mean()), 4), "n": int(len(g_te))}
        print(f"[execute轴] GRU W=3 全命中: {gru_ms['win3']['all_hit']:.4f}")
        results["models"]["execute轴_族GRU"] = gru_ms
    except Exception as e:
        print("GRU 失败:", e)
        results["models"]["execute轴_族GRU"] = {"error": str(e)}

    # ========== ④ 资源多步 ==========
    res_ms = {}
    for key, d in res_data.items():
        Xf, y, spl = d["X"], d["y"], d["split"]
        r_tr = np.where(spl == "train")[0]
        r_te = np.where(spl == "test")[0]
        per_k = {}
        for k in range(1, K_MAX + 1):
            k_tr = r_tr[y[r_tr, k] > 0]
            k_te = r_te[y[r_te, k] > 0]
            if len(k_tr) < 50 or len(k_te) == 0:
                per_k[k] = {"n": int(len(k_te))}
                continue
            m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                 random_state=42).fit(Xf[k_tr], y[k_tr, k])
            pred = m.predict(Xf[k_te])
            mae, rel, bc = metrics(pred, y[k_te, k])
            per_k[k] = {"mae": mae, "rel": rel, "bucket": bc, "n": int(len(k_te))}
            print(f"[资源:{key}] k={k}: MAE={mae:.0f} rel={rel:.1%} bucket={bc:.1%} (n={len(k_te)})")
        res_ms[key] = per_k
    results["models"]["资源_多步"] = res_ms

    with open(D + "results/processed/multistep_experiment_20260807.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("已写入 results/processed/multistep_experiment_20260807.json")


if __name__ == "__main__":
    main()
