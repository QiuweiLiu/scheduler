#!/usr/bin/env python3
"""BiLSTM/GRU 补测:next 语义 + 最优特征集(单级 + 整体对照)

补上 nn_improve 系列遗漏的序列模型重测
对照:XGBoost 0.8950(单级)/ 0.8493-0.9520-0.9059(整体);MLP 0.8772
"""
import json
import math
import statistics as st
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
TGT_ROLES = ["plan", "execute", "aggregate", "terminate"]
EXECUTOR = {"select_frames": "TOOL", "visual_qa": "LLM", "temporal_ops": "TOOL",
            "summarize": "LLM", "detect": "YOLO", "other": "?"}
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
D = "/root/autodl-tmp/scheduler/"


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


def eval_pipeline(te, y_role_next, nf_arr, nm_arr, p_role, p_fam_prob):
    n = len(te)
    join = route = layer = 0
    for i, j in enumerate(te):
        tr_ = y_role_next[j]
        tf_ = nf_arr[j]
        pr = p_role[i]
        if pr == 1 and max(p_fam_prob[i]) >= 0.5:
            pf = int(np.argmax(p_fam_prob[i]))
        else:
            pf = None
        if tf_ >= 0:
            if pr == 1 and pf == tf_:
                join += 1
        else:
            if pr == tr_:
                join += 1
        ex_pred = EXECUTOR.get(FAMS[pf]) if pf is not None else None
        ex_true = executor_of(nm_arr[j]) if tf_ >= 0 else None
        if ex_pred == ex_true:
            route += 1
        l_true = (FAMS[tf_] if tf_ >= 0 else None, ex_true)
        l_pred = (FAMS[pf] if pf is not None else None, ex_pred)
        if l_true == l_pred:
            layer += 1
    return {"join": round(join / n, 4), "route": round(route / n, 4), "layer": round(layer / n, 4)}


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
            "q_chars": (ts.get("question_chars") if isinstance(ts, dict) else None) or 0,
            "q_tokens": (ts.get("question_tokens") if isinstance(ts, dict) else None) or 0,
            "opt_count": (ts.get("option_count") if isinstance(ts, dict) else None) or 0,
            "duration": ((p.get("state_features") or {}).get("video") or {}).get("duration_s"),
            "fps": ((p.get("state_features") or {}).get("video") or {}).get("fps"),
        }
    for r in rows:
        m = vid_meta.get(r["video_id"], {})
        r["_dom"] = m.get("domain", "unknown")
        r["_q_chars"] = m.get("q_chars", 0)
        r["_q_tokens"] = m.get("q_tokens", 0)
        r["_opt_count"] = m.get("opt_count", 0)
        r["_duration"] = float(m.get("duration") or 0)
        r["_fps"] = float(m.get("fps") or 0)

    by_run = defaultdict(list)
    for i, r in enumerate(rows):
        by_run[r["run_id"]].append(r)
    seq_of = {}
    next_fam = {}
    next_model = {}
    for rid, rs in by_run.items():
        rs.sort(key=lambda x: x["event_index"])
        seq_of[rid] = [fam_of(r["raw_action"]) if r["role"] == "execute" else "non_execute"
                       for r in rs]
        prev_rt = prev2_rt = 0.0
        prev_fam = None
        for k, r in enumerate(rs):
            r["_prev_runtime"] = prev_rt
            r["_prev2_runtime"] = prev2_rt
            r["_prev_fam"] = prev_fam
            prev2_rt = prev_rt
            prev_rt = float(r.get("runtime_ms") or 0)
            prev_fam = fam_of(r["raw_action"]) if r["role"] == "execute" else None
            if k + 1 < len(rs):
                a = seq_of[rid][k + 1]
                next_fam[id(r)] = fam_of(rs[k + 1]["raw_action"]) if a != "non_execute" else None
                next_model[id(r)] = rs[k + 1]["model_id"]
            else:
                next_fam[id(r)] = None
                next_model[id(r)] = None

    max_ev = max(r["event_index"] for r in rows)
    max_rt = max((float(r.get("runtime_ms") or 0) for r in rows), default=1.0)
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

    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "validation"]
    te = [r for r in rows if r["split"] == "test"]

    gmed2 = defaultdict(list)
    gmed3 = defaultdict(list)
    for r in tr:
        gmed2[(r["_prev_fam"], r["model_id"])].append(r["runtime_ms"])
        gmed3[(r["_prev_fam"], r["model_id"], min(r["event_index"], 9) // 3)].append(r["runtime_ms"])
    cross2 = {k: st.median(v) for k, v in gmed2.items()}
    cross3 = {k: st.median(v) for k, v in gmed3.items()}
    g = st.median([r["runtime_ms"] for r in tr if (r.get("runtime_ms") or 0) > 0] or [0])

    def onehot(vals, size):
        v = np.zeros(size, dtype=np.float32)
        v[vals] = 1.0
        return v

    def feat(r, last_fam):
        attr = MODEL_ATTR.get(r["model_id"], (0, 0, 0))
        cross = cross2.get((last_fam, r["model_id"]), g) if last_fam else g
        c3 = cross3.get((last_fam, r["model_id"], min(r["event_index"], 9) // 3), g) if last_fam else g
        return np.concatenate([
            onehot([model_idx[r["model_id"]]], len(models)),
            onehot([base_idx[r["baseline"]]], len(bases)),
            onehot([qt_idx[r["question_type"]]], len(qts)),
            onehot([exe_idx[r["_exe"]]], len(exes)),
            onehot([stk_idx[r["_stack"]]], len(stacks)),
            onehot([dom_idx[r["_dom"]]], len(doms)),
            np.array([r["event_index"] / max_ev,
                      1.0 if r.get("model_resident_before") else 0.0,
                      1.0 if r.get("yolo_batch") else 0.0,
                      float(attr[0]), float(attr[1]), float(attr[2]),
                      cross, c3,
                      r["_prev_runtime"] / max_rt, r["_prev2_runtime"] / max_rt],
                     dtype=np.float32),
            onehot([fam_idx[last_fam]], 6) if last_fam else np.zeros(6, dtype=np.float32),
        ])

    X_all = []
    seq_windows = []
    W = 5
    for r in rows:
        s = seq_of[r["run_id"]]
        idx = r["event_index"]
        last_fam = None
        win = []
        for a in s[:idx]:
            if a != "non_execute":
                win.append(fam_idx[a])
                last_fam = a
        win = win[-W:]
        seq_windows.append(win + [6] * (W - len(win)))
        X_all.append(feat(r, last_fam))
    X_all = np.stack(X_all)
    S = np.array(seq_windows, dtype=np.int64)
    ns = len(models) + len(bases) + len(qts) + len(exes) + len(stacks) + len(doms)
    ne = ns + 10
    tr_idx = [i for i, r in enumerate(rows) if r["split"] == "train"]
    mu = X_all[tr_idx, ns:ne].mean(0)
    sd = X_all[tr_idx, ns:ne].std(0).clip(min=1e-6)
    X_all[:, ns:ne] = (X_all[:, ns:ne] - mu) / sd

    y_role = np.array([role_idx[r["next_role"]] if r["next_role"] in role_idx else 0
                       for r in rows], dtype=np.int64)
    nf_arr = np.array([fam_idx[next_fam[id(r)]] if next_fam[id(r)] else -1 for r in rows],
                      dtype=np.int64)
    nm_arr = np.array([next_model[id(r)] or "" for r in rows], dtype=object)

    idx = {id(r): i for i, r in enumerate(rows)}
    itr = [idx[id(r)] for r in tr]
    iva = [idx[id(r)] for r in va]
    ite = [idx[id(r)] for r in te]
    nf_tr = [i for i in itr if nf_arr[i] >= 0]
    nf_va = [i for i in iva if nf_arr[i] >= 0]
    nf_te = [i for i in ite if nf_arr[i] >= 0]

    import torch
    import torch.nn as nn
    import xgboost as xgb
    from sklearn.model_selection import GroupKFold
    torch.manual_seed(42)
    DIM = X_all.shape[1]

    def align6(pb, classes):
        pb6 = np.zeros((pb.shape[0], 6), dtype=np.float32)
        for c, col in enumerate(classes):
            pb6[:, col] = pb[:, c]
        return pb6

    # ---- XGB 动作族:5 折 GroupKFold OOF logits(nf_tr)+ 全量 test/val logits ----
    groups = np.array([rows[i]["video_id"] for i in nf_tr])
    kf = GroupKFold(n_splits=5)
    oof_logits = np.zeros((len(X_all), 6), dtype=np.float32)
    nf_tr_arr = np.array(nf_tr)
    for tr_idx, va_idx in kf.split(nf_tr_arr, groups=groups):
        m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
        m.fit(X_all[nf_tr_arr[tr_idx]], nf_arr[nf_tr_arr[tr_idx]])
        pb = align6(m.predict_proba(X_all[nf_tr_arr[va_idx]]), m.classes_)
        oof_logits[nf_tr_arr[va_idx]] = np.log(pb.clip(1e-6))
    m_full = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
    m_full.fit(X_all[nf_tr], nf_arr[nf_tr])
    te_xgb_logits = np.log(align6(m_full.predict_proba(X_all[nf_te]), m_full.classes_).clip(1e-6))
    va_xgb_logits = np.log(align6(m_full.predict_proba(X_all[nf_va]), m_full.classes_).clip(1e-6))

    # ---- 角色 XGB(整体用)----
    l1 = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
    l1.fit(X_all[itr], y_role[itr])
    pr_role = l1.predict(X_all[ite])

    xa = torch.from_numpy(X_all)
    sa = torch.from_numpy(S)

    class GRURes(nn.Module):
        def __init__(self, use_logits):
            super().__init__()
            self.use_logits = use_logits
            self.emb = nn.Embedding(7, 16)
            self.rnn = nn.GRU(16, 64, batch_first=True)
            d_in = 64 + DIM + (6 if use_logits else 0)
            self.head = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 6))
        def forward(self, s, x, lg=None):
            e = self.emb(s)
            _, h = self.rnn(e)
            z = [h[-1], x] + ([lg] if self.use_logits and lg is not None else [])
            return self.head(torch.cat(z, dim=1))

    def train_net(net, lg_tr, lg_va, epochs=120):
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        best, best_state = -1, None
        for ep in range(epochs):
            net.train(); opt.zero_grad()
            logits = lg_tr + net(sa[nf_tr], xa[nf_tr], lg_tr)
            loss = nn.functional.cross_entropy(logits, torch.from_numpy(nf_arr[nf_tr]))
            loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                lg_v = lg_va + net(sa[nf_va], xa[nf_va], lg_va)
                acc = (lg_v.argmax(1) == torch.from_numpy(nf_arr[nf_va])).float().mean().item()
            if acc > best:
                best, best_state = acc, {k: v.clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_state); net.eval()
        return net, best

    results = {"models": {}}
    xgb_pb = np.exp(te_xgb_logits)
    xgb_pb = xgb_pb / xgb_pb.sum(1, keepdims=True)

    # 对照:纯 GRU(无残差)
    net_gru = GRURes(use_logits=False)
    lg0_tr = torch.zeros(len(nf_tr), 6)
    lg0_va = torch.zeros(len(nf_va), 6)
    net_gru, gv = train_net(net_gru, lg0_tr, lg0_va)
    with torch.no_grad():
        gru_pb = torch.softmax(net_gru(sa[nf_te], xa[nf_te]), 1).numpy()
    print(f"纯GRU 单级={float((gru_pb.argmax(1)==nf_arr[nf_te]).mean()):.4f} (val={gv:.3f})")

    # 残差:GRU(logits 条件)
    net_res = GRURes(use_logits=True)
    lg_tr = torch.from_numpy(oof_logits[np.array(nf_tr)])
    lg_va = torch.from_numpy(va_xgb_logits)
    net_res, rv = train_net(net_res, lg_tr, lg_va)
    with torch.no_grad():
        res_logits = torch.from_numpy(te_xgb_logits) + net_res(sa[nf_te], xa[nf_te], torch.from_numpy(te_xgb_logits))
        res_pb = torch.softmax(res_logits, 1).numpy()
    print(f"XGB+GRU残差 单级={float((res_pb.argmax(1)==nf_arr[nf_te]).mean()):.4f} (val={rv:.3f})")

    # ---- 长尾指标 ----
    def longtail(name, pb):
        yp = pb.argmax(1)
        from collections import Counter
        recall = {}
        for c in range(6):
            mask = nf_arr[nf_te] == c
            if mask.sum() == 0:
                recall[str(c)] = None
            else:
                recall[str(c)] = round(float((yp[mask] == c).sum() / mask.sum()), 4)
        mf1 = round(float(__import__('sklearn').metrics.f1_score(
            nf_arr[nf_te], yp, average='macro', zero_division=0)), 4)
        cost = np.ones((6, 6))
        cost[2, 0] = 5.0   # temporal_ops -> select_frames 高成本
        cost[4, 0] = 5.0   # detect -> select_frames
        cost[2, 1] = 3.0   # temporal -> visual_qa
        wrong = yp != nf_arr[nf_te]
        c_err = float((cost[nf_arr[nf_te], yp][wrong]).sum() / len(nf_te))
        results["models"][name] = {"top1": round(float((yp == nf_arr[nf_te]).mean()), 4),
                                   "macro_f1": mf1, "cost_err": round(c_err, 4),
                                   "recall": recall}
        print(f"{name:20s} top1={results['models'][name]['top1']:.4f} macroF1={mf1:.4f} cost_err={c_err:.4f}")
        print(f"    recall={recall}")

    longtail("XGB", xgb_pb)
    longtail("GRU", gru_pb)
    longtail("XGB+GRU残差", res_pb)

    # ---- 整体(角色 XGB + fam 概率)----
    def overall(name, p_fam_te):
        p_map = {i: p_fam_te[k] for k, i in enumerate(nf_te)}
        p_full = np.stack([p_map.get(i, np.zeros(6)) for i in ite])
        m = eval_pipeline(ite, y_role, nf_arr, nm_arr, pr_role, p_full)
        results["models"][name + "_整体"] = m
        print(f"{name:20s} 整体 {m}")

    overall("XGB", xgb_pb)
    overall("GRU", gru_pb)
    overall("XGB+GRU残差", res_pb)

    with open(D + "results/processed/nn_residual_model_20260809.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("已写入 results/processed/nn_residual_model_20260809.json")


if __name__ == "__main__":
    main()
