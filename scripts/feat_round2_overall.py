#!/usr/bin/env python3
"""行为侧整体输出评估(最优特征集,next 语义,全量 test 1,520 行)

四模型对比:
  XGBoost:角色 XGB + 动作族 XGB,级联触发
  ① 单任务 MLP + 角色 XGB:动作族级用 NN
  ③ 级联 NN:角色 MLP → 概率 → 动作族 MLP
  ② 多任务 NN:单模型双头(λ=2.0)
指标:联合准确率 join / 执行者路由 / 资源分层
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


def eval_pipeline(te, y_role_next, next_fam_arr, next_model_arr, p_role, p_fam_prob):
    n = len(te)
    join = route = layer = 0
    for i, j in enumerate(te):
        tr_ = y_role_next[j]
        tf_ = next_fam_arr[j]
        pr = p_role[i]
        if pr == 1 and max(p_fam_prob[i]) >= 0.5:
            pf = int(np.argmax(p_fam_prob[i]))
        else:
            pf = None
        # join
        if tf_ >= 0:
            if pr == 1 and pf == tf_:
                join += 1
        else:
            if pr == tr_:
                join += 1
        # route
        ex_pred = EXECUTOR.get(FAMS[pf]) if pf is not None else None
        ex_true = executor_of(next_model_arr[j]) if tf_ >= 0 else None
        if ex_pred == ex_true:
            route += 1
        # layer
        l_true = (FAMS[tf_] if tf_ >= 0 else None, ex_true)
        l_pred = (FAMS[pf] if pf is not None else None, ex_pred)
        if l_true == l_pred:
            layer += 1
    return {"join": round(join / n, 4), "route": round(route / n, 4),
            "layer": round(layer / n, 4)}


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

    # run 序列(全量事件,fam 或 non_execute)+ next 语义
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

    # 统计特征(train)
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

    # 全量特征(每行的 last_fam = 当前行之前的 execute 族)
    X_all = []
    for r in rows:
        s = seq_of[r["run_id"]]
        idx = r["event_index"]
        last_fam = None
        for a in s[max(0, idx - 3):idx]:
            if a != "non_execute":
                last_fam = a
        X_all.append(feat(r, last_fam))
    X_all = np.stack(X_all)
    ns = len(models) + len(bases) + len(qts) + len(exes) + len(stacks) + len(doms)
    ne = ns + 10
    mu = X_all[0:len(tr), ns:ne].mean(0)
    sd = X_all[0:len(tr), ns:ne].std(0).clip(min=1e-6)
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
    # 动作族训练集:next 是 execute 的行
    nf_tr = [i for i in itr if nf_arr[i] >= 0]
    nf_va = [i for i in iva if nf_arr[i] >= 0]
    nf_te = [i for i in ite if nf_arr[i] >= 0]
    print(f"全量 train={len(tr)} va={len(va)} te={len(te)}; 动作族行 train={len(nf_tr)} va={len(nf_va)} te={len(nf_te)}")

    import torch
    import torch.nn as nn
    import xgboost as xgb
    torch.manual_seed(42)
    W = 128

    results = {"n_feat": int(X_all.shape[1]), "models": {}}
    xa = torch.from_numpy(X_all)

    # XGBoost(角色 + 动作族,级联)
    l1 = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                           random_state=42).fit(X_all[itr], y_role[itr])
    l2 = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                           random_state=42).fit(X_all[nf_tr], nf_arr[nf_tr])
    pr = l1.predict_proba(X_all[ite])
    pf_raw = l2.predict_proba(X_all[nf_te]) if len(nf_te) else np.zeros((0, 6))
    pf_full = np.zeros((len(nf_te), 6))
    for k, cls in enumerate(l2.classes_):
        pf_full[:, int(cls)] = pf_raw[:, k]
    pf_map = {i: pf_full[k] for k, i in enumerate(nf_te)}
    p_fam = np.stack([pf_map.get(i, np.zeros(6)) for i in ite])
    results["models"]["XGBoost(级联)"] = eval_pipeline(ite, y_role, nf_arr, nm_arr, pr.argmax(1), p_fam)
    print(f"  XGBoost(级联)        {results['models']['XGBoost(级联)']}")

    # 角色 MLP(级联 NN 与混合共用)
    yr_t = torch.from_numpy(y_role)
    rnet = nn.Sequential(nn.Linear(X_all.shape[1], W), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(W, W), nn.ReLU(), nn.Linear(W, 4))
    optr = torch.optim.Adam(rnet.parameters(), lr=1e-3)
    best, best_state = -1, None
    for ep in range(80):
        rnet.train(); optr.zero_grad()
        loss = nn.functional.cross_entropy(rnet(xa[itr]), yr_t[itr])
        loss.backward(); optr.step()
        rnet.eval()
        with torch.no_grad():
            acc = (rnet(xa[iva]).argmax(1) == yr_t[iva]).float().mean().item()
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in rnet.state_dict().items()}
    rnet.load_state_dict(best_state); rnet.eval()
    with torch.no_grad():
        rp_all = torch.softmax(rnet(xa), 1).numpy()

    # ① 单任务 MLP(动作族,next 语义)
    net = nn.Sequential(nn.Linear(X_all.shape[1], W), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(W, W), nn.ReLU(), nn.Linear(W, 6))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    nf_t = torch.from_numpy(nf_arr)
    best, best_state = -1, None
    for ep in range(100):
        net.train(); opt.zero_grad()
        loss = nn.functional.cross_entropy(net(xa[nf_tr]), nf_t[nf_tr])
        loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            acc = (net(xa[nf_va]).argmax(1) == nf_t[nf_va]).float().mean().item()
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        mlp_fam = torch.softmax(net(xa), 1).numpy()
    pr_mlp = rp_all[ite].argmax(1)  # 角色用 MLP?混合版:角色用 XGB 或 MLP。①定义:角色 XGB + 动作族 MLP
    # ① 用角色 XGB + 动作族 MLP
    results["models"]["① 单任务MLP(角色XGB)"] = eval_pipeline(
        ite, y_role, nf_arr, nm_arr, pr.argmax(1),
        np.array([mlp_fam[i] for i in ite]))
    print(f"  ① 单任务MLP+角色XGB  {results['models']['① 单任务MLP(角色XGB)']}")

    # ③ 级联 NN:角色 MLP 概率 → 动作族 MLP
    X_cas = np.hstack([X_all, rp_all])
    ns2 = ne
    mu2 = X_cas[itr, ns2:].mean(0)
    sd2 = X_cas[itr, ns2:].std(0).clip(min=1e-6)
    X_cas[:, ns2:] = (X_cas[:, ns2:] - mu2) / sd2
    xc = torch.from_numpy(X_cas)
    cnet = nn.Sequential(nn.Linear(X_cas.shape[1], W), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(W, W), nn.ReLU(), nn.Linear(W, 6))
    optc = torch.optim.Adam(cnet.parameters(), lr=1e-3)
    best, best_state = -1, None
    for ep in range(100):
        cnet.train(); optc.zero_grad()
        loss = nn.functional.cross_entropy(cnet(xc[nf_tr]), nf_t[nf_tr])
        loss.backward(); optc.step()
        cnet.eval()
        with torch.no_grad():
            acc = (cnet(xc[nf_va]).argmax(1) == nf_t[nf_va]).float().mean().item()
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in cnet.state_dict().items()}
    cnet.load_state_dict(best_state); cnet.eval()
    with torch.no_grad():
        cas_fam = torch.softmax(cnet(xc), 1).numpy()
    results["models"]["③ 级联NN"] = eval_pipeline(
        ite, y_role, nf_arr, nm_arr, rp_all[ite].argmax(1),
        np.array([cas_fam[i] for i in ite]))
    print(f"  ③ 级联NN             {results['models']['③ 级联NN']}")

    # ② 多任务 NN(λ=2.0,角色头全量 + 动作族头 next-execute 行)
    class MT(nn.Module):
        def __init__(self):
            super().__init__()
            self.back = nn.Sequential(nn.Linear(X_all.shape[1], W), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(W, W), nn.ReLU())
            self.rh = nn.Linear(W, 4)
            self.fh = nn.Sequential(nn.Linear(W, 64), nn.ReLU(), nn.Linear(64, 6))
        def forward(self, x):
            h = self.back(x)
            return self.rh(h), self.fh(h)
    mt = MT()
    optm = torch.optim.Adam(mt.parameters(), lr=1e-3)
    m_tr = torch.from_numpy((nf_arr >= 0).astype(np.float32))
    lam = 2.0
    best, best_state = -1, None
    for ep in range(80):
        mt.train(); optm.zero_grad()
        rp, fp = mt(xa[itr])
        loss = (nn.functional.cross_entropy(rp, yr_t[itr])
                + lam * (nn.functional.cross_entropy(fp, nf_t[itr].clamp(0, 5), reduction="none")
                         * m_tr[itr]).mean())
        loss.backward(); optm.step()
        mt.eval()
        with torch.no_grad():
            rp_v, _ = mt(xa[iva])
            acc = (rp_v.argmax(1) == yr_t[iva]).float().mean().item()
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in mt.state_dict().items()}
    mt.load_state_dict(best_state); mt.eval()
    with torch.no_grad():
        rp_mt, fp_mt = mt(xa)
        rp_mt = torch.softmax(rp_mt, 1).numpy()
        fp_mt = torch.softmax(fp_mt, 1).numpy()
    results["models"]["② 多任务NN λ=2"] = eval_pipeline(
        ite, y_role, nf_arr, nm_arr, rp_mt[ite].argmax(1),
        np.array([fp_mt[i] for i in ite]))
    print(f"  ② 多任务NN λ=2       {results['models']['② 多任务NN λ=2']}")

    with open(D + "results/processed/feat_round2_overall_20260806.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("写入 results/processed/feat_round2_overall_20260806.json")


if __name__ == "__main__":
    main()
