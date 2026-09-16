#!/usr/bin/env python3
"""Part B:NN 行为预测三形态对比(最优特征集:全关系 + prev2 + cross3)

① 单任务 MLP(execute 子集,动作族)
② 分层多任务(共享编码 + 角色头 + 动作族头,λ 0.5/1.0/2.0)
③ 级联 NN(角色 MLP → 概率 → 动作族 MLP)
④ 角色级单任务 MLP(全量 4 类)
对照:XGBoost(同特征)
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
ROLES = ["plan", "execute", "aggregate", "terminate"]
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


def top1(y, pb):
    y = np.asarray(y)
    return float((np.argmax(pb, 1) == y).mean())


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
    for rid, rs in by_run.items():
        rs.sort(key=lambda x: x["event_index"])
        prev_rt = prev2_rt = 0.0
        prev_fam = None
        for k, r in enumerate(rs):
            r["_prev_runtime"] = prev_rt
            r["_prev2_runtime"] = prev2_rt
            r["_prev_fam"] = prev_fam
            prev2_rt = prev_rt
            prev_rt = float(r.get("runtime_ms") or 0)
            prev_fam = r["_fam"]

    max_ev = max(r["event_index"] for r in rows)
    max_rt = max((float(r.get("runtime_ms") or 0) for r in rows), default=1.0)
    fam_idx = {f: i for i, f in enumerate(FAMS)}
    role_idx = {r: i for i, r in enumerate(ROLES)}
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

    # 统计特征(train 拟合)
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

    # 特征(全关系 + prev2 + cross3)
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

    # execute 子集 + 全量(角色)
    exe_rows = [r for r in rows if r["role"] == "execute"]
    by_e = defaultdict(list)
    for r in exe_rows:
        by_e[r["run_id"]].append(r)
    for rs in by_e.values():
        rs.sort(key=lambda x: x["event_index"])
    seq_e = {rid: [r["_fam"] for r in rs] for rid, rs in by_e.items()}

    def build_x(rs):
        Xs = []
        for r in rs:
            if r["role"] == "execute":
                s = seq_e[r["run_id"]]
                pos = s.index(r["_fam"]) if r["_fam"] in s else len(s) - 1
                last = s[pos - 1] if pos >= 1 else None
            else:
                last = None
            Xs.append(feat(r, last))
        return np.stack(Xs)

    X_all = build_x(rows)
    X_exe = build_x(exe_rows)
    # z-score(数值列)
    ns = len(models) + len(bases) + len(qts) + len(exes) + len(stacks) + len(doms)
    ne = ns + 10
    mu = X_all[0:len(tr), ns:ne].mean(0)
    sd = X_all[0:len(tr), ns:ne].std(0).clip(min=1e-6)
    X_all[:, ns:ne] = (X_all[:, ns:ne] - mu) / sd
    X_exe[:, ns:ne] = (X_exe[:, ns:ne] - mu) / sd

    y_role = np.array([role_idx[r["next_role"]] if r["next_role"] in role_idx else 0
                       for r in rows], dtype=np.int64)
    y_fam = np.array([fam_idx[r["_fam"]] for r in exe_rows], dtype=np.int64)

    idx = {id(r): i for i, r in enumerate(rows)}
    eidx = {id(r): i for i, r in enumerate(exe_rows)}
    itr = [idx[id(r)] for r in tr]
    ite = [idx[id(r)] for r in te]
    etr = [eidx[id(r)] for r in exe_rows if r["split"] == "train"]
    eva = [eidx[id(r)] for r in exe_rows if r["split"] == "validation"]
    ete = [eidx[id(r)] for r in exe_rows if r["split"] == "test"]
    rtr = [idx[id(r)] for r in tr]
    rva = [idx[id(r)] for r in va]
    rte = [idx[id(r)] for r in te]

    import torch
    import torch.nn as nn
    torch.manual_seed(42)
    W = 128

    results = {"n_feat": int(X_all.shape[1]), "models": {}}

    def report(name, pb, y, is_fam=True):
        t1 = top1(y, pb)
        results["models"][name] = {"top1": round(t1, 4)}
        print(f"  {name:28s} Top1={t1:.4f}")

    # XGBoost 对照(标签版,同特征信息)
    import xgboost as xgb
    base_idx2 = base_idx; qt_idx2 = qt_idx; exe_idx2 = exe_idx; stk_idx2 = stk_idx; dom_idx2 = dom_idx
    mid_idx2 = model_idx; fmid2 = fam_idx
    Xt = np.array([[mid_idx2[r["model_id"]], base_idx2[r["baseline"]], qt_idx2[r["question_type"]],
                    exe_idx2[r["_exe"]], stk_idx2[r["_stack"]], dom_idx2[r["_dom"]],
                    r["event_index"] / max_ev, r["_prev_runtime"] / max_rt]
                   for r in exe_rows], dtype=np.float32)
    last_col = np.zeros((len(exe_rows), 6), dtype=np.float32)
    for i, r in enumerate(exe_rows):
        s = seq_e[r["run_id"]]
        pos = s.index(r["_fam"]) if r["_fam"] in s else len(s) - 1
        if pos >= 1:
            last_col[i, fam_idx[s[pos - 1]]] = 1.0
    Xt = np.hstack([Xt, last_col])
    xgb_m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                              random_state=42).fit(Xt[etr], y_fam[etr])
    report("XGBoost(最优特征)", xgb_m.predict_proba(Xt[ete]), y_fam[ete])

    # ① 单任务 MLP
    xt = torch.from_numpy(X_exe[etr]); yt = torch.from_numpy(y_fam[etr])
    xv = torch.from_numpy(X_exe[eva]); yv = torch.from_numpy(y_fam[eva])
    net = nn.Sequential(nn.Linear(X_exe.shape[1], W), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(W, W), nn.ReLU(), nn.Linear(W, 6))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best, best_state = -1, None
    for ep in range(100):
        net.train(); opt.zero_grad()
        loss = nn.functional.cross_entropy(net(xt), yt)
        loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            acc = (net(xv).argmax(1) == yv).float().mean().item()
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        pb = torch.softmax(net(torch.from_numpy(X_exe[ete])), dim=1).numpy()
    report("① 单任务 MLP", pb, y_fam[ete])

    # ③ 级联 NN:角色 MLP(全量)→ 概率 → 动作族 MLP
    xa = torch.from_numpy(X_all)
    yr = torch.from_numpy(y_role)
    rnet = nn.Sequential(nn.Linear(X_all.shape[1], W), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(W, W), nn.ReLU(), nn.Linear(W, 4))
    optr = torch.optim.Adam(rnet.parameters(), lr=1e-3)
    best, best_state = -1, None
    for ep in range(80):
        rnet.train(); optr.zero_grad()
        loss = nn.functional.cross_entropy(rnet(xa[rtr]), yr[rtr])
        loss.backward(); optr.step()
        rnet.eval()
        with torch.no_grad():
            acc = (rnet(xa[rva]).argmax(1) == yr[rva]).float().mean().item()
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in rnet.state_dict().items()}
    rnet.load_state_dict(best_state); rnet.eval()
    with torch.no_grad():
        role_pb = torch.softmax(rnet(xa[rte]), 1)
    role_top1 = top1(yr[rte], role_pb.numpy())
    results["models"]["角色级单任务MLP"] = {"top1": round(role_top1, 4)}
    print(f"  {'④ 角色级单任务 MLP':28s} Top1={role_top1:.4f}")

    # 级联:动作族 MLP 输入加角色概率(execute 行)
    with torch.no_grad():
        role_prob = torch.softmax(rnet(xa), 1).numpy()
    X_cas = np.hstack([X_exe, role_prob[[idx[id(r)] for r in exe_rows]]])
    xs = ns
    mu2 = X_cas[etr, xs + 10:].mean(0)
    sd2 = X_cas[etr, xs + 10:].std(0).clip(min=1e-6)
    X_cas[:, xs + 10:] = (X_cas[:, xs + 10:] - mu2) / sd2
    xc = torch.from_numpy(X_cas)
    cnet = nn.Sequential(nn.Linear(X_cas.shape[1], W), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(W, W), nn.ReLU(), nn.Linear(W, 6))
    optc = torch.optim.Adam(cnet.parameters(), lr=1e-3)
    best, best_state = -1, None
    for ep in range(100):
        cnet.train(); optc.zero_grad()
        loss = nn.functional.cross_entropy(cnet(xc[etr]), yt)
        loss.backward(); optc.step()
        cnet.eval()
        with torch.no_grad():
            acc = (cnet(xc[eva]).argmax(1) == yv).float().mean().item()
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in cnet.state_dict().items()}
    cnet.load_state_dict(best_state); cnet.eval()
    with torch.no_grad():
        pb = torch.softmax(cnet(xc[ete]), dim=1).numpy()
    report("③ 级联 NN(角色概率)", pb, y_fam[ete])

    # ② 分层多任务:共享编码 + 角色头 + 动作族头(λ)
    for lam in (0.5, 1.0, 2.0):
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
        fmask_tr = torch.from_numpy((y_fam >= 0).astype(np.float32)) if False else None
        # 动作族监督只对 execute 行;这里全量行里 execute 行的 fam 标签通过 eidx 对齐
        fam_label = np.full(len(rows), -1, dtype=np.int64)
        for i, r in enumerate(exe_rows):
            fam_label[idx[id(r)]] = fam_idx[r["_fam"]]
        fl = torch.from_numpy(fam_label)
        m_tr = torch.from_numpy((fam_label >= 0).astype(np.float32))
        best, best_state = -1, None
        for ep in range(80):
            mt.train(); optm.zero_grad()
            rp, fp = mt(xa[rtr])
            loss = (nn.functional.cross_entropy(rp, yr[rtr])
                    + lam * (nn.functional.cross_entropy(fp, fl[rtr].clamp(0, 5), reduction="none")
                             * m_tr[rtr]).mean())
            loss.backward(); optm.step()
            mt.eval()
            with torch.no_grad():
                rp_v, _ = mt(xa[rva])
                acc = (rp_v.argmax(1) == yr[rva]).float().mean().item()
            if acc > best:
                best, best_state = acc, {k: v.clone() for k, v in mt.state_dict().items()}
        mt.load_state_dict(best_state); mt.eval()
        with torch.no_grad():
            _, fp_all = mt(xa)
            pb = torch.softmax(fp_all[[idx[id(r)] for r in exe_rows if r["split"] == "test"]], 1).numpy()
        report(f"② 多任务 λ={lam}", pb, y_fam[ete])

    with open(D + "results/processed/feat_round2_partB_20260806.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("写入 results/processed/feat_round2_partB_20260806.json")


if __name__ == "__main__":
    main()
