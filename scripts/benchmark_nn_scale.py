#!/usr/bin/env python3
"""NN 参数量消融:同结构 MLP 扫隐藏层宽度 32/128/512/2048,报告参数量与指标

行为侧:动作族预测(全关系特征 one-hot+std),指标 Top-1,对照 XGBoost 0.8505
资源侧:runtime 回归(增强特征),指标 rel MAE,对照 XGBoost 13.7%
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


def n_params(model):
    return sum(p.numel() for p in model.parameters())


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
            "q_chars": (ts.get("question_chars") if isinstance(ts, dict) else None) or 0,
            "q_tokens": (ts.get("question_tokens") if isinstance(ts, dict) else None) or 0,
            "opt_count": (ts.get("option_count") if isinstance(ts, dict) else None) or 0,
            "domain": (ts.get("domain") if isinstance(ts, dict) else None) or "unknown",
            "duration": ((p.get("state_features") or {}).get("video") or {}).get("duration_s"),
            "fps": ((p.get("state_features") or {}).get("video") or {}).get("fps"),
        }
    for r in rows:
        m = vid_meta.get(r["video_id"], {})
        r["_q_chars"] = m.get("q_chars", 0)
        r["_q_tokens"] = m.get("q_tokens", 0)
        r["_opt_count"] = m.get("opt_count", 0)
        r["_dom"] = m.get("domain", "unknown")
        r["_duration"] = float(m.get("duration") or 0)
        r["_fps"] = float(m.get("fps") or 0)

    by_run = defaultdict(list)
    for i, r in enumerate(rows):
        by_run[r["run_id"]].append(r)
    for rid, rs in by_run.items():
        rs.sort(key=lambda x: x["event_index"])
        acc_steps = acc_rt = prev_rt = 0.0
        prev_fam = None
        for r in rs:
            r["_acc_steps"] = acc_steps
            r["_acc_runtime"] = acc_rt
            r["_prev_runtime"] = prev_rt
            r["_prev_fam"] = prev_fam
            acc_steps += 1
            acc_rt += float(r.get("runtime_ms") or 0)
            prev_rt = float(r.get("runtime_ms") or 0)
            prev_fam = r["_fam"]

    max_ev = max(r["event_index"] for r in rows)
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

    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "validation"]
    te = [r for r in rows if r["split"] == "test"]
    exe_rows = [r for r in rows if r["role"] == "execute"]
    exe_tr = [r for r in exe_rows if r["split"] == "train"]
    exe_va = [r for r in exe_rows if r["split"] == "validation"]
    exe_te = [r for r in exe_rows if r["split"] == "test"]

    # 交叉统计(上一事件族×模型,仅 train)
    gmed = defaultdict(list)
    for r in tr:
        gmed[(r["_prev_fam"], r["model_id"])].append(r["runtime_ms"])
    cross_med = {k: st.median(v) for k, v in gmed.items()}
    g = st.median([r["runtime_ms"] for r in tr])

    def onehot(vals, size):
        v = np.zeros(size, dtype=np.float32)
        v[vals] = 1.0
        return v

    # ===== 行为侧全关系特征 =====
    by_run_e = defaultdict(list)
    for r in exe_rows:
        by_run_e[r["run_id"]].append(r)
    for rs in by_run_e.values():
        rs.sort(key=lambda x: x["event_index"])
    seq_of = {rid: [r["_fam"] for r in rs] for rid, rs in by_run_e.items()}

    def beh_feat(r, last_fam):
        attr = MODEL_ATTR.get(r["model_id"], (0, 0, 0))
        cross = cross_med.get((last_fam, r["model_id"]), g) if last_fam else g
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
                      float(attr[0]), float(attr[1]), float(attr[2]), cross],
                     dtype=np.float32),
            onehot([fam_idx[last_fam]], 6) if last_fam else np.zeros(6, dtype=np.float32),
        ])

    BX = []
    for r in exe_rows:
        s = seq_of[r["run_id"]]
        pos = s.index(r["_fam"]) if r["_fam"] in s else len(s) - 1
        last = s[pos - 1] if pos >= 1 else None
        BX.append(beh_feat(r, last))
    BX = np.stack(BX)
    bs = len(models) + len(bases) + len(qts) + len(exes) + len(stacks) + len(doms)
    be = bs + 7
    mu = BX[0:len(exe_tr), bs:be].mean(0)
    sd = BX[0:len(exe_tr), bs:be].std(0).clip(min=1e-6)
    BX[:, bs:be] = (BX[:, bs:be] - mu) / sd
    By = np.array([fam_idx[r["_fam"]] for r in exe_rows], dtype=np.int64)
    eidx = {id(r): i for i, r in enumerate(exe_rows)}
    e_tr = [eidx[id(r)] for r in exe_tr]
    e_va = [eidx[id(r)] for r in exe_va]
    e_te = [eidx[id(r)] for r in exe_te]

    # ===== 资源侧全关系特征 =====
    rt = [r for r in rows if (r.get("runtime_ms") or 0) > 0]
    r_tr = [r for r in rt if r["split"] == "train"]
    r_te = [r for r in rt if r["split"] == "test"]

    def res_feat(r):
        attr = MODEL_ATTR.get(r["model_id"], (0, 0, 0))
        cross = cross_med.get((r["_prev_fam"], r["model_id"]), g)
        return np.concatenate([
            onehot([fam_idx[r["_fam"]]], len(FAMS)),
            onehot([model_idx[r["model_id"]]], len(models)),
            onehot([base_idx[r["baseline"]]], len(bases)),
            onehot([qt_idx[r["question_type"]]], len(qts)),
            onehot([dom_idx[r["_dom"]]], len(doms)),
            np.array([r["event_index"] / max_ev,
                      1.0 if r.get("model_resident_before") else 0.0,
                      1.0 if r.get("yolo_batch") else 0.0,
                      float(r.get("frame_count") or 0),
                      r["_q_chars"], r["_q_tokens"], r["_opt_count"],
                      r["_duration"], r["_fps"],
                      r["_acc_steps"], r["_acc_runtime"], r["_prev_runtime"],
                      float(attr[0]), float(attr[1]), float(attr[2]), cross],
                     dtype=np.float32),
        ])

    RX = np.stack([res_feat(r) for r in rt])
    rs_ = len(FAMS) + len(models) + len(bases) + len(qts) + len(doms)
    re_ = rs_ + 16
    mu_r = RX[0:len(r_tr), rs_:re_].mean(0)
    sd_r = RX[0:len(r_tr), rs_:re_].std(0).clip(min=1e-6)
    RX[:, rs_:re_] = (RX[:, rs_:re_] - mu_r) / sd_r
    RY = np.array([math.log1p(r["runtime_ms"]) for r in rt], dtype=np.float32)
    ridx = {id(r): i for i, r in enumerate(rt)}
    r_tr_i = [ridx[id(r)] for r in r_tr]
    r_te_i = [ridx[id(r)] for r in r_te]
    ys_rt = [r["runtime_ms"] for r in r_te]

    import torch
    import torch.nn as nn
    torch.manual_seed(42)

    results = {"behavior": {}, "resource": {}}

    # 行为侧容量消融
    for w in (32, 128, 512, 2048):
        net = nn.Sequential(nn.Linear(BX.shape[1], w), nn.ReLU(), nn.Dropout(0.2),
                            nn.Linear(w, w), nn.ReLU(), nn.Linear(w, 6))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        xt = torch.from_numpy(BX[e_tr])
        yt = torch.from_numpy(By[e_tr])
        xv = torch.from_numpy(BX[e_va])
        yv = torch.from_numpy(By[e_va])
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
            pb = net(torch.from_numpy(BX[e_te])).argmax(1).numpy()
        top1 = float((pb == By[e_te]).mean())
        np_ = n_params(net)
        results["behavior"][f"w{w}"] = {"params": int(np_), "top1": round(top1, 4)}
        print(f"行为MLP w={w:5d} 参数={np_:9,d} Top1={top1:.4f}")

    # 资源侧容量消融
    for w in (32, 128, 512, 2048):
        net = nn.Sequential(nn.Linear(RX.shape[1], w), nn.ReLU(), nn.Dropout(0.1),
                            nn.Linear(w, w), nn.ReLU(), nn.Linear(w, 1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        xt = torch.from_numpy(RX[r_tr_i])
        yt = torch.from_numpy(RY[r_tr_i]).unsqueeze(1)
        for ep in range(100):
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(xt), yt)
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pl = net(torch.from_numpy(RX[r_te_i])).squeeze(1).numpy()
        preds = [math.expm1(p) for p in pl]
        errs = [abs(p - y) for p, y in zip(preds, ys_rt)]
        mae = sum(errs) / len(errs)
        rel = mae / (sum(ys_rt) / len(ys_rt))
        np_ = n_params(net)
        results["resource"][f"w{w}"] = {"params": int(np_), "mae": round(mae, 1),
                                        "rel": round(rel, 4)}
        print(f"资源MLP w={w:5d} 参数={np_:9,d} rel={rel:.1%}")

    with open(D + "results/processed/benchmark_nn_scale_20260806.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("写入 results/processed/benchmark_nn_scale_20260806.json")


if __name__ == "__main__":
    main()
