#!/usr/bin/env python3
"""资源预测器对比实验(对应远端 docs/predictor_selection_plan_20260805.md)

模型:R0 分位表 / R1 分位表+backoff / R5 roofline / R2 LightGBM / R3 MLP / R4 ProD式分布
split:video_id -> train/validation/test(从 prefix 数据映射)
指标:MAE / RMSE / P95(log1p 逆变换),整体 + 按角色×模型分桶
前置:噪声半径(同 video+baseline+planner 重复 run 同位置 runtime 相对差)
"""
import argparse
import json
import math
import statistics as st
from collections import defaultdict, Counter
import numpy as np

NODE_TO_ROLE = {
    "run_control": "init", "planner": "plan",
    "videotool_spatial": "execute", "videotool_temporal": "execute",
    "videotool_generalist": "execute", "answer_generation": "aggregate",
}


def load_rows(path):
    return [json.loads(l) for l in open(path)]


def parse_scale(s):
    fc = qic = yb = None
    if isinstance(s, dict):
        fc, qic, yb = s.get("frame_count"), s.get("qwen_image_count"), s.get("yolo_batch")
    return fc, qic, yb


def build_features(rows):
    feats, ys = [], []
    for r in rows:
        rt = r.get("runtime_ms")
        if not rt or rt <= 0:
            continue
        fc, qic, yb = parse_scale(r.get("input_scale"))
        feats.append({
            "role": NODE_TO_ROLE.get(r.get("node_type"), "other"),
            "model": r.get("model_id", "?"),
            "baseline": r.get("baseline", "?"),
            "frame_count": float(fc) if fc else 0.0,
            "qwen_image_count": float(qic) if qic else 0.0,
            "yolo_batch": float(yb) if yb else 0.0,
            "video_id": r.get("video_id"),
        })
        ys.append(float(rt))
    return feats, ys


def mae_rmse_p95(preds, ys):
    errs = [abs(p - y) for p, y in zip(preds, ys)]
    mae = sum(errs) / len(errs)
    rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
    p95 = sorted(errs)[int(len(errs) * 0.95)]
    return mae, rmse, p95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute-events", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    events = load_rows(args.compute_events)
    prefixes = load_rows(args.prefix)
    # video_id -> split 映射(prefix 里取多数)
    vsplit = defaultdict(Counter)
    for p in prefixes:
        vsplit[p["video_id"]][p.get("split")] += 1
    video_split = {v: c.most_common(1)[0][0] for v, c in vsplit.items()}
    print(f"视频 split: train={sum(1 for v in video_split.values() if v=='train')} "
          f"val={sum(1 for v in video_split.values() if v=='validation')} "
          f"test={sum(1 for v in video_split.values() if v=='test')}")

    feats_all, ys_all = build_features(events)
    print(f"有效 runtime 事件: {len(feats_all)}")

    # 前置:噪声半径(同 video+baseline+planner 重复 run 同位置 runtime 相对差)
    by_run = defaultdict(list)
    for r in events:
        if (r.get("runtime_ms") or 0) > 0:
            by_run[r["run_id"]].append(r)
    groups = defaultdict(list)
    for rid, rs in by_run.items():
        groups[(rs[0].get("video_id"), rs[0].get("baseline"),
                rs[0].get("model_id"))].append(rs)
    dup = sum(1 for v in groups.values() if len(v) >= 2)
    diffs = []
    for (v, b, m), runs in groups.items():
        if len(runs) < 2:
            continue
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                rt1 = [r.get("runtime_ms") or 0 for r in sorted(runs[i], key=lambda x: x.get("event_index", 0))]
                rt2 = [r.get("runtime_ms") or 0 for r in sorted(runs[j], key=lambda x: x.get("event_index", 0))]
                for a, c in zip(rt1, rt2):
                    if a > 0 and c > 0:
                        diffs.append(abs(a - c) / max(a, c))
    noise = {"dup_groups": dup, "n_pairs": len(diffs)}
    if diffs:
        noise["rel_diff_median"] = round(st.median(diffs), 4)
        noise["rel_diff_p90"] = round(sorted(diffs)[int(len(diffs) * 0.9)], 4)
    print(f"噪声半径: 重复组={dup} 对={len(diffs)} 相对差中位={noise.get('rel_diff_median')} p90={noise.get('rel_diff_p90')}")

    # split 划分
    tr_idx = [i for i, f in enumerate(feats_all) if video_split.get(f["video_id"]) == "train"]
    va_idx = [i for i, f in enumerate(feats_all) if video_split.get(f["video_id"]) == "validation"]
    te_idx = [i for i, f in enumerate(feats_all) if video_split.get(f["video_id"]) == "test"]
    print(f"事件划分: train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")

    def split_data(idx):
        return [feats_all[i] for i in idx], [ys_all[i] for i in idx]

    tr_f, tr_y = split_data(tr_idx)
    te_f, te_y = split_data(te_idx)

    def eval_model(name, pred_fn):
        preds = [pred_fn(f) for f in te_f]
        mae, rmse, p95 = mae_rmse_p95(preds, te_y)
        rel = mae / (sum(te_y) / len(te_y))
        print(f"  {name:28s} MAE={mae:9.1f}ms  RMSE={rmse:10.1f}  P95={p95:10.1f}  相对MAE={rel:.2%}")
        return {"name": name, "mae": round(mae, 1), "rmse": round(rmse, 1),
                "p95": round(p95, 1), "rel_mae": round(rel, 4)}

    results = {"noise_radius": noise, "n_train": len(tr_idx), "n_test": len(te_idx),
               "models": []}

    # R0/R1: 分位表(训练集拟合)
    group_med = defaultdict(list)
    for f, y in zip(tr_f, tr_y):
        group_med[(f["role"], f["model"])].append(y)
    med_by_group = {k: st.median(v) for k, v in group_med.items()}
    model_med = defaultdict(list)
    for f, y in zip(tr_f, tr_y):
        model_med[f["model"]].append(y)
    med_by_model = {k: st.median(v) for k, v in model_med.items()}
    global_med = st.median(tr_y)

    def r0(f):
        return med_by_group.get((f["role"], f["model"]), global_med)

    def r1(f):
        k = (f["role"], f["model"])
        if len(group_med.get(k, [])) >= 10:
            return med_by_group[k]
        if f["model"] in med_by_model:
            return med_by_model[f["model"]]
        return global_med

    results["models"].append(eval_model("R0 分位表(角色x模型)", r0))
    results["models"].append(eval_model("R1 分位表+backoff", r1))

    # R5: roofline(per model: log1p(runtime) ~ frame_count 线性)
    from sklearn.linear_model import LinearRegression
    roofline = {}
    for model in set(f["model"] for f in tr_f):
        X, y = [], []
        for f, yy in zip(tr_f, tr_y):
            if f["model"] == model:
                X.append([f["frame_count"], f["qwen_image_count"]])
                y.append(math.log1p(yy))
        if len(set(tuple(x) for x in X)) > 1 and len(y) >= 10:
            lr = LinearRegression().fit(np.array(X), np.array(y))
            roofline[model] = lr
    def r5(f):
        if f["model"] in roofline:
            return math.expm1(roofline[f["model"]].predict([[f["frame_count"], f["qwen_image_count"]]])[0])
        return global_med
    results["models"].append(eval_model("R5 roofline(per model)", r5))

    # 特征编码(共享)
    all_f = tr_f + te_f
    role_cat = {v: i for i, v in enumerate(sorted(set(f["role"] for f in all_f)))}
    model_cat = {v: i for i, v in enumerate(sorted(set(f["model"] for f in all_f)))}
    base_cat = {v: i for i, v in enumerate(sorted(set(f["baseline"] for f in all_f)))}

    def encode(f):
        return [role_cat[f["role"]], model_cat[f["model"]], base_cat[f["baseline"]],
                f["frame_count"], f["qwen_image_count"]]

    Xtr = np.array([encode(f) for f in tr_f], dtype=np.float32)
    Xte = np.array([encode(f) for f in te_f], dtype=np.float32)
    ytr_log = np.array([math.log1p(y) for y in tr_y], dtype=np.float32)

    # R2: LightGBM
    import lightgbm as lgb
    lgb_m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31,
                              verbose=-1, random_state=42)
    lgb_m.fit(Xtr, ytr_log)
    results["models"].append(eval_model("R2 LightGBM(log1p)", lambda f: math.expm1(
        lgb_m.predict(np.array([encode(f)], dtype=np.float32))[0])))

    # R3: MLP 回归
    import torch
    import torch.nn as nn
    torch.manual_seed(42)
    mlp = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(ytr_log).unsqueeze(1)
    mlp.train()
    for ep in range(60):
        opt.zero_grad()
        loss = lossf(mlp(xt), yt)
        loss.backward()
        opt.step()
    mlp.eval()
    with torch.no_grad():
        preds_log = mlp(torch.from_numpy(Xte)).squeeze(1).numpy()
    errs = sorted(abs(math.expm1(p) - y) for p, y in zip(preds_log, te_y))
    results["models"].append({"name": "R3 MLP(log1p)", "mae": round(sum(errs) / len(errs), 1),
                              "rmse": round(math.sqrt(sum(e * e for e in errs) / len(errs)), 1),
                              "p95": round(errs[int(len(errs) * 0.95)], 1),
                              "rel_mae": round((sum(errs) / len(errs)) / (sum(te_y) / len(te_y)), 4)})
    print(f"  R3 MLP(log1p)            MAE={results['models'][-1]['mae']}  RMSE={results['models'][-1]['rmse']}  P95={results['models'][-1]['p95']}")

    # R4: ProD 式分布输出(32 bin)
    n_bins = 32
    max_log = max(ytr_log)
    edges = np.linspace(0, max_log + 0.1, n_bins + 1)
    ybin = np.clip(np.searchsorted(edges, ytr_log, side="right") - 1, 0, n_bins - 1)
    class DistMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Dropout(0.1),
                                     nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_bins))
        def forward(self, x):
            return self.net(x)
    dm = DistMLP()
    opt2 = torch.optim.Adam(dm.parameters(), lr=1e-3)
    yb_t = torch.from_numpy(ybin).long()
    for ep in range(60):
        opt2.zero_grad()
        logits = dm(xt)
        loss = nn.functional.cross_entropy(logits, yb_t)
        loss.backward()
        opt2.step()
    dm.eval()
    with torch.no_grad():
        probs = torch.softmax(dm(torch.from_numpy(Xte)), dim=1).numpy()
    centers = (edges[:-1] + edges[1:]) / 2
    preds_dist = np.array([np.dot(p, centers) for p in probs])
    errs2 = sorted(abs(math.expm1(p) - y) for p, y in zip(preds_dist, te_y))
    results["models"].append({"name": "R4 ProD式分布(32bin)", "mae": round(sum(errs2) / len(errs2), 1),
                              "rmse": round(math.sqrt(sum(e * e for e in errs2) / len(errs2)), 1),
                              "p95": round(errs2[int(len(errs2) * 0.95)], 1),
                              "rel_mae": round((sum(errs2) / len(errs2)) / (sum(te_y) / len(te_y)), 4)})
    print(f"  R4 ProD式分布(32bin)     MAE={results['models'][-1]['mae']}  RMSE={results['models'][-1]['rmse']}  P95={results['models'][-1]['p95']}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已写入: {args.out}")


if __name__ == "__main__":
    main()
