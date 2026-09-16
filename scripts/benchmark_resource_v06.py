#!/usr/bin/env python3
"""资源预测器统一评估(对应远端 docs/resource_predictor_v0_6_plan_20260806.md)

预测目标:runtime_ms(主)/ peak_allocated_mb / load_ms
预测器:R0 全局中位 / R1 动作族×模型分位表 / R2 +backoff / R3 LightGBM /
        R4 XGBoost / R5 MLP / R6 ProD式分布 / R7 roofline / R8 KNN
输出:分位三元组(P50/P90/P99);评估:MAE/RMSE/P95 + P99 覆盖率
噪声半径参照:同 (video,baseline,model) 重复 run 同位置 runtime 相对差
"""
import argparse
import json
import math
import statistics as st
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
LLM_SET = {"Qwen3-VL-8B-Instruct", "Qwen2.5-VL-3B-Instruct"}


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def fam_of(raw):
    return FAMILY.get(raw, "other")


def executor_of(model_id):
    if model_id in LLM_SET:
        return "LLM"
    if "yolo" in str(model_id):
        return "YOLO"
    return "TOOL"


def q(vals, q_):
    if not vals:
        return None
    return float(sorted(vals)[min(len(vals) - 1, int(q_ * len(vals)))])


def metrics(preds, ys):
    errs = [abs(p - y) for p, y in zip(preds, ys)]
    mae = sum(errs) / len(errs)
    rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
    p95 = sorted(errs)[int(len(errs) * 0.95)]
    return round(mae, 1), round(rmse, 1), round(p95, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--prefix", required=True, help="prefix 数据(question 文本/视频元数据 join 来源)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_jsonl(args.dataset)
    prefixes = load_jsonl(args.prefix)
    for r in rows:
        r["_fam"] = fam_of(r.get("raw_action"))
    print(f"总行数: {len(rows)}")

    # 任务级静态特征(video_id join,prefix 来源)
    vid_meta = {}
    for p in prefixes:
        ts = p.get("task_structure") or {}
        sf = p.get("state_features") or {}
        vd = sf.get("video") if isinstance(sf, dict) else None
        vid_meta[p["video_id"]] = {
            "q_chars": (ts.get("question_chars") if isinstance(ts, dict) else None) or 0,
            "q_tokens": (ts.get("question_tokens") if isinstance(ts, dict) else None) or 0,
            "opt_count": (ts.get("option_count") if isinstance(ts, dict) else None) or 0,
            "duration": vd.get("duration_s") if isinstance(vd, dict) else None,
            "fps": vd.get("fps") if isinstance(vd, dict) else None,
        }
    for r in rows:
        m = vid_meta.get(r["video_id"], {})
        r["_q_chars"] = m.get("q_chars", 0)
        r["_q_tokens"] = m.get("q_tokens", 0)
        r["_opt_count"] = m.get("opt_count", 0)
        r["_duration"] = float(m.get("duration") or 0)
        r["_fps"] = float(m.get("fps") or 0)

    # 累计资源特征 + 上一事件 runtime(自相关,推理时可得)
    by_run = defaultdict(list)
    for i, r in enumerate(rows):
        by_run[r["run_id"]].append(r)
    for rid, rs in by_run.items():
        rs.sort(key=lambda x: x["event_index"])
        acc_steps = 0
        acc_rt = 0.0
        prev_rt = 0.0
        for r in rs:
            r["_acc_steps"] = acc_steps
            r["_acc_runtime"] = acc_rt
            r["_prev_runtime"] = prev_rt
            acc_steps += 1
            acc_rt += float(r.get("runtime_ms") or 0)
            prev_rt = float(r.get("runtime_ms") or 0)

    # 噪声半径(同 video+baseline+model 重复 run 同位置 runtime 相对差)
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run_id"]].append(r)
    groups = defaultdict(list)
    for rid, rs in by_run.items():
        groups[(rs[0].get("video_id"), rs[0].get("baseline"), rs[0].get("model_id"))].append(rs)
    diffs = []
    for (v, b, m), runs in groups.items():
        if len(runs) < 2:
            continue
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                rt1 = [r.get("runtime_ms") or 0 for r in sorted(runs[i], key=lambda x: x["event_index"])]
                rt2 = [r.get("runtime_ms") or 0 for r in sorted(runs[j], key=lambda x: x["event_index"])]
                for a, c in zip(rt1, rt2):
                    if a > 0 and c > 0:
                        diffs.append(abs(a - c) / max(a, c))
    noise = {"n_pairs": len(diffs)}
    if diffs:
        noise["rel_diff_median"] = round(st.median(diffs), 4)
    print(f"噪声半径: 相对差中位 = {noise.get('rel_diff_median')}")

    tr = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] == "test"]
    rt_rows = [r for r in rows if (r.get("runtime_ms") or 0) > 0]
    rt_tr = [r for r in rt_rows if r["split"] == "train"]
    rt_te = [r for r in rt_rows if r["split"] == "test"]
    pk_rows = [r for r in rows if (r.get("peak_allocated_mb") or 0) > 0]
    pk_tr = [r for r in pk_rows if r["split"] == "train"]
    pk_te = [r for r in pk_rows if r["split"] == "test"]
    ld_rows = [r for r in rows if (r.get("load_ms") or 0) > 0]
    ld_tr = [r for r in ld_rows if r["split"] == "train"]
    ld_te = [r for r in ld_rows if r["split"] == "test"]
    print(f"runtime>0: train={len(rt_tr)} test={len(rt_te)}; "
          f"peak>0: train={len(pk_tr)} test={len(pk_te)}; load>0: train={len(ld_tr)} test={len(ld_te)}")

    results = {"dataset": args.dataset, "noise_radius": noise, "runtime": {}, "peak": {}, "load": {}}

    # 特征编码
    fam_idx = {f: i for i, f in enumerate(FAMS)}
    models = sorted(set(r["model_id"] for r in rows))
    model_idx = {m: i for i, m in enumerate(models)}
    bases = sorted(set(r["baseline"] for r in rows))
    base_idx = {b: i for i, b in enumerate(bases)}
    qts = sorted(set(r["question_type"] for r in rows))
    qt_idx = {q: i for i, q in enumerate(qts)}
    max_ev = max(r["event_index"] for r in rows)

    def feat(r):
        fc = float(r.get("frame_count") or 0)
        max_acc_rt = max((rr.get("_acc_runtime", 0) for rr in rows), default=0)
        max_rt = max((float(rr.get("runtime_ms") or 0) for rr in rows), default=1.0)
        return [fam_idx[r["_fam"]], model_idx[r["model_id"]], base_idx[r["baseline"]],
                r["event_index"] / max_ev,
                1.0 if r.get("model_resident_before") else 0.0,
                1.0 if r.get("yolo_batch") else 0.0,
                qt_idx[r["question_type"]], fc,
                r["_q_chars"], r["_q_tokens"], r["_opt_count"],
                r["_duration"], r["_fps"],
                r["_acc_steps"], r["_acc_runtime"] / max(max_acc_rt, 1.0),
                r["_prev_runtime"] / max_rt]

    def bucket_acc(preds, ys):
        def b(v):
            return 0 if v < 1000 else (1 if v < 10000 else 2)
        hit = sum(1 for p, y in zip(preds, ys) if b(p) == b(y))
        return round(hit / len(ys), 4)

    def run_target(name, t_tr, t_te, get_y, pred_fn, out_dict):
        ys = [get_y(r) for r in t_te]
        preds = [pred_fn(r) for r in t_te]
        mae, rmse, p95 = metrics(preds, ys)
        mean_y = sum(ys) / len(ys)
        # P99 覆盖率(预留正确率)
        covered = sum(1 for r, y in zip(t_te, ys) if pred_fn(r) * 1.5 >= y)
        # 用 P90 做预留检查:每样本 P90 由分组提供?简化:P50×1.5 作为预留
        out_dict[name] = {"mae": mae, "rmse": rmse, "p95": p95,
                          "rel_mae": round(mae / mean_y, 4),
                          "bucket_acc": bucket_acc(preds, ys),
                          "reserve_cover(1.5xP50)": round(covered / len(ys), 4)}
        print(f"  {name:26s} MAE={mae:10.1f} RMSE={rmse:10.1f} P95={p95:10.1f} "
              f"rel={mae/mean_y:.1%} bucket={bucket_acc(preds, ys):.2%} cover={covered/len(ys):.2%}")

    # ===== runtime(主目标,log1p 域回归) =====
    def y_rt(r):
        return math.log1p(r["runtime_ms"])

    # R0 全局中位
    g = st.median([r["runtime_ms"] for r in rt_tr])
    run_target("R0_全局中位", rt_tr, rt_te, lambda r: r["runtime_ms"],
               lambda r: g, results["runtime"])
    # R1 动作族×模型分位表
    gmed = defaultdict(list)
    for r in rt_tr:
        gmed[(r["_fam"], r["model_id"])].append(r["runtime_ms"])
    med1 = {k: st.median(v) for k, v in gmed.items()}
    run_target("R1_族x模型分位表", rt_tr, rt_te, lambda r: r["runtime_ms"],
               lambda r: med1.get((r["_fam"], r["model_id"]), g), results["runtime"])
    # R2 +backoff
    mmed = defaultdict(list)
    for r in rt_tr:
        mmed[r["model_id"]].append(r["runtime_ms"])
    med2 = {k: st.median(v) for k, v in mmed.items()}
    def r2(r):
        k = (r["_fam"], r["model_id"])
        if len(gmed.get(k, [])) >= 10:
            return med1[k]
        return med2.get(r["model_id"], g)
    run_target("R2_分位表+backoff", rt_tr, rt_te, lambda r: r["runtime_ms"], r2, results["runtime"])

    # R7 roofline(per model: log1p(runtime) ~ frame_count)
    from sklearn.linear_model import LinearRegression
    roofline = {}
    for m in set(r["model_id"] for r in rt_tr):
        Xs, ys = [], []
        for r in rt_tr:
            if r["model_id"] == m:
                Xs.append([float(r.get("frame_count") or 0)])
                ys.append(y_rt(r))
        if len(set(tuple(x) for x in Xs)) > 1 and len(ys) >= 10:
            roofline[m] = LinearRegression().fit(np.array(Xs), np.array(ys))
    def r7(r):
        if r["model_id"] in roofline:
            return math.expm1(roofline[r["model_id"]].predict([[float(r.get("frame_count") or 0)]])[0])
        return g
    run_target("R7_roofline", rt_tr, rt_te, lambda r: r["runtime_ms"], r7, results["runtime"])

    # 特征矩阵(训练型)
    X = np.array([feat(r) for r in rows], dtype=np.float32)
    idx_tr = [rows.index(r) for r in rt_tr]
    idx_te = [rows.index(r) for r in rt_te]
    y_tr = np.array([y_rt(r) for r in rt_tr], dtype=np.float32)
    Xtr, Xte = X[idx_tr], X[idx_te]

    # R3 LightGBM
    import lightgbm as lgb
    lm = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                           verbose=-1, random_state=42).fit(Xtr, y_tr)
    def r3(r):
        return math.expm1(lm.predict(X[rows.index(r):rows.index(r) + 1])[0])
    run_target("R3_LightGBM", rt_tr, rt_te, lambda r: r["runtime_ms"], r3, results["runtime"])

    # R4 XGBoost
    import xgboost as xgb
    xm = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                          random_state=42).fit(Xtr, y_tr)
    def r4(r):
        return math.expm1(xm.predict(X[rows.index(r):rows.index(r) + 1])[0])
    run_target("R4_XGBoost", rt_tr, rt_te, lambda r: r["runtime_ms"], r4, results["runtime"])

    # R5 MLP 回归
    import torch
    import torch.nn as nn
    torch.manual_seed(42)
    mlp = nn.Sequential(nn.Linear(Xtr.shape[1], 64), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    xt = torch.from_numpy(Xtr); yt = torch.from_numpy(y_tr).unsqueeze(1)
    for ep in range(80):
        opt.zero_grad()
        loss = nn.functional.mse_loss(mlp(xt), yt)
        loss.backward(); opt.step()
    mlp.eval()
    preds_log = mlp(torch.from_numpy(Xte)).squeeze(1).detach().numpy()
    preds_mlp = [math.expm1(p) for p in preds_log]
    ys_rt = [r["runtime_ms"] for r in rt_te]
    mae, rmse, p95 = metrics(preds_mlp, ys_rt)
    results["runtime"]["R5_MLP"] = {"mae": mae, "rmse": rmse, "p95": p95,
                                    "rel_mae": round(mae / (sum(ys_rt) / len(ys_rt)), 4),
                                    "bucket_acc": bucket_acc(preds_mlp, ys_rt)}
    print(f"  R5_MLP                     MAE={mae:10.1f} RMSE={rmse:10.1f} P95={p95:10.1f} bucket={bucket_acc(preds_mlp, ys_rt):.1%}")

    # R6 ProD 式分布(32 bin)
    n_bins = 32
    max_log = float(y_tr.max())
    edges = np.linspace(0, max_log + 0.1, n_bins + 1)
    ybin = np.clip(np.searchsorted(edges, y_tr, side="right") - 1, 0, n_bins - 1)
    dm = nn.Sequential(nn.Linear(Xtr.shape[1], 64), nn.ReLU(), nn.Linear(64, n_bins))
    opt2 = torch.optim.Adam(dm.parameters(), lr=1e-3)
    yb = torch.from_numpy(ybin).long()
    for ep in range(80):
        opt2.zero_grad()
        loss = nn.functional.cross_entropy(dm(xt), yb)
        loss.backward(); opt2.step()
    dm.eval()
    with torch.no_grad():
        probs = torch.softmax(dm(torch.from_numpy(Xte)), dim=1).numpy()
    centers = (edges[:-1] + edges[1:]) / 2
    preds_dist = [math.expm1(float(np.dot(p, centers))) for p in probs]
    mae, rmse, p95 = metrics(preds_dist, ys_rt)
    results["runtime"]["R6_ProD分布"] = {"mae": mae, "rmse": rmse, "p95": p95,
                                         "rel_mae": round(mae / (sum(ys_rt) / len(ys_rt)), 4),
                                         "bucket_acc": bucket_acc(preds_dist, ys_rt)}
    print(f"  R6_ProD分布                MAE={mae:10.1f} RMSE={rmse:10.1f} P95={p95:10.1f} bucket={bucket_acc(preds_dist, ys_rt):.1%}")

    # R8 KNN
    tr_feats = np.array([feat(r) for r in rt_tr], dtype=np.float32)
    tr_rts = np.array([r["runtime_ms"] for r in rt_tr], dtype=np.float32)
    te_feats = np.array([feat(r) for r in rt_te], dtype=np.float32)
    # 简单 KNN:标准化后欧氏距离,取 top-20 中位
    mu, sd = tr_feats.mean(0), tr_feats.std(0).clip(min=1e-6)
    tn, en = (tr_feats - mu) / sd, (te_feats - mu) / sd
    preds_knn = []
    for e in en:
        d = np.abs(tn - e).sum(1)
        idx = np.argsort(d)[:20]
        preds_knn.append(float(np.median(tr_rts[idx])))
    mae, rmse, p95 = metrics(preds_knn, ys_rt)
    results["runtime"]["R8_KNN"] = {"mae": mae, "rmse": rmse, "p95": p95,
                                    "rel_mae": round(mae / (sum(ys_rt) / len(ys_rt)), 4),
                                    "bucket_acc": bucket_acc(preds_knn, ys_rt)}
    print(f"  R8_KNN                     MAE={mae:10.1f} RMSE={rmse:10.1f} P95={p95:10.1f} bucket={bucket_acc(preds_knn, ys_rt):.1%}")

    # ===== peak VRAM(次要,有值行) =====
    if pk_te:
        gp = defaultdict(list)
        for r in pk_tr:
            gp[(r["_fam"], r["model_id"])].append(r["peak_allocated_mb"])
        mp = {k: st.median(v) for k, v in gp.items()}
        gg = st.median([r["peak_allocated_mb"] for r in pk_tr])
        preds = [mp.get((r["_fam"], r["model_id"]), gg) for r in pk_te]
        ys = [r["peak_allocated_mb"] for r in pk_te]
        mae, rmse, p95 = metrics(preds, ys)
        results["peak"]["R1_族x模型分位表"] = {"mae": mae, "rmse": rmse, "p95": p95}
        print(f"  [peak] R1 分位表: MAE={mae:.0f}MB RMSE={rmse:.0f} P95={p95:.0f}")

    # ===== load_ms(次要,冷启动) =====
    if ld_te:
        gl = defaultdict(list)
        for r in ld_tr:
            gl[(r["_fam"], r["model_id"])].append(r["load_ms"])
        ml = {k: st.median(v) for k, v in gl.items()}
        glb = st.median([r["load_ms"] for r in ld_tr])
        preds = [ml.get((r["_fam"], r["model_id"]), glb) for r in ld_te]
        ys = [r["load_ms"] for r in ld_te]
        mae, rmse, p95 = metrics(preds, ys)
        results["load"]["R1_族x模型分位表"] = {"mae": mae, "rmse": rmse, "p95": p95}
        print(f"  [load] R1 分位表: MAE={mae:.0f}ms RMSE={rmse:.0f} P95={p95:.0f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已写入: {args.out}")


if __name__ == "__main__":
    main()
