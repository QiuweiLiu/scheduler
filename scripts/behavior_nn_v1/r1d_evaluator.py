#!/usr/bin/env python3
"""R1-evaluator:新评估器 + fixtures(behavior_nn_experiment_plan 3.7/5.1/11)

- joint:真实非 execute 只要求 role 正确;真实 execute 要求 role+fam 均正确
- route:分母只能是 execute 子集
- macro-F1:observed-class 与 fixed-schema 分开;无支持类不进 observed 分母
- support 报告
产物:results/processed/behavior_nn_v1/tests/evaluator_fixture_report.json
"""
import json
import os

D = "/root/autodl-tmp/scheduler/"
OUT = D + "results/processed/behavior_nn_v1/tests/"
os.makedirs(OUT, exist_ok=True)

ROLES = ["plan", "execute", "aggregate", "terminate"]
FAMS = ["select_frames", "visual_qa", "temporal_ops", "summarize", "detect", "other"]


def macro_f1(y_true, y_pred, classes):
    """observed-class macro-F1:只统计出现在 y_true 中的类"""
    f1s = []
    for c in classes:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == c and b == c)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == c and b != c)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def per_class_metrics(y_true, y_pred, classes):
    """per-class precision/recall/F1/support"""
    out = {}
    for c in classes:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == c and b == c)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == c and b != c)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        out[str(c)] = {"precision": round(p, 4), "recall": round(r, 4),
                       "f1": round(f1, 4), "support": sum(1 for a in y_true if a == c)}
    return out


def confusion_matrix(y_true, y_pred, classes):
    n = len(classes)
    cm = [[0] * n for _ in range(n)]
    idx = {c: i for i, c in enumerate(classes)}
    for a, b in zip(y_true, y_pred):
        cm[idx[a]][idx[b]] += 1
    return cm


def evaluate_joint(rows):
    """rows: [{role_true, role_pred, fam_true(None 非 execute), fam_pred(None|idx)}]"""
    n = len(rows)
    joint = route_n = route_ok = 0
    for r in rows:
        role_ok = r["role_true"] == r["role_pred"]
        if r["fam_true"] is None:
            ok = role_ok  # 非 execute:只要求角色
        else:
            ok = role_ok and r["fam_pred"] == r["fam_true"]
            route_n += 1
            route_ok += (r["fam_pred"] == r["fam_true"])
        joint += ok
    return {
        "joint": round(joint / n, 4),
        "n": n,
        "route_denominator": route_n,
        "route": round(route_ok / route_n, 4) if route_n else None,
    }


def run_fixtures():
    results = {}

    # F1:非 execute 角色预测错误,joint 必须记错
    rows = [
        {"role_true": "aggregate", "role_pred": "plan", "fam_true": None, "fam_pred": None},
        {"role_true": "execute", "role_pred": "execute", "fam_true": 0, "fam_pred": 0},
    ]
    e = evaluate_joint(rows)
    results["f1_non_execute_role_err_counts_wrong"] = e["joint"] == 0.5
    print("f1:", e, "-> non-execute role err counted:", e["joint"] == 0.5)

    # F2:route 分母 = execute 子集
    rows2 = [{"role_true": "plan", "role_pred": "plan", "fam_true": None, "fam_pred": None},
             {"role_true": "execute", "role_pred": "execute", "fam_true": 1, "fam_pred": 1},
             {"role_true": "execute", "role_pred": "execute", "fam_true": 2, "fam_pred": 3}]
    e2 = evaluate_joint(rows2)
    results["f2_route_denominator"] = e2["route_denominator"] == 2 and e2["route"] == 0.5
    print("f2: route denom =", e2["route_denominator"], "route =", e2["route"])

    # F3:无支持类不进 observed macro-F1
    yt = [0, 0, 1, 1, 2]
    yp = [0, 1, 1, 1, 2]
    mf_obs = macro_f1(yt, yp, [0, 1, 2])           # observed 3 类
    mf_fixed = macro_f1(yt, yp, list(range(6)))     # fixed-schema 6 类(3 个零 F1)
    results["f3_observed_vs_fixed_macro"] = {
        "observed_3class": round(mf_obs, 4),
        "fixed_6class": round(mf_fixed, 4),
        "observed_gt_fixed": mf_obs > mf_fixed,
    }
    print("f3:", results["f3_observed_vs_fixed_macro"])

    # F4:execute-gated 语义:真实 execute 且 fam 错,joint 记错
    rows4 = [{"role_true": "execute", "role_pred": "execute", "fam_true": 0, "fam_pred": 1},
             {"role_true": "execute", "role_pred": "aggregate", "fam_true": 0, "fam_pred": 1}]
    e4 = evaluate_joint(rows4)
    results["f4_execute_gated"] = e4["joint"] == 0.0
    print("f4: execute-gated joint =", e4["joint"])

    # F5:support 报告:每类真实样本数
    from collections import Counter
    sup = Counter(r["role_true"] for r in rows + rows2 + rows4)
    results["f5_support"] = dict(sup)
    print("f5 support:", dict(sup))

    # F6:confusion matrix / per-class metrics(从 y_true/y_pred 重算,含 predictions.jsonl 重算一致性)
    yt = [0, 0, 1, 1, 2, 2, 2, 3]
    yp = [0, 1, 1, 1, 2, 0, 3, 3]
    cm = confusion_matrix(yt, yp, list(range(4)))
    pc = per_class_metrics(yt, yp, list(range(4)))
    mf_direct = round(macro_f1(yt, yp, [0, 1, 2, 3]), 4)
    # predictions.jsonl 重算:写一个预测文件,从文件重算与直接计算一致
    import tempfile
    pred_path = OUT + "eval_fixture_predictions.jsonl"
    with open(pred_path, "w") as f:
        for a, b in zip(yt, yp):
            f.write(json.dumps({"y_true": a, "y_pred": b}) + "\n")
    yt_r, yp_r = [], []
    for line in open(pred_path):
        d = json.loads(line)
        yt_r.append(d["y_true"]); yp_r.append(d["y_pred"])
    mf_recompute = round(macro_f1(yt_r, yp_r, [0, 1, 2, 3]), 4)
    results["f6_confusion"] = {"matrix": cm,
                               "per_class": pc,
                               "recompute_consistent": mf_direct == mf_recompute,
                               "macro_from_predictions": mf_recompute}
    print("f6: confusion matrix rows:", cm)
    print("f6 recompute consistent:", results["f6_confusion"]["recompute_consistent"])

    results["all_fixtures_pass"] = all(
        [results["f1_non_execute_role_err_counts_wrong"], results["f2_route_denominator"],
         results["f3_observed_vs_fixed_macro"]["observed_gt_fixed"],
         results["f4_execute_gated"], results["f6_confusion"]["recompute_consistent"]])
    with open(OUT + "evaluator_fixture_report.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("fixtures all pass:", results["all_fixtures_pass"])


if __name__ == "__main__":
    run_fixtures()
