#!/usr/bin/env python3
"""抽象维度预测器验证(对应远端 docs/abstract_dimension_predictor_20260805.md)

验证 A:抽象角色熵降(A1 compute_events 5 类角色序列 / A2 prefix 3 类角色序列)
验证 B:资源意图区分度((角色 x model) 分位表)
验证 C:角色序列循环数分布
"""
import argparse
import json
import math
from collections import Counter, defaultdict


def load_rows(path):
    return [json.loads(l) for l in open(path)]


def entropy(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def conditional_entropy(joint):
    marginal_x = Counter()
    for (x, _), c in joint.items():
        marginal_x[x] += c
    total = sum(marginal_x.values())
    if total == 0:
        return 0.0
    h = 0.0
    for x, cx in marginal_x.items():
        p_x = cx / total
        inner = Counter({y: c for (xx, y), c in joint.items() if xx == x})
        h += p_x * entropy(inner)
    return h


def entropy_reduction(hm, hc):
    return 1.0 - hc / hm if hm else 1.0


def diag(seqs, cond_fields=None):
    """seqs: list of (label_seq, cond_list_per_row)。计算一阶熵降 + 条件熵降。"""
    marginal = Counter()
    joint = Counter()
    for seq, conds in seqs:
        for i in range(len(seq) - 1):
            ctx = (seq[i],)
            if cond_fields:
                ctx += tuple(conds[i + 1].get(f, "unknown") for f in cond_fields)
            joint[(ctx, seq[i + 1])] += 1
            marginal[seq[i + 1]] += 1
    hm = entropy(marginal)
    hc = conditional_entropy(joint)
    return {"H_marginal": round(hm, 4), "H_cond": round(hc, 4),
            "R": round(entropy_reduction(hm, hc), 4), "n": sum(joint.values())}


def quantile(sorted_vals, q):
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, int(q * len(sorted_vals)))
    return round(sorted_vals[idx], 1)


# 角色映射
NODE_TO_ROLE = {
    "run_control": "init",
    "planner": "plan",
    "videotool_spatial": "execute",
    "videotool_temporal": "execute",
    "videotool_generalist": "execute",
    "answer_generation": "aggregate",
}
ACTIVITY_TO_ROLE = {
    "sample_seek": "execute",
    "spatial_qa": "execute",
    "temporal_qa": "execute",
    "object_detection": "execute",
    "summarize": "execute",
    "answer": "aggregate",
    "__END__": "terminate",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute-events", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    events = load_rows(args.compute_events)
    prefixes = load_rows(args.prefix)
    out = {"plan_doc": "docs/abstract_dimension_predictor_20260805.md"}

    # ===== A1: compute_events 层面,node_type -> 角色序列 =====
    by_run = defaultdict(list)
    for r in events:
        by_run[r["run_id"]].append(r)
    seqs = []
    for rid, rs in by_run.items():
        rs_sorted = sorted(rs, key=lambda x: x.get("event_index", 0))
        roles = [NODE_TO_ROLE.get(r.get("node_type"), "other") for r in rs_sorted]
        conds = [{"model": r.get("model_id", "?"), "baseline": r.get("baseline", "?")}
                 for r in rs_sorted]
        seqs.append((roles, conds))
    a1 = {
        "runs": len(seqs),
        "role_dist": dict(Counter(r for s, _ in seqs for r in s)),
        "diag_1st": diag(seqs),
        "diag_1st_model": diag(seqs, ["model"]),
        "diag_1st_model_baseline": diag(seqs, ["model", "baseline"]),
    }
    out["A1_compute_role_sequence"] = a1
    print("== A1 compute 角色序列(5类) ==")
    for k, v in a1.items():
        if k.startswith("diag"):
            print(f"  {k}: {v}")
    print(f"  角色分布: {a1['role_dist']}")

    # ===== A2: prefix 层面,target -> 角色序列(3 类) =====
    p_by_run = defaultdict(list)
    for r in prefixes:
        p_by_run[r["run_id"]].append(r)
    p_seqs = []
    for rid, rs in p_by_run.items():
        rs_sorted = sorted(rs, key=lambda x: x["position"])
        seen = set()
        roles, conds = [], []
        for r in rs_sorted:
            if r["position"] in seen:
                continue
            seen.add(r["position"])
            roles.append(ACTIVITY_TO_ROLE.get(r.get("target_next_activity"), "other"))
            conds.append({"planner": r.get("planner_model_id", "?"),
                          "baseline": r.get("baseline", "?")})
        p_seqs.append((roles, conds))
    a2 = {
        "runs": len(p_seqs),
        "role_dist": dict(Counter(r for s, _ in p_seqs for r in s)),
        "diag_1st": diag(p_seqs),
        "diag_1st_planner": diag(p_seqs, ["planner"]),
        "diag_1st_planner_baseline": diag(p_seqs, ["planner", "baseline"]),
    }
    out["A2_prefix_role_sequence"] = a2
    print("\n== A2 prefix 角色序列(3类) ==")
    for k, v in a2.items():
        if k.startswith("diag"):
            print(f"  {k}: {v}")
    print(f"  角色分布: {a2['role_dist']}")

    # ===== B: 资源意图区分度(角色 x model) =====
    g = defaultdict(list)
    for r in events:
        role = NODE_TO_ROLE.get(r.get("node_type"), "other")
        if (r.get("runtime_ms") or 0) > 0:
            g[(role, r.get("model_id", "?")[:20])].append(r)
    b = {}
    print("\n== B 资源分位表(角色 x model, runtime>0) ==")
    for (role, model), rs in sorted(g.items(), key=lambda x: -len(x[1])):
        rt = sorted(r["runtime_ms"] for r in rs)
        pk = sorted((r.get("peak_allocated_mb") or 0) for r in rs)
        ld = sorted((r.get("load_ms") or 0) for r in rs)
        row = {"n": len(rs), "runtime": {"P50": quantile(rt, 0.5), "P90": quantile(rt, 0.9),
                                         "P99": quantile(rt, 0.99)},
               "peak_MB": {"P50": quantile(pk, 0.5), "P99": quantile(pk, 0.99)},
               "load_ms_P50": quantile(ld, 0.5)}
        b[f"{role}|{model}"] = row
        print(f"  {role:10s} {model:22s} n={len(rs):5d} rt_P50={row['runtime']['P50']} "
              f"rt_P90={row['runtime']['P90']} peak_P99={row['peak_MB']['P99']}")
    # 组间分离度
    meds = [(k, v["runtime"]["P50"]) for k, v in b.items() if v["runtime"]["P50"]]
    max_ratio = 1.0
    for i in range(len(meds)):
        for j in range(i + 1, len(meds)):
            r = max(meds[i][1], meds[j][1]) / max(min(meds[i][1], meds[j][1]), 1e-6)
            max_ratio = max(max_ratio, r)
    out["B_resource_separation"] = {"groups": b, "max_runtime_P50_ratio": round(max_ratio, 2)}
    print(f"  组间 runtime P50 最大比值: {max_ratio:.1f}x")

    # ===== C: 循环数分布(plan->execute 循环计数) =====
    loop_counts = []
    for roles, _ in seqs:
        c = 0
        in_loop = False
        for r in roles:
            if r == "plan":
                in_loop = True
            elif r == "execute" and in_loop:
                c += 1
            elif r == "aggregate":
                break
        loop_counts.append(c)
    lc = Counter(loop_counts)
    out["C_plan_execute_loop"] = {"per_run": dict(sorted(lc.items())),
                                  "mean": round(sum(loop_counts) / len(loop_counts), 2),
                                  "p90": sorted(loop_counts)[int(len(loop_counts) * 0.9)]}
    print(f"\n== C plan->execute 循环数分布(per run) ==\n  {dict(sorted(lc.items()))}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n结果已写入: {args.out}")


if __name__ == "__main__":
    main()
