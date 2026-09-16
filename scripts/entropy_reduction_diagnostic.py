#!/usr/bin/env python3
"""熵降证据增量诊断 (对应远端 docs/entropy_evidence_diagnosis_20260805.md)

用法:
  python3 scripts/entropy_reduction_diagnostic.py PREFIX_FILE [--out OUT] [--no-end] [--group-key FIELD]
"""
import argparse
import json
import math
from collections import Counter, defaultdict


def entropy(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def conditional_entropy(joint_counts):
    marginal_x = Counter()
    for (x, _), c in joint_counts.items():
        marginal_x[x] += c
    total = sum(marginal_x.values())
    if total == 0:
        return 0.0
    h = 0.0
    for x, cx in marginal_x.items():
        p_x = cx / total
        inner = Counter({y: c for (xx, y), c in joint_counts.items() if xx == x})
        h += p_x * entropy(inner)
    return h


def entropy_reduction(hm, hc):
    if hm == 0:
        return 1.0
    return 1.0 - hc / hm


def qcut(values, n_bins=3):
    """按分位数把连续值切成 n_bins 档,返回 (bin_edges, bucket_fn)。"""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return [], lambda v: "missing"
    edges = []
    for i in range(1, n_bins):
        q = i / n_bins
        idx = min(len(vals) - 1, int(q * len(vals)))
        edges.append(vals[idx])
    edges = sorted(set(edges))

    def bucket(v):
        if v is None:
            return "missing"
        for i, e in enumerate(edges):
            if v <= e:
                return f"bin{i}"
        return f"bin{len(edges)}"
    return edges, bucket


def flatten_row(row, buckets):
    """把嵌套字段拍平成离散特征 dict。buckets: {field: bucket_fn}"""
    f = {}
    ts = row.get("task_structure") or {}
    f["planner_model_id"] = row.get("planner_model_id") or "unknown"
    f["baseline"] = row.get("baseline") or "unknown"
    f["question_type"] = ts.get("question_type") if isinstance(ts, dict) else None
    f["question_type"] = f["question_type"] if f["question_type"] not in (None, "") else "unknown"
    f["domain"] = ts.get("domain") if isinstance(ts, dict) else None
    f["domain"] = f["domain"] if f["domain"] not in (None, "") else "unknown"
    f["temporal_scope"] = ts.get("temporal_scope") if isinstance(ts, dict) else None
    f["temporal_scope"] = f["temporal_scope"] if f["temporal_scope"] not in (None, "") else "unknown"

    sf = row.get("state_features") or {}
    ev = sf.get("evidence") if isinstance(sf, dict) else None
    cov = ev.get("coverage_ratio") if isinstance(ev, dict) else None
    f["coverage_ratio"] = buckets.get("coverage_ratio", lambda v: "missing")(cov)
    px = sf.get("prefix") if isinstance(sf, dict) else None
    prog = px.get("progress_ratio") if isinstance(px, dict) else None
    f["progress_ratio"] = buckets.get("progress_ratio", lambda v: "missing")(prog)

    feats = row.get("features") or {}
    cat = feats.get("categorical") if isinstance(feats, dict) else None
    f["raw_prefix_tail2"] = (cat or {}).get("raw_prefix_tail2") or "unknown"

    tp = row.get("teacher_probs")
    if isinstance(tp, (list, tuple)) and len(tp) > 0:
        mx = max(tp)
        if mx >= 0.8:
            f["teacher_bucket"] = "high"
        elif mx >= 0.5:
            f["teacher_bucket"] = "mid"
        else:
            f["teacher_bucket"] = "low"
    else:
        f["teacher_bucket"] = "missing"
    return f


def build_sequences(rows, buckets, target_map=None):
    """按 run_id 排序 position,重建动作序列;返回 (run_id, planner, baseline, seq, flats)。"""
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run_id"]].append(r)
    seqs = []
    for run_id, rs in by_run.items():
        rs_sorted = sorted(rs, key=lambda x: x["position"])
        seen = set()
        seq = []
        flats = []
        for r in rs_sorted:
            if r["position"] in seen:
                continue
            seen.add(r["position"])
            seq.append(r)
            flats.append(flatten_row(r, buckets))
        if target_map:
            seq = [{**r, "target_next_activity": target_map.get(
                r["target_next_activity"], r["target_next_activity"])} for r in seq]
        seqs.append((run_id, rs_sorted[0].get("planner_model_id", "unknown"),
                     rs_sorted[0].get("baseline", "unknown"), seq, flats))
    return seqs


def collect_transitions(seqs, order=1, include_end=True, cond_fields=None):
    """收集 (context, next) 计数。cond_fields 为扁平特征字段名列表。"""
    joint = Counter()
    marginal = Counter()
    for _, _, _, seq, flats in seqs:
        acts = [r["target_next_activity"] for r in seq]
        if not include_end:
            acts = [a for a in acts if a != "__END__"]
        for i in range(len(acts) - 1):
            ctx = list(acts[max(0, i + 1 - order): i + 1])
            if cond_fields:
                ctx += [flats[i + 1].get(k, "unknown") for k in cond_fields]
            nxt = acts[i + 1]
            joint[(tuple(ctx), nxt)] += 1
            marginal[nxt] += 1
    return marginal, joint


def pos_bucket(pos, total):
    if pos <= 1:
        return "start"
    if pos >= total - 2:
        return "end"
    return "mid"


# 增量测试顺序(与远端文档第 4.3 节一致)
INCREMENTAL_STEPS = [
    ("step0 一阶基线 [A_t]", []),
    ("step1 +planner", ["planner_model_id"]),
    ("step2 +baseline", ["planner_model_id", "baseline"]),
    ("step3 +question_type", ["planner_model_id", "baseline", "question_type"]),
    ("step4 +domain", ["planner_model_id", "baseline", "question_type", "domain"]),
    ("step5 +temporal_scope", ["planner_model_id", "baseline", "question_type", "domain", "temporal_scope"]),
    ("step6 +coverage_ratio", ["planner_model_id", "baseline", "question_type", "domain", "temporal_scope", "coverage_ratio"]),
    ("step7 +progress_ratio", ["planner_model_id", "baseline", "question_type", "domain", "temporal_scope", "coverage_ratio", "progress_ratio"]),
    ("step8 +raw_prefix_tail2", ["planner_model_id", "baseline", "question_type", "domain", "temporal_scope", "coverage_ratio", "progress_ratio", "raw_prefix_tail2"]),
    ("step9 +teacher_bucket", ["planner_model_id", "baseline", "question_type", "domain", "temporal_scope", "coverage_ratio", "progress_ratio", "raw_prefix_tail2", "teacher_bucket"]),
]

# 粒度映射(与远端 docs/entropy_granularity_diagnosis_20260805.md 第 4.1 节一致)
GRANULARITY_MAPS = {
    5: {"spatial_qa": "vision_qa", "temporal_qa": "vision_qa"},
    4: {"sample_seek": "observe", "spatial_qa": "observe", "temporal_qa": "observe",
        "object_detection": "observe"},
    3: {"sample_seek": "visual", "spatial_qa": "visual", "temporal_qa": "visual",
        "object_detection": "visual", "summarize": "text", "answer": "text"},
}


def diagnose(seqs, include_end, buckets):
    """按增量顺序输出全部诊断条目。"""
    results = []
    for label, cond in INCREMENTAL_STEPS:
        marginal, joint = collect_transitions(seqs, order=1, include_end=include_end,
                                              cond_fields=cond)
        hm = entropy(marginal)
        hc = conditional_entropy(joint)
        r = entropy_reduction(hm, hc)
        n_cond = len(set(k for (k, _) in joint))
        per_cond = sum(joint.values()) / n_cond if n_cond else 0
        warn = " ⚠️样本稀释" if per_cond < 20 else ""
        results.append({"label": label, "fields": cond,
                        "H_marginal_bits": round(hm, 4), "H_cond_bits": round(hc, 4),
                        "R": round(r, 4), "n_transitions": sum(joint.values()),
                        "n_cond_keys": n_cond, "per_cond_mean": round(per_cond, 1),
                        "dilution_warning": per_cond < 20})
        print(f"{label:34s} H_marg={hm:.3f} H_cond={hc:.3f} R={r:.4f} "
              f"n={sum(joint.values())} 每条件组均样本={per_cond:.1f}{warn}")

    _, joint1 = collect_transitions(seqs, order=1, include_end=include_end)
    acts = Counter()
    for (_cond, nxt), c in joint1.items():
        acts[nxt] += c
    print("目标动作分布 (top-15):")
    for a, c in acts.most_common(15):
        print(f"  {a:22s} {c:6d}  {c/sum(acts.values()):.3f}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix_file")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-end", action="store_true", help="去掉 __END__ 目标")
    ap.add_argument("--group-key", default=None,
                    help="按该字段分组分别统计(如 dataset_partition / planner_model_id)")
    ap.add_argument("--granularity", default="all", choices=["all", "7", "5", "4", "3"],
                    help="目标粒度消融:7/5/4/3 或 all(默认)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.prefix_file)]

    if args.granularity == "all":
        granularities = sorted([7] + list(GRANULARITY_MAPS.keys()))
    else:
        granularities = [int(args.granularity)]

    # 分箱边界在全量数据上按分位数计算(诊断用,非模型训练)
    buckets = {}
    for field, path in [("coverage_ratio", ("state_features", "evidence", "coverage_ratio")),
                        ("progress_ratio", ("state_features", "prefix", "progress_ratio"))]:
        vals = []
        for r in rows:
            d = r
            ok = True
            for k in path:
                if not isinstance(d, dict):
                    ok = False
                    break
                d = d.get(k)
            vals.append(d if ok and d is not None else None)
        _, bucket = qcut(vals, 3)
        buckets[field] = bucket

    if args.group_key:
        groups = defaultdict(list)
        for r in rows:
            groups[r.get(args.group_key, "unknown")].append(r)
        print(f"分组 ({args.group_key}): {dict((k, len(v)) for k, v in groups.items())}")
        grouped = list(groups.items())
    else:
        grouped = [("all", rows)]

    include_end = not args.no_end
    tag = "all(含END)" if include_end else "非终止"

    report = {"source": args.prefix_file, "group_key": args.group_key,
              "scope": tag, "granularities": {}, "groups": []}

    for g in granularities:
        target_map = GRANULARITY_MAPS.get(g)
        gname0 = f"粒度{g}类" + ("(映射)" if target_map else "(原样)")
        print(f"\n########## {gname0} ##########")
        g_groups = []
        for gname, g_rows in grouped:
            seqs = build_sequences(g_rows, buckets, target_map=target_map)
            print(f"===== 分组 [{gname}]  prefix 行数: {len(g_rows)}  run 数: {len(seqs)} ({tag}) =====")
            results = diagnose(seqs, include_end, buckets)
            g_groups.append({"group": gname, "prefix_rows": len(g_rows),
                             "runs": len(seqs), "scope": tag, "results": results})
        report["granularities"][str(g)] = {"map": target_map or {}, "groups": g_groups}

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n报告已写入: {args.out}")


if __name__ == "__main__":
    main()
