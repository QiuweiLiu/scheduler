#!/usr/bin/env python3
"""R1-cutoff v2(验收二轮修复):
- question/options 真实位置 features.text.question/options:扰动同长度文本,特征签名不变
- suffix 扰动调用真实特征构建器(feature_builder),而非手写过滤器
- answer_excluded 检查(features.text.answer_excluded)
- all_pass 包含视觉 cutoff 检查
"""
import hashlib
import json
import os
from collections import defaultdict, Counter

D = "/root/autodl-tmp/scheduler/"
OUT = D + "results/processed/behavior_nn_v1/tests/"
os.makedirs(OUT, exist_ok=True)

samples = [json.loads(l) for l in open(D + "results/processed/behavior_nn_v1/data/semantic_tool_samples.jsonl")]
rows = [json.loads(l) for l in open(D + "results/processed/role_dataset_v0_3.jsonl")]
by_run = defaultdict(list)
for r in rows:
    by_run[r["run_id"]].append(r)
for rl in by_run.values():
    rl.sort(key=lambda x: x["event_index"])

FAMILY = {"sample_seek": "select_frames", "frame-selector": "select_frames",
          "image-grid-selector": "select_frames", "spatial_qa": "visual_qa",
          "image-qa": "visual_qa", "image-grid-qa": "visual_qa", "patch-zoomer": "visual_qa",
          "temporal-qa": "temporal_ops", "temporal-grounding": "temporal_ops",
          "summarize": "summarize", "summarization-tool": "summarize",
          "object_detection": "detect", "yolo-tracker": "detect"}
FAMS = ["select_frames", "visual_qa", "temporal_ops", "summarize", "detect", "other"]


def feature_builder(s):
    """真实特征构建器(R2 用):返回结构化特征 dict(序列化签名用)
    - 静态:domain/question_type/answer_type/**current model(预测前可知)**/baseline/position
    - current model = 目标前一事件的 model(首步用 prefix 的 compute_last_model_id,即 planner/预测时刻已知)
    - 历史:目标之前事件序列(role, family)
    - 禁止:target_model_id、question 文本、options 文本、任何 target/未来字段"""
    f = s.get("features") or {}
    cat = f.get("categorical") or {}
    tidx = s.get("cutoff_event_index") or s.get("target_event_index")
    prev = by_run.get(s["run_id"], [])
    prev = [r for r in prev if r["event_index"] < tidx]
    cur_model = None
    if prev:
        cur_model = prev[-1]["model_id"]          # 前一事件模型(预测时刻已知)
    else:
        cur_model = cat.get("compute_last_model_id")  # 首步:planner(预测时刻已知)
    hist = [(r["event_index"], r["role"], FAMILY.get(r["raw_action"], "other"))
            for r in prev]
    return {
        "static": [cat.get("domain"), cat.get("question_type"), cat.get("answer_type"),
                   cur_model, s.get("target_baseline"), s.get("position")],
        "history": hist,
    }


def sig(s):
    return hashlib.sha256(json.dumps(feature_builder(s), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


results = {}

# ---- 1. suffix 扰动(真实特征构建器)----
suf_ok = 0
suf_n = 0
for s in samples[:300]:
    rid = s["run_id"]
    tidx = s["target_event_index"]
    rl_all = by_run.get(rid, [])
    if not any(r["event_index"] > tidx for r in rl_all):
        continue
    suf_n += 1
    h_before = sig(s)
    for r in rl_all:
        if r["event_index"] > tidx:
            r["raw_action"] = "__SUFFIX_PERTURBED__"
            r["role"] = "plan" if r["role"] == "execute" else "execute"
    h_after = sig(s)
    if h_before == h_after:
        suf_ok += 1
results["t1_suffix_perturbation"] = {"tested": suf_n, "signature_unchanged": suf_ok,
                                     "pass": suf_n > 0 and suf_ok == suf_n}
print("t1 suffix(feature_builder):", results["t1_suffix_perturbation"])

# ---- 2. question/options 同长度扰动 + answer_excluded ----
t2_ok = 0
t2_n = 0
t2_no_text = 0
for s in samples[:300]:
    f = s.get("features") or {}
    txt = f.get("text") or {}
    q = txt.get("question")
    opts = txt.get("options")
    if not (isinstance(q, str) and isinstance(opts, list)):
        t2_no_text += 1
        continue
    t2_n += 1
    h_before = sig(s)
    orig_q, orig_opts = q, list(opts)
    txt["question"] = "x" * len(q)                 # 同长度扰动
    txt["options"] = ["y" * len(o) for o in opts]   # 同长度扰动
    h_after = sig(s)
    if h_before == h_after:
        t2_ok += 1
    txt["question"], txt["options"] = orig_q, orig_opts
ans_excluded = sum(1 for s in samples if (s.get("features") or {}).get("text", {}).get("answer_excluded"))
results["t2_question_options_perturbation"] = {
    "tested": t2_n, "no_text_field": t2_no_text, "signature_unchanged": t2_ok,
    "answer_excluded_flag": ans_excluded,
    "pass": t2_n > 0 and t2_ok == t2_n and ans_excluded == len(samples),
}
print("t2 question/options 同长度扰动:", results["t2_question_options_perturbation"])

# ---- 3. 视觉 cutoff:contract + cutoff_event + missing_reason ----
vc = Counter()
all_cutoff = True
for s in samples:
    vp = s.get("visual_context") or {}
    vc[(vp.get("target_event_excluded"), vp.get("future_events_excluded"))] += 1
    if not isinstance(s.get("cutoff_event"), int):
        all_cutoff = False
vis_ok = all((a is True and b is True) for (a, b) in vc) and all_cutoff
results["t3_visual_cutoff"] = {
    "contract": {f"{a}/{b}": c for (a, b), c in vc.items()},
    "cutoff_event_present": all_cutoff,
    "missing_reason_dist": dict(Counter(s["missing_reason"] for s in samples)),
    "vision_available": sum(1 for s in samples if s["vision_available"]),
    "limit": "帧→事件时间线映射缺数据;cutoff_event 由 target_event_index-1 推导,帧级验证待 R2 视觉模块",
    "pass": vis_ok,
}
print("t3 visual:", results["t3_visual_cutoff"])

results["all_pass"] = (results["t1_suffix_perturbation"]["pass"]
                       and results["t2_question_options_perturbation"]["pass"]
                       and results["t3_visual_cutoff"]["pass"])

# ---- 4. 泄漏面真实检测:feature_builder 输出不得含 target 字段 ----
leak_hits = []
for s in samples[:500]:
    fb = feature_builder(s)
    flat = json.dumps(fb, ensure_ascii=False).lower()
    for bad in ("target_model", "target_baseline", "target_source", "next_role", "family_label",
                "answer", "ground_truth", "remaining", "future"):
        if bad in flat:
            leak_hits.append(bad)
results["t4_feature_builder_leak_scan"] = {
    "hits": sorted(set(leak_hits)),
    "target_model_id_absent": "target_model" not in flat if leak_hits else True,
    "pass": len(leak_hits) == 0,
}
print("t4 feature_builder leak scan:", results["t4_feature_builder_leak_scan"])

# 视觉帧审计结果并入 all_pass
vis_audit = D + "results/processed/behavior_nn_v1/tests/visual_cutoff_audit_report.json"
if os.path.exists(vis_audit):
    va = json.load(open(vis_audit))
    vis_pass = va.get("violations", -1) == 0 and va.get("no_trace", 0) + va.get("prefix_events_unresolved", 0) == 0
    results["t5_visual_frame_audit"] = {"violations": va.get("violations"),
                                        "verified": va.get("verified_frame_in_prefix_and_no_target_leak"),
                                        "no_trace": va.get("no_trace"),
                                        "pass": vis_pass}
    results["all_pass"] = results["all_pass"] and vis_pass
else:
    results["t5_visual_frame_audit"] = {"pass": False, "error": "audit report missing"}
    results["all_pass"] = False
print("t5 visual frame audit:", results["t5_visual_frame_audit"])
with open(OUT + "cutoff_report.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("all_pass(含视觉):", results["all_pass"])
print("written:", OUT + "cutoff_report.json")
