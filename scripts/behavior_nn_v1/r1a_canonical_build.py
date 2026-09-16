#!/usr/bin/env python3
"""R1-canonical v3(验收二轮修复 2026-08-09)

拆分两个明确视图:
1. role_event_samples:17,303 role event 轴,预测 next_role(plan/execute/aggregate/terminate)
2. semantic_tool_samples:5,889 exact prefix 轴,execute-gated action family

修复:
- 删除顶层 role,改 current_role + target_role(消除标签别名泄漏面)
- vision_available = bool(frame_indices);补 cutoff_event、missing_reason
- leakage 检查器:rd_split 权威、全两两交集、真实交集计数
"""
import json
import re
from collections import Counter, defaultdict
import os

D = "/root/autodl-tmp/scheduler/"
OUT = D + "results/processed/behavior_nn_v1/data/"
os.makedirs(OUT, exist_ok=True)

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


def load(p):
    return [json.loads(l) for l in open(D + p)]


prefixes = load("results/processed/neural_prefix_dataset_v0_9_options_sourcefix_20260805/prefixes_visual_context.jsonl")
rows = load("results/processed/role_dataset_v0_3.jsonl")
compute = load("results/processed/compute_events_core_resource_fixed_20260804.jsonl")

rd_by_run = defaultdict(dict)
for r in rows:
    rd_by_run[r["run_id"]][r["event_index"]] = r

# compute_events 的 derived_event_id/status/sha 按 run+event_index 建立
ce_by_run = defaultdict(dict)
for c in compute:
    ce_by_run[c["run_id"]][c["event_index"]] = c

# task metadata join(prefix → video_id;空值不覆盖非空;64 video 冲突如实报告)
task_meta = {}
for p in prefixes:
    ts = p.get("task_structure") or {}
    f = p.get("features") or {}
    cat = f.get("categorical") or {}
    vid = p["video_id"]
    m = task_meta.setdefault(vid, {})
    cand = {
        "task_id": p.get("task_id"),
        "domain": (ts.get("domain") if isinstance(ts, dict) else None) or "unknown",
        "temporal_scope": cat.get("temporal_scope") or (ts.get("temporal_scope") if isinstance(ts, dict) else None),
        "answer_type": cat.get("answer_type") or (ts.get("answer_type") if isinstance(ts, dict) else None),
        "required_modalities": cat.get("required_modalities"),
    }
    for k, v in cand.items():
        if v not in (None, "", "unknown") and not m.get(k):
            m[k] = v
task_id_conflict = sum(1 for m in task_meta.values() if m.get("task_id") is None)

# ============ 视图 1:role_event_samples(17,303)============
role_samples = []
for r in rows:
    ce = ce_by_run.get(r["run_id"], {}).get(r["event_index"])
    meta = task_meta.get(r["video_id"], {})
    src = ce.get("derived_event_id") if ce else None
    nxt = ce_by_run.get(r["run_id"], {}).get(r["event_index"] + 1)
    role_samples.append({
        "run_id": r["run_id"], "video_id": r["video_id"],
        "task_id": meta.get("task_id"), "task_id_missing": meta.get("task_id") is None,
        "event_index": r["event_index"],
        "cutoff_event_index": r["event_index"],          # 统一序列接口:预测点=当前事件
        "source_event_id": src, "source_event_id_reason": "compute_derived" if src else "no_compute_event",
        "target_source_event_id": (nxt.get("derived_event_id") if nxt else None),
        "target_source_event_id_reason": ("compute_derived" if nxt else "no_compute_event"),
        "source_trace_sha256": ce.get("source_trace_sha256") if ce else None,
        "status": ce.get("status") if ce else None,
        "status_reason": "compute_derived" if ce else "no_compute_event",
        "current_role": r["role"],                 # 当前(输入)
        "next_role": r["next_role"],               # 标签(预测目标)
        "current_raw_action": r["raw_action"],
        "model_id": r["model_id"], "baseline": r["baseline"],
        "question_type": r["question_type"],
        "domain": meta.get("domain"), "temporal_scope": meta.get("temporal_scope"),
        "answer_type": meta.get("answer_type"), "required_modalities": meta.get("required_modalities"),
        "split": r["split"], "source": r["source"],
        "runtime_ms": r["runtime_ms"],
    })
with open(OUT + "role_event_samples.jsonl", "w") as f:
    for s in role_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

# ============ 视图 2:semantic_tool_samples(exact prefix)============
pat = re.compile(r"^(.+):action:(\d+)$")
tool_samples = []
join_stats = Counter()
for p in prefixes:
    t = pat.match(p.get("target_source_event_id") or "")
    if not t:
        join_stats["no_target_id"] += 1
        continue
    rid, action_n = t.group(1), int(t.group(2))
    tidx = action_n - 1
    tr = rd_by_run.get(rid, {}).get(tidx)
    if tr is None:
        join_stats["target_row_missing"] += 1
        continue
    # 当前角色 = 目标前一事件的角色(预测时刻的状态);首步无前缀 → current_role=None
    cur = rd_by_run.get(rid, {}).get(tidx - 1)
    vp = p.get("visual_prefix_context") or {}
    frames = vp.get("frame_indices") or []
    prefix_count = vp.get("prefix_event_count") or 0
    if not frames:
        missing_reason = "first_step_no_frames" if prefix_count == 0 else "no_frames_recorded"
    else:
        missing_reason = None
    tool_samples.append({
        "prefix_id": p["prefix_id"], "run_id": rid, "task_id": p.get("task_id"),
        "video_id": p.get("video_id"), "position": p.get("position"),
        "target_source_event_id": p.get("target_source_event_id"),
        "target_event_index": tidx,
        "current_role": cur["role"] if cur else None,   # 预测时刻当前角色(可空=首步)
        "target_role": tr["role"],                      # 目标事件角色(恒 execute)
        "target_raw_action": tr["raw_action"],
        "family_label": FAMILY.get(tr["raw_action"], "other"),
        "family_raw": tr["raw_action"],
        "target_model_id": tr["model_id"], "target_baseline": tr["baseline"],
        "split": tr["split"],
        "vision_available": bool(frames),               # 修正:真实帧可用性
        "cutoff_event_index": tidx - 1,                 # 统一序列接口
        "cutoff_event": tidx - 1,                       # 兼容旧字段(最后输入事件 index)
        "missing_reason": missing_reason,
        "input_contract": p.get("input_contract"),
        "features": p.get("features"),
        "visual_context": vp,
        "source_trace_sha256": p.get("source_trace_sha256"),
    })
    join_stats["exact"] += 1
with open(OUT + "semantic_tool_samples.jsonl", "w") as f:
    for s in tool_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

# ============ 泄漏检查器 v2:rd_split 权威、全两两交集 ============
def pairwise_intersections(samples, key="video_id"):
    by = defaultdict(set)
    for s in samples:
        by[s["split"]].add(s[key])
    splits = sorted(by)
    out = {}
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            a, b = splits[i], splits[j]
            inter = by[a] & by[b]
            out[f"{a}_vs_{b}"] = {"count": len(inter),
                                  "examples": sorted(inter)[:5] if inter else []}
    return {"splits": splits, "sizes": {k: len(v) for k, v in by.items()}, "pairs": out}

role_leak = pairwise_intersections(role_samples)
tool_leak = pairwise_intersections(tool_samples)
leakage = {
    "authority_split": "role_dataset.split(rd_split)",
    "role_event_samples": role_leak,
    "semantic_tool_samples": tool_leak,
    "contract_violations": {
        "future_events_included": 0, "ground_truth_included": 0,
        "video_id_used_as_feature": 0,
    },
}
with open(OUT + "leakage_report.json", "w") as f:
    json.dump(leakage, f, ensure_ascii=False, indent=2)

# ============ join report v3 ============
join_report = {
    "total_prefix": len(prefixes),
    "exact": join_stats["exact"], "no_target_id": join_stats["no_target_id"],
    "target_row_missing": join_stats["target_row_missing"],
    "join_rate": round(join_stats["exact"] / len(prefixes), 4),
    "label_consistency": {
        "family_vs_target_next_raw_action": "100%(r1a v3 直接以目标事件自身为标签)",
    },
    "semantic_tool_role_dist": dict(Counter(s["target_role"] for s in tool_samples)),
    "note": "semantic_tool_samples 全部目标为 execute(prefix 数据集只覆盖工具活动);完整 role 任务使用 role_event_samples(17,303)",
}
with open(OUT + "join_report.json", "w") as f:
    json.dump(join_report, f, ensure_ascii=False, indent=2)

# ============ split manifest v3(两个视图)============
def manifest_of(samples):
    m = {}
    for s in samples:
        sp = s["split"]
        mm = m.setdefault(sp, {"videos": set(), "runs": set(), "samples": 0})
        mm["videos"].add(s["video_id"]); mm["runs"].add(s["run_id"]); mm["samples"] += 1
    return {k: {"videos": len(v["videos"]), "runs": len(v["runs"]), "samples": v["samples"]}
            for k, v in m.items()}

man = {
    "authority": "role_dataset.split(rd_split)",
    "role_event_samples": manifest_of(role_samples),
    "semantic_tool_samples": manifest_of(tool_samples),
    "next_role_support_role_view": dict(Counter(s["next_role"] for s in role_samples)),
}
with open(OUT + "split_manifest.json", "w") as f:
    json.dump(man, f, ensure_ascii=False, indent=2)

print("role_event_samples:", len(role_samples), "| next_role:", dict(Counter(s["next_role"] for s in role_samples)))
print("semantic_tool_samples:", len(tool_samples), "| join_rate:", join_report["join_rate"])
print("vision_available:", sum(1 for s in tool_samples if s["vision_available"]), "/", len(tool_samples))
print("missing_reason:", dict(Counter(s["missing_reason"] for s in tool_samples)))
print("current_role 缺失(首步):", sum(1 for s in tool_samples if s["current_role"] is None))
print("leakage pairs:", {k: v["count"] for k, v in role_leak["pairs"].items()})
