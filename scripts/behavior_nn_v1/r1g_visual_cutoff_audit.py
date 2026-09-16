#!/usr/bin/env python3
"""R1-visual-cutoff-audit:帧→事件映射审计(验收三轮修复,视觉硬门槛)

利用 trace.jsonl action 事件 input.frame_indices 构建"事件→帧号"映射,
对每个 semantic_tool 样本验证:
1. visual.frame_indices ⊆ 事件帧并集(prefix 内事件)
2. 目标事件帧不在 visual 中(若可定位)
产物:tests/visual_cutoff_audit_report.json
"""
import glob
import json
import os
from collections import defaultdict

D = "/root/autodl-tmp/scheduler/"
OUT = D + "results/processed/behavior_nn_v1/tests/"
os.makedirs(OUT, exist_ok=True)

# ---- 索引 trace:run_id -> {event_id: frame_indices} ----
trace_dirs = (glob.glob(D + "results/raw/phase3_videomme_expansion236_stackb_20260805/*/")
              + glob.glob(D + "results/raw/phase2_videomme_formal_flash_120/*/"))
run_frames = {}
run_loaded = 0
for d in trace_dirs:
    tp = os.path.join(d, "trace.jsonl")
    if not os.path.exists(tp):
        continue
    try:
        evs = {}
        for line in open(tp):
            x = json.loads(line)
            inp = x.get("input")
            fi = inp.get("frame_indices") if isinstance(inp, dict) else None
            if x.get("event_type") == "action" and fi:
                evs[x.get("event_id")] = list(fi)
        if evs:
            run_frames[os.path.basename(d.rstrip("/"))] = evs
            run_loaded += 1
    except Exception:
        pass
print("索引 run(含帧事件):", run_loaded, "/", len(trace_dirs))

# ---- 验证 tool 样本 ----
samples = [json.loads(l) for l in open(D + "results/processed/behavior_nn_v1/data/semantic_tool_samples.jsonl")]
ok = violation = no_trace = no_events = 0
viol_examples = []
frames_total = 0
for s in samples:
    vp = s.get("visual_context") or {}
    vis_frames = vp.get("frame_indices") or []
    if not vis_frames:
        continue
    frames_total += 1
    rid = s["run_id"]
    evs = run_frames.get(rid)
    if not evs:
        no_trace += 1
        continue
    src_ids = vp.get("source_event_ids") or []
    tgt_id = s["target_source_event_id"]
    prefix_events = [e for e in src_ids if e in evs]
    if not prefix_events:
        no_events += 1
        continue
    prefix_frames = set()
    for e in prefix_events:
        prefix_frames.update(evs[e])
    tgt_frames = set(evs.get(tgt_id, []))
    vis_set = set(vis_frames)
    in_prefix = vis_set <= prefix_frames
    tgt_leak = bool(vis_set & tgt_frames)
    if in_prefix and not tgt_leak:
        ok += 1
    else:
        violation += 1
        if len(viol_examples) < 5:
            viol_examples.append({
                "run_id": rid[:24], "n_vis": len(vis_frames),
                "in_prefix": in_prefix, "tgt_leak": tgt_leak,
                "vis": sorted(vis_set)[:6], "prefix": sorted(prefix_frames)[:6],
                "tgt": sorted(tgt_frames)[:6]})

rep = {
    "samples_with_frames": frames_total,
    "verified_frame_in_prefix_and_no_target_leak": ok,
    "violations": violation,
    "no_trace": no_trace,
    "prefix_events_unresolved": no_events,
    "violation_examples": viol_examples,
    "frame_to_event_mapping_source": "trace.jsonl action input.frame_indices(事件级帧号)",
    "coverage_note": f"trace 索引覆盖 {run_loaded} run(phase3 472 + phase2 formal 120);未覆盖 run 无法验证",
}
with open(OUT + "visual_cutoff_audit_report.json", "w") as f:
    json.dump(rep, f, ensure_ascii=False, indent=2)
print(json.dumps({k: v for k, v in rep.items() if k != "violation_examples"}, ensure_ascii=False, indent=1))
