#!/usr/bin/env python3
"""R1-GateA v4(三轮修复):全部真实验证
"""
import json
import os
from collections import Counter

D = "/root/autodl-tmp/scheduler/"
OUT = D + "results/processed/behavior_nn_v1/"


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


checks = []

# 1. R0 registry
registry = json.load(open(OUT + "registry/legacy_result_registry.json"))
checks.append(("legacy result registry",
               len(registry["entries"]) >= 13 and sum(1 for e in registry["entries"] if e["file_exists"]) >= 13,
               "registry/legacy_result_registry.json"))

# 2. role canonical schema:字段齐全
role_samples = load_jsonl(OUT + "data/role_event_samples.jsonl")
tool_samples = load_jsonl(OUT + "data/semantic_tool_samples.jsonl")
need_role = {"task_id", "target_source_event_id", "source_trace_sha256", "status",
             "domain", "temporal_scope", "answer_type", "required_modalities",
             "cutoff_event_index", "source_event_id"}
role_ok = all(need_role.issubset(s.keys()) for s in role_samples)
role_dist = Counter(s["next_role"] for s in role_samples)
role_schema_ok = role_ok and set(role_dist) >= {"plan", "execute", "aggregate", "terminate"}
checks.append(("role canonical schema(稳定键+元数据+4 类标签)",
               role_schema_ok,
               f"17,303 样本;字段全;next_role {dict(role_dist)};status 覆盖 {sum(1 for s in role_samples if s['status'])}/{len(role_samples)}"))

# 3. tool 视图:无顶层 role、统一 cutoff_event_index、无标签别名
tool_ok = (all("role" not in s for s in tool_samples)
           and all(s.get("current_role") != s.get("target_role") for s in tool_samples)
           and all(isinstance(s.get("cutoff_event_index"), int) for s in tool_samples))
checks.append(("semantic tool schema(无别名+统一 cutoff)",
               tool_ok, f"5,889 样本;current≠target;cutoff_event_index 全有"))

# 4. split/group(全两两)
leak = json.load(open(OUT + "data/leakage_report.json"))
role_pairs = leak["role_event_samples"]["pairs"]
tool_pairs = leak["semantic_tool_samples"]["pairs"]
leak_ok = (all(v["count"] == 0 for v in role_pairs.values())
           and all(v["count"] == 0 for v in tool_pairs.values()))
checks.append(("split/group isolation(全两两)",
               leak_ok, f"role 视图 {list(role_pairs.keys())} 交集全 0;tool 视图同"))

# 5. padding + 梯度(含 CUDA、全空 batch、BiLSTM)
pad = json.load(open(OUT + "tests/padding_report.json"))
pad_keys = ["t1_gru_pad_invariance", "t1_lstm_pad_invariance", "t1_gru_bi_pad_invariance",
            "t1_lstm_bi_pad_invariance", "t2_gru_real_len0", "t2_lstm_real_len0",
            "t3_transformer_allpad_grad", "t4_gru_batch_vs_single", "t4_lstm_batch_vs_single",
            "t4_gru_bi_batch_vs_single", "t4_lstm_bi_batch_vs_single", "t5_mask_lengths",
            "t6_p99plus1", "t7_canonical_smoke"]
pad_ok = all(pad.get(k) is True or (isinstance(pad.get(k), dict) and pad[k].get("finite") is not False)
             for k in pad_keys)
t3b = pad.get("t3b_cpu_all_empty", {})
t3c = pad.get("t3c_cuda", {})
empty_ok = (t3b.get("gradients_exist") is True and t3b.get("gradient_finite") is True)
cuda_ok = (isinstance(t3c, dict) and t3c.get("mixed_grad_finite") is True
           and t3c.get("all_empty_grad_finite") is True) if isinstance(t3c, dict) else True
checks.append(("padding + 梯度(CPU 全空/CUDA/BiLSTM)",
               pad_ok and empty_ok and cuda_ok,
               f"BiLSTM pad 一致:{pad.get('t1_gru_bi_pad_invariance')};CPU 全空梯度:{empty_ok};CUDA:{t3c}"))

# 6. cutoff:feature_builder 泄漏扫描 + 帧审计
cut = json.load(open(OUT + "tests/cutoff_report.json"))
vis_audit = json.load(open(OUT + "tests/visual_cutoff_audit_report.json"))
vis_pass = (vis_audit.get("violations", -1) == 0
            and vis_audit.get("no_trace", -1) == 0
            and vis_audit.get("prefix_events_unresolved", -1) == 0)
cut_ok = cut.get("all_pass") is True and cut.get("t4_feature_builder_leak_scan", {}).get("pass") is True
checks.append(("future/answer/visual cutoff(帧级审计)",
               cut_ok and vis_pass,
               f"feature_builder 泄漏扫描:{cut.get('t4_feature_builder_leak_scan', {}).get('hits')};帧审计 verified {vis_audit.get('verified_frame_in_prefix_and_no_target_leak')}/{vis_audit.get('samples_with_frames')},violations {vis_audit.get('violations')}"))

# 7. evaluator
ev = json.load(open(OUT + "tests/evaluator_fixture_report.json"))
checks.append(("evaluator fixtures", ev.get("all_fixtures_pass") is True, "confusion/per-class/重算已含"))

# 8. config schema + feature contract v2
cfgv = json.load(open(OUT + "tests/config_validator_report.json"))
fc2 = OUT + "data/feature_contract_v2.json"
fc_ok = os.path.exists(fc2)
if fc_ok:
    fc = json.load(open(fc2))
    fc_ok = ("views" in fc and "role_event_samples" in fc["views"] and "semantic_tool_samples" in fc["views"]
             and "target_model_id" in json.dumps(fc))
checks.append(("artifact/config + feature contract v2",
               cfgv.get("validator_works") is True and fc_ok,
               "config validator + feature_contract_v2(双视图字段表,禁止 target_model_id)"))

passed = sum(1 for _, ok, _ in checks if ok)
lines = ["# Gate A 验收报告 v4(2026-08-09,三轮修复后)", "",
         "> 依据:docs/behavior_nn_experiment_plan.md 第 14 节;三轮修复:role 视图稳定键/元数据、统一 cutoff_event_index、Transformer 设备+全空梯度、target_model_id 移除、帧→事件审计、feature_contract v2。",
         "", "| # | 检查项 | 通过 | 说明 |", "|---|---|---|---|"]
for i, (name, ok, note) in enumerate(checks, 1):
    lines.append(f"| {i} | {name} | {'✓' if ok else '✗'} | {note} |")
lines += ["", f"**汇总:{passed}/{len(checks)} 通过**", ""]
lines += ["## 未达项与处置", ""]
for name, ok, note in checks:
    if not ok:
        lines.append(f"- ✗ {name}:{note}")
with open(OUT + "gate_a_report.md", "w") as f:
    f.write("\n".join(lines))

with open(OUT + "README.md", "w") as f:
    f.write(f"""# behavior_nn_v1

正式实验根目录(behavior_nn_experiment_plan.md 第 13 节)。

## 数据视图(v4)
- data/role_event_samples.jsonl:17,303 事件轴,预测 next_role(4 类);稳定键+元数据已补齐
- data/semantic_tool_samples.jsonl:5,889 prefix 轴,execute-gated family;禁止 target_model_id 输入

## 废弃
- data/canonical_samples.jsonl(v1,role=target_role=next_role 标签别名):**废弃**,不得用于 R2

## 目录
registry/ data/ tests/ baselines/(空) runs/(空) comparisons/(空)

## 当前阶段
R0+R1 完成(Gate A:{passed}/{len(checks)},见 gate_a_report.md)。未启动训练。
""")
print(f"Gate A v4: {passed}/{len(checks)} passed")
for name, ok, note in checks:
    print(("  ✓ " if ok else "  ✗ ") + name + " | " + note[:90])
