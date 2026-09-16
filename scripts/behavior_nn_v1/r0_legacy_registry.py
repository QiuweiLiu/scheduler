#!/usr/bin/env python3
"""R0: legacy result registry(旧结果清算,behavior_nn_experiment_plan 第 2 节)

分级表来源:docs/behavior_nn_experiment_plan.md 第 2 节
产物:
  results/processed/behavior_nn_v1/registry/legacy_result_registry.json
  results/processed/behavior_nn_v1/registry/legacy_result_registry.md
"""
import hashlib
import json
import os

D = "/root/autodl-tmp/scheduler/"
OUT = D + "results/processed/behavior_nn_v1/registry/"
os.makedirs(OUT, exist_ok=True)

LEGACY = [
    {"id": "role_xgb_0.9322", "file": "results/processed/feat_round2_partB_20260806.json",
     "task": "next role Top-1", "data": "role_dataset_v0_3", "split": "240/30/30",
     "defect": "元数据(vid_meta)按 video 最后一条覆盖,64/300 视频 domain 被空记录覆盖,影响约 729/1520 test 事件",
     "grade": "暂定基线", "action": "R2 重跑锁定", "note": "数字本身来自正确 split 过滤;需修 metadata join 后复核"},
    {"id": "family_xgb_0.8060", "file": "results/processed/nn_residual_model_20260809.json",
     "task": "next action family Top-1 (oracle-gated, 562)", "data": "role_dataset_v0_3", "split": "240/30/30",
     "defect": "完整特征 XGB;仍受 metadata 覆盖影响",
     "grade": "暂定基线", "action": "R2 重跑锁定"},
    {"id": "joint_xgb_0.8507", "file": "results/processed/nn_residual_model_20260809.json",
     "task": "pipeline joint", "data": "role_dataset_v0_3", "split": "240/30/30",
     "defect": "route/layer 旧实现分母不正确(非 execute 样本 executor=None 抬高),joint 本身有意义",
     "grade": "暂定基线", "action": "用新 evaluator 重算"},
    {"id": "ft_v2_0.8060", "file": "results/processed/nn_improve_script2_20260807.json",
     "task": "next family Top-1", "data": "role_dataset_v0_3", "split": "240/30/30",
     "defect": "静态结构已修(z-score 泄漏修复+2-block),但共享 metadata 仍可能错误",
     "grade": "部分有效", "action": "R2 重跑"},
    {"id": "bilstm_0.8043", "file": "results/processed/nn_bilstm_retest_20260807.json",
     "task": "next family Top-1", "data": "role_dataset_v0_3", "split": "240/30/30",
     "defect": "右 padding 后直接读隐藏状态,无 pack/mask;z-score 已修但 padding 未修",
     "grade": "作废待重跑", "action": "正确 mask 后重跑"},
    {"id": "gru_0.8096", "file": "results/processed/nn_bilstm_retest_20260807.json",
     "task": "next family Top-1", "data": "role_dataset_v0_3", "split": "240/30/30",
     "defect": "同 bilstm:右 padding 无 mask",
     "grade": "作废待重跑", "action": "正确 mask 后重跑"},
    {"id": "allnn_0.8059", "file": "results/processed/nn_both_gru_20260807.json",
     "task": "全 NN(GRU+GRU)joint", "data": "role_dataset_v0_3", "split": "240/30/30",
     "defect": "角色/动作 GRU 均有 padding 问题",
     "grade": "作废待重跑", "action": "正确 mask 后重跑"},
    {"id": "residual_0.8025", "file": "results/processed/nn_residual_model_20260809.json",
     "task": "XGB+GRU residual family Top-1", "data": "role_dataset_v0_3", "split": "240/30/30",
     "defect": "padding 未修;未与无序列 residual(B09)比较;temporal 改善可能是校准而非序列",
     "grade": "诊断性线索", "action": "R2/N1 重跑,加 B09 对照"},
    {"id": "oof_0.7900", "file": "results/processed/nn_improve_script1_20260807.json",
     "task": "OOF stacking(GroupKFold)family Top-1", "data": "role_dataset_v0_3", "split": "240/30/30",
     "defect": "GroupKFold 已修;共享 metadata 仍错误",
     "grade": "部分有效", "action": "作为简单 ensemble 对照重跑"},
    {"id": "fm_0.7705", "file": "results/processed/nn_improve_script2_20260807.json",
     "task": "MLP+FM(真交互)family Top-1", "data": "role_dataset_v0_3", "split": "240/30/30",
     "defect": "真 FM 已实现且无增益(历史负结果)",
     "grade": "历史负结果", "action": "核心 MLP 变化 <1pp 时不重跑"},
    {"id": "focal_cw", "file": "results/processed/nn_improve_script1_20260807.json",
     "task": "focal/class weight family", "data": "role_dataset_v0_3", "split": "240/30/30",
     "defect": "曾降低总体表现",
     "grade": "历史负结果", "action": "不作为第一轮主实验"},
    {"id": "finegrained_v04", "file": "results/processed/benchmark_role5_*.json(历史)",
     "task": "fine-grained v0.4 role", "data": "64 视频旧数据集", "split": "旧 split",
     "defect": "标签空间、样本和 split 与当前 coarse-role 不同",
     "grade": "历史独立口径", "action": "只在附录单独报告"},
    {"id": "multistep_behavior", "file": "results/processed/multistep_experiment_20260807.json",
     "task": "多步(事件轴角色/execute 轴族/资源)per-horizon", "data": "role_dataset_v0_3", "split": "240/30/30",
     "defect": "execute 轴 NN 长尾丢失;资源侧 load 样本不足;角色多步为直接 XGB 可信",
     "grade": "部分有效", "action": "角色多步保留参考;族/资源多步 R2 后重审"},
    {"id": "nn_ablation_multistep", "file": "results/processed/nn_ablation_multistep_20260807.json",
     "task": "NN 容量/模型族消融(execute 轴 K=3)", "data": "role_dataset_v0_3", "split": "240/30/30",
     "defect": "与 multistep 同管线;长尾类丢失已证实;验证确定性/seed 敏感(0-2/783)",
     "grade": "诊断性", "action": "作 R2 序列模型的动机来源,不直接引用"},
]

registry = {"version": "v1", "created": "2026-08-09", "plan_doc": "docs/behavior_nn_experiment_plan.md",
            "entries": []}
for e in LEGACY:
    path = D + e["file"]
    exists = os.path.exists(path)
    sha = None
    if exists:
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    entry = dict(e)
    entry["file_exists"] = exists
    entry["sha256_prefix"] = sha
    registry["entries"].append(entry)

with open(OUT + "legacy_result_registry.json", "w") as f:
    json.dump(registry, f, ensure_ascii=False, indent=2)

md = ["# Legacy Result Registry(v1, 2026-08-09)",
      "",
      "> 来源:docs/behavior_nn_experiment_plan.md 第 2 节;旧文件不删除不覆盖。",
      "> 正式根目录:results/processed/behavior_nn_v1/",
      "",
      "| ID | 数字 | 分级 | 已知缺陷 | 后续动作 | 文件存在 |",
      "|---|---|---|---|---|---|"]
for e in registry["entries"]:
    md.append(f"| {e['id']} | {e['grade'].split()[0]} {e.get('note','')} | {e['grade']} | {e['defect']} | {e['action']} | {e['file_exists']} |")
md += ["", "每条原始文件与 sha256 前缀见 legacy_result_registry.json。"]
with open(OUT + "legacy_result_registry.md", "w") as f:
    f.write("\n".join(md))
print("registry written:", OUT)
print(f"entries: {len(registry['entries'])}, exists: {sum(1 for e in registry['entries'] if e['file_exists'])}")
