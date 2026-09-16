# VideoSeek 调度研究项目

## Purpose

围绕视频 Agent 的动态行为 trace，验证有限前瞻预测如何改善资源预测和多 GPU 调度；先保证行为、资源、到达和调度输入可复现，再开展 Optimizer-H、RL-H 与预测式策略比较。

## Scope

- In scope: trace schema/采集适配器、行为/资源预测器、因果 workload、节点级调度模拟、C1 回顾性与 C2 全新视频验证。
- Out of scope: 不把历史诊断结果冒充正式 C1/C2；不覆盖原始视频/模型/trace；不在本阶段引入真实电梯视频迁移。

## Constraints

- 当前独立本地执行副本优先使用：`/Volumes/Lenovo/scheduler`；远端 `root@connect.westc.seetacloud.com:12469:/root/autodl-tmp/scheduler` 仅作为已同步源/备份，后续实验不默认依赖远端。
- 本地源码唯一当前路径：`src/tracing/`；包导入名保持 `tracing`，通过 `PYTHONPATH=src` 解析。
- 768 条旧轨迹继续可用，但标记为 `trace_collection_v1_legacy_core`；600 视频扩展产生 `trace_collection_v2_expanded`，不覆盖 v1。
- Git 不包含视频、模型权重、原始 trace、缓存和大型临时输出；清理和删除仍需单独确认。

## Architecture Overview

`src/tracing` 提供 schema、collectors、validators、analysis 和 workloads；`scripts/` 负责可复现入口；`data/manifests/` 登记远端/本地数据版本；`experiments/` 保存正式实验配置和指标；`.project/` 是唯一项目控制面。

## Canonical Locations

- Code: `src/`、`scripts/`
- Tests: `tests/`
- Data: `data/raw/`（只读）、`data/interim/`、`data/processed/`、`data/manifests/`
- Experiments: `experiments/EXP-xxx_name/`
- Outputs: `outputs/`（临时）
- Durable design: `docs/`
- Control plane: `.project/`
