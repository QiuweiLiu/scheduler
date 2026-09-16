# VideoSeek 调度项目说明

本文件只记录本项目约束；用户指令和全局 Codex 规则优先级更高。

## Canonical control plane

续接任务时按以下顺序读取：`.project/PROJECT.md`、`.project/STATE.md`、`.project/PLAN.md`、`.project/HANDOFF.md`，再读取相关 `docs/`、`experiments/`、`data/` 和脚本。当前计划只保留在 `.project/PLAN.md`，旧计划进入 `.project/archive/`，不创建平行的 `plan_v2` 或 `summary_final`。

## 项目边界

- 执行环境以远端 `/root/autodl-tmp/scheduler` 为准；本地目录是控制面、代码镜像和小型清单。
- `src/tracing/` 是 tracing 源码的唯一当前路径；Python 包名仍为 `tracing`，运行时使用 `PYTHONPATH=src`，禁止恢复第二份长期源码。
- `data/raw/` 和远端原始视频、模型权重只读；768 条旧轨迹保留为 `trace_collection_v1_legacy_core`，扩展到 600 视频后的轨迹必须登记为新 collection，不覆盖旧数据。
- Git 只提交代码、schema、配置、清单、文档和小型指标；不提交视频、模型权重、原始 trace、缓存和大型临时输出。

## 修改与验证

- 先 inspect，再执行已确认的修改；移动/归档/删除前确认目标和可回滚路径。
- 源码路径变化后必须同步导入、CLI、shell 脚本、测试命令和当前文档，并运行编译/导入 smoke test。
- 远端改动必须记录主机、路径、命令、结果和失败原因；不能把远端存在误写成本地已同步。

## 受保护路径

- 原始视频、模型目录、`results/raw/`、已有 768 条轨迹及其哈希不可覆盖。
- `outputs/` 只放临时运行产物；长期事实写入 `data/manifests/`、`experiments/` 或 `.project/`。
