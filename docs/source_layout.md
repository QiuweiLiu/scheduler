# 当前源码与数据布局

更新时间：2026-08-13。

## 当前约定

`tracing` 是项目源码包，不是数据集。因此当前唯一源码目录是：

```text
src/tracing/
  schema/       # trace/state/task manifest schema
  collectors/   # trace collectors and model adapters
  validators/   # raw trace validation
  analysis/     # enrichment, predictors, simulators
  workloads/    # trace -> template -> episode
```

包名不变，仍然使用：

```python
from tracing.analysis.trace_predictor import PlannerAwareTracePredictor
```

从项目根目录运行时需要：

```bash
export PYTHONPATH="$PWD/src"
python -m tracing.analysis.phase4_trace_simulator --help
```

直接执行源码文件时使用 `src/tracing/...` 路径；shell 入口已经同步这一约定。

## 为什么不放进 `data/`

`data/` 只承载原始数据、派生数据和数据清单；源码需要版本控制、导入和测试，生命周期与视频/trace 不同。把 tracing 放进 `data/` 会让数据清理或挂载操作误伤代码，也无法表达源码和数据的独立版本。

## 迁移边界

- 本地：根目录 `tracing/` 已移动为 `src/tracing/`。
- 远端：`/root/autodl-tmp/scheduler/tracing/` 已移动为 `/root/autodl-tmp/scheduler/src/tracing/`；迁移前确认无活跃采集进程。
- 原始视频、模型、`results/raw/` 和旧 trace 没有移动或覆盖。
- 历史报告中的 `tracing/...` 是当时的路径记录；当前可执行命令以本文件和 `.project/PLAN.md` 为准。

## 同步验收

1. `find src/tracing -name '*.py' -print0 | xargs -0 -n1 python3 -m py_compile`
2. `PYTHONPATH=src python3 -c 'from tracing.collectors.structured_state import canonical_action; print(canonical_action("image-grid-selector"))'`
3. 远端使用 finetooling Python 和 `PYTHONPATH=/root/autodl-tmp/scheduler/src` 重复第 2 项。
