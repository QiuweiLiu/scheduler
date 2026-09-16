# J 预测器 → 调度器接口：打包与契约 smoke（2026-09-11）

状态：**打包完成 + P_dev 侧契约 smoke 通过**；S_* 侧 smoke 待做（需要 anchor feature builder）。

## 1. 打包产物（J3:seed11）

命令：
```sh
PYTHONPATH=src python scripts/pack_j_predictor_artifacts.py \
  --config experiments/EXP-20260911_p9d_j_predictor_acceptance/config.json \
  --split validation --output-root outputs/j_predictor_artifacts/validation
```

产物（`outputs/j_predictor_artifacts/validation/`，2029 anchors，12s）：

| 文件 | 内容 | sha256 |
|---|---|---|
| `j_future_h1.jsonl.gz` | 1 步事件场景（legacy 风格） | `f4281f5e…` |
| `j_future_h3.jsonl.gz` | 3 步 | `0a7a67e3…` |
| `j_future_h5.jsonl.gz` | 5 步 | `11ad3809…` |
| `j_future_h5_layers.jsonl.gz` | layer-H5 侧车（schema `scheduling-future-h5-layer-v1`） | `207ae9d4…` |

- 预测节点数：mean 3.55，min 0，max 5；单场景（prob 1.0）、不做 DAG 宽度声明。
- 每步字段：role/action_family/model_id + 概率、execution_lane（启发式映射）、raw_action（train-mode 映射）、merged/retry，以及 `resource` 扩展块（runtime p50/p90/p95、load occurrence prob、load duration p50/p90/p95）。
- 契约校验：`project_event_scenarios_to_unary_layers` + `validate_layer_scenarios` 通过。

## 2. 调度器 loader smoke（通过）

把产物按调度器命名放入 mock artifact 目录后，直接调用调度器自己的加载器：

```
load_future_artifacts(.scratch/j_artifact_smoke/prediction_artifacts)
→ loaded nodes: 2029
→ row keys: future_h1 / future_h3 / future_h5 / future_h5_layers
→ layer scenario: 5 layers，契约字段齐全
→ event step: 全部 legacy 字段 + resource 扩展
```

## 3. 关键集成事实（会影响步骤 3–4 的设计）

1. **调度器现有策略不消费我们的资源分位数**：它用自带的资源表（按 model_id/role 的统计）估算成本；我们的 `resource` 块是新增字段，需要新策略或适配层才会被使用。→ 步骤 3 里要明确"是否/如何用预测器的 runtime 分位数"。
2. **模型输出与 legacy 字段的映射**：`raw_action` 由 train-mode 映射（action_family→raw_action）填补，`execution_lane` 为启发式；这两项是近似，不是模型预测。
3. **单场景确定链**：我们输出一条最可能的执行链（长度=argmax），与 legacy 的 top-3 场景格式兼容但只给 1 个场景。

## 4. S_* 侧 smoke 的剩余前置

- S_* traces 已定位：`results/raw/r7_trace_full_20260817`（640 runs，含 `trace.jsonl`/`run_manifest.json`/`states/`）。
- 缺：**anchor feature builder**——把 S_* 事件链转成预测器输入（history/current/context），任务上下文（answer_type/domain/…）需要从 trace manifest 与事件输入重建；P9d builder（`scripts/build_p9d_topology_dataset.py`）可作模板，但它的源是 P_dev 专用数据集，不能直接复用。
- 目标：对 S_train/S_val 的每个事件生成 anchor → 跑 J3:seed11 推理 → 产出 scheduler artifacts → 用同样的 loader smoke + 覆盖率/契约检查（只推理，不拟合）。

## 5. S_* 推理 smoke（2026-09-11 完成）

数据管道（新增 `scripts/build_sstar_predictor_anchors.py`，复用 P9d builder 的字段派生）：
- 输入：`results/raw/r7_trace_full_20260817`（640 runs）；输出：`outputs/sstar_predictor_anchors/features_sstar.jsonl.gz`。
- 结果：**9,575 anchors，0 跳过，23 秒**（每个事件一个锚点）。
- OOV（相对冻结词表）：event_type/node_type/raw_action/action_family = **0%**；model_id 6.0%；model_stack_id 48.9%（stack_a/yolo26n 为新栈）；temporal_scope、planner_model_id 100%（新取值，映射 UNK——训练机制支持）。

推理 + 打包（`scripts/pack_j_predictor_artifacts.py --anchors-file`）：
- 产出 `outputs/sstar_predictor_artifacts/`（h1/h3/h5 + layer 侧车），**24 秒**；layer 校验通过。
- 调度器 loader 兼容性：与 P_dev 相同路径可读取。

对 S_* 真实执行结果的审计（`scripts/sstar_inference_smoke_audit.py`，5 秒；只对比不调参）：

| 指标 | S_*（9,575 anchors） | P_dev validation（域内参照） |
|---|---|---|
| next-role 准确率 | 0.912 | 0.944 |
| next-family 准确率 | 0.803 | 0.840 |
| runtime pinball（raw ms） | 1,045.5 | 845.0 |
| load 发生 Brier | 0.0225 | 0.0135 |
| 长度绝对误差（剩余事件数，cap 5） | 0.408 | —（定义不同） |
| load>0 步的 p50 绝对误差 | 888 ms | —（口径不同） |

分组：
- s_train vs s_val 几乎一致（role 0.911 vs 0.913；family 0.801 vs 0.811；pinball 1,042 vs 1,058）→ 无 split 特异异常，且没有用于选择。
- 按基线：**star 明显好于 langgraph_react**（role 0.947 vs 0.878；family 0.831 vs 0.777）。
- 结论：跨到 S_*（含新栈 stack_a/yolo26n、未见上下文取值）**没有崩溃，但可测地退化**（role −3.2pp、family −3.7pp、runtime 误差 +24%、Brier +67%）——属于预期的分布偏移，调度器实验需按此预期校准。

## 6. 结论（更新）

- 步骤 1（接口打包）与步骤 2（契约 smoke：P_dev + S_* 两侧）**已完成**，总计算成本 <1 分钟。
- 剩余：步骤 3（调度器级验收标准 + 是否接资源适配层）与步骤 4（小规模 PredOpt 矩阵）。
