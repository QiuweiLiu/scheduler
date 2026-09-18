# Phase 17 — runtime / load 契约审计（P0）

**Date:** 2026-09-17
**Status:** `audit_done_fix_pending`（只读审计完成；**未修改任何消费者、未重跑任何实验、未改动任何既有数字**）
**Trigger:** GPT 审阅（`docs/research/2026-09-17_fas_phase16_sameshape_review_gpt.md`）提出的 P0：
"确认 J 预测器的 `runtime_ms_quantiles` 是 total runtime 还是 compute-only；若含 load，则 `runtime + load` 有重复计 load 风险。"

## 结论

**`runtime_ms` 是包含 load 的完整墙钟时间；J 预测器的 runtime 头直接以它为标签。**
因此凡是在 runtime 之上再加一个 load 项的消费者，都在**对 load 重复计数**。这不是个别臂的问题，而是表路径与预测路径共有的系统性契约缺陷。

## 证据链（四层，全部 file:line）

| 层 | 事实 | 证据 |
|---|---|---|
| schema/Node | runtime = compute + load（load 是 runtime 的子区间） | `src/tracing/analysis/workload_v02_simulator.py`：`compute_ms = max(0.1, self.runtime_ms - number(self.load_ms))` |
| 文档 | "完整墙钟写入 runtime_ms"；"首次冷运行 runtime = load + 服务时间 + 协议开销"；并明确警告"若 runtime_ms 已含 cold loading，就不能再加 load" | `docs/VideoSeek_R0a_只读审计报告_20260813.md`；`docs/VideoSeek_有限前瞻调度_Optimizer-H_RL-H_实施规划.md` |
| J 标签 | runtime 头目标 = **原始 `runtime_ms`**（不做任何 load 扣减）；load 由独立头预测 | `scripts/build_j_dataset.py`（`resource.get("runtime_ms")` 直接进 targets）；`j_series_common.py`：`runtime_target = torch.log1p(batch['runtime_ms'])`、`dur_target = torch.log1p(batch['load_ms'])` |
| artifact | `resource.runtime_ms_quantiles` = runtime 头输出，**packer 不做 load 扣减**；load 分位数来自独立头 | `scripts/pack_j_predictor_artifacts.py`：`rt = torch.expm1(o["resource"]["runtime_log_quantiles"])` 与 `ld = torch.expm1(o["resource"]["load_dur_log_quantiles"])` 并列发射 |

### 数据端交叉验证（不是只靠注释）

在 `results/processed/j_series_dataset_v1/j_validation.jsonl.gz` 上（2,029 rows / 7,195 steps）：

- `load_ms > 0` 的步：**546**；
- 其中 `runtime_ms < load_ms` 的：**0 条（0.000%）**；
- `load_ms / runtime_ms`：p50 **0.373**、p90 0.444、max 0.539。

若 `runtime_ms` 是 compute-only，冷加载占主导的步必然出现 `runtime < load`；实测一条都没有，且比值被紧紧限制在 (0, 0.55) —— 与"runtime 是含 load 的完整墙钟"完全一致，与 `compute_ms` 定义一致。

### 影响量化（冻结 artifact `b05_future_h5`，9,575 anchors / 35,362 steps）

- GPU 预测步 **24,973**；其中被 `load_occurrence_probability ≥ 0.5` 门控（即会加 load 项）：**1,921 = 7.69%**；
- 这些被门控步上，**附加 load p95 / runtime p95：p50 = 0.402**（mean 0.401，max 0.568）。

即：约 7.7% 的 GPU 未来步被系统性**高估约 40%**。这也解释了为什么 `predopt_h5_rt95`（不带 load 项）与 q95 只差 +11 ms —— 受影响步占比小，但它恰好集中在**冷加载步**，也就是机制叙事最关心的那部分。

## 受影响的实现（均在 `workload_v02_simulator.py`）

"runtime + load" 模式（在已含 load 的 runtime 上再加 load）：

- 表路径：`_step_estimate_cost`（`runtime_p50_ms + load_p50_ms`，而 `runtime_p50_ms` = `Node.runtime_ms` 的 p50，已是总量）；
- 预测路径：`_mix_step_cost`、`_q95_step_cost`（**当前冠军**）、`_risk_step_cost`、`_mix95_step_cost`、`_split95_step_cost`、`_jres_step_cost`、`_jrt_step_cost`；
- 各策略 key 的 `current` 项统一为 `runtime_p50_ms + (非驻留 ? load_p50_ms : 0)`，**同样重复计入**（对所有策略一致，因此对相对比较影响较小，但绝对量级被抬高）。

**干净的实现**：`_runtime_only_step_cost`（docstring 明确 "no future load"，只返回 runtime 分位数）——即 `predopt_h5_r95` / `predopt_h5_r50` 族不受此缺陷影响。

## 为什么这很严重

1. 冠军 `predopt_h5_q95` 的**语义描述是错的**：它不是"逐步骤 runtime p95 求和"，而是"逐步骤 **总 runtime** p95 求和 + 部分步再叠加 load p95"。
2. 它影响"冠军 = runtime 尾部"这条语义描述（描述失真，见 1）。但**不影响"load 维度无用"这条结论**——
   此处初版写"ld95 更差可能来自重复计数"，经复核**该说法不成立**，已撤回：`predopt_h5_ld95` 走的是
   `_split95_step_cost(step, stats, runtime_lam=0, load_lam=1)`，即**纯 load 维度作 future 信号**；
   而 `predopt_h5_rt95`（runtime-only，`runtime_lam=1, load_lam=0`）与 q95 只差 +11ms，说明"多出来的那个 load 项"
   在决策上近乎中性，无法解释 ld95 的 +15.1k。→ ld95 的负结论保留，只是其描述应写清"用 load 维度作整体信号"。
3. 它同时污染了表 vs 预测的比较（两侧都重复，但重复比例不同：表侧用 p50 load，预测侧用 p95 load 且带 occurrence 门控）。

## 未做（需要批准）

- **不修改**任何消费者、**不重跑**任何实验：修 `_*_step_cost` 会改变冠军的定义与全部已发布数字，
  属受保护的科学定义变更，需要单独批准 + 新门禁 + 重跑 dev700。
- 待定方案（三选一，供决策）：
  (a) **runtime-only 化**：所有消费者只用 runtime 分位数，彻底删除 load 项（与 `_runtime_only_step_cost` 统一）；
  (b) **compute-only 标签**：J 的 runtime 头改为 `runtime_ms - load_ms`，消费端继续显式加 load（改动最大，需重训/重打包）；
  (c) 只在 `predopt_h5_q95` 上做 (a) 的最小修，作为 Phase16-B 的干净基线。

## 复现命令

```sh
PYTHONPATH=src python -c "..."   # 见本文件"数据端交叉验证"与"影响量化"两节的统计脚本
```
（统计只用 `gzip`/`json`/`statistics`，无需 GPU；本次在 macOS 上以 `/opt/miniconda3/bin/python` 执行。）
