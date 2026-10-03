# RESULT — EXP-20261003_substrate_preemption_resume_v1（P1：抢占往返验证）

**状态**：完成（2026-10-03）。**远端脚本 `p1_resume_roundtrip.py`；产物 `artifacts/p1_resume.json`。**

## 目的

验证调用级抢占的恢复代价记账：**R_m = F2 prefill 曲线** 是否适用于"打断→丢 KV→重算→继续"这一真实操作。
（此前 R_m 只被"正常 prefill"数据验证过；本实验直接测打断-恢复往返。）

## 方法

同一 prompt（L=1000 token）三种跑法对比（手动 greedy decode，CUDA events 计时）：
- **full**：prefill + N=48 个 decode step
- **interrupted**：prefill + k 个 step → 丢弃 KV → 重算(L+k) → 继续 N−k 个 step
- **theory**：重算段应 = F2 曲线 `intercept + rate×(L+k)`（4B: 8.1+0.11457/tok；8B: 9.8+0.19001/tok）

3 reps，中位数。

## 结果

| 配置 | 总时长差 (int−full) | 重算实测 | F2 预测 | **残差** |
|---|---|---|---|---|
| 4B ∣ L1000 ∣ k=20 | +107.4ms | 120.5ms | 125.9ms | **−4.3%** |
| 4B ∣ L1000 ∣ k=40 | +144.4ms | 121.0ms | 128.2ms | **−5.6%** |
| 8B ∣ L1000 ∣ k=20 | +188.0ms | 201.5ms | 205.1ms | **−1.8%** |

## 结论

1. **R_m = F2 prefill 曲线验证通过**：三配置残差 −1.8%~−5.6%（全部略快于预测，保守方向）。
2. **总时长差 ≈ 重算时间**（107–188ms）：抢占-恢复的附加成本就是 R_m + 极小编排开销（<40ms）。
3. **DES 记账确认**：抢占时"丢弃已花时间、恢复时加 R_m"的模型与真机行为一致。

## 边界

- 手动 decode 循环（与 F2 同款）；单卡共享环境；3 reps。
- 未测多次抢占链、多任务并发下的抢占（留给 DES 仿真验证）。

## 仿真器集成修正（2026-10-03 晚，评审 P1-1/P1-2）

- **P1-1 修正**：RequestSplit 改为以**内在工作量**表达（`prefill_work_ms`/`decode_step_work_ms`），
  相位/token 状态从 `consumed = total_work − remaining_work` 推导——batching 因子、F1 减速、
  伙伴退出复位、加性挡停都会一致移动相位时钟（不再用派发时固定的墙钟偏移）。
  合法抢占点 = `prefill_work + m×d_r`（m≥0）；落在 token 中间时推迟到下一合法边界
  （`victim_deferred_to_token_boundary` + `preemption_boundary` 唤醒事件）。
- **P1-2 修正**：受害者先由调度器可见信息唯一选出；执行层只执行或在物理不可执行时报告/推迟，
  不再用隐藏相位过滤候选或换选 victim。
- 集成冒烟（真实 profile，4B planner victim + 3B priority target，arrival=150ms）：
  推迟到首个 token 边界 **150.108ms**、tokens_done=10、R_m=132.065；恢复完成 342.065 ✓。
- 边界：R_m 验证仍只有 4B/8B 格；3B 来源缺口（评审 P1-5）待决策。
