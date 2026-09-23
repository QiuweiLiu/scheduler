# HANDOFF

## Goal
复现四条调度路线并接入项目调度器，回答：agent workflow 的未来信息应以
runtime / risk / topology / information value 中的哪种形式进入 GPU 调度。

## Done
- **主 baseline 冻结为 F0**（sameshape_h5_p95，F0 包）；A0(J3) 降为历史基线；
  delta_NI = 485 ms 继续用不重算。已写入 DECISIONS + EXPERIMENT_GATE
  （新实验 EXP-20260921_scheduler_replication_v1）。
- **SRTF+Aging 实现完成**：新增 policy srtf_h5_p95_aging；
  key = (hard_priority, R - wait_age, R, -wait_age, ...)，R = current + sum_h future_p95；
  纯函数 srtf_aging_key() 抽出以便精确断言。
- **F0 包重建**：补 cvar95_ms（原缺失导致 loader 拒绝）、视图改 float64
  （原 float32 导致 1.18e-03 误差）。现 loader 接受：
  nodes=9575 steps=35362 view_err=3.64e-12 nonresource_mismatch=0
  sha256 57f9627a3f5f7b61f50416f01955cf2c944511260e7425e378d02b5bbca95d9b。

## Verified
- **SRTF+Aging 第一层门禁 PASS**：	ests/test_srtf_aging_fidelity.py 10/10。
  覆盖：手算 key 精确、1ms 等待=1ms credit、等待单调、硬优先级不可跨越（测到 1e12 ms）、
  负等待钳零、future 项 = H5 链成本。
- **无回归**：sameshape 19/19、j_series_resource_dist 23/23、phase_r_reproduction 4/4、
  histres_causal_guard 7/7。
- 3 集冒烟跑通：F0 基准 80,006 / srtf_h5_p95_aging 87,872 / sjf_pred 83,187
  （**3 集无统计意义，仅证明可运行**）。

## Rejected
- 未采用「F0 包无需重建」的判断 —— 验证器抓到缺 cvar95_ms 与 float32 视图两个真实缺陷。

## Open
- SRTF+Aging 的 **shadow scoring** 与 **300 配对性能跑**未做。
- TIE / Pythia-Graph / LLMSched 未开始。
- SRTF+Aging 的命名限制须写进论文：**非抢占** SRTF/SRPT-inspired（simulator 无节点级抢占）。
- 第二层门禁（performance）判据：Delta = new - F0；非劣 CI_upper < +485；
  统计改善 CI_upper < 0；实质改善 point <= -485 且 CI_upper < 0。

## Active
无长时任务在跑。

## Next
1. SRTF+Aging：shadow scoring（同池 6 分数 → rank/分歧/决策开销）→ 30 集冒烟 → 300 配对跑。
2. 按 GPT 排序做 **TIE**（16 格分布现成，公式近乎原样移植，性价比最高）。
3. 再 Pythia-Graph（需新增 per-model 队列深度状态），最后 LLMSched（需 train-only MI profiler）。
4. 每条路线走同一流程：fidelity → shadow → 30 集冒烟 → 300 配对跑。
