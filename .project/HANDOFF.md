# HANDOFF

## Goal
四条**联合基线**（faithful adaptation）+ 完全体证据链。
**substrate Freeze v1 已正式签字**（HEAD `9aedf2b`，implementation closure `af78ec8`）；
下一步按 LLM-1…LLM-6 修 LLMSched。

> **Freeze 声明**：除非 baseline 实现暴露出**可复现的 substrate correctness bug**，
> 否则不再改动 topology / composite / admission / TIE / raw rebuild / preemption 语义。
> LLMSched 实施方案全文见 `docs/research/2026-09-24_llmsched_adapted_implementation_plan.md`
> （LLM-1…6 + L1–L7 gate + artifact manifest 的 faithful/adapted 清单）。

## Done
- **拓扑 edge 回归**已修复（v04）；**节点本体**已对齐 v3.1（**v04.1**）
- **substrate Freeze v1 批准**（GPT，HEAD `2052a8f` + `af78ec8`）。范围：
  v04.1 ontology/topology、composite 非抢占执行语义、GPU placement/admission、TIE、gate 与 raw rebuild。
  `myopic_preempt` / `batch_myopic_preempt` 作为**非正式 extension**，已由 sentinel 8 关闭 reservation-preemption 交互。
- **substrate 关闭证据**：
  1. **provenance-based gate**（`scripts/scheduler_projection_gate_v041.py`）**640/640 PASS，0 违规**
  2. **169 对嵌套时间窗 VERIFIED**：169/169 包含、169/169 `parent_runtime == interval`
  3. **648 在执行 run_id 层闭环**：640 formal + 8 pilot = 648，交集 0
  4. **169 集合相等**：独立从 raw 重发现 vs projection → 0 假阳、0 假阴
- v04.1：8,295 → **8,126 节点**（合并 169 嵌套）、`resource_applicable` **强制契约**、
  `sequence_index` 重编号、契约名 `scheduler_projection_of_verified_serial_control_flow_v3_1`
- **composite CPU+GPU 段**（option b）状态机：`parent_start → nested_ready → nested_gpu_start →
  nested_gpu_finish → parent_finish`；`R_total = R_pre + R_nested + R_post`。
  **sentinel 8/8**（含 sentinel 7 抢占、sentinel 8 queued-reservation 抢占）
- TIE **fidelity 21/21**（含 Lq 唯一性 3 条 + load 归属 3 条）；真实 bank：5 组、0 缺字段、
  `fraction_lt_10=0`、`fraction_singleton=0`、`tie_load_estimate` 3978.7–6538.4 ms
- SRTF+Aging 退役为经典调度**负对照臂**；F0 打包器 attr_feature bug 已修（新 sha `586ae65d…`）

## Verified
- v04.1 gate 640/640；嵌套包含 169/169；执行层 648 交集 0；集合相等 EXACT；
- composite sentinel 8/8；TIE 21/21；全量 **295 测试**（5 个失败套件**全部预先存在**）

## Open（GPT 复核指出）
- **LLMSched fidelity FAIL**：`H(X) ≠ I(X;Y)`；真实 duration 未进 posterior。
  修复走 **LLM-1…LLM-6** 六阶段（删假 BN frontend，旧的留作 legacy oracle），按 **L1–L7** sentinel 验证
- **Pythia**：`baseline` 不是 role alphabet。应改名 `workflow-family progress prior` 或重构为 role-PFA
- **Latency-Aware**：key 不是 Eq.12（`-boundaries_removed` 是发明的 tie-break）；**用了真值 `compute_ms` = 真值泄漏**；
  还需**自己的 train-only request-conditioned predictor** + 非退化 gate
- **49.3% fusion 素材需在 v04.1 上重算**
- `round_robin` 实跑 myopic（预先存在，未修）
- 预取测试用 3-GPU 合成数据（真实 2 GPU），需补代表性测试
- `SchedulerTopologyContractGate` + `BaselineFidelityManifest` 未冻结
- 三条共享模块建议（`verify_raw_run` → `canonicalize_v31` → `project_scheduler_chain`）未做

## Active
LLM-1…LLM-5 已写出并推送（`111b3e4`），但**消费端尚未迁移**——这是刻意的中间状态，不是遗漏。
两个已实测的阻塞点见下。

`llmsched_bn_legacy.py` 保存退役实现（逐字节），其测试套件已改为显式指向它，使「退役」可重放。
`src/tracing/analysis/workload_v02_simulator.py` 中 `policy == "llmsched"` 分支暂时 import
`llmsched_bn_legacy`，并标注 **NOT YET MIGRATED**。

### 已完成（`111b3e4`）
- **LLM-2** `src/tracing/analysis/llmsched_stage.py`：canonical stage ontology
  = `(lane, role, action_family, raw_action)` + **prefix-only** occurrence；
  `sequence_index` 只作**顺序**、绝不作 identity。v04.1 上得 **55 个 stage / 480 train 模板**，
  6 个 duration bin（log 空间分位、完全均衡），**76.8 % vocabulary 条目为 ABSENT**。
  不用 `template.baseline` 作 application family；非 train 模板跳过并计数。
- **LLM-3** `llmsched_bn.py` v2：每 stage 一个离散变量 `{ABSENT, D0..D5}`，
  按条件互信息贪心选父，CPD **对所有父组合补全** + 平滑。净得 **68 边 / max indegree 2 /
  induced width 4**（guard 6 内）。两项额外约束记录为 adaptation 并附原因：**lag window**
  （纯因果序贪心在 480 样本上过拟合到 induced width 53）与 **min gain**。
- **LLM-4** 精确变量消元（证据折入因子）。**L1 全过**：`P(A)` / `P(B|A)` / `P(C|A)` /
  `P(B,C|A)` / `P(B|A,C)` / `P(C|A,B)` 与手工值**精确一致**。
- **LLM-5** `uncertainty_reduction = 互信息 × Range`；`draw_mode` 保持**每次决策一枚硬币**。

### L1 抓出的三个真 bug（都是「静默返回貌似合理的错值」）
1. 观测到的**后代**被「只走祖先」的剪枝丢掉 → `P(B=t|A=t,C=f)` 返回 0.9 而非 0.75
2. 因子相乘时用**错误的操作数**去翻译 key 索引
3. 观测到的**子节点**保留了自身坐标轴 → 被当作隐变量求和掉而非钉住

### 两个阻塞点（均已实测，未修）
1. **精确推断 ~2 s / 次查询**：消元循环每步都对每个剩余变量重扫全部因子。调度内循环付不起。
2. **`∏ Range` 在 30 个后代上爆到 1e+72**：一个 stage 在 55-stage vocabulary 上有多达 30 个
   可达后代，`∏ Range` 被用在了远大于「单个 workflow 剩余 stage」的集合上。

## Next
1. **修 LLM-4 性能**：按 canonical order 做消元 + 预计算 pairwise 边际（或按 evidence 签名缓存），
   目标 µs 级；然后把消费端从 legacy 切到 v2（去掉 NOT YET MIGRATED 标记）
2. **定 `Range` 的 scope**：把 Y 限制为「当前 workflow 剩余 canonical stage」或最近 k 个后代，
   使 `∏ Range` 与原论文语义一致
3. 补 **L2–L7** gate（`tests/test_llmsched_bn_v2.py`），尤其 **L2 同 entropy 不同 MI**
4. Pythia role-PFA 重构；Latency-Aware 换成自训练 request-conditioned predictor
5. 冻结 `SchedulerTopologyContractGate` + `BaselineFidelityManifest`
6. 重跑 30 集 smoke（只看机制）；在 v04.1 上重算 fusion 素材与调度机会

