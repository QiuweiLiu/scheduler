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
无长时任务在跑。

## Next
1. **LLMSched 修复**：LLM-1 替换假 BN frontend（删 `P(length|baseline)` / position histogram / `entropy(position)`），
   旧实现保留为 legacy test oracle；按 L1–L7 建 sentinel（含 same-H-different-MI 与 E2E mutation）
2. Pythia role-PFA 重构；Latency-Aware 换成自训练 request-conditioned predictor
3. 冻结 `SchedulerTopologyContractGate` + `BaselineFidelityManifest`
4. 重跑 30 集 smoke（只看机制）
5. 在 v04.1 上重算 fusion 素材与调度机会
