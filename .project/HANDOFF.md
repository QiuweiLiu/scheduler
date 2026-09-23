# HANDOFF

## Goal
四条**联合基线**（faithful adaptation）+ 完全体证据链。
**当前被 fidelity 缺口阻塞**（GPT 复核：不批准开始正式跑）。

## Done
- **拓扑 edge 回归**已修复（v04）；**节点本体**已对齐 v3.1（**v04.1**）
- **v04.1**：嵌套合并 **169**（与 GPT 计数一致）、`run_control` 排除、
  `resource_applicable` **成为真正执行契约**（Node 字段 + loader 读取 + 3 处 fail-closed）、
  `sequence_index` 重编号、契约改名
  `scheduler_projection_of_verified_serial_control_flow_v3_1`
  - **8,295 → 8,126 节点**；**seriality gate 640/640 PASS**
  - **640 + 8 pilot = 648 provenance 闭环**（pilot-only = 0）
- 四条基线**架构骨架**已搭好（TIE / Pythia / LLMSched / Latency-Aware 独立调度器）
- SRTF+Aging 退役；F0 打包器 attr_feature bug 已修

## Verified
- v04.1 gate 640/640 PASS；现有臂回归正常（16 job 全完成、0 失败）
- 640+8=648 已从 manifest 机械验证

## Open（GPT 复核指出，全部未修）
- **TIE P0**：`estimate()` 丢掉 `runtime_mean_ms`/`runtime_cvar90_ms` → 实际跑的是
  `p50 + β·p90`，**不是 TIE**；我那个 10/10 是**假阳性**。
  还需：专用 `tie_estimate_current()`（缺 mean/CVaR 就 fail）、waiting decay、
  `B` 改成预冻结常量
- **LLMSched**：`H(X) ≠ I(X;Y)`；已完成节点的真实 duration 未进 posterior。**fidelity FAIL**
- **Pythia**：`baseline`（workflow 家族）**不是 role alphabet**。应改名为
  `workflow-family progress prior` 或重构为真正的 role-PFA
- **Latency-Aware**：key **不是 Eq.12**（`-boundaries_removed` 是我发明的 tie-break）；
  且**用了真值 `compute_ms` = 真值泄漏**。91,134 vs 90,535 只能写 "integration probe"
- **49.3% fusion 素材需在 v04.1 上重算**（169 个节点被合并，链结构变了）
- `round_robin` 实跑 myopic（预先存在，未修）
- 预取测试用 3-GPU 合成数据（真实是 2 GPU），需补代表性测试
- `SchedulerTopologyContractGate` + `BaselineFidelityManifest` 未冻结

## Active
无长时任务在跑。

## Next
1. **第 2 步**：修四条基线 fidelity P0，顺序 **TIE → LLMSched → Pythia → Latency-Aware**，
   每条加 **E2E sentinel mutation test**
2. 第 3 步：冻结两个 gate
3. 第 4 步：重跑 30 集 smoke（只看机制）
4. 在 v04.1 上重算 fusion 素材与调度机会
