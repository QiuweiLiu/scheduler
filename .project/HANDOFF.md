# HANDOFF

## Goal
四条**联合基线**（faithful adaptation）+ 完全体证据链。
**第 1 步已冻结**；下一步进 TIE。

## Done
- **拓扑 edge 回归**已修复（v04）；**节点本体**已对齐 v3.1（**v04.1**）
- **第 1 步正式冻结**，三个关闭证据齐全：
  1. **provenance-based gate**（`scripts/scheduler_projection_gate_v041.py`）
     **640/640 PASS，0 违规** —— 语义检查只读 `source_step_ids`/`parent_step_ids`/`retry_of`/raw 词表，
     `sequence_index` 仅作 projection-index 连续性检查
  2. **169 对嵌套时间窗 VERIFIED**（原始 trace）：169/169 包含、169/169 `parent_runtime == interval`
  3. **648 在执行 run_id 层闭环**：640 formal + 8 pilot = 648，交集 0
- v04.1：8,295 → **8,126 节点**（合并 169 嵌套）、`resource_applicable` 成为**强制契约**、
  `sequence_index` 重编号、契约名 `scheduler_projection_of_verified_serial_control_flow_v3_1`
- 四条基线架构骨架已搭好；SRTF+Aging 退役；F0 打包器 attr_feature bug 已修

## Verified
- v04.1 gate 640/640；嵌套包含 169/169；执行层 648 交集 0；现有臂回归正常

## Open（GPT 复核指出）
- **P0 未修**：**合并嵌套调用的资源归属** —— 169 个 nested 全是 GPU、169 个父全是 CPU；
  删 nested 保 CPU 父 → **真实 GPU 工作从 GPU 竞争消失**。
  需 v3.1 的 composite signature（父节点携带 `nested_model_class` + simulator 收内层 GPU 成本）。
  **第 1 步之后的第一优先项。**
- **TIE P0**：`estimate()` 丢 mean/CVaR → 实际跑 `p50 + β·p90`；10/10 是假阳性。**7 条修复清单见 gate**
- **LLMSched**：`H(X) ≠ I(X;Y)`；真实 duration 未进 posterior。fidelity FAIL
- **Pythia**：`baseline` 不是 role alphabet。应改名 `workflow-family progress prior` 或重构为 role-PFA
- **Latency-Aware**：key 不是 Eq.12（`-boundaries_removed` 是发明的 tie-break）；**用了真值 `compute_ms` = 真值泄漏**
- **49.3% fusion 素材需在 v04.1 上重算**
- `round_robin` 实跑 myopic（预先存在，未修）
- 预取测试用 3-GPU 合成数据（真实 2 GPU），需补代表性测试
- `SchedulerTopologyContractGate` + `BaselineFidelityManifest` 未冻结
- 三条共享模块建议（`verify_raw_run` → `canonicalize_v31` → `project_scheduler_chain`）未做

## Active
无长时任务在跑。

## Next
1. **修合并嵌套调用的资源归属**（第 1 步的 P0 尾巴）
2. **第 2 步**：TIE → LLMSched → Pythia → Latency-Aware，每条加 **E2E sentinel mutation test**
3. 冻结两个 gate
4. 重跑 30 集 smoke（只看机制）
5. 在 v04.1 上重算 fusion 素材与调度机会
