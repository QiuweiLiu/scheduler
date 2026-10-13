# P0 动作合法性修复(顾问准入审查第 1 优先项)

日期:2026-10-13。依据:`docs/research/2026-10-13_milp_v0_admission_ruling.md`(P0:GPU 放置合法性)。

## 1. 修复内容

**① 合法性掩码(advisor 公式 `d_ig ≤ A_ig^0`)**
- `MILPObservation` 新增 `legal_first`(当前合法的 (node, gpu) 对集合;None=离线默认无掩码;空集=当前无合法动作→必须回退)。
- `solve_v0a`:立即派发变量从"按节点"改为**逐 (node, gpu) 对**的 `d_ig`;新增约束 `d_ig ≤ A_ig^0`(掩码外强制 0)、`d_ig ≤ x_ig`(派发必须落在自身分配上)、`Σ d_ig ≥ 1`(至少一个合法立即派发,除非显式回退)。
- 首动作提取改为**只从 `d_ig = 1` 的对中选**(掩码+分配双保证);提取器保留掩码安全检查作兜底。
- 掩码只约束**首动作**;未来 H5 规划不受影响(顾问要求)。

**② 全部已观测 Ready 节点参与规划**
- 在线 `milp_v0a` 与批次脚本的 ready 列表改为直接取**全部就绪节点**(此前从过滤后的候选池构造,会漏掉等待中的节点、缩小 H5 目标覆盖)。

**③ 批次验证的掩码来源**
- 先跑 DES 探针取候选池(同刻池),再以该池作为掩码求解;探针首决策时刻 ≠ 快照时刻时掩码为空(该时刻无合法派发)。

## 2. 测试证据(GATE PASS)

- **T11 动作掩码 4/4**:掩码只留劣后节点→首动作=该节点;掩码排除自然放置→首动作在掩码内;空掩码→显式回退(infeasible);无掩码→默认行为不变。
- T7 穷举 36/36 一致(无掩码行为逐位不变);dispatch-now 3/3;T6b/T8 等全过;GATE PASS。
- 证据:`artifacts/tests_evidence.json`。

## 3. 批次验证(183 状态,掩码版)

- **首动作非法数 = 0**(验收标准:被采用首动作 100% 在真实 DES 候选池内)✓
- 执行 50(全部在池内)/回退 133;求解 P50 4.2ms / P95 455ms(掩码剪枝后 P50 从 14.4ms 降至 4.2ms)。
- 变化类型(vs PDRS 基线):same 30 / ordering 20 / **placement 0**(8 个非法放置全部消失)。
- **INFEASIBLE 拆分**(顾问放行门槛第 4 条):`当前无合法动作` 46 状态 vs `有合法动作但 v0a 无驱逐计划不可行` 87 状态。
- 转变:pre-mask 的 11 个非法首动作中,4 个改为合法动作执行、7 个正确回退;另有 3 个此前"同动作"状态因探针时刻校验更严(快照时刻实际无可派发)而回退。

## 4. 11 例非法首动作逐一诊断(顾问要求)

逐例输出(真实 GPU 状态 / 是否在 free 集 / can_dispatch / 准入结果 / 候选池 / MILP 预测可用时间):
**10/11 = 预测可用时间误差**——MILP 观测的可用时间 ≈ now(认为卡空闲),真实 `busy_until` 仍在未来约 200–1600ms(运行中任务的剩余工作量被低估:预测工作量 × 慢化 − 已执行 ≤ 0,但任务实际仍在跑)。
**1/11 = 平局重选**(重解后同一节点换到空闲卡,已合法)。
**0 例**来自显存准入 / 候选池筛选 / 派发规则。
- 证据:`artifacts/illegal_action_diagnosis.json`;pre-mask 参照:`artifacts/batch_validation_v2_premask.json`。
- 结论:非法性来源是**观测可用时间的乐观偏差**(对运行中任务),不是求解器约束错误;掩码已从机制上消除非法执行,该偏差的建模改进(如运行中任务的保守剩余上界)留待后续评估。

## 5. 复现

```sh
PYTHONPATH=src python scripts/milp_v0_tests.py            # GATE PASS(T11 4/4)
PYTHONPATH=src python scripts/milp_v0_batch.py --episodes 60 --cap 4 --batch-a 30
PYTHONPATH=src python scripts/milp_v0_illegal_diag.py     # 11 例根因
```
