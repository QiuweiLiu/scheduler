# HANDOFF

## 2026-09-26 (5) Pythia S_unblock 修正（GPT P0）（未 commit）

GPT 复核（`docs/research/2026-09-26_pythia_algorithm3_review.md`）判定：无泄漏，但 **S_unblock 距离量错 = P0，不能 freeze**。

### 已修
- **P0**：`expected_distance_to_role()`（current→future 首次命中距离）取代 `expected_remaining_steps(future)`；贡献 = `idle_weight / E[D(current,a)]`；方向 gate 钉死。
- **P1**：raw sum（非均值）；按 `role_model_dist` 的 P(model|role) 加权（planner 纯度 0.53、answer 0.50）；GPU-lane 过滤；改名 `model_has_zero_visible_demand` + `visible_model_demand()`；常量统一为预注册固定值；清单矛盾文案清除。
- sentinel 24→32；gate 155/155；全量 391（仅预存/环境）；fidelity manifest PASS。
- Non-degeneracy：aging 改变结果；both≠completion-only；S_unblock 非零率 80.5%（单加未改 makespan，已记录）。

### Active
待 commit/push；随后请 GPT 终审 Pythia freeze。

### Next
1. commit + push
2. GPT 终审 Pythia（预期 ACCEPT/FREEZE）
3. Agentix 披露收尾

---

# HANDOFF

## 2026-09-26 (4) Pythia Algorithm 3 补齐 + Agentix 去伪声明（未 commit）

### 本轮已实现
- **Pythia 补齐**：新增 `pythia_methods.py`（`omega1*S_completion + omega2*S_unblock` + 无量纲 aging）；
  `pythia_profiler` 增 `role_model`（train-only）与 `reachable_future_roles()`，schema → `pythia-role-pfa-v3`；
  模拟器 Pythia 分支改为消费全部三项 + 当前 model-demand 代理。**取代 completion-only 版本为正式主基线。**
- **非退化实测**：aging 与 S_unblock 都改变结果；S_unblock 非零率 **90.5%**（3 confirm 集）。
- **Agentix**：删除 `critical_path_service_ms` 的"ATLAS 等价"错误声明，改标 ATLAS VARIANT。
- **未补**（GPT 判定）：Agentix K 队列/抢占/防饿死；LLMSched sample_tasks。

### 验证
聚焦 gate 147/147；全量 383 测试（仅预存 round_robin + 8 环境 error）；fidelity manifest PASS。

### Active
待 commit/push。之后按 GPT 建议：按新 head **重新 freeze Pythia**，并（可选）把本轮实现交 GPT 复核。

### Next
1. commit + push（含 `docs/research/2026-09-26_{baseline_strengthening_adjudication,agentix_baseline_review}.md`）
2. 重新 freeze Pythia（head/常量/清单）
3. Agentix 主臂固定 PLAS；补披露（MLQ/时间片/防饿死/抢占 unavailable）

---

# HANDOFF

## 2026-09-26 (3) 复核修复 + Latency 契约更正 + 新增 Agentix（未 commit）

详见 `DECISIONS.md` 同名三节。要点：

### 已实现并验证
- 复核 P0/P1 修复：Latency resolved-graph view、TIE cold-load、Pythia aging omission、LLMSched interval、runner `--formal`。
- Latency 契约由 `logical_graph_assumed_known` **更正为 `runtime_resolved_logical_graph`**（论文只观察 resolved `L_t`）。
- 实证：本 workload 是决策驱动 ReAct 流程，**Latency 的 fusion 无合法机会**（0 合法边），需论文披露。
- **新增 Agentix-adapted**（NSDI 2026）：`agentix_methods.py`（PLAS/ATLAS）+ `agentix` 策略；gate 10/10；真实 1 集跑通。

### 验证
聚焦 gate 136/136；全量 372 测试（仅预存 round_robin + 8 环境 error）；fidelity manifest PASS。

### Active
已提交并推送 **`122e184`**（`c06a170..122e184`，含 3 份 GPT 复核文档、Latency 契约更正、Agentix-adapted）。
**Agentix 的 GPT 审核暂受阻**：ChatGPT 网页该标签页反复「无法加载你的账户」，项目会话页尤其加载失败；旧会话页可用但已达长度上限。待浏览器恢复后重发 brief。

### Next
1. commit + push
2. GPT 审核 Agentix 实现与 Latency 契约更正
3. （用户定）Agentix 进五臂 / 是否补 Maestro；Latency 降级披露措辞

---

# HANDOFF

## 2026-09-26 (2) 按 GPT 复核执行 P0/P1 修复（工作树未 commit）

**依据**：`docs/research/2026-09-26_freeze_breaking_fixes_review.md`。详见 `DECISIONS.md` 同名小节。

### 已修
- **P0 Latency resolved-graph view**：新增 `resolved_graph_view`；fusion 只允许落在 resolved 窗口内；near-ready 只取 running 的唯一后继；
  删 `FusedChain.summed_runtime_ms`；新增 identical-prefix/different-future sentinel。**后果：v04.1 上 fusion 不再触发**（改前 3 集约 39 次），
  Latency 变弱（starts 172→211，1 集 Δ +4087→+5850ms）——这是 GPT 接受的保守解，**需用户确认**。
- **P1 TIE**：nonresident 改用 `E[load | load>0]`（cold-load mean）；[0,10] sentinel。gate 24/24。
- **P1 Pythia**：删错误注释，aging 明确列为 omission。gate 13/13。
- **P1 LLMSched**：interval 全 stage 条件在 known-present 集合。gate 38/38。
- **P1/P2 runner**：`--formal` 硬断言 fidelity/topology gate、split 700/300、git HEAD 非 "unknown"。

### 验证
聚焦 gate 138/138；全量 362 测试（仅预存 round_robin + 8 环境 error）；fidelity manifest PASS；`--formal --smoke 1` 通过。

### Open / 待用户决定
- **Latency fusion 权衡**：接受「弱但不泄漏」还是改成「仅有唯一后继时不泄漏地保留 fusion」（行为不变，但不满足 sentinel）？
- 本轮**未 commit / 未 push**；复核要求按新 head 重走一次 freeze 声明。

### Next
1. （用户）确认 fusion 取舍 → commit/push
2. 按新 head 重跑 GPT freeze 声明
3. 30 集 freeze-qualified smoke → 300×5

---

# HANDOFF

## 2026-09-26 网页版 GPT 复核结论：NEEDS CHANGE（未据复核改代码）

**复核文档**：`docs/research/2026-09-26_freeze_breaking_fixes_review.md`（GPT-5.6 Sol High，基于公开仓库 diff）。

**复核接受**：LLMSched whole-job + conditional 数学、Pythia 步数修复、Latency load head 与 prefetch 时序（对当前 ready）、runner survivor-mean P0。

**复核推翻 / 新增 blocker**（正式测量前最小项）：
1. **P0 — Latency-Aware 仍泄漏**：论文假设 **resolved logical window**，不是完整 realized 图；
   `latency_aware_fusion.py` / `latency_aware_lifecycle.py` 读完整 `job.template` = 真值结构泄漏。
   → 我在 2026-09-25 §3 记的 `logical_graph_assumed_known` **信息契约结论已作废**。
2. **P1 — TIE**：nonresident 的 load 应用 **conditional cold-load mean**（`mean(load_ms>0)`），不是含 0 的 unconditional mean。
3. **P1 — Pythia**：waiting-time aging 未实现；注释错误。
4. **P1 — LLMSched**：interval 未把 known-present 条件传播到其它 stage（现为 conservative envelope）。
5. **P1 — runner**：formal 模式需 hard-assert freeze/gate/hash，禁止 `git_head == "unknown"`。P2：删 `FusedChain.summed_runtime_ms`。

**Gate**：30 集可作**工程诊断**；**freeze-qualified smoke 与 300×5 均不批准**；`c06a170` re-pin ≠ freeze 签字。

### Active
复核文档已落盘；**未**据复核改动任何代码。等待用户决定是否执行上述 5 项最小修复。

### Next
1. （用户决定）执行 P0+P1 最小修复 → 重新生成 manifest/新 SHA → 重新 freeze 声明 → 30 集 freeze smoke → 300×5
2. 复核文档与本次控制面更新是否提交/推送（待批准）

---

# HANDOFF

## 2026-09-25 (2) 审计 P0 修复后（工作树未 commit；上一提交 `2fbb44a`）

### 本轮已修（全部 freeze-breaking；权威记录见 `DECISIONS.md` 同名小节）
- **LLMSched EXPLOIT**：新增 `expected_job_remaining_ms` + `conditional_state_probs`，覆盖**全部**未完成 stage 并条件在
  `current_stage != ABSENT`；`job_duration_interval_ms(..., known_present=)` 去掉 ready stage 的虚假 0；删除死代码 load。
  Eq.6 不动。**gate 38/38**（含新 `L8WholeJobRemainingAndKnownPresent`）。
- **Pythia**：`V(role)` 改为**未来 role 转移步数**，`S_completion = 1/(1+V)`，consumer 不再加毫秒；schema → `pythia-role-pfa-v2`。**gate 13/13**。
- **Latency-Aware**：Eq(5) prefetch 移到 **dispatch 之后、只在空闲设备**执行（旧时序 250 ms → 新 150 ms 可区分）；
  新增 `GRAPH_VISIBILITY_CONTRACT`。**gate 45/45**（含新 `test_prefetch_does_not_delay_ready_work`）。
- **清理**：`latency_aware_predictor.py` 重复定义去重；`four_joint_baselines.py` bootstrap 文案改 **episode-cluster**。
- 四份 freeze 清单新增 `freeze_breaking_fixes`；`baseline_fidelity_manifest_v1.json` 重新生成 **PASS**。

### 仍未修
- **TIE load adaptation 的「窄条件统计」问题**：审计原文在 LLMSched cache 处被截断、未保存，细节未知 → **未改（不臆测）**，等补齐原文。
- 跨基线要点仍有效：主表评估**完整系统**，需 same-interface 2×2；**F0 future-chain identity 仍来自 J3**（论文不能藏）。
- 环境性测试：8 个 error（缺 torch / py3.9 `write_text(newline=)`）、1 个 failure（预存 `round_robin` myopic）。

### Active
已提交并推送 **`54a2d25`**（`2fbb44a..54a2d25 main`）；冻结清单 `head` 与 fidelity manifest 已重新 pin 到该 commit。

### Next
1. 取得被截断的 TIE 审计原文后处理 TIE 的「窄条件统计」项
2. 在 v04.1 重跑 30 集 smoke，再进 300×5 正式测量
3. 如需，按新 head 重新走一次 GPT freeze 声明

---

## 2026-09-25 全面审查后的状态（HEAD 3342dac）

### 已修并验证

**P0-01 预取在过去开始**（workload_v02_simulator.py:4572,4605）
start = max(0.0, cursor)，cursor 只由 gpu.busy_until 播种 → 设备空闲时预取排到**已过去的时间**。
真实 workload 前 3 集：12 次预取中 **10 次早于发出事件、5 次已「完成」、最大回溯 ~373 s**。
→ cursor 播种与使用点都改为以 
ow 为下界。
**复测（3 个 confirm 集）：9 次预取，早于发出 0，已完成为 0，最大回溯 0.000 s。**

**P0-02 runner 从未读过冻结分区**（our_joint_baselines.py）
原来读 1000 集验证文件、--episodes 300 取**前 300**；按 seed-20260914 shuffle 那只有
**212 dev + 88 confirm**。→ 改为按 data/manifests/validation_split_dev700_confirm300.json
的 **id** 选取，默认 --split confirm，--split development 仅供调参，并断言选择等于 manifest 集合。
**验证：confirm=300 / dev=700；文件前 300 确实只含 88 个 confirm（复现了 bug）。**

**Runner fail-closed（GPT 审计项）**

un_arm 现在对每个 episode 断言 ailed_jobs==0、completed_jobs==jobs、metric 有限；
projection SHA **断言**等于冻结值（不再只是记录）；记录 split manifest 与 episode 文件 hash；
git_head 改从公开镜像读取（工作树不是 git 仓库，原来记成空字符串）。

验证：**345 测试 / 5 个预存失败**（无回归）；F0 参考在 confirm 集上仍 16/16 完成。

### 仍未修（GPT 全面审查的其余 P0）

| 基线 | 问题 |
|---|---|
| LLMSched | EXPLOIT 的 remaining **未条件化 X != ABSENT**（而 EXPLORE 与 current-service 都条件化了），且只算 BN descendants、漏掉与 X 无关仍属 job remaining 的 stage |
| Pythia | 算的是「duration-weighted remaining **毫秒**」，Algorithm 3 用的是 **regex 上 expected remaining distance（步数）** |
| Latency-Aware | Constructor / prefetch **读 realized template 未执行后缀** = 真值结构泄漏 |
| Latency-Aware | load 头是 request-conditioned，但论文是 T_load(d,g)（deployment/device） |
| Latency-Aware | prefetch 在 ready dispatch 前跑，**可能延迟 ready work** |
| TIE | load adaptation 的窄条件统计问题（GPT 后半段未读到细节） |

另两条跨基线要点（改进论文，非改代码）：
- 主表评估**完整系统**，不能单独证明「你的 scheduler 更强」；需 same-interface 2×2。
- F0 的 future-chain identity 仍来自 J3，**这个 caveat 论文不能藏**。

### 通道
ChatGPT 换了 DOM（输入框 = ProseMirror contenteditable，turn 选择器失效）→ 已重写
.scratch/cdp_send3.py（CDP Input.insertText + Enter，并校验 composer 内容）与
.scratch/cdp_bottom_dump.py（滚动到底 + main.innerText）。

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

### 三个真 bug（L1 抓出，都是「静默返回貌似合理的错值」）
1. 观测到的**后代**被「只走祖先」的剪枝丢掉 → `P(B=t|A=t,C=f)` 返回 0.9 而非 0.75
2. 因子相乘时用**错误的操作数**去翻译 key 索引
3. 观测到的**子节点**保留了自身坐标轴 → 被当作隐变量求和掉而非钉住

### 两个阻塞点已修复（`105ba07`）
1. **速度**：宽因子是 6 变量 = 7^6 = 117649 项，纯 Python dict 乘不开。
   改为 **numpy 密集数组**（各因子 reshape 到 union 轴序后一次广播相乘）。
   另一独立开销：cheapest-first 用 `len(array)`（只是第一轴）当成本，且每步重扫全部因子。
   改为**按 canonical order 消元**——builder 已用 induced-width guard 认证该顺序。
   **实测 12.6 ms/次，159× 加速**；L1 五项手工 posterior 仍精确一致。
2. **Range scope**：`Y` 曾取「stage 的所有可达后代」，在 55-stage 词表上达 30 个、乘积 1e72。
   **这是我的误读，不是论文问题**：`Y` 是**该工作流剩余的工作**，词表比任一工作流大得多。
   新增 `workflow_future_stages()` 返回本工作流**未完成的后续 stage**。
   结构可见是合法的（预测动作链正是本 baseline 的前提）；**绝不读取未执行节点的时长**。
   **顺带定下排序**：同一决策内所有候选共享同一剩余集合 → `∏ Range(Y)` 是**正常数**，
   不参与排序 → **EXPLORE 完全由互信息排序**，正是 **L2** 要测的。
   省略 `future_stages` 的调用者拿到**有上限的默认值**，避免调用点静默丢失 scope。

### 四条联合基线 —— 实现层全部冻结，两个总门已建并 PASS（HEAD e6dc902）

| 基线 | 冻结 HEAD | 状态 |
|---|---|---|
| LLMSched | 618b26b | FROZEN |
| Pythia | 26484a4e… | FROZEN |
| TIE | c6cbec9… | FROZEN |
| Latency-Aware | c6cbec9… | FROZEN |

**SchedulerTopologyContractGate v1**（scripts/scheduler_topology_contract_gate.py）— **PASS**
重新推导而非信任旧产物：640 模板 / 8126 节点 / 169 嵌套合并 / 契约名
scheduler_projection_of_verified_serial_control_flow_v3_1 / 全节点 
esource_applicable /
640 个 chain_tail_is_answer / 串行因果链（无节点多前驱、无悬挂引用）。
按内容哈希钉住投影：**15c62dafd99701b3883a3fe4…**。并把四份支撑结论（provenance gate、
seriality gate、嵌套集合相等、嵌套资源守恒）一并收入同一产物。

**BaselineFidelityManifest v1**（scripts/baseline_fidelity_manifest.py）— **PASS**
**检查**而非罗列：每条基线的 freeze manifest 必须存在且 status=APPROVED、arm 正确；
gate 文件必须存在；gate 必须**实际跑绿**。缺失或失败即整体 FAIL，不会报陈旧批准。
四条全部 reeze=APPROVED + gate=green。
产物明确写清含义边界：**fidelity 是可比性的前提，不是性能主张**。

**下一步**：正式跑四条联合基线实验。此前**不再改**任何 baseline semantics。

### 仍未做（LLMSched 之外）
- 上面「Pythia / Latency-Aware / 两个 gate / fusion 重算」各项

## Next
1. **LLM-4 再优化**：`posterior_joint` 限制到相关祖先窗口 + evidence 签名缓存；目标 <1 ms/查询
2. **切消费端**到 v2，去掉 `NOT YET MIGRATED` 标记；E2E 走通
3. **补 L2–L7** gate，尤其 **L2 同 entropy 不同 MI** 与 **L7 E2E mutation**
4. Pythia role-PFA 重构；Latency-Aware 换成自训练 request-conditioned predictor
5. 冻结 `SchedulerTopologyContractGate` + `BaselineFidelityManifest`
6. 重跑 30 集 smoke（只看机制）；在 v04.1 上重算 fusion 素材与调度机会

