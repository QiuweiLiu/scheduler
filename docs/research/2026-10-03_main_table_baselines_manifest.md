# 主表基线适配 manifest（Parrot / QLM / Hermes / Torpor）— **已冻结**

**状态**：已按网页版独立审核（generation 10 会话，2026-10-04）的裁定冻结；
审核回复见 `docs/research/2026-10-04_main_table_baselines_review_gpt.md`。
**当前稿曾被判"不批准冻结"**：Parrot 的 critical-path 与 Torpor 的加权分是原文没有的调度逻辑（strawman 风险）；
QLM 需改为 SLO-oriented stochastic program；Hermes 只需小修（Gittins 原式 + 在线 prewarm trigger）。

## 已冻结的前提

- **主表**：Parrot + QLM + LLMSched（已实现）+ Hermes + Torpor + Ours；Myopic 下界。
- **substrate**：单卡 node-level DES；原子 node 不可抢占；引擎已有 F1/F2/F3/F4、mp 部署、请求级抢占、批处理。
- **信息契约**：可见 = 已揭示前缀（completed/ready 及其因果边）、train-only 估计、train split（480 模板）
  训练的模型、引擎状态、已完成节点的 `observed_intrinsic_ms`；禁止 = 未执行 suffix 真值、真实
  `runtime_ms`、H5/future artifacts、任何经 scheduler_state / dispatchable_gpu_indices 的真值泄漏。
- **工作负载**：v04 因果模板为串行链（480 train / 160 validation；每 job 任一时刻 1 个 ready 节点）。
- **顺序（审核追加 gate）**：implementation → unit/fidelity tests → leakage gate →
  mechanism-activation audit → independent code review → formal run。
  正式跑前须产出 **baseline activation report**（各基线计数器，见文末）；**不设人为激活率门槛**，
  只要求"机制真实进入决策路径"并原样报告真实激活率。

---

## 1) `Parrot-adapted (App-FIFO projection)`（policy id `parrot_appfifo`）

**保留**：application-centric topological / **App-FIFO** 调度——同一 service class 内按
**application arrival order**（最老的 application 先），并尽量连续推进同一 application 的 ready node
（application-level depth-first）。readiness 本身保证拓扑合法。
**禁止**：canonical remaining-work / critical-path / 任何"剩余大者优先"的 LJF 变体。
**省略**：Semantic Variable API、prefix/context sharing、batching、并行 task-group scheduling。

**冻结规则**：
```
priority = (service_priority, application_arrival_ms, node_ready_ms, job_index, node_id)
```
即同 service class 内 oldest-application-first；一个 application 的连续 stage 获得 continuation 偏好
（其 ready node 的 arrival key 仍是该 application 的 arrival，故持续优先）。

**表脚注**：Semantic Variable API, prefix/context sharing, batching, and parallel task-group scheduling
are unavailable; the serial-chain projection retains Parrot's application-centric topological/App-FIFO
scheduling.

**测试**：App-FIFO 与 request-FIFO 至少一次 action flip；application 连续 stage 的 continuation 偏好；
改不可见 suffix 的 runtime/role 不改变决定；serial chain 下 `task_group_size == 1` 显式统计并报告。

---

## 2) `QLM-queue-adapted`（policy id `qlm_queue`）

**保留**：uncertainty-aware **stochastic queue reordering** + model-transition（load/swap）成本进入
completion time；目标 = **SLO attainment**（chance 约束 + expected SLO penalty）。
**省略**：token-level batching、request eviction、KV state swapping、autoscaling。
**禁止**：在线 same-key Bayesian posterior 更新（原文没有）；把目标换成平均 JCT。

**冻结规则（SAA）**：
- duration 分布：train-only 经验分布，key = `(model_id, role)`；样本不足 fallback `model_id → global`；
  不细分 seq position，不把 service priority 放进分布。
- 场景：固定 **S=64 common-random-number scenarios**（固定 seed）。
- completion：`C_i = waiting_i + runtime_i + load/swap_i`（F3 load 直接进入）；
  **scenario 时钟从决策时刻起算**（deadline 是时间轴绝对 deadline，等待时间必须进入 C_i——
  review 2424f67 P1-2 修正）。
- 目标（lexicographic）：① 最大化满足 chance-SLO 的 request 数（`P(C_i ≤ d_i) > Δ`）；
  ② 最小化 expected SLO penalty `ΣE[(C_i − d_i)_+]`；③ 最小化 `ΣE[C_i]`。
- **Δ = 0.90**（adaptation hyperparameter，非论文原参数；appendix 做 {0.8, 0.9, 0.95} 敏感性）。
- chance 约束全不可满足时（无 autoscaling）：lexicographic fallback
  `min(#violations, ΣE[(C_i−d_i)_+], ΣE[C_i])`。
- 首槽选择：对每个候选作为首元素、其余按 EDD 续排，评估上述 lexicographic 目标（可复现的
  tractable 近似；实现时冻结并在 docstring 说明）。

**表脚注**：Retains QLM's uncertainty-aware stochastic queue reordering and model-transition cost;
token-level batching, request eviction, KV state swapping, and autoscaling are unavailable.

**测试**：**同 mean runtime、仅改 variance → 至少一个构造 case 的 queue solution 不同**
（否则退化为 deterministic）；固定 seed 确定性；无泄漏。

---

## 3) `Hermes-PDGraph-adapted`（policy id `hermes_gittins` + 在线 prewarm）

**保留**：conditional PDGraph + **论文原式 Gittins rank** + **在线 analytical prewarm trigger**。
**省略**：原系统多 backend 部署细节；backend→model/model-class 映射必须披露。

**冻结规则**：
- PDGraph：train-only、**按 workflow/application type 条件化**（`workflow_type_id`），保存 aligned
  历史 tuple（prefix demand、suffix demand、next role、next model）；已完成节点的**实际观测需求**
  （`observed_intrinsic_ms` 之和）在 **Pearson ρ > 0.5** 时筛选历史 tuple（±25%，少于 3 条回退全量）；
  backend/model 取条件**联合** (next_role, next_model) 分布（非全局 role-majority）——review 2424f67 P1-4 修正。
- Gittins 原式（**G 越小优先级越高**）：`G(D,a) = inf_{Δ>0} E[min(X_D − a, Δ) | X_D > a] / P(X_D − a ≤ Δ | X_D > a)`；
  主实现为**精确经验 Gittins**（在 distinct support 值上取 inf，前缀和 O(n)）。
  **更正**：论文的 "10 buckets" 属于 cross-unit demand correlation/Pearson 分析，**不是** Δ 枚举
  （上一轮冻结的错误绑定）；旧等质量 10 点网格仅作 appendix 敏感性——review 2424f67 P1-3 修正。
- prewarm（**在线，非仅 idle**）：`p_e = p_s · P(t_c > t_s + t_p)`；`p_s < K=0.5` 不预热；
  `t_c` 必须来自**调度器可见预测**（派发时冻结的 `predicted_work_ms`），**不得读真实 finish/runtime**
  （review 2424f67 P1-1 修正）；F4 未覆盖 → serial fallback 并计数。
- 不把 deadline 项混进 Gittins（Hermes-DDL/worst-case LSTF 仅未来 appendix 备选）。

**表脚注**：PDGraph → online refinement → Gittins → analytical prewarm 保留；backend 映射为
model/model-class；多 backend 部署细节省略。

**测试**：`gittins != mean-SPT` 至少一个构造 state；prefix 观测后 conditional PDGraph 变化；
`p_s < 0.5` 不 prewarm；`p_s ≥ 0.5` 按 analytical trigger prewarm；prewarm 发生在 active-node
执行窗口内（非仅 idle）；F4 uncovered fallback 可计数。

---

## 4) `Torpor-lifecycle-projection`（policy id `torpor_lifecycle`）

**保留**：late binding、residency-aware model swapping、interference-aware loading、
swap-cost-aware eviction。
**省略（必须披露）**：RRC tail-SLO queueing（workflow deadline 无法干净映射为 function P98 SLO，
不伪造）；NVLink GPU→GPU swap（单卡）；原版 heavy/light 二元阈值（推广为连续 burden，见下）。
**禁止**：`runtime + load + λ·interference` 加权分（原文没有）；runtime-SJF 排序。

**冻结规则**：
- job selection：硬 service priority 第一键，同 priority 内 **FCFS**（无 runtime 项）。
- GPU/placement（lexicographic）：
  `resident+available ≻ covered low-interference load ≻ covered higher-interference load ≻ uncovered serial fallback`；
  coverage 必须用**canonical `(infer_model, infer_shape, load_model)` 三元组**判定（复用 substrate 的
  `prefetch_interference_extra_ms`），**不得**用 model-level 集合或跨格 max——review 2424f67 P1-5 修正。
  （当前 admission 拓扑下冷加载只进空闲设备 → 忙卡干扰分支结构上不可达；已在审计登记。）
- eviction：从 `swap_burden(m) = load_time(m) + interference_cost(m)` 最低开始
  （model-level conservative aggregate 仅作 eviction burden estimator）；
  同成本 → **确定性 model_id tie-break**（引擎无驻留时间戳，LRU 不可表示，不假装实现）。

**表脚注**：Retains Torpor's late binding, residency-aware model swapping, interference-aware loading,
and swap-cost-aware eviction. RRC tail-SLO queueing and NVLink GPU-to-GPU swapping are unavailable in
our substrate.

**测试**：resident 优先/冷加载路径选择激活；eviction 按 burden 选择；未覆盖干扰回退可计数；无泄漏。

---

## 激活审计计数器（formal run 前产出）

（v2，review 2424f67 P1-6 修正后）
- Parrot：`app_fifo_action_flip_count`（真实 top-1 对比）
- QLM：`stochastic_reorder_count`、`load_present_count`、`swap_cost_affected_count`
  （**反事实**：load 归零后选择改变）
- Hermes：`gittins_vs_mean_flip`（同一 tie-break、仅主键不同）、`pdgraph_conditional_used`、
  `pdgraph_refined`、`prewarm_trigger`
- Torpor：`resident_hit`、`cold_swap`、`covered_load_choice`、`uncovered_load_choice`、
  `interference_aware_choice`、eviction 事件
- 审计分两相：A = 真实 dev 子集（无 profile；排序机制）；B = **声明式合成冒烟**（真实模型/形状/身份 +
  真实 extension artifacts；profile 依赖机制）。真实 dev 集缺 shape/identity → 全 profile 审计在契约补齐前
  标 **PARTIAL**。

要求：机制真实进入决策路径；真实激活率原样报告，不调 workload 凑比例。
