# RESULT — EXP-20261004_main_table_baselines_activation_v1（主表基线 activation audit）

**状态**：完成（2026-10-04）；本地、无 GPU；**dev 子集 30 集（development，confirm300 未触碰）**。

## 目的与判据

按冻结顺序（manifest 审核后）：implementation → unit/fidelity tests → **leakage gate** →
**mechanism-activation audit** → 独立 code review → 正式对比运行。本实验回答：
**四条新基线的独门机制是否真实进入决策路径**；激活率原样报告，**不设人为比例门槛**。

## 结果（30 个 dev episode，causal_v3 视图，v041 模板 + v03 episodes）

| 基线 | 决策数 | 激活计数器（真实值） | 备注 |
|---|---|---|---|
| `parrot_appfifo` | 3984 | `app_fifo_action_flip_count` = **1067（26.8%）** | App-FIFO 与 request-FCFS 在串行链上确实分叉 |
| `qlm_queue` | 3984 | `stochastic_reorder_count` = **1700（42.7%）**；`swap_cost_affected_count` = **1636（41.1%）** | SAA 排序与队列序分叉;load 项实际参与 |
| `hermes_gittins` | 3988 | `pdgraph_conditional_used` = **3508（88.0%）**；`pdgraph_conditional_hit` = 3508；`gittins_vs_mean_flip` = **1（0.03%）** | 条件 PDGraph 活跃;Gittins 排序与 mean 排序几乎重合（见发现 1） |
| `torpor_lifecycle` | 3992 | `resident_hit` = **3112（77.9%）**；`cold_swap` = 880（22.0%）；`uncovered_load_choice` = 880；`eviction_events` = 730 | 驻留优先与冷加载选择均真实发生 |

**Leakage gate**：新增测试 `test_main_table_baselines_ignore_the_invisible_suffix`（四臂首个决策在
不可见 suffix（runtime/role/model）突变下不变）+ 各臂既有无泄漏单测 → **通过**（simulator 53/53）。

## 发现（必须随报告披露）

1. **Gittins 离散化敏感性**：按审核冻结的 "10-bucket support boundaries 枚举"，在真实稠密条件分布上
   `G` 与均值几乎相等（例：state (0,'') n=480，G=mean=53280.9），因此 `gittins_vs_mean_flip≈0`。
   精确 inf 更低（state (5,'planner')：精确 14351.5 = 0.6×mean，而 10-bucket 枚举给 23930=mean）。
   → Hermes arm 在本 workload 上数值上退化为 mean 排序（机制按冻结公式实现，构造用例中
   Gittins≠mean 已由 `test_hermes_gittins_prefers_the_lower_index_application` 证明）。
   **处置**：保持冻结实现不变；把该敏感性作为待审事项（可选：exact inf 或更多 bucket 的敏感性）。
2. **Profile 依赖机制未在本审计激活**（如实登记）：Hermes 在线 prewarm 与 Torpor covered-load 排序
   需要 workload 的 `workload_shape` / GPU identity 契约（当前模板/集缺失 → fail-closed）。
   其激活证据目前由逐机制 fidelity 测试承担（在线 prewarm+干扰计费、covered cell 匹配）。
3. **nested_gpu_no_capacity 失败**：30 集中 5–8 集有 nested GPU 节点失败；**fcfs/myopic 同样失败
   （6/30、5/30）→ 与基线无关的预存工作负载行为**（无 extension config 的运行条件）。

## 附带的 substrate 修复（审计过程中发现）

- **loader fail-closed 洞**：v041 的 `topology_contract` 为
  `scheduler_projection_of_verified_serial_control_flow_v3_1`（带前缀），而旧检查是精确匹配
  `verified_serial_control_flow_v3_1` → **v041 以 legacy 视图加载不会报错，静默产生无边（全并行）图**。
  已改为 marker 包含匹配 + 回归测试 `test_causal_contract_marker_blocks_legacy_load`。
  影响面：正式五臂（`four_joint_baselines.py`）显式传 `causal_v3` ✓ 不受影响；
  任何以默认视图加载 causal 投影的脚本现在会**报错而不是静默错跑**。

## 交付物

- `artifacts/activation_report_v1.json`（逐臂计数器、比率、fail-closed 原因、失败集统计）
- 脚本：`scripts/baseline_activation_audit.py`（`--topology-view` 显式，默认 causal_v3）

## v2 重跑（2026-10-04 晚，review 2424f67 P1-6 修正后）

**裁定**：本审计在真实 dev 集上仍标 **PARTIAL / NOT PASSED**（真实 workload 缺 shape/identity，
profile 依赖机制无法在全量集上配置；见下）。计数器已按 review 修正为"反事实"口径。

### Phase A（真实 dev 30 集，causal_v3，无 profile；`activation_report_v2.json`）

| 基线 | 修正后计数器 |
|---|---|
| Parrot | `app_fifo_action_flip_count` = 1067 / 3984（26.8%） |
| QLM | `stochastic_reorder_count` = 1683（42.2%）；`load_present_count` = 1587；**`swap_cost_affected_count` = 256（load 归零后选择改变；旧口径 1636 是"存在冷候选"的自欺计数）** |
| Hermes | `gittins_vs_mean_flip` = **7**（同一 tie-break、仅主键不同）；`pdgraph_conditional_used` = 3511（88%）；**`pdgraph_refined` = 493（12.3%，观测精化真实生效）** |
| Torpor | `resident_hit` = 3112（77.9%）；`cold_swap` = 880；**`covered_load_choice` = 880（canonical 判定把空闲设备冷加载正确归为 covered；旧代码误标 uncovered）** |

### Phase B（声明式合成冒烟，真实模型/形状/身份 + 真实 extension artifacts）

- **Hermes 在线 prewarm 真实触发**：`prewarm_trigger` = 1；对 `Qwen2.5-VL-3B-Instruct` 的
  `prefetch_start` 发生在运行节点执行窗口内（t≈5329.7ms），干扰计费 **17.7ms = 实测 (4B, medium, 3B) 格**
  （与当时运行节点 `Qwen3-4B|medium` 的 canonical 三元组一致）。
- Torpor 空闲设备冷加载 → covered rank-1；`interference_aware_choice` 为 0 且**结构上不可达**：
  admission 拓扑只允许冷加载进空闲设备（忙卡要求候选模型已驻留），已登记。

### 仍需（契约补齐后）

- 真实 dev/正式集上带完整 profile 的审计（需 `workload_shape` / GPU identity 契约）；之后本 gate 才可转 PASS。
