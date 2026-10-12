# MILP_DES_CONTRACT.md — 约束↔DES 语义映射契约

- **实验**: `EXP-20261012_milp_v0`(第一阶段交付物之二)
- **目的**: 把 MILP_FORMULATION.md 的每个变量/约束映射到 `workload_v02_simulator.py` 的具体函数与状态,确保:v0 的数学假设与"串行配置下的真实 DES"逐条对齐;所有一致性都可以被独立测试(T1–T10)检验。
- **基准环境**: `serial_v0_extension.json`(由 v7 派生:共驻/预取干扰/抢占/批处理 `enabled=false`, `preemption_enabled=false`, `prefetch_overlap=false`;冒烟证据:16/16 完成、prefetch 0、preemptions 0、colocation disabled、驱逐 3)。

## 1. 状态获取(与实验 I 同一机制)

| MILP 输入 | DES 来源 | 说明 |
|---|---|---|
| 就绪节点 R | ready 堆(`release_ready`, `ready_nodes`) | `node_state=ready`;含 `(priority, ready_since, job_index, node_id)` |
| 卡可用时刻 F_g | `GPU.busy_until` / `active_tasks`(kind=node) | 在跑任务的 `finish_ms`;快照中为冻结参数 |
| 驻留 ρ_{g,m} | `GPU.resident`(MB 由模板 `resident_model_mb`) | 快照原样注入 |
| 在途加载 | `GPU.prefetch_pending` / `loading_model_ids()` | 冻结;v0 串行配置下通常为空 |
| 预测时长/加载/显存 p_i, ℓ_i, w_i | `estimate(node, train_stats)`(train-only 经验分位) | 与 DES 调度器读同一估计,不读真值 |
| H5 链 K_i | 冻结产物 `future_h5[0].steps`(J3:seed11) | 贪婪 top-1;`model_id/role/lane/resource.*_quantiles` |
| 容量 C_g | episode `gpu_topology_mb` | 32760 |

**状态注入**:复用实验 I `serialize_des_state` / `apply_injected_state`(规划世界);MILP 的输入只来自该快照与冻结产物。

## 2. 约束逐条映射

| MILP | 物理含义 | DES 对应 | 一致性检验 |
|---|---|---|---|
| C1 节点唯一 | 每节点执行一次 | `build_jobs`(node_state 全 pending)、`ready_nodes`(前驱 complete 才就绪)、链式 `predecessors` | T2:规划顺序 == 链顺序;虚拟节点从不出现在 DES 派发事件 |
| C1 因果 | 链顺序不可改 | 模板线性链(已验证 8126 节点 0 分叉) | T2 |
| C2 卡可用 | 不早于卡空闲 | `dispatchable_gpus`(`gpu.busy_until ≤ now` 或 `active_node is None`)、`gpu_can_dispatch_node` | T1/T10 |
| C2 互斥 | 同卡不重叠 | 串行配置下 `active_node` 唯一;`GPUActiveTask` 区间账本 | T10(时间线比对) |
| C3 冷加载 | 未驻留先加载 | 派发路径 `model_load` 事件;加载时长 `estimate.load_p50_ms`;`loading_model_ids()` | T4/T10 |
| C3 首用者一次加载 | 同卡同模型只加载一次 | 加载后 `gpu.resident[m]=allocation`;后续节点 `cache_hit` | T4 |
| C4 容量 | 驻留+工作区≤容量 | `plan_gpu_admission`(resident + pending load + allocation + workspace 投影);`workspace_incremental_mb` | T3/T10(显存轨迹) |
| C5 驱逐安全 | 不驱逐在用模型 | `plan_gpu_admission` 的 `protected = active_model_ids()`;可驱逐=非 active | T3 |
| C6 冻结初始态 | 运行中任务/在途加载不可改写 | 注入快照的 finish_heap/active_tasks 为参数;任何优化变量不参与其区间 | T6(隔离)+ T10 |
| 目标 J0 | 边界节点完成时间 | DES 事件 `node_finish` 时间 | T10(预测 vs 实际) |
| 回退 | 超时/不可行→基线 | 策略名 `pdrs_resident`(冻结主线) | T8 |

## 3. 执行契约(首动作)

- **首动作提取**:MILP 计划中最早开始的被选节点(并列按冻结 F0 键 `(priority, current+future, …)`)→ `(node_key, gpu_index)`;
- **执行**:实验 I `des_rollout` 策略的强制首动作机制(`policy_context["des_action"]`;无预取、驱逐模式=冻结基线默认);
- **合法动作保证**:执行前校验首动作 ∈ DES 可派发集合(`dispatchable_gpus` × 就绪池);不在则**视为 T1 失败**,回退基线并记录;
- **续跑**:其后全部决策委托 `pdrs_resident`(非预知;与实验 I 同纪律)。

## 4. 一致性测试的真值对齐(关键)

DES 的执行真值(模板 `runtime_ms`)与 MILP 的参数(预测 p50)**天然不同**。T10 的比对协议:

1. 用实验 I 的场景注入(`build_scenario_templates`)把 DES 的执行真值**替换为 MILP 的同一参数**(时长/加载/显存 = 预测值);
2. 在同一注入状态下运行串行 DES,导出事件时间线(加载/开始/完成/驱逐/显存峰值);
3. 与 MILP 的计划时间线逐项比对:顺序、开始/完成时刻、加载事件数、驱逐集合、容量占用;
4. 差异容忍:仅允许 ≤1e-6 的浮点差;任何结构性差异记为不一致项并报告(裁定:"报告全部不一致项")。

## 5. 隔离契约

- 优化器输入清单 = 快照(可观测状态)+ 冻结产物 + train_stats;**无** episode 真值、无未来到达表、无隐藏状态;
- 验证:复用实验 I 门2 协议(替换真实未来执行真值 → 首动作不变)+ 输入审计(构建 MILP 的数据结构逐一列出字段来源);
- 未到达任务:不在就绪池;其到达表不进入 MILP(到达模型为 v1+ 议题)。

## 6. 串行配置的机制对应(与 v7 差异)

| 机制 | v7 | serial_v0 | DES 开关 |
|---|---|---|---|
| 共驻 | enabled(cells 11 个) | `colocation_profile.enabled=false` | `_colocation_profile()` 返回 None |
| 预取干扰 | enabled | `prefetch_interference.enabled=false` | `_prefetch_interference_profile()` |
| 抢占 | `preemption_enabled=true` | `false` | 抢占路径短路 |
| 批处理 | enabled | `batching_engine.enabled=false` | `_batching_engine_profile()` |
| 主动预取 | `prefetch_overlap=true` + 策略计划 | `prefetch_overlap=false` | invoke gate 不触发 `initialize_prefetch()` |
| 请求级抢占 | enabled | `request_preemption.enabled=false` | `_request_preemption_profile()` |
| 加载/驻留/驱逐/准入 | 保留 | **保留** | `plan_gpu_admission` 等不变 |
| profile contract | 需要(cells) | **保留**(键保留、开关关) | `covered_strata` 仍可用,模板处理与主实验一致 |

**F0/PDRS 基线必须在 serial_v0 下重跑**(裁定),作为 v0 的对照;不覆盖 v7 原配置与既有冻结结果。

## 7. 审计输出(每次求解记录)

`status / objective / best_bound / gap / solve_ms / fallback / planned_nodes / selected(first action) / eviction_set / load_events / boundary_completion_ms`,并附输入指纹(serial 配置 sha、产物路径、train_stats 来源、代码 commit)。
