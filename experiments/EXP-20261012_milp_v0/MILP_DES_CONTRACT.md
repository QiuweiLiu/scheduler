# MILP_DES_CONTRACT.md — 约束↔DES 语义映射契约(v2 修订版)

- **实验**: `EXP-20261012_milp_v0`;v2 修订依据:审查 M1–M6 + P0–P2(`docs/research/2026-10-12_milp_v0_review.md`)
- **v2 变更**: ①MILPObservation 严格投影(§2)②加载/计算时长口径(§3)③驱逐复现与 v0a/v0b(§4)④CPU 链(§5)⑤首动作合法集(§6)
- **基准环境**: `serial_v0_extension.json`(冒烟:16/16、prefetch 0、preemptions 0、colocation off、驱逐保留)

## 1. 状态获取(与实验 I 同一机制)

| MILP 输入 | DES 来源 | 说明 |
|---|---|---|
| 就绪节点 R | ready 堆(`release_ready`,`ready_nodes`) | `node_state=ready` |
| GPU 可用时刻 | **now + Σ 活动任务预测剩余**(`GPUActiveTask.predicted_work_ms × slowdown − elapsed`) | **禁止** `GPU.busy_until`/真实 `finish_ms`/`finish_heap`(M5) |
| 驻留 | `GPU.resident`(MB)+ 派生 **ρ01 标志** | MB 只进容量式 |
| 在途加载 | `GPU.prefetch_pending` / `loading_model_ids()` | 冻结参数 |
| 预测估计 | `estimate(node, train_stats)` | 与 DES 调度器同源 |
| H5 链 | 冻结产物 `future_h5[0].steps` → **合成虚拟节点**(`pred:<anchor>:<k>`) | **禁止** `greedy_chain_scenario()` 的真实 `pending_after`(M5) |
| 容量/驻留占用 | `gpu_topology_mb` / 模板 `resident_model_mb` | — |

## 2. MILPObservation 投影(M5,新增)

**定义**:优化器构建代码只能通过 `MILPObservation` 数据类读取状态;其字段=§1 表全部来源。

**禁止清单**(出现即失败,T6b 静态审计):
- `GPU.busy_until`、`GPUActiveTask.finish_ms`(真实)、`finish_heap` 中真实未来结束时间;
- 任何真实 `node.runtime_ms / load_ms` 真值(只能用 `estimate`);
- episode 的未到达任务表/未来到达;`template` 中隐藏后缀(超出 H5 链)节点;
- 实验 I 场景注入的真实 `pending_after` 占位节点。

**执行证据**:每次求解导出输入清单(字段名+来源函数)到审计输出。

## 3. 加载/计算时长口径(M3,新增)

| 量 | MILP 定义 | DES 对应 |
|---|---|---|
| 纯计算 `p_i` | `max(0.1, runtime_p50_ms − load_p50_ms)` | `Node.compute_ms` 属性(`max(0.1, runtime_ms − load_ms)`) |
| 加载 `ℓ_i` | `load_p50_ms`,仅当 `z_i=1` | 模型加载事件(`model_load`),未驻留时发生 |
| 完成 | `c_i = s_i + p_i`(等式) | `node_finish` 事件时刻 |
| 占用区间 | `[s_i − ℓ_i·z_i, c_i]` | 加载+计算连续占用(串行配置) |

**约定**:`runtime_p50_ms` **可能已含加载**,故纯计算必须扣除 `load_p50_ms`;一致性测试(T10a)按此口径核对单步转移。

## 4. 驱逐复现(v0b)与 v0a 范围(M4,新增)

| 版本 | 规则 | DES 对应 |
|---|---|---|
| v0a | 不新增驱逐;需要驱逐的状态 → 回退(不计成功) | 空驱逐集时的 `plan_gpu_admission` |
| v0b | 驱逐集=**按 DES 规则从计划派生**(串行配置:主线 min-V `eviction_values`;动作指定时 largest-first) | `plan_gpu_admission(eviction_preference/eviction_values)`;仅 `active=False` 模型可驱逐 |

**统一性约束**:计划阶段使用的驱逐集与 DES 执行时该规则给出的驱逐集**必须一致**;不一致=契约失败(T3b)。

## 5. CPU 链节点(M6,新增)

- H5 链中的 `lane=cpu` 节点:DES 中不占 GPU 槽(CPU/API 车道);
- MILP:无 `x/z/y` 变量,不占容量;仅链式时间约束(`s ≥ 前驱 c`,`c = s + p`);计入链与边界目标;
- 一致性:CPU 节点在 DES 事件中不产生 `node_dispatch`(GPU)事件;时间线按链顺序核对。

## 6. 首动作与执行(P0 修订)

1. **首动作提取**:仅从真实就绪节点 R 中取"最早开始"操作(并列按冻结 F0 键);虚拟节点因 `s_v ≥ c_anchor` 不可能成为最早操作;
2. **合法集校验**:执行前断言首动作 ∈ `dispatchable_gpus()` × 就绪池;
3. **无合法动作**:显式等待或回退 `pdrs_resident`,**不得派发预测节点**;
4. **执行**:实验 I `des_rollout` 强制首动作(无预取);其后委托 `pdrs_resident`;
5. **v0b 驱逐执行**:首动作派发时的驱逐按 §4 的复现规则(经动作通道传递,若首动作含驱逐变更;否则按主线默认)。

## 7. 一致性测试的真值对齐(T10a/T10b)

- **T10a(单步转移)**:将 DES 执行真值注入为 MILP 参数(实验 I 场景注入),比对**单步**转移:加载事件、驻留变化、驱逐集合、计算区间、容量占用;
- **T10b(完整计划可执行性)**:按 MILP 计划生成**脚本化动作序列**,在 DES 中以"逐步强制"回放(需扩展 `des_rollout` 支持动作序列);**不得**用"首动作强制 + PDRS 续跑"代替;回放中的每一步都要通过合法集校验;
- 浮点容差 ≤1e-6;结构性差异全部报告。

## 8. 串行配置差异表(不变)

| 机制 | v7 | serial_v0 | 开关 |
|---|---|---|---|
| 共驻/预取干扰/请求抢占/批处理 | enabled | `enabled=false` | 四个 profile getter |
| 抢占/主动预取 | true | false | `preemption_enabled` / `prefetch_overlap` |
| 加载/驻留/驱逐/准入 | 保留 | 保留 | `plan_gpu_admission` 等 |
| profile contract | 需要 | 保留(键在、开关关) | `covered_strata` 可用 |

F0/PDRS 基线必须在 serial_v0 下重跑;不覆盖 v7 与既有冻结结果。

## 9. 审计输出

`status/objective/best_bound/gap/solve_ms/fallback/planned_nodes/first_action/eviction_set/load_events/boundary_completion_ms` + **MILPObservation 输入清单** + 指纹(serial sha、产物路径、代码 commit)。
