# EXP-20261012_milp_v0 — 阶段一证据

## 1. 串行配置派生与冒烟

- 派生自 `substrate_extension_v7.json`(sha256 `165d7631da595772…`),**未覆盖原文件**;
- 派生方式:保留全部键,置 `colocation_profile / prefetch_interference / request_preemption / batching_engine` 的 `enabled=false` + `preemption_enabled=false` + `prefetch_overlap=false`;
- 产物:`artifacts/serial_v0_extension.json`(sha256 `96fc531cf7f8edf7…`);
- 冒烟(训练集第 0 集,pdrs_resident):
  - 完成 16/16,失败 0;
  - `prefetch_count=0`(主动预取关闭 ✓)、`preemptions=0`(抢占关闭 ✓)、`colocation_profile_enabled=False`(共驻关闭 ✓);
  - `gpu_evictions=3`(驱逐/驻留机制保留 ✓);mean 46,808.4 ms。

## 2. 求解器环境

- 本机 print 环境:scipy **1.17.1** 已装,`scipy.optimize.milp`(HiGHS)可用;
- 仓库既有先例:`milp_sanity.py`(scipy/HiGHS 接口 + 回退 + gap 报告)、`cp_rho.py`(CP-SAT 候选/首动作/状态报告)——复用清单见 `MILP_FORMULATION.md` §10。

## 3. 复用调研结论(裁定最看重项)

| 结论 | 证据 |
|---|---|
| 不新建第二套调度框架 | 候选表示 `CpRhoCandidate`、候选构建 `candidate_from_scheduler_row`、scipy/HiGHS 接口模式、回退语义、状态/gap 报告、执行路径(实验 I `des_rollout`)全部复用 |
| v0 增量仅为数学扩展 | H5 链虚拟节点、驻留/加载/驱逐变量、按加载事件容量约束、H5 边界目标、串行环境契约 |

## 4. 待办(审查通过后)

1. 实现 MILP 求解模块(v0 变量/约束/目标,复用 `milp_sanity` 骨架);
2. 实现并运行 T1–T10(T11 为 v2 预留);
3. 全部通过后进入 v0 性能验证准备。
