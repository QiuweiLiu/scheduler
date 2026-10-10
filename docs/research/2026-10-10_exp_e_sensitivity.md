# EXP-20261010_prefetch_sensitivity_v1 — 预取参数敏感性(实验 E)

**状态:** 设计冻结(运行前) · 2026-10-10
**动机(顾问清单 + 参数出处审计遗留项):** 排除"性能只是某个参数碰巧选得好"。
`2026-10-05_parameter_provenance_main_line.md` 列出的未敏感性设计选择:
**预取深度 k=1、并发 1、V/U 聚合范围 ready+running**。
本实验每个未测选择做一个单变量替代臂,其余与主线逐位相同。

## 1. 臂(自然 confirm300;全部 = F0 排序 + 最小 ΣV 驱逐 + 保守预取,仅一处不同)

| 臂 | 改变 | 精确语义 |
|---|---|---|
| `pdrs_resident`(主线,冻结三件套) | — | k=1;**单次调用最多 1 个预取目标**;V/U 聚合 ready+running |
| `pdrs_resident_k2` | 预取深度 k=1→2 | 预取目标的 U(m) 按**未来两步**累积需求打分(其余不变;V/驱逐不受影响) |
| `pdrs_resident_inflight2` | 同时预取数 1→2 | 单次计划可给出 **top-2 目标**,各配无驱逐可容纳且未被本计划占用/无 pending 的设备;其余不变 |
| `pdrs_resident_readyonly` | 聚合范围 | V/U 只聚合 **ready** 节点(排除 running);预取与驱逐同时生效于该口径 |

- 未实现/未测的第四项(披露):"预取可触发驱逐"变体——执行器层(initialize_prefetch)不执行驱逐,
  实现需动内存账本;本轮不做,登记为遗留。

## 2. 预注册判据(冻结)

- 头条:`变体 − 主线`,mean 与 p95,逐集配对 bootstrap 2000/seed 11;零失败硬断言。
- 判据:
  - **设计选择被支持**:变体不显著优于主线(CI 含 0 或为正)→ 冻结参数不是"碰巧选好",敏感性检验通过;
  - **设计选择被证伪**:任一变体在 mean 或 p95 上显著优于主线(CI_upper<0)→ 如实报告并重评冻结选择
    (注意:单一确认集上的多臂比较,需按探索性口径披露)。
- 机制计数(预取触发/使用/浪费/容量拒绝,驱逐,冷加载)全量输出作旁证。

## 3. 协议与成本

- 同一 commit;natural confirm300;v7 substrate;冻结 projection + 同一 pack;单次运行(5 臂)。
- 成本:5 臂 × 300 集 ≈ 25 分钟(本地 CPU)。
- 运行器:`scripts/prefetch_sensitivity.py`(--formal 要求本文件 + git HEAD);
  产物 `experiments/EXP-20261010_prefetch_sensitivity_v1/{RESULT.md,artifacts/prefetch_sensitivity_v1.json}`。
