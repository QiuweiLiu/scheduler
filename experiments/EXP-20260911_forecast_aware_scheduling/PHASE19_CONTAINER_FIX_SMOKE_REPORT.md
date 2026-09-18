# Phase 19 — container-excluded workload：配对冒烟对照（v02 vs v03）

**Date:** 2026-09-17
**剧本：** GPT 规划（`docs/research/…` 无，见会话）→ 本报告为执行结果
**状态：** **情况 B（必须重审机制链）** —— 策略排序改变，且 E2 对 E0 的优势消失

## 做法

- v03 workload：`results/processed/r7_workload_v03_no_run_container/`
  - `job_templates_r7_v03.jsonl`：删除 640 个 `event_type == "run"` 整条流墙钟容器
    （纯删终端叶子，无重连；removed 640 / remaining 8,295 / templates 640 / roots 640 / dangling 0，
    `runtime_removed_ratio = 0.5024`）
  - `episodes/workload_{train,validation}_r7_v03.jsonl`：用同一套构造公式重算（episode_id、template_id、
    到达**位置**、压力、拓扑、初始驻留全部与 v02 一致；窗口按新工作量重算，**实测负载仍为 0.5000**）
- **可信性锚点**：重建脚本的自检模式用 v02 模板重算 v02 的 21,000 条 episode，**逐字段等于存档值（0 不匹配）**，
  说明重建是构造器的忠实逆变换。
- 冒烟：`validation_000000`–`validation_000099`（100 集）× 5 策略 × 两个 workload，同一 episode id 配对；
  bootstrap 以 **episode_id** 为聚类（B=2000）。
- 产物：`outputs/phase19_container_fix_smoke_v03/{legacy_v02,v03_no_run_container}/`，对比 `paired_diff.json`。
- legacy v02、既有 `outputs/phase*`、预测 artifact **均未改动**。

## 排序（mean completion，越小越好）

| 排名 | legacy v02 | v03（容器已排除） |
|---|---|---|
| 1 | `predopt_h5_r95` 145,877 | `predopt_h5_r95` 71,468 |
| 2 | `predopt_h5`(E2) 150,337 | `predopt_h5_r50` 73,307 |
| 3 | `predopt_h5_r50` 160,141 | `myopic`(E0) 74,334 |
| 4 | `fcfs` 160,627 | `predopt_h5`(E2) 74,597 |
| 5 | `myopic`(E0) 160,958 | `fcfs` 86,146 |

## 配对对照（ms，负 = 左边更好；[95% CI]，episode 聚类）

| 对照 | legacy v02 | v03 | 修正带来的变化 |
|---|---|---|---|
| r95 − E2 | **−4,460 [−5,693, −3,336]** | **−3,129 [−3,950, −2,417]** | +1,331 [+349, +2,284] |
| **r95 − E0** | **−15,081 [−19,074, −11,543]** | **−2,866 [−3,510, −2,254]** | **+12,215 [+8,930, +15,944]** |
| r50 − E0 | −817 [−1,736, +78] | −1,027 [−1,466, −564] | −210 [−1,315, +873] |
| **E2 − E0** | **−10,621 [−13,680, −7,862]** | **+263 [−392, +956]** | **+10,884 [+7,975, +14,070]** |
| fcfs − E0 | −331 [−1,659, +882] | +11,812 [+9,354, +14,400] | +12,143 [+9,289, +15,191] |

## 判定（按预注册判据）

- **排序改变** → 情况 B；
- **r95 仍是最优**，且 r95−E2 仍显著为负（−3.1 s，CI 排除 0）→ "尾部消费优于点消费"这条**存活**；
- **E2−E0 由 −10.6 s 变成 +0.26 s（CI 跨 0）** → **"未来信息本身有价值"这条不成立**（至少在点消费形式下）；
- **r95 相对 E0 的收益从 −15.1 s 缩到 −2.9 s**：删除重复的 workflow 级容器节点后，
  观测到的收益缩小约 **81%**，说明旧 workload 构造**显著放大了**调度收益。
  （措辞纪律：**不写** "81% of the gain was caused by the container artifact"——那是因果断言；
  正确写法：*Removing duplicated workflow-level container nodes reduced the observed improvement of r95 over E0
  by approximately 81%, indicating that the previous workload construction substantially amplified the scheduling benefit.*）
- FCFS 在 v03 下比 E0 差 11.8 s（v02 下几乎无差）→ 说明容器那个 54 秒巨型节点此前主导了队列动态。

## 结论

1. **必须重审机制链**：至少三条既有表述要改——"E2 显著优于无未来（−17,065ms, 1000 集）"、
   "未来信息价值"、以及"收益随 contention 增强"（后者需在 v03 上重跑压力 sweep 才能定性）。
2. **存活**：r95/q95 作为最保守消费仍最优；r95 vs E2 仍显著（收益变小但方向不变）。
3. **待决**：是否把主实验全部迁移到 v03（最小集的基线矩阵 + 6-cell sweep + 敏感性都要重跑），
   以及旧结果在论文中的处置（GPT 建议标为 legacy workload sensitivity，只作披露、不作主结果）。

## GPT 审阅后的措辞纪律与迁移计划（2026-09-17）

- **保留**：q95 > p50；aggregation matters；conservative aggregation 补偿系统性预测乐观并改善排序鲁棒性。
- **删除**："future information itself improves scheduling"；"contention amplified benefit"；"risk 来自 execution correlation"。
- **v02 结果处置**：不删除，作为 appendix / ablation 的 *legacy workload sensitivity*；
  论文写 "We identified and corrected a workflow-level container artifact during workload auditing.
  Results on the original workload are reported as sensitivity analysis."
- **v03 的到达时间必须声明**：*We preserve normalized arrival pattern and offered compute load rather than absolute timestamps.*
- **Phase 18 改名与限定**：应表述为 **forecast optimism calibration analysis**，不是 comonotonic risk validation；
  它支持"p50 偏乐观 / p95 更保守"，**不**支持"执行时尾部相依"或"随机 runtime 相关"。
- **迁移计划**：Phase 20-A 基线矩阵（validation 1000，FCFS/myopic/E2/r50/r95）→ 20-B 6-cell sweep（必须，因
  "contention 放大收益"已不可信）→ 20-C Phase16 四臂（必须，优先级低）。CP-SAT/MPC/BC 留到 P2。
