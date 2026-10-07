# 报告材料(Report Materials)

> 本文件夹 = 汇报/论文用**交付物**的唯一工作副本(图 + 表)。
> 源数据(逐集指标、机制计数、运行配置)在 `experiments/EXP-*/artifacts/`(冻结、可追溯)。

## 内容

### `figures/`(5 张数据图,PNG + SVG)

| 文件 | 内容 | 对应汇报页 |
|---|---|---|
| `fig3_main_table_mean_forest` | 主表 forest:F0 vs 五条适配基线 + FCFS(Δmean + CI);角注 p95 反转 | 结果-1 |
| `fig4_design_space_pruning` | 设计空间裁剪热图:PDRS(两种口径)/oracle/抢占/驻留 × mean/p95 | 为什么没有更复杂 |
| `fig5_residency_result_and_mechanism` | 驻留动作四指标 Δ+CI;机制柱(驱逐/冷加载/重载/预取命中) | 核心结果 |
| `fig6_attribution` | 因果拆解:point/distribution/shuffle/oracle × mean/p95 | 归因 |
| `fig7_tail_story` | 尾部故事:Parrot/驻留/抢占/无动作 的 p95 | 尾部机制 |

### `tables/`(4 组结果表,MD + CSV)

| 文件 | 内容 |
|---|---|
| `main_table_vs_F0` | 主表:五基线+FCFS vs F0(mean/p95/违约率/makespan + CI) |
| `residency_arms_vs_F0` | 全部驻留臂 vs F0(含 v1/p95/preempt 三个 run) |
| `mechanism_counters` | 机制计数(300 集合计 + 相对 F0 百分比) |
| `system_level_vs_main_line` | 系统级:各基线 vs 主线 `pdrs_resident`(逐集配对) |

## 重新生成

```bash
# 图(需要 conda 环境 print)
MPLCONFIGDIR=/tmp/mpl-cache conda run -n print python scripts/make_report_figures.py
# 表
python3 scripts/make_report_tables.py
```

## 重要口径说明

- 所有数字来自 **frozen confirm300**;参照 F0 在各 run 中逐位复现(69,788.8878 ms);
- **机制更正**:`−18%` 驱逐降幅属于"仅驱逐"臂(pdrs_evict);**主线 `pdrs_resident` 的精确值为
  驱逐 −10.3% / 冷加载 −21.5% / 重载 −22.3%,预取命中 41.9%**;
- 系统级表为**跨接口**对比(基线不带驻留动作)→ 只作端到端性能,不作机制归因;
- 主表主张仅限**均值**;p95 上 Parrot 优于 F0(原空间);makespan 上 Torpor 更好。
