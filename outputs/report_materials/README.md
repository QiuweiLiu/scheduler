# 报告材料(Report Materials)

> 本文件夹 = 汇报/论文用**交付物**的唯一工作副本(图 + 表)。
> 源数据(逐集指标、机制计数、运行配置)在 `experiments/EXP-*/artifacts/`(冻结、可追溯)。

## 内容

### `final/` —— **正式用图**(定稿一张、进一张)

汇报/论文**直接使用**的版本放这里(工作区里仍保留全部历史版本):

| 文件 | 用途 |
|---|---|
| `fig15_predictor_architecture_zh.png` | 预测器架构图(图像生成版,中文;用户提供)—— **暂定** |
| `fig14_predictor_structure_zh.png` / `.svg` | 多步结构预测跨族对比(中文,汇报用)—— **已定稿** |
| `fig14_predictor_structure.png` / `.svg` | 多步结构预测跨族对比(英文,论文用)—— **已定稿** |
| `fig12_lanes_zh.png` / `.svg` | 时间线泳道图(中文,汇报用)—— **已定稿** |
| `fig12_lanes.png` / `.svg` | 时间线泳道图(英文,论文用)—— **已定稿** |
| `fig9_workflow_zh.png` / `.svg` | 视频 Agent 工作流图(中文,汇报用)—— **已定稿** |
| `fig9_workflow.png` / `.svg` | 视频 Agent 工作流图(英文,论文用)—— **已定稿** |
| `fig2_system_zh.png` / `.svg` | 系统框图(中文,汇报用)—— **已定稿** |
| `fig2_system.png` / `.svg` | 系统框图(英文,论文用)—— **已定稿** |

### `figures/`(工作区:15 张图,含 fig10 系列备选;PNG + SVG;每张都有英文版与 **`_zh` 中文版**)

**中文版**:文件名加 `_zh` 后缀(如 `fig2_system_zh.png`);数据图由 `scripts/make_report_figures_zh.py` 生成,
示意图由英文 SVG 翻译生成(字体 STHeiti)。

| 文件 | 内容 | 对应汇报页 |
|---|---|---|
| `fig3_main_table_mean_forest` | 主表 forest:F0 vs 五条适配基线 + FCFS(Δmean + CI);角注 p95 反转 | 结果-1 |
| `fig4_design_space_pruning` | 设计空间裁剪热图:PDRS(两种口径)/oracle/抢占/驻留 × mean/p95 | 为什么没有更复杂 |
| `fig5_residency_result_and_mechanism` | 驻留动作四指标 Δ+CI;机制柱(驱逐/冷加载/重载/预取命中) | 核心结果 |
| `fig6_attribution` | 因果拆解:point/distribution/shuffle/oracle × mean/p95 | 归因 |
| `fig7_tail_story` | 尾部故事:Parrot/驻留/抢占/无动作 的 p95 | 尾部机制 |
| `fig1_problem` | 问题示意:已知前缀 vs 未知后缀 + 决策时刻 + 两张 GPU | 问题页 |
| `fig2_system` | 系统框图:prefix→冻结预测器→{排序,驻留控制}→GPU;机器表只在 evaluator 层 | 方法页 |
| `fig8_positioning` | 相关工作二维定位(五近邻 + Ours) | 论文用(可选汇报) |
| `fig13_predictor_families_single_step` | 预测器跨族(单步):动作族 top-1(18 个模型)/ 运行时 rel-MAE(9 个模型,含噪声参考线) | 预测器-1 |
| `fig15_gru_architecture` | 预测器架构图:共享因果 GRU + 多任务头(部署 J3/F0;因果边界/冻结区标注) | 方法-预测器 |

### `tables/`(4 组结果表,MD + CSV)

| 文件 | 内容 |
|---|---|
| `main_table_vs_F0` | 主表:五基线+FCFS vs F0(mean/p95/违约率/makespan + CI) |
| `residency_arms_vs_F0` | 全部驻留臂 vs F0(含 v1/p95/preempt 三个 run) |
| `mechanism_counters` | 机制计数(300 集合计 + 相对 F0 百分比) |
| `system_level_vs_main_line` | 系统级:各基线 vs 主线 `pdrs_resident`(逐集配对) |

## 重新生成

```bash
# 数据图(需要 conda 环境 print)
MPLCONFIGDIR=/tmp/mpl-cache conda run -n print python scripts/make_report_figures.py
MPLCONFIGDIR=/tmp/mpl-cache conda run -n print python scripts/make_report_figures_zh.py
# 预测器跨族图(fig13/fig14)
MPLCONFIGDIR=/tmp/mpl-cache conda run -n print python scripts/make_predictor_figures.py
MPLCONFIGDIR=/tmp/mpl-cache conda run -n print python scripts/make_predictor_figures_zh.py
# 表
python3 scripts/make_report_tables.py
# 示意图(fig1/fig2/fig8):SVG 由 GPT 第 9 轮提供,渲染:
MPLCONFIGDIR=/tmp/mpl-cache conda run -n print python -c "
import cairosvg; from pathlib import Path
d = Path('outputs/report_materials/figures')
for n in ('fig1_problem','fig2_system','fig8_positioning'):
    cairosvg.svg2png(url=str(d/(n+'.svg')), write_to=str(d/(n+'.png')), scale=2.0)"
```

## 重要口径说明

- 所有数字来自 **frozen confirm300**;参照 F0 在各 run 中逐位复现(69,788.8878 ms);
- **机制更正**:`−18%` 驱逐降幅属于"仅驱逐"臂(pdrs_evict);**主线 `pdrs_resident` 的精确值为
  驱逐 −10.3% / 冷加载 −21.5% / 重载 −22.3%,预取命中 41.9%**;
- 系统级表为**跨接口**对比(基线不带驻留动作)→ 只作端到端性能,不作机制归因;
- 主表主张仅限**均值**;p95 上 Parrot 优于 F0(原空间);makespan 上 Torpor 更好。
