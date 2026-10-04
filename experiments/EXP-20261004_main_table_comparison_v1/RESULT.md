# RESULT — EXP-20261004_main_table_comparison_v1（主表正式对比）

**状态**：正式运行完成（2026-10-04）。固定 commit `30ce2bdeab8cb23e5182e741289df4247546de29`；
config = v7 substrate（sha `165d7631…`）+ profile 契约；confirm300 全集；300 集 × 6 臂，
wall 1512.1 s（≈25 分钟，本地 CPU）。产物 `artifacts/main_table_formal_v1.json`
（含逐集 episode_values 与各臂激活计数）；冒烟 `main_table_smoke_v1.json`（10 集，82 s，0 失败）。

## 设计（运行前冻结）

- **参照**：F0 `sameshape_h5_p95`（deployment 预测器，overlay artifacts + horizon 5）。
- **臂**：`parrot_appfifo` / `qlm_queue` / `llmsched` / `hermes_gittins` / `torpor_lifecycle`
  （冻结适配；各自独立前端，互不借产物）。
- **substrate**：v7 实测配置（真实档位共置 + 加性加载干扰 + 请求级抢占 + 批处理）+ 证据契约
  （shape=节点角色证据、episode 身份=节点统一证据）。
- **指标**：`mean_completion_ms`；Δ = 候选 − F0；paired episode bootstrap 2000，seed 11；
  non-inferior = CI_upper < +485 ms；statistically better = CI_upper < 0。
- **零失败硬断言**：任何 episode failed_jobs>0 或未完成即中止（不允许 survivor mean）。
- 运行器 `scripts/main_table_comparison.py`；运行前门禁已登记（EXPERIMENT_GATE）。

## 结果（confirm300；Δ 正 = 比 F0 差）

| 臂 | mean (ms) | Δ (ms) | CI95 | 更差集数 | 判定 |
|---|---:|---:|---|---:|---|
| **F0（参照）** | **69,788.9** | — | — | — | — |
| hermes_gittins | 70,139.3 | **+350.4** | [+46.4, +616.5] | 200/300 | inferior |
| llmsched | 70,728.2 | +939.3 | [+597.0, +1,317.2] | 218/300 | inferior |
| parrot_appfifo | 71,836.9 | +2,048.0 | [+1,608.7, +2,491.8] | 223/300 | inferior |
| qlm_queue | 73,168.0 | +3,379.1 | [+2,631.9, +4,212.9] | 235/300 | inferior |
| torpor_lifecycle | 73,310.5 | +3,521.6 | [+2,763.8, +4,350.9] | 193/300 | inferior |

**全部五条基线 CI 下界 > 0 → F0 在 confirm300 上统计显著优于全部主表基线。**
（CI 上界均 < +485 ms 的非劣判定不适用：五条全部越过 inferior 线。）

## 机制激活（正式运行，真实计数）

- Parrot：app_fifo flip 6,134 / 85,122 决策。
- QLM：load_present 13,795；stochastic reorder 14,309；swap-cost 反事实 1,188。
- Hermes：条件 PDGraph 74,978；观测精化 10,457；在线预热触发 **54**；Gittins-vs-mean flip 284。
- Torpor：resident_hit 76,348；canonical 三元组 covered_load 8,774；cold_swap 8,774。

## 与旧正式运行的关系

旧五臂（tie/pythia/llmsched/latency/agentix，694e6fc，无扩展配置）在主表口径下**不可直接比较**：
substrate 不同（旧=串行近似，新=多进程实测面）。旧结果仅作历史记录；主表以本产物为准。

## 边界（随表披露）

- 这是**同一 substrate 上的完整系统对比**；单独主张"我们的调度器更强"仍需 same-interface 2×2
  （信息对等）——已在 PLAN。
- 各基线带冻结并披露的 adaptation/omission（见 manifest 与两轮审核报告）。
- episode-cluster bootstrap 只描述冻结 300 集内的不确定性，不外推到未见视频。
- 本地测试/运行为 UNVERIFIED execution report（审核方未独立运行）。
