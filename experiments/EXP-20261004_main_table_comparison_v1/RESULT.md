# RESULT — EXP-20261004_main_table_comparison_v1（主表正式对比）

**状态**：正式运行完成（2026-10-04，v2 = 审查修复后重跑）。固定 commit
`a7ab49367337e0234cfec7e4bda52d752ff3cc98`（运行器 + 修复）；config = v7 substrate（sha `165d7631…`）
+ profile 契约；confirm300 全集；300 集 × 6 臂，wall 1362.0 s（≈23 分钟，本地 CPU）。
产物 `artifacts/main_table_formal_v2.json`（含逐集 episode_values、激活计数、机制计数、
参照产物 manifest 校验）；冒烟 `main_table_smoke_v2.json`。
**v1（`30ce2bd`）与 v2 各臂均值逐位一致**（确定性校验：修复只改判定/记录，不改实验处理）。

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

判定按**三条独立陈述**（不把"非劣未建立"写成"已证明 inferior"）：
`statistically_worse := CI_lower > 0`；`non_inferior := CI_upper < +485 ms`；
`margin_inferior := CI_lower > +485 ms`。

| 臂 | mean (ms) | Δ (ms) | CI95 | 更差集数 | stat. worse | non-inferior | margin-inferior |
|---|---:|---:|---|---:|---|---|---|
| **F0（参照）** | **69,788.9** | — | — | — | — | — | — |
| hermes_gittins | 70,139.3 | +350.4 | [+46.4, +616.5] | 200/300 | **是** | 否 | **否** |
| llmsched | 70,728.2 | +939.3 | [+597.0, +1,317.2] | 218/300 | 是 | 否 | 是 |
| parrot_appfifo | 71,836.9 | +2,048.0 | [+1,608.7, +2,491.8] | 223/300 | 是 | 否 | 是 |
| qlm_queue | 73,168.0 | +3,379.1 | [+2,631.9, +4,212.9] | 235/300 | 是 | 否 | 是 |
| torpor_lifecycle | 73,310.5 | +3,521.6 | [+2,763.8, +4,350.9] | 193/300 | 是 | 否 | 是 |

**五条全部"统计显著更差"（CI 下界 > 0）；四条越过 +485 ms margin（margin-inferior）；
Hermes 的 margin-inferior 未建立**（CI 下界 +46.4 < 485）——即 Hermes 显著差于 F0，
但未证明差超过 margin。

## 已执行 vs 声明的 substrate 机制（审查修复，运行期计数）

冻结设计列了四个能力；本表**实际执行**的环境机制（运行期计数，v2）：

- **共置**：2,624,836 次准入（拒绝 1,901,244）；
- **引擎级同模型批处理**（按实测每请求延迟因子合并）：780,498 次准入（拒绝 1,310,626）；
- **加性加载干扰**：经 Hermes 预热路径实际消费（激活计数见下）；
- **请求级抢占**：0 事件——模拟器将抢占门控为"抢占型策略"能力（`myopic_preempt` 族），
  六臂均非抢占型策略（冻结适配不含抢占）；
- **策略侧批调度器（`batch_enabled`）**：设计关闭——它会把各臂排序目标替换成 myopic 代价，
  破坏冻结适配语义（审查方亦明确警告不得直接开启）。

## 机制激活（正式运行，真实计数）

- Parrot：app_fifo flip 6,134 / 85,122 决策。
- QLM：load_present 13,795；stochastic reorder 14,309；swap-cost 反事实 1,188。
- Hermes：条件 PDGraph 74,978；观测精化 10,457；在线预热触发 **54**；Gittins-vs-mean flip 284。
- Torpor：resident_hit 76,348；canonical 三元组 covered_load 8,774；cold_swap 8,774。
- 抢占事件：六臂 + F0 全部 0（声明为策略门控能力，非缺陷）。

## 参照产物 provenance（审查修复）

- `data/manifests/f0_reference_artifacts_manifest_v1.json`（入库）：base pack 与 F0 overlay
  逐文件 SHA256 + 树哈希（base `3f350705…`、overlay `944fb8b8…`）+ overlay 内部来源
  （artifact_sha256 `586ae65d…`、producer checkpoint、frozen_j3）。
- 运行器启动时**对 manifest fail-closed 校验**；校验摘要与 manifest SHA（`c986ed17…`）写入正式产物。

## 与旧正式运行的关系

旧五臂（tie/pythia/llmsched/latency/agentix，694e6fc，无扩展配置）在主表口径下**不可直接比较**：
substrate 不同（旧=串行近似，新=多进程实测面）。旧结果仅作历史记录；主表以本产物为准。

## 边界（随表披露）

- 这是**同一 substrate 上的完整系统对比**；单独主张"我们的调度器更强"仍需 same-interface 2×2
  （信息对等）——已在 PLAN。
- 各基线带冻结并披露的 adaptation/omission（见 manifest 与两轮审核报告）。
- episode-cluster bootstrap 只描述冻结 300 集内的不确定性，不外推到未见视频。
- 本地测试/运行为 UNVERIFIED execution report（审核方未独立运行）。
