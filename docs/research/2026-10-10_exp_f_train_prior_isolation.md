# EXP-20261010_train_prior_isolation_v1 — 训练集先验严格隔离重跑(实验 F)

**状态:** 设计冻结(运行前) · 2026-10-10
**动机:** 实验 A 的池化先验使用了**全部预测行(含 confirm300 的预测)**构建,不够严格。
本实验:(a) 先做**数据隔离审计**;(b) 历史先验**只用预测器训练集**(j-series `j_train`,拟合用
split;`j_validation` 仅选型、`j_test` 训练未读)构建;(c) 新增 **Full-Prior** 臂(全部预测通道替换);
(d) 三组同一调度规则/GPU 环境/驱逐与预取机制,唯一差异 = 预测信息来源。**不覆盖旧实验文件**
(新实验目录)。

## 1. 数据隔离审计(运行前已执行,产物内嵌)

| 层级 | 事实 |
|---|---|
| Episode 级 | confirm300 ∩ dev700 = **0**(episode id 互斥) |
| 模板级 | 工作负载 640 模板 = 480 train + 160 validation;confirm300 全部使用 **validation 模板**(160 个);dev700 与 confirm 共用同一批 160 个 validation 模板(episode 不同) |
| 预测器训练数据(j-series) | train 13,754 行 / 988 runs / 240 视频;validation 2,029 / 144 / 30;test 1,520 / 108 / 30 |
| **j-train vs 工作负载** | **视频级 0 重叠、run 级 0 重叠**;validation/test 同样 0 |
| j-train vs confirm / dev | run 级 0 / 0 |

**结论:先验池(j_train)与工作负载(confirm 及 dev)在视频/run 级完全不相交;confirm 与 dev 在
episode 级不相交。** 对照口径:先验 = 预测器训练集合的真实未来目标统计(非预测值;无标签泄漏)。

## 2. 三臂(其余全部逐位相同)

| 臂 | 预测来源 | 说明 |
|---|---|---|
| **Ours** | 冻结实例级 H5 预测包(原样) | 主线 `pdrs_resident`(F0 排序 + 最小 ΣV 驱逐 + 保守预取) |
| **Partial-Prior** | pack 变换:仅**未来模型概率 + 剩余长度概率** → j_train 先验 | 排序不消费这两通道 → 排序 byte 一致(可证);信念走先验 |
| **Full-Prior** | pack 变换:模型 + 长度 + **运行时 + 加载 + lane** → j_train 先验 | 排序消费运行时/加载 → 排序也变先验驱动(设计如此);其余规则不变 |
| (参照) | 冻结 pack,无动作 | `sameshape_h5_p95` |

## 3. 先验通道定义(j_train,精确冻结)

- **模型**:每步 k=1..5,`future[k-1].model_class` 频率(按有 ≥k 步的行归一;含 CPU 伪模型/终止标记)。
- **长度**:`len(future)` 直方图 {0..4},索引 5 = "≥5"(有界截断);归一化。
- **运行时**:每步 k:`targets.runtime_ms` 的 mean / p50 / p90 / p95 / **cvar95**。
- **加载**:每步 k:occurrence = P(load_ms>0);正样本 p50/p90/p95。
- **lane**:每步 k 多数类(cpu-metadata-adapter-v1 vs 其余 → cpu/gpu)。

## 4. pack 变换字段清单(冻结;其余字段不动)

- Partial:step `model_probabilities`(→先验分布)、`model_probability`(→先验 max)、
  row `length_probabilities`、`predicted_future_length`(→先验 argmax)。
- Full:以上 + step `execution_lane`、`resource.runtime_mean_ms`、`resource.runtime_ms_quantiles`、
  `resource.cvar95_ms`、`resource.load_occurrence_probability`、`resource.load_duration_ms_quantiles`。
- 未替换:role/action_family/retry/merged/prototype 等(现无调度消费者,披露);
  `resource.bin_schema_id`/`resource_head_id` 为来源标识,保留。
- 当前节点估计 = 工作负载 train_stats,**三臂相同**(非差异来源)。

## 5. 指标与统计(冻结)

- 主指标:mean / p95 完成时间;机制:**驱逐次数、冷加载次数、预取次数、有效预取比例**(used/prefetch);
  全部逐集配对 bootstrap 95% CI(2000, seed 11);零失败硬断言。
- 有效比例:每臂总量比 + 仅在"两臂都有预取"的 episode 子集上配对(子集大小披露)。
- 判据:
  1. **实例预测独立收益**:`Ours − Partial`(或 `− Full`)在 mean 或 p95 上 CI_upper<0;
  2. **机制贡献**:`Partial/Full − 参照` 是否仍显著(先验下动作是否仍有价值);
  3. **非显著规则**:CI 含 0 **既不支持差异也不支持等效**——不作"两者等效"结论(未做等效性检验);
     如需"等效"主张需预注册等效边界 + TOST/更大功效(登记为后续)。
- 披露:先验来自训练集合(不同视频集,统计可用但分布可能偏移——这正是"历史先验"语义);
  Full-Prior 的排序退化是其设计的组成部分;单轮 confirm300;所有对照同刻同 seed。

## 6. 产物

- `experiments/EXP-20261010_train_prior_isolation_v1/{RESULT.md,artifacts/train_prior_v1.json}`
  (产物含:隔离审计、先验统计全文、prior sha、j_train sha、逐集六指标、全部配对 CI、比例子集);
- 运行器 `scripts/train_prior_isolation.py` + 变换库 `scripts/train_prior_lib.py`;
- 成本:~4 臂 × 300 集 ≈ 25 分钟(本地 CPU);smoke 10 集先行。
