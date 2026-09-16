# Forecast-aware scheduling 规划（ChatGPT Web 评审，2026-09-11）

核心判断：**问题已从"预测器准不准"转向"未来行为预测是否改变调度决策并带来系统级收益"**。
建议停止优化 predictor，做一个 **forecast-aware scheduling ablation**；贡献叙事从"我们能预测未来"改为"未来预测在有限视野调度中是否提供可利用的信息"。

## 实验矩阵（优先级从高到低）

| 臂 | 未来拓扑 | 未来资源 | 说明 |
|---|---|---|---|
| E0 No-Future | ❌ | ❌ | 仅当前 ready node + 当前资源状态 + 历史统计表（Myopic 基线） |
| E1 Resource-only | ❌ | J3 预测 | 隔离"预测资源本身有没有调度价值" |
| E2 Topology-only | J3 预测（h1/h3/h5 + 属性） | 旧统计表 | **最重要的新实验**：仅知道未来会怎么走，有没有帮助 |
| E3 Full J3 | J3 预测 | J3 预测 | 完整预测器（主实验） |
| E4 Oracle Future | oracle | oracle | 上界；非部署方案 |

预期叙事：E3 > E1,E2 > E0；E4 给剩余空间。

## 指标与统计

- Primary：mean completion time（JCT）；deadline miss / lateness（若可用）。
- Secondary：GPU 利用率、排队等待、GPU 空闲、抢占次数、load stall 次数。
- 统计：同 episode seed/到达/配置；paired bootstrap（1,000 episodes，episode 为统计单位）。
- 事前成功标准：**强成功** = E3 vs E0 completion CI 下界 > 0 且 deadline miss 不恶化；**机制成功** = E3>E1（拓扑有贡献）、E3>E2（资源预测有额外贡献）。否则只能陈述"预测精度未转化为调度收益"（合理负结果）。

## 资源表 vs J3 runtime（消融，不作为默认替换）

- A：future topology + 静态资源表
- B：future topology + J3 预测 runtime
- C：future topology + oracle runtime
- memory：保持不进主调度（R0 已证不可靠），仅作未来方向。

## OOD 退化处理

- 不隐藏：S_* 上 role/family 降 3–4pp、runtime 误差 +24%。
- 分两层报告：predictor 层（已测）+ scheduler 层（关键：预测退化是否导致调度退化）。
- 两种可能：A 预测降但调度仍升 → 调度只需粗粒度未来信息；B 两者都降 → 需要 domain adaptation / uncertainty-aware scheduling。
- 不做 S_* 调参、不做 OOD calibration；只做 robustness analysis。

## 执行阶段与成本

- **Phase 0 接口验证（~1 天）**：J3 artifact → scheduler 的 node id 映射、资源块消费；100 episodes；gate：所有策略输出一致、无崩溃。
- **Phase 1 小 pilot（~1 天）**：100 episodes，E0/E2/E3/E4；看 effect size；gate：若 E3−E0 < 1%，不直接跑全量。
- **Phase 2 正式矩阵（3–5 天）**：1,000 episodes；策略：Myopic、PredOpt-H5(E0)、E1、E2、E3、Oracle；每策略 3 seeds；RTX 3060 可完成。

## 结论边界

- E3>E2>E0：可说"未来行为预测为 GPU 调度提供可操作信息，拓扑+资源结合带来额外收益"。
- 只有 E3>E0：只能说"future-aware scheduling 有效"，不能归因 topology vs resource。
- E2>E0 且 E3≈E2：未来结构有价值，但资源预测没有转化 → 转向 objective mismatch / uncertainty handling。
- 全部无提升：**不否定 predictor**；结论 = "prediction accuracy does not automatically imply scheduling benefit under current scheduler and workload"。

## 最高优先级建议

先做 E0/E1/E2/E3 四臂 pilot（不改 predictor，只验证核心科学问题）；有收益再谈 predicted vs oracle gap 与 uncertainty-aware scheduler；无收益则转查调度器是否真正利用未来、H=5 是否有效、目标函数是否匹配。
