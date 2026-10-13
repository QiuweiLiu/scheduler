# 预注册:confirm300 正式对照(冻结)

日期:2026-10-13。依据:顾问准入裁定(`docs/research/2026-10-13_milp_v0_admission_ruling.md`)的协议草案;
本轮按"先出对比、净收益标记未测"的加速口径执行(用户指示)。

## 冻结指纹

| 项 | 值 |
|---|---|
| 代码 commit | `7c7e642`(冻结后不得改动;改动即作废重跑) |
| 串行环境配置 | `serial_v0_extension.json` sha256[:16] = `96fc531cf7f8edf7` |
| 评测脚本 | `scripts/milp_v0_performance.py` sha256[:16] = `3e06517235806a41` |
| 评估集 | confirm300(300 个 episode,`validation_split_dev700_confirm300.json`;工作流来源隔离;此前未用于 MILP 调参)ids sha256[:16] = `e08d213e16b488d7` |
| 未来产物/资源目录 | `sstar_predictor_artifacts_dist_sched` + `resource_v2_artifacts/f0_seed11`(冻结,同主线) |
| Bootstrap 种子 | 11(2000 次,episode 配对) |

## 臂与预算

- **M0** = `sameshape_h5_p95`(F0 参照)/ **M1** = `pdrs_resident`(主线)/ **M2** = `milp_v0a`(在线,回退到 M1);
- **主预算 50ms**(HiGHS time_limit);敏感性 100ms、500ms 另跑,不用于主结论;
- 回退策略:M2 失败/不可行/无合法动作 → 回退 M1;全部回退与非法候选记录。

## 指标与口径(顾问要求的修正)

- **零开销算法收益**(主口径):mean JCT;**P95 双口径**——合并 P95(全部请求)与每集 P95 均值,分开报告,不混用;
- **独立在线成本**:每决策端到端墙钟(观测构建+建模+求解+提取)P50/P95/mean/max;求解状态计数(OPTIMAL/TIME_LIMIT);回退率;暴露延迟;
- **净 JCT:未测**(未做事件级延迟注入,按顾问允许的替代方案分列两段,不得声称净正/净负);
- 统计:episode 配对,聚类 Bootstrap 95% CI(seed=11);不根据正式结果继续调参。

## 已知限制(诚实披露)

1. 50ms 预算**不约束端到端墙钟**(开发集实测 p95 墙钟 ≈512ms,与 500ms 档几乎相同——尾部成本在 Python 侧建模;顾问已预警);
2. 求解时限受墙钟影响,边界状态可能在 TIME_LIMIT/OPTIMAL 间波动(开发集重复运行观测到小幅漂移);
3. v0 串行受限环境,结论不外推到完整 v7 环境("v0 的受限子环境实验与 v1 的完整 DES 实验分开报告")。

## 命令

```sh
PYTHONPATH=src python scripts/milp_v0_performance.py --episodes 300 --budget 0.05 --split confirm \
  --out experiments/EXP-20261012_milp_v0/artifacts/performance_confirm300_50ms.json
# 敏感性:--budget 0.1 / 0.5
```
