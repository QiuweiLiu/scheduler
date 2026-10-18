# Gate 1A + Arrival Summary 统一冻结文档(2026-10-16,探索性证据冻结)

> 状态:**EVIDENCE_FROZEN(探索性)**。本线(串行环境首动作 × 未来到达信息)正式收束;Arrival-Holdout(validation ep10-19)全程未使用。下一阶段=共驻资源语义与可优化性(另立契约)。

## 1. 设计

- **问题**:固定调度前缀下,未来到达信息是否改变最优首动作?其价值能否被低维可预测摘要保留?
- **环境**:串行配置(serial_prefetch_v1,sha256 前缀记录于报告 des_fingerprint);冻结 Workload r150_b15(train 来源 20 集/480 模板池)。
- **状态**:8 个预冻结状态(分层:多未来 45-56 / 中等 21-25 / 少 1-2;eps 7-14;按剩余到达数排序选顶 3/中 3/末 2,非按收益)。
- **场景**:13 个/状态(原场景 + 12 捐赠;捐赠 eps 0-6+15-19;已到达数量锚定 + 首个到达 ≥ t+500ms)。
- **动作**:全部合法 (node_id, gpu_index);强制首动作 + 固定 PDRS 续跑;同场景共享同一未来。
- **断言(全过,hard_failures=0)**:派发时刻/动作池/场景指纹一致、强制动作完整执行(node+gpu)、任务全完成、捐赠到达唯一匹配;失败 fail-fast 非零退出。

## 2. 结果(13 场景口径)

**完美到达信息 VOI**:8 状态等权平均 **623.7ms**(范围 0~1344.9ms);不同最优动作 1-7 种。

**低维摘要消费(防作弊留一评价,13 场景)**:
| 摘要 | 匹配覆盖率 | LOO 增益(相对 LOO-I0) | **LOO 相对 PDRS** | 最严重恶化 |
|---|---|---|---|---|
| S1 粗分桶(0/1/2-3/≥4) | 0.77-1.0 | 平均 **−71.9ms**(+134/+90/+43 vs −488/−316/−39) | 混合(+497/+91/+80/+39 vs −689/−437;3 状态 ≈0) | **−4,807.7ms** |
| S3 双 30s 桶模型集合 | 0.23-0.92 | 平均 −37.6ms | +345/+280 vs −914/−11 | −6,459.3ms |
| S2 逐模型计数 / S1 精确 / S3 细版 | ≈0(11-13 单例) | 不可用(场景识别偏差) | — | — |

**附录敏感性(一次性,min-match≥4,不再调参)**:覆盖率降至 0.46-1.0;ep13 由 +51→−437、ep08 +39→+123,无稳定改善。

**关键修正**:LOO-I0 基线自身常低于 PDRS(ep12@0.55 −598.7、ep13 −436.6、ep09 −224.3)——此前报的"摘要增益"是相对一个有缺陷的估计基线,不是相对 PDRS。按对 PDRS 的口径,摘要策略与基线均**未证实稳定正收益**。

## 3. 结论(论文可用表述)

1. **Future arrivals can change first-action optimality**——固定前缀下仅变未来到达即可改变最优首动作,存在非零样本内完美信息价值(有限场景池上界);
2. **Information value is state- and scenario-dependent**——不同状态/动作集合/捐赠场景 VOI 差异大(0~1.3s);同一状态换捐赠集可从 +285ms 变 0;
3. **Low-dimensional information is not necessarily actionable**——短窗口到达计数/模型需求摘要在当前消费规则下未通过留一评价,且下行风险(单场景 −4.8~−6.5s)远超平均收益;**"存在 VOI" ≠ "可可靠兑现"**。

**禁止声称**:低维摘要普遍无信息;未来信息对所有方法无价值;已证明在线调度性能改善;中等竞争甜点区为一般规律。

## 4. 限制与排除

- 场景=捐赠拼接("到达场景敏感性集成"),非真实条件未来分布;状态仅 6 episode(ep12 被 3 次使用),非独立统计样本;
- ep03@0.50 排除主表(派发时刻状态审计不一致,148.4ms 降附录);ep12@0.25 池=1 结构性 0;ep02@0.75 无未来到达不适用;
- 模板-模型映射完整性已核查:480/480 映射成功、0 缺失、0 空集;映射文件 sha256=`15c62dafd99701b3`(results/processed/r7_workload_v041_ontology_no_run_container/job_templates_r7_v041.jsonl,10.7MB,重建=冻结 Workload 生成流程);
- S3 语义=两个时间桶内模型**出现集合**,非动态需求预测。

## 5. 证据与复现

- `artifacts/gate1a_p2.json` / `gate1a_p2_expanded.json`(8/13 场景完整成本矩阵、配对 ΔJ、PDRS 成本、逐场景最优、指纹、捐赠元数据、Top-K、LOO);
- `artifacts/arrival_summary_analysis.json` / `_expanded.json`(摘要分组/保留率/LOO);
- `artifacts/summary_freeze_checks.json`(映射核查 + LOO-I0/Summary vs PDRS + min-match 附录);
- `artifacts/gate1a_ep03_audit.json`;运行器 `scripts/jct_milp_v1_gate1a_p2.py`、`scripts/jct_milp_v1_summary_analysis.py`;
- 复现:`PYTHONPATH=src python scripts/jct_milp_v1_gate1a_p2.py && PYTHONPATH=src python scripts/jct_milp_v1_summary_analysis.py`。

## 6. 下一阶段(仅规划,未启动)

**共驻资源语义与可优化性**:先独立验证真实 GPU 共驻/冷加载/显存/干扰的物理证据与 DES 语义,再做有限 PDRS 对比与全动作离线反事实 Headroom;**新核心假设**:未来模型需求能否改善**跨时间承诺决策**(驻留/预取/驱逐/放置)?不预设正收益,不立即训练预测器/调度器。
