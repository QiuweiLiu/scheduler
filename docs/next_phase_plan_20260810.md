# 下一阶段执行规划 v10(2026-08-10):A-D + F + PROJECT_STATE

- 状态:规划 v10(吸收 v9 审查 2 P0 + 9 P1;全部判定写死)
- 核心预期(条件算术,非证据):若 B05 相对 B01/B03 的 paired video CI 半宽 ≈0.0115,点估计 +2.9~3.0pp → 下界 ≈+0.0175 > superiority margin +0.005 → **Gate C 有真实通过路径**;若实际 d 不足,输出"无决定→停止"(诚实)

## 正文步骤(同 v9)
Step 0 PROJECT_STATE → 1 联合 split → 2 OOF v3 → 3 重跑 → 4 A 统计 → 5 B N1 → 5.5 finalist CV → 6 D → 7 F → 8 C

# 附录 A:统计契约(v10 修订项加粗)

## A-1 比较家族与方向
- d = candidate − comparator;8 对 confirmatory;B05 vs B01/B03 为 Gate C primary(2 对),其余 6 对 secondary(输出不触发)
- **V 冻结(v9 P2)**:预注册共同 video manifest(V_common = 8 对共有的支持 video 集);每对报告 V_pair 与排除原因;横向比较用 V_common
- estimand 区分:Gate C 判定 **seed-mean route effect**(mean_s acc_{s,v});最终模型为 3-seed 概率 ensemble,**ensemble accuracy 单独报告,不参与 Gate C**(v9 P1-09:明确不混淆)

## A-2 Holm-TOST(伪代码级)
- **判定伪代码(冻结)**:按 TOST p 值升序做 Holm 步进(m=8);superiority 单侧检验 p_sup = P(d ≤ 0.005);调整后判定条件为**严格 >**(`CI_lower > 0.005`,非 ≥);CI 用 percentile 方法,bootstrap 5,000 次 seed 42,95% 描述性 CI 并存(标注非判定)
- 边界示例写死:半宽 0.0115 时,d 需 > 0.0165 才能 CI 下界 > 0.005(下界=0.005 恰好时不通过)

## A-3 head guardrail:role/family top1 非劣 **CI 下界 > −0.005**;NLL/ECE 方向+CI 描述

## A-4 Gate C 决策表(primary 2 对)
| 条件(全部) | 输出 |
|---|---|
| primary 2 对 superior(调整后 CI 下界>0.005,按 A-5 部分识别规则)+ guardrail 全过 + 3 seeds 方向一致 + hard subset 非劣 | 通过 |
| 任一 primary inferior 或 guardrail 失败 | 不通过 |
| 其余 | 无决定 → 停止序列扩展 |

## A-5 84 行部分识别(v9 P0-02:fail-closed)
- 主结论定义为**部分识别区间**:枚举缺失行 4 种 paired outcome(受已知 role 结果约束)的可行集;**Gate C 通过条件 = 所有可行补全下调整后 CI 下界均 > 0.005**;否则输出无决定
- complete-case(1,494)仅作声明性 estimand,不得用于全量 Gate C 结论;报告 84/1,578 coverage 与缺失行分布

## A-6 Req95 与 SLO(v9 P0-01:状态闭集)
- **terminal_status 闭集**:{completed, failed, timeout, oom, unadmitted, pending_end, cancelled, evicted, unknown};仿真终点固定(到达终点仍 pending → pending_end 视为失败);**每个到达 job 恰好一条 final 记录**;缺失/未知/admitted 未 completed → 默认失败(无删失规则)
- 失败规则:admitted 终态 ∈ {failed, timeout, oom, pending_end, cancelled, evicted, unknown} → episode 失败;admitted 比例阈值预注册 **0.90/0.95/1.00 三档敏感性**(16/32/64 jobs 允许 unadmitted 0/1/3 的离散性分层报告);critical path job unadmitted → episode 失败
- **critical 标记进 job_outcome schema**(版本化,由不可变 template/episode 生成,禁后验标记;DAG/分支/retry 语义冻结)
- deadline:completion ≤ deadline_ms(绝对);**deadline_miss 独立口径**:完成但超 deadline 计入 job 级 miss 率(≥0.05 → episode 失败);baseline 构造固定(不依赖结果,预注册)
- Req95_e = admitted completed job 的 (completion−arrival) P95(线性插值);completed==0 → 失败;SLO_e = 上述全过;总体 = ΣI(SLO_e)/N_e

## A-7 job_outcome 双表(job_outcome final 一行 + job_attempt 多行;时间不变量;SLO 只消费 final)

## A-8 cell(v9 P1-05/P1-06:estimand 精确化)
- **estimand = (pressure, arrival_pattern, initial_state) 联合生成分布**(实际非零组合 ≈ 5×3×3=45);topology/job count/deadline multiplier 为**派生条件**(共线 i%3),禁止独立主效应/因果解释
- θ = n_c/N(固定);45 cell 逐 cell 报告(22-23 episodes/cell)标 **descriptive**(禁无校正 cell headline);零 cell 报告缺失
- **bootstrap 衔接(v9 P1-06)**:按 cell 分层 + 同 episode 配对重采样,固定 θ;明确目标是**固定 workload estimand**(不是随机 workload)

## A-9 cluster 与发布边界(v9 P1-07)
- 主 CI:episode cluster bootstrap(同 episode×seed 先 seed 均值);**发布边界机器门禁**:主表/摘要强制 "fixed workload instance" 标签 + θ;禁一般策略优越性宣称;video/template cluster 仅敏感性(预注册)
- 若要一般调度结论 → 另设独立 workload mix 实验(不在本规划)

## A-10 seed(v9 P1-08:5-seed 敏感性)
- 模型:3 seeds(11/22/33);**另跑 2 seeds(44/55)作 5-seed 敏感性**(报告逐 seed、3-seed/5-seed ensemble、seed 方差;看结果前冻结)
- 仿真:3 seeds(101/202/303);policy 间同 episode/job 集合;per-seed endpoint → seed mean → paired diff → cell 分层 episode bootstrap

# 附录 B:工程契约(v10 增补)
- B-1 lineage(逐 job + ordinal/arrival/deadline digest)
- B-2 identity registry(quarantine + 覆盖率门槛)
- B-3 effective_split 逐条裁决表
- B-4 DatasetContext.verify_consumer(全 consumer + template/workload/resource 入链)
- B-5 r3 统一入口(config v3)
- B-6 OOF v3 四 block 主键元组
- B-7 finalist CV(checkpoint 冻结 epoch;fold manifest)
- B-8 资源表(F3 前置;unknown 阻塞)
- B-9 pilot 独立 gate(Step 7a)
- B-10 simulator(golden fixtures/determinism/旧入口禁导入)
- **B-11 job_outcome validator**(闭集枚举/唯一 final/时间不变量/critical 字段;负测覆盖未知状态与缺失记录)

# 实现期 DoD(同 v9 结构,增补)
- Step4 DoD 增:A-5 部分识别区间计算 + Gate C 按"所有可行补全下界"机器判定;A-6 状态闭集 validator 通过
- Step7 DoD 增:job_outcome 闭集校验;固定 workload 标签门禁;5-seed 敏感性报告

## 审查记录
v1(9)→v2(8)→v3(12)→v4(9)→v5(10)→v6(11)→v7(14)→v8(20)→v9(2 P0+9 P1)→v10 吸收;待复核
