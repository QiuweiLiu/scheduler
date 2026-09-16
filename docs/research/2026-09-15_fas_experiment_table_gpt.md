# FAS 补充实验总表 + 工业场景评估（GPT，2026-09-15）

来源：ChatGPT Web 会话（第三轮追问：实验表格 / 基线 / 真实 replay / 电梯检测 agent 私有数据是否加分）。
引用的论文与系统（Parrot、SOLA、SuperServe、Katz 等）仍需逐篇核验。

---

## 5. assistant
A. 投稿前补充实验总表
实验	目的（审稿质疑）	设计/臂	基线	指标	通过判据	成本与依赖	优先级
真实 GPU replay / 小 prototype	“仿真收益是否是真实系统 artifact？”	replay 单位应是 episode：保持多 workflow 到达序列、依赖、deadline、scheduler decision point；每个 workflow 是 episode 内 job。优先远端 2×同型号 ≥24GB GPU，固定 3–5 个代表模型组成可完整运行的 stack；100–200 episodes，E0/E2/q95/Oracle-H5，随机策略执行顺序并重复 3 次	E0、E2、q95、Oracle	workflow completion/JCT、miss、queue、GPU util；sim↔real 相对误差和策略排序	q95 对 E0/E2 在真机仍显著改善；策略排序不反转；sim 对主要指标误差最好 <10–15%	2 GPU 足以证明“调度闭环有效”，无需集群。1×3060 6GB 只能做节点 latency/replay fidelity，不能证明多 GPU scheduler；若只能单卡，可缩模型验证 simulator，但不能替代 P0	P0
投稿基线矩阵	“是否只打赢弱 baseline？”	仿真统一跑：FCFS；SJF/LTR-style remaining-work；E0；E2；q95；STS-style termination-aware；SOLA/SlackFit-style current-state；Parrot-like known-DAG；Oracle-H5/H10	同左	completion、p95 JCT、miss、queue	q95 至少显著优于 FCFS/SJF/E0/E2/state-aware；known-DAG/oracle 保持合理上界	约 8–9 臂，现有 simulator 数小时内；应称 adapted baseline，不声称复现原系统	P0/P1
6-cell pressure sweep	“只在一个压力点有效？”	arrival={0.7,1.0,1.3}×base × deadline={tight,loose}；每 cell 跑 E0/E2/q95/Oracle-H5	E0/E2	completion、miss、gain	q95 对 E0 ≥5/6 cells 不劣且明显获益；对 E2 ≥4/6 有正收益；高压力区收益应不消失	24 arms≈4 GPU/CPU 仿真小时	P1
Forecast-quality sensitivity	“单 checkpoint 偶然命中 scheduler？”	length 分布分别混入 10/25/50% uniform/noise；runtime-tail 分别 shrink q95→p50 25/50/75%；content/model identity 随机替换 10/25/50%	原 q95	scheduling gain vs corruption	length/tail 信息破坏呈剂量-反应式退化；content corruption 若影响小反而支持你们机制	约 12–20 arms，2–4h	P1
q95 机制：相关尾风险	“q95-sum 是否只是调参冠军/长度惩罚？”	matched length 下比较 p50/p90/p95；再构造 aggregate-tail：ρ=0/.5/1 的相关性近似，或控制节点 tail 共振强度	length-only、CVaR	completion/miss；收益随压力/ρ变化	高 contention/deadline pressure 下保守 tail 的优势系统增强；若与ρ无关，则只能称 heuristic	需少量消费者代码，<1天	P1+
接口 factorial	“到底消费了什么？”	matched：length-only / runtime-tail-only / length+tail / content+length+tail	E0	completion、miss	length+tail ≥ 单独两者；content 不应成为隐藏混淆	大部分已有结果可复用	已有/补齐
H10-lite	“既然 H10 oracle 更好，为何系统只 H5？”	H5 backbone/near head 保持，6–10步 coarse auxiliary head；3 seeds	frozen H5	前5步质量、H10调度收益	前5步 NI≤2%；至少关闭 oracle H5→H10 gap 的 ~30%	重新训练，3060 可做；风险最高	P2

这里要注意基线语义：Parrot 确实通过 Semantic Variables 暴露 application-level dataflow，因此“known-DAG oracle”是合理对照，而不是声称复现 Parrot；SOLA 是 request/system current-state aware；SuperServe 面向不可预测 workload 做 reactive model selection；Katz 则针对已知 diffusion workflow 做缓存/并行/异步加载。
USENIX
+3
USENIX
+3
MLSys Proceedings
+3

三档实验集我建议这样冻结：

最小可发表集：现有主实验 + FCFS/SJF/state-aware/termination-aware 基线 + 6-cell pressure sweep + forecast-quality sensitivity。适合 ICPP/CCGrid/Cluster/FGCS/JPDC 路线。

推荐集：最小集 + 2-GPU真实 replay + sim-real fidelity + q95相关尾风险机制。这个组合最可能把工作推到 MLSys/ATC/EuroSys 可认真尝试的状态。

完整集：推荐集 + H10-lite + 第二 workload（最好就是电梯 Agent）+ 最终冻结后一次性 T_final。这是最完整的论文版本。

一个重要执行细节：真机不要追求完整原模型栈。 两张 24GB GPU 上固定一个代表性 stack 更科学，例如 planner/LLM + VLM + detector/segmenter + CPU tool，保留“模型切换、异构节点、并发和 dependency”即可。模型集合在实验前冻结。若远端只有 1 GPU，则做 sim-real latency calibration 有价值，但无法替代多 GPU scheduling evidence。

B. 电梯检测 Agent：建议做，但定位为第二 workload / industrial case study

它能明显加分，前提是它真的构成不同 workload distribution。最大的价值不是“工业故事更漂亮”，而是解决你们当前最难防的质疑：所有结论是否只来自一套 VideoQA workflow family。

如果电梯 Agent 的分支触发机制、模型组合、链长分布、deadline/SLO、Repair 分支概率都和现有 workload 不同，那么它可以同时增强三点：external validity；真实业务约束；对“future-action information 是否跨 agent 类型有效”的验证。尤其如果仍观察到“continuation/length + tail-risk 消费”有效，这比再做几十个同域 simulator ablation 更有说服力。

减分也很明确。私有视频本身无法公开，审稿人无法完整复现；如果只展示几个案例，会被视为 demo；若 predictor 或 scheduler 专门为电梯调参，则反而削弱“通用接口”的主张。

让它净加分的做法是：不把私有数据作为主 benchmark，而是作为 Case Study / External Workload。冻结现有 Future-Action 接口和 q95 consumer，不针对电梯重新设计 scheduler；只允许重新生成该 workload 的预测 artifacts。论文公开：匿名化 DAG/action trace、模型类别、节点 runtime/load trace、arrival/deadline 分布、branch/repair 统计、schema 与 replay driver；不能公开视频就提供脱敏 trace + synthetic trace generator。这样别人至少可以复现实验调度层。

最小工业实验我建议只做四臂：E0 / E2 / q95 / known-future oracle，在真实到达 trace replay 上报告 completion、miss、queue；若有两张 GPU，直接真机跑。如果只能仿真，它仍是有价值的跨 workload 验证，但加分明显小于真实 replay。

我的投入排序非常明确：

第一优先仍然是 2-GPU replay。 它补的是当前论文最致命的 internal/system validity 缺口。

第二优先才是电梯 Agent。 最理想不是二选一，而是让电梯 Agent 直接成为 2-GPU replay 的第二 workload：主 Video Agent 做完整基准，电梯 Agent 做较小规模 external case study。这样一次工程投入同时解决“无真机”和“单 workload”两个问题，证据增量最大。

如果资源只够做一个，我选 2-GPU真实 replay；如果再多出约 30–50% 工程时间，就把电梯 Agent 接入同一 Future-Action contract，跑一个不调参的外部 case study。这样最划算。
