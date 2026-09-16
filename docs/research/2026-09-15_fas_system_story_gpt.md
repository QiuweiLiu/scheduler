# FAS 系统论文主线（GPT 重分析，2026-09-15）

来源：ChatGPT Web 会话 `https://chatgpt.com/c/6a9822da-e278-83e9-9c1a-675923acda0e`
（输入：系统全貌简报 = 预测器 + 接口 + 调度器 + 全部实验；输出：系统论文定位/创新点重排/最小补充实验/投稿档次）。
说明：引用的论文（Parrot/OSDI 2024、SOLA/MLSys 2025、SuperServe/NSDI 2025、Katz/ATC 2025、Vidur/MLSys 2024、FATE 2026）
**尚未逐篇核验**，进论文前需核对。

---

## 5. assistant
我会把它重构成一篇“系统发现 + 系统机制 + 端到端实现”的论文，而不是“预测器论文 + 调度器论文”的拼接。

主线标题级故事可以是：
Predicting What Agents Will Do Next: Future-Action-Aware Scheduling for Multi-Model Video Agents
核心问题不是“能不能预测 runtime”，而是：

当前节点级 scheduler 看不到 Agent 请求未来会继续多久、调用哪些模型，因此在动态多模型 workflow 上做局部决策；我们建立 Future-Action Interface，预测未来 workflow evolution，并研究什么 future information 真正具有 scheduling value，以及怎样消费它。

三个贡献应重新排序：

C1 主贡献：Future-action-aware scheduling system。 预测“请求接下来会不会继续、还能执行多少节点及其未来执行风险”，将 workflow future 作为一等调度状态。
C2 核心机制贡献：不是越精细越好。 oracle factorial 表明 scheduling value 主要来自 continuation/horizon；精确 node identity 和 point resource 并非主要瓶颈，而 runtime-tail future cost 才是有效消费形式。
C3 supporting：完整 forecast→artifact→scheduler→oracle/ablation 评测管线。 J 系列预测器本身应该作为系统组件，而不是单独声称 SOTA predictor。

与近两年工作的边界：信息层差异基本成立，但必须准确表述

最危险的 prior 是 Parrot, “Efficient Serving of LLM-based Applications with Semantic Variable”, OSDI 2024。它已经明确指出 request-level API 丢失 application-level information，并把已知的 LLM application dataflow/DAG 暴露给 serving system，用于 DAG-aware scheduling。
USENIX
+1

所以不能写“首个利用 agent workflow 的 scheduler”。你们的区别是：

Parrot 要求/暴露已知 application graph；你们面对的是运行时由 agent 决策动态展开、未来尚未显式存在的 workflow，因此预测 future action sequence。

这是目前最重要的新颖性边界。

SOLA, “Optimizing SLO Attainment for Large Language Model Serving with State-Aware Scheduling”, MLSys 2025 根据单请求当前状态和系统状态进行细粒度 scheduling；它说明 state-aware scheduling 已经很成熟，但并不预测下游 agent workflow evolution。
MLSys Proceedings

SuperServe, “Fine-Grained Inference Serving for Unpredictable Workloads”, NSDI 2025 的 SlackFit 根据 latency slack 动态选择可执行模型，属于 reactive/state-aware scheduling；仍不是预测未来多阶段 action chain。
USENIX

Katz, “Efficient Workflow Serving for Diffusion Models with Many Adapters”, USENIX ATC 2025 已经是真正的 heterogeneous workflow serving：ControlNet、LoRA、base model 有不同 compute/loading 特性，并利用已知 workflow 做并行、缓存和异步加载。它和你们负载形态很接近，但 workflow structure 是应用定义的，而不是 agent 在线决策产生、需要预测。
USENIX

Vidur, MLSys 2024 则直接削弱你们“仿真框架本身”的创新性：它已经做 profiling/predictive modeling 驱动的高保真 LLM inference simulator，并用真实 GPU 验证误差。
MLSys Proceedings
 因此 C3 只能 supporting，除非你们也证明 simulator fidelity。

所以建议 Related Work 中明确一句：

Prior systems exploit known workflow semantics, current execution state, or request-level workload predictions; our target is the latent future control flow of dynamically unfolding agents.

这句话目前是站得住的，但投稿前还应继续检索 2026 已接收论文；像 FATE 这种 2026 preprint 已经非常接近 future-state scheduling，但当前搜索到的是 arXiv，不应拿它当已同行评审 prior。
arXiv

让“完整系统论文”成立，最少补 3 件事

P0：真实 GPU replay / 小 prototype。 这是当前最大短板。你现在的数字证明的是 simulator 上 scheduling policy 有效，而不是实际 GPU system 有效。最低配置不需要大集群：2 GPUs、几十/几百条 replay workload，比较 E0、E2、q95、oracle/subset，验证 completion/JCT 排序与 simulator 一致。最好同时报告 simulation-vs-real latency error。Vidur 都专门做了这一层验证。
MLSys Proceedings

验收： q95 相对 current-node baseline 在真实 replay 仍有明确正收益；仿真对策略排序不反转。这个实验对论文档次的提升最大。

P1：证明收益不是这一个 workload pressure 点的偶然。 不需要新数据，用现有 600-video pool 改 arrival rate、deadline multiplier、GPU contention、模型 mix，至少形成 low/medium/high load × tight/normal deadline 的 6 个 cell。主比较只跑 E0/E2/q95/oracle。
验收： future-aware 在多数 cell（建议 ≥5/6）不劣，q95 在至少 4/6 显著优于普通 future-cost；不能只在当前压力参数有效。

P1：forecast-quality sensitivity。 单 predictor 会被质疑“系统是不是绑死 J3”。不一定要再训练模型：对 frozen artifact 人工控制 length error、runtime quantile calibration、content corruption，画 scheduling gain vs forecast quality 曲线。再加 oracle endpoints。
验收： 存在连续、可解释的 degradation，而不是某个 checkpoint 偶然命中 scheduler。

H10-lite 我反而放 P2。它会增强故事，但不是当前论文成立的必要条件。已有 oracle H5→H10 已足够说明 future horizon 有未利用价值；朴素 H10 失败可以诚实作为系统边界。除非 H10-lite 能至少保住 H5 近端质量并明显关闭 oracle gap，否则不要让论文被它拖住。

你现有的主证据可以直接使用：E2 vs E0 证明 future interface 有价值；oracle topology/resource factorial 证明价值来自哪里；H5/H10/unbounded oracle 定位 horizon 上限；16 个消费者 + frozen 300 confirm 证明 q95-tail consumption 的决策价值；所有负结果则构成 unusually strong mechanism section，而不是附件垃圾。

档次判断

维持“纯 CPU 仿真 + 单 workload family”：我不会主攻 OSDI/SOSP/NSDI。系统 claim 太强，而 physical-system evidence 太弱。更现实是 ICPP / CCGrid / Cluster；如果写得完整、补大量 sensitivity 和算法分析，TPDS/TCC 也合理。

补 2-GPU 或小规模真实 prototype/replay + simulator fidelity + workload-pressure sweep 后，我认为可以认真冲 MLSys；USENIX ATC / EuroSys 也进入“值得尝试”区间，尤其如果系统代码是真实 runnable 而不是只 replay trace。Parrot、Llumnix、SuperServe、Katz 这些近作都以真实系统实现和实际硬件实验支撑系统结论。
USENIX
+3
USENIX
+3
USENIX
+3

要到 OSDI/NSDI/SOSP，通常还需要再跨一层：多 workload / 多模型栈、更真实的并发部署规模，以及一个比“q95-sum 实测最好”更强的调度算法或理论解释。

所以我现在会把项目定位为：

一个预测动态 Agent future、量化 future information value，并把预测尾部风险转化为 GPU scheduling decision 的完整系统。

真正决定它从“不错的仿真论文”跃迁到“MLSys/系统主会候选”的，不是再把预测准确率提高 2%，而是真实硬件闭环验证 + 跨压力区间稳定性。
