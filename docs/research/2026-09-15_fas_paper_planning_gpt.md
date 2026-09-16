# FAS 论文规划（GPT，2026-09-15）

来源：ChatGPT Web 会话 `https://chatgpt.com/c/6a9822da-e278-83e9-9c1a-675923acda0e`（Phase 6/7 结果 + 创新点查新与补实验清单）。
说明：文中引用的论文（Parrot/OSDI 2024、Pythia 2026、PBKV 2026、FATE 2026、SOLA/MLSys 2025、Vidur/MLSys 2024、
DistServe/Sarathi-Serve/OSDI 2024、Llumnix、ServerlessLLM）**尚未逐篇核验**，进入论文前需核对标题/会议/年份。

---

## 5. assistant
结论先说：你们现在最有潜力的主干不是“预测器本身”，而是 Agent future 的哪些信息真正值得预测、以及调度器怎样消费它们。按创新强度我会排：② 风险消费机制 > ① 信息价值分解 > ③ 多尺度 horizon > ④ 评测框架。

创新点 1：未来工作流“长度/终止”是主导信号，而精细内容/单节点资源并非主导。新颖度：中等偏高，适合主贡献之一。 最近最接近的是 Parrot（Parrot: Efficient Serving of LLM-based Applications with Semantic Variable, OSDI 2024），它证明暴露 application-level workflow/DAG information 能显著改善端到端 serving；Pythia（Pythia: Toward Predictability-Driven Agent-Native LLM Serving, arXiv 2026）进一步把 agent workflow 的 predictability 当作系统优化资源；PBKV（Efficient Serving for Dynamic Agent Workflows with Prediction-based KV-Cache Management, arXiv 2026）直接预测未来 agent invocations 来管理 KV cache。换言之，“未来 workflow 有价值”已经不新。
USENIX
+2
arXiv
+2

你们较新的点是：通过 oracle factorial decomposition 发现 continuation/horizon 比 identity/resource accuracy 更重要，甚至更精确的 identity 会负优化。现有 Parrot/PBKV 并未给出这种“信息价值排序”。要立住，必须补：① length-only / content-only / resource-only / full 的 matched factorial；② 至少在 star/langgraph、不同 deadline/load cell 上复现；③ oracle 与 predicted 两套都做。支撑门槛建议：length signal 在 ≥3/4 workload cells 中相对 no-future 改善 ≥5%，paired CI 不跨 0；oracle resource 相对 table 的增益稳定 <2%；content identity 加入不能稳定优于 length-only。反之若换 workload 后 resource/content 成为主要收益，此点只能降级为“本 workload characterization”。

创新点 2：逐步 q95 相加这种“完全相关式尾部代价”比 CVaR/场景平均/自适应风险更有效。新颖度：潜力最高，但目前只是观察，不是算法贡献。 DistServe（DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving, OSDI 2024）把 tail-latency SLO 直接变成 goodput 优化；Sarathi-Serve（Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve, OSDI 2024）围绕 tail latency 设计 stall-free batching；SOLA（SOLA: Optimizing SLO Attainment for Large Language Model Serving with State-Aware Scheduling, MLSys 2025）根据 request/system state 动态调整调度。它们都说明“风险/SLO-aware consumption”本身不新。
USENIX
+2
USENIX
+2

真正可能新的，是把你们的 q95-sum 形式化成一个 correlated-tail robust surrogate：

C
robust
	​

=
h
∑
	​

Q
.95
	​

(T
h
	​

)

并解释为何在 agent 链中，保留“共振式最坏情况”比独立场景采样后方差收缩更适合排序。必须补两类实验：一是 workload stress sweep（deadline、offered load、chain-length distribution）；二是合成相关性实验，人为控制未来节点 runtime correlation ρ，比较 p50 / q95-sum / scenario-CVaR。若 q95-sum 在高相关或高 deadline-pressure 区域系统领先，并且随着 ρ 增大优势增大，这就从“经验冠军”升级成机制贡献。若跨 stress 后只在当前 cell 赢，则只能当 engineering heuristic。

创新点 3：oracle H10 有价值，但朴素 H10 predictor 破坏近端质量；需要 multi-resolution horizon prediction。新颖度：中等，当前还不是贡献。 “长 horizon 帮助、长预测更难”本身是标准 forecasting 问题；FATE（FATE: Future-State-Aware Scheduling for Heterogeneous LLM Workflows, arXiv 2026）已经使用 horizon-aware planning，而 PBKV 也直接预测多个 future steps。
arXiv
+1

要把它变成创新，必须让 H10-lite 成功：保留 H5 backbone/near-term heads，另加 6–10 步 coarse/auxiliary head，避免共享 slot capacity 稀释。验收建议：前 5 步 runtime/content 指标相对 H5 NI ≤2%；6–10 步长度/终止明显优于 trivial continuation baseline；调度在 300 confirm 上较 H5 q95 改善 ≥3%，CI<0，并至少关闭 30% 的 oracle H5→H10 gap。若做不到，这一点应写成 negative finding，不当贡献。

创新点 4：forecast-aware evaluation methodology。新颖度：低，适合作为 supporting contribution。 Vidur（VIDUR: A Large-Scale Simulation Framework for LLM Inference, MLSys 2024）已经建立 profiling + predictive modeling + simulator-based configuration search，并验证 simulator fidelity；因此“仿真+预测+调度评测”本身不新。
MLSys Proceedings
 你们只有在公开/完整提供 workload contract、oracle/predicted factorial、clustered bootstrap、dev/confirm discipline，并证明 simulator replay/fidelity 时，才能叫 benchmark/methodology contribution；否则应放 artifact/reproducibility。

论文最强结构我建议是三条：C1 信息价值分解——future continuation dominates fine-grained resource/content；C2 Robust Future Cost——提出并机制化解释 q95-sum 作为 correlated-tail surrogate；C3 Multi-resolution horizon——如果 H10-lite 成功，再加入，否则作为 limitation/negative result。

最值得补的实验依次是：① q95-sum 的 correlation × load/deadline stress mechanism sweep；② 信息价值 factorial 在不同 workflow/pressure cell 上复现；③ H10-lite。前两个完全可用 CPU 仿真，成本大约几十个臂 × 10 分钟，1–2 天可完成；第三个需要一次预测器新立项和 3 seeds。

投稿档次方面，如果维持“单套 predictor + CPU simulator + 单 workload family”，OSDI/SOSP/NSDI/EuroSys 主会很难，这些近年相关论文通常有真实 distributed serving prototype 与多硬件/多 workload，例如 Parrot、Llumnix、ServerlessLLM 都做了完整系统实现与真实集群评测。
USENIX
+2
USENIX
+2

如果补齐多 workload stress、机制实验、H10-lite，并把算法写清楚，ICPP / CCGrid / Cluster / TPDS/TCC 更现实；其中 TPDS/TCC 需要更完整的算法分析和大量仿真，ICPP/CCGrid 对当前形态最匹配。MLSys main 有机会，但最好至少增加一个真实 GPU replay/prototype 小实验或高保真验证；仅 CPU 仿真会是明显短板。SC 除非扩到大规模 heterogeneous GPU workflow，否则主题匹配度一般。

所以我会把接下来资源集中在 “q95-sum 为什么有效” 上。若能证明它不是偶然调参冠军，而是由 agent future 的相关尾风险导致，并跨 workload/load/deadline 稳定成立，这会比再提升 2–3 个百分点 predictor accuracy 更像一篇系统论文的核心创新。
