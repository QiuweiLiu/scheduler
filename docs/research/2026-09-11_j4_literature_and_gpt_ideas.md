# J4 后续方向：网页版建议 + 文献检索记录（2026-09-11）

背景：J4（duration 解耦）负结果；缺口 = 可部署条件下 load-duration 预测修复。本文汇合 ChatGPT Web 的独立建议与本地文献检索。

## 1. 网页版建议（要点）

- **不补数据即可试**（按推荐度）：
  1. **条件化 hurdle**：把 duration 头显式读取 occurrence 的隐状态 `P(load>0|A,c)·P(duration|load>0,A,c)`（当前两者独立）。
  2. **model-class 层级化 / 专家混合**：`P(y|x)=Σ_m P(m|x)·P(y|x,m)`，用 q(model_class) 做门控——数据事实（VL-3B 占 63%、VL-8B 长尾误差大）支持该结构。
  3. **分布参数化**：log-normal / Gamma / Weibull survival / mixture density network 替代裸 pinball。
  4. **load>0 子集专用表示**：保留 occurrence→duration 信息流的 adapter（区别于 J4 的断开式解耦）。
  5. **校准/后处理**（低成本、上限有限）：quantile recalibration、isotonic、conformalized quantile regression（Romano et al., NeurIPS 2019）。
- **文献关键词**：serverless cold start prediction（SAND USENIX ATC'18、FaastLane ATC'19）、learned cost models（Neo VLDB'19、Kipf CIDR'19）、heavy-tail quantile regression（Koenker & Bassett 1978）、MTL negative transfer（PCGrad NeurIPS'20、GradNorm ICML'18）。
- **路线排序**：① 接受 trade-off + 改写叙事（推荐，★★★★★）；② model-conditioned hierarchical duration（★★★★）；③ 补采 workload/cold-start（收益最大成本最高，★★★）。
- 其明确判断：J4 负结果说明"问题不是共享容量不足，而是 duration 的强条件结构未被当前接口显式表达"。

## 2. 文献检索结果（本地 web 检索，均为可追溯论文）

### A. 与 hurdle / 零膨胀性能预测直接对应（最贴题）
- **From Wait Time to Slowdown: A Hurdle Framework for Performance-Centric Metric Prediction in HPC Clusters**（Dakkak, Black Sea Journal of Engineering and Science, 2026, doi:10.34248/bsengineering.1963790）：HPC 作业 wait-time/slowdown 预测；单阶段回归在零膨胀目标上结构性崩溃（R²<0），hurdle 两阶段（分类器 + 仅正样本 LightGBM + log1p）显著更优；且"零膨胀消失时简单模型更好"。→ **直接支持我们的 hurdle 设计，并提出条件耦合/阈值选择改进**。
- 统计侧：Mullahy (1986) 静态 hurdle、Baetschmann & Winkelmann 动态 hurdle（zero-inflated count）。

### B. Serverless 冷启动 / 模型加载耗时预测（问题形态最像）
- MASTER: Machine Learning-Based Cold Start Latency Prediction Framework in Serverless Edge Computing（IEEE J. Sel. Areas Sensors 2024）。
- SPFaaS: A Sparse Function Prediction Approach for Cold Start Optimization（IEEE TPDS 2025）：稀疏调用预测 + GRU/TCN。
- Serverless Cold Starts and Where to Find Them（Joosen et al., 2025, doi:10.1145/3689031.3696073）。
- HydraServe（arXiv 2502.15524）：冷启动分阶段（容器创建/权重拉取/CUDA 初始化），可重叠；LLM 冷启动可达 40s。
- Chameleon（MICRO'25）：adapter 加载耗时达 30ms、强重尾；缓存命中可消除。
- 结论：load 时长 = **多阶段、多模态、重尾**，且强依赖**常驻/缓存状态**（我们缺的正是这个变量）。

### C. Learned cost model / 查询优化（方法论迁移）
- Neo: A Learned Query Optimizer（VLDB 2019）；Learned Cardinalities（CIDR 2019）。
- 迁移点：层级化成本、不确定性、workload-conditioned 预测。

### D. LLM-Agent 工作流调度（应用场景同类）
- Astraea（arXiv 2512.14142）：状态感知分层调度；service time predictor（I/O 与 compute 分类）。
- Murakkab（arXiv 2508.18298）：agentic workflow 编排，含 video QA；profile-guided。
- Chimera（arXiv 2603.22206）：异构集群 + 剩余输出长度预测。
- LLMSched（arXiv 2504.03444）：不确定性感知 + Bayesian network 时长估计。
- Pythia（arXiv 2604.25899）：工作流可预测性利用。
- **Latency-Aware Orchestration for Multi-Agent LLM Workflows on Heterogeneous GPUs（arXiv 2609.03335）**：predictor 显式估计"activation latency、peak memory、**model-loading cost**"——与我们最接近的系统工作，但其依赖运行时池状态/物理执行图，而非纯执行前预测。
- 说明：该领域普遍用"profile + 结构化模型 + 部分 oracle"；**纯执行前、可部署的 load-duration 概率预测仍是空缺**。

### E. 多任务负迁移 / 任务分组（解释 J3/J4）
- Multi-Task Learning 1997–2024（Harvard Data Science Review 2025 三部曲）；Deep MTL review（2025）。
- MTLinear（AISTATS 2025）：按相关性聚类变体来分组任务并做梯度平衡；RMT 分析指出噪声项导致负迁移。
- 与 J4 一致：runtime（连续大质量）与 duration（稀疏条件）任务相关性弱 → 过度硬共享收益有限；但也说明"完全拆开"丢失共享信息。

## 3. 汇合判断（本地）

1. 文献支持"**结构化建模优于裸多任务回归**"：hurdle 论文（A）与冷启动分阶段证据（B）都指向 duration 是多阶段/条件重尾分布，而非单峰回归目标。
2. 无需新数据、成本最低的两条改进：**条件化 hurdle 耦合**（网页版 1）与 **model-class 条件化/专家门控**（网页版 2）——两者都能直接接入现有 q(A) 接口，且不破坏 J 系列协议。
3. 分布参数化（Weibull/log-normal/MDN）是第三条技术线；校准后处理只能改善区间不能补信息，优先级最低。
4. 真正确认"能修好"的关键变量（model residency/cold-start、workload 量级）确实缺失（B/C 两族文献都依赖系统侧状态）——若要彻底解决需新采集或系统改造。
5. 论文叙事选项（网页版方案 1）与继续修复（方案 2/3）不冲突：可先做方案 2 的小成本实验，若收益显著则继续，否则以 trade-off 叙事收尾。

## 4. 建议的下一步（待用户选择）

- **J5a（低成本，优先）**：条件化 hurdle + model-class 门控 duration 头（网页版 1+2 合体），validation 判定、~15 GPU-min。
- **J5b**：分布参数化 duration（Weibull/log-normal，hurdle 同上）。
- **J5c**：仅校准/后处理（conformal/isotonic）。
- **停止/叙事**：以 J3 主结果 + J4 负结果 + 权衡边界收束联合训练线。
