# VideoSeek 预测与节点级调度实验阶段总结

> 状态日期：2026-08-12  
> 本地文档：`/Users/liuqiuwei/Documents/调度/docs/videoseek_experiment_summary_20260812.md`  
> 主要实验环境：`root@connect.westc.seetacloud.com:12469`  
> 远端项目：`/root/autodl-tmp/scheduler`  
> 文档性质：对截至当前已经执行的实验、正式结果、历史结果和未完成事项进行统一归档；不把下一阶段规划写成已完成结果。

## 1. 项目现在要解决的问题

本项目当前主线是研究视频 Agent 在执行路径尚未完全确定时，能否利用已经执行的前缀预测有限未来，并据此进行节点级 GPU 调度：

```text
Agent 已执行 prefix
        ↓
行为预测器：未来可能执行什么
        ↓
资源预测器：这些候选节点需要多少时间、显存和加载代价
        ↓
有限前瞻场景
        ↓
调度器选择当前 node × GPU
```

项目已经完成了 trace 采集、行为预测、资源预测、正式 workload 和基础调度门禁。尚未完成的是：把有限前瞻真正接入数学优化器和 RL，并验证其相对无前瞻调度的收益。

## 2. 当前完成度总览

| 模块 | 已完成内容 | 当前状态 |
|---|---|---|
| Agent baseline | ST fixed、STAR、LangGraph/ReAct 外层兼容复现 | 已完成 |
| 动态 trace | 本地 Qwen、API planner、Qwen VLM、YOLO11x/YOLO26n 介入 | 已完成 |
| 正式调度 collection | 768 条 trace、9,366 个可调度节点 | 已冻结 |
| DVD 扩展 | 本地兼容 adapter 轨迹 | 117 条成功，独立于主线，未达到128条目标 |
| 细粒度预测器 | Markov、RAG、AutoTool/Pythia-lite 适配、v0.4 | 已完成，作为历史/对照 |
| 粗粒度行为预测器 | XGB、MLP、FT-Transformer、GRU、LSTM、Causal Transformer、残差模型 | 已完成正式未见视频评估 |
| 多步/多模态预测 | H=1…5、图约束、GRU/LSTM/GNN、InternVideo、公共事件预训练 | 已运行，未稳定超过旧 v0.4 |
| 资源预测器 | runtime/load/peak-memory 分位预测与正式契约 | 已完成，但 stall/OOM/queue 仍不支持 |
| Workload v0.2 | 20k/2k/6,750 episodes，135 个测试 cell | 已完成并审计通过 |
| 基础调度门禁 | Round Robin、Myopic、Oracle | 已完成，Oracle gap 充分 |
| 有限前瞻调度 | PEFT-H、数学优化器-H、RL-H 公平比较 | 方法定义已收敛，尚未实现和运行 |
| 真实 contention | 单卡/多卡真实并发校准 | 尚未完成 |
| 电梯 Agent 迁移 | 角色映射已设计 | 尚无真实电梯 trace 验证 |

## 3. 第一阶段：统一 trace 与三类 Video Agent baseline

### 3.1 做了什么

固定 VideoTool 上游控制流，在不修改第三方业务逻辑的前提下，用外层 adapter 运行三类 baseline：

1. `st_fixed`：数据集专用的固定工具链，作为固定流程对照；
2. `star`：planner 在 temporal/spatial 工具间循环决策，控制器保留交替约束和最大步数；
3. `langgraph_react`：模型读取工具结果后继续发起标准 tool call，直到回答或触发递归上限。

同时建立统一 trace 记录和校验：

- 保存模型原始输出、解析后的决定、实际执行工具和控制器修正；
- 区分 `api_wait_ms`、`local_runtime_ms`、`load_ms` 和视觉推理时间；
- 保存 GPU 显存快照、模型 ID、视频和代码来源；
- 保存错误、解析失败、重试和 `retry_of`；
- API key 不写入 trace、manifest 或日志；
- 空答案不再被误判为成功。

### 3.2 模型和工具替代

官方 VideoTool/STAR 依赖 Grounded-Video-LLM、LLaVA 等重量组件，当前实验没有声称完整模型级复现，而是保留控制流和工具语义，使用已有模型替代：

- Qwen3-VL-8B：planner、视觉问答或答案生成；
- Qwen3-4B：拆分栈文本 planner；
- Qwen2.5-VL-3B：拆分栈视觉和答案；
- YOLO11x：重型检测和 batch 资源干预；
- YOLO26n：轻型检测对照；
- CPU：抽帧、网格、裁剪和元数据处理。

因此这里复现的是“Agent 控制流和动态 trace 生成方式”，不是原论文的准确率复现。

## 4. 第二阶段：动态性和资源异构性实验

### 4.1 32 视频、三 baseline 的本地 Qwen cohort

在32个 Video-MME 视频上，对每个视频运行三种 baseline，共得到96条轨迹：

- 96/96 运行成功；
- 96/96 通过 trace validator；
- planner、视觉和答案由本地 Qwen3-VL-8B 串行服务；
- YOLO 实际介入并记录资源字段。

这一批支持了路径动态性和资源异构性的初步假设，但当时的 Phase 4 简化模拟中 Oracle gap 为0，说明旧 workload 没有形成足够竞争，不能据此判断预测式调度无效。

### 4.2 YOLO11x batch 干预

在8个视频、STAR/ReAct 两条动态 baseline 上测试 YOLO batch：

```text
batch = 1 / 2 / 4 / 8 / 16 / 32 / 64
```

共得到112条 pilot：

- 112/112 success；
- 112/112 validator valid；
- YOLO allocated memory 从约372.7MB增长到约5,474.4MB；
- batch 1到64的路径变化比例由68.75%升到100%。

这个实验说明资源配置不仅改变显存和时间，也可能通过视觉观测结果改变 planner 后续路径；它是资源干预层，不作为独立任务样本重复计入 core collection。

## 5. 正式 trace collection

### 5.1 主线768条

正式主线最终冻结为：

| 组成 | 数量 | 说明 |
|---|---:|---|
| Stack A | 384 | Qwen3-VL-8B + YOLO11x，含动态和固定 baseline |
| Stack B | 256 | Qwen3-4B planner + Qwen2.5-VL-3B + YOLO26n，STAR/ReAct |
| Core 合计 | 640 | 64视频、128任务的核心轨迹 |
| Resource layer | 128 | YOLO batch、解析/恢复和资源条件轨迹 |
| 正式主线合计 | **768** | 冻结 collection index |

对应资产：

- 64个 Video-MME 视频；
- 视频级48/8/8划分；
- 768个 job template；
- 9,366个 measured nodes；
- template split：566/94/108；
- 768/768 run success；
- 初始解析错误177次，retry成功144次，retry后仍错误33次；
- 严格 OOM 样本0；
- 正 queue/wait 样本0。

正式 collection：

```text
/root/autodl-tmp/scheduler/results/processed/
  collection_index_core_resource_fixed_20260804.jsonl
```

需要注意：128条 resource layer 满足数量和资源差异门槛，但没有产生真实 OOM 和等待，因此不能称为已经完成真实 contention 数据采集。

### 5.2 Grounded-VideoLLM

Grounded-VideoLLM 的权重组件存在，但真实 import/依赖兼容门禁未通过。因此没有用低质量替代强行补32条，正式主线保持768而不是800。

### 5.3 DVD 独立扩展

Microsoft Deep Video Discovery 的官方代码和工具思路已检查，并实现了 `dvd_local` 兼容 adapter。当前可审计快照为：

- 117条成功 trace；
- 60个视频；
- 756个 prefix 样本；
- 固定48/8/8预期未满足，`fixed_split_gate=false`；
- 没有并入768条主线 collection。

它只能写成“DVD 方法兼容重实现/独立扩展”，不能写成完整官方 baseline 复现，也不能按原计划写成已经完成128条。

## 6. 早期细粒度预测器实验

### 6.1 预测对象

早期预测器直接预测下一工具或下一细粒度活动，并尝试预测剩余路径长度。复现/适配了：

- Markov-1、Markov-2；
- AutoTool next-tool adapter；
- RAG next-activity adapter；
- Pythia-lite path-length adapter；
- 结构化 prefix predictor；
- 图约束多步预测。

这些实现大多是对论文预测思想的轻量适配，不等于按官方端到端系统复现。

### 6.2 v0.4结果

在768条正式 collection 派生的锁定划分上，v0.4曾达到：

- fine-grained Top-1/Top-3：0.8157/0.9803；
- 非终止子集：0.7891/0.9791；
- coarse routing Top-1/Top-3：0.9589/0.9946。

该结果说明粗粒度资源路由更容易预测，但细粒度下一工具在早期 planner 高熵位置仍难以确定。这也是后续从具体节点转向跨 Agent 角色/动作族表示的原因。

### 6.3 修复过的问题

这部分实验发现并修复了多类可信度问题：

- 空 prefix 的全局计数污染；
- 使用随意长度分桶导致信息不足；
- 路径 coverage 曾错误使用近似恒真的 prefix+target 口径；
- target/current-action 语义混淆；
- baseline 类型可能成为捷径特征；
- 不能把同视频的重复事件当成独立样本做统计。

修复后保留旧结果作为历史，不再用被废弃数字做 headline。

## 7. 300视频粗粒度行为预测实验

### 7.1 数据规模和标签

后来将视频库扩展到300个有效 Video-MME 视频，并把预测目标改成跨场景粗粒度角色和动作族。

| Split | 视频 | Run | Role事件 |
|---|---:|---:|---:|
| train | 240 | 988 | 13,754 |
| validation | 30 | 144 | 2,029 |
| test | 30 | 108 | 1,520 |
| 总计 | **300** | **1,240** | **17,303** |

角色 schema 为：

```text
init / plan / execute / verify / aggregate / terminate
```

VideoMME 当前实际评价 `plan / execute / aggregate / terminate`；`verify` 没有真实样本，只为未来电梯 Agent 预留，不能把零支持类别混入宏平均制造结果。

execute 下再预测动作族：

```text
select_frames / visual_qa / temporal_ops / summarize / detect / other
```

### 7.2 数据和实现修复

在正式 NN 比较前完成了：

- 视频级 split，video/run 交集为0；
- canonical sample key 和 source event 对齐；
- metadata 空记录不能覆盖有效记录；
- GRU/LSTM 使用真实长度，不读取右侧 padding；
- Transformer 使用 attention mask；
- train-only scaler、vocab、transition graph 和 OOF teacher；
- prefix-only视觉帧，目标及未来帧违规为0；
- 答案、正确选项、未来状态、target runtime/VRAM 和 video ID 禁止进入输入；
- OOF缓存哈希、主键集合和负向篡改测试；
- 远端 R2 全套154项测试通过。

还发现过一次真实泄漏：交叉特征首版错误使用“当前目标动作族”，造成 XGBoost 虚高到1.0。该结果已作废，特征修复为上一事件动作族与模型的组合，并纳入可得性审计。

### 7.3 比较过的模型

| 编号 | 模型 |
|---|---|
| B00 | 全局先验、Markov |
| B01 | XGBoost |
| B02 | Logistic regression |
| B03 | 静态 MLP |
| B04 | FT-Transformer |
| B05 | Masked GRU |
| B06 | Masked LSTM |
| B07 | Causal Transformer |
| B08 | OOF XGB + 线性 meta |
| B09 | XGB + MLP residual |
| B10 | XGB + GRU residual |

### 7.4 Validation与序列消融

在固定 validation 上：

- B01 XGB joint：0.8501；
- B03 MLP：约0.851；
- B05 Masked GRU：0.8802±0.0019；
- B06 Masked LSTM：0.8798±0.0005；
- B07 Causal Transformer：0.8782±0.0007；
- B10 XGB+GRU residual：0.8768±0.0011。

按视频配对 bootstrap：

- B05相对B01：+2.92pp，95% CI `[+1.86,+4.11]pp`；
- B05相对B03：+2.17pp，95% CI `[+1.06,+3.37]pp`。

序列消融：

- 清空历史后下降2.14pp，95% CI `[+1.01,+3.31]pp`；
- 保留历史但打乱顺序后下降0.61pp，95% CI `[+0.08,+1.24]pp`。

这支持“历史内容有用，历史顺序也有较小增量作用”，但不是因果证明。

### 7.5 Finalist CV与未见视频holdout

在270个 development 视频的5折 finalist CV 中：

- B01：0.8355±0.0059；
- B03：0.8395±0.0060；
- B05：0.8754±0.0060。

之后锁定40个未见视频，每个视频重新运行 `st_fixed/star/langgraph_react`，得到120/120条成功轨迹。模型 epoch 由 CV 冻结，不根据 holdout 调参。

| 模型 | Joint accuracy | Family Top-1 | Role Top-1 |
|---|---:|---:|---:|
| B01 XGB | 0.7406 | 0.5494 | 0.9123 |
| B03 MLP | 0.7377±0.0026 | 0.5615 | 0.9046 |
| B05 Masked GRU | **0.8046±0.0048** | **0.6521** | **0.9348** |

B05相对B01提升6.40pp，相对B03提升6.69pp。这是当前最扎实的行为预测结果，但只覆盖 VideoMME 和当前三种 baseline，不能直接外推到电梯 Agent。

## 8. 多步、图结构、多模态和公开预训练实验

### 8.1 做了什么

已实现或运行：

- GRU、LSTM、原生消息传递 GNN；
- H=1…5 direct heads；
- 图约束 beam Top-K future path；
- entropy、margin、预计剩余长度；
- public event GRU预训练后迁移；
- task text、结构化视觉证据和资源 prefix；
- InternVideo2-1B prefix-only视觉特征；
- shuffled vision和zero+mask负对照设计。

InternVideo探针使用4帧输入，输出 `[1,1025,1408]`，缓存向量为CLS与patch均值拼接的2,816维；BF16单视频峰值约2.17GiB。首步没有历史帧时使用显式 `visual_mask=0`。

### 8.2 结果

旧64视频/640 runs多模态正式比较：

| 模型 | H1 | H3 | H5 |
|---|---:|---:|---:|
| GRU + structured + InternVideo | 0.798 | 0.506 | 0.339 |
| LSTM + structured + InternVideo | 0.811 | 0.503 | 0.335 |
| GNN + structured + InternVideo | 0.812 | 0.505 | 0.328 |
| 锁定v0.4参考 | **0.821** | **0.539** | **0.340** |

结论是：多模态和图结构已经真实接入，但在当时的标签规模上没有稳定超过v0.4。公开事件预训练也只改善部分NLL、Top-3或多步路径，没有形成稳定的headline优势。

当前最新B05证明了粗粒度序列建模有效，但尚未把B05编码器与H=3/5概率路径头整合成正式有限前瞻输出。这是下一阶段行为预测侧的关键缺口。

## 9. 资源预测器实验

### 9.1 预测目标和输入

资源预测器消费候选节点及当前可见信息：

- candidate activity/node/model；
- 已观察prefix；
- GPU容量；
- cold/warm；
- input scale。

正式输出契约 `scheduler-resource-contract-v1`：

```text
runtime_p50_ms
runtime_p90_ms
load_p50_ms
peak_memory_p95_mb
uncertainty
```

queue由调度模拟自然形成，不由资源预测器预测；strict OOM、stall风险和resident weight/workspace严格分离当前仍标记为unsupported。

### 9.2 数据与泄漏控制

- development：8,301行、64视频；
- final holdout：1,380行、40个未见视频；
- 只使用静态任务信息、候选节点和已发生历史；
- 禁止答案、future event、target runtime/load/peak和video ID进入输入；
- 所有统计量只在train侧拟合。

### 9.3 正式结果

正式选择由development validation完成，holdout只评估一次。

| Target | Holdout结果 | 解释 |
|---|---|---|
| runtime，正常服务样本 | 相对MAE约9.96%，P95覆盖约95.48% | 可作为正常服务时间预测 |
| runtime，包含全部事件 | 相对MAE约62.25% | 4个极端stall造成重尾，不能称为全面高精度 |
| load | 选定模型相对MAE约10.54%，P95覆盖约86.67% | 未见分布下覆盖不足，需要继续校准 |
| peak memory | 选定模型相对MAE约1.05%，P95覆盖约95.39% | 当前最稳定的资源目标 |

极端stall共有4条，占成功runtime行约0.32%；训练集中没有正stall样本，因此当前没有训练因果stall分类器，也没有静默删除这些长尾事件。

## 10. Workload v0.2与节点级模拟器

### 10.1 为什么重做workload

旧v0.1存在：压力定义粗糙、hot状态可能超显存、deadline读取真实总耗时、DAG边恢复不完整、只按顺序回放以及缺少节点级事件等问题。因此保留v0.1作历史smoke，重新构建v0.2。

### 10.2 节点级定义

一条真实trace编译成一个job template；planner、VLM、YOLO、时序工具、validator和answer generation等可测动作成为node。调度动作从job级下沉为：

```text
ready node × free GPU
```

模拟器产生：

```text
job_arrive
node_ready
node_dispatch
node_wait
model_load_start / model_load_finish
node_start / node_finish
model_evict
job_finish / job_fail
```

### 10.3 Workload规模

| Split | Episodes | 用途 |
|---|---:|---|
| train | 20,000 | 后续优化器/RL训练 |
| validation | 2,000 | 参数、早停和策略选择 |
| locked test | 6,750 | 一次性正式比较 |
| 合计 | **28,750** | 共1,066,080个job实例 |

测试矩阵：

```text
3种到达 × 5种计算压力 × 3种GPU拓扑 × 3种初始驻留
= 135个cell
```

每个cell 50个episode，并平衡16/32/64 jobs。到达包括Alibaba真实提交间隔重放、Poisson和burst；Alibaba数据只提供到达随机性，不替代本项目服务时间和显存画像。

### 10.4 W0–W6验收

- W0：冻结768条collection、资源契约和Alibaba到达块；
- W1：修复parent→predecessor、DAG、retry和edge provenance；
- W2：加入三类到达、分层无放回抽样、可验证初始驻留和无泄漏deadline；
- W3.1：修复GPU显存准入账本；
- W3.2：改成ready node×free GPU联合动作；
- W4：30个episode、480 jobs小规模门禁通过；
- W5：正式生成28,750 episodes，135/135 cell审计通过；
- W6：validation与locked test三策略配对重放和独立事件审计通过。

正式目录：

```text
/root/autodl-tmp/scheduler/results/processed/
  workload_v0_2_formal_20260812/
```

## 11. W6基础调度实验

目前运行的三种策略不是论文调度器复现，而是模拟器门禁：

- Round Robin：轮流分配，主要验证基础公平分发；
- Myopic：只根据当前ready节点和当前GPU状态做一步选择；
- Oracle：读取真实剩余路径和资源真值，作为信息上界。

### 11.1 Locked-test结果

| 策略 | 平均完成时间 | 平均排队时间 | Deadline miss | 平均GPU驱逐 |
|---|---:|---:|---:|---:|
| Round Robin | 182,017.6ms | 130,343.7ms | 11.31% | 46.49 |
| Myopic | 178,514.8ms | 102,314.7ms | 10.49% | 26.55 |
| Oracle | **140,675.5ms** | **63,214.1ms** | **4.22%** | 26.19 |

- Myopic相对Oracle gap：26.90%；
- Round Robin相对Oracle gap：29.39%；
- 6,750 episodes、248,400 jobs/策略；
- 三种策略failed jobs均为0。

Validation上Myopic/Oracle gap为28.50%，Round Robin/Oracle为29.21%。事件审计覆盖35个gzip分块、约4,760万条事件，时间顺序、显存账本、arrival配对和dispatch/start/finish关系均通过。

这证明当前workload已经有足够调度空间，但还没有证明预测器能回收该gap。

## 12. 当前可以支持的实验结论

1. 三种不同自由度的Video Agent控制流能够在统一schema下产生可比较的动态trace。
2. Qwen VLM、拆分planner/VLM栈和YOLO batch产生了真实的路径与资源差异。
3. 粗粒度角色/动作族比过细的具体节点更稳定，也更适合作为跨Agent接口。
4. Masked GRU在validation配对检验中得到优势支持，并在未见视频holdout上明显高于静态XGB/MLP点估计；历史内容提供了可测的增量信息。
5. 多步、图结构和InternVideo已经跑通，但旧数据上尚未稳定超过v0.4，不能包装成正结果。
6. 正常服务runtime和peak memory具有较好可预测性；极端stall、严格OOM和queue仍缺真实标签。
7. 串行真实trace可以在不伪造新trace的情况下编译成并发、多GPU、节点级workload。
8. 当前Oracle gap约27%，说明有限前瞻调度具有值得验证的潜在空间。

## 13. 尚不能声称完成的内容

- 尚未证明行为预测+资源预测能够改善正式调度指标；
- 尚未实现统一信息条件下的PEFT-H、数学优化器-H和RL-H；
- 尚未完成真实单GPU contention参数校准；
- 尚未在2张及以上GPU上运行真实调度；
- 尚无严格OOM、真实queue/wait和可训练stall风险数据；
- 尚未完成Grounded-VideoLLM环境门禁；
- DVD只有117条、60视频，未达到128条和固定split目标；
- 尚未用真实电梯Agent trace验证角色迁移和`verify`类别；
- 300视频行为数据不等于768条正式调度collection，二者用途不同；
- 28,750个workload episode和1,066,080个job实例是由真实模板派生的模拟负载，不是新增真实Agent trace。

## 14. 下一阶段已经收敛的方法定义

后续正式比较固定所有方法可见的信息：

```text
当前系统状态
+ 当前ready nodes
+ GPU状态
+ 行为预测器给出的未来H步概率路径
+ 资源预测器给出的时间/显存/加载分布
```

只改变信息利用方式：

| 方法 | 如何利用同一信息 |
|---|---|
| Myopic | 只做当前一步贪心 |
| PEFT-H | 用有限未来计算启发式前瞻代价 |
| Optimizer-H | 用CP-SAT/MILP做滚动时域联合求解 |
| RL-H | 用GNN/策略网络从训练交互中学习动作 |
| Oracle | 使用完整真实未来，仅作上界 |

核心消融应包括H=0/1/3/5/8，并比较：

- `Optimizer-H vs Optimizer-0`；
- `RL-H vs RL-0`；
- `PEFT-H vs Optimizer-H vs RL-H`；
- `Predicted-H vs True-H`；
- `True-H vs Oracle`。

这部分目前是下一阶段实验定义，不属于已经完成的实验结果。

## 15. 当前权威产物索引

### Trace与预测

```text
/root/autodl-tmp/scheduler/results/processed/
  collection_index_core_resource_fixed_20260804.jsonl
  job_templates_core_resource_fixed_20260804.jsonl
  trace_enrichment_core_fixed_canonical_v02_20260804/
  trace_predictor_core_fixed_canonical_v04_20260804.json
  prediction_baselines_core_fixed_canonical_v03_20260804.json
```

### 粗粒度行为预测

```text
/root/autodl-tmp/scheduler/results/processed/behavior_nn_v1_r2/
  comparisons/r2_metrics_summary.json
  comparisons/n1_paired_bootstrap.json
  comparisons/n1_sequence_ablation.json
  comparisons/finalist_cv/finalist_cv.json
  final_holdout/evaluation/final_holdout_report.json
```

### 资源预测

```text
/root/autodl-tmp/scheduler/results/processed/
  resource_predictor_v1_final_q99_20260811/
    resource_predictor_report.json
    scheduler_resource_contract.json
    holdout_predictions.jsonl
```

### Workload与调度

```text
/root/autodl-tmp/scheduler/results/processed/
  workload_v0_2_formal_20260812/
    train.jsonl
    validation.jsonl
    test.jsonl
    generation_manifest.json
    audit_summary.json
    w6_validation_report.json
    w6_formal_report.json
    w6_event_audit.json
```

## 16. 一句话总结

到目前为止，项目已经把“动态Video Agent真实trace”转化成了可审计的行为预测数据、资源预测契约和有压力的节点级多GPU workload，并证明序列行为预测有效、Oracle调度空间充足；下一步真正需要验证的是：有限前瞻预测能否被数学优化器或RL转化为可重复的调度收益。
