# 神经网络预测器迁移训练方案

## 目标与边界

目标是先用公开工作流/工具调用数据学习“节点和边的常见结构”，再用本项目自己的视频 agent trace 做域适配，得到可用于下一节点、多步路径和资源估计的预测器。该方案是可审计的工程复现路线，不宣称复现论文中未公开的私有训练代码或权重。

第一阶段不把 Qwen3-VL-8B 全量训练作为前提：视频语义模型可作为冻结特征提取器，预测器本体先保持轻量，以便和现有图约束、Markov 及调度模拟器做公平比较。

## 数据层

### 公开预训练数据

按“工作流结构”和“观察到的动作”分层使用，所有公开数据只进入预训练的 train split：

| 数据层 | 可借用的信号 | 不直接当作什么 |
| --- | --- | --- |
| BPI/Sepsis 等过程日志 | 节点、边、重复路径、完成长度 | 不当作视频语义标签 |
| Mind2Web、ToolBench、API-Bank | 观察→工具/动作、工具参数及失败重试模式 | 不把网站/API 名称当作本项目节点名 |
| 公开视频 agent/VideoQA trace（若字段可抽取） | 视觉观察、检索、回答的顺序 | 不混入本项目验证/测试视频 |

公开数据先映射到粗粒度本体（observe、retrieve、ground、summarize、answer、retry、end 等），保留 `source_dataset` 和 `domain_id`，避免不同数据集的同名动作被误认为完全相同。

### 自有域适配数据

使用当前核心数据的固定视频级划分：48 个 train 视频、8 个 validation 视频、8 个 test 视频；当前核心 prefix 文件约 640 条 run、4,441 个 prefix 行。后续资源层（runtime、峰值显存、OOM/等待、重试）只在相应字段真实存在的自有 trace 上训练资源头，不把这些未来标签放进节点预测输入。

统一样本至少包括：`run_id`、`video_id`（仅用于分组）、`source_dataset`、`baseline`、`position`、`prefix_nodes`、`observed_state`、`target_next_node`、可选完整 suffix、`status/retry/error`，以及单独的 runtime/VRAM/OOM 监督字段。

## 模型结构

1. **时序编码器**：对已观察节点和状态字段使用小型 GRU/Transformer，产生 prefix 表示。
2. **图编码器**：在训练集构建有向节点图，用 GraphSAGE/GAT 或关系型 Graph Transformer 聚合合法邻居；图只由 train 轨迹构建。
3. **多头输出**：
   - 下一节点分类：`P(next_node | prefix)`；
   - 多步路径：对合法边自回归解码，支持 H=1/2/3/5；
   - 长度/终止：`P(END)` 与剩余长度；
   - 资源头：runtime、峰值显存、OOM/等待概率（只用资源标签存在的自有样本）。
4. **域条件**：加入 baseline/model-stack/source-domain embedding；不把 `video_id` 作为输入特征。

首个可训练版本建议使用 2--20M 参数的小模型；Qwen3-VL-8B 只提供冻结的视觉/文本 embedding 或作为离线 teacher，避免显存和数据量把结构预测实验变成大模型微调实验。

## 两阶段训练

### A. 公开数据预训练

- 监督任务：next-node/next-tool 分类、next-edge 分类、完成长度回归。
- 自监督任务：随机 mask 节点、边重建、prefix 对比学习、suffix/长度排序。
- 每个样本带 `source_dataset` 和 `domain_id`；按数据集分层采样，防止某个 API 数据集主导图结构。
- 只保存 backbone、粗粒度节点词表和映射版本；本项目的细粒度输出头另建。

### B. 本项目 trace 域适配

- 先冻结 backbone，仅训练本项目节点头、图边头、baseline/domain adapter 和 LoRA/小型状态投影。
- 用 train 视频拟合；validation 只用于 early stopping 和超参选择，test 到最终一次才打开。
- 若适配后验证集稳定改善，再解冻时序编码器最后一层；不默认解冻全部视觉模型。
- 训练多步时使用 teacher forcing 只在训练前缀内部；推理时必须按图约束逐步解码。

## 对照、指标与验收

至少比较四条路线：

1. train-from-scratch Markov/图模型；
2. 小型神经时序模型 from scratch；
3. 公开数据预训练 → 自有 trace 微调；
4. 冻结 Qwen embedding → 轻量图/时序头。

节点层报告 Top-1/Top-3、NLL、校准误差；路径层报告 H=2/3/5 的 prefix-hit、exact-suffix-hit、suffix edit distance 和终止/长度 MAE；资源层报告 runtime/VRAM MAE、OOM/等待 AUROC 或 F1。最后把预测器接入 phase4 调度模拟器，报告 completion time、wait time、OOM、cache/load 次数。

正式验收条件：无未来字段泄漏；按视频分组的 48/8/8 评估可重跑；预训练迁移模型在 validation 和至少 3 个随机种子下相对 from-scratch 有稳定收益，或至少不损害节点/调度指标。若收益只出现在训练集或只来自更强先验，则保留图基线，不把它称作迁移成功。

## 风险与处理

- **域偏移**：公开过程日志能补足长路径和稀有边，但不能替代视频语义；用 domain embedding 和自有适配头隔离。
- **词表不一致**：保留粗粒度 ontology 与细粒度 canonical mapping 版本，未知节点进入 `other`，不强行同义合并。
- **稀疏与过拟合**：先做图/Markov sanity check，再增大模型；报告按 baseline、路径长度和是否重试分层指标。
- **标签泄漏**：`remaining_steps`、未来 suffix、答案文本、test/validation sidecar 只能做评估标签，不能进入输入；任何新增字段先过 leakage guard。
- **资源监督不足**：如果 OOM/等待样本数量不够，不训练复杂分类头，先使用分层统计或校准后的概率先验。

## 落地执行版（v0.2）

本节把上述方案收敛成可执行的第一条链路。第一版的公开数据只预训练时序/前缀编码器；本项目自己的有向图、canonical 节点头和资源头在自有 trace 上重新训练，避免把 BPI、API 和网页动作误当作同一套节点。

### 阶段 0：环境与数据门禁

1. 在远端 GPU 上只读检查 Python、PyTorch/CUDA、显存、磁盘、驱动和 Qwen3-VL-8B 权重路径；本机只承担转换器、schema 检查和小样本 CPU smoke test。
2. 公开数据只从官方入口获取，先下载每个数据集的少量 train 子集验证解析器，再扩大下载；记录 URL、许可证、版本、SHA256 和原始文件大小。
3. 不下载公开 test split 到预训练目录；Mind2Web 等数据集的测试集单独保存并只用于最终评估。

### 阶段 1：公开 trace 标准化

首批数据源为 BPI 2012/2014、Sepsis Cases、ToolBench、Mind2Web train。标准化输出采用 `neural-trace-v0.1`：

```text
dataset, case_id, group_id, split, domain_id
events[node, raw_action, position, status, duration]
task_context, observed_state, source_metadata
```

每条完整轨迹再切成 `neural-prefix-v0.1`：

```text
input:  prefix_nodes, task_context, observed_state, position, domain_id
target: next_node, suffix, end, length_bucket
```

`target` 永不进入模型输入。公开数据保留 dataset-local 动作词表，同时映射到 `observe/retrieve/tool_call/summarize/answer/retry/end` 等粗粒度类型；未知动作进入 `other`，不强行同义合并。

### 阶段 2：编码器预训练

模型从 2–10M 参数起步：节点/动作 embedding、两层 GRU 或小型 Transformer、位置和 domain embedding；每个公开数据集使用独立输出头。训练目标为 next-node、masked-node、END/长度和短 suffix；按数据集平衡采样，避免 ToolBench 的规模压过过程日志。

第一版不预训练跨数据集 GNN。公共数据只学习“prefix 的结构表示”；自有 trace 的图编码和 legality mask 另行拟合。这样可验证迁移是否来自路径结构，而不是来自错误的节点词表共享。

### 阶段 3：自有视频 trace 迁移

使用当前核心数据的 48/8/8 视频划分（480/80/80 runs，3327/555/559 prefix 行）：

1. 丢弃公开数据的 local output heads，重新初始化本项目 canonical action embedding/head；
2. 冻结公共时序编码器，只训练自有节点头、planner/baseline adapter 和 train-only 图 mask；
3. validation 有稳定收益后，再解冻时序编码器最后一层；
4. runtime、峰值显存、OOM、等待和重试头只使用自有资源 trace；
5. Qwen3-VL-8B 只做当前帧/已观察窗口的冻结 embedding，和 prefix/state 做消融比较，不把生成答案或未来帧作为特征。

### 阶段 4：验收与调度闭环

保留当前 graph/Markov 模型和自有 trace 从头训练神经模型作为对照。报告 H=1/2/3/5 的 Top-1/Top-3、NLL、ECE、prefix/exact suffix hit、长度误差，以及 phase4 的 completion time、wait time、OOM、模型加载次数。模型选择只看 validation；test 在方案冻结后打开一次。公开预训练只有在至少 3 个随机种子下稳定改善 validation 和调度指标时才保留。

### 当前执行边界

本轮已完成方案文档、数据源清单和自有 trace 标准化链路的实现准备；本机检查显示 M1/8GB、无 CUDA/PyTorch、剩余磁盘约 13GiB，不适合公开数据全量预训练。远端 GPU 环境、依赖安装和公开数据下载需要在当前新远程上完成只读门禁后再执行，避免把大文件和凭据写入本机。

本轮对本机 SSH 配置中的 `autodl-weste` 做了只读连接检查；连接在远端端口 `38739` 被关闭，未得到 hostname、GPU 或磁盘信息，也未在远端执行任何写操作。该 alias/端口不能作为已验证的训练环境，后续需使用当前有效的新远程配置重新做门禁。
