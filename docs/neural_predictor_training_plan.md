# 神经网络预测器训练与迁移方案（远端落地版 v0.1）

- 文档状态：300 视频库已完成；prefix-only InternVideo2 视觉分支 v0.3 三 seed 正式对比已完成
- 更新时间：2026-08-05 02:26（远端）
- 远端根目录：/root/autodl-tmp/scheduler
- 约束：视频、模型、trace、训练、评测全部在远端执行；本机不参与数据处理或训练

> 本文只记录已核对的资产、训练设计和验收门槛，不把尚未运行的模型、指标或下载结果写成已完成事实。原始 trace 永远不覆盖，所有派生文件保留 source hash。

## 1. 结论和研究边界

第一版按以下顺序实现：

1. 结构化 prefix → 小型因果序列模型（GRU 主线，Transformer-lite 对照）。
2. 用 train 视频学习的合法转移图做 logit mask，预测 next-node 及 H=3/5 未来路径。
3. runtime、peak VRAM、load cost 使用独立资源预测器，不把 GPU 当前状态混入 Agent 语义预测。
4. 公开事件日志只用于通用事件编码器预训练；自有 Video Agent trace 用视频级 48/8/8 split 迁移并锁定评测。
5. Qwen3-VL-8B 只做冻结、可选的视觉特征提取；不反向微调 8B 基座。若权重或显存门禁不通过，结构化模型仍是主线。

本方案是 Phase 3 统计/分层基线的神经扩展，不替代 Markov、prefix、hierarchical、RAG、Pythia-style 和资源 baseline。暂不以 GNN、RL 或完整 VLM 微调作为第一版必需项。

## 2. 远端现状（已核对）

### 2.1 数据资产

| 资产 | 当前事实 | 用途 |
|---|---|---|
| core collection | 640 runs、64 videos | 内容/路径预测主线 |
| resource collection | 128 runs | 资源条件和 workload 分层 |
| 视频 split | train 48 / validation 8 / test 8 | headline 评测固定口径 |
| core prefix | canonical 约 4,441 条 | next-node 监督 |
| final templates | 768 templates、约 9,366 nodes | resource/workload |
| workload | train/validation/test = 20,000/1,000/1,000 episodes | Phase 4 回放，不是内容标签 |

远端输入：

~~~text
/root/autodl-tmp/scheduler/results/processed/collection_index_core_resource_fixed_20260804.jsonl
/root/autodl-tmp/scheduler/results/processed/trace_enrichment_core_fixed_canonical_v02_20260804/
/root/autodl-tmp/scheduler/results/processed/prefix_samples_core_resource_fixed_20260804.jsonl
/root/autodl-tmp/scheduler/results/processed/job_templates_core_resource_fixed_20260804.jsonl
/root/autodl-tmp/scheduler/results/processed/workload_{train,validation,test}_core_resource_fixed_20260804.jsonl
~~~

现有参考结果：

~~~text
/root/autodl-tmp/scheduler/results/processed/prediction_baselines_core_fixed_canonical_v03_20260804.json
/root/autodl-tmp/scheduler/results/processed/trace_predictor_core_fixed_canonical_v04_20260804.json
/root/autodl-tmp/scheduler/results/processed/resource_predictor_core_final_20260803.json
~~~

当前 fine-grained next-node 参考约为 Top-1 0.80、Top-3 0.97；scheduler routing class 更高，不能冒充细粒度节点结果。资源预测器 test 有 1,035 条样本，best baseline 的 peak-VRAM MAE 约 577、runtime MAE 约 3,094（均按报告原单位）。

### 2.2 单卡环境

已核对远端 RTX 4080 SUPER，32,760 MiB 总显存、约 32,230 MiB 空闲。finetooling Python 环境已有 torch、transformers、accelerate、safetensors、scikit-learn；peft 未安装。

远端可见 Qwen3-4B、Qwen2.5-VL-3B 和 Grounded-VideoLLM。Qwen3-VL-8B 的实际权重路径尚未独立核验；不能仅凭 trace 的 model_stack_id 假设 8B 可加载。Grounded-VideoLLM 约 26G，只作为后续可选分支。

## 3. 预测任务

### 3.1 语义/路径 head

输入是 t 时刻以前的 prefix：

    x_≤t = (task, observed_state_≤t, action_≤t, provenance)

输出：

- y[t+1]：canonical activity，如 planner、sample_seek、spatial_qa、temporal_qa、object_detection、summarize、answer、retry、end；
- c[t+1]：调度 routing class，如 observe、answer、summarize、retry、end；
- y[t+1:t+H]：H=3、5 自回归未来路径；
- legal mask、unknown/fallback 和置信度。

语义 head 不读取答案、未来 suffix、目标事件的 runtime/VRAM、测试视频 ID 或目标之后字段。

### 3.2 资源 head

给定 prefix 时刻可见的任务、候选节点、模型栈、输入规模和 cold/warm 条件，单独预测：

- runtime_ms；
- peak_workspace_mb / peak_allocated_mb；
- load_ms；
- 可选 queue/wait 风险。

GPU 空闲状态和其他 job 的队列只进入调度器，不进入 Agent 语义 head。常驻模型显存未知时必须保留 unknown，不能把 workspace peak 宣称为完整模型显存。

## 4. 样本构造和数据来源

### 4.1 自有 trace

1. 按原始 event_id 排序，不只按 step_id。
2. 每个可预测位置生成 prefix-target pair；H=3/5 标签只在标签侧读取。
3. prefix 只包含 t 以前的 action、state 和 provenance。
4. 失败、解析错误、等待、OOM、retry 进入 failure/recovery 层；不能伪造成功后缀。
5. 同一 run 的相同 prefix hash 去重，并保留 source_trace_sha256、source_event_ids、split、leakage marker。
6. 128 条 resource run 用于资源分层；本轮试验发现其中部分 canonical target 为 other，无法与 core 的 7 类 semantic head 对齐，因此未混入 semantic 训练；如用于语义消融，必须先完成 ontology 映射并显式加入 yolo_batch/资源条件，不能混入 640 core 指标。

### 4.2 公开预训练

候选：BPI Challenge 2012、BPI Challenge 2014、Sepsis、ToolBench、API-Bank、Mind2Web。

公开数据的 prefix→next-event 记录就是 N1 阶段的真实训练样本，而不是仅用于统计或初始化的“背景数据”。N1 先在公开事件序列上训练通用因果事件编码器；N2 再将可迁移的编码器参数加载到自有 Video Agent 模型，并只用自有 train 视频微调 canonical activity head。公开 activity label 不直接当作 Video Agent canonical label，也不进入自有 48/8/8 的 headline 指标。转换记录：

~~~text
public_action_id
public_dataset
case_id_hash
event_index
event_name
timestamp_bucket
argument_schema_summary
~~~

通过 ontology mapper 映射到 observe/tool_call/answer/retry/end 等公共层级；未知动作保留 unknown。每个数据集记录许可证、下载时间、SHA256、转换版本和自己的 train/validation/test。公开数据与自有 8 个 test 视频隔离。

## 5. 特征契约与防泄漏

### 5.1 语义 head 允许的特征

- task：question_type、temporal_scope、answer_type、required_modalities、option_count、official_task_type；
- observed state：coverage_ratio、frames_seen、modality/object/OCR/temporal evidence、累计 retry/error；
- prefix action：最近 1–3 个 canonical activity、position、terminal flag；
- provenance 消融：baseline、planner_model_id、model_stack_id；
- 数值归一化统计量只由 train 拟合。

正式报告同时给出含 provenance 和不含 provenance 两种结果，防止把 planner/模型栈差异误报成内容增益。

### 5.2 资源 head 允许的特征

candidate model、node_type、activity、input_scale、frame_count、qwen_image_count、yolo_batch、GPU capacity、cold/warm/load state，以及 prefix 时刻已经观测到的资源字段。禁止使用 target runtime/VRAM。

### 5.3 永远禁止

- video_id、video_sha256 或能记住测试视频的 ID；
- 答案、答案文本、正确选项；
- target/future 的 action、runtime、VRAM、error；
- 完整 trace embedding 后再切 prefix；
- 用 test 视频构建 RAG、词表、scaler、转移图；
- 用合成 workload 的 future job 当内容预测标签；
- 将同一视频的重复运行跨 split。

## 6. 模型结构

### 6.1 主线 Structured GRU

- categorical embedding：activity、node_type、task taxonomy、planner/stack condition、status；
- numerical MLP：coverage、frames、modality/target counts、position、retry/error；
- 每个时间步拼接为 256 维 event vector；
- 2-layer causal GRU，hidden=256，dropout=0.1；
- padding 使用 mask；
- Transformer-lite（2 layers、d_model=256、4 heads）作为对照，不只报告最优结构。

输出头：

- fine next-activity；
- scheduler routing class；
- autoregressive H=3/5；
- uncertainty/temperature calibration；
- train-only graph mask，未见上下文回退到 train-only global/position prior。

### 6.2 独立资源模型

首版用 MLP/GRU 的两个回归头：

- log1p(runtime_ms)；
- log1p(peak_workspace_mb)。

使用 Huber/SmoothL1，反变换后报告 MAE、RMSE、P95。global median、activity median、model-activity median 和 park-song-style 仍为控制组。

### 6.3 可选 Qwen3-VL-8B 分支

仅在环境门禁通过后执行：

1. 冻结 8B，只读取 prefix 时刻已经采样的帧/证据。
2. 提取视觉/视频表示或短文本摘要表示，不读取答案 logits。
3. batch=1/2、fp16/bf16、有限帧数，顺序提取并缓存 safetensors/pt。
4. 训练 projection + GRU/Transformer head，不更新 VLM。
5. 比较 structured-only、Qwen-feature-only、structured+Qwen。

当前阶段的硬门槛：主 Top-1、Top-3、NLL 和 H=3/5 prefix-hit 必须相对于锁定的 v0.4 参考分别报告；只有在 validation 选出的配置在 test 上超过 v0.4，才可称为“优于当前最优”。若只改善 NLL/Top-3/多步路径而 Top-1 未超过，必须分指标陈述，不能笼统宣称胜出。
6. 权重不存在、OOM、吞吐过低或 cutoff 无法对齐时跳过并记录，不偷偷替换模型。

## 7. 训练阶段

### N0：审计和转换

生成 train-only vocab、task ontology、合法 transition graph、numeric scaler、prefix-target、H=3/5、resource regression、source hash 和 leakage report。

门槛：每条样本可回溯到 source event；48/8/8 隔离；词表/scaler/graph 只看 train；解析率 100%；失败样本单独计数。

### N1：公开事件编码器预训练

任务：next-event、masked-event，可选 case-level contrastive。建议 max length 32/64、batch 64（显存允许 128）、AdamW lr 1e-3、5–20 epochs、按 validation NLL early stop。

输出：

~~~text
results/processed/public_event_gru_pretrain_v0_1/
  public_gru_checkpoint.pt
  pretrain_metrics.json
  pretrain_rows.jsonl
~~~

### N2：自有 trace 迁移

当前已执行版本：加载公开预训练 checkpoint 中 shape-compatible 的 2-layer GRU recurrent weights，重新初始化自有 token embedding 和 canonical label heads；在自有 train split 上端到端微调，teacher prior 和 ensemble 权重只在 validation 选择。冻结 encoder warm-up 仍作为下一轮消融，不把尚未执行的阶段写成正式结果。

后续可选阶段：冻结 encoder，只训练 next-activity/routing/uncertainty heads 5 epochs；再解冻最后一层，lr 降至 1e-4 或更低训练 10–20 epochs，按 validation NLL/Top-1 early stop。

阶段 C：只在 validation 做 temperature scaling、class weight 和 graph mask 阈值选择；test 只正式运行一次。

输出：

~~~text
results/processed/neural_trace_predictor_structured_v0_1/
  checkpoint_best.pt
  artifacts.json
  metrics_summary.json
  leakage_report.json
~~~

### N3：图约束和多步

分别评估 unconstrained、hard mask、soft penalty；生成 H=3/5，记录 top-K、legal rate、prefix_hit@1/3/5。不能用 test future 重新选择 mask 或 decoding 参数。

### N4：资源迁移

semantic model 不读 target runtime/VRAM；resource model 只读 compute_view 的 prefix-visible 条件。先 structured-only，再做 Qwen/YOLO batch 消融，按 activity/model/input-scale 分桶。

### N5：Phase 4 回放

P3 通过后，把 top-K future candidate 和资源预测交给 predictive policy。与 myopic、static-template、rule/predictive、oracle 使用相同 workload、seed 和 GPU topology。Oracle 相对 Myopic 的完成时间/尾延迟差异稳定超过 1% 后，再增加调度复杂度。

## 8. 单卡执行规则

- 结构化模型训练与 YOLO/Qwen 推理不并发；
- structured 模型先从 max sequence 32、batch 64 开始，OOM 按 32/16/8 退让并记录；
- 使用 fp16；仅在 torch.cuda.is_bf16_supported() 通过时选 bf16；
- 用 gradient accumulation，不偷偷缩短任务；
- Qwen 特征 batch 1/2、顺序加载、缓存后卸载；
- 不同时驻留 Qwen3-4B、Qwen2.5-VL-3B、Grounded-VideoLLM 和训练模型；
- 每个 run 记录 peak allocated/reserved、runtime、load、OOM/retry；
- peft 未安装，第一版不依赖 LoRA；需要改环境时另行确认。

## 9. 评估、消融和判据

必报语义指标：Top-1、Top-3、MRR、NLL、ECE、reliability bucket、fallback rate、graph legality、H=3/5 prefix-hit@1/3/5，并按 baseline、planner、stack、task type、首步/中间步分桶。

必报资源指标：runtime/peak-VRAM MAE、RMSE、P95、cold/warm/load 分桶，并与现有 median/park-song-style baseline 比较。

调度指标：makespan、平均/尾部完成时延、deadline miss、queue、model-load、OOM/recovery，以及 predictive 相对 myopic/static/oracle 的跨 seed 差值。

建议的继续门槛（不是预先宣称成功）：

1. 固定 48/8/8、3 seeds、按视频 bootstrap 95% CI；
2. 相对 planner-state v0.3，Top-1 提升至少 2 个百分点，或 Top-1 不下降且 NLL/ECE 显著降低；
3. graph-constrained legality ≥99.5%，不能通过把未知样本全回退到一个先验实现；
4. H=3/5 prefix-hit 有可解释提升；
5. 资源 MAE/P95 按分桶改善；
6. 无稳定增益时，统计/分层模型继续作为正式主结果。

消融矩阵：

| 编号 | structured | provenance | graph | public pretrain | Qwen feature |
|---|---|---|---|---|---|
| A | 是 | 否 | 否 | 否 | 否 |
| B | 是 | 是 | 否 | 否 | 否 |
| C | 是 | 是 | 是 | 否 | 否 |
| D | 是 | 是 | 是 | 是 | 否 |
| E | 是 | 是 | 是 | 否 | 冻结 Qwen |
| F | 是 | 是 | 是 | 是 | 冻结 Qwen |

## 10. 计划新增的远端实现文件

以下文件已经在远端落地；其中带有 smoke/formal 后缀的结果目录分别保留诊断和正式候选，不覆盖原始 trace：

~~~text
/root/autodl-tmp/scheduler/configs/neural_predictor_train_v0_1.yaml
/root/autodl-tmp/scheduler/tracing/analysis/public_event_pretrain.py
/root/autodl-tmp/scheduler/tracing/analysis/augment_teacher_features.py
/root/autodl-tmp/scheduler/tracing/analysis/neural_trace_predictor.py
/root/autodl-tmp/scheduler/data/public/raw/
/root/autodl-tmp/scheduler/results/processed/public_event_gru_pretrain_v0_1/
/root/autodl-tmp/scheduler/results/processed/neural_prefix_dataset_v0_1/
/root/autodl-tmp/scheduler/results/processed/neural_prefix_dataset_v0_2_teacher/
/root/autodl-tmp/scheduler/results/processed/neural_trace_predictor_public_teacher_v0_1/
~~~

实现时必须复用现有 trace loader、split manifest、canonical activity 和 leakage validator，不另造互不兼容的事件接口。

## 11. 当前状态和下一步

已完成并核验：

- 核对远端项目、GPU、Python 依赖、collection、prefix、template、workload 和现有预测器；
- 锁定当前 fine-grained v0.4 参考：test Top-1=0.815742、Top-3=0.980322、NLL=0.583295；
- 远端获取 10 个公开事件日志 CSV（总 2,746 条 prefix→next-event 样本），保存源码 blob SHA、文件 SHA256 和转换后的 JSONL；
- 完成 N1 公开事件 GRU 预训练：CUDA，train=2,190、validation=556；checkpoint 只导出可迁移的 2-layer GRU recurrent weights；
- 完成 N0 自有 prefix 转换：4,441 rows、640 runs、64 videos，48/8/8 视频 split，无 split conflict、prefix mismatch 或 leakage rejection；
- 完成 train-only v0.4 teacher 概率特征附加，并完成公开预训练→自有 trace 的多 seed 迁移烟囱/正式候选；
- 迁移模型目前已经改善部分 NLL、Top-3 和多步路径，但单个 seed/ensemble 的 Top-1 尚未稳定超过 v0.4，因此尚未宣称“优于当前最优”。

当前正式候选结果：

- /root/autodl-tmp/scheduler/results/processed/neural_trace_predictor_public_teacher_v0_1/：公开预训练 GRU + 自有 teacher feature，3 seeds，供多步/分布分析；
- /root/autodl-tmp/scheduler/results/processed/neural_trace_predictor_public_teacher_prior_h1_v0_1/：teacher-prior residual，3 seeds已完成；validation 只选 α=0.05 的保守融合；
- smoke 目录仅作诊断，不进入 headline 结论。

下一步严格按远端执行：

1. 已完成 prior-h1 3-seed ensemble；validation 选择 α=0.05 的保守融合；
2. 已在同一 test split 报告 v0.4、teacher-only、neural transfer、ensemble 的 Top-1/Top-3/NLL 和 H=3/5 prefix-hit；ECE 仍需在最终汇总脚本中补齐；
3. 当前结果允许报告“小幅 Top-1 和 NLL 改善”，但 Top-1 bootstrap CI 包含 0，且未达到 2 个百分点/10% 大幅增益门槛；v0.4 继续作为主结果，神经模型作为迁移消融；
4. 只有后续冻结 warm-up、更多公开事件源或冻结 Qwen 特征在 validation 锁定后带来稳定增益，才升级主结果；否则进入 Phase 4 时同时保留 v0.4 和迁移 ensemble 两个策略。


## 10. 2026-08-05 实际落地更新

### 10.1 300 视频库和并行下载

已在远端生成 300-video manifest、source、runtime 和 annotations：

    configs/phase3_videomme_300_download.jsonl
    configs/phase3_videomme_300_source.jsonl
    configs/phase3_videomme_300_runtime.jsonl
    configs/phase3_videomme_300_annotations.jsonl
    configs/phase3_videomme_300_summary.json
    configs/phase3_video_pool_300_v0_1.jsonl

下载使用 hf-mirror.com，当前保持 4 个视频任务并行、每个视频 4 个 HTTP range 分段并行（最多约 16 个 range 请求）。下载器与模型训练是两个独立后台进程；下载不中断训练。临时分段目录固定在数据盘：

    VIDEOMME_REPO_BASE=https://hf-mirror.com/datasets/lmms-eval/Video-MME/resolve/main
    TMPDIR=/root/autodl-tmp/cache/videomme_tmp

系统盘 /tmp 曾因旧临时目录达到 100%，已确认无进程占用后清理；后续增加递归清理器，只删除已有完整 mp4 的压缩分段。2026-08-05 01:25 已验证 197 个有效视频，下载仍在继续；该数字是动态状态，不是最终数量。

### 10.2 无泄露的多模态结构化输入

已落地：

    tracing/analysis/build_multimodal_prefix_dataset.py
    tracing/analysis/add_multimodal_horizon_labels.py
    tracing/analysis/join_teacher_probs.py
    results/processed/neural_prefix_dataset_v0_2/prefixes_labeled.jsonl
    results/processed/neural_prefix_dataset_v0_2/metadata.json
    results/processed/neural_prefix_dataset_v0_2/labels_report.json

v0.2 每个 prefix 同时保留：

- 已观察的 semantic activity 和 raw tool action 序列；
- question/options 文本（不保存答案）；
- 已观察的视觉证据统计（覆盖率、帧数、OCR、对象/模态计数等）；
- 目标事件之前严格截止的 compute 事件累计（模型、API/action 次数、peak allocated/reserved、runtime/load、输入帧数、Qwen 图像数、YOLO batch 等）。

重要边界修正：原始 action:3 事件编号从 1 开始，而 compute 的 source_event_index 从 0 开始；增强器现使用 target_event_number - 1，目标事件本身不进入输入。审计结果为 4,441 rows、640 runs、64 videos、3327/555/559 split，答案 key=0、future/ground truth=excluded、compute cutoff 严格生效。

### 10.3 神经网络结构与详细输出

    tracing/analysis/neural_multimodal_predictor_v02.py

同一模型脚本支持：

- GRU、LSTM；
- 不依赖 torch_geometric 的原生两轮 message-passing GNN；
- semantic prefix、raw-action prefix、任务文本、结构化视觉证据、资源前缀和可选 train-only teacher prior 的融合；
- next node H=1..5、图约束 beam top-K、每个 horizon 的 top-3/entropy/margin、预测剩余步数和长度熵；
- 自回归路径作为单独诊断输出，不与一次性多头主指标混淆。

teacher prior 只来自 train-only 拟合、只看 prefix 的旧模型：

    results/processed/neural_prefix_dataset_v0_2/prefixes_teacher.jsonl
    results/processed/neural_prefix_dataset_v0_2/teacher_join_report.json

### 10.4 已完成的 v0.2 正式三 seed 对比

报告：

    results/processed/multimodal_predictor_v0_2_formal_comparison.json

模型在锁定 test 上的三 seed 均值（主指标为一次性多头+train-only图约束）：

| 模型 | H=3 prefix-hit | H=5 prefix-hit | 结论 |
|---|---:|---:|---|
| GRU | 0.483 | 0.300 | 未超过 v0.4 |
| LSTM | 0.494 | 0.320 | 未超过 v0.4 |
| GNN | 0.481 | 0.317 | 未超过 v0.4 |

锁定 v0.4 参考为 H=3=0.539、H=5=0.340；因此本轮只能结论为“输入更丰富、输出更详细、GNN 验证集较强，但尚未达到当前最优多步精度”。不能把 smoke 或 validation 的高点写成 test 改进。

自回归诊断短烟测显示，若把预测 semantic 节点直接伪装成 raw tool action 回灌，会产生输入分布偏移，H=3/H=5 明显下降；后续必须使用 semantic/raw 双通道的合法状态更新，不能简单字符串回填。

### 10.5 当前下一步

1. 保持 300 视频下载到 300，并核验 CRC/SHA256、文件数和空间。
2. 下载完成后为 expansion_train 视频重新生成 trace，再按同一字段契约离线增强。
3. 在真实 prefix frame clip 上做冻结 InternVideo2/8B 特征可行性探针；只读当前已观察时间窗，缓存后再训练投影层，不能把完整视频 embedding 当 prefix 特征。
4. 对 teacher-prior、structured-only、structured+frozen-video 三组按 validation 选择，锁定 test 只看一次。
5. 只有在 test H=3/H=5（及需要的 Top-1/NLL/ECE）超过 v0.4，才宣称神经网络优于当前最优；否则保留为丰富输入/详细输出基线。

## 11. 2026-08-05 视觉分支与300视频库实际结果

### 11.1 采集状态

- configs/phase3_videomme_300_download.jsonl 对应的 300 个 Video-MME 视频已全部通过校验：有效 MP4 300/300。
- 远端下载使用镜像源和 --workers 4 --segment-workers 4，即最多 4 个视频任务、每个任务 4 个 range 分片并行；本轮实际新增下载 206 个、已有缓存复用 94 个，约 27 GiB。训练期间未停止下载器；下载完成后才自然退出。
- 数据盘完成时约剩 34 GiB；临时分片目录已清空。300 视频中只有原锁定 64 个已有 trace，新增 236 个目前属于扩展视频池，不能冒充已经有标签的训练样本。

### 11.2 prefix-only 视觉输入和泄漏审计

离线从原始 trace 对齐 prefix_source_event_ids 之前的 input.frame_indices，生成：

~~~text
results/processed/neural_prefix_dataset_v0_2/prefixes_visual_context.jsonl
results/processed/neural_prefix_dataset_v0_2/visual_context_report.json
~~~

4441 个 prefix、640 个 run 中，3797 行有历史帧，644 行是首步缺少前置帧；目标事件及其之后的帧全部排除，future_frame_violations=0。此步骤没有使用答案、remaining_steps、target runtime/VRAM、video_id 作为输入，也没有先做整段视频 embedding 再切 prefix。

Grounded-VideoLLM 的 InternVideo2-stage2-1B-f4 权重已实际探针通过：checkpoint 是 4 帧位置编码，因此使用 4 帧输入，输出 [1,1025,1408]，BF16 单视频峰值约 2.17 GiB。188 个唯一 prefix 上下文抽取了 124 个非空 embedding，保存为：

~~~text
results/processed/neural_prefix_dataset_v0_2/internvideo_prefix_embeddings.pt
results/processed/neural_prefix_dataset_v0_2/internvideo_prefix_embeddings_report.json
~~~

每个向量为 CLS 与 patch 均值拼接的 2816 维；没有历史帧的首步使用显式 visual_mask=0，不是伪造的视觉证据。

### 11.3 三种神经结构正式结果

实现文件：

~~~text
tracing/analysis/neural_multimodal_predictor_v03_visual.py
~~~

输出同时包括 H=1..5 的候选及概率、entropy、margin、graph-constrained top-K path、预测剩余长度，以及单独标记的 autoregressive diagnostic path。正式三 seed (11/13/17) 均使用原锁定 48/8/8 视频划分：

~~~text
results/processed/multimodal_predictor_v0_3_visual_formal_seed_11/
results/processed/multimodal_predictor_v0_3_visual_formal_seed_13/
results/processed/multimodal_predictor_v0_3_visual_formal_seed_17/
results/processed/multimodal_predictor_v0_3_visual_formal_comparison.json
~~~

| 模型 | test H1 mean | test H3 mean | test H5 mean |
|---|---:|---:|---:|
| GRU + structured + InternVideo | 0.798 | 0.506 | 0.339 |
| LSTM + structured + InternVideo | 0.811 | 0.503 | 0.335 |
| GNN + structured + InternVideo | 0.812 | 0.505 | 0.328 |
| 锁定 v0.4 gated reference | 0.821 | 0.539 | 0.340 |

结论：视觉分支已经真实接入并通过无泄漏审计，但在当前 64 视频/640 run 标签集上没有稳定超过 v0.4；它应作为可复现实验和消融，不应被写成当前 headline 最优模型。最接近的是 GRU 的 H5（均值仅低约 0.001），但 H3 仍低约 0.033。要让多模态优势显现，下一步需要给新增 236 个视频生成同样的真实 trace，再按视频隔离重新训练，而不是调 test 解码参数。

### 11.4 后续门槛

1. 为扩展视频池生成可审计的 baseline trace，补齐 prefix labels 后再训练；300 视频池不能直接改变当前 48/8/8 headline test。
2. 保持 visual_context_report.json 的目标/未来帧零违规，并对新 trace 重跑同一审计。
3. 只有 validation 选出的模型在锁定 test 上同时改善关键 H3/H5，才进入 Phase 4 调度回放；否则保留 v0.4 reference，视觉分支作为 ablation。
