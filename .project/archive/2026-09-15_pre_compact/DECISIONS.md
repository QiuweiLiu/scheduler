# Decisions

Append important decisions. Do not rewrite history.

## Decision Log

<!--
### YYYY-MM-DD — Decision title
- Decision:
- Reason:
- Alternatives considered:
- Consequence:
-->

### 2026-08-13 — Project OS 与源码布局迁移
- Decision: 以 `.project/` 为唯一项目控制面；将本地和远端当前 tracing 源码统一放在 `src/tracing/`，Python 包名保持 `tracing`，运行时通过 `PYTHONPATH=src` 解析。
- Reason: `tracing` 是源码而不是数据；把它放入 `data/` 会混淆原始数据、派生数据和可执行代码的生命周期，也会破坏现有导入语义。
- Alternatives considered: 保留根目录第二份源码、改成 `from src.tracing`、把源码放入 `data/`；均会造成双源、导入漂移或数据/代码边界混乱。
- Consequence: shell/CLI 路径已改为 `src/tracing/...`；历史文档中的旧路径只作为历史证据，当前命令以 `docs/source_layout.md` 为准。

### 2026-08-13 — Legacy trace collection 不覆盖
- Decision: 768 条历史轨迹登记为 `trace_collection_v1_legacy_core`，600 视频扩展登记为新的 collection ID；所有派生数据必须记录 source collection 和 hash。
- Reason: 旧轨迹仍可用于回顾性分析和对照，但预测器见过的视频不能伪装成 C2 全新测试数据。
- Consequence: R5/R6 重新生成扩展 collection 的 split、causal prefix、资源视图和 workload；旧 v1 只读保留。

### 2026-08-15 — 远端只同步控制面并采用加法式目录规范化
- Decision: 将本地 `AGENTS.md`、`.project/` canonical 文件和小型 registry 同步到远端；只新增 Project OS 标准目录，不移动或删除已有数据、模型、源码、结果和缓存。
- Reason: 远端无 Git 且磁盘紧张，必须先建立可恢复的控制面；历史目录的所有权、引用关系和清理边界尚未逐项确认。
- Alternatives considered: 直接合并或删除重复模型和脚本目录；风险是破坏运行时引用、旧轨迹可复现性或正在使用的缓存，因此暂不采用。
- Consequence: 远端控制文件由本地 canonical copy 驱动，源同步使用 hash manifest；重复目录与缓存仅登记在 `data/manifests/remote_layout_audit_20260815.json`，后续需单独确认后再处理。

### 2026-08-15 — 640×360 先做 pilot，不直接替换 R5 原始视频
- Decision: 保留原始 Video-MME MP4；640×360 只作为独立派生版本。六个 1280×720 样本的 `libx264 -preset fast -crf 23` aggregate output/input ratio 为 0.755591，暂不据此切换全部 260 个视频。
- Reason: 实测节省仅 24.4409%，且单视频 ratio 为 0.494763–0.960285；部分原视频已经低码率，强制缩放并不能稳定带来预期的空间收益。
- Alternatives considered: 直接按 25%–40% 比例估算并批量转码；该估算未经测量，且会压缩远端 trace/workload 的剩余空间预算，因此不采用。
- Consequence: R5 继续按批次处理；如仍需压缩，先做更激进编码参数与视觉/轨迹质量对比，再冻结新的派生 collection 和容量 manifest。

### 2026-08-15 — 视频 provenance 先于清理与派生 collection 冻结
- Decision: 以 `src/tracing/schema/video_provenance_v0_1.json` 作为视频来源、原始文件、派生文件和 retention 的统一记录格式；原视频保留为 source-of-truth，640×360 只作为带 `parent_sha256` 的独立 variant。
- Reason: 仅有 `video_id/source_url/video_path` 无法证明原始字节、官方 ZIP 成员、视觉条件或派生父子关系；在远端空间紧张时，缺 provenance 直接删除会破坏 C2、重采和结果审计。
- Alternatives considered: 只保留文件名/URL后删除原视频，或把派生文件覆盖到原路径；均无法恢复同一字节内容，也会混淆 trace 的输入条件，因此不采用。
- Consequence: R5 顺序固定为 provenance 回填与校验 → split/容量 gate → 可选派生 → 归档恢复测试 → 单独删除确认；在此之前 `deletion_eligible=false`。

### 2026-08-15 — 下载采用本地获取后校验上传
- Decision: 不把远端国际网络直连作为下载前提。官方视频、ZIP 尾部索引和其他非中国大陆来源默认在本地获取；完成大小/CRC32/SHA-256 校验后，用不带 `--delete` 的 rsync 分批上传远端。远端只有在已知、已验证的本机代理端口存在时才可走端口转发。
- Reason: 实测远端访问 `huggingface.co:443` 返回连接失败；继续重试同一远端 URL 只会阻塞任务，且无法证明数据完整性。
- Alternatives considered: 猜测代理端口、在远端循环重试、直接把未校验文件放入正式数据目录；均会引入不可审计的网络或数据完整性风险，因此不采用。
- Consequence: 下载任务拆成“本地获取/索引 → 本地校验 → 小批 rsync → 远端 SHA-256 复核”；网络失败时保留失败报告，不生成已完成标记。

### 2026-08-16 — R6 使用 causal-v2 与 engine-only truth
- Decision: 以远端 `r6_causal_v2_20260816_fix1` 作为唯一当前 R6 派生结果；C1 继续使用 48/8/8 视频级 split，C2 只冻结 32+8 个 predictor-unseen pending 候选。prefix/current-node identity、future provider、resource predictions 和 scheduler-visible workload 分开保存；runtime/load/memory/status、source video refs 和完整后缀只保留给 engine/audit。
- Reason: 旧 B05/resource artifact 使用混杂 split、同 run future prototype 和整条 workload 估计，不能作为正式有限前瞻输入；旧模板还缺少显式 execution lane。R6 用 train-only prototype/quantile 重建，并对输入泄漏、causal-chain edge、概率归一和 workload 字段做 gate。
- Alternatives considered: 直接复用 `scheduling_future_v1_20260812` 或 `workload_v0_2_formal_20260812`；它们保留历史诊断价值，但信息边界/评估 split 不满足 R6，因此不选。
- Consequence: R7 必须在统一事件引擎中显式实现 StateView/Truth/FutureProvider；C2 轨迹采集只能在 predictor/resource 冻结后进行，不能把 R6 pending registry 当成已完成 C2 数据。

### 2026-08-16 — 后续预测器与调度实验采用最新 NN 数据口径
- Decision: 后续行为预测器、资源预测器和调度器主线以已冻结的 `behavior_nn_v1_r2` 数据体系为准：300 个 VideoMME 开发视频用于 train/development（内部 role split 为 240/30/30，旧 test 不作为最终选择集），另设 40 个完全未参与开发的视频作为 final holdout。旧 64 视频/48/8/8/768 legacy collection 只保留为经典基线、R6 回溯和兼容性对照，不再作为新主线训练集。
- Reason: 最新正式行为预测器 B05 Masked GRU 的训练、CV 和 final holdout 都已在该数据体系上完成；继续以 64 视频作为主口径会把历史经典预测器、R6 因果派生和当前 NN 行为预测混成一个数据集，并破坏未见视频评估边界。
- Alternatives considered: 继续沿用 64 视频作为所有预测器主线，或把 300 个开发视频与 40 个 holdout 混合训练；前者与最新训练结果不一致，后者会污染最终未见评估，均不采用。
- Consequence: 新的 resource/scheduler 实验必须从 300-video development pool 拟合，在 40-video holdout 上只做冻结后的最终评估；R6 legacy 结果不覆盖、不重写，电梯/C2 新轨迹另建独立 collection。

### 2026-08-17 — R7 采用600视频六组边界和最小主线矩阵
- Decision: 固定 `P_dev=300`、`P_holdout_diag=40`、`S_train=120`、`S_val=40`、`T_final=80`、`T_backup=20`。行为/资源预测器共享P_dev；调度器只使用另外260个predictor-unseen视频。调度主线先运行 Myopic、Optimizer-0、PredOpt-H1/H3/H5、TrueOpt-H1/H3/H5；只有True-H确认可利用未来信息后才进入成对RL-0/RL-H，validation只选择一个H*。
- Reason: 这直接检验“有限前瞻是否改善调度”，同时避免把预测器见过的视频用于调度训练或最终测试，也避免无信息增益的H=8、完整噪声网格、结构扫参和全量真实GPU回放。
- Alternatives considered: 继续沿用旧64视频/48/8/8作为主线，或一次性展开全部H×模型×噪声×真实回放矩阵；前者污染新数据口径，后者在当前磁盘和数据边界下产生大量防御性结果，均不采用。
- Consequence: R7-0先完成ID级split同步和零交集gate；未通过前不采集正式scheduler trace。唯一边界清单为 `data/manifests/video_split_registry_r7_v1.json`，旧R6只保留compatibility replay。

### 2026-08-17 — R7 pilot 允许保留可恢复解析错误
- Decision: pilot gate 以“最终 run 成功、trace schema valid、错误有明确成功 retry、信息边界无泄漏”为准，不要求错误事件数量为零；解析错误和 retry 事件必须原样保留。
- Reason: 解析失败/重试本身是动态 trace 的有效行为，也是后续压力与资源预测的输入；把它们静默删除会低估真实路径复杂度。pilot 中 2 个解析错误均由成功 retry 恢复，8/8 run 最终成功。
- Consequence: 640-run 扩展沿用相同规则；不可恢复失败进入失败清单，不用替补或重复运行掩盖失败率，只有通过完整质量 gate 后才生成正式 workload。

### 2026-08-17 — 明确授权清理旧 P_dev 原始视频
- Decision: 在当前 R7 采集继续运行的条件下，只删除旧 `P_dev` 的 300 个远端原始 MP4；保留 `P_holdout_diag`、`S_train`、`S_val`、`T_final`、`T_backup`，以及所有 trace、模型、实验输出和 provenance。由于用户明确要求不保留本地归档，本次对默认 archive-restore 删除闸门作范围限定的例外，以来源 URL 和官方 archive 定位作为再获取依据。
- Reason: `P_dev` 与当前 R7 活跃 160 个视频零交集，300 个视频均已有成功 raw trace；远端磁盘空间紧张，删除可释放 28,436,477,904 bytes，而不影响已保存的 trace-level predictor/workload/scheduling 输入。
- Alternatives considered: 保留原始视频会继续占用约 28.44 GB；删除 holdout 或当前/未来 scheduler 视频会损害结果审计或最终测试边界，因此不采用。
- Consequence: `video_provenance_v2_expanded_available.jsonl` 保留每个目标的 SHA256、字节数、分辨率、时长、source URL 和官方 ZIP 定位，并把目标 retention 标为 `deleted_after_gate`；若以后需要视觉重跑，必须按 URL/archive 定位重新获取并重新校验。

### 2026-08-20 — 保留当前 workload 设置，记录改进方向
- Decision: 不修改现有 workload 代码，保留当前 2-GPU、deadline 1.5-3×、PredOpt 手调权重的设置。改进建议记录在 `docs/experiment_improvement_notes.md`，供后续版本参考。
- Reason: 当前实验已完成大量训练（BC、PPO 5 seed、CP-RHO 矩阵），修改 workload 将导致所有结果作废。当前结果已足够支持论文关于"视频 Agent 调度场景下，简单评分函数优于复杂数学优化器"的结论。
- Alternatives considered: 立即修改 workload 重跑所有实验；时间成本高，且当前结果已形成完整故事线。
- Consequence: 下一版本或扩展实验可使用新 workload 设置（4-8 GPU、deadline 1.1-1.3×、统一目标函数）。

### 2026-08-24 — Transition profile 采用 opt-in 严格 GPU-class overlay
- Decision: 将实测 transition profile 接入现有 simulator 的 load/eviction cost resolver，但保持单 GPU slot、node runtime、prefetch 和 full-node recompute 语义不变；profile 只对已测 RTX 4080 SUPER/32760 MB class 生效，并对不匹配 topology 严格拒绝。正式 paired pilot 从冻结 150 episodes 中按源顺序筛出 55 个 `[32760,32760]` episodes；不把 24576 MB 或混合 GPU 当作已测覆盖。
- Reason: 实测 benchmark 只有单张 RTX 4080 SUPER，混合 stress input 同时包含 24576 MB GPU；放宽匹配会把未测 load/eviction 成本伪装成真实测量，破坏 profile provenance 和容量契约。
- Alternatives considered: 对所有 GPU 复用同一 profile、把 strict 改成 silent fallback、或立即补齐异构 GPU benchmark；前两者不具备证据基础，后者是独立的资源/时间扩展，不阻塞本轮单 class 接入，因此暂不采用。
- Consequence: paired simulator pilot 已通过但仅是 single-GPU-slot execution-cost overlay；后续若要覆盖异构节点，必须为每个 GPU class 补测并扩展 profile schema，不能直接解封 R8-P7 或 `T_final`。

### 2026-08-25 — WAIT 延后，先冻结 full-event Q/value contract
- Decision: 不因 Action-Value Audit 的 local-Q 结果立即加入 `WAIT`/`RESERVE`。先完成 full-event counterfactual value 的定义和 deployable surrogate objective audit。
- Reason: 在 15,364 个 common states 上，PredOpt-H5 相对 Myopic 的 local-Q top-1 仅高 0.47 个百分点，mean regret 反而略高；在 100-state/717-branch full-event follow-up 中，PredOpt-H5 与 Myopic top-1 都为 32%，而 local `TrueOpt-H5` 只有 40%，说明现有 `Q_local_H5` 不是 full-event value 的可靠代理。
- Alternatives considered: 立即扩大动作空间加入 WAIT/RESERVE；当前证据不能区分“动作空间不足”和“价值函数/目标错配”，因此暂不采用。
- Consequence: 下一步是 R8-P8c objective/Q contract audit；Prefetch、PPO、preemption、多 GPU 资源契约和 `T_final` 继续冻结。full-event follow-up 仅是 diagnostic pilot，不替代正式 1,000 集验证。

### 2026-08-26 — RiskAwareScore 仅保留为诊断候选
- Decision: 借鉴论文的“硬可行性过滤 + 当前候选集归一化线性动作评分”，新增 opt-in `risk_aware`，但不替换 Myopic/PredOpt-H5，不直接扩大到正式选择矩阵。
- Reason: 10 集 paired smoke 中 RiskAwareScore 的 mean completion 低于 Myopic 和 PredOpt-H5，但仅对 Myopic 6/10 集、对 PredOpt-H5 5/10 集获胜；同时 queue 和 GPU eviction 更高，且该版本不使用 future artifacts，因此尚不能证明全局目标或未来信息利用得到改善。
- Alternatives considered: 立即把 RiskAwareScore 提升为正式候选、加入 future-risk 权重网格、或解封 `T_final`；小样本和指标权衡不足以支持这些操作，均暂不采用。
- Consequence: 保留 `experiments/EXP-20260826_risk_aware_score/` 作为 diagnostic evidence；下一步继续做 action-score/terminal decomposition 与 full-event value contract audit，WAIT/RESERVE、PPO、多 GPU 和 `T_final` 继续封存。

### 2026-08-31 — 统一 H5 评分契约完成但不升级主策略
- Decision: 新增并完成 opt-in `aligned_predopt_h5`/`aligned_trueopt_h5` 评分契约审计；保持旧 `myopic`、`predopt_h5`、`trueopt_h5` 不变，不把 aligned PredOpt 提升为正式候选。
- Reason: 统一当前动作、H=5 后继层、缓存推进、硬优先级和 tie-break 后，aligned PredOpt 在 15,364 个 common decisions 上只比旧 PredOpt 小幅改善；100 状态 full-event audit 中两者 top-1 都为 31%，mean regret 也基本相同，10 集 smoke 中 aligned PredOpt 反而略慢于旧 PredOpt。剩余差距主要来自 H5 surrogate 与完整 episode continuation/JCT 目标不一致。
- Alternatives considered: 继续调 H5 权重、立即加入 WAIT/RESERVE、扩大到 1,000 集或解封 `T_final`；当前诊断证据不能支持这些扩大动作，因此均暂不采用。
- Consequence: `EXP-20260831_aligned_h5_score_contract` 作为已通过但非选择性的 diagnostic evidence 保留；下一步若继续，应先设计与 full-event episode objective 对齐的可部署 surrogate/terminal cost。WAIT/RESERVE、prefetch、preemption、多 GPU 契约、PPO 和 `T_final` 继续封存。

### 2026-09-01 — 先修正 H5 future topology，再判断预测器或调度器问题
- Decision: 完成 `EXP-20260901_pred_true_score_decomposition` 的 100 集同状态逐动作审计；不扩大正式矩阵，不加入新动作语义，不解封 `T_final`。下一步优先修复 Pred/True 的 future topology contract。
- Evidence: 15,364 decisions、44,690 candidates 中没有 raw/model/strict filter 差异；Pred/True scored action set、priority/tie-break、score additivity 和原策略 top-1 重现均通过。Pred/True 确定性 top-1 一致率为 65.1979%，future mean absolute error 为 38,531.9ms，高于 current 的 7,650.2ms；Qwen3-4B planner 的 future signed bias 为 -49,724.9ms。
- Reason: Pred `future_h5` 对所有候选 node 都是固定 5 个 synthetic event steps，而 True `aligned_h5_score` 展开后继 5 个 DAG layers；加权真值后继 node 数为 5.3889，Qwen3-4B 为 9.1858，后者 77.8287% 的 candidate occurrence 真值后继数超过预测 event 数。说明此前“aligned H5 只在信息源上不同”的说法过宽。
- Consequence: 当前不能把剩余问题归因于候选动作空间或“调度器不会用信息”；先使两边使用同一未来单位并重跑同一审计，再继续判断 H5 surrogate 与 full-event/JCT 的目标差异。WAIT/RESERVE、prefetch、preemption、多 GPU、PPO 和 `T_final` 继续封存。

### 2026-09-01 — 先落地 layer schema，不伪造缺失的 topology predictor
- Decision: 用户确认后仅在本地新增 `future_h5_layers` sidecar、`aligned_predopt_h5_layer` 和独立实验 `EXP-20260901_h5_layer_contract_repair`；旧 artifact/策略保持不变。因本地没有 B05 checkpoint/dataset，使用显式标记的 unary projection 做契约和回归验证，不把它当作正式 topology predictor。
- Evidence: 8,935 layer rows、26,805 scenarios、15,364 decisions、44,690 candidates；新策略与旧 aligned Pred 的逐 candidate score/action 完全相同；10 集 smoke 3 policies、160 jobs/policy、0 failed jobs；full unittest 60/60。
- Reason: 仅把旧 event steps 改名为 layers 不能制造未来分支信息；若直接从当前目标模板读取并行节点会泄漏 future truth。先固定无泄漏 schema 和评分入口，等待 train-only topology 输入，才能区分“接口问题”和“预测器缺信息”。
- Consequence: R8-P9c 保持未完成但已登记恢复点；不扩大 300/1,000 集，不连接远端，不解封 `T_final`，不加入 WAIT/RESERVE、抢占或多 GPU。

### 2026-09-01 — 先用 train-only empirical baseline 闭合 H5 topology unit
- Decision: 在不改变动作空间、GPU 语义和旧策略的前提下，用 `r7_s_train` DAG labels 拟合 conditional empirical multi-node topology baseline，在 `r7_s_val` 上生成 identity-free `future_h5_layers`，并沿同一 `aligned_predopt_h5_layer` 入口重跑诊断审计。
- Evidence: canonical evidence is `experiments/EXP-20260901_topology_predictor_baseline/metrics.json`; baseline, topology audit, aligned action audit and scheduler smoke all passed their execution gates, with no remote access and no `T_final` read.
- Reason: 该模型只使用训练分区的 DAG predecessor structure 生成 topology target，不把验证模板后缀、未来事件、successor identity、执行真值或资源真值变成预测特征；因此可以先验证多节点层契约是否真正影响评分，再决定是否值得引入 learned predictor。
- Consequence: `R8-P9c` 标记为 diagnostic completed；`aligned_predopt_h5_layer` 不升级为正式候选，不扩大 300/1,000 集，不解封 `T_final`。learned topology predictor 只有在本基线评审后才可另行确认。

### 2026-09-01 — layer baseline 改善排序但未解决 future-cost 校准
- Decision: 使用同一份当前 action audit 重跑旧 `aligned_predopt_h5` 与新 `aligned_predopt_h5_layer` 的同状态 Pred/True decomposition；保留新策略为 diagnostic，不扩大正式矩阵。
- Evidence: 两版均为 15,364 decisions/44,690 candidates，过滤、priority/tie-break、scored set 和加法契约一致；新 layer 的 tie-aware top-1 为 71.0817%（旧版 66.0570%），mean true regret 为 6,686.3ms（旧版 7,978.7ms），但 future MAE 仍为 37,327.7ms，且预测拓扑规模低于 True DAG。
- Reason: 差异只出现在 future 项；layer baseline 带来更好的相对排序，但不同模型/角色的 future 偏差方向不一致，不能把局部 ranking 改善解释成已完成的预测器校准或 episode-level 收益。
- Consequence: 下一步优先做按条件的 future-cost calibration 或 learned train-only topology predictor；不加入 WAIT/RESERVE、抢占、多 GPU，不解封 `T_final`，不把 smoke 或同状态 audit 当作正式策略选择依据。

### 2026-09-02 — topology predictor 归入 predictor block，改用预测器数据边界
- Decision: topology predictor 的正式训练/验证必须复用行为与资源预测器的同一视频数据体系：`P_dev=300`（内部 train/validation/test）和 `P_holdout_diag=40`；`S_train/S_val/T_final` 只保留给模型冻结后的调度器集成评估，不参与 topology predictor 拟合、选型或校准。
- Reason: topology predictor 与行为/资源预测器都是预测模块，应该在一致的视频级泛化边界上比较；此前使用 R7 `S_train/S_val` 会把调度器专用数据误当成预测器训练/验证数据，无法证明预测器在统一数据协议下的能力。
- Consequence: 现有 `EXP-20260901_topology_predictor_baseline` 仅保留为 scheduler-side diagnostic，不升级为正式 topology predictor。重新训练前必须在 `P_dev/P_holdout_diag` 对应视频上核验或构建完整未来 DAG labels，并通过因果输入 gate；未完成前不改代码、不训练、不解封 `T_final`。

### 2026-09-02 — P9d topology labels 使用同边界 enriched behavior anchors
- Decision: topology dataset 复用行为 role 样本的 17,303 个 P_dev 行顺序键、行为标签和视频级 split，并使用已存在的 enriched candidate 文件补齐扩展集 `source_event_id`；P_holdout 使用 `final_holdout_v1` source 行。未来 DAG labels 只从对应 raw `trace.jsonl.parent_step_ids` 重建，features 与 labels 分 split 独立保存。
- Evidence: 旧 `behavior_nn_v1/data/role_event_samples.jsonl` 与 enriched candidate 版本的 17,303 行顺序键、`current_role/next_role/current_raw_action/split/source` 全部一致；旧版本扩展 7,169 行为 `source_event_id=null`，enriched 版本 17,303 行均有事件 ID。生成数据覆盖 train/validation/test/holdout=`13,754/2,029/1,520/1,380` anchors、视频=`240/30/30/40`，features/labels 逐 split sample_id 对齐，causal forbidden-key violation=0，parent missing/forward refs=0，与所有 scheduler groups 零交集。
- Reason: 没有 raw event ID 的行为初始化行无法构造可审计拓扑目标；同一行身份的 enriched 版本只补 provenance/事件锚点，不改变行为标签或视频边界。registry 已提供 21 个截断 ID 的 canonical mapping，raw manifest 缺 hash 时计算 trace 实际 hash，禁止猜测或静默替换。
- Consequence: `data/manifests/topology_predictor_p9d_v1.json` 登记远端 canonical dataset；原始 trace、旧 R7 artifact、`S_*`、`T_final` 均保持边界不变。当前只完成数据准备/审计，learned topology predictor 的 formal experiment 与训练另行按 empirical → tabular → shared causal GRU 顺序执行。

### 2026-09-02 — P9d empirical baseline 先行并冻结 holdout 验收
- Decision: 创建唯一正式实验 `EXP-20260902_p9d_topology_empirical_baseline`；只用 P_dev/train 拟合三个预注册条件频数 schema，在 P_dev/validation 以 NLL 选择，在 P_dev/test 做诊断，并在选择冻结后评估 P_holdout_diag/holdout。最终选择 `current_full`，不接 scheduler。
- Evidence: 本地/远端编译和 5/5 单元测试通过；远端实验状态为 `passed_empirical_baseline`；metrics、model summary、validation/test/holdout predictions 的本地/远端 SHA-256 一致；预测产物行数、top-3 概率归一和 identity/edge/resource forbidden-field 审计均通过。
- Reason: 先用无训练依赖的最低复杂度模型测量拓扑支持、条件稀疏和规模偏差，才能把后续 learned 模型的收益归因到表示/模型，而不是数据或输出契约。NLL 是预注册选择指标，因此不因 history schema 在规模误差上的局部优势而改选。
- Consequence: 该实验闭合 empirical baseline gate，但结果显示 test/holdout 的未来层数和节点数仍整体低估，holdout 条件键未命中训练条件；不把它宣称为 learned topology predictor，也不解封 T_final。下一步按计划进入 tabular comparator，再比较 shared causal GRU。

### 2026-09-02 — P9d tabular comparator 完成但不通过最终 topology gate
- Decision: 创建唯一正式实验 `EXP-20260902_p9d_topology_tabular`；只用 P_dev/train 的 causal model_input 训练两个预注册 LightGBM 结构配置，在 validation 按 layer-count MAE 选择，在 test 做诊断，并在选择冻结后评估 P_holdout_diag/holdout。最终选择 `lgbm_small`；节点 prototype 继续使用 train-only 条件解码，不接 scheduler。
- Evidence: 本地/远端编译、4/4 单元测试、预测行数、top-3 概率归一、禁止 identity/edge/resource 字段审计和本地/远端 SHA-256 均通过。holdout 的 layer/node/width MAE 为 `0.1696/0.5435/0.1655`，bias 为 `+0.0290/+0.0232`，结构误差显著优于 empirical baseline；但 prototype accuracy=`0.5160`、top-3 完整 signature coverage=`0.0870`，仍有明显原型/分布偏移。
- Reason: 该阶段先验证当前可见状态能否学习未来 DAG 的规模结构，再把 prototype identity 的缺口与结构误差分开测量；按预注册 primary metric 选择 `lgbm_small`，不因 `lgbm_wide` 在 factorized NLL/shape exact 上局部更优而事后改规则。
- Consequence: tabular comparator gate 闭合，但完整 topology predictor gate 未闭合；不把 `lgbm_small` 宣称为最终模型，不读取或拟合 `S_train/S_val`，不接 scheduler，不解封 `T_final`。下一步才创建 shared causal GRU formal experiment，并以 empirical/tabular 作为同一 holdout 对照。

### 2026-09-02 — shared causal GRU 完成 learned topology diagnostic，但暂不接 scheduler
- Decision: 创建并完成唯一正式实验 `EXP-20260902_p9d_shared_causal_gru`；使用单向共享 causal GRU、分开的 behavior/topology heads，以及 `behavior_only`、`topology_only`、`shared_multitask`、`shared_first_layer_consistency` 四个预注册变体和 seeds `11/22/33`。fit 只用 `P_dev/train`，validation 选 epoch/变体，test 做诊断，holdout 只在冻结后读取。
- Evidence: 远端 12 个 run 全部成功；本地/远端 source/config/metrics/run-manifest hashes 一致；PyTorch forward/loss/backward 4/4 通过；36 个 prediction files 的 row coverage、top-3 概率归一、identity/edge/resource-truth leakage 和输出 shape 审计均通过。validation 结构 primary score 选择 `topology_only=0.1284`；holdout 上 `shared_multitask` 的 layer/node/width MAE=`0.1442/0.3995/0.1378`，行为 role/family/joint=`0.9630/0.6901/0.8449`，优于同实验 behavior-only 的 `0.9621/0.6679/0.8355`。
- Reason: shared encoder 确实能从同一可见 prefix 学到多节点未来 DAG 结构，并在当前 holdout 上优于 tabular comparator；但 validation 选择与 holdout 平衡排序不完全一致，不能使用 holdout 事后改选。首层 consistency loss 没有稳定超过 plain shared multitask，因此不增加该交互复杂度。
- Consequence: `EXP-20260902_p9d_shared_causal_gru` 标记为 `passed_diagnostic`；`topology_only` 是按预注册 topology primary metric 的选择结果，`shared_multitask` 仅作为更有潜力的平衡候选记录，不直接升级为调度器输入。下一步先做冻结候选的 future-cost/resource calibration 和分层误差审计，再在 `S_train/S_val` 运行只推理 contract/smoke；不读取/解封 `T_final`，不改变 raw trace、旧 R7 artifact、WAIT/RESERVE、抢占或多 GPU 语义。

### 2026-09-03 — 完成 all-future-node content P1，但不冻结 scheduler 候选
- Decision: 创建并完成唯一正式实验 `EXP-20260903_p9d_future_content_multitask`；使用同一 shared causal GRU 和分开的 behavior/structure/content heads，比较 `N0/A/B/D × 3 seeds`。A 按预注册 validation `layer_count_mae + full-H width_vector_mae` 选中；B/D 的 next-step auxiliary 保留为诊断，不据 holdout 事后改选。
- Evidence: Stage 0 的 18,663 行标签/泄漏门禁通过；12 个远端 run 全部完成；本地/远端测试、实际数据 forward/loss/backward smoke、62 个非 checkpoint 产物 SHA-256、36 个 prediction 文件契约和边界审计均通过。A 的 validation structure score=`0.1272`；D 的 holdout structure exact/layer/width=`0.8244/0.1437/0.1371`，但 holdout 不参与选择；B/D validation next-family F1=`0.7904/0.7829`，低于 N0=`0.8075`。
- Reason: 该实验先验证“共享因果表示 + 未来结构/节点内容 + next-step auxiliary + soft topology conditioning”的输出和训练链路，再分离结构误差、内容分布偏移与行为辅助任务影响。保持固定 split/seeds 能与前序 P9d 实验直接比较；不把 holdout 最优结构结果改写成选择结果。
- Consequence: `EXP-20260903_p9d_future_content_multitask` 标记为 `passed_diagnostic`；A 仅是 validation-selected structural reference，D 仅是 holdout structural diagnostic，尚无模型升级为 scheduler 输入。下一步只做 train-only content/future-cost/resource calibration 和分层审计；在校准通过前不推理到 `S_train/S_val`，不读取/解封 `T_final`，不改变 raw trace、R7 artifact、WAIT/RESERVE、抢占或多 GPU 语义。

### 2026-09-03 — reduced future-node role/family decoder 完成，但不冻结 scheduler 候选
- Decision: 创建并完成唯一正式实验 `EXP-20260903_p9d_future_role_family_multitask`；在同一 P9d predictor 数据边界上把未来节点内容从四字段收缩为 `role + action_family`，保留 `next_role + next_family_if_execute` 辅助行为头，比较 `N0/A/B/D × 3 seeds`。按预注册 validation `layer_count_mae + full-H width_vector_mae` 选择 `B`。
- Evidence: Stage 0 覆盖 18,683 行，split/causal/leakage errors=`0`；12 个远端 run 完成；真实 8-row forward/loss/backward smoke 和远端 P9e 契约测试 `4/4` 通过；36 个 prediction files、62 个非 checkpoint 输出的 manifest hash 和独立输出契约审计均通过。B 的 validation structure score=`0.12847`、future role+family F1=`0.72873`；frozen holdout structure score/exact=`0.27986/0.82923`，future role/action-family F1=`0.90151/0.40360`。共同 action-family 在 validation/test/holdout=`0.50666/0.49099/0.40360`，未超过旧四字段 B 的 `0.52043/0.50889/0.40982`。
- Reason: 用户希望未来节点只保留能直接服务拓扑和行为语义的两个字段；该消融可以判断减少输出字段是否降低训练负担，同时保持结构头和单步行为辅助头可审计。两字段总 macro-F1 与四字段总 macro-F1 的标签集合不同，不能直接作为整体提升；共同 action-family 才可作谨慎对照。
- Consequence: `EXP-20260903_p9d_future_role_family_multitask` 标记为 `passed_diagnostic`；B 是当前简化结构参考，但不宣称 action-family 已改善，也不把它接入 scheduler。下一步先做 train-only content/future-cost/resource calibration 和分层误差审计；不使用 `S_train/S_val` 拟合或选择，不读取/解封 `T_final`，不改变旧四字段实验、raw trace、R7 artifact、WAIT/RESERVE、抢占或多 GPU 语义。

### 2026-09-03 — P9f 结构诊断通过，但 calibration/resource gate 关闭
- Decision: 创建并完成唯一正式实验 `EXP-20260903_p9f_predictor_acceptance_audit`；只加载冻结 P9e reduced `B`/seeds=`11/22/33`，不重新训练、不改选模型、不接 scheduler。temperature 只在 `P_dev/train` 拟合，holdout 只在既有选择冻结后读取。
- Evidence: Stage 0、远端 `4/4` 回归测试、编译、真实 forward smoke、JSON/有限值/manifest-output 审计和远端/本地逐文件哈希均通过。holdout learned B 的 layer/width/node MAE=`0.1442/0.1357/0.3807`，均优于 empirical 的 `1.9080/0.9274/4.6370`，exact=`0.8292`。但 train-fit temperature 在 layer/width/next-role/next-family 四头的 holdout NLL 分别由 `0.33361/0.42303/0.13095/0.80845` 变为 `0.33446/0.42390/0.13171/0.81524`，所以 raw probabilities 保留。role/family P50 runtime proxy 的三 seed holdout coverage=`0.8449/0.8435/0.9152`，top-1 runtime MAE 均值=`118,798.7 ms`，系统性低估长尾。
- Reason: P9f 把“结构能否预测”和“结构能否直接转成 scheduler future cost”分开验收。当前 reduced node contract 没有 `model_id/node_type/input_scale/cold_warm`，因此 load/memory 无法辨识；不应把 coarse role/family runtime proxy 或 incomplete coverage 当作可部署 resource predictor。
- Consequence: P9f 标记为 `completed_diagnostic`；结构比较 gate 通过，但 calibration 不提升、resource/model-aware/scheduler gates 均关闭。下一步需另行设计 resource-aware future-content 或 conditional aggregation 实验；在其通过前不做 `S_train/S_val` 推理集成，不加入 WAIT/RESERVE、抢占或多 GPU，不解封 `T_final`。

### 2026-09-07 — P9d 数据集 v2 重建（依赖重建 + 最长路径分层 + workload_scale）

- Decision: 修改 `scripts/build_p9d_topology_dataset.py` 并重建 v2 数据集到外接盘：step 级父依赖取 last-event 语义、分层改最长路径深度、标签加逐节点 workload_scale、加 termination_status 删失语义、宽度合并 4+；schema 版本保持 v1（纯加法字段），v2 身份由目录 + manifest + 契约字符串承载；旧 v1 数据集保留不动。
- Reason: 网页评审把依赖重建定为 P0 地基问题；全量审计确认 4.22% 层内边、17.4% 截断混淆、缺 input_scale 字段；v2 审计层内边清零、行数与 v1 逐 split 一致。
- Alternatives considered: 只修分层不修边（会在错图上算出更自信的错 DAG）；删除宽度 5 样本（会扭曲拓扑分布，改为合并）。
- Consequence: v2 登记为 `data/manifests/topology_predictor_p9d_v2.json`；独立 reviewer 门因认证 token 过期未能执行，记为 pending，待恢复后补审；R0 上界实验仍需等本契约冻结确认后才可启动。

### 2026-09-08 — 工作区迁移到外接盘并双向对账同步
- Decision: 以后所有操作默认在 `/Volumes/Lenovo/scheduler` 执行；本地仅保留控制面编辑入口。本轮做双向对账：本地新增 493 个文件复制到外接盘；外接盘更新的 7 个脚本 + 600 视频 provenance 清单（含 8/17 P_dev 清理记录）回填本地；6 个本地新版代码/文档覆盖外接盘旧版（经 diff 确认本地严格新增）；7 个同字节大文件跳过复制。
- Reason: 外接盘已有完整数据/模型/实验产物且空间充足；对账发现外接盘有本地缺失的远端更新，直接覆盖会丢记录，故改为按文件判向。
- Alternatives considered: 单向本地覆盖外接盘；会回退 provenance 清单和远端脚本修正，故不采用。
- Consequence: 46 个视频/mp4 文件（约 3.37GB，含 staging 临时目录）暂缓复制，待用户明确是否需要上外接盘；远端同步因拒连继续 pending。

### 2026-09-04 — 后续实验迁移到外接盘独立执行副本
- Decision: 将远端 `/root/autodl-tmp/scheduler` 的完整内容同步到 `/Volumes/Lenovo/scheduler`，包含代码、数据、模型、实验产物和 `.project/`；同名内容以远端为准，不使用 `--delete`，目标目录独有文件保留。后续实验默认在外接盘副本执行，不依赖远端计算。
- Evidence: 内容级增量同步实际传输 `501` 个文件、约 `58,382 kB`；同步阶段对远端 `45,983` 个 regular files 做 SHA-256 验收，`0` 个 mismatch；六个原始控制面文件和最新 P9d/P9e/P9f 实验目录均已核对。外接盘剩余空间约 `782GiB`。
- Reason: 外接盘已有完整历史副本和足够空间，继续依赖远端会增加网络、主机状态和连接中断风险；保留远端为源/备份可以维持恢复能力，同时不删除外接盘中的历史额外产物。
- Consequence: 后续新实验先在 `/Volumes/Lenovo/scheduler` 准备本地 Python/PyTorch 环境并做导入、checkpoint 和数据路径 smoke；远端不再作为默认执行位置。`P_dev/P_holdout_diag`、`S_train/S_val`、`T_final`、WAIT/RESERVE、抢占和多 GPU 边界不变。

### 2026-09-08 — ChatGPT Web 送审改用中文
- Decision: 后续向绑定会话 `6a9822da-e278-83e9-9c1a-675923acda0e` 发送的 brief 改用中文撰写，并明确要求对方用中文回答（含 VERIFIED/INFERENCE/UNVERIFIED 标签保留英文原词以便审计）。
- Reason: 用户要求中文可读性；既有英文 brief/回答增加理解成本。
- Alternatives considered: 中英双语 brief；会使输入变长且回答语言不可控，因此不采用。
- Consequence: `.scratch/chatgpt_*_brief.md` 与 `save-report` 产物默认中文；控制面引用原文时保留关键英文术语对照。

### 2026-09-10 — P9d v3 标签契约冻结：经验证串行控制流 + 嵌套合并 + 复合资源签名

- Decision: 依据 step-structure 审计与网页评审（MODIFY 后接受），把 P9d topology 标签重建升级到 v3.1 并冻结：节点集 = supported 事件 − `run_control` − 嵌套答案调用；边 = 经验证的顶层串行控制流直接执行依赖（Seriality gate fail-closed）；嵌套答案调用合并进其 Summarizer 工具节点（`nested_calls` + composite resource signature，`R_node = R_total`，禁止重复核算）；失败 planner 与 retry 保留为链节点；最终 `answer` 定为 terminal/non-resource 标记（不占 H=5 槽位）；width 恒为 1 并降级为 deterministic compatibility field；研究表述改为 bounded future execution-trace forecasting。builder 增加 `--label-policy v2|v3`（默认 v2 保持旧行为），v3 在新目录 `results/processed/topology_predictor_p9d_v3` 重建并登记 `data/manifests/topology_predictor_p9d_v3.json`；契约文档 `docs/p9d_topology_label_contract_v3.md`。
- Evidence: 12/12 本地 unittest 通过（v2 8 项 + v3 4 项）；重建 1,360 run 全部通过 gates（multi-tool=0、非上一步父引用=0、步跳变=0、retry 邻接 502/502、tool step id 6,630/6,630、planner-tool 名称 6,630/6,630、嵌套合并 482、terminal marker 1,360）；行数/split/视频/runs 与 v2 完全一致（13,754/2,029/1,520/1,380；240/30/30/40；988/144/108/120）；features 解压 SHA-256 与 v2 四 split 逐字节一致；v2→v3 前驱修正 7,117（tool:tool→planner 5,565、generate:tool→tool 1,113、planner:tool→planner 365、其余 74）；label 侧 width 直方图 = {1: 65,275}；答案生成嵌套在未来节点中出现 1,548 次、全部 `resource_applicable=true`。评审记录 `docs/research/2026-09-10_p9d_v3_contract_review.md`（`status=valid`）。
- Reason: v2 的“父步骤→最后事件”把同一步 planner 与其工具变成假并行（train 13,754 行中 5,856 行含该模式），width/层结构被系统性高估；采集器源码 + 2,008 条 run 审计证明当前三类 workflow 为串行控制流，应按真实执行语义重建。网页评审明确要求把契约建立在 serial-control invariant 上、并冻结 resource node ontology，否则 R0 的 oracle 资源上界不可解释。
- Alternatives considered: 保留 v2 规则（标签结构性偏差，已证伪）；按“文件顺序”直接建边而不做独立 seriality 验证（评审否定：相邻≠因果，需 fail-closed）；把嵌套调用保留为独立节点（会重复核算 runtime，资源语义错误）；把 width 继续当学习任务（无信息量）。
- Consequence: predictor 侧标签契约 v3.1 冻结，R0 可在 v3 标签上解冻设计；v1/v2 与所有既有实验只读保留；R7/R8 scheduler 侧 `job_templates`/future artifacts 的同源问题另行审计，首次 scheduler integration 前必须过独立 gate；`T_final` 继续封存；holdout 表述改为 contract-integrity inspected（不得再声称 completely untouched）。

### 2026-09-11 — R0 设计冻结：oracle-signature 经验资源上界（v3 ACCEPT）

- Decision: 冻结 R0 设计，正式名称 **oracle-signature empirical resource ceiling**，实验 ID 拟 `EXP-20260911_p9d_r0_oracle_signature_ceiling`；设计文档 `docs/p9d_r0_oracle_signature_ceiling_design.md`。要点：只在 v3.1 的 15,481 个 resource-applicable 节点上做 oracle 签名资源上界；runtime 为主 gate（`PB_primary=mean(PB.50/.90/.95)`，raw scale，唯一 Core baseline = exec_class 条件中位数，改善 ≥15% 且 video-cluster bootstrap CI 下界 >0）；load 按 hurdle（`load_ms=0` 为合法 no-load；occurrence 用 LightGBM binary + Brier + 固定 logistic 参考）；memory 限定 instrumented 子集（worker 每请求 reset 的节点级峰值）；`residency_state` 因 `model_resident_before` 100% 缺失不可用，C4 用 `prefix_model_reuse` proxy；统计以 video-cluster bootstrap 为主 + per-run/per-video macro；C0–C4 消融、阈值/网格/校准 gate 全部预注册；holdout 一次性验收，评估后不得再改 contract/模型/特征。
- Reason: 网页评审 v1→v3 先后指出并关闭 5 个 P0（load→regime 泄漏、G3 覆盖率数学错误、memory 语义、`load_ms=0` 语义、load 计数口径）与全部 P1，最终裁决 “R0 DESIGN v3 ACCEPT（可冻结）”。R0 是 J 系列（联合训练）前的接口可辨识性 gate，必须在 v3.1 标签契约上先行，否则资源映射误差与签名预测误差无法分离。
- Evidence: 评审记录 `docs/research/2026-09-11_p9d_r0_design_review{,_v2,_v3,_final}.md`；load 计数在 v3.1 节点本体上重算（planner 7,211 = 67/5,960/1,184；tool 7,078 = 0/6,742/336；post-loop generation 1,192 = 9/530/653；合计 15,481；互斥三项合计成立；nested 482 排除）`.scratch/v3_load_accounting.{py,json}`；采集器证据（`load_ms = model_load_ms if request_index==1 else 0.0`）与 worker 证据（`reset_peak_memory_stats()` per request）。
- Alternatives considered: 跳过 R0 直接 J 系列（接口误差与预测误差混淆）；把 `load_ms=0` 当哨兵剔除（改变目标语义）；使用 residency 特征（数据不可用）；memory 全量外推（仅约 66% 覆盖且按类非随机缺失）。
- Consequence: R0 设计冻结；下一步注册实验、更新 `.project/EXPERIMENT_GATE.json` 并跑 Stage 0 + smoke（本地 CPU，<10 CPU-min）；不训练模型、不接调度器、不触碰 `S_*/T_final`；holdout 一旦打开即失去再改选资格。

### 2026-09-11 — R0 完成：runtime/load 可辨识（Core GO PASS），memory 降级 descriptive

- Decision: R0 按冻结设计执行完毕；裁决 **Core GO PASS**（runtime 主 gate + load 次要），memory 因校准不达标按契约降级为 descriptive-only；不做任何事后模型/特征调整；holdout 已一次性使用，失去再改选资格。
- Evidence: validation 改善 74.9%（CI [0.7035, 0.8239]），test 82.1%，holdout 20.2%（CI [0.099, 0.815]）；校准 `max_τ` = 0.0338/0.0461/0.0161（≤0.05）；消融 C0→C4 = 941.4→809.2→809.6→809.6→**380.0**（`workload_scale` 全池零方差导致 C3 持平）；load occurrence Brier holdout 0.0013（logistic 参考 0.096），duration PB val 41.7ms；memory 校准 val 0.127 → descriptive（覆盖 9,816/15,481）；holdout 2 个 stall（45min/68min）主导 Inclusive 长尾，Normal 子集 PB 319.8ms。完整结果 `experiments/EXP-20260911_p9d_r0_oracle_signature_ceiling/RESULT.md`，metrics sha256 `832ae291…`。
- Reason: 冻结设计规定 Core GO 只看 runtime（+load）；memory 不可解释时自动降级；R0 的定位是接口可辨识性 gate，不是方法比较。
- Alternatives considered: 因 memory 不达标暂停整个 R0（契约已给出降级路径）；剔除 holdout 长尾样本（契约禁止删除 stall，改为 Inclusive/Normal 分别报告）；事后加入 token/frame 特征重跑（会破坏预注册与 holdout 身份）。
- Consequence: J 系列可在限定范围启动（runtime 为主、load 为辅、memory/tail 不在本期可辨识声明内）；接口扩展候选（真实 token 长度、帧数/分辨率、cold/warm 状态、batch 上下文）列为下一轮独立议题；holdout 评测后不得再改 contract/模型/特征。

### 2026-09-11 — J 系列设计冻结（v3）：五变体隔离 + runtime primary + NI manifest + test 判定

- Decision: 冻结 J 系列设计 v3（`docs/p9d_j_series_design.md`），进入实现阶段前不再改动架构。要点：五变体（J0 probe / B1 两段式 / J1 / J2 / J3）+ OracleAttr 仅诊断；全部变体共享同一 backbone（attribute-only）初始化与统一 attribute-head manifest；资源监督测试 = 共享表示适配（J0→J1）、predicted-interface 信息（J1→J2）、同软接口联合适配（B1→J2）、端到端接口重塑（J2→J3）。主指标恢复为 runtime 单一 primary（`RuntimeQScore = mean(PB.50/.90/.95)`，与 R0 相同）；load occurrence Brier 与 duration 分位数分开作关键次要；Core success 与 load NI **在冻结后的 P_dev/test 上一次性判定**（train 拟合 → validation 仅 checkpoint/配置选择 → test 判定）；NI 采用逐 endpoint 的 metric+direction+δ manifest（越低越好者用 `CI_upper(Δ)<+δ`，越高越好者用 `CI_lower(Δ)>−δ`）；checkpoint 先 NI-feasible 再 argmin RuntimeQScore；width 退出正式训练目标（structure = future length/termination）；T=1 不做温度调参；第一轮不用 GradNorm/PCGrad；holdout 完全退出正式决策。
- Reason: 网页评审两轮（4 P0 + 3 P0）全部关闭；第二轮发现的 CI 方向错误与“成功判定不得复用 validation”属数学/流程错误，已按评审修正为 v3；评审明确表示除此之外无新的架构阻塞。
- Alternatives considered: 继续用旧 ResourceQScore（与 hurdle load 不兼容，已废弃）；在 validation 上宣称成功（selection optimism，评审否决）；首轮加 PCGrad/GradNorm（会同时改变优化器，破坏 J2→J3 隔离）；保留 width 作为学习目标（v3.1 下 width 恒为 1，无学习意义）。
- Consequence: 下一步在本地实现 J：安装 torch（当前缺失；本机 RTX 3060 6GB 可用）、构建 J 数据管线（v3.1 标签 + 实测资源关联）、依次跑 J0 → backbone → B1 → J1/J2/J3（3 matched seeds，预计 <2 GPU-hour）；不接 scheduler、不读 `S_*/T_final`、holdout 保持不可见。

### 2026-09-11 — J 系列正式判定完成：无 Core GO；J2/J3 runtime 显著优于 B1

- Decision: J 系列（topology x resource 联合训练）按冻结 v3.1 协议执行并完成一次性 test 判定。正式结论：**无 Core GO**（J2/J3 的 load-duration 未过 5% 非劣）；J2/J3 的 runtime 主端点显著优于 B1（3/3 seeds，CI 上界<0）；J0/J1 显著更差。
- Evidence: `experiments/EXP-20260911_p9d_j_series_joint_resource/RESULT.md`；test_eval.json（冻结，未重跑）；审计 `docs/research/2026-09-11_j_result_audit.md`（hash 闭环，AUDIT 一致）；评审 `docs/research/2026-09-11_j_result_review.md`（P1 已修：聚合规则+seed22 披露）；J3 回归检查 max_abs_delta=0。
- Reason: 预注册规则要求 runtime CI_upper<0 且 load 双端点非劣；J2/J3 仅在 load-duration 上以 +5.2%/+5.3%（逐 seed 匹配）越界，按规则不得宣称 Core GO；不得事后调整容差。
- Consequence: 联合训练线以“runtime 显著收益 + load-duration 贴线未过”的负结果存档；test 已消耗一次，后续确认性声明需新 sealed 评估集。

### 2026-09-11 — J 线收束（用户决策 A）：J4 负结果 + 分布重分配归因，不再做修复实验

- Decision: 用户选择收束 J 联合训练线。J4（duration 解耦：J4a 独立分支 / J4b 冻结共享特征）已按 frozen v2 执行并**否定**；随后完成“为什么退化”的收尾诊断；**不再执行** J5 类修复实验（条件化 hurdle/model 门控/分布参数化/校准），除非未来重新立项。
- Evidence: `experiments/EXP-20260911_p9d_j4_duration_branch/RESULT.md`（J4a 0/3 可行；J4b 仅 ep7-9 可行且 runtime 1187/1285 vs B1 912/895；H1/H2 validation CI 全 fail）；机制收尾 `docs/research/2026-09-11_j_mechanism_analysis.md` §6（τ=.90 变差、τ=.95 3/3 seeds 变好、中位数不变且 J3 校准更好；VL-3B 略差、VL-8B 2/3 seeds 改善；退化结构随 seed 不稳定）；GPT/文献 `docs/research/2026-09-11_j4_literature_and_gpt_ideas.md`。
- Reason: 退化是分布内的拟合重分配（几十 ms、贴线、无单独坏路径），J4 解耦修不好具有必然性；继续修需要系统侧缺失变量（model residency/cold-start、workload 量级），属新一轮数据/系统工作，不在当前线内。
- Alternatives considered: J5a 条件化 hurdle + model-class 门控（成本低但预期收益受限于缺失变量）；J5b 分布参数化；J5c 纯校准（天花板低）；均记录在文献笔记中备查。
- Consequence: predictor 资源线当前定格在“R0 上界 + J3 runtime 收益 + load-duration 权衡边界”；论文/交付叙事按 trade-off 边界撰写；下一阶段的候选方向（数据补采 / 证据综合 / 调度器主线 R8-P4-P5）另行决策。
