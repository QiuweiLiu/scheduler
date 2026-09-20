# Decisions

> 决策日志。2026-09-10 之前的历史决策全文已归档到 `.project/archive/2026-09-15_pre_compact/DECISIONS.md`；
> 下方索引保留每条决策的日期与标题（不删除记录）。

## Archived index (2026-08-13 → 2026-09-08)

- 2026-08-13 — Project OS 与源码布局迁移
- 2026-08-13 — Legacy trace collection 不覆盖
- 2026-08-15 — 远端只同步控制面并采用加法式目录规范化
- 2026-08-15 — 640×360 先做 pilot，不直接替换 R5 原始视频
- 2026-08-15 — 视频 provenance 先于清理与派生 collection 冻结
- 2026-08-15 — 下载采用本地获取后校验上传
- 2026-08-16 — R6 使用 causal-v2 与 engine-only truth
- 2026-08-16 — 后续预测器与调度实验采用最新 NN 数据口径
- 2026-08-17 — R7 采用600视频六组边界和最小主线矩阵
- 2026-08-17 — R7 pilot 允许保留可恢复解析错误
- 2026-08-17 — 明确授权清理旧 P_dev 原始视频
- 2026-08-20 — 保留当前 workload 设置，记录改进方向
- 2026-08-24 — Transition profile 采用 opt-in 严格 GPU-class overlay
- 2026-08-25 — WAIT 延后，先冻结 full-event Q/value contract
- 2026-08-26 — RiskAwareScore 仅保留为诊断候选
- 2026-08-31 — 统一 H5 评分契约完成但不升级主策略
- 2026-09-01 — 先修正 H5 future topology，再判断预测器或调度器问题
- 2026-09-01 — 先落地 layer schema，不伪造缺失的 topology predictor
- 2026-09-01 — 先用 train-only empirical baseline 闭合 H5 topology unit
- 2026-09-01 — layer baseline 改善排序但未解决 future-cost 校准
- 2026-09-02 — topology predictor 归入 predictor block，改用预测器数据边界
- 2026-09-02 — P9d topology labels 使用同边界 enriched behavior anchors
- 2026-09-02 — P9d empirical baseline 先行并冻结 holdout 验收
- 2026-09-02 — P9d tabular comparator 完成但不通过最终 topology gate
- 2026-09-02 — shared causal GRU 完成 learned topology diagnostic，但暂不接 scheduler
- 2026-09-03 — 完成 all-future-node content P1，但不冻结 scheduler 候选
- 2026-09-03 — reduced future-node role/family decoder 完成，但不冻结 scheduler 候选
- 2026-09-03 — P9f 结构诊断通过，但 calibration/resource gate 关闭
- 2026-09-07 — P9d 数据集 v2 重建（依赖重建 + 最长路径分层 + workload_scale）
- 2026-09-08 — 工作区迁移到外接盘并双向对账同步
- 2026-09-04 — 后续实验迁移到外接盘独立执行副本
- 2026-09-08 — ChatGPT Web 送审改用中文

## Active decisions (2026-09-10 → )

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

### 2026-09-14 — validation 切 700 dev / 300 frozen confirm，冠军只在 confirm 跑一次

背景：1,000 条 validation episodes 已被多轮消费器探索使用，存在过拟合风险。
决策：固定切分（seed 20260914，manifest `data/manifests/validation_split_dev700_confirm300.json`）；
所有新消费器调参只在 dev700 报告；最终冠军只在 confirm300 运行一次并单独报告。运行器新增 `--episode-ids-file`。

### 2026-09-14 — 消费方式冠军 = q95（逐步骤 runtime p95 求和）

背景：16 个消费变体在同一预测输入上比较；场景采样+CVaR、共单调、自适应风险等均更差。
决策：以 `predopt_h5_q95` 为当前冠军配置（H5 预测器 + q95 消费）；`predopt_h5_rt95` 视为等价实现（仅 runtime 尾部）；
不再为消费器家族继续投入（除非有新证据）。机制结论：价值来自 runtime 尾部的完全相关式惩罚，load 维度与收益无关。

### 2026-09-17 — oracle 对照存在 key 形状混淆：撤回两处 oracle 差值，新增 same-shape 消费者臂（opt-in）

- Decision: 认定 `baseline_matrix_status` 中"q95 胜过两个 oracle 臂"为 confounded 并**撤回该表述**；
  新增三个同 key 形状的 opt-in 臂 `sameshape_h5_p50` / `sameshape_h5_p95` / `sameshape_h5_truth`；
  **既有 legacy 臂（`predopt_h5`、`trueopt_h5`、`predopt_h5_q95`、`aligned_*`）不做任何修改**，历史数字保持可复现。
- Evidence: `src/tracing/analysis/workload_v02_simulator.py` 中 greedy 预测族（`predopt_h*` 与 `_q95/_lam` 分支）
  priority 均为第 1，而 `trueopt_h*` 分支把 `limited_future_truth_cost` 放第 1、priority 第 2 且**不含当前节点**；
  `aligned_h5_policy_key()` 文档要求 priority 先于 cost。`tests/test_sameshape_consumer.py` 断言 `sameshape_h5_p95`
  与 `predopt_h5_q95` 在同一 episode 上 summary 逐字段相等；详见
  `experiments/EXP-20260911_forecast_aware_scheduling/PHASE16_SAMESHAPE_CONSUMER.md`。
  **勘误（2026-09-17，审阅后复核）**：本条曾写"三臂 priority 分别第 1/第 2/第 3 位"，其中"`predopt_h5` priority 第 3"
  是错的——第 3 种形状只存在于 MPC/rollout 路径（`_predicted_candidate_key()`），不在 `choose_action` 的 greedy 派发里。
  混淆只有一处（trueopt_h5 vs 预测族），撤回旧"q95 胜 oracle"的结论不变。

### 2026-09-17 — runtime/load 契约缺陷认定（P0）：load 被部分重复计入；**修法待定，不擅自改**

- Decision: 认定"在 runtime 之上再加 load 项"是系统性契约缺陷；**本轮只审计、不修改任何消费者、不重跑**。
  修改 `_*_step_cost` 会改变冠军定义与全部已发布数字，属受保护的科学定义变更，须单独批准 + 新门禁。
- Evidence: `Node.compute_ms = runtime_ms - load_ms`；`build_j_dataset.py` / `j_series_common.py` 的 runtime 头目标
  是原始 `runtime_ms`（含 load）；`pack_j_predictor_artifacts.py` 的 runtime/load 分位数来自两个独立头且无扣减；
  `j_validation` 上 546 个 `load>0` 步中 **0 个** `runtime_ms < load_ms`，`load/runtime ∈ (0, 0.55)`。
  影响量化：冻结 artifact 中 7.69% 的 GPU 未来步被 `occ≥0.5` 门控并高估约 40%（load p95/runtime p95 p50=0.402）。
  详见 `experiments/EXP-20260911_forecast_aware_scheduling/PHASE17_RUNTIME_LOAD_CONTRACT_AUDIT.md`。
- Reason: 若 runtime 标签已含 load，则再加 load 是重复计数；这会让"冠军 = 逐步骤 runtime p95 求和"的语义描述失真。
  **注意**：这**不**混淆"load 尾部无用（ld95 +15.1k）"——ld95 走 `_split95_step_cost(load_lam=1)`，是纯 load 信号；
  且多出的 load 项近乎决策中性（rt95 vs q95 仅 +11ms）。初版曾写 ld95 结论受混淆，已撤回。
- Alternatives considered: (a) 全部消费者改为 runtime-only；(b) runtime 头改为 compute-only 标签（需重训/重打包）；
  (c) 只在 `predopt_h5_q95` 做最小修作为 Phase16-B 的干净基线。**尚未选择**。
- Consequence: 在选定修法并重跑 dev700 之前，"load 维度无价值"与"q95 = runtime 尾部"两条表述**暂停对外使用**；
  `predopt_h5_r95` / `predopt_h5_r50` 族不受此缺陷影响，可作干净参照。
- Reason: `r95=180.4s` 与 `trueopt_h5≈183.8s` 的差值同时混合信息源、排序 key 形状与 future 项定义，
  不能支撑任何"预测优于真值"结论；先对齐消费函数形状，再回答"真值为何打不赢预测值"。
- Alternatives considered: 直接修 `trueopt_h5`/`predopt_h5` 的 key 顺序（会改变已发布 baseline 语义、需重跑全部矩阵，已拒绝）；
  使用既有 `aligned_trueopt_h5`（其 Pred 用 5 个 synthetic event step、True 用 5 个 DAG layer，H 语义不同，已判定过宽）。
- Consequence: Truth-SameConsumer 成为下一步正式实验（未跑，需门禁）；不可消除的剩余差异（预测链 vs 真后继链）
  必须在论文中显式声明；`trueopt_h5` 历史结果只能标注为 "legacy key shape"，不得当作 oracle 上界。

### 2026-09-18 — Phase 21 结案（预注册负结果）：**task_context 缺失不是 S_* 域外失准的原因**；不重生成、不重跑

- Decision: 在 Windows/GPU 机执行 16 视频 paired masked/fixed pilot（`scripts/preprocess/phase21_pilot.ps1`，23 秒）。
  **判定 PILOT FAILED**：四条预注册 calibration gate 全部不通过；按预注册规则**不执行全量 S_* 重生成**，
  不重跑任何调度族，不触发 Phase 18/20 重做链。**预测器、artifact、v03 workload、`T_final` 均未改动**。
- Evidence: 管线 gate 全 PASS——975 anchors（两侧一致，576 skipped）、anchor id 顺序/集合 sha 一致、
  **`prefix_hash` 975/975 完全一致**、`source_trace_sha256` 一致、`unknown_rate(domain/official_task_type/sub_category)=0`
  且 `source_coverage=1.0`（masked 对照 1.0/0.0）、`task_context_failures=[]`、checkpoint SHA256 未变
  （`0ee8ded4…`）、差异仅限三字段（2,925/2,925 全变）+ 审计块 → **干净单变量对照**。
  校准门 FAIL：fixed p50 `R=0.3971`（门限 [0.70,1.30]）、cov `0.3960`（[0.45,0.65]）、p90 cov `0.8443`（[0.85,0.97]）、
  p95 cov `0.9157`（[0.92,0.995]）；paired Δpinball p50 **+51.59ms [+27.23,+74.42]（显著更差）**，
  p90 +9.12 [−25.80,43.60] 与 p95 +4.75 [−23.45,32.88] 不显著 → "pinball 降 ≥10%" 亦 FAIL。
  报告 `experiments/EXP-20260911_forecast_aware_scheduling/PHASE21_TASKCTX_PILOT_REPORT.md`。
- Reason: 修复被正确实现但未改善校准 → 失准是**结构性**的（逐步 p50 求和 vs 聚合真值的错配、执行期分布偏移），
  不是特征管线遗漏。与 Phase 10（content 无可测效应）、Phase 18（Σp50=0.456× 真值、Σp95=4.9×）、
  Phase 20-B（oracle-truth 仅比 p95 好 1.3s）互证 → **该负结果支撑 C2**（保守尾部聚合是纠偏机制），写入论文。
- Alternatives considered: 继续全量重生成（作废全部调度结果且预期无收益，已否决）；
  顺带修 `temporal_scope`/`planner_model_id`/`model_stack_id` 的 OOV（值是冻结 fallback，超出本问题范围，
  已登记为残余同类问题，未修）。
- Consequence: 预测侧修补线关闭；下一步需用户在 A/B/C 中选择（A=转入已批准的 P2 sweep + v03-confirm300，
  推荐；B=Phase 17 契约修复，会重定义冠军且需预注册 δ；C=H10-lite/cache-aware/真实 replay）。
- 附带修复（端到端路径缺陷，Mac 侧 11/11 单测未覆盖；`scripts/build_sstar_predictor_anchors.py`、
  `scripts/analysis/analysis_phase21_taskctx_calibration.py`）：
  ① `oov_rate_all` KeyError → 键回退；② `required_modalities` KeyError → 该字段为**列表型**并经
  `vocabs.modalities`（而非 `vocabs.context`）编码，新增独立审计块，且字面量 `"unknown"` 视为缺失而非类型错误；
  ③ `resolve_artifacts` 的 FileNotFoundError → packer 写 `j_future_h*` 于根目录而脚本找 `b05_future_h*` 于
  `prediction_artifacts/`，改为兼容两种命名/两种布局并回退父目录。
  另记：masked baseline 记录 n=3,226 vs 实测 3,275（+1.5%），但 R/cov 近乎逐位复现 → 属 slot 过滤口径差异，
  不影响判定；**下次需在脚本内记录 slot 对定义**。

### 2026-09-18（同日）— 一次误报的撤回：S_* context 输入**没有** null 遮蔽缺陷；Phase 21 总括结论**恢复成立**

- Decision: 用户质疑负结果并索要代码审查后，我先提出"`baseline` / `model_stack_id` / `planner_model_id` 被 null 遮蔽
  → 这三个字段 100% 变成 UNK"的 P0，并据此收窄了 Phase 21 结论。**该 P0 是误报，已撤回**；
  Phase 21 的总括结论（恢复三个 registry 内容字段不改善校准，且**无**同类未修的管线输入缺口）**恢复成立**。
  **未做任何管线改动**（不需要修）。
- Evidence（误报根因 + 决定性反证）：
  - 误报根因：探针用 `dict.get(field)` —— "键不存在"与"键存在但值为 null"都返回 `None`，**无法区分**；
    "键存在且为 null"是**推断**出来的，不是实测的。
  - 决定性检查（显式测键存在性）：`task_context` 的键集 =
    {answer_type, domain, official_task_type, question_type, required_modalities, sub_category, temporal_scope}，
    **975/975 行完全一致**；`baseline` / `model_stack_id` / `planner_model_id` 的 `key_present = False`（975/975）。
    即 `task_context` 恰好只有 7 个键，与 `build_p9d_topology_dataset._task_context()` 的输出、
    以及域内 J 的布局**完全一致**。
  - 因此编码器 `j_series_common.py:178-183` 正常走 `else` 分支、从 `stack_context` 读到真实值；
    按编码器口径复算：`baseline = langgraph_react`（**in_vocab**），
    `answer_type / question_type / domain / official_task_type / sub_category` 全部 in_vocab。
  - 真正的残余边界（均非 bug）：`temporal_scope = "unknown"` → UNK，但 registry **无此字段**（数据边界，GPT 方案已决定保留 fallback）；
    `model_stack_id = stack_a_qwen3_vl8b_yolo11x` → OOV，是**数据集命名漂移**（域内为近似值 `stack_a_qwen3_vl8b`），
    另一半 486 个锚点（`stack_b_..._yolo26n`）在词表内、能拿到真实 token；
    `planner_model_id` 域内词表**只有 `unknown`** 一个值 → 该字段在域内从来不含信息；
    `required_modalities` 的字符串形态与逐字符迭代**域内也存在**
    （J train：list(1)=8156 / list(2)=2029 / str=3439 / list(3)=130）。
- Reason: 该 P0 建立在"`coverage_report.json` 从 `stack_context` 解析（报 missing=0）而编码器从 `task_context` 解析"
  这个矛盾上；显式测键存在性后发现**矛盾根本不存在**（审计与编码器口径一致）。
- Alternatives considered: 按误报直接去"修" builder（**已否决** —— 会把本来正确的布局改坏）；
  保留误报并加免责声明（否决 —— 会污染记录）。
- Consequence: 记录更正四处（门禁 `post_pilot_code_audit_20260918`、本文件、PHASE21 报告 §7、HANDOFF/STATE）。
  **教训已登记**：审计代码必须显式测键存在性（`field in mapping`），**不得**用 `.get()` 推断存在性；
  建议 coverage 审计直接断言"编码器等价解析"的结果，而不是另写一套解析逻辑。
- 审查路由：用户已把绑定会话切到 **Sol + High**，brief 已提交（待回收）。本节为本地端到端验证，
  **以 GPT 审查回复做最终确认**。

### 2026-09-18（同日）— 独立审查裁决：核心负结果通过、误报撤回通过；但"无残余输入缺口"被驳回（新增 21-B1/B2）

- Decision: 把 Phase 21 的代码与判定提交绑定会话（GPT-5.6 Sol + High）独立审查，并按裁决更新记录。
  **采纳**：核心结论（恢复三个 registry 内容字段不改善校准）与"null 遮蔽误报的撤回"均通过；
  **撤回**我另加的两句——"不存在同类未修输入缺口"与"误差主要来自结构错配/执行期分布偏移"。
- Evidence:
  - **GPT 指出的关键事实**：冻结 `Vocab` 有两个不同概念 —— 特殊 OOV 桶 `__UNK__` 与普通类别 token `"unknown"`。
    域内 `planner_model_id` 只有字面量 `"unknown"`（模型学的是它的 embedding）；S_* 传的是
    `Qwen3-VL-8B-Instruct` → 落到 `__UNK__`（该 context embedding 很可能训练中几乎未被更新）。
    因此"域内词表只有 unknown → 没有信息损失"在 **token 级是错的**。
    同理适用于 `model_stack_id` 的 stack_a 命名变体与 `temporal_scope="unknown"`。
    历史字段 `model_id` 也约有 6% OOV。
  - **GPT 要求的两个封存前检查，已在 Windows 机执行并全部通过**：
    ① 真值顺序等价性：按纯 `sequence_index` 与按 `(source_step_id, sequence_index)` 排序，
    **640/640 全 R7 模板 + 64/64 pilot 模板完全一致（0 不一致）**（7,073/8,295 个节点的 `step_of != sequence_index`，
    但**诱导顺序不变**）→ Q1 由 INFERENCE 转 **VERIFIED**，pilot 数字无需重跑。
    ② 两个 pack 的 manifest 均 `min_steps=5` / `horizon=5` / checkpoint `0ee8ded4…` → P0-2 关闭。
  - **零成本分层（GPT 建议的第一步，用现有输出）**：按 `model_stack_id` 分层
    （fixed：stack_b(in-vocab) R50 **0.5820** / cov90 **0.9012** / cov95 **0.9699**；
    stack_a(OOV) R50 **0.1751** / cov90 0.7878 / cov95 0.8620）
    → 命中 GPT 判据"stack_b 正常、stack_a 极差 → 先查 stack canonicalization"。
    **但存在混淆**：两者本就是不同执行栈、runtime 分布不同，分层不证明因果。
- Reason: 只排除了一个"简单解释"，不等于排除了所有输入侧解释；`__UNK__` 与 `"unknown"` 的区别是我漏掉的关键点。
- Alternatives considered: 直接全量重生成（否决：先做几十秒的配对反事实）；
  凭名字相似直接把 stack_a 变体映射到 `stack_a_qwen3_vl8b`（**否决/需前置**：必须先确认 stack identity 定义）。
- Consequence: 新增两个待批的几十秒级配对实验 **21-B1**（`planner_model_id` → 字面量 `"unknown"`，恢复训练期
  feature-availability contract）与 **21-B2**（stack canonicalization，需前置确认）。
  判据：mean runtime quantile pinball 改善 ≥10% 且 CI upper < 0，且 p50 coverage / R50 朝目标移动。
  两者都失败后才可把主因归为分布偏移。**论文措辞以 GPT 给的三句为准，禁用清单见门禁 `forbidden_wording`。**
  审查归档 `docs/research/2026-09-18_fas_phase21_review_gpt.md`。

### 2026-09-18（同日）— Phase 21-B 封存：B1 有效但小；B2 拒绝正确；分层只能做 association

- Decision: 执行 GPT 指定的 21-B1、拒绝 21-B2，并把结论按审核意见收紧。**feature-patching 线到此停止**，
  回到已批准的 P2 sweep。
- Evidence:
  - **B1**（planner_model_id → 训练期字面量 `"unknown"`，975/975 锚点，同 checkpoint）：p50 pinball
    **−24.70ms [−38.02, −9.87]（显著改善）**，但仅 **2.16%**（预注册门槛 10%）→ **FAIL**；
    相对原始 masked +26.88 [−3.76, 55.51]（不显著）。故 `planner_model_id` 的 deployment-contract mismatch
    是**可检出但工程上很小**的贡献者。
  - **B2 拒绝依据**：`configs/phase3_stacka_old32_q2.jsonl`（裸 id 的唯一来源）**没有配置检测器**、
    `qwen_max_new_tokens=384`、`answer_model_id=finish_argument`；R7 的 stack_a 是 **yolo11x.pt batch 8**、
    256 tokens、answer_model=Qwen3-VL-8B-Instruct → **不同执行配置**，改名 = 错误身份替换。审核确认拒绝正确。
  - **零成本分层分解（GPT 最高优先项）**：B1 下预测 p50 **scale 近乎平坦**（各格 193–764ms），
    真值中位在 345–5978ms；**`langgraph_react` 在两个栈上都是最差格**（pred/truth = 0.069 与 0.114），
    **含 in-vocab 的 stack_b** → **baseline 的影响大于"栈是否在词表内"**，"未见栈是主因"读法不完整。
    p90/p95 头仍保守（stack_b 0.918/0.971；stack_a 0.819/0.878），失效的是 **p50 scale 头**。
- Reason: 审核明确要求把因果叙事收紧：stack identity 的 OOV 与真实执行配置差异**完全共线**，
  因此只能写 association（"失准高度集中于训练词表未覆盖的 stack_a 配置"），
  不能写"未见栈本身导致失准"，也不能写"in-vocab 栈上模型校准良好"（只有一个 in-vocab 部署栈，n=1，且 p50 未校准）。
- Alternatives considered: 继续做 stack canonicalization（**否决**：身份不符）；把 model_stack_id 也 mask 成
  训练期 `unknown`（**审核排除**：该字段训练词表中**不存在**字面量 `unknown`，实验不合法）；
  在本线重训 runtime 头（否决：改变冻结假设，属另立新线）。
- Consequence: 冻结预测器的**栈相关 OOD 限制被接受**，不再改输入；下一步回到 P2 sweep（v03 pressure × horizon）
  与 v03-confirm300 一次性验证。**写入论文前必做 3 个 P1**：128 missing-template anchor 分类 + fail-closed、
  `paired_delta` 配对一致性断言、输出写入 resolved path/SHA/checkpoint/min_steps。
  论文定位：不单开一章，归入 **Deployment input-contract audit**（A = registry task-context restoration，
  B = stack/planner contract diagnostics）。归档 `docs/research/2026-09-18_fas_phase21b_review_gpt.md`。

### 2026-09-18（同日）— 指标口径审计：**"预测器严重失准"这条前提撤回**；R7 上按主指标反而更好

- Decision: 用户质疑"预测器没道理这么差"后，用**同一把尺子**复核。结论：**之前那串"预测器很差"的判断不成立**，
  以它为前提的推论（含"C2 保守尾部聚合是在补偿预测误差"）需重新论证。**模型、artifact、调度实验均未改动。**
- Evidence:
  - 用 `pack_j_predictor_artifacts.py --split validation --min-steps 5` 在 **J 验证集（2,029 行）**上跑冻结 J3:seed11，
    再用 **R7 那套统计代码**打分 → **逐位复现验收报告的全部数字**：
    p50/p90/p95 覆盖 `0.5555 / 0.9166 / 0.9680`（验收 `0.556 / 0.917 / 0.968`）；
    RuntimeQScore `845.0 = (1194.3+797.9+542.9)/3`（验收 `845.0`）。⇒ **两套打分是同一把尺子。**
  - **关键事实一**：域内 `Σp50/Σ真值 = 0.6322`，**不是 1.0**。验收报告只公布覆盖率与 pinball，从未公布这个比值。
  - **关键事实二**：按项目主指标 RuntimeQScore，**R7（782.1）优于域内（845.0）**（越低越好）；
    R7 三个 pinball `1117.3 / 737.4 / 491.6`。
  - **为什么 0.63 是结构性的**：耗时分布极度右偏 —— 域内每槽位真值中位 2209 ms、均值 3877 ms（**1.76 倍**）。
    "逐步中位数之和" ≠ "总和的中位数"，完美模型也只能给出 ≈1/1.76 = 0.57；实测 0.632 正在该量级。
  - **90%+ 是另一个头**：验收表里的 90%+（7 字段平均 0.9013、next-role 0.9444、next-family 0.8396）是
    **分类头**（结构/行为）的准确率，与**资源头**（覆盖率/pinball）无关 —— 两者混用才产生了"预测器很差"的误判。
  - 资源头覆盖率三个分位**都略高于目标**（0.5555 vs 0.50；0.9166 vs 0.90；0.9680 vs 0.95）→ 预测分位数偏保守，方向安全。
- Reason: "Σ预测/Σ真值 只有 0.44"被当成对比 1.0 来读，但域内该值本就是 0.63；且 0.44 里还混着真实的（较小）R7 退化。
- Alternatives considered: 直接把 R7 塞进 J 的标签流水线重造数据集（成本高；先用"同一把尺子"在 J 验证集上验证了代码等价，已足够定案）。
- Consequence:
  - **Phase 18 的"Σp50 = 0.456× 真值 ⇒ 预测乐观"必须改基准**：对照域内 0.632 而非 1.0。乐观依然存在（0.456 vs 0.632）
    但比原描述小得多。
  - **C2 需要重新论证**：若"Σp50 天然低估总量"，则 q95 式聚合可能是**数学上正确的聚合方式**，
    而不是"补偿预测误差"。此为**假设**，尚未证实。
  - **禁止**在任何对外文本中写"预测器低估约 60%"。
  - 唯一真实残余：p50 覆盖率 R7 0.427 vs 域内 0.556（对应已知的 OOV 输入字段与栈分层）。
  - 产物：`outputs/a2a_jval_full5/`（J 验证集预测包）；门禁 `metric_audit_20260918`。

## 2026-09-20 · resource-v2 artifact 契约审查（GPT，提交 78f000a）

- **裁决**：地基正确（Q2(a)(b)(c) 全部 VERIFIED，无需返工 Phase R）；但 artifact 契约未达
  fail-closed 标准，**正式 300 集 smoke 不得启动**，直到 P0 修完。
- **P0（阻塞正式 smoke）**
  1. pack_resource_v2_artifacts.py 的 missing-node / missing-scenario continue 与越界 reak 全部静默
     → 改为 fail-closed 断言；收尾断言 written == len(rows) == len(base_pack) 且 steps_written == base_step_count。
  2. workload_v02_simulator.py:158 的 load_future_artifacts() 强制要求 H1/H3/H5，
     resource-v2 目录只有 H5 → 必须新增 **overlay loader**（读 base root + 单独读 H5，断言节点集相同、
     非 resource 内容与 base 相同、每次只替换 future_h5），**不要复制 H1/H3 文件**。
  3. 缺 **打包后 validator**：probs→q50/q90/q95/mean/CVaR 重推导一致、每个 step 都已升级、
     无静默回退、非 resource 字段未变。
  4. manifest 缺 immutable SHA（artifact/head/base/bin + producer commit），runner 无法真正冻结
     predictor_artifact_id。
  5. 新消费臂**不得同时**改 runtime 与 load 规则，否则 consumer effect 被混淆。
- **Q3 消费臂规格（已定）**
  - 先抽出 _legacy_p95_load_cost(step)，原 _q95_step_cost 重构为
    
untime_p95 + _legacy_p95_load_cost(step)（注意：现有 q95 臂**不只是 runtime q95**）。
  - B1 sameshape_h5_condmean / consumer sum_conditional_runtime_mean_plus_legacy_load_v1，**fail-closed 不复用 fallback**。
  - B2 sameshape_h5_stepcvar95 / consumer sum_marginal_step_cvar95_plus_legacy_load_v1。
  - _SAMESHAPE_PREDICTED_STATS = (p50, p95, condmean, stepcvar95)。
  - **论文措辞**：B2 必须写成「各预测未来步边际 CVaR95 的加和风险分数」，
    **严禁**写成 CVaR_0.95(Σ_h T_h)（我们没有联合分布）。
- **Q4 O 臂**：用仓库**已有**的 sameshape_h5_truth，不需新造 artifact；但必须命名为
  **joint-future-truth headroom**，**不得**称为 resource-head oracle（它同时消除 runtime 误差与
  topology/length 误差）。当前 artifact 不存在无歧义的 resource-only oracle 映射。
- **Q5 runner 规格（已定）**
  - 300 集 = 135 个 cell 各取 2（=270）+ 30 个 cell 各加 1；30 个 extras 需满足
    arrival 各 +10、load 各 +6、GPU topology 各 +10、state/deadline 各 +10；cell 内按 episode_id 排序。
    最终边际：arrival 100/100/100，load 60×5，GPU 100/100/100，state 100/100/100。
    **禁止**直接取 alidation_000000:000299。选出 
esource_v2_smoke_v1_episode_ids.txt 后**永久冻结并记 SHA256**。
  - 不要调 
un(...)（无 per-policy artifact binding）；逐 episode × arm 调用 simulate_episode(...)，天然 paired。
  - **NI margin δ = 485 ms = 0.1 × 4853**，其中 4853 ms 来自 Phase 20B 已冻结的 p95 − E0（dev700）。
    **这是一项治理选择，需项目负责人在跑 300 集之前签字冻结 10% 比例。**
  - A2（R3a-U）已违反原 calibration integrity gate → 必须标
    eligible_for_model_selection = false、
ole = diagnostic_only；即使 scheduler 偶然最好也不得成为最终 winner。
  - **启动前必须全部 PASS**：节点集一致 / 每行恰好一个 scenario / 每个 step 都已升级 / 无静默回退 /
    probs→各视图重推导一致 / 非 resource 字段未变 / artifact-head-bin-base SHA 冻结 /
    overlay loader 可用 / sameshape_h5_p95 回归测试逐位不变。

## 2026-09-20 · 用户签字冻结：6 臂 smoke 的 NI margin

- **决定**：接受 GPT 提议，δ_NI = 485 ms = 0.1 x 4853 ms。
- **来源**：4853 ms 是 **Phase 20B 已冻结**的 dev700 上 p95 − E0（平均完成时间），
  **完全来自本次 Phase R 之前的数据**。
- **签字时点**：2026-09-20，**在任何 smoke 结果产生之前**。用户原话：就按照这个吧。
- **判据（预注册）**：
  - 非劣：CI_upper95(Δ) < +485 ms
  - 实质改进：CI_upper95(Δ) < 0 且 Δ ≤ −485 ms
  - 相对 O 饱和：CI_upper95(A1 − O) ≤ 485 ms
- **性质**：治理选择（INFERENCE），不是数学定理；10% 这个比例是人为约定，论文中必须如实标注。
- **落盘位置**：.project/EXPERIMENT_GATE.json
  → experiments.EXP-20260919_j_series_resource_dist_v1.smoke_preregistration_20260920（status = FROZEN_BEFORE_RESULTS），
  其中同时冻结了 6 臂的 immutable identity、contrasts、300 集抽样规则、统计设计（paired bootstrap B=2000）、
  9 条 preflight 与 5 个阻塞 P0。
- **纪律**：此后**不得**再改 δ、抽样规则或 contrasts；任何修改都必须作为新的 amendment 并显式声明未失效的部分。
