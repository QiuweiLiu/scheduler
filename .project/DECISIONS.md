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

## 2026-09-21 · F0 选定为部署预测器；遥测负结果收口

- **遥测负结果（预注册判据未达成）**：`F1−F0` 的 RuntimeQScore 逐 seed 为 −0.159 / −0.250 / +0.721，
  均值 +0.104（+0.02%）；配对 video-cluster bootstrap（B=2000，30 clusters）**CI95 = [−0.3613, +0.7922]，
  跨 0**，`P(mean ≤ 0) = 0.2075`。8 个指标里 **7 个符号在 seed 间翻转**；唯一一致的是 `termination_bce`
  （+0.00008，且方向变差）。判定：**no measurable improvement detected**。
  **不得**写成"F0/F1 统计等价"（未预注册等价边界），也**不得**写成"历史遥测无信息"。
- **排除"训练不够"的替代解释**（GPT 复核 + 我方实测）：
  ① 三个 seed 下 F0/F1 **选中的 epoch 完全相同**（28/28、18/18、20/20）；
  ② 末轮训练损失几乎重合（seed11 resource：F0 1.825572 vs F1 1.825395）；
  ③ validation 在最佳 epoch 后**退化**（seed22 602.59@18 → 613.27@30），是 plateau/overfit 形态。
  → **不建议**为这个反解释补 60/100 epoch（那会成为 post-hoc training search）。
- **实现级审计（我方独立做）**：F1 遥测参数位移 0.554/0.335/0.459，**F0 恰好 0.0**；
  在训练好的 F1 上做推理消融，开关遥测只改变 ~0.1% 且符号翻转；
  `||E_telemetry|| / ||E_categorical+E_position||` ≈ **1%**（init 与 final 几乎相同）。
  **结论：接线正确、通道被训练过，但模型基本不依赖它。**
  ⚠️ **一个未排除的替代解释**：新分支初始化 `normal_(0, 0.02)`，比类别嵌入的 `N(0,1)` **弱约 50 倍**，
  且训练后仍弱 50 倍 → "信息少"与"初始化太弱学不起来"**尚未分离**。已向 GPT 提出，未解决。
- **全量重训 + 16-bin 头的效应**：`F0−J3` = −28.7%，但这是**两个因素的打包**
  （全量重训 + 16-bin 头），**不可拆分归因**；必须写成 "full retraining + 16-bin distributional head"。
- **F0 选定为部署预测器**（`F0 seed11`），理由：GPT 判定"无可测收益时不应把额外输入依赖带进最终系统"，
  且 **seed11 是本项目的 canonical seed**（J3、R1b 主线都用它）；选 seed33（数字最好）会引入 best-of-3 optimism。
  **F1 的定位 = 历史遥测消融臂。**
- **真值参照的排序诊断（新增，零训练）**：在 7,663 个可解析锚点上，
  **F0 的全局 Spearman 0.6517 / 首选一致率 0.916**，**均优于 J3（0.5917 / 0.895）**；
  R1b（0.5544 / 0.884）与 R3a-U（0.5461 / 0.650）**均差于 J3**。
  → **F0 是第一个在"真值参照的排序质量"上真正超过 J3 的预测器**，不只是"更保守"。
  ⚠️ 三点限定：① 三个尺子不完全一致（模板内两两一致率把 R1b 排在 J3 之上）；
  ② 1,280 个锚点（主要是 `:run:1` 根节点）无法解析；③ 真值走查用了"空驻留 GPU"固定代理
  （绝对值可能有偏，但四包面对同一组真值，**比较公平**）。
- **方法论教训（已记录）**：把"与现任冠军 A0 的排序一致性"当作质量指标是**错的** ——
  它只衡量**扰动幅度**，不衡量**质量**。R1b 与 A0 一致性 0.754 却在真值上更差（0.554 < 0.592），
  即"变了很多而且变差了"。**质量必须对真值，不对 A0。**

## 2026-09-21 · 主 baseline 改为 F0（用户确认）

- **决定**：新调度方法的**主对比基准改为 `F0 sameshape_h5_p95`**（F0 是已选定的部署预测器）。
  旧 `A0`（J3）**保留为历史基线**，但新方法不再主要跟旧预测器比。
- **理由**：F0 是当前部署预测器，且是第一个在真值参照排序上超过 J3 的预测器
  （Spearman 0.6517 / 首选一致 0.916 vs J3 的 0.5917 / 0.895）。
- **`δ_NI = 485 ms` 继续使用，不重算** —— 同 workload、同 primary metric 下已冻结的绝对 practical margin。
- **判据**：`Δ = new − F0Baseline`；
  非劣 `CI_upper(Δ) < +485 ms`；统计改善 `CI_upper(Δ) < 0`；
  实质改善 `point(Δ) ≤ −485 ms` **且** `CI_upper(Δ) < 0`。
- **两层门禁**（来自复现审查）：第一层 **implementation fidelity**（即使性能差也算成功复现），
  第二层 **performance**。若 fidelity PASS 但性能 FAIL，写法是
  "scheduler successfully reproduced/adapted, but did not improve this workload" —— **合法的负结果**。

## 2026-09-23 · F0 打包器 attr_feature bug（已修复）+ 撤回"F0 更保守"

### bug
`pack_f0_artifacts.py` 手工调用 `model.resource(encoded, None)`，而 `j_series_common.py`
把 `None` 替换成 `torch.zeros(...)`。训练时走的是 `j_series_train_eval.forward()`：

    attribute_logits = model.attribute_logits(repr_vec)
    feature, _       = model.attribute_distribution(attribute_logits)
    resource         = model.resource(repr_vec, feature)

F0 的 config 是 `use_attribute_distribution=True, attribute_gradient='full'`，所以资源头
**是条件在预测属性特征上的**。传 `None` 等于把它喂在全零输入上，5 个 horizon 槽位拿到同一份
输入，资源头只能靠位置编码区分槽位。

### 实测影响（512 个 validation 锚点，F0 seed11）
| 视图 | None（坏） | 真实 attr_feature（对） | 相同比例 |
|---|---|---|---|
| p50 | 2604.9 | 2599.1 | 26.1% |
| p90 | 8223.6 | 5808.2 | 20.6% |
| p95 | 10353.9 | 7229.2 | 22.4% |
| mean | 3408.0 | 3088.7 | 0.0% |
| RuntimeQScore | 7060.81 | 5212.16 | 比值 1.355 |

### 影响面（已逐个核实调用点）
- **安全**：`pack_j_predictor_artifacts.py` 用 `jte.forward(...)`；`pack_resource_v2_artifacts.py`
  用 `base.forward(...)`。所以 **J3 base pack 与 R1b/R3a overlay 都正确**，
  `A1-A0`、`B2-A1`、pack-vs-truth 排序等已发表结果**不受影响**。
- **安全**：`F0-J3 = -28.7%` 来自训练报告，走 `forward()`，**有效**。
- **受影响并已重跑**：F0 包本身，以及**今天**用坏包跑的 aging shadow / aging smoke /
  F0 pack-vs-truth。
- 其余 `resource(..., None)` 调用点：`histres_smoke.py`、`histres_train_f0f1.py`、
  `f0f1_contrast_bootstrap.py`、`histres_channel_audit.py`、`tests/test_distribution_training_path.py`
  —— **待逐个判断是否影响结论**（这些多为诊断/探针，不是已发表主结果）。

### 修复
`pack_f0_artifacts.py` 改为精确复现训练 forward 的属性路径，并加 fail-closed 守卫：
若该臂 config 不再声明 `use_attribute_distribution`，打包器直接报错退出。
**刻意不使用 `jte.make_ctx`**，因为它会读封存的 `j_test.jsonl.gz`。
修复后包 sha256 `586ae65d75e8993ed72176b7920bf9878a3fb47b2cbdfcb7a222482de606a086`，
loader 接受（view_err 3.64e-12, nonresource_mismatch 0）。

### 撤回
**"F0 的 consumed score 是 J3 中位数的 1.327 倍（F0 明显更保守）" 撤回。**
那是 bug 造成的假象。修复后 F0/J3 的 p95 总和比 = **0.8764**，即 F0 比 J3 **低 12%（更不保守）**。
独立交叉验证：修复后中位 p95 = 7201.3，与直接测正确路径得到的 7229.2 一致。

### 教训
模型是联合训练的、训练指标有效，但**打包器手工绕过了联合路径**。
"联合"必须在**打包产物**上验证，不能只看训练代码。

## 2026-09-23 · 基线集合修订为四条"联合基线"（项目负责人指定）

四条**全部是联合基线**（Predictor->Scheduler 整条链一起迁移的 faithful adaptation），
**SRTF+Aging 不在其中**：

1. **LLMSched-adapted**（IEEE ICDCS 2025，IEEE Xplore 11183728，身份已确认）
   前端 DAG+Bayesian Network 建模结构/duration 不确定性，随已完成 stage 更新 posterior；
   后端 entropy 衡量 uncertainty reduction + JCT/SRTF 优先级。
   检验：传统概率工作流建模 + uncertainty-reduction scheduling 是否已经足够。
2. **Pythia-adapted**（arXiv:2604.25899，v2 题名 "Exploiting Workflow Predictability..."，
   **venue UNVERIFIED**）：历史 traces -> PFA -> bounded workflow -> E[remaining distance]
   -> completion-aware priority。检验：历史模式预测能否替代 instance-specific prefix forecasting。
3. **Latency-Aware-Orchestration-adapted**（arXiv:2609.03335，2026-09-03，cs.DC，13 页，
   **venue UNVERIFIED**，已核实存在）：Predictor（device-specific latency/memory/loading）
   + **Constructor（fusion / model-lifecycle alternatives）** + Scheduler（joint selection/
   placement/order）。检验：已知 DAG + 资源预测 + 物理图联合优化是否已经足够。
4. **TIE-adapted**（ICML 2026，arXiv:2604.00499，身份已确认）：**独立的 current-node-only
   分布预测器**（log-t 或等价参数化）+ `E + beta*CVaR` tail-aware score。
   检验：单节点 runtime 从 point 换成 distribution 是否已经足够。

**完全体** = prefix-conditioned predictor -> 尚未展开 future 的 structure + runtime/resource
distributions -> uncertainty-aware node-level GPU scheduler。

**证据链**：E2E 优于四类联合基线 -> "固定 scheduler 换 predictor" + "固定 predictor 换
scheduler" 的 2x2 -> 消融 prefix future / distribution / joint modeling / uncertainty consumer。

**实施顺序**：`Gate-0（Latency-Aware 可行性）-> TIE -> Pythia -> LLMSched -> Latency-Aware`。

**SRTF+Aging 定位**：`classical scheduling diagnostic / negative-control arm`，
退出联合基线主链；不补 T 臂来救它。

## 2026-09-23 · P0：scheduler 侧拓扑契约回归（已定位、已重建、已过 gate）

### 定性
**`P0 scheduler experiment validity issue，不是 predictor validity issue`。**
GPT 独立复核确认，用词：**scheduler-side topology contract regression /
incomplete contract migration**。

### 证据链
- **2026-08-13** `docs/VideoSeek_R0a_只读审计报告_20260813.md` 已规定主实验用因果链：
  "其余节点仅以前一个节点为 predecessor；保留原有 predecessor 的 provenance 作为旁路审计字段；
  主实验先使用 causal-chain；raw-DAG 只作…"。并**预言了**："预测器按 sequence_index 线型
  prefix 构造特征，与 raw parent 产生的稠密 DAG **同时使用会造成不一致**"。
- **2026-08-16** `r6_causal_v2` **实现了**双视图：`causal_predecessor_node_ids` +
  `raw_predecessor_node_ids` + `scheduler_visibility`，`edge_provenance =
  canonical_sequence_chain_v1`，因果视图是一条完美直链。
- **2026-08-17** `r7_workload_20260817/job_templates_r7_v02.jsonl` **退回**
  `value_source = measured_trace_events_with_raw_parent_edges`，
  **`causal_predecessor_node_ids` 与 `scheduler_visibility` 消失**。
- v03 = v02 减 `event_type=="run"` 容器（`scripts/preprocess/remove_run_container_nodes.py`，
  **no rewiring**）→ 原样继承缺陷。
- **2026-09-10** `docs/p9d_topology_label_contract_v3.md` §9 明确：
  "本轮只冻结 predictor 侧 v3 标签契约；R7/R8 的 job_templates/future artifacts
  **存在同源问题但不同步修改**"，顺序为 predictor v3 → … → **（独立 gate）
  scheduler-side topology contract audit → scheduler integration**；
  "首次 scheduler 集成前，两侧契约语义必须一致"。
  **这个 scheduler 侧审计从未执行，调度集成却已发生。**

### 机制（代码级确认，非统计推断）
`src/tracing/workloads/build_workload_v02.py::_resolve_edges()`：
```python
by_step[step_id] = 该 step 的全部 node_id
for parent in parent_step_ids:
    predecessors.extend(by_step[parent])
node["predecessor_node_ids"] = sorted(set(predecessors))
node["edge_provenance"] = "raw_parent_step_ids"
```
step N 展开为 {u1,u2,u3}、step N+1 展开为 {v1,v2} 时，每个 vi 把 {u1,u2,u3} 全收为前驱
→ 必然产生 `{2,3} → {4,5} → {6,7}` 的分层全连接。

### 关键量化：源数据是干净的，错只在边推导
对 640 个 v03 模板跑 v3.1 §7 P0 gate 1：
```
multi_tool_per_iteration   0   ✅
multi_parent_step          0   ✅
step_jump                  0   ✅
not_single_root            0   ✅
dangling_predecessor       0   ✅
missing_resource_applicable 640  ← schema 字段缺失，非语义违规
```
**`parent_step_ids` 的串行不变量完全成立（0 违规）** —— 原始 trace 数据正确，
假并行**完全由 `_resolve_edges()` 的 step 展开引入**。

### 重建
`scripts/rebuild_scheduler_templates_causal_v31.py` →
`results/processed/r7_workload_v04_causal_v31_no_run_container/job_templates_r7_v04.jsonl`
- `causal_predecessor_node_ids` = 已验证串行序列中的相邻节点（单前驱单后继）
- `raw_predecessor_node_ids` = 旧 step 展开边，**仅作旁路审计字段**
- `resource_applicable` = 链尾 answer marker 为 false，其余 true（v3.1 §3）
- `topology_contract = verified_serial_control_flow_v3_1`
- 节点数 **8,295 不变**（纯边重建）；**`chain_tail_is_answer = 640/640`**
  （v3.1 §4 path gate："每条 run = 单链、单 root、终点为 answer"）
- v3.1 seriality gate：**640/640 PASS，0 违规**

### 影响（GPT 复核并补充）
**需要重新审计（scheduler-facing evidence chain）**：
- `A1-A0` / `B2-A1` / `O-A1` 那批 scheduler smoke（300 配对集）
- 72,773 decisions / 291,285 candidates 的 decision trace
- winner disagreement / Kendall tau
- aging 的激活度与 +12% 结果
- Latency-Aware 当前 6 对 fusion 的统计
- 任何依赖 v03 successor walk 的 future truth / ranking 诊断

**不受影响**：预测器侧全部结论（training / validation / `forward()` / RuntimeQScore /
distribution calibration），**包括 `F0 相对 J3 = -28.7%`**。

### GPT 复核中必须遵守的修正
1. 回归点在 **r7/v02（08-17）**，不是 v03（v03 只是 no-rewiring 继承）。
2. **R6 causal-chain 与 P9d v3.1 不是同一个契约**：R6 是 operational repair，
   v3.1 是语义级 node ontology（run_control 排除、nested call 合并、answer terminal、
   retry 显式、width=1、seriality fail-closed）。
   **修复必须追上 v3.1，不能退回 R6。**
3. 不要把 "0.2% → 91.2%" 当作修复后的 fusion 提升；fusion 仍需唯一前驱/后继、
   同部署、配置兼容、语义保持。
4. **不要把"预测器输出是直链"当作主要证据**（单独不够）；硬证据是
   workflow 源码语义 + `parent_step_ids` 审计 + 2,008 traces 串行性 gate + R0a/R6/P9d 冻结决策。
5. 确认 640 个正式模板自己过 gate（**已做：640/640 PASS**）。
6. **不要只改边、不改节点定义**；`missing_resource_applicable` 正是这一层的缺口信号。

### 待办
- **调度机会量化（GPT 预注册）**：`Pr(feasible_actions >= 2)`、`Pr(ready_GPU_nodes >= 2)`、
  竞争决策率、候选数 p50/p90、每决策活跃 job 数、top-1 分歧、队列占用、GPU 利用率、
  驱逐/加载次数。**若这些比率骤降到个位数，需要重新设计 workload pressure，
  而不是恢复假并行。**
- **冻结 `SchedulerTopologyContractGate`**：主实验启动前要求两侧契约一致、640 模板全过
  seriality gate、raw 边永不作为执行边、两侧 SHA 冻结；每个 scheduler 输出记录
  `topology_contract_id` / `template_sha256` / `future_artifact_sha256` /
  `resource_pack_sha256` / `simulator_commit`。
- **640 vs 648**：契约文档说审计 1,360 predictor + 648 R7 scheduler = 2,008 runs，
  我们的 workload 是 640 accepted templates。**不要在重建时猜**，需从 manifest 确认。

## 2026-09-23 · 四条论文复现基线全部实现（含独立调度器）

### 顺序与定位
GPT 指定的顺序 `TIE -> Pythia -> LLMSched -> Latency-Aware`，全部完成并过 fidelity。
**SRTF+Aging 已退役**为 classical negative-control，不在主链上。

### TIE-adapted（ICML 2026，arXiv:2604.00499）
- 前端：**独立的 current-node-only 分布**，来自 train-only 的 `(model, lane)` 直方图
  （`train_resource_stats` 新增 `runtime_mean_ms` / `runtime_cvar90_ms`）。
  **刻意不复用 F0**，符合 GPT "不能拿 F0 16-bin 算 CVaR 就叫 TIE" 的要求。
- 后端：`tie_beta(L_q,B) = clip(0.1·L_q/B, 0.1, 0.5)`；`tie_current_score = E + β·CVaR + load`。
  纯函数已抽出，可直测。
- **测试抓出的真实性质**：`E + β·CVaR` 在单样本组上退化为 `mean` ——
  current-node-only 前端**只能区分 `(model,lane)` 不同的候选**。这是方法的结构性局限。
- 不迁移：vLLM continuous batching、DeBERTa encoder、token 语义、异步预测线程。

### Pythia-adapted（arXiv:2604.25899，venue UNVERIFIED）
- 迁移 Pythia-core：`train-only traces -> profiler -> E[remaining distance] -> completion-aware priority`
- 480 个 train-only 模板 → 2 个家族：`langgraph_react`（E[D_rem|k=0]=62,237ms）、`star`（45,394ms）
- `S_completion = 1/E[D_remaining]`；**ω1=1, ω2=0**
- **S_unblock 明确省略**：本模拟器无 model-server 队列抽象。GPT 原话："这比生造
  'GPU 空闲 ≈ downstream model idle' 要可信得多"。
- 不迁移：cache routing、prefix caching、model-replica idleness、autoscaling

### LLMSched-adapted（IEEE ICDCS 2025，Xplore 11183728，身份已确认）
- **policy wrapper，不是静态 key**：每决策抽**一次** ε 硬币选模式，再排序候选池
  - EXPLOIT = 最小估计剩余时长（JCT/SRTF）
  - EXPLORE = 最大信息增益
- BN/CPD 从 train-only 学（**不用 F0 的 marginal 拼 joint**，那才是被禁止的人为联合假设）
- `R(X) = (结构信息 + duration 熵) × Range(Y_m)`，对应论文的 `I × ΣRange`
- **测试抓出的真实性质**：`info_gain(k=0) = 0.0000` —— 因为最短的链也有 3–6 个节点，
  `n≥0` 与 `n≥1` 是同分布。**结构不确定性在链式负载上早期为零**（与 fusion 同类现象）。
  所以 information 项必须由 **duration 不确定性**承担（H_dur ≈ 2.88 bits）。
- 明确省略：`sample_tasks(r)` 的 stage 内部分执行（我们的执行单元已不可分）。
  **没有偷偷设 r=1 假装完整复现。**

### Latency-Aware-adapted（arXiv:2609.03335，2026-09-03，cs.DC，venue UNVERIFIED）
- **独立调度器**（`src/tracing/analysis/latency_aware_scheduler.py`），按项目负责人的架构：
  **`choose_action` 及其 26 处 `min(pool,key=)` 一字未动**，其他 84 个策略完全不受影响。
- 实现论文 Eq (3)(4)(5)(7)(8)(11)(12)：
  - Eq (4) 融合时长 = 各成员 **`compute_ms`**（= `runtime_ms - load_ms`）之和 + **一次**加载
  - Eq (7) `start = max(release, gpu.busy_until, now)`
  - Eq (11) 准入显存检查（已驻留模型不重复计）
  - Eq (12) `Φ = (hard_priority, C_F, -boundaries_removed, ready, gpu, node)`
- 动作词汇显式化：`{"type":"start","candidate":…,"fused":[…]}` + 预取
- **融合素材从 0.2% → 49.3%**（拓扑修复后，Eq (3) 全条件：唯一前驱/后继 + 同部署 + 配置兼容），
  2,076 个可省边界，最长 13 节点链
- 实测（3 集探针）：**132 个融合单元**（旁路版只有 5），14 次预取，
  但 mean_completion 91,134 vs myopic 90,535 —— **激进融合减少准入次数却拖长完成时间**，
  这是个有价值的发现
- **仍待做并已显式记录**：reclaim 的 victim 选择（论文"优先冗余副本、其次下次使用距离远"）、
  Eq (7) 的**跨候选时间线传播**
- 不迁移：Qwen/vLLM 特定实现、原测试集、原 GPU 型号、output-length confidence bound、KV/prefix cache

### 顺带发现的两个预先存在的 bug
1. **`round_robin` 从未真正运行**：`if policy == "round_robin"` 是裸 `if` 且**缺 `return`**，
   而紧跟其后的 `if policy == "fcfs"` 打破了 if/elif 链，于是穿透到链尾的 `else:`（myopic）
   并覆盖 `chosen`。**已在公开仓库 commit f592591 中确认是预先存在的。**
   证据：所有基准里 `myopic` 与 `round_robin` 数字**每次都完全相同**（90,327/90,327）。
   **未修**（项目负责人指示先不管）。**影响**：任何把 round_robin 当独立基线的对比不成立。
2. `fcfs` 是安全的（它后面跟的是 `elif`，链没断）。

### 命名纪律（GPT 的硬要求）
- 四条**全部是 faithful adaptation**，不是完整系统复现；每条都列出了不迁移的部分
- Latency-Aware 的 Scheduler 缺两块（reclaim victim、跨候选时间线传播），
  应命名为 `LatencyAware-fusion-lifecycle-sched-adapted` 并在 note 列明
- **不声称 round_robin 的结果**（因为它实际是 myopic）

## 2026-09-23 · 第 1 步：v04 -> v04.1（节点定义对齐 v3.1）

GPT 复核指出：v04 只修了**边**，**节点本体未迁移**。已按 v3.1 的权威实现
（`scripts/build_p9d_topology_dataset.py::build_chain` 及其 `_resource_applicable` /
`_is_terminal_marker` / 嵌套合并规则）重建 v04.1。

### 嵌套合并（v3.1 §4 规则 3）
- 规则：`node_type == "answer_generation"` 且 `raw_action == "generalist.generate"`，
  且**同一 step 内**紧跟着 `raw_action == "summarization-tool"` → 前者是**嵌套在
  Summarizer 工具内部**的调用，必须合并进父节点
- **实测 169/169 匹配**（与 GPT 从 commit 读出的计数一致）
- 父节点获得 `merged_nested_call=true` + `nested_calls=[...]`
- **节点数 8,295 → 8,126（−169）**
- 理由：v3.1 规定 `R_node = R_total(parent summarizer)`，**禁止重复核算**；
  保留独立节点会让 simulator 把嵌套调用和父工具**分开执行 → 工作量与时长双重核算**

### run_control 排除（v3.1 §3）
`node_type == "run_control"` 不是拓扑节点，已排除。

### terminal marker 语义修正（v3.1 §3）
- 权威定义：`event_type == "run"` **且** `action == "answer"` → `resource_applicable = false`
- **我原先写的 `node_type == "answer_generation"` 是错的**
- 本投影（v03 起）已由 `remove_run_container_nodes.py` 删除全部 `run` 容器，
  **因此不含任何 terminal marker** —— 所以契约**诚实改名为**
  **`scheduler_projection_of_verified_serial_control_flow_v3_1`**，
  不再声称与 predictor 侧 v3.1 完全相同（GPT 明确给了这个选项）

### `resource_applicable` 成为真正的执行契约（不是 annotation）
- `Node` dataclass 新增该字段
- `load_templates` 读取；**causal 视图下缺该字段 → 直接拒绝**
- **3 处候选池构建点都会拒绝不可调度节点**（fail-closed）
- 修好了 GPT 指出的"JSON 里有字段但程序完全不读"的问题

### `sequence_index` 重编号
合并掉节点后原位置出现空洞，导致 gate 的 `step_jump` 误报 117 个模板。
已重编号为 `0..n-1` 连续。

### 结果
- seriality gate：**640/640 PASS，0 违规**
- 现有臂回归正常：fcfs 105,708 / myopic 90,289 / tie_current 90,367，16 job 全完成

### 640 + 8 = 648 provenance 闭环（GPT 建议的机械验证）
```
data/manifests/r7_trace_manifest_v1.jsonl       = 640 rows
data/manifests/r7_trace_pilot_manifest_v1.jsonl =   8 rows
sum = 648
pilot-only (not in formal) = 0
```
**确认：640 formal + 8 pilot = 648，没有遗漏 8 个正式 scheduler workload。**

### 仍未做（第 2–4 步）
- 第 2 步：修四条基线的 fidelity P0（TIE / LLMSched / Pythia / Latency-Aware），
  每条需重构前端数学模型 + 加 sentinel mutation 测试
- 第 3 步：冻结 `SchedulerTopologyContractGate` + `BaselineFidelityManifest`
- 第 4 步：重跑 30 集 smoke

## 2026-09-23 · 第 1 步正式冻结（三个关闭证据）

GPT 对 e8b6248 的复核判定：`~90% 完成，conditional PASS`，还差三个洞。**三个洞已全部补上。**

### 洞 1：scheduler gate 是"自证循环"（必须修）
GPT 的批评：
> 你把"重新编号以后 gate PASS"误认为 semantic gate PASS。这是典型的
> **用 transform 产生的字段验证 transform 本身** → 自证循环。
> 比如 builder 强制 `sequence_index = 0..n-1`，gate 检查 `sequence_index` 是否每次 +1
> —— 当然永远通过。
> **semantic gate 必须使用独立于被验证输出的 provenance 字段。**

它逐条指出我的空门禁：
| 我声称检查 | 实际 |
|---|---|
| `step_jump` | 重编号后**永远通过** |
| `parent_step_ids` 规范性 | 只查 `len(ps) > 1`，**没查 `()/[N-1]`** |
| `single successor` / `cycle` / `answer last` | docstring 声称有，**代码没写** |

**修复**：新建 `scripts/scheduler_projection_gate_v041.py`，语义检查全部读
`source_step_ids` / `parent_step_ids` / `retry_of` / raw node 词表；
`sequence_index` 只作 **projection-index-contiguity** 检查并**显式标注非串行性证明**。
覆盖：templates==640、nested merged==169、`parent_step_ids` 规范、`source_step_ids`
连续 same/+1、每 step ≤1 工具、retry 相邻、单 root、单 tail、无环、链覆盖每节点恰好一次、
**全部 `resource_applicable == true`**、无嵌套子节点残留。

**结果：640/640 PASS，0 违规。**

### 洞 2：169 对嵌套时间窗 —— VERIFIED
回原始 trace `results/raw/r7_trace_full_20260817`（640 run 目录）验证：
```
pairs checked               : 169
contained (nested ⊆ parent) : 169
parent_runtime == interval  : 169
missing raw trace           : 0
violations                  : 0
```
**结论（可正式写进文档）**：嵌套调用的时长**已经包含在父工具测得的 wall-clock 区间内**，
删掉嵌套调度节点**消除重复核算且不改变总耗时**。
（GPT 强调：**绝对不要再从父节点减 nested runtime**，否则会变成 exclusive runtime，
不再是 v3.1 冻结的 `R_total(parent)`。）

### 洞 3：648 必须在【执行 run_id】层闭环 —— **更正我上轮的结论**
GPT 指出我上轮写错了：
> `pilot-only (not in formal) = 0` 只说明 8 个 pilot **任务定义**都在 formal manifest 里，
> 所以**按 task identity：`|formal ∪ pilot| = 640`，不是 648**。
> 必须核 **execution run_id**。

**我查证后确认 GPT 对**：两个 manifest 只有 `task_id`，**没有 `run_id`**（它们是任务定义）。
pilot 的执行记录在 `experiments/EXP-20260817_r7_trace_pilot/remote_pilot_report.json`
（`expected_runs=8`，`raw_root=results/raw/r7_trace_pilot_20260817`）。

**执行层闭环**：
```
formal workload executions : 640
pilot executions           :   8
intersection               :   0
union                      : 648    ← 与 P9d 引用的 648 一致
```

### 关于 nested 合并的资源归属（GPT 发现的另一个 P0）
GPT 指出：**169 个 nested `generalist.generate` 全是 GPU 节点，169 个父
`summarization-tool` 全是 CPU 节点**（104 × Qwen3-VL-8B、65 × Qwen2.5-VL-3B → cpu-metadata-adapter）。
**我实测确认**：
```
nested node lanes : {'gpu': 169}
parent node lanes : {'cpu': 169}
```
所以直接删 nested、保留父节点的 `execution_lane=cpu`，**会让这些真实 GPU 工作从 GPU 竞争里消失**。
**这一条我还没有修**（v3.1 的 `composite signature` 正是为此设计：父节点应携带
`nested_model_class`，且 simulator 需为内层 GPU 模型收取加载/驻留成本）。
**列为第 1 步之后的第一优先项。**

### GPT 的其他判定
- **Q3**：`resource_applicable=false` 为 0 是可以接受的 fail-closed schema 契约；
  **它反而建议不要把 terminal marker 塞回 scheduler**（那只会增加"它何时完成/是否占 GPU/是否算 JCT"的无意义问题）
- **Q4**：我做的是**语义复刻**，不是代码复用。不阻塞，但建议抽三层共享模块
  （`verify_raw_run` → `canonicalize_v31` → `project_scheduler_chain`），
  **并让 scheduler 读冻结的 `canonical_v31_manifest`，而不是从 v03 推断本体**
- **TIE 修复清单（7 条）**：独立 bank 不走 `estimate()`；缺字段 `raise` 不 `or p50`；
  lookup 只用 `(model_id, lane)`；support 提前写死否则叫 `Empirical-TIE-adapted`；
  CVaR 小样本严格定义并报告样本数直方图；补 waiting decay；**`B` 不能用 `len(free_gpus)+1`**；
  **加 E2E sentinel mutation test**

## 2026-09-23 · 合并嵌套调用的资源归属（option b，已实现）

### 问题（GPT 发现）
169 个被合并的 nested `generalist.generate` **全是 GPU 节点**，169 个父
`summarization-tool` **全是 CPU 节点**。直接删 nested、保留 CPU 父，会让
**真实的 GPU 工作从 GPU 竞争里消失**。

### 数据侧（已完成）
v04.1 的 169 个父节点现在携带 v3.1 的 composite signature 资源信息：
```
nested_model_class   内层 GPU 模型（Qwen3-VL-8B ×104 / Qwen2.5-VL-3B ×65）
nested_model_mb      实测 peak_allocated_mb（约 16.8 GB）
nested_reserved_mb   实测 peak_reserved_mb
nested_load_ms       实测加载成本
nested_runtime_ms    内层自身区间（仅作 audit）
```
`Node` dataclass 新增这 4 个字段，`load_templates` 读取。
守恒审计：**−0.35%**（残差来自 v03 的 `workspace_peak_mb` 与 raw `peak_allocated_mb`
的推导差异）。

### 执行侧（option b，已完成）
**用户选择 (b)：复合单元同时占 CPU 和 GPU。**

新增 `admit_nested_gpu_work(node, job, job_index, now, sequence_ref)`，在 CPU 准入路径调用：
- 优先选**已驻留**内层模型的 GPU，否则选**装得下**的
- 内层模型不在驻留时**收取加载成本**并加入 `resident`
- **设备被内层调用占住自己的区间**（`busy_until = now + load + nested_runtime_ms`）
- 推一个**独立的 finish entry**（lane `gpu_nested`）在该区间结束时释放设备
- **fail-closed**：内层分配未测量时**直接抛错**，不静默跳过
  （静默跳过正是这个 P0 本身）

CPU 半不变：仍立即运行、在 `now + R_total` 完成（v3.1 的 `R_node = R_total`）。

### 验证
```
3 集探针：nested_gpu_admit = 9（三个策略一致），completed=16，failed=0
结果确实变了：fcfs 105,708 -> 101,757；myopic 89,749 -> 88,456；tie_current 89,688 -> 88,464
=> 字段真的被消费了，不是 annotation
```
**全量 276 个测试，5 个失败套件全部预先存在**（4 个缺数据 + `round_robin` bug）。

### 教训
我在这个 P0 上**差点重犯同一个错误**：先把字段写进 JSON 就以为修好了
（"JSON 里有字段但程序不读"正是 F0 打包器和 `resource_applicable` 的同一个坑）。
**必须有"改动该字段 → 结果必须改变"的测试才算修完。**

## 2026-09-23 · 第 2 步之 TIE 修复（GPT 8 条清单全部完成）

### GPT 复核发现的 P0
> `train_resource_stats()` 里有 `mean`/`CVaR`，但**真实 scheduler 经过 `estimate()` 时它们被丢掉了**，
> 只返回 `p50`/`p90`。**所以 TIE 实际跑的是 `p50 + β·p90`，不是 TIE。**
> **你那个 10/10 是假阳性**：测试了 bank 和纯函数，却没测真实 policy 路径最终消费了它们。

### 修复（8 条）
| # | 要求 | 实现 |
|---|---|---|
| 1 | 独立 TIE bank，不走通用 `estimate()` | `src/tracing/analysis/tie_methods.py` |
| 2 | 缺 mean/CVaR 必须 `raise`，绝不 `or p50` | `tie_current_distribution` 直接抛错 |
| 3 | lookup 只用 `(model_id, lane)` | `tie_key()`，**刻意不用 `resource_keys()`**（含 `sequence_index`，会泄漏 workflow 位置）|
| 4 | support 提前写死否则改名 | 改名 **`Empirical-TIE-adapted`**，deviation 写进 `TIE_DEVIATION` |
| 5 | CVaR 小样本严格定义 + 样本数直方图 | `upper_cvar` 用 `k=ceil((1-α)n)`；`bank_sample_report` |
| 6 | waiting decay，不复用 aging | `tie_wait_adjust`（**乘法**），测试断言它 ≠ `srtf_aging_key`（**减法**）|
| 7 | `B` 不能用 `len(free_gpus)+1` | 从 episode 的 `gpu_topology_mb` 长度取（系统常量）|
| 8 | **E2E sentinel mutation test** | 三条：跟 mean/CVaR 走、percentile 臂必须选另一个、只改 CVaR 必须翻转 |

### 实测（v04.1）
```
TIE bank: 5 组 / 6128 样本 / singleton 0% / <10 0%
样本直方图: {36: 1, 249: 1, 1552: 1, 2035: 1, 2256: 1}
tie_current 91,354（旧版 90,869，变了 => 新前端被消费）
myopic 88,456 / fcfs 101,757
```
**TIE fidelity 15/15；全量 281 个测试，5 个失败套件全部预先存在。**

### 测试又抓出我两个错误
1. `estimate()` **有最小样本数要求** —— 我最初的测试模板每模型只有 1 个样本，`estimate()` 在 TIE 之前就拒绝了
2. 样本数期望写错（模板改成 3 节点后应是 3 不是 1）

### 剩余（未做）
- **LLMSched**：`H(X) ≠ I(X;Y)`；真实 duration 未进 posterior。**fidelity FAIL**
- **Pythia**：`baseline` 不是 role alphabet。应改名 `workflow-family progress prior` 或重构为 role-PFA
- **Latency-Aware**：key 不是 Eq.12（`-boundaries_removed` 是发明的 tie-break）；**用了真值 `compute_ms` = 真值泄漏**

## 2026-09-23 · composite CPU+GPU 段做成独立状态机 + gate/TIE 收尾

### GPT 对 option (b) 首版的判定：概念 PASS，实现 P0 FAIL
它给了正确的模型：
```
R_total = R_pre + R_nested + R_post
nested_ready  = t + R_pre
nested_start  = max(nested_ready, device availability)
parent_finish = nested_finish + R_post
```
并指出首版的两个时间错误（**丢了 start offset**、**GPU 被争用后 parent 不随之延迟**）
以及 6 个具体 bug。

### 修复：独立状态机
```
parent_start -> nested_ready -> nested_gpu_start -> nested_gpu_finish -> parent_finish
```
- **数据侧**：169 个父节点新增 `nested_pre_ms` / `nested_post_ms` / `nested_inner_ms`
  （从 raw trace 的父子时间戳算）。**验证：169/169 满足 `pre + inner + post == R_total`（1ms 内）。**
- `GPU` 新增 `composite_until`（**替代 `active_node`**）：
  段**排队**而非抢占；`active_node` **完全不被 composite 触碰**
- `free_gpu` 判定加入 `composite_until`
- `process_finish` **特判 `gpu_nested` lane**：只释放**预留**，**绝不 complete parent、绝不 release 后继、
  绝不碰别的 job 的 `active_node`**
- 抢占路径跳过 composite 段
- **不再加 load**（`runtime_ms` 已含内层区间 → 原来会重复计算）
- `sequence` 改为**返回值传递**（原来传新 list，外层计数不更新 → tie-break 可能重复）

### 测试抓出的两个真 bug（都是我自己的断言/测试抓的）
1. **`gpu_nested` 事件时间写成了 `nested_start`**（应为 `nested_finish`）→ `parent_finish = 300` 而非 1000
2. **设备忙时 `nested_finish > busy_until` 恒真 → composite 覆盖了别的 job 的 `active_node`**
   （断言 `nested segment for A released a device holding B` 直接命中）→ 改为 `composite_until` 排队

### 4 条 event-level sentinel（GPT 指定，全过）
1. **无争用** → `parent_finish` **精确等于** `t + R_total`
2. **GPU 被占用** → parent 延迟 **=** nested 延迟
3. **nested finish** → **不 release parent 后继**
4. **nested finish** → **不释放不属于自己的 GPU job**

### gate 的两个洞（GPT 指出）
1. `templates != 640` / `nested_merged != 169` 原来只打印 MISMATCH **不影响退出码** →
   现在**进入聚合，fail-closed**
2. **`_has_cycle()` 原来是错的**（colour DFS 未保持递归栈）→ 换成 **Kahn 入度扫描**（精确、独立）
3. 新增 **provenance 断言**：`node_type != run_control` 且 `NOT(event_type=="run" AND raw_action=="answer")`
   —— 不只靠 `resource_applicable == true`
- **gate 仍 640/640 PASS，exit 0**

### 169 合并的集合等价（GPT 指出我只验证了 precision）
从 **raw trace 独立重新发现**，再与 projection 做 **exact set compare**：
```
projected 169 / discovered 169 / false positive 0 / false negative 0
EXACT SET EQUALITY: True
```

### TIE 的 B（GPT 纠正了自己上一轮的建议）
> 论文里的 `B` **不是 GPU 数量，而是 configured maximum batch size**。
所以 `tie_B = len(gpu_topology_mb)` **是 adaptation，不是论文原义** ——
已写进 `TIE_DEVIATION`，并要求**不得描述成 "paper's B"**。
（`γ=0.9, τ=30s` GPT 确认**是论文原值**。）

### 仍然未做
- **LLMSched**：`H(X) ≠ I(X;Y)`。GPT 说"先不要碰 LLMSched"，等 substrate 冻结
- **Pythia**：不是 role-PFA
- **Latency-Aware**：key 不是 Eq.12 + **真值泄漏**（用 `compute_ms`）
  - GPT 给了边界：**Latency-Aware 必须有自己独立的 train-only predictor**，
    **绝对不共享 F0 的输出**；可共享的只有 train split / 当前可观察 raw request fields / simulator

## 2026-09-23 · composite 预约语义修复（真正的 nested_ready 事件）

GPT 的反例：`parent_start=0, pre=1000, inner=100, post=100` 时设备在 0~1000 本应自由，
但旧代码 t=0 就设 `composite_until=1100`，把整段准备期都挡住。
且两个 composite 会重叠（第二个只看 busy_until），先完成的还会把别人的预约一起清掉。
GPT 原话：**"你成功建模了 future reservation，但把 future reservation 错当成了 current occupancy。"**

### 修复：把"占用"推迟到真正需要的时刻
```
parent_start      -> 只 push 一个 nested_ready 事件(now+pre)，【完全不碰设备】
nested_ready      -> start  = max(now, busy_until)
                     finish = start + inner
                     【此时才】收取驻留、延长 busy_until、记 composite_tail
                     设备空闲才设 active_node
                     push nested_gpu_finish(finish)
nested_gpu_finish -> 释放占用；【绝不缩小 composite_tail】
                     push parent_finish(finish + post)
正常 GPU 完成     -> busy_until = max(finish, composite_tail)
```
- 删掉 `composite_until` 与 `active_composite`（后者**从不设为 True**，抢占里那个保护**永远不生效**）
- 驻留从 `parent_start` **推迟到 `nested_ready`**（原来模型显存在真正调用前就被占）
- `busy_time_ms += inner`（原来 gpu_utilization 会漏掉这段 GPU 工作）

### 6 条 sentinel 全过（按 GPT 的确定性构造，无 skip）
1. 无争用 → `parent_finish` 精确 = `t + R_total`
2. **强制争用**（长 GPU job arrival=0/3000ms，composite arrival=10）→ `nested_start == 3000`
3. 后继必须等父节点
4. 不扰动别的 job 的窗口
5. **两个 composite 串行、不重叠**
6. **准备阶段 GPU 仍可用**（短 GPU job 在 `nested_ready` 前完成）← 抓旧 bug 的那条

### 仍待做（GPT 冻结前清单的剩余两项）
- **从 raw trace 重建 `nested_pre/post/inner` 的 producer**（loader 支持但 repo 无生成路径）
- **TIE 的独立 `Lq` + load estimator 归属**

**287 个测试，5 个失败套件全部预先存在。**

## 2026-09-23 · GPT 冻结前清单的最后两项

### 4. 从 raw trace 重建 nested_pre/post/inner 的 producer（reproducibility 收尾）
GPT 指出：loader 支持 `nested_pre_ms`/`nested_post_ms`/`nested_inner_ms`，
**但 repo 里没有生成并持久化它们的 producer** —— 测试 fixture 能造，
但"从 raw trace 干净重建 v04.1"的路径不闭合。

**修复**：把提取折进真正的生成器 `scripts/rebuild_scheduler_templates_v041_ontology.py`：
- 新增 `RAW_TRACE_ROOT`、`load_raw_trace()`、`attach_nested_resources()`
- 在 `rebuild()` 里对每个带 `nested_calls` 的父节点调用
- **硬后置条件**：`nested_pre_ms + nested_inner_ms + nested_post_ms == runtime_ms`（1ms 容差），
  不满足直接抛错 —— **干净重建 169/169 通过**
- manifest 新增 `composites_with_pre_inner_post` 与 `reconstruction_rule`

**验证**：从 v04 干净重建 v04.1 成功（8,295 → 8,126 节点，169 合并）。

### 3. TIE 的独立 `Lq` + load 归属
GPT 指出：`Lq` 必须是 TIE 独有、定义唯一的量；load 项必须来自 **TIE 自己的前端**，
而不是共享的 `estimate()` 路径。

**修复**（全部进 `tie_methods.py`）：
- `tie_queue_length(pool, competitive_priority)` —— 定义唯一，
  兼容候选池的 7 元组结构（`entry[0][0]` 才是内层 priority）
- `tie_load_estimate(bank, node)` —— 从 TIE 自己的 bank 读
- `tie_score_for(bank, node, beta, resident=...)` —— **完整分数只来自 TIE 自己的前端**
- bank 新增 `load_mean_ms` / `load_sample_count`（train-only 的 load 均值）

**验证**：`tie_current` 从 **91,354 变为 97,414** → **新归属确实被消费**；
TIE fidelity 15/15、composite 6/6、全量 287 个测试（5 个失败套件全部预先存在）。

### 至此 GPT 第 4 轮冻结前清单的 4 项全部完成
1. ✅ composite 升级为 nested_ready 事件 + queue tail
2. ✅ 6 条 sentinel
3. ✅ TIE 的 unique `Lq` + load ownership
4. ✅ raw-trace producer

## 2026-09-24 — substrate Freeze v1 获批；queued-composite 保留量必须挡住抢占

**决定**
1. GPT 复核（HEAD `2052a8f`）判定 substrate **Freeze v1 批准**。范围：v04.1 ontology/topology、
   composite 非抢占执行语义、GPU placement/admission、TIE、gate 与 raw rebuild。
   `myopic_preempt` / `batch_myopic_preempt` 列为**非正式 extension**，不再阻塞全局 freeze。
2. **queued composite 的保留量优先于抢占**：设备上有 composite 段持有或排队时，
   该设备的 active node **不允许被抢占**。

**原因**
- 抢占路径只从 `active_node` / `busy_until` 判断空闲。sentinel 7 覆盖的顺序里保留量尚不存在；
  GPT 给出未被覆盖的顺序：N 占 0..4000 → composite 在 nested_ready 入队 [4000,4700] 并把
  `busy_until`/`composite_tail` 置 4700 → 抢占在 t=200 到达，把 `busy_until` 回绕到 200，
  而 `composite_tail` 仍是 4700。此时设备看起来空闲，普通 dispatch 会穿过已保留窗口。
- 备选方案 `busy_until = max(now, composite_tail)` 被否：composite 起点仍按旧的前驱 finish 计算，
  会留下一段保守空闲，语义更差且更难解释。
- 选中的方案代价最多是一次前驱重算，且只发生在本来就属于 extension 的路径上；
  语义简单、确定、可解释。

**替代方案与拒绝原因**
- `busy_until = max(now, composite_tail)`：留下保守空闲窗口，拒绝。

**验证**
- sentinel 8：保留窗口仍 start 一次 / finish 一次 / 父完成一次；优先级作业既不在
  [4000,4700] 内 start 也不在其中 finish。
- composite sentinel 8/8；TIE 21/21；全量 295 测试（5 个失败套件全部预先存在）。
- 真实 v04.1 bank：5 组、0 缺 `load_mean_ms`、`fraction_lt_10=0`、`fraction_singleton=0`。

## 2026-09-24 — 缺失的 load 字段不是零

**决定**：`tie_load_estimate` 在 `load_mean_ms` **字段缺失**时 `raise KeyError`；
只有**显式 null** 才合法地表示 0（记录该组没有 load 样本）。

**原因**：原实现 `group.get("load_mean_ms", 0.0)` 把字段缺失静默读成零负载，
会让一个 bank 构造 bug 表现为「这个模型加载免费」，从而污染 TIE 分数。
真实 `build_tie_bank()` 始终写该字段，正式路径不受影响。

**验证**：真实 bank 检查 0 缺失；手工 fixture 已补齐字段；TIE 21/21。

## 2026-09-25 — LLMSched Baseline Freeze v1 获批

**决定**：LLMSched-adapted 正式冻结。

- **HEAD**：`618b26bcf717c2bfb00b3545d58dccfb027f494e`
- **状态**：APPROVED / FROZEN（GPT 于第 11 轮复核后签字）

**冻结范围**
canonical-stage ontology；train-only BN structure/CPDs/discretizer；ABSENT + duration bins 状态语义；
completed intrinsic-duration evidence；exact posterior inference；Eq.6 uncertainty score；
`MAX_JOINT_FUTURE=4` 与其 shortest-path selection；whole-job remaining-duration estimator / support interval；
Algorithm 1 non-overlapping duration sets；epsilon-greedy EXPLORE/EXPLOIT consumer；以及 L1–L7 / L4b 与相关 sentinel gates。

**后续不得改动**
stage ontology；BN structure-learning / CPD semantics；evidence definition；posterior semantics；Eq.6；
`MAX_JOINT_FUTURE=4` 或其 selection rule；remaining-work / interval semantics；
non-overlapping-set construction / order；epsilon policy；LLMSched-specific simulator interface。

**唯一例外**：出现**可复现 correctness bug**。若发生，必须单独记录为 **freeze-breaking fix**，
**不得为改善结果而调参**。特别是 freeze 后不得原地修改 `cpds` / `stage_order` ——
当且仅当该假设成立，profiler 上的 posterior memoization 才是安全的（缓存键含 CPD 表与
`stage_order` 的 identity，替换即失效）。

**Freeze 前关闭的全部缺陷（供论文 limitation 引用）**
1. **拓扑泄漏（P0）**：`Y` 曾取自 realized template 的未执行后缀，即泄漏结构真相。改为自学习到的网络 − 已观测。
2. **`Range` 连乘**：应为 `Σ Range`；`max(1.0, range)` 地板移除。
3. **pairwise sum ≠ joint MI**：改为截断集合上的精确 joint。
4. **L2 用一致率冒充互信息**：熵型打分器仍能通过。改为对手工网络直接断言 production scorer。
5. **三个静默错值**：`stage_range_ms` 缺字段返回 0.0；`profiler_sha256` 未覆盖 CPDs；
   证据回退到 template 时长。
6. **MI 与 Range 用了不同集合**：Eq.6 两边必须同一组 `Y₁..Y_M`。
7. **ready candidate 未条件化 `X != ABSENT`**：全部变量带 ABSENT 态会给出虚高探索分。
8. **interval 是期望不是支撑**：`P(ABSENT)=P(D=100)=0.5` 报 `[50,50]`，真实支撑 `[0,100]`。
9. **interval 只覆盖候选的相关后代**：应为整个 job 的模型侧未解工作。
10. **后验缓存未绑定网络身份**：浅拷贝共享缓存 → 改动 CPD 后返回旧后验。

**替代方案与拒绝原因**
- 保留 `∑_i I(X;Y_i|E)`：与论文的 `I(X;Y₁..Y_M|E)` 不等（`Y₁=Y₂=X` 时差一倍），拒绝。
- 不截断 `Y` 做精确 joint：`|Y|` 实测 max 35 / 均值 16.8，joint 为 7^17，不可行，拒绝。
- 用 `P(present) × min/max` 作为区间：把支撑塌成期望，会让两个都真正不确定的 job 被当成时长已知来排序，拒绝。

**验证**
v2 gate `tests/test_llmsched_bn_v2.py` **34 项**；全量 **329 测试**（5 个失败套件全部预先存在）；
端到端 EXPLOIT 190479.7 / EXPLORE 178539.8，两模式确实不同，均完成 6 个 validation job。

## 2026-09-25 — Pythia Baseline Freeze v1 获批

**决定**：Pythia-adapted 正式冻结。

- **HEAD**：`26484a4e545e88a55556e70c91e2b1cb62272ac`
- **状态**：APPROVED / FROZEN（声明已收到）

**冻结范围**
role-level PFA（状态 = role、转移 = train-only 经验分布、低于 `min_prob` 的转移剪枝并归一化）、
有限 horizon 期望剩余距离、`current_role` 索引、`S_completion = 1/E[D_remaining]` 与
`omega1=1, omega2=0` 的 completion-aware priority、以及 `tests/test_pythia_fidelity.py` 的 12 项 gate。

**后续不得改动**
role alphabet 的定义（`role:action_family:raw_action`，**不含** occurrence 后缀）；PFA 结构与转移语义；
剪枝阈值与归一化；`DEFAULT_HORIZON`；期望剩余距离的递归；`current_role` 索引语义；
`S_completion`；epsilon 无关的 completion priority；Pythia-specific simulator 接口。

**唯一例外**：可复现 correctness bug，须单独记为 **freeze-breaking fix**，不得为改善结果调参。

**Freeze 前关闭的缺陷**
1. **alphabet 用错（P0）**：旧 frontend 以 `template.baseline` 为 alphabet —— 那是收集时记录的
   workflow family 标签，不是 role alphabet，且**根本没有建模 role**。改为真正的 role-level PFA。
2. **状态用错 + 当前节点双算（P0）**：consumer 以 `last_observed_role` 索引，而 `V(role)` 是
   **该 role 之后**的剩余工作。链条 A→B→C（10/20/40）在 B ready 时得 `d(B)+V(A)=80`，
   真实应为 `d(B)+V(B)=60`。改为从**候选自己的 role**（= Pythia 的 `agent_id`）索引。
   实测端到端 makespan 182738.8 → **182025.1**，证明双算在生效路径上。
3. **措辞不实**：曾把有限 horizon 写成「论文的 bounded probable future」。论文实际是
   **低概率转移剪枝 + 对循环 role 的 repetition bounds**（`engineer^{3,6}`）；`H=6` 是我们的
   **固定、train-independent 计算近似**，不是论文公式。已改。

**替代方案与拒绝原因**
- 保留 `baseline` 作为 alphabet：不是 role alphabet，拒绝。
- 保留 occurrence 后缀（`#0/#1/#2`）：论文用**重复 role** 表达循环，加后缀会把一个循环 role
  拆成多个位置状态，恰好破坏 automaton 的用途，拒绝。
- 用平稳解代替有限 horizon：环状 automaton 上无折扣期望不收敛，拒绝。

**验证**
Pythia gate `tests/test_pythia_fidelity.py` **12 项**；全量 **333 测试**（5 个失败套件全部预先存在）；
v04.1 上 11 roles / 26 retained edges / 4 pruned / 480 train runs，`E[rem]` 跨 0–20627 ms；
6 个 validation job 全部完成。

### Pythia freeze 声明的补充（GPT 最终回复）

**不得根据最终实验表现调整** horizon、role 粒度、pruning threshold 或 priority 形式；
需要调整时必须作为**新的 variant**，不覆盖 frozen baseline。

**措辞更正（重要）**：manifest 里不要再写「finite horizon 代替平稳解」——
Pythia 原文**并不是**用 stationary solve 求这个量。正确说法是：

> finite-horizon recursion replaces Pythia's bounded-regex / repetition-bound
> representation for expected remaining-distance estimation.

**论文 limitation 要点（GPT 指定要写进 manifest 的）**
- faithful：历史 train traces → role-level PFA；alphabet 表示 **agent roles** 而非
  workflow-family label；同一循环 role 保持同一状态（不加 occurrence-position suffix）；
  train-only 经验 next-role 转移；低概率 transition pruning + retained-row 重新归一化
  （论文明确用 dominant-percentile filtering，约 5%）；当前 ready request 以**当前
  agent_id/role** 访问 workflow profile；由预测的 workflow future 计算 expected remaining distance；
  `S_completion = 1/E[D_remaining]` 作 completion-aware priority。
- adapted：原论文 `agent_id` → VideoSeek 的 `role:action_family:raw_action`（裸 role 过粗）；
  PFA/duration statistics → VideoSeek intrinsic runtime 与统一 GPU simulator；
  node 是本实验的调度执行单元；原系统 global scheduling semantics → 统一 candidate-pool dispatch；
  `ω1=1, ω2=0`，因本 simulator 无等价于论文 model-replica queue 的 `S_unblock` 状态。
- deviation：`H=6` 固定有限 horizon 是我们的近似（论文用低概率 transition filtering 加
  循环 repetition bounds，如 `(engineer)^{3,6}`，构造 bounded probable workflow）；
  不迁移 `S_unblock` / DownstreamIdleRisk；不迁移 speculative/prefix cache routing、
  model-replica idleness、autoscaling 及原 Pythia 的 serving/batching infrastructure
  （不在统一 GPU abstraction 内）。论文 Algorithm 3 原版 priority 同时含
  `S_completion` 与 `S_unblock`。

## 2026-09-25 — TIE 与 Latency-Aware Baseline Freeze v1 获批；四条基线实现层全部冻结

**HEAD**：`bc6cbec97dc35d407f8ec250fa729fdfc08ca357`　**状态**：APPROVED / FROZEN

### Empirical-TIE-adapted Baseline Freeze v1
**冻结范围**：train-only `(model_id, lane)` empirical **compute** bank；CVaR(α=0.9)；
`TIE = E[X] + β·CVaR`；`β = clip(0.1·L_q/B, 0.1, 0.5)`；unique-ready-unit `L_q`；
episode GPU service concurrency 映射为 `B`；residency-aware load surcharge；
multiplicative waiting decay；hard-priority-preserving consumer；全部 TIE gates。

**不得改动**：bank key；**compute/load decomposition**；empirical distribution semantics；
CVaR α=0.9；small-sample rule；`L_q` 定义；`B` 映射；β 公式；waiting decay γ/τ；
load residency semantics；fail-closed 行为；consumer ordering。

### Latency-Aware-Orchestration-adapted Baseline Freeze v1
**冻结范围**：train-only request-conditioned predictor；四级 tier backoff；`min_support=3`；
**compute/load/total-peak 三头语义**；predictor mandatory / fail-closed contract；
fusion constructor；predicted completion planning；memory feasibility；near-ready prefetch；
Eq.12 单次 commit 字典序映射；独立 `latency_aware` scheduler；全部 truth-leak /
non-degeneracy / fusion / scheduler gates。

**不得改动**：四级 feature tier；`min_support=3`；finest-supported-tier backoff；
`compute = runtime − load` target；load 使用**同一** predictor；total-peak memory semantics；
predictor mandatory contract；fusion eligibility；fusion duration/load semantics；
memory admission；Eq.12 ordering；near-ready prefetch mapping；正式 consumer 接口。

### 两条的共同例外
可复现 correctness bug；必须留下 **freeze-breaking provenance**，
**不允许根据最终性能调 tier、support threshold 或 key**。

### Freeze 前关闭的关键缺陷（供论文 limitation 引用）
- **TIE zero-load builder bug**：`if getattr(n, "load_ms", None):` 把合法的 `load_ms == 0.0`
  （模型已驻留）当成**没有样本**。训练 load `[0, 10]` 时 bank 得 mean **10** 而非 **5**。
- **TIE Lq 重复计数**：`tie_queue_length` 数的是 candidate tuple 而非 waiting unit
  （2 节点 × 2 空闲 GPU 报 4）。
- **统一 decomposition 错误（两条都中）**：substrate 的 `compute_ms = runtime_ms − load_ms`，
  即 `runtime_ms` **已含 load**。两条都在其上再加一个 load 项 → non-resident **双算**、
  resident 则把历史 load 留在均值里。另有准入显存重复加 `resident_model_mb`
  （`workspace_peak_mb` 本已是含 resident model 的**总 peak**）。
- **Latency-Aware 真值泄漏**：`predicted_duration_ms` 读 `node.compute_ms`（真实内在计算时间）。
- **Latency-Aware 发明的 tie-break**：Eq.12 key 里的 `-boundaries_removed` 论文没有。
- **truth fallback 未结构性闭合**：simulator wrapper 挡住了，但 scheduler 自己的入口仍接受
  `predictor=None` 并读 `node.workspace_peak_mb`。
- **non-degeneracy gate 假阳性**：直接比较 `tables[1]`，绕过 `min_support` 与 production backoff。

### 下一步顺序（GPT 锁定，严格执行）
1. `SchedulerTopologyContractGate` freeze
2. `BaselineFidelityManifest` freeze
3. **四条联合基线正式实验**

正式实验前**不再改** LLMSched / Pythia / TIE / Latency-Aware 的任何 baseline semantics。
之后再发现问题，只按「可复现 correctness bug → freeze-breaking commit → 重做受影响结果」处理。

## 2026-09-25 — 四基线的 freeze-breaking 修复（LLMSched / Pythia / Latency-Aware）

**背景**：全链路审计（`docs/research/2026-09-25_four_baseline_review_gpt.md`）判定「先不要启动 300×5 正式测量」，
列出若干 freeze-breaking correctness 项。本轮按「只修可复现 correctness bug、不为改善结果调参」的例外条款处理，
逐项修复并在 gate 上钉死。冻结清单新增 `freeze_breaking_fixes` 字段记录同一内容，并置
`pending_freeze_breaking_commit = true`（尚未提交）。

### 1. LLMSched — EXPLOIT 的信息状态与覆盖范围
- **缺陷**：`expected_remaining_ms()` 只走 `_future_from_network`（候选的网络后代），且 future posterior 只用
  completed evidence；而 EXPLORE 与 current service 都条件在 `X != ABSENT`。后果：与候选无路径关系、但仍属
  job remaining 的 stage 被静默丢掉；同一决策内三个消费者处在不同信息状态。
- **修复**：新增 `expected_job_remaining_ms(profiler, evidence, current_stage)`，遍历**全部**未完成 stage，
  每个 posterior 用新增的 `conditional_state_probs()` 条件在 `current_stage != ABSENT`；
  `job_duration_interval_ms(..., known_present=...)` 让 ready stage 不再贡献虚假的 0；删除死代码 load 项
  （intrinsic duration 已含冷加载）。simulator EXPLOIT 改用它，grouping 的 interval 传入 ready stage 集合。
  **Eq.6（correlated descendants + top-4）不动。**
- **验证**：新增 `L8WholeJobRemainingAndKnownPresent`（4 项），LLMSched gate **38/38**。

### 2. Pythia — 期望剩余「距离」应为步数而非毫秒
- **缺陷**：有限 horizon DP 累加的是 role 的**毫秒**均值，consumer 又加 `runtime_p50 + load`，使该臂变成
  duration-weighted / SRTF 风格，偏离 Algorithm 3 的 expected remaining **distance**。
- **修复**：`V(role)` 改为**未来 role 转移数的期望**（到 `<end>` 的转移不算一个 role、不计数）；
  `S_completion = 1 / (1 + V(role))`；consumer 不再加任何毫秒项。schema 升为 `pythia-role-pfa-v2`，旧 ms 产物 fail-closed。
- **验证**：新增 `test_the_distance_does_not_depend_on_durations`，并把手算值与角色索引测试改成步数；Pythia gate **13/13**。

### 3. Latency-Aware — load head 与 prefetch 时序
- **load head（`2fbb44a`）**：run/peak/load 曾共用四级 request tier，而论文是 `T_load(d,g)`。改为 load 单独
  `(model_id, lane)` key + train-wide 默认；非退化 gate 不再把 load 计入。
- **prefetch 时序（本轮）**：Eq (5) prefetch 原在**主循环顶部、ready dispatch 之前**运行，eligibility 只看
  `active_node`/`prefetch_pending`，可能抢占用作 ready work 的设备并把 `busy_until` 推后。改为
  **dispatch 之后**、且只在「所有 runnable 单元都安置完后仍空闲」的设备上执行。
- **信息契约**：新增模块常量 `latency_aware_predictor.GRAPH_VISIBILITY_CONTRACT = "logical_graph_assumed_known"`，
  冻结清单写入 `information_contract`：论文 Eq (3)/(5) 以**已知逻辑图**为前提，Constructor / near-ready 读
  realized template 是其自身前提，**不是** LLMSched 式的结构泄漏；因此主表比较的是完整系统。
- **验证**：新增 `test_prefetch_does_not_delay_ready_work`（旧时序 250 ms / 新时序 150 ms 可区分）；LA gate 45 全绿。

### 4. 尾巴清理
- `latency_aware_predictor.py` 的 `_load_key` ×3 / `LOAD_TIER` ×2 重复定义去重。
- `four_joint_baselines.py` bootstrap 文案由 "video-cluster" 改为 **episode-cluster**（实现本就按 episode 重采样；
  不再暗示可泛化到新视频）。

### 未修 / 待定
- **TIE load adaptation 的「窄条件统计」问题**：原始审计回复在 LLMSched cache 处被截断，细节**没有保存在仓库**，
  无法从本地证据确定具体缺陷。**本轮不臆测、不改 TIE**，待补齐原文后再处理。
- 上述均为 freeze-breaking 修复，**尚未 commit**；`baseline_fidelity_manifest_v1.json` 已按新 gate 重新生成，
  但冻结清单 `head` 仍指向旧 commit，需在提交后重新 pin。

### 全量验证
- `python3 -m unittest discover -s tests`：356 测试，8 error 全为环境（缺 torch / py3.9 `write_text(newline=)`），
  1 failure 是**预存的** `round_robin` myopic（已用 HEAD 原文件复现确认与本次无关）。
- `baseline_fidelity_manifest.py` PASS（四臂 freeze=APPROVED + gate=green）。

### 5. 独立 review 追加发现并修复
- **P0（预存）**：`scripts/four_joint_baselines.py` 使用了 `args.smoke`（209/234/280）但 `--smoke` 从未注册，
  任何 CLI 调用都会 `AttributeError`。已补 `--smoke N`（文档早已声明）。1 集 smoke 在真实 v04.1 上端到端跑通（18 s，四臂全跑，`git_head` 已非空）。
- **P1**：`BaselineFidelityManifest` 原先只跑 `test_latency_aware_fidelity.py`，不含新的 prefetch 回归。
  已给 latency 臂加 `extra_gate_files = [test_latency_aware_scheduler.py]`，两门都绿才 PASS；
  并在 fidelity gate 增加 `InformationContractTests` 机器校验 `GRAPH_VISIBILITY_CONTRACT`。
- **P1（已知）**：冻结清单 `head` 仍指向旧 commit；提交后重新 pin。

### 6. 提交与重新 pin（收尾）
- 已提交并以 token 推送：**`54a2d25088832049afb0cf860cdabfe5152c8c3d`**（`2fbb44a..54a2d25 main`）。
  提交只含本轮 17 个文件，未带入工作树其余 CRLF 行尾噪声（提交前把这些文件转回 LF，匹配仓库约定）。
- 四份冻结清单已去 `pending_freeze_breaking_commit`；LLMSched / Pythia / Latency 的 `freeze.head` 与
  `BaselineFidelityManifest` 的 `freeze_head` 重新 pin 到 `54a2d25`（TIE 未改，保留 `bc6cbec9`）。
- 重新生成的 `baseline_fidelity_manifest_v1.json` **PASS**。
- 说明：本轮修复属 freeze 例外（可复现 correctness bug），**未**重开 GPT 的 freeze 签字流程；
  若需按新 head 重新走一次 freeze 声明，另行安排。

## 2026-09-26 — 网页版 GPT 复核本轮 freeze-breaking 修复（结论 NEEDS CHANGE）

**来源**：`docs/research/2026-09-26_freeze_breaking_fixes_review.md`（ChatGPT Web，GPT-5.6 Sol High；
基于公开仓库 `2fbb44a...54a2d25` 与 `c06a170` 的实际 diff）。这是**独立复核**，不是本地测试的复述。

### 复核接受
- LLMSched whole-job remaining 与 `conditional_state_probs` 的条件化/归一化数学正确，且未误删 query 自身的 ABSENT。
- Pythia 毫秒→步数修复正确；`1/(1+V)` 的 +1 是自洽 distance convention。
- TIE 核心公式、compute/load decomposition、CVaR、adaptive beta 无问题。
- Latency 的 load head（deployment-only）与 prefetch 移到 dispatch 之后：对**当前 ready** 的 work 已修好。
- runner 的 survivor-mean P0 已关闭。

### 复核推翻（重要，纠正本日志 2026-09-25 §3）
- **Latency-Aware 读完整 realized template 仍是 P0，不是「论文前提」。** 复核指出论文假设的是
  **resolved logical window**（只含已解析 branch + ready/immediate near-ready），而非完整 realized 未来图。
  我在 §3 记的 `GRAPH_VISIBILITY_CONTRACT = "logical_graph_assumed_known"` **解释作废**；
  `latency_aware_fusion.py` / `latency_aware_lifecycle.py` 读完整 `job.template` 属真值结构泄漏，
  需改为只读 resolved/visible 部分，并加 **identical-prefix / different-future sentinel**。
- **TIE 被截断项补全**：`load_mean_ms` 是 resident(0)+nonresident(>0) 的 **unconditional mean**；live 已知道
  candidate 为 nonresident，应使用 **E[load | load required] = mean(load_ms>0)**。sentinel: `[0,10]` → cold mean `10`，
  resident surcharge `0`。**这条要正式实验前修，因为它可能改变 TIE 排序。**
- **Pythia waiting-time aging 实际未实现**：注释称由 hard priority 承担，但 hard priority 是固定 0/1，不是论文的
  accumulated-wait aging。→ 实现，或在 freeze manifest 明确列为 omission 并删掉错误注释。
- **LLMSched interval 未把 known-present 条件传播到其它 stage**：现为 conservative envelope，不宜称 exact support。
- **runner formal gate 仍不够 fail-closed**：formal 模式需硬断言 freeze manifest/gate/episode+split+resource hash，
  且不允许 `git_head == "unknown"`。
- **P2**：`FusedChain.summed_runtime_ms` 是无消费的 truth 字段，建议删除。

### 复核结论与 gate
- **NEEDS CHANGE**；Latency-Aware 为 **REJECT（P0）**。
- 30 集**工程诊断** smoke：可以；30 集作为 **freeze-qualified** smoke / 300×5：**不批准**。
- `c06a170` 的 re-pin **不等于**独立 freeze 签字；freeze-breaking 语义改动后应按新 head 重走一次 freeze 声明。

### 下一步最小项（待用户决定是否执行）
1. Latency resolved/visible graph view（P0，先做）
2. TIE conditional cold-load estimate（P1）
3. Pythia aging：实现 或 明确 omission + 删错误注释（P1）
4. LLMSched interval 全 stage 条件化 或 改称 envelope + over-approx 测试（P1）
5. runner formal hard gate（P1）+ FusedChain truth 字段清理（P2）
→ 然后重新生成 manifest、新 SHA、重新走 freeze 声明，再 30 集 freeze smoke，最后 300×5。

## 2026-09-26 — 按复核执行 P0/P1 修复（第二轮）

依据 `docs/research/2026-09-26_freeze_breaking_fixes_review.md` 的清单逐项修复；均为 correctness/fidelity，未调参。

### P0 — Latency-Aware resolved/visible graph view
- 新增 `latency_aware_lifecycle.resolved_graph_view(job)`：只暴露已 resolved 的节点（ready/running/completed/failed）
  加 **RUNNING 且唯一后继** 的近 ready 窗口；`near_ready_deployments` 只在该视图内取 running 的后继。
- `latency_aware_fusion.maximal_fusible_chains(template, visible_ids=...)`：链只落在 resolved 视图内；
  `rank_ready` 改为 `chains_for(job)`（状态相关，不再按 template 缓存）。
- 删除 `FusedChain.summed_runtime_ms`（无消费的真值字段，P2）。
- sentinel：`ResolvedWindowTests`（相同可见前缀 / 不同未来必须不可区分；running 的分叉节点不暴露后继；
  唯一后继的 running 节点进 near-ready）。
- **行为影响（必须让用户拍板）**：v04.1 上 fusion 因此**完全不再触发**（改前 3 集约 39 次融合，改后 0）；
  Latency-Aware 变弱：3 集 starts 172/161/163 → 211/198/205，1 集 smoke Δ 从 +4087ms → +5850ms。
  near-ready 不变（workload 全节点后继数 ≤1）。这是 GPT 明确接受的「比论文弱但不泄漏」保守解；
  **备选**：仅当后继唯一（无分支）时才允许跨 ready 节点融合——行为不变，但不满足 GPT 的 identical-prefix sentinel。

### P1 — TIE conditional cold-load
- bank 新增 `load_required_count / load_occurrence_rate / load_conditional_mean_ms`；consumer 对 nonresident 用
  `E[load | load>0]`（不再用含 resident 0 的 unconditional mean）。
- sentinel：`[0,10]` → occurrence 0.5、cold mean 10、resident surcharge 0 / nonresident 10；gate 24/24。

### P1 — Pythia aging
- 删除「aging 由 hard priority 承担」的错误注释，明确 hard priority 是固定 0/1、不是论文的 accumulated-wait aging；
  在 freeze manifest `omissions` 记录未迁移 + `V_plus_one_convention`。gate 13/13。

### P1 — LLMSched interval
- 新增 `conditional_state_probs_for_set`；`job_duration_interval_ms` 对所有未完成 stage 都条件在 known-present 集合上，
  与 EXPLOIT / current service 同一信息状态。gate 38/38。

### P1/P2 — runner formal gate
- 新增 `--formal`：硬断言 `BaselineFidelityManifest.pass`、`SchedulerTopologyContractGate.pass`、split=700/300、
  `git_head` 为 40 位 commit（不允许 "unknown"）。`--formal --smoke 1` 端到端通过。

### 验证
- 聚焦 gate 138/138；全量 362 测试，仅预存 `round_robin` failure + 8 个环境 error；fidelity manifest PASS。
- **未 commit/未 push**；未重走 GPT freeze 声明。

## 2026-09-26 — 按复核执行 P0/P1 修复 + Latency 信息契约更正 + 新增 Agentix 基线

### A. 复核 P0/P1 修复（freeze-breaking）
- **Latency resolved-graph view**：新增 `latency_aware_lifecycle.resolved_graph_view(job)`（只暴露已 resolved 节点 +
  RUNNING 且唯一后继的近 ready 窗口）；`maximal_fusible_chains(template, visible_ids=)` 只允许落入该视图；
  `rank_ready` 改用状态相关的 `chains_for(job)`；删除 `FusedChain.summed_runtime_ms`（无消费真值字段）。
  新增 `ResolvedWindowTests` sentinel。
- **TIE cold-load**：bank 增 `load_required_count/load_occurrence_rate/load_conditional_mean_ms`；nonresident 用
  `E[load | load>0]`。[0,10] sentinel。gate 24/24。
- **Pythia**：删错误 aging 注释；freeze manifest 记为 omission + `V_plus_one_convention`。gate 13/13。
- **LLMSched**：`conditional_state_probs_for_set`，interval 全 stage 条件化。gate 38/38。
- **runner**：`--formal` 硬断言 fidelity/topology gate、split 700/300、git HEAD 非 "unknown"。
- **全量**：372 测试，仅预存 round_robin failure + 8 环境 error；fidelity manifest PASS。

### B. Latency 信息契约更正（依论文原文，纠正 2fbb44a 的过度声明）
我读了 arXiv:2609.03335v1 §3.2/§2.1/§3.4/Eq(3)：论文观察的是 **resolved portion `L_t`**，
"A branch enters only after its control result resolves"，planning window 不含 unresolved branch；
fusion legality（Eq 3）是 **resolved 图上的一对一边**。
- → 冻结清单的 `logical_graph_assumed_known` 过强，**更正为 `runtime_resolved_logical_graph`**（代码常量、测试、清单同步）。
- GPT 复核第二轮也据此**撤回**了它"ready 节点的后继一律不算 resolved"的严格读法（A 才是对的、B 会泄漏）。

### C. 实证：本 workload 上 Latency 的 fusion 无合法机会（重要披露）
对 v04.1（640 模板；`videomme.langgraph_react`/`videomme.star`）统计：
- planner 之后接什么**不定**（temporal 1268 / spatial 416 / generalist 142 / planner 104 / answer 3）；
- tool 之后是 planner(3011) 或 answer(596) → 继续/终止也是运行时决定；
- **没有任何 `tool → tool`**；所有相邻同模型对都跨一个运行时决策点。
→ 本 workload 是**决策驱动 ReAct 流程**，读 template 往前看 = 偷看 planner 决定。**Latency 的 Constructor 在此
workload 上结构性 N/A**（改前约 39 次融合全部是越界融合，已正确移除）。这不是"把对手改弱"，而是 workload 本身
没有论文那种"固定连续同模型段"。论文需如实披露（GPT 给的英文措辞见
`docs/research/2026-09-26_baseline_applicability_review.md`）。

### D. 新增基线：Agentix-adapted（NSDI 2026，原 Autellix）
依据 `docs/research/2026-09-26_baseline_applicability_review.md`：GPT 建议**不删 Latency-Aware**，而是把它降为
"可用部件对手"（fusion=N/A），并补一个真正匹配"未来未展开"场景的对手——**Agentix**。
- 实现：`src/tracing/analysis/agentix_methods.py`（PLAS = 已完成调用服务之和；ATLAS = 已完成关键路径最大值）；
  模拟器新增 `agentix` 策略（`policy_context['agentix_mode']` = plas|atlas），优先级 = 程序已达服务，小的先跑。
- **non-clairvoyant 由 gate 钉死**：只读 `job.completed`；未执行节点的 runtime 改动**不影响**分数。
- 迁移声明（写入 `AGENTIX_DEVIATION`）：只移植优先级规则，原论文的引擎内 batching / 跨引擎路由 / KV 局部性**未迁移**。
- gate：`tests/test_agentix_fidelity.py` 10/10；真实 confirm 集 1 集跑通（16/16，mean 82449）。
- 实证核实：Agentix 真实（USENIX NSDI 2026，arXiv:2502.13965）。

### E. 下一步（待用户）
- Agentix 是否进正式五臂（或替换 Latency 进四臂）；是否再补 Maestro-adapted（ICDCS 2026, arXiv:2606.12950）。
- 本轮改动**待 commit/push**。

## 2026-09-26 — Pythia Algorithm 3 补齐（S_unblock + aging）+ Agentix ATLAS 去伪声明

依据 `docs/research/2026-09-26_baseline_strengthening_adjudication.md`（GPT 原则：**可表达的机制就补；缺原语的写 omission，不硬造**）。

### Pythia 补齐（取代 completion-only 版本成为正式主基线）
- 新增 `src/tracing/analysis/pythia_methods.py`：
  - `pythia_base_priority = omega1*S_completion + omega2*S_unblock`（补回论文的第二项）；
  - `pythia_effective_priority = S_base + lambda*(wait/tau)`，**无量纲**（未复用 SRTF 的"毫秒-毫秒"aging）。
- **S_unblock = DownstreamIdleRisk**：对 PFA **可达未来 role**（`reachable_future_roles`）取 `S_completion(V(a))` 的**均值**，
  仅当该 role 的**训练期映射 model**（`profiler["role_model"]`）当前**没有** ready/running 需求（**队列需求代理**）时计入。
  **只读训练期 PFA + 当前可见 demand，绝不读模板。**
- `pythia_profiler` 新增训练期 `role_model`（role→多数 model）与 `reachable_future_roles()`；schema → **`pythia-role-pfa-v3`**。
- 冻结常量：`omega1=1, omega2=1, lambda=1, tau=30000ms`（论文只给形式不给值；**预注册默认**，待 dev 网格一次性选定后冻结）。
- 迁移声明（写入清单）：replica queue depth → **queue-demand proxy**；aging 是我们的无量纲 re-score，非论文 worker loop；
  S_unblock 取**均值**（论文是未标定尺度的求和）。

### Non-degeneracy（3 个 confirm 集实测）
- `completion only` vs `+aging` vs `+S_unblock` vs `both` **互不相同** → 两机制都生效。
- **S_unblock 非零率 660/729 = 90.5%**（候选级评分）。

### 未补（GPT 判定，已记录）
- **Agentix K 队列 / 抢占 / 防饿死**：依赖我们没有的 token/KV 抢占原语，**硬补比 omission 更不忠实** → 正式臂固定 **PLAS non-preemptive**，其余列 unavailable。
- **LLMSched `sample_tasks(r)`**：需要"stage 内可部分执行的 task 集"，当前 node 抽象不提供 → 保持 omission。

### Agentix 去伪声明
- 删除 `critical_path_service_ms` 里"重算与论文等价"的错误声明，改标 **ATLAS VARIANT**（精确重算 ≠ 论文的"每程序一标量继承"）。

### 验证
- 聚焦 gate **147/147**（Pythia 13→24）；全量 **383** 测试，仅预存 `round_robin` + 8 环境 error；fidelity manifest **PASS**。
- **待 commit/push**；GPT 建议补齐后按新 head **重新 freeze Pythia**（清单已写 `algorithm3_restoration_20260926`）。
