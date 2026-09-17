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
- Reason: `r95=180.4s` 与 `trueopt_h5≈183.8s` 的差值同时混合信息源、排序 key 形状与 future 项定义，
  不能支撑任何"预测优于真值"结论；先对齐消费函数形状，再回答"真值为何打不赢预测值"。
- Alternatives considered: 直接修 `trueopt_h5`/`predopt_h5` 的 key 顺序（会改变已发布 baseline 语义、需重跑全部矩阵，已拒绝）；
  使用既有 `aligned_trueopt_h5`（其 Pred 用 5 个 synthetic event step、True 用 5 个 DAG layer，H 语义不同，已判定过宽）。
- Consequence: Truth-SameConsumer 成为下一步正式实验（未跑，需门禁）；不可消除的剩余差异（预测链 vs 真后继链）
  必须在论文中显式声明；`trueopt_h5` 历史结果只能标注为 "legacy key shape"，不得当作 oracle 上界。
