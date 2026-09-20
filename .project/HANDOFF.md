# Handoff

**Last Updated:** 2026-09-20

## 当前状态（2026-09-20，权威）

**阶段：Phase R 资源头 v2 → 6 臂调度 smoke 的「写代码」阶段。**

- **Phase R 已完成并封存**：R1b（MLP + 16 格 mass-balanced 离散分布）在**原验收协议**下 pinball **−21.0%**、
  校准同时改善；R3a-U pinball −23.6% 但**违反原 calibration integrity gate**（cov90 0.803）→ 只能作诊断臂。
- **资源头 v2 artifact 打包器已写完并推送**：`scripts/pack_resource_v2_artifacts.py`，
  提交 `78f000a`，实测每臂 9,575 节点 / 35,362 步 / 22 秒，已生成 `outputs/resource_v2_artifacts/{r1b,r3a_u}`。
- **GPT 审查已完成（2026-09-20，Sol+High，思考 10m02s）**：
  `docs/research/2026-09-20_fas_resource_v2_packer_review_gpt.md`
  - 裁决：**地基正确，不需要返工 Phase R**；Q2(a) 受控对比 / Q2(b) res_hidden 重放等价性 /
    Q2(c) probs-quantiles 不会新旧混用 —— **三项全部 VERIFIED**。
  - 但 artifact 契约仍是「能用的研究脚本」，**未达 fail-closed 标准**：
    **消费臂可以写，正式 300 集 smoke 不得启动**，直到 5 个 P0 修完。
  - 三个最关键修复（都不是算法）：**严格完备性断言 / resource-v2 overlay loader / 打包后分布-视图不变性审计**。
- **下一步（按序）**：
  0. ~~用户签字冻结 NI margin~~ **已于 2026-09-20 冻结**：`δ_NI = 485 ms = 0.1 × 4853`
     （4853 = Phase 20B 已冻结的 dev700 上 `p95 − E0`），落盘于
     `.project/EXPERIMENT_GATE.json → ...smoke_preregistration_20260920`（`status = FROZEN_BEFORE_RESULTS`）。
  1. ~~修 packer 的 P0-1 / P0-3 / P0-4~~ **已完成并实测通过**（2026-09-20）：
     - P0-1 fail-closed：锚点/节点集不符、重复 node_id、scenario ≠ 1、step 多于预测槽位**全部 raise**；
       收尾硬后置条件 `len(records) == len(rows) == len(base_pack)` 与 `steps_upgraded == base_step_count`。
     - P0-3 post-pack validator：`validate_post_pack()` 复核节点集、scenario 数、每步是否升级、
       `probs→各视图`重推导、非 resource 字段不变、bin_schema_id、load 字段未丢、无 NaN；
       同时写 `resource_v2_validation.json`。
     - P0-4 manifest 补不可变 SHA：`artifact_sha256` / `base_pack_sha256` / `bin_spec_sha256` /
       `resource_head_sha256` / `producer_checkpoint_sha256` / `producer_commit_sha`，
       另加 `resource_head_training` 溯源与 `consumer_compat`。
     - **验证器抓到两个真实缺陷并已修**：① 打包器在 float32 上算视图（CVaR 跨界权重是近乎相消，
       误差被放大到 6.99e-02）→ 改为全程 float64，视图误差现为 **0.0**；
       ② r3a checkpoint 不带 bin spec → 改为**从权重推导维度契约**（`n_bins`/`hidden`/`in_dim`/
       `res_hidden` 输出维度逐项交叉断言），比信任元数据更强。
     - **实测四臂全绿**（各 9,575 节点 / 35,362 步）：`j3` 语义身份（逐记录深度相等）已断言；
       `r1b` / `r3a_u` / `r3a_f` 均 `unupgraded=0`、`nonresource_mismatch=0`、`bin_schema_mismatch=0`、
       `load_field_missing=0`、`nan_prob=0`、`canonical_view_max_abs_error=0.0`。
  2. 写 **overlay loader**（P0-2）：`load_resource_v2_overlay(base_root, resource_v2_h5)`，
     断言节点集相同、非 resource 内容与 base 一致、逐节点只替换 `future_h5`；**不要复制 H1/H3 文件**。
  3. 写两个消费臂：`sameshape_h5_condmean`（`sum_conditional_runtime_mean_plus_legacy_load_v1`）与
     `sameshape_h5_stepcvar95`（`sum_marginal_step_cvar95_plus_legacy_load_v1`）；
     先抽出 `_legacy_p95_load_cost()`，且**不得同时改 load 规则**。
  4. 写 300 集 runner（规格见 DECISIONS 2026-09-20）：135 cell × 2 + 30 extras，冻结 episode 清单 + SHA256；
     逐 episode × arm 调 `simulate_episode(...)`；O 臂直接用**已有** `sameshape_h5_truth`
     （命名 **joint-future-truth headroom**）。
  5. 九个 preflight 全 PASS 后，才跑 300 × 6。

## 2026-09-20 · resource-v2 6 臂调度 smoke（已跑完，结果与预期相反）

**跑完**：300 集配对 × 6 臂，586 秒。产物 outputs/resource_v2_scheduler_smoke_v1/
（per_episode.jsonl、contrasts.json、episode_selection.json、mechanism_conservatism.json）。
清单冻结 SHA256 c98556c8…；边际**精确命中**预注册（arrival 100/100/100、load 60×5、gpu 100×100×100、state 100×100×100）。

**主指标（mean_completion_ms，越低越好）**
O 84246.2 < **A0 85789.2** < B2 86158.1 < A1 86585.8 < B1 86631.2 < A2 86880.6

**预注册 contrasts（δ_NI = 485 ms）**

| contrast | point | CI95 | 判定 |
|---|---|---|---|
| A1−A0（预测器效应） | **+796.6** | [471.0, 1104.7] | **显著更差**，且超过 +485 |
| A2−A1（解冻附加） | +294.9 | [−18.1, 624.8] | inconclusive |
| B1−A1（均值 vs 旧风险） | +45.4 | [−290.6, 394.3] | non_inferior |
| B2−A1（CVaR vs 旧风险） | **−427.7** | [−687.7, −145.2] | non_inferior（未达 −485 的实质改进） |
| O−A1（剩余 headroom） | **−2339.6** | [−2856.3, −1855.0] | material_improvement |

**核心结论：更好的预测器（R1b）在原验收协议下 pinball −21%、校准同时改善，但在真实调度器上显著更差。**
预测质量提升**没有**转化为调度收益。

**机制（已量化，mechanism_conservatism.json）**：调度器实际消费的是**保守性**。
按各臂真实喂给调度器的每节点未来成本分数排序：B2(CVaR95) 48020 > A0(J3 p95) 37885 >
A1(R1b p95) 36560 > A2(R3a-U p95) 32702 > B1(condmean) 15079；
三个 p95 臂的调度排序 **A0 < A1 < A2 被完全预测**（分数越大越好），
Spearman(分数, 完成时间) = **−0.800**。
R1b 的 p95 **比 J3 低 6.2%**（中位比 0.938），因为 J3 的 q95 是直接分位头、
R1b 的 q95 由离散 CDF 反演取 bin 代表值 → **同一个消费规则看到系统性不同的数字**。
CVaR95 在每个节点上都是 p95 的 1.32×（100% 节点），因此 B2 比 A1 更保守、调度更好。

**不得写成**「R1b 预测更准」或「预测质量提升了调度」——两者都不成立。

## 2026-09-20 - 推送 eb88a7b + GPT 审查受阻

- **已推送 `eb88a7b`**：resource-v2 六臂 smoke 的全部实现代码、测试、结果与报告（仓库 170.0 MB / 1,035 文件）。
- **审查状态：未完成**。GPT 明确声明其检索拿不到 `QiuweiLiu/scheduler@eb88a7b`（返回的是别的同名项目），
  **拒绝在未读文件的情况下伪造行级结论**，要求上传文件或提供 `git show` 补丁。
  已构建审查包 `.scratch/review_bundle_eb88a7b.md`（129 KB，内联全部指定文件 + 模拟器 diff + 数据）；
  bridge `upload` 报 `upload_already_present` 但 CDP 复核 `attachment_nodes = 0`，
  随后宣告附件就绪的跟进消息**未成功提交**（send 再被 MCP 超时打断）。
  **=> 下一步必须先解决文件投递（重试 upload，或把审查包分段内联粘贴）。**
- **GPT 的重要预判（显式声明非代码审查结论）**：我的"调度器消费保守性"表述**大概率要改写** ——
  因为 p95 是 A0>A1>A2 而 **p90 是 A1>A0>A2**，"R1b 更不保守"解释不了 p90 变大。
  它建议改为：**调度器消费的是预测分布经 consumer function 映射后的排序信号**
  （`score_i = f(pred_i)`，而非 `d(pred_i, P_i)`）；pinball/NLL 优化分布距离，
  而调度器需要决策分数的**排序质量** -> 可引出 decision-focused / ranking / regret-aware。
  它要求用数据判别 `mechanism_conservatism.json` 到底证明的是
  **(1) scale 改变 / (2) pairwise ordering 改变 / (3) calibration 改变 / (4) 单纯数值变化**。
- **用户已决定**：不管审查结果如何，**都要先整体解冻预测器重训一次**。
  因此下一步顺序 = (1) 打通审查投递并拿到 Q1-Q6 -> (2) 按审查修代码 ->
  (3) 写整体解冻训练代码（训练目标待 GPT 的 Q4 建议）-> (4) 训练 -> (5) 回看调度器。
  注意：本次 smoke 已证明**预测指标改善不代表调度改善**，所以解冻重训的**成功判据必须与调度结果挂钩**。

## 2026-09-20 - GPT 审查 eb88a7b（已完成，13/13 文件读到）

归档：`docs/research/2026-09-20_fas_smoke_review_gpt.md`。
它独立下载了完整 1,800 行 `per_episode.jsonl` 并复算：A1−A0 +796.5545 / B2−A1 −427.6650 / O−A1 −2339.6176，与仓库一致。

### 一句话结论
本次**没有**证明"越保守越好"，而是证明：**提高概率预测质量不足以保证下游收益** ——
predictor replacement 同时改变了 scheduler 消费的 future-cost statistic 的
**尺度、当前/未来相对权重、局部候选排序**。下一轮应**保留 calibrated distribution 目标** +
**显式学习 decision-aligned scheduler score**，以**真实调度结果**为最终 gate。

### Q1 代码：5 个新 P0（全部阻塞下一轮）
1. `pack_resource_v2_artifacts.py:393-397` —— `max(0.0, nan)` 会**静默吞掉**缺失/NaN 的 canonical view。
2. `workload_v02_simulator.py:380-383` —— overlay 的 probs 检查允许 NaN/Inf/bool/负值部分漏过。
3. `workload_v02_simulator.py:310-313` —— loader 只读 manifest 里的 SHA，**没有现场重算** artifact/base SHA。
4. `resource_v2_scheduler_smoke.py:392-415` —— 第二轮 `A1-A0` 写入**覆盖**第一轮，导致正式 verdict 丢失
   （这就是 `contrasts.json` 里 A1-A0 没有 verdict 的原因）。
5. `resource_v2_scheduler_smoke.py:364-415` —— pairing 用 episode set 交集，**某臂少 episode 会静默缩样本**。
- **Q1(a) 两套 validator 确实分歧（VERIFIED）**：存在 `validate_post_pack PASS → load_resource_v2_overlay FAIL`
  的理论路径。它要求抽出公共的 `validate_resource_v2_step()` / `validate_resource_v2_record()`，两者调同一实现。
- Q1(b) `_required_runtime_field()` **实现正确**（0 ✅ / int ✅ / bool ❌ / NaN,Inf ❌ / 负 ❌），建议补 0 与 int 的测试。
- Q1(c) bootstrap 核心正确；verdict 缺 `materially_inferior` / `statistically_worse` 分支；
  `prob_le_zero` 应改名 `bootstrap_fraction_mean_le_zero`。
  **措辞红线**：可以写"A1 显著差于 A0、点估计 +796.6 超过 485、不满足 non-inferiority"；
  **不能**写"95% 置信下退化幅度超过 485 ms"（因为 CI_low = 471 < 485）。
- Q1(d) `balanced_extra_cells()` 对 v03 **验证成立**；但 `n_extra` 参数没真正传进构造（P1 API bug）。

### Q2 机制：我的"保守性"表述**过强**（INFERENCE）
- (1) score scale/distribution 改变：**VERIFIED**。R1b/J3 p95 ratio 中位 0.938，但 **p10 0.775 / p90 1.444，
  仅 61.5% 节点 <1** → 是**异质性变换**，**不是**统一降低保守性。
- (2) pairwise ordering 是否改变：**UNVERIFIED** —— `mechanism_conservatism.json` 没测
  同一 decision state 内的 Kendall τ / flip rate / top-1 disagreement / margin crossing。
- (3) calibration 改善：VERIFIED，但 scheduler **不直接读** calibration。
- (4) "越保守越好"作为一般规律：**被数据否定** —— B1 是反例（0.372× 却略优于 A1）。
- **决定性新证据**：它从 1,800 行重算三个 p95 臂的 episode 级排序，**六种 permutation 全都出现**
  （A0<A1<A2 83 / A0<A2<A1 69 / A1<A0<A2 49 / A1<A2<A0 32 / A2<A0<A1 32 / A2<A1<A0 35）。
  所以 **Spearman −0.8 只是 5 个 arm 均值上的描述性相关**，不是 episode 级机制。
- **更准确的机制**：决策 key 含 `current + future, future, ...`（`workload_v02_simulator.py:3017-3038`），
  future score 从 15k→36k→48k 改变的是 **future cost 相对 current cost 的隐式权重**。
- **它给的论文级表述**（英文/中文均已给出，可直接用）："prediction-consumption mismatch" ——
  尤其是 **被消费的决策统计量的尺度与局部排序几何**的错位，
  **不是**"更保守的预测对调度更好"。

### Q3 A1−A0 的正确定性
**可以**叫 `predictor replacement effect`（scheduler 代码/consumer ID/workload/load 规则全同，只换 artifact）；
**不能**叫 `prediction-quality effect`（替换同时改变了参数化、分位提取、marginal scale、局部排序、不确定性形状）。
- **零训练最小机制实验**（它推荐，先别动模型）：记录每个 decision 的
  `candidate_node_ids / A0_current,future,total / A1_current,future,total / chosen_A0,A1 / truth_future_cost`，
  算 within-decision `Kendall τ`、`pairwise_flip_rate`、`top1_disagreement_rate`、`margin_crossing_rate`、
  score-ratio 分布、decision disagreement → downstream delta。
- **分离 scale 与 ordering 的诊断**：单调分位映射 `g = F_J3^{-1} ∘ F_R1b`，把 R1b 的 marginal score 分布对齐到 J3
  但**保持其 ranking 不变**（g 单调）。比较 A0 / A1 / **A1-M**：
  若 A1-M ≈ A0 → **scale 是主因**；若 A1-M ≈ A1 → **局部排序几何是主因**。
  仅作机制诊断，**不是**部署校正，也**不是**"p50 乘系数"。

### Q4 整体解冻重训（它给了完整可执行方案）
- **(a) pinball/NLL 没有被否定**（VERIFIED）：被否定的只是"predictive score 更好 ⇒ scheduler 自动更好"。
  R1b 的 RuntimeQScore −21% / calibration / log-MAE 改善都是**真实进展**，完整分布仍应用 proper loss 训练。
- **(b) 不建议**把 Σq95 / ΣCVaR95 当唯一目标 —— 会让分布不再 calibrated，损失 Phase R 最有价值的成果。
- **(c) 它推荐的做法**：把"概率预测"与"调度消费量"**拆开** ——
  - 保留 `P(T_h|x)` 作为 calibrated distribution，`L_dist = NLL + 0.5·RPS + 0.25·pseudoHuber(mean) + 0.25·bandCE`（不变）
  - **新增 scheduler-facing head** `scheduler_runtime_score_ms ≥ 0`（明确不叫 q95/CVaR/mean），
    `S_j = Σ_h s^sched_{j,h} + legacy load`
  - `L_horizon = SmoothL1[log(1+Ŝ_j), log(1+S*_j)]`，权重 0.25（解决"单步不错但聚合尺度不对"）
  - `L_decision` = **decision-state 内**的 pairwise ranking（logistic），
    `w_ij` 用真实 cost gap 加权（近乎 tie 的候选权重低），权重 0.25
    —— 与之前被降级的 **global Spearman 完全不同**：这里问的是"scheduler 正在多选一时有没有把更便宜的排前面"
  - 三项先除以 **train-only** moving mean 归一化
- **数据纪律（重要）**：**不能**用这 300 集 smoke episode 构造 decision loss（已经看过结果并据此设计 loss）。
  必须来自 scheduler train/dev-training split，或按冻结的 v03 generator **新生成 train-only episodes**。**J test 继续封存。**
- **现在就该冻结的预注册成功门**：
  - Prediction integrity：`RuntimeQScore < 845.0`（不得退回 J3 以下）、`calibration error ≤ 0.0501`、`quantile crossing = 0`
  - Dev scheduler gate：consumer 冻结为 `sum_scheduler_runtime_score_v1`，primary 仍 `mean_completion_ms`，
    δ_NI = 485 ms；viability `CI_high < +485`；strong `CI_high < 0` **且** `point ≤ −485`
  - Final test gate：全部锁死 → 只留一个 winner → **打开 J test 一次** → 复用同一 metric/contrast/485 规则 → 不再据 test 调模型

### Q5 全解冻前必须修的（阻塞）
上面 5 个 P0 + 冻结 `scheduler_runtime_score_v1` + 冻结 decision-state training protocol。
**另有一条 provenance 缺陷**：`manifest_r1b.json` 记 `producer_commit_sha = 78f000a`，
但最终逻辑（float64 validator 等）在工作树里、后来才提交为 `eb88a7b` ——
典型的"**未提交代码生成 artifact → commit 后 manifest 指向旧 HEAD**"。
要求：**artifact 只允许在 clean git tree 上生成**，并保存 `producer_commit_sha` / `packer_source_sha256` / `git_dirty = false`。

### Q6 "相对 O 还有 2339.6 ms headroom" 的边界
**能写**：在 300 集开发 smoke 中，以同一 scheduler key 形状使用真实 H=5 后继执行信息时，
平均完成时间相对 A1 降低 2339.6 ms，表明距 **truth-informed future-information reference** 仍有显著空间。
**不能写**："resource predictor 还有 2339.6 ms 提升空间"（O 同时消除了 runtime 误差、topology/suffix 误差、length 误差）；
**更不能写** "O 是理论 oracle lower bound / ceiling" —— 它从 300 集重算发现
**O 在约 76% episode 优于 A1，但在约 24% episode 反而更差**，真正的下界不该如此。
统一改称 **joint-future-truth reference** / **truth-informed H5 reference**。

## 2026-09-20 - 老族验证 + 属性分布混合路线规划中

### 实测：老族（`predopt_h1/h3/h5`）**没有坏，全部能跑**
在 v03 validation 第 1 集上（`templates=640, train_stat_groups=78, artifact_nodes=9575`）：

| policy | mean_completion_ms |
|---|---|
| `predopt_h1` | 60695.2 |
| `predopt_h3` | 60090.3 |
| `predopt_h5` | 59996.6 |
| `predopt_h5_jres` | 59776.2 |
| `predopt_h5_jrt` | 65007.8 |
| `sameshape_h5_p95`（冠军） | 59550.2 |

**项目负责人的怀疑（"老族用的旧模型，代码可能不能用了"）不成立。**

### 关键发现：老族就是"属性 + 统计量"，但只用 argmax
- `predicted_future_cost()`（`workload_v02_simulator.py:1124`）→ `_step_estimate_cost()`（:1107）
  → `_hierarchical_train_row()`（:1073）三级查表 → `runtime_p50_ms + load_p50_ms`，**不用学习头**
- 被 `_predicted_candidate_key`(:2418) / `_predicted_visible_terminal_work`(:2496) /
  `choose_action`(:3152,3405,3429) / `_rl_features`(:2107) / `_cp_rho_choose`(:2185) 当**主路径**
- **缺口**：artifact 里存了 `role` / `role_probability`（`pack_j_predictor_artifacts.py:145,154-155`），
  **但没有任何消费者读概率** —— 正是 GPT 上轮列为 P1 的 "artifact observability 缺失"。
  项目负责人的"属性**分布** + 统计量混合"就是填这个缺口。

### `current` 差距的静态量化（已完成，`mechanism_conservatism.json` 之外的新证据）
- 恒等式验证：`truth == compute + load` **逐位相等**（mean 6133.2 / median 5208.3）
  → 证实 `runtime_p50_ms` 已含加载，再加 `load_p50_ms` 是重复计
- `estimate_cold / truth`：中位 **1.075**，但 **p10 0.502 / p90 2.398**，**14.3% 的节点差 >2 倍**
- 重复计的加载：中位 0（一半的组训练时没发生过加载），发生时占估计值 **p90 达 27%**
- 组中位数偏差：mean −192 ms（总体几乎无偏）
- **分模型**：`Qwen3-VL-8B` p10 = **0.081**（低估到 8%）；`Qwen2.5-VL-3B` p90 = **30.8×**（高估 30 倍）
- **驱逐成本 `current` 里完全没有**，但实际执行时长含它 —— 只能从事件日志拿
- 推论：决策间距很小，系统处在"小扰动就翻盘"的区间（这解释了 6% 的 future 变化造成 797 ms 差异）

### 项目负责人的新指示（2026-09-20）
原话："老族用的是之前的模型我怀疑代码可能不能用了。我希望重新跑一下**属性分布 + 统计量混合**，
不过**得先能够看到每一个方式的分数以及排序情况以及所需时间**。
你让网页版 gpt 规划一下 然后你来实现 写完代码之后上传到 github 让网页版 gpt 审查 审查完你修改完再跑代码"

**流程**：GPT 规划 → 我实现 → 推 GitHub → GPT 审查 → 我修改 → 才跑。

已发出规划请求 `.scratch/gpt_plan_request.md`（含需求 A 可观测性 / 需求 B 混合构造 / 需求 C 分步计划），
GPT 正在生成。**硬前置 = 可观测性（逐决策、逐候选的分数 + 排序 + 耗时）必须先做。**

### 上一轮审查的 5 个 P0 仍未修（阻塞项，见下条）
`packer:393-397` nan 吞错 / `simulator:380-383` probs 检查宽松 / `simulator:310-313` 不重算 SHA /
`runner:392-415` verdict 覆盖 / `runner:364-415` 静默缩样本。

## Goal

验证"视频 Agent 工作流的未来预测能否改进 GPU 调度"，并形成可发表的论文主线
（未来工作流不确定性 + 风险消费；不写"预测 runtime 帮助调度"）。

## Done

- Phase 0–7 完成并归档（接口验证 → 信息价值 → 形态匹配 → 消融 → horizon → 消费锦标赛 → dev/confirm 确认）。
- Phase 11–15 完成：runtime-only 正交族（C2 严格检验）、CVaR 重写、优化型参考重跑、fixed-L 对照、CP-RHO executed 变量修复 + 300 集配对重跑。
- trace 依赖检验 T1/T2/T3 完成：**公共慢化因子 / 尾部共动假设被否**，叙事转为 forecast-error-aware ranking surrogate。
- 公开仓库两轮 GPT 审阅 + 修复：CP-RHO 量纲/结构/executed、runner 指纹（含 gzip 载荷摘要 + fail-closed）、normalizer manifest、tail-shuffle 脚本公开、requirements、LICENSE/PROVENANCE/reproduce_main.sh、B≥2000 双侧统计。
- **Phase 16（本次）**：oracle 对照 confounded 的 P0 认定 + 撤回；新增同 key 形状三臂并单元验证（未跑实验）；
  已推送到公开仓库（`ae61c0a`）；GPT 审阅回复已存档（`docs/research/2026-09-17_fas_phase16_sameshape_review_gpt.md`）。
- **Phase 16 勘误 + 审阅结论**（2026-09-17）：原"三臂 priority 第 1/2/3 位"有误——greedy 预测族（含 `predopt_h5`）
  都是 priority 第 1，只有 legacy `trueopt_h5` 不同（future 第 1、且不含当前节点）；第 3 种形状只在 MPC/rollout 路径。
  混淆只有一处，撤回旧"q95 胜 oracle"不变。GPT 判定 Phase16-A 只能作 **same-key system sensitivity**，
   严格机制实验需 **Phase16-B**（runtime-only 四臂 `Pred50_R → Pred95_R → Truth|PredShape → Truth|TrueShape`）。
- **Phase 21 pilot（2026-09-18，Windows 执行）**：S_* `task_context` 修复的 16 视频 paired masked/fixed pilot 已跑完，
  **FAIL 全部四条预注册 calibration gate**（fixed p50 R=0.397 vs 门限 [0.70,1.30]；cov p50/p90/p95 = 0.396/0.844/0.916；
  p50 pinball **显著恶化** +51.6ms [27.2, 74.4]）。管线 gate 全过（join 1.0、unknown_rate 0、`prefix_hash` 975/975 一致、
  checkpoint SHA 未变）。报告 `experiments/EXP-20260911_forecast_aware_scheduling/PHASE21_TASKCTX_PILOT_REPORT.md`；
  证据副本 `experiments/.../artifacts/phase21/`；执行中修复 3 个端到端路径缺陷（见 DECISIONS 2026-09-18）。
- **Phase 21 事后代码审查（同日，用户质疑后）**：曾提出"三个 stack context 字段被 null 遮蔽 → 100% 变 UNK"的 P0，
  **经复查为误报并已撤回**（根因：探针用 `.get()`，无法区分"键不存在"与"键存在但值为 null"）。
  显式测键存在性：`task_context` 恰好只有 7 个键（975/975，与域内 `_task_context()` 布局一致），
  三个 stack 键 `key_present = False` → 编码器正常从 `stack_context` 读到真实值
  （`baseline = langgraph_react`，**在词表内**）。**Phase 21 总括结论恢复成立**：
  恢复三个 registry 内容字段不改善校准，且无同类未修管线缺口；**未做任何管线改动**（不需要）。

## Verified

- 冠军 = r95/q95（逐步骤 runtime p95 求和）：dev700 −17,229ms、frozen confirm300 −18,765ms [−21,948, −15,932] vs E2。
- C2 机制：收益来自尾部形状 + 状态对齐，不是放大尺度（Pred95R − ScaledPred50R −12,963；− ShuffledTail −7,016）。
- 压力 sweep 6/6 cell 显著为正且随负载单调（低→高约 3.4×）；质量敏感性呈剂量-反应（tail 最强，content 无可测效应）。
- `sameshape_h5_p95` 与 `predopt_h5_q95` 在同一 episode 上 summary 逐字段相等（新代码钉在已验证冠军上）。
- **runtime/load 契约审计（Phase 17）**：`runtime_ms` 是**含 load 的完整墙钟**，J 的 runtime 头直接以它为标签
  → 所有"runtime + load"的消费者都在**重复计 load**。数据证据：j_validation 546 个 `load>0` 步中
  **0 个** `runtime_ms < load_ms`，`load/runtime ∈ (0, 0.55)`；冻结 artifact 中 **7.69%** 的 GPU 未来步被
  `occ≥0.5` 门控、被高估约 **40%**（load p95/runtime p95 p50=0.402），这也解释了 rt95 与 q95 只差 +11ms。
  受影响：`_q95_step_cost`（冠军）、`_mix/_risk/_mix95/_split95/_jres/_jrt/_step_estimate_cost` 与所有 key 的 `current` 项；
   干净的是 `_runtime_only_step_cost`（`predopt_h5_r95/r50` 族）。**未修、未重跑**（需批准）。
- **Phase 21 负结果（预注册，2026-09-18，差分成立）**：恢复三个**内容**字段（domain / official_task_type / sub_category）后
  S_* 仍 p50 R=0.397、cov 0.396/0.844/0.916，p50 pinball 显著更差 +51.6ms[27.2,74.4] → **只证明"内容字段无助校准"**。
  ~~排除"缺元数据"解释~~ **已撤回**（见下条）。与 Phase 10（content 无可测效应）一致，但不足以支撑总括结论。
- **Phase 21 事后审查（2026-09-18，最终结论）**：context 块**无** null 遮蔽缺陷。按编码器口径复算 9 个字段，
  只剩 `temporal_scope`（registry 无源数据，数据边界）与 `model_stack_id` 的 stack_a 命名变体
  （数据集命名漂移，域内为近似值 `stack_a_qwen3_vl8b`）以 UNK 进入编码器；
  `baseline`（`langgraph_react`）与 stack_b 变体（486 个锚点）**in_vocab**；
  `planner_model_id` 域内词表只有 `unknown`（该字段从来不含信息）；
  `required_modalities` 的字符串形态与逐字符迭代**域内也存在**（J train 有 3439 行是字符串）。

## Rejected（不得重跑，除非有新证据）

- 朴素 H10 重训；场景采样+CVaR / 共单调 / 自适应风险 / 机会约束 / 生存 / 缓存 / 内容 / 期望成本 / load 尾部全部未超过 r95。
- **trace 公共慢化因子 / 尾部共动机制**（T1 未约束 MOM 分量 −0.139 CI 全负、截断 ICC 0；T3 lift<1 但双侧 p=0.54/0.20/0.25 不显著）→ 不得写"证明负相关"。
- 优化型参考（CP-SAT/pred-MPC）作"优化不是瓶颈"的证据：修好后仍落后 r95 约 19s，且 15.4% 决策撞 0.25s 上限、平均 gap 10.3% → 只能 exploratory。
- **"q95 胜过 oracle 臂"**：key 形状未对齐（Phase 16），已从门禁撤回。
- **"S_* 域外失准源于 task_context 缺失"**：Phase 21 pilot 已用干净单变量对照否掉（2026-09-18）。
  **不得全量重生成 S_* 预测包**（会作废全部调度结果，且预期无收益）。
- **"stack context 被 null 遮蔽"的 P0（我提出）**：**误报，已撤回** —— `dict.get()` 无法区分"键缺失"与"值为 null"。
  不得重提；审计代码必须显式测键存在性（`field in mapping`）。

## Open

- Truth-SameConsumer 三臂**已实现未跑**（门禁：dev700 配对，判据见 PHASE16）。
- **Phase16-B 未实现**：runtime-only 四臂 + 若 v3.1 evaluated continuation 内 outdegree ≤ 1 则做
  `TruthRuntime | PredictedShape`（ordinal successor 对齐，需 path-invariant 断言）。
- **Phase 17 修法未选**，但**用户已定约束：不动预测器**（"runtime 头改 compute-only 标签"方案排除），
  只能在消费端修（全消费者 runtime-only / 只最小修冠军）。在选定并重跑 dev700 前，
  "q95 = runtime 尾部"这条描述暂停对外使用（"load 维度无价值"不受影响，已复核）。
- H10-lite 未实现；cache-aware 缺 residency/reuse 数据；真实 2-GPU replay 未做（论文最大短板，阻塞于硬件）。
- 既有测试失败（与本次改动无关）：`tests.test_workload_v02_simulator.AdmissionTests.test_round_robin_joint_action_keeps_oldest_ready_node`。

## Active

- 无运行中的实验（CPU/GPU 空闲）。
- 控制面 `STATE.md` 顶部的 09-15 段落仍是旧快照；09-17 状态见新增段落与 `EXPERIMENT_GATE.json`。

## Next

- **Phase 18（进行中）**：聚合乐观性诊断 —— GPT 审阅判定"设计可用、代码 6 个 P0 未过"（`docs/research/2026-09-17_fas_phase18_code_review_gpt.md`）；
  6 个 P0 已全部修完（video_id 取模板字段 + 断言 160 视频、unjoined fail-closed、**保留预测零步锚点**、
  分位缺失 fail-closed、bootstrap 成对过滤、linear_k 改用全局 sequence_index + 正确的分支/单链判定），
  6 个 P0 + 复审新增的 2 个 P0（`_stat_block` KeyError、zero-step 混入主人群）与 1 个 P1 全部修完，24/24 单测通过
  （含"真实 anchor_metrics → summarize"集成测试）；干跑找回被静默丢掉的 **1,272** 个零步锚点。
  **首跑已完成但结果判定为 NOT REPORTABLE**：R7 模板的"未来"是**分层事件 DAG**（每个 agent step 有 2–4 个事件），
  严格单链子集恰好只选中"下一个事件就是终止 `:run:*` 事件"的锚点（197 个/129 视频，均只有 1 个预测步与 1 个后继），
  于是 r≈0.008 而 branching 人群 r≈2.8–3.9 —— 是粒度错配而非校准结论。
  真实尺度：R7 叶节点 runtime 中位数 api_call 4,658ms / action 0.1ms / **run 53,778ms**；
  预测 p50 中位 324ms、p95 中位 7,673ms。
- **Phase 18 口径已按数据重定**（三条规则）：步骤耗时 = 该步非容器事件之和；`event_type='run'` 的
  **整条流墙钟容器**排除并单独计数；时间戳只作事件点、耗时一律用 `runtime_ms`。
  脚本内置校验通过：640 个模板的"非容器之和 / 容器墙钟"**中位 0.982**（p10 0.970）；干跑 join 8,295（+640 容器 +640 根，均计数排除）。
- **新 P0（已确认未处理）**：该容器事件被**模拟器当作普通 GPU 节点调度执行** —— 640/8,935 个节点却占总 runtime 的
  **50.2%**（GPU 的 50.5%）；episode `validation_000000` 记录 `nodes=224 / completed_nodes=224`，其中 16 个正是容器。
  即**仿真 workload ≈ 真实执行时间的 2 倍**，且每个 job 末尾多出一个 ~54 秒的调度对象。
  影响：绝对 makespan/completion/utilization/eviction 全部偏高；相对排序是否改变**尚未检验**。
  门禁登记 `workload_container_double_count_20260917`。
- **Phase 18 已完成（可报告）**：主口径 = 事件级 next-K，4,052 锚点 / 160 视频。
  p50 求和平均只到真值 **0.456**（93% 锚点低估）；p95 求和平均是真值 **4.9 倍**（聚合 2.8 倍）
  → **"p95≈聚合纠偏"的替代解释被否**，它是保守上界而非校准期望。
  报告 `experiments/EXP-20260911_forecast_aware_scheduling/PHASE18_AGGREGATE_OPTIMISM_REPORT.md`，
  数据 `artifacts/phase18_aggregate_optimism_v2/`（v1 作废仅存证）。
- **GPT 对容器 P0 的裁决**：确认是 workload construction error，不是参数偏差；**不建议只做单臂 included/excluded**；
  要求新建 `r7_workload_v03_no_run_container`（legacy v02 不许覆盖），先跑最小多臂矩阵（E0/E2/r50/r95 + FCFS 或 SJF），
  看 Δ(r95−E2)、Δ(r95−E0) 与**排序**是否保持，再决定扩到哪些主实验；论文按 workload-construction correction 简短披露。
  受影响最大的既有结论：**contention 相关**（r95 gain 随 contention 增强可能被 54 秒巨型节点放大）、
  其次 tail-risk 机制（tail-shuffle / scaled-p50 需复查）、再次 load/cache 结论（真实 load 占比被稀释）。
  相对稳的：预测器 artifact 分位结构、tail-shuffle 的纯预测侧现象、T1–T3 trace 依赖分析。
- **Phase 19 执行完成（判定 = 情况 B，必须重审机制链）**：
  - v03 workload 已建：`results/processed/r7_workload_v03_no_run_container/`（模板 8,295 节点 / 0 容器；
    episodes 重算；**legacy v02 未动**）。重建可信性锚点：用 v02 模板重算 21,000 条 v02 episode **逐字段 0 不匹配**。
    episodes 窗口按新工作量重算 → **实测负载仍 0.5000**（无"顺带减压"混淆）。
  - 冒烟：100 集配对 × 5 策略 × {v02,v03}，`outputs/phase19_container_fix_smoke_v03/`。
  - **排序改变**：legacy `r95 < E2 < r50 < fcfs≈E0` → v03 `r95 < r50 < E0 < E2 < fcfs`。
  - **E2−E0 从 −10.6s 变 +0.26s（CI 跨 0）** → "未来信息本身有价值"不成立；
    **r95−E0 从 −15.1s 缩到 −2.9s**（约 81% 收益来自容器构造）；
    r95−E2 仍显著（−4.5s → −3.1s，CI 排除 0）→ **"尾部消费优于点消费"存活**；
    fcfs−E0 从 −0.3s 变 **+11.8s**。
  - 报告：`experiments/EXP-20260911_forecast_aware_scheduling/PHASE19_CONTAINER_FIX_SMOKE_REPORT.md`。
- **GPT 已审核 Phase 19**（判定与措辞已记入 PHASE19 报告 + 门禁）：
  - 读法成立，但**措辞要收敛**：不写 "81% of the gain was caused by the container artifact"（因果），
    改写 "removing duplicated container nodes **reduced the observed improvement** by ~81%, indicating the previous
    construction substantially **amplified** the scheduling benefit"。
  - **保留**：q95>p50；aggregation matters；conservative aggregation 补偿系统性预测乐观、改善排序鲁棒性。
  - **删除**："future information itself improves scheduling"；"contention amplified benefit"；"risk 来自 execution correlation"。
  - **Phase 18 改限定**：应称 **forecast optimism calibration analysis**（不是 comonotonic risk validation）；
    支持 p50 偏乐观 / p95 保守，**不**支持执行时尾部相依。
  - **v03 必须声明**：保留的是归一化到达模式与 offered load，不是绝对时间戳。
  - **迁移计划**：20-A 基线矩阵（v03, validation 1000, fcfs/myopic/E2/r50/r95）→ 20-B 6-cell sweep（必须）→
    20-C Phase16 四臂（必须，优先级低）；CP-SAT/MPC/BC 留 P2。
- **Phase 20-A 已完成**（v03 全 1000 集 × 5 臂，配对，B=2000；`outputs/phase20a_matrix_v03/`）：
  - **复现性**：legacy v02 的 1000 集结果与归档 Phase 11 leaderboard **逐位相同**（r95 180,407.2 / E2 187,614.9）。
  - 排序：legacy `r95 < E2 < fcfs < r50 < myopic`；v03 `r95 < E2 < r50 < myopic < fcfs`。
  - r95−E2：−7,208 → **−4,136 [−4,488, −3,805]**（存活）；
    r95−E0：−34,709 → **−4,929**（缩 86%）；
    **E2−E0：−27,501 → −793 [−1,062, −507]**（缩 **97%**，仍显著——**不是消失**）；
    fcfs−E0：−1,835 → **+27,977**。
  - **对 100 集冒烟的修正**：冒烟说"E2 优势消失（CI 跨 0）"，全 1000 集显示仍显著但只剩 0.79 s。
    正确表述 = **缩小约 97%**，不是"消失"。
  - 报告：`experiments/EXP-20260911_forecast_aware_scheduling/PHASE20A_BASELINE_MATRIX_V03_REPORT.md`。
- **GPT 裁决 Phase 20-A：P0 PASS**，**v03 成为新主 workload**；论文主线改为
  "Forecast information alone provides limited gains; the key challenge is uncertainty-aware consumption
  of imperfect future estimates"。
  - 措辞纪律：E2 写 "statistically detectable but **practically modest**"（**不写** negligible）；
    FCFS 写 "substantially altered queue dynamics"（**不写**"由 54 秒节点导致"）。
  - **P1 设计（GPT）**：消费臂最小集 = p50 / p90 / p95 / scaled-p50 / tail-shuffle / oracle-truth；
    **暂不做 CVaR**；新增指标 = per-anchor forecast bias、**aggregation gap**（ΣQ0.95(x_i) vs Q0.95(Σx_i)）、
    决策级（ranking agreement / regret / 被选任务的真实剩余成本）；划分改名为 **v03-dev700 / v03-confirm300**。
  - **P2 设计（GPT）**：旧 6-cell 不再沿用（contention 已被污染）；主轴改为 **pressure {0.25,0.5,0.75,1.0} × horizon {1,3,5}**，
    deadline sweep 降为 sensitivity。
  - **两个残余风险已当场关闭**：artifact 对 v03 覆盖率 **8,295/8,295 = 100%**（无静默跳过）；
    deadline/slack v02→v03：deadline 均值 1,887,546 → 927,097，slack 均值 231,318 → 148,378，两侧负 slack 率均为 0.000。
- **P1 已完成（Phase 18 新指标 + Phase 20-B 消费矩阵，v03-dev700）**：
  - 矩阵 9 臂排名：**oracle-truth 88,644 < p90 89,822 ≈ p95 89,946 < E2 94,125 ≈ tail-shuffle 94,200 ≈ p50 94,614 ≈ E0 94,800 < scaled-p50 105,090 < FCFS 121,987**。
  - 相对 E0：p90 −4,978 / p95 −4,853（显著）；**oracle-truth −6,156**；**p50 −185（不显著）**；
    scaled-p50 **+10,291（显著更差）**。
  - 相对 p95：**oracle-truth 仅 −1,303**（同形状上界只高 1.3 s）；p90 −125（**不可分**）；
    tail-shuffle **+4,253**（打乱对齐 → 收益全失）；scaled-p50 +15,144。
  - **新机制表述（可写）**：**预测本身收益有限；关键是"与未来步对齐的保守尾部聚合"**，
    它接近同形状 oracle 上界；**不是**"有未来信息"（E2 仅 −0.68 s）、**不是**"尺度放大"、**不是**"难度信号"。
  - Phase 18 新增指标：**Σp90 ≈ 真实工作量的 95 分位（比值 1.010）**，Σp95 = 1.41×；
    **尾部间隙不标记难度（Spearman −0.088）**。
  - 报告：`PHASE20B_CONSUMER_MATRIX_V03_REPORT.md`、`artifacts/phase18_aggregate_optimism_v3/`。
- **新 P0（已定位、待执行）：预测输入特征缺口 —— 用户决定"修管线，不动模型"**。
  - 现象：S_* 锚点的 `task_context` **全部** 为 "unknown"（9,575/9,575），域内训练/验收数据只有约 22% 缺失。
  - 根因：R7 的 trace 与 template **根本没有**任务元数据（`task_context` 字段不存在；注册表字段不在 trace 里），
    `scripts/build_sstar_predictor_anchors.py` 因此产出全 unknown。
  - **可修**：`data/manifests/videomme_600_source_v2.jsonl` 里有 domain / official_task_type / sub_category /
    duration / question_id / answer / options / question，且 **160/160 命中 R7 视频**；
    字段构造的参考实现 = `scripts/build_p9d_topology_dataset.py::_task_context(sample)`。
  - **为什么原有检查没发现**：`coverage_report.json` 显示这些字段 oov_rate = **0.0000** —— 因为 "unknown" 本身
    就在 J 词表里；该检查测的是"词表外率"，不是"值为 unknown"。（temporal_scope 的 oov 是 1.0，被忽略。）
  - 域内参考：J3:seed11 在自己验证集上分位覆盖率 0.556/0.917/0.968（目标 0.50/0.90/0.95）、pinball 845ms、长度 MAE 0.047
    → **模型在特征完整时是校准良好的**；域外则 p50 只有实测表的 0.38 倍。
  - **代价**：重生成 S_* 预测包（同一个冻结 checkpoint、只换输入）→ **所有调度结果必须重跑**；
    且必须在训练机（Windows/torch）上执行，本机不行。
- **GPT 已给 Phase 21 方案**：只补 `domain + official_task_type + sub_category`（其余字段保持冻结 fallback；
  **禁止**把 question/answer/options/duration 塞进模型）；先做 **16 视频 paired masked/fixed pilot**，
  pilot 的 calibration gate 通过后才全量重生成；不需要重做 v03 workload。
- **代码已就绪（本机，11/11 单测通过）**：
  `scripts/preprocess/sstar_task_context.py`（纯 join 逻辑 + 确定性 pilot 选择）、
  `scripts/build_sstar_predictor_anchors.py`（+`--task-registry` / `--video-allowlist` / `--mask-registry-task-context`，
  coverage gate 增加 missing/unknown/OOV(non-missing) 与 fail-closed 规则）、
  `scripts/analysis/analysis_phase21_taskctx_calibration.py`（逐步校准审计 + paired Δpinball）、
  `scripts/preprocess/phase21_pilot.ps1`（Windows 执行，含 checkpoint SHA 校验）、
  pilot 清单 `experiments/.../artifacts/phase21/pilot_videos.txt`（16 视频，sha256 预注册）。
- **修复前基线（同 pilot，3,226 slot 对）**：p50 R=0.415 / cov=0.392；p90 cov=0.829；p95 cov=0.906
  → **四条 calibration gate 全部 FAIL**，作为 before 参照。
- **阻塞**：无（Phase 21 pilot 已于 2026-09-18 在 Windows 机上执行完毕）。
- **Phase 21 已结案（负结果）**：pilot FAIL → 不重生成、不重跑；结论文本已写入
  `PHASE21_TASKCTX_PILOT_REPORT.md` 与门禁 `predictor_input_feature_gap_20260917`。
  关键数据事实（仍有效，供后续真值定义使用）：预测 artifact 的 anchor 来自 **R7 job templates**
  （640/640 run 重叠，7,663 anchors 可 join，640 个 `:run:1` 工作流根不可 join）；
  **J 数据集与 R7 workload 完全不相交**（0/640 run、0/160 video）→ 真值必须取自 R7 templates。
  模板是 **DAG**（outdegree 0–5，均值 ≈2.08）而预测是单条线性链 → 真值需并列报告三种定义（linear_k / layers_h / all）。

1. **Phase 21-B 已完成并封存（GPT 审核通过）**：
   - **B1**（`planner_model_id` → 训练期字面量 `"unknown"`，975/975，同 checkpoint）：
     p50 pinball **−24.70ms [−38.02, −9.87]（显著）**，但只有 **2.16%**，**未过 10% 门槛**；
     相对原始 masked 是 +26.88 [−3.76, 55.51]（不显著）→ **方向正确、可检出、工程上很小**，不得升级为"主因"。
   - **B2**（stack canonicalization）**拒绝执行且被审核确认正确**：`stack_a_qwen3_vl8b`（phase-3 配置，**无检测器**、
     384 tokens、answer_model=finish_argument）与 R7 的 `stack_a_qwen3_vl8b_yolo11x`（**yolo11x batch 8**、256 tokens、
     answer_model=Qwen3-VL-8B）是**不同执行配置**，改名会注入错误身份。
     可写："The exact R7 stack_a configuration is out of support for the frozen categorical stack vocabulary and cannot be
     faithfully repaired by input relabeling alone."
   - **零成本分层分解（GPT 指定的最高优先项）**：预测的 p50 **scale 几乎不随执行配置变化**
     （各格 pred50 中位都落在 193–764ms），而真值中位在 345–5978ms 变化。
     **`langgraph_react` 在两个栈上都最差**（ratio 0.069 / 0.114），**包括 in-vocab 的 stack_b**。
     → **baseline（langgraph_react vs star）比"栈是否在词表内"影响更大**；"未见栈是主因"的读法**不完整**。
     p90/p95 头仍相当保守（stack_b 0.918/0.971；stack_a 0.819/0.878），**失效的是 p50 scale 头**。
   - **审核结论**：只能做 **association**，不能做因果归因；必须紧跟一句限定
     "Because the two stacks also differ in model composition and runtime distribution, this stratification does not
     isolate the causal effect of vocabulary support."
   - **审核指定的下一步**：**停止 feature-patching 线** → 接受冻结预测器的栈相关 OOD 限制 → 回到已批准的 **P2 sweep**。
     审核还**排除**了"把 model_stack_id 也设成训练期 unknown"的实验（该字段训练词表里**没有**字面量 `unknown`）。
   - **写入论文前的 3 个必做 P1**：① 128 个 missing-template anchor 分类并 fail-closed（root=64/container=64/unexpected=0）；
     ② `paired_delta` 补严格配对一致性断言；③ 输出里写入 resolved path + SHA + checkpoint + min_steps。
   - 归档：`docs/research/2026-09-18_fas_phase21b_review_gpt.md`；产物 `outputs/phase21b_stackctx/`。
2. **独立审查（Phase 21）已完成（GPT，Sol + High）**：裁决 = **核心负结果 PASS + 误报撤回 PASS**，
   但**驳回**我"不存在同类未修输入缺口"与"误差主要来自结构错配/分布偏移"两句（已撤回）。
   归档 `docs/research/2026-09-18_fas_phase21_review_gpt.md`。
   **它要求的两个封存前检查，我已跑完并全部通过**：
   - 真值顺序等价性：按纯 `sequence_index` 排序 vs 按 `(source_step_id, sequence_index)` 排序，
     **640/640 全 R7 模板 + 64/64 pilot 模板完全一致（0 不一致）** → Q1 转 VERIFIED，pilot 数字无需重跑。
   - 两个 artifact manifest 均为 `min_steps=5` / `horizon=5` / checkpoint `0ee8ded4…` → P0-2 关闭。
2. **GPT 指出的新关键点（已用现有输出做零成本分层验证）**：`__UNK__`（特殊 OOV 桶）**不等于**训练时的字面量
   `"unknown"` token —— 模型学的是 `"unknown"` 的 embedding，S_* 喂的是几乎没被训练过的 `__UNK__` embedding。
   按 `model_stack_id` 分层（现有 pilot 输出，零 GPU）：
   - **stack_b（in-vocab）**：fixed R50 **0.5820**、cov90 **0.9012**、cov95 **0.9699**（接近域内水平）
   - **stack_a（OOV）**：fixed R50 **0.1751**、cov90 0.7878、cov95 0.8620（差得多）
   → 命中 GPT 的判据"stack_b 正常、stack_a 极差 → 先查 stack canonicalization"。
   **但有混淆**：两个 stack 本身就是不同执行栈、runtime 分布不同，分层本身不证明因果，必须做配对反事实。
3. **下一步（GPT 指定，都是几十秒级、同 16 视频同 checkpoint）**：
   - **21-B1**：把 `planner_model_id` 设为字面量 `"unknown"`（恢复训练时的 feature-availability contract，
     不是猜类别）→ 与 masked/fixed 配对比较。
   - **21-B2**：把 `stack_a_qwen3_vl8b_yolo11x` 映射到词表内的 `stack_a_qwen3_vl8b`。
     **前置条件：先确认两者的 stack identity 定义相同**（不许仅凭名字相似就映射）。
   - 成功判据：mean runtime quantile pinball 改善 ≥10% 且 video-cluster CI upper < 0，
     同时 p50 coverage / R50 朝目标方向移动。
   - 若两者都失败，才可把主因归到 resource-distribution shift / 属性→资源头传播误差 / target alignment / 执行栈差异。
4. **GPT 给的 P1 代码加固（未做）**：`collect()` 对 128 个 missing-template anchor 静默丢弃
   （GPT 推断是每 run 预期的 root + container 各一个，128/64=2，要求分类并 assert unexpected=0）；
   `paired_delta()` 缺 pair-set/truth/video 相等性断言；builder 的 `required_modalities` 审计不是 encoder-equivalent
   （应逐元素编码，且 `is_missing([])` 语义不对）；`resolve_artifacts` 需记录 resolved path + sha256
   并 assert fixed≠masked；若要把 +51.6ms 恶化写进正文，补 16-video exact sign-flip test。
5. **论文措辞（GPT 给了可直接用的三句 + 禁用清单）**：可写"恢复这三个 registry 字段不改善校准 + 那三句"；
   **不可写**"metadata missing is not the cause"、"no remaining input pipeline mismatch exists"、
   "planner_model_id 无信息损失"、"误差主要来自结构错配/分布偏移"、"Phase 21 支持 C2"。
   术语：`R` 叫 aggregate scale ratio，coverage 才叫 quantile calibration。
2. **需用户选定方向（A/B/C）**：A 转已批准的 P2（v03 pressure × horizon）+ v03-confirm300；
   B 转 Phase 17 runtime/load 契约修复（会重定义冠军，需预注册 δ）；C H10-lite / cache-aware / 真实 replay。
3. 补真实 replay 的硬件确认（本地 1×3060 6GB 只能做节点级 fidelity）。
4. 剩余仓库项：`trueopt_h5`/`predopt_h5` 命名与 legacy key shape 在论文中的标注；round-robin 既有测试失败定位。

## Environment / Recovery

- **执行机（Windows，正式实验）**：解释器 `D:\anaconda\envs\scheduler\python.exe`；`PYTHONPATH=src`；RTX 3060 Laptop。
- **本机（macOS，控制面/代码/轻量验证）**：项目在 `/Volumes/Lenovo/scheduler`；
  可用 `PYTHONPATH=src /opt/miniconda3/bin/python`（3.13，无 numpy/torch）跑**纯 stdlib 的模拟器与单测**；
  **不下载** Windows 的 GPU/模型实验。
- **公开镜像**：`/Volumes/Lenovo/scheduler_public_repo` → `github.com/QiuweiLiu/scheduler`；staging 脚本 `.scratch/stage_public_repo_v2.ps1`（源自 `F:\scheduler`）。
  注意：该脚本复制后会在工作区留下大量 **CRLF/BOM-only** 差异（34 个文件，改变行数对称），提交前需只 stage 目标文件。
- **ChatGPT 桥接（本机已验证）**：
  `"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --remote-debugging-port=9222 --user-data-dir="$HOME/Library/Application Support/Google/Chrome-Automation" https://chatgpt.com/`
  —— 必须用 `Chrome-Automation` profile（已登录；`~/.chrome-chatgpt-bridge` 是空 profile，会落在登录页）；
  该 profile 与 `chatgpt-automation` MCP **共用**，两者不可同时运行。
  本机**直连** chatgpt.com 可用，**不需要** `--proxy-server=7897`（那是 Windows 的坑）。
  绑定会话：`https://chatgpt.com/c/6a9822da-e278-83e9-9c1a-675923acda0e`（标题「架构设计评估」）。


## Phase 21-C（2026-09-18，用户质疑"预测器没道理这么差"）

**已推仓库 `ed32351`**（含 v03 workload：templates + validation episodes；Phase 21/21-B 脚本、审计、报告、测试；.gitignore）。
GPT 审查 + 本机 raw-trace 审计已完成，**三个候选解释全部被排除或量级不足**：

- **nested 重复计时**（GPT 的 P0，`videotool_phase1.py` 行号已核）：包含关系真实
  （169/169 summarizer 内嵌 `generalist.generate`，内/外 runtime 比中位 0.758、p90 0.992），
  但只占总 runtime **0.48%** → R50 只会从 0.436 → 0.438。**不足以解释。**
- **`answer` 残留容器**：`answer` 的 `runtime_ms` 与 `run` 容器**逐 run 完全相等（640/640，比值恒 1.000）**，
  是同一节点的双标签 → v03 已正确移除，**无残留 50% 膨胀**。
- **尺度域偏移**：J 训练槽位中位 **2223.0 ms** vs R7 v03 非容器节点中位 **2213.0 ms**；
  J `planner` 中位 4992 ms vs R7 `api_call` 中位 4658 ms → **尺度相同**。

**留下的矛盾**：同 checkpoint、同尺度，J 域内 p50 覆盖 **0.556**，R7 S_* 锚点只有 **0.39–0.43**（R50 0.40–0.44）。
⇒ 差异**不在数据分布、也不在模型**，只能在 **S_* 推理路径**（锚点/history/current_node 的构造）。

结构性差异（尚未量化）：J 行来自 **chain**（`build_p9d_topology_dataset`：去 `run_control`、
把 nested `generalist.generate` 合并进 summarizer、处理 retry），而 S_* 锚点来自**原始事件表**
（每个原始事件一个锚点，`history = events[:index]`）→ S_* 的 history/`current_node` 可能包含
J chain 从未出现过的 `event_type='run'` 容器与 nested generate。

- 报告：`experiments/EXP-20260911_forecast_aware_scheduling/PHASE21C_PREDICTOR_SUSPICION_AUDIT.md`
- 审查：`docs/research/2026-09-18_fas_predictor_suspicion_review_gpt.md`
- **下一步（仍无需 GPU）**：对同一 run 比对 chain 口径 vs raw-event 口径的 model_input
  （history token 序列、`current_node`、`stack_context`、编码后的 context id），
  再用 **chain 口径锚点**重跑 pilot，看 p50 覆盖是否回到 0.55 附近。
- GPT 另纠正两处我的误判：① artifact 没序列化 `node_type` ≠ runtime head 不知道预测的 node_type
  （J3 内部吃的是 `softmax(node_type/role/action_family/model_class)` 的完整分布，属 artifact observability 缺失，P1）；
  ② `load-duration p50 > runtime p50`（82.5%）不是 packer bug，而是"条件 load 时长 vs 无条件 runtime 分位"的合法差异
  （runtime 含 load；load head 只在 load 发生的槽位上训练），但多头非 joint-coherent，manifest 需明确标注。


## 指标口径审计（2026-09-18，用户质疑"预测器没道理这么差"）

**结论：前提撤回——预测器没有严重失准。**

- 用 packer 在 **J 验证集**（2,029 行 → 7,195 槽位）跑冻结 J3:seed11，再用 **R7 那套统计代码**打分，
  **逐位复现验收报告全部数字**：覆盖 `0.5555/0.9166/0.9680`（验收 0.556/0.917/0.968）、
  RuntimeQScore `845.0`（验收 845.0，= (1194.3+797.9+542.9)/3）⇒ **同一把尺子**。
- **域内 `Σp50/Σ真值 = 0.6322`，不是 1.0**；验收报告从未公布这个比值。
- **主指标 RuntimeQScore：R7 = 782.1，优于域内 845.0**（越低越好）。
- **0.63 是结构性的**：域内每槽位真值 中位 2209 ms / 均值 3877 ms（1.76×）→ 完美模型也只能 ≈0.57；
  实测 0.632 正在该量级。**"逐步中位数之和 ≠ 总和的中位数"**。
- **90%+ 是分类头**（7 字段 0.9013 / next-role 0.9444 / next-family 0.8396），与资源头（覆盖率/pinball）无关。
- 资源头三个覆盖率**都略高于目标**（0.5555/0.9166/0.9680 vs 0.50/0.90/0.95）→ 偏保守，方向安全。
- **唯一真实残余**：p50 覆盖率 R7 0.427 vs 域内 0.556（对应已知 OOV 字段与栈分层）。

**连带修正**：
1. Phase 18 的"Σp50 = 0.456× ⇒ 预测乐观"必须**改基准为域内 0.632**（乐观仍在，但远小于原描述）。
2. **C2 需重新论证**：若 Σp50 天然低估总量，q95 式聚合可能是**数学上正确的聚合**而非"补偿预测误差"（假设，未证实）。
3. **禁止**对外写"预测器低估约 60%"。

产物 `outputs/a2a_jval_full5/`；门禁 `metric_audit_20260918`；待与 GPT 讨论 C2 如何重述。
