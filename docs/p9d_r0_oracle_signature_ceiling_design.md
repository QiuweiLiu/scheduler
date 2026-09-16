# P9d R0：oracle-signature 经验资源上界实验设计（frozen v3）

状态：**frozen（2026-09-11）**。依据网页评审 v1→v3 修订与最终裁决 **“R0 DESIGN v3 ACCEPT（可冻结）”**。
关联：契约 `docs/p9d_topology_label_contract_v3.md`；评审记录 `docs/research/2026-09-11_p9d_r0_design_review{,_v2,_v3,_final}.md`；数据集登记 `data/manifests/topology_predictor_p9d_v3.json`。

## 0. 定位与命名

- 名称：**R0：oracle-signature empirical resource ceiling（冻结接口与估计器族下的经验上界）**。
- 形式化：`R₀: (Z_oracle, workload_scale, c_pre) → (runtime, load, memory)`。
- 只回答：当未来节点“是什么已知”（oracle 签名）时，当前接口能把单节点资源估到多准；作为 J 系列（J0/J1/J2/J3）的 go/no-go gate。
- 明确不是理论界：结果受估计器族、当前 Z、当前 workload descriptors、当前测量质量限制。
- 非目标：不预测结构/签名、不接调度器、不产出可部署模型、不算聚合 future cost、不触碰 `S_*/T_final`。

## 1. 数据与节点本体

- 数据集：`topology_predictor_p9d_v3`（契约 v3.1，经验证串行控制流）。
- 目标节点：15,481 个 resource-applicable 计算节点 —— planner 7,211、tool 7,078、post-loop generation 1,192；嵌套 `generalist.generate` 482 个已合并进 Summarizer，**不是独立 target**；terminal `answer`（1,360）为 non-resource 标记，不进入目标。
- 划分：P_dev/train 拟合 → P_dev/validation 选型 → P_dev/test 诊断 → P_holdout_diag 冻结后一次性验收；视频级划分（同一视频所有 run 同侧）。
- 统计单位：video-cluster bootstrap（video 为重采样单位）；报告 node-pooled、per-run macro、per-video macro；不建 run-level split，run 级只作宏观敏感性。

## 2. 目标与测量契约

- 目标：`runtime_ms`（主 gate）、`load_ms`（次，hurdle）、`peak_memory_inclusive_mb = max(peak_reserved_mb, peak_allocated_mb)`（次，instrumented 子集）。
- 有效行：`status=success` 且目标为正有限值（load 见下）；缺失不得填 0。
- **load 语义（冻结）**：采集器为 `load_ms = model_load_ms if request_index == 1 else 0.0`（帧加载路径同理）→ `load_ms=0` 是**合法 no-load 观测**；缺失=unavailable 并报告覆盖率。评价按 hurdle：`occurrence = 1[load_ms>0]`（全部非缺失行）+ `duration = load_ms | load_ms>0`（正样本）+ 可选 `expected_load = P(occ)·E[dur|occ]`。
- **load 计数（冻结，v3.1 本体）**：planner 7,211（missing 67 / 0: 5,960 / >0: 1,184）；tool 7,078（0 / 6,742 / 336）；post-loop generation 1,192（9 / 530 / 653）；合计 15,481（missing 76 / zero 13,232 / positive 2,173）；互斥三项之和 == 类总数。
- **residency**：`residency_state` 定义保留（`{resident, nonresident, unknown}`，源自执行前 `model_resident_before`），但当前池该字段 100% null → 不可用于特征（记录为数据限制）。C4 上下文使用 `prefix_model_reuse`（执行前可见：该节点 model_class 是否已在同一 run 困难可见前缀出现），明确标注 **pre-execution proxy**，由 C3→C4 消融检验。
- **memory 语义（冻结）**：worker 每请求读取 `torch.cuda.max_memory_allocated()/max_memory_reserved()` 后立即 `torch.cuda.reset_peak_memory_stats()` → 按请求窗口的节点级峰值（含常驻权重，inclusive）；memory 结论仅限 instrumented 子集，并报告 `coverage(memory | exec_class)` 与 `coverage(memory | model_class)`。
- stall 规则（仅由 train 构造并冻结）：`runtime_ms > 10 × median_train(exec_class)` 或 `> 300,000 ms`（先触发者）。主表 **R0-Inclusive** 保留全部合法行；**R0-Normal** 副表排除 stall 行（同一 inclusive 训练模型的子集诊断，不另训模型）；stall 计数与分布单独报告。
- tail 阈值：`T_tail = q90(runtime | train)`，各 split 统一使用；tail 子集误差单独报告。

## 3. 特征与角色白名单

- `Z`：`exec_class`（node_type + raw_action）、`role`、`model_class`（exact `model_id`；coarse 版本另设）、`planner_mode`、`merged_nested_call`、`nested_model_class`、`is_retry`。
- `workload_scale`：`clip_len`、`query_char_len`、`nested_api_call_count`（+缺失标志）。
- `c_pre`：`stack/baseline`、`prefix_model_reuse`。
- 角色白名单（Stage 0 逐列标注 `pre_execution` / `oracle_label` / `target_only`）：特征只用前两类；`target_only` 永不进特征。禁止：执行后 frame count、实际 decode/output token 数、执行后 `qwen_image_count`、load/runtime/memory、后验 batch/interference 状态。

## 4. 消融（C0–C4，字段冻结）

- `C0={exec_class}`；`C1=C0+{role, planner_mode, is_retry}`；`C2=C1+{model_class(exact|coarse), nested_model_class, merged_nested_call}`；`C3=C2+workload_scale`；`C4=C3+{stack/baseline, prefix_model_reuse}`。
- 报告每级 `ΔMAE`、`Δpinball`，回答“哪个变量修复了可辨识性”。exact/coarse 两臂分别跑（`R0_exact` = known-model ceiling，不外推未见模型；`R0_coarse` 为后续 resource-equivalence 方向）。

## 5. 估计器

1. 全局中位数（train）——very weak 描述性对照。
2. 条件中位数：按完整签名键分层 + **预注册回退层级**；记录每级 support count 与 validation/test/holdout 实际使用层级。
3. 主模型：LightGBM quantile（τ=0.50/0.90/0.95，log1p 训练，C4 特征）；固定网格 `num_leaves∈{31,63}`、`min_data_in_leaf∈{20,50}`、`learning_rate=0.05`、`n_estimators=400`、subsample/colsample=1.0；seeds={11,22,33}；**选型 = 三个 seed 的 validation `PB_primary` 均值**（worst seed 仅报告）；平票取更小 `num_leaves`；不做 early stopping、不按 split 调参。
4. hurdle occurrence 主模型：LightGBM binary（固定 `num_leaves=31`、`min_data_in_leaf=20`、`learning_rate=0.05`、`n_estimators=200`、subsample/colsample=1.0；seeds={11,22,33}），特征 = C4，按 validation **Brier score** 选型；参考基线 = 固定 L2 logistic（C=1.0，one-hot C4，不调参）。
5. 不加入 conformal；XGBoost 点回归仅可选对照。

## 6. 指标

- Core loss：`PB_primary = mean(PB_0.50, PB_0.90, PB_0.95)`（runtime）；`Improvement = 1 − PB_R0 / PB_baseline`。
- 校准：逐分位 `C_τ = P̂(Y ≤ q̂_τ)`；overall gate `max_τ |C_τ − τ| ≤ 0.05`（τ∈{.50,.90,.95}，含 q95）；major strata ≤ 0.10；mean QCalError 仅描述；`q05` 本轮不做，`[q50,q95]` 仅描述性区间，不声明 90% 覆盖。
- scale：训练 log1p；正式 gate 在 raw physical scale 上计算 pinball/MAE；跨资源比较才做 train-only normalization。
- major strata 规则（与结果无关）：validation 中 `n ≥ 200 节点且 ≥ 10 个独立 source video` 才应用 0.10 校准 gate，其余仅描述。
- quantile crossing：报告 raw crossing rate；指标计算前对每行预测做单调重排，并报告重排前数值作为敏感性。
- 分层与聚合：exec_class × model_class、residency（如可用）、workload_scale 分位桶、tail；node-pooled / per-run macro / per-video macro + video-cluster bootstrap CI。
- 泛化稳定性：`I_s = (Loss_baseline,s − Loss_R0,s) / Loss_baseline,s`；要求 `I_holdout > 0` 且 bootstrap CI 不支持大规模反转；validation→holdout 相对退化 ≤15% 为次要稳定性余量。

## 7. Go/No-Go 与诊断表

- **Core GO**（全部满足）：validation 上最佳预注册模型相对唯一 Core baseline（`exec_class-conditional train median`）的 `PB_primary` 改善 ≥15%（项目工程余量，UNVERIFIED）且 video-cluster bootstrap 改善 CI 下界 > 0；`max_τ` 校准 ≤0.05（major strata ≤0.10）；R0-Inclusive tail 无灾难性失败；memory 在 instrumented 主要类别可解释（否则 memory 仅 descriptive，Core GO 用 runtime + load）。
- **机制诊断（不 veto）**：条件中位数 vs 全局中位数；主模型 vs 条件中位数；C0–C4 逐级增益。
- **冻结 holdout**：一次性验收；`I_holdout` + bootstrap CI。
- **四象限（诊断假设，非结论）**：① Normal/Inclusive 都好 → 进入 J 系列；② Normal 好、Inclusive 差 → 优先查 cold/startup/stall 的执行前上下文与测量质量；③ runtime 好、memory 差 → 查 memory 测量与签名覆盖；④ 全 oracle 仍差 → 先确认 “current interface + measurement + estimator family 不足”，依次排查测量质量→估计器误配→缺失协变量→再决定是否暂停 J。
- **holdout failure 规则**：holdout 评估后若再修改 contract/模型/特征，该 holdout 立即失去 confirmatory 身份；绝不复用于改选。

## 8. Stage 0 / 执行计划

Stage 0 硬断言（全过才进 smoke）：
1. `load_ms` 不出现在任何 input/regime 派生路径；load 语义与 hurdle 结构已冻结；
2. **load join 表严格落在 15,481 个 v3.1 resource-applicable 节点**；逐类 missing/0/positive 互斥、合计等于类总数；nested 不成为独立 target；
3. `peak_*` 的 node-window/reset 语义已冻结（或 memory 降级为 descriptive）；
4. feature role 白名单逐列完成（pre_execution / oracle_label / target_only）；
5. 条件中位数回退层级与 support count 固定；
6. stall/tail 阈值仅由 train 构造并冻结。

执行：smoke 约 20 run / ≤300 节点；正式全量 + holdout 一次性评估；本地 CPU，预计 <10 CPU-min；无 GPU、无下载。正式前更新 `.project/EXPERIMENT_GATE.json`。
实验 ID：`EXP-20260911_p9d_r0_oracle_signature_ceiling`。

## 9. 交付物与可复现性

- 实验目录：`config.json`、`run.sh`、`metrics.json`、`RESULT.md`、逐节点预测/真值 artifacts（label 侧）、`run_manifest.json`（源哈希、seed、环境、命令、阈值）。
- 唯一控制面更新；v1–v3 数据集与既有实验只读。

## 10. Claims 边界

- 允许：在上述冻结契约、已观测 workload descriptors、执行前上下文与 oracle 签名条件下，所测节点类别具有经验可辨识性（empirical ceiling）。
- 禁止：resource prediction solved；future resources are predictable；topology predictor can predict the resource signature；observed-memory subset 结果外推为全部节点；known-model ceiling 外推为未见模型泛化。

## 11. 评审与冻结依据

- v1 评审：定位成立，MODIFY（3 P0：load 泄漏 / G3 覆盖率数学 / memory 语义）。
- v2 复核：3 P0 关闭；新 P0（`load_ms=0` 语义）+ 11 项 P1。
- v3 复核：load 计数口径错误（嵌套计入）+ occurrence 模型未预注册；条件式接受。
- 最终轮：load 计数与节点本体对齐（15,481；互斥合计成立；nested 排除）；hurdle 契约与 occurrence 预注册关闭；其余 P1 达预注册状态；无新 P0。
- 最终裁决：**R0 DESIGN v3 ACCEPT（可冻结）**；实现细节备注：主 LightGBM 配置选择用 3 seed validation `PB_primary` 均值，worst seed 仅报告（已并入 §5）。
