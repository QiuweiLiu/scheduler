# State

## Current Stage (2026-09-18)

- 主线不变：视频 Agent 未来预测 → GPU 调度。**主 workload 已切到 v03**（`r7_workload_v03_no_run_container`，移除 640 个重复计数容器节点）。
- **本回合（Windows 执行）**：Phase 21 pilot 结案 = 预注册负结果（**差分成立**）。
  - S_* `task_context` 修复（16 视频 paired masked/fixed）**FAIL 全部四条 calibration gate**：
    fixed p50 R=0.397（门限 [0.70,1.30]）、cov p50/p90/p95 = 0.396/0.844/0.916、p50 pinball **显著恶化 +51.6ms [27.2,74.4]**。
  - 管线 gate 全过（join 1.0、unknown_rate 0、`prefix_hash` 975/975 一致、checkpoint SHA 未变）→ **干净单变量对照**。
  - → **不重生成 S_* 预测包、不重跑调度族**（该结论对"内容字段"仍成立）。
- **同日事后代码审查（最终结论）：context 块无缺陷，Phase 21 总括结论成立**
  - 期间曾提出"三个 stack 字段被 null 遮蔽 → 100% UNK"的 P0，**经复查为误报并已撤回**。
  - 根因：探针用 `dict.get(field)`，**无法区分"键不存在"与"键存在但值为 null"**；"键存在且为 null"是推断而非实测。
  - 决定性反证（显式测键存在性）：`task_context` 的键集 975/975 =
    {answer_type, domain, official_task_type, question_type, required_modalities, sub_category, temporal_scope}
    —— 恰好 7 个键，与 `_task_context()` 输出及域内 J 布局**完全一致**；
    `baseline`/`model_stack_id`/`planner_model_id` 的 `key_present = False`（975/975）
    → 编码器正常走 `else` 分支、从 `stack_context` 读真实值（`baseline = langgraph_react`，**in_vocab**）。
  - 真正残余（均非 bug）：`temporal_scope = "unknown"` → UNK（**registry 无此字段**，数据边界）；
    `model_stack_id = stack_a_qwen3_vl8b_yolo11x` → OOV（**数据集命名漂移**，域内为近似值 `stack_a_qwen3_vl8b`），
    另一半 486 个锚点（`stack_b_..._yolo26n`）在词表内；`planner_model_id` 域内词表只有 `unknown`（从来不含信息）；
    `required_modalities` 字符串形态域内也存在（J train 3439 行是字符串）。
  - **未做任何管线改动**（不需要）；教训已登记：审计必须显式测键存在性，不得用 `.get()` 推断。
- **独立 GPT 审查已完成（Sol + High）**：**核心负结果 PASS + 误报撤回 PASS**；但驳回"不存在同类未修输入缺口"
  与"误差主要来自结构错配/分布偏移"（已撤回）。归档 `docs/research/2026-09-18_fas_phase21_review_gpt.md`。
  - **它要求的两个封存前检查我已跑完并全部通过**：真值顺序等价性 640/640 + 64/64 模板完全一致
    （Q1 转 VERIFIED，pilot 无需重跑）；两 pack 均 `min_steps=5`（P0-2 关闭）。
  - **新关键点**：`__UNK__` ≠ 训练时的字面量 `"unknown"` token → `planner_model_id` / `model_stack_id`(stack_a) /
    `temporal_scope` 仍是真实的 deployment input mismatch。
  - **零成本分层（现有输出）**：in-vocab 的 stack_b 校准明显好于 OOV 的 stack_a
    （fixed R50 0.5820 vs 0.1751；cov90 0.9012 vs 0.7878；cov95 0.9699 vs 0.8620）→ 命中"先查 stack canonicalization"。
    但**有混淆**（两者本就是不同执行栈、runtime 分布不同），需配对反事实才能定因果。
- **下一步（GPT 指定，均几十秒级）**：21-B1 把 `planner_model_id` 设为字面量 `"unknown"`；
  21-B2 把 `stack_a_qwen3_vl8b_yolo11x` 映射到词表内的 `stack_a_qwen3_vl8b`（**前置：先确认 stack identity 定义相同**）。
  判据：mean pinball 改善 ≥10% + CI upper < 0 + coverage/R50 朝目标移动。两者都失败才归因到分布偏移。
- **待用户决策**：① 是否执行 21-B1 / 21-B2（前置的 stack identity 确认）；② 方向 A（P2 sweep + v03-confirm300）/
  B（Phase 17 契约修复）/ C（H10-lite 等）。
- 冠军与机制表述仍按 09-17 状态：**v03 上 `predopt_h5_q95`（=r95）为当前冠军**，机制 = "与未来步对齐的保守尾部聚合"。

## Current Stage (2026-09-17)

- 主线不变：视频 Agent 未来预测 → GPU 调度；冠军仍是 r95/q95（逐步骤 runtime p95 求和）。
- **09-16 → 09-17 新增**：
  - Phase 11–15：runtime-only 正交族（C2 严格检验通过）、CVaR 重写（固定长度后场景族仍落后 15–17.5s）、
    优化型参考重跑（exploratory negative，仍落后 ~19s）、CP-RHO executed 变量修复 + 300 集配对。
  - **trace 依赖假设被否**：未约束 MOM run 层分量 −0.139（CI 全负）、截断 ICC=0、尾部 lift 双侧不显著
    → 放弃"共单调/尾部共动"物理叙事，改挂 **forecast-error-aware ranking surrogate**。
  - 公开仓库两轮审阅的 P0/P1/P2 全部落地（见 `POSTFIX_REPORT.md`）。
  - **Phase 16（本回合）**：oracle 对照 confounded 认定 + 撤回；新增 `sameshape_h5_{p50,p95,truth}` 三臂（opt-in，未跑）。
- **09-15 段落以下仍是有效细节快照**，但其"下一步"列表已过时（压力 sweep 实为已完成，6/6 cell 显著为正）。

## Current Stage (2026-09-15)

- 主线：视频 Agent 工作流未来预测 → GPU 调度（forecast-aware scheduling）；Phase 0–7 全部完成并归档。
- **当前冠军配置**：H5 预测器（J3:seed11，v3.1 契约） + **q95 消费**（逐步骤 runtime p95 相加；load 分量与收益无关）。
  - dev700：179,125ms，**−17,229 [−19,037, −15,476]** vs E2；miss −1.40pp。
  - frozen confirm300（首次使用）：183,248 vs 202,012，**−18,765 [−21,948, −15,932]**，miss −1.36pp。
- 评测纪律：validation 已切 **700 dev / 300 frozen confirm**（`data/manifests/validation_split_dev700_confirm300.json`，seed 20260914）；
  调参只在 dev；冠军只在 confirm 跑一次。运行器支持 `--episode-ids-file`。
- 外部评审：GPT 文献 + 计划（`docs/research/2026-09-14_fas_phase6_gpt_literature.md`）；系统论文定位已定稿（`docs/research/2026-09-15_fas_system_story_gpt.md`）：主贡献 = future-action-aware scheduling system；机制贡献 = continuation/horizon 主导 + runtime-tail 消费；关键新颖性边界 = 预测**动态展开、尚不存在**的未来控制流（区别于 Parrot 的已知 DAG）；最小补充实验 = P0 真实 GPU replay（最大短板）→ P1 压力 sweep + 预测质量敏感性 → P2 H10-lite。故事线、基线清单与投稿概率见 `docs/research/2026-09-15_fas_novelty_venue_gpt.md`：纯仿真 FGCS/JPDC/ICPP/CCGrid 现实，补 2-GPU replay 后 MLSys 12%→25–35%、ATC/EuroSys 进入可冲区间；审稿三大攻击（simulator artifact / 单 workload / q95 调参）必须用 replay + pressure sweep + 机制分析防御。补充实验总表与三档实验集见 `docs/research/2026-09-15_fas_experiment_table_gpt.md`；工业电梯 Agent 定位为第二 workload / case study（冻结接口与 q95，只重生成 artifacts；最好与 2-GPU replay 合并做）。

## 活跃约束（必须遵守）

- `T_final` 封存；`S_*` 只允许推理、不拟合；holdout 不进入正式决策（历史 holdout 已用于 R0/J 判定，保持密封语义）。
- 运行器是 resume 式：同一 `--output-dir` 会跳过已有 `(episode_id, policy)`；重跑同一策略请换新目录。
- 可用 artifacts：`outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts`（带每步 model_probabilities 与每行 length_probabilities）
  与 `outputs/sstar_predictor_artifacts_sched/prediction_artifacts`（无分布）。H10：`outputs/sstar_predictor_artifacts_h10[_sched]`。
- 环境：解释器 `D:\anaconda\envs\scheduler\python.exe`（torch 2.6.0+cu124，RTX 3060 Laptop），所有调度运行 `PYTHONPATH=src`。
- 桥接：Chrome for Testing (`E:\chrome-for-testing\chrome-win64\chrome.exe`) 必须带
  `--proxy-server=http://127.0.0.1:7897`（本机直连 chatgpt.com 超时）；会话 URL `https://chatgpt.com/c/6a9822da-...`。
- Git 只提交代码/schema/配置/清单/文档/小型指标；不提交视频、权重、原始 trace、缓存、大型输出。

## 已验证结论（按主题）

1. **信息价值**：未来结构预测显著优于无未来（E2 vs E0 −17,065ms，1000 集）；
   oracle 资源 ≈ 静态表（183,845 vs 184,514）→ **资源预测不是瓶颈**；真值内容反而更差（+14,222ms），
   去掉模型身份更好（−12,334ms）→ 内容/身份无正价值，**长度/终止结构是主要价值来源**。
2. **Horizon**：真值 H10 vs H5 = −10,231ms [−11,115, −9,363]；H10 ≈ 无界 oracle；H20=H10 饱和 → H5→H10 是最大可部署杠杆。
3. **消费方式（16 变体，结论稳定）**：**q95（逐步骤 runtime p95 求和）最优**；
   场景采样+CVaR（独立 −13.2k / 共单调 −7.8~−10.5k）、自适应风险（−11.4k）、机会约束、生存、缓存、内容全部更差；
   **纯均值 +13.8k、load 尾部（ld95）+15.1k**；rt95 ≈ q95（+11ms，统计等价）。
   → 机制：价值来自对预测链 **runtime 尾部** 的"完全相关式"惩罚；对 load 维度或平均化处理会削弱它。
4. **J 线（预测器）**：runtime 显著优于 B1（J2/J3 3/3 seeds），但 load-duration 贴线越界（+5.2%/+5.3%）→ 无 Core GO；
   J4 解耦负结果；归因（τ=.90 差、τ=.95 好、中位数校准更好）已结案；J 线收束。
5. **H10 重训（朴素延长窗口）**：负迁移 —— 前 5 步退化（next-family 0.803→0.640、runtime pinball +62%），调度 +14.6s；
   需 H10-lite（H5 backbone + 后段辅助头）。

## 未决问题与风险

- H10 负迁移未修复（H10-lite 未实现）；cache-aware 消费缺 residency/reuse 数据（暂缓）。
- 全是仿真实验（无真实系统）；论文可行性/短板待 GPT 评估（可能需跨 workload/期限稳健性与预测器×消费联合实验）。
- S_* 域外退化已量化（role 0.912/family 0.803/runtime pinball 1,045）；消费结论只在当前 workload/期限设置验证过。
- 1,000 集已被多次使用 → 依赖 dev/confirm 纪律控制过拟合，后续新消费器必须先在 dev 上报。

## 下一步（按优先级）

1. 先核验 GPT 引用论文的真伪/出处（Parrot、Pythia、PBKV、FATE、SOLA、Vidur 等），再更新计划。
2. H10-lite（H5 backbone + 后段 horizon 辅助头/降权，multi-resolution）。
3. 备选：7-D queue-aware rollout（预期低）、跨期限/λ 稳健性、预测器 seed 稳健性（J3 seed22/33）。

## ChatGPT Web 绑定（权威源，2026-09-18 从归档恢复）

2026-09-15 压缩时该块从活跃 STATE 丢失，仅存归档；现恢复并更新。skill 规定此块是会话绑定的唯一真相源。

```yaml
chatgpt_web:
  status: active
  generation: 7
  conversation_id: "6a9822da-e278-83e9-9c1a-675923acda0e"
  conversation_url: "https://chatgpt.com/c/6a9822da-e278-83e9-9c1a-675923acda0e"
  title: "架构设计评估"
  model: "GPT-5.6 Sol"
  reasoning: "High"
  created_at: "2026-09-02 21:30 CST"
  last_verified_at: "2026-09-08 17:19 CST"
  last_used_at: "2026-09-08 17:19 CST"
  parent_conversation_id: null
  last_rollover_reason: null
```

**2026-09-18 实测（Windows 侧桥接）**：会话可达、线程健康（末轮为 Phase 21 计划），模型选择器当前显示
**`DS Flash`** 而非登记的 `GPT-5.6 Sol` → model gate 拒绝提交（`selected_model: null`、`high_selected: false`），
**未提交任何 brief**。在用户把该会话的手动模型切回 `GPT-5.6 Sol` + High 之前，网页审查不可用。

## 记录指针

- 实验记录：`experiments/EXP-20260911_forecast_aware_scheduling/PHASE{0..7A,7BC}_REPORT.md`；
  `experiments/EXP-20260911_p9d_j_*`（R0/J/J4/H10/predictor acceptance）。
- 文献/评审：`docs/research/2026-09-14_fas_phase6_gpt_literature.md`、`docs/research/2026-09-11_fas_*`、`docs/p9d_*`。
- 控制面历史（本次归档）：`.project/archive/2026-09-15_pre_compact/`。
- 门禁：`.project/EXPERIMENT_GATE.json`（活跃实验 + 指针）。


