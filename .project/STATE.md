# State

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

## 记录指针

- 实验记录：`experiments/EXP-20260911_forecast_aware_scheduling/PHASE{0..7A,7BC}_REPORT.md`；
  `experiments/EXP-20260911_p9d_j_*`（R0/J/J4/H10/predictor acceptance）。
- 文献/评审：`docs/research/2026-09-14_fas_phase6_gpt_literature.md`、`docs/research/2026-09-11_fas_*`、`docs/p9d_*`。
- 控制面历史（本次归档）：`.project/archive/2026-09-15_pre_compact/`。
- 门禁：`.project/EXPERIMENT_GATE.json`（活跃实验 + 指针）。


