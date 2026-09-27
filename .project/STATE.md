## 2026-09-27 当前阶段：Stress regime —— **v1 作废 / v2 已冻结 α*=0.80，待一次性四臂测量**

- **目标**：把争用提到足够高，使五条基线与 F0 的对比有意义（Natural confirm300 的 H5 信息价值只有 **851 ms**）。
- **v1（绝对目标 ρ）作废**：construct-validity 审计失败——各集自带 ρ_old∈[0.50,1.05]，
  `arrival×(ρ_old/ρ)` 对 ρ_old>ρ 的集实为**拉伸**，只选到"均匀化"点（ρ'=0.83，competitive 反低于 Natural）。
- **v2（乘性到达缩放 α）已冻结 α\*=0.80**：
  - 变换：对**每集每个到达间隔**同乘 α；保持模板/图/成员/顺序/异质性，**严格压缩**；deadline 保持绝对预算。
  - 对照：**A0B0 natural(dev)** 与 **B-only(α=1.0 all-normal)**；**B 结构上无效**（Δcomp −0.0048、Δutil 0.0000）。
  - 机械选取（dev 100 集，**禁用性能指标**）：从 α=0.95 往小扫，取**满足全门的最大 α** →
    α=0.95/0.90/0.85 落选（0.85 仅差 0.0008 且 CI 跨阈），**α=0.80 通过**（comp 0.5651、util 0.8321、Δcomp +0.1462 [0.1194,0.1766]、Δutil +0.1039、p50 2.14、orderRet 1.0、0 失败）。
  - 评审：r1/r2 REJECT → **r3 ACCEPT（0 P0 / 0 P1）**；披露项已写入门禁。
- **限制（必须随文披露）**：α\* 是固定网格最大合格值；`ordering_retention` 仅 **capacity proxy**（不建模 residency/eviction/admission）；
  标定仅 100 集 dev；本地无 git；stress 全程 **post-hoc**，Natural confirm300 不变。
- **下一步**：一次性跑 Myopic → F0 → Truth-H5 → Oracle，揭晓 `O_H5=Myopic−TruthH5`、`O_full=Myopic−Oracle`、`H5Coverage=O_H5/O_full`；
  若 H5 仍小 → 如实报告"提高到达争用未显著提高 H5 可操作价值"，**不得再改 workload**。

## 2026-09-18 指标口径审计 — 前提撤回

- **"预测器严重失准"不成立**：用同一套统计代码给 J 验证集打分，**逐位复现**验收数字
  （覆盖 0.5555/0.9166/0.9680、RuntimeQScore 845.0）⇒ 两套打分同尺。
- **域内 Σp50/Σ真值 = 0.6322（不是 1.0）**；这是"逐步中位数之和 ≠ 总和的中位数"的必然结果
  （真值中位 2209 ms vs 均值 3877 ms，1.76×）。
- **主指标 RuntimeQScore：R7 = 782.1 优于域内 845.0**。
- 90%+ 是**分类头**（结构/行为），与资源头（覆盖/pinball）无关。
- 唯一真实残余：p50 覆盖 R7 0.427 vs 域内 0.556。
- 连带：Phase 18 的"乐观"基准要从 1.0 改为 0.632；**C2 需重新论证**；禁止写"低估约 60%"。

# State

## Current Stage (2026-09-27)

- **主线**：agent workflow 的未来信息**以什么形式**进入 GPU 调度（runtime / risk / topology / information value）。
- **预测器**：**F0（seed 11）为部署预测器**（`EXP-20260921_histres_causal_input_v1`，commit `a01e7eb`）。
  历史执行遥测**无增量价值**（CI 跨 0）；完整重训 + 16-bin 离散头相对冻结 J3 **提升约 28.7%**，
  且 F0 是第一个同时改善"真值参照排序"的预测器。
- **主实验**：`EXP-20260921_scheduler_replication_v1`（commit `b6d5077`），主臂 F0 `sameshape_h5_p95`、历史臂 A0（J3）。
  **formal 5 臂 confirm300 已完成**（commit `694e6fc`，300 集，1879.8 s）：F0 **显著优于全部五条基线**
  （LLMSched +7,049 / TIE +14,714 / Latency-Aware +16,353 / Pythia +33,510 / Agentix +47,055 ms，Δ CI 下界全部 > 0；
  F0 均值 99,967 ms）。**范围限定**：这是同一 substrate 的"完整系统对比"，**不能**单独主张"我们的调度器更强"
  （需 same-interface 2×2）；五条基线各自的 adaptation/omission 必须随表披露。
- **workload**：主用 **v04 因果模板**（`r7_workload_v04_causal_v31_no_run_container`，schema v0.3，
  `verified_serial_control_flow_v3_1`，`chain_tail_is_answer` 640/640，8,295 节点，容器已排除）；
  模拟器 `load_templates` 有显式 `topology_view`（`legacy` / `causal_v3`）与两条 fail-closed 规则。
  另有 v03（去容器，保留）、v041（ontology）、以及 **v03 stress ρ=1.05**。
- **压力 regime**：进行中（见本文件顶部 09-27 段与 `.project/PLAN.md`）。
- **纪律**：Natural confirm300 冻结保留作确认性评测与 QoS sensitivity；stress 只作 post-hoc。

## 近期阶段摘要（2026-09-15 → 09-26）

> 说明：本文件原 09-15/09-17/09-18 三段正文在 2026-09-27 发现**不可逆编码损坏**（详见文末"控制面完整性"），
> 已按归档与门禁**重写为摘要**；逐字原文见 `.project/archive/`（09-15 快照）与
> `.project/archive/2026-09-27_handoff_log/HANDOFF_log_20260926.md`（09-18 → 09-26 流水）。

- **09-15**：Phase 0–7 完成；消费方式锦标赛定出 q95 冠军；dev700 / frozen confirm300 切分建立；
  GPT 给出系统论文定位与最小补充实验表。
- **09-17**：oracle 对照 key 形状混淆被认定并撤回（`trueopt_h5` 与预测族不同 key 形状）；
  runtime/load 契约审计发现 load 被部分重复计入（修法待定，未动预测器）。
- **09-17 → 09-21**：发现 R7 workload 把"整条流墙钟容器"当普通 GPU 节点调度（640/8,935 节点，占 node runtime **50.24%**）
  → 建 **v03 去容器 workload**（重建自检 21,000 条 episode 逐字段 0 不匹配）；Phase 20-A/20-B 给出消费层证据。
- **09-18**：Phase 21（补 task_context）pilot 在 Windows 执行 → **预注册负结果**（四条 calibration gate 全 FAIL、
  p50 pinball 显著恶化 +51.6 ms）；事后审查判定"context 块缺失"为**误报并撤回** → 不重生成预测包、不重跑调度。
- **09-19 → 09-21**：资源分布预测器线（`EXP-20260919_j_series_resource_dist_v1`，R1/R1b viability PASS、strong FAIL）；
  因果历史遥测线（`EXP-20260921_histres_causal_input_v1`）定出 **F0**。
- **09-23**：拓扑契约回归修复 → **v04 因果模板**；调度机会量化（修正图下 `Pr(feasible≥2)=52.82%`、
  `Pr(ready GPU≥2)=68.24%`）；四条论文基线全部实现；新增 6 个测试套件（全量 276，5 个失败套件均为预先存在）。
- **09-25 → 09-26**：五条基线的 adaptation 收尾（Pythia ACCEPT + re-pin、Agentix PLAS 服务口径 + discrete 敏感性、
  LLMSched freeze-breaking 修复）→ **formal 5 臂测量完成**。

## 活跃约束（必须遵守）

- `T_final` 封存；`S_*` 只允许推理、不拟合；holdout 不进入正式决策（历史 holdout 已用于 R0/J 判定，保持密封语义）。
- **Natural confirm300 冻结保留**；stress regime 只能作 post-hoc，且标定期间禁用一切性能指标。
- 运行器是 resume 式：同一 `--output-dir` 会跳过已有 `(episode_id, policy)`；重跑同一策略请换新目录。
- **本机（macOS）**不能跑 torch / predictor packer / GPU 实验；训练机为 Windows（`D:\anaconda\envs\scheduler\python.exe`，仓库 `F:\scheduler`）。
  其余调度运行 `PYTHONPATH=src`。
- 可用 artifacts：`outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts`（含每步 model_probabilities 与每行 length_probabilities）、
  `outputs/sstar_predictor_artifacts_sched/prediction_artifacts`（无分布）、H10：`outputs/sstar_predictor_artifacts_h10[_sched]`。
- Git 只提交代码/schema/配置/清单/文档/小型指标；不提交视频、权重、原始 trace、缓存、大型输出。
- 论文纪律：基线必须随表披露 adaptation/omission；不得声称"调度器更强"（除非 same-interface 2×2）；
  Phase 18 类"乐观"表述须以 **0.632** 为基准，禁止写"低估约 60%"。

## 已验证结论（按主题）

1. **消费方式（v03-dev700，Phase 20-B 九臂）**：`oracle-truth` 仅比 **p95** 好 **1.3 s**；**p90 ≈ p95**；
   p50 ≈ E0（不显著）；**tail-shuffle 打乱对齐后收益全失**（+4.25 s）；scaled-p50 显著更差（+15.1 s）。
   → 收益来自"**与未来步对齐的保守尾部聚合**"，不是"有未来信息"、不是"尺度放大"。
2. **workload 构造**：整条流容器节点 640/8,935 却占 node runtime **50.24%**（GPU 的 50.5%）；v03 已排除，
   重建自检 21,000 条 episode 逐字段 0 不匹配；v04 因果契约修复后 `chain_tail_is_answer` 640/640。
3. **指标口径**：域内 `Σp50/Σ真值 = 0.6322`；RuntimeQScore **R7 782.1 优于域内 845.0**；
   p50 覆盖 R7 0.427 vs 域内 0.556（唯一真实残余）。
4. **预测器**：F0（seed 11）为部署预测器；历史执行遥测无增量价值；重训 + 16-bin 头 ≈ +28.7%。
5. **基线复现**：五条基线（LLMSched / TIE / Pythia / Latency-Aware / Agentix）已实现并在 confirm300 上与 F0 正式对比（结果见上）。
6. **J 线（历史）**：runtime 显著优于 B1（J2/J3 3/3 seeds），但 load-duration 贴线越界 → 无 Core GO；
   J4 解耦负结果；H10 朴素重训负迁移（需 H10-lite）。

## 未决问题与风险

- **Stress-v2 已冻结 α\*=0.80**（见顶部；评审 r3 ACCEPT）；**confirm300 的 v2 α=0.80 配对产物已生成**，四臂尚未跑。
- 门禁中另存一份 **v1 stress 四臂 ladder 数字**（O_H5=7176 ms），但**无任何产物文件可追溯**；按 2026-09-27 决定**不采用**，仅留 historical。
- **same-interface 2×2 未做** → 目前**不能**主张"调度器更强"。
- 09-26 → 09-27 工作**已 commit/push**（GitHub `QiuweiLiu/scheduler`，`bf0da99`）。
- 两个**预先存在**的 bug 已定位未修（09-23 记录）：`round_robin` 实跑 myopic；`fcfs` 安全。
- 全是仿真实验（无真实系统回放）；S_* 为域外推断。
- 控制面完整性（编码损坏）已处理，见文末。

## 下一步（按优先级）

1. **在已冻结的 v2 上一次性跑四臂**：`experiments/.../r7_workload_v03_stressv2_confirm080`（confirm300，α=0.80）
   → Myopic → F0 → Truth-H5 → Oracle，揭晓 `O_H5 / O_full / H5Coverage`。**不得再改 workload/gate**。
2. 若 H5 仍小 → 如实报告"提高到达争用未显著提高 H5 可操作价值"，不回头调参。
3. 五条冻结基线**仅在科学上值得时**在 stress 上跑；论文随表写 disclosure 与零机会统计。
4. （可选）same-interface 2×2；随后再谈 Phase 17 契约修复 / H10-lite 等方向。

## 记录指针

- 计划：`.project/PLAN.md`（顶部 = **Stress-v2** 协议；v1 段落已标 HISTORICAL）。
- 门禁：`.project/EXPERIMENT_GATE.json`（`stress_regime_v1_20260927`、`EXP-20260921_scheduler_replication_v1`、
  `EXP-20260921_histres_causal_input_v1`、`EXP-20260919_j_series_resource_dist_v1`、`EXP-20260911_*`）。
- 决策：`.project/DECISIONS.md`（尾部为 09-26/09-27 条目）。
- 实验记录：`experiments/EXP-20260921_scheduler_replication_v1/`（含 `artifacts/four_baseline_formal_v1.json`）、
  `experiments/EXP-20260911_forecast_aware_scheduling/PHASE*`。
- 文献/评审：`docs/research/2026-09-*.md`（含 `2026-09-18_fas_phase21_review_gpt.md`、`2026-09-25_four_baseline_review_gpt.md`、
  `2026-09-26_pythia_and_agentix_final_review.md`）。
- 归档：`.project/archive/2026-09-15_pre_compact/`（09-15 快照）、
  `.project/archive/2026-09-27_handoff_log/HANDOFF_log_20260926.md`（09-18→09-26 流水）、
  `.project/archive/2026-09-27_state_corrupted/`（本次重写前的损坏副本）。

## ChatGPT Web 绑定（权威源，2026-09-18 从归档恢复）

```yaml
chatgpt_web:
  status: active
  generation: 8
  conversation_id: "6ab6b895-43b8-83ec-982a-4c13ac404115"
  conversation_url: "https://chatgpt.com/g/g-p-6a45fd9c56e48191a5e1a006582fcdef-diao-du/c/6ab6b895-43b8-83ec-982a-4c13ac404115"
  title: "鏋舵瀯璁捐璇勪及"
  model: "GPT-5.6 Sol"
  reasoning: "High"
  created_at: "2026-09-02 21:30 CST"
  last_verified_at: "2026-09-27 (opencode session: URL re-confirmed by the project owner; page loaded, project scope 调度)"
  last_used_at: "2026-09-27"
  parent_conversation_id: null
  last_rollover_reason: null
  # NOTE 2026-09-27: this project-scoped conversation is the authoritative one.
  # The older conversation 6a9822da-e278-83e9-9c1a-675923acda0e (recorded in the
  # 09-15 HANDOFF) is at its length limit and must not be used for new briefs.
```

## 2026-09-23 状态追加

- **拓扑契约回归已修复**：v04 因果模板（`r7_workload_v04_causal_v31_no_run_container`），
  seriality gate 640/640 PASS，`chain_tail_is_answer` 640/640。
- **`load_templates` 有显式 `topology_view`**（`legacy` / `causal_v3`），两条 fail-closed 规则已验证。
- **调度机会量化**：修正图下 `Pr(feasible_actions>=2)=52.82%`、`Pr(ready_GPU_nodes>=2)=68.24%`
  —— **调度问题依然成立**，不需要重设 workload pressure，也不能恢复假并行。
- **四条论文基线全部实现**：TIE / Pythia / LLMSched / Latency-Aware（含独立调度器）。
- **新增 6 个测试套件**（tie / pythia / llmsched / latency_aware_fusion / latency_aware_fidelity /
  latency_aware_scheduler），全量 276 个测试，5 个失败套件**全部预先存在**。
- **两个预先存在的 bug 已定位未修**：`round_robin` 实跑 myopic；`fcfs` 安全。

## 控制面完整性说明（2026-09-27）

- **发现**：本文件自 `# State` 以下的 09-15/09-17/09-18 三段正文与部分小标题存在**不可逆编码损坏**
  （UTF-8 文本被按 GBK 解读后再编码，部分字节已丢失为 `?`）。已验证 gbk 反解仅能部分恢复（86/150 行仍含损坏字符），
  **不做自动批量修复**。
- **处置（本次）**：损坏文件原样归档到 `.project/archive/2026-09-27_state_corrupted/STATE_corrupted_20260927.md`；
  本文件按**归档 + 门禁 + 09-26 HANDOFF 流水**重写为可读版本；可读段落（09-18 口径审计、09-23 追加、ChatGPT 绑定块）**逐字保留**。
- **教训**：跨机器（Windows ↔ macOS）写同一份 UTF-8 文件时，必须避免用按 GBK 默认编码的工具读写；
  控制面文件的编码应显式声明 UTF-8。
- **同时**：`.project/HANDOFF.md` 已由日志（403 行、11 个 `# HANDOFF` 块）收敛为快照；
  旧日志归档在 `.project/archive/2026-09-27_handoff_log/`。
