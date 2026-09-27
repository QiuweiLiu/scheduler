# Handoff

**Last Updated:** 2026-09-27（opencode 会话；日志式 HANDOFF 已收敛为快照并归档在
`.project/archive/2026-09-27_handoff_log/HANDOFF_log_20260926.md`；`STATE.md` 编码损坏已重写归档在
`.project/archive/2026-09-27_state_corrupted/`；stress ρ 网格标定完成）

## Goal

回答并论证：**agent workflow 的未来信息应以什么形式进入 GPU 调度**——runtime / risk / topology / information value。
部署预测器 = **F0（seed 11）**；主结果 = 与五条已发表基线在同一 substrate 上的完整系统对比；
另建 **post-hoc High-Contention Stress Regime** 以获得足够争用下的公平对比。

## Done（最近，09-18 → 09-26）

- **正式 5 臂测量（frozen confirm300）**：commit `694e6fc`，300 集，1879.8 s。
  Δ = 基线 − F0（正 = 更差）：LLMSched **+7,049** [5,941, 8,259]（260/300 更差）、TIE **+14,714** [12,359, 17,316]、
  Latency-Aware **+16,353** [13,707, 19,252]、Pythia **+33,510** [28,900, 38,620]、Agentix(PLAS) **+47,055** [40,606, 54,058]。
  F0 均值 **99,967 ms**；五条基线 Δ 的 CI 下界全部 > 0。
  产物：`experiments/EXP-20260921_scheduler_replication_v1/artifacts/four_baseline_formal_v1.json`。
- 基线收尾：**Pythia 算法 ACCEPT**（provenance re-pin `head_full_sha = d01d202…`）；**Agentix 主臂 PLAS 服务口径修正**
  + `agentix_mode=discrete` 敏感性（train-only 分位档界 `(0,13785,23111,35100)`，K_nonempty=4，防饿死 9.7%）；
  **LLMSched freeze-breaking 修复**（验证期 OOV stage 导致崩溃 → `on_unknown="skip"` 并计数，待 re-pin head）。
- **F0 成为部署预测器**：`EXP-20260921_histres_causal_input_v1`（commit `a01e7eb`）——历史执行遥测**无增量价值**（CI 跨 0），
  但完整重训 + 16-bin 头相对冻结 J3 **提升约 28.7%**，且 F0 是第一个同时改善"真值参照排序"的预测器。
- **指标口径审计（09-18）撤回一个前提**：用同一套统计代码给 J 验证集打分**逐位复现**验收数字
  （覆盖 0.5555/0.9166/0.9680、RuntimeQScore 845.04）；域内 `Σp50/Σ真值 = 0.6322`（不是 1.0，是"逐步中位数之和 ≠ 总和的中位数"的必然结果）；
  **RuntimeQScore：R7 = 782.1 优于域内 845.0** → 不再主张"预测器严重失准"。
- **Phase 21 结案（预注册负结果）**：S_* `task_context` 修复的 16 视频 paired pilot **四条 calibration gate 全 FAIL**，
  且 fixed 的 p50 pinball **显著恶化 +51.6 ms [27.2, 74.4]**；管线 gate 全过（join 100%、unknown_rate 0、prefix_hash 975/975 一致、checkpoint SHA 未变）。
  同日事后审查判定"context 块缺失"为**误报并撤回**（探针用 `dict.get()` 无法区分"键不存在"与"键存在但为 null"）
  → **不重生成 S_* 预测包、不重跑调度**。
- **Stress regime v1 ρ 网格标定（09-27）**：变换 `stress-arrival-compression-all-normal-v1` 已实现，
  网格 {1.05…1.50} 与细化网格 {0.80…0.95} 均以 **Myopic-only 结构门**在 300 集 confirm 上测过
  （`experiments/EXP-20260921_scheduler_replication_v1/artifacts/stress_rho_calibration_v1.json`，
  gate = `scripts/preprocess/stress_structural_metrics.py`，templates = v041，topology_view = causal_v3）。
  **预注册网格 {1.05…1.50} 无一合格**（competitive ≥ 0.8168、util ≥ 0.9225）；细化后 **最小合格 ρ' = 0.83**
  （comp 0.5384、util 0.8039、p50 2、0 失败）；Natural 参照 comp 0.5866 / util 0.7703。
  **前提问题**：ρ'=0.83 < Natural 中位负载 0.85，且其 competitive 反低于 Natural → 合格点是"均匀化"而非"提高争用"，**已暂停冻结并上报**。
- **workload 与消费层（09-17 起，已并入主线）**：主 workload = **v03 episodes + v041 causal 模板 + `topology_view=causal_v3`**（删除 640 个整条流容器节点）；
  Phase 20-A 1000 集配对：E2−E0 由 −27.5 s 缩到 **−0.79 s**（缩 97%，仍显著）、r95−E2 由 −7.2 s 缩到 **−4.1 s**（存活）、FCFS 由"与 E0 持平"变 **+28.0 s**；
  Phase 20-B 9 臂消费矩阵（v03-dev700）：**oracle-truth 仅比 p95 好 1.3 s**，p90 ≈ p95，p50 ≈ E0，**tail-shuffle 收益全失**，scaled-p50 显著更差。

## Verified（可直接引用）

- 五条基线与 F0 的 confirm300 对比（上表）——注意这是**同一 substrate 的完整系统对比**，
  **不能**单独主张"我们的调度器比别人的调度器强"（那需要 same-interface 2×2）；每条基线的 adaptation/omission 必须随表披露。
- 域内指标口径：Σp50/Σ真值 = **0.6322**；RuntimeQScore R7 **782.1** vs 域内 **845.0**。
- Phase 21 pilot：pipeline gates 全过、calibration gates 全 FAIL、pinball 显著恶化。
- v03 workload 正确性：重建自检 21,000 条 v02 episode **逐字段 0 不匹配**；容器占原 node runtime **50.24%**。

## Rejected（不得重跑，除非有新证据）

- **重生成 S_* 预测包 / 修复 task_context**（Phase 21 预注册负结果）。
- "预测器严重失准"这一前提（09-18 已撤回；Phase 18 的乐观基准须由 1.0 改为 **0.632**，禁止写"低估约 60%"）。
- **重新采样 stress episodes**（会同时改变 arrival 密度与 program 数，无法归因）。
- 为让某条基线变强而调 workload 参数（所有 stress 参数只能由 dev-only 结构指标机械选出）。

## Open

- **Stress-v2 已冻结 α\*=0.80**（本地评审 r1/r2 REJECT → **r3 ACCEPT**；其后 **GPT GitHub 审阅**给出 NEEDS SMALL FIX，
  6 个 P1 已修：p50 macro 的 median/mean bug、配对 fail-closed、capacity 语义、rev3 重跑、控制面 stale、confirm300 产物）。
  rev3 结论不变：α=0.80 通过全门（Δcomp +0.1462 [0.1194,0.1766]、Δutil +0.1039、p50macro 2.0、simulated_oom 0、0 失败）；
  α=0.85 差 0.0008 且 CI 跨阈。
  **待办**：在 `r7_workload_v03_stressv2_confirm080`（confirm300，α=0.80，已生成并校验配对）上一次性跑 Myopic→F0→Truth-H5→Oracle。
- **门禁里另有一份 v1 stress 四臂 ladder（O_H5=7176 ms），无产物可追溯 → 2026-09-27 决定不采用，仅留 historical。**
- same-interface 2×2（若要主张"调度器更强"）。
- 论文：五条基线 disclosure + 零机会统计随表写出；Natural regime 的 post-hoc 措辞。
- 09-26 → 09-27 工作**已 commit/push**（GitHub `QiuweiLiu/scheduler`：`0603b43` + `bf0da99`）。
- 既有 bug（09-23 记录，已定位未修）：`round_robin` 实跑 myopic；`fcfs` 安全——可能影响引用 round_robin 臂的结论。

## Active

- 无运行中的调度实验（据控制面；本机为 macOS，不能跑 torch/packer）。

## Next

1. **在冻结的 stress 上一次性跑四臂**：Myopic → F0 → Truth-H5 → Oracle（confirm300 配对上），
   揭晓 `O_H5=Myopic−TruthH5`、`O_full=Myopic−Oracle`、`H5Coverage=O_H5/O_full`（Natural 现值 851 ms / 2852 ms / ≈29.8%）。
2. 若 H5 仍小 → 如实报告"提高到达争用未显著提高 H5 可操作价值"，**禁止再改 workload**，转路线 (d)。
3. 冻结披露随文写出（α=0.80 网格边界、α=0.85 边界、capacity-proxy 限制、dev-only、本地无 git、post-hoc）。
3. 论文表格与披露（含 cache/lifecycle 放大诊断：load-positive rate、evictions/dispatch、load+eviction time / GPU busy time）。
4. commit + push（09-26 → 09-27 工作）。

## Environment / Recovery

- **ChatGPT 绑定（权威）**：项目会话
  `https://chatgpt.com/g/g-p-6a45fd9c56e48191a5e1a006582fcdef-diao-du/c/6ab6b895-43b8-83ec-982a-4c13ac404115`
  （GPT-5.6 Sol + High）。旧会话 `6a9822da-…` 已达长度上限，**不要再用**。
- **本机（macOS）**：只能做控制面/代码/纯 stdlib 分析；**不能**跑 torch、predictor packer、GPU 实验。
  桥接 Chrome：`--remote-debugging-port=9222 --user-data-dir="$HOME/Library/Application Support/Google/Chrome-Automation"`（已登录）。
- **训练机（Windows）**：解释器 `D:\anaconda\envs\scheduler\python.exe`；仓库 `F:\scheduler`。
- 关键指针：计划 `.project/PLAN.md` 顶部；门禁 `.project/EXPERIMENT_GATE.json`（`stress_regime_v1_20260927`、
  `EXP-20260921_scheduler_replication_v1`、`EXP-20260921_histres_causal_input_v1`、`EXP-20260919_j_series_resource_dist_v1`）；
  决策 `.project/DECISIONS.md` 尾部两条 09-27 / 09-26 条目。
