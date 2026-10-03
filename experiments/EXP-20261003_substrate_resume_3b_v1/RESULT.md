# RESULT — EXP-20261003_substrate_resume_3b_v1（P1-5：3B resume 代表性格）

**状态**：完成（2026-10-03）；**运行前已冻结门禁**（见 `.project/EXPERIMENT_GATE.json`）。

## 目的

评审 P1-5：`request_preemption_profile_v1.json` 给 3B 发布了 recompute rates，但 P1 往返实验只测了
4B/8B 格，来源链不闭合。本实验补 **3B 代表性格**（k=20 与 k=40），并加一格 **4B 对照**用于分离
"模型差异"与"实例间 GPU 差异"。

## 协议（与 P1 相同，未改）

- 脚本：`p1_resume_roundtrip.py`（同协议）+ 3B 模型/系数扩展；reps=3、交替 full/interrupted、取中位数。
- 判据：重建段 `t_rebuild` vs F2 曲线预测 `f2_pred = intercept + rate×n_ctx`；
  F2-3B 系数取自 `EXP-20260929_substrate_phase_profile_v1`（10.2 ms + 75.29 µs/tok）。
- **运行前冻结的 gate**：每格 `|residual_ms| ≤ 100` 且 `|residual_pct| ≤ 10%`；
  失败则 3B 不发布 measured 覆盖（fail-closed），不放宽 gate。

## 环境（必须随文披露）

- **新实例 GPU 报告为 `NVIDIA GeForce RTX 4080`（非 SUPER）**；SM max 3105 MHz / MEM max 11201 MHz /
  32760 MiB（旧 SUPER 实例：MEM max 11501 MHz，其余同）。
- 同协议 4B 对照格在克隆实例上复现了 SUPER 的小残差（−1.8% vs SUPER 的 −4.3%），
  说明两实例对本协议性能等价（差异在共享卡噪声量级，F0 协议：中位数+交替）。
- 主环境：`/root/miniconda3/bin/python`（torch 2.8.0+cu128 / transformers 5.17.0，与旧实例一致）。
- 单实例单卡共享环境；L=1000 固定 prompt。

## 结果（每格 3 reps 中位数）

| cell | in_tokens | k | n_ctx | t_rebuild (ms) | F2 pred (ms) | residual | gate |
|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-3B k20 | 1019 | 20 | 1039 | **90.72** | 88.43 | **+2.29 (+2.6%)** | ✅ |
| Qwen2.5-VL-3B k40 | 1019 | 40 | 1059 | **91.27** | 89.93 | **+1.34 (+1.5%)** | ✅ |
| Qwen3-4B k20（对照） | 1008 | 20 | 1028 | **123.57** | 125.88 | **−2.31 (−1.8%)** | ✅ |

- 3B 重建代价被 F2-3B 曲线在 **3% 以内**预测；残差方向与 4B/8B 一致（编排开销量级）。
- 总时长差 ≈ t_rebuild（3B k20：91.0 vs 90.7；k40：94.8 vs 91.3）→ 与 4B/8B 相同的"恢复≈重建"结构。
- rebuild_runs 三次重复离散度 ≤0.1ms（3B）。

## 结论与影响

- **P1-5 关闭**：3B request-level preemption 的来源链闭合（F2-3B 率 + 3B 代表性格 + 4B 对照）。
  `request_preemption_profile_v1.json` 的 3B 覆盖不再是未验证外推。
- 披露：3B 格在克隆实例（RTX 4080）上测量；profile 的 gpu_identity 保持 SUPER（F2 来源），
  验证实例与等价性证据写入 profile 的 `probe_metadata`。

## 交付物

- `artifacts/p1_resume_3b.json`（结果）、`artifacts/p1_resume_3b.log`（运行日志）、
  `artifacts/p1_resume_3b.py`（本次使用的脚本快照）。
- 远端：`/root/autodl-tmp/scheduler_measurements/resume_roundtrip_3b/p1_resume.json`。
