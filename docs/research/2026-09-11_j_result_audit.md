# J 系列结果审计记录（AUD-20260911-J-TEST-01）

日期：2026-09-11。审计对象：`experiments/EXP-20260911_p9d_j_series_joint_resource/RESULT.md` 与原始产物一致性。
审计答复（原文要点）与后续闭环：

## Audited findings

1. 主表 12 个 runtime Δ+CI（J0/J1/J2/J3 × 3 seeds）— **PASS（12/12 精确一致）**：J0 +115.093[87.215,137.095]/+119.369[94.997,138.590]/+147.441[125.126,167.297]；J1 +26.401[3.955,50.438]/+19.887[0.196,36.772]/+73.764[53.333,95.295]；J2 −48.995[−69.983,−25.908]/−65.387[−79.615,−52.875]/−29.882[−45.481,−16.301]；J3 −78.125[−98.600,−54.116]/−77.093[−95.914,−60.386]/−63.965[−80.108,−49.179]；均值 +127.301/+40.017/−48.088/−73.061、B1 点 914.743/900.892/923.862 一致；per_variant 全 NO-GO。
2. Load Brier/duration 列 — **FAIL（口径不一致，已修正）**：duration CI 上界全部一致（J2 仅 seed22 `duration_ok=1`；阈值 11.19/10.20/10.29ms 一致）；但 RESULT 的 Brier 列混用了 delta_mean 与不同 seed 的 CI 上界（J0 列实为 delta_mean；J2/J3 用了非最差 seed）。gate 判定本身正确。**修正：§2 表统一为 worst-seed CI 上界（J0 0.0141/0.0117/0.0133；J1 ≤+0.0034；J2 ≤−0.0035；J3 ≤−0.0036）。**
3. selected_epoch/status — **PASS**：15 条 variant + 3 条 backbone 全 ok；test_eval 与 run.json 逐条一致。
4. 数据集计数与 config sha — **PASS**：13,754/2,029/1,520 行、48,343/7,195/5,387 槽位；config sha `a201c9ff…` 全库一致。
5. run_manifest vs config — **PASS**（batch 256、cuda、seeds [11,22,33]、B=1000 seed=20260911）。
6. test 仅执行一次 — **PASS**：无第二份 test 产物；脚本对已存在 test_eval.json 默认拒绝覆盖；时间线（backbone 16:47 → variants 16:53 → test 17:16）一致。
7. 梯度探针 — **PASS（数值）**：9 条范围 g_R 4.779–5.068、g_S 1.328–2.084、g_B 1.163–1.781、g_C 0.291–0.414、cos(g_R,g_B) −0.193~−0.097、cos(g_R,g_C) −0.008~0.122；口径（epoch1/batch64）无法从产物验证，已在 RESULT §4 注明实现口径。
8. artifacts checkpoints — **闭环通过**：`scripts/.scratch/verify_j_hashes.py` 复核 12 个变体 checkpoint 源=复制=test_eval `model_sha256`（0 mismatch）；另单独复核 B1×3 复制一致（共 15/15）；config 实际 sha256=`a201c9ff…` 与记录一致。

## 次要发现（已处理）

- 变体 run.json duration 合计 1371.5s（非 1393s wall）；RESULT 已改为 run.json 合计 + wall 标注。
- `.project/EXPERIMENT_GATE.json` / `HANDOFF.md` 状态滞后 → 已更新。
- 任务包写 15 个 run.json，实际 18 个（含 backbone×3），已全部检查。

审计原文结论：`AUDIT_VERDICT: inconsistent`（仅因第 2 项统计口径标签与第 8 项未闭环）；两项均已在 RESULT/产物中修正或闭环，无计算级错误。
