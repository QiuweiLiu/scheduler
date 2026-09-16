# Forecast-aware Phase 2 审计记录（AUD-20260911-FAS-P2-01）

日期：2026-09-11。对象：`experiments/EXP-20260911_forecast_aware_scheduling/PHASE2_REPORT.md`。
原结论：AUDIT_VERDICT: inconsistent（因审计通道无法执行复算/解压的三项未验证）。

## 审计覆盖（PASS）
1. §1 聚合表 6 臂 × 5 指标与两个 matrix report 完全一致（逐值核对，含四舍五入）。
2. §2 配对 Δ 点估计与聚合差值一致（E2−E0 −17,064.7；E2−legacy −12,583.4；E3a−E2 +16,012.2；E3b−E3a −15.0；E4−E2 −25,318.8）。
3. §0 运行事实：J 5 臂 result_rows=5,000、B05 1 臂=1,000、failed_jobs=0、status=passed。
4. §5 产物路径存在。
5. §4 覆盖率 8,935/8,935 与 `b05_artifact_manifest.json` 的 template_coverage 一致。

## 未验证项 → 已由本地复算闭环
- **CI 复算**：`.scratch/phase2_stats_verify.py`（seed 20260911、B=2,000）独立重算全部 15 项 Δ+CI，与报告一致（worst |Δ|=0.037、worst |CI|=0.050，均在打印精度内）。
- **行数**：5,000 / 1,000（实测）。
- **manifest sha 闭环**：J 侧 `b05_artifact_manifest.json` sha256 = `0d0c9668b1b27819…`，与 J matrix report `future_artifacts_manifest_sha256` 一致；B05 侧 `aa943b60…`。
- **空步回退审计**（审计提出）：J artifacts 在 template 节点上 0 步比例 **14.2%**（1,272/8,935），B05 为 0%（固定 5 步 × 3 场景）→ 已作为关键混淆写入报告 §2b，并把 E2-vs-legacy 降级为"制品级"表述。
