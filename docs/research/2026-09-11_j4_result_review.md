# J4 结果独立评审记录（REV-20260911-J4-01）

日期：2026-09-11。Procedure：experiment-reviewer。对象：`exp4 RESULT.md` 与原始产物。
结论：**CONDITIONAL PASS → 4 项 P1，已全部修正**。

## 逐项结论
1. **协议执行 — PASS**：validation 唯一正式层、test 仅 exploratory（无 confirmatory 声明）、两层门实现（NI AND duration → argmin runtime）、duration 门基线钉死在冻结 B1（未重训）均落实；J4b/11 可行 ep7–9 选 ep9、J4b/22 可行 ep7–8 选 ep8 与记录一致。
2. **负结果推理 — CONDITIONAL**：断"竞争假设被否定、拒绝解耦方向"成立；但 H1/H2 未执行 validation CI（设计要求的 CI 规则），且存在未披露混淆；`【P1】` RESULT 补 §2b validation CI（B=1000 冻结 draws：H1 三项全 fail；H2 dur vs J3 CI 上界 +14.2/+15.0 > 0 fail）。`【P1】` 混淆披露：config `attribute_gradient=stopgrad` 标注 vs 实现实际为 J3 同款全梯度（字段对 J4 无效）——已在 §4 披露，并说明不存在属性头梯度路径差异；为保 config sha256 未改 config。
3. **公平性 — PASS（有条件）**：同数据/种子/预算；继承冻结参照正当并注明；未拿早期 checkpoint 的 duration 宣称修复；门实现与设计一致。`【P1】` 选择规则不对称（J4 可行集 ⊂ J3 可行集，早期选中是该规则的后果）——已在 §3 末注补披露。
4. **泄漏与实现 — PASS**：解耦真实（梯度断言 0.0 vs adapter >0）；无解释性 bug；J3 回归 3 seeds max_abs_delta=0.0（P2 局限：仅 7 指标点比对，未比对逐行分布）。
5. **结论边界 — CONDITIONAL**：§3 同期权衡有证据；"共享特征携带 duration 信息"仅在 J4b>J4a 意义上成立（合规）。`【P1】` 探索性 test 的 −4.9 均值掩盖异质性（seed11 −14.6 CI[−34.6,−0.3] vs seed22 +4.7 CI[−7.4,+14.4]，且两 seed Brier 显著变差）——已改分 seed 报告并注明均值不可单独引用；§3 epoch30 数据出处已补注（run.json history epoch 30）。
6. **后续方向 — PASS（有条件）**：J4c/停止由证据支撑；不建议继续拆 hidden/调参/复用 test 合规。

## 必改清单（已全部落实）
- [x] RESULT §2b 补 validation 层 CI（H1/H2 正式判定）。
- [x] RESULT §4 披露 attribute_gradient 标注与实现偏差（无梯度路径混淆；config 不改以保哈希）。
- [x] RESULT §3 探索性 test 分 seed 报告；移除单点均值表述。
- [x] RESULT §3 补选择规则不对称说明；epoch30 数据出处注记。

`REVIEW_VERDICT: modify`（不重跑 test；`test_exploratory.json` 保持原样）。
