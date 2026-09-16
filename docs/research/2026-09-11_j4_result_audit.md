# J4 结果审计记录（AUD-20260911-J4-01）

日期：2026-09-11。对象：`experiments/EXP-20260911_p9d_j4_duration_branch/RESULT.md` 与原始产物一致性。
结论：**AUDIT_VERDICT: consistent（7/7 PASS）**。

## Findings（全部 PASS）
1. §1 表 6 行 duration 最小值/epoch、ep29/30 duration、ep30 runtime、门限、可行 epoch 数与 run.json history 完全一致（抽查含逐行出处；四舍五入均为常规 1 位）。
2. J4b 选中值（seed11 ep9 / 1187.4 / 288.7；seed22 ep8 / 1284.6 / 279.4）与 selected_record 一致。
3. §2 H1/H2 输入（B1 选中 rt 912.7/894.5/957.8、dur 283.8/266.4/257.5；J3 选中 rt 814.4/818.5/870.2、dur 286.6/276.7/260.0）与冻结 J 运行一致；J3 seed33 选中 epoch 29 的细节 RESULT 未声明 epoch 号，不构成不一致。
4. §3 探索性 test 数字（J4b mean Δruntime +377.63、mean Δdur vs J3 −4.93、seeds [11,22]；J4a 无条目）与 test_exploratory.json 一致。
5. §0 config sha256 `b97879f8…`、environment/device/batch、6 runs 合计 374.7s 与 run_manifest/config 一致。
6. Stage 0 的 j3_regression_check（3 seeds max_abs_delta=0.0）与 routing（J4a/J4b shared-hidden duration grad=0.0、adapter 1.70/0.98；J3 对照 2.00）与 stage0_audit.json 一致。
7. test_exploratory.json 带 exploratory 标注（top-level status + per-variant note），无独立 confirmatory verdict 字段；RESULT 未将其当作 confirmatory 使用。

## Coverage gaps（不影响结论）
- 未重算 sha256/行哈希/bootstrap 均值（只读解析，依赖文件内记录）。
- RESULT 中“hidden=64”指 duration_branch_hidden；模型主 hidden 仍为 128（语义歧义，已在后续版本措辞中注意）。
- 未全量重排中段 epoch 最小值（用 grep 抽查确认唯一）。
