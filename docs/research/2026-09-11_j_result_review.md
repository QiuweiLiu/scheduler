# J 系列结果独立评审记录（REV-20260911-J-TEST-01）

日期：2026-09-11。Procedure：experiment-reviewer。对象：`experiments/EXP-20260911_p9d_j_series_joint_resource/RESULT.md` + 原始判定。
评审结论：`REVIEW_VERDICT: modify`（无 P0；1 个 P1 为治理/披露层，其余为报告口径 P2）。

## 逐项结论（7 项）

1. Selection optimism — PASS（1 P2）：test 仅由 `stage_test` 执行且拒绝覆盖；时间线（backbone 16:47:53 → variants 16:53:21 → test 17:16:59）单次；run.json 无 test 指标。P2：backbone 选择准则未在设计逐字写死（与 J 变体准则不同，但仅为共享初始化）。
2. Bootstrap — PASS：video-cluster、paired 同 mask（NaN 逐行成对丢弃）、固定 indices 全程复用、valid_ratio<95% fail-closed、未把 seed 当样本；全部 endpoint valid_ratio=1.0。
3. NI 方向规则 — PASS：5 接口端点方向/δ 与设计一致；lower→CI_upper<+δ、higher→CI_lower>−δ 且要求 reliable；load 阈值 = 0.05×同 seed B1 参考均值（分母设计未指定，已标注）。
4. 泄漏检查 — PASS（2 P2）：c 仅执行前字段且 v3 白名单；`prefix_model_reuse` 在 J 代码中不存在；nested N/A mask 正确；资源真值仅在 `targets`；复合键 join fail-closed。P2：设计文本"仅 stack/baseline"与实现消费 task 元数据不符（无泄漏，文档需改准）；merged 槽位 nested 缺失会被编码为 UNK 而非 N/A（契约上不应出现，缺断言）。
5. 结论边界 — CONDITIONAL（P1-1 + P2 若干）：证据支持 J2/J3 runtime 3/3 seed CI 上界<0、J1/J0 显著更差；"无 Core GO"在 all-seed/worst-seed 读法下成立。缺陷：官方判定使用了未预注册的 ≥2/3 跨 seed 规则；J2 seed22 通过全部逐 seed 冻结判据未披露；Brier 表统计量混用；"5.2%–7.9%" 不可复现；接口 Δ 上界写错；duration 局部化表述无产物支持。
6. 公平性/一致性 — PASS：18/18 run device=cuda、batch=256、fp32、单脚本；非 backbone 变体 init 均为同 seed backbone 状态；J0/B1 仅资源头可训练、J1–J3 全参数；J0/B1 接口 Δ 严格=0 已逐 endpoint 验证；B1 实现=冻结 backbone+sg(q(A))，与设计一致。
7. 负结果/边际解释 — CONDITIONAL：RESULT 明确禁止事后调容差；但"边际"表述对 J2 seed11（2.4× 阈值）与 J3 三 seed 不准确；J2 seed22 单独通过应明示。

## Findings

- **P1-1（唯一需修改项，已处理）**：≥2/3 聚合规则未预注册，且 J2 seed22 通过全部逐 seed 判据未披露。修正：RESULT §0 登记官方聚合规则（all-seed/worst-seed），§1 注明确披露 J2 seed22（duration CI 上界 9.942 < 10.195，差 2.5%）及"不得作为 NI/Core GO 证据"；test_eval.json 的 ≥2/3 判据标注为事后次级统计。**不重跑 test，test_eval.json 保持冻结。**
- P2-1（已修正）：Brier 列统一 worst-seed CI 上界（J0 0.0141/0.0117/0.0133；J1 ≤+0.0034；J2 ≤−0.0035；J3 ≤−0.0036）。
- P2-2（已修正）："5.2%–7.9%" → 逐 seed 匹配基线 J2 5.36%/7.26%/3.23%、J3 8.54%/8.56%/6.92%。
- P2-3（已修正）：接口 Δ 上界 → content ≤ +0.0022、next-role ≤ +0.0079、next-family ≤ +0.0109。
- P2-4（已修正）：删除"duration 恶化集中在少数长 duration 槽位"的未验证表述。
- P2-5（已修正）：改为"J3 的 NI FAIL ⇒ 不能声称端到端接口学习有效（负迁移条款）；未做 J2-vs-J3 配对检验"。
- P2-6（已修正）：R0 表述改为"方向/机制对齐，幅度不可比（R0 为 oracle 上界 74.9%/82.1%）"。
- P2-7（已记录）：设计文本 c 白名单措辞与实现差异，待设计文档小修（不影响结果）。
- P2-8（已记录）：backbone 选择准则未冻结；nested missing 缺 Stage 0 断言；OracleAttr 未运行（R0 承担其角色）——已在 RESULT §6 注明。
- P2-9（已修正）：run.sh 补 `--stage`；RESULT 记录正式运行命令。
- P2-10（已修正）：EXPERIMENT_GATE/HANDOFF/INDEX 更新；RESULT 补 duration 覆盖（n_rows=474，31.2%）与 valid_ratio。

## Required next check（评审原文）

1. 预注册跨 seed 判定规则并重算官方 verdict ✅（RESULT §0/§1）。
2. 显式披露 J2 seed22 ✅。
3. 修订 RESULT 数字与措辞 ✅。
4. 记录正式命令并修复 run.sh ✅。
5. 更新控制面；补 duration 覆盖；注明 OracleAttr 未跑 ✅。
6. 不重跑 test，保持 test_eval.json 冻结 ✅。
