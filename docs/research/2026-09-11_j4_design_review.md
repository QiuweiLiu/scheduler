# J4 设计 v1 评审记录（ChatGPT Web，2026-09-11）

会话：bridge `6a9822da-e278-83e9-9c1a-675923acda0e`。结论：**MODIFY**（方向认可，2 P0 + 3 P1；关闭后可执行）。
用户裁决要点（原文精简）：

## P0
1. **test 复用方案**：不接受方案 A（复用 test + Bonferroni 97.5% CI）。理由：J4 的方向由 J 系列 test 失败驱动，属 post-selection bias，alpha 调整不能恢复 test 的 confirmatory 身份。推荐方案 C（最严格：只报告机制修复）；方案 B 可接受（validation 做 checkpoint/比较/机制判定，test 仅 exploratory second look；未来 confirmatory claim 需新 sealed 评估集）。
2. **checkpoint 选择**：在 J 规则（5 接口端点 NI-feasible）之上新增 J4-specific duration validation gate：`LoadDurationQ_val ≤ B1 + δ`，再 argmin RuntimeQScore。属预注册 success definition，不改评价目标。

## P1
1. H2 自创 δ=+2%（对 J3）无统计依据；应沿用既有规则（runtime 保全用 `CI_upper(Δ_runtime vs B1)<0`；J4 vs J3 仅描述性）。
2. J4a/J4b 差异不只梯度流，还包括 representation access；措辞降级为 decoupling ablation，勿声称证明梯度竞争。
3. branch capacity（hidden=64）固定，不得后续调参。
4. J3 冻结结果作为参照可接受，但报告必须注明 "inherited frozen reference, not re-estimated"。
5. q(A) 变硬不干预（不要同时引入熵正则/温度调整）；3 seeds 足够；负结果规则提前冻结。

## 落地
以上全部并入 `docs/p9d_j4_duration_branch_design.md` v2。

## v2 复核（同日，第二次评审）

结论：**ACCEPT（可冻结执行）**，无 P0；两个非阻塞措辞建议（已落实）：
1. H3 的 `J4b>J4a` 只说明 shared feature access 提供增量信息，不是"梯度冲突是唯一原因"的证明（两臂同时改变输入表示、参数隔离与头容量）。
2. duration gate 的 B1 基线必须钉死在 **J 系列冻结的 B1 validation 预测**上，不得重训 B1（否则门漂移）。
复核并确认：test 身份（方案 B）、两层选择门、H2 改用既有规则、臂命名与解释降级、hidden=64 冻结、不干预 q(A)、3 seeds、30 epochs、负结果规则预冻结，均无阻塞。
最终：**J4 DESIGN v2 ACCEPT（可冻结执行）**。后续结果解释预置：修复 duration + runtime 保持 → 支持 duration decoupling 改善资源多任务接口；修复 duration 但 runtime 下降 → 记录 Pareto trade-off；未修复 → 问题不在 shared duration capacity，应转向 workload/model-side context 方向而非继续拆 hidden。
