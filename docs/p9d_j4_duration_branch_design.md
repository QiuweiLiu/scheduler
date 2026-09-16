# J4 设计：duration 解耦分支（frozen v2）

状态：**frozen v2**（2026-09-11；v1 评审 MODIFY → v2 修订 → 复核 ACCEPT）。评审记录：`docs/research/2026-09-11_j4_design_review.md`。
依据：J 正式结果（`RESULT.md`）与机制分析（`docs/research/2026-09-11_j_mechanism_analysis.md`）；协议继承 `docs/p9d_j_series_design.md`（frozen v3.1）。

## v1 → v2 变更（评审关闭表）

| 项 | v1 | v2 处理 |
|---|---|---|
| **P0-1 test 复用** | 方案 A：test 复用 + 97.5% CI | **改方案 B**：validation 为正式判定层（checkpoint/变体比较/机制判定）；**test 仅作 exploratory report 并明示 second look**；任何未来 confirmatory claim 需新 sealed 评估集（不在本实验）。理由：J4 由 test 失败驱动，属 post-selection，Bonferroni 不能恢复 test 的 confirmatory 身份 |
| **P0-2 checkpoint 选择** | 仅接口 NI-feasible → argmin runtime | **两层选择**：第一层保留 J 规则（5 接口端点 NI-feasible）；第二层新增 J4 duration 门：`LoadDurationQ_val ≤ LoadDurationQ_val(B1) + δ_dur`（δ_dur=5%·B1_val），再 argmin RuntimeQScore。属 J4 预注册 success definition，不改评价目标 |
| **P1-1 H2 runtime δ** | 自创 +2%（相对 J3） | 删除自创 δ。runtime 保全改用**既有规则**：`CI_upper(Δ_runtime vs B1) < 0`（与 J 主端点同规则）；J4 vs J3 的 runtime 差仅描述性报告 |
| **P1-2 分支解释** | 声称区分"梯度竞争" | 降级措辞：两臂差异 = **representation access + decoupling** 的消融，不直接声称证明梯度竞争；臂名改为 J4a=duration-decoupled head、J4b=duration-decoupled head + frozen shared feature |
| **P1-3 容量** | hidden=64 | 固定为 **hidden=64，全程不得调参**（避免 branch capacity 与 decoupling 混淆） |

### v2 复核补充（2 个非阻塞 P1，已落实）

- H3 表述约束：`J4b > J4a` 只说明 **shared feature access 提供增量信息**，不等价于证明 J3 的梯度冲突是唯一原因（两臂同时改变输入 representation、参数隔离与 duration 头容量；属机制支持证据，非严格因果证明）。
- duration gate 基线钉死：第二层门的 `LoadDurationQ_val(B1)` **必须取自 J 系列冻结的 B1 validation 预测**（不重训 B1；实现时从冻结 run 的 validation 预测重算或复用其数值），否则门会漂移。

## 0. 目标与非目标

- 目标：修复 J 系列唯一门禁失败点（load-duration 5% 非劣），同时**保持 runtime 对 B1 的既有优势**；以 validation 为正式判定层检验"duration 解耦容量"是否消除共适应代价。
- 非目标：不做 confirmatory test 声明；不改数据/backbone/属性接口/NI manifest 的 5 个端点与 δ；不改 bootstrap；不调 q(A) 温度或加熵正则（D2 只作机制解释）；不接调度器；不动 holdout。

## 1. 机制依据

- D1'（头交换）：原配仅差 1–4%，交叉差 +17%（B1 头↔J 表示）与 +80%~+170%（J 头↔B1 表示）⇒ 表示-头共适应，共享资源隐性层是嫌疑点。
- D3：log/raw 尺度不是主因。D2：接口变硬是伴随现象，不干预。
- treatment = 把 duration 输出路径从共享资源隐性层解耦。

## 2. 架构与变体

共享部分与 J3 完全一致：encoder、属性头 q(A)、structure/behavior 头、资源共享隐性层（仅服务 runtime/occurrence）。

| 臂 | duration 路径 | 资源梯度路径 |
|---|---|---|
| **J4a** duration-decoupled head | 独立参数分支：输入 `[repr, slot_emb, q(A)]` → Linear(→64) → tanh → Linear(→3τ) | 不经过共享资源层 |
| **J4b** decoupled + frozen shared feature | 独立参数分支：输入 `[sg(shared_hidden), slot_emb, q(A)]` → 同结构（hidden=64） | 共享层只收 runtime/occ 梯度 |
| 对照 J3（继承冻结参照，不重估） | duration 与 runtime/occ 共用 hidden | — |
| 对照 B1（继承冻结参照，不重估） | 冻结 backbone + 资源头 | — |

- 差分语义（降级表述）：J4a vs J4b 反映"共享特征访问是否有用"；J4b vs J3 反映解耦本身的效果；二者合起来是 decoupling ablation。
- 其余协议与 J frozen 完全一致：等权 L_total、30 epochs、batch 256、lr 3e-4、wd 1e-4、seeds 11/22/33、q(A) 软分布不调温、hidden=64 冻结。

## 3. 假设与判据（validation 为判定层；test 仅 exploratory）

- **H1 Core（对 B1）**：`CI_upper(Δ_runtime)<0` 且 load 双端点非劣（Brier `CI_upper<+0.005`；duration `CI_upper<+5%·B1`）→ **validation-level Core GO**（明确标注低于 J 系列 test 判定级别）。
- **H2 修复（对 J3）**：`CI_upper(Δ_load_duration)<0`；runtime 保全用既有规则 `CI_upper(Δ_runtime vs B1)<0`（不引入新 δ）。
- **H3 机制**：J4b>J4a ⇒ 共享特征访问有价值；J4a≈J4b 且均修复 ⇒ 独立容量足够；均不修复 ⇒ 共适应不在输出路径，转 J4c（条件化增强）。
- **负结果预冻结**：duration 未修复 → 拒绝该分支并结束该方向；duration 修复但 runtime 保全失败 → 记录 trade-off 不宣称 GO；J4a/J4b 无差异 → 停止该方向。

## 4. 数据、泄漏与公平性

- 数据 `j_series_dataset_v1` 不变；上下文 c 仅执行前字段；nested N/A mask 不变；holdout 不参与。
- 代码：扩展 `scripts/j_series_common.py` / `scripts/j_series_train_eval.py` 增加 J4a/J4b 分支；J 变体行为必须不变（用已有 J3 checkpoint 在 validation 上做改动前后一致性回归检查，逐 metric 比对）。
- 新实验目录与 config；J 实验产物不改；正式臂统一 device/batch；不跨臂调参。

## 5. 判定层级与声明边界（P0-1 落实）

- **正式层（validation）**：checkpoint 选择、J4a/J4b 比较、H1/H2/H3 判定。
- **探索层（test）**：仅作 exploratory report（明确标注 "second look on previously used P_dev/test"），不产生 Core GO 声明。
- 未来如需 confirmatory claim：必须采集/封存新的评估集（记入 next steps，不在本实验）。

## 6. 执行与治理

- 顺序：Stage 0（含新变体路由断言 + J3 回归检查）→ smoke（J4a/J4b 各 2 epochs 子集）→ J4a/J4b × 3 seeds × 30 epochs → validation 判定 → test exploratory 报告。
- 成本 ≈15 GPU-min（RTX 3060 本地）；无新数据/远端；`S_*/T_final` 不动。
- 新增门禁登记与结果文档；评审记录与结论一并落盘。

## 7. 变更历史

- v1（2026-09-11）：初稿（两臂 + 方案 A + 自创 δ）。
- v1 评审：MODIFY（P0 test 身份、P0 selection 门、P1 δ/命名/容量）→ v2 修正。
