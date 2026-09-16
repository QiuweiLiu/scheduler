# EXP-20260911_p9d_j4_duration_branch — RESULT

状态：**正式判定完成（validation 层）**；结论：**duration 解耦未修复时长端点，按预注册负结果规则拒绝该方向**。
设计：`docs/p9d_j4_duration_branch_design.md`（frozen v2，评审 ACCEPT）；门禁：`.project/EXPERIMENT_GATE.json`。

## 0. 运行事实

| 项 | 值 |
|---|---|
| 数据 | `results/processed/j_series_dataset_v1`（不变；行 13,754/2,029/1,520；哈希与注册表一致） |
| 参照 | J3/B1 = **继承的冻结参照（inherited frozen reference, not re-estimated）**，来自 `outputs/j_series_joint_resource` |
| 臂 | J4a（duration-decoupled head）、J4b（decoupled + frozen shared feature）；hidden=64；q(A)/lr/epochs/batch 均按冻结协议 |
| config sha256 | `b97879f8c451aecd924e011dffc1f31036cd358e1c88e3787aa4b77a6934aa72` |
| Stage 0 | rows/hashes 通过；**J3 回归检查：3 seeds 全部指标 max_abs_delta = 0.0**（代码扩展未改变 J 行为）；路由断言：J4a/J4b duration-only 梯度对共享隐性层 = 0，对 adapter 1.70/0.98 > 0；J3 对照 = 2.00 > 0 |
| 成本 | 6 runs × 30 epochs 合计 374.7s（≈6.2 GPU-min）+ stage0/smoke/test ≈ 8 GPU-min |
| 判定层级 | validation 为正式层；test = exploratory second look（`test_exploratory.json`，无 confirmatory 声明） |
| 运行产物 | `outputs/j4_duration_branch/`（run_manifest、stage0_audit、runs/*/run.json+checkpoint、test_exploratory.json、validation_ci.json）；`experiments/EXP-20260911_p9d_j4_duration_branch/`（config、run.sh、RESULT.md） |

## 1. 主结果（validation；duration 门 = LoadDurationQ ≤ B1 参考 ×1.05）

| 臂 | seed | 可行 epoch | 选中 | duration 最小值（epoch） | epoch29/30 duration | epoch30 runtime | 门限 | 状态 |
|---|---|---|---|---|---|---|---|---|
| J4a | 11 | 0 | — | 301.7 (9) | 321.6 / 317.9 | 741.4 | 298.0 | fail |
| J4a | 22 | 0 | — | 288.0 (8) | 317.3 / 327.0 | 767.9 | 279.8 | fail |
| J4a | 33 | 0 | — | 290.7 (8) | 324.1 / 320.2 | 797.6 | 270.3 | fail |
| J4b | 11 | 3（ep7–9） | ep9 | 288.2 (8) | 318.7 / 312.9 | 752.2 | 298.0 | ok（但 sel RuntimeQScore=1187.4） |
| J4b | 22 | 2（ep7–8） | ep8 | 270.8 (7) | 298.8 / 304.2 | 769.5 | 279.8 | ok（sel RuntimeQScore=1284.6） |
| J4b | 33 | 0 | — | 274.7 (7) | 308.9 / 304.0 | 798.5 | 270.3 | fail |

对照（各自选中 epoch，validation）：B1 283.8/266.4/257.5（runtime 912.7/894.5/957.8）；J3 286.6/276.7/260.0（runtime 814.4/818.5/870.2）。

## 2. 判定（按冻结规则；validation 层 CI 于 `validation_ci.json`）

**2a. 选择门与点数**（见 §1 表）：
- **H1（对 B1）**：J4a 无可用 checkpoint；J4b 的可行 checkpoint 出现在 epoch 7–9，此时 RuntimeQScore 1187/1285 ≫ B1（912/895）→ 点数级已不成立。
- **H2（对 J3）**：J4b 在可行 checkpoint 的 duration（288.7 / 279.4）与 J3 选中值（286.6 / 276.7）基本同级。

**2b. validation 层 bootstrap CI（配对 video bootstrap，B=1000，冻结 draws）**：

| seed | H1: Δruntime vs B1 | H1: ΔBrier | H1: Δduration vs B1 | H2: Δduration vs J3 | Δruntime vs J3 |
|---|---|---|---|---|---|
| 11 (ep9) | +274.7 [+241.5, +330.1] ✗（上界>0） | +0.0173 [+0.0148, +0.0191] ✗（>+0.005） | +4.9 [−7.7, +17.0] ✗（上界>14.2=5%·B1） | +2.1 [−15.8, +14.2] ✗（上界>0） | +373.0 [+335.0, +422.3] |
| 22 (ep8) | +390.1 [+340.5, +469.4] ✗ | +0.0389 [+0.0352, +0.0443] ✗ | +13.0 [−1.3, +27.5] ✗（上界>14.0） | +2.7 [−10.1, +15.0] ✗ | +466.1 [+409.6, +550.3] |
| 33 | 无 checkpoint（failed_no_feasible_epoch） | — | — | — | — |

- **H1 不成立**（runtime 显著更差、Brier 显著更差、duration CI 上界越界；所有 reliable 判定为真）；**H2 不成立**（duration vs J3 的 CI 上界 > 0，未显示修复）。
- **H3（机制）**：J4b（保留共享特征访问）在 duration 上清楚优于 J4a（完全独立分支）（最小 270.8–288.2 vs 288.0–301.7）⇒ **共享特征确实携带 duration 相关信息**；但两臂均未修好 duration ⇒ **"共享隐性层竞争是 duration 损失主因"的假设被否定**（移除竞争后 duration 没有变好，反而在同期更差）。
- 按预注册负结果规则：**duration 未修复 → 拒绝该分支方向**；不再继续拆 hidden。

## 3. 观察到的真实权衡（诊断性，不作声明）

- 同期对比（run.json history 的 epoch 30 条）：J4a/J4b 的 runtime 741–798 **优于** J3 的 814–898；但 duration 304–327 **差于** J3 的 260–291。
- 即：把 duration 从共享隐性层移出，会略微改善 runtime、但恶化 duration。共享隐性层的多任务塑形对 duration **有益**（与 J3 > J4 一致），此前"竞争导致退化"的解释不成立；J 系列的 duration 微降更可能来自 J3 表示-头共适应的整体偏移（D1'）而非可分离的竞争项。
- 探索性 test（仅 J4b 2 seeds，早期 checkpoint）**分 seed**：seed11 Δruntime vs B1 = +313.3、Δduration vs J3 = −14.6（CI [−34.6, −0.3]）；seed22 Δruntime = +442.0、Δduration vs J3 = +4.7（CI [−7.4, +14.4]）；两 seed 的 load Brier 均显著变差（+0.015 / +0.039，CI 不含 0）。方向不一致且以运行时代价巨大为背景——**不构成任何 confirmatory 结论**（均值 −4.9 仅为两 seed 平均，不应单独引用）。
- 注（选择规则不对称）：J4 的可行集在 J3 规则上多加了一层 duration 门，因此可行 checkpoint 是 J3 可行集的子集；早期 epoch 被选中（runtime 较差）是该更严格规则的直接后果，而非训练不稳定。

## 4. 边界与限制

- validation 层判定；test 已被 J 系列使用一次，本次仅 exploratory（治理决议，评审记录 `docs/research/2026-09-11_j4_design_review.md`）。
- hidden=64 与 30 epochs 为冻结配置，未调参；不排除更大分支/更长训练会改变结论（未做，不得事后调参宣称）。
- q(A) 未干预；J3/B1 参照未重估。
- **配置标注与实现的偏差（披露）**：J4 config 中 J4a/J4b 的 `attribute_gradient` 标为 `stopgrad`，但当前实现仅对 `B1`/`J2` 应用 detach/stopgrad；J4 臂实际执行的是与 `J3` 相同的全梯度接口（实现生效值 = full）。因此 J4 vs J3 的对比中**不存在属性头梯度路径的额外差异**；该标注字段对 J4 无效果，属配置与实现不一致（为保持已记录的 config sha256 不变，本次不修改 config；下一版本应使实现遵循该字段或移除之）。
- 选择规则不对称：J4 的第二层 duration 门使可行集为 J3 可行集的子集；早期 epoch 被选中是该规则的直接后果（见 §3 末注）。
- 判定产物：`validation_ci.json`（validation 层 H1/H2 CI）、`test_exploratory.json`（探索性 second look）、`stage0_audit.json`（回归检查 + 路由断言）、`runs/*/run.json`（逐 epoch 门与选择记录）。
- 无调度器集成；`S_*/T_final` 未动；J 系列产物未改动（回归检查 max_abs_delta=0 佐证）。
- 本结果已通过独立审计（`docs/research/2026-09-11_j4_result_audit.md`，PASS）与独立评审（`docs/research/2026-09-11_j4_result_review.md`，CONDITIONAL PASS → 4 项 P1 修正已并入本文件）。

## 5. 后续方向（供决策）

- **J4c（条件化增强）**：mechanism 分析预测 duration 需要 model-side/执行前上下文（workload scale、cold-start/residency proxy）——该方向尚未被本轮否定。
- **停止该线**：接受"deployable 场景下 duration 端点无法在 J 系列预算内修复"，回到已证实主线（属性接口 + R0 校准）。
- 不建议：继续拆 hidden、调分支容量/epochs、或多重性不成立下复用 test。
