# R0 RESULT — oracle-signature empirical resource ceiling

实验：`EXP-20260911_p9d_r0_oracle_signature_ceiling`
设计：`docs/p9d_r0_oracle_signature_ceiling_design.md`（frozen v3；网页评审 v1→final ACCEPT）
数据集：`topology_predictor_p9d_v3`（契约 v3.1，`data/manifests/topology_predictor_p9d_v3.json`）
运行：本地 CPU（Windows 挂载 `F:` = `/Volumes/Lenovo`），LightGBM 4.7.0 / numpy 2.5.3 / scipy 1.18.1；全量 36.57 s。
配置：`config.json`（sha256 `cca9874f…`）；完整哈希见 `run_manifest.json`。

## 1. 裁决摘要（按冻结 design §7）

| Gate | 结果 | 数值 |
|---|---|---|
| Core GO：runtime 改善 ≥15% 且 CI 下界 >0 | **PASS** | validation 改善 **74.9%**，bootstrap CI [0.704, 0.824]；test 82.1% [0.793, 0.842]；holdout 20.2% [0.099, 0.815] |
| Core GO：校准 `max_τ|C_τ−τ| ≤ 0.05` | **PASS** | validation 0.0338、test 0.0461、holdout 0.0161 |
| Core GO：memory 可解释（否则 descriptive-only） | **FAIL（按契约降级）** | validation 校准 0.127 > 0.10；memory 转为 descriptive，Core GO 用 runtime + load |
| load（次要目标） | **PASS** | occurrence Brier：val 0.0175 / test 0.0033 / holdout 0.0013（logistic 参考 0.1055/0.1036/0.0963）；duration（正样本）PB：val 41.7 ms / test 45.5 ms / holdout 305.1 ms |
| tail 无灾难性失败 | **附条件** | Inclusive 下 holdout 有 2 个极端事件（planner 2,694,757 ms、image-qa 4,073,736 ms）主导误差；Normal 子集（排除 stall）holdout PB=319.8 ms |

**结论：当前接口在 runtime 上具有可利用的经验可辨识性（Core GO PASS）；memory 无法在本池校准（descriptive-only）；长尾（cold-start/stall）不可由现有执行前上下文辨识。**

## 2. Stage 0 与执行门禁

- Stage 0 六项断言全部通过（`stage0_audit.json`）：节点数 15,481（planner 7,211 / tool 7,078 / post-loop gen 1,192）；load 计数逐类互斥且合计 = 类总数（planner 67/5,960/1,184、tool 0/6,742/336、gen 9/530/653）；memory 公式断言通过；feature role 白名单无 target 泄漏；回退层级固定；stall/tail 阈值仅由 train 构造（tail q90 = 11,688 ms）。
- smoke：80 run / 932 节点，9.0 s，链路通过（`metrics_smoke.json`）。
- 正式：15,481 节点，36.6 s；逐节点预测产物 4 份（train/validation/test/holdout）。

## 3. Runtime 主结果（目标 = runtime_ms；PB_primary = mean(PB0.50, PB0.90, PB0.95)，raw scale）

| split | R0 模型 PB | Core baseline（条件中位数） | 改善 | 次级诊断基线（条件经验分位） | 校准 max_τ |
|---|---:|---:|---:|---:|---:|
| train | 304.9 | 1424.5 | 78.6% | 799.7 | 0.0117 |
| validation | 380.0 | 1517.0 | 74.9% | 809.4 | 0.0338 |
| test | 261.1 | 1455.6 | 82.1% | 795.5 | 0.0461 |
| holdout | 5264.0 | 6597.9 | 20.2% | 5785.2 | 0.0161 |

- 三个 seed 的 worst-seed PB 与 mean-seed 完全一致（确定性训练），见 `metrics.json`。
- 相对更强的条件经验分位基线，validation 改善仍为 53.0%（380.0 vs 809.4），说明改善不只来自“中位数基线不会给上分位”。

## 4. 消融（机制诊断，不 veto）与 exact/coarse

| 级别 | validation PB | test PB | 增量解释 |
|---|---:|---:|---|
| C0（exec_class） | 941.4 | 885.9 | 类别本身解释大部分方差 |
| C1（+role/planner_mode/is_retry） | 809.2 | 792.4 | 小幅改善（role 与 exec_class 冗余；planner_mode/is_retry 有信息） |
| C2（+model/nested） | 809.6 | 792.5 | 与 C1 持平 |
| C3（+workload_scale） | 809.6 | 792.5 | **与 C2 持平** |
| C4（+baseline/prefix_model_reuse） | **380.0** | **261.1** | 主要增益来源 |

- `exact` vs `coarse` model_class：validation 均为 380.0（本池 model_class 值少，粗化不改变划分）——结论限定 known-model ceiling。

## 5. 关键发现

1. **`workload_scale` 在本池零方差**（Stage 0 警告 + 全量扫描）：`clip_len` 0/15,481 非空；`query_char_len`、`nested_api_call_count` 全部为 0（原始 trace 不含 `input.parameters`）。C3 与 C2 完全持平即其直接后果。这是**接口缺口**而非模型缺陷。
2. **`prefix_model_reuse` + baseline 是 runtime 可辨识性的主增益**（C3→C4：809.6→380.0）。加载/驻留相关的执行前上下文对 planner/工具资源估计贡献显著。
   - **口径说明**：`prefix_model_reuse` 的构造需要知道节点自身的 `model_class`（属 oracle 签名 Z 的一部分），因此该增益是"oracle 条件下"的；它不是纯执行前可见量。deployable J 必须将其移除或改为由预测分布派生的 soft reuse（J 设计 v3.1 已修订）。
3. **load 可辨识性好于预期**：occurrence（是否发生加载）在 holdout 上 Brier 0.0013，远优于固定 logistic（0.096）；`load_ms=0` 合法 no-load 的 hurdle 契约成立。
4. **memory 校准不达标**：train/validation/holdout 的 max_τ 分别为 0.104/0.127/0.373 → 按契约降级为 descriptive-only；memory 覆盖仅 9,816/15,481（planner、generate、image-qa、image-grid-qa、yolo 有值，其余工具无测量）。
5. **长尾由 2 个 stall 主导**：holdout 2 个事件（planner 45 min、image-qa 68 min）把 Inclusive PB 从 Normal 319.8 ms 拉到 5,264 ms；这指向“缺少 cold/startup/stall 的执行前系统上下文”，与四象限诊断 ②一致。

## 6. 四象限定位与后续建议

- 定位：**quadrant ②（Normal 良好、Inclusive 被 stall 拉低）+ 接口缺 workload 规模信息**。
- 建议（不自动执行）：
  1. **J 系列可在限定范围内启动**：以 runtime 为主、load 为辅；在 RESULT/论文中声明 memory 与 tail 不在本期可辨识范围。
  2. **接口扩展候选（下一轮 pre-registration）**：token/字符的真实长度（现为查询文本长度代理）、实际帧数/分辨率、cold-start/驻留状态（`model_resident_before` 在 P9d 池 100% 缺失，需要新的采集字段或 worker 前缀状态）、以及批处理/并发上下文。
  3. **holdout 一次性验收已使用**：按契约，若之后修改 contract/模型/特征，该 holdout 失去 confirmatory 身份。

## 7. Claims 边界（不得外推）

- 允许：在上述冻结契约、oracle 签名、已观测描述符与执行前上下文的条件下，`runtime`（以及 `load` occurrence/duration）在本池主要节点类别上具有经验可辨识性。
- 禁止：resource prediction solved；future resources predictable；topology predictor can predict the resource signature；unknown-model generalization；memory 全量可辨识；长尾冷启动已解决。

## 8. 复现

```sh
# Stage 0（六项断言）
python scripts/r0_oracle_signature_ceiling.py --config experiments/EXP-20260911_p9d_r0_oracle_signature_ceiling/config.json --stage stage0
# smoke
python scripts/r0_oracle_signature_ceiling.py --config experiments/EXP-20260911_p9d_r0_oracle_signature_ceiling/config.json --smoke
# 正式
python scripts/r0_oracle_signature_ceiling.py --config experiments/EXP-20260911_p9d_r0_oracle_signature_ceiling/config.json
```

产物：`metrics.json`、`stage0_audit.json`、`run_manifest.json`、`artifacts/node_table.jsonl.gz`、`artifacts/predictions_{split}.jsonl.gz`（哈希见 `run_manifest.json`）。
