# P9d J 系列设计：topology × resource 联合训练（frozen v3）

状态：**frozen（2026-09-11）**。经历：v1 设计 → 网页评审（4 个 P0）→ v2 修订 → 复核（3 个 P0：CI 方向两处、成功判定流程一处）→ v3 修正并冻结。
关联：`docs/p9d_topology_label_contract_v3.md`（标签契约）、`docs/p9d_r0_oracle_signature_ceiling_design.md`（R0 上界）、评审记录 `docs/research/2026-09-11_j_series_design_review{,_v2}.md`。

## 0. 目标与非目标

- 目标：检验**资源监督能否通过共享表示 / 软条件化改善可部署的未来资源估计**（runtime 主、load 次；memory/tail 不作可辨识声明）。
- 非目标：不预测聚合 future cost；不接调度器；不重采数据；不读 `S_*/T_final`；holdout 冻结前不可见且不进正式决策。

## 1. 数据与标签

- 样本 = P9d 行为锚点；视频级 split：P_dev train/validation/test；`P_holdout_diag` 完全退出正式决策（可选 exploratory，默认不报）。
- 未来 H=5 链节点属性标签（v3.1）：`exec_class`、`role`、`action_family`、`model_class`、`merged_nested_call`、`is_retry`、`nested_model_class`（含 N/A mask）。
- 资源目标按 node_id 关联同 trace 实测值：`runtime_ms`（primary）、`load_ms`（hurdle：occurrence + 正样本 duration）；memory 仅描述。
- 执行前上下文 c 白名单（**v3.1 修订：移除 `prefix_model_reuse`**）：仅 `stack/baseline`。原因：`prefix_model_reuse` 需要知道节点自身的 `model_class`，属 future-label 派生；R0 作为 oracle 签名用途合法，但 deployable J 必须删除（或改为由预测分布 `q(model)` 派生的 soft reuse，列入后续版本）。禁止任何执行后/未来信息。

## 2. Backbone 与变体（评审 P0-1 修订）

Backbone（attribute-only，无资源监督）：共享 causal GRU + structure 头（future length/termination）+ content 头（多属性，N/A mask）+ next-step 行为头；训练后冻结，作为 B1 的预测器与全部 J 变体的**共同初始化**。

**structure target（v3.1 修订）**：`bounded_future_length = min(剩余 compute 节点数, 5) ∈ {0,1,2,3,4,5}`（含 0：当前 anchor 后可能只剩 terminal marker，此时长度为 0）；`termination = 1[在 H 内到达 answer]`。censored 样本的 `L_H=5` 是窗口内精确长度（不是"最终总长度=5"），因此所有样本都保留在 length CE 中，不视为 missing。length 头 = 6 类 CE；NI 的 length metric = `MAE(argmax L_H, L_H_GT)`。

| 变体 | Encoder | 属性分布 → 资源头 | 资源梯度路径 |
|---|---|---|---|
| J0 probe | 冻结 | 不使用 | 仅资源头 |
| **B1（deployable 两段式）** | 冻结 | `sg(q(A))` | 仅资源头 |
| J1 | 可训练 | 不使用 | → encoder |
| J2 | 可训练 | `sg(q(A))` | → encoder，不进属性头 |
| J3 | 可训练 | `q(A)`（无 stopgrad） | → encoder + 属性头 |
| OracleAttr（仅诊断） | 冻结 | GT 属性 | 仅资源头 |

差分：J0→J1 表示适配；J1→J2 predicted-interface 信息；B1→J2 同软接口下的联合表示适配；J2→J3 资源梯度重塑语义接口。`q(A)` 用部署同套 soft 分布（**T=1 固定**，不做温度调参）。

## 3. 属性头清单（接口一致性）

所有正式变体共享完全一致的 heads/labels/loss：`exec_class`(CE)、`role`(CE)、`action_family`(CE)、`model_class`(CE)、`merged_nested_call`(BCE)、`is_retry`(BCE)、`nested_model_class`(CE+N/A mask)。B1 的冻结预测器即 backbone；不得出现 J 变体有某 head 而 B1 没有。

## 4. Loss（写死）

```
L_total = 1.0·LS + 1.0·LC + 1.0·LB + 1.0·LR
LS = future length (MAE) + termination (BCE)
LC = 属性头 CE/BCE 的均值（按有效槽位归一化）
LB = next-role CE + next-family CE
LR = mean( runtime-quantile pinball(log1p, τ=.50/.90/.95),
           load-occurrence BCE,
           load-duration pinball(log1p, 仅 load>0) )
```

组内先按有效节点/字段归一化，再固定等权；运行前不得改权重；第一轮不用 GradNorm/PCGrad，仅记录 `||g_R||/||g_C||/||g_B||` 与 cosine（供 J4 决策）。

## 5. 指标与成功判据（评审 P0-2/P0-3 修订）

- Primary：`RuntimeQScore = mean(PB.50, PB.90, PB.95)`（raw scale，与 R0 一致）。
- **判定数据流（评审修订）**：`train` 拟合 → `validation` 仅做 checkpoint/配置选择 → **`P_dev/test` 冻结后一次性判定 Core success 与 load NI**（防止 selection optimism；validation 不得再作为成功证据）。
- Core success（在 test 上）：`Δ_rt = RuntimeQScore_Jx − RuntimeQScore_B1` 的配对 95% source-video bootstrap **上界 < 0**（loss 越低越好 → 要求 Jx 不高于 B1）。
- Load 非劣（在 test 上，均为 loss 越低越好）：`CI_upper(Δ_Brier) < +0.005`；`CI_upper(Δ_LoadDurationQ) < +5%`（相对；工程容差）。
- 三个 matched seeds（11/22/33）匹配初始化/数据顺序；报 mean + worst seed；方向明显不一致再扩 seed。

## 6. NI manifest 与 checkpoint 选择（评审 P0-4 修订）

每个 endpoint 明确 **metric + direction + δ**，禁止统一方向规则：

| endpoint | metric | 方向 | δ | 检查（paired video bootstrap） |
|---|---|---|---|---|
| future length | MAE | 越低越好 | 0.02 | `CI_upper(Δ) < +0.02` |
| termination | BCE | 越低越好 | 0.02 | `CI_upper(Δ) < +0.02` |
| future content | accuracy | 越高越好 | 0.02 | `CI_lower(Δ) > −0.02` |
| next-role | accuracy | 越高越好 | 0.01 | `CI_lower(Δ) > −0.01` |
| next-family | accuracy | 越高越好 | 0.02 | `CI_lower(Δ) > −0.02` |

- Checkpoint 流程：validation 上先算 NI（上表），只在 **NI-feasible** 集合内按 `argmin RuntimeQScore` 选择；某变体无 feasible checkpoint → 该变体 fail。
- **训练预算（v3.1 修订）**：取消 early stopping/patience；所有正式变体固定训练 30 epochs，结束后从全部 epoch 中做 NI-feasible → argmin RuntimeQScore 选择（避免 patience 与 NI 可行时间的交互）。
- **bootstrap 固定（v3.1 修订）**：正式运行前一次性生成 B=1000 的 validation video-cluster bootstrap indices，全部 epoch/variant/seed 复用同一 index set；逐 endpoint 使用**成对相同**的有效 mask；空 replicate 标为 invalid，有效比例 <95% 的 endpoint 判为不可靠并 fail-closed。
- **width=1 退出正式训练目标**：不训练 width classifier、不算 width loss；structure 任务 = future bounded length/termination；width 仅兼容字段。

## 7. 执行与治理

- 顺序（v3.1 修订）：Stage 0 → smoke → **backbone 训练** → J0/B1 → J1/J2/J3 → 冻结 test 判定（J0 依赖冻结 backbone，不能提前）。
- holdout：默认数据管线只构建 train/validation/test；holdout 仅在显式 `--include-nonconfirmatory-holdout` 的独立路径下可作 exploratory，不进入任何正式决策。
- 正式运行一致性（v3.1 修订）：所有正式变体使用同一 device / torch / CUDA / precision / batch（effective batch 一致；必要时统一 gradient accumulation），smoke 阶段先选定；禁止跨变体单独调 batch，禁止 CPU/GPU 混跑。
- 成本：本地单卡 6GB（或 CPU，二选一统一），小 GRU，预计 <2 GPU-hour；需本地安装 torch（当前缺失）。
- 结果解释表：J 全不优于 B1 → 停止联合训练主张；J1 好而 J2/J3 无增益 → 表示监督有效、软接口无增益；J2>J1 → predicted attributes 含资源信息；J3>J2 且 NI PASS → 端到端接口学习有效；J3 NI FAIL → 负迁移拒绝；load 变差而 runtime 无改善 → 不算 Core GO。
- 部分完成规则：若因预算/故障只完成 backbone + J0/B1，只能标记 `partial / incomplete`，不得生成 joint-training 科学结论。

## 8. 变更历史

- v1：初始设计（J1/J2/J3 + 旧 ResourceQScore + 两段式基线）。
- v1 评审：4 P0（B1 定义、ResourceQScore 兼容 hurdle、joint loss 未写死、checkpoint/NI/width）→ v2。
- v2 复核：3 P0（load NI CI 方向写反、NI 各 endpoint 方向规则、成功判定须在冻结后的 test）→ v3 修正并冻结。
