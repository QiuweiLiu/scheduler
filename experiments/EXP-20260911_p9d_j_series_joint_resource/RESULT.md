# EXP-20260911_p9d_j_series_joint_resource — RESULT

状态：**正式 test 判定已执行（一次）**；严格成功规则判定 **无 Core GO**（load-duration 非劣边际失败）。
关联：`docs/p9d_j_series_design.md`（frozen v3.1）、`.project/EXPERIMENT_GATE.json`、评审记录 `docs/research/2026-09-11_j_series_design_review{,v2}.md`、`docs/research/2026-09-11_j_impl_plan_review{,v2}.md`、结果审计/评审 `docs/research/2026-09-11_j_result_audit.md`、`docs/research/2026-09-11_j_result_review.md`。

## 0. 运行事实

| 项 | 值 |
|---|---|
| 数据集 | `results/processed/j_series_dataset_v1`（行 13,754/2,029/1,520；监督槽位 48,343/7,195/5,387；注册 `data/manifests/j_series_dataset_v1.json`） |
| 资源表 | `experiments/EXP-20260911_p9d_r0_oracle_signature_ceiling/artifacts/node_table.jsonl.gz`（sha256 见 run_manifest） |
| config sha256 | `a201c9ff007c5ee8ad0f19856d278d0767eeed190ac1628998a2231c4e4e0d58` |
| 环境 | conda env `scheduler`；Python 3.10.21；torch 2.6.0+cu124；RTX 3060 Laptop（6GB）；device=cuda；batch=256（全部变体一致） |
| bootstrap | B=1000，seed 20260911（validation）/ +1（test），video-cluster，全部 epoch/变体/seed 复用固定 indices；有效比例 1.000 |
| 训练预算 | backbone 3×30 epochs（run.json 合计 311.98s）+ 变体 15×30 epochs（run.json 合计 1371.5s；wall 1393s）+ test 13s ≈ 28.3 GPU-min |
| 选择规则 | 固定 30 epochs，无 early stopping；validation NI-feasible → argmin RuntimeQScore；**test 仅执行一次** |
| 运行产物 | `outputs/j_series_joint_resource/`（checkpoints、run.json、test_eval.json）；小结果已复制到本目录 |
| 运行命令 | `D:\anaconda\envs\scheduler\python.exe scripts/j_series_train_eval.py --config experiments/EXP-20260911_p9d_j_series_joint_resource/config.json --stage {stage0\|smoke\|backbone\|variants\|test}`（依次执行；smoke 在独立子目录 `smoke/`） |
| 判定聚合规则 | 官方 Core GO 采用**逐 seed（all-seed/worst-seed）**读法：3/3 seeds 均需 runtime CI 上界<0 且 load 双端点通过；`test_eval.json` 中同时记录的 ≥2/3 判据为**事后次级聚合**，不作为冻结判定依据（见 §1 注） |

Stage 0 断言：行数/哈希与注册表一致；模型 127,500 参数；梯度路由断言全部符合设计（J0/B1 无 encoder 资源梯度；J1/J2 encoder 有、属性头 0；J3 属性头收到资源梯度 0.0687；backbone 资源头不受影响）。

## 1. 主判定（P_dev/test 冻结，Δ = 变体 − B1，RuntimeQScore 越低越好）

| 变体 | seed 11 Δ [95% CI] | seed 22 Δ [95% CI] | seed 33 Δ [95% CI] | 均值 Δ | Core gate |
|---|---|---|---|---|---|
| J0 | +115.09 [+87.21,+137.09] | +119.37 [+95.00,+138.59] | +147.44 [+125.13,+167.30] | +127.30 | 否（runtime 更差） |
| **B1（参照）** | 0（point 914.7） | 0（point 900.9） | 0（point 923.9） | 0 | — |
| J1 | +26.40 [+3.95,+50.44] | +19.89 [+0.20,+36.77] | +73.76 [+53.33,+95.30] | +40.02 | 否（runtime 更差） |
| J2 | **−49.00 [−69.98,−25.91]** | **−65.39 [−79.61,−52.87]** | **−29.88 [−45.48,−16.30]** | **−48.09** | 否（load-duration） |
| J3 | **−78.12 [−98.60,−54.12]** | **−77.09 [−95.91,−60.39]** | **−63.97 [−80.11,−49.18]** | **−73.06** | 否（load-duration） |

- Runtime 主端点：J2/J3 在 **3/3 seeds** 上 CI 上界 < 0（对 B1 显著更优）；J1/J0 在 3/3 seeds 上显著更差。
- 排序（均值 Δ）：**J3 (−73.1) > J2 (−48.1) > B1 (0) > J1 (+40.0) > J0 (+127.3)**。

注（评审披露，审计要求）：
- 跨 seed 聚合的官方判据为 all-seed（worst-seed）读法（见表"判定聚合规则"）；在 all-seed 读法下 J2/J3 均因 load-duration 逐 seed 失败而无 Core GO；`test_eval.json` 中的 ≥2/3 判据为未预注册的事后次级统计，不改变上述结论。
- **J2 seed22 是唯一通过全部冻结逐 seed 判据的 run**（runtime CI 上界 −52.87<0；Brier CI 上界 −0.00349<+0.005；duration CI 上界 +9.94ms < 阈值 10.195ms，差 2.5%）。该通过幅度在 bootstrap 噪声尺度内，且其余 2/3 seeds 失败，故**不得**作为 load 非劣或 Core GO 的证据引用。

## 2. Load 非劣（越低越好；δ_Brier=+0.005，δ_duration=+5%·B1 参考均值）

下表 Brier 列统一使用**各变体的最差 seed CI 上界**（与 duration 列同口径）；duration 阈值 = 0.05×同 seed B1 参考均值 = 11.19/10.20/10.29 ms。

| 变体 | Brier CI 上界（worst seed） | duration CI 上界（ms；worst seed） | 判定 |
|---|---|---|---|
| J0 | +0.0141/+0.0117/+0.0133（全 fail） | +16.1/+27.8/+22.6（全 fail） | fail |
| J1 | ≤+0.0034（pass） | +26.1/+19.9/+43.7（全 fail） | fail |
| J2 | ≤−0.0035（pass） | +27.2/**+9.9**/+17.4（仅 seed22 通过） | fail（2/3 seed 越界） |
| J3 | ≤−0.0036（pass） | +25.1/+14.2/+16.8（全 fail） | fail |

- J2 的 duration 点估计相对恶化（逐 seed 匹配基线）：+7.3%/+3.0%/+5.4%；J3：+6.7%/+4.7%/+4.3%。越界幅度不均（J2 seed11 为 2.4× 阈值，属明确越界；其余在 5%–40% 超阈值区间），并非全部"边际"。
- Brier 端 J2/J3 显著更优（CI 上界 < 0）；只有 J0 的 Brier 明确变差。
- 严格 Core success 要求 runtime CI 上界 < 0 **且** load 双端点通过 → J2/J3 因 duration 逐 seed 失败而 **无 Core GO**；J2 seed22 的单独通过按 §1 注处理。

## 3. 接口端点（test，诊断用，不进 gate）

- J1/J2/J3 的 length MAE Δ vs B1 在 −0.005 ~ +0.017；content Δ ≤ +0.0022；next-role Δ ≤ +0.0079；next-family Δ ≤ +0.0109。三个联合变体未出现接口崩塌（全部远小于对应 δ）。
- J0/B1 的接口 Δ 严格 = 0（冻结预测器；验证了无资源→接口泄漏）。
- duration endpoint 的逐 seed 覆盖：n_rows=474（占 test 1,520 行的 31.2%，仅 load>0 槽位行），bootstrap valid_ratio=1.000。

## 4. 机制诊断

- 梯度探针（实现口径：epoch 1、batch=64；run.json 记录 norms/cosines）：‖g_R‖≈4.8–5.1，显著大于 ‖g_S‖≈1.3–2.1、‖g_B‖≈1.2–1.8、‖g_C‖≈0.3–0.4；cos(g_R,g_B)≈−0.10~−0.19（轻度冲突），cos(g_R,g_C)≈0.0~0.12（9 条完整范围见 run.json）。资源损失在共享强度上占主导，为后续加权/J4 提供依据。
- NI-feasible epochs：J1 29/29/26，J2 29/29/26，J3 30/29/26；接口训练总体与 backbone 持平或略优（content/next-role/family 略升）。
- 与 R0 的关系（方向/机制对齐，幅度不可比）：R0 是 oracle 属性签名的上界（validation/test 改善 74.9%/82.1%）；J 以 deployable 形式显示 B1/J2/J3 ≫ J0/J1，说明收益方向来自**属性接口信息**而非单纯资源头容量。

## 5. 结论（不超过证据边界）

1. **J2/J3（软属性接口 + 联合表示）显著优于 B1**：RuntimeQScore 逐 seed 相对降低（匹配同 seed B1 基线）J2 = 5.36%/7.26%/3.23%，J3 = 8.54%/8.56%/6.92%；3/3 seeds CI 上界 < 0。方向稳定，无种子反转。
2. **J1（联合表示、无接口）劣于 B1**：说明仅共享表示不足以带来增益；增益依赖 predicted-attribute 接口。
3. **J0（冻结表示、纯资源头）最差**：与"接口/表示适应"解释一致。
4. **J3 的 NI 未通过 ⇒ 不能声称端到端接口学习有效**（设计负迁移拒绝条款）；本实验未做 J2-vs-J3 配对检验，两者的均值差异（−73.1 vs −48.1）不构成显著性声明。
5. Load Brier 端 J2/J3 显著更优；duration 端 J2/J3 的恶化（2/3 或 3/3 seed 越界）如实报告，未做槽位/长尾归因（无分层产物），属后续工作。

## 6. 边界与限制

- 仅 P_dev train/validation/test；`P_holdout_diag` 未参与任何训练/选择/判定。
- 单一 backbone 配置（hidden=128、GRU、30 epochs、batch=256、3 seeds）；未做 hidden/温度/权重敏感性扫描。
- RuntimeQScore 为 raw-ms pinball(τ=.5/.9/.95) 在"存在 future slot"上的均值；LoadDurationQ 为 raw-ms pinball（load>0 槽位）。load-duration 相对 δ 的分母为同 seed B1 的 test 点均值（设计未指定分母，本口径已在 config/RESULT 标注）。
- 5% duration 容差为设计期工程容差；结果中 2/3 或 3/3 seed 越界，**不得**在事后调整容差或改用其它统计量来宣称成功。
- 未运行 OracleAttr 诊断变体（设计表列出的诊断项）；其角色（oracle 属性上界）由 R0 结果承担。
- backbone 的选择准则（validation structure+content+behavior 损失最小）在设计中未逐字写死；由于其仅为所有变体共享的初始化且验收以 J 变体为准，不构成本实验的差异混淆。
- 无调度器集成；`S_*/T_final` 未动；v1–v3 数据集与既有实验未改动。
- 本结果已通过独立审计（`docs/research/2026-09-11_j_result_audit.md`）与独立评审（`docs/research/2026-09-11_j_result_review.md`）；相应修正（Brier 统计口径、seed22 披露、聚合规则、相对改善复现、接口上界）已并入本文件。
- 现象机制探索（load-duration 微降的成因）见 `docs/research/2026-09-11_j_mechanism_analysis.md`（validation-only 探索性诊断：头交换、接口熵、log/raw 分解；不改变本文件的 test 判定）。

## 7. 后续选项（供决策，不自动执行）

- A. 复现/稳健性：增加 seeds（如 44/55/66）确认 J2/J3 排序与 load-duration 边际是否稳定。
- B. 预注册修订（需新的门禁）：对 load-duration 的 hurdle/加权单独建模（例如 duration 只在 load 发生槽位、或改为 log 尺度 + 新的 δ 事前注册），并重新跑一次正式判定。
- C. J4 方向：按梯度探针调整组权重（如 PCGrad/不确定性加权）以缓解 cos(g_R,g_B)<0 的冲突。
- D. 停止联合训练主张，转向已证实的主线（属性接口 + R0 校准）并记录负结果。

## 8. 收束记录（2026-09-11，post-J4）

- **J4（duration 解耦）已执行并否定**：见 `experiments/EXP-20260911_p9d_j4_duration_branch/RESULT.md`（J4a 0/3 可行；J4b 仅早期 epoch 可行且运行时代价大；duration 未修复；共享塑形对 duration 有益）。按预注册负结果规则拒绝解耦方向。
- **"为什么退化"的收尾诊断已完成**（validation-only）：见 `docs/research/2026-09-11_j_mechanism_analysis.md` §6——退化是**分布内的拟合重分配**（τ=.90 变差、τ=.95 3/3 seeds 变好、中位数不变且 J3 校准更好），幅度小、结构随 seed 不稳定，属贴线级形状偏移；无"坏路径"可修。
- **GPT 建议与文献线索**：`docs/research/2026-09-11_j4_literature_and_gpt_ideas.md`（条件化 hurdle、model-class 门控、分布参数化、校准后处理；serverless 冷启动与 HPC hurdle 文献支持"结构化建模优于裸多任务回归"，且确认纯执行前 load-duration 预测为文献空缺）。
- **用户决策（2026-09-11）**：选择 **A. 收尾** —— 联合训练线到此为止；不再做修复实验（B/C 暂不执行）。记录于 `.project/DECISIONS.md`。
- 本 RESULT 的主判定（无 Core GO）不受后续分析影响；`test_eval.json` 保持冻结，未重跑 test。
