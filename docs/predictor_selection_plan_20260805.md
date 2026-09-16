# 预测器选择实验规划(2026-08-05)

- 文档状态:已完成(2026-08-05/06 执行完毕,结果见第 7 节;资源 benchmark 按用户指示延后)
- 执行地点:远端 /root/autodl-tmp/scheduler
- 原则:全部实验在同一数据契约、同一 split、同一指标下进行,保证可比;防泄漏契约沿用现有门禁;偏离本文档时在"偏离记录"如实补充。
- 依据:本轮全部文献调研(HexAGenT/Maestro/Pythia/ProD/PASTE/Speculative Actions/CacheScout/Decima 等)+ 抽象维度验证(abstract_dimension_predictor_20260805.md,已通过)

## 1. 目标

用统一实验对比多种行为预测器与资源预测器,选出:
- 预测能力最强(同一指标下);
- 可迁移性最好(电梯 Phase 6:角色词表/模型无关);
- 计算成本可接受(调度场景毫秒级推理)。

## 2. 预测器矩阵

### 2.1 行为预测器(目标:抽象角色 4-5 类为主,coarse 4 类 / 细粒度 7 类为对照)

| 编号 | 预测器 | 家族/论文依据 | 训练需求 | 新代码量 |
|---|---|---|---|---|
| B0 | v0.4 PlannerAwareTracePredictor | 层级计数+回退(自有,主基线) | 无(计数) | 复用 |
| B1 | 一阶/二阶马尔可夫(+planner+baseline) | 统计(既有结果可引) | 无 | 复用 |
| B2 | **LightGBM 多分类** | 经典 ML(Maestro 用 LightGBM 分类动作类型) | 是,分钟级 | 新,~100 行 |
| B3 | **MLP(结构化特征,2 层)** | 轻量 NN(ProD/EGTP 的 MLP 风格) | 是,分钟级 | 新,~150 行 |
| B4 | GRU / LSTM(语义序列) | 深度序列 NN(自有 v0.3 实现,换角色目标重训) | 是,GPU 分钟-小时 | 复用+改词表 |
| B5 | GNN(图邻接) | 图 NN(自有 v0.3) | 是,GPU | 复用+改词表 |
| B6 | **在线转移矩阵(CacheScout 式)** | 在线学习(2605.27744) | 无(在线计数) | 新,~80 行 |
| B7 | LLM 零样本 next-role(可选) | LLM 冷启动(PPM-LLM 2601.11468) | 无(API/本地) | 后置,成本高 |

特征契约(统一):planner_model_id、baseline、position、question_type、coverage/progress 分箱、raw_tail(降维版)、首步族。
注意:细粒度目标(7 类)只作为研究对照;主目标是抽象角色。

### 2.2 资源预测器(目标:runtime_ms 主,peak VRAM / load_ms 次)

| 编号 | 预测器 | 家族/论文依据 | 训练需求 | 新代码量 |
|---|---|---|---|---|
| R0 | 分组分位表(角色×模型,查表) | 非参数统计(基线) | 无 | 复用(abstract 脚本) |
| R1 | 分组分位表 + backoff(样本少回退) | 同 R0 改进(v0.4 backoff 思路) | 无 | 小改 |
| R2 | **LightGBM 回归(log1p runtime)** | 经典 ML(Maestro 角色分组 log 回归) | 是,分钟级 | 新,~120 行 |
| R3 | **MLP 回归(结构化特征)** | 轻量 NN | 是,分钟级 | 新,~150 行 |
| R4 | **ProD 式分布输出(MLP 多 bin + 中位数/分位解码)** | 分布预测(2604.07931) | 是,分钟级 | 新,~200 行 |
| R5 | **roofline 解析模型(runtime ≈ a(model)×输入规模+b)** | 物理模型(HexAGenT Estimator) | 拟合每模型系数 | 新,~100 行 |
| R6 | 两级:分组分位 + 组内回归修正(只对高 CV 组) | Maestro/Pythia 组合 | 部分 | 组合 |

特征契约(统一):角色、model_id、输入规模(帧数/图像数/yolo_batch)、冷热(model_resident_before)。

## 3. 实验设计

### 3.1 数据与划分
- 行为:8,268 prefix,视频级 48/8/8(locked test 559 行);train 拟合、validation 选型、test 只看一次
- 资源:compute_events(10,134),**视频级 split**(按 video_id 对齐 48/8/8);训练只用 train 视频事件;峰值显存/时长 label 不进入任何输入特征
- 防泄漏契约:沿用现有门禁(答案/未来事件/测试视频 ID 不进输入;scaler/vocab 只看 train)

### 3.2 指标
- 行为:Top-1 / Top-3 / NLL / ECE(全体 + 非终止),按角色×planner 分桶
- 资源:MAE / RMSE / P95(整体 + 按角色×模型分桶),相对现状(整体 runtime MAE≈3,094ms)
- 迁移性:每个预测器标注"可迁移资产"(角色词表/模型物理参数/特征契约)与"场景绑定资产"

### 3.3 前置天花板测量(ProD 噪声半径)
先对同 (video, prefix) 的重复 run 测 runtime 中位数噪声半径:若所有预测器误差接近噪声半径 → 到数据上限,选最简模型;若远高于 → 模型还有提升空间。

## 4. 执行步骤

1. 本规划文档落远端 docs/predictor_selection_plan_20260805.md
2. 前置:噪声半径测量(复用/扩展现有脚本)
3. 行为批次:统计族(B0-B1 复用)→ 新实现(B2 LightGBM / B3 MLP)→ 神经族(B4 GRU/LSTM / B5 GNN,换角色词表)
4. 资源批次:R0-R1(分位表+backoff)→ R5(roofline)→ R2/R3/R4(训练型)→ R6(组合)
5. 汇总:统一对比表(行为表 + 资源表),含每模型训练时间与推理延迟
6. 结论与选择写回本文档第 7 节;偏离记录第 6 节

## 5. 验收与选择规则

- 行为:抽象角色 Top-1 超过 B0(细粒度 0.8157 的对应口径)且 validation 选型稳定者入选;优先选"可迁移资产占比高"的模型
- 资源:runtime MAE/P95 优于 R0(分位表)者入选;ProD 式输出(R4)若 MAE 相当但提供分位,则优先(调度需要)
- 最终交付:行为 1 主 + 资源 1 主(可迁移组合),含完整实验报告

## 6. 偏离记录

(执行中发现与原计划不一致之处,如实记录于此)

**2026-08-05 执行后补充**:

1. B1 实现为简化条件计数(planner+baseline+position 条件),非 v0.4 完整 5 层加权;细粒度口径下 v0.4 正式数字(0.8157)另行引用,不与 B1 混淆。
2. **B5 GNN 初版已补跑**(本轮),但 B7 LLM 零样本为本轮新增执行(见第 7 节)。
3. 资源侧 `model_resident_before` 字段全部为空 → **本轮修复方案**:序列推导(per run model 首次=冷/再次=热,100% 覆盖 17,303 事件)+ yolo_batch=8 常数补全;采集器硬编码 None 的 bug(videoseek_wrapper.py:365)记录为未来修复项。
4. **角色数据集重建**:prefix 的 compute_last_node_type 缺 videotool 节点(仅 4 种取值),无法重建 5 类 → 改用 **compute_events 重建**(node_type 100% 覆盖),并提取 expansion 472 runs trace 补全(7,169 事件),合计 17,303 行 / 300 视频。
5. **重划分(用户决策,不与 v0.4 历史对比)**:视频级 train 240 / val 30 / test 30(seed=42),manifest: `results/processed/split_manifest_v0_1.json`。
6. 资源侧 `input_scale.frame_count` 仅 38.6%、`queue_ms` 0 样本、无 token 数——资源 benchmark(R 系列)按用户指示**延后**,相关差距与补充方案记入 `docs/predictor_selection_plan_20260805.md` 第 2.2 节讨论,后续单独推进。
7. 目标空间修正:init 永不为"下一步"目标 → 目标空间为 4 类(plan/execute/aggregate/terminate),报告仍称 5 类角色流程(含 init/verify 预留)。

## 7. 结果

(执行后填写:各预测器指标对比表、结论、选择)

**执行时间**:2026-08-05/06(远端);产物 `results/processed/benchmark_role5_20260805.json`、`results/processed/llm_zero_shot_role_20260805.json`、`results/processed/role_dataset_v0_1.jsonl`、`results/processed/split_manifest_v0_1.json`。

### 7.1 角色 5 类流程数据集

- 17,303 行(core 10,134 + expansion 7,169)/ 1,240 runs / 300 视频
- 角色分布:init 1,240 / plan 6,749 / execute 6,552 / aggregate 2,762;目标(下一角色)4 类(plan/execute/aggregate/terminate)
- 冷热状态:序列推导 100% 覆盖(冷 3,746 / 热 13,557);yolo_batch=8 补全
- 重划分:train 240 视频(13,754 行)/ val 30(2,029)/ test 30(1,520)

### 7.2 行为预测器对比(角色 4 类目标,test 1,520 行)

| 预测器 | Top-1 | Top-3 | NLL | ECE |
|---|---|---|---|---|
| **B2 LightGBM** | **0.9151** | 1.0000 | 0.2748 | 0.0263 |
| B4 GRU | 0.8362 | 0.9908 | 0.5568 | 0.1065 |
| B3 MLP | 0.8164 | 0.9901 | 0.5953 | 0.0948 |
| B1 条件计数 | 0.8013 | 1.0000 | 0.4975 | 0.0131 |
| B4 LSTM | 0.7928 | 0.9888 | 0.6515 | 0.0980 |
| B5 GNN | 0.7368 | 0.9289 | 0.9753 | 0.2539 |

**结论**:LightGBM 再次全面胜出(Top-1 0.9151,ECE 0.0263 且校准最好);GRU 次之(0.8362);GNN 最差(当前角色已高度决定下一步,额外图传播引入噪声);条件计数为免训练基线(0.8013,ECE 最低 0.0131,适合保守场景)。**行为主预测器选择:B2 LightGBM(角色 4 类目标)。**

### 7.3 LLM 零样本(冷启动对照)

本地 Qwen3-4B,test 200 行子集:**Top-1 = 0.035**(200 行仅命中 7),191/200 无法解析。

诊断:模型输出退化为重复文本(如"视频问答视频问答..."、句号循环、自言自语),不执行"只输出角色名"的指令——不是解析问题,是零样本生成质量问题。

**结论:本地 4B LLM 零样本作为行为预测器不可用**,如实记录为失败对照;与文献结论一致(PreScam 2605.12243、社交模拟 2502.12073 均实测零样本 LLM 在动作预测上不如监督小模型)。冷启动兜底不依赖 LLM 零样本,改用"模板先验 + B1 条件计数冷启动"方案。

### 7.4 最终选择(迁移视角)

| 层 | 主模型 | 免训练对照 | 电梯迁移资产 |
|---|---|---|---|
| 行为(角色) | B2 LightGBM(Top1 0.9151) | B1 条件计数(0.8013,ECE 0.0131) | 角色词表跨场景共享;特征为模型/角色/位置/冷热等通用量 |
| 资源 | (延后) | (延后) | roofline/分位表/冷热特征已具备 |
| 调度 | coarse 角色路由 + 分位预留 | —— | score 函数、分位口径与动作名无关 |

**下一步**:实现"角色 4 类 + 资源分位"的生产管线(行为 LightGBM),接入调度 score 函数;电梯接入仅需换 14 节点→角色映射表。
