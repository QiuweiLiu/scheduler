# 动作族预测器规划(2026-08-06)

- 文档状态:已完成(2026-08-06 执行完毕,结果见第 9 节)
- 执行地点:远端 /root/autodl-tmp/scheduler
- 原则:统一数据契约、统一 split、统一评估;偏离本文档时在"偏离记录"如实补充。

## 0. 数据规范(强制)

- **expansion 数据必须进入所有后续训练/测试集**(core + expansion 合并,不再单独用 core)
- 全部实验在合并数据集(17,303 行)上产出

## 1. 数据准备

### 1.1 重建 role_dataset_v0_2.jsonl

| 新增字段 | 来源 | 验证状态 |
|---|---|---|
| activity(core) | join compute_events(run_id+event_index) | 10,134/10,134 匹配 ✅ |
| action(expansion) | 从 trace 提取 | 7,169 事件 0 缺失 ✅ |
| question_type | 视频级 join prefix,取非 unknown 多数;52 个不一致视频取多数、21 个缺失标 unknown | 用户已确认 |

split manifest 复用(240/30/30,seed 42)。

### 1.2 动作族映射表(双命名,仅 execute 事件参与)

| 动作族 | core activity | expansion action |
|---|---|---|
| select_frames | sample_seek | frame-selector / image-grid-selector |
| visual_qa | spatial_qa / image-qa / image-grid-qa / patch-zoomer | image-qa / image-grid-qa |
| temporal_ops | temporal-qa / temporal-grounding | temporal-qa / temporal-grounding |
| summarize | summarize | summarization-tool |
| detect | object_detection | (yolo 事件) |
| other | 其余 | generalist.generate 待核对归属 |

### 1.3 合并后 execute 动作族分布核对(含 expansion,与 core 版 3,911 条对比)

## 2. 预测器矩阵(14 个,统一评估框架)

**统计族(免训练)**:全局先验 / 条件计数(位置+baseline+model)/ n-gram(窗口 1/2/3 消融)
**经典 ML 族**:逻辑回归 / 朴素贝叶斯 / 随机森林 / LightGBM / XGBoost(GradientBoosting 备选)
**神经网络族**:MLP / CNN-1D / GRU(one-hot vs embedding 编码消融)/ BiLSTM / Transformer-lite / **GNN(转移邻接图,第一轮就做)**
**检索族**:KNN 相似 trace(前缀 LCS → 相似轨迹下一动作投票)

## 3. 输入输出规格(统一契约)

| 预测器 | 输入 |
|---|---|
| 统计/ML/MLP | 扁平特征:position / model / baseline / 冷热 / yolo_batch / question_type / 最近动作族 one-hot |
| n-gram / CNN / GRU / BiLSTM / Transformer | 前 N 步动作族序列(窗口 1/2/3)+ 扁平特征拼接 |
| GNN | 动作族转移邻接图(train 构建)+ 当前节点 |
| KNN | 前缀序列相似度(LCS) |

输出统一:**6 类动作族概率分布**。

## 4. 评估口径

- 动作族 Top-1 / Top-3(test)
- 执行者路由正确率(动作族 → 执行者映射 vs 真实)
- 全部在 execute 子集、合并数据集、同一 split(240/30/30)上

## 5. 验证门槛

- 动作族 Top-1 ≥ 0.6(条件计数 0.55 为参照)
- 执行者路由正确率 ≥ 0.8

## 6. 执行步骤

1. 本规划文档落远端 docs/action_family_predictor_20260806.md
2. 重建 role_dataset_v0_2(expansion 动作 + question_type)+ 动作族分布核对 + generalist.generate 归属核对
3. 实现统一评估框架 + 14 个预测器
4. 跑实验,结果 + 偏离写回本文档,上传远端

## 7. 风险与预案

- expansion 动作族分布偏移 → 合并后重验,统一口径
- XGBoost 装包失败 → GradientBoosting 替代(如实记录)
- 序列模型训练时间 → 后台运行,窗口消融控制规模

## 8. 偏离记录

(执行中发现与原计划不一致之处,如实记录于此)

**2026-08-06 执行后补充**:

1. XGBoost 装包成功(结果含 08_XGBoost,未走 GradientBoosting fallback)。
2. generalist.generate 归属核对:其 node_type 为 answer_generation(588 条),**不属于 execute**,不参与动作族预测——映射表无需覆盖。
3. role_dataset_v0_2 的 question_type:视频级多数 join,300 视频全覆盖(unknown 4,882 / identification 3,969 / temporal 3,077 / count 1,689 / causal 1,162 / summary 1,092 / spatial 751 / comparison 681)。
4. 合并动作族分布(execute 6,552)与 core 版差异显著:select_frames 3,779(57.7%,core 版 29%)——expansion 的 frame-selector 主导;无 other 类(映射全覆盖)。
5. GRU 编码消融、n-gram 窗口消融均按计划执行。

## 9. 结果

(执行后填写)

**执行时间**:2026-08-06(远端);数据 role_dataset_v0_2 execute 子集(6,552 行,含 expansion),train 5,207 / val 783 / test 562;产物 `results/processed/benchmark_action_family_20260806.json`。

### 9.1 动作族预测对比(test,562 行)

| 排名 | 预测器 | Top-1 | Top-3 | 执行者路由 |
|---|---|---|---|---|
| 1 | **08_XGBoost** | **0.8381** | 0.9947 | 0.9342 |
| 2 | 07_LightGBM | 0.8292 | 0.9929 | 0.9235 |
| 3 | 08_GradientBoosting(备选) | 0.8274 | 0.9947 | 0.9324 |
| 4 | 06_随机森林 | 0.8149 | 0.9893 | 0.9235 |
| 5 | 04_逻辑回归 | 0.8114 | 0.9769 | **0.9377** |
| 6 | 05_朴素贝叶斯 | 0.7883 | 0.9733 | 0.9235 |
| 7 | 09_MLP | 0.7616 | 0.9626 | 0.9288 |
| 8 | 02_条件计数 | 0.7598 | 0.9858 | 0.9004 |
| 9 | 03_n-gram_w2 | 0.7580 | 0.9858 | 0.9021 |
| 10 | 03_n-gram_w3 | 0.7509 | 0.9858 | 0.9004 |
| 11 | 03_n-gram_w1 | 0.7153 | 0.9769 | 0.8968 |
| 12 | 15_KNN | 0.7420 | 0.9804 | 0.9004 |
| 13 | 10_CNN-1D / 12_BiLSTM / 13_Transformer-lite | 0.7402 | 0.97-0.98 | 0.89 |
| 14 | 11_GRU_embedding | 0.7313 | 0.9733 | 0.8968 |
| 15 | 11_GRU_onehot | 0.6708 | 0.9555 | 0.8399 |
| 16 | 01_全局先验 | 0.6584 | 0.9555 | 0.8256 |
| 17 | 14_GNN | 0.6584 | 0.8968 | 0.8274 |

### 9.2 消融结果

- **n-gram 窗口**:窗口 2 最优(0.7580),窗口 3 略降(0.7509),窗口 1 最差(0.7153)——与文献(2604.21629)一致,窗口是重要超参
- **GRU 编码**:embedding(0.7313)> one-hot(0.6708)——与 Weinzierl 结论一致

### 9.3 结论

1. **验证门槛全部达成**:动作族 Top-1 = 0.8381(≥0.6 ✓),执行者路由 = 0.9342(≥0.8 ✓)
2. **XGBoost 胜出**(0.8381),LightGBM 紧随(0.8292)——树模型族统治细粒度预测
3. **NN 族(MLP 0.7616 / GRU 0.73 / Transformer 0.74 / GNN 0.66)全部低于树模型**——与角色预测、资源预测的结论一致:结构化表格 + 离散类别下树模型占优
4. **GNN 最差**(0.6584,与全局先验持平):当前节点 + 邻接传播对本数据无增益,与角色 5 类 benchmark 结论一致
5. **执行者路由最优是逻辑回归(0.9377)**,XGBoost 次之(0.9342)——路由正确率整体高,因为 TOOL 类占多数
6. **expansion 纳入后动作族分布大幅右偏**(select_frames 57.7%),全局先验即达 0.66——树模型在分布偏移下依然稳健

**最终选择**:动作族预测主模型 = **XGBoost(Top-1 0.8381,执行者路由 0.9342)**,LightGBM 为近等备选;条件计数(0.7598)为免训练兜底。两级行为管线:角色 LightGBM(0.9151)→ execute 内动作族 XGBoost(0.8381)→ 执行者派生 → 资源分层路由。
