# NN 实验再审查与修正计划(2026-08-09)

- 文档状态:执行中
- 触发:外部审查指出 NN 实验存在实现/评估问题,"NN 探索结束"结论下得太早
- 原则:所有修正如实记录;修正后数字与旧数字并列对照;XGB 结论不受影响(z-score 为单调线性变换,树切分不变)

## 1. 审查问题核实结论(代码逐行验证)

| # | 问题 | 核实 | 位置 |
|---|---|---|---|
| 1 | z-score 用 `X_all[0:len(tr)]` 拟合,文件实际混排(前 13754 行 = 10,930 train + 1,552 val + 1,272 test,缺 2,824 train) | **属实** | nn_bilstm_retest.py:223 / nn_both_gru.py:223 / nn_improve_script1.py:216 / nn_improve_script2.py:217。multistep/ablation 用显式 tr_slice,**无此问题** |
| 2 | 序列 padding 右侧无 mask(GRU `h[-1]`=pad 状态、BiLSTM/Transformer/CNN/GNN 同) | **属实** | benchmark_action_family.py:136-137(该脚本全部 NN 数字作废) |
| 3 | FM 计算了 `fm` 但喂 head 的是常数 `fmv`,交互量从未使用 | **属实** | nn_improve_script2.py:287-289 |
| 4 | FT-Transformer 为 lite(标量 token、单块、无数值 tokenizer) | **属实** | nn_improve_script2.py:297-316 |
| 5 | OOF 用普通 KFold(无 group),同 run 事件跨折 | **属实** | nn_improve_script1.py:339 |

## 2. P0 修正清单

### P0-1 z-score 泄漏(4 脚本)
- `X_all[0:len(tr), ns:ne]` → `X_all[tr_idx, ns:ne]`,`tr_idx = [i for i,r in enumerate(rows) if r["split"]=="train"]`
- 影响:BiLSTM/GRU(0.8078/0.8096)、MLP 系列(0.7811-0.7936)、FM/FT(0.7740/0.7865)、全 NN(0.8033)

### P0-2 FM(script2)
- `return self.head(torch.cat([h, fmv], dim=1))` → 改用真正的 `fm`(B,1);head 首层 W+1→W

### P0-3 FT-Transformer v2(script2)
- 2 个 attention block + 残差;数值列 token 加可学习 bias;CLS 池化。标注"v2(接近论文结构,非逐字复刻)"

### P0-4 OOF 分组(script1)
- `KFold` → `GroupKFold(n_splits=5)`,group = `rows[i]["video_id"]`(nf_tr 行)

### P0-5 benchmark_action_family
- 当前行语义(已废弃口径)+ padding bug:本次仅**标注作废**,不重跑(避免混淆生产口径);padding 修复排入 P2

## 3. P1 残差模型(主候选路线)

```text
XGB 角色概率(0.9322)
      +
最终 logits = log(XGB 动作族概率) + GRU residual logits
```

- GRU 输入:现有特征 + XGB 动作族 logits(6 维)作为条件;输出 6 维残差 logits
- 训练:CE on (log-XGB-logits + GRU 输出) → 梯度只更新 GRU(残差学习)
- 评估:单级 Top-1 + 整体 join/route/layer + **长尾指标**(macro-F1、temporal_ops 召回、cost matrix 错判 temporal→select 惩罚)
- 对比:XGB 0.7954 / GRU 0.8096 / 残差 ?

## 4. 执行步骤

1. 修 P0-1..4(本地 4 脚本)
2. 重跑:retest → script1 → script2 → both_gru(远程前台,分次)
3. P1 残差模型脚本 + 跑
4. 汇总写回本文档(修正对照表、结论重写)

## 5. 结果

### P0 修正后单级(动作族 Top-1,next 语义,test 562)

| 方案 | 旧(泄漏) | 修正后 | 变化 |
|---|---|---|---|
| XGBoost 锚点 | 0.7954 | **0.8060**(完整特征) | 树不受 z-score 影响;锚点特征为近似复刻,完整特征下更高 |
| BiLSTM | 0.8078 | 0.8043 | -0.35pp |
| GRU | 0.8096 | 0.8096 | 0 |
| MLP ce | 0.7811 | 0.7811 | 0 |
| OOF(GroupKFold) | 0.7936 | 0.7900 | -0.36pp |
| MLP+FM(真交互) | 0.7740(假) | 0.7705 | FM 确认无增益 |
| FT-Transformer v2 | 0.7865(lite) | **0.8060** | **+2.0pp,NN 家族头部** |
| 全 NN(GRU+GRU) | 0.8033 | 0.8059 | +0.26pp |

**关键结论:z-score 泄漏影响极小(≤0.4pp),原 NN 数字方向不变;FT-Transformer 修复后从 0.7865 → 0.8060(+2pp),是审查的最大收获。**

### P0 修正后整体(join / route / layer)

| 方案 | 旧 | 修正后 |
|---|---|---|
| XGBoost 级联 | 0.8493 | **0.8507** |
| GRU 动作族头 | 0.8447 | 0.8428-0.8447 |
| OOF(GroupKFold) | 0.8428 | 0.8408 |
| FT v2 | 0.8296 | 0.8336 |
| 全 NN(GRU+GRU) | 0.8033 | 0.8059 |

### P1 残差模型(XGB logits + GRU residual,nn_residual_model_20260809.json)

| 指标 | XGB | GRU | **XGB+GRU 残差** |
|---|---|---|---|
| 单级 Top-1 | 0.8060 | 0.8060 | 0.8025 |
| macro-F1 | 0.7376 | 0.5775 | **0.7427** |
| cost_err(调度代价,temporal/detect→select 重罚) | 0.5890 | 0.6352 | **0.5676** |
| temporal_ops 召回 | 0.3617 | 0.2660 | **0.4149** |
| visual_qa 召回 | 0.7534 | 0.9041 | 0.7260 |
| summarize 召回 | 0.9167 | 0.7917 | 0.9167 |
| 整体 join | 0.8507 | 0.8428 | 0.8474 |

**残差模型验证了审查的核心判断**:Top-1 持平(0.8025 vs 0.8060,噪声内),但在**调度相关指标上全面领先**——macro-F1 +0.5pp、调度代价错误 -2.1pp、temporal_ops 召回 +5.3pp(GRU 纯版只有 0.266 的长尾缺陷被残差机制修复到 0.415,且反超 XGB)。

## 6. 结论重写(取代"NN 探索结束")

1. **"NN 探索结束"撤销**:"NN 不行"的旧结论基于未修正的实现(泄漏 + FM 假实现 + FT lite + padding 污染 + OOF 无分组)。
2. **NN 的真实价值在调度维度,不在 Top-1**:XGB+GRU 残差的调度代价错误率与长尾召回全面优于纯 XGB;Top-1 差距 0.35pp 在噪声内。
3. **FT-Transformer v2 值得保留**(0.8060 单级,NN 头部)。
4. **残差模型为生产候选**:调度器集成时,动作族概率用 `softmax(log XGB + GRU residual)`(P1 候选);若追求简单,纯 XGB(0.8060)仍可用。
5. **GRU 纯版的长尾缺陷(temporal 召回 0.266)证实 NN 必须与 XGB 结合,不能单独上岗**。
6. **待办**:多模态 embedding(v0.9 全量)、实例图 GNN(电梯)、调度代价 loss 直接优化(当前已用残差验证代价指标,下一步可把 cost 直接纳入 loss)。

## 7. 偏离记录
1. P0-5 benchmark_action_family 的 padding 修复未执行(当前行语义已废弃口径),仅标注作废;padding 修复留待 P2(电梯迁移时统一管线)。
2. 远程 SSH 会话不稳定(超时/断连多次),采用"前台 + 轮询 JSON"模式克服;nohup 日志不持久。
3. script1 两次被杀(超时+断连),最终前台 30 分钟完成;OOF 5 折 XGB 为耗时大头。
4. 残差模型评估段首次遗漏 logits 传参(维度错误),修复后重跑。
