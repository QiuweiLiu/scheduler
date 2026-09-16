# EXP-20260902 P9d topology tabular comparator

## 结论

P9d 的第二阶段 tabular comparator 已在远端完成。它使用 train-only 的 one-hot causal
model_input 训练 LightGBM 结构头，预测 H=5 的 layer count 和各层 width；节点原型仍由
train-only 条件表解码。validation 在两个预注册配置中选择 lgbm_small，test 和冻结后的
holdout 只作诊断/验收。精确数值只以 metrics.json 为准。

相对前一阶段 empirical baseline，tabular 模型显著改善了未来层数、宽度和节点总量的
结构误差，说明“宽度/层数确实可由当前可见状态学习”。但完整 identity-free topology
signature 的覆盖仍受原型解码限制，尤其在 holdout 上；因此它是成功的结构 comparator，
还不是可以直接接入 scheduler 的最终 topology predictor。

## 方法与边界

- LightGBM 只消费 P9d model_input 的堆栈、任务、当前节点、最近两步历史和 prefix length；
  不消费 video_id、future events/edges、行为或资源 target、执行 truth 或资源 truth。
- 结构输出为 layer_count、width_1 到 width_5；原型字段由 train-only empirical decoder
  补全，保持 node ID/父子边 identity-free。
- fit/selection/test/holdout 严格对应 train/validation/test/holdout；没有读取
  S_train/S_val/T_final，也没有启动 scheduler integration。

## 验收与下一步

本地/远端编译和单元测试通过；预测行数、top-3 概率、节点输出禁止字段和本地/远端
SHA-256 均通过。tabular 阶段完成，但完整拓扑预测仍需解决原型联合建模和分布偏移。

下一步按既定路线比较 shared causal GRU 的 behavior/topology 分头，并以 empirical 与
tabular 两个结果共同作为 holdout 对照；在 learned predictor 通过冻结 holdout gate 前，
不做 scheduler 集成，不解封 T_final。
