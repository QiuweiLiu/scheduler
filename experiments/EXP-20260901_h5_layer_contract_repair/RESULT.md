# EXP-20260901 H5 layer contract repair

## 结论

本轮完成了 Pred/True H5 评分所需的 layer-aware 数据契约和 opt-in 评分入口，
并在同一 100 集输入上通过了逐候选审计和 10 集 scheduler smoke。

但本地没有 B05 checkpoint/dataset，无法生成真正的多节点未来拓扑。因此本轮使用的
`legacy_event_to_unary_layer_projection` 只是把旧的 5 个预测事件放进 5 个层，
每层固定 1 个节点；它证明了新契约可表达多节点层，但不能证明 topology prediction
已经完成。

## 已完成

- 新增 `src/tracing/scheduling/future_topology.py`，定义 `future_h5_layers`：每个 scenario
  按 `layer_offset=1..H` 排列，每层有 `nodes` 列表；禁止把 `node_id`、前驱或后继字段
  放进预测 artifact。
- `load_future_artifacts` 以可选 sidecar 方式加载
  `b05_future_h5_layers.jsonl.gz`，旧 artifact 缺少 sidecar 时行为不变。
- 新增 opt-in `aligned_predopt_h5_layer`，当前 action、cache 更新、priority 和 tie-break
  与 aligned H5 共享；预测未来按层、按层内节点累加。
- `scripts/scheduling_future_r7_artifacts.py --emit-layer-h5` 可在具备 B05 输入时输出
  layer sidecar；当前生成器明确标记 unary projection，不声称是正式 topology model。
- 新增本地 adapter 和 topology audit，输出均在本实验目录，未改写 `.scratch` 冻结输入。

## 结果

| 检查项 | 结果 |
|---|---:|
| layer artifact rows / scenarios | 8,935 / 26,805 |
| 预测层数 / scenario | 5 / 5 |
| 预测节点数 / scenario | 5 / 5 |
| True 侧加权非空层数 | 2.5107 |
| True 侧加权 DAG 节点数 | 5.3889 |
| 100 集 decisions / candidates | 15,364 / 44,690 |
| `aligned_predopt_h5_layer` top-1 / mean regret | 66.06% / 7,978.7 ms |
| 与旧 `aligned_predopt_h5` score/action | 完全相同 |
| 10 集 smoke failed jobs | 0 |
| full unittest | 60/60 |

新策略与旧 aligned Pred 完全相同是预期现象：unary projection 没有增加预测信息。
这次实验排除了“只改 H5 字段/评分读取入口就能恢复收益”的可能；剩余核心问题仍是
**如何从 train-only 输入预测每个未来 DAG layer 的节点数量、节点类型和资源身份**。

## 边界与下一步

- 没有连接远端，没有真实 GPU 测量，没有读取 `T_final`。
- 没有改变动作空间、单节点单 GPU 语义、WAIT/RESERVE、抢占或多 GPU。
- 本轮结果是 `passed_diagnostic_unary_projection`，不提升为正式 topology predictor，
  也不进入 300/1,000 集策略选择。
- 下一步需要补齐本地或获授权读取的 P_dev topology training input/checkpoint，训练或
  固化一个 train-only layer-width/node-prototype predictor，再沿同一 sidecar schema
  重跑 10 集 smoke 和 100 集逐动作审计。
