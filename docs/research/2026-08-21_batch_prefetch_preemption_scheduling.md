---
research_backend: primary_papers_and_official_docs
chatgpt_web_status: unavailable_connection_closed
timestamp: "2026-08-21"
status: valid
scope: "batch size, real model preload cost, preemption, and unchanged binary priority"
---

# Batch、预加载与抢占：现有证据和文献核验

## 0. 先纠正一个口径

“当前调度器没有 batch”不等于“项目以前没有测过 batch”。两件事不同：

1. **已经测过的 batch**：YOLO11x 在单个 `yolo-tracker` 工具动作内，对同一视频抽取多帧，真实调用 `batch=1/2/4/8/16/32/64`。它改变了 YOLO 的显存、检测摘要和后续 planner 轨迹。
2. **当前调度动作**：仍然是 `ready GPU node × free GPU`。调度器还没有“把多个请求组成一个 batch”或“选择某个 batch size”的动作字段。

因此，之前把“batch 未进入调度器动作”说成“batch 没测过”是不准确的；本报告把两层含义分开。

## 1. 已完成的 batch 测量（远端可复核）

来源：远端 `docs/phase2_dynamic_trace_collection.md`、
`results/processed/phase3_yolo11x_batch_pilot8_batch_summary.json` 和
`phase3_yolo11x_batch_pilot8_trajectory_comparison.json`。

- 8 个 Video-MME 视频 × `star/langgraph_react` × 7 个 batch；每档 16 条，共 **112 条 trace**。
- 每条先真实运行 YOLO11x 多帧检测，再把检测摘要交给 Qwen3-VL-8B planner。
- 112/112 summary 成功，112/112 validator `VALID`，实际有效 batch 与请求值一致。

| YOLO batch | YOLO peak allocated (MB) | 检测数均值 | unique paths | 相对 batch=1 的路径变化 |
|---:|---:|---:|---:|---:|
| 1 | 372.672 | 0.125 | 16 | — |
| 2 | 419.535 | 1.000 | 14 | 68.75% |
| 4 | 581.660 | 3.375 | 16 | 93.75% |
| 8 | 906.910 | 11.875 | 14 | 93.75% |
| 16 | 1,560.410 | 25.250 | 13 | 93.75% |
| 32 | 2,864.410 | 54.125 | 15 | 100% |
| 64 | 5,474.410 | 92.250 | 15 | 100% |

这证明了 **YOLO 的 batch 已经是一个真实的资源/行为因子**，但它证明的是“单个工具动作的多帧 batch”，不是“跨 job 的动态 batching”。不能把这 112 条 pilot 直接当成 scheduler 已经支持 batch action 的证据。

## 2. 当前 simulator 的真实边界

远端 `src/tracing/analysis/workload_v02_simulator.py` 的只读核对结果：

- 没有 `batch_size` 字段，也没有 scheduler batch action；
- 没有 active-node 的 preempt/cancel/resume 事件；
- `service_class` 仍是二元 `priority/normal`，现有 workload 约每 4 个 job 中 1 个为 priority；**本轮保持不动**；
- `initial_residency_hint` 只是 episode 开始时的缓存初态，不是一次真实预加载动作；
- 节点启动时若模型不在 GPU，会走 `model_load_start` 并把 `load_ms` 计入该节点；这部分已有真实 load 成本。

所以后续要做的是：把已测的 YOLO batch 曲线接进资源画像，并新增一个真实计费的 prefetch/load 机制；不是重做 batch pilot，也不是先改 priority。

## 3. 论文和官方实现怎么处理

### 3.1 Clockwork：把 load 当成调度动作和 SLO 成本

Clockwork 的核心做法是中央控制器显式决定何时把模型从 worker 内存加载到 GPU、何时执行 inference；控制器维护全局 cache/scheduling 状态，并把 `LOAD + INFER` 一起纳入可预测的完成时间。它与我们的模型驻留、load、eviction 状态最接近。

**对我们的启示（VERIFIED → ADAPTATION）**：prefetch 不能是免费状态修改；应有 `prefetch_start → prefetch_end`，占用实际加载时间和内存，并可能与其他 load/compute 竞争。只有模型在真正执行前仍驻留，执行节点才可以不重复付 load；若中途被驱逐，则再次支付真实 reload。

来源：[Clockwork, USENIX OSDI’20](https://www.usenix.org/conference/osdi20/presentation/gujarati)；[论文 PDF](https://www.usenix.org/system/files/osdi20-gujarati.pdf)。

### 3.2 Triton：动态 batch 有最大值、等待窗口和优先级队列

Triton Dynamic Batcher 将**同一模型的多个请求**动态合并；配置包括最大 batch、可选 preferred batch、为等更多请求加入而允许的最大 queue delay，以及 priority levels。官方建议先用 Performance Analyzer 测 latency/throughput，再调最大 batch 和 delay，而不是直接猜一个 batch。

Triton 的模型管理也提供显式 `load/unload`；显式模式下，加载和卸载由控制 API 发起，服务器还可用专门的 model-load threads 在后台加载。即便是后台加载，也不是零成本：加载动作有时间、内存和并发资源边界。

**对我们的启示（VERIFIED → ADAPTATION）**：保持现有 binary priority 语义，只在同一 priority 规则下记录 queue age/slack；batch 实验要同时记录吞吐、单请求等待和显存，不能只看 completion mean。

来源：[Triton Batchers](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)；[Triton Model Management](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_management.html)。

### 3.3 Orca：适合多迭代生成，不应直接套到一次性 YOLO 节点

Orca 将调度粒度从整个 request 改成每次 generation iteration，并用 selective batching 处理可批处理和不可批处理的算子。它的要点是：当一个请求还没结束时，新请求不必等整个旧 batch 完成。

**对我们的启示（INFERENCE）**：如果未来把 Qwen/VideoAgent 拆成可恢复的多轮或 token/step 节点，可以研究 iteration-level batching；当前 YOLO `yolo-tracker` 是一次性工具动作，不能用 Orca 的结果声称已经有跨 job batch。

来源：[Orca, USENIX OSDI’22](https://www.usenix.org/conference/osdi22/presentation/yu)；[论文 PDF](https://www.usenix.org/system/files/osdi22-yu.pdf)。

### 3.4 FastServe/vLLM：抢占必须有可恢复状态和明确成本

FastServe 在 LLM 的输出 token 粒度抢占，并主动在 GPU/CPU 之间搬运中间状态；vLLM 的调度器明确区分 recompute 和 swap 两种 preemption：前者释放状态、恢复时重算，后者把状态换出再换回。两者都不是“暂停一下就免费让路”。

**对我们的启示（VERIFIED → ADAPTATION）**：当前 video tool 节点没有 checkpoint/resume 合约，先不要加入抢占动作。若以后加入，至少要记录 `preempt_start/end`、已完成工作、checkpoint/swap 或 recompute 时间、恢复后的剩余 runtime；否则会人为制造 scheduler 优势。

来源：[FastServe, USENIX NSDI’26](https://www.usenix.org/conference/nsdi26/presentation/wu-bingyang)；[vLLM scheduler preemption modes](https://docs.vllm.ai/en/v0.10.2/api/vllm/core/scheduler.html)。

## 4. 对本项目的最小落地路径

### A. 先利用已经测过的 batch，不立即扩大动作空间

把 `batch ∈ {1,2,4,8,16,32,64}` 作为 **YOLO 节点的资源配置/特征**，用已有测量建立查表或分段模型：

```text
(tool=yolo-tracker, batch) → runtime_ms, peak_allocated_mb, detection_summary
```

这样可以立即让资源预测器知道 batch 的影响，同时不把“单个视频内多帧 batch”误称为“跨 job batching”。

### B. 若要让调度器真的选择 batch

只有在定义了“哪些 ready 节点可以合批、合批后结果如何拆回、是否必须同一模型/输入形状兼容”后，才把动作扩成：

```text
(ready-node 或 compatible-node-group, GPU, batch_size)
```

动作的可行性必须同时检查显存、模型兼容、最大 batch 和 queue delay；每个 batch 的 runtime/peak memory 由真实测量曲线提供。现有 YOLO pilot 只能提供单节点多帧曲线，不能替代跨 job 合批测量。

### C. 预加载必须显式、真实计费

建议的最小状态/事件是：

```text
prefetch_start(model, gpu, load_ms, bytes)
prefetch_end(model, gpu)
resident(model, gpu)=true
```

prefetch 需要占用一个明确的 load/PCIe 资源（最初可按每 GPU 一个串行 load lane），并占用显存。执行节点命中驻留时不重复收费；被 eviction 后重新 load 时重新收费。`initial_residency_hint` 只保留为 cold/warm 对照，不当作 prefetch 结果。

### D. priority 不动，先把它作为控制变量

保留当前 `priority/normal` 二元语义和现有比例；不增加第三类 priority、不重写排序规则。新增的 batch/prefetch 实验只比较：

- 无 prefetch vs 显式 prefetch（真实 load）；
- 固定 batch 配置 vs batch-aware resource profile；
- 同一 binary priority 下的 completion、P95/P99、deadline miss、queue age、load/eviction。

## 5. 当前结论

- **Batch：已经测过，而且是有效测量。** 但测的是 YOLO 单工具动作的多帧 batch，不是 scheduler 的跨 job dynamic batching。
- **Priority：按你的要求保持不动。** 现有二元 priority 先作为控制变量。
- **Preload：应该做成真实动作并计入 load。** 当前 `initial_residency_hint` 不能冒充预加载。
- **Preemption：暂缓。** 没有 checkpoint/resume 语义前，模拟抢占会高估调度收益。

本轮只读检查和文献核验，没有修改代码、workload、模型或远端作业。

