# VideoSeek R0a 只读审计报告

日期：2026-08-13（Asia/Shanghai）  
远端：`root@connect.westc.seetacloud.com:12469`  
项目根目录：`/root/autodl-tmp/scheduler`  
审计性质：只读；未修改代码、未重训预测器、未运行新的 validation/test/C2 策略

## 1. 任务边界与结论

### 1.1 本轮目标

1. 冻结当前 C1 workload、B05、资源预测器及关键代码的路径、行数、schema 和 SHA256；
2. 核验 64 个 C1 视频在 workload、B05 和资源预测器之间的 split；
3. 核验 9,366 个节点与 B05/resource artifact 的连接覆盖；
4. 核验 `runtime_ms`、`load_ms`、`peak_memory`、resident/workspace 的真实含义；
5. 量化 raw-DAG 与静态线性 prefix 的不一致；
6. 审计 C2 未见视频候选；
7. 给出 R0b/R1 的进入门禁，不执行 R0b/R1。

### 1.2 总结论

R0a 证据收集已完成，但当前状态只能给出以下分级结论：

| 门禁 | 结论 | 原因 |
|---|---|---|
| 历史产物可否继续作为诊断证据 | **通过** | 9,366 节点连接完整，50-episode replay 完全一致，历史 workload 确有调度空间 |
| 可否直接把 B05/resource v1 接入正式调度 | **不通过** | split 不一致、decision-time prefix 不因果、B05 同 run future prototype、资源真值与预测耦合 |
| 可否进入 R0b/R1 修复阶段 | **有条件通过** | 先由用户验收本报告，并建立远端代码回滚基线；R0b/R1只使用 train/validation/smoke |
| 可否进入 R2/R3 正式预测/优化实验 | **不通过** | 统一 split、causal-v2、资源 v2 和 StateView/Truth 边界尚未完成 |
| 当前视频库能否建立 32+8 的 C2 | **不通过** | 340 个真实 VideoMME 视频已全部进入旧 predictor development 或 final holdout，未见候选为 0 |

最重要的判断是：当前问题不是“没有 gap”，而是历史 gap 的信息边界和执行真值尚不满足正式实验要求。应先完成 R0b/R1，再讨论 Optimizer-H、RL-H 和正式比较。

## 2. 远端环境与可回滚性

### 2.1 当前环境

- GPU：NVIDIA GeForce RTX 4080 SUPER；C1 实测物理总显存 `32230.812 MiB`；
- 当前磁盘：约 250 GiB 总量，约 33 GiB 可用；
- 资源预测器正确运行环境：`/root/miniconda3/envs/finetooling/bin/python`；
- 该环境：Python 3.10.20、NumPy 2.2.6、scikit-learn 1.7.2，并安装 LightGBM/XGBoost；
- `/root/miniconda3/bin/python` 当前是 Python 3.12.3，不能加载 `models.pkl`：scikit-learn 版本不一致且缺少 LightGBM；
- 远端项目根目录当前没有 `.git`；本地仓库虽然执行过 `git init`，但 `main` 没有任何 commit，项目文件基本均为 untracked。

### 2.2 回滚门禁

R1 会修改调度器核心代码。在没有基线 commit 的状态下直接修改，无法可靠回答“哪些行属于本轮修改”以及“如何回滚”。因此 R1 前必须二选一：

1. 推荐：在远端建立只跟踪代码、配置、测试和文档的 Git 基线，明确排除 `data/`、`results/`、模型权重和大日志；
2. 备选：生成带 SHA256 manifest 的代码快照包。

本轮没有擅自建立 Git、commit 或快照。

## 3. 冻结输入与产物血缘

### 3.1 C1 workload 与预测产物

| 文件（相对远端项目根目录） | 行数 | schema | SHA256 |
|---|---:|---|---|
| `results/processed/scheduling_future_v1_20260812/workload_c1_v1/job_templates_c1_v1.jsonl` | 768 | `job-template-v0.2` | `9ea458b0a2d2835813f241e48e21c0b0ab55bcd2dc1340bddf10ab43daeae45c` |
| `.../workload_train_c1_v1.jsonl` | 20,000 | `workload-episode-v0.2` | `2f1dd100519012a532c0baa4da454800e004437eea10964e3c92c495064913eb` |
| `.../workload_validation_c1_v1.jsonl` | 1,000 | `workload-episode-v0.2` | `63536b717e86ff09995a7882f34a5a67df4817e6eaa439f14b6aa11ed764902b` |
| `.../workload_test_c1_v1.jsonl` | 6,750 | `workload-episode-v0.2` | `a2fd8bd57a6647358562d55c17c1f54dd6f8b4d6d03bcae616637a9c47a205db` |
| `results/processed/scheduling_future_v1_20260812/prediction_artifacts/b05_node_h1.jsonl.gz` | 9,366 | `scheduling-future-b05-node-v1` | `628646b0dba706a797625746707c9fc5bd2e61cdc8e9de5dcda159fac47ee646` |
| `.../b05_future_h3.jsonl.gz` | 9,366 | 同上 | `e9e28c6b28837fb61ff3b37b4d7946e46908beedce01de0efd656d579852f608` |
| `.../b05_future_h5.jsonl.gz` | 9,366 | 同上 | `05f0f8f816e2ed8a9d17359b3a51fba3621c40e023c9a15bb4136fe702825031` |
| `.../resource_predictions_v1.jsonl.gz` | 9,366 | `scheduling-future-resource-node-v1` | `5653ee725ed9711701dbd85245f7f469ccd2997fddf08d48495a13a6633a473f` |

模板直接输入：

| 文件 | 行数 | SHA256 |
|---|---:|---|
| `results/processed/workload_v0_2_smoke/snapshot768.jsonl` | 768 | `7e56d3617d6ffb96d8a384eb1b5522f68ea0183ef75711fc37ade67d20788646` |
| `results/processed/compute_events_core_resource_fixed_20260804.jsonl` | 10,134 | `7f8227e0b95b1cd219f1af6b94119ba78c136334d23e1c8fc0ba1a1286dc4c05` |
| `results/processed/split_manifest_v0_1.json` | 300 videos | `b49ecf21824f9cf7d901b28cda60b907cf9ca79064790a3eb0fe92c9111c1f4d` |
| `results/processed/behavior_nn_v1_r2/final_holdout/split_manifest.jsonl` | 40 videos | `69a51de21e9adb430d3973dd86f1fbfbd64790f0eaf1a28b2a99c917b1d22972` |

### 3.2 资源预测器产物

| 文件 | schema | SHA256 |
|---|---|---|
| `results/processed/resource_predictor_v1_final_q99_20260811/models.pkl` | pickle bundle | `612aeba62b79436fd2e488f0d985d69b678a745d2e242e934b406b4fbe4645ff` |
| `.../resource_predictor_report.json` | `resource-predictor-v1.0` | `49697fd5edc91517f9c443a4887feab988dd4d85b439d8bf1557929c397d21a3` |
| `.../feature_schema.json` | `resource-predictor-features-v1.1` | `b0186228e65ae43f903dd6fe2fe0550635202aca91264cf9b663178666cb0d05` |
| `.../scheduler_resource_contract.json` | `scheduler-resource-contract-v1` | `9a1f6d4efc963fed8cdceee3a8091311f62edef2a0da663b00c66f3eb5755c14` |

资源预测器入口唯一定位为：

```text
tracing/analysis/resource_predictor_v1.py
```

原始 shell command 没有保存在输出目录、`/root/.bash_history` 或 `/root/.zsh_history`。下列命令是根据 CLI 参数和报告中六个唯一输入哈希重建的等价命令，不能冒充历史原始命令：

```bash
PYTHONPATH=. /root/miniconda3/envs/finetooling/bin/python -B \
  tracing/analysis/resource_predictor_v1.py \
  --dev-compute results/processed/trace_enrichment_core_fixed_20260804/compute_events_v0_1.jsonl \
  --dev-split results/processed/split_manifest_v0_1.json \
  --holdout-compute results/processed/behavior_nn_v1_r2/final_holdout/enrichment/compute_events_v0_1.jsonl \
  --holdout-split results/processed/behavior_nn_v1_r2/final_holdout/split_manifest.jsonl \
  --dev-metadata results/processed/neural_prefix_dataset_v0_9_options_sourcefix_20260805/prefixes_visual_context.jsonl \
  --holdout-metadata results/processed/behavior_nn_v1_r2/final_holdout/enrichment/prefix_samples_v0_1.jsonl \
  --output-dir results/processed/resource_predictor_v1_final_q99_20260811 \
  --seed 42
```

哈希唯一对应的训练契约：

- development：8,301 rows / 64 videos；train 6,771、validation 911、旧 test 619（最终 fit 排除旧 test）；
- final artifact fit：`development_train + development_validation`；
- final holdout：1,380 rows / 40 videos；
- 目标：`runtime_ms`、`load_ms`、`peak_memory_inclusive_mb`；
- 不支持：queue、strict OOM、stall risk、resident weight separation。

这与 C1 workload 的 48/8/8 video split 不是同一训练/评估契约，因此只能复用流程和结构，不能直接复用该 checkpoint 作为正式 C1 predictor。

### 3.3 关键代码基线哈希

| 文件 | 行数 | SHA256 |
|---|---:|---|
| `tracing/analysis/workload_v02_simulator.py` | 810 | `6c820e874cfe393ee756875809b887eb020b3851263277bb094b25b28e9c5e9b` |
| `tracing/workloads/build_workload_v02.py` | 807 | `01bd4bb9ec98ed5d2c5c6fa7cdca9dee5a6cf6885ca53543838ae6543d081fe5` |
| `tracing/analysis/resource_predictor_v1.py` | 836 | `9a064f59923c04fc44f5ec0117586765a272e69b657979b9bab06bf33e5297cb` |
| `scripts/scheduling_future_b05_artifacts.py` | 705 | `2a54cb5db6ca2ea64a3c2fe27fcfffd65884eea821c137874341e51f93c2d46f` |
| `scripts/build_c1_workload.py` | 111 | `8ec8223798a576a492fd1a50ce40e08ffcc38b409f3119d10c058ec77098dc96` |
| `scripts/build_c1_resource_profile.py` | 127 | `53bfd42a201dc03fad8c1de94de15086b749f55eb0a47ca64e0bcb49ca3140c5` |

## 4. C1 数据与 split 审计

### 4.1 C1 内部隔离

内部视频隔离通过：64 个视频均只属于一个 workload split，没有 mixed video。

| C1 split | 视频 | 模板 | 节点长度中位数 | 节点长度 P95 |
|---|---:|---:|---:|---:|
| train | 48 | 566 | 14 | 16 |
| validation | 8 | 94 | 14 | 15 |
| test-retrospective | 8 | 108 | 14 | 15 |

baseline 分布：

| split | langgraph_react | star | st_fixed |
|---|---:|---:|---:|
| train | 235 | 235 | 96 |
| validation | 39 | 39 | 16 |
| test | 46 | 46 | 16 |

### 4.2 C1 与旧 B05/resource split 的交叉表

模板级：

| C1 \ 旧 split | train | validation | test |
|---|---:|---:|---:|
| train | 454 | 64 | 48 |
| validation | 94 | 0 | 0 |
| test | 64 | 34 | 10 |

视频级：

| C1 \ 旧 split | train | validation | test |
|---|---:|---:|---:|
| train | 39 | 5 | 4 |
| validation | 8 | 0 | 0 |
| test | 5 | 2 | 1 |

结果：

- 768 个模板中只有 464 个落在同名 split，304 个需要重新归属；
- 64 个视频中只有 40 个落在同名 split，24 个需要重新归属；
- C1 validation 的 94 个模板全部被旧 B05/resource 当作 train；
- C1 test 的 108 个模板中，98 个曾属于旧 train/validation；
- 旧 B05/resource checkpoint 不能作为 C1 validation/test holdout predictor。

### 4.3 domain 与时长分布

| split | 时长中位数 | P95 | domain（视频数） |
|---|---:|---:|---|
| train | 290.80s | 840.50s | Knowledge 27；Sports 7；Artistic 6；Film & TV 6；Life Record 2 |
| validation | 102.74s | 375.15s | Knowledge 6；Life Record 2 |
| test | 102.37s | 331.01s | Knowledge 6；Life Record 2 |

这不是泄漏，但存在明显分布偏移：validation/test 比 train 更短，且缺少 train 中三个内容域。正式报告必须把这一点作为 C1 仅代表给定 8 个 test 视频的限制；C2 应按冻结的元数据分层规则补足外部泛化证据。

### 4.4 model stack 标记

模板原始字段：

- 明确 Stack A：96；
- 明确 Stack B：272；
- `model_stack_id=unknown`：400。

400 个 unknown 模板的 GPU 模型集合全部只有 `Qwen3-VL-8B-Instruct`，可以在 R0b 用可审计规则归为 legacy Stack A，但不能在本轮只读审计中改写原文件。正式 registry 必须保留 `original_model_stack_id` 与 `normalized_model_stack_id` 两列，不能静默覆盖。

## 5. 9,366 节点连接与 B05 审计

### 5.1 连接覆盖

| 检查 | 结果 |
|---|---:|
| template node_id | 9,366，重复 0 |
| B05 H1 rows | 9,366，missing 0，extra 0，重复 0 |
| resource rows | 9,366，missing 0，extra 0，重复 0 |
| B05 source_event_id 与模板 source_event_id 不一致 | 0 |

因此“连接键覆盖”通过。

### 5.2 B05 v1 仍不可正式使用

`family_source`：

| 来源 | 行数 |
|---|---:|
| `observed_tool_prefix` | 3,120 |
| `same_run_tool_template` | 4,749 |
| `unavailable` | 1,497 |

代码证据位于 `scripts/scheduling_future_b05_artifacts.py:595-620`：当前 cutoff 没有 tool sample 时，会回退到同一 run 的第一条 tool sample；rollout 还读取同一 run 的 future cutoff map。旧 label-mutation audit 没有覆盖“删除 cutoff 后同 run 行，预测是否不变”。因此 v1 仅能作为 diagnostic artifact；正式版必须重建 `causal-v2`。

## 6. raw-DAG 与线性 prefix

### 6.1 实测差异

- 768/768 模板含分叉或汇合；
- GPU 节点 6,693 个；其中 1,828 个（27.312%）并不以所有更早 `sequence_index` 节点为祖先；
- 受影响模板 636 个；
- 现有模板 predecessor edge 共 17,402 条；
- proposed causal-chain edge 为 8,598 条；
- 9,366 个节点中 7,774 个 predecessor 集合会变化，涉及 768/768 模板；
- overlay 将删除 12,874 条 edge、增加 4,070 条 edge。

当前 raw parent step 会把父 step 内的多个事件全部映射成 predecessor，产生稠密且不等价于原始串行执行的 DAG。与此同时，B05/resource artifact 又按完整 `sequence_index` 线性 prefix 构造特征。两者同时使用会造成 ready 节点读取尚未完成 sibling 的信息。

### 6.2 proposed overlay

R0b 应新建而不覆盖：

```text
job_templates_causal_chain_v1.jsonl
```

规则：每个模板按 canonical `sequence_index` 排序；首节点无 predecessor；其余节点仅以前一个节点为 predecessor。保留原始 predecessor 和 provenance 作为旁路审计字段。主实验先使用 causal-chain；raw-DAG 只作为后续扩展，不与静态 predictor 混用。

## 7. 资源字段语义审计

### 7.1 runtime 与 load

代码链路：

- Qwen worker 在模型加载时单独测 `model_load_ms`，生成时单独测 `inference_ms`；
- parent 首次调用 `_ensure_process()`，因此首次工具/API `runtime_ms` 的墙钟区间包含 worker 启动、模型加载和推理；
- YOLO 的 BaseTool 墙钟同样包围“子进程启动 + 模型加载 + 推理”；
- `videotool_phase1.py:430-480` 将完整墙钟写入 `runtime_ms`，`538-563` 和 `620-665` 把 load/inference 作为子项写入 resource。

数值交叉验证：

- 所有 raw traces 中正 load 行 3,463；`runtime < load` 为 0；
- 其中同时有 inference 的行 3,149；`runtime < load + inference` 为 0；
- C1 template 正 load 节点 1,177；`runtime < load` 为 0；
- `runtime-load` 中位数 8,974.10ms，P95 20,006.52ms。

所以当前测量含义是：

```text
首次冷运行 runtime = load + inference/服务时间 + 启动及协议开销
后续暖运行 runtime = inference/服务时间 + 协议开销
```

`Node.compute_ms = runtime-load` 的方向合理，但当前 simulator 在冷加载时使用：

```text
load = max(真实 node.load_ms, 预测 load_p50_ms)
duration = (真实 runtime - 真实 load) + load
```

见 `workload_v02_simulator.py:649-659`。这会让预测值改变 execution truth；预测高估时，模拟中的真实执行时间被人为拉长。正式 R1 必须改成：预测只进入 SchedulerStateView；执行 duration 只读冻结 truth provider。

### 7.2 peak memory、resident 与 workspace

资源预测器 schema 明确定义：

```text
peak_memory_inclusive_mb = max(peak_reserved_mb, peak_allocated_mb)
```

它是 resident + workspace 的 inclusive label。可是 workload builder 把这个值命名为 `workspace_peak_mb`；simulator 再减去来自 NVML C1 plateau 的 `resident_model_mb`。这混用了 PyTorch allocator peak 与 NVML plateau 两种测量口径。

当前直接后果：

| 模型 | GPU nodes | trace inclusive peak 低于 resident | predicted peak ≤ resident |
|---|---:|---:|---:|
| Qwen3-VL-8B | 4,357 | 131 | 131 |
| Qwen3-4B | 1,735 | 1,723 | 1,735 |
| Qwen2.5-VL-3B | 595 | 323 | 323 |
| YOLO11x | 6 | 6 | 6 |

因此当前相减后：

- Qwen3-4B 的 P95 incremental workspace 为 0；
- Qwen2.5-VL-3B 的全部有测量节点 incremental workspace 为 0；
- YOLO11x 的全部节点 incremental workspace 为 0；
- 这不是“模型真的没有 workspace”，而是口径不一致和 conservative resident 大于 allocator peak。

正式 v2 必须统一成一种可加和账本，例如：

```text
resident_model_mb：同一测量口径下的常驻权重/上下文
active_workspace_mb：同口径 active peak - resident
admission_total_mb：当前 GPU 所有 resident + 当前 node active_workspace
```

不能把 inclusive peak 再与 target resident 相加，也不能把 NVML resident 直接从 PyTorch peak 相减后把负值静默截成 0。

### 7.3 静态 cold/warm 不是调度状态

`scripts/scheduling_future_resource_artifacts.py:74-154` 按模板线性顺序用 `seen_models` 生成静态 `cold_warm`，并把此前线性行的 runtime 累积进 prefix。它既不是某个 GPU 的实际 cache 状态，也不适用于 raw-DAG sibling ready 情况。所有 9,366 个节点的 input scale 还都是 unknown。

正式 resource v2 应：

1. 预测 warm service runtime；
2. 单独输出 model-level cold-start load penalty；
3. 在调度时由目标 GPU 的真实 resident set 决定是否加 load；
4. 携带 dispatch 前已知的 frame/image/batch/input token scale；
5. queue 由 simulator 产生，不让 resource predictor 猜；
6. OOM/stall 在没有受控标注前继续标记 unsupported。

## 8. C1 容量与 replay

### 8.1 容量

- workload topology 使用 `32760 MiB`；
- C1 实测卡容量 `32230.812 MiB`；
- 差值 `529.188 MiB`。

R0b capacity registry 必须把 32GB class 映射到实测 `32230.812 MiB`。`24576 MiB` 只能标为未在本机物理复现实测的 topology/extrapolation，不能伪装成第二类真实 GPU。

### 8.2 deterministic replay

现有 50-episode smoke 与 replay 完全一致：

- `simulation_results.jsonl` SHA256 均为 `715ba185b6325d9c1846e6bd3a140840bf1bf16dea80289dc7690e82caa88d6d`；
- `simulation_report.json` SHA256 均为 `931e9c9b42de113dc7e5ce51f146d4403d76ad046b27cbc3fc23a068c5fbe4a4`。

这证明旧引擎确定性可复现，但不证明其信息边界或资源语义正确。

## 9. 历史调度空间与仍缺的指标

### 9.1 C1 validation 1,000 episodes

历史结果：

| policy | mean completion | deadline miss |
|---|---:|---:|
| Myopic | 199,097.66ms | 11.856% |
| historical oracle | 155,534.70ms | 4.653% |
| Round Robin | 200,579.86ms | 12.866% |

Myopic 相对 historical oracle gap 为 28.01%。该结果证明 workload 有调度空间，但不能作为正式结果，因为 historical oracle 读取完整 suffix，且 simulator 的 load/memory truth 仍受预测值影响。

### 9.2 历史 135-cell × 50 test/replay

旧 6,750-episode 结果中：

- Myopic mean completion：178,514.77ms；
- historical oracle：140,675.54ms；
- 相对 gap：26.90%；
- 135/135 cells 的 gap 均为正；
- cell gap 最小 3.25%，中位数 24.70%，P95 41.17%，最大 46.57%；
- 按压力档平均 gap：0.50→13.69%，0.70→17.98%，0.85→22.47%，0.95→29.11%，1.05→38.79%。

压力越高，历史 gap 越大，说明资源压力有效。但这仍是旧 input/truth contract 下的 retrospective diagnostic。

### 9.3 model stack 关联，不是因果贡献

旧 6,750 episodes 的 248,400 个 jobs 中，按实际 GPU 模型集合归类：Stack A 179,820，Stack B 68,580。每个 episode 都同时包含 A/B，B 占比只有 25.0%–29.7%，不存在 A-only 或 B-only 对照。

- B 占比 ≤25% 的 episodes：平均相对 gap 16.61%，97.45% 为正；
- B 占比 >25% 的 episodes：平均相对 gap 31.33%，99.49% 为正；
- B 占比与相对 gap 的 Pearson 相关约 0.429。

这只说明 gap 与 Stack B 比例存在关联，不能说明“Stack B 贡献了多少”。R0b preliminary 应增加固定 mix 或 leave-one-stack 诊断，但只能在 validation/smoke 上进行。

### 9.4 decision opportunity rate 暂时不可恢复

当前 1,000-episode 和 6,750-episode 持久化文件只有 episode summary；full decision events 已未保留。报告只保存 dispatch 总数，没有每次决策的：

```text
candidate_actions
available_gpu_slots
feasible node×GPU mappings
```

因此不能从现有结果科学计算用户要求的 decision opportunity rate。本轮没有为了补数而重新运行 validation/test。R1 必须把该计数加入 summary-only 聚合，随后只在 10/50 smoke 和 validation preliminary 上补测。

## 10. C2 未见视频审计

### 10.1 当前真实库存

`data/phase3/public/videos/videomme`：

- `.mp4` 文件 348；
- 其中真实视频 340；
- 另有 8 个 `._*.mp4`，每个仅 163 bytes，是 macOS AppleDouble 元数据，不是视频。

旧 predictor registry：

- development：300 个视频；
- final holdout：40 个视频；
- 二者交集 0；
- 合计正好覆盖当前 340 个真实 VideoMME 视频。

与 `results/raw/**/run_manifest.json` 交叉后也没有产生新的可用候选。此前“351 unique stems、排除 C1 64 后剩 287”只是按文件名排除 C1 的粗筛，不能代表 predictor-unseen。

### 10.2 结论与需求

当前可用于独立 C2 的真实 VideoMME 视频：**0**。

要建立目标 32 主样本 + 8 替补，必须新增至少 40 个从未进入以下任何集合的真实视频：

1. 300-video development registry；
2. 40-video final holdout registry；
3. C1 64-video workload；
4. 所有 raw trace、feature、resource、B05 registry；
5. content hash / perceptual hash 近重复集合。

按现有 340 个视频平均约 92.75 MiB 估算，40 个新视频本体约 3.6 GiB；当前约 33 GiB 可用，视频下载本身可承受，但采集 traces、临时帧和日志仍需单独预算。正式下载前应先冻结来源、ID、分层元数据和清理策略。

## 11. R0b/R1 的具体进入门禁

### 11.1 R0b（只生成派生文件，不覆盖历史）

在用户验收本报告并授权写入后：

1. 建立远端代码基线与排除规则；
2. 新建 `video_split_registry_v1`：保留原 split、统一 C1 48/8/8、normalized stack、C2=`pending_new_video`；
3. 新建 `gpu_capacity_registry_v1`：32760 class→32230.812 measured；24576→extrapolation；
4. 新建 causal-chain overlay，保留 raw predecessor 旁路字段；
5. 生成 train-only stack/tool/resource prototype coverage，不生成正式预测 artifact；
6. 将所有 8 个 `._*.mp4` 标记 invalid，不删除；
7. 只做 schema/hash/coverage/determinism 验证。

### 11.2 R1（统一信息边界与执行真值）

1. 在唯一事件引擎中引入 `SchedulerStateView`、policy callback、`ExecutionTruthProvider`；
2. RR/Myopic 通过 adapter 复现旧行为；
3. prediction 只能影响排序、准入估计和规划，不能改 truth duration/memory；
4. runtime 分成 warm service truth 与 cold load truth；cold/warm 由目标 GPU resident set 动态决定；
5. memory 使用统一可加和口径；
6. summary-only 聚合 decision opportunity、fallback、unsupported/unknown、capacity violation；
7. 先跑 10/50 smoke parity；不运行 C1 test-retrospective 或 C2。

### 11.3 R2 及以后仍禁止的事项

在以下全部通过前，不进入正式预测/优化/RL：

- C1/B05/resource/workload 三方 split 完全对角；
- causal-v2 在 current/post-current mutation 和 deletion audit 下不变；
- resource v2 只用 C1 train fit，validation 只选参数；
- 预测与执行真值隔离测试通过；
- decision opportunity rate 在修复后的 validation 仍非零；
- C2 新视频清单、content/pHash 去重与 32+8 registry 冻结。

## 12. 本轮没有做的事

- 没有修改任何 `.py`、config、workload、prediction artifact；
- 没有训练 B05/resource/RL；
- 没有运行新的 1,000 validation、6,750 C1 test 或 C2；
- 没有读取 C2 调度结果；
- 没有删除 AppleDouble 文件；
- 没有初始化 Git 或 commit；
- 没有创建 `PROJECT_STATE.md`（项目当前不存在该文件，需另行授权）。

## 13. 验收建议

建议将本报告作为 R0a 唯一验收件。若用户确认结论，下一步不是直接实现 Optimizer-H，而是按顺序执行：

```text
远端代码回滚基线
→ R0b split/capacity registry + causal-chain overlay
→ R1 StateView/Truth/Policy 重构与 10/50 parity
→ 修复后 validation preliminary（含 decision opportunity）
→ 再决定是否进入 R2 resource-v2 / B05 causal-v2
```
