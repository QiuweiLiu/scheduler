# Phase 2：小样本动态性与资源画像

## 1. 目标与边界卡

本阶段对应 `CODEX_WORK_GUIDE.md` 的 Phase 2：先收集 20–50 条 preliminary trace，
检查路径动态性（H1）和资源异构性（H3），再决定是否进入 100–300 条正式初始样本。
本阶段不训练预测器、不搭模拟器、不做真实多 GPU 调度，也不把替代模型结果表述为论文精度复现。

用户已明确：热成像视频不进入主线；Qwen3-VL-8B 和 YOLO 必须真正介入；API key 不轮换，
但只通过临时进程环境传递，不写入命令行、文件、trace、manifest 或日志。

## 2. 公开数据策略

| 来源 | 官方入口 | 代码/数据事实 | 本阶段用法 |
|---|---|---|---|
| LongVideoBench/LVBench | `https://huggingface.co/datasets/longvideobench/LongVideoBench` | LongVideoBench 官方数据卡提供 `lvb_val.json`、视频和字幕归档；VideoTool `lvb` loader 读取 `videos/<video_path>`、`subtitles/<subtitle_path>` 和多选题 | 复用远端已有的公开视频 ID `wgBlACG927Y` 及 Phase 0 已记录问题；重复运行只作为 preliminary repetition，明确不当作不同题目 |
| VideoMME | `https://huggingface.co/datasets/lmms-lab/Video-MME` | VideoTool loader 读取 Parquet 的 `videoID/question/options/answer`，视频按 `videoID.mp4` 查找 | 已下载 8 个公开视频和对应官方 metadata 题目；每个视频取 1 道题进入跨媒体 preliminary，不伪造视频与题目的配对 |
| NExT-QA | `https://github.com/doc-doc/NExT-QA` | VideoTool loader 读取 CSV 与按视频 ID 查找的 MP4 | 作为后续公开子集候选；当前远端直连下载超时，暂不把未落盘媒体算入样本 |

远端对 Hugging Face 的 `curl` 直连在 2026-07-31 超时，因此本阶段不下载完整归档，
不绕过公开数据的许可/访问条件。现有可运行媒体的哈希、来源和问题已记录在
`configs/phase0_videoseek_qwen3_vl_plus.yaml`。

当前 preliminary manifest：

```text
configs/phase2_public_lvbench_repeated.jsonl
```

它包含 1 个公开视频/问题记录、7 次重复、3 条 baseline，即 21 个有明确标签的运行槽位。
运行 ID 中的 `_rNN` 是重复编号，不代表 21 个独立 benchmark 样本。

## 3. 真实模型与资源记录

### Qwen3-VL-8B

远端路径：`/root/autodl-tmp/Qwen3-VL-8B-Instruct`。

`tracing/collectors/qwen3_vl_worker.py` 在 `/root/miniconda3/envs/finetooling` 中单独启动，
模型只加载一次并通过 JSONL 请求处理 `ImageQA/ImageGridQA`。父进程仍运行在轻量
`videotool_phase1` 环境，不安装第二份 Torch/Transformers。每次视觉 action 记录：

- 冷启动 `load_ms` / worker 启动时间；
- warm inference 时间 `qwen_inference_ms`；
- `peak_allocated_mb`、`peak_reserved_mb`；
- 帧数和 `qwen_model_resident`。

远端单帧 probe 已验证：模型加载约 12.6 秒、显存约 16.3 GiB、生成约 2.9 秒；
多帧 Phase 2 pilot 中同样产生非空显存和推理时间字段。

### YOLO11x

模型路径：`/root/autodl-tmp/upload/models/yolo11x.pt`，Python：
`/root/miniconda3/envs/finetooling/bin/python`。`YOLOTracker` 调用实际 Ultralytics
模型，按 action 串行加载并记录 `load_ms`、`yolo_inference_ms`、显存峰值、实际 batch
和检测摘要。默认仍为单帧 batch=1；多 batch 实验通过显式 `--yolo-batch` 与
`--yolo-preobserve` 开关启用。
它不是占位字符串；若没有 `--yolo-model`，trace 会明确记录跳过原因。

## 4. 批量运行与统计

批量入口：

```bash
python -m tracing.collectors.videotool_phase2_batch \
  --manifest configs/phase2_public_lvbench_repeated.jsonl \
  --output-root <phase2-output> \
  --videotool-root <videotool-checkout> \
  --planner-mode api \
  --qwen-model /root/autodl-tmp/Qwen3-VL-8B-Instruct \
  --qwen-python /root/miniconda3/envs/finetooling/bin/python \
  --yolo-model /root/autodl-tmp/upload/models/yolo11n.pt \
  --yolo-python /root/miniconda3/envs/finetooling/bin/python
```

统计入口：

```bash
python tracing/analysis/phase2_trace_stats.py --root <phase2-output>
```

输出包含路径长度、唯一路径、工具频次、转移频次、条件下一动作熵、失败/重试数、
运行/API/本地时间变异系数、加载时间和显存峰值样本数。H1/H3 字段是门槛观察值，
不是因果结论：至少需要更多公开视频/任务后才能决定 Phase 3。

## 5. 已完成验证

### 单条模型介入 smoke

| baseline | run | 结果 |
|---|---|---|
| ST fixed | `wgBlACG927Y_st_fixed_1785479268` | Qwen3-VL-8B `ImageQA` 成功，validator `VALID` |
| LangGraph/ReAct | `wgBlACG927Y_langgraph_react_1785479381` | Qwen3-VL-8B 两次视觉调用 + YOLO11n 一次真实检测，validator `VALID` |

该 ReAct trace 的实测资源字段包括 Qwen 冷启动约 6908 ms、视觉推理约 4634/4048 ms、
峰值约 19.1 GiB，以及 YOLO 加载约 87 ms、推理约 958 ms；这证明 8B/YOLO 已真实介入，
不是 metadata placeholder。

### API pilot

输出目录：

```text
/root/autodl-tmp/scheduler/results/raw/phase2_pilot_api
```

3 条 API trace（ST fixed、STAR、LangGraph/ReAct）均 `VALID`，路径各不相同，动作数为
2–9，统计器报告：

```text
runs=3, unique_paths=3, next_action_entropy=1.001304 bits
local_runtime_ms_cv=4.097098, load_ms_cv=2.236517
H1_path_dynamicity_observation=pass
H3_resource_heterogeneity_observation=pass
```

这只是 pilot，不足以直接进入 Phase 3；批量 21 槽位完成后必须重新运行 validator 和统计器。

### Preliminary 批次（已完成）

输出目录：

```text
/root/autodl-tmp/scheduler/results/raw/phase2_prelim
```

批次已正常退出，`phase2_batch_summary.jsonl` 逐条记录 21 个 run；每个 run 都有独立的
`trace.jsonl/run_manifest.json/run_status.json`。最终验收：

```text
21/21 success；21/21 validator VALID；0 个凭据命中
unique_paths=14，path_ratio=0.666667
action_count mean/min/max=5.142857/2/9
next_action_entropy=1.751348 bits
retry_count=36，error_count=36（均为 STAR planner 的真实解析失败→一次 retry；run 仍成功）
local_runtime_ms_cv=3.987209，load_ms_cv=2.042224，peak_allocated_mb_cv=0.173488
H1_path_dynamicity_observation=pass
H3_resource_heterogeneity_observation=pass
```

按 baseline 的路径现象：ST fixed 的 7 次重复均为固定两步；STAR 的 7 次重复产生 7 条不同
的 7-action 路径，并发生真实解析失败/重试；LangGraph/ReAct 的 7 次重复动作数为 4–9，
其中至少一条实际调用了 `yolo-tracker`。这说明“动态”来自决策输出和约束分支，而不是人为
给固定流程加随机节点。

这组旧的 21 条仍全部来自同一公开视频/问题，因此它本身不能证明 H1 的跨内容相关性；后续
Video-MME 跨视频批次的结果见下节。当前两批结果都只作为 preliminary 轨迹证据，不能直接声称
Phase 3 或论文调度收益。

### Video-MME 跨媒体 preliminary（已完成）

已将 8 个已下载并上传的 Video-MME 公公开视频各取 1 道官方 metadata 题目，建立以下可复查
manifest：

```text
configs/phase2_public_videomme_prelim.jsonl
configs/phase2_public_videomme_remaining_flash.jsonl
configs/phase2_public_videomme_remaining_plus.jsonl
```

其中，首轮错误模型名 `openai/qwen3-vl-plus` 的 21 个 error 目录保留在
`/root/autodl-tmp/scheduler/results/raw/phase2_videomme_prelim`，不计入统计。修正为此前
成功 pilot 使用的 `qwen3-vl-plus` 后，保留 4 条成功 trace；为控制 API 配额，剩余主批次改用
`qwen3-vl-flash`，完成 17 条成功 trace。两个未完成目录只作为中断证据，不计入有效样本：

```text
/root/autodl-tmp/scheduler/results/raw/phase2_videomme_prelim_retry/N1cdUjctpG8_star_1785488753773
/root/autodl-tmp/scheduler/results/raw/phase2_videomme_flash/X30VlGh3HwQ_langgraph_react_1785490403533
```

原计划 24 个槽位中，21 条成功；`N1cdUjctpG8` 的 STAR 目录和 `X30VlGh3HwQ` 的 ReAct
目录在配额/耗时控制下中断，`N1cdUjctpG8` 的 ReAct 槽位尚未启动。有效样本汇总目录及统计结果为：

```text
/root/autodl-tmp/scheduler/results/raw/phase2_videomme_valid_20260731
/root/autodl-tmp/scheduler/results/raw/phase2_videomme_valid_20260731/phase2_trace_stats.json
```

验收结果：

```text
21/21 success；21/21 validator VALID；0 个凭据命中
覆盖 8 个 Video-MME 视频；st_fixed=8、star=7、langgraph_react=6
模型分布：qwen3-vl-plus=4、qwen3-vl-flash=17
unique_paths=14，path_ratio=0.666667
path_length mean/min/max=4.571429/2/9
next_action_entropy=1.747605 bits
retry_count=6，error_count=6（真实 planner 解析失败及其 retry 链）
runtime_ms_cv=4.637008，local_runtime_ms_cv=5.108826
api_wait_ms_cv=9.901454，load_ms_cv=1.885357，peak_allocated_mb_cv=0.289646
H1_path_dynamicity_observation=pass
H3_resource_heterogeneity_observation=pass
```

该批次已满足 20 条 preliminary 最低数量，并首次提供跨 8 个公开视频的内容覆盖。由于模型
分组是为控制免费 API 配额而做的，正式扩展时必须把 `model_name` 作为实验条件记录，避免把
模型差异误解释为纯内容差异。尚未运行 `qwen-plus` 分组，以免在已有 21 条有效样本后继续
消耗配额。

### Formal fixed-model cohort（已完成 96 条有效运行）

为消除 preliminary 阶段的模型混杂，正式 cohort 固定使用 qwen3-vl-flash，并采用
8 个 Video-MME 视频 × 3 个 baseline × 5 个重复的计划，共 120 次尝试。实际结果为：

120 attempts；96 success；24 quota-error
有效分布：st_fixed=32、star=32、langgraph_react=32
每个视频 12 条有效运行；planner/API model=qwen3-vl-flash

第 5 个重复批次开始时免费 API 配额耗尽，24 个目录保留为真实错误证据，不计入有效统计；
不重试、不切换付费额度、不把失败目录伪装成成功。有效 cohort 的可复查产物为：

/root/autodl-tmp/scheduler/results/raw/phase2_videomme_formal_flash_120
/root/autodl-tmp/scheduler/results/raw/phase2_videomme_formal_valid_20260731
/root/autodl-tmp/scheduler/results/raw/phase2_videomme_formal_valid_20260731/phase2_videomme_formal_stratified_stats.md
/root/autodl-tmp/scheduler/results/raw/phase2_videomme_formal_valid_20260731/phase2_videomme_formal_stratified_stats.json

96 条有效运行全部通过 validator；固定模型后正式 cohort 的 unique_paths=52、
path_ratio=0.541667，下一动作熵为 2.070288 bits，runtime_ms_cv=2.455804、
local_runtime_ms_cv=2.796424、api_wait_ms_cv=5.794226、load_ms_cv=2.060967、
peak_allocated_mb_cv=0.235009，H1/H3 仍为 pass。
由于配额耗尽，正式样本暂止于 96 条，尚未达到指导书规定的 100 条最低正式样本；
该处是配额耗尽当时的暂停记录；用户随后接受 96 条作为阶段性正式样本并启动了离线 Phase 3。

### Phase 3 H2 preliminary validation（已执行）

在用户接受 96 条有效 cohort 作为阶段性正式样本后，已对其执行离线 Phase 3 验证，
不再调用 API。评估使用 leave-one-video-out，8 个视频的所有重复运行固定在同一折；
共展开 601 个 prefix 样本（505 个下一动作目标、96 个 `__END__` 目标），问题文本
96/96 可从正式 manifest 复原。

结果如下：

```text
markov2:        top1=0.517471, nll=1.444317, video_macro_top1=0.519129
prefix_only_nb: top1=0.599002, nll=1.342965, video_macro_top1=0.600345
prefix_task_nb: top1=0.625624, nll=1.409401, video_macro_top1=0.626808
remaining_steps: static_mae=1.860, markov1_mae=1.438
remaining_runtime: static_mae=7420.8ms, markov1_mae=5306.1ms
H2_observation=mixed_evidence
```

`prefix_only_nb` 已明显优于 Markov，说明执行前缀本身包含未来动作信号；加入问题词后
Top-1 又提升 0.026622，但 NLL 增加 0.066435、Top-3 下降 0.019967。因此当前只能
确认“前缀可预测”，不能确认“内容特征带来稳定且校准良好的额外收益”。完整方法与
按 baseline 分解见 `docs/phase3_predictor_validation.md`；Phase 4 调度收益实验暂缓。

### YOLO11x batch sensitivity pilot（2026-08-02，已完成）

为验证“大 batch 是否真的改变动态轨迹”，先在 1 条视频上修复并验证了末端抽帧边界，
随后使用 8 个 Video-MME 视频 × `star/langgraph_react` 两个 local-Qwen 动态 baseline，
对 `batch=1/2/4/8/16/32/64` 各跑 16 条，共 112 条新 trace。每条 trace 先执行真实的
YOLO11x 多帧检测，再把检测摘要交给 Qwen3-VL-8B planner；旧 96 条 cohort 未覆盖。

远端产物：

```text
/root/autodl-tmp/scheduler/results/raw/phase3_yolo11x_batch_pilot8_b01_clean
/root/autodl-tmp/scheduler/results/raw/phase3_yolo11x_batch_pilot8_b02_clean
/root/autodl-tmp/scheduler/results/raw/phase3_yolo11x_batch_pilot8_b04_clean
/root/autodl-tmp/scheduler/results/raw/phase3_yolo11x_batch_pilot8_b08_clean
/root/autodl-tmp/scheduler/results/raw/phase3_yolo11x_batch_pilot8_b16_clean
/root/autodl-tmp/scheduler/results/raw/phase3_yolo11x_batch_pilot8_b32_clean
/root/autodl-tmp/scheduler/results/raw/phase3_yolo11x_batch_pilot8_b64_clean
/root/autodl-tmp/scheduler/results/processed/phase3_yolo11x_batch_pilot8_batch_summary.json
/root/autodl-tmp/scheduler/results/processed/phase3_yolo11x_batch_pilot8_trajectory_comparison.json
```

验收：112/112 summary rows 成功、112/112 trace validator `VALID`、所有 YOLO action 成功，
实际有效 batch 与请求值一致。YOLO11x `peak_allocated_mb` 均值从 372.672（batch=1）
增至 5,474.410（batch=64）；检测数均值从 0.125 增至 92.250。相对同一视频/同一
baseline 的 batch=1 路径，batch=2/4/8/16/32/64 的路径变化比例分别为
68.75%/93.75%/93.75%/93.75%/100%/100%；说明 batch 进入 planner 输入后确实产生了
不同的后续工具轨迹。各档 H1/H3 统计门槛均为 `pass`；这仍是 8 视频 pilot，不等同于
完整 32 视频正式 cohort。

## 6. Go/No-Go 规则

- 若批次及后续公开视频扩展仍出现单一路径、低下一动作熵和资源 CV 接近 0：按指导书降级
  VideoSeek/固定流程，把后续主线转向 STAR/VideoTool，并缩小动态调度故事；
- 若 H1/H3 在 20–50 preliminary 中保持非平凡：进入 100–300 条正式初始 trace，先做静态先验、
  一/二阶 Markov、Logistic/XGBoost/LSTM/小型 Transformer；
- 当前 96 条正式有效 trace 已完成平衡 cohort；用户已接受其作为阶段性正式样本，
  已执行离线 Phase 3，但 H2 目前为 mixed_evidence，不进入 Phase 4 调度收益实验；
- 不因为 API 失败、模型错误或缺少公开媒体而伪造重试/失败事件；trace 中只保留真实错误和真实
  `retry_of` 链。

## 7. 变更记录

| 日期 | 变更 | 原因 | 验证 |
|---|---|---|---|
| 2026-07-31 | 新增 Qwen 串行 worker、YOLO 实测计时、Phase 2 batch/stats 入口 | 让 8B 与 YOLO 真实介入并满足资源画像字段 | 本地 py_compile/14 项测试通过；远端 Qwen+YOLO ReAct trace `VALID` |
| 2026-07-31 | 删除热成像视频作为主线候选，改用公开 benchmark manifest | 用户明确要求公开数据为主、热成像跳过 | 远端 pilot 使用已有公开 LVBench 媒体；未下载完整数据包 |
| 2026-07-31 | API key 条款改为继续使用现有 key、不自动轮换 | 用户明确指示 | key 仅通过当前远端 shell 的临时环境变量传递；产物无 key |
| 2026-07-31 | 上传 8 个 Video-MME 公共视频并完成跨媒体 preliminary；按配额切换到 qwen3-vl-flash | 补足 H1 的跨视频内容覆盖，同时避免集中消耗单一 API 模型配额 | 21 条有效 trace、21/21 validator VALID、H1/H3 pass；错误模型名批次和中断目录排除并保留 |

## local_qwen 真动态 cohort（2026-08-01，已完成）

为区分 scripted 兼容性控制与真正由模型决定下一步的轨迹，新增本地串行
Qwen3-VL-8B planner/answer worker。单视频 smoke3 已显示 STAR 与 ReAct 产生不同的
工具路径；随后对 32 个独立 Video-MME 视频运行 3 个 baseline、每视频 1 次重复，共
96 条动态 trace。运行期间未调用 API，不消耗千问配额；YOLO11n 作为可选工具继续介入。

验收结果：

```text
96/96 success；96/96 validator VALID；32 个视频；model=Qwen3-VL-8B-Instruct
unique_paths=50，path_ratio=0.520833，path_length mean/min/max=3.302083/2/4
next_action_entropy=2.086501 bits；retry_count=1，error_count=2（真实 parser 事件）
runtime/local_runtime CV=1.562125，load CV=2.480231，peak allocated CV=0.064338，api_wait CV=0
H1_path_dynamicity_observation=pass；H3_resource_heterogeneity_observation=pass
```

路径按 baseline 分层为 `st_fixed=1`、`star=23`、`langgraph_react=28` 唯一路径；因此
H1 的动态性来自 local_qwen STAR/ReAct 的内容条件决策，而不是把固定流程包装成长轨迹。
正式动态 cohort 的 runtime manifest、batch summary 和报告分别为：

```text
configs/phase3_structured_videomme_32_localqwen_r01_remote.jsonl
results/raw/phase3_localqwen_videomme_32_r01/phase2_batch_summary.jsonl
results/processed/phase3_localqwen_videomme_32_r01_stats.json
results/processed/phase3_localqwen_h2_20260801/phase3_h2_report.json
results/processed/phase4_localqwen_burst_20260801/phase4_report.json
results/processed/phase4_localqwen_staggered_20260801/phase4_report.json
```

结构化 H2 使用 32-video leave-one-video-out、413 个 prefix 样本，结论为
`insufficient_evidence`：prefix Top-1 虽比 Markov2 高 0.111380，但 NLL 增高 0.093111；
加入任务特征的 Top-1 增益仅 0.007264 且 NLL 再增 0.117757，evidence 增量反而降低
Top-1 0.014528 并增加 NLL 0.241826。Phase 4 burst/staggered 的 Oracle gap 均为 0，
六个调度策略聚合结果相同，门控为 `stop_or_refine_workload`，所以当前不租用/部署真实
多 GPU，也不声称预测式调度收益。
