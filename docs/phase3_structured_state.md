# Phase 3：结构化状态与任务输入 v0.2

## 1. 本轮目的

旧版 Phase 3 使用 8 个独立视频、96 条重复轨迹、问题词袋和
`min(prefix_length, 6)`。它可以检验前缀是否有信号，但不能严格检验“视频内容是否
影响未来动作”。本轮先把输入和证据记录改成可审计的结构化表示，再扩展视频数量；不
改写旧 cohort，也不新增 API 调用。

本轮新增：

- `tracing/schema/state_v0_2.json`：每个决策点的 `state_t` schema；
- `tracing/schema/phase3_task_manifest_v0_2.json`：运行时任务 manifest schema；
- `tracing/collectors/structured_state.py`：任务结构派生、动作族归一化、证据累积和
  state sidecar 写入；
- `configs/phase3_structured_videomme_seed8.jsonl`：从现有 8 个 Video-MME 记录转换的
  不含答案标签的 seed manifest；
- `tests/test_structured_state.py`：5 个结构化状态/泄漏防护测试。

`videotool_phase1.py` 已接入最小 sidecar：运行启动时写 `state_0000.json`，每个工具
动作完成后按 observed-prefix 长度写一个新状态。下一步仍先用一个视频做 smoke，再进行
批量采集；这样可以把 schema 错误和采集控制流问题分开定位。

## 2. 数学状态

预测器的输入定义为：

```text
s_t = [q, v, e_t, p_t, b]
```

其中：

- `q` 是任务向量：`question_type`、`answer_type`、`temporal_scope`、所需模态、
  选项数、公开数据集的 domain/sub_category，以及问题/选项的长度统计；
- `v` 是视频元数据：时长、帧率、分辨率、字幕可用性、镜头数；
- `e_t` 是截至决策点已经获得的证据：32-bin 时间覆盖、观察区间、帧数、模态计数、
  YOLO 目标计数、OCR 字符数、时序关系数和可选置信度均值；
- `p_t` 是前缀状态：已观察步数、未截断的进度比例、动作族直方图、最近两个动作、
  本地 runtime、API wait、重试数和错误数；
- `b` 是 baseline 条件，不被当作视频内容。报告必须同时给出 baseline 内结果和
  pooled-with-baseline 结果。

GPU 当前状态不进入内容预测输入；它只在后续资源预测和调度器中使用。`video_id` 只
用于分组切分，`answer`、标准答案和未来事件均禁止进入特征。

输出定义为：

```text
P(next_canonical_action | s_t)
P(END | s_t)
E[next_runtime_ms | s_t]
E[next_peak_vram_mb | s_t]
```

框架动作保留在原始 trace 中，同时映射到 `sample_seek`、`spatial_qa`、
`temporal_qa`、`object_detection`、`ocr`、`summarize`、`answer`、`retry` 和
`other` 等 canonical action family。

## 3. 任务 manifest 的安全边界

转换命令：

```bash
python3 tracing/collectors/structured_state.py \
  --input configs/phase2_public_videomme_prelim.jsonl \
  --output configs/phase3_structured_videomme_seed8.jsonl
```

转换器保留给 Agent 使用的 `question` 和 `options`，但不复制源 manifest 中的
`answer`。`task_structure` 的 question type 是固定规则派生值，不是由 API 生成的自由
文本；字段中保存 `derivation.version` 和 `answer_label_used=false`。

## 4. Evidence sidecar 约定

每个已经完成的视觉工具结果都可以更新 `EvidenceAccumulator`。它只接收已观察的时间
区间、帧数、模态、YOLO 计数、OCR 数量、时序关系数和可选置信度。累积快照通过
`write_state_sidecar()` 写入独立 JSON 文件，原始 `trace.jsonl` 不覆盖。

sidecar 必须满足：

1. `future_events_included=false`；
2. `ground_truth_included=false`；
3. `video_id_used_as_feature=false`；
4. `answer_text_used_as_feature=false`；
5. 时间覆盖只来自当前及之前的工具调用。

## 5. 数据扩展目标

当前 8 个视频保留为 legacy/API cohort，不通过增加重复次数代替独立视频。下一批建议：

| 规模 | 独立视频 | baseline | 每视频重复 | 轨迹数 |
|---|---:|---:|---:|---:|
| 最低正式批 | 32 | 3 | 3 | 288 |
| 推荐批 | 48 | 3 | 3 | 432 |

新批次使用本地 Qwen3-VL-8B 和 YOLO11n，和旧的 `qwen3-vl-flash` cohort 分开统计；
Video-MME 与 LongVideoBench 分层评估，不能直接混成一个分布。划分单位是视频，必要
时进一步按来源/主题分组。

## 6. 下一轮验收

1. 一个视频的一次运行能写出 `trace.jsonl` 和多个合法 `state_t` sidecar；
2. validator 能拒绝 future evidence 或答案标签；
3. 不同 prefix 长度不会被强制合并为 `6+`；
4. 结构化 manifest 的行数、视频数、baseline 数和重复数可自动核对；
5. 通过单视频 smoke 后，才创建 32/48 视频正式 manifest 和批量运行。

## 7. 已执行的正式扩展（2026-08-01）

smoke 已通过后，本轮按“独立视频优先、重复只用于估计运行方差”的原则固定了 32 个
Video-MME 视频：旧的 8 个 legacy 视频加 24 个新增公开视频；每个视频使用
`st_fixed`、`star`、`langgraph_react` 三个控制流，每个控制流重复 3 次，目标为
`32 × 3 × 3 = 288` 条 trace。数据只取 Video-MME 官方 metadata/viewer 的公开视频和
题目，官方入口为：
`https://huggingface.co/datasets/lmms-eval/Video-MME`。

新增的可复查输入/下载产物为：

- `configs/phase3_videomme_32_source.jsonl`：含答案的审计源记录，只供评估映射使用；
- `configs/phase3_structured_videomme_32_local.jsonl`：本地 runtime manifest，不含答案；
- `configs/phase3_structured_videomme_32_remote.jsonl`：远端 runtime manifest，不含答案；
- `configs/phase3_videomme_32_eval_annotations.jsonl`：独立答案映射，不进入 agent state；
- `scripts/download_videomme_subset.py`：按官方 ZIP 中央目录和字节范围提取 MP4，并校验
  uncompressed size、CRC32、SHA256；
- `scripts/build_phase3_videomme_manifest.py`：从 source manifest 生成 runtime/eval manifest。

本地新增 24 个 MP4 共 `224,322,679` bytes；远端目录通过逐文件大小、SHA256 和
`ffprobe` 检查，24/24 可读。运行使用远端本地 Qwen3-VL-8B-Instruct 和 YOLO11n，不调用
API，也不把答案标签复制到运行时 manifest。

远端环境没有 `langgraph`/`langchain_core`，因此两个依赖它们的 baseline 在采集器中使用
保留标准 tool-call envelope 的兼容控制循环：ReAct 仍记录五步观察工具调用，STAR 仍记录
严格的 temporal/spatial 交替 planner 决策。trace 中明确标注 `compat_react_loop` 或
`compat_star_loop`；这不是声称上游 LangGraph 依赖已安装，而是可审计的资源受限复现路径。

## 8. 结构化 H2 验证入口

正式批次完成后运行：

```bash
PYTHONPATH=. python3 tracing/analysis/phase3_structured_predictor.py \
  --root results/raw/phase3_structured_videomme_32 \
  --output results/processed/phase3_h2_structured_20260801/phase3_h2_report.json \
  --csv results/processed/phase3_h2_structured_20260801/phase3_h2_metrics.csv
```

评估器只读取成功 run 的 `trace.jsonl` 和 `states/state_*.json`，按视频做
leave-one-video-out。`task_only_structured_nb`、`prefix_structured_nb`、
`task_prefix_structured_nb` 和 `structured_state_nb` 分别用于拆分任务元数据、执行前缀、
任务增量和已观测证据增量；同时保留 static/Markov 和剩余动作数、runtime、峰值显存的
均值基线。H2 的 preliminary pass 要求前缀模型同时优于 Markov 的 Top-1、NLL 和视频宏
平均 Top-1，且加入结构化任务特征后仍同时改善；否则记录为 mixed/insufficient，不进入
收益宣称。

## 9. Phase 4 回放入口

结构化 H2 完成后，使用：

```bash
PYTHONPATH=. python3 tracing/analysis/phase4_trace_simulator.py \
  --root results/raw/phase3_structured_videomme_32 \
  --output results/processed/phase4_replay_20260801/phase4_report.json \
  --csv results/processed/phase4_replay_20260801/phase4_metrics.csv \
  --capacities 32760,24576 --seeds 0,1,2,3,4
```

回放器固定真实 trace 的动作序列，只将测得的 `runtime_ms` 拆成 compute 与 cold-load，
并在反事实 GPU 容量上维护模型驻留、显存峰值和加载开销。首批策略为 Round-Robin、
Least-Loaded、Myopic、Static template、Oracle 和 `predictive`；后者在每个 job 上排除
同视频样本后，用 baseline/最近动作的转移先验估计下一模型。Oracle gap 小于 1% 时停止
把该 workload 作为调度收益证据；只有 Myopic 明显落后 Oracle，且预测式策略回收其中一部分
差距，才进入真实多 GPU 校准。报告中 `real_gpu_claim=false`，不会把反事实结果写成实机测量。

## 10. local_qwen 动态复验（2026-08-01）

scripted cohort 仅作为兼容性控制，不能证明 planner 由内容决定下一步。为满足这一
边界，新增 `planner_mode=local_qwen`：planner、视觉工具和答案生成共用一个串行常驻的
Qwen3-VL-8B worker；planner 输出完整 JSON 时直接解析，若只输出一个已知工具 token 则
记录 `parse_status=bare_tool_token` 后透明归一化，其他非 JSON 输出仍计为真实解析错误。
YOLO11n 继续作为可选 `yolo-tracker` 工具介入，API 不参与这一 cohort。

单视频 smoke3 已验证：三基线均成功，STAR 路径为
`image-grid-qa → temporal-qa → image-qa → image-qa`，ReAct 路径为
`temporal-grounding → frame-selector → image-grid-qa → temporal-grounding`；planner
事件均被记录，Qwen worker 的加载/推理/峰值显存字段非空。正式动态 cohort 使用同一
32 视频、3 baseline、每视频 1 次重复，共 96 条独立动态 trace；之前已完成的
32×3×3=288 条 scripted trace 保留为控制，不覆盖。动态 cohort 的远端输入为：

```text
configs/phase3_structured_videomme_32_localqwen_r01_remote.jsonl
results/raw/phase3_localqwen_videomme_32_r01
```

自动收尾脚本为 `scripts/finalize_phase3_localqwen_remote.sh`。96/96 条动态 trace 已
成功、96/96 通过 validator；统计得到 50 条唯一路径（path ratio=0.520833）、路径长度
均值 3.302083（2–4）、下一动作条件熵 2.086501 bits。按 baseline，`st_fixed` 为 1 条
固定路径，`star` 为 23 条，`langgraph_react` 为 28 条；H1/H3 observation 均为 pass。
资源 CV 为 runtime/local-runtime=1.562125、load=2.480231、peak allocated=0.064338，
API wait=0；共有 1 条 retry 链和 2 个真实解析错误事件，但没有失败 run。

结构化 H2 在 32 个视频、413 个 prefix 样本上为 `insufficient_evidence`：
`prefix_structured_nb` 相对 Markov2 的 Top-1 提升 0.111380，但 NLL 增加 0.093111；
加入任务特征后 Top-1 仅再升 0.007264，而 NLL 再增 0.117757；证据特征增量使 Top-1
下降 0.014528、NLL 增加 0.241826。因此只能报告“前缀存在预测信号但概率质量/内容增量
不稳定”，不能宣称 H2 完整通过。

Phase 4 burst 与 staggered 回放都得到 `oracle_gap_small`（Myopic−Oracle 平均完成时间
差为 0，六种策略聚合指标相同），门控结果为 `stop_or_refine_workload`；按照指导书，
不进入真实多 GPU 校准。报告与原始 batch summary 分别保存在：

```text
results/processed/phase3_localqwen_videomme_32_r01_stats.json
results/processed/phase3_localqwen_h2_20260801/
results/processed/phase4_localqwen_burst_20260801/
results/processed/phase4_localqwen_staggered_20260801/
results/raw/phase3_localqwen_videomme_32_r01/phase2_batch_summary.jsonl
```

96 条 trace 的轻量 `trace.jsonl`、`run_manifest/status` 与 413 个 state sidecar 已下载
到本地独立复核：本地 validator=96/96 VALID；本地重算的 H1/H3、H2 和两种 Phase 4
报告与远端收尾结果一致（Phase 4 aggregate 逐项相同）。
