# Phase 3–5 轨迹、训练 Workload 与并发显存实验规划

## 1. 文档状态与任务边界

- 目标：规划从公开视频下载、动态 Agent 轨迹采集、预测器数据构造，到并发调度与真实显存竞争验证的完整流程。
- 当前环境：本地项目 `/Users/liuqiuwei/Documents/调度`；远端主机使用 RTX 4080 SUPER 32GB，项目目录为 `/root/autodl-tmp/scheduler`。
- 本文授权范围：形成并维护规划、数量口径、执行顺序和验收标准。
- 2026-08-03 执行修订：先冻结不可变的 candidate snapshot，再从原始 trace 生成 semantic/compute 两套派生事件视图和 enrichment sidecar；预测器结果独立保存，禁止写回原始 trace。
- 本轮已确认的 DVD 扩展范围：更新本规划、在指定远端检出官方 DVD 代码、建立隔离依赖、完成 pilot，并启动独立的 128 条正式 DVD 扩展；不改变 768/800 条主线口径。
- 总原则：先串行采集不受干扰的真实节点画像，再从轨迹生成并发 workload；并发不是通过伪造轨迹获得，而是通过多个真实任务的到达与节点重叠获得。
- 正式收集目标冻结为：`768` 条必做轨迹；Grounded-VideoLLM 通过兼容性门禁后增加 `32` 条时序轨迹，最多 `800` 条。历史 scripted、API cohort 和未入选重复轨迹保留审计，但不混入正式 collection index。

### 当前权威状态（2026-08-04）

- 主线 collection 已完成并冻结：`640` core + `128` resource = `768` 条，64 个视频，
  core split 为视频级 48/8/8；Grounded-VideoLLM 未通过独立兼容门禁，因此没有把
  32 条可选时序轨迹加入主线。
- 768 条已编译为模板并生成 workload：train/validation/test 分别为
  `20,000/1,000/1,000` episodes，作业数为 `746,640/37,312/37,312`，三套
  `split_isolation_gate=true`。
- 最新 predictor 是 v0.4；locked test fine Top-1/Top-3=`0.8157/0.9803`，非终止
  子集=`0.7891/0.9791`；coarse routing=`0.9589/0.9946`。这表示总体目标已过，
  非终止 Top-1 仍作为明确的未满足子门槛记录。
- 原始 trace、canonical 派生输入和 workload 均保留可追溯 hash；权威报告和实现路径
  见本文第 15 节。第 3、8、14 节中的“计划/中间快照”是历史记录，不覆盖上述最终口径。

## 2. 研究目标

Phase 3 验证轨迹前缀是否能够预测未来执行：下一工具、剩余节点数、剩余时间、未来 K 个节点及分支概率。Phase 4 验证这些预测是否对多 GPU 调度产生反事实价值，即预测式策略相对 Myopic 等策略是否降低完成时间、尾延迟、模型换入和 OOM 风险，并与 Oracle 保持可解释的差距。Phase 5 只在 Phase 4 得到稳定正收益后，用真实多 GPU 或受控单 GPU 并发校准仿真模型。

核心假设：

- H1：结构化前缀状态比静态任务属性、长度档位和仅历史动作更能预测下一节点。
- H2：模型栈、任务类型和视觉证据会改变后续工具选择与剩余耗时。
- H3：未来节点预测能提前判断模型复用、换入和显存冲突。
- H4：当 workload 具有足够到达并发与模型异质性时，预测式调度优于无前瞻策略。

## 3. 当前资产与缺口

截至 2026-08-03，正式 DVD 目标集合冻结为 64 个唯一 Video-MME 视频：旧 32 个视频与新增 32 个视频。官方标注已补齐为每视频 2 个问题并写入
`configs/phase3_dvd_formal128.jsonl`（128 行、运行时不含答案）。远端当前实际存在 56 个视频文件；旧集合中仍缺 8 个文件，正式采集采用“已有 56 个视频先跑 112 条、缺失 8 个文件下载验证后补跑 16 条”的可恢复批次，不把缺失文件或重复问题计入成功样本。

已有归档不能简单按目录数相加：

- 288 条 scripted 轨迹仅为控制，不证明模型真正决定下一步；
- 112 条 YOLO batch pilot 只覆盖 8 个视频，属于资源干预层；
- `phase3_localqwen_yolo11x_videomme_32_r01` 已有旧 32 视频 × 1 问题 × 3 baseline，共 96 条成功 Stack A 轨迹，可在重新验证后计入核心层；
- 对应 r02/r03 共 192 条成功轨迹只作为重复 seed/运行方差候选，不重复计入核心独立任务；
- `phase3_localqwen_yolo11x_videomme_expansion32_r01` 的目标为新增 32 视频 × 2 问题 × 3 baseline，共 192 条；完成且通过 validator 后可计入核心层；
- DVD（Microsoft Deep Video Discovery）官方代码、隔离依赖和 `dvd_local` adapter 已通过检查；官方仓库未发布评测 trajectory，因此正式结果仍是兼容 adapter 的本地重实现。DVD pilot 与正式扩展和主线 collection 分层保存，不替代 STAR/ReAct/ST fixed；
- 最终使用独立 `collection_index.jsonl` 冻结纳入的 768/800 条。原始目录可以更多，但不得用未分层的总数宣称样本量。

2026-08-03 远端 live 审计显示 raw 运行目录包含历史 cohort、pilot、重复和失败尝试，不能直接相加为正式样本。最近一次派生快照口径为：Stack A/YOLO11x 384 条成功、Stack B formal 109 条成功且 1 条仍在运行、DVD formal 117 条成功（独立扩展）；这是 `trace_enrichment_core_live_20260803` 的中间值，不是最终 collection index。Stack B 结束后必须重新生成 snapshot、validator 报告并排除活动目录。

以上是 2026-08-03 的中间审计快照；Stack B 后续已完成并重新冻结，当前正式口径见“当前权威状态”和 §15，不再使用 56 视频/109 条 Stack B 的数字作为最终样本量。

当前可用模型：

| 角色 | 模型 | 当前状态 | 计划用途 |
|---|---|---|---|
| 重型统一 VLM | Qwen3-VL-8B-Instruct | 已使用，约 17–19.5 GiB | 参考模型栈 A，兼任 planner 与视觉判断 |
| 轻型 VLM | Qwen2.5-VL-3B-Instruct | 7.1GB 权重分片完整；已有三 baseline 单模型 smoke 成功，峰值约 9.25 GiB | 模型栈 B 的视觉与答案工具；现有 smoke 不等于拆分栈验收 |
| 文本 Planner | Qwen3-4B | 7.6GB、三个权重分片完整；尚无正式 PlannerDecision cohort，严格 tool call 仍待验证 | 通过独立文本 worker、50 次解析门禁后进入模型栈 B |
| 重型检测器 | YOLO11x | 可运行，batch 1–64 约 0.37–5.47 GiB | 目标检测与可控资源压力 |
| 轻型检测器 | YOLO26n | 可运行，约 64 MiB | 轻量检测对照 |
| 时序定位工具 | Grounded-VideoLLM Phi-3.5 | 26GB 权重组件已存在且分片索引完整；当前环境依赖不兼容、真实 import 失败 | 隔离环境验证后，仅作为可选 32 条时序子集工具 |

Grounded-VideoLLM 不计入 768 条必做目标，避免其修复阻塞主线；只有通过独立环境、短视频加载和时间区间输出门禁后才增加 32 条，使总量达到 800。

## 4. 视频与任务数据设计

### 4.1 视频数量

主数据集优先使用 Video-MME，保持统一的视频、问题和答案协议；LVBench/LongVideoBench 或 NExT-QA 只作为外部泛化层，确认下载许可、文件完整性和标注协议后再纳入。

- 最低规模：48 个独立视频；本轮正式主线采用已经落盘的 64 个 Video-MME 视频。
- 每个视频固定选择 2 个不同类型的官方问题，共 128 个任务。新增 32 视频已经满足；旧 32 视频仍需补选第二问题。
- 另留 8–16 个外部数据集视频做最终域外验证，但它们不阻塞 768/800 条主数据采集。
- 同一视频的全部问题、模型栈和 baseline 必须进入同一数据划分，防止画面泄漏。

64 个主视频建议按视频分组为 48/8/8 的 train/validation/test；Phase 3 也可在训练部分做 GroupKFold，但最终 8 个测试视频保持锁定。

### 4.2 任务构成

每个视频从官方问题中选择两个互补任务，并平衡以下类别：

- 时间顺序、事件定位或动作阶段；
- 物体、场景、人物和动作识别；
- 计数、跟踪或跨镜头一致性；
- 因果、多跳证据或状态变化。

运行时 manifest 只包含问题、选项和视频路径；标准答案单独保存在评测标注中，禁止进入 planner 状态、提示词或 trace sidecar。

### 4.3 下载与完整性流程

1. 先生成下载清单，记录数据集、视频 ID、来源 URL、问题类型、预计大小和目标路径。
2. 下载前检查本地与远端剩余空间；远端至少保留 30GB，新增视频缓存建议设 15GB 软上限，超过上限先复核样本组成。
3. Video-MME 复用 `scripts/download_videomme_subset.py`，从官方 Hugging Face 数据源按 ZIP 索引和 byte-range 抽取所需视频，避免下载完整压缩包。
4. 若远端链路慢，在本地下载后上传远端；上传前后比较 SHA256，并用 `ffprobe` 验证时长、编码和可解码性。
5. 保存下载报告：成功、跳过、失败原因、字节数、CRC32/SHA256、重试次数。失败项不自动更换视频，先区分临时网络失败和源文件缺失。

建议目录：

```text
data/public/videos/<dataset>/<video_id>.mp4
data/public/manifests/source_manifest.jsonl   # 含来源和答案，仅评测侧可读
data/public/manifests/runtime_manifest.jsonl  # Agent 运行输入，不含答案
data/public/download_reports/
```

## 5. 模型栈与 baseline 矩阵

不做所有模型的笛卡尔积，只保留能回答研究问题的组合：

| 编号 | Planner | 视觉理解 | 检测器 | 目的 |
|---|---|---|---|---|
| A：重型统一 | Qwen3-VL-8B | Qwen3-VL-8B | YOLO11x | 高质量参考、较高显存占用 |
| B：轻型拆分 | Qwen3-4B | Qwen2.5-VL-3B（兼任答案） | YOLO26n | 观察模块拆分、复用与切换；YOLO11x 只进入资源子集 |
| C：时序扩展 | Qwen3-4B | Qwen2.5-VL-3B + Grounded-Video-LLM | YOLO11x | 修复后用于时间定位子集，不阻塞主实验 |
| D：DVD 扩展 | DVD API pilot：o3 + GPT-4.1-mini；DVD local：Qwen3-4B/8B + Qwen2.5-VL-3B | DVD 的 global browse / clip search / frame inspect 工具 | 不接入官方 DVD baseline | 验证搜索型、多粒度工具轨迹；独立于 768/800 主线 |

动态 baseline 使用 STAR 风格 planner 和 LangGraph ReAct；ST fixed 只做固定流程控制。模型必须真正输出下一步决策，保存原始输出、解析结果和实际执行动作。Qwen3-4B 进入正式采集前必须通过至少 50 次严格 `PlannerDecision`/标准 tool-call smoke test，并报告一次通过率、修复率和错误分布；解析器不得静默替模型改变决策。

### 5.1 第二模型栈的接口改造

当前采集器只有一个 `QwenVLClient`，planner、视觉工具和答案生成共用同一个 VLM。正式 Stack B 必须真实拆分，不能只修改 `model_name`：

```text
planner_client = Qwen3-4B
visual_client  = Qwen2.5-VL-3B
answer_client  = Qwen2.5-VL-3B
detector       = YOLO26n
```

执行时复用现有 JSON-line worker 协议，新增独立文本 planner worker，并为采集器增加下列运行参数；旧 Stack A 参数继续兼容：

```text
model_stack_id
planner_model_path / planner_python
visual_model_path  / visual_python
answer_model_path
detector_model_path
seed
residency_policy
thermal_state
```

当前实现落在 `tracing/collectors/qwen_text_worker.py` 与
`tracing/collectors/videotool_phase1.py`：`--planner-mode local_split` 启动
Qwen3-4B 文本 worker；`--qwen-model` 继续负责视觉工具；`--answer-model`
可指定独立的 Qwen2.5-VL-3B 答案 worker，同路径时复用视觉 worker，避免重复驻留。
Planner、视觉、答案和 detector 节点通过 `TraceRecorder.record_event(model_id=...)`
写入各自实际模型 ID，Stack A 的旧参数和旧 trace 结构保持兼容。phase2 manifest
可逐条覆盖这些模型路径、ID、worker Python 和生成长度。

运行 manifest 必须记录 planner、VLM、answer、detector 的真实模型 ID、权重 hash、运行环境、seed 和驻留策略；每个 trace 事件的 `model_id` 必须对应实际执行该节点的模型。任务 manifest 只保存视频与问题结构，不混入模型条件。

### 5.2 Qwen3-4B PlannerDecision 门禁

正式 Stack B 采集前先运行 50 次决策级 smoke，覆盖四类问题、STAR/ReAct、短/中/长前缀。保存原始输出并区分：`json`、`non_json`、`truncated_json`、`schema_error`、`unknown_tool`、`wrong_argument_type`、`multiple_objects`、`timeout` 和 `worker_exit`。

门禁脚本为 `scripts/run_qwen_planner_gate.py`；它通过与采集器相同的
JSON-line 文本 worker 发送 50 个真实请求，只允许一次严格 retry，并把每次原始输出
和资源字段保存为 JSONL，最终成功率低于门槛时返回非零退出码。

通过条件：

- 第一次结构化解析成功率不低于 90%；
- 最多一次 retry 后的成功率不低于 98%；
- retry 后未知工具数为 0；
- 语法约束只能限制 JSON 结构，不能替模型决定工具；
- 不允许把 bare token、非法工具或第二次失败静默改成预设路径；
- 加载时间、推理时间、峰值显存与原始输出字段完整。

若门禁失败，Stack B 暂停，不得仍以 Qwen3-4B Planner 名义采集。可另行评估并明确改名为 `Qwen2.5-VL-3B unified` 的备选栈，但必须重新做 pilot，不能与原定义混用。

本轮实际门禁结果（2026-08-03）：Transformers 5.8.1 环境使用
`lm-format-enforcer` 的 JSON Schema prefix constraint；为避免长 reasoning 在词表过滤器上造成不必要的等待，gate 使用
`--max-new-tokens 96`，并对 `reasoning`/`tool_input` 设置长度上限。首次 46/50，最多一次 retry 后 50/50（100%）通过。
正式采集通过 `--planner-constrained-json` 将同一约束传入 `local_split` planner worker；该开关默认关闭，Stack A 旧轨迹不受影响。

### 5.3 第二模型栈 pilot 与驻留门禁

选择覆盖 train/validation/test 的 8 个视频，每视频 2 个问题，运行 STAR 与 ReAct：

```text
8 视频 × 2 问题 × 2 动态 baseline = 32 条
```

pilot manifest 由 `scripts/build_stackb_pilot_manifest.py` 生成，当前候选文件为
`configs/phase3_stackb_pilot8.jsonl`；它去除标准答案字段、固定两个动态 baseline，
并逐任务写入真实 planner/VLM/answer/detector 路径与模型 ID。

pilot 启动后由 `scripts/auto_stackb_after_pilot.sh` 等待 32 条任务完成，先检查 summary
和逐条 trace validator；只有 32/32 成功且全部 `VALID` 才自动启动 formal 256 条输出目录，
失败则停止并保留原始失败证据。

配置冻结后，这 32 条可以计入 Stack B 正式 256 条。pilot 同时测量 Qwen3-4B 单独驻留、Qwen2.5-VL-3B 单独驻留、二者同时驻留、二者加 YOLO26n，以及二者加 YOLO11x batch 16。只有 `常驻模型显存 + workspace P95` 不超过 32GB GPU 容量的 90% 时才采用同时驻留；否则采用显式卸载/加载策略并记录切换代价。

### 5.4 Grounded-VideoLLM 当前状态与可选落地

远端已核对：

```text
代码：/root/autodl-tmp/scheduler/grounded-video-llm
commit：e26da4e4b681357fd911d5ace3467f031af29208
权重：/root/autodl-tmp/scheduler/Grounded-VideoLLM
```

代码约 19MB；工作树表面为 dirty，但忽略行尾差异后没有内容 diff。权重约 26GB，Phi-3.5 四个 safetensors 分片、8.12GB SFT checkpoint、2.82GB InternVideo2、视觉模型和 projector 均存在。当前没有 LLaMA3 版本权重，Phi-3.5 版本足够用于可选时序工具。

当前 `finetooling` 环境为 Torch 2.12/Transformers 5.8，且没有 `flash_attn` 和 `peft`；官方要求 Torch 2.1.2、Torchvision 0.16.2、Transformers 4.40.1、PEFT 0.3.0、FlashAttention 2.3.3。真实执行 `inference.py --help` 已在导入 `flash_attn` 时失败。因此：

1. 不修改现有 dirty 代码目录；从固定 commit 建立 clean worktree；
2. 在数据盘创建隔离环境 `/root/autodl-tmp/conda/envs/grounded-videollm`，不占用只剩约 8GB 的根盘；
3. 先做 import/help，再做 1 个短视频 16/32 帧加载 smoke，最后用 2 个时序视频、96 帧检查时间区间输出；
4. 通过后封装为独立 temporal-grounding worker；
5. 在 16 个时序型视频上各取 1 个任务，运行 STAR/ReAct，共 32 条 Stack C 轨迹；
6. 任一门禁失败即停止于 768 条，不用低质量重复强行补到 800。

### 5.5 DVD 扩展 cohort（主线之外）

DVD 按 Microsoft Deep Video Discovery 处理。它的原生工具集合是
`global_browse_tool`、`clip_search_tool`、`frame_inspect_tool` 和 `finish`，每一轮由
orchestrator 通过 function call 选择工具；DVD 的 global/clip 工具主要依赖片段字幕、
向量检索和摘要，frame inspect 才直接产生视觉 VLM 节点。官方实现默认使用 `o3`、
`gpt-4.1-mini` 与 `text-embedding-3-large`，不能把本地 Qwen 运行结果写成官方精确复现。

当前目标分两层：

1. `dvd_api_pilot`：8–16 条，优先验证官方 function-call schema、数据库/字幕输入和
   `TraceRecorder` 适配；只通过临时环境变量提供已有 API key，不写入文件或日志。
2. `dvd_local`：pilot 通过后再生成 128 条，即 64 个 Video-MME 视频 × 2 个问题 × 1
   DVD baseline。使用本地 Qwen 作为替代 orchestrator/视觉工具时，必须将
   `agent_family=dvd`、`implementation=local_reimplementation` 和真实模型 ID 写入
   manifest/trace，不能标记为官方 DVD。

DVD 复用主线 64 个视频、128 个任务和 48/8/8 视频 split，不新增视频泄漏。每条 trace
   保存原始 planner/function-call 输出、工具参数、检索/帧区间、观察摘要、实际执行
   节点、模型显存和时间字段。DVD 不接入 YOLO；若需要检测器资源压力，另建带检测器
   的 `dvd_resource_extension`，不与 DVD baseline 混计。

DVD 128 条只作为主线完成后的独立扩展：

```text
768 条必做主线（或通过 Grounded 后的 800 条）
+ 128 条 DVD 扩展
= 896 条（或 928 条）总轨迹上限
```

DVD pilot 的通过条件：所有 trace schema validator 通过率 ≥95%；工具名和参数严格可
解析；每条成功运行有 `finish` 终止节点；保留自然 parse error/retry；跨视频至少出现
两种非终止工具路径。pilot 未通过时保留失败审计，不启动 128 条正式扩展。

2026-08-03 已完成 `dvd_local` 的 8 条 pilot（4 个 Video-MME 视频 × 2 个问题；1 条
单条 smoke 加 7 条串行批次）。8/8 run 成功，8/8 通过
`tracing/validators/validate_trace.py`，每条都有 `finish`，且工具调用均可还原为标准
function-call：共 8 次 `global_browse_tool`、7 次 `clip_search_tool`、3 次
`frame_inspect_tool` 和 8 次 `finish`。路径至少包含 global→clip→finish、
global→frame→finish 与 global→clip→frame→finish 三类；成功样本没有重试。使用的是
本地 Qwen3-VL-8B 的 `local_reimplementation`，不是官方 API 精确复现。此前修正轮中
保留了 6 次自然失败审计（3 次 planner JSON 截断、2 次 VFR 帧抽取失败、1 次终止步
未收敛），没有混入成功 pilot。结果目录为
`results/raw/phase3_dvd_local_pilot8_r03` 和 `results/raw/phase3_dvd_local_pilot8_final`；
因此已满足 pilot 门禁。正式 manifest 已冻结为
`configs/phase3_dvd_formal128.jsonl`；采集按 112+16 两个可恢复批次执行，成功后再生成独立 DVD index。

2026-08-03 已启动正式采集：远端 `/root/autodl-tmp/scheduler` 的 batch-1 使用
`configs/phase3_dvd_formal112_available.jsonl`，输出到
`results/raw/phase3_dvd_local_formal128`；该批次包含当前已验证存在的 56 个视频、112
 条任务，仍使用本地 Qwen3-VL-8B 和 5 步 DVD 工具循环。缺失 8 个视频由独立下载进程按
官方 Video-MME shard 提取并校验，完成后从同一冻结 manifest 补采剩余 16 条。两个进程
均不改变主线 768/800 collection index。

首条正式任务在 `qwen_max_new_tokens=192` 下自然产生了 planner JSON 截断；该失败目录
保留在正式输出根目录作为 failure/recovery 审计。已用 `384` 做同任务 smoke，运行成功
且 validator 通过，因此 batch-1 以 `384` 上限重新开始，不覆盖原始失败目录。
远端直连 Hugging Face 的下载进程因 443 超时停止；随后在本机用同一官方 shard 脚本
下载了 8 个文件（总计 60,780,469 bytes），逐文件 SHA256 与远端上传结果一致后才标记
为可用。此下载失败本身保留在 `logs/dvd_missing8_download.log`，不计入 DVD trace。

## 6. 为什么先串行生成轨迹

主轨迹采集采用单任务串行执行，同一时刻只让一个 GPU 工具节点运行。这样得到的节点耗时、冷启动、模型加载和峰值显存是无竞争基准，便于学习节点本身的资源需求。若一开始并发采集，排队、CUDA 竞争和显存驱逐会混入服务时间，预测器难以区分“任务本身很慢”和“被其他任务拖慢”。

每个模型栈至少分别采集：

- cold：模型未驻留，记录加载时间和首次推理；
- warm：模型已驻留，记录稳定服务时间；
- 关键节点重复 2 次不同 seed，用于估计 planner 随机性；
- YOLO11x 对小型资源子集采集 batch 1/8/16/32/64，不把这些重复计作独立视频。

串行采集完成后，再把完整 trace 当作一个带依赖关系的 job template，在 Phase 4 中并发到达与交错执行。

## 7. 轨迹字段与质量门禁

每条轨迹必须至少记录：

- 身份：`dataset`、`source_video_id`、`task_id`、`task_type`、`baseline`、`seed`、`schema_version`；
- 模型条件：`model_stack_id`、planner/VLM/detector 模型与版本、量化方式、YOLO batch；
- 决策：每步结构化状态、原始 planner 输出、解析后的决定、实际执行工具和参数；
- 证据：帧区间、检测结果摘要、证据引用，不嵌入标准答案；
- 时间：排队、加载、预处理、推理、后处理和节点总耗时；
- 资源：GPU 型号、峰值 allocated/reserved、模型驻留前后、CPU/RAM；
- 鲁棒性：错误类型、重试次数、重采原因、最终状态；
- 输出：节点依赖、终止原因、答案与评测结果。

正式纳入训练的门禁：JSON/schema 可解析；工具名合法；时间单调；视频可解码；资源字段非负；成功轨迹有终止节点；失败轨迹有错误类别；同一视频没有跨 split；答案没有泄漏。解析失败、OOM 和重试轨迹不删除，进入单独的 failure/recovery 数据层。

### 7.1 真实错误、等待与恢复的定义

允许通过受控实验条件产生真实运行错误，但禁止手工构造模型输出、直接写入伪造错误事件或用 `sleep` 伪装排队。所有样本标记来源：

- `natural`：模型自然产生的解析错误、worker 退出、解码失败或运行时错误；
- `controlled_stress`：真实并发、deadline 或超容量请求引起的等待、timeout、CUDA OOM 与恢复；
- `synthetic_trace`：禁止纳入正式数据。

Planner 首次输出无法解析时，先写一条真实 `planner.generate` error 事件，保存原始输出与错误类别；只允许一次 JSON-schema 修复调用，新事件的 `retry_of` 指向第一次事件。第二次仍失败时 run 真实失败，不允许由采集器替模型挑选工具。

等待样本由多个真实 job 同时提交到同一 GPU worker/service 产生：

```text
queue_ms = node_start_at - job_ready_at
```

必须保存 ready、admission、start、finish 时间和 timeout/deadline 状态，不能用固定休眠生成等待值。

OOM 只在主轨迹完成、GPU 空闲后，以隔离 subprocess、watchdog 和独立输出目录执行。关闭 admission 的场景用于观察真实 CUDA OOM；开启 admission 的对应场景用于验证等待、驱逐、卸载或降 batch 能否避免 OOM。恢复动作只允许：清理失败 worker、卸载无关模型、YOLO batch 减半、改为串行准入并重试同一工具。原始 OOM 事件不得被成功 retry 覆盖。

鲁棒性门禁：至少 8 个真实 parse-error/retry 链、8 个非零 `queue_ms` 等待样本和 8 个真实 CUDA OOM 事件；OOM 恢复成功率至少 75%。若模型自然解析错误不足，可扩展真实 stress 尝试但设置上限，并从重复 seed 子集中等量扣减，保持正式总量不变；仍不足时如实报告，不伪造。

### 7.2 Trace enrichment 与双事件视图（2026-08-03 修订）

原始 `trace.jsonl` 保持 schema v0.1、只追加不覆盖。新增派生数据使用独立版本和来源引用：

```text
candidate_snapshot_v0_1.jsonl       # 当前可审计候选快照，不等同最终 collection index
semantic_events_v0_1.jsonl          # 预测 Agent 下一步的语义事件
compute_events_v0_1.jsonl           # 预测 runtime/VRAM 和构造 workload 的计算事件
trace_enrichment_v0_1.jsonl         # workflow/activity/dependency 等派生字段
prefix_samples_v0_1.jsonl           # 可重建的 Phase 3 输入/标签
```

每条派生记录必须包含 `source_event_ids`、`source_trace_sha256`、`derivation_version` 和 `value_source`（`measured`、`derived`、`unknown`）。无法从原始事件证明的参数来源、未选择分支、候选 GPU 运行时间或 speculative 状态必须保持 `unknown`，不得补猜。

两套事件视图的规则固定如下：

| 视图 | 保留事件 | 用途 |
|---|---|---|
| `semantic_view` | planner 的实际决定、实际工具、规范化 retry/finish | DyOrc/Markov、AutoTool-lite、Pythia-lite、RAG、SuTraN |
| `compute_view` | planner/API、视觉工具、YOLO、answer、load/unload、queue、retry | Park–Song-style resource predictor、job template、Phase 4 |

`step_id` 不是事件唯一 ID；派生事件必须按原始文件顺序使用 `event_id` 建立顺序，并把同一逻辑决策的 API retry 记录为边而不是无条件的新语义节点。成功轨迹必须有终止事件；失败轨迹保留错误和 retry，但不能当作成功后缀训练样本。

Future Predictor 只允许使用任务、视频元数据、已执行 semantic prefix 和已观测证据；GPU 当前队列、显存和模型驻留只进入 Resource Predictor 或 Scheduler。待预测节点的真实 runtime/VRAM 只能作为标签，不能作为该节点的输入。

## 8. 轨迹规模与采集阶段

### 8.1 核心内容轨迹：640 条

以下是采集前的配额设计；实际是否已完成以 §15 的冻结 collection index 为准，下面的
“仍需采集”只描述当时的计划快照。

```text
Stack A，STAR + ReAct：64 视频 × 2 问题 × 2 baseline = 256
Stack B，STAR + ReAct：64 视频 × 2 问题 × 2 baseline = 256
Stack A，ST fixed：    64 视频 × 2 问题              = 128
核心合计：                                               640
```

ST fixed 只保留 Stack A 参考控制，不再乘第二模型栈。Stack A 使用 Qwen3-VL-8B + YOLO11x；Stack B 使用 Qwen3-4B planner + Qwen2.5-VL-3B visual/answer + YOLO26n。

当前可计入口径为：旧 32 视频第 1 问题的 Stack A/YOLO11x r01 共 96 条；新增 32 视频两个问题的 Stack A 批次完成后共 192 条。二者全部通过 validator 后，核心层达到 288/640；仍需采集旧 32 视频第 2 问题的 Stack A 三 baseline 共 96 条，以及全部 64 视频两个问题的 Stack B 两动态 baseline 共 256 条，即还需 352 条核心轨迹。

### 8.2 资源与鲁棒性轨迹：128 条

| 子集 | 设计 | 数量 |
|---|---|---:|
| Planner 解析错误与 retry | 6 个高难任务 × 2 baseline × 2 stack | 24 |
| Planner 随机性/重复 seed | 6 视频 × 2 baseline × 2 额外 seed | 24 |
| YOLO batch 资源曲线 | 4 视频 × 2 baseline × batch 8/16/32/64 | 32 |
| cold/warm | 4 任务 × 2 stack × cold/warm | 16 |
| 实际 queue/wait | 8 个双 job 并发场景 | 16 |
| 实际 OOM/recovery | 8 个双 job 超容量场景 | 16 |
| 合计 |  | **128** |

重复 seed 优先从现有 r02/r03 中筛选；YOLO batch 优先从已验证的 batch pilot 中筛选，batch=1 使用核心轨迹作参考。预计可复用 56 条、需要新采约 72 条。解析、等待和 OOM 样本即使最终失败也保留，但只进入 failure/recovery 和调度资源层，不进入 H1/H2 内容主训练集。

### 8.3 正式总量与可选时序层

```text
640 核心内容轨迹
+ 128 资源与鲁棒性轨迹
= 768 条必做正式轨迹

+ 32 Grounded-VideoLLM 可选时序轨迹
= 800 条上限

+ 128 DVD 扩展轨迹
= 896 条（或含 Grounded 的 928 条）总轨迹上限；DVD 不进入 768/800 的主线 collection index。
```

预计每条轨迹产生 4–5 个前缀样本，768–800 条可形成约 3,000–4,000 个监督前缀，适合 Logistic/XGBoost、Markov、LSTM 或小型 Transformer；不适合直接微调 4B planner。

原始目录允许因为失败尝试、resume 和历史 cohort 超过 800；正式样本量以通过校验的 `collection_index.jsonl` 为准。核心槽位若失败，保留失败 trace，再对同一槽位重采成功样本；失败样本从 128 条鲁棒性 reservoir 中选入，避免不透明删除或重复计数。任何批次若 schema 通过率低于 95% 或 planner 一次解析成功率低于 90%，先停止扩量并修正接口。

## 9. 从轨迹生成训练 Workload

### 9.1 预测器训练样本

对每条 trace 的每个决策点生成一个 prefix sample：

```text
Future Predictor 输入 X_t = 任务结构 + 视频元数据 + 已执行 semantic 节点
                            + 结构化证据和工具输出摘要
Future Predictor 标签 Y_t = 下一语义节点、剩余节点数、未来 K 节点、终止/分支概率

Resource Predictor 输入 R_t = 候选节点 + 输入规模 + 模型 + GPU 类型 + cold/warm
                               + 历史资源画像
Resource Predictor 标签 Z_t = 节点 runtime、峰值 VRAM、load cost、queue/等待代价
```

所有连续值使用真实数值或分位数归一化，不再用随意文本或把所有长度大于 6 粗暴并为一档。分类字段保留显式枚举，历史动作使用序列编码。先比较静态、Markov、prefix+task、prefix+task+evidence 四级模型，并分别报告各 baseline、任务类型、模型栈和 held-out 视频的指标。

正式评估固定使用 48/8/8 视频 split；leave-one-video-out 只作为训练侧诊断。RAG 的检索索引只能由训练视频构建，测试视频的任何重复运行不得进入索引。baseline/model stack 同时报告“允许作为部署条件”和“不作为语义特征”的两套消融，避免把 Agent 类型差异误认为内容预测增益。

### 9.2 可复现预测器契约

第一批只声明以下核心组件：

| 组件 | 实施标签 | 输出 | 主要验收 |
|---|---|---|---|
| DyOrc/Markov | `local_reimplementation` | 下一节点概率、未来 K 步样本 | Top-1/3、NLL、K-step recall |
| AutoTool next-tool | `adapted_next_tool_only` | 工具候选和置信度 | 与 Markov 的增益、参数缺失率 |
| Pythia path/length | `adapted_path_length_only` | 路径候选、长度区间 | 覆盖率、候选宽度、区间覆盖率 |
| RAG Next Activity | `adapted_retrieval` | 相似 prefix 和下一节点 | train-only index、Top-1/3、检索泄漏检查 |
| SuTraN | `small_model_adapter` | 剩余后缀、下一时间、剩余时间 | suffix recall、time MAE、3-seed 方差 |
| Park–Song predictor | `predictor_only` | runtime/VRAM 估计 | MAE/P95 误差；不宣称完整资源优化 |

PASTE 和 SpecFaaS 暂不列为完整复现；它们需要参数来源图、完整控制/数据依赖和 speculative commit/discard 事件。

### 9.3 调度器 workload 模板

每条完整轨迹转换为一个 job template，保存 planner、VLM、YOLO、Grounded、answer、模型加载/卸载、等待与 retry 节点，以及节点 DAG、模型 ID、cold/warm 服务时间分布、常驻模型显存、活动 workspace、加载代价、失败概率和重试边。生成 workload 时抽样完整模板，而不是独立打乱节点，以保持真实依赖和工具转换。

每个 workload episode 包含：

- 到达过程：低/中/高强度 Poisson、零间隔 burst、staggered arrival；
- 任务混合：短/长视频、四类问题、A/B 模型栈、轻/重检测器；
- GPU 集群：2 卡起步，覆盖 32760+32760、24576+24576、32760+24576 MiB；
- 服务等级：普通/高优先级、软 deadline；
- 系统初态：冷缓存、热缓存、偏斜驻留三类。

本轮固定生成：

```text
训练：20,000 episodes
验证： 1,000 episodes
测试： 1,000 episodes
```

每个 episode 抽取 16/32/64 个完整 job，覆盖 50%、70%、85%、95% 和 >100% 请求压力；deadline 取预计服务时间的 1.5×、2×、3×。仿真 episode 可以大量生成，但其基础节点分布必须来自真实 trace，且测试 episode 的视频模板不得出现在训练侧。训练/验证/测试继续按 48/8/8 视频分组，鲁棒性样本继承其源视频 split。

## 10. 从串行轨迹构造并发显存竞争

并发来自多个 job 的节点同时 ready，而不是改变单条 trace：

```text
真实串行 trace → job template → 抽样到达时间 → 多个 job 节点同时 ready
                  ↓
       调度策略选择 GPU、等待、复用、驱逐或延迟准入
                  ↓
     模型驻留显存 + 活动 workspace + 安全余量与 GPU 容量比较
```

显存准入条件建议为：

```text
resident_model_memory
+ sum(active_workspace_p95)
+ allocator_fragmentation_margin
<= gpu_capacity
```

压力档位按“驻留模型 + 已准入活动节点”的预计显存占容量比例定义为 50%、70%、85%、95% 和 >100% 请求压力。85% 以上用于观察等待、驱逐、模型复用和 OOM 防护；不能只靠把 YOLO batch 放大，因为那只制造单节点大显存，不能代表异构模型并发。

现有 `phase4_trace_simulator.py` 已支持 round-robin、least-loaded、myopic、static-template、predictive 和 oracle，以及 GPU 容量和固定到达间隔，但还不能称为完整 workload：它当前只读取 `event_type=action`，漏掉 planner/answer GPU 成本；把轨迹按顺序列表回放而非完整 DAG；每张 GPU 同时只运行一个节点；没有分离模型常驻显存与活动 workspace；没有真实 queue/OOM/recovery 状态；驱逐近似为清空全部 resident model。正式 Phase 4 前需要补充：

1. 活动节点 workspace 与模型权重显存分离；
2. 多节点在同一 GPU 上重叠运行或受控并发槽位；
3. 模型驱逐、重新加载及缓存命中代价；
4. 显存碎片与安全余量；
5. 并发导致的服务时间膨胀系数；
6. OOM/等待/降级/重试状态转移；
7. Poisson 与 burst 到达生成器，而不只有固定间隔。

实现时保留现有入口并拆出可复用的 trace-to-template 编译层，新增带版本的 `job_template` 和 `workload_episode` schema/validator。所有 derived 数据都必须能从原始 trace 和 collection index 重建；不得覆盖原始 trace。

首版预测式调度器保持可解释：

```text
cost = predicted_queue_time
     + predicted_runtime
     + model_load_cost
     + deadline_risk
     + memory_conflict_risk
     - model_reuse_gain
```

先验证 Oracle 相对 Myopic 的完成时间差是否稳定超过噪声，再比较 predictive。如果 Oracle gap 仍接近 0，说明 workload 压力或资源模型仍不足，不应直接进入强化学习或真实多 GPU 大实验。

## 11. 真实并发校准与 Phase 5

串行服务时间不能直接代表并发性能。Phase 4 通过门禁后，执行小型 microbenchmark：

- Qwen3-VL-8B 与 YOLO11x；
- Qwen2.5-VL-3B 与 Qwen3-4B；
- 两个 VLM 请求；
- YOLO11x batch 16/32/64 与轻型 planner；
- 每组测 1、2、4 个并发请求或进程，至少 3 个 seed。

使用高频 NVML 采样记录显存时间序列、利用率、功耗、节点开始/结束、模型加载和 OOM。由此拟合：并发耗时膨胀、实际 workspace 叠加、碎片余量和驱逐代价，再回填 Phase 4。

单张 32GB GPU 可以做压力校准，但必须标注为“单 GPU contention/stress”，不能冒充多 GPU 调度。若人为预留显存，只用于校准准入和 OOM 防护，不能计作 Agent trace。最终 Phase 5 建议至少 2 张 GPU、100–200 个真实调度 episode，或 30–50 种 workload mix × 3 seeds。

## 12. 分阶段执行与验收

| 阶段 | 动作 | 通过条件 |
|---|---|---|
| D0 数据审计 | 冻结 64 视频、128 任务、容量预算、去重与 48/8/8 split | 无跨 split 视频；答案隔离；全部可解码 |
| D1 模型兼容 | Qwen3-4B 50 决策测试；Stack B 8 视频/32 轨迹 pilot | schema ≥95%；planner 一次解析 ≥90%、retry 后 ≥98%；资源字段完整 |
| D2 主轨迹 | 完成 640 核心 + 128 资源/恢复轨迹 | collection index=768；核心槽位完整；模型栈/任务平衡 |
| D2-C 可选时序 | Grounded 隔离环境与 32 条时序轨迹 | 通过 import、加载、时间区间与显存门禁后 collection index=800 |
| D2-D DVD 扩展 | 官方 DVD 依赖检查、8–16 条 pilot；主线完成后再扩展到 128 条 | pilot schema ≥95%、工具路径可解析；通过后单独生成 DVD index，不改变主线 768/800 |
| P3 预测验证 | 固定 48/8/8 split；LOVO 仅诊断；运行 enrichment 契约内的预测器 | 所有输出可追溯；无未来泄漏；报告 Top-K、NLL/校准、后缀/时间误差；不能用 baseline 混杂冒充内容增益 |
| P4 仿真验证 | 20k/1k/1k episodes，多压力 workload，比较 6 策略 | Oracle gap >1% 且跨 seed 稳定；predictive 优于 myopic |
| C1 并发校准 | 真实 contention microbenchmark | 获得可复现的显存与耗时膨胀参数 |
| P5 真实验证 | 2+ GPU 或明确标注的单 GPU stress | 趋势与仿真一致，无未解释 OOM，报告 P50/P95/吞吐 |

若 P3 不通过，优先增加视频与结构化证据质量；若 P3 通过而 P4 Oracle gap 小，优先修正 workload 并发、模型驻留和显存模型，而不是继续堆轨迹；若 P4 通过但 P5 不一致，校准资源模型和并发膨胀系数。

## 13. 预期产物

```text
data/public/                         # 视频、运行 manifest、来源清单和下载报告
tracing/runs/<cohort>/               # 原始、不可覆盖的轨迹
tracing/derived/collection_index/    # 冻结纳入的 768/800 条正式轨迹及分层标签
tracing/derived/prefix_samples/      # 可重建的 Phase 3 样本
tracing/workloads/templates/         # trace 到 job template 的转换结果
tracing/workloads/episodes/          # 仿真训练/验证/测试 episode
tracing/reports/phase3/              # 预测指标与分层结果
tracing/reports/phase4/              # 策略、压力档位和 Oracle gap
tracing/reports/phase5/              # 真实并发校准与调度结果
```

本轮实际使用的可重建入口为：

```text
tracing/analysis/build_trace_enrichment.py
tracing/analysis/reproduce_prediction_baselines.py
tracing/analysis/reproduce_resource_predictor.py
tracing/analysis/build_collection_index.py
tracing/workloads/build_workload.py
scripts/build_video_split_manifest.py
```

其中 `build_workload.py` 只在模板引用层生成 arrival、压力、GPU 拓扑和初始驻留状态；它不把合成节点写回 raw trace，也不把 workload episode 计作新的 Agent trace。

### 14. 2026-08-04 执行记录：core 冻结、修复后预测与 workload

- Stack B 已自然结束，正式 core 为 **640/640**：Stack A 384、Stack B 256；validator
  640/640，重复槽位 0，64 视频，48/8/8 视频 split 门禁通过。原始目录中的旧
  尝试和失败目录没有被相加计数。
- 修复后的派生目录为
  `results/processed/trace_enrichment_core_fixed_20260804/`。它保留原始
  `v0_1` 文件名以兼容下游，但 prefix 增加 `state_features`；4,441 个 prefix
  中 3,801 个有对应的 state_t（缺失的 640 个是终止后没有 pre-action sidecar，
  不是伪造补齐）。所有 state leakage guard 仍为 future/ground-truth excluded。
- 旧的 0.5 概率问题已确认既有代码问题也有信息/任务动态性问题：空前缀的全局
  计数污染已修复，position/history 回退与 train-only vocabulary 已加入；但前 3
  个节点在 STAR/ReAct 中仍然是高熵 planner 行为，不能把 exact next-tool 的
  Top-1 强行解释成确定性。修复后 locked test 的 fine-grained 自有 predictor 为
  Top-1 **0.7352**、Top-3 **0.9428**；非终止行 Top-1 **0.6931**、Top-3
  **0.9353**。在 position≥4 的 warm-prefix 子集为 Top-1 **0.8506**、Top-3
  **0.9544**。面向调度 admission 的四类 routing（observe/summarize/answer/end）
  为 Top-1 **0.9589**、Top-3 **0.9946**（非终止行 0.9645/0.9979），满足调度
  级别目标；fine-grained 全体目标尚未满足，不能用粗粒度数字冒充。
- 修复后的基线 test 结果（Top-1/Top-3/NLL）为：Markov-1
  **0.6959/0.9428/0.8517**，Markov-2 **0.7227/0.9213/0.9108**，AutoTool
  **0.6762/0.8819/1.0467**，RAG **0.6601/0.9195/1.0514**，Pythia-lite
  **0.7227/0.9177/1.0087**；Pythia 路径 coverage@1/@3 为 **0.5921/0.6601**，
  现已按完整未来 suffix 计算，不再使用旧的 tautological prefix+target 指标。
- 修复后的 core workload 为
  `collection_index_core_fixed_20260804.jsonl`、640 个 job templates（7,661
  measured nodes）以及 `workload_{train,validation,test}_core_fixed_20260804.jsonl`
  的 **20,000/1,000/1,000** episodes。五档 pressure、三种 GPU topology、三种
  initial residency 和三种 arrival pattern 均有计数，split isolation errors=0。
- resource layer 已完成 **128/128**：既有 112 条 YOLO batch 资源候选，加上新
  completion cohort 中 round-robin 选出的 16 条（STAR/ReAct 各 8 条）。最终合并
  collection index 为 `collection_index_core_resource_fixed_20260804.jsonl`，正式
  模板为 768 条、9,366 个 measured nodes。事件审计文件为
  `event_audit_core_resource_fixed_20260804.json`：768/768 run success，初始解析
  错误 177，retry 成功 144，retry 仍错误 33，严格 OOM=0，正 queue/wait=0。数量
  门槛通过，但真实等待/OOM/recovery robustness 子门槛不足，不能把 resource 层
  描述成已经完成 contention 校准。原收尾脚本生成的 `*_core_final_20260803` 保留
  为 legacy 对照，正式修复后结果以 `*_fixed_20260804` 为准。
- 最终 workload 为 `workload_{train,validation,test}_core_resource_fixed_20260804.jsonl`，
  episode 数为 20,000/1,000/1,000，五档 pressure、三种 GPU topology、三种初态和
  三种 arrival pattern 均有计数，split isolation gate=true。

原始 trace 只追加不覆盖；derived 数据必须由带版本的脚本从 trace 重建。每次正式队列保存配置快照、Git commit、模型哈希、视频哈希、随机种子和汇总报告。

## 14. 完整执行顺序

该表保留原始执行顺序。按当前验收，步骤 1–16 已完成并有 §15 产物，步骤 17 的
Phase 3 predictor 已完成但 Phase 4 六策略 Oracle-gap 尚未宣称通过，步骤 18 尚未
开始；不要把表中的历史“计划”行理解为当前仍缺少 768 条 trace。

| 步骤 | 动作 | 输出 | 通过条件/停止规则 |
|---:|---|---|---|
| 1 | 等待并收尾当前新增 32 视频 × 2 问题 × 3 baseline 批次 | batch summary、192 个 run、validator 报告 | 192 个核心槽位完成；schema ≥95%；失败槽位保留并 resume |
| 2 | 审计 64 个视频并冻结 48/8/8 split | 视频 hash、解码报告、split manifest | 64 个唯一视频；无跨 split；答案只在 eval 侧 |
| 3 | 从官方 Video-MME 标注为旧 32 视频补选第二问题 | 128 任务 source/runtime/eval manifest | 每视频 2 个互补任务；runtime 无答案；任务类型平衡 |
| 4 | 扩展运行 schema 与采集器，增加独立 Qwen3-4B planner worker 和拆分客户端 | 向后兼容的 Stack A；真实 Stack B run config | 旧测试通过；事件 model_id 与实际模型一致；无静默策略替换 |
| 5 | 执行 Qwen3-4B 50 次 PlannerDecision 测试 | 原始输出、解析统计、错误分布 | 首次 ≥90%；一次 retry 后 ≥98%；失败则暂停 Stack B |
| 6 | 执行 Stack B 8 视频/32 轨迹 pilot 与驻留测量 | 32 条候选正式轨迹、显存/加载报告 | schema ≥95%；资源字段完整；驻留策略明确且可复现 |
| 7 | 补采旧 32 视频第二问题的 Stack A 三 baseline | 96 条核心轨迹 | 96 个槽位完成并通过 validator |
| 8 | 扩展 Stack B 到 64 视频 × 2 问题 × STAR/ReAct | 256 条核心轨迹 | 256 个槽位完成；任务/模型栈平衡 |
| 9 | 校验并筛选 r02/r03 与 YOLO batch pilot | 24 条重复 seed + 32 条 batch 资源轨迹 | 配置/hash 可追溯；不跨 split；不重复计核心 |
| 10 | 采集解析 stress、cold/warm、真实 wait 和真实 OOM/recovery | 约 72 条新鲁棒性轨迹 | 正式资源层合计 128；错误/等待/OOM 门禁满足或如实报告不足 |
| 11 | 冻结必做 collection index | `collection_index.jsonl` 与统计报告 | 精确 640 核心 + 128 资源/恢复 = 768 条 |
| 12 | 在独立数据盘环境验证 Grounded-VideoLLM | import、短视频、96 帧时序 smoke | 通过则采集 32 条并将 index 扩为 800；失败则保持 768 |
| 13 | 检出 DVD 官方代码、建立隔离环境并运行 8–16 条 pilot | DVD 依赖报告、pilot trace、工具/路径统计 | 通过 DVD pilot 门禁；失败则只保留审计，不启动 128 条扩展 |
| 14 | pilot 通过后运行 DVD 64 视频 × 2 问题 × 1 baseline（先 56 视频，再补 8 视频） | 128 条独立 DVD 扩展轨迹 | 独立 DVD index；不混入 768/800 主线；缺失视频失败不计数 |
| 15 | 从冻结 snapshot 生成 semantic/compute views、enrichment、prefix samples，再编译 job templates | candidate snapshot、派生 schema、template schema、validator、模板统计 | 所有派生记录可追溯到 raw event；无未来泄漏；planner/answer/工具/模型生命周期/retry 边完整 |
| 16 | 生成完整 workload | 20k train、1k validation、1k locked test episodes | 48/8/8 视频隔离；三种 GPU 拓扑、五档压力、三类初态齐全 |
| 17 | 运行 Phase 3 预测器复现与 Phase 4 六策略实验 | P3/P4 报告、CSV、种子与配置快照、implementation/deviation 清单 | P3 split/泄漏门禁通过；预测器指标可复查；Oracle gap >1%；predictive 跨 seed 优于 myopic |
| 18 | 仅在 Phase 4 通过后执行真实 contention/Phase 5 | NVML 时间序列、膨胀系数、真实调度报告 | 单卡明确标注 stress；2+ GPU 才声明多 GPU；趋势与仿真一致 |

每一步完成后先验证再进入下一步。涉及代码修改、安装 Grounded 依赖、受控 OOM 或租用/切换多 GPU 时，按任务边界单独确认。原始 trace、失败事件和现有 dirty 第三方目录不得覆盖或重置。

## 15. 当前权威产物与偏差记录（2026-08-04）

| 产物 | 当前路径/状态 | 验收证据 |
|---|---|---|
| 正式 collection | `results/processed/collection_index_core_resource_fixed_20260804.jsonl` | 768 rows；640 core、128 resource；formal gate=true |
| prefix 输入 | `results/processed/trace_enrichment_core_fixed_canonical_v02_20260804/prefix_samples_v0_2.jsonl` | 4,441 rows；source hash 与泄漏 marker 保留 |
| job templates | `results/processed/job_templates_core_resource_fixed_20260804.jsonl` | 768 templates、9,366 nodes；split 566/94/108 |
| workload | `results/processed/workload_{train,validation,test}_core_resource_fixed_20260804.jsonl` | 20k/1k/1k episodes；split isolation gate=true |
| predictor | `results/processed/trace_predictor_core_fixed_canonical_v04_20260804.json` | locked test Top-1/Top-3=`0.8157/0.9803` |
| baseline audit | `results/processed/prediction_baselines_core_fixed_canonical_v03_20260804.json` | 含 confidence/ECE/bucket hit rate；structured backoff 单独标记 adapter |

与采集前规划的差异必须按以下方式解释：

1. Grounded-VideoLLM 的独立环境门禁未通过，因此主线保持 768，而不是强行扩到 800；
   失败原因和权重状态保留在 Grounded 小节，不把不可运行组件计入 trace。
2. 资源层虽然满足数量门槛，但 768 条中严格 OOM=0、正 queue/wait=0；因此它能生成
   workload 的资源条件和压力档位，却不能被描述为已经完成真实 contention 校准。
3. v0.4 的 overall exact gate 已通过，但 non-terminal Top-1 为 0.7891；后续若要
   宣称“所有子集均达到 0.80”，必须新增带完整 planner/stack provenance 和结构化
   intake 的独立视频并重新锁定 split，不能用 test 标签或合成 terminal 行修饰。
4. Phase 4/5 的并发策略和真实 NVML contention 校准尚未由本轮验收声称完成；下一步
   应使用上述 workload 先运行 Oracle/Myopic/predictive 的策略差异门禁，再决定是否
   进入单 GPU stress 或 2+ GPU 实验。
