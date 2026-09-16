# Phase 1：三类 Video Agent baseline 复现计划与过程记录

## 1. 目标

在用户指定的远端环境中，使用同一批输入和统一的 trace schema，生成以下三类可比较的执行轨迹：

1. **ST fixed baseline**：VideoTool 中按数据集写死的时空工具链；
2. **STAR baseline**：带 temporal/spatial 交替约束的动态 Planner 循环；
3. **LangGraph/ReAct baseline**：由模型在工具结果返回后自由选择下一工具的通用 Agent。

本阶段的验收目标是得到足够动态、可复查、可测量的真实 trace，用于验证路径动态性和资源异构性假设；不把替代模型的结果表述为原论文精度复现。

本轮已完成两种运行层：`scripted` 只用于先验控制流 smoke，`api` 使用远端
OpenAI-compatible Qwen 端点产生真实 Planner/Agent 输出。正式比较优先使用 `api`
目录中的结果；scripted 结果保留为控制流回归样本。

上游来源固定为 [VideoTool](https://github.com/fansunqi/VideoTool)，checkout commit
为 `9ede28ea4ca633c1878484cdc0563467c2865a64`；本地 runner 只调用其控制流，不修改
第三方源码。

## 2. 任务边界卡

### 目标

- 固定上游 VideoTool/STAR 控制流和工具语义；
- 以外层 adapter 替换当前硬件无法同时常驻的 Grounded-Video-LLM、LLaVA 等模型；
- 记录每轮 Planner 输出、解析结果、实际执行工具、约束修正、帧数、API 等待、本地运行时间和 GPU 快照；
- 让同一输入分别跑三条 baseline，产出可由 validator 校验的 JSONL trace。

### 环境

| 项目 | 当前已验证值 |
|---|---|
| 本地项目 | `/Users/liuqiuwei/Documents/调度` |
| 远端 | 用户指定的 `connect.westc.seetacloud.com:12469` |
| 远端容器 | `autodl-container-41d846924e-d62d0601` |
| 远端工作根目录 | `/root/autodl-tmp/scheduler` |
| GPU | NVIDIA GeForce RTX 4080 SUPER，32760 MiB |
| 远端可用空间 | `/root/autodl-tmp` 约 66G；根 overlay 约 9.5G |
| 已有视觉模型 | `/root/autodl-tmp/Qwen3-VL-8B-Instruct` |
| 已有检测模型 | `/root/autodl-tmp/upload/models/yolo11n.pt` |
| 已有分割模型 | `/root/autodl-tmp/upload/yolo11n-seg.pt` |
| 已有 VideoSeek | `/root/autodl-tmp/scheduler/third_party/videoseek`，固定 commit 已验收 |
| 当前已有 trace 工具 | 本项目 `tracing/schema`、`tracing/validators`、VideoSeek wrapper |

### 已获得的范围

- 用户已要求创建本计划并开始三基线复现；
- 可以在远端创建隔离环境、clone 官方仓库、安装最小依赖和生成实验产物；
- 可以在本地项目中新增过程文档、外层 adapter、测试和配置。

### 明确不做

- 不修改 VideoTool/STAR 上游业务逻辑，优先使用 wrapper/adapter；
- 不在第一轮下载完整 VideoMME、NExT-QA 或 LongVideoBench；
- 不在没有数据授权、大小和目标路径记录前下载大规模数据；
- 不把 API 等待时间当成本地 GPU 推理时间；
- API key 只通过临时进程环境传递，不写入命令行、文件或产物；手工 TTY 探测曾发生回显，
  按用户明确指示继续使用该 key，不自动轮换；
- 不伪造失败、重试或动态分支；
- 不把 Phase 1 smoke trace 当作论文准确率结果；
- 不创建或修改 API key、模型 key 的持久化配置。

### 验收标准

每条 baseline 至少完成一条真实输入并满足：

- 产生 `trace.jsonl`、`run_manifest.json`、`run_status.json`；
- trace 通过现有 validator；
- 至少包含一条实际工具事件和一条最终回答/结束事件；
- 记录 `raw_model_output`、`parsed_decision`、`executed_tool`、`fallback_reason`、`cache_hit` 等控制流证据；
- 资源字段分离 `api_wait_ms`、`local_runtime_ms`、`decode_ms`、`load_ms`，没有实测值时写 `null`；
- API key 不出现在配置、命令行、trace、manifest 或日志中；
- 任何实现偏离都在本文档的“变更记录”中追加原因、影响和验证结果。

## 3. 三条 baseline 的定义

### 3.1 ST fixed

复用 VideoTool `reasoning.py` 的数据集专用流程：

- `nextqa`：temporal grounding → patch zoom → image grid QA → image QA → summarizer → temporal QA，并在意见不一致时补充 frame selector；
- `videomme`：循环 image-grid QA，不确定时调用 image-grid selector；
- `lvb`：image QA → summarizer。

这条 baseline 的步骤主要由代码控制，模型输出只参与少量“不确定/是否补帧”判断。它作为固定流程对照组，不应包装成完全动态 Agent。

### 3.2 STAR

复用 `star_reasoning.py` 的 LangGraph StateGraph：

```text
PlannerDecision
  -> should_continue
  -> tool_executor
  -> 更新 visible_frames / tool_history
  -> PlannerDecision
  -> generalist / END
```

Planner 输出结构为：

```json
{
  "reasoning": "...",
  "tool_name": "FrameSelector",
  "tool_input": "...",
  "info_sufficient": false
}
```

控制器强制 temporal/spatial 交替、限制最大迭代次数，并对非法工具名做兜底。trace 同时保存模型原始输出和最终执行结果，以区分模型决定与控制器修正。

### 3.3 LangGraph/ReAct

复用 `reasoning.py::langgraph_reasoning` 的 `create_react_agent` 路径。工具执行结果返回给模型后，由模型发起下一次标准 tool call，直到回答或达到 recursion limit。

这一条比 STAR 更自由，但当前官方入口存在两个需要适配的问题：

1. `main.py` 使用拼写为 `langgrah` 的模式名；
2. `main.py` 将 `tool_planner_llm` 初始化为 `None`，不能直接交给 `create_react_agent`；
3. VideoTool 的 `engine.ChatOpenAI` 是自定义 EngineLM，不应假定具备 LangChain `BaseChatModel`/`bind_tools` 接口。

因此只在 adapter 中解决模型兼容性，不改 ReAct 业务逻辑。

## 4. 数据策略

### 4.1 第一轮 smoke 数据

先复用远端已有的单个 LVBench 视频：

```text
/root/autodl-tmp/scheduler/data/phase0/lvbench/wgBlACG927Y.mp4
```

已有输入问题和选项保存在本地 `configs/phase0_videoseek_qwen3_vl_plus.yaml`，直接转换成 VideoTool 所需的 `video_path/question/options` manifest。该输入用于确认三个控制流能跑通，不用于准确率比较。

### 4.2 第二轮动态性数据

按用户最新边界，热成像视频不再进入主线数据。Phase 2 只使用这些 agent 已采用的公开 benchmark 媒体/问题；没有公开标注或无法复核来源的本地视频不纳入正式统计。

### 4.3 正式 benchmark（后置）

需要正式对比时再按最小子集准备：

| 数据集 | 官方代码期望 | 计划用途 |
|---|---|---|
| NExT-QA | CSV + 按视频 ID 查找 MP4 | 短视频、多选和时序问题 |
| Video-MME | Parquet + `videoID.mp4` | 多主题、4 选项视频 QA |
| LongVideoBench | JSON + `videos/` 和 `subtitles/` | 长视频、字幕和更多时序分支 |

下载前记录来源、大小、许可证、目标目录和剩余空间。第一轮不下载完整包。

## 5. API 与模型适配方案

### 5.1 PlannerDecision adapter

当前 VideoTool `engine/openai.py` 仅对硬编码 GPT 模型名使用 `beta.chat.completions.parse`；Qwen/DeepSeek/vLLM 通常返回普通字符串。因此 adapter 必须按以下顺序解析：

1. 已有 Pydantic 对象：直接校验；
2. `message.parsed`：转成 `PlannerDecision`；
3. `message.content`：去除 markdown code fence 后执行 JSON parse，再用 `PlannerDecision.model_validate`；
4. 兼容 Qwen 常见的 `tool/input/is_sufficient` 别名，归一化为
   `PlannerDecision`，并把原始 payload 记录在 `raw_model_output`；
5. 失败：记录 `parse_error`，最多做一次严格 JSON 重试；重试事件通过
   `retry_of` 关联，仍失败时由官方 STAR 控制流安全结束，不能静默伪造工具调用。

### 5.2 标准 tool call adapter

LangGraph 路径需要模型对象支持 `bind_tools()`，并将返回值暴露为：

```text
message.tool_calls[].function.name
message.tool_calls[].function.arguments
```

若 OpenAI-compatible endpoint 不支持原生 tool call，则使用一个小型 LangChain wrapper：请求 JSON schema，解析 JSON 后转换为标准 `ToolCall`，同时保留原始响应。adapter 必须在单元测试中覆盖：正常调用、非法 JSON、未知工具、参数缺失和模型结束回答。

### 5.3 模型分工

- Planner：优先使用远程 Qwen/DeepSeek API，避免与本地 8B 模型争抢显存；
- 视觉验证：已有 Qwen3-VL-8B，限制帧数和上下文；
- 空间工具：已有 YOLO11n/YOLO 分割模型；
- 抽帧、裁剪、网格和 ROI：CPU 工具；
- 不在第一轮加载官方 Grounded-Video-LLM 和 LLaVA 全套模型。

## 6. 执行批次

### Batch A：源码和环境固定

- clone VideoTool，记录 commit、许可证和 requirements；
- 创建隔离环境，不污染已验收的 `finetooling` VideoSeek 环境；
- 安装最小 LangGraph/LangChain/OpenAI/Pydantic/OmegaConf 依赖；
- 先运行 import/工具注册 smoke，不加载大型模型；若官方模块导入会级联加载不可用的完整视觉栈，改用外层轻量注册器而不修改上游。

### Batch B：通用 trace recorder 与 manifest

- 复用本项目 trace v0.1 字段和 validator；
- 新增 VideoTool/STAR 通用 recorder，不修改第三方源码；
- 记录 planner 原始响应、解析状态、工具调用、帧变化和资源时间；
- 先用 fake tool/假的模型响应跑单元测试。

### Batch C：模型适配与工具替换

- 实现 `PlannerDecision` JSON adapter；
- 实现 LangChain-compatible planner wrapper；
- 将官方 temporal/image QA 工具替换为 Qwen3/CPU/YOLO 适配器；不直接导入会强制加载 Grounded-Video-LLM/LLaVA 的官方工具模块；
- 对每个替换保留 `original_tool_name` 和 `replacement_model_id`。

### Batch D：三条单输入 smoke

对同一视频/问题各运行一次：

```text
st_fixed
star
langgraph_react
```

分别验证执行事件、结束原因、trace 字段和输出解析。

### Batch E：动态性扩展

- 使用公开 LVBench/LongVideoBench 样本 manifest；当前远端已有的 `wgBlACG927Y` 视频作为可复查媒体，重复运行明确标记为 preliminary repetition，不当作独立题目；
- 每条 baseline 运行多条任务和重复 seed/请求，保留 API planner 的自然分支；
- 汇总路径长度、工具序列、帧数、API 等待、本地 runtime、显存和失败/重试；
- 若路径仍近似固定，记录 Go/No-Go，而不是人为增加分支。

本轮 Batch A–D 已执行完成；Phase 2 的批量入口和统计器见
`tracing/collectors/videotool_phase2_batch.py` 与
`tracing/analysis/phase2_trace_stats.py`，正式过程记录见
`docs/phase2_dynamic_trace_collection.md`。

## 7. 变更记录

| 时间 | 变更 | 原因 | 影响 | 验证 |
|---|---|---|---|---|
| 2026-07-31 | 创建本复现计划 | 用户要求规划并开始三基线复现 | 尚未改动第三方源码 | 已完成本地/远端只读核对 |
| 2026-07-31 | 固定 VideoTool checkout `9ede28ea4ca633c1878484cdc0563467c2865a64`，创建 `videotool_phase1` 隔离环境并安装最小 LangGraph/API 依赖 | 远端没有官方 checkout，`finetooling` 缺少 LangGraph 组件 | 不污染已验收 VideoSeek 环境 | `PlannerDecision` 与 `star_reasoning` import smoke 通过 |
| 2026-07-31 | 不安装官方完整 `requirements.txt`；ST/ReAct 改用轻量外层工具注册器 | 官方 `reasoning.py` 导入会级联 `cv2`、Torch、Transformers、LLaVA/Detectron2 等重量级依赖，当前单卡和 Phase 1 trace 目标不需要全部常驻 | 保留控制流语义，工具实现标记为替代模型；不能宣称原始模型级复现 | 官方 `reasoning.py` import 在 `cv2` 缺失处停止，问题已复现并记录 |
| 2026-07-31 | 新增 `tracing/collectors/videotool_phase1.py`，不改 VideoTool checkout | 需要统一运行 ST、官方 STAR 图和 LangGraph `create_react_agent` | 统一输出 trace/manifest/status；视觉工具由 CPU metadata adapter 和可选 YOLO11n adapter 承载 | 本地语法检查与既有 13 项测试通过；远端三条 scripted 成功 trace 均 `VALID` |
| 2026-07-31 | LangGraph fake chat 初次使用错误字段 `messages`，远端第一次 ReAct smoke 失败 | `langchain-core 0.3.x` 的 `FakeMessagesListChatModel` 实际字段是 `responses` | 保留失败目录作为问题证据，修正后 ReAct 步骤号和标准 tool call ID 正常递增 | 修正后 scripted ReAct 成功，步骤 1→2→3→4，trace `VALID` |
| 2026-07-31 | Qwen Planner 增加 `tool/input` 别名归一化和一次严格 JSON retry | OpenAI-compatible 端点返回了非 STAR 原生字段，且偶发长解释文本无 JSON | 不改上游 Planner；保留原文、解析状态、`retry_of` 和失败事件 | 真实 API STAR 成功完成，包含 4 个解析错误/重试对并通过 `VALID` |
| 2026-07-31 | Phase 2 外层视觉 adapter 接入 Qwen3-VL-8B 和 YOLO11n；新增串行 Qwen worker、批量 manifest 和动态性统计器 | 用户要求 8B 与 YOLO 真正介入，并进入指导书规定的 Phase 2 路径/资源画像 | Qwen 只在独立 `finetooling` worker 中单卡常驻；YOLO 按工具调用串行加载；trace 分开记录 `load_ms`、视觉推理时间、显存峰值和 YOLO 推理时间 | 1 条 scripted + Qwen/YOLO ReAct 和 3 条 API pilot 均通过 `VALID`；Qwen 实测约 16 GiB 峰值，YOLO 实测加载/推理字段非空 |
| 2026-07-31 | Phase 2 数据边界改为公开 benchmark 主线，明确不使用热成像视频；API key 条款改为按用户指示继续使用现有 key、不自动轮换 | 用户最新边界和凭据要求 | 公开数据来源、重复性质和下载限制写入 Phase 2 过程文档；key 仍只进临时环境 | 远端 public metadata 直连超时已记录，未下载完整数据集或把不可复核视频纳入统计 |
| 2026-07-31 | 空最终答案不再标记为成功 | 一次真实 ST API 调用出现 generalist API 错误但旧 runner 仍写入 success | 运行器现在抛出 `empty final answer after baseline execution` 并写 error run event | 修正已通过语法检查；后续运行按该门槛验收 |
| 2026-07-31 | 为 LangGraph `ChatOpenAI` 增加 `_generate` 外层计时记录，并修正 ReAct 步数摘要 | 初版 ReAct trace 只有标准 tool call，没有独立 `api_wait_ms` 事件；后续核对发现实际为 7 个工具调用 | API 等待与本地工具 runtime 分离；不改变 LangGraph tool binding | 新 ReAct run 17 events，7 个 tool call，`api_wait_ms` 9083.4ms、`local_runtime_ms` 11648.9ms，trace `VALID` |
| 2026-08-01 | 新增 `planner_mode=local_qwen`，planner/视觉/答案共用串行 Qwen3-VL-8B worker，并完成 32 视频 × 3 baseline × 1 的 96 条动态 cohort | scripted 控制只有固定路径，不能作为“模型决定下一步”的证据；用户要求 8B 与 YOLO 真正介入 | 保留 raw planner output、JSON/bare-tool 解析状态、Qwen load/inference/VRAM 与 YOLO 字段；不调用 API、不改第三方 checkout | 96/96 success、96/96 validator VALID；H1/H3 pass；H2 insufficient_evidence；Phase 4 oracle gap=0，停止真实多 GPU 校准 |
| 2026-08-02 | 收集器新增 `--yolo-batch`/`--yolo-preobserve`，使用 YOLO11x 多帧检测驱动 Qwen planner，并完成 8 视频 × 2 动态 baseline × 7 batch 的 112 条 pilot | 原 96 条 local-Qwen cohort 中 planner 实际未选择 YOLO，单纯更换权重无法验证 batch 对决策轨迹的影响 | 新增真实多帧 batch、实际有效帧数、检测数与显存字段；旧 cohort 不覆盖；保留混合目录作为中断审计，不纳入正式统计 | 112/112 success、112/112 validator VALID；batch 1→64 路径变化比例 68.75%→100%；YOLO allocated 372.672→5474.410 MB |

后续每次遇到依赖、API、显存、数据或控制流问题，都在此表追加一行；若改变执行路径，也同步修改第 3–6 节。

## 8. 已完成运行与证据

远端统一输出根目录：

```text
/root/autodl-tmp/scheduler/results/raw/phase1_videotool/
/root/autodl-tmp/scheduler/results/raw/phase1_videotool_api/
```

### 8.1 Scripted control-flow smoke

| baseline | run_id | 状态 | 事件数 | 实际工具序列 | validator |
|---|---|---|---:|---|---|
| `st_fixed` | `wgBlACG927Y_st_fixed_1785471244` | success | 5 | `image-qa → summarization-tool` | VALID |
| `star` | `wgBlACG927Y_star_1785471245` | success | 13 | `frame-selector → image-qa → temporal-qa → image-grid-qa → summarization-tool` | VALID |
| `langgraph_react` | `wgBlACG927Y_langgraph_react_1785471377` | success | 6 | `FrameSelector → ImageQA → TemporalQA → ImageGridQA` | VALID |

### 8.2 Real API run

| baseline | run_id | 状态 | 事件数 | 关键结果 | validator |
|---|---|---|---:|---|---|
| `st_fixed` | `wgBlACG927Y_st_fixed_1785471611` | success | 5 | Qwen generalist 输出 `(D) Lion`，解析为 `D` | VALID |
| `star` | `wgBlACG927Y_star_1785472077` | success | 21 | 动态 temporal/spatial 序列；4 次 Planner parse error 后各重试成功；最终 `D` | VALID |
| `langgraph_react` | `wgBlACG927Y_langgraph_react_1785472625` | success | 17 | 7 个标准 tool call，步骤 1→7；另记录 8 次 ReAct API 调用；最终回答为 “lion” | VALID |

ReAct trace 的每个 action 都包含 `standard_tool_call.name`、`args`、`id`，例如
`TemporalGrounding`、`FrameSelector`、`ImageGridQA`、`ImageQA`。这证明当前
OpenAI-compatible 输出可以被 LangGraph 解析为标准工具调用；本轮不把自然语言最终回答
强行转换成选项字母。

### 8.3 失败与重试证据

- `wgBlACG927Y_langgraph_react_1785471247`：初次 fake chat 构造使用了不存在的
  `messages` 字段，run error；保留以证明实现问题没有被隐藏。
- `wgBlACG927Y_star_1785471749`：Qwen 返回的 Planner payload 字段不符合原生
  `PlannerDecision`，记录 parse error 后安全结束；随后修正别名归一化并重跑。
- `wgBlACG927Y_star_1785472077`：运行中出现 4 个 parse error，每个都有一个
  `retry_of` 指向严格 JSON 重试，最终完成成功。
- 一次较早的 ST API generalist 失败产生空答案；运行器已加空答案门槛，后续不再把它当作
  成功结果。

### 8.4 解释边界

当前 trace 的动态性来自真实 Qwen Planner/Agent，但视觉工具实现是明确标注的轻量替代：
CPU 帧元数据、CPU lazy 抽帧和可选 YOLO11n；没有把 Grounded-Video-LLM、LLaVA 或
本地 Qwen3-VL-8B 全套权重同时常驻。因此这些结果可用于验证控制流、解析、失败/重试和
资源字段假设，不可作为 VideoTool/STAR 原始模型的准确率复现。

### 8.5 local_qwen 动态 cohort

单视频 smoke3 的 STAR 序列为 `image-grid-qa → temporal-qa → image-qa → image-qa`，
ReAct 序列为 `temporal-grounding → frame-selector → image-grid-qa → temporal-grounding`；
两条序列的 planner JSON 与标准 tool-call envelope 均在 trace 中可复查。正式 96 条 cohort
的结果为：50 条唯一路径、path ratio 0.520833、路径长度 2–4（均值 3.302083）、下一动作
熵 2.086501 bits；按 baseline 唯一路径数为 `st_fixed=1`、`star=23`、`langgraph_react=28`。
runtime/local-runtime CV=1.562125，load CV=2.480231，peak allocated CV=0.064338，API
wait=0；1 条 retry 链和 2 个真实解析错误事件均保留。

结构化 H2 在 32 个视频/413 个 prefix 样本上为 `insufficient_evidence`，而不是完整通过：
prefix Top-1 比 Markov2 高 0.111380，但 NLL 高 0.093111；任务增量 Top-1 仅高 0.007264
且 NLL 高 0.117757；evidence 增量 Top-1 降 0.014528、NLL 高 0.241826。burst/staggered
Phase 4 的 Myopic 与 Oracle 平均完成时间差均为 0，六策略指标相同，门控为
`stop_or_refine_workload`。

复查产物：

```text
results/raw/phase3_localqwen_videomme_32_r01/phase2_batch_summary.jsonl
results/processed/phase3_localqwen_videomme_32_r01_stats.json
results/processed/phase3_localqwen_h2_20260801/
results/processed/phase4_localqwen_burst_20260801/
results/processed/phase4_localqwen_staggered_20260801/
```

## 9. 当前状态与下一步

当前已确认的边界：

- 本地/远端已有 1 个 LVBench 视频和 32 个 Video-MME 公公开视频（新增 24 个已校验 MP4），
  已完成 21 条跨媒体 preliminary；
- 已完成 120 次正式固定模型尝试，其中 96 条有效成功 trace；有效 cohort 为 8 视频 ×
  3 baseline × 4 重复，另有 24 条真实 quota-error 目录被排除；
- 一张 32GB GPU 不能原样常驻官方 temporal model + LLaVA；当前采用串行 Qwen3-VL-8B worker；
- YOLO11n 已作为真实工具调用介入并记录加载/推理/显存字段；这些结果仍不是原始 VideoTool/STAR
  模型级准确率复现。
- local_qwen 96 条动态 cohort 已通过 H1/H3 observation，但结构化 H2 未通过完整规则，
  Phase 4 Oracle gap=0，当前不进入真实多 GPU 校准。

下一步：

1. 正式 cohort 已固定 planner/API model 并完成分层统计；不再把模型差异解释为内容差异；
2. 免费 quota 已耗尽，不补齐第 100 条，也不启动新的 API 运行；
3. 已使用 scripted 288 条控制和 local_qwen 96 条动态轨迹完成 Static/Markov/结构化状态
   离线验证；下一步是改进证据/概率校准或换用更有调度差异的 workload，不是立即租用真实多 GPU。

Phase 3 已对 scripted 控制与 local_qwen 动态 cohort 分别执行离线验证；local_qwen 的
prefix Top-1 提升但 NLL 变差，任务/证据增量不稳定，H2 记为 `insufficient_evidence`。
Phase 4 已完成反事实回放但 Oracle gap=0，因此不启动真实多 GPU 调度；详细结果见
`docs/phase3_predictor_validation.md` 与 `docs/phase3_structured_state.md`。
