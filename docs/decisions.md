# 项目决策记录

## 2026-07-31 — Phase 0：VideoSeek 原生基线

### 已确认事实

1. 官方 VideoSeek 固定 commit `443f8710bd1f9023f4a52ac95cb83610ad29c5ac` 能在远端 `finetooling` 环境中正常运行一条本地视频任务。
2. 一次原生运行成功退出（289 秒），产生可解析的 `prediction.json`、`trajectory.json`、完整日志和前后 GPU 快照；源码工作树未改动。
3. 实际执行是 Decord CPU 解码/图像打包加外部 Qwen API 调用。运行前后 GPU 都是 1 MiB、0%，且未观察到本地 GPU 计算进程。
4. 受远端出网速度限制，样本采用同一官方视频的 144p 版本。该选择适合最小可运行性验证，不适合视觉准确率或资源性能结论。

### 决策

| 决策 | 状态 | 理由 |
|---|---|---|
| 将本次运行作为 Phase 0 原生可运行性证据 | 通过 | 输入、代码 commit、产物与日志均可复查。 |
| 将本次预测作为模型准确率证据 | 不通过 | Agent 在轨迹中未获得明确视觉证据，且 144p 输入降低了细粒度识别可信度。 |
| 从该基线直接主张本地多 GPU 调度收益 | 不通过 | 当前路径没有本地 GPU 推理或显存占用。 |
| 进入 Phase 1 Trace Adapter v0.1 | 条件通过 | 仅以外层 wrapper 采集真实节点、CPU/API 时间和 GPU 采样；不改 VideoSeek 决策逻辑。 |
| 进入 Phase 2 | 已允许，按 preliminary gate 执行 | Phase 1 已有 3 条完整 trace；Qwen3-VL-8B/YOLO11n 介入后的 API pilot 已出现多路径和资源差异，但仍需完成 20–50 条 preliminary 后再决定 Phase 3。 |

### 后续质量门槛

1. 实现 schema、外层 collector 与 validator；用本次原始 trajectory 对照字段映射。
2. 连续采集至少 3 条完整 trace，包含一次成功节点、一次 API/工具失败或重试样例；若无法自然获得失败样例，记录缺失原因，不能伪造。
3. 对每条 trace 分离 `decode_ms`、`api_wait_ms`、本地 runtime 和 GPU 资源。没有本地 GPU 工具时明确记为 `null`，不得填零伪装测量值。
4. 只在 20–50 条预实验显示真实路径/资源差异后，才训练下一节点或资源预测基线。

### 数据与密钥政策

- 继续只使用官方可追溯入口与最小样本；不下载完整 LVBench 或许可证不一致的第三方整包。
- 所有未来下载必须先记录预估大小、来源、目标路径和可用磁盘空间。
- API Key 仅由单次进程环境变量提供，禁止写入 YAML、日志、Git 或 shell 命令行；按用户明确指示继续使用现有 Key，不自动轮换。

## 2026-07-31 — Phase 1：Trace Adapter v0.1 真实采集

### 已确认事实

1. 在用户指定的新远端 `connect.westc.seetacloud.com:12469` 上，使用未修改的 VideoSeek commit `443f8710bd1f9023f4a52ac95cb83610ad29c5ac` 完成 3 条真实运行。
2. 3 条 `trace.jsonl` 均通过 validator；事件数分别为 37、41、21，且 manifest 计数与 JSONL 实际行数一致。
3. 所有事件必需字段和 `runtime_ms` / `api_wait_ms` 资源字段均完整；三条运行均成功，未自然产生错误或重试。
4. API key 只通过远端进程环境变量使用，未进入代码、配置、manifest、trace、日志或 Git。

### 决策

| 决策 | 状态 | 理由 |
|---|---|---|
| Trace Adapter v0.1 进入真实采集验收 | 通过 | 3 条真实 trace、manifest、validator 和字段完整性均可复查。 |
| 将本轮预测作为准确率证据 | 不通过 | 输入为 144p，且本地路径无视觉 GPU 推理；本轮目标是采集而非评测。 |
| 伪造失败/重试事件补齐样本 | 禁止 | 三条真实运行没有自然失败/重试；报告记录缺失原因，不篡改 trace。 |
| 进入 Phase 2 | 已完成 preliminary，正式扩展暂缓 | 3 条 Phase 1 trace 已通过验收；21 槽位 preliminary 已完成并显示路径/资源差异，失败/重试只记录真实发生者。 |

## 2026-07-31 — Phase 2：preliminary 动态性与资源画像

### 已确认事实

1. 公开 LVBench 媒体 `wgBlACG927Y` 上完成 21 条（7 次重复 × 3 baseline）API preliminary；21/21 成功，21/21 通过 validator。
2. 21 条包含 14 条唯一路径，动作数均值 5.142857、范围 2–9，条件下一动作熵 1.751348 bit。
3. Qwen3-VL-8B 的冷/热视觉调用与 YOLO11n 的实际调用产生非空加载、推理时间和显存峰值；资源 CV 为 `local_runtime_ms=3.987209`、`load_ms=2.042224`、`peak_allocated_mb=0.173488`。
4. STAR 真实产生 36 个解析错误及其 36 条 retry 事件，但所有 run 仍成功；没有伪造失败或 retry。

### 决策

| 决策 | 状态 | 理由 |
|---|---|---|
| H1/H3 preliminary operational gate | 通过（初步） | 14/21 唯一路径和明显资源 CV；统计器观察值均为 pass。 |
| 直接进入 100–300 条正式 trace | 暂缓 | 21 条都来自同一视频/问题，尚未满足跨视频/跨问题的内容相关性证据；先补第二个可复查公开媒体。 |
| 进入 Phase 3 预测器 | 暂缓 | 需先补公开媒体多样性并复跑 Phase 2 gate，避免把同一输入重复当成内容样本。 |

### Video-MME 跨媒体补充（2026-07-31）

1. 8 个公开视频各取 1 道官方 metadata 题目，完成 21 条有效 trace；21/21 通过 validator，覆盖 8 个视频。
2. 有效样本的 `unique_paths=14`、`path_ratio=0.666667`、下一动作熵 `1.747605 bits`；
   `runtime_ms_cv=4.637008`、`local_runtime_ms_cv=5.108826`、`api_wait_ms_cv=9.901454`、
   `load_ms_cv=1.885357`、`peak_allocated_mb_cv=0.289646`，H1/H3 preliminary observation 均为 pass。
3. 为保护免费 API 配额，4 条成功 trace 使用 `qwen3-vl-plus`，17 条使用 `qwen3-vl-flash`；
   模型分组必须在后续分析中显式记录，不把模型差异当作内容因果证据。
4. 首轮错误模型名批次和两个中断目录均保留作审计证据，但不计入有效统计；没有伪造失败或 retry。

| 决策 | 状态 | 理由 |
|---|---|---|
| H1/H3 跨媒体 preliminary gate | 通过（初步） | 21 条有效 trace 覆盖 8 个公开视频，路径与资源统计均非平凡。 |
| 立即扩到 100–300 条正式 trace | 暂缓 | 先固定/分层 API 模型条件并审查 21 条跨视频样本的模型混杂，再决定正式扩展规模。 |
| 启动 Phase 3 预测器 | 暂缓 | 仍需明确模型条件、失败/重试分布和正式样本设计后再建预测基线。 |

## 2026-07-31 — Phase 2：固定模型正式 cohort

### 已确认事实

1. 正式 manifest 计划 120 次尝试：8 个 Video-MME 视频 × 3 个 baseline × 5 个重复。
2. 为消除模型混杂，120 次尝试统一使用 qwen3-vl-flash；前四个重复批次完成
   96 条成功运行，严格平衡为每个 baseline 32 条、每个视频 12 条。
3. 第五批次的 24 次尝试因免费 API 配额耗尽而失败；失败目录和错误事件均保留，未重试或伪造成功。
4. 96 条有效 trace 全部通过 validator；固定 cohort 统计为 52 唯一路径、下一动作熵
   2.070288 bits，H1/H3 preliminary observation 均为 pass。

| 决策 | 状态 | 理由 |
|---|---|---|
| 正式 cohort 模型混杂控制 | 通过（本 cohort） | planner/API model 固定为 qwen3-vl-flash，baseline 与视频分层平衡。 |
| 100 条正式样本最低门槛 | 未达到 | 有效样本为 96 条；第 5 批次因免费配额耗尽停止。 |
| 重试 quota-error 或自动切换付费 | 禁止 | 用户未授权新增付费额度；错误必须保留为真实失败证据。 |
| 启动 Phase 3 预测器 | 当时暂缓，后续已启动离线验证 | 用户随后明确接受 96 条阶段性样本，Phase 3 使用现有 trace 离线执行。 |

## 2026-07-31 — Phase 3：H2 离线预测验证

### 已确认事实

1. 用户接受 96 条固定模型有效轨迹作为阶段性正式 cohort；本阶段没有新增 API 调用。
2. 使用 leave-one-video-out，在 8 个视频之间分组，展开 601 个 prefix 样本（505 个动作目标、
   96 个终止目标）；问题文本 96/96 可从正式 manifest 复原。
3. `prefix_only_nb` 相比 `markov2` 的 Top-1 从 0.517471 提升到 0.599002，NLL 从 1.444317
   降到 1.342965，说明执行前缀本身包含预测信号。
4. `prefix_task_nb` 的 Top-1 进一步达到 0.625624，但 NLL 升至 1.409401、Top-3 降至
   0.811980；按 baseline 看，STAR 有判别性提升，LangGraph/ReAct 没有稳定提升。

| 决策 | 状态 | 理由 |
|---|---|---|
| 前缀可预测性 | 初步支持 | Prefix-only 在留一视频验证中优于 Markov。 |
| 内容特征的额外收益 | 证据混合 | Top-1 提升但概率质量与 Top-3 变差，H2 不能完整通过。 |
| 进入 Phase 4 调度收益模拟 | 暂缓 | 先扩充独立公开视频或补充可复查视觉证据，再重跑内容消融。 |

## 2026-08-01 — Phase 3：结构化状态改造

### 已确认事实

1. 现有 Phase 3 的 96 条轨迹来自 8 个独立视频；问题输入只是词袋特征，且没有把每步
   已获得的视觉证据纳入预测输入。
2. `min(prefix_length, 6)` 是小样本下的工程性截断，不是研究假设；它不应继续作为正式
   状态定义。
3. 用户确认先实现结构化状态、evidence sidecar 和任务 manifest，再扩充独立视频；本轮
   不新增 API 调用、不覆盖旧结果。

| 决策 | 状态 | 理由 |
|---|---|---|
| 采用 `state_t` v0.2 结构化输入 | 通过 | 将任务、视频元数据、已观测证据、执行前缀和 baseline policy 分开表示。 |
| 将原始问题词袋作为正式内容特征 | 否决 | 语义不稳定且不能代表视觉证据；只保留为 agent runtime input，不直接进入预测特征。 |
| 将 6+ 步合并为一个长度档 | 否决 | 改用未截断步数、进度比例和累计时间；如需分箱只在训练折拟合。 |
| 扩大独立视频而非继续增加同视频重复 | 通过 | 正式目标先定为 32 个视频，推荐 48 个；旧 8 视频作为 legacy cohort。 |
| 立即启动批量新数据采集 | 暂缓 | 先完成单视频 sidecar smoke 和 validator 验收，避免扩大错误 schema。 |

## 2026-08-01 — local_qwen 真动态复验（已完成）

1. scripted 288 条轨迹已作为兼容性控制完成：288/288 成功且 validator 全部通过，但仅有
   3 条固定路径，H1 不能据此通过；这批结果不再被当作“模型决定下一步”的证据。
2. 单视频 `local_qwen` smoke3 使用串行 Qwen3-VL-8B worker 与 YOLO11n，STAR 和 ReAct
   都产生了不同的、可解析的真实工具序列，并记录了 Qwen 冷启动、推理和峰值显存。
3. 正式动态批次改为 32 个独立视频 × 3 baseline × 1 重复 = 96 条，避免在同一题目上重复
   冷启动而把时间误当作独立证据；原 288 条 scripted 控制保留，不覆盖。
4. 远端 batch、validator、H1/H3、结构化 H2 和 Phase 4 均已完成：96/96 success、
   96/96 validator VALID；50 条唯一路径（path ratio=0.520833），H1/H3 observation
   均为 pass。
5. H2 为 `insufficient_evidence`：prefix Top-1 比 Markov2 高 0.111380，但 NLL 高
   0.093111；加入任务特征后 Top-1 仅再高 0.007264、NLL 再高 0.117757；证据增量
   Top-1 下降 0.014528、NLL 再高 0.241826。
6. burst/staggered Phase 4 的 Myopic 与 Oracle 平均完成时间差均为 0，六种策略指标
   相同，门控为 `stop_or_refine_workload`；不进入真实多 GPU 校准，避免把无收益 workload
   包装成调度贡献。
7. 已下载 96 条轻量 trace、run manifest/status 和 413 个 state sidecar；本地独立 validator、
   H1/H2/Phase4 重算与远端报告一致，远端收尾结果可复核而非只信日志摘要。
