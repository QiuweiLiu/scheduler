# Phase 21 — S_* task-context restoration: pre-registered pilot result

**日期**：2026-09-18
**执行机**：Windows / `F:\scheduler`（`D:\anaconda\envs\scheduler\python.exe`，torch 2.6.0+cu124，RTX 3060 Laptop）
**判定**：**PILOT FAILED — 四条预注册 calibration gate 全部不通过；按预注册规则，不得全量重生成 S_* 预测包**
**命令**：`powershell -ExecutionPolicy Bypass -File scripts\preprocess\phase21_pilot.ps1`（23 秒）
**产物**：`experiments/EXP-20260911_forecast_aware_scheduling/artifacts/phase21/`（本报告的持久副本）

---

## 1. 问题与预注册

Phase 20-B 之后定位的新 P0：S_* 锚点的 `task_context` **全部**为 `"unknown"`（9,575/9,575），
而 J 域内训练/验证行约 78% 有值。用户决定**修管线、不动模型**（同一冻结 checkpoint）。
GPT 方案（`docs/research/2026-09-17_fas_phase21_pipeline_fix_gpt.md`）：只补
`domain + official_task_type + sub_category`，其余字段保持冻结 fallback，禁止把
`question/answer/options/duration` 塞进模型；**先做 16 视频 paired masked/fixed pilot，gate 通过才全量**。

预注册 gate（改写自 `.project/EXPERIMENT_GATE.json` → `predictor_input_feature_gap_20260917`）：

- 管线：registry join 100%；`unknown_rate(domain/official_task_type/sub_category) = 0`；无全 unknown 行；
  checkpoint SHA 不变；anchor id / prefix hash 不变。
- 校准：`R50 ∈ [0.70, 1.30]`；`p50 coverage ∈ [0.45, 0.65]`；`p90 ∈ [0.85, 0.97]`；`p95 ∈ [0.92, 0.995]`；
  mean pinball 相对 masked 下降 ≥10%；next-role/family 不劣化 >2pp。

修复前基线（同 pilot，masked，n=3,226）：p50 R=0.415 / cov=0.392；p90 cov=0.829；p95 cov=0.906
→ 四条 gate 全部 FAIL（before 参照）。

---

## 2. 管线 gate：全部 PASS

| 检查 | 结果 |
|---|---|
| anchors / skipped | 975 / 576（masked 与 fixed 一致） |
| anchor 行数、id 顺序、id 集合 sha | 975 / 975，顺序一致，集合 sha 一致 |
| `prefix_hash` | **975/975 完全一致** |
| `source_trace_sha256` | 975/975 完全一致 |
| `domain` / `official_task_type` / `sub_category`（fixed） | `unknown_rate = 0.0`，`source_coverage = 1.0` |
| 同上（masked 对照） | `unknown_rate = 1.0`，`source_coverage = 0.0`（符合设计） |
| `task_context_failures` | `[]` |
| checkpoint SHA256 | `0ee8ded4f92553853026ee24a3c320f21d524f9d2c60de841091430d14949c77`（**未变**） |
| masked vs fixed 差异范围 | 仅 `model_input.task_context` 的三字段（2,925/2,925 全变）+ `task_context_audit` 溯源块；其余全同 |

**结论：修复被正确实现，masked/fixed 对比是干净的单变量对照**（唯一变化 = 三个分类字段）。

---

## 3. 校准 gate：全部 FAIL

pilot 共 **3,275** 个 slot 对（baseline 记录为 3,226；见 §5 复现性说明）。

| 分位 | masked R | fixed R | masked cov | fixed cov | gate | 判定 |
|---|---|---|---|---|---|---|
| p50 | 0.4152 | **0.3971** | 0.3911 | **0.3960** | R∈[0.70,1.30], cov∈[0.45,0.65] | **FAIL**（R 更差） |
| p90 | 1.8816 | **2.0691** | 0.8290 | **0.8443** | cov∈[0.85,0.97] | **FAIL**（差 0.006） |
| p95 | 2.6153 | **2.8285** | 0.9056 | **0.9157** | cov∈[0.92,0.995] | **FAIL**（差 0.004） |

Paired Δpinball（fixed − masked，video-cluster bootstrap）：

| 分位 | Δ mean (ms) | 相对 | CI | 显著 |
|---|---|---|---|---|
| p50 | **+51.59** | **+4.73%（更差）** | [27.23, 74.42] | **是** |
| p90 | +9.12 | +1.24% | [−25.80, 43.60] | 否 |
| p95 | +4.75 | +0.97% | [−23.45, 32.88] | 否 |

- **pinball 下降 ≥10% 的 gate 也 FAIL**：p50 反而显著恶化 +4.73%。
- 分槽（fixed，slot 1→5）：p50 R = 0.453 / 0.394 / 0.415 / 0.344 / 0.361；cov = 0.460 / 0.306 / 0.432 / 0.347 / 0.435。
  晚期槽低估更严重（slot4 R=0.344），无槽位达标。
- masked 与 fixed 的 by-slot 结构几乎同形 → 修复只带来**轻微上移**，未改变误差结构。

---

## 4. 结论

1. **metadata 缺失不是 S_* 域外失准的原因。** 恢复三个任务元数据字段（domain / official_task_type /
   sub_category）后，p50 仍只有真值的 **0.40 倍**，p50/p90/p95 覆盖率仍全部低于预注册下限，
   p50 pinball 反而显著变差。域外失准是**结构性**的，不是特征管线遗漏。
2. **按预注册规则：停止 Phase 21 全量重生成。** 不重生成 S_* 锚点/artifact，不重跑调度族，
   不触发 Phase 18/20 的重做链。
3. **对论文的正面价值**（可写）：把"域外退化"从"缺元数据"这一平凡解释中干净地排除，
   与 Phase 10（content 无可测效应）、Phase 18（Σp50 仅 0.456× 真值、Σp95 = 4.9× 真值）、
   Phase 20-B（oracle-truth 仅比 p95 好 1.3 s）互相印证：
   **`task_context` 的内容信号弱，误差主要来自"逐步预测 vs 聚合真值"的结构错配与执行期分布偏移**。
   这正是 C2（保守尾部聚合是纠偏机制）的支撑证据，而非反例。
4. **残余特征问题（同类，未修，新登记）**：`temporal_scope` OOV = 1.0、`planner_model_id` OOV = 1.0、
   `model_stack_id` OOV = 0.5015（均为冻结 fallback 字段，不属本次修复范围）。若将来再查输入侧，
   `planner_model_id`/`model_stack_id` 的 OOV 需要与本次同样口径的审计。

---

## 5. 执行中发现并修复的 3 个代码缺陷（均为端到端路径缺陷，单测未覆盖）

Mac 侧 11/11 单测通过，但**端到端 Windows 路径从未被执行过**，因此以下缺陷在 pilot 首跑暴露：

1. `scripts/build_sstar_predictor_anchors.py:246` — `KeyError: 'oov_rate_all'`：
   某字段统计缺该键 → 改为 `v.get("oov_rate_all", v.get("oov_rate", 0.0))`。
2. `scripts/build_sstar_predictor_anchors.py:238` — `KeyError: 'required_modalities'`：
   gate 引用了该字段但审计循环从未计算它。且它是**列表型字段**，经 `vocabs.modalities`（而非
   `vocabs.context`）编码，不能并入原循环 → 新增独立审计块，并把 **字面量 `"unknown"` 视为缺失**
   （冻结 fallback 的合法标记）而非类型错误。修复后 `task_context_failures` 为空。
3. `scripts/analysis/analysis_phase21_taskctx_calibration.py::resolve_artifacts` —
   `FileNotFoundError: ...\artifacts_fixed_full5\prediction_artifacts`：
   脚本按 `b05_future_h{h}.jsonl.gz` + `prediction_artifacts/` 布局查找，而 packer 实际写
   `j_future_h{h}.jsonl.gz` 于根目录 → 改为兼容两种命名、两种布局，并在传入未创建子目录时回退父目录。

**复现性说明**：baseline 记录 masked n=3,226，本次实测 3,275（+49，1.5%），
但 masked 的 R/cov 几乎逐位复现（0.415 vs 0.4152；0.392 vs 0.3911；pb 1088 vs 1090.4）
→ 判定差异来自 slot 过滤口径的轻微不同，不影响结论。**下次应记录脚本内 slot 对定义。**

---

## 6. 后续选项（需用户决策，均已预注册判据）

- **A. 停止预测侧修补，接受现状**（推荐）：把本负结果作为"域外失准非元数据问题"的证据写入论文，
  主线回到已批准的 P2（v03 pressure × horizon sweep）+ v03-confirm300 一次性验证。
- **B. 转向 Phase 17 契约修复**（`runtime_ms` 含 load 的重复计数；干净臂是 `predopt_h5_r95/r50`）：
  用户已定约束"不动预测器"，只能在消费端修；修法未选，且会重定义冠军 → 需先批准 + 预注册 equivalence margin δ。
- **C. H10-lite / cache-aware / 真实 replay**：与前两者独立，H10-lite 优先级最低。

**未做**：未改预测器、未重生成 artifact、未重跑任何调度族、未动 `T_final` 与 v03 workload。

---

## 7. 事后代码审查（2026-09-18）——并附一次**误报的撤回**

触发：用户质疑"感觉不可能"，要求独立代码审查。

### 7.1 误报（已撤回）

第一版审查声称：`baseline` / `model_stack_id` / `planner_model_id` 被 builder 以 **null** 写进 `task_context`，
遮蔽了 `stack_context` 的真实值，导致 100% 的 S_* 锚点这三个字段被编码为 UNK。
**该结论是错的，已撤回。**

**根因**：探针用 `dict.get(field)` —— 它在"键不存在"和"键存在但值为 null"两种情况下都返回 `None`，
无法区分；"键存在且为 null"是**推断**出来的，而不是测出来的。

**决定性检查**（显式测键存在性）：

```text
task_context keys = {answer_type, domain, official_task_type, question_type,
                     required_modalities, sub_category, temporal_scope}   # 975/975
baseline            key_present = False (975/975)
model_stack_id      key_present = False (975/975)
planner_model_id    key_present = False (975/975)
```

即 `task_context` **恰好只有 7 个键**，与 `build_p9d_topology_dataset._task_context()` 的输出、
以及域内 J 的布局**完全一致**。三个 stack 字段的键**不在** `task_context` 里，
所以编码器 `j_series_common.py:178-183` 的 `else` 分支正常生效、从 `stack_context` 读到真实值。

**按编码器口径逐字段复算**（真实 S_* 锚点）：

| 字段 | 键在 task_context | 解析出的值 | 编码 |
|---|---|---|---|
| answer_type | 是 | `unknown` | in_vocab |
| domain | 是 | `Film & Television` | in_vocab |
| official_task_type | 是 | `Information Synopsis` | in_vocab |
| question_type | 是 | `unknown` | in_vocab |
| sub_category | 是 | `News Report` | in_vocab |
| temporal_scope | 是 | `unknown` | **UNK** |
| **baseline** | **否** | **`langgraph_react`** | **in_vocab** |
| model_stack_id | 否 | `stack_a_qwen3_vl8b_yolo11x` | UNK |
| planner_model_id | 否 | `Qwen3-VL-8B-Instruct` | UNK |

### 7.2 修正后的残余发现（都是"有据可查的边界"，不是 bug）

- `temporal_scope`：S_* 是 `"unknown"`（OOV；域内是 duration / global / ordered_events / unspecified）。
  **registry 里没有这个字段**，GPT 批准的 Phase 21 方案明确决定保持冻结 fallback → 属**数据边界**。
- `model_stack_id`：解析正确，但**值**在约一半锚点上 OOV —— R7 用 `stack_a_qwen3_vl8b_yolo11x`，
  而域内词表里是近似的 `stack_a_qwen3_vl8b`；另一半（`stack_b_..._yolo26n`，486 个）**在词表内、能拿到真实 token**。
  这是**数据集命名漂移**，不是管线 bug。
- `planner_model_id`：解析正确，但域内词表**只有 `unknown`** 这一个值 → 该字段在域内从来不含信息，**没有信息损失**。
- `required_modalities`：S_* 是字符串 `"unknown"`；域内 J 本身就是混合的
  （`list(1)=8156 / list(2)=2029 / str=3439 / list(3)=130`）→ 字符串形态与逐字符迭代**域内也存在**，行为一致。

### 7.3 对 Phase 21 判定与论文的影响

- **§4 的总括结论恢复成立**：context 块已经"能补的都补了"；恢复三个 registry 内容字段不改善校准，
  且**不存在同类的、尚未修复的管线输入缺口**。
- 唯一仍以 UNK 进入编码器的 context 输入是 `temporal_scope`（项目里根本没有源数据）
  与 `model_stack_id` 的 stack_a 命名变体（命名漂移）。
- 论文写法：把这次管线修正作为 **reproducibility note** 披露（补 registry join、checkpoint 未动），
  不要把"输入缺口"讲成事故。

### 7.4 教训（已登记）

审计代码**必须显式测键存在性**（`field in mapping`），**不得**用 `.get()` 推断存在性 —— 这是本项目
第二次同类静默失败。建议 coverage 审计直接断言"编码器等价解析"的结果，而不是另写一套解析逻辑。

### 7.5 未做 / 审查状态

- **未做任何管线改动**：不需要修；anchor builder、编码器解析规则、pilot 产物均未变。
- 用户要求的独立 GPT 审查：桥接会话已由用户切到 **Sol + High**，brief 已提交，结论待回收；
  本节为本地端到端验证结果，**以 GPT 审查回复为准做最终确认**。
