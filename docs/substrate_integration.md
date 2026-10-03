# 模拟器实测数据接入：实现范围与验证

本次是工程接入，不是新一轮正式实验，也不代表全部机制已经实现或真实系统回放已通过。

## 单任务与双任务使用同一套记录

`GPU.active_tasks` 是显卡占用的唯一记录。普通任务、嵌套显卡阶段和提前加载都在其中登记。
忙碌时间按占用区间的并集计算，重叠时间只计一次；它是**模拟设备占用率**，不是实测显卡计算核心活跃率。

共置（让同一显卡同时运行两个任务）目前支持最多两个普通任务：模型已经驻留、显式指定
`workload_shape`，且硬件身份和模型/形状组合在 F1 实测表中有覆盖。模型只占一份驻留显存，
所有运行任务的工作区显存合计，不能驱逐正在使用的模型。
任务加入、退出或被抢占时更新剩余工作量、减速系数与完成事件。加载和驱逐不按推理减速系数缩放。

加载结束与提前加载结束都是显式时间边界：普通任务的 `gpu_transition` 事件提交模型可执行驻留，
该时刻重新调度；预取的结束时刻无论显卡上是否还有推理，都会参与下一事件时间计算。
不允许追溯消费早于仿真当前时间的完成事件。普通加载和预取加载都先计预留显存，不提前算可执行驻留。

组合节点的字段来自 `attach_nested_resources()`：`nested_model_mb` 实际是内层调用的已分配显存峰值，
`nested_reserved_mb` 是保留显存峰值，**不是独立测量的权重大小/精确工作区拆分**。
继续沿用已分配峰值作为模型缓存的保守代理，并把峰值需求超过该缓存的部分计入增量显存；
存在 reserved 值时取两种峰值的较大值。没有 reserved 时只支持 allocated 峰值代理，
不能声称覆盖未知的额外 allocator reservation。暂时被其他任务工作区占用时延后重试，不当成永久 OOM。
新组合模型在内层调用结束后才标记可执行；内层的加载已包含在原始内层耗时，不重复添加。

F1 是整段任务平均减速测量；把该系数用于重叠期间的工作进度是一种分段近似，**不是已经验证的瞬时速度模型**。
后续必须通过真实回放校准，不能仅凭单元测试宣称端到端精度或正式策略排序正确。

明确未支持：超过两个普通任务、冷加载/驱逐与推理共置、融合链的共置和抢占，以及预测式多步推演策略
（`pred_mpc_*`）的共置推演。后者会直接拒绝运行，不悄悄使用串行推演评估并发动作。

**策略公平性限制**：原有策略评分没有因 F1 接入而自动获得并发代价模型。
启用共置时，它们是“不感知共置代价的对照”，输出
`colocation_policy_semantics=colocation_unaware_control_not_faithful_concurrent_baseline`。
报告必须明确标成 `<原策略>-colocation-unaware` 敏感性对照，不能称已忠实复现论文的并发调度器。
本次不静默给论文策略添加 F1 惩罚或更改其核心评分；没有正式并发基线比较结论。

## 部署口径：多进程（deployment=multi_process）

共置表分两种部署口径，由 `deployment` 字段区分：

- `single_process`（默认）：旧的单进程 harness 测量面（线程+双流）。它描述的是单进程 Python
  下的共置行为，其主导机制是 GIL 发射串行化；**不能当作多进程部署的通用共置代价**。
- `multi_process`：与真实采集系统一致的每模型 worker 进程部署（`EXP-20261003_mp_calibration_v1`
  实测面）。该口径下：
  - **同模型对 = 串行**：并发同模型请求在共享 worker 中排队，不作为重叠收益处理
    （多进程 profile 禁止包含同模型单元，查表直接返回未覆盖 → 等待空闲显卡）；
  - **异模型对**：使用多进程实测减速（8B 涉及的对 1.24–1.60；小模型对 1.02–1.18）。

多进程共置面为筛选后的单元格（7 格），未覆盖的模型/形状组合仍按原有 fail-closed 规则延后串行。

## 提前加载与抢占

F4 保留纯拷贝与完整框架加载两种实测证据，并提供两代干扰模型：

- **乘性（legacy）**：`full_slowdown`/`full_dilation`，仅重叠加载期间按系数减速，加载结束复位；
  实测证据仅对应中等档（`medium`）探针，不能推广到任意输入长度、图像或运行引擎。
- **加性（多进程部署）**：`interference_model=additive_extra_ms`，按实测
  **(infer_model, infer_shape, load_model)** 格给出 `extra_ms`，在重叠开始时**一次性**
  加入运行任务的剩余工作量（同时进入 `stall_remaining_ms` 预算，见请求级抢占一节的相位语义）；
  加载时长不缩放（无 dilation），加载结束不复位。
  支持域 = 产物 `prefetch_interference_additive_v2.json` 的 7 个实测格；同一被加载模型跨推断
  形状跨度可达 35×（3B 载入：4B medium +17.7ms vs 4B long +633.5ms），因此**未覆盖组合明确
  登记 `missing_measured_cell` 并延后串行加载**，不用 load-only 中位数冒充。
  依据：F4 与 mp 校准显示干扰是"加性挡停"而非乘性倍数（小任务的 18× 是比值错觉）。

配置构建器对 legacy 证据选择完整加载（`full`）；实测 `copy` 模式暂时拒绝执行，因为纯拷贝证据
不保证模型已经可执行；合成数据的 copy 模式仅用于验证事件数学。
未覆盖组合/形状/因子会明确登记原因并延后串行加载，不把未知干扰算成零成本。
提前加载完成之前，模型只占用预留显存，不算可执行的驻留模型。
同卡任务提前完成/恢复单任务速度时，尚未开始的串行加载和组合阶段预约也会重排，避免旧预约留下虚假空闲。

抢占默认采用原有的**整节点重算**规则。启用请求级 profile 后切换为**调用级断点恢复**（见下节）；
两种语义绝不混用。选择受害者前检查释放其工作区后的显存与目标可执行性；保留并更新同卡其他任务。
不为无法与剩余任务一起派发的目标无意义地抢占，不抢占仍有预取或组合阶段预约的显卡。
普通模型处于加载/驱逐阶段时也不能抢占；加载时间不会计成已经丢弃的推理进度。

## 请求级抢占（REQUEST_RECOMPUTE，opt-in）

在既有"整节点重算"之外新增**调用级抢占**作为独立模式（`request_preemption.mode=REQUEST_RECOMPUTE`）。
未启用请求级 profile 时保持原节点级语义，两者不混用。

- **时间线拆分（执行真值，从不进入任何调度器可见视图/评分/动作可用性/受害者选择）**：
  `load -> prefill（原子，不可抢） -> decode（按 token 边界可抢）`。
  拆分以**内在工作量**表达：`prefill_work_ms = P̂ = intercept + rate×n_in`（n_in 取 token 先验层中位）；
  `decode_step_work_ms = d_r = (compute_ms − P̂)/n_out`；**总服务时间仍钉在 trace 真值**。
- **相位时钟按已消费内在工作量推进**（`consumed = total_work − intrinsic_remaining_work`，由 GPU
  账本维护）：batching 因子、F1 共置减速、伙伴退出复位都会一致地移动 prefill/decode/token 状态，
  不再用派发时固定的墙钟偏移。**加性干扰**进入独立的 `stall_remaining_ms` 预算：先消耗干扰工作量、
  期间 request token 进度**暂停**（既不倒退、也不把干扰算成 token 进度），预算耗尽后相位时钟恢复。
- **token 边界**：合法抢占点 = `prefill_work + m×d_r`（m ≥ 0，含 prefill 结束点）。到达/决策
  落在 token 中间时**推迟到下一合法边界**（`victim_deferred_to_token_boundary`），prefill 期间
  报告 `victim_prefill_in_progress`；两者都会调度一个边界唤醒（`preemption_boundary` 事件）。
  唤醒只是提示：相位在每次决策时从工作量重导，早醒/晚醒都不会在 token 中间抢占。
- **受害者选择（信息边界）**：先由**调度器可见信息**（service class、队列年龄、deadline slack、
  结构/显存/驻留检查）选出唯一 victim；执行层只能执行该动作、或在物理不可执行时报告/推迟，
  **不得读取隐藏相位后换选另一个 victim**。
- **恢复**：重新派发时工作量 = `R_m(n_ctx) + 剩余 token × d_r`，其中
  `R_m = intercept + rate/1000 × n_ctx`、`n_ctx = n_in + k`。
  R_m 已由 P1 往返实验验证（4B/8B SUPER 残差 −1.8%~−5.6%；**3B 补测 +1.5%/+2.6%**（克隆实例，
  4B 对照 −1.8%，`EXP-20261003_substrate_resume_3b_v1`））；`preempt_recompute_ms` 记 R_m。
- **阻止/推迟条件（显式登记原因）**：`victim_prefill_in_progress`、`victim_no_request_split`、
  `victim_decode_complete`、`victim_model_not_covered`、`victim_deferred_to_token_boundary`。
- **不适用范围**：CPU 节点、YOLO、融合链、组合/嵌套节点（无拆分 → 不可请求级抢占）。
- **配置**：`request_preemption` 需要 `preemption_enabled=true`（构建器 `--request-preemption` 会同时置位）。
  产物 `experiments/EXP-20261003_substrate_preemption_resume_v1/artifacts/request_preemption_profile_v1.json`
  （token 先验层中位 + F2 相位率）。
- **估算口径**：token 数取层中位（确定性），不称"恢复真实 token"；真实 trace 无逐节点 token 字段。

## 批处理引擎模式（vllm_batched，opt-in）

多进程部署下同模型并发默认串行（共享 worker 排队）。启用 `batching_engine` 后，同模型对按
**实测批处理曲线**合并为一个批次执行（vLLM 连续批处理是部署升级假设，微基准为证据）：

- **语义（每请求延迟因子，不是吞吐加速比）**：曲线值 `f(B) = wall_ms(B) / wall_ms(1)`（实测同质批次
  的墙钟比，≥1）。批内每个请求按速率 `1/f(B)` 运行；聚合吞吐增益 = `B / f(B)`。
  **不得**把吞吐加速比 `T1/(wall/B)` 当每请求延迟——那会把"两个请求已经并行"再算一次，
  使 B=2 的 aggregate throughput 虚增约一倍（评审 P0）。
- **准入（engine admission）**：`model|role` 完全相同（mixed-role 未测 → 串行）；KV 池已知且
  `cap = pool // (in+out) ≥ 2`（缺失即拒绝，fail-closed）；两请求的**调度器可见预测时长**比值
  ≤ `homogeneity_tolerance`（产物 1.25）——运行任务在派发时冻结自己的预测值，候选用估计行；
  **不使用真实 trace 时长**（否则会经 `dispatchable_gpu_indices` 泄漏未来真值）。B2 探针只测
  同质批次，明显异质对未测 → 串行。
- **批次内伙伴离开**：先完成者离开时幸存者复位到单任务速度（已消费工作量按因子折算，无重复计时）。
- **跨模型**：仍走 HF 多进程共置表（`cross_model_policy=mp_table_proxy`）。B3 双引擎对照显示
  8B 侧 vLLM 与 HF 一致（1.29 vs 1.24），小模型侧不对称（3B 0.98 vs 1.27），差异已在产物登记。
  validator 强制：`vllm_batched` 不得与单进程 F1 表组合（cross-profile 一致性门）。
- **冷加载/驱逐过渡区间不批处理**（显式报错），与共置同规则。
- **配置**：构建器 `--batching-engine experiments/EXP-20261003_substrate_batching_curves_v1/artifacts/batching_engine_profile_v2.json`。
- **不包含**：前缀缓存（无前缀身份证据）、跨任务混合模型批、vLLM 调度器内部重排。

## 模型阶段、批处理与缓存的真实边界

F2 接入仅暴露实测系数。原始节点没有完整请求长度和图像数量，因此不推断图像数量、
不输出伪造的阶段耗时。嵌套显卡阶段暂不在该观测覆盖范围内。

vLLM 第 5/6/7 项只支持读取来源并明确报告未支持状态：
- 连续批处理（第 5 项）已作为独立引擎模式 `vllm_batched` 接入（见上节），以**每请求延迟因子** +
  KV 池上限 + 同质容差表达；未覆盖层/超池/异质组合 fail-closed 退回串行。
- 前缀缓存缺少可核对的公共输入前缀身份——**永久不支持**，不以任意假设替代。
- 第 7 项是显存压力总代价，缺少实际抢占次数，不能当精确抢占代价。

这些证据不用于缩放 HF 主执行路径；已有 YOLO 的帧内批处理也不等于跨任务大模型请求批处理。
**e 的批处理部分已完成（opt-in）；前缀缓存部分永久不可做（无前缀身份数据）。**

## 使用已有配置入口

单进程口径（旧）：

```sh
export PYTHONPATH=src
python scripts/build_substrate_extension_config.py \
  --transition-config experiments/EXP-20261001_substrate_transition_integration_v1/transition_config.json \
  --colocation experiments/EXP-20260929_substrate_colocation_surface_v1/artifacts/colocation_cost_table_v1.json \
  --prefetch-interference experiments/EXP-20260929_substrate_overlap_matrix_v1/artifacts/f4_overlap_matrix.json \
  --output /tmp/substrate-extension.json
```

多进程口径（部署 of record，含加性加载干扰）：

```sh
python scripts/build_substrate_extension_config.py \
  --transition-config experiments/EXP-20261001_substrate_transition_integration_v1/transition_config.json \
  --colocation experiments/EXP-20261003_mp_calibration_v1/artifacts/mp_colocation_table_v1.json \
  --prefetch-interference experiments/EXP-20261003_mp_calibration_v1/artifacts/prefetch_interference_additive_v2.json \
  --output /tmp/substrate-extension-mp.json
```

多进程 + 请求级抢占 + 批处理引擎（当前完整口径）：

```sh
python scripts/build_substrate_extension_config.py \
  --colocation experiments/EXP-20261003_mp_calibration_v1/artifacts/mp_colocation_table_v1.json \
  --prefetch-interference experiments/EXP-20261003_mp_calibration_v1/artifacts/prefetch_interference_additive_v2.json \
  --request-preemption experiments/EXP-20261003_substrate_preemption_resume_v1/artifacts/request_preemption_profile_v1.json \
  --batching-engine experiments/EXP-20261003_substrate_batching_curves_v1/artifacts/batching_engine_profile_v2.json \
  --output /tmp/substrate-extension-v4.json
```

输出通过已有运行器 `scripts/r8_scheduler_extensions_matrix.py --extension-config` 使用。
提前加载需要原有显式 `prefetch_plan` 或 Latency-Aware 在线计划；构建器不会凭空生成计划。
旧任务数据缺少显式形状/硬件身份时必须补齐有证据的输入契约，不能根据未来真值或任意默认值猜测。
不能把该配置直接称为旧数据上已验证的完整共置实验。

## 最小验证

```sh
PYTHONPATH=src python -m unittest discover -s tests -p test_workload_v02_simulator.py
PYTHONPATH=src python -m unittest discover -s tests -p test_composite_segment.py
PYTHONPATH=src python -m unittest discover -s tests -p test_workload_v02_state_adapter.py
```

测试包含手算双任务非对称减速、反向查表、真实 F1 表加载、正冷加载成本下的驻留任务共置、
区间去重、显存保护、抢占剩余任务、未知 F4 因子延后、GPU 身份不匹配、加载时间不消耗推理进度、
请求级断点恢复与泄漏检查、token 边界推迟、批处理准入与幸存者复位、异质对串行、mixed-role 串行、
KV 池上限/缺元数据拒绝、加性支持域 fail-closed 以及非法批处理曲线拒绝。
真实实测表测试使用人工短任务，只验证接入链路，不构成原始 workload 的真实回放证据。
