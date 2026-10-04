# RESULT — EXP-20261004_real_strata_calibration_v1（真实档位全表校准）

**状态**：完成（2026-10-04）。测量 runner 原样入库：`scripts/real_strata_calibration.py`
（SHA256 `1c5c806ee8fdf1f6c5df053738b0ef88b2604817e5cade6767fed0d710210c50`，
与克隆实例 `:19351` 上实际执行的字节一致；该实例 RTX 4080 32GB，设备名 `NVIDIA GeForce RTX 4080`，
已关机）。原始产物 `artifacts/real_strata_calibration.json`（逐轮原始值 + 每格耗时）。

## 背景与目的

冻结工作负载的 profile 依赖机制（共置、加性加载干扰）此前按"通用档位"（xs/short/medium/long/xl）
测量，与真实请求的 **per-(model, role) 尺寸**对不上号（审计需求探针：真实组合集中 8B↔4B）。
本实验按真实档位**全表**测量：覆盖冻结工作负载能产生的**每一个**组合，消除 fail-closed 缺口。

**档位定义**（= `request_preemption_profile_v1` 的 token 层中位，与批处理/抢占 profile 同一词表）：

| model | role | in | out | images |
|---|---|---:|---:|---:|
| Qwen3-VL-8B | planner | 1055 | 47 | 0 |
| Qwen3-VL-8B | videotool_spatial | 264 | 50 | 1 |
| Qwen3-VL-8B | answer_generation | 574 | 2 | 0 |
| Qwen3-4B | planner | 1072 | 36 | 0 |
| Qwen2.5-VL-3B | answer_generation | 442 | 3 | 0 |
| Qwen2.5-VL-3B | videotool_spatial | 353 | 35 | 1 |

两个 spatial 档位按 token 先验的 image_count（p50=1）带 1 帧图，文本长度自动校准到
"文本+图像 = in 中位"；文本档位与既有测量谱系一致（text = in，模板开销另计）。

## 协议

- 1 预热 + 3 轮、交替 solo 顺序、取中位；镜像 `mp_calibration.py`。
- **worker 复用**：Part A 每个模型对、Part B 每个推理模型只加载一次 bf16 权重，探针按请求档位重建；
  每格计算与逐格重启一致（节省重复加载时间）。
- Part A（共置）：3 组模型对 × 全档位组合 = **11 格**。
- Part B（加载干扰）：6 个推理档位 × 其余 2 个被加载模型 = **12 格**。
  同模型加载**结构上不可达**（模型在跑即已驻留，不会再加载）→ 不测、明确登记。

## 结果 A：共置（11 格，slowdown_a/b）

| 格子 | slow a/b | | 格子 | slow a/b |
|---|---|---|---|---|
| 8B[answer]∣4B[planner] | **1.786** / 1.085 | | 8B[planner]∣3B[answer] | 1.028 / **1.453** |
| 8B[planner]∣4B[planner] | 1.164 / 1.150 | | 8B[planner]∣3B[spatial] | 1.088 / 1.158 |
| 8B[spatial]∣4B[planner] | 1.135 / 1.090 | | 8B[spatial]∣3B[answer] | 1.039 / 1.155 |
| 8B[answer]∣3B[answer] | **1.403 / 1.481** | | 8B[spatial]∣3B[spatial] | 1.063 / 1.110 |
| 8B[answer]∣3B[spatial] | 1.387 / 1.072 | | 4B[planner]∣3B[answer] | 1.046 / **1.552** |
| | | | 4B[planner]∣3B[spatial] | 1.038 / 1.081 |

**发现**：小请求（8B answer ~175ms、3B answer ~118ms）拿到最大的相对 slowdown（1.4–1.8×）——
固定争用开销对小任务占比高；大请求（8B planner ~2.2s）对 1.03–1.16。与通用档位时代
"极端不对称小任务异常"结论方向一致，但数值按真实档位重定。

## 结果 B：加载干扰（12 格，extra_ms）

| infer ← load | 8B | 4B | 3B |
|---|---:|---:|---:|
| 8B[planner] | — | 3.6 | 29.0 |
| 8B[spatial] | — | **−7.5 → 0** | 33.8 |
| 8B[answer] | — | 10.7 | 13.1 |
| 4B[planner] | 33.1 | — | 8.4 |
| 3B[answer] | 5.5 | 4.0 | — |
| 3B[spatial] | 7.1 | 24.8 | — |

**发现**：真实档位下加性干扰 **≤ 34ms**（通用档位时代 17.7–634ms 是长输出探针产物）；
负值（−7.5ms）为 3 轮分辨率的噪声，按加性模型非负语义**归零**并在产物保留原始值
（`extra_ms_raw` / `extra_ms_clamped_to_zero`）。load 侧 dilation 0.89–1.11（与既有结论一致）。

## 派生产物与契约

- `artifacts/real_strata_colocation_table_v1.json`（schema `colocation-cost-table-v1`，11 格）
- `artifacts/real_strata_interference_additive_v1.json`（schema `prefetch-interference-additive-v2`，12 格）
- 派生脚本 `scripts/derive_real_strata_profiles.py`；两者均通过仿真器 loader 校验。
- **契约** `src/tracing/analysis/profile_contract.py`：`node.workload_shape := node_type`（覆盖模型
  的每个档位必须有实测格，否则 fail-closed）；`episode.gpu_identity :=` 节点级 `gpu_model` 统一证据
  （混合/缺失/跨类 fail-closed）。Node 增加 `gpu_model` 证据字段（loader 透传）。
- **GPU 身份门改为同类比较**：本次测量活动横跨两张同档 32GB RTX 4080 卡（原始实例驱动名
  "RTX 4080 SUPER"、克隆实例 "RTX 4080"）；门比较设备类（仅忽略尾部 SUPER），精确字符串保留在
  产物与事件日志；其他型号仍 fail-closed。回归测试
  `test_gpu_identity_gate_compares_the_device_class_not_the_suffix`。

## 全 profile 审计（真实 dev 30 集，`activation_report_v3.json`）

- 契约 + v7 配置（新共置/干扰 + 抢占 + 批处理 + `prefetch_overlap`）：四臂 **0 失败**、无 fail-closed。
- 机制激活（真实计数）：Parrot flip 271；QLM load_present 653 / reorder 692 / swap-cost 反事实 67；
  Hermes 条件 PDGraph 3517 / 观测精化 493 / **在线预热触发 1**（容量不足时 `prefetch_fail`，机制已进入路径）；
  Torpor resident 3583 / **canonical 三元组 covered 414**。

## 边界

- 3 轮中位；单卡（测量时无其他负载）；同模型对与 yolo 不覆盖（前者部署语义串行、后者非 LLM profile）。
- 探针为确定性定长请求（档位中位），应用到真实节点时长是**分段近似**（既有 F1 边界同样适用）。
- 负 extra 归零是噪声下限处理，非物理结论；原始值保留。
