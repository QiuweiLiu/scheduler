# RESULT — EXP-20261003_substrate_batching_curves_v1（B2：批处理曲线）

**状态**：完成（2026-10-03）。**远端脚本 `b2_batch_curves.py`（经 run_vllm.sh）；产物 `artifacts/b2_batch_curves.json`。**

## 目的

DES 新增"批处理引擎模式"所需的核心数据：**同模型多请求的批处理加速比**，
按 workload 真实形状测量（8B/3B/4B × 文本/图像）。

## 方法

vLLM（隔离 venv 0.30.0）microcurve：B∈{1,2,4,8,16}，每档 1 warmup + 3 reps；
固定输出长度（min=max, ignore_eos）；图像条件按 vLLM 要求带
`<|vision_start|><|image_pad|><|vision_end|>` 占位符 + `multi_modal_data`。

## 结果（每请求加速比 = B=1 每请求耗时 ÷ 当前每请求耗时）

| 模型\|条件 | B=1 每请求 | B=2 | B=4 | B=8 | B=16 |
|---|---|---|---|---|---|
| **8B planner**（文本 ~1055/48） | 1120ms | **1.94×** | 3.80× | 7.23× | **13.46×** |
| **8B spatial**（1图+~42/50） | 1149ms | **1.95×** | 3.85× | 7.59× | **14.64×** |
| **3B answer**（文本 ~442/3） | 45ms | 1.63× | 3.08× | 4.77× | 7.19× |
| **3B spatial**（1图+~52/35） | 382ms | **1.88×** | 3.75× | 7.41× | 13.70× |
| **4B planner**（文本 ~1072/36） | 487ms | **1.86×** | 3.64× | 6.67× | 11.75× |

**引擎 KV 池实测**（vLLM 启动日志/属性）：8B = 3155块×16 = **50,480 tokens**；
3B = 29760×16 = **476,160 tokens**；4B = 8164×16 = **130,624 tokens**（gmu=0.85）。

## 结论

1. **B=2（两任务共置的现实场景）≈ 1.9×**（除 3B answer 因输出极短为 1.63×）；
   B=16 达 **13–15×**（图像条件最高 14.64×）。
2. **输出越短、批处理收益越低**（3B answer 输出仅 3 token → 7.19×；长输出接近线性 14×）。
3. 文本与图像条件收益接近（图像不改变批处理规律）。
4. KV 池实测值可直接用于 DES 的"批处理并发上限"记账：
   `最大并发 = KV池 ÷ (KV/token × 上下文)`（例：8B 在 1k 上下文 ≈ 50 并发）。

## 边界

- vLLM 引擎口径（HF 主 substrate 无批处理能力；本曲线用于"升级为 vLLM 式引擎"的建模假设）。
- 单卡共享环境；固定输出长度；3 reps。
- 3B answer 的 B=1 仅 45ms（含 vLLM 调度开销），比例受固定开销影响。

## 仿真器集成（2026-10-03；2026-10-03 晚按评审 P0 修正）

- **语义修正（评审 P0）**：曲线在仿真器中使用的是**每请求延迟因子**
  `f(B) = wall_ms(B)/wall_ms(1)`（≥1），不是吞吐加速比 `T1/(wall/B)`。
  用吞吐加速比当每请求延迟会把"两个请求已经并行"重复计算，B=2 的聚合吞吐虚增约一倍。
  派生产物 `artifacts/batching_engine_profile_v2.json`（schema `batching-engine-profile-v2`）：
  5 条 f(B) 曲线（8B planner B2 1.0338 / B16 1.1885；3B answer B2 1.2274 / B16 2.2097；…）+
  3 个 KV 池 + 6 层 token 中位 + `homogeneity_tolerance=1.25`；吞吐值保留在 `probe_metadata` 作上下文。
  跨模型 `cross_model_policy=mp_table_proxy`（B3 不对称登记）。
- **准入（engine admission，fail-closed）**：`model|role` 必须相同（mixed-role 未测→串行）；
  KV 池已知且 cap≥2（缺元数据即拒绝）；两请求实际内在运行时长的比值 ≤ 1.25
  （B2 只测同质批次，明显异质对未测→串行）。
- 消费方：`workload_v02_simulator.py` 的 `vllm_batched` 模式（准入/速率/KV 上限/异质 fail-closed）。
- 集成冒烟（真实 v2 产物）：4B planner 100/100ms → 完成 [107.67, 107.67] = 100×1.0767 ✓；
  100/300ms（比值 3）→ 串行 [0, 100] ✓。
