# RESULT — EXP-20261002_substrate_token_prior_v1（第 1 步：小样本重放 / token 先验）

**状态**：完成（2026-10-02）；**2026-10-02 晚补提取全字段 + 数据审计**。

## 交付物

- `artifacts/token_replay_raw.json`（351 行，旧提取，仅 token/timing 字段）
- `artifacts/token_prior_v1.json`（分层先验）
- `artifacts/replay_full_extract_v1.jsonl`（**571 行全字段补提取**：api_call + action；351 带 token、72 带 first_token_ms、400 带显存峰值）
- `replay_runs.json`（48 run 清单）

## 方法

重放 12 个可用视频对应的 48 个 run（24 local_qwen + 24 local_split），在采集器中加
token/计时/显存插桩后重跑。远端输出 `/root/autodl-tmp/scheduler_measurements/token_replay/`。

## 结果

### 分层 token 先验（实测）

| model | node_type | n | in_p50 | out_p50 | in范围 |
|---|---|---|---|---|---|
| Qwen3-VL-8B | planner | 138 | 1055 | 48 | 899–1243 |
| Qwen3-4B | planner | 72 | 1072 | 36 | 912–1582 |
| Qwen3-VL-8B | videotool_spatial | 56 | 264 | 50 | 220–276 |
| Qwen2.5-VL-3B | videotool_spatial | 28 | 353 | 35 | 348–358 |
| Qwen3-VL-8B | answer_generation | 31 | 574 | 2 | 154–916 |
| Qwen2.5-VL-3B | answer_generation | 26 | 442 | 3 | 72–1314 |

### 全字段补提取的关键结论（2026-10-02 晚）

1. **F2 prefill 被真实数据验证**：4B 稳态 `first_token = −8.1ms + 115.3µs×tok`（R²=0.40）
   vs F2 `8.1 + 114.6µs` → 斜率差 0.6%；中位比值 0.86。冷启动 540–750ms（一次性预热）。
2. **refit α/β 计划证伪**：组内 input 方差太窄，从 inference_ms 不可辨识 prefill 斜率。
3. **decode 速率**：稳态中位 33.8 ms/tok vs F2 tpot 29.0（慢 ~17%）。
4. **显存增长**：真实 0.24/0.18/0.11 MB/tok（8B/4B/3B，R²≈0.99）< F2 扫描 0.44/0.44/0.33（未决）。

## 数据质量限定（必须随用）

- **text worker 协议 bug（本轮插桩引入，已修复）**：`_TimedStreamer` 的 `super()` 把生成文本
  print 到 stdout，污染 JSON-line 协议。24 条 local_split run 全部受影响：
  12 条 langgraph_react planner 全灭（0 成功）、12 条 star 决策错位（数据为真实生成、归属偏移）。
  修复：不再调 `super()`（保留计时）；协议验证 3/3 干净。原始 640-run 数据零此错误。
- VL worker（8B/3B，279 条）协议干净。
- 4B planner 72 条来自 star run 错位轨迹；12 条 langgraph_react local_split run 无 planner 数据。
- **F2 斜率对真实任务的"5.04× 偏差"归因更新**：该偏差不与 input 相关（每调用开销/噪声/共享卡），
  不是 prefill 斜率问题；total 由 trace 钉死，不影响 R_m。

## 未做

- 第 2 步（KV/抢占定向标定）、第 3 步（共置外推）、第 4 步（F4 扩展）、第 5 步（F9）未开始。
