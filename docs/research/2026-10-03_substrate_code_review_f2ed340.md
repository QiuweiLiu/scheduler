# 模拟器 ①②③ 代码审核（网页版独立审核）

## 审核来源和范围

网页版 GPT（绑定 generation 9 会话，GPT-5.6 Sol + High）直接读取公开仓库固定提交
`f2ed340930972bbd4ed812ba1937ef1163443202`（相对 `1e8bdf8`）；`1ee3a0b8...` 仅用于核对
token-prior 来源。无附件、无控制面、无凭据上传。审核为静态审查，未独立运行测试。

结论：**REVIEW — NEEDS CHANGE（P0 1 条，P1 5 条）**；旧单进程/未启用扩展路径未发现新 P0/P1 回归。
审核回复原文如下（未改动）。

---

炜，结论：**REVIEW — NEEDS CHANGE**。

我按固定 SHA `f2ed340930972bbd4ed812ba1937ef1163443202` 审了源码、测试和该提交里的实验产物；`1ee3a0b8...` 只用于核对 token-prior 来源。没有把后续代码混进判断。**P0 1 条，P1 5 条。** 旧单进程/未启用扩展路径没有看到新的 P0/P1 回归。

另外，你报告的“440 tests = 1 个既存失败 + 10 个无 torch 错误”我没有独立执行，因此这里只视为 **UNVERIFIED execution report**；下面结论来自固定源码和产物静态审查。

## P0 — vLLM batching 把 throughput speedup 当成 request latency speedup，B=2 被重复加速

位置：

- `src/tracing/analysis/workload_v02_simulator.py:1726–1780`
- `src/tracing/analysis/workload_v02_simulator.py:6873–6898`
- `experiments/EXP-20261003_substrate_batching_curves_v1/artifacts/b2_batch_curves.json`
- `experiments/EXP-20261003_substrate_batching_curves_v1/RESULT.md`
- 对应测试 `tests/test_workload_v02_simulator.py:~680–735`

`batching_admission_factor()` 返回：

```python
return 1.0 / speedup
```

随后这个 `<1` factor 同时作用于 active 和 candidate。fileciteturn122file0L2-L2 fileciteturn114file0L2-L2

问题是 B2 产物里的 `speedup_per_req` 定义为：

\[
\frac{T_{B=1}}{T_{B}/B},
\]

其中 `per_req_ms = wall_ms / batch_size`。这实际上是**吞吐归一化指标**，不是每个并发 request 的 wall-clock completion latency。产物本身例如 8B planner：

- B=1 wall = 1119.6 ms
- B=2 wall = 1157.4 ms
- `per_req_ms = 578.7`
- `speedup_per_req = 1.935`

fileciteturn105file0L2-L2

当前 DES 却会把两个约 1119.6 ms 的 request 都按 `1/1.935` 加速，约 **579 ms** 就完成。真实 B=2 probe 的整个 batch 明明花了 **1157.4 ms**。

也就是说，同一个 B=2 并行度被算了两次：

1. 两个 request 本来已经同时存在；
2. 又把 `wall/B` 当每个 request 的执行时间。

这会把 aggregate throughput 再虚增约一倍。

新增测试也把这个错误语义固化了：synthetic speedup=2 时期待 100 ms request 在 50 ms 完成。fileciteturn115file0L2-L2

**最小反例**

两个完全相同的 8B planner 同时开始。按真实 B2 probe，它们应共享约 1157 ms 的 batch wall interval；当前模型会让两者约 579 ms 完成。

**修复要求**

不能再把 `wall/B` 转成两个并发任务各自的 duration。至少应二选一：

- 对完全同质 B=2 probe，把 batch 建成一个 measured wall-time interval；
- 或建立每个 request 的 concurrent service-rate 模型，使最终 batch makespan 对得上 measured `wall_ms`。

对于两个 runtime 明显不同的 trace request，现有 homogeneous B2 probe **不足以**推出各自 completion time，应先 fail-closed，而不是像当前测试那样把 100/200 ms 两个任务都乘同一个 throughput factor。

这一条本身就足以阻止 `vllm_batched` 进入正式结果。

---

## P1-1 — REQUEST_RECOMPUTE 实际没有保证“只在 token 边界抢占”，且 phase clock 与动态速率失同步

位置：

- `workload_v02_simulator.py:6444–6474`
- `workload_v02_simulator.py:6530–6577`
- `workload_v02_simulator.py:6840–6930`

抢占时做的是：

```python
k_elapsed = int(
    (now - split.prefill_end_ms) // split.decode_step_ms
)
```

之后**立即在 `now` 删除 victim**。没有检查 `now` 是否刚好落在 token boundary，也没有推迟到下一 boundary。fileciteturn123file0L2-L2

**最小反例**

- prefill end = 100 ms
- decode step = 10 ms
- priority request 在 105 ms 到达

当前代码：

- `k_elapsed = 0`
- 105 ms 立即抢占。

但合法 token boundary 是 **110 ms**。这让 challenger 提前 5 ms 上 GPU，同时 victim 正在进行中的 token work 被截断。

现有测试恰好选择 `t=150`, `decode_step=2`，正好是 token 边界，因此没有覆盖这个问题。fileciteturn116file0L2-L2

更严重的是，`RequestSplit` 的：

```python
prefill_end_ms = work_start + p_hat
decode_step_ms = ...
```

是按原始 wall-clock 固定的，但 GPU task 的真实推进使用 `remaining_work / slowdown`。批处理 `<1`、F1 共置 `>1`、伙伴退出后的 rate reset、additive stall 都不会同步更新 `RequestSplit`。fileciteturn114file0L2-L2

例如 prefill work=40 ms：

- batching factor=0.5：GPU work 模型里约 20 ms 已做完 prefill，但 `RequestSplit` 仍认为要到 40 ms；
- coloc slowdown=2：实际约 80 ms 才做完，`RequestSplit` 却从 40 ms 就允许 decode preemption。

因此“三个机制交互”下 token progress 会错。

**修复要求**

`RequestSplit` 不能用固定绝对 wall-clock 来代表 phase progress。应该根据**已消费 intrinsic work**推进 prefill/decode/token state，或者在修好前明确禁止：

`REQUEST_RECOMPUTE × batching/F1/additive-interference`

联合启用。

同时抢占事件必须 snap/schedule 到下一 token boundary。

---

## P1-2 — 声明为 execution-only 的 RequestSplit 实际参与了动作可用性和 victim selection

这是明确的信息边界违约。

位置：

`workload_v02_simulator.py:6444–6514`

代码在 victim 加入 `victim_rows` **之前**依次读取私有：

- `request_split`
- `prefill_end_ms`
- `tokens_done`
- `remaining_tokens`
- `n_ctx`
- `rm_ms`

并根据这些值：

```python
continue
```

过滤 victim。之后再：

```python
min(victim_rows, ...)
```

选择真正的受害者。fileciteturn123file0L2-L2

所以虽然新增测试证明这些字段**没有直接序列化进 scheduler_state**，它们仍然通过 simulator control flow 影响：

1. 此刻有没有 preemption action；
2. 哪个任务能够成为 victim。

测试只检查了第一类“字段泄漏”，没有检查第二类“行为泄漏”。fileciteturn116file0L2-L2

而文档明确声称：

> execution truth “从不进入……动作可用性/受害者选择”。

fileciteturn117file0L2-L2

**最小反例**

构造两个 scheduler-visible state 完全一样的 episode，只改 execution-private `prefill_end_ms`：

- A：`prefill_end=40`, `now=50` → victim 可以被选；
- B：`prefill_end=100`, `now=50` → victim 被过滤。

policy 没看到任何不同输入，最终可执行 action 却不同。

如果同 GPU 有两个候选 victim，一个 hidden state 仍在 prefill，一个已 decode，私有 phase 甚至会改变**选谁**。

**修复要求**

先用 scheduler-visible information 决定明确的 victim/action，然后 execution layer 只能：

- 执行该 action；
- 或报告物理上暂不可执行/推迟至下一个合法边界。

不能读取 hidden truth 后偷偷换选另一个 victim。

另一种方案是把 `preemptible/phase` 明确纳入 scheduler-visible contract，但这会改变你们当前研究的信息边界，不能静默做。

---

## P1-3 — batching 的 fail-closed 有两个洞：mixed-role pair 被当成已测同质 batch；缺失 KV metadata 反而放行

位置：

`workload_v02_simulator.py:1726–1780`。fileciteturn122file0L2-L2

### 3a. Mixed-role 同模型没有验证 active side 的 role

准入只检查：

```python
active_task.model_id == node.model_id
```

然后 speedup 只查**candidate**：

```python
profile[f"{node.model_id}|{node.role}"]
```

`GPUActiveTask` 没保存 batching role/key。

因此实际可能发生：

- active = `Qwen2.5-VL-3B|answer_generation`
- candidate = `Qwen2.5-VL-3B|videotool_spatial`

candidate 有 spatial curve，于是两者一起得到 **1/1.881** rate factor。

但 B2 只测了 homogeneous：

- 3B answer × 3B answer
- 3B spatial × 3B spatial

没有测 answer × spatial。fileciteturn105file0L2-L2

8B 同样存在 planner/spatial/answer 角色混配风险。

这直接违反“未覆盖组合 fail-closed”。

### 3b. KV cap 缺失时是 fail-open

`batching_kv_cap()` 缺 pool 或 layer 时返回 `None`。

调用方却是：

```python
if cap is not None and cap < 2:
    return None
return 1.0 / speedup
```

也就是：

> cap 不知道 → 仍允许 batching。

和 docstring 写的 “KV pool too small **or unavailable → not admissible**” 不一致。

**修复要求**

至少：

```text
active.batch_key == candidate.batch_key
```

即 exact `model|role` match，除非另有 mixed-role 实测 cell。

同时：

```python
if cap is None or cap < 2:
    return None
```

validator 也应该保证每个 curve key 都存在对应 layer 和 model KV pool。

---

## P1-4 — multi-process additive F4 把强烈依赖 infer/shape 的数据压成“仅 load-model 常数 + shape=any”，违反本轮要求的 fail-closed

位置：

- `prefetch_interference_additive_v1.json:1–43`
- `mp_calibration.json` Part B
- `workload_v02_simulator.py:1520–1543, 5860–5960`
- docs additive F4 section

原始 mp 数据对于 **load 3B**：

- 4B medium ← 3B：+17.7 ms
- 4B long ← 3B：+633.5 ms
- 8B medium ← 3B：+47.2 ms

fileciteturn102file0L2-L2

也就是说同一个 loaded model 的 extra 跨格约 **35×**。

但派生产物变成：

```json
{
  "load": "Qwen2.5-VL-3B",
  "extra_ms": 47.2,
  "range_ms": [17.7, 633.5]
}
```

并且：

```json
"supported_workload_shape": "any"
```

fileciteturn104file0L2-L2

运行时函数又只接受 `load_model`：

```python
prefetch_interference_extra_ms(profile, load_model)
```

所以那个**真实测得 +633.5 ms 的 4B-long←3B cell，在正式模型中反而会被记成 +47.2 ms**。未测过的 shape 也因为 `any` 被视为 covered，而不会 fail-closed。fileciteturn99file0L2-L2

实验 RESULT 自己也明确承认 “single-cell span large”，只是选择了按 loaded-model median 作为 DES modeling choice。fileciteturn119file0L2-L2

这个做法可以作为**粗粒度 sensitivity model**，但不能同时声称：

> measured + fail-closed + unsupported shape 不猜。

**修复要求**

要么按至少：

\[
(\text{infer model},\text{infer shape},\text{load model})
\]

保存支持域，没测到则串行；

要么明确把当前 load-only median 降级成 `approximate_extrapolation`，不能标 `supported_workload_shape=any`，也不能作为 deployment-of-record fidelity surface。

---

## P1-5 — REQUEST_RECOMPUTE 对 3B 宣称 measured support，但 P1 resume source 并没有测 3B

`request_preemption_profile_v1.json` 为三个模型都发布了 recompute rates，包括 3B，代码因此允许 3B request-level preemption。fileciteturn96file0L2-L2

文档又写：

> `R_m` 已由 P1 往返实验验证（残差 −1.8%~−5.6%）。

fileciteturn117file0L2-L2

但 `p1_resume.json` 实际只有三格：

- 4B, k=20
- 4B, k=40
- 8B, k=20

**没有任何 3B resume/rebuild cell。** fileciteturn106file0L2-L2

后续 `1ee3a0b` 的 token-prior 确实包含 3B token 数据，但那不是 3B resume-cost validation；它甚至明确区分了 token replay 与后续定向抢占标定。fileciteturn108file0L2-L2

所以当前：

```text
3B victim
→ request_recompute_ms()
→ F2 slope/intercept
→ 当成 measured REQUEST_RECOMPUTE
```

来源链是不闭合的。

**修复要求**

二选一即可：

- 补最少一个 3B representative resume cell，并通过预先冻结误差 gate；
- 在此之前 3B `request_recompute_ms()` 返回 unsupported，显式 `victim_model_not_covered`。

---

## 已核实没有阻断的问题

这几项实现本身我认为是对的：

- **multi_process 同模型默认串行**：profile 明确禁止 same-model cell，`colocation_slowdowns()` 也返回 `None`；测试覆盖。fileciteturn92file0L2-L2 fileciteturn115file0L2-L2
- **mp 异模型表来源一致**：A1–A5 与 raw calibration 对得上；A5 的 `<1` slowdown 被 clamp 到 1.0，而且 artifact 明示，不是静默处理。fileciteturn101file0L2-L2
- **additive interference 的事件数学**正确：`extra_ms` 一次加入 remaining work，load finish 不复位，不再 dilation；测试也覆盖了这一点。问题在于“用哪个 extra_ms / 支持哪些 shape”，而不是 one-shot 实现本身。fileciteturn99file0L2-L2
- **普通 batching partner 完成后 survivor 恢复 1×**：`process_finish()` 会先结算剩余 work，再 `set_node_slowdown(...,1.0)`；没有看到重复计时。只是当前输入的 batch factor 本身是错误的。fileciteturn100file0L2-L2
- **GPU identity 检查**在 colocation/request-preemption/batching/prefetch profile 都存在。
- **cold-load / eviction transition 不与 batching/co-location 静默重叠**：准入和 dispatch 都有保护。
- **未启用 request profile 时旧 node-recompute 路径仍在**；未启用 batching 时旧共置路径仍在。静态审核未发现 P0/P1 legacy 回归。

B2 profile 与 `b2_batch_curves` 的数字本身一致，B3 “8B 1.29 vs mp 1.24 / 3B 0.98 vs mp 1.27”的 metadata 也与实测文件一致。fileciteturn95file0L2-L2 fileciteturn111file0L2-L2

## 非阻断但建议顺手补的测试/契约

当前新增测试缺少四类关键 interaction case：

1. `request_preemption + batching` / `request_preemption + F1 slowdown`；
2. 非 token-aligned arrival，例如 105 ms 落在 100–110 ms token 内；
3. same-model **mixed-role** batching；
4. batching profile 有 curve、但缺 layer/KV pool 的 fail-closed case。

另外 `cross_model_policy="mp_table_proxy"` 目前基本是 metadata；代码没有强制 batching profile 必须和 `deployment=multi_process` colocation profile 成对出现。 canonical 文档命令确实同时传了二者，所以我暂不把它单列 P1，但建议 validator 做 cross-profile consistency gate，防止未来手工配置把 single-process F1 表接到 vLLM mode 上。fileciteturn121file0L2-L2

后续 `1ee3a0b` 还有一个很小的 provenance 文案不一致：`token_prior_v1.json` 写 `"17 video replay, 48 runs"`，RESULT 写“12 个可用视频对应 48 run”。数值层本身没有因此失配，但建议统一来源描述。fileciteturn109file0L2-L2 fileciteturn108file0L2-L2

### Gate 结论

因此当前不是“补几个测试就 ACCEPT”的状态。至少要先关闭：

**P0 batching throughput/latency 双计数**，以及 **P1 request phase/token-boundary、hidden-truth victim selection、mixed-role batching fail-open、additive-F4 支持域、3B resume 来源缺口**。

这些修完之后，我认为旧 substrate 不需要推倒重来；问题基本都集中在这次新增的三个 opt-in extension 的接口和证据边界。