# 模拟器审核修复链（第二轮 + 第三轮）— 网页版独立审核记录

## 审核来源和范围

网页版 GPT（绑定 generation 9 会话，GPT-5.6 Sol + High）直接读取公开仓库固定提交，无附件。
审核为静态审查，未独立运行测试；本地测试结果按 UNVERIFIED execution report 处理。

修复链：
- `f2ed340` → NEEDS CHANGE（P0×1 + P1×5，报告见 `2026-10-03_substrate_code_review_f2ed340.md`）
- `e74fd02` → NEEDS CHANGE（P0=0；原 P1-2/3/4/5 关闭；2 个新 P1）
- `2c727ca` → **ACCEPT**（2 个新 P1 关闭；仅剩 1 个非阻断 P2 docstring，已顺手修正）

---

## 第二轮（e74fd02）审核回复原文

炜，结论：**NEEDS CHANGE**。

固定复验提交：`e74fd02325057f6404f67e420f2cc736b2c5940b`。本轮静态复验结果是 **P0 = 0，P1 = 2**。上一轮 batching 吞吐/延迟 P0 的核心数学错误已经关闭；原 P1-2/3/4/5 在阻断级别也已关闭。原 P1-1 的主体修复正确，但与 additive F4 的交互仍有一个阻断问题；此外 batching 修复新引入了一处 execution-truth → action availability 泄漏。

本地 40/40、63/63、446 测试结果我没有独立执行，仍按 UNVERIFIED execution report 处理。

### P1 — additive F4 会让 RequestSplit 的“已消费内在工作量”倒退，token/phase 状态被回拨

文件：`src/tracing/analysis/workload_v02_simulator.py`

- `request_split_phase()`：约 **1677–1735**
- `add_node_work()`：约 **6296–6311**
- `GPUActiveTask.total_work_ms` 初始化：约 **640–665**

新的 phase clock 定义本身是合理的：

```python
consumed = total_work_ms - remaining_work_ms
```

并由此决定 prefill/decode、`tokens_done` 和下一 token boundary。fileciteturn160file0

但 additive F4 仍然这样收费：

```python
task = gpu.update_task_progress(task_id, at_ms)
task.remaining_work_ms += extra_ms
```

它**只增加 `remaining_work_ms`，不改变 `total_work_ms`**。fileciteturn159file0

而 `total_work_ms` 在任务创建时只初始化一次。fileciteturn161file0

所以一旦运行中的 REQUEST_RECOMPUTE task 遭遇 additive load interference：

\[
consumed'=total-(remaining+extra)
=consumed-extra
\]

phase progress 会直接倒退。

最小反例：

```text
total_work = 200 ms
prefill_work = 100 ms
decode_step = 2 ms

t=150:
remaining = 50
consumed = 150
=> decode 已完成 25 token

此时 additive extra = 100 ms:
remaining = 150
total 仍 = 200
consumed = 50
=> RequestSplit 突然重新判成 prefill
=> tokens_done 从 25 退回 0
```

之后抢占会错误地报告 `victim_prefill_in_progress`，重新安排边界，并且后续 `n_ctx / R_m / remaining_tokens` 都会错误。

这不是理论上的不可达组合：文档给出的“当前完整口径”明确同时启用 `prefetch_interference_additive_v2` 和 `request_preemption_profile_v1`；文档还明确声称 additive stall 会和 batching/F1 一样“一致地移动 phase clock”。当前实现并不满足这句话。fileciteturn153file0

**修复要求：**不要再用包含 interference overhead 的通用 `remaining_work_ms` 直接推导 request intrinsic progress。最稳妥的是给 request phase 单独维护 intrinsic-progress ledger / stall budget。或者在这套记账没完成前，明确 fail-close 禁止 `REQUEST_RECOMPUTE × additive F4` 联合启用。

仅仅做 `total_work_ms += extra_ms` 可以避免“瞬间倒退”，但仍会把后续消耗的 interference work 当成 request token progress，因此不足以完整解决语义问题。

---

### P1 — batching 的 homogeneity gate 使用隐藏 trace 真值，并通过 `dispatchable_gpu_indices` 泄漏到调度器

文件：`src/tracing/analysis/workload_v02_simulator.py`

关键位置：

- `batching_admission_factor()`：约 **1887–1924**
- 真值读取：**1917–1918**
- `gpu_can_dispatch_node()`：约 **2013–2036**
- scheduler state 的 `dispatchable_gpu_indices`：约 **5635–5648**

当前 homogeneity gate 是：

```python
candidate_work = node.compute_ms
active_work = active_task.total_work_ms
...
if high / low > tolerance:
    return None
```

fileciteturn156file0

其中 `node.compute_ms` 来自真实 `node.runtime_ms - load_ms`，也就是 execution truth，不是 scheduler prediction。fileciteturn154file0

这本来如果只留在 execution engine 内部还可以讨论，但 `batching_admission_factor()` 被 `gpu_can_dispatch_node()` 调用：fileciteturn158file0

而 `gpu_can_dispatch_node()` 的结果直接写进调度器可见状态：

```python
"dispatchable_gpu_indices": tuple(
    gpu.index for gpu in gpus
    if gpu_can_dispatch_node(...)
)
```

fileciteturn157file0

因此 hidden true runtime 已经改变了**动作可用性**。

最小反例：

```text
scheduler-visible prediction 完全相同：
active predicted runtime = 100 ms
candidate predicted runtime = 100 ms
tolerance = 1.25

World A hidden truth:
active=100, candidate=120
ratio=1.20 -> GPU 0 dispatchable

World B hidden truth:
active=100, candidate=130
ratio=1.30 -> GPU 0 不 dispatchable
```

两个 scheduler-visible world 本应完全相同，但 policy 收到的 `dispatchable_gpu_indices` 不同，因此能够间接推断未来真实 duration。

这一点尤其值得修，因为 `batching_engine_profile_v2.json` 自己写的其实是：

> admission requires ... **scheduler-visible runtime estimates** within homogeneity_tolerance

fileciteturn145file0

代码实现和产物契约反而没有对齐。

**修复要求：**homogeneity gate 改用 scheduler-visible estimate。候选任务直接使用 `estimate_row["runtime_p50_ms"]`；运行任务在首次 dispatch 时冻结一份对应的 predicted/estimated intrinsic runtime 到 `GPUActiveTask`，后续 batching admission 比较这两个预测值。不要使用 `node.compute_ms` 或真实 `total_work_ms` 决定 scheduler-visible dispatchability。

---

因此这轮距离 ACCEPT 已经很近：**原来的 1×P0 数学问题确实修掉了，P1-2/3/4/5 也没有看到阻断级残留；现在只剩上述两个 P1。** 特别是第二个是修复 homogeneity gate 时新引入的信息边界问题，第一个则是 P1-1 在 additive F4 联合路径上还没有完全闭环。
---

## 第三轮（2c727ca）审核回复原文

炜，结论：**ACCEPT**。

固定提交 `2c727ca8eb368ed3d39a0557d32b2e0c1e93a447` 相对 `e74fd02` 的 focused 复验中，上一轮留下的 **2 个 P1 均已关闭**；我没有发现这两处修复引入新的 P0/P1。

R2-P1-A 已关闭。现在 `GPUActiveTask` 将总剩余工作、request 内在剩余工作和 additive stall 分开记账：`update_task_progress()` 先消费 `stall_remaining_ms`，只有剩余 delta 才减少 `intrinsic_remaining_work_ms`；`add_node_work()` 同时增加 wall/service ledger 的 remaining work 和 stall budget；`request_split_phase()` 则只从 `total_work_ms - intrinsic_remaining_work_ms` 推导 token/phase。因此 additive charge 不再导致 phase 倒退，也不会被错误计成 token progress。fileciteturn165file0 fileciteturn175file0 fileciteturn176file0

对应新测试也正好覆盖了上一轮的最小反例：t=150 已经完成 25 token，加入 +100 stall 后 t=160 仍保持 decode/25 token；stall 消耗完以后 t=260 才继续推进到 30 token。fileciteturn180file0

R2-P1-B 也已关闭。`predicted_compute_ms()` 明确只从 scheduler-visible estimate row 的 `runtime_p50_ms - load_p50_ms` 计算；batching gate 比较 candidate 的该预测值与 active task 派发时冻结的 `predicted_work_ms`，不再读取 `node.compute_ms` 或 `total_work_ms`。fileciteturn173file0 fileciteturn167file0 fileciteturn168file0

这条边界在两个关键入口都保持一致：`gpu_can_dispatch_node()` 用传入的预测 estimate 判断 batching 是否可派发，实际 dispatch 时 active task 保存的也仍是同口径预测值。fileciteturn170file0 fileciteturn183file0 fileciteturn184file0 因而“隐藏真值不同、scheduler-visible prediction 相同”的两个 world 不会再通过 `dispatchable_gpu_indices` 暴露差异。新增测试用真实 runtime 100/200、共同预测 100 的 pair 验证仍会同样批处理；预测值真正相差 2× 时则串行。fileciteturn178file0

文档也已经改成与实现一致的两条语义：additive interference 是独立 stall budget；batch homogeneity 使用 scheduler-visible predicted runtime。fileciteturn185file0

唯一看到的非阻断小问题是 `request_split_phase()` 的函数 docstring 仍写着旧式的

`consumed = total_work - remaining_work`

而实际代码已经正确改成 `total_work - intrinsic_remaining_work_ms`。fileciteturn176file0 这是纯文档 P2，不影响 ACCEPT，顺手改一下即可。

因此这条修复链现在可以记为：

**f2ed340: NEEDS CHANGE → e74fd02: NEEDS CHANGE → 2c727ca: ACCEPT**

你报告的 `42/42`、`63/63`、全量 `448` 测试结果我仍未独立执行，所以运行结果本身保持 **UNVERIFIED**；但就本轮要求的固定源码静态审核而言，**没有剩余阻断级 P0/P1**。