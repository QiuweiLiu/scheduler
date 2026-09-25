# 四条联合基线全面审查（GPT，HEAD d7e90d4）

**结论：不建议现在启动 300×5 正式测量。**
这次全链路审查抓到的不是性能问题，而是几处**之前局部 freeze 没覆盖到的跨模块 correctness / fidelity 问题**。
其中 LLMSched、Pythia、Latency-Aware 都有 **freeze-breaking** 项；TIE 核心公式没问题，
但 load adaptation 还有一个窄的条件统计问题。

这次审查的定位：这是正式开跑前的**最后一次 freeze-breaking 审计**，
只保留能被代码证据支持的 P0/P1/limitation，不重开已证明无问题的项。

---

## P0 — 必须修（freeze-breaking correctness / fidelity）

| 优先级 | 位置 | 问题 | 判定 |
|---|---|---|---|
| P0 | `llmsched_bn.py:797,854` + simulator ~3541 | EXPLOIT 的 remaining **没条件化 current X != ABSENT**，且只算 BN descendants，不是 whole-job unresolved work | **BREAK LLMSched freeze** |
| P0 | `pythia_profiler.py:108~160` + simulator ~3490 | 算的是「duration-weighted remaining ms」，但 Pythia Algorithm 3 用的是 **regex 上的 expected remaining distance（步数）**；且当前节点还有 `runtime_p50 + load` 的重算/口径问题 | **BREAK Pythia freeze** |
| P0 | `latency_aware_fusion.py:86`、`lifecycle.py:31`、simulator ~3674/~4482 | Constructor / prefetch **直接读 realized `job.template` 的未执行后继**，在本项目「未来 Agent workflow 未展开」的设定下是**真值结构泄漏** | **BREAK Latency freeze** |
| P0 | `latency_aware_predictor.py:69` | run/peak/load 三头**全部共用 request tiers**；但论文的 load 是 deployment/device 条件 `T_load(d,g)`，不是 request-conditioned | **BREAK Latency freeze** |
| P0 | simulator ~4482,4558 | Latency prefetch 在 ready dispatch **之前**运行，只检查 GPU active/prefetch，**没有保证不延迟 ready work** | **BREAK Latency freeze** |
| P0-实验 | `four_joint_baselines.py:107` | runner **不检查 `failed_jobs==0` / `completed_jobs==jobs`**；只要还有已完成 job，`mean_completion_ms` 就可能返回一个**看似正常的 survivor mean** | 正式测量前必须加 |

### 前三条最要紧

**① LLMSched 的信息状态不一致**
`joint_mutual_information()` 已正确对 `X != ABSENT` 条件化，`current_service_ms()` 也通过 `absorb()` 做了同样条件化；
**但 `expected_remaining_ms()` 对每个 future stage 调用的是 `posterior_state_probs(..., evidence)`，没有把「当前 ready X 已确定存在」传进去。**
`job_duration_interval_ms()` 同样只有 completed evidence。
→ EXPLORE、current service、future remaining **实际处在三个不完全一致的信息状态**。

而且 `expected_remaining_ms()` 仍然调用 `_future_from_network()`，**只加 X 的 BN descendants**。
这与已修正的 `job_duration_interval_ms()` 不一致：若存在与 X 独立、无 X→Z path 的未来 stage Z，
Z 确实仍属于 job 的 remaining work，但 **EXPLOIT 会漏掉它**。

**建议一次关掉**：新增 `expected_job_remaining_ms(profiler, evidence, current_stage)`，
遍历**全部 model-side unresolved stages**，所有 posterior 都条件在 `current_stage != ABSENT`；
interval 使用同一个 known-present 条件。**Eq.6 仍保持 correlated descendants + top-4，不动。**
理由：LLMSched 论文核心是 BN uncertainty + JCT-efficient scheduling，两者不应消费不同的 posterior state。

**② Pythia 是量纲/语义错，不只是 double-count**
`build_pythia_profiler()` 第 108 行记的是 **role 的 intrinsic duration**，再由 horizon DP 得到
`expected_remaining_ms_by_role`；live consumer 又把 shared `estimate_row["runtime_p50_ms"] + load` 加到它前面。
→ 这实际上已变成 **duration-aware / SRTF 风格**，不是 Pythia。

Pythia Algorithm 3 明写：
```
S_completion = 1 / E[D_remaining]
```
其中 `D_remaining` 是 **probabilistic regex 上「到 terminal 的 expected distance」**；
论文解释也是「离完成还有几步」，DownstreamIdleRisk 同样以「一步 vs 十步」的 expected distance 表述。
随后 worker 再单独加 wait-time aging。

→ **最 faithful 的修法不是继续修 ms decomposition，而是**：
```
V(role) = E[number of future role transitions within the bounded PFA]   # 步数，不是毫秒
S_completion = 1 / (1 + V(current_role))
```

### Latency-Aware 的四项
- **fusion / prefetch 读 realized suffix**：与我在 LLMSched 里修掉的泄漏同类。原论文可以假设逻辑
  workflow graph 已知，但**本项目的核心设定是未来 workflow 尚未展开**。需确认 runner/live path 是否真的消费到它，再定级。
- **load 头不该是 request-conditioned**：论文是 `T_load(d,g)`（deployment/device 条件）。
- **prefetch 可能延迟 ready work**：需加「不延迟 ready work」的保证。
- 另有 `latency_aware_predictor.py:69` 的三头共用 tier 问题。

---

## 待补
GPT 的回复我读到了 P0 表与前三条的展开（约在字符 27500 处截断），
**TIE 的窄项、P1 清单、limitation 清单、以及第 11/12 问（smoke 异常与 O(n²) 风险）尚未读到。**
需要重新取到回复的后半段。
