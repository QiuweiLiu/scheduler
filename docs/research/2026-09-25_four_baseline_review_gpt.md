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

---

## 跨基线的公平性（重要，必须改进论文）

每条基线使用**自己的** paper-derived / adapted predictor–scheduler interface，
因此主表评估的是**完整系统**，而**不是**「同等信息前提下孤立的调度器」。论文必须写明：

> Each baseline uses its own paper-derived/adapted predictor–scheduler interface; therefore the
> comparison evaluates complete systems, not an isolated scheduler under an equal-information oracle.

→ 主表能回答**「完整方案谁更好」**，但**不能单独证明「你的 scheduler 比别人强的 scheduler 好」**。
后一个结论必须依赖已规划的 **same-interface 2×2 / ablation**。

另：ledger 自己记录着 **F0 deployed future-chain identity 仍来源于 J3，而不是 F0 自己的 structure head**。
这个 caveat **正式论文不能藏**，否则「F0 完整 joint predictor」的说法过强。

## Bootstrap：结论正确，但文案要改

每个 episode 最终只产生一个 `mean_completion_ms`，再 bootstrap episode delta，
因此 **episode 内多 job 的相关性已经被保留，不应该按 job 重采样** —— 这里**没问题**。

但代码/文档写的是 **"video-cluster bootstrap"**，而 `paired_bootstrap_ci()` 实际是
**普通 paired episode bootstrap**。若 300 episodes 之间会复用同一 video/template 并希望推断到新视频，
则要按 video/template cluster 重采样；否则**把文案改成 episode-cluster bootstrap**。

## 正式 runner 还有两个必须 fail-closed 的点

`four_joint_baselines.py:107` 的 `run_arm()` 至少应：

```python
assert failed_jobs == 0
assert completed_jobs == jobs
assert metric is finite
```

否则「部分 job 失败但幸存者很快」会得到一个**漂亮且错误的**平均完成时间。

另外 runner 目前只是**记录** projection SHA，没有 **assert 它等于冻结 SHA**；episodes file 甚至没有 hash。
正式版应在启动时硬断言：

```python
projection_sha == 15c62d...
BaselineFidelityManifest.pass is True
TopologyGate.pass is True
episode_file_sha == frozen_episode_sha
F0 overlay/base hashes == frozen hashes
git HEAD / experiment implementation hash 被记录
```

**smoke artifact 里的 `git_head` 目前还是空字符串**，formal 前必须修。

## Smoke 数字与运行时间

3 集 smoke 的五个数字**没有哪一个「数学上不可能」**。Pythia ≈ F0、TIE +4.6 s、LLMSched +9.2 s、
Latency +9.8 s 都可能出现；三集样本也不足以从性能倒推 implementation。

值得注意的是：**Latency 即使在当前拥有 future-template leak 的情况下仍明显更慢**，
这反而与「prefetch 可能占用 ready GPU / single-commit approximation」相容，**不是反证**。

**61.6 分钟是合理的一阶估算**，但不应当作可靠上限；**没看到随 episode 数必然爆成 O(n²) 的路径**，
TIE / Pythia / Latency 都比较轻。

**LLMSched 有一个更现实的风险**：`posterior_joint()` 的 `_posterior_cache` 上限是 200,000 entries，
而 top-4 joint 单个最坏数组有 7^5 = 16807 项（原文在此处截断，需重新取全）。

## 待补

回复在 LLMSched cache 风险处截断。**TIE 的窄项、P1 清单、以及 limitation 清单尚未读到**，
需要重新取到回复末尾。
