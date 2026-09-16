# FAS 消费策略代码审查（GPT，2026-09-15）

来源：ChatGPT Web 会话（输入：9 条自查疑点 + 消费策略全部实现代码；输出：逐项裁决、遗漏的实现层问题、重写清单、结论稳定性判断）。
**关键结论：q95 公式本身没有致命 bug，但"q95 是最佳消费机制"目前只能是 provisional**——
E2 对照被表查询/加载语义污染，CVaR 家族有更严重的实现问题（截断、人造 CDF、缓存污染、N=32），
因此"q95 战胜所有分布方法"的强表述应**暂停使用**，先做 P0 正交重跑。

---

## 5. assistant
结论先说：q95 本身的实现相对简单，暂未看到明显“写错公式”的致命 bug；但目前“q95 是最佳消费机制”只能算 provisional。 E2 对照被表查询/加载语义污染，而 CVaR 家族存在更严重的实现问题，当前不能据此宣称“q95 本质上优于分布消费”。

1. 你列的 9 个疑点逐项裁决

E2 vs q95 两变量同时变化：确认。 不是代码 bug，是实验混淆。E2→pred-p50(lam0) 实际还恶化约 +16.0s；pred-p50→q95 则改善约 −33.7s，净值才是 −17.7s。因此“预测资源优于表”明显不成立，真正有效的是 predicted-tail 消费。

λ 只扫 p50→p90：确认。 不是 bug，但不能称“完整风险扫描”。lam100=risk=p90，没有覆盖 q95。应另做统一参数：

Q
λ
	​

=p50+λ(p95−p50),λ={0,.25,.5,.75,1}

load 语义不一致：确认，而且较严重。 表统计的 load_p50 只从 load>0 样本计算，本质是 conditional-positive duration；E2 却每个匹配步骤都加，预测臂则 occ>=.5 才加。不是同一随机变量。由于 rt95≈q95，建议主消费实验先完全去掉 future load，load 单独消融。

prefix 表查询：确认，属于实现缺陷。 _step_estimate_cost 没有复用 estimate() 的 hierarchical fallback，而是把 exact/model_lane/lane 三层“统计量本身”混在一起再取 median，相当于对聚合结果二次聚合且重复计权。应改为：
model_lane(count>=3) → lane(count>=3) → fail，不要 prefix 搜全部 row。

scenario 长度只能缩短：确定 bug。 如果 steps 已按 argmax L 截断，则采样 L>L_argmax 永远实现不了。当前 CVaR 不是完整的 length-distribution sampling。必须让 artifact 提供完整 H 个 slot，再由 sampled L 截断。

inverse CDF 下半段 0→p50：确认有问题。 它人为规定 Q(0)=0，没有预测依据；而 u>.95 又人为延伸至 p95+9(p95-p90)。这两个假设都足以严重改变 mean/CVaR。不是 Python bug，而是未识别分布被人为造出来，因此 CVaR 结果目前不公平。

comonotone 只耦合 runtime：确认。 若名称叫 joint/comonotone，应连 load occurrence/duration（以及最好 length）一起规定相关结构；否则只能叫 runtime-comonotone。

current=table、future=prediction 尺度混用：确认，但所有 predicted-resource 臂共同存在。 它不会污染 p50/p90/p95 的相对比较，却会污染“风险机制”的解释：p95 可能只是碰巧把 0.34× 的 future scale 拉回合理权重。

occ≥.5：合理 heuristic，不是 bug。 但阈值任意且不连续。鉴于 load 已证明无贡献，最干净的办法不是继续调阈值，而是 primary consumer 只比较 runtime。

2. 你还漏了几个真正实现层问题

P0：_SCEN_COST_CACHE 键有严重错误。

Python
运行
key=(node_id,horizon,samples,coupling)

没有 future_artifacts、train_stats 或实验版本。只要同一进程中对同 node_id 使用过原 artifact，再跑扰动 artifact/CVaR，可能直接返回旧结果。敏感性/不同 provider 尤其危险。

至少改成：
(artifact_version, stats_version,node_id,H,N,coupling,sampler_version)，或每臂显式清 cache。

_JRES_COST_CACHE 也不够稳：用 id(cost_fn) 标识 lambda，有 Python object-id 重用风险，而且未包含 train_stats。应使用显式 (policy_name, parameters)，不要用函数 id()。

Scenario sampler 还有三点：

永远只取 scenarios[0]，没有采 scenario_probability，也没有消费 model distribution；

32 samples、CVaR90 实际尾部只有约 3 个样本，方差非常大；至少 128/256 或 stratified/quasi-MC；

mean + κ·CVaR 不是标准 convex risk blend，会同时改变整体 scale。更干净是

(1−κ)E[C]+κCVaR

或直接纯 CVaR。

quantile 输入缺乏防御： _mix_step_cost 只验证 p50，却直接 float(p90)；_split95 没验证 p50 类型；也没有检查 p50≤p90≤p95。建议 Stage0 assert quantile monotonicity + finite/nonnegative。

adaptive 有一个实质遗漏： adapt_lambda() 估计 slack 时只加 current runtime，没有加入当前 cold-load，而最终 score 的 current 又包含 load。因此 cold model 的 λ 判定与实际 score 不一致。

3. q95 对照应该怎么重建

不要把 E2 改名/替换成 pred-p50。 E2 是历史 static-table baseline，应保留。

新增一套真正正交的消费族：

Pred50-R: Σ predicted runtime p50

Pred90-R

Pred95-R

ScaledPred50-R: k·Σp50，k 在 700-dev 预冻结，使平均 future-cost 与 p95 匹配

ShuffledTail: 保留 (p95-p50) 分布，但跨 step/node 打乱

五者使用完全相同 current_cost、长度链、fallback，future load=0。

这样才能区分：

p95 提升来自 tail 信息，还是仅来自把 future cost 放大。

如果 Pred95 >> ScaledPred50 且 Pred95 >> ShuffledTail，C2 才真正站稳。

4. 必须重写的顺序

P0（重跑后才能正式引用）

修 _SCEN_COST_CACHE / _JRES_COST_CACHE；

修 table lookup 为真正 hierarchical lookup；

建 runtime-only 的 p50/p90/p95 正交族；

Stage0 校验 quantile、missing/fallback，并报告 fallback rate。

P1（CVaR 结论要保留则必须）
5. artifacts 保留完整 H slots，使 sampled length 可增可减；
6. 重写 quantile distribution reconstruction；
7. N≥128，规范 CVaR objective；
8. scenario/model/length distribution 真正采样。

P2
9. comonotone 全分量耦合；
10. adaptive slack 加 current load；load occurrence 语义统一。

5. 哪些结论可能变？

我认为最可能保持的是：

future information > myopic；

runtime tail 越保守，从 p50→p90→p95 越有效的现象；

load-tail 基本无价值；

length/continuation 有价值；

高负载下 q95 类策略收益更大。

最可能明显变化的是：

E2 与 q95 的绝对差距（因为 E2 table lookup 本身不干净）；

“CVaR/场景采样不如 q95”——当前证据我会暂时撤回，因为 sampler 有截断、CDF 人造、缓存污染和 N=32 四重问题；

“q95 是 tail-risk 机制”——可能最后变成“q95 恰好是最合适的 future-weight scaling”。

所以现在不用怀疑整个项目，但应该暂停使用“q95 战胜所有分布方法”的强表述。先做 P0 runtime-only 正交重跑。 如果 q95 在 ScaledPred50 和 ShuffledTail 之后仍明显领先，那么这个冠军才真正有机制含义。
