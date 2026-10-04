# 主表基线实现复验（第二轮，网页版独立审核）— ACCEPT

## 来源与范围

网页版 GPT（GPT-5.6 Sol + High；generation 11 会话 `6ac1f777-6728-83ec-9281-18a3cb787f23`，
generation 10 达长度上限后换代）对固定提交 `e9eacdeb6aeaaaa7737c2b4dec3753a0d9e9b687`
（相对 2424f67：六项 P1 修复 39717fd + 跟进 e9eacde）做静态复验，直接读 GitHub。

**结论：ACCEPT — P0 0 / P1 0；上一轮六项 P1 全部 CLOSED；基线实现可冻结。**

关键边界（必须保持）：**P1-6 CLOSED ≠ profile activation gate PASSED**；
真实 dev 缺 shape/identity → profile gate 维持 **PARTIAL / NOT PASSED**，直到真实 benchmark
contract 带上 profile 字段，**不得因本次 ACCEPT 改绿**。

非阻断跟进：① Hermes 部分 docstring 残留旧措辞；② 真实 v0.4.1 workload artifact 未随提交入库，
建议 artifact freeze 时保存 `workflow_type_id` 分布 + workload SHA。

审核回复原文如下（未改动）。

---

炜，结论：**ACCEPT**。

我按固定提交 `e9eacdeb6aeaaaa7737c2b4dec3753a0d9e9b687` 复查了六项 P1 的实际实现、simulator 调用链、对应回归测试以及 `activation_report_v2` / audit 逻辑。**六项 P1 均可关闭，本轮没有发现新的 P0/P1。**

| 项目 | 复验结论 | 关键原因 |
|---|---|---|
| P1-1 Hermes 真值时长泄漏 | **CLOSED** | `_hermes_prefetch_plan` 已改为 `work_start + predicted_work_ms × slowdown`；预测缺失 fail-closed；没有再从 `finish_ms` 推未来完成时刻。测试也覆盖“预测相同、真实 runtime 不同但 prewarm 不变”。 |
| P1-2 QLM 时钟原点 | **CLOSED** | `qlm_saa_choice` 明确接收 `now_ms`，scenario clock 从当前决策时刻开始；simulator 调用处也实际传入 `now`，不是只修 helper。反事实 swap-cost audit 两边还复用了相同 RNG state。 |
| P1-3 Gittins | **CLOSED** | 主 `gittins_index` 现在确实对经验分布的 distinct support 计算精确 inf；旧近似已独立为 `gittins_index_bucketed`。`{10,190}` 得 20，而 bucketed 得 38，和理论一致。 |
| P1-4 PDGraph | **CLOSED** | 不只是加字段：构建、lookup、refinement、联合 `(next_role,next_model)` sampling 和 simulator consumer 都串起来了。ρ 门、±25% 筛选、样本不足 fallback 都实际存在；主 dispatch 的 observed prefix 只来自已完成节点。 |
| P1-5 Torpor coverage | **CLOSED** | coverage 已由 canonical `(infer_model,infer_shape,load_model)` 决定；model-level missing estimator 不再影响 coverage。更重要的是我追了 admission path：当前 substrate 上忙卡若目标模型不 resident 就不能 dispatch，因此“忙卡 cold-load interference”确实结构不可达，RESULT 中的披露成立。 |
| P1-6 audit | **CLOSED** | v2 已经从“统计某条件出现次数”改成真正的反事实/机制 activation 统计；Phase A/B 被明确分开，且没有把合成 profile 冒烟伪装成真实 dev 证据。formal gate 明确保持 **PARTIAL / NOT PASSED**，这是正确行为。 |

有一个容易混淆的地方需要明确：**P1-6 CLOSED ≠ profile activation gate PASSED**。这里关闭的是“audit 设计和报告自欺”的代码问题。真实 dev 没有 shape/identity，因此正式 profile gate 仍然必须维持：

> `PARTIAL / NOT PASSED`

直到真实 benchmark contract 带上对应 profile 字段。**不要因为本次 ACCEPT 就把这个 gate 改绿。**

另外我专门检查了一个潜在的新泄漏点：Hermes prewarm 会为了估计“当前运行节点结束后的 prospective state”，把当前节点的 `predicted_work_ms` 加到 completed observed prefix 后再查询 PDGraph。这个和主 dispatch 路径“只累计 completed observed intrinsic”表面上不完全一样，但它使用的是**派发时冻结的预测值，不是真实 runtime**，而且这是为未来 prewarm 时点构造 scheduler-visible projected state，因此我不认为构成 P1。

有两点非阻断问题，可以以后顺手收掉：

1. Hermes 部分 docstring 仍残留旧实现措辞，例如曾经的 scheduled finish / bucketed Gittins 描述，与当前代码已经略不同。属于文档卫生问题，不影响算法。
2. `activation_report_v2` 所引用的真实 v0.4.1 dev workload artifact 没有跟固定提交一起进入 GitHub，因此我无法从“仅 GitHub”独立确认这些真实 rows 中 `workflow_type_id` 的实际非空分布。代码 loader/PDGraph/test 对 workflow conditioning 已经闭环，所以我不把它升为 P1；但正式 artifact freeze 时建议至少保存 `workflow_type_id` 分布 + workload SHA。否则以后 reviewer 想复现实验时，`pdgraph_conditional_used=3511` 本身并不能证明多个真实 workflow type 确实存在。

我也**没有把你给出的 `59/59、63/63、465` 当作通过依据**，因为你已经明确标了 `UNVERIFIED execution report`。这次 ACCEPT 是依据固定 commit 的源码调用链、回归测试内容和 audit 实现本身作出的。

所以本轮状态我建议正式登记为：

**Baseline implementation review round 2 — ACCEPT**  
**P0: 0**  
**P1: 0**  
**Previous six P1: 6/6 CLOSED**  
**Profile activation gate: PARTIAL / NOT PASSED（保持不变）**

这意味着：**基线实现本身可以 freeze；下一步可以进入真实 profile/测量契约，而不需要再开第三轮 baseline 代码修复。**