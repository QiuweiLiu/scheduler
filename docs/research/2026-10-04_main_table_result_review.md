# 主表正式结果复验（网页版独立审查）— ACCEPT / freeze

## 来源与范围

网页版 GPT（GPT-5.6 Sol + High；generation 11）对固定提交
`62be12616494d439d5f025d64bd7f46f77c6cf88`（并回溯运行 commit `a7ab493`）做独立静态审查
（首轮 3×P1 → 修复 → 二轮 ACCEPT）。程序：experiment-reviewer 口径
（泄漏、公平基线、混杂、指标偏差、统计不确定性、声明边界）。

**结论：ACCEPT — P0 0 / P1 0；主表结果 freeze 通过。**

冻结结论：**F0 在冻结 confirm300 上相对五条主表基线均统计显著更好；其中 LLMSched / Parrot /
QLM / Torpor 的劣势超过预注册 +485 ms margin，Hermes 只建立"显著更差"、未建立"差超过 margin"。**

边界（保留）：不得单独推出"F0 的 scheduler 本身更强"（留给 same-interface 2×2）；
旧无扩展 substrate 的五臂结果不得与新表直接横比。正式 300 集运行保持 UNVERIFIED execution report。

审核回复原文如下（未改动）。

---

炜，**ACCEPT**。

**P0 = 0，P1 = 0。** 我按固定提交 `62be12616494d439d5f025d64bd7f46f77c6cf88`，并回溯实际运行 commit `a7ab49367337e0234cfec7e4bda52d752ff3cc98` 重新审查，上一轮三个阻断项均已关闭。

1. **P1-1 CLOSED — substrate 实际语义与声明已对齐。**  
我确认上一轮这里有一部分判断需要修正：`batch_enabled` 控制的是策略侧 `choose_action_with_batch`，它确实会引入自己的 myopic 排序目标，因此不能给五条冻结 baseline 直接打开；而引擎级同模型 batching 是另一条 substrate admission 路径。运行器现在也明确把 co-location / engine batching 定义为环境 admission，把 request preemption 定义为 policy-gated capability，并记录运行期计数。fileciteturn246file0  
v2 固定产物实际保存了 co-location `2,624,836 / 1,901,244`、batching `780,498 / 1,310,626` 的 admission/rejection，以及六臂/F0 的 `preemption_events=0`；RESULT 也明确解释了抢占为何为 0、策略 batch dispatcher 为何关闭。fileciteturn238file0 fileciteturn237file0  
直接检查 simulator 后，没有发现 F0 特有的 batching/co-location admission 路径；这部分是六臂共享环境能力。Hermes/Torpor 等自身的 prewarm/eviction 行为属于冻结 baseline mechanism，不属于 substrate 对 F0 的特殊优待。

2. **P1-2 CLOSED — F0 reference bytes 已被不可变绑定。**  
`f0_reference_artifacts_manifest_v1.json` 已入库，逐文件记录 SHA256、size、pack tree hash，并记录 overlay 的 artifact/checkpoint/frozen-J3/base-pack provenance。fileciteturn239file0  
更重要的是 runner 不是“只记录 manifest”：正式运行开始前会逐文件验证存在性、size、SHA256，再重算 tree hash，任一不符直接中止；验证后的 manifest SHA/tree hash/provenance 随正式产物保存。fileciteturn246file0 v2 artifact 中确实钉住 manifest SHA `c986ed17…`、base tree `3f350705…`、overlay tree `944fb8b8…` 和 artifact SHA `586ae65d…`。fileciteturn238file0  
因此上一轮的最小反例——“同一路径换成另一套本地 F0 bytes，结果 provenance 不变”——现在已经不能通过 formal gate。

3. **P1-3 CLOSED — 统计判定已正确拆开。**  
当前实现分别计算 `CI_lower>0`、`CI_upper<485`、`CI_lower>485`，不再把“non-inferiority 未建立”误写成“margin-inferior”。fileciteturn245file0  
Hermes 在正式产物中正确成为：`statistically_worse=true / non_inferior=false / margin_inferior=false`，对应 `+350.4 [46.4,616.5]`；其余四臂才是 margin-inferior。fileciteturn243file0 RESULT 的文字结论也与此完全一致。fileciteturn237file0

因此这张表现在可以按当前声明边界**冻结为主表正式 full-system comparison**：

**F0 在冻结 confirm300 上相对五条联合基线均统计显著更好；其中 LLMSched / Parrot / QLM / Torpor 的劣势超过预注册 +485 ms margin，Hermes 只建立“显著更差”，没有建立“差超过 margin”。**

同时继续保留现有边界是正确的：这不能单独推出“F0 的 scheduler 本身更强”，scheduler 因果贡献仍交给 same-interface 2×2；旧无扩展 substrate 的五臂也不应与新表直接横向比较。fileciteturn237file0

**最终裁定：ACCEPT。主表结果 freeze 可以通过。**

正式 300 集运行本身仍按你声明保持 **UNVERIFIED execution report**；我的 ACCEPT 是对固定代码、正式结果产物、provenance、统计实现和声明边界的独立静态审查。