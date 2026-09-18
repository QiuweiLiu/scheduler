---
export_backend: "ChatGPT Web DOM"
conversation_id: "6a9822da-e278-83e9-9c1a-675923acda0e"
conversation_url: "https://chatgpt.com/c/6a9822da-e278-83e9-9c1a-675923acda0e"
title: "架构设计评估"
source: "visible_dom"
completeness: "visible_dom_only"
message_count: "8"
timestamp: "2026-09-18T22:58+08:00"
---

# Conversation Export

## 1. user
analysis_phase18_step_level_v4.py
Python
test_phase18_step_level_v4.py
Python
两个新附件（v4，避免同名去重）：analysis_phase18_step_level_v4.py、test_phase18_step_level_v4.py。
sha256 前 16 位：29cdb8a32c043e7f / 4194641330091403。送达自检：文件里应出现
s_true_next_k、remaining_after、CONTAINER_EVENT_TYPE、wall_clock_check。

## 一、口径问题已按数据定死（不是拍的）

你上轮问的"第 2 步花多久"，我去 120 条原始 trace 里量了：

1. api_call 的 timestamp_end == timestamp_start（889/889 条）→ 区间不可测，耗时只能用 runtime_ms；
2. 同一步骤内的事件不重叠（action 有真实区间且紧接在调用之后）→ **步骤耗时 = 该步非容器事件之和**；
3. **event_type == "run" 的节点是"整条流的墙钟容器"**：120 条样本里"容器 runtime / 整文件时间跨度"中位 = 1.000。

第 3 条是关键：那条 44.6 秒的 run:14 不是"下一步要花的 44 秒"，而是整条工作流的墙钟汇总。
把它当未来工作求和 = 把整条流重复算一遍（首跑 r≈0.01 就是这么来的）。

脚本现在内置校验：**非容器事件之和 / 容器墙钟**，640 个模板实测**中位 0.982**（p10 0.970、p90 0.988）。
真值口径：s_true_next_k（锚点之后 K 步的非容器事件）与 s_true_remaining_all（锚点之后全部剩余事件），
容器单独报告（s_true_remaining_with_containers）保持可见。26/26 单测通过；干跑 join 8,295 个步骤锚点
（另有 640 个容器锚点 + 640 个 :run:1 根锚点被计数排除），unjoined_other = 0，视频 160。

## 二、顺带查出一个更严重的问题（新 P0，需你裁决）

既然容器事件 = 整条流墙钟，我顺手查了**模拟器怎么用它**：

- 容器节点 execution_lane=gpu、runtime_ms≈54 秒（api_call 中位 4.7 秒）；
- load_templates() 里**没有任何 event_type 过滤** → 容器被当普通 GPU 节点参与调度；
- 实测确认：episode validation_000000 的 16 个 job 模板共 **224 个节点**（其中 **16 个是容器**），
  而真实运行记录里 nodes = 224、completed_nodes = 224 —— **容器确实被执行了**；
- 量级：容器只占 640/8,935 个节点，却占总 node runtime 的 **50.2%**（GPU runtime 的 50.5%）。

也就是说：**仿真里的 GPU 工作量约为真实测量执行的 2 倍**，而且每个 job 末尾都会多出一个 ~54 秒的巨型调度对象。

## 三、请裁决

1. 真值口径（步骤级 + 排除容器 + 只用 runtime_ms）是否通过？通过我就跑正式 Phase18 统计。
2. 容器 P0 该怎么处理？我倾向：把 event_type == "run" 从 workload 里排除后**重跑主对照**（基线矩阵 + q95/E2/E0），
   先看**策略排序是否改变**（绝对数字肯定会变）。是否同意？还是先只做"excluded vs included"的单臂对照？
3. 这个容器问题会不会影响既有结论的**表述**（不只是数字）？例如 "load/contention 无关"、"q95 的收益来自尾部"
   这类机制结论，是否可能被一个每 job 一个的 54 秒巨型节点放大或掩盖？
4. 是否需要在论文里报告这个发现（作为 workload 构造的修正），还是仅内部修正后重跑即可？

（约束不变：不动预测器；不重跑 1,000 集；不改 legacy 臂数值。未获批准前我不会修改冻结 workload。）

## 2. user
analysis_phase19_container_fix_v1.py
Python
phase19_paired_diff_v1.json
文件
phase19_report_v1.md
文件
phase19_v03_template_summary_v1.json
文件
rebuild_episodes_no_container_v1.py
Python
remove_run_container_nodes_v1.py
Python
Phase 19 已按你的规划执行完，结果**是 情况 B**。附件 6 个（报告 / paired_diff / v03 模板摘要 / 三个脚本）。

## 执行了什么（你的验收点全部满足）

- v03 模板：删 640 个容器，剩 **8,295** 节点 / **640** 模板 / **640** 根 / **悬空前驱 0**，
  runtime_removed_ratio = 0.5024（与预期 50.2% 一致）。
- v03 episodes：用同一构造公式重算；episode_id / template_id / 到达位置 / 压力 / 拓扑 / 初始驻留 与 v02 一致，
  窗口按新工作量重算 → **实测负载仍是 0.5000**（不是"顺带把负载减半"）。
- **重建可信性锚点（关键）**：重建脚本有 self-check 模式，用 **v02 模板**重算 **v02 的 21,000 条 episode**，
  **逐字段 0 不匹配**（容差 1e-2 ms）→ 说明重建是构造器的忠实逆变换，而不是我另写了一套公式。
- 冒烟：validation_000000–000099（100 集）× 5 策略（myopic/E2/predopt_h5_r50/r95/FCFS）× {v02, v03}，
  同 episode id 配对；bootstrap 聚类 = **episode_id**，B=2000。
- legacy v02、outputs/phase*、预测 artifact **一律未改**。

## 结果

**排序变了：**
- legacy v02：r95 < E2 < r50 < fcfs ≈ E0
- v03：r95 < r50 < E0 < E2 < fcfs

**配对对照（ms，负 = 左边更好）**

| 对照 | legacy v02 | v03 | 修正效应 |
|---|---|---|---|
| r95 − E2 | **−4,460 [−5,693, −3,336]** | **−3,129 [−3,950, −2,417]** | +1,331 [+349, +2,284] |
| **r95 − E0** | **−15,081 [−19,074, −11,543]** | **−2,866 [−3,510, −2,254]** | **+12,215 [+8,930, +15,944]** |
| r50 − E0 | −817 [−1,736, +78] | −1,027 [−1,466, −564] | −210 [−1,315, +873] |
| **E2 − E0** | **−10,621 [−13,680, −7,862]** | **+263 [−392, +956]** | **+10,884 [+7,975, +14,070]** |
| fcfs − E0 | −331 [−1,659, +882] | **+11,812 [+9,354, +14,400]** | +12,143 [+9,289, +15,191] |

## 我的读法（请核）

- **排序改变 → 情况 B**；
- **r95 仍最优**且 r95−E2 仍显著为负（CI 排除 0）→ "尾部消费优于点消费"**存活**；
- **E2−E0 由 −10.6 s 变 +0.26 s（CI 跨 0）** → **"未来信息本身有价值"不成立**（至少在点消费形式下）；
- **r95 相对 E0 的收益缩到 ~19%**（−15.1 s → −2.9 s）→ 约 81% 的既有收益来自容器构造；
- fcfs 在 v03 下比 E0 差 11.8 s（v02 下几乎无差）→ 那个 54 秒巨型节点此前主导了队列动态，
  这也解释了"收益随 contention 增强"为何可能是假象。

## 请裁决

1. 上述读法是否成立？有没有我漏掉的混淆（例如：v03 的窗口缩短导致到达更密集，是否引入了新的耦合）？
2. **主实验集要不要整体迁到 v03 重跑**？范围与顺序建议（基线矩阵 / 6-cell sweep / 敏感性 / Phase 16 四臂）？
3. 既有 v02 结果在论文中的处置：是否就按你上轮给的措辞标为 legacy workload sensitivity？
4. 机制叙事需要怎么改写？现在能保留的强主张是什么（我理解只剩"保守尾部消费 > 点消费"，且幅度小得多）？
5. Phase 18 的聚合乐观性诊断结论（p50 偏低 / p95 是保守上界）是在**不含容器**的真值上做的，是否不受本次影响、可继续使用？

## 3. user
phase20a_paired_diff_v1.json
文件
phase20a_report_v1.md
文件
附件：phase20a_report_v1.md、phase20a_paired_diff_v1.json。P0 已按你的优先级执行完，请审。

## 设置

- **v03 全 1000 集**（validation_000000–000999），与 legacy v02 **同 episode id 配对**；
- 策略：fcfs / myopic(E0) / predopt_h5(E2) / r50 / r95；bootstrap 聚类 = episode_id，B=2000；
- 两 workload 用同一份冻结 artifact；legacy v02 与旧输出未改动。
- **复现性检查通过**：legacy v02 的 1000 集结果与已归档 Phase 11 leaderboard **逐位相同**
  （r95 = 180,407.2；E2 = 187,614.9）→ 对照管线与历史一致，v02↔v03 可比。

## 结果

排序：legacy r95 < E2 < fcfs < r50 < myopic → v03 r95 < E2 < r50 < myopic < fcfs。

| 对照 (ms) | legacy v02 | v03 | 修正效应 |
|---|---|---|---|
| r95 − E2 | −7,208 [−8,016, −6,464] | **−4,136 [−4,488, −3,805]** | +3,071 |
| r95 − E0 | −34,709 [−37,792, −31,773] | **−4,929 [−5,257, −4,618]** | +29,780 |
| r50 − E0 | −757 [−1,090, −395] | −102 [−343, +148] | +655 |
| **E2 − E0** | −27,501 [−30,103, −25,007] | **−793 [−1,062, −507]** | +26,708 |
| fcfs − E0 | −1,835 [−2,683, −1,082] | **+27,977 [+25,727, +30,382]** | +29,811 |

绝对量级约减半；GPU 利用率 0.735 → 0.768。

## 需要你确认的一处修正（重要）

100 集冒烟时 E2 − E0 = **+263，CI [−392, +956]**，我据此写了"E2 的优势消失"。
全 1000 集是 **−793，CI [−1,062, −507]** —— **仍显著**。

所以我把表述改成：**"E2 相对 E0 的优势缩小约 97%（27.5 s → 0.79 s），而不是消失"**。
请确认这个修正是否就是你要的；以及 **0.79 s 这个残余**在论文里该怎么处理
（我还是倾向写成"practically negligible"并给 CI，而不是声称"仍有显著收益"）。

## 请裁决的问题

1. 上述修正与措辞是否认可？残余 0.79 s 的正确写法？
2. **FCFS 从"与 E0 持平"变成"+28 s"**，这个反转在论文里怎么用？（我理解它证明旧 workload 的队列动态被那个 54 秒节点主导。）
3. **P1 的具体设计**：在 v03 上重做 Phase 18（forecast optimism）+ 消费函数对照。
   - 消费臂的最小集是什么？（我倾向 r50 / r90 / r95 / scaled-p50（scale-matched）/ shuffled-tail，是否还要 CVaR / comonotone？）
   - 目标是把"预测误差 → 消费函数 → 调度效果"串起来，需要哪些量（例如 per-anchor bias、tail-gap 与真实误差的关系、以及这些量与调度 Δ 的关联）？
   - 用 dev700 还是全 1000？confirm300 语义在 v03 下是否要重新定义？
4. **P2 sweep 设计**：v03 上的 cell 与臂怎么选（沿用 6 cell？arrival×deadline？还是换成 pressure×horizon）？
5. **绝对数字**（v03 上 r95 = 90,939 ms 等）能否进论文正文，还是只报对比？
6. 我在 v03 构造里还有没有没检查的残余混淆？（我的三条声明：删的是终端叶子无重连；episode 构造公式逐字段自检 0 不匹配；
   负载守恒 0.5000；保留归一化到达模式而非绝对时间戳。）

## 4. assistant
我审完 Phase20A。结论先说：P0 通过，v03 可以作为新的主 workload 基线。你的修正表述是正确的，而且比 100 集冒烟时严谨很多。现在研究问题已经从“未来信息有没有用”收敛到了“预测信息如何被消费”。

Phase20A 的关键事实已经稳定：

v02→v03 不是随机重跑：1000 集同 episode id 配对，策略和 artifact 固定，v02 旧结果逐位复现 Phase11。

phase20a_report_v1

v03 排序稳定为 r95 > E2 > r50 > E0 > FCFS（按 completion 从小到大）。

phase20a_report_v1

r95-E2 仍显著：−4.1s，CI 不跨 0。

phase20a_report_v1

E2-E0 仍存在，但从 27.5s 缩小到 0.79s。

phase20a_report_v1

下面逐项裁决。

1. E2 的表述：你的修正完全正确

之前“E2 优势消失”是不准确的。

原因：

100 集：

263ms, CI=[−392,956]

只能说明：

小样本无法确认。

不能说明：

不存在。

1000 集：

−793ms, CI=[−1062,−507]

说明：

存在统计显著差异。

但是论文里我不建议写：

E2 still provides significant improvement

虽然统计上成立，但会误导。

因为：

27.5s → 0.79s

下降：

97.1%

这个效应已经接近工程噪声尺度。

建议写：

Future-aware scoring with point estimates retains a statistically detectable but practically small advantage over myopic scheduling after workload correction (0.79s, 95% CI [0.51s,1.06s]), suggesting that forecast availability alone contributes limited gains without uncertainty-aware aggregation.

中文：

未来信息本身仍提供小幅统计收益，但经过 workload 修正后，其工程收益显著降低。

不要写：

“practically negligible”。

因为 reviewer 可能问：

“你凭什么定义 negligible？”

更安全：

practically modest

2. FCFS 反转怎么解释？

你的理解基本正确。

但是不要直接写：

54秒节点主导队列动态

这个需要因果证据。

更严谨：

The workload artifact substantially altered queue dynamics, as evidenced by the large FCFS/E0 reversal after removing duplicated workflow-level container nodes.

证据：

v02:

FCFS-E0:

−1835ms

几乎一样。

v03:

+27977ms

FCFS明显变差。

phase20a_report_v1

所以可以说：

“队列行为被改变”。

不能说：

“完全由54秒节点导致”。

后者需要额外 attribution experiment。

3. P1 Phase18 + consumer design

这里是现在最关键的一步。

目标：

建立：

forecast error
        ↓
aggregation bias
        ↓
consumer decision
        ↓
scheduler performance

我建议不要一下加太多 consumer。

第一阶段最小矩阵：

必须有：

Consumer	目的
p50	乐观基准
p90	中等保守
p95	当前冠军
scaled-p50	检验是不是简单校准即可
tail-shuffle	检验相关结构
oracle truth	上界

暂时不要：

CVaR

原因：

现在你的问题不是 distribution optimization。

而是：

forecast quantile calibration。

CVaR会引入：

tail definition

sampling

risk measure

变量太多。

comonotonic

保留，但作为分析，不作为主 consumer。

Phase18 新版指标

必须增加：

1. forecast bias

每个 anchor：

b=S
true
	​

−S
pred
	​


报告：

mean bias

median bias

underestimate rate

2. aggregation gap

重点：

比较：

∑Q
0.95
	​

(x
i
	​

)

和：

Q
0.95
	​

(∑x
i
	​

)

如果：

前者明显大：

说明 q95收益来自保守聚合。

3. 决策相关指标

不是只看预测误差。

增加：

对于每个 scheduler decision：

candidate ranking agreement

chosen action regret

selected job true remaining cost

这是连接预测和调度的桥。

数据划分

不要重新定义。

建议：

dev700

用于：

consumer设计

threshold选择

confirm300

保持：

最终一次验证

但是：

v03 是新 workload。

所以名字应该改：

v03-dev700
v03-confirm300

不要沿用旧 confirm300。

4. P2 sweep

不要沿用旧 6-cell。

因为旧机制：

contention

已经被污染。

现在更合理：

主轴：
pressure × horizon

例如：

pressure:

0.25
0.5
0.75
1.0

horizon:

1
3
5

原因：

你的贡献：

未来消费。

所以应该证明：

预测窗口越长，不确定性越重要。

deadline sweep：

作为 sensitivity。

不是主实验。

5. v03绝对数字是否进正文？

可以，但不要作为核心。

正文：

主要：

relative improvement。

例如：

r95 reduces completion time by 4.9s compared with myopic.

绝对：

90,939ms。

可以放表。

原因：

workload correction改变绝对尺度。

更重要的是：

relative ordering。

6. v03还有没有残余混淆？

我检查你的三条声明。

(1) 纯删除终端叶子

通过。

代码和summary支持：

remove event_type run

terminal leaf

no rewiring

remove_run_container_nodes_v1

(2) episode重建自检

通过。

self-check重新生成v02：

字段匹配。

rebuild_episodes_no_container_v1

(3) load保持0.5

通过。

但论文注意：

不是“workload完全不变”。

你改变了：

absolute runtime

arrival timestamp

保持：

normalized arrival order

offered compute load

这是正确设计。

还有两个小风险：

P1-remaining-1：artifact覆盖问题

你的预测artifact包含原8,935节点。

v03只有8295。

需要确认：

consumer lookup不会静默跳过。

必须报告：

artifact matched nodes / workload nodes

否则 reviewer 会问：

“删除节点后预测覆盖率？”

P1-remaining-2：v03上的deadline分布

因为：

deadline:

arrival+multiplier×total_p90

改变了。

所以需要报告：

v02 vs v03：

deadline mean

deadline slack

否则：

性能提升可能来自 deadline distribution 改变。

最终裁决
已通过：

✅ v03作为主 workload
✅ E2结论修改为“小幅统计优势”
✅ q95核心故事保留
✅ Phase18继续

论文主线改成：

以前：

Future information improves scheduling.

现在：

Forecast information alone provides limited gains; the key challenge is uncertainty-aware consumption of imperfect future estimates. Tail-aware aggregation improves scheduling robustness under forecast uncertainty.

下一步：

我建议不要直接跑 CP-SAT/MPC。

先做 Phase18 + consumer matrix。

因为现在真正的论文贡献已经从：

“预测器”

转移到了：

forecast-to-control interface（预测到调度决策的消费层）。

这条线比原来的调度算法故事更统一。

## 5. user
查出一个**新的 P0：预测输入特征缺口**，用户已决定"**修管线、让预测器表现回正常水平；模型不动**"。请给可执行的修复与验证方案。

## 证据链（全部已在本地核实）

1. **S_* 锚点的任务元数据 100% 缺失**：
   outputs/sstar_predictor_anchors/coverage_report.json 显示
   domain / answer_type / official_task_type / question_type / sub_category 在 **9,575/9,575** 个锚点上全是 "unknown"。
   域内（J 数据集验证集）只有约 22% 是 unknown。
2. **R7 的 trace 与 template 里根本没有这些字段**：
   raw trace 的 keys 只有 action/dataset/input/model_id/node_type/resource/step_id/timestamp…；
   template 有 dataset="videomme"、task_id、workflow_type_id="videomme.langgraph_react"，
   但 **没有 task_context**，且 task_structure = {}（空）。
3. **元数据其实在项目里就有，而且 100% 覆盖 R7 视频**：
   data/manifests/videomme_600_source_v2.jsonl（600 条）含
   domain / official_task_type / sub_category / duration / question_id / answer / options / question；
   R7 的 160 个视频 **160/160 命中**。
4. **字段构造的参考实现**：scripts/build_p9d_topology_dataset.py::_task_context(sample) ——
   从 sample 读 answer_type, domain, question_type, required_modalities, temporal_scope, sub_category, official_task_type，
   回退到 sample["task_structure"]，再回退 "unknown"。J 数据集就是用它造的。
5. **为什么原有检查没发现（静默失败）**：
   锚点 builder 的 coverage 检查测的是 **OOV 率**；因为 "unknown" 本身就在 J 词表里，
   这些字段的 oov_rate = **0.0000** → 一路绿灯。（只有 temporal_scope 的 oov 是 1.0，被忽略。）
6. **后果量化**：同一批步骤上比较，预测器的 p50 平均只有"实测资源表"的 **0.38 倍**（p95 是 2.3 倍）；
   域内 J3:seed11 的分位覆盖率是 **0.556 / 0.917 / 0.968**（目标 0.50/0.90/0.95）、pinball 845ms、长度 MAE 0.047。

## 现状与约束

- **模型不动**：同一个冻结 checkpoint（J3:seed11）、同一权重；只修**输入特征管线**。
- 必须在**训练机**（Windows + torch）上执行；本机（macOS）没有 torch，跑不了 packer。
- 现有 S_* 预测包与基于它的**全部调度结果**都作废（若重生成）。
- legacy v02 workload、旧 outputs、预测器权重都不许覆盖。

## 请给的方案（尽量到文件/命令粒度）

1. **改哪里**：build_sstar_predictor_anchors.py 里 join 注册表的正确做法（按 video_id？注册表还含
   question/answer，是否要把这些都作为特征？还是只补 task_context 的那几个字段）？
   required_modalities / temporal_scope 在注册表里没有，是否留 "unknown" 可以接受？
2. **小规模验证设计（优先，成本最小）**：先只重生成一个子集（比如 500~1,000 个锚点），
   只回答一个问题：**补上元数据后，p50 相对实测表的 0.38 倍偏差是否明显收敛？**
   请给判据（例如中位比值进入 [0.7,1.4]、分位覆盖率回到 [0.45,0.60]/[0.85,0.95]/[0.92,0.99]）。
3. **是否要同时补强 coverage 检查**：把"所有锚点都是 unknown"变成硬失败（而不是只看 OOV 率）？还有哪些类似的静默失败值得一并加断言？
4. **验证通过后的完整顺序**：重生成全量 S_* 预测 → 重跑哪些实验（基线矩阵？消费矩阵？Phase 18 重做？）
   → 哪些旧结论需要重审；是否需要重跑 v03 workload 相关的全部内容。
5. **回滚与隔离**：命名约定（例如 sstar_predictor_artifacts_v2_...）、旧包如何标注、如何保证不覆盖。
6. **论文影响**：如果修复后预测器明显变好，我们的"弱预测下稳健消费"叙事怎么调整？
   以及"工作负载/输入管线修正"这条要不要按之前的 workload-construction correction 一样披露。
7. 有没有**更强的验证**方式：例如同时报告"修复前 vs 修复后"的 p50/p95 校准曲线，作为论文的附录证据？

约束：不动模型权重；不覆盖旧产物；不跑 1,000 集调度重跑，除非你明确说值得。

## 6. assistant
这个 P0 我认可，而且这次修复边界应该非常严格：

只恢复冻结 J3:seed11 训练时本来就存在的输入上下文，不改模型、不新增语义特征、不重新训练。

我核了当前仓库实现。build_sstar_predictor_anchors.py 现在把 run_manifest 直接作为 sample_row 交给 P9d 的 make_feature()；而 _task_context() 正是从 sample_row 读取 answer_type/domain/question_type/required_modalities/temporal_scope/sub_category/official_task_type。所以你发现的缺口确实发生在 S_ anchor construction → frozen encoder* 之间。另外 J 的实际 encoder 只消费这些 categorical context 和 required_modalities multi-hot；question/answer/options 并不属于模型输入，其中 answer 甚至属于禁止进入 model_input 的字段。

我建议把这轮定义为 Phase21：S_ task-context pipeline repair*。

1. 代码怎么改

只改 scripts/build_sstar_predictor_anchors.py，不要改 build_p9d_topology_dataset.py、j_series_common.py 或 checkpoint。

新增两个 CLI：

PowerShell
--task-registry data/manifests/videomme_600_source_v2.jsonl
--video-allowlist <optional pilot list>

当前 registry 我已经核过：600 行、600 个唯一 video_id，没有重复。所以当前版本按 video_id join 是安全的；但代码必须先 assert uniqueness，不能把“现在刚好唯一”写成隐含假设。

建议实现：

Python
运行
REGISTRY_TASK_FIELDS = (
    "domain",
    "official_task_type",
    "sub_category",
)

每个 run：

Python
运行
registry_row = registry_by_video[video_id]

enriched_sample = dict(run_manifest)

for field in REGISTRY_TASK_FIELDS:
    old = enriched_sample.get(field)
    new = registry_row.get(field)

    if old not in (None, "", "unknown") and new not in (None, ""):
        if str(old) != str(new):
            raise ValueError(...)

    if old in (None, "", "unknown"):
        enriched_sample[field] = new

然后把：

Python
运行
"sample_row": run_manifest

换成：

Python
运行
"sample_row": enriched_sample

并把 question_id、registry SHA、join key 放在 model_input 外面的 audit metadata 即可。

不要把下面这些复制进模型：

question
answer
options
duration
question_id

尤其不要用 answer。这不但不是训练输入，而且会破坏 causal contract。

answer_type 和 question_type：继续 unknown。不要根据 options 或 official_task_type 猜。

temporal_scope：继续 unknown。不要把 Video-MME 的 duration=short/medium/long 擅自映射成 temporal_scope，除非你找到 J 数据构造阶段已经存在的完全相同映射。

required_modalities：同样不要发明。先保持冻结训练契约的 fallback；但我建议顺手审一下 J train 中该字段的实际类型，因为当前 _task_context() 缺失时产生字符串 "unknown"，而 encoder 按列表迭代 required_modalities。这类类型错位应该进入 coverage audit，别顺手改 encoder——那会变成第二项模型输入变更。

换句话说，这轮真正补的是：

domain + official_task_type + sub_category
	​


而不是“把 registry 里所有信息都塞给模型”。

2. 先做约 1000 anchor 的 pilot，不要直接全量

不要用 --limit 1000 anchors，因为按文件顺序切会让视频/baseline 分布偏。

建议固定 16 个 video，每个 video 保留全部 R7 runs。你现在大约 9575 anchors / 160 videos，16 videos 应接近 950–1000 anchors。

选择方法要在看结果前固定，例如：

Python
运行
sorted(video_ids, key=lambda x: sha256(x.encode()).hexdigest())[:16]

输出：

outputs/phase21_taskctx_fix/
├── pilot_videos.txt
├── anchors_old_masked/
├── anchors_fixed/
├── artifacts_old_masked_full5/
├── artifacts_fixed_full5/
└── audit/

这里建议做一个非常强的 paired A/B：

A = 原始输入：

domain = unknown
official_task_type = unknown
sub_category = unknown

B = repaired registry context。

除这三个字段之外：

anchor id 相同；

history 相同；

current node 相同；

stack_context 相同；

prefix_hash 相同；

checkpoint SHA 相同。

这样如果输出变化，你可以非常干净地归因于 context repair。

Windows 执行骨架

假设仓库还是冻结 config 中的 F:\scheduler：

PowerShell
$PY   = "D:\anaconda\envs\scheduler\python.exe"
$ROOT = "F:\scheduler"
Set-Location $ROOT

$CFG = "experiments\EXP-20260911_p9d_j_predictor_acceptance\config.json"
$REG = "data\manifests\videomme_600_source_v2.jsonl"
$VOC = "results\processed\j_series_dataset_v1\j_train.jsonl.gz"
$RUNS = "results\raw\r7_trace_full_20260817"
$SPLIT = "results\processed\r7_workload_20260817\split_manifest_train_validation.jsonl"

生成 repaired pilot：

PowerShell
& $PY scripts\build_sstar_predictor_anchors.py `
  --runs-root $RUNS `
  --manifest $SPLIT `
  --vocab-source $VOC `
  --task-registry $REG `
  --video-allowlist outputs\phase21_taskctx_fix\pilot_videos.txt `
  --output outputs\phase21_taskctx_fix\anchors_fixed\features_sstar.jsonl.gz

原始 masked control 最好也由修改后的同一个 builder生成，只加例如：

--mask-registry-task-context

这样不会拿旧文件和新程序混比。

然后做 calibration-only pack：

PowerShell
& $PY scripts\pack_j_predictor_artifacts.py `
  --config $CFG `
  --anchors-file outputs\phase21_taskctx_fix\anchors_fixed\features_sstar.jsonl.gz `
  --output-root outputs\phase21_taskctx_fix\artifacts_fixed_full5 `
  --min-steps 5 `
  --emit-distributions

masked control 完全相同，只换 anchors/output。

这里 --min-steps 5 仅用于 predictor calibration audit，这样预测长度错误不会把 runtime head 的后续 slot 隐藏掉。最终 scheduler artifact 不加这个参数，仍使用模型 argmax length。

3. Pilot 的验收判据

我建议分“管线 gate”和“性能 gate”。

管线 gate 必须全部 PASS：

registry：600 rows = 600 unique video IDs；

pilot video registry join = 100%；

pilot anchor join = 100%；

domain / official_task_type / sub_category unknown rate = 0%；

三字段之外 model_input byte-equivalent / canonical-hash equivalent；

anchor IDs 逐个一致；

prefix hashes 逐个一致；

checkpoint SHA 必须仍为冻结 J3:seed11：
0ee8ded4...49c77；

forbidden model-input audit 仍为 0。

然后性能 gate 用相同 anchor、相同真实 future slot paired 比较。

我建议报告：

R
τ
	​

=
∑y
∑q
τ
	​

	​


以及 empirical coverage：

C
τ
	​

=P(y≤q
τ
	​

)

再加 raw-ms pinball。

当前最关心的强通过标准可以定成：

R50_fixed ∈ [0.70, 1.30]；

p50 coverage ∈ [0.45, 0.65]；

p90 coverage ∈ [0.85, 0.97]；

p95 coverage ∈ [0.92, 0.995]；

runtime mean pinball 相比 masked control 至少下降 10%；

next-role / next-family 不允许退化 >2pp。

我不建议要求完全恢复 J validation 的 0.556/0.917/0.968，因为 R7 stack/baseline 仍然存在真实 domain shift。

再加一个“最低可接受改善”兜底：

∣R50
fixed
	​

−1∣<0.6∣R50
old
	​

−1∣

且 R50_fixed ≥ 0.60。

如果连这个都做不到，就不要立刻全量；说明缺失 task metadata 只是 0.38 偏差的一部分来源。

4. Coverage 检查必须升级

这次已经证明：

OOV=0 不等于输入正常。

所以以后 coverage 至少同时报告：

missing_rate
unknown_rate
oov_rate_all
oov_rate_nonmissing
distinct_count
top_values
source_coverage

其中硬失败建议如下：

domain / official_task_type / sub_category：registry 明明提供，任何 unknown 都 fail；

registry join <100%：fail；

duplicate registry video_id：fail；

所有 task-context fields 同时 unknown 的 row >0：fail；

anchor count / unique sample_id 不一致：fail；

required_modalities 类型异常：fail 或至少显式 gate；

history/current/stack_context 与旧 builder 非预期变化：fail。

而这些字段可以是“允许缺失，但必须显式报告”：

answer_type
question_type
temporal_scope
required_modalities

不要再把“100% unknown + OOV=0”显示成绿色 PASS；应该显示：

SOURCE_UNAVAILABLE / EXPECTED_UNKNOWN

这是完全不同的状态。

5. Pilot 通过后的完整顺序

这次不用动 v03 workload。workload correction 与 predictor input correction 是两个正交修复。

顺序建议：

全量重新生成 S_* anchors。

全量重新 pack J3:seed11 artifacts。

生成新的 scheduler-facing b05_* 目录。

在 v03 template 上做 artifact coverage：

workload nodes = 8295；

covered = 8295；

missing = 0；

artifact extra anchors 可以存在，但必须报告（旧 run/container/root 等不会被 v03 scheduler 消费）。

先重跑 Phase18 predictor calibration。

跑 100-episode scheduler smoke：

E2

r50

r90

r95

E0 可直接复用同 v03 workload 的结果作为 invariant baseline。

如果方向稳定，再跑 dev700 consumer matrix：

E2

r50

r90

r95

scaled-p50

tail-shuffle

重做 pressure/horizon sweep。

CP/optimizer 之后再说。

这时我不建议马上重跑1000集。

但是如果 pilot + dev700 都稳定，最终论文主表值得用 corrected artifact 在 v03 上再跑一次 1000 核心矩阵。因为你现有 Phase20A 中：

E0 / FCFS：仍有效；

E2 / r50 / r95：都因旧 S_ artifact 输入缺口而失效*。

所以最终 1000 只需要重跑真正消费 artifact 的 arms，不需要把所有 artifact-independent baseline 全重算。

6. 哪些旧结论作废，哪些不受影响

直接作废：

旧 S_* 的 runtime calibration / pinball / quantile coverage；

“S_*只是正常 OOD 退化”的旧解释；

Phase18 基于旧 artifact 的 aggregate optimism 数值；

所有使用旧 S_* prediction artifact 的 scheduler policy 结果；

scaled-p50 / tail-shuffle 等从旧 artifact 派生的结果；

使用这些 predictions 的 CP/MPC/forecast-aware 结果。

不受这次修复影响：

J3:seed11 域内 validation/test 本身；

checkpoint；

v03 container-excluded workload；

v03 episode rebuild；

E0 / FCFS 等不读取 future artifacts 的策略；

T1–T3 原始 trace dependence 分析。

所以不需要重新生成 v03 workload。

7. 隔离与命名

不要叫简单的 v2，容易和 workload v02/v03 混。

我建议：

outputs/
├── sstar_predictor_anchors_legacy_missing_taskctx/
├── sstar_predictor_artifacts_legacy_missing_taskctx/
│
├── sstar_predictor_anchors_taskctxfix_v1/
├── sstar_predictor_artifacts_taskctxfix_v1/
└── sstar_predictor_artifacts_taskctxfix_v1_dist_sched/
    └── prediction_artifacts/

manifest 至少记录：

JSON
{
  "checkpoint_sha256": "...J3 seed11...",
  "anchor_builder_version": "taskctxfix_v1",
  "task_registry": "...videomme_600_source_v2.jsonl",
  "task_registry_sha256": "...",
  "join_key": "video_id",
  "recovered_fields": [
    "domain",
    "official_task_type",
    "sub_category"
  ],
  "intentionally_unavailable_fields": [
    "answer_type",
    "question_type",
    "required_modalities",
    "temporal_scope"
  ]
}

旧包显式打：

LEGACY_MISSING_TASK_CONTEXT

绝不能覆盖。

8. 对论文故事的影响

现在不要提前保“弱预测下稳健消费”这条故事。

修完之后会有三种可能：

A. predictor 明显恢复，E2 也明显增强，而 r95-E2 缩小。

故事变成：

Forecast quality matters; conservative consumption mainly compensated deployment-side miscalibration.

这时“稳健消费”降级成辅助结论。

B. predictor 恢复，但 r95 仍稳定优于 E2/r50。

这是最漂亮的情况：

Even after restoring the intended task context and improving forecast calibration, uncertainty-aware consumption still provides independent scheduling value.

这才是真正能证明 consumer 贡献不是 bug compensation。

C. predictor 恢复后 r50/E2 接近或超过 r95。

那就应该承认：

旧 q95 优势主要是在补偿错误输入导致的 forecast optimism。

这不是坏结果，但论文主贡献要转回 predictor + forecast-to-control interface，而不是 tail-risk consumer。

9. 论文要不要披露这个 input-pipeline correction？

要。

但和 container correction一样，写成 reproducibility correction，不要在正文讲成事故：

During scheduler-side inference auditing, we found that benchmark task metadata available at request time was not propagated into the frozen predictor's task-context input, causing categorical fields to collapse to unknown values. We corrected the feature-construction pipeline by joining the benchmark registry while keeping the predictor architecture and checkpoint fixed. All reported scheduler-facing predictions use the corrected input pipeline.

这里最关键的一句是：

architecture and checkpoint fixed

这样 reviewer 很清楚这不是重训后挑了一个更好模型。

10. 我最推荐再加一个“强证据”

除了 before/after calibration curves，我更推荐做一个输入遮蔽反事实：

同一个 repaired anchor：

FIXED:
domain = real
official_task_type = real
sub_category = real

然后复制一份：

MASKED:
domain = unknown
official_task_type = unknown
sub_category = unknown

同 checkpoint、同 anchor、同 batch、同 truth。

你直接得到：

Δq
τ
	​

=q
τ
fixed
	​

−q
τ
masked
	​


以及：

ΔPB=PB
fixed
−PB
masked

这比“旧文件 vs 新文件”更强，因为除了这三个输入 token 没有任何东西改变。

附录里建议放：

old/masked vs fixed 的 p50/p90/p95 empirical coverage；

p50/truth ratio 分布；

pinball；

按 future slot h=1…5；

按 baseline / stack；

按 domain / official_task_type；

paired Δpinball 的 video-cluster bootstrap CI。

如果 fixed 明显改善，这几乎能把“metadata omission → forecast miscalibration”这条因果链钉死。

所以执行顺序我建议非常明确：

先修 builder → 16-video paired masked/fixed pilot → 只做 predictor audit。

现在不要跑任何 scheduler。

只有 pilot 达到上面的 calibration gate 后，才全量重生成 artifact；然后重新做 Phase18，再做100集 scheduler smoke，再进入 dev700。

这比直接全量重打包安全很多，也能避免我们第三次在下游才发现上游输入契约问题。

## 7. user
phase21_calibration_analysis_20260918.py
Python
phase21_anchor_builder_20260918.py
Python
phase21_task_context_20260918.py
Python
phase21_pilot_report_20260918.md
文件
Research Question:
独立审查 Phase 21（S_* task-context 管线修复）的 pilot 代码与判定，并特别复核我刚刚**撤回**的一次误报。

Background / Project Context:
项目 F:\scheduler：视频 Agent 工作流的未来行为预测 → GPU 调度。冻结预测器 J3:seed11（H5，
checkpoint sha256 0ee8ded4…49c77），**模型不动**，本轮只允许改输入管线。主 workload 已切到
r7_workload_v03_no_run_container。本轮（Phase 21）：S_* 锚点的 task_context 在旧锚点里 100% 是 "unknown"
（域内约 78% 有值），怀疑是"输入缺口"，决定只补 domain + official_task_type + sub_category
（来源 data/manifests/videomme_600_source_v2.jsonl），先做 16 视频 paired masked/fixed pilot，
**通过预注册 gate 才全量重生成**。

What We Already Know（全部已在本地端到端实测）:
1. pilot 已跑完（scripts/preprocess/phase21_pilot.ps1，23 秒）。**管线 gate 全过**：975 anchors（两侧一致，576 skipped）、
   anchor id 顺序/集合一致、**prefix_hash 975/975 完全一致**、source_trace_sha256 一致、
   unknown_rate(domain/official_task_type/sub_category)=0 且 source_coverage=1.0（masked 对照 1.0/0.0）、
   task_context_failures=[]、checkpoint SHA 未变、两侧差异**仅限那三个字段**（2925/2925 全变）+ 审计块。
2. **校准 gate 四条全 FAIL**：fixed p50 R=0.3971（门限 [0.70,1.30]）、cov p50/p90/p95 = 0.3960/0.8443/0.9157
   （门限 0.45–0.65 / 0.85–0.97 / 0.92–0.995）；paired Δpinball(fixed−masked) p50 = **+51.59ms [27.23, 74.42]**
   （显著更差），p90 +9.12 [−25.80,43.60]、p95 +4.75 [−23.45,32.88] 不显著。
   masked 侧 p50 R=0.4152 / cov 0.3911；n=3275 对（预注册 baseline 记 3226，另一已记录的 1.5% 口径差）。
3. **我的误报与撤回**：我曾提出"baseline/model_stack_id/planner_model_id 被以 null 写进 task_context，
   遮蔽了 stack_context 的真实值，导致这三个字段 100% 变 UNK"的 P0，并据此收窄了 Phase 21 结论。
   随后**显式测键存在性**发现：task_context 的键集 975/975 =
   {answer_type, domain, official_task_type, question_type, required_modalities, sub_category, temporal_scope}
   （恰好 7 个键），三个 stack 键 key_present = False（975/975）→ 编码器 j_series_common.py:178-183
   正常走 else 分支从 stack_context 读真实值。**误报根因：探针用 dict.get(field)，无法区分
   "键不存在"与"键存在但值为 null"**。该 P0 已撤回，未做任何管线改动。
4. 按编码器口径复算 9 个 context 字段（真实锚点）：answer_type=unknown(in_vocab)、domain/official_task_type/sub_category
   (in_vocab)、question_type=unknown(in_vocab)、temporal_scope=unknown(**UNK**)、baseline=langgraph_react(in_vocab)、
   model_stack_id=stack_a_qwen3_vl8b_yolo11x(**UNK**，域内词表是近似值 stack_a_qwen3_vl8b)、
   planner_model_id=Qwen3-VL-8B-Instruct(**UNK**，域内词表只有 "unknown")。
5. 域内 J train（13,754 行）required_modalities 是**混合类型**：list(1)=8156 / list(2)=2029 / str=3439 / list(3)=130。
6. 我另外修了 3 个端到端路径缺陷（Mac 侧 11/11 单测没有覆盖真实路径）：
   ① oov_rate_all KeyError；② required_modalities KeyError（该字段是 **list 型**、经 vocabs.modalities
   而非 vocabs.context 编码，故单独审计并把字面量 "unknown" 视为缺失）；③ resolve_artifacts 路径不匹配
   （packer 写 j_future_h* 于根目录，脚本找 b05_future_h* 于 prediction_artifacts/）。
7. 真值定义 analysis_phase21_taskctx_calibration.py::truth_sequence = 按 (source_step_ids[0], sequence_index)
   展平后取锚点之后 K 个**非容器**事件（K=5），用模板 runtime_ms。R7 模板是 DAG
   （outdegree 分布 {0:201, 2:401, 3:220, 4:25}，无 outdegree 1）；slot1 有 **51.9%** 的"下一个事件"
   **不是锚点的后代**（是**同一步骤内的并行兄弟事件**，见例子：锚点 api_call:4 → 下一个 action:5，
   而 api_call:6/action:7 同时依赖 api_call:4 与 action:5 的菱形结构）。
   但 S_* 锚点是**事件级**的（current_node_id = event_id，history = events[:index]），
   所以我判断真值定义与预测目标**在粒度上一致**。请复核这个判断。

Questions To Answer（请逐条给判定）:
Q1. truth_sequence 这个真值定义，对"事件级、单链、K=5"的预测输出，是否构成公平的配对标尺？
    若你认为不公平，请给出正确的目标定义，并定性说明它会让 R50 上升/下降多少。
Q2. paired masked/fixed 的机制（同一 builder、只切 --mask-registry-task-context、按 (anchor, slot) 配对、
    video-cluster bootstrap 2000 次）是否有统计或实现缺陷？特别请看：
    (a) summarise 用 ratio_of_sums = Σq/Σy 而不是分布分位比；(b) collect 用 steps[:len(truths)] 截断；
    (c) step_of 混用 source_step_ids[0] 与 sequence_index 回退；(d) 128/975 个锚点因无模板被静默丢弃。
Q3. 预注册门限（R50∈[0.70,1.30]、cov 0.45–0.65 / 0.85–0.97 / 0.92–0.995）是从 J3:seed11 **域内验收**
    （0.556/0.917/0.968）推来的，但 R7 的目标是事件级 next-K、且 stack/baseline 分布不同。
    这些门限对本次 pilot 是否成立？如果不成立，正确的 gate 应该怎么定？
Q4. 复核我的撤回是否正确：在 j_series_common.py:178-183 + Vocab.encode(None)=UNK 的规则下，
    S_* 锚点的 baseline 是否**真的**拿到了真实值？请指出我可能**仍然漏掉**的点
    （这是最容易出错的地方——我已经错过一次）。
Q5. 结论"恢复三个 registry 内容字段不改善 S_* 域外校准，且不存在同类未修的管线输入缺口"是否成立？
    若成立，剩余 ~60% 低估最可能是什么原因？给出你建议的下一步实验（含判据与预期成本）。
Q6. 三个修复（附件 2）是否正确、是否有副作用？

Required Deliverables:
- 逐条 Q1–Q6 的判定，每条标 VERIFIED / INFERENCE / UNVERIFIED
- 你发现的任何 P0/P1 缺陷清单（文件:行 粒度）
- 对 Phase 21 **最终写法**的具体措辞建议（可写 / 不可写）

Preferred Sources: 项目内代码（附件）优先于外部资料。

Constraints:
- 不要建议重训模型或改 checkpoint。
- 不要建议改编码器解析规则（j_series_common.py 属冻结契约）。
- 不要建议全量重生成 S_* artifact，除非先证明 pilot gate 可达成。
- 不要上传或引用原始 trace、问题/答案内容、密钥。

What Must Be Verified:
- Q4 的解析链（我已在 dict.get 上错过一次，请用"键存在性"口径独立复核）。

附件（**以这 4 份为准**，均为 2026-09-18 版本；会话里若存在同名旧附件请忽略）:
1. phase21_calibration_analysis_20260918.py —— scripts/analysis/analysis_phase21_taskctx_calibration.py
   （含 truth_sequence / collect / summarise / paired_delta / resolve_artifacts）
2. phase21_anchor_builder_20260918.py —— scripts/build_sstar_predictor_anchors.py
   （含 registry join、masked 对照、coverage/任务上下文 gate）
3. phase21_task_context_20260918.py —— scripts/preprocess/sstar_task_context.py
   （纯 join 逻辑：TASK_CONTEXT_FIELDS / REGISTRY_UNAVAILABLE_FIELDS / enrich_sample_context）
4. phase21_pilot_report_20260918.md —— Phase 21 pilot 报告（含 §7 事后审查与误报撤回）

补充：冻结编码器的解析规则在 scripts/j_series_common.py 第 178-183 行
（task_context[field] if field in task_context else stack_context[field]），Vocab.encode 在同类文件
（if value is None: return self.index[UNK]）。这两段**未作为附件**上送，但 Q4 的复核依赖它们；
如需可再上传。

## 8. assistant
我先给总裁决：**Phase 21 的核心负结果是可信的——恢复 domain / official_task_type / sub_category 没有改善这个 pilot，p50 甚至显著变差；你撤回“stack 字段被 task_context 的 null 遮蔽”的误报也是正确的。**但我不同意报告里更强的两句结论：

“不存在同类未修的管线输入缺口”
“误差主要来自结构错配与执行期分布偏移”

目前都还证据不足。

我发现一个很重要的新点：**planner_model_id 的实际值 OOV 后会编码成特殊 __UNK__，它不是训练时的字面量 "unknown" token。**因此“训练词表只有 unknown → 没有信息损失”这个判断不成立。这可能是下一项最值得做的小 pilot。

Q1. truth_sequence 是否公平？

判定：INFERENCE — “事件级 next-K”这个思想正确，但当前实现还差一个顺序契约的硬验证。

你关于“51.9% 下一事件不是 DAG 后代并不代表 truth 错”的判断，我同意。

冻结 J v3 的预测目标本身不是“DAG descendant”，而是一个 verified serial control-flow chain：未来 slot 是当前事件之后的下一个可调度 compute event。因此 R7 workload 后来根据 parent_step_ids 构造出的 diamond DAG，和 predictor 的线性目标不是同一个拓扑语义。

所以：

api_call:4 → action:5 即使从 R7 DAG 的 successor 语义看不是你想象中的“未来后代链”，只要它是实际 raw control-flow event order 中的下一个 resource event，它仍然可以是正确的 slot-1 truth。

这点是对的。

但当前 truth_sequence() 不是单纯按 event/trace order：

Python
运行
considered.sort(key=lambda node_id: (step_of[node_id], order[node_id]))

其中 step_of 优先用了 source_step_ids[0]。

phase21_calibration_analysis_20…

而 R7 template 的 sequence_index 本来就是 workload builder 按原 compute-event event_index 排序后赋的。因此对 predictor contract 来说，更自然的 truth 应该是：

anchor 后按 sequence_index 的下 K 个 non-container compute events
	​


而不是先按 step id 再排序。

这不一定已经改变了你现在的数字——step id 很可能恰好单调——但代码里没有证明。

跑任何进一步解释前必须补一个非常便宜的 gate：

for every pilot template:
    order_by_sequence_index
    ==
    order_by_(source_step_id, sequence_index)

至少要报告 mismatch nodes / anchors。

如果 mismatch=0，当前 Phase21 数字无需重跑，Q1 转为 VERIFIED。

如果不为0，就用 sequence_index truth 重算 calibration。R50 上升还是下降无法从现有信息推断，取决于被交换事件的真实 runtime；“51.9% 非 descendant”本身不能决定方向。

另外最好再确认 template 的 compute node 合并规则与 J v3 的 nested-call merge 是一致的，否则要把 nested merge 差异单列。

Q2. masked/fixed paired 机制是否干净？

判定：核心对照 VERIFIED；实现还有两个 P0-guard 和几个 P1。

masked/fixed 的单变量设计很好。builder 只恢复三个 registry-backed 字段，明确禁止 question/answer/options/duration 进入 model input。

phase21_task_context_20260918 +1

你已经验证：

975/975 anchor一致；

prefix hash一致；

source trace hash一致；

checkpoint一致；

差异仅2925个目标字段 + audit。

phase21_pilot_report_20260918

因此“这三个字段的因果效果”这个问题回答得很干净。

逐个看你问的四点：

(a) ratio_of_sums = Σq/Σy：VERIFIED，适合衡量 aggregate scale bias，但不能单独叫 quantile calibration。

真正 quantile calibration 是：

P(Y≤Q
τ
	​

)

你已经同时报告 coverage，因此没问题。

phase21_calibration_analysis_20…

建议术语：

ratio_of_sums → aggregate scale ratio

coverage → quantile calibration

不要把 R=0.397 解读成“p50 coverage 只有39.7%”；这是两个不同量。

(b) steps[:len(truths)]：条件通过。

如果 masked/fixed pack 确实都是 --min-steps 5 的 full5 pack，那么这个写法正确：near-terminal anchor 的 truth 只有2步，就比较前2步。

phase21_calibration_analysis_20…

但 analysis script 没有自己验证 packer 的 min_steps=5。

如果不是 full5，那么你实际上是在：

只对模型愿意输出的长度做 runtime calibration。

这会混入 length prediction。

所以建议 analysis script fail-closed：

manifest.min_steps == H

或者至少输出：

anchors_with_lt_H_emitted_steps

本轮如果你已经从 packer manifest 确认两个包都是 full5，则无需重跑。

(c) step_of：这是 Q1 的唯一真正 P0。

先做顺序一致性 audit。

(d) 128/975 模板缺失：不应该静默，但我认为很可能是预期排除。

你 pilot 实际包含：

640−576=64 runs

而：

128/64=2

恰好每 run 少两个 anchor。

这与目前已知的：

workflow root；

whole-run container；

每 run 各一个完全吻合。

所以我高度怀疑这128个就是预期的两个非 scheduler-node anchor，而不是随机缺失。

INFERENCE，但非常强。

代码仍应改成：

missing_template_anchor:
    expected_root
    expected_container
    unexpected

并要求：

unexpected == 0

不要静默 continue。当前 collect() 确实直接丢弃 template missing。

phase21_calibration_analysis_20…

paired bootstrap 本身方向也对；paired_delta(masked,fixed) 实际算的是 fixed−masked，和报告一致。

phase21_calibration_analysis_20…

P1建议再加三个 assert：

fixed pair keys == masked pair keys
truth_fixed == truth_masked
video_id_fixed == video_id_masked

现在它只是 inner lookup，没有 fail-closed。

Q3. 原来的 calibration gate 合理吗？

判定：INFERENCE — 不适合作为绝对 deployment gate，但不影响本次“修复失败”的结论。

你原来的：

R50∈[0.7,1.3]

以及 coverage 接近：

0.50/0.90/0.95

是参考 J 域内表现定的。

phase21_pilot_report_20260918

问题是 R7 同时改变了：

stack；

model IDs；

workflow population；

truth selection population；

deployment context。

所以不能要求“修三个 metadata token 后，R7 必须恢复到 J validation 水平”。

更合理的 pilot gate应该主要是paired improvement，而不是绝对恢复：

ΔPB
τ
	​

=PB
τ
fixed
	​

−PB
τ
masked
	​


主 gate 我建议改成：

平均 runtime quantile pinball 至少改善10%；

video-cluster CI upper <0；

calibration distance

D=
3
1
	​

τ
∑
	​

∣C
τ
	​

−τ∣

明显减小；

aggregate p50 scale error：

∣R50−1∣

至少减少25%；

role/family 不退化 >2pp。

域内 bands 保留为 reference，不作为 deployment PASS/FAIL。

不过这不会拯救当前 Phase21。

因为你的 p50 paired result是：

+51.59ms, CI=[27.23,74.42]

也就是 fixed 显著更差。p90/p95又没有显著改善。

phase21_pilot_report_20260918

所以即使换成更合理的 gate：

恢复这三个字段没有改善 predictor。

这个负结论仍成立。

16个 video clusters 不算多；作为 pilot 足够，但如果要把 p50恶化作为论文定量结论，附录可增加一个16-video exact sign-flip/permutation test，成本几乎为零。

Q4. 撤回“stack字段被 null 遮蔽”是否正确？

判定：VERIFIED。撤回完全正确。

我独立核了当前 repo 的 j_series_common.py。

解析逻辑确实是：

Python
运行
if field in model_input["task_context"]:
    value = task_context[field]
else:
    value = stack_context[field]

而不是 .get(task_context) or stack_context。

你显式检查得到 task_context 恰好只有七个 task fields，baseline / model_stack_id / planner_model_id 三个键全部不存在。

phase21_pilot_report_20260918

因此：

baseline key absent
→ else
→ stack_context["baseline"]
→ "langgraph_react"
→ in-vocab

这个链条是正确的。

你撤回误报是对的。

但你仍然漏了一个重要点：

planner_model_id OOV ≠ 训练时的 "unknown"

冻结 Vocab 有两个不同概念：

__UNK__    # special OOV bucket
"unknown"  # ordinary categorical token

如果训练集的 planner_model_id 只有字面量：

"unknown"

那么模型训练的是 "unknown" 对应的 embedding。

S_* 输入：

"Qwen3-VL-8B-Instruct"

不在训练 vocab时，会被编码成：

__UNK__

这是另一个 embedding。

而且因为 vocab 是从 train 构造，context 的 __UNK__ embedding 很可能训练中几乎/完全没被更新。

所以报告中的：

“planner_model_id：域内词表只有 unknown → 没有信息损失”

不成立。

你当前实测表已经显示它是 OOV。

phase21_pilot_report_20260918

这不是 parser bug，但它是一个真实的 deployment input mismatch。

同理：

model_stack_id stack_a 命名变体 → __UNK__

temporal_scope="unknown" 如果 "unknown" 本身不在该字段训练 vocab → __UNK__

都需要和“训练时 literal unknown token”严格区分。

当前 j_series_common.py

因此 Q4 的最终答案是：

baseline 正确；撤回正确；但你对 planner_model_id OOV 的影响低估了。

Q5. “三个字段不是原因，且不存在同类缺口”成立吗？

这里必须拆开。

VERIFIED：

恢复 domain / official_task_type / sub_category 没有改善这个16-video pilot的runtime calibration。

这个可以下结论，而且很强。fixed p50 R从0.415降到0.397，p50 pinball显著恶化。

phase21_pilot_report_20260918

不成立：

不存在同类未修的输入缺口。

至少还有三个：

planner_model_id → OOV special __UNK__；

model_stack_id → 大约一半 OOV；

temporal_scope → 100% OOV。

另外历史中的 model_id 也曾有约6% OOV。

所以现在最多能写：

The missing registry-backed task metadata is not the primary cause.

不能写：

input mismatch has been ruled out.

下一步我建议先不做任何新推理

先用现有 Phase21 输出按 model_stack_id 分层看：

R50, coverage, pinball

比较：

stack_b（in-vocab）
vs
stack_a（OOV）

如果两边都约0.4：

stack OOV 很可能不是主因。

如果 stack_b 明显正常、stack_a 极差：

先查 stack canonicalization。

这是零额外GPU成本。

然后做两个非常小的 paired counterfactual：

Experiment 21-B1：planner contract normalization

仅把：

planner_model_id = actual OOV value

改成训练时唯一支持的：

"unknown"

这不是猜类别，而是恢复训练时 feature-availability contract。

Experiment 21-B2：stack canonicalization

只在有明确语义依据时，把：

stack_a_qwen3_vl8b_yolo11x

映射到冻结 vocab 中设计上对应的：

stack_a_qwen3_vl8b

不能仅因为名字相似就做；先确认它们的 stack identity 定义。

还是同16视频、同checkpoint。

成功标准：

ΔPB
mean
	​

≤−10%

且video-cluster CI upper<0，同时 p50 coverage / R50 向目标方向移动。

每个arm按你现有运行速度大约几十秒级，不需要全量。

如果这两个也失败，再把主因归到：

resource-distribution shift；

predicted attributes → resource-head propagation error；

target alignment；

execution-stack differences。

我尤其建议先完成Q1的 truth-order audit。否则现在的“0.4”还有真值配对是否完全正确这一层不确定性。

Q6. 三个代码修复是否正确？
① oov_rate_all KeyError fallback

VERIFIED，正确，无实质副作用。

Python
运行
v.get("oov_rate_all", v.get("oov_rate", 0.0))

只是兼容不同统计块schema。

phase21_anchor_builder_20260918

② required_modalities 单独审计

INFERENCE：解决了 KeyError，但审计实现还有 P1 语义错误。

你正确识别了它不是 vocabs.context，而是 vocabs.modalities。

phase21_anchor_builder_20260918

但是当前：

Python
运行
rendered = "|".join(items)
if rendered not in vocabs.modalities.index:
    oov_all += 1

不等价于真实 encoder。

真实 encoder 是：

Python
运行
for value in required_modalities:
    vocab.encode(value)

即逐 modality token 编码，不是把整个 list join 后查一次。

例如：

["video","audio"]

如果 video 和 audio 都 in-vocab，

你的 audit 会查：

"video|audio"

然后错误报告 OOV。

所以这个 block 应改成“encoder-equivalent audit”：

list → 对每个元素逐项编码；

str → 完全按照冻结 encoder 当前行为迭代字符串；

不要自己定义另一套语义。

另外：

Python
运行
is_missing([])

现在是 False，因为 str([])=="[]"。

但encoder对空list实际不产生任何 modality。

这也应该在audit里单独表示。

当前 pilot因为S_*都是literal "unknown"，这个问题大概率没影响三字段 masked/fixed结论，但该审计不能作为一般正确的 modalities OOV 统计。

③ resolve_artifacts

VERIFIED for this pilot；P1 robustness risk。

兼容：

j_future_h*

b05_future_h*

root

prediction_artifacts/

的确修复了实际路径问题。

phase21_calibration_analysis_20…

但“如果path不存在就找parent”有一个未来风险：

路径拼错时可能静默捡到 parent 中另一个旧artifact。

建议报告中写入：

resolved_fixed_path
resolved_masked_path
sha256
manifest checkpoint_sha
min_steps

并 assert fixed/masked不是同一个文件。

我发现的 P0 / P1

P0-1 — analysis_phase21_taskctx_calibration.py:81–94
truth_sequence 用 (source_step_id, sequence_index) 排序，但冻结目标是raw serial-event order。先做全量顺序等价检查；不等价则改成 sequence_index。

P0-2 — calibration script未验证full5
如果本轮两个 pack 不是 min_steps=5，collect() 会把runtime calibration条件化在模型预测长度上。当前 pilot只有确认manifest为full5后才能正式封存。

P0-3 — 结论层，不是代码崩溃
“不存在同类未修输入缺口”必须撤回。planner_model_id/model_stack_id/temporal_scope 仍有真实 OOV→__UNK__ mismatch。

P1-1 — collect():109–114
128个missing template anchors静默丢弃。应分类并 assert unexpected_missing=0。

P1-2 — paired_delta():168–179
缺少duplicate pair、truth equality、video equality、pair-set equality断言。

P1-3 — builder required_modalities audit 212–246
不是encoder-equivalent，list join OOV算法错误。

P1-4 — resolve_artifacts():43–61
parent fallback可能静默选错artifact；需记录resolved path+SHA。

P1-5 — statistical
16个video cluster适合pilot，但正文若强调+51.6ms恶化，建议补exact video-level sign-flip test。

Phase 21最终论文写法
可以写

We identified that three registry-available task-context fields were absent from scheduler-side predictor inputs. Keeping the J3:seed11 checkpoint fixed, we restored domain, official_task_type, and sub_category and evaluated them against a paired masked control on a pre-registered 16-video pilot.

Restoring these fields did not improve runtime-quantile calibration. The p50 pinball loss increased by 51.6 ms (95% video-cluster bootstrap CI 27.2–74.4 ms), while p90 and p95 changes were inconclusive. 

phase21_pilot_report_20260918

Thus, omission of these three registry-backed fields does not explain the observed deployment-side p50 miscalibration.

这三句我认可。

不可以写

不要写：

metadata missing is not the cause.

范围太大。

不要写：

no remaining input pipeline mismatch exists.

目前明显不成立。

不要写：

planner_model_id carries no lost information because training only contained "unknown".

这是 token-level 上错误的；"unknown" ≠ special __UNK__。

不要写：

error mainly comes from structural mismatch and execution distribution shift.

现在最多是候选解释，还没排除 stack/planner OOV 和 truth ordering。

也不要把 Phase21直接写成：

supports the C2 tail-aggregation mechanism.

Phase21只排除了一个简单解释；它不能正向证明为什么p95有效。

所以我的最终裁决是：

Phase21“三个registry字段修复失败”这一核心结论通过。误报撤回通过。

但在正式封存Phase21前，我要求两个非常小的检查：

truth_sequence 的 step-first order 与 pure sequence_index order 做全pilot/full-R7 ID一致性审计；

确认masked/fixed artifact manifest均为 min_steps=5。

如果这两项都通过，现有pilot无需重跑。

然后不要全量重生成。先用现有输出看 in-vocab stack_b vs OOV stack_a；只有它提示输入OOV可能重要时，再做 planner/stack 的几十秒级paired pilot。
