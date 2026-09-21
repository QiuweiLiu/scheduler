# GPT 对因果边界澄清的裁决 + Stage 0 实施规格

- 日期：2026-09-21
- 请求：`.scratch/gpt_causal_boundary_clarification.md`
- 项目负责人的边界（逐字）：
  > 历史信息（包括属性和运行用时）都可以用；当前节点的类别可以用；
  > **但是耗时不可以用**；未来节点的所有信息**绝对不可以用**

## 唯一因果契约（GPT 的最终表述）

```
i <  anchor : attributes + observed resource outcomes
i == anchor : attributes only
i >  anchor : nothing
```
并且：**「身份字段可以用于证明这条边界，但身份本身不进入模型。」**

## 冻结的字段边界

| 信息 | 历史 `< anchor` | 当前 `== anchor` | 未来 `> anchor` |
|---|---|---|---|
| event_type / node_type / role / raw_action / action_family / model_id | ✅ | ✅ | ❌ |
| runtime_ms | ✅ | ❌ | ❌ |
| load_ms | ✅ | ❌ | ❌ |
| peak_allocated_mb / peak_reserved_mb | ✅ | ❌ | ❌ |
| status_class | ✅ | ❌ | ❌ |
| is_retry | ✅ | **条件允许**（本版不开放） | ❌ |
| merged_nested_call | ✅ | **禁止**（它是执行结果，不是类别） | ❌ |
| nested-call 属性 | ✅ | ❌ | ❌ |
| nested-call runtime | 原理上可用，**本版不使用**（避免与 outer composite runtime 重复计） | ❌ | ❌ |
| node_id / event_id / run_id / video_id | **仅 JOIN/AUDIT** | **仅 JOIN/AUDIT** | ❌ |
| successors / predecessors / remaining / future labels | ❌ | ❌ | ❌ |

- **`is_retry` 的当前节点**：`node_table.is_retry` 来自 trace 的 `retry_of`，属执行结果 → 第一版**不开放**
- **`node_id` 等高基数身份**：只用于 JOIN，**不进入模型**（防止模型记模板而非学规律）

## 数值表示（冻结）

`u = log(1 + max(x, 0))` → **仅 train split** 估计 `median` 与 `IQR/1.349` →
`z = clip((u − median)/max(scale, eps), −6, 6)` → **每个字段独立 presence mask**

- **不做** per-model / per-family 归一化（会把"Qwen 历史 8 s、metadata 0.1 ms"这种**绝对信息抹掉**）
- **不做**分桶作为主版本（16-bin 是**输出**分布的接口，不代表**输入**也要离散化）
- **不做** K-step 手工聚合（GRU 已经看到完整历史序列；先验证原始观测有没有增益）
- `status_class` 用 **categorical embedding**，不转数值；当前节点固定为 `UNOBSERVED`

## join 规则（解决了我发现的 92.8% 位置错配）

**放弃 `history[i] → chain_position i−1`**（position 不是可靠主键，只能作 audit signal）。

**第一层 = identity join**：每个 anchor 自己告诉我们 `(run_id, current_event_index) → current_node_id`；
对历史 token `i`，若 `(run_id, i)` 有唯一身份 → 直接 `node_table[(run_id, node_id)]`。
（`current_event_index = len(history) − 1`，因为 history token 的 `position` 就是 `enumerate()` 的序号。）

**第二层 = monotonic categorical alignment**（仅在需要时）：以已解析的 identity 为固定锚点，
在锚点之间做单调序列对齐，签名 `(node_type, role, action_family, model)`；允许两侧 skip，**不许 reorder**；
**只有唯一匹配才接受，否则标记 missing，绝不猜**。

**join provenance 必须落盘**（模型看不到）：
`identity_exact` / `monotonic_unique` / `unresolved` / `current_forbidden` / `run_container`

**GPT 最保守偏好**：主实验**只用 `identity_exact`**；`monotonic_unique` 只做 coverage 诊断。

## 覆盖率 gate

**硬 gate（无条件 100%）**：`current resource exposed = 0`、`future resource exposed = 0`、
`ambiguous joins accepted = 0`、`duplicate identity joins = 0`、`split mismatch = 0`

**历史资源 coverage**：预注册 **≥ 90%**，且跨 split `max − min ≤ 5 pp`。
若 identity-only < 90% → 启用 monotonic_unique 后重新报告；仍 < 90% 则标为
**partial historical telemetry coverage**，**绝不偷偷补齐**。

## 泄漏守卫（必须同时检查身份与时间）

```python
assert token_event_index < current_event_index
assert token_node_id != current_node_id
assert joined_row["node_id"] != current_node_id
```
**不得**以 `chain_position < anchor_position` 作为因果证明（那正是已被证明会错位的地方）。

## 三个变异测试（GPT 的验收核心）

1. **current mutation**：把 node_table 里当前节点的 runtime/load/peak_allocated/peak_reserved/status
   改成极端哨兵（如 `9.87654321e11` / `__CURRENT_LEAK_SENTINEL__`）→ **model_input 必须逐字节相同**
2. **future mutation**：改任意 future 行 → **逐字节相同**
3. **past mutation**：改一个 identity_exact 的历史节点 → **只有对应 historical 通道变化**

---

==========================================================================================
负责人给出的边界很清楚，而且我认为你整理的主表基本忠实。这次最重要的变化是：不需要再争论“resource information 能不能进模型”，只需要确保它严格满足

past outcome 可见；current identity 可见；current outcome 与 future 全部不可见
	​


我重新核了现有 builder 和 R0 node_table 的生成逻辑。下面我会把规则收紧到可以直接编码的程度。

Q1. 字段边界核验

你的表总体正确。我建议正式冻结成下面这一版：

信息	历史 < anchor	当前 == anchor	未来 > anchor
event_type/node_type/role/raw_action/action_family/model_id	✅	✅	❌
runtime_ms	✅	❌	❌
load_ms	✅	❌	❌
peak_allocated_mb	✅	❌	❌
peak_reserved_mb	✅	❌	❌
status_class	✅	❌	❌
is_retry	✅	条件允许	❌
merged_nested_call	✅	禁止	❌
nested-call 属性	✅，若历史已发生	❌	❌
nested-call runtime	原理上历史可用；本版不使用	❌	❌
node_id/event_id/run_id/video_id	仅 JOIN/AUDIT	仅 JOIN/AUDIT	❌
successors / predecessors / remaining / future labels	❌	❌	❌

有几个边界值得单独解释。

is_retry

历史节点当然可以用。

当前节点只有在 scheduler/agent 派发前就明确知道“这是一个 retry node” 时才能用。现有 node_table.is_retry 来自 trace 的 retry_of，因此我不会仅凭 node_table 就把 current is_retry 放进模型。

第一版最稳：

历史 is_retry 可以；current is_retry 暂不新增。

等以后证明 ready-node descriptor 本身就带这个字段，再开放。

merged_nested_call

历史可以，因为执行已经结束。

当前不能。现有 merged_nested_call 是 build_chain() 通过观察实际上是否出现 nested event 后构造出来的，属于执行结果。

所以：

current merged_nested_call 是 outcome，不是 category
	​


应该禁止。

nested_calls 的历史耗时能不能用？

这里分“因果原则”和“本次实现”。

因果原则：可以

如果一个 nested call 已经在 anchor 前完整执行完：

t
nested,end
	​

<t
anchor
	​


那么它的 runtime 属于历史已观测结果，符合负责人定义。

但本次我建议：不要使用 nested-call 独立 runtime

原因不是因果问题，而是资源语义问题。

我重新看了 R0 node table builder：

node_table 的一行对应 build_chain() 合并后的 top-level schedulable node；其中：

merged_nested_call

nested_model_class

会保存；

但资源字段：

runtime_ms
load_ms
peak_allocated_mb
peak_reserved_mb

来自这个 outer event 的 resource。

并没有独立保存 nested child's runtime。

而现有 J-series contract 明确把 merged nested call 作为 composite node。

所以如果未来从其他地方再拿 nested child runtime，又同时保留 outer runtime，很容易：

double count
	​


本轮最好冻结：

历史 merged node 使用 node_table 的 composite runtime；nested 子调用只使用已经存在的 categorical descriptor，不单独加入 nested runtime。

如果以后拿回 raw trace，可以另开一个 nested-resource ablation，但别混进这次主实验。

node_id 能不能用？

项目负责人说“历史属性都可以用”并不意味着应该把 ID 当模型特征。

我强烈建议：

node_id只用于 JOIN，不进入模型
	​


同样适用于：

event_id
run_id
video_id

原因很简单：它们是高基数身份标识，容易让模型记模板/样本，而不是学习规律。

现有 builder 把这些列入 forbidden keys 是合理的，继续保持。

Q2. 历史 runtime 应该怎么表示？

我的推荐现在很明确：

真实数值保留→log(1+x)→train−onlyglobalrobustnormalization
	​


不要分桶作为主版本，也不要第一版做 K-step 手工统计。

为什么不直接 raw milliseconds？

虽然 raw ms 是最“原始”的观测，但不等于最好的神经网络表示。

你现在真实数据跨度类似：

0.1 ms metadata
几百 ms
几秒
10+ 秒 LLM

至少跨 5 个数量级。

如果直接把：

0.1, 7336, 24254

扔进一个小 MLP/GRU：

大值会主导梯度；

极端 tail 对表示影响很大；

模型更容易记绝对机器速度；

以后硬件/负载稍有变化，分布漂移风险更大。

所以我不会直接 raw-ms。

推荐变换

对每个连续字段分别：

u=log(1+max(x,0)).

然后只在 train split 上估计：

m=median(u)

和

s=
1.349
IQR(u)
	​

.

输入：

z=clip(
max(s,ϵ)
u−m
	​

,−6,6).

并且每个值必须有独立 presence mask：

runtime_z
runtime_present

load_z
load_present

peak_alloc_z
peak_alloc_present

peak_reserved_z
peak_reserved_present

所以：

真实的 0 ms

与：

没有测量

绝对不会混淆。

status

不要转 numeric。

直接：

hist_status_embedding

如：

success
failed
timeout
unknown
...

只允许历史 token。

current 固定：

UNOBSERVED
要不要“按 model/action group 中位数归一化”？

主实验不要。

例如：

runtime/median(runtime∣model,family)

虽然可以消除不同 model 的速度尺度，但恰恰可能把你想给模型的信息抹掉。

假如：

Qwen historical runtime = 8s
metadata = 0.1ms

这是非常有价值的绝对信息。

而模型已经知道：

model_id
action_family
role

让 GRU 自己学习：

某个 model 正常大概多快，以及这个 sample 最近偏快还是偏慢，

比我们事先把组间差异消掉更合理。

因此第一版本：

global per-field normalization
	​


不是：

per-model / per-family normalization.
分桶呢？

作为 neural input，我也不推荐作为主版本。

分桶会丢掉：

2.1svs7.8s

这种连续信号。

16-bin 是 resource output distribution 的合理接口；并不意味着 history input 也要 discretize。

“上 K 步同类动作均值/中位数”呢？

第一阶段不做。

GRU 已经看到完整历史 sequence：

(x
1
	​

,r
1
	​

),(x
2
	​

,r
2
	​

),...

如果另外再给：

last_same_model_mean
last_3_runtime
same_family_median

会引入一批研究者设计特征，自由度迅速变大。

先验证：

原始历史观测通过简单数值编码有没有信息增益。

如果 F1 有收益，再做 aggregation ablation。

因此建议 numeric token 最终就是

例如历史 planner：

categorical:
  event_type
  node_type
  role
  action_family
  model_id

numeric:
  runtime_z
  load_z
  peak_alloc_z
  peak_reserved_z

masks:
  runtime_present
  load_present
  peak_alloc_present
  peak_reserved_present

categorical outcome:
  status_class

输入 model 时：

e
t
	​

=e
t
cat
	​

+e
t
resource
	​

+e
t
status
	​

+e
t
position
	​

.

当前 token：

categorical = 正常值
resource_present = 0,0,0,0
resource_z = 0,0,0,0
status = UNOBSERVED

这非常干净。

Q3. 10.9% 位置错配怎么处理？

这里我不建议继续：

history[i]→chain_position=i−1.

你发现的 89.1% 已经足以证明：

position不是可靠主键
	​


它最多作为 audit signal。

第一优先级：identity join

现有 topology feature 行本身有：

run_id
current_event_index
current_node_id

这一点非常有用。

对于同一个 run，可以建立：

(run_id,event_index)→current_node_id.

因为每个 anchor 自己告诉我们：

raw history 的这个 event index 对应哪个 event/node identity。

所以第一层 join 应该：

Python
运行
anchor_identity[(run_id, current_event_index)] = current_node_id

如果同一个 (run,index) 出现两个不同 node_id：

FAIL

然后对一个 sample 的历史 token i：

i == 0

run container：

resource_present = 0
i == current_event_index

明确 current：

resource_present = 0

即使 node_table 能查到它，也绝不读取资源字段。

i < current_event_index

如果：

(run_id, i)

能得到唯一 event/node identity，则：

Python
运行
node_table[(run_id, node_id)]

直接 identity join。

这是最高可信级别。

为什么 identity join 比 chain_position 好？

R0 的 node_table 是：

Python
运行
for position, node_id in enumerate(chain["nodes"])

生成。

而 J feature history 来自：

Python
运行
events[:event_index+1]

也就是说一个是：

verified merged serial chain

另一个是：

raw supported-event prefix

两者在 nested merge / 特殊 retry 情况下并没有“一格对一格”的保证。

所以你发现 10.9% mismatch 是很合理的。

第二层：对于没有 anchor identity 的历史 token

这里我建议不要直接 fallback 回 position。

可以做一个严格的 monotonic categorical alignment。

对每个 run：

左边是 history event sequence，已有：

event_type
node_type
role
raw_action
action_family
model_id

右边是 node_table sequence，已有：

exec_class
role
action_family
model_class
chain_position
is_retry
merged_nested_call

构造 signature：

S
h
	​

=(nodeType,role,actionFamily,model)
S
n
	​

=(nodeType,role,actionFamily,model).

使用已解析的 identity anchors 作为固定锚点。

在两个固定 anchor 之间做单调 sequence alignment：

exact signature match 优先；

history 允许 skip（例如 merged nested child）；

node rows 也允许 skip；

不允许 reorder。

只有当一个 token 的最优匹配是唯一的时才接受。

如果两个 planner：

planner / Qwen / plan
planner / Qwen / plan

无法唯一判断是谁：

mark missing，不猜
	​

我建议 join provenance 一起写入 token
resource_join_status:
  identity_exact
  monotonic_unique
  unresolved
  current_forbidden
  run_container

模型本身不需要看到这个字符串。

但 dataset audit 必须保存。

一个更保守、我更偏好的第一版

事实上，因为你们数据不大，我甚至建议：

正式主实验只使用 identity_exact 的历史 resource。

monotonic_unique 先生成，但只做 coverage diagnostic，不进入 F1。

原因是这一轮最重要的是：

零泄漏

而不是榨干最后 5–10% feature coverage。

如果 identity-only coverage 已经高，就完全没有必要冒对齐错误风险。

覆盖率门槛怎么定？

我建议分成正确性 gate 与 viability gate。

硬 gate：无条件 100%

必须：

current resource exposed = 0
future resource exposed = 0
ambiguous joins accepted = 0
duplicate identity joins = 0
split mismatch = 0

任何一个不为 0：

FAIL
	​

历史资源 coverage

不是要求 100%。

建议预注册：

≥90%
	​


的eligible historical top-level resource tokens有可靠 resource join。

并且各 split 间：

maxCoverage−minCoverage≤5 percentage points.

为什么 90% 而不是 95%？

因为你已经知道 raw-event / merged-chain 语义天然不完全相同，而且当前 node-id cross-check 大约 89.1%。设 95% 很可能迫使你写危险 fallback。

更重要的不是某个神奇百分比，而是：

unresolved 必须变 missing，绝对不能猜。

如果 identity-only coverage <90%，我不会为了过 gate 用 position 强行补。我会启用 monotonic_unique，然后重新报告 coverage。

如果仍 <90%，这个实验仍然可以做，但应标为：

partial historical telemetry coverage

而不是偷偷补齐。

泄漏守卫要比 join gate 更强

你已经指出：

当前节点真值在 node_table 中确实存在。

所以仅靠：

Python
运行
if history_index < anchor_position

还不够。

最终必须同时检查身份与时间关系。

对每个 resource-bearing token：

Python
运行
assert token_event_index < current_event_index
assert token_node_id != current_node_id

如果有 chain position：

Python
运行
assert joined_row["node_id"] != current_node_id

但不要仅以：

chain_position < anchor_position

作为因果证明。

因为这正是已经被证明会错位的地方。

最强 sentinel test

我继续推荐这个自动测试。

随机抽每个 split 至少 100 个 sample。

把 node_table 中current node：

runtime_ms
load_ms
peak_allocated_mb
peak_reserved_mb
status_class

全部改成极端 sentinel：

runtime = 9.87654321e11
load = 8.7654321e11
status = "__CURRENT_LEAK_SENTINEL__"

重新构造 feature。

要求：

modelInputBefore=modelInputAfter
	​


byte-identical。

再修改一个 identity-exact 的 historical node：

要求：

只有对应 historical numeric/status channel 改变。

最后修改任意 future row：

要求 model input byte-identical。

这样就把负责人的三条边界自动编码成测试了。

关于 node_table 本身还有一个好消息

我重新看了 r0_oracle_signature_ceiling.py。

node_table 的确是通过：

run_id
node_id
chain_position

明确落盘，而且 node resource 来自对应 event。

所以它非常适合作为：

historical resource sidecar
	​


但不应该拿 chain_position 当唯一 join key。

正确角色是：

identity/resource truth store。

Q4. 上一轮其余规划需要怎么调整？

大方向不变，但有四个重要调整。

调整 1：不再改 raw topology builder

你已经确认：

data/raw = empty

而且所需信息已经存在于：

j_series_dataset_v1
topology_predictor_p9d_v3
node_table

所以不要让 Codex 去恢复远端 raw trace。

新建一个独立 augment stage，比如：

scripts/build_j_series_history_resource_v2.py

输入：

features_{split}.jsonl.gz
labels_{split}.jsonl.gz
node_table.jsonl.gz

输出：

results/processed/j_series_dataset_histres_v2/

或 historical-resource sidecar。

这样旧数据完全不动。

调整 2：F0 / F1 对照更加重要

负责人已经明确：

历史资源允许用。

因此真正研究问题现在就是：

这些额外合法信息到底有没有增益？
	​


所以仍然必须：

F0

新架构、全模型重训，但是 historical-resource channel 全 mask。

F1

完全同架构、初始化、训练协议，historical-resource channel 正常开启。

真正归因：

F1−F0.

J3 继续作为 frozen historical baseline。

不要只跑：

F1vsJ3.

否则没法区分：

输入信息收益

和：

重训/新架构收益。

调整 3：训练目标暂时仍不加 L_decision

这一点不变。

这次最重要的是先验证：

past resource telemetry 是否增加预测可辨识性。

因此仍建议：

L=L
S
	​

+L
C
	​

+L
B
	​

+L
R
	​

.

不要同时改 decision objective。

Risk/CP 仍然等 predictor artifact 冻结后再接。

调整 4：Risk / CP / Truth-Q 规划基本不变

前一轮：

Risk-v1 no future
Risk-v2 predicted future
Risk-v2 truth-H5

继续。

CP 统一 future coefficient 的：

2×2

继续：

	Predicted future	Truth-H5
Greedy	G-P	G-T
CP	CP-P	CP-T

Truth-Q sampled counterfactual 继续。

唯一变化只是：

Predicted future 改用本次 F0/F1 选出的冻结 predictor artifact。

我建议你现在让 Codex 先只实现 Stage 0

不要马上训练。

第一提交最好只包含：

historical-resource augmentation
identity join
causal guards
scaler
dataset audit
tests

不碰 model。

我审核时重点看六个数字：

current_truth_exposure_count = 0
future_truth_exposure_count = 0
ambiguous_join_accepted_count = 0
identity_exact_coverage
monotonic_unique_coverage
unresolved_coverage

以及三个 mutation tests：

current mutation → input unchanged
future mutation  → input unchanged
past mutation    → expected historical channel changes

这批通过后再做 model branch，风险最低。

最终可以把负责人的契约写成一句不可歧义的程序规则：

⎩
⎨
⎧
	​

i<anchor
i=anchor
i>anchor
	​

:attributes+observed resource outcomes
:attributes only
:nothing
	​

	​


并再补一句：

身份字段可以用于证明这条边界，但身份本身不进入模型。

这就是我建议后面所有 dataset、模型、测试都围绕的唯一因果契约。
