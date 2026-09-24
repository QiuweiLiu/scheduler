# LLMSched-adapted 实施方案（LLM-1 … LLM-6）与 L1–L7 fidelity gate

来源：GPT 复核（HEAD `9aedf2b` 之后）。L1–L7 建议**原封不动**写成正式 gate。

论文身份（IEEE 已确认）：**LLMSched: Uncertainty-Aware Workload Scheduling for Compound
LLM Applications**, Botao Zhu / Chen Chen / Xiaoyi Fan / Yifei Zhu, **ICDCS 2025**, pp. 527–537。

论文核心：DAG 表示 compound workflow → Bayesian Network 建模 stage 间相关性 →
**已完成 stage 的真实 duration 作为 evidence** 更新未来 posterior →
用 **mutual-information-based uncertainty reduction** 做探索，配 SRTF/JCT exploitation + ε-greedy。
论文 uncertainty-reduction 形式：
`I(X;Y₁..Y_k) × ∏_i Range(Y_i)`

---

## LLM-1：替换「假 BN frontend」，旧实现封存为 legacy oracle

现有 `src/tracing/analysis/llmsched_bn.py` 的正式路径**退役**：

- `P(length | template.baseline)`
- position-wise runtime mean/histogram
- `H(length)`
- `H(runtime_position)`
- `H_now − H_next`

这些量**保留但迁移**到 `llmsched_bn_legacy.py`（或明确命名 `legacy_length_profiler` /
`legacy_entropy_proxy`），仅供回归测试；**正式 `policy="llmsched"` 不得再消费它们**。

frontend 版本号升为：

```
BN_SCHEMA = "llmsched-bn-v2"
```

正式数据流固定为：

```
train traces → canonical stage representation → train-only duration discretizer
→ joint stage table → Bayesian Network + CPDs → online evidence → posterior inference
```

**验收（LLM-1）**：正式 llmsched consumer 中不存在 `posterior_length_probs()`、
不存在 `duration_entropy()`、不存在用 `consumed = len(job.completed)` 作 posterior surrogate。

---

## LLM-2：冻结 canonical stage ontology + duration state

### 2.1 不把 `sequence_index` 当 stage identity

v3.1 是 serial causal chain，但 `sequence_index = 4` 在不同 trace 里未必是同一语义 stage。
新增确定性 mapper：

```
canonical_stage_key(node, prefix_state) -> stage_id
```

基础字段只用**当前已合法可见**的 ontology：`role` / `node_type`、`action_family`、`raw_action`、`lane`。

重复 loop 用 **prefix-only** 的 occurrence index：

```
plan#0  execute:generalist#0  summarize#0
plan#1  execute:generalist#1  summarize#1
```

occurrence **只能由已出现的 prefix 计算，不能看未来 length**。
语义无法稳定对齐的动作使用一个 **frozen dynamic slot**，不得偷用未来 node ID。

### 2.2 不默认用 `tpl.baseline` 作 application family

先删掉 `family = tpl.baseline`。除非能证明 `star` / `langgraph_react` / … 在
**job admission 时已知**、真正代表 application identity、且**不编码未来执行结果**。
第一版最稳：**VideoSeek = 单个 application BN**。

### 2.3 每个 canonical stage 建离散变量

```
X_i ∈ {ABSENT, D_1 … D_K}
```

- `ABSENT` = 该 canonical future stage 本次没有出现
- `D_k` = 出现，且 intrinsic service duration 落入第 k 个 bin

同一个变量同时表达 **structure uncertainty**（ABSENT / PRESENT）与
**duration uncertainty**（D1…DK）。

duration bins：**只用 train、冻结 edge**；validation / test 不允许重新分箱。
`K` 若正文无可靠确认，**明确记为 adaptation hyperparameter**（先固定 小值 4/6，不做搜索）。

### 2.4 composite 的 duration 定义

merged CPU+GPU composite 的 BN stage duration = **frozen node intrinsic `R_total`**。
**不要把 simulator 人工产生的 nested GPU queue delay 放进训练变量**。

---

## LLM-3：从 train-only traces 学真正的 joint BN

每个 train workflow 编成一行：

```
trace_001: X0=D2 X1=D4 X2=D1 X3=ABSENT X4=ABSENT …
trace_002: X0=D1 X1=D3 X2=D5 X3=D2 X4=ABSENT …
```

拟合 `P(X₁…X_n) = ∏_i P(X_i | Pa(X_i))`。

### 3.1 结构约束

数据量不大，**不做完全自由结构搜索**：

- 只允许 **earlier canonical stage → later canonical stage**
- **max indegree = 2～3**
- 禁止反时间边

在 train 上做 constrained structure learning（BIC/MDL 类打分），或先用
**canonical workflow partial order 作 allowed-edge mask**，再在 mask 内选 parents。

必须写明 **deviation**：原论文用 BN 描述 stage-duration dependencies；我们在 VideoSeek
adaptation 中**加了 causal-order edge constraint**，防止有限样本学出 future→past 的不可解释依赖。
**不要声称具体结构学习算法与论文相同**，除非后续从正文确认。

### 3.2 CPD

CPD 必须来自 **train**，并加固定 **smoothing**，避免开发集没出现过的 state 得到零概率。

产物字段：

```
schema, stage_vocab, stage_order, bin_edges, BN_edges, CPDs,
train_sample_count, support_per_stage, support_per_CPD, smoothing, training_split_hash
```

### 3.3 Formal gate（fail-closed）

- 0 validation samples / 0 test samples
- 每个 edge 满足 causal order
- 每个 CPD 每行 sum = 1
- 所有 state 属于 frozen vocabulary
- 不存在未执行的 node_id / template suffix truth 字段

---

## LLM-4：真正的 per-job evidence → posterior engine

**不能再写** `consumed = len(job.completed)`。每个 job 维护：

```python
llmsched_evidence[job_instance_id] = {"plan#0": D2, "execute#0": D4, ...}
```

### 4.1 evidence 只能在完成后产生

唯一合法数据流：

```
node_finish → 该 stage 的 observed intrinsic service duration → canonical_stage_key
→ frozen discretizer → X_i = D_k → job evidence update
```

`ABSENT` **只能在因果上已确定不存在时**才加入；**不能因为 simulator template 知道
future 不存在就提前加入 ABSENT**。

### 4.2 observed duration 的定义（关键）

**不能**用 `finish_ms − node_start_ms`：composite 的这个差值包含 simulator 人工产生的
GPU queue delay。训练 BN 学的是 raw trace **intrinsic** service duration，在线完成后必须
暴露**相同定义**。simulator 已经 role 为 truth_provider，因此合法做法是：

**在 `node_finish` 之后读取该已完成 node 的 intrinsic observed runtime**——
此时它已成为历史 observation，不再是 future leakage。

### 4.3 exact posterior API

```python
posterior_state_probs(profiler, query_stage, evidence)
posterior_joint(profiler, query_stages, evidence)
```

第一版**直接 exact inference**（BN 小、state 数有限），不要提前上 sampling。

---

## LLM-5：真正的 uncertainty reduction + posterior SRTF consumer

取代当前的 `(structural_entropy + duration_entropy) * spread`。

### 5.1 EXPLORE

ready candidate stage `X_j` 的存在性已知，因此 posterior 要**额外条件化** `X_j = PRESENT`
（不是 ABSENT）。设未来 relevant variables `Y = {Y₁…Y_m}`，第一版定义为
**BN 中 X 的尚未完成的 descendants**（与 causal BN 一致，不从 future truth 定义 Y）。

```
I(X;Y|E) = Σ_{x,y} P(x,y|E) log [ P(x,y|E) / (P(x|E) P(y|E)) ]
R_E(X)   = I(X;Y|E) × ∏_{Y_i ∈ Y} Range(Y_i)
```

`Range(Y_i)` 第一版**直接用 train-frozen duration support 的跨度**，
不根据 validation/test 动态重定标。EXPLORE 取 **larger R 更好**。

### 5.2 EXPLOIT

不再用 shared `runtime_p50` + static positional means。由 BN 计算：

```
E[D_remaining|E=e] = Σ_i  P(X_i ≠ ABSENT | E) × E[D_i | X_i ≠ ABSENT, E]
```

当前 ready stage 也由它自己的 posterior duration 得到 expected current service。

**Common substrate cost**：LLMSched BN expected service time **+** common cold-model load surcharge；
memory feasibility / eviction / residency 仍由 frozen substrate 处理。
但论文中没有的 shared `runtime_p50` **不能**重新进入 LLMSched duration ranking。

### 5.3 ε-greedy

保留「一次 scheduling decision 只抽一个 coin」：

```python
mode = EXPLORE if rng.random() < epsilon else EXPLOIT
```

整个 candidate pool 共用该 mode。EXPLORE → `max R_E(X)`；EXPLOIT → `min` posterior
expected remaining service。

论文的 **intra-stage `r-sampling`**（先执行一部分任务以降低不确定性）因为我们的节点是
不可再分割的执行单位，继续明确记录为 **adaptation omission**，不需要制造 fake partial stage。

### 5.4 decision trace 必须留证据链

每条 LLMSched decision 记录：

```
decision_index, job_id, node_id, canonical_stage, evidence_hash, evidence_count, mode,
candidate posterior E[current], candidate posterior E[remaining], candidate CMI,
candidate sum_range, candidate uncertainty_reduction, cold_load_surcharge,
final_primary_score, chosen
```

以后这就是论文 baseline fidelity 的直接证据。

---

## LLM-6：L1–L7 fidelity gate + freeze LLMSched

### L1 — Hand-computable posterior
极小 BN `A→B→C`，手工指定 CPD。给定 `A=a1`，手算 `P(B|A=a1)`、`P(C|A=a1)`；
断言代码 posterior == 手工 posterior 且 `sum(probabilities) == 1`。
再加入 evidence `B=b2`，断言 posterior 再次按手算变化。**PASS 条件：数值一致。**

### L2 — Same entropy, different MI（杀掉旧 entropy proxy 的核心 sentinel）
候选 A：`X_A ~ Bernoulli(.5)`, `Y_A = X_A` → `H(X_A)=1`, `I(X_A;Y_A)=1`。
候选 B：`X_B ~ Bernoulli(.5)`, `Y_B ~ Bernoulli(.5)` 独立 → `H(X_B)=1`, `I(X_B;Y_B)=0`。
两边**相同 marginal entropy、相同 Range(Y)**。
断言 `R(A) > R(B)` 且 **EXPLORE chooses A**。
如果 `duration_entropy()` 还能通过这条测试，测试就写错了。

### L3 — Completed-duration evidence mutation
已完成 stage `Z` + 当前 ready `A`/`B` + future `Y`；BN CPD 设计成 `Z=D1 → A 更好`、
`Z=D2 → B 更好`。**保持**同一 ready set / GPU state / topology / 同一 future truth，
**只改变已完成 `Z` 的 observed duration bin**。
断言至少：posterior changes、`expected_remaining` 与 CMI changes、最终 A/B winner flips。
最好固定 `epsilon=0` 与 `epsilon=1` 分别做确定性 exploit / explore 测试，避开 RNG。

### L4 — Future-truth invariance（leakage gate）
同一 scheduling state（相同 current prefix / completed evidence / ready nodes / GPU state），
**只修改尚未执行的 future truth**：future actual runtime、future presence/absence、
future model/resource、future node IDs。
断言当前 decision 下 `evidence`、`posterior`、`CMI`、`expected_remaining`、
`chosen action` **全部 identical**。
即：未执行 future truth 无论被改成什么，当前 LLMSched 看不到。

### L5 — Same marginals, different joint correlation
两个 profiler P / Q，**所有 X/Y marginal probabilities 一致、所有 duration means 一致、
所有 ranges 一致**；唯一变化是 **P 中 X 与 Y 强相关，Q 中独立/弱相关**。
断言 `CMI_P != CMI_Q`、`uncertainty_reduction` changes、EXPLORE ordering flips 或显著变化。
若 winner 完全由 marginals 决定，这条会失败。

### L6 — Structural uncertainty / ABSENT state
构造 `X → Y`，其中 `Y ∈ {ABSENT} ∪ duration bins`。
例如 `E=e1 → P(Y=ABSENT)=0.8`、`E=e2 → P(Y=ABSENT)=0.1`。
断言 posterior absence probability 正确变化、`expected_remaining` 在 `e1` 下更小、
**ABSENT stage 不贡献 duration**。
**再加一条**：当前 ready stage `X` 必须条件化 `X != ABSENT`——
scheduler 不能一边看到 `X` 已 ready、一边 posterior 还给它 30%「不存在」。

### L7 — Full E2E consumer sentinel（最终签字项）
必须走真实 `simulate_episode()`，完整路径：

```
node_finish → observed intrinsic service duration → canonical stage mapping
→ frozen binning → per-job evidence update → BN posterior → CMI / expected remaining
→ epsilon mode → llmsched_score → choose_action → actual node_start
```

构造两个 episode，**只修改前面已完成节点的 observed duration bin**
（Episode A: `Z=D1`；Episode B: `Z=D2`），断言 decision trace 的 evidence 不同、
posterior 不同、score 不同、**实际 first chosen node 不同**。

同时断言 formal consumer **没有调用** legacy `posterior_length_probs` /
`duration_entropy` / `info_gain`。这条一过才能说：
**LLMSched 的 BN 不只是单元函数存在，而是真正控制了实际 dispatch。**

### L1–L7 之外再保留 5 个普通 unit gate

`train-only`、unknown canonical stage `fail-closed`、missing CPD/state `fail-closed`、
one epsilon coin per decision、fixed seed → deterministic decisions。

### artifact manifest 必须明确写

```
paper:
  LLMSched, ICDCS 2025
faithful:
  DAG uncertainty
  BN joint dependencies
  completed-duration evidence
  posterior update
  MI × Range uncertainty reduction
  ε-greedy uncertainty / SRTF
adapted:
  VideoSeek canonical stage ontology
  causal-order-constrained BN learning
  GPU/node execution unit
  no intra-stage sample_tasks(r)
  common frozen GPU lifecycle / admission substrate
```

---

## 开工顺序

1. **LLM-1** — 封存旧 frontend，formal path 断开 `entropy` / `length` profiler
2. LLM-2 canonical ontology + duration state
3. LLM-3 joint BN + fail-closed builder
4. LLM-4 per-job evidence + exact posterior
5. LLM-5 uncertainty reduction + posterior SRTF consumer
6. LLM-6 L1–L7 gate
