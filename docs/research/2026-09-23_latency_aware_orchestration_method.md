# Latency-Aware Orchestration — 论文算法摘要（实现依据）

来源：Jinghao Wang, Yifeng Zhang, Xiao Zhou, Yao Lu, Yihui Zhang, Xiaoyang Sun,
Tianyu Wo, Xu Wang, Chunming Hu, Renyu Yang,
*Latency-Aware Orchestration for Multi-Agent LLM Workflows on Heterogeneous GPUs*,
**arXiv:2609.03335v1 [cs.DC], 2026-09-03, 13 pages**（venue UNVERIFIED，arXiv preprint）。
Beihang University + University of Leeds。

以下为**逐式抄录**，作为 faithful adaptation 的实现依据。凡我们无法忠实迁移的部分，
在同一处标注 `[NOT MIGRATED]` 及理由。

---

## 0. 动机数字（论文自己的测量，可用于对比）

- **Figure 1**：模型加载消耗的 GPU 时间**多于 token 生成**；idle residency 占 28.1–33.0%；
  可在请求就绪前就开始的加载占**总 GPU 时间的 10.5–11.5%**。
- **Figure 2**：同模型节点边界的重复 acquire 等待 —— **中位数只占 2–3%，但 p95 占 39–42%**。
- **QMSum 场景**：两条并行三节点 Qwen3-4B 链 + 三个 Qwen3-8B 聚合节点；
  **9 次调用中 6 次有同模型后继**。
- **4.5 消融**：限制 cross-workflow lifecycle coordination 会显著增加 makespan；
  **关闭 fusion 会显著影响 p95 和 makespan**。

---

## 1. 执行模型（3.2）

- 逻辑图 `L_t^w = <V_t^w, E_t^w>`：每个 activation 是一次独立调用。
- **ready（`F_t`）**：所有前驱完成。
- **near-ready（`N_t`）**：所有前驱完成**或正在运行**，且至少一个仍在运行。
- 规划窗口 `W_t = G_t^0[F_t ∪ N_t]`（fusion 归一化之后）。
- 物理实现 `G_t^p = <U_t^0 ∪ H_t^p, E_t^{L,p}>`，`H_t^p` 是 device-symbolic 的
  **loading / prefetching / reclamation** 动作；边保证生命周期动作排在其消费者之前。
- **deployment identity `d`** = (model + version + serving configuration)。
  同 identity 的 unit 可复用 replica，但保留各自 request state 与输出。
- **execution grant**：把一个 unit 准入到一个 replica；其 lease 在该 unit 完成前阻止该
  replica 被回收。
- 只 **commit 到下一个执行边界**，未提交的 suffix 可随事件修订。

## 2. Predictor（3.3）

**Eq (1)**
```
(T̂^run_{m,r,g}, M̂^peak_{m,r,g}, T̂^load_{d,g}) = Predict(G_m, r, h_g)
```
- `G_m` = 模型表示为 operator data-flow graph（算子类型/算力量/参数量/tensor shape）
- `r` = request 配置（serving phase、batch size、序列长度）
- `h_g` = device 特征（算力、显存容量、带宽）
- 三个头分别出 run / peak memory / load

**缓存剖面**：`C_κ`，`κ = (d, g, b, ℓ_in, ℓ_out)`；另有 `C^load_{d,g}`。
- 计时用覆盖"近期同逻辑节点完成输出的**经验 p90 输出长度**"的最小输出 bucket，
  无历史则回落到配置上限。
- **admission 一律用覆盖"配置的输出上限"的 bucket。**

**admission 预留显存**
```
M^adm_κ(t) = M̂^peak_κ + ε_M + ρ_κ(t)
```
`ε_M` 覆盖未建模波动；`ρ_κ(t)` 在观察到 OOM 后增长（历史只能细化完成估计，
**不能减少为可行性预留的显存**）。

**Eq (2) 预测 release**
```
r̂_u = t                                  if u ∈ F_t
    = max_{v ∈ pred_{G⁰_t}(u)} ĉ_v        if u ∈ N_t
```

`[NOT MIGRATED]` graph encoder 本身（算子图 + 设备特征）。替换为本项目的 F0 预测器；
**论文明确说调度路径上用的是 cached lookup 而非现跑 graph encoder**，所以用缓存的
per-(model, device) 剖面是忠实等价的做法。

## 3. Constructor（3.4）

**Eq (3) 可融合链条件** —— 边 `(u,v)` 属于某个极大可融合链当且仅当：
```
u, v ∈ V_A,
succ_{L_t}(u) = {v},  pred_{L_t}(v) = {u},     ← 一一对应，把 branch/join 留在链边界外
d(u) = d(v),
θ(u) ~_cfg θ(v)                                 ← 配置兼容（到 output-length 上限）
```
注：`V_A` 是 Agent 节点；`d(·)` 是 deployment identity；`~_cfg` 是配置兼容。

对每条极大链 `c = (v_1,…,v_h)` 收缩得到 `G⁰_t = Fuse(L_t)`：
- 外部依赖挂到链的**端点**
- `v_1 ≺ … ≺ v_h` 在**一次 grant + 一份 replica lease** 下执行
- 每个 activation **保留自己的 request 配置**，并把输出传给下一个（数据流不变）

**Eq (4) 融合后的准入边界与时长**
```
Ī_i = P_i + Σ_{j ∈ A_i} O_j                    ← 输入包络：固定 prompt + 消费的先前输出
I^adm_c = max_{1≤i≤h} Ī_i
O^adm_c = max_{1≤i≤h} O_i
T̂^run_{c,g} = Σ_{i=1}^{h} T̂^run_{m(c), r_i, g}  ← 时长是各 activation 之和
```
长度为 h 的链**用一次 grant，去掉 h−1 个中间调度边界**。

**Eq (5) 生命周期备选**
```
H^X_t(u) = { ⟨α_X(d(u))⟩, ⟨Reclaim, α_X(d(u))⟩ }
```
其中 `α_F = Load`（ready work），`α_N = Prefetch`（near-ready work）。
生命周期动作**在被调度前不绑定位置**；reclaim 选一个 **lease-free** 的 victim。

## 4. Scheduler（3.5）

**Eq (7) 起止时间**（同时受逻辑前驱与资源顺序前驱约束）
```
ŝ^{G,π}_s = max{ r̂_s, max_{v ∈ pred_G(s)} ĉ^{G,π}_v, max_{v ∈ rpred_π(s)} ĉ^{G,π}_v }
ĉ^{G,π}_s = ŝ^{G,π}_s + T̂_{s, a_π(s)}
```

**Eq (8) 可行设备域**
```
D_t(u,ν) = { a ∈ D | I_u + O_u ≤ L^max_{d(u)}  且  M^adm_{κ(u,g(a),ν)}(t) ≤ C^M_a }
```
这**只是第一道可行性剪枝** —— 融合单元可能过上下文限制，也可能在小 GPU 上超显存；
反过来，设备可以进入域内即使 deployment 未驻留。

**Eq (9) 图–调度一致性**：`Σ_a x_{s,a} = y_s`；loading/prefetch 前驱与消费者同设备
（`x_{h,a} = x_{u,a}`）；`Σ_q β_{u,q} + λ_u(G) = y_u`（每个 unit 要么绑到接受它的 replica，
要么有 loading/prefetch 前驱）；`β_{u,q} ≤ x_{u,a(q)}`。reclaim 同理绑一个 lease-free victim。

**Eq (10) 资源冲突排序**：同一非并发资源上的操作对 `(i,j,a)` 由一个二元方向变量定向，
`z_{ij,a} + z_{ji,a} = 1`，并要求先后关系；合成后的前驱关系保持无环。

**Eq (11) 累积显存可行性**（在每个预测 transition boundary `τ` 检查）
```
R^{G,π}_{a,t}(τ) + Σ_s x_{s,a} M^inc_{s,a} 1[ŝ_s ≤ τ < ĉ_s] ≤ C^M_a     ∀a, τ
```
并发绑到 replica `q` 的 unit 还要满足 batch 上限 `B_q`。

**贪心策略文字（关键）**
- 先**利用不需要再加载就能服务 ready work 的容量**：驻留且接受的 replica 立即复用。
- 需要加载时，其价值随**它能解锁的 ready work**（相对预测加载时间）增长，
  **并计入 priority 与 aging**。
- **scale-out 只在"新 replica 能比当前队列更早完成该 unit"时考虑**，且受 replica 上限约束。
- **reclaim**：只考虑 idle、lease-free 的 replica，**优先冗余副本**，其次是
  **下次使用距离远**（相对于 reload 成本）；缺估计时用 LRU。
- **prefetch**：near-ready 的 deployment 在"剩余 release 时间接近 loading 时间 + dispatch slack"
  时才有资格，**且不得延迟选中计划里的 ready work**。

**Eq (12) 字典序排序键**
```
Φ_t(G, π) = ( -A_1, …, -A_K,  Ĉ_F,  -P_N,  H_life )
```
- `A_k` = 被准入的 level-k unit 数（按 priority class）
- `Ĉ_F` = ready unit 的**最晚完成时间**
- `P_N` = 及时 prefetch 数
- `H_life` = ready-work 加载延迟 + reclaim 引起的 reload 成本
- **顺序：先按 ready-work 覆盖（跨优先级类），再按完成时间，最后按及时 prefetch 与生命周期开销**

**Algorithm 1**
```
1:  R ← RankReady(F_t, S_t)                 ⊳ priority 与 aging
2:  Γ_t, L, B ← ∅
3:  for u ∈ R do
4:      b_u ← BestFeasibleBinding(u, S_t, C)   ⊳ 一个绑定
5:      if b_u 复用驻留且接受的 replica then Γ_t ← Γ_t ∪ {(u, b_u)}
7:      else if b_u ≠ ⊥               then L   ← L ∪ {(u, b_u)}
9:      else                              B   ← B ∪ {u}
11: A_t ← PlanReady(L, B, S_t, C)            ⊳ load / scale-out / reclaim
12: if Γ_t = ∅ ∧ A_t = ∅ then
13:     A_t ← PlanPrefetch(N_t, S_t, C)
14: (G_t, π_t) ← Propagate(W_t, Γ_t, A_t, S_t)  ⊳ 排冲突；查可行性
15: (Γ_t, A_t) ← Prefix_t(G_t, π_t, S_t)
16: return (G_t, π_t, Γ_t, A_t)
```

---

## 5. 我们的迁移决定

| 论文部件 | 我们的做法 | 状态 |
|---|---|---|
| Predictor 的 graph encoder | 用本项目 F0 预测器 + per-(model, device) 缓存剖面 | 待实现 |
| Eq (2) release 传播 | 直接实现（我们有 ready/near-ready 概念） | 待实现 |
| Eq (3) 融合链判定 | **直接实现**（我们有 succ/pred、model_id、lane） | **本次实现** |
| Eq (4) 融合准入与时长 | 直接实现（我们有 workspace_peak_mb / context 长度代理） | 待实现 |
| Eq (5) 生命周期备选 | 直接实现（模拟器 prefetch 管道已存在，需策略化） | 待实现 |
| Eq (7) 起止时间 | 直接实现 | 待实现 |
| Eq (8)(9)(10)(11) 可行性 | 直接实现（我们有 capacity_mb / resident / batch） | 待实现 |
| Eq (12) 字典序键 | 直接实现 | 待实现 |
| Qwen/vLLM 特定实现、原测试集、原 GPU 型号 | `[NOT MIGRATED]` | — |
| output-length confidence bound | `[NOT MIGRATED]`（我们用 runtime 分布代替） | — |
| KV cache / prefix cache 复用 | `[NOT MIGRATED]`（模拟器无 KV 抽象） | — |
| 跨 workflow 的 model-replica sharing | 待判定：模拟器有 `resident` 但无 replica 概念 | 待定 |

**架构**：按项目负责人决定，每条基线**自带调度器**（`choose_action_v2` 返回带类型的动作
+ `dispatch(action)`），**现有 `choose_action` 与其 33 个 return 一字不动**，
保证已冻结结果可比。
