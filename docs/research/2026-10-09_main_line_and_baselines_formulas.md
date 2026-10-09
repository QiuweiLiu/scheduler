# 主线与基线:评分 / 决策公式(实现对齐版)

日期:2026-10-09 · 用途:汇报/论文公式表;与代码逐行对齐(行号基于当前 repo)。
配套图:`fig16_main_vs_families`(主线 vs 三类对照)。
设计依据(保留/省略/禁止项):`docs/research/2026-10-03_main_table_baselines_manifest.md`。
代码位置:`src/tracing/analysis/{workload_v02_simulator,pdrs_methods,residency_methods,hermes_methods,qlm_methods,torpor_methods}.py`。

## 0. 通用记号

- 统一形式:`chosen = argmin_{c ∈ pool} key(c)`(字典序);pool = ready GPU 节点 × 可容纳 GPU 的全部候选。
- 候选 `c = (item, j, n, m, g, est)`;
  `item = (prio, ready_time, j, n)`(优先级、节点就绪时间、任务号、节点号);
  `est = {runtime_p50_ms, load_p50_ms}`;`resident ⇔ m ∈ g.resident`。
- 公共契约:原子节点不可抢占(抢占臂除外);工程 substrate 与信息契约见 manifest §前提。

---

## 1. 我们:主线 `pdrs_resident`

主线 = **F0 底座排序(逐字节沿用)** + **PDRS 信念** + **最小 ΣV 驱逐** + **保守预取**。
(代码:RESIDENCY_POLICIES 分支 L5036,注释:"Ordering is byte-identical to sameshape_h5_p95"。)

### 1.1 F0 排序键(底座,`sameshape_h5_p95`)

```
current(c) = p50_runtime(m) + 1[m ∉ resident] · p50_load(m)
W5(n)      = Σ_{k=1..5} [ p95_runtime(s_k) + 1{lane(s_k)=gpu ∧ occ_k ≥ 0.5} · p95_load(s_k) ]
key_F0(c)  = ( prio, current(c) + W5(n), W5(n), ready_time, j, n, gpu )
```

- p95/occ 全部来自冻结预测包(部署:F0 包);缺失 p95 时 fail-closed 回退 train 统计(资源 v2 臂故意不做回退)。
- 实现:`sameshape_score` L4736;`_q95_step_cost` L2785;`_legacy_p95_load_cost` L2765。

### 1.2 信念(只喂驻留动作,不改排序)

```
D(j, k, m) = q_k · p_k(m) · ℓ_k          (实例 j 的第 k 步、模型 m 的需求值)
```

| 消费方式 | q_k | p_k(m) | ℓ_k |
|---|---|---|---|
| **pdrs(主线)** | P(L ≥ k) = Σ_{t≥k} length_probabilities[t] | 完整 model_probabilities | occ_k · p95_load_k |
| f0point(对照) | 1 | one-hot(argmax, 平局按字典序) | 1[occ_k ≥ 0.5] · p95_load_k |
| oracle(对照) | 同 pdrs | 真值 one-hot | 同 pdrs |

实现:`residency_methods.step_demand` L62;`survival_weights`(`pdrs_methods` L47);三条规则以外的动作规则**完全一致**。

### 1.3 动作①:最小 ΣV 驱逐(dispatch 需要腾显存时)

```
V(m) = Σ_j Σ_k D(j,k,m)                    (驻留"价值")
驱逐子集 S* = argmin_{S: Σ_{m∈S} mem(m) ≥ required_free}  ( Σ_{m∈S} V(m),  释放量−需求,  字典序 )
```

平局:tie-break = 最小多余释放内存 → 模型名元组字典序;无可行子集时返回全部可驱逐(由上层投影检查裁决,绝不静默放行)。
GPU 选择(同节点多卡候选):min Σ_{被逐模型} V(m);平局 → F0 放置键。
实现:`choose_eviction_subset` L144;`eviction_penalty` L203;分支内 GPU 选择 L5036+。

### 1.4 动作②:保守预取

```
目标 m* = argmax_{m ∈ GPU 可执行模型} U(m),   U(m) = Σ_j D(j,1,m)      (仅 k=1)
条件:m* 未驻留且不在加载中;存在"无需驱逐即可容纳(m* 及其工作区)"的卡(排除预取在途卡)
选卡:空闲显存最大者(平局 → gpu id);预取永不触发驱逐;仅在真实 ready 任务派发之后触发
```

实现:`_residency_prefetch_plan` L6791;`choose_prefetch_gpu` L188。

### 1.5 (已有结论:未采纳)抢占

```
R_B(j) = 可见剩余工作量 + Σ 预测未来后缀工作量
规则:仅当 R_B(目标) < R_B(victim) 时抢占;p=t 边界恢复(R_m + 剩余 decode);零进展活性守卫
结果:p95 +2,559 [998,4113]、makespan +3,376(显著变差)→ 不进入主线
```

---

## 2. 基础基线(主表,冻结)

### 2.1 FCFS

```
key = ( prio, ready_time, j, n, gpu )          # 先到先服务(实现 L4568)
```

### 2.2 Myopic(补充参考,无前视下界)

```
key = ( prio, p50_runtime + 1[m∉resident]·p50_load, 1[m∉resident], ready_time, j, n, gpu )
```

只评估**当前一步**成本(实现 L6332;主表外补充测:mean +1,491.4 [+945.4,+2,102.1])。

### 2.3 Parrot(App-FIFO 适配)

```
key = ( prio, job.arrival_ms, ready_time, j, n, gpu )     # 同 service class 内最老 application 先;连续 stage 承袭 arrival → 深度优先
```

省略:Semantic Variable API / prefix 共享 / batching / 并行 task-group;禁止:critical-path 等 LJF 变体(实现 L5186)。

### 2.4 Torpor(lifecycle 适配)

```
key = ( prio, ready_time, j, n, rank, placement_cost, gpu )
rank: 0 = resident+可用;1 = 已覆盖干扰的冷加载;2 = 未覆盖(回退串行路径,不加臆造惩罚)
placement_cost(rank=1) = 实测 load + 干扰附加(由 (infer_model, infer_shape, load_model) 单元格查得)
驱逐(事件内):swap_burden(m) = load_estimate(m) + max 覆盖干扰(m);按 burden 升序,平局 → model_id
```

实现 L5285;`torpor_methods.torpor_placement_rank` L99、`swap_burden_order` L76。

### 2.5 Hermes(PDGraph + 论文原式 Gittins 适配)

```
key = ( prio, G_j, ready_time, j, n, gpu )      # G 越小优先级越高
G(D,0) = inf_{Δ>0} E[min(X, Δ)] / P(X ≤ Δ)      # 精确经验 Gittins:在支持点上取 inf,O(n) 前缀和
在线 prewarm: p_e = p_s · P(t_c > t_s + t_p);p_s < 0.5 不预热;t_c 只读调度器可见预测
```

实现 L5379;`hermes_methods.gittins_index` L203。

### 2.6 QLM(不确定性感知的随机队列,SAA 适配)

```
场景:S = 64 个公共随机数场景(每集固定种子)
完成时间:C_i = wait_i + runtime_i + load/swap_i(loader 直接进入 C_i,不另设权重)
目标(字典序):① max #{ i : P(C_i ≤ d_i) > Δ }  ② min Σ_i E[(C_i − d_i)_+]  ③ min Σ_i E[C_i]
首槽:每个候选当首元素、其余按 EDD 续排,在 CRN 场景上评估上式;Δ = 0.90(适配超参,附录扫 0.8/0.9/0.95)
```

实现 L5433;`qlm_methods.qlm_saa_choice` L70。

### 2.7 LLMSched(信息获取 vs JCT 利用)

```
每次决策掷一次 ε 硬币(ε = 0.1):
  EXPLOIT: key = ( prio, current_bn + job_remaining_bn, ready_time, j, n, gpu )
  EXPLORE: key = ( prio, group_id, −ΔH(阶段), ready_time, j, n, gpu )
证据 = 已完成阶段的观测时长(后验推进);Y 来自学习网络 + 观测,不读模板未执行后缀
```

实现 L5599;后验/信息量 `llmsched_bn`。

---

## 3. 附:两种"未来感知排序折算"(图 16 B 组,无动作)

```
suffix_expected_cost      = Σ_{k=1..5} q_k · ( runtime_mean_k + occ_k · p95_load_k )        # 均值口径
suffix_expected_cost_p95  = Σ_{k=1..5} q_k · ( p95_runtime_k  + occ_k · p95_load_k )        # p95 口径
q_k = P(L ≥ k) (生存权重)
```

实现:`pdrs_methods.suffix_expected_cost` L82 / `suffix_expected_cost_p95` L102。
纯排序(无动作)结果:均值口径 mean +61 [−217,+329] / p95 +1,666 [+285,+3,000];p95 口径 mean −8 / p95 +226(n.s.)→ 不叠加动作时尾部受损。

---

## 4. 对齐与边界说明

- 所有臂共享同一 substrate 与信息契约(manifest §前提);本文件只给公式,不复述保留/省略清单。
- 主线沿用的 F0 排序键与全部基线/对照臂在**同一模拟器内**逐字节复现(参照 F0 mean 69,788.8878 ms)。
- 所有 p95/occ/model_probs 均来自冻结预测包(部署:J3 结构骨干 + F0 资源头);预测契约与消费盘点见 `docs/research/2026-10-05_parameter_provenance_main_line.md` 与实验 `RESULT.md`。
