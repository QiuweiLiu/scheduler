# EXP-20261010_matched_consumer_v1 — matched-consumer 机制对照(实验 B)

**状态:** 设计冻结(运行前) · 2026-10-10
**动机(外部建议实验 B):** 主表结论是"完整系统对比",不能归因"决策规则更强"。Hermes 有概率未来需求
与预热、Torpor 有驻留/换模/驱逐,但它们**动作触发条件与所用信号**与主线不同。本实验把
**同一份预测输入、同一 GPU 状态、同一合法动作集**(排序 + 最小 ΣV 驱逐 + 保守预取)**固定**,
把主线框架中的**单个决策组件**替换为相关基线的**可迁移决策规则**,做机制对照。
**不是原论文复现;faithful-adapted 主表臂不动。**

## 1. 臂(全部消费同一冻结 pack;单组件替换)

| 臂 | 替换组件 | 规则来源(冻结) | 其余组件 |
|---|---|---|---|
| `sameshape_h5_p95` | —(参照,无动作) | — | — |
| `pdrs_resident`(主线) | —(全主线规则) | 主线 | 主线 |
| `hermes_order_mainline` | **排序键** | Hermes Gittins `G(D,0)`(`hermes_methods.gittins_index`,原式) | 主线驱逐/预取/安置 |
| `hermes_prefetch_mainline` | **预取规则** | Hermes 在线预热触发条件(忠实臂同条件:p_s≥K=0.5、单运行任务、装载在预测完成前;目标=运行中节点的下一单元) | F0 排序/主线驱逐 |
| `torpor_evict_mainline` | **驱逐规则** | Torpor `swap_burden_order`(冻结实测负担序,直接复用) | F0 排序/主线预取 |

## 2. 各替换组件的精确语义(冻结)

### 2.1 `hermes_order_mainline` 的排序键
- 候选的**剩余需求经验分布**从 pack 推导(而非 PDGraph):给定 (job 前沿节点) 行,
  链长结果 k=1..H 的取值 = 累积保守步成本 `W_k = Σ_{i≤k} q95_step_cost(step_i)`
  (与 F0 排序同函数);质量 = `probs[k]`(k<H,个体质量)与 `Σ_{i≥H} probs[i]`(k=H,尾部),
  归一化 `q1=Σ probs[1:]`;整数重复 scale=200 构成经验样本。
- `G = gittins_index(samples)`(下界检查原式;None → +inf),分数元组
  `(item[0], G, item[1], item[2], item[3], gpu.index)`(Hermes 忠实臂同构)。
- 缺失行/空分布 → 样本 [] → G=+inf(与忠实臂 fail 行为一致)。
- 放置:同主线——被选节点的多卡候选中按 (最小驱逐 V 罚, 排序键) 重选;计数
  `hermes_order_placement_flips`。

### 2.2 `hermes_prefetch_mainline` 的预取规则
- 触发条件(与 faithful `_hermes_prefetch_plan` 同构):设备恰有 1 个活跃运行任务、无 pending 预取、
  已开始执行;需求 = **运行节点** pack 行的第一步模型分布(即"该节点完成后的下游单元");目标 =
  argmax(字典序平局);`p_s ≥ K=0.5` 才触发;装载估计 `t_p = model_load_estimate_ms`(同一实测中位);
  需在**调度器可见预测完成**前装完:`now+t_p ≤ t_c`,`t_c = work_start + predicted_work_ms × slowdown`
  (不用真值);重复保护保持**同卡范围**(忠实臂原样;主线预取的"全设备"重复保护属其规则本身,
  不混入本替换组件——披露);目标设备 = 运行中的同一张卡(与忠实臂一致)。
- 计数:`hermes_style_prewarm_trigger`。

### 2.3 `torpor_evict_mainline` 的驱逐规则
- `eviction_preference = swap_burden_order(gpu, train_stats, interference_profile)`(Torpor 冻结规则原样;
  信息盲——其投影没有未来需求视图,这是其规则的固有性质,作为对照事实披露)。
- 其余同主线(排序 F0、预取 U-argmax、V 不参与驱逐)。

## 3. 预注册统计(冻结)

- confirm300;seed 11;配对 episode bootstrap 2000;零失败断言;单次运行(5 臂同 run)。
- 头条对比(负 = 候选更好):`pdrs_resident − 每个 swap 臂`(mean 与 p95),各自 CI。
- 判据(证伪矩阵):

| 场景 | 判据 | 含义 |
|---|---|---|
| "主线决策规则贡献" | 主线在 mean 或 p95 上优于**每个** swap 臂(CI_upper<0),且激活审计确认替换组件真实触发 | 优势来自决策规则,而非信息/动作接口 |
| "组件等价" | 某 swap 臂 ≈ 主线(该指标 CI 含 0) | 该组件的**规则形态**不是差异来源(信息/动作接口或触发率解释差异) |
| "替换组件未激活" | 某臂的激活计数 ≈ 0 | 该对照无效,不得解释为"规则等价"(如实登记) |

- 机制审计:每臂激活计数(排序翻转/预取触发/驱逐翻转)+ 机制计数器全量输出。

## 4. 协议与披露

- 同一 commit 运行;config: confirm300 + v7 substrate + 冻结 projection + 同一 pack(sha 记录)。
- 披露:(a) 三臂均为机制对照,非论文复现;Hermes 的 PDGraph 在线精炼保持其原有适配边界不变;
  (b) 预取臂目标设备=运行卡、重复保护=同卡(忠实形状),与主线"最大空闲卡、全设备重复保护"
  的差异属规则差异本身;(c) Torpor 驱逐规则的"信息盲"是其投影的固有性质。
- 成本:5 臂 × 300 集 ≈ 30 分钟(本地 CPU);smoke 10 集先行。

## 5. 实现清单

- `simulator`:`pack_gittins_samples` 模块级 helper;`hermes_order_mainline` 排序分支;
  `_hermes_pack_prefetch_plan` + 分派;`torpor_evict_mainline` 计入驱逐偏好分支;
  三臂注册进 RESIDENCY_POLICIES(排序忽略例外)/ RESIDENCY_PREFETCH_POLICIES(预取臂除外)。
- 运行器:`scripts/matched_consumer_comparison.py`(--formal 要求本文件 + git HEAD)。
- 测试:`tests/test_matched_consumer_arms.py`(样本构造、注册、触发条件、端到端、泄漏)。
- 产物:`experiments/EXP-20261010_matched_consumer_v1/artifacts/matched_consumer_v1.json` + RESULT.md。
