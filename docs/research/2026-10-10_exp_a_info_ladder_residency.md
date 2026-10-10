# EXP-20261010_info_ladder_residency_v1 — 信息梯度实验(历史先验 → 实例预测 → 打乱 → 真值)

**状态:** 设计冻结(运行前) · 2026-10-10
**动机:** 第 6 轮驻留线已验证"动作有用,但分布消费 vs 点消费 n.s.;打乱信念 mean 仍赢"。
缺失的对照 = **历史统计先验 + 完全相同的驻留控制机制**。本实验补齐信息梯度,回答:
*实例级 H5 预测在驻留动作下,比"不依赖前缀的历史先验"多提供了多少价值?*
(师兄/顾问三组实验之 A;见 2026-10-10 外部建议。)

## 1. 与已有实验的关系

- 复用第 6 轮全部设计(冻结 F0 排序、统一动作规则、统一信念接口 `D(j,k,m)=q_k·p_k(m)·ℓ_k`)。
- 新增臂:**`pdrs_resident_prior`** = pdrs 消费(存活/全分布/软加载)但把身份通道
  `model_probabilities`(全部未来步)与链长通道 `length_probabilities` 替换为
  **池化边际**(跨全部 9,575 条冻结预测行;仅概率、无标签真值)。
- 已有臂直接进入同一运行:reference / pdrs_p / f0point_resident / pdrs_resident /
  pdrs_resident_shuffle / pdrs_resident_oracle(RESIDENCY_POLICIES 内,行为不变)。

## 2. 先验臂的精确定义(冻结)

1. **池化** 对 pack 中每条行 r 与步 k=0..len(steps_r)-1 累加 `p_k(m)`;`p̄_k(m) = acc_k(m)/n_k`。
   链长:`ℓ̄ = mean_r length_probabilities`(逐分量)。
2. **替换** 对每行:所有未来步的 `model_probabilities := p̄_k`;`length_probabilities := ℓ̄`。
3. **保留** 每步 runtime/load 字段(与第 5 轮 `pdrs_prior` 口径一致:先验控制只剥离
   身份+链长通道;载荷尺度属资源预测通道,在全部信息臂中一致保留;打乱臂仍是更重的对照)。
4. **作用域** 仅驻留信念路径(`residency_artifacts_for_policy`),**排序始终是真实冻结 F0**
   (与 shuffle 相同机制;F0 键消费 runtime/load 字段,不受替换影响——加断言测试)。
5. 池化含全部行(含 confirm 行的预测;无标签/真值)。**披露项**:该池化先验是"测试分布边际"
   的强先验,对"实例信息有价值"的主张是保守方向(先验越强,越难显著);仍是边际统计,
   不含任何执行真值。

## 3. 预注册统计(冻结)

- 数据:confirm300(冻结);seed 11;配对 episode bootstrap 2000;零失败硬断言;单次运行
  (7 臂同 run,同 episode 集配对)。
- 参照:Δ = 臂 − `sameshape_h5_p95`(负=更好);主指标 mean,次指标 p95/违约率/makespan。
- 头条对比(直接配对,优先于交互项):
  1. **Δ_instance = pdrs_resident − pdrs_resident_prior**(实例 vs 历史先验)。
  2. Δ_shuffle = pdrs_resident − pdrs_resident_shuffle(实例特异性;复验第 6 轮)。
  3. Δ_ceiling = pdrs_resident_oracle − pdrs_resident(天花板余量)。
  4. 动作效应 = 各 resident 臂 − F0;先验臂 vs F0。
- **判据(证伪矩阵)**:

| 场景 | 判据 | 含义 |
|---|---|---|
| "实例信息有价值" | Δ_instance 在 mean 或 p95 上 CI_upper < 0 | 实例级预测优于历史先验 |
| "先验已足够" | Δ_instance 在 mean 与 p95 上均 CI 含 0,且 prior 臂显著优于 F0 | 动作可被聚合统计充分驱动;实例前缀信息无额外调度价值 |
| "动作本身无价值" | prior 臂 ≈ F0(CI 含 0) | 连先验驱动的动作都不产生收益 |
| "天花板太低" | Δ_ceiling CI 含 0 且 pdrs≈prior | 预测空间本身无信息可挖 |

## 4. 协议与可追溯

- 同一 commit 运行(代码含先验臂+运行器);config: confirm300 + v7 substrate +
  冻结 projection(sha 15c62daf…)+ 同一 pack(sha 记录)+ horizon 5。
- 零失败断言、非有限值断言与第 6 轮运行器一致;机制计数器全量记录。
- 预计成本:7 臂 × 300 集 ≈ **40 分钟**(本地 CPU;第 6 轮 8 臂 45 分钟)。
- 产物:`experiments/EXP-20261010_info_ladder_residency_v1/artifacts/info_ladder_v1.json` + RESULT.md。

## 5. 实现清单

- `pdrs_methods.build_prior_artifacts(..., all_steps=True)`:默认 False = 第 5 轮行为**逐位不变**;
  all_steps=True 池化每步并替换每步(链长同口径池化)。
- `simulator`:RESIDENCY_POLICIES / RESIDENCY_PREFETCH_POLICIES 增加 `pdrs_resident_prior`;
  `residency_artifacts_for_policy` 增加该臂的先验包分支(episode 级缓存,与 shuffle 相同)。
- 运行器:`scripts/info_ladder_comparison.py`(--formal 要求本文件存在 + git HEAD)。
- 测试:池化正确性(每步行、链长、前缀无关性)、默认路径不变、F0 排序不受替换影响、
  fail-closed、泄漏四证照旧。

## 6. 披露边界

- 先验池跨全部预测行(含 confirm 预测;无真值)。
- 先验臂保留载荷尺度通道(资源预测),与第 5 轮口径一致;打乱臂是更重对照。
- 单卡/全驻留负控制仍缺(注册为后续,不属本实验)。
