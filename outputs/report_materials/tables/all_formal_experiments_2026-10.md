# 完整实验表格 — 调度证据链 + 全部正式实验清单

更新: 2026-10-10 · 生成源: `.project/EXPERIMENT_GATE.json`(44 个实验)+ 各 EXPERIMENT/RESULT 记录
说明: Δ 一律 = 候选 − 参照(负 = 更好);CI95 为逐集配对 bootstrap(2000, seed 11);
所有调度实验数据 = 冻结 confirm300(逐集配对);运行全部本地 CPU。

## 表 1. 调度证据链总表(A–E 为 2026-10-10 新增实验程序)

| # | 实验 | 问题 | 设计 | 关键数字 | 预注册判据落点 |
|---|---|---|---|---|---|
| 1 | **主表对比** `EXP-20261004_main_table_comparison_v1`(冻结 62be126) | 冻结 F0 与五条主表基线 | confirm300 × 6 臂;v7 substrate | F0 = 69,788.9 ms;Δmean:Hermes **+350** / LLMSched **+939** / Parrot **+2,048** / QLM **+3,379** / Torpor **+3,522**(全部显著更差) | F0 显著优五基线;第四条 margin-inferior;「调度器更强」不单独主张 |
| 2 | **主表 p95 补测** | 尾部口径是否翻转排序 | 同冻结构,指标扩展 | Parrot p95 **−2.40%**(唯一显著更好);QLM +4.08%;其余中性 | 尾部与均值排序不同(饥饿/公平主导尾部) |
| 3 | **PDRS 三版** `EXP-20261005_pdrs_comparison_v1` | 复杂分布排序是否值得 | v1/v2/v3(均值/p95 口径) | v1 全臂 +46..+83 n.s.;v2 四指标无胜利;v3 p95 口径 −113 n.s.;oracle 平 | **负结果**:停止规则触发;oracle 无剩余空间 |
| 4 | **驻留动作第 6 轮** `EXP-20261005_residency_comparison_v1` | 分布信念 + 驻留动作是否解锁价值 | confirm300 × 8 臂;统一 F0 排序 | 动作包 mean **−536..−884**(全显著)/p95 −1.5k..−2.5k;信念增量 n.s.(打乱亦赢);实例特异性仅 p95(vs shuffle **−1,615**);抢占净有害(p95 +1,859..+2,559、mk +3,085..+3,376) | 动作有用;分布信念增量未支持;抢占排除;机制:驱逐 −10.3%/冷 −22%/重载 −22%/预取命中 |
| 5 | **系统级对照** | 完整系统 vs 6 基线 | 主线为基准逐集配对 | mean +1.73%~+7.49% **全胜**;p95 除 Parrot 打平(−0.47% n.s.)全胜;makespan 胜 3 平 2 负 Torpor(−1,629);违约率全胜 | 完整系统比较;Parrot 尾部反转消解为打平 |
| 6 | **A 信息梯度** `EXP-20261010_info_ladder_residency_v1`(运行 `2bea092` · 入库 `a1cd92a`) | 实例级预测比历史先验多值多少 | 同排序+同动作,7 臂 | 先验臂 **−600 [−948,−292]** / p95 −2,005;实例−先验 **−241 [−605,+108] n.s.** / p95 −543 n.s.;特异性仅 vs shuffle p95 −1,615;先验预取 24 次 vs 主线 1,432 | **「先验已足够」**:实例增量 n.s.;聚合边距无法为预取选靶;先验收益≈驱逐通道 |
| 7 | **B matched-consumer** `EXP-20261010_matched_consumer_v1`(运行 `30284d9` · 入库 `154e413`) | 同信息同动作下决策规则的贡献 | 单组件替换(3 臂) | 主线−Hermes 排序 **−5,732 [−6,984,−4,650]** / p95 −11,850(Gittins 换入后 vs F0 反升 +4,890/+9,303);主线−Hermes 预热 **−275 [−564,−4]**(其触发 173 次仅 11 装载,162 容量拒);主线−Torpor 驱逐 **−158 [−259,−60]** | **「决策规则贡献成立」(mean)**:优势不只来自信息/动作接口 |
| 8 | **C 压力矩阵** `EXP-20261010_pressure_grid_v1`(运行 `b9d1deb` · 入库 `09d43b0`) | 什么条件下最有效 | 3 负载 × 3 容量 × 3 臂 | 主线 **9/9 格显著**优于 F0(−289..−1,707);H1 负载基本成立(8/9 单调);H2 容量**反向**(−587/24·24 < −1,491/32·24 < −1,707/32·32 @α0.8);无失效格;Myopic 9/9 显著更差 | 收益随负载↑、余量↑;紧容量压缩动作空间;主线预取容量拒绝全表 0 |
| 9 | **D 反事实消除** `EXP-20261010_future_info_counterfactual_v1`(运行 `b8c0826` · 入库 `186bf24`) | 收益是否依赖模型切换/驻留压力 | 单模型 + 全驻留 反事实 | 单模型格主线 **≡F0(Δ=0.0,激活全 0)**;全驻留格 −920→**−250**(收缩 73%,驱逐归零,残余=预取 405 次);Δgain **+670 [+300,+1,020]** / **+920 [+639,+1,208]** 均显著 | 驱逐通道←容量压力;预取通道←模型切换(无切换即死)、不依赖容量压力 |
| 10 | **E 参数敏感性** `EXP-20261010_prefetch_sensitivity_v1`(运行 `b8c0826` · 入库 `186bf24`) | 冻结参数是否碰巧选好 | 单变量替代臂 × 3 | ready-only **+405 [+170,+637]** 显著更差(预取仅 147 次);k2 **−324 [−609,−67]**、inflight2 **−334 [−601,−84]** 显著优于冻结主线(mean,~−0.47%);p95 无增益(k2 +143 n.s.;inflight2 +704 边缘偏坏) | 聚合范围=正确;k=1/并发=1 按**探索性**口径被证伪(幅度小、对比族内);未改冻结主线,建议 dev 复核 |

## 表 2. 全部正式实验清单(gate 44 项,按登记顺序)

| # | 实验 ID | 目的/问题 | 状态 | 一行结论(如有) |
|---|---|---|---|---|
| 1 | `EXP-20260919_j_series_resource_dist_v1` |  | active | R1 and R1b both viability PASS / strong FAIL (2 of 4); R3a partial-unfreeze is the next pre-registered step |
| 2 | `EXP-20260911_forecast_aware_scheduling` |  | phase7bc_done_consumer_champion_q95 | q95 consumer is the champion (dev -17,229; confirm -18,765 ms vs E2); scenario sampling+CVaR, comonotone, ad… |
| 3 | `EXP-20260911_p9d_j4_duration_branch` |  | completed_negative | decoupling did not fix load-duration; root cause = in-distribution fit re-allocation. |
| 4 | `EXP-20260911_p9d_j_h10_predictor` |  | completed_negative_transfer | naive horizon extension degrades first-5-step quality (family 0.803->0.640, runtime pinball +62%); scheduler… |
| 5 | `EXP-20260911_p9d_j_predictor_acceptance` |  | frozen_candidate_in_use | J3:seed11 frozen (the H5 predictor used by all scheduler phases); memory head untrained (never expose). |
| 6 | `EXP-20260911_p9d_j_series_joint_resource` |  | completed_no_core_go | runtime endpoint significantly better than B1 (J2/J3, 3/3 seeds) but load-duration marginally out of the NI … |
| 7 | `EXP-20260911_p9d_r0_oracle_signature_ceiling` |  | completed_core_go_pass | runtime/load identifiable (runtime val +74.9%); memory descriptive-only. |
| 8 | `EXP-20260911_p9d_r0p_workload_proxy_probe` |  | completed_no_gain | workload size is not recoverable from existing records. |
| 9 | `EXP-20260921_histres_causal_input_v1` |  | active | Historical execution telemetry adds no measurable predictive value under this workload (F1-F0 CI95 [-0.3613,… |
| 10 | `EXP-20260921_scheduler_replication_v1` |  | active |  |
| 11 | `stress_regime_v1_20260927` |  | failed_construct_validity_audit |  |
| 12 | `stress_regime_v2_20260927` |  | FROZEN_alpha_0.80_rev3_reverified |  |
| 13 | `EXP-20260929_substrate_f0_hardware_stability_v1` | substrate 地基 F0：硬件身份 + 频率/功耗/温度稳定性 + 干扰来源归因 | active | UNAVAILABLE: GPU IS SHARED/time-sliced. 150-run 10.4-min timeline: fast mode 75.3% (28.95 ms/step, util 52.2… |
| 14 | `EXP-20260929_substrate_colocation_surface_v1` | substrate 地基 F1：完整卡内 co-location 代价面（45 格），供 simulator 拓宽动作域 | active | DONE 45/45 cells. 35/45 显存可行。共置代价(makespan/串行) 中位 1.149，范围 [1.008,1.296]，**0/35 格优于串行** → LLM↔LLM 共置对 makesp… |
| 15 | `EXP-20260929_substrate_phase_profile_v1` | substrate 地基 F2：prefill/decode/vision-encode/KV 曲线（28 行） | active | prefill 严格线性：8B 190.0 / 4B 114.6 / 3B 75.3 µs/token；tpot 恒定 28.42/29.04/23.30 ms（CV<1.5%）。⚠️ 8B 与 4B 的 decod… |
| 16 | `EXP-20260929_substrate_lifecycle_profile_v1` | substrate 地基 F3+F3b：模型生命周期四层 | active | L2 warm 全量重载 8B 5191 / 4B 2785 / 3B 2790 ms；L4 allocator evict 154-196 ms；L3 纯搬运 H2D 12.13 GB/s vs D2H 1.86 … |
| 17 | `EXP-20260929_substrate_overlap_matrix_v1` | substrate 地基 F4：inference ∥ load(B) ordered-pair 重叠矩阵 | active | **预热的主要代价不是搬运数据**：纯拷贝只让推理慢 1.085×（中位），且拷贝本身 dilation≈1.0008；完整框架加载让推理慢 1.722×（中位），比纯拷贝贵 1.59×。干扰源是框架加载的非拷贝开销… |
| 18 | `EXP-20260929_substrate_overlap_audit_v1` | substrate overlap 审计（GPT Gate 3 要求）；远端无 nsys，改用 torch.profiler chrome trace | active | 6 case。GPU occupancy 中位 **0.389**（卡在共置期间 61% 空转）；两臂 overlap ratio 中位 **0.233**（基本轮流跑）；span 比'两臂 GPU 忙时间之和'长 … |
| 19 | `EXP-20260929_vllm_batching_v1` | 第5项：dynamic / continuous batching（隔离 vLLM venv） | active | A) microcurve B=1→16：吞吐 78.1→1043.0 tok/s（**13.4×**），wall 只涨 1.20×。B) 64 请求异构载荷：seq 120,315ms / static_16 15… |
| 20 | `EXP-20260929_vllm_prefix_cache_v1` | 第6项：KV / prefix cache reuse（隔离 vLLM venv）——验证 Pythia 机制可触发 | active | 净收益随共享前缀比例 r 单调上升：r=0 1.003× / 0.25 1.189× / 0.5 1.745× / 0.75 2.570× / **r=1.0 5.888×**。OFF 对照全程平坦（0.992-1.… |
| 21 | `EXP-20260929_vllm_preemption_v1` | 第7项：KV 压力下的 request-level preemption/recompute 代价（隔离 vLLM venv） | active | 16 个并发请求、max_model_len=12288。KV 压力档 gmu=0.40（KV≈5GiB）。代价：ctx=1024 **1.01×**（KV 充足，阴性对照通过）/ 4096 **4.64×** / … |
| 22 | `EXP-20261001_substrate_transition_integration_v1` | 1b 接入: F3 实测 transition 成本以 opt-in transition_profile 进仿真器 (零代码改动) | active | 接入链路已验证。config 通过 validate_transition_profile (3 topologies) 且 strict 下 4 个 model_id 全部解析。limit=2 paired pil… |
| 23 | `EXP-20261001_substrate_phase_integration_v1` | 1c 接入: F2 phase_profile 观测性分解 (零服务时间扰动) | active | 更正：早期 limit=3 开关对照验证了执行结果不受影响，但 25 图像节点、5550 图像 tokens、2787.1 ms prefill 的归因解释已撤回（缺少真实图像数量和输入长度）。当前仅暴露实测系数；输… |
| 24 | `EXP-20261002_substrate_token_prior_v1` | 步1: 48-run 小样本重放,采集每节点 token 先验(原 trace 无 token 字段) | active | 完成:48/48 rc=0,351 条带 token 记录,6 层先验。补提取 571 行(含 first_token_ms 72 条、显存 400 条)。F2 prefill 斜率被 4B 真实数据验证(115.3… |
| 25 | `EXP-20261003_substrate_colocation_extrapolation_v1` | 步3: F1 共置表 OOD 外推(3 XL×XL + 1 XS×XL + 1 staggered) | active | 完成。M3 对 XL 长任务外推成立（三个自对误差 −1.8%~−3.4%）；staggered 剩余工作积分正确（−1.0%~−1.3%）；极端不对称 XS 侧 M3 低估 32%（160ms 小任务实测 3.16… |
| 26 | `EXP-20261003_substrate_prefetch_shape_extend_v1` | 步4: F4 预取干扰形状扩展(top-2 转移对 × XS/XL) | active | 完成。加载干扰强烈依赖推理时长：full-load slowdown 从 XS 的 18.6–21.6× 降到 XL 的 1.18–1.30×；constant-factor 不可外推；加载侧 dilation≈1（… |
| 27 | `EXP-20261003_substrate_replay_reconciliation_v1` | 步5: F9 真机回放对账(4 episode × 3 policy × 3 paired,同口径) | active | 完成。4 episode × 3 policy × 3 reps：P1 预取在 E2 净省 1.3s（−7%，公式预测 +0.9s 同号同量级）；E4 被加载方差淹没；P2 单进程并发惩罚 +24%/+11%（继承 … |
| 28 | `EXP-20261003_mps_comparison_v1` | C组: MPS 对照(单进程是否放大共置代价) | active | 完成。F1 单进程 ~2× 共置代价主要是 GIL 假象：多进程下 4B/3B 对 slowdown 仅 1.02–1.04；8B/3B 仍有真实残差 1.22–1.27；MPS 相对多进程无增益。F1 表语义限定为… |
| 29 | `EXP-20261003_substrate_small_items_v1` | 小项打包：显存口径（生成 vs 全量前向）+ YOLO 子进程归因 | active | 完成。显存口径解决：generate 路径增长 0.243 MB/tok（与重放 0.24 吻合），F2 扫描 0.438（含全量 logits 物化）→ 裁定 DES 用 generate 口径。YOLO 61% … |
| 30 | `EXP-20261003_mp_calibration_v1` | mp 口径校准：DES 部署切换（同模型对=串行；异模型对 mp 共置表；mp 加载干扰） | active | 完成（含 B 组扩展至 7 格）。mp 共置：8B 对最贵（1.24–1.60），小模型对 1.02–1.18；极端不对称小任务 1.60（单进程 3.16）。mp 加载干扰全表：extra ≤634ms，按被加载模… |
| 31 | `EXP-20261003_substrate_preemption_resume_v1` | P1: 抢占往返验证（打断→丢KV→重算→继续 的 R_m 是否 = F2 曲线） | active | 完成。三配置残差 −1.8%~−5.6%（全部略快于预测）；总时长差 ≈ 重算时间（107–188ms）。R_m=F2 曲线确认，DES 记账模型验证通过。 评审 P1-1/P1-2 修正：RequestSplit … |
| 32 | `EXP-20261003_substrate_batching_curves_v1` | B2: 批处理曲线（vLLM，workload 真实形状，8B/3B/4B × 文本/图像） | active | 完成。B=2 ≈ 1.9×（3B answer 1.63×）；B=16 达 13–15×；输出越短收益越低。KV 池实测：8B 50,480 / 3B 476,160 / 4B 130,624 tokens。 已由仿… |
| 33 | `EXP-20261003_substrate_cross_model_vllm_v1` | B3: 跨模型并发 vLLM 双引擎 vs HF-mp 表 | active | 完成。8B 侧 1.29 vs HF-mp 1.24（一致）；3B 侧 0.98 vs 1.27（不对称：vLLM 下小引擎免付）。异模型并发建模建议用不对称模型或保守上限。 |
| 34 | `EXP-20261003_substrate_resume_3b_v1` | P1-5 闭环：3B resume（R_m）代表性格 + 4B 对照格（同一协议 p1_resume_roundtrip.py，reps=3） | active | 完成，gate 全过：3B k20 +2.29ms(+2.6%)、3B k40 +1.34ms(+1.5%)、4B 对照 −2.31ms(−1.8%)，均满足 /res/≤100ms 且 /res%/≤10%。P1-… |
| 35 | `EXP-20261004_main_table_baselines_activation_v1` | 主表四条基线 activation audit（pre-run 证据；本地无 GPU；dev 子集，不碰 confirm300） | active | v2 重跑（review 2424f67 修正后）：计数器改反事实口径（QLM swap_cost_affected 256/3989、Hermes gittins_vs_mean_flip 7、pdgraph_re… |
| 36 | `EXP-20261004_real_strata_calibration_v1` | 真实档位全表校准：共置 11 格 + 加性加载干扰 12 格（冻结工作负载的每一个可达组合） | active |  |
| 37 | `EXP-20261004_main_table_comparison_v1` | 主表正式对比：F0 (sameshape_h5_p95) vs 五条主表基线（Parrot/QLM/LLMSched/Hermes/Torpor），v7 substrate +… | frozen |  |
| 38 | `EXP-20261005_pdrs_comparison_v1` | PDRS 对比 v1:前缀条件的分布化滚动视野调度(表外方法)vs F0/Myopic + 信息对照(先验/打乱/oracle) | completed (negative result) | PDRS v1 does not beat F0 (all arms +46..+83 ms, CI include 0); oracle ceiling flat (+46) |
| 39 | `EXP-20261005_residency_comparison_v1` | Round-6 residency actions: does the distributional future belief become actionable once … | completed | actions useful, distributional belief increment unsupported (primary metric) |
| 40 | `EXP-20261010_info_ladder_residency_v1` | 信息梯度(历史先验→实例→打乱→真值)在驻留动作下的对照(外部建议实验 A;补第 6 轮缺失的 prior 臂) | completed | '先验已足够' 落点:实例 vs 先验 n.s.(mean −241 [−605,+108]; p95 −543 [−1840,+823]); prior 臂显著优于 F0(mean −600/p95 −2005/m… |
| 41 | `EXP-20261010_matched_consumer_v1` | matched-consumer 机制对照(实验 B):同 pack/同状态/同合法动作集,单组件替换为 Hermes/Torpor 可迁移决策规则 | completed | '主线决策规则贡献'成立(mean 口径):主线显著优于全部三个单组件替换臂(hermes_order −5732/−11850; hermes_prefetch −275 [−564,−4] 边缘; torpor_… |
| 42 | `EXP-20261010_pressure_grid_v1` | 负载强度×显存容量 3×3 压力矩阵(实验 C):Myopic/F0/主线;回答'方法在什么条件下最有效' | completed | 主线 9/9 格显著优于 F0; H1(负载)基本成立(8/9 单调步); H2(容量)反向拒绝——容量越宽收益越大(−587/24/24 < −1491/32/24 < −1707/32/32 @α0.8), 机制… |
| 43 | `EXP-20261010_future_info_counterfactual_v1` | 未来信息消除反事实(实验 D):全驻留/单模型下主线增益是否消失 | completed | 单模型:增益完全消失(主线≡F0 逐位相同, 激活全 0); 全驻留:增益收缩 73%(−920→−250, Δgain +670 [300,1020] 显著)但残余=预取通道(驱逐归零); 参照格与实验 C 逐位一致 |
| 44 | `EXP-20261010_prefetch_sensitivity_v1` | 预取参数敏感性(实验 E):k=1→2 / 并发 1→2 / V-U 聚合 ready-only 三个单变量臂 | completed | readyonly 显著更差(+405 [170,637])→范围冻结选择正确; k2(−324 [−609,−67])与 inflight2(−334 [−601,−84])在 mean 上显著优于冻结主线(均 ~… |
