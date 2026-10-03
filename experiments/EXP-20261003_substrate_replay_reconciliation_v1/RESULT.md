# RESULT — EXP-20261003_substrate_replay_reconciliation_v1（步5：F9 真机回放对账）

**状态**：完成（2026-10-03）。**远端脚本 `f9_episode_replay.py`；产物 `artifacts/f9_physical.json`**
（36 runs = 4 episode × 3 policy × 3 reps；12 warmups）。

## 目的与设计

在真机上执行 **4 个 policy-independent episode**（节点级调度循环，构造输入、真实 HF 调用），
3 个机制性 policy 控制 dispatch/加载/预取/并发，检验**机制组合**是否与 substrate 代价模型一致。
**同口径**：物理侧与预测侧共用同一批实测 solo 时长（消除 8/10 月时段混淆）。

- Episodes（仅 4B/3B，避免 8B 显存压力）：
  - E1 warm/light：2 jobs × [small/medium]（无加载）
  - E2 cold/churn：单 job，冷启动 3B→4B 换模型（2 次加载）
  - E3 colocation：2 jobs（3B ∥ 4B）medium
  - E4 prefetch：单 job [4B large → 3B large → 4B large]（1 次可隐藏加载）
- Policies：**P0_seq**（顺序+阻塞加载）/ **P1_prefetch**（顺序+当前 op 期间后台加载下一模型）/
  **P2_crev**（双 job 就绪 op 并发 + 预取）。

## 结果（makespan 中位数，ms）

| episode | P0_seq | P1_prefetch | P2_crev | Δ(P1−P0) | Δ(P2−P0) |
|---|---|---|---|---|---|
| E1 warm_light | **12,868** | 12,838 | **15,923** | −30 | **+3,055 (+24%)** |
| E2 cold_churn | **18,958** | **17,644** | 18,297 | **−1,314 (−7%)** | −661 |
| E3 colocation | 17,487 | 17,481 | 19,436 | −6 | **+1,949 (+11%)** |
| E4 prefetch | 29,224 | 29,520 | 29,479 | +296 | +255 |

逐 op 证据（rep0）：
- E2|P1：4B 加载（4.1s）藏进 op1（3B medium）——op1 被干扰 3.1→6.6s（+3.5s），wait_load 4.1→0 → 净收益存在。
- E4|P1：3B 预取藏进 op0（op0 7.9→9.1s），wait_load 3.2→0；但首个 4B 加载本底波动（3.7→6.0s）抵消净收益。
- E1|P1 ≈ P0（无加载可藏）→ 健全性检查通过。

## 对账（公式级，采用 substrate 实测常数）

- **E2 预取收益预测**：`min(load≈4.1s, op1≈3.1s) − 3.1×(1.7−1) ≈ +0.9s`；观测 **+1.3s** → 符号与量级一致。
- **E4 预取收益预测**：`min(load≈3.2s, op0≈7.9s) − 7.9×(1.3−1) ≈ +0.8s`；观测 **−0.3s** → 被首加载方差（±2.3s）淹没，不能判定机制失败。
- **P2 并发代价**：预测（F1 单进程口径）并发拖慢 → makespan 上升；观测 E1 +24%、E3 +11% → 同号。
- **结论**：机制组合与代价模型**符号一致**；预取净收益的定量受加载时间方差限制（需更多 reps 或更大隐藏窗口）。

## 边界（必读）

1. **单进程执行器**：P2 的并发惩罚继承单进程 GIL 假象（见 MPS 实验）；真实多进程部署下该惩罚预计消失。
2. 3 policies 为**机制性策略**（非论文主基线）；本实验是对"机制组合 ↔ 代价公式"的对账，不是完整 DES 端到端回放。
3. 3 reps；单卡共享环境；E4 的加载本底波动 ±2s 量级。
4. `lm_head.weight MISSING` 加载报告出现于 7/48 runs（transformers 加载日志，复现测试未见；
   输出长度被 min_new_tokens 强制，计时语义不受影响）——记录在案。
