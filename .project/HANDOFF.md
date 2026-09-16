# Handoff

**Last Updated:** 2026-09-15

## Goal

验证"视频 Agent 工作流的未来预测能否改进 GPU 调度"，并形成可发表的论文主线
（未来工作流不确定性 + 风险消费；不写"预测 runtime 帮助调度"）。

## Done

- Phase 0–7 全部完成：接口验证 → 信息价值 → 形态匹配/归因 → 内容与长度消融 → horizon → 消费方式锦标赛 → dev/confirm 确认。
- 消费器共 16 变体（表分位数、λ 混合、q95、机会约束、期望、生存、场景概率、缓存、内容、场景采样+CVaR、共单调、自适应风险、分量拆分）。
- validation 切分（700 dev / 300 frozen confirm）与运行器支持；冠军 q95 已在 confirm 确认（−18,765 [−21,948, −15,932]）。
- H10 重训完成（负迁移，含 S_* 质量审计与分解臂）；分布 artifacts 打包（`--emit-distributions`）。
- 桥接修好（Chrome 需 7897 代理）；GPT 文献调研 + 计划已归档；论文规划咨询已发出（回复待收）。

## Verified

- **q95 消费 = 当前冠军**，dev 与 confirm 一致；rt95 等价（+11ms）→ 收益来自 runtime 尾部。
- E2（未来结构 + 静态资源表）显著优于无未来（1000 集 −17,065ms，CI[−18,692,−15,458]）。
- oracle 资源 ≈ 静态表；真值内容更差；长度/终止结构主导。
- 真值 H10 vs H5 = −10,231ms，H10≈无界 oracle（可部署 lever，但朴素重训失败）。

## Rejected（不得重跑，除非有新证据）

- 朴素 H10 重训（表示稀释；需 H10-lite 配方）。
- 场景采样 + CVaR / 共单调耦合 / 自适应风险 / 机会约束 / 生存加权 / 缓存感知 / 内容身份 / 期望成本 / load 尾部 —— 全部未超过 q95。
- J4 duration 解耦（负结果）；资源 p50 直接替换（+16,012ms）。

## Open

- H10-lite（H5 backbone + 后段辅助头）未实现。
- cache-aware 需 residency/reuse 数据（缺）；真实系统实验缺（论文短板）。
- 稳健性未做：期限档位、预测器 seeds（J3 seed22/33）、跨 workload。

## Active

- 无运行中的实验（CPU/GPU 空闲）。
- 系统论文定位已定稿（GPT 重分析）：主贡献 C1 future-action-aware scheduling；机制 C2；管线 C3。故事句 + 基线 + 投稿概率 + 审稿攻击见 docs/research/2026-09-15_fas_novelty_venue_gpt.md。最小补充：P0 2-GPU replay（阻塞：硬件确认）> P1 6-cell pressure sweep / 预测质量敏感性 > P2 H10-lite。

## Next

0. 实验总表已定（docs/research/2026-09-15_fas_experiment_table_gpt.md）：最小集（基线矩阵+6-cell sweep+质量敏感性）/ 推荐集（+2-GPU replay+相关性机制）/ 完整集（+H10-lite+电梯第二 workload）。
1. 确认 P0 硬件可行性（本地 1×RTX 3060 6GB / 远端 autodl GPU 情况），设计最小真实 replay 协议。
2. P1 压力 sweep（现有 600 视频池，dev-only，E0/E2/q95/oracle × ~6 cells）。
3. 核验 GPT 引用论文（Parrot/SOLA/SuperServe/Katz/Vidur/FATE），再写 Related Work 定调句。

## Environment / Recovery

- 解释器 `D:\anaconda\envs\scheduler\python.exe`；`PYTHONPATH=src`；RTX 3060 Laptop。
- 桥接：Chrome for Testing 需 `--proxy-server=http://127.0.0.1:7897`；会话 `https://chatgpt.com/c/6a9822da-e278-83e9-9c1a-675923acda0e`。
- 控制面历史归档：`.project/archive/2026-09-15_pre_compact/`。

