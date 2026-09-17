# Handoff

**Last Updated:** 2026-09-17

## Goal

验证"视频 Agent 工作流的未来预测能否改进 GPU 调度"，并形成可发表的论文主线
（未来工作流不确定性 + 风险消费；不写"预测 runtime 帮助调度"）。

## Done

- Phase 0–7 完成并归档（接口验证 → 信息价值 → 形态匹配 → 消融 → horizon → 消费锦标赛 → dev/confirm 确认）。
- Phase 11–15 完成：runtime-only 正交族（C2 严格检验）、CVaR 重写、优化型参考重跑、fixed-L 对照、CP-RHO executed 变量修复 + 300 集配对重跑。
- trace 依赖检验 T1/T2/T3 完成：**公共慢化因子 / 尾部共动假设被否**，叙事转为 forecast-error-aware ranking surrogate。
- 公开仓库两轮 GPT 审阅 + 修复：CP-RHO 量纲/结构/executed、runner 指纹（含 gzip 载荷摘要 + fail-closed）、normalizer manifest、tail-shuffle 脚本公开、requirements、LICENSE/PROVENANCE/reproduce_main.sh、B≥2000 双侧统计。
- **Phase 16（本次）**：oracle 对照 confounded 的 P0 认定 + 撤回；新增同 key 形状三臂并单元验证（未跑实验）。

## Verified

- 冠军 = r95/q95（逐步骤 runtime p95 求和）：dev700 −17,229ms、frozen confirm300 −18,765ms [−21,948, −15,932] vs E2。
- C2 机制：收益来自尾部形状 + 状态对齐，不是放大尺度（Pred95R − ScaledPred50R −12,963；− ShuffledTail −7,016）。
- 压力 sweep 6/6 cell 显著为正且随负载单调（低→高约 3.4×）；质量敏感性呈剂量-反应（tail 最强，content 无可测效应）。
- `sameshape_h5_p95` 与 `predopt_h5_q95` 在同一 episode 上 summary 逐字段相等（新代码钉在已验证冠军上）。

## Rejected（不得重跑，除非有新证据）

- 朴素 H10 重训；场景采样+CVaR / 共单调 / 自适应风险 / 机会约束 / 生存 / 缓存 / 内容 / 期望成本 / load 尾部全部未超过 r95。
- **trace 公共慢化因子 / 尾部共动机制**（T1 未约束 MOM 分量 −0.139 CI 全负、截断 ICC 0；T3 lift<1 但双侧 p=0.54/0.20/0.25 不显著）→ 不得写"证明负相关"。
- 优化型参考（CP-SAT/pred-MPC）作"优化不是瓶颈"的证据：修好后仍落后 r95 约 19s，且 15.4% 决策撞 0.25s 上限、平均 gap 10.3% → 只能 exploratory。
- **"q95 胜过 oracle 臂"**：key 形状未对齐（Phase 16），已从门禁撤回。

## Open

- Truth-SameConsumer 三臂**已实现未跑**（门禁：dev700 配对，判据见 PHASE16）。
- H10-lite 未实现；cache-aware 缺 residency/reuse 数据；真实 2-GPU replay 未做（论文最大短板，阻塞于硬件）。
- 既有测试失败（与本次改动无关）：`tests.test_workload_v02_simulator.AdmissionTests.test_round_robin_joint_action_keeps_oldest_ready_node`。

## Active

- 无运行中的实验（CPU/GPU 空闲）。
- 控制面 `STATE.md` 顶部的 09-15 段落仍是旧快照；09-17 状态见新增段落与 `EXPERIMENT_GATE.json`。

## Next

1. 跑 Phase 16 三臂（dev700 配对 → 判据分支），决定叙事走"校准"还是"不确定性溢价"。
2. 补真实 replay 的硬件确认（本地 1×3060 6GB 只能做节点级 fidelity）。
3. 剩余仓库项：`trueopt_h5`/`predopt_h5` 命名与 legacy key shape 在论文中的标注；round-robin 既有测试失败定位。

## Environment / Recovery

- **执行机（Windows，正式实验）**：解释器 `D:\anaconda\envs\scheduler\python.exe`；`PYTHONPATH=src`；RTX 3060 Laptop。
- **本机（macOS，控制面/代码/轻量验证）**：项目在 `/Volumes/Lenovo/scheduler`；
  可用 `PYTHONPATH=src /opt/miniconda3/bin/python`（3.13，无 numpy/torch）跑**纯 stdlib 的模拟器与单测**；
  **不下载** Windows 的 GPU/模型实验。
- **公开镜像**：`/Volumes/Lenovo/scheduler_public_repo` → `github.com/QiuweiLiu/scheduler`；staging 脚本 `.scratch/stage_public_repo_v2.ps1`（源自 `F:\scheduler`）。
  注意：该脚本复制后会在工作区留下大量 **CRLF/BOM-only** 差异（34 个文件，改变行数对称），提交前需只 stage 目标文件。
- **ChatGPT 桥接（本机已验证）**：
  `"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --remote-debugging-port=9222 --user-data-dir="$HOME/Library/Application Support/Google/Chrome-Automation" https://chatgpt.com/`
  —— 必须用 `Chrome-Automation` profile（已登录；`~/.chrome-chatgpt-bridge` 是空 profile，会落在登录页）；
  该 profile 与 `chatgpt-automation` MCP **共用**，两者不可同时运行。
  本机**直连** chatgpt.com 可用，**不需要** `--proxy-server=7897`（那是 Windows 的坑）。
  绑定会话：`https://chatgpt.com/c/6a9822da-e278-83e9-9c1a-675923acda0e`（标题「架构设计评估」）。
