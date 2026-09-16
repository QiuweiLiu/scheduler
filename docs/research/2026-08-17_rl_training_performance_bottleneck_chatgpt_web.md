---
research_backend: ChatGPT Web
requested_model: GPT-5.6 Sol
requested_reasoning: High
verified_before_submit: true
verified_after_response: false
timestamp: "2026-08-17 19:15 +08:00"
status: incomplete
reason: post_response_model_state_check_timeout
---

# RL 训练性能瓶颈：网页版复核

## Executive Summary

这次网页版复核支持当前本地诊断，但不能替代远端基准实验：

1. 六个进程各自启用 64 个 intra-op 和 64 个 inter-op 线程，最可能造成线程池过度订阅和上下文切换；这是解释“并发后明显变慢”的第一嫌疑。
2. 模拟器主要是 Python/CPU 事件循环，单进程的约 15.1 秒/episode 很可能是长期的结构性瓶颈；多进程不会共享同一个 GIL，但每个进程仍可能主要只使用一个 Python 执行流。
3. 每个 episode 约 335 次决策，逐决策保留小 autograd 图再在 episode 末尾反向传播，可能让 6.5 秒 backward 中大部分时间花在 Python/Autograd 调度，而不是矩阵计算。
4. 当前小策略网络（约 1.86 万参数）不值得先把每个 decision 搬到 GPU；CPU→GPU→CPU 的频繁同步很可能更慢。只有在 rollout 无梯度、批量 learner 结构稳定后，才值得做 GPU batch 对照。

## Root-cause ranking

| 排名 | 判断 | 说明 |
|---|---|---|
| 1 | VERIFIED + INFERENCE | PyTorch/OpenMP/MKL 线程过度订阅，优先解释六进程并发退化。 |
| 2 | VERIFIED + INFERENCE | Python 事件循环解释单进程 simulation wall-time；不是六个进程争一个 GIL。 |
| 3 | INFERENCE（需验证） | 335 个 tiny autograd graph 在 episode 末端统一 backward，可能产生大量调度开销。 |
| 4 | INFERENCE | cache、内存带宽、NUMA、上下文切换等一般进程竞争，在线程修正后再判断。 |
| 5 | VERIFIED（非根因） | GPU 未使用是当前架构现象，不足以解释 CPU 并发退化。 |

## 建议的最小 A/B 矩阵

先不改变 RL 算法：

| 组 | 进程数 | intra/inter | OMP/MKL | rollout 方式 | 目的 |
|---|---:|---:|---:|---|---|
| B0/B1 | 1/6 | 64/64 | 当前 | 当前实现 | 原始基线 |
| T1/T2 | 1/6 | 1/1 | 1/1 | 当前实现 | 验证过度订阅 |
| T3/T4 | 6 | 2/1、4/1 | 同步设置 | 当前实现 | 找到小网络的线程甜点点 |
| C1 | 1/2/3/6 | 最佳线程设置 | 同步设置 | 当前实现 | 分离进程竞争与线程竞争 |
| G1/G2 | 1/6 | 最佳线程设置 | 同步设置 | rollout `no_grad`，episode-end 批量重算 log-prob | 验证 autograd 图是否主因 |
| D1/D2 | 1 | 最佳线程设置 | 同步设置 | decision 级 GPU / 批量 GPU learner | 只在前两轮后判断 GPU 是否有收益 |

每个 cell 应记录 episode/simulation/forward/backward/optimizer-step wall-time、decisions/s、nodes/s、RSS、CPU 利用率、线程数和上下文切换；GPU cell 额外记录 GPU 利用率、H2D 时间/字节和显存。

## 关键正确性约束

不要直接对 `log_prob` 做 `detach()`，这会切断 policy gradient。更安全的路线是 rollout 阶段只保存 observation、候选动作、选中动作、reward、mask 和重建输入所需 metadata；episode 末尾把 observation 堆成 batch，重新计算 log-prob，再做一次 loss/backward。必须用同一条 trajectory 比较 old/new 的 loss、gradient norm、参数梯度 cosine 和 optimizer step 差异。

## 慢、死锁、活锁、OOM 的判别

- 慢：episode/decision 计数持续增长，CPU time 增长，RSS 大致稳定，只是 wall-time 高。
- 死锁：计数不变，CPU time/利用率接近不变，进程仍存活，可能在等待锁、队列、I/O 或 join。
- 活锁：计数不变但 CPU 持续消耗，说明在重复工作而没有进展。
- OOM：RSS 单调增长后进程消失或出现 CUDA/系统内存错误。

当前日志每 250 episode 一次会放大“像卡住”的错觉；至少增加 episode/decision heartbeat 和轻量 timing。

## 推荐顺序

1. 固定 `torch`、OMP、MKL 的线程数为 1/1，先做 P=1 与 P=6 对照；再测试 2、4 个线程。
2. 固定最佳线程设置，测 P=1/2/3/6 的 aggregate decisions/s 和 p95 episode latency。
3. 对一条完全相同的 trajectory 做 rollout-no-grad + episode-end batch recompute，并先验证梯度等价。
4. 只有当 learning 仍占总 wall-time 的明显比例时，再做批量 GPU learner；不优先做每 decision 的 CPU↔GPU 往返。

## 证据状态与限制

- `VERIFIED`：网页版在发送前可见地显示 `GPT-5.6 Sol` 与“高”；脱敏 brief 已发送并收到报告。
- `INFERENCE`：以上根因排序、线程甜点点和 GPU 是否有收益，仍需远端 A/B 实验确认。
- `UNVERIFIED`：当前 RL-0/RL-H5 的精确 loss 是否能完整重建；是否存在 RNN hidden state、dropout、batch-dependent normalization 或隐式中间张量依赖。
- `status=incomplete`：分析完成后页面网络请求持续超时，未能再次读取模型菜单确认 `GPT-5.6 Sol + 高`，因此没有宣称后验模型核验通过。

## 网页版报告引用的来源类别

PyTorch multiprocessing/CPU oversubscription、PyTorch threading environment variables、Intel oneMKL threading、Python multiprocessing/threading 与 GIL、PyTorch Autograd Mechanics、PyTorch Profiler、PyTorch CPU↔GPU transfer、REINFORCE 原论文、IMPALA 原论文。网页版返回了来源标签，但当前未逐条提取稳定 URL；重要结论仍需在本地或远端用官方文档独立核验。
