# Phase 0 — VideoSeek 仓库与原生运行审计

## 结论

已在一台 AutoDL RTX 4080 SUPER 实例上完成一条原生 VideoSeek 任务。运行正常退出，原生 `prediction.json`、`trajectory.json`、完整 stdout/stderr 与 GPU 快照均可复查。此次结果证明了 **VideoSeek → LiteLLM → Qwen → 工具调用** 路径可运行；它不证明答案准确率、路径动态性或本地 GPU 调度价值。

## 受审计对象

| 项目 | 已验证值 |
|---|---|
| 官方仓库 | `https://github.com/jylins/videoseek.git` |
| 固定 commit | `443f8710bd1f9023f4a52ac95cb83610ad29c5ac` (`Initial commit`, 2026-03-23) |
| 软件许可 | MIT（仓库 `LICENSE`） |
| 远端 checkout | `/root/autodl-tmp/scheduler/third_party/videoseek` |
| 工作树 | 干净，未修改上游源码 |
| Python 环境 | `finetooling`, Python 3.10.20 |
| 关键包 | VideoSeek 0.1.0、LiteLLM 1.94.0、OpenAI 2.37.0、Decord 0.6.0 |
| GPU | NVIDIA GeForce RTX 4080 SUPER, 32760 MiB, driver 595.58.03 |

## 数据与网络边界

样本为 VideoSeek README 所示的 LVBench 视频 ID `wgBlACG927Y`。LVBench 的官方元数据和 VideoSeek README 给出该视频/问题入口；原始媒体由公开视频源提供。只下载了一条视频，未下载完整数据集。

远端对 Hugging Face 与 YouTube 的直连不可用或不稳定，因此通过临时、仅回环可见的 SSH 反向隧道使用本机 HTTP 代理。一次 720p 尝试因吞吐过低中止；为完成最小原生运行，下载了同一视频的 144p H.264/AAC 版本。文件属性为 256×144、25 fps、2331.423 秒、39,024,195 字节；SHA-256 为：

```text
adf60db34dad3d81b0a091045e356b1722c6d32d0139a5c2db2e4fee0f7e76e7
```

该分辨率选择是网络与存储约束下的 Phase 0 最小化取舍，**不得用于准确率、视觉细粒度能力或资源画像结论**。

## 已验证的原生入口

`videoseek-cli` 需要本地视频路径与问题文本，并会在 `--output_dir/<video_id>_<timestamp>/` 写出 `prediction.json` 和 `trajectory.json`。

实际运行使用以下等价的无密钥配置（密钥只通过单次进程环境变量传递）：

```bash
export OPENAI_API_BASE='https://dashscope.aliyuncs.com/compatible-mode/v1'
export OPENAI_API_KEY='<in-process only; never persist or print>'
export LITELLM_DROP_PARAMS=True

/root/miniconda3/envs/finetooling/bin/videoseek-cli \
  --video_path /root/autodl-tmp/scheduler/data/phase0/lvbench/wgBlACG927Y.mp4 \
  --user_query '<LVBench question recorded in configs/phase0_videoseek_qwen3_vl_plus.yaml>' \
  --output_dir '<new run directory>' \
  --verbose \
  --model_name openai/qwen3-vl-plus
```

`LITELLM_DROP_PARAMS=True` 是运行时兼容设置，用于避免 LiteLLM 将 VideoSeek 固定传入的 `reasoning_effort` 作为 Qwen 不支持的参数拒绝；没有修改任何上游文件。

## 本次运行与原始证据

| 项目 | 值 |
|---|---|
| 远端运行根目录 | `/root/autodl-tmp/scheduler/results/raw/phase0/videoseek_wgBlACG927Y_20260731T003811+0800` |
| 退出状态 / 耗时 | `0` / 289 秒 |
| 原生预测 | `D` |
| 轨迹 | 7 步，`finish_reason=stop` |
| 工具序列 | `overview → skim → focus → skim → skim → focus → answer` |
| stdout/stderr | `stdout_stderr.log`, 48,745 bytes |
| 原始轨迹 | `videoseek_output/wgBlACG927Y_1785429495/trajectory.json`, 49,622 bytes |
| 原始预测 | `videoseek_output/wgBlACG927Y_1785429495/prediction.json`, 23 bytes |

完整性哈希：

```text
fd2c49f6cdd328c0af8f563154e93e2ab74afceacbb906be214ac07feb5d74a6  stdout_stderr.log
2131a2982620e2c648c7ddf89ba5c10ceb550f1bcbd3c2d13cc08f435bd389b9  prediction.json
c759055d6cf28bc647a04dc265b18dfb2ecc34e6976fa37a1abfdaee86052ecb  trajectory.json
```

轨迹顶层字段为 `question`、`steps`、`final_answer`、`finish_reason`、`total_steps`。每个步骤包含 `step_id`、`thought`、`action`、`observation`，能够作为 Trace Adapter 的原始基线。

## 实际执行路径与资源事实

本次路径由 Decord `VideoReader` 读取本地 MP4；`overview`、`skim`、`focus` 在 CPU 上抽帧、拼图、JPEG/Base64 编码，随后经 LiteLLM 调用远端 Qwen；`answer` 也是远端 API 调用。

运行前后 `nvidia-smi` 都为 1 MiB、0%，且运行中未观察到本地 GPU 计算进程。因此：

- 此运行的端到端时间应拆分为 CPU 解码/打包、外部 API 等待和少量本地开销；
- 不得把 API 等待时间记为 GPU 运行时间；
- 该工作负载尚未证明本地 GPU 资源异构性，不能直接支持多 GPU 调度收益主张。

## 推荐的最小 Trace Adapter hook 点

保持第三方源码不变，在项目外层 wrapper 中做运行时包装：

1. 在 `VideoSeekAgent.run()` 外记录任务级 manifest：代码 commit、输入文件哈希、环境、GPU、模型、种子和启动/结束时间。
2. 在实例的 `_VideoSeekAgent__exec_action` 外记录每个工具节点的 `action`、输入参数、单调时钟起止、状态、观察摘要和异常；这是最小侵入的逐节点边界。
3. 在外部 wrapper 中分别包装 `videoseek.agent.call_llm_api` 与各工具模块的 `call_llm_api`，将 API 等待单列为 `api_wait_ms`，不伪装成 GPU runtime。
4. 在 action 前后采样 `nvidia-smi`/NVML、进程 RSS 与 Decord 解码时间；仅在真实 GPU 工具出现时记录 CUDA allocated/reserved。

上述方案为运行时外层 hook，不要求编辑 `third_party/videoseek`。

## 已知限制与下一步

- 144p 输入使小物体与文字证据不可靠；本次答案不应用于准确率评测。
- 当前 Agent 的本地执行没有实际 GPU 推理；若 Phase 2 仍无 GPU 异构性，应按指导书将 VideoSeek 降级为 trace 采集补充，并转向具有真实本地视觉模型的工作负载。
- Phase 1 的下一步是实现外层 Trace Adapter v0.1，并在不改 Agent 决策逻辑的前提下采集 3 条完整 trace；在此之前不要开始预测器、模拟器或真实多 GPU 部署。
