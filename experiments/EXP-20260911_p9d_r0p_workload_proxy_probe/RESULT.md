# R0.1 RESULT — workload proxy probe（现有字段能否补出"工作量大小"）

实验：`EXP-20260911_p9d_r0p_workload_proxy_probe`
问题：现有 P9d trace 里已存在的工作量代理字段（工具请求长度、帧数、题目大小、视频时长/分辨率）能否恢复"工作量大小"信号？
协议：复用 R0 的节点本体与估计器族；`P_dev/train` 拟合、`validation` 选型、`test` 诊断；**冻结 holdout 未使用**（R0 已消费；特征变更后旧 holdout 不再具备确证身份）。
运行：本地 CPU 63.3 s；LightGBM 4.7.0。

## 1. 数据覆盖（15,481 节点）

| 代理字段 | 可用节点 | 覆盖率 |
|---|---:|---:|
| `tool_input_len` / `frame_count_before` / `frame_count`（工具事件） | 7,078 | 45.7%（全部工具节点） |
| `question_chars` / `question_tokens` / `option_chars_mean` / `option_count` | 11,889 | 76.8% |
| `video_duration_s` / `video_width` / `video_height`（provenance） | 14,939 | 96.5% |

## 2. 结果（runtime；PB_primary 越小越好）

| 特征集 | validation PB | test PB | 相对 F0 |
|---|---:|---:|---|
| F0 = R0 的 C4（对照） | 380.0 | 261.1 | — |
| F1 = F0 + trace 代理（请求长度/帧数） | 384.6 | 258.0 | val −1.2%，test +1.2% |
| F2 = F1 + 题目大小 + 视频元数据 | 386.0 | 278.9 | val −1.6%，test −6.8% |

**结论：没有有意义的改善。** 差异在选型/噪声范围内，且方向不一致（F1 在 test 略好、val 略差；F2 两侧都不好）。

模型确实使用了这些字段（F2 的 gain：`tool_input_len` 5,161、`video_duration_s` 3,210、`question_chars` 2,538、`option_chars_mean` 2,412，仅次于 `exec_class` 13.5万与 `prefix_model_reuse` 8,417），但它们与已有特征高度冗余，**没有把可辨识性往上抬**。

## 3. 解释与结论

1. **"工作量大小"不能靠现有记录补出来。** 请求文本长度、帧数（本池几乎恒为 3–4）、题目大小、视频时长/分辨率都不构成一阶解释变量。
2. 真正缺的仍是**模型侧一阶量**：实际 prompt/completion token 数、图像/张量尺寸、批处理/并发状态。这些当前 trace 未记录，只能通过给采集器加字段并重采获得。
3. 另一种同样成立的解释：在 `exec_class + 运行上下文` 已给定的条件下，剩余 runtime 方差本来就以噪声/不可见因素（含 2 个 stall）为主——即便补上一阶字段，增量也可能有限。

## 4. 建议（需用户决策）

- **A. 接受当前边界，进入 J 系列**（runtime 为主、load 为辅；memory/tail 不作可辨识声明）。成本最低，符合 R0 已通过的裁决。
- **B. 设计"插桩重采"议题**：给 worker/collector 增加 token/图像/批处理字段，重新采集一批 trace。前置问题：P_dev 的 300 个视频已删除（需重下约 30GB 或新建视频池），且新特征需要新的 dev/holdout 划分与独立验收协议。这是一个独立的中期项目，需单独批准。
- **C. 折中**：先只做 B 的"采集器改造 + 小规模 pilot（用现存 holdout/调度器池之外的可行视频）"，验证 token/图像字段是否真的提升可辨识性，再决定是否大规模重采。

## 5. 复现

```sh
python scripts/r0p_workload_proxy_probe.py --config experiments/EXP-20260911_p9d_r0p_workload_proxy_probe/config.json
```

产物：`metrics.json`、`stage0_audit.json`、`run_manifest.json`、`artifacts/joined_table.jsonl.gz`（哈希见 run_manifest）。未修改任何已冻结数据集；holdout 未使用。
