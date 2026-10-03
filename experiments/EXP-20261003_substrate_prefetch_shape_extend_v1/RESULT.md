# RESULT — EXP-20261003_substrate_prefetch_shape_extend_v1（步4：F4 形状扩展）

**状态**：完成（2026-10-03）。**远端脚本 `f4_shape_extend.py`；产物 `artifacts/f4_shape_extend.json`。**

## 目的

F4 原有 6 对 × medium 档。真实 workload 跨模型转移频率（0-GPU 分析 v04 模板）：
仅 **4B→3B（7.5%）与 3B→4B（1.0%）**（8B→8B 54.7%、4B→4B 29.9% 同模型不触发加载）。
本实验补 top-2 转移对 × {XS, XL} 四格，检验"加载干扰"的**形状外推**。

## 协议

ordered pair（A 推理 ∥ B 加载）；variant copy（纯拷贝，下界）与 full（完整框架加载，上界）；
指标 slowdown = overlap_infer/solo_infer；dilation = overlap_load/solo_load；reps=3 + 1 warmup。

## 结果

| 格子 | in/out | solo | copy (slow/dil) | **full (slow/dil)** |
|---|---|---|---|---|
| 4B[XS] ← load 3B | 72/4 | 163 ms | 2.62 / 1.00 | **18.58 / 1.08** |
| 4B[XL] ← load 3B | 4104/512 | 15231 ms | 1.03 / 1.00 | **1.18 / 1.09** |
| 3B[XS] ← load 4B | 83/4 | 117 ms | 3.48 / 1.00 | **21.58 / 1.05** |
| 3B[XL] ← load 4B | 4115/512 | 11864 ms | 1.04 / 1.00 | **1.30 / 0.92** |

## 结论

1. **加载干扰强烈依赖推理任务的时长**：full-load slowdown 从 XS 的 **18.6–21.6×**
   降到 XL 的 **1.18–1.30×**。**constant-factor（medium 的 ~1.7）不可外推**。
2. 机制解释：加载（~3–7s）对毫秒级任务是"几乎全阻塞"；对十秒级任务只摊薄掉一小段。
3. 加载侧 dilation ≈ 1.0（一方向性干扰），与既有 F4 一致。
4. copy 变体（纯带宽）在小任务上也造成 2.6–3.5× slowdown —— 说明干扰不仅是框架开销，纯拷贝也阻塞。
5. **Hermes 类预取决策需要按任务时长建模**：短任务旁不宜并发加载（得不偿失）；长任务旁预取代价 <10–30%。

## 边界

- 只测 top-2 真实转移对；XL=4096/512；3 reps（screening 级）；单卡共享环境。
- 与 medium 档的对比：medium full slowdown（既有表）≈1.7，本扩展给出形状依赖的两端。
