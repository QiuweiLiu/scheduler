# EXP-20261010_future_info_counterfactual_v1 — 未来信息消除反事实(实验 D)

**状态:** 设计冻结(运行前) · 2026-10-10
**动机(顾问清单):** "构造单模型、全驻留等反事实场景,观察你的驻留策略与普通策略的差距是否下降——
收益是否真正依赖未来模型需求?"(纯 CPU 仿真反事实,非新硬件测量)

## 1. 三格设计(同一 confirm300,id 保留逐集配对;α=1.0)

| 格 | 变换 | 预期 |
|---|---|---|
| `c_cap3232`(参照) | α=1.0 + 容量钉死 (32760,32760)(= 实验 C 的 a100_cap3232,用于自身一致对账) | 主−F0 ≈ −920 |
| `c_allres`(全驻留) | 容量钉死 **(131072,131072) MB** → 所有模型一次装载后常驻,驱逐≈0 | 主−F0 向 0 收缩 |
| `c_single`(单模型) | **所有 LLM GPU 节点重标为 `Qwen3-VL-8B-Instruct`**(8B 角色覆盖是 3B/4B 的超集:answer_generation/planner/videotool_spatial);内存/加载归一为 8B 记录;episode 初始驻留提示合并去重;**pack 的 LLM 身份质量全部并入 8B**(CPU/finish 质量保留);yolo 保留(512MB,非切换部署) | 主−F0 向 0 收缩 |

- 反事实只消除"模型身份/切换"维度;**runtime 预测与链长分布保持原记录**(披露:
  不改变工作量,只改变身份结构)。契约检查:c_single 的 (8B,role) 全部在覆盖集内(已验证)。
- 臂:myopic / F0 / 主线 `pdrs_resident`(3 格 × 3 臂)。

## 2. 预注册判据(冻结)

- 格内:Δ = 臂 − 同格 F0(配对 bootstrap 2000/seed 11;零失败硬断言);
- **跨格增益变化(头条)**:`Δgain = (主−F0)_格 − (主−F0)_c_cap3232`,逐集配对;消除主张 =
  点估计为正(增益收缩)且 CI 支持/至少方向一致;**若 c_allres/c_single 的增益不降,则"收益依赖
  模型切换/驻留压力"的机制解释被证伪**,如实报告;
- 机制审计:驱逐/冷加载/重载/预取计数在反事实格的变化(全驻留格驱逐→≈0;单模型格
  驱逐/重载大幅下降)必须与主张一致。

## 3. 披露与成本

- 容量 128GB 是纯反事实上限;单模型重标保留原 runtime/load 记录(上限近似);yolo 不参与 LLM 单模型化;
  episode id 保留(cross-cell 配对);confirm300 复用(不引入新方法)。
- 成本:~3 格 × 3 臂 × 300 集 ≈ 12 分钟(本地 CPU)。

## 4. 产物

- `experiments/EXP-20261010_future_info_counterfactual_v1/{RESULT.md,artifacts/counterfactual_v1.json}`;
  运行器 `scripts/future_info_counterfactual.py` + 变换库 `scripts/counterfactual_lib.py`。
