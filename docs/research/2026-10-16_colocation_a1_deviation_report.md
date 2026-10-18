# 共驻阶段 A1 偏差报告:分段 slowdown 语义未复现物理终点(2026-10-16)

## 结论

**A1a(查表)PASS:11/11 组合经真实 `colocation_slowdowns()` 双向查表精确匹配(1e-9)。**
**A1b(终点)FAIL:仅 4/11 组合在预注册容差内(max(3%, 30ms))。**

## 测量(11 对微型双任务,用模拟器真实方法:add_active_task/set_task_slowdown/update_task_progress)

| 组合 | 长边偏差 | 相对 |
|---|---|---|
| 8B[planner]\|4B[planner] | **−113.0ms** | −4.5% |
| 8B[answer]\|4B[planner] | −103.0ms | −6.3% |
| 8B[videotool]\|4B[planner] | −99.5ms | −3.9% |
| 8B[videotool]\|3B[answer] | −78.3ms | −3.5% |
| 8B[answer]\|3B[videotool] | −75.7ms | −5.6% |
| 8B[planner]\|3B[videotool] | −71.1ms | −3.0% |
| 4B[planner]\|3B[answer] | −62.5ms | −3.9% |
| 8B[planner]\|3B[answer] | −55.5ms | −2.6% |
| 8B[videotool]\|3B[videotool] | −53.2ms | −2.3% |
| 8B[answer]\|3B[answer] | −21.7ms | −8.8% |
| 4B[planner]\|3B[videotool] | −8.7ms | −0.5% |

**方向系统性:全部为负(长边被低估),无一为正;短边恒精确(构造恒等)。**

## 诊断

1. **实测两端点均精确等于 `solo × slowdown`**(表的 slowdown 字段按定义 = coloc/solo,两端点同时满足)→ 物理语义表现为:一旦共驻,**双方各自按 slowdown 全程延展,同伴离开后无可见加速**。
2. 当前 DES 语义:同伴完成时幸存者**重置 slowdown=1**(`workload_v02_simulator.py` L8962),即"分段 slowdown"→ 长边终点 = t₁+(solo_long−t₁/slow_long) < solo_long×slow_long,**结构性低估**。
3. 两种解释无法由冻结的中位数分辨:
   - (a) 真实物理存在**持续延展**(离开后不恢复全速);
   - (b) 探针方法学把"重叠期测得的 slowdown"报告为整请求延展时间(原始三轮数据不在冻结产物内)。
4. 候选修复语义:**共驻后 slowdown 持续到各自完成(不做 survivor reset)**——对同时启动可精确复现两端点;但对"晚到候选"场景的语义需另行裁定。

## 处置(按顾问指示)

- **不调整容差、不替换物理表、不简化语义来"通过"**;
- A1b 标记 **语义校准失败**;阶段 A **暂停**(A2/A3 与阶段 B 均不动),提交本偏差报告请求裁定:
  1. 是否将模拟器共驻语义改为"持续延展"?(涉及 v7 配置语义与全部下游结论的再现性,须单独批准)
  2. 或先补测/索取原始三轮数据以分辨 (a)/(b)?
- 全部证据:`artifacts/colocation_a1_micro.json`(含逐组合预测/实测/偏差/诊断);脚本 `scripts/colocation_a1_micro.py`(可复现)。
