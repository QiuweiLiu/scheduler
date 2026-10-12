# 实验 I:DES 对齐的高预算 Rollout(规划 DES)

- **日期**: 2026-10-12
- **状态**: 工程验证门 1/2 通过;确定性管线构建中
- **实验 ID**: `EXP-20261012_des_aligned_rollout_v1`
- **背景**: 顾问方案"DES-Aligned High-Budget Rollout"。修复此前简化沙盘(exp G/H)与真实 DES 不一致的病根:
  规划世界必须复用真实仿真器的事件推进/显存准入/加载/干扰/驱逐规则,后续策略固定为主线(pdrs_resident)。

## 设计

### 规划 DES = 真实 DES + 状态注入
- **状态快照**(`serialize_des_state`):在真实运行的主循环迭代开头捕获完整调度状态:
  jobs(node_state/ready_since/started/completed/counters/observed 等)、gpus(resident/active_tasks/prefetch_pending/counters)、
  finish_heap、ready 堆、未到达队列、active_jobs、wait_logged、sequence/rr_cursor、now。
- **注入**(`apply_injected_state`):规划运行的 episode 携带 `injected_state`,仿真器在初始化时装入快照
  (仅当该键存在;默认路径不变)。
- **场景**(`build_scenario_templates`):把"未来执行真值"替换为场景 —— 未启动节点的 model_id/runtime_ms/load_ms/
  workspace 等属性被替换;超出场景视界的尾节点被裁剪;链式因果顺序不变(工作负载为纯线性链,已验证 8126 节点 0 分叉)。
- **规划运行**:`run_planning` 用 `des_rollout` 策略:第一个派发决策执行强制动作
  (排序 node_key/gpu + 预取目标 + 驱逐模式,经 `policy_context["des_action"]`),其后全部委托主线
  (pdrs_resident),保证续跑非预知且对所有候选动作一致。
- **成本**:`future_flow_cost` = 快照时刻之后的未来流转时间(等待面积),与论文指标 mean(finish−arrival) 同口径。

### 工程验证门(顾问要求,先于扩大场景数)
- **门 1(机制对齐)**:真实运行 + 中途快照 → 注入快照重放(真实未来、真实到达、真实模板)→ 从捕获点起
  **逐事件一致**(event_type/time/job/node/gpu 及全部载荷;仅剔除 episode_id 后缀与 decision_index 两个惰性记账字段)。
  - 结果:**PASS**,4 案例(train_000000/1 × 捕获点 0.3/0.6),624/446/728/558 事件全一致。
- **门 2(隔离性)**:把首次决策观测集之外的未启动节点执行真值替换为随机模型/时长(含裁剪变体),
  首次派发决策必须不变;同时后续事件流必须确实改变(非空测试)。
  - 结果:**PASS**,12 案例(3 集 × 2 随机种子 × 2 变体);首次决策 (job,node,gpu) 全同,后续事件改变 146–719 条。
  - 备注:测试设计中发现并修正一个测试缺陷——随机化必须排除"首次决策时已可见"的节点(其身份已进入观测),否则测的不是隔离性。

### 修复的真实缺陷(实现过程中)
1. 快照恢复时**非活跃任务被当作活跃任务**(GPU active_task_count 超限)→ 只序列化 `task.active` 的任务。
2. 规划运行误触发**初始化期预取**(真实运行在 now=0 空状态调用,规划运行在注入状态上调用)→ 注入运行跳过该初始化调用。
3. 捕获点从"迭代中部"移到**迭代开头**(否则真实运行在同一迭代内的循环顶部预取事件被记在捕获之前,造成事件流错位)。

## 下一步

1. 场景生成器:确定性版(产物贪婪链 → 场景属性;资源=逐步预测值/统计)。
2. 场景生成器:概率版(GRU J3 自回归采样;需 torch,安装中)。
3. Rollout 驱动:动作空间(排序×预取×驱逐,含消融)× 共同场景 → (s, a, Ĝ, 不确定性)。
4. 终端价值:V=0 / 剩余工作量公式 / 学习 V̂(标签用训练集真实延续) / 完整延续参照。
5. 分级预算:smoke 20×8 → 收敛 100×{32..256} → 标签 500–1000 → confirm300 最终评估。
