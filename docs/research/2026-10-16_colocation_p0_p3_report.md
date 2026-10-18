# 共驻阶段 P0–P3 报告(2026-10-16,按顾问 A1 终审指令执行)

## 状态

- **A0 PASS**(2935531);**A1a PASS**(查表 11/11 精确);**A1b FAIL**(终点 4/11,长边结构性低估 −8.7~−113.0ms,全部为负);
- **A2/A3/B 封锁中**;不修改正式 DES、不替换 sticky 语义、不动 v7 与下游基线;
- 本报告 = P0(逐轮数据审计)+ P1(测量协议审计)+ P2(语义诊断)+ P3(补测协议)。

## P0:原始逐轮数据审计(`scripts/colocation_p0_round_audit.py`)

证据:`artifacts/colocation_p0_round_audit.json`(Phase A 11 格逐轮 + Phase B 12 探针概要)

1. **逐轮数据存在**(顾问纠正成立):每格 solo/coloc 各 3 轮(例:8B[answer] solo 180.3/179.1/178.1,coloc 319.8/320.8/319.9);
2. **结构性偏差数学核对 11/11 OK**:piecewise 偏差 ≡ −(sL−1)(TL−tS/sL)(逐格误差 <0.5ms)→ 负号是数学必然,非新物理证据;
3. **长边 piecewise 预测 0/11 落入物理逐轮区间**:多数格偏差远超声速轮间离散度(如 P-2 格:离散 0.4%≈10ms vs 偏差 −113ms);最小偏差格(4B|3B[videotool] −8.7ms)与离散度同量级;
4. 短边 11/11 恒等(构造使然);coloc 离散度:多数 0.4–1.3%,个别 3.1%/5.8%/13.8%。

## P1:原始测量协议审计(`scripts/real_strata_calibration.py`)

1. **无共同启动屏障**:共驻轮为 `wa.go(); wb.go()` 顺序下发(L268-269),两 worker 各自收到 go 后才计 t0,GPU 实际起点存在未测偏移;
2. **wall_start/wall_end 被丢弃**:worker 已返回该两字段(L146),主程序只保留 `["ms"]`(L270-271);
3. artifacts 目录无原始运行日志(仅 4 个 JSON);
4. **结论:实际重叠区间不可重建**;不得据此推断退出后速率。

## P2:候选语义诊断(`scripts/colocation_p2_semantics_diag.py`)

证据:`artifacts/colocation_p2_semantics_diag.json`

1. 同步启动:piecewise 长边 −8.7~−113.0ms;sticky ≈0.0ms——**但 sticky 零偏差是定义恒等式**(=solo×slowdown=median coloc),**非校准证据**;
2. 晚加入网格:两模型**仅在"同伴先于长任务退出"时发散**;同伴在长任务结束后加入→两模型与无重叠一致(负对照);
3. **sticky 对"连续换伴"(A+B 后 A+C)无实测规则**;11-cell 表不含链式共驻/偏移数据;
4. 结论:两种语义在同步启动下不可区分(都能"重现"原表);辨识必须靠偏移启动补测。

## P3:最小物理辨识协议(仅规划,不租 GPU)

`docs/research/2026-10-16_colocation_p3_remeasure_protocol.md`

- 3 组合(P-1 4B+8Banswer;P-2 8Bplanner+4Bplanner;P-3 8Bplanner+3Bspatial)× 4 条件(T0 共同屏障 / T1 短晚加入 / T2 中段加入 / T3 无重叠负对照)× ≥3 轮;
- 必须新增:保存 wall 时间戳、共同启动屏障、逐轮重叠区间;建议逐 token/分段进度时间戳(直接观察退出前后速率);
- 拟合(T0)与检验(T1-T3)严格分离;三者都不能解释→记录不可辨识,不强行选模型;
- 同硬件不可得时:共驻阶段冻结为"物理干扰已测量、动态执行语义未校准"。

## 下一步

将 P0–P3 提交顾问(下一轮语义裁定);在此之前 A2/A3/B 维持封锁,Arrival-Holdout 继续封存。
