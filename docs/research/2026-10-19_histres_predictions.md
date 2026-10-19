# 最新预测器(histres F0/F1)在调度数据集上的预测结果(2026-10-19,远端 RTX 4080 SUPER)

## 背景与模型选择

- 目标:用**最新的预测器**在调度数据(train + validation 来源)上跑预测、保存逐行预测(数据集更新组件)。
- 关键发现:2026-09-11 的 **J3(joint_resource)checkpoint 与当前代码不兼容**(模型已演进出 histres 层:`hist_res_proj`/`hist_status_emb`);**最新预测器线 = `EXP-20260921_histres_causal_input_v1`(F0/F1)**,其 checkpoint 与当前代码匹配。
- F0 = 无历史遥测通道;F1 = 含因果历史遥测通道(telemetry_grad 0.029~0.085,路由守卫通过)。

## 执行(远端)

- 设备:RTX 4080 SUPER 32GB;环境 torch 2.13.0+cu130 / transformers 5.17.0;
- 数据:`j_series_dataset_histres_v2`(histres_train 13754 行 / histres_validation 2029 行);**sealed J test 从未读取**(j test loaded: False);
- 预测:F0/F1 × seeds {11,22,33} × {train, validation} → 12 个逐行预测 JSONL(含 next_role/next_family/内容字段预测 + runtime q50/q90/q95 + load q50/occ + termination 概率)+ `predict_report.json`;
- 脚本:`scripts/histres_predict_splits.py`(远端执行,可复现)。

## 结果(validation;train 同表见报告)

| 指标 | F0(3 种子) | F1(3 种子) | F1−F0 |
|---|---|---|---|
| next_role_acc | 0.9433~0.9438 | 0.9433~0.9438 | ≈0.0000 |
| next_family_acc | 0.8425~0.8454 | 0.8425~0.8454 | ≈0.0000 |
| content_accuracy | 0.8980~0.8993 | 0.8980~0.8993 | ≈0.0000 |
| runtime_qscore | 599.0~605.9 | 599.8~605.8 | ±0.7 内 |
| load_dur_qscore | 230.7~255.3 | 229.9~254.8 | ±0.8 内 |
| load_brier / termination_bce / length_mae | — | — | ≤0.002 |

**结论**:①最新预测器在调度数据集上的质量:next_role ≈ 94%、next_family ≈ 84%、内容 ≈ 90%(与训练报告一致,复算吻合);②**F1−F0 ≈ 0:因果历史遥测通道在这些指标上无增量**;③与旧结论一致——预测器质量不是瓶颈(此前的消费实验显示:即使完美未来也不兑现)。

## 交付物

- `artifacts/predict_splits_report.json`:全指标(每 arm × seed × split)+ 词表;
- `artifacts/predict_splits_jsonl.tgz`:12 个逐行预测 JSONL(39MB 原始 / 2.4MB 压缩)——即"更新后的数据集"组件(可直接 join 回 histres 数据集行);
- 备注:如需把逐行预测 join 回正式工作负载的 future_h5 产物(Burst64),需下一步的格式转换(另议)。

## 追加:调度数据集 train + **test** 预测(2026-10-19 晚,用户要求)

- **口径**:histres F0/F1 × seeds {11,22,33} × {train(13754), **test(1520)**} → 12 个逐行预测 JSONL + 报告;
- **封存声明**:这是 histres 线 **j_test 的首次打开**(此前 f0f1 报告声明 "J test remains sealed";本产物用途 = 数据生产,按用户指令执行;后续若要用 test 做确证性 claim,须单独申请裁定);
- 脚本新增显式开关 `--allow-test`(不传则拒绝读取 test)。

**test 指标(n=1520)**:

| 指标 | F0(3 种子) | F1(3 种子) |
|---|---|---|
| next_role_acc | 0.9335~0.9351 | 0.9335~0.9351 |
| next_family_acc | 0.8570~0.8578 | 0.8570~0.8578 |
| content_accuracy | 0.8815~0.8821 | 0.8815~0.8820 |
| runtime_qscore | 667.3~672.1 | 667.4~672.1 |

- train 指标与之前一致(role ≈ 0.966,fam ≈ 0.887~0.890);F1−F0 ≈ 0(与 validation 结论一致);
- 产物:`artifacts/predict_splits_train_test_report.json`、`artifacts/predict_splits_train_test_jsonl.tgz`(12 JSONL,2.4MB 压缩)。
