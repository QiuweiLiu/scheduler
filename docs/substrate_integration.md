# 模拟器实测数据接入：实现范围与验证

本次是工程接入，不是新一轮正式实验，也不代表全部机制已经实现或真实系统回放已通过。

## 单任务与双任务使用同一套记录

`GPU.active_tasks` 是显卡占用的唯一记录。普通任务、嵌套显卡阶段和提前加载都在其中登记。
忙碌时间按占用区间的并集计算，重叠时间只计一次；它是**模拟设备占用率**，不是实测显卡计算核心活跃率。

共置（让同一显卡同时运行两个任务）目前支持最多两个普通任务：模型已经驻留、显式指定
`workload_shape`，且硬件身份和模型/形状组合在 F1 实测表中有覆盖。模型只占一份驻留显存，
所有运行任务的工作区显存合计，不能驱逐正在使用的模型。
任务加入、退出或被抢占时更新剩余工作量、减速系数与完成事件。加载和驱逐不按推理减速系数缩放。

F1 是整段任务平均减速测量；把该系数用于重叠期间的工作进度是一种分段近似，**不是已经验证的瞬时速度模型**。
后续必须通过真实回放校准，不能仅凭单元测试宣称端到端精度或正式策略排序正确。

明确未支持：超过两个普通任务、冷加载/驱逐与推理共置、融合链的共置和抢占，以及预测式多步推演策略
（`pred_mpc_*`）的共置推演。后者会直接拒绝运行，不悄悄使用串行推演评估并发动作。

## 提前加载与抢占

F4 保留纯拷贝与完整框架加载两种实测证据。配置构建器选择完整加载（`full`），避免把纯数据拷贝
当成完整模型初始化。实测 `copy` 模式暂时拒绝执行，因为纯拷贝证据不保证模型已经可执行；
合成数据的 copy 模式仅用于验证事件数学。推理干扰与加载时间变化仅在已覆盖的重叠条件下应用。
实测证据仅对应中等档（`medium`）探针，不能推广到任意输入长度、图像或运行引擎。
未覆盖组合/形状/因子会明确登记原因并延后串行加载，不把未知干扰算成零成本。
提前加载完成之前，模型只占用预留显存，不算可执行的驻留模型。

抢占采用原有的**整节点重算**规则，不冒充能够保存/恢复中途进度。
选择受害者前检查释放其工作区后的显存与目标可执行性；保留并更新同卡其他任务。
不为无法与剩余任务一起派发的目标无意义地抢占，不抢占仍有预取或组合阶段预约的显卡。

## 模型阶段、批处理与缓存的真实边界

F2 接入仅暴露实测系数。原始节点没有完整请求长度和图像数量，因此不推断图像数量、
不输出伪造的阶段耗时。嵌套显卡阶段暂不在该观测覆盖范围内。

vLLM 第 5/6/7 项只支持读取来源并明确报告未支持状态：
- 连续批处理缺少请求级输入/输出长度及执行队列契约。
- 前缀缓存缺少可核对的公共输入前缀身份。
- 第 7 项是显存压力总代价，缺少实际抢占次数，不能当精确抢占代价。

这些证据不用于缩放 HF 主执行路径；已有 YOLO 的帧内批处理也不等于跨任务大模型请求批处理。
**因此 e（连续批处理与缓存复用执行）仍未完成。**

## 使用已有配置入口

```sh
export PYTHONPATH=src
python scripts/build_substrate_extension_config.py \
  --transition-config experiments/EXP-20261001_substrate_transition_integration_v1/transition_config.json \
  --colocation experiments/EXP-20260929_substrate_colocation_surface_v1/artifacts/colocation_cost_table_v1.json \
  --prefetch-interference experiments/EXP-20260929_substrate_overlap_matrix_v1/artifacts/f4_overlap_matrix.json \
  --output /tmp/substrate-extension.json
```

输出通过已有运行器 `scripts/r8_scheduler_extensions_matrix.py --extension-config` 使用。
提前加载需要原有显式 `prefetch_plan` 或 Latency-Aware 在线计划；构建器不会凭空生成计划。
旧任务数据缺少显式形状/硬件身份时必须补齐有证据的输入契约，不能根据未来真值或任意默认值猜测。
不能把该配置直接称为旧数据上已验证的完整共置实验。

## 最小验证

```sh
PYTHONPATH=src python -m unittest discover -s tests -p test_workload_v02_simulator.py
PYTHONPATH=src python -m unittest discover -s tests -p test_composite_segment.py
PYTHONPATH=src python -m unittest discover -s tests -p test_workload_v02_state_adapter.py
```

测试包含手算双任务非对称减速、反向查表、真实 F1 表加载、正冷加载成本下的驻留任务共置、
区间去重、显存保护、抢占剩余任务、未知 F4 因子延后、GPU 身份不匹配以及加载时间不消耗推理进度。
真实实测表测试使用人工短任务，只验证接入链路，不构成原始 workload 的真实回放证据。
