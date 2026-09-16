# State

## Current Stage

R0–R6 已完成并冻结为历史/兼容性基线；`behavior_nn_v1_r2` 和 R7-1 资源预测器已冻结。R7-2 的 640 条 predictor-unseen trace、节点级 workload 和 workload gate 已通过；R7-3 的 1,000-episode、10-policy validation 矩阵已通过；R7-4 的 RL 六 seed 运行完成，R7-5 的 paired validation statistics 已通过，但 RL seed-consistency gate 未通过。R8-P1 CP-RHO V1 已完成执行/审计矩阵，R8-P2 MILP sanity 已通过；CP-RHO 未超过 `PredOpt-v2`，`T_final` 仍封存。

**Latest execution-copy sync (2026-09-04):** 远端 `/root/autodl-tmp/scheduler` 已按内容级增量同步到外接盘 `/Volumes/Lenovo/scheduler`，包含代码、数据、模型、实验产物和 `.project/`。SHA-256 清单检查远端 `45,983` 个 regular files，`0` 个内容差异；本次实际传输 `501` 个文件、约 `58,382 kB`，不使用 `--delete`，目标独有文件保留。后续实验默认从外接盘执行，远端仅作源/备份；本地依赖环境尚未安装或锁定。

**Latest P9d update (2026-09-02):** predictor-aligned topology dataset 的独立经验基线已完成。模型只在 P_dev/train 拟合，validation 按预注册 NLL 选择 `current_full`，test 做诊断，holdout 在选择冻结后评估。结果确认 identity-free 多节点 DAG-layer 输出契约可运行，但 layer/node count 在 test/holdout 仍整体低估，holdout 条件键全部回退到全局分布；尚不足以作为 learned predictor 或接入 scheduler。精确数值见 `experiments/EXP-20260902_p9d_topology_empirical_baseline/metrics.json`，执行与哈希见同目录 `run_manifest.json`。

**Latest P9d tabular update (2026-09-02):** 在同一 P_dev/P_holdout_diag 数据边界上完成第二阶段 tabular comparator。只用 P_dev/train 的 one-hot causal model_input 训练两个预注册 LightGBM 结构配置，validation 按 layer-count MAE 选择 `lgbm_small`，test 做诊断，holdout 在冻结后验收；节点原型仍由 train-only 条件表解码。holdout 的 layer MAE=`0.1696`、node MAE=`0.5435`、width vector MAE=`0.1655`，layer/node bias=`+0.0290/+0.0232`，结构明显优于 empirical baseline；但 prototype accuracy=`0.5160`、top-3 完整 signature coverage=`0.0870`，仍存在 holdout 分布偏移。因此 tabular 仅闭合结构 comparator，不是最终 topology predictor，也未接入 scheduler。精确数值见 `experiments/EXP-20260902_p9d_topology_tabular/metrics.json`，执行、边界和哈希见同目录 `run_manifest.json`。

**Latest P9d shared-GRU update (2026-09-02):** 在同一 P9d 边界完成 `EXP-20260902_p9d_shared_causal_gru` 的 4 变体×3 seeds 远端实验。单向共享 causal GRU 的 topology heads 在 holdout 上显著优于 tabular comparator；按预注册 validation `layer MAE + width-vector MAE` 选择 `topology_only`（`0.1284`），但 holdout 的结构综合表现最好的是 `shared_multitask`：layer/node/width MAE=`0.1442/0.3995/0.1378`，exact shape=`0.8217`，top-3 signature=`0.2995`。`shared_multitask` 同时保持/提高行为 holdout role/family/joint accuracy=`0.9630/0.6901/0.8449`，相对同实验 `behavior_only`=`0.9621/0.6679/0.8355`；首层 consistency 没有显示清晰增益。远端/本地产物哈希、36 个 prediction 文件、top-3 概率、输出契约审计全部通过；该结果是 learned topology diagnostic，不接 scheduler，`T_final` 继续封存。权威解释见 `experiments/EXP-20260902_p9d_shared_causal_gru/RESULT.md`。

**Latest P9d future-content update (2026-09-03):** `EXP-20260903_p9d_future_content_multitask` 的 `N0/A/B/D × 3 seeds` 共 12 个远端 run 已完成，Stage 0、实际数据 smoke、远端测试和本地逐文件哈希审计均通过。A 按 validation `layer_count_mae + full-H width_vector_mae` 选中（`0.1272`）；D 的 frozen holdout structure 最好（exact=`0.8244`、layer MAE=`0.1437`、width MAE=`0.1371`），但不事后改选。A/B/D 的 validation predicted-content F1=`0.6715/0.6748/0.6738`，B/D next-family F1=`0.7904/0.7829` 低于 N0=`0.8075`，因此辅助行为头的 validation non-inferiority 未通过。该实验验证了共享 causal GRU + structure/content/next-step 分头的可运行性，但 content holdout gap 仍大，尚不能冻结为 scheduler 输入；60 个 artifact 文件和根级 stage0/metrics 共 62 个非 checkpoint 产物已恢复本地，12 个 checkpoint 保留远端。权威解释见 `experiments/EXP-20260903_p9d_future_content_multitask/RESULT.md`。

**Latest P9e reduced role/family update (2026-09-03):** `EXP-20260903_p9d_future_role_family_multitask` 在同一 P9d predictor 数据边界完成 `N0/A/B/D × 3 seeds` 共 12 个远端 run。未来节点输出收缩为每节点 `role + action_family`，同时保留 `next_role + next_family_if_execute` 辅助行为头；B 按 validation structure score 选中（`0.12847`）。B 的 frozen holdout structure score/exact=`0.27986/0.82923`，未来节点 role/action-family F1=`0.90151/0.40360`；共同 action-family 相比旧四字段 B 未提升。Stage 0、实际 forward/loss/backward smoke、远端 4/4 契约测试、本地逐文件 hash 和 36 个预测文件的输出审计均通过；62 个非 checkpoint 产物已恢复本地，12 个 checkpoint 保留远端。该实验确认简化后的输出契约可行，但不把 B 冻结为 scheduler 输入；`S_train/S_val`、原始 trace 和 `T_final` 仍不触碰。权威解释见 `experiments/EXP-20260903_p9d_future_role_family_multitask/RESULT.md`。

**Latest P9f acceptance-audit update (2026-09-03):** 在不重新训练的前提下，对冻结 P9e `B`/seeds=`11/22/33` 完成 train-only calibration、P_dev/test/holdout 分层审计和粗粒度 future-runtime proxy。Stage 0、远端 `4/4` 单元测试、编译、真实 forward smoke、本地 JSON 有限值/manifest-output 审计和远端/本地逐文件 SHA-256 均通过。holdout 结构均值为 layer/width/node MAE=`0.1442/0.1357/0.3807`、exact=`0.8292`，三项结构误差均优于 empirical baseline 的 `1.9080/0.9274/4.6370`；但 train-fit temperature 在四个核心 head 上的 holdout NLL 均变差，因此保留 raw probabilities。role/family→P50 runtime proxy 的 holdout coverage 为 `0.8449/0.8435/0.9152`，top-1 runtime MAE 均值=`118,798.7 ms`且显著低估长尾；当前 reduced node contract 缺少 `model_id/node_type/input_scale/cold_warm`，load/memory 不可辨识。故 P9f 为 `completed_diagnostic`，structure gate 通过但 calibration/resource/model-aware/scheduler gates 关闭；不进入 `S_train/S_val` 推理，不接 scheduler，不读取 `T_final`。权威解释见 `experiments/EXP-20260903_p9f_predictor_acceptance_audit/RESULT.md`。

chatgpt_web:
  status: active
  generation: 7
  conversation_id: "6a9822da-e278-83e9-9c1a-675923acda0e"
  conversation_url: "https://chatgpt.com/c/6a9822da-e278-83e9-9c1a-675923acda0e"
  title: "架构设计评估"
  model: "GPT-5.6 Sol"
  reasoning: "High"
  created_at: "2026-09-02 21:30 CST"
  last_verified_at: "2026-09-08 17:19 CST"
  last_used_at: "2026-09-08 17:19 CST"
  parent_conversation_id: null
  last_rollover_reason: null

**Latest ChatGPT Web architecture discussion (2026-09-02):** 脱敏架构 brief 已在绑定会话“架构设计评估”完成提交并返回完整回答；研究报告与原始来源链接整理在 `docs/research/2026-09-02_future_dag_structure_content_decoder.md`，并完成 Deep Sets、Set Transformer、TSPN、GraphRNN、GRAN、D-VAE、Scheduled Sampling 等原始页面核验。其建议属于研究输入而非已批准架构决策：优先比较 shared causal GRU + structure head + layer-wise permutation-invariant set decoder + all-future-node content decoder，并保留 next-step behavior auxiliary head；使用 soft topology conditioning 和直接 `h_t` 路径，暂不采用硬 topology 串联或完整 graph decoder。当前只写入研究记录和会话生命周期，不修改代码、不改变 P9d gate、不解封 `T_final`。

**Latest ChatGPT Web experiment-plan discussion (2026-09-03):** 在同一绑定会话中完成“保留 next-step behavior auxiliary 后的 all-future-node predictor”实验设计评审，研究记录为 `docs/research/2026-09-03_next_step_auxiliary_experiment_plan.md`。网页建议首轮收缩为 `N0/A/B/D`：next-only reference、future-only、future+next auxiliary、future+next+soft topology conditioning；将 layer-1 consistency、显式 edge 和 autoregressive/graph decoder 后置。建议的 5-fold source-video grouped CV×3 seeds、非劣性 margin 和 stop-gradient conditioning 均标为项目推论/待验证假设，不自动替换当前 R8-P9d 计划。未改代码、未创建正式实验、未训练、未接 scheduler，`S_train/S_val/T_final` 继续排除/封存。

**Latest ChatGPT Web topology-resource discussion (2026-09-06):** 在同一绑定会话提交脱敏 brief（拓扑→资源接口可辨识性问题），收到完整研究报告并存档为 `docs/research/2026-09-06_topology_resource_interface.md`（`status=valid`，发送前可见 High 已核验，回答后已通读核验）。核心建议（研究输入，非决策）：把瓶颈定为接口可辨识性而非拓扑预测本身，引入 `z=(exec_class, model_resource_class, input_scale, mode)` 资源等价签名 + 运行时上下文 `c_t` 的两段式接口，用不确定性边缘化代替 hard top-1 展开；先做 oracle-signature 可辨识性上界实验（R0），再做 C0–C4 契约消融；引用的 Vulcan/Vidur/vLLM/DistServe/ServerlessLLM/WfCommons/PeakBench 等需本地独立核验后才可引用。未改代码、未创建正式实验、未训练、未接 scheduler，`S_train/S_val/T_final` 继续排除/封存。

**Latest ChatGPT Web joint-training discussion (2026-09-06):** 同一绑定会话追加聚焦 follow-up（拓扑×资源联合训练如何让资源更准），收到完整研究报告并存档为 `docs/research/2026-09-06_joint_topology_resource_training.md`（`status=valid`，发送前可见 High 已核验，回答后已通读核验）。核心建议（研究输入，非决策）：首轮只做 J1/J2/J3 三阶段（共享编码器+独立资源头 → stopgrad 软条件 → 端到端软条件），资源标签优先用同 trace 后续实测值离线 join，禁用 role/family 中位数伪造连续真值；主指标用 ResourceQScore（.50/.95 pinball 均值），J3 晋升要求结构/行为 NI 全过且校准不塌；引用的 VIDUR/USHER/Llumnix/FairGrad/FAMO/保形推断等需本地独立核验。未改代码、未创建正式实验、未训练、未接 scheduler，`S_train/S_val/T_final` 继续排除/封存。

**Latest ChatGPT Web plan-review discussion (2026-09-06):** 同一绑定会话上传 3 个安检后文件（整理后规划文档、P9e 模型代码、脱敏数据样本各 1 条 feature/label）并请求评审，收到完整评审报告并存档为 `docs/research/2026-09-06_joint_plan_code_data_review.md`（`status=valid`，发送前可见 High 已核验，回答后已通读核验）。结论（研究输入，非决策）：J1→J2→J3 架构保留，但规划文档不予原样冻结；开出 5 个 P0 必改项（同层依赖语义、运行时上下文白名单、实测资源标签确认、holdout 可见性、checkpoint 选择改按 ResourceQScore）与若干 P1/P2 项，另建议加非正式 J0 冻结编码器探针。引用的 FAMO/FairGrad/保形推断/CVPR 部分标注 MTL 等需本地独立核验。未改代码、未创建正式实验、未训练、未接 scheduler，`S_train/S_val/T_final` 继续排除/封存。

**Latest ChatGPT Web three-revisions review (2026-09-07):** 同一绑定会话上传 2 个安检后文件（三条修订说明 + 3 个随机 raw trace 脱敏样本）并请求评审，收到完整评审并存档为 `docs/research/2026-09-07_three_revisions_review.md`（`status=valid`，发送前可见 High 已核验，回答后已通读核验）。结论（研究输入，非决策）：J0 接受且不加随机编码器对照；input_scale 修订接受但改名 workload_scale，首版用 `(clip_len, query_char_len, nested_api_call_count)`，分辨率/帧数暂不强制；分层与展开必须都修且先修依赖重建（step 级父依赖取 last-event 语义）再做最长路径分层，修完后可恢复 DAG 层术语。未改代码、未创建正式实验、未训练、未接 scheduler，`S_train/S_val/T_final` 继续排除/封存。

**Latest ChatGPT Web dataset-audit verdict (2026-09-07):** 同一绑定会话发送全量数据集审计数（54,687 层、18,683 行），收到裁决并存档为 `docs/research/2026-09-07_dataset_audit_verdict.md`（`status=valid`，发送前可见 High 已核验，回答后已通读核验）。结论（研究输入，非决策）：三条修订冻结，但 Finding C 升为 P0 地基问题——R0 上界实验必须等事件图契约冻结，否则 oracle 本身不可信；D 按生存分析删失处理（`L≥5` 不等式约束而非降权）；E 把宽度 5 并入 4+ 而非删除；B 接受 workload_scale 首版。核心转向：下轮正式实验先是标签契约稳定化实验，其次才是联合学习。引用的 WfCommons/DeepHit 等需本地独立核验。未改代码、未创建正式实验、未训练、未接 scheduler，`S_train/S_val/T_final` 继续排除/封存。

**Latest P9d dataset v2 rebuild (2026-09-07):** 本地执行 builder 修改并在外接盘重建 `topology_predictor_p9d_v2`：last-event 展开（14,736 次，丢弃 16,971 条脑补边）、最长路径分层、逐节点 workload_scale、termination_status、宽度 4+ 合并；8/8 本地测试通过（含 3 个新测试）。v2 行数与 v1 逐 split 一致（13,754/2,029/1,520/1,380），层内依赖层从 2,307 清零，删失/终止为 3,315/15,368，workload_scale 覆盖 68,731 个未来节点；宽度 5 在新分层下自然消失，合并路径未触发。旧 v1 数据集、原始 trace、模型代码、`S_*`/`T_final` 均未动。登记见 `data/manifests/topology_predictor_p9d_v2.json`。

**Latest ChatGPT Web v2 rebuild review (2026-09-08):** 同一绑定会话发送脱敏 v2 重建包（5 项修复 + 审计数 + 6 个评审问题），收到完整评审并存档为 `docs/research/2026-09-08_p9d_v2_rebuild_review.md`（`status=valid`，发送前 `high_ui_mapping` 可见 High 已核验，`select-model` 为 `already_visible`，回答后已通读核验；另存 `docs/research/2026-09-08_p9d_v2_rebuild_review_export.md` 为可见 DOM 导出）。结论（研究输入，非决策）：CONDITIONAL GO for R0——C 按重建策略接受（last-event 为最小可操作化，非观测真值，1,227 条 fallback 链仍为策略边）、A 接受、D 标签接受但 loss 改为 H 内深度 + 独立终止头（仅在预测超 H 总深度时才用生存式不等式）、E 接受 4+、B 按首版代理接受；剩余 P0-1 为全图无环需显式断言（`cycle count=0`），P0-2 为 fallback 边语义仅对边敏感 claim 阻塞，节点级 R0 不必整体等待。R0 必须框定为冻结 v2 重建契约条件下的资源上界，不声称观测因果 DAG。引用的 WfCommons/WfBench/Vidur/生存分析/长尾分类等需本地独立核验。未改代码、未创建正式实验、未训练、未接 scheduler，`S_train/S_val/T_final` 继续排除/封存。

**P9d step-structure audit (2026-09-10, 只读):** 用户质疑 v2 的“步骤级展开”标签规则丢失事件间真实关系（任务流程仅 2–3 类、类内步骤关联应相似），对 P9d 预测器池全部 1,360 条 run（`results/raw`，与 behavior 锚点 run_id 完全匹配，0 missing）做只读结构审计。结果确认三类流程：`langgraph_react` 596、`star` 596、`st_fixed` 168 run；step 级父引用只出现 `()` 或单步 `(k,)`，多父事件=0，即类内步骤是一条链。trace 内含可恢复的事件级关联信号：tool 事件 `input.standard_tool_call.id = compat-{star|react}-step-N` 与本步号 100% 一致（langgraph 3,383/3,439、star 3,247/3,303 步），planner `parsed_decision.tool_name` 与同一步 tool action 名称匹配 100%（0 mismatch）；`retry_of` 共 502 个事件（langgraph 268、star 234）全部同一步；`st_fixed` 为固定序列 `[ImageQA, Summarizer]`（无 planner，168/168 完全同构）。v2 当前的 last-event 规则把同一步的 planner 与其 tool 变成同层兄弟：v2 图上 langgraph 2,849、star 1,929 个 tool 与其 planner 同层（合计 4,778/6,630 ≈72% 可匹配 tool 事件）；对 v2 train labels 全量扫描，13,754 行中 5,856 行含这种同层 planner+tool 对（共 18,028 对），70.7%（18,422/26,054）的未来层含多个节点，tool 的 predecessor 91.6%（18,236/19,914）指向上一步的 tool 而非本步 planner。另核对 R7 `job_templates_r7_v02` 使用“父步骤全部事件”展开（比 v2 更粗），同样缺 planner→tool 边（scope 只到 P9d/R7 模板证据，不外推）。结论：v2 的 layer/width 标签存在系统性“假并行”膨胀，需要按类别的 v3 重建规则（planner→tool、retry→failed、step 链、st_fixed 模板）；该发现待用户决策后再设计 v3，未改代码、未重建数据、未训练、未接 scheduler，`S_*/T_final` 未动。证据：`.scratch/p9d_step_structure_audit.py`、`.scratch/p9d_step_structure_audit.json`、`.scratch/p9d_step_supp.py`、`.scratch/p9d_step_supp.json`。

**P9d v3 契约草案与网页评审 (2026-09-10):** 依据上述审计与采集器源码语义（`src/tracing/collectors/videotool_phase1.py`：`parent_step_ids` 为采集器粗粒度约定；`Summarizer` 内部调用 answer model 形成嵌套事件；`retry_of` 记录失败→重试）设计 v3 重建契约草案：节点=事件−`run_control`−嵌套答案调用（合并进 Summarizer 的 `nested_calls`），边=执行顺序链，每条 run 单根单链；原型在 1,360 条 run 上全部通过（langgraph 8,289 节点/7,693 边、star 8,048/7,452、st_fixed 504/336；无分叉、answer 恒最后、post-loop generate 恒倒数第二、步号单调），相对 v2 修正 7,117 个节点前驱（工具 5,575、post-loop 生成 1,177、重试 365），planner→tool 同一步配对 6,630、名称不一致 0。脱敏中文草案 `.scratch/chatgpt_p9d_v3_contract_brief_zh.md` 送绑定会话评审（附件首次未送达、独立 `upload` 重传后完整返回），存档 `docs/research/2026-09-10_p9d_v3_contract_review.md`（`status=valid`；另存 `..._export.md` 可见 DOM 导出）。网页裁决（研究输入，非决策）：**MODIFY 后接受**，方向正确但需 4 项契约级修改才可冻结——(1) 把“文件顺序链”改写为“经源码语义+独立审计验证的顶层串行控制流直接执行链”，并新增独立 Seriality gate（多工具/fork/无法解释的时间窗重叠/新 workflow 类型一律 fail-closed）；(2) nested Summarizer 用 composite resource signature（outer_exec_class/outer_tool_family/nested_model_class），不要用 `effective_model_id` 覆盖节点语义、不做 runtime 重复核算；(3) final answer 定为 terminal/non-resource 事件（`resource_applicable=false`，不占 H=5 计算槽位），post-loop generate 才是 compute node；(4) width=1 标为 deterministic compatibility field，放弃 width prediction 作为正式结构任务，研究表述改为 bounded future execution-trace forecasting。另新增 P0：R0 前冻结 schedulable/resource node ontology（planner/tool/merged Summarizer/post-loop generate/retry=resource target；nested/final answer/run_control 不是）；holdout 说明改为“contract-integrity inspected, never used for fitting/selection”（全量审计已触及 holdout 统计，不再声称 completely untouched）。scheduler 侧同源 `job_templates` 问题接受分阶段处理：predictor v3→R0→J 系列冻结后再做 scheduler contract audit。另对 R7 调度池（`r7_trace_full_20260817` + pilot，648 条 run、8,878 节点）做相同串行性检查：多工具步=0、非上一步父引用=0、步跳变=0、answer 恒最后、post-loop 生成恒倒数第二，同一步相邻对全部为合法串行类（planner→tool、planner→planner 重试 230、planner→Summarizer 171 等），嵌套合并 171、重试事件 230；即当前全部 2,008 条已采集 run（预测器 1,360 + 调度器 648）在事件粒度上均为串行控制流。证据：`.scratch/r7_seriality_check.py`、`.scratch/r7_seriality_check.json`。未改代码/数据、未重建、未训练、未接 scheduler，`S_*/T_final` 未动。

**P9d v3.1 标签契约冻结与数据集重建 (2026-09-10):** builder 增加 `--label-policy v2|v3`（默认 v2 保持旧行为），实现经验证串行控制流重建（节点 = supported 事件 − `run_control` − 嵌套答案调用；边 = 执行序列链；Seriality/信号一致性/嵌套核算/路径/可调度本体 gates fail-closed）；嵌套答案调用合并进 Summarizer 节点（`nested_calls` + composite `resource_signature`，`R_node = R_total`）；失败 planner 与 retry 保留为链节点；最终 `answer` 为 terminal/non-resource 标记（不占 H=5 槽位）；width 标记为 deterministic compatibility field。12/12 本地 unittest 通过（新增 4 项 v3 测试 + LF 换行回归，v2 8 项保持）。v3 数据集重建到 `results/processed/topology_predictor_p9d_v3`，0 violations：1,360 run、16,841 链节点、15,481 边、嵌套合并 482（langgraph 272 / star 42 / st_fixed 168）、retry 502/502 邻接、tool step id 6,630/6,630、planner-tool 名称 6,630/6,630、terminal marker 1,360、宽度直方图 `{1: 65,275}`；行数/split/视频/runs 与 v2 完全一致（13,754/2,029/1,520/1,380；240/30/30/40；988/144/108/120）；features 解压 SHA-256 与 v2 四 split 逐字节一致；v2→v3 前驱修正 7,117（tool:tool→planner 5,565、generate:tool→tool 1,113、planner:tool→planner 365、其余 74）。登记 `data/manifests/topology_predictor_p9d_v3.json`（8 个数据文件 + 2 个报告 SHA-256），契约文档冻结于 `docs/p9d_topology_label_contract_v3.md`。重建过程保留两个 invalidated 目录（v2diff 统计 bug、CRLF 换行）供审计。未训练、未接 scheduler，`S_*/T_final` 未动。

**R0 设计草案与网页评审 (2026-09-11):** 在 v3.1 契约上设计 R0（oracle-signature 经验资源上界：`(Z_oracle, workload_scale, c) → runtime/load/memory`，仅 resource-applicable 节点；作为 J 系列前置 go/no-go gate），草案 `chatgpt_p9d_r0_design_brief_zh` 送绑定会话评审（中文，`upload` 附件已验证、回答完整返回），存档 `docs/research/2026-09-11_p9d_r0_design_review.md`（`status=valid`；另存 `..._export.md`）。裁决：**定位成立但须 MODIFY 后冻结**；3 个 P0——(1) 删除全部 `load_ms → regime/context` 路径（`load_ms` 只能是 target；residency 只用执行前可见的 `model_resident_before`，改名 `residency_state`）；(2) 修正 G3 覆盖率定义（`[P50,P95]` 名义覆盖 45%，不是 90–98%；改为 q50/q90/q95 单侧校准 QCalError，或加 q05 后检查 `[q05,q95]` 的 90% coverage）；(3) 明确 memory measurement semantics（`peak_*` 必须证明是 node-window/reset/baseline 校正后的单节点测量；若为进程级累计峰值，memory 先从 go/no-go 移出）。另要求：stall 由“主表排除”改为 R0-Inclusive 主表 + R0-Normal 副表（tail 本身就是调度关心项）；G1/G2 降为机制诊断，不得作为 veto；统计单位改为 **video-cluster bootstrap** + per-run/per-video macro 敏感性（不加 run-level split）；tail 用 train-frozen `q90` 阈值；新增 C0–C4 递增 contract ablation（exec_class→role→model→workload_scale→residency/context）；exact model_id/stack 结论限定 known-model regime 并加 coarse ablation；Stage 0 增加 6 项硬断言（load 不进特征、memory 语义冻结、nested 不重复计、feature 角色白名单、fallback 层级固定、tail 阈值 train-only）。未改代码/数据/实验，`S_*/T_final` 未动。

**R0 设计评审循环与修正 (2026-09-11):** R0 设计经三轮网页评审并收敛：v2 复核确认原 3 P0 关闭、新 P0（`load_ms=0` 语义）+ 11 项 P1；v3 确认又发现 2 项——load 计数口径错误（原扫描把 482 个嵌套生成计入，15,887 > 15,481）与 occurrence 模型未预注册。本轮修正并本地验证：(1) 采集器源码证据 `load_ms = model_load_ms if request_index == 1 else 0.0`（帧加载路径同理）→ **0 = 合法 no-load 观测**（情况 A），按 hurdle（occurrence + positive duration + expected load）评价，缺失=unavailable 并报覆盖率；(2) `model_resident_before` 在 P9d 池 100% null → `residency_state` 不可用（记录为数据限制），C4 上下文改用 `prefix_model_reuse`（执行前可见，明确标注 proxy）；(3) worker 每请求读取 `max_memory_allocated/reserved` 后立即 `reset_peak_memory_stats()` → memory 为按请求窗口的节点级峰值（含常驻权重，inclusive 语义），P0-3 证据闭环；(4) load 计数在 v3.1 节点本体上重算：planner 7,211（missing 67 / zero 5,960 / positive 1,184）、tool 7,078（0 / 6,742 / 336）、post-loop generation 1,192（9 / 530 / 653），合计 15,481（missing 76 / zero 13,232 / positive 2,173），nested 482 已排除且不构成独立 target；(5) 全部 P1 预注册写死：`PB_primary = mean(PB_0.50, PB_0.90, PB_0.95)`、唯一 Core baseline = exec_class 条件中位数、raw-scale gate、major strata 规则（n≥200 且 ≥10 独立视频）、`max_τ|C_τ−τ| ≤ 0.05`（含 q95）、q05 不做、quantile crossing 单调重排+报告、C0–C4 精确字段、LightGBM 固定网格/seeds/选型（quantile 用 PB_primary；occurrence 用 Brier）、Normal 为同一 inclusive 模型的子集诊断、holdout failure 规则。评审给出条件式接受（“把 load 计数/节点本体对齐并预注册 occurrence classifier 后即可给出 R0 DESIGN v3 ACCEPT”）；两项已修复并本地验证；是否再送最终确认轮待用户决定（本轮未自动发送，遵循送审轮次约束）。证据：`.scratch/v3_load_accounting.py`、`.scratch/v3_load_accounting.json`、`.scratch/chatgpt_p9d_r0_design_brief_v3_zh.md`、`docs/research/2026-09-11_p9d_r0_design_review_v2.md`、`docs/research/2026-09-11_p9d_r0_design_review_v3.md`。未改代码/数据/实验，`S_*/T_final` 未动。

**R0 设计冻结 (2026-09-11):** 最终确认轮裁决 **“R0 DESIGN v3 ACCEPT（可冻结）”**：load 计数与 v3.1 节点本体对齐（planner 67/5,960/1,184=7,211；tool 0/6,742/336=7,078；post-loop generation 9/530/653=1,192；合计 15,481；nested 482 排除）；hurdle 契约（`load_ms=0` 合法 no-load）与 occurrence 模型预注册（LightGBM binary 固定配置 + Brier 选型 + 固定 logistic 参考）关闭；其余 P1 达预注册状态；无新 P0；唯一实现备注（主 LightGBM 配置选择 = 3 seed validation `PB_primary` 均值，worst seed 仅报告）已并入设计 §5。冻结文档 `docs/p9d_r0_oracle_signature_ceiling_design.md`；评审存档 `docs/research/2026-09-11_p9d_r0_design_review_final.md`（同目录含 v1/v2/v3 报告与 exports）。下一步：注册 `EXP-20260911_p9d_r0_oracle_signature_ceiling`、更新 `.project/EXPERIMENT_GATE.json`、跑 Stage 0 + smoke（本地 CPU，<10 CPU-min）。未训练、未接 scheduler，`S_*/T_final` 未动。

**Latest override (2026-08-18 09:17 +08:00):** workload audit 为 `gate=true`（640 templates、20,000 train + 1,000 validation episodes、135 scenario cells、777,840 jobs）；scheduler matrix 为 `passed`（10,000 result rows、0 failed jobs、0 capacity violations）。Paired statistics 覆盖 1,000 episodes/135 cells/10,000 bootstrap；PredOpt-H5 的 completion/deadline 改善通过 Holm 校正，RL-H5 相对 RL-0 的三个 seed 方向不一致，因此 RL 尚未冻结，`T_final` 仍封存。

**R8 override (2026-08-18 12:28 +08:00):** P0 首次 PredOpt-v2 审计在 1,000 validation episodes 上完成，0 failed jobs；priority=0.25/future=1.0/memory=0.05 的候选比旧 PredOpt-H5 慢 6,770.44 ms，且 deadline miss 更高，虽然 94.4% episodes 出现动作差异。随后在 `S_train` 的 300 episode 校准 pilot 测试 priority=1/2、future=1/2 三组配置；无配置改善 completion，因此 score-only tuning 停止，下一步转 CP-RHO V1。远端 base/finetooling/videotool 均未安装 `ortools`，P1 需先获得依赖安装确认；`T_final` 仍封存。

**R8-P1 update (2026-08-19 21:03 +08:00):** 远端 `finetooling` 已安装并核验 `ortools==9.9.3963`；CP-RHO V1 完整 S_val 矩阵通过执行/审计 gate：1,000 episodes、6,000 policy rows、0 failed jobs。CP-RHO H1/H3/H5 的 timeout/fallback 分别为 1,866/1,760/1,787，平均完成时间均高于对应 PredOpt-v2（CP-RHO H5=223,464.12 ms，PredOpt-v2 H5=217,524.48 ms），因此 CP-RHO V1 只保留为失败/开销审计，不提升为最终候选。`T_final` 仍封存。

**R8-P2 update (2026-08-19 23:30 +08:00):** fix2 已作为 canonical 结果通过复核。SciPy 1.15.3/HiGHS MILP 与 CP-RHO 在 20 个四节点/双GPU小案例上均为 `OPTIMAL`；node-level 首动作一致率=1.0，exact node+GPU 一致率=0.8（4 个案例为同节点不同 GPU 标签），最大目标相对差 `2.6319902025099073e-13`，低于 `1e-6` gate；CP-RHO/MILP 平均求解时间分别为 39.51/40.81 ms。报告记录了完整 command、seed、time limit、cwd/PYTHONPATH、运行时/硬件信息，以及 `cp_rho.py`、`milp_sanity.py`、`workload_v02_simulator.py`、runner 四份源码哈希。该结果只验证窄版 assignment/NoOverlap 模型目标/约束实现，不声称已验证未来 DAG precedence、显存硬约束或完整 cache 生命周期，也不改变 CP-RHO 的 P1 性能结论，不把 MILP 接入在线调度。canonical 远端输出为 `/root/autodl-tmp/scheduler/results/processed/r8_optimizer_rl_upgrade/p2_milp_sanity_20260819_fix2/`。

## Stress pilot update (2026-08-21)

用户授权后完成了独立的 `EXP-20260821_stress_pilot`。现有 v0.2 workload builder 没有压力参数，因此只新增实验专属派生脚本，不改 R7 builder、模板、预测器或 R7 结果。以同一批 30 个 R7 validation episodes 生成 S0 baseline、S1 ready-width、S2 tight-deadline、S3 offered-load、S4 combined，共 150 episodes/5,120 jobs；RR/Myopic/PredOpt-H5/Oracle 运行 600 rows，0 failed jobs、0 capacity violations，summary gate=passed。

关键观察：S1 将 Myopic/PredOpt-H5 event-audit action-width 中位从 2 提高到 12，S3 将 queue/wait 压力明显提高；整体 Oracle 相对 Myopic 的完成时间差扩大到约 55%（S3）。S2 只改变 deadline miss（Myopic 11.4%→23.6%），不改变 completion/queue，因为当前策略排序不使用 deadline slack；PredOpt-H5 在 S1/S3/S4 与 Myopic 完全同分，说明 finite-horizon 预测尚未有效改变动作顺序。该 pilot 只用于压力机制校准，不能替代扩大 S_val 的正式统计，也不能解封 `T_final`。

权威本地记录：`experiments/EXP-20260821_stress_pilot/metrics.json`、`RESULT.md`；远端报告：`/root/autodl-tmp/scheduler/results/processed/EXP-20260821_stress_pilot/pilot_report.json`。R7 与 `T_final` 保持只读。

## GPU topology pilot update (2026-08-21)

已完成 4/8-GPU 与异构 GPU topology pilot：以 stress pilot 的 S0/S4 同源 episode 为基础，8 cells × 30 episodes、4 policies，共 960 rows、8,192 jobs；0 failed jobs、0 capacity violations，事件审计 8 cells × 1 episode 通过。拓扑为 G4/G8 同容量和 H4/H8 `[32760,24576,...] MB`。相对上一轮 2-GPU控制，Myopic 在 S0 从约 214.7s 降至 G4 111.7s、G8 98.7s；S4 从约 1,097.6s 降至 G4 570.6s、G8 308.9s。4→8 GPU 的收益依赖压力：S0 约 -11.7%，S4 约 -45.9%。异构容量对完成时间影响小于 0.6%，但 RR 的 cache eviction 明显上升。G8 baseline action-width 中位数从 G4 的 3 增至 8；S4 ready-node 数限制下约为 10。PredOpt-H5 仍在压力 cell 与 Myopic 同分；增加 GPU 缓解排队但未让预测真正改变动作排序。权威记录为 `experiments/EXP-20260821_gpu_topology_pilot/metrics.json`、`RESULT.md`；远端报告为 `/root/autodl-tmp/scheduler/results/processed/EXP-20260821_gpu_topology_pilot/gpu_topology_report.json`。R7、`T_final` 保持只读。

## Scheduler extension pilot update (2026-08-21)

用户澄清本轮要扩展的是 batch-size、预加载和节点级抢占，而不是把 YOLO26n 加进 Stack B。已停止误启动的 YOLO26n 完整仿真；本轮新实验不读取或覆盖 R7/T_final。新增的扩展是 opt-in：`(ready_node, GPU, batch_size)` 只对有实测 YOLO per-tool profile 的节点生效；prefetch 以独立 `prefetch_start/end` 事件占用 GPU load lane 和 resident memory；preemption 使用无 checkpoint 的 node-level recompute，且 pilot 每 episode 上限32次，priority 仍是原有二元字段。

同一 R7 validation 前100集、同一到达/任务集合的 paired matrix 已通过：5个条件均为 `status=passed`，每条件1,600 completed jobs，0 failed jobs，0 capacity violations。未扩展 Myopic mean completion=160,957.96ms；batch_myopic=161,529.53ms（+0.36%，evictions 18.57→20.01，batch-action width p50均值2.805）；paid prefetch=161,421.58ms（+0.29%，每集load 9,181.90ms，wasted 1.78）；myopic_preempt=165,792.31ms（+3.00%，每集31.73次抢占、丢弃重算17,046.53ms，deadline miss +0.625pp）；三者组合=166,546.95ms（+3.47%，evictions 21.94、wasted prefetch 1.78、width 2.770）。batch_throughput 的完成时间+0.29%，但 miss rate比baseline低0.0625pp。

本结果表明：在当前 workload 上，扩展动作空间本身已可观测，但 batch/prefetch 单独收益很小，预加载与抢占的成本反而提高等待/驱逐；抢占若没有 checkpoint/resume 合约不应宣称有优势。batch pilot 本轮只接入已验证的 YOLO11x peak-memory 曲线，runtime/load沿用冻结模板，避免臆造未聚合的 batch runtime。权威记录：本地 `experiments/EXP-20260821_scheduler_extensions/metrics.json`、`RESULT.md`；远端报告目录 `/root/autodl-tmp/scheduler/results/processed/EXP-20260821_scheduler_extensions/matrix_*_100/`。统一源码哈希：runner=`b06fddae6fbb4abfab3eece4dad639bce187f985e923c19c6bfc61c2fdfccc2e`，simulator=`f900cc7cfd7b309461bed2435188b9e906f47d2f292ffffd7aed07be654740b8`。该100集仍是扩展 pilot，不替换R7主表或解封T_final。

2026-08-22 batch profile update：远端 YOLO11x 112-run pilot 的原始 `yolo-tracker` 事件已聚合为 batch=1/2/4/8/16/32/64 的 runtime/load profile，并写入 `experiments/EXP-20260821_scheduler_extensions/batch_profiles.json`；当前可选动作仍为 batch={1,8,16,32,64}，2/4 只保留测量不激活。profile 的 runtime/load 使用事件 p50，runtime p90 作为估计字段，显存沿用原有保守 profile。原100集扩展结果没有重跑，R8-P7仍待重新验证。

## R8 Transition Profile (2026-08-24)

已完成独立的 `EXP-20260824_transition_profile` 真实模型 transition benchmark。远端 `finetooling` 环境在单张 RTX 4080 SUPER 上按顺序测量 Qwen3-VL-8B-Instruct、Qwen3-4B、Qwen2.5-VL-3B-Instruct 和 yolo11x.pt，各 2 repeats，共 8/8 success、0 error、0 missing。结果区分了 cold load、first operation、resident operation 和显式 unload/`cuda.empty_cache` proxy；重复 operation 只作为 recompute proxy。四个模型均未暴露可审计的 checkpoint/restore API，因此没有填写伪造的 checkpoint 成本。

权威本地记录为 `experiments/EXP-20260824_transition_profile/metrics.json`、`RESULT.md` 和 `artifacts/run_20260824T100219+0800/`；远端原始输出为 `/root/autodl-tmp/scheduler/experiments/EXP-20260824_transition_profile/run_20260824T100219+0800/`。运行后远端 GPU 回到 0%/1 MiB，未下载视频、未重采 trace、未读取或修改 `R7/T_final`。该 profile 目前只支持下一步设计 opt-in simulator transition cost，不自动解封 `T_final`。

## R8 Transition Profile Simulator Pilot (2026-08-24)

已完成 `EXP-20260824_transition_profile_simulator` 的最小 opt-in 接入与远端配对 pilot。simulator 新增独立 transition-profile resolver：对非 resident 模型使用已测 cold-load，真实 eviction 时计入显式 unload/`cuda.empty_cache` proxy；resident/first-operation 不替换 node runtime，checkpoint/restore 保持 unsupported，preemption 继续 full-node recompute。profile-off 默认路径的 transition 字段和成本保持为 disabled/zero。

严格容量检查发现冻结的 150 个 stress episodes 含 55 个 `[32760,32760]`、50 个 `[24576,24576]`、45 个混合 topology；由于真实 profile 只覆盖 RTX 4080 SUPER/32760 MB，正式 paired 条件按源顺序筛出 55 个同类 episode，不把未测 GPU 视为已覆盖。profile-off/profile-on 各完成 55×4=220 rows、2,320 jobs，event audit 完成 10×4=40 rows，全部 `status=passed`、0 failed jobs、0 capacity violations，最大模拟峰值 32,756 MB。profile-on 总计计入 load 30,751,273.877 ms、eviction 982,906.773 ms；mean completion 相对 off 增幅为 RR +2.65%、Myopic +1.17%、PredOpt-H5 +1.19%、Oracle +1.57%。远端 102,722 行 event audit 与 summary load/eviction 成本守恒，原始 450 MB event samples 仅留远端。

权威本地记录为 `experiments/EXP-20260824_transition_profile_simulator/metrics.json`、`RESULT.md` 和 `artifacts/paired_rtx4080_super/`；远端 paired 输出为 `/root/autodl-tmp/scheduler/results/processed/EXP-20260824_transition_profile_simulator/paired_rtx4080_super/`。本 pilot 只验证单 GPU slot 的 transition cost overlay，不是 R8-P7 的 1,000 集重跑，不解封 `T_final`，也不新增多 GPU 资源契约。

## R8 Action-Value Audit (2026-08-25)

已完成 `EXP-20260825_action_value_audit` 的 100 集 common-state 审计和
sampled full-event follow-up。local-Q 阶段覆盖 15,364 个 dispatch states，所有
策略的 strict-feasible chosen coverage=1.0；PredOpt-H5 的 local-Q top-1=63.37%，
Myopic=62.89%，但 PredOpt-H5 mean local-Q regret=18,404.7 ms，高于 Myopic 的
18,249.4 ms，PredOpt-H5 的 predicted-vs-true mean Spearman=-0.0208。最大误差
集中在 Qwen3-4B/planner，而不是单纯 cache 或 action width。

full-event follow-up 在同一批记录中分层抽取 100 个 common states，对 717 个
strict-feasible first actions 做完整事件仿真；0 failures，100/100 reference
consistency 通过。Myopic/PredOpt-H5/PredOpt-v2-H5 的 full-event top-1 分别为
32%/32%/33%，现有 local TrueOpt-H5 也只有 40%，说明 `Q_local_H5` 不是可靠的
full-event oracle proxy。当前不把 WAIT/RESERVE 当作已证实的瓶颈，先做
full-event Q/value contract 与 surrogate objective audit；Prefetch、PPO、preemption、
多 GPU 契约和 `T_final` 继续封存。本实验是 diagnostic pilot，不替代正式 1,000 集验证。

权威记录：`experiments/EXP-20260825_action_value_audit/metrics.json`、
`rollout_audit/metrics.json`、`RESULT.md`；审计工具为
`scripts/r8_action_value_audit.py` 和 `scripts/r8_action_value_rollout_audit.py`。

## R8 PredMPC Smoke (2026-08-26)

已创建 `EXP-20260826_pred_mpc_rolling_horizon`，实现 opt-in `PredMPC-H3/H5` 和
`PredMPC-H5-no-terminal`。代码 review 发现并修正两类信息边界问题：rollout 不再遍历
未进入可见状态的 job，也不再读取当前 job 的完整 template suffix；最终版本只使用当前
ready-node candidate window、train-only resource estimates 和冻结 future-artifact continuation
cost。远端编译与定向测试通过。

最终 visible-window smoke 在 10 个固定 validation episodes 上运行 6 policies，共 60 rows；
每个 policy 完成 160/160 jobs，0 failed jobs。mean completion 为：Myopic 176,937.1 ms、
PredOpt-H5 175,860.0 ms、CP-RHO-H5 183,881.3 ms、PredMPC-H3 186,010.2 ms、
PredMPC-H5 187,089.5 ms、PredMPC-H5-no-terminal 185,218.1 ms。
PredMPC-H5 相对 Myopic 为 +10,152.4 ms（+5.74%），相对 PredOpt-H5 为 +11,229.5 ms；
terminal continuation 在该实现中比 no-terminal 再慢 1,871.3 ms。该结果是 smoke 诊断，不是正式方法选择，
不进入 300/1,000 集验证。

权威本地记录为 `experiments/EXP-20260826_pred_mpc_rolling_horizon/metrics.json`、
`RESULT.md` 和 `artifacts/smoke_10_visible_window/`；远端输出为
`/root/autodl-tmp/scheduler/results/processed/EXP-20260826_pred_mpc_rolling_horizon/smoke_10_visible_window/`。
前三版 smoke 目录保留但标记为 invalidated diagnostic。`T_final`、PPO、WAIT/RESERVE、
preemption 和多 GPU 语义继续封存；下一步先做 action-score/terminal decomposition audit，
不自动扩大样本。

## R8 RiskAwareScore Pilot (2026-08-26)

已新增 opt-in `risk_aware` 策略：先按 `plan_gpu_admission` 保留严格可行的
`(ready_node, free_gpu)`，再对当前候选集的紧迫度、GPU/cache-transition 风险和等待年龄做
min-max 归一化，使用等权 `-urgency + resource_risk - age` 排序。该策略不读取未来 DAG
后缀、future artifacts 或 execution truth，仍保持单节点单 GPU、非抢占语义。

远端 10 个固定 validation episodes 上运行 Myopic、RiskAwareScore、PredOpt-H5 和 privileged
TrueOpt-H5，共 40 rows；每个策略完成 160/160 jobs，0 failed jobs。RiskAwareScore mean
completion=174,733.3 ms，Myopic=176,937.1 ms，PredOpt-H5=175,860.0 ms；相对 Myopic
在 10 集中 6 胜 4 负，相对 PredOpt-H5 为 5 胜 5 负。RiskAwareScore mean queue=106,960.8 ms、
GPU evictions=20.8，高于 Myopic 的 85,629.5 ms 和 18.2；deadline miss=0.0%，Myopic=0.625%，
PredOpt-H5=1.25%。每个 RiskAwareScore decision 的 strict-feasible rate=1.0；未触发模拟 OOM/failed-job 路径。

权威记录为 `experiments/EXP-20260826_risk_aware_score/metrics.json`、`RESULT.md`、
`artifacts/smoke_10/`；远端输出为
`/root/autodl-tmp/scheduler/results/processed/EXP-20260826_risk_aware_score/smoke_10/`。
该结果只说明透明当前状态评分在小样本上值得继续审计，不提升为最终候选，不解封 `T_final`，
也不证明未来信息已被有效利用；本版 RiskAwareScore 明确不消费 future artifacts。

## R8 Aligned H5 Score Contract (2026-08-31)

已在本地完成 `EXP-20260831_aligned_h5_score_contract`。新增的
`aligned_predopt_h5`/`aligned_trueopt_h5` 为 opt-in 策略；二者共享当前动作计入、H=5
DAG 层范围、resident-plus-workspace 缓存推进、硬优先级和 tie-break，Predicted 只读
train-only 资源统计与冻结 future artifacts，Truth 只在审计 evaluator 中读取模板
execution truth。transition profile 关闭，旧策略默认路径保持不变。

100 集 common-state action audit 覆盖 15,364 个 decisions、0 failed jobs、无缺失
future artifact。统一 truth key 下，`aligned_trueopt_h5` 内部一致率为 100%；
`aligned_predopt_h5` top-1=66.06%、mean regret=7,978.7ms，旧 `predopt_h5` 为
66.03%/8,065.5ms，Myopic 为 69.54%/7,659.2ms。说明评分口径不一致确实存在，但
对本数据的局部排名改善很小。

100 个状态的 full-event forced-action audit 完成 774 个 branches、0 failures、
100% reference consistency；`aligned_predopt_h5`/旧 `predopt_h5`/Myopic 的
full-event top-1 为 31%/31%/32%，mean regret 为 3,280.0/3,263.0/3,295.1ms。
统一契约没有实质缩小 full-event 差距，主要剩余问题仍是 H5 surrogate 与完整 episode
continuation/JCT 目标不一致。

10 集 paired smoke 为 50 rows，所有策略均完成 160/160 jobs、0 failed jobs；
`aligned_predopt_h5` mean completion=177,058.3ms，旧 `predopt_h5`=175,963.2ms，
Myopic=176,937.1ms；最大模拟峰值显存 32,756MB，低于 32,760MB，0 capacity
violations。该结果是诊断实验，不把 aligned 策略提升为正式候选，不扩大到 1,000 集，
不解封 `T_final`；WAIT/RESERVE、prefetch、preemption、多 GPU 契约和 PPO 继续封存。

权威记录：`experiments/EXP-20260831_aligned_h5_score_contract/metrics.json`、
`RESULT.md`、`artifacts/action_value_audit/`、`artifacts/rollout_audit/` 和
`artifacts/smoke_10/`。

## R8 Pred/True Score Decomposition (2026-09-01)

已完成 `EXP-20260901_pred_true_score_decomposition` 的本地只读审计。输入为上一轮
100 集 aligned H5 common-state records；在同一 `episode_id + decision_index + state_hash`
上逐项比较 Pred/True candidate filter、priority、current/future/total score、rank 和
top-1。覆盖 15,364 decisions、44,690 candidate rows，输入哈希和逐动作 raw score
复核一致，58/58 unittest 仍通过。

关键结论：候选集合不是瓶颈；raw/model/strict 过滤、Pred/True scored action set、
priority/tie-break 和 total 加和全部一致。Pred/True 确定性 top-1 一致率为 65.1979%，
tie-aware 为 66.0570%，两两排序翻转率为 39.0751%。candidate-level future mean
absolute error 为 38,531.9ms，current 为 7,650.2ms；92.45% candidate 的 future
误差大于 current 误差，Pred future 的 signed bias 为 -27,729.1ms。

更重要的是发现 aligned H5 仍存在未来拓扑单位差异：Pred `future_h5` 对所有 1,546
个候选 node 都是 3 个 synthetic scenario、每个固定 5 个 event step；True
`aligned_h5_score` 展开的是后继 5 个 DAG layer，允许每层多个 node。加权真值后继数
为 5.3889，Qwen3-4B planner 为 9.1858；后者 77.8287% 的 candidate occurrence
处于真值后继 node 数大于预测 event 数的状态。此前“Pred/True 只在信息源不同”的
表述需收窄为：动作集合和 tie-break 已对齐，但 future topology 尚未对齐。

权威记录：`experiments/EXP-20260901_pred_true_score_decomposition/metrics.json`、
`artifacts/metrics.json`、`artifacts/topology_metrics.json`、
`artifacts/decision_comparisons.jsonl` 和 `artifacts/candidate_score_comparisons.jsonl`。
下一步先修正 H5 future-topology contract，再重跑同一 100 集审计；不扩大正式矩阵，
不解封 `T_final`，不加入 WAIT/RESERVE、抢占或多 GPU 语义。

## R8 H5 Layer Contract Repair (2026-09-01)

用户确认后仅在本地执行 `EXP-20260901_h5_layer_contract_repair`，远端未连接。
新增 `future_h5_layers` sidecar 和 opt-in `aligned_predopt_h5_layer`；每个 scenario
使用 `layer_offset=1..5`，每层使用 `nodes` 列表，预测节点禁止携带 target DAG identity
字段。旧 `future_h5`、旧策略和 `.scratch` 冻结输入未覆盖；动作空间、单节点单 GPU
语义、WAIT/RESERVE、抢占和多 GPU 均未改变。

本地只找到 8,935 行冻结 future artifact，没有 B05 checkpoint/dataset，因而生成的是
明确标记的 `legacy_event_to_unary_layer_projection`：每个 scenario 恰好 5 层、每层
1 个节点。schema/loader/score 支持多节点层，但本轮不把 unary projection 当作真实
topology predictor。100 集 audit 覆盖 15,364 decisions、44,690 candidates、0 failed
jobs；新 layer policy 的 top-1=66.0570%、mean true regret=7,978.7ms，与旧 aligned
Pred 完全相同。10 集 smoke 为 3 policies、160 jobs/policy、0 failed jobs；全套
unittest 60/60 通过。

拓扑 audit 的真值参考为加权非空层数 2.5107、加权 DAG 节点数 5.3889；预测侧为 5 层
和 5 节点，说明接口已修复但预测信息没有增加。该实验状态为
`passed_diagnostic_unary_projection`，R8-P9c 不标记为完全完成；下一步需补齐
train-only topology predictor 输入后再重跑同一 smoke 和 100 集逐动作审计。正式
矩阵、`T_final`、远端同步继续封存。

权威记录：`experiments/EXP-20260901_h5_layer_contract_repair/metrics.json`、
`RESULT.md`、`artifacts/layer_topology_audit.json`、
`artifacts/action_value_audit/` 和 `artifacts/smoke_10/`。

## R8 Conditional Empirical Topology Baseline (2026-09-01)

已在独立实验 `EXP-20260901_topology_predictor_baseline` 中完成 train-only conditional
empirical topology baseline。训练目标从 `r7_s_train` 模板的 `predecessor_node_ids` 重建
DAG layers，预测分区为 `r7_s_val`；预测 sidecar 使用 `future_h5_layers`，只输出无身份的
节点 prototype，不携带 successor identity、边、执行真值或资源真值。B05 checkpoint、未来
事件和目标标签均未作为特征，旧 artifact、R7 和 `T_final` 未覆盖或读取。

同一 aligned-H5 action audit、拓扑审计和 10 集 scheduler smoke 均通过；多节点层已被评分
入口实际消费，且预测层宽不再固定为 unary。该基线改善了局部 action-ranking 诊断，但 smoke
没有证明 JCT/完成时间收益，预测拓扑规模仍低于 True DAG 参考。因此本实验状态为
`passed_diagnostic_empirical_topology_baseline`，不把 `aligned_predopt_h5_layer` 提升为正式
策略，也不扩大到正式矩阵。

权威记录：`experiments/EXP-20260901_topology_predictor_baseline/metrics.json`、
`RESULT.md`、`artifacts/prediction_artifacts/`、`artifacts/action_value_audit/`、
`artifacts/layer_topology_audit.json` 和 `artifacts/smoke_10/`。下一步只在评审该基线后考虑
learned train-only topology predictor；WAIT/RESERVE、抢占、多 GPU、PPO 和 `T_final` 继续封存。

## R8-P9d Predictor-aligned Topology Plan (2026-09-02)

用户已明确 topology predictor 属于 predictor block。正式 topology predictor 不再使用
R7 scheduler 专用的 `S_train/S_val` 作为训练/验证，而是复用行为/资源预测器的视频边界：
`P_dev=300`（内部 train/validation/test）和 `P_holdout_diag=40`。三类预测器共享视频级
split 和当前可见输入边界，但 topology 使用独立的未来 DAG-layer labels；冻结后才允许
把预测结果推理到 `S_train/S_val/T_final` 做 scheduler 集成评估。

阶段 0 已完成 `P_dev/P_holdout_diag` 的 label coverage/provenance gate：行为 role 样本提供
精确锚点，原始 trace 的 `parent_step_ids` 可重建事件级 DAG；没有用 `S_*` 模板补标签。当前
还未创建 learned experiment、修改模型代码或训练。

## R8-P9d Derived Topology Dataset (2026-09-02)

已在远端生成独立数据集
`/root/autodl-tmp/scheduler/results/processed/topology_predictor_p9d_v1`，本地登记为
`data/manifests/topology_predictor_p9d_v1.json`。P_dev 使用与旧行为 role 文件 17,303 行
顺序键、行为标签和视频 split 一致的 enriched 版本；旧文件的扩展 7,169 行只有
`no_compute_event`/null `source_event_id`，因此没有伪造拓扑锚点。registry 中 21 个截断 ID
通过已登记 `raw_development_to_canonical` 映射解析；raw manifest 缺 hash 时计算 trace 实际
SHA-256 并与样本 provenance hash 比对。

数据按 train/validation/test/holdout 分开写出，行数为 `13,754/2,029/1,520/1,380`，视频数
为 `240/30/30/40`，run 数为 `988/144/108/120`；每个 split 的 features/labels sample_id
完全一致、无重复、无 unmatched anchor。图重建共 18,683 raw events、17,323 future graph
nodes、32,934 edges，missing/forward parent refs 均为 0；同一步无显式 parent 的链式补边为
1,227 条，并已登记为 reconstruction policy。真实未来 labels 保留 node identity/edges，
model_input 只含 observed history/current node/task/stack categorical fields；独立 causal audit
发现 0 个 forbidden-key violation，且与 `S_train/S_val/T_final/T_backup` 视频零交集。原始
trace、旧 artifact、`T_final` 未覆盖/解封；execution_lane 因 raw trace 未提供而标记
`unknown`，不从 runtime 或 model 名称推断。

本地 `scripts/build_p9d_topology_dataset.py` 的 `py_compile` 和 5/5 端到端测试通过，旧
topology baseline 回归测试在 `PYTHONPATH=src:.` 下 5/5 通过。当前结果只闭合数据准备门禁，
不代表 topology predictor 已训练或 scheduler 已获得未来预测收益。

## Same-state Pred/True Score Decomposition (2026-09-01)

为避免把输入版本差异误判为 predictor 差异，使用当前
`EXP-20260901_topology_predictor_baseline` action audit 的同一份记录，按相同
`episode_id + decision_index + state_hash` 重跑旧 `aligned_predopt_h5` 和新
`aligned_predopt_h5_layer`，共同对比 `aligned_trueopt_h5`。两次均覆盖 15,364 decisions、
44,690 candidate rows，输入 SHA-256 相同；候选过滤、priority/tie-break、Pred/True scored
set 和加法契约全部通过。

新 layer policy 的 tie-aware top-1 为 71.0817%，旧 policy 为 66.0570%；mean true regret 为
6,686.3ms 对 7,978.7ms，p95 为 45,452.5ms 对 54,561.9ms。新旧 current 预测逐候选完全相同，
future 预测平均相差 -9,559.9ms，因此总分差异完全来自 future 项。candidate future MAE 从
38,531.9ms 降到 37,327.7ms，但 candidate total MAE 从 35,520.9ms 升到 36,213.5ms；这说明
layer baseline 改善了排序相对关系，却没有完成数值校准。

拓扑审计仍显示预测侧加权 layer/node 数为 1.5885/3.1149，低于 True DAG 的 2.5107/5.3889。
按 candidate context，Qwen3-4B planner 的 future 误差多数被修正，而 Qwen3-VL-8B planner
的部分 cache-hit 场景反而更乐观；下一步不应直接扩大样本或提升策略，而应先做按条件的
future-cost calibration 或 learned train-only topology predictor。权威明细见
`experiments/EXP-20260901_topology_predictor_baseline/artifacts/old_pred_true_decomposition/metrics.json`
和 `experiments/EXP-20260901_topology_predictor_baseline/artifacts/pred_true_layer_decomposition/metrics.json`。

## Verified Facts

- 本地根目录：`/Users/liuqiuwei/Documents/调度`；远端根目录：`/root/autodl-tmp/scheduler`。
- 远端只读审计时间：2026-08-13 14:54（Asia/Shanghai）；主机为 `autodl-container-41d846924e-d62d0601`。
- 远端无 Git 仓库；旧 `P_dev` 清理后，`data/phase3/public/videos/videomme` 有 308 个 `.mp4` 文件名（其中 300 个是仍保留的目标视频，8 个是 `._*` 元数据）；历史审计剩余约 31G，本轮派生生成前实测约 21G，仍不得下载大文件。
- 历史核心 768 条轨迹仍保留并可复用；其正式语义和 split 以现有审计/清单为准，不能与扩展 collection 混称。
- R1 已用 Project OS v1 初始化；R2 已将本地和远端源码目录从 `tracing/` 移到 `src/tracing/`，Python 包导入名保持 `tracing`。目录路径已统一，但远端有 57 个源码/备份文件，本地当前镜像有 29 个文件，内容不是完全镜像。
- 本地两个关键文件与远端迁移前后哈希一致：`videotool_phase1.py`、`phase4_trace_simulator.py`。
- 当前有限前瞻实施规划已同步到远端；本地/远端 SHA256 均为 `f034e1625c1e7f90f260f05286549927cd5fd20ba087a1b713b7593d82181812`。
- 2026-08-15 远端控制面规范化完成：已将本地 `AGENTS.md`、`.project/` 六类 canonical 文件及 archive 通过不带 `--delete` 的 `rsync` 同步到 `/root/autodl-tmp/scheduler`，并创建标准模块目录；远端无 Git 仍保持不变。远端控制文件 SHA256 已逐项与本地一致。
- R5 初步审计：官方 Video-MME 元数据为 2700 条题目、900 个独立视频；600 视频清单包含 340 个可复用视频和 260 个待下载视频。待下载视频按官方 ZIP 尾部索引估算约 30.94 GB 解压空间，接近远端剩余空间，不能一次性落盘。
- 本地下载 pilot 已成功取得 9/10 个文件、约 783 MB；其余重试遇到 Hugging Face CDN SSL 错误，下载器已改为直接使用官方原始 URL 跟随重定向，pilot 不计为完整数据集。
- 历史 R5 batch01 pilot 仅有 9 个文件完成官方大小/CRC/SHA256 校验并上传；该 pilot 不计入当前 r5_batch02–11 正式扩展，未完成的第 10 个不进入正式清单。
- 主库样本为 1280×720、约 1.8–2.1 Mbps，最大单文件约 872 MiB；旧 phase2/phase0 样本已有 640×360 和 256×144 版本。当前采集器 `VisibleFrames.ensure_frame_file` 抽帧时统一缩放到宽 640，未发现下游依赖原始 1280×720 像素。
- 640×360 pilot（`EXP-20260815_video_640x360_pilot`）已在远端完成 6 个 1280×720 视频：输入 1,068,528,024 bytes，输出 807,370,470 bytes，aggregate ratio 0.755591，节省 24.4409%；单视频 ratio 为 0.494763–0.960285。该结果只用于容量估算，不代表最终 R5 编码策略。
- 已建立 `src/tracing/schema/video_provenance_v0_1.json` 和 `docs/video_provenance_and_derivative_policy.md`：每个逻辑视频记录官方 archive 定位、原始文件 SHA-256/字节数/ffprobe 条件、派生文件 `parent_sha256` 以及 retention/deletion gate；当前 600 视频逐视频回填已通过 provenance gate。
- provenance schema、策略文档、collection registry 及 `.project/PLAN|STATE|DECISIONS|HANDOFF` 已同步到远端 `/root/autodl-tmp/scheduler`；2026-08-15 逐文件 SHA-256 与本地一致。远端镜像没有 Python/Node/JSON CLI，本地 JSON 解析通过，远端以字节哈希完成同步验收。
- 2026-08-15 网络边界已验证：远端访问 `huggingface.co:443` 返回 `curl: (7) Failed to connect`，不能依赖远端直接访问中国大陆以外的下载源。后续官方视频/元数据优先在本地获取，再通过带 SHA-256 校验的 rsync 上传；只有在明确提供并验证本机代理端口时才使用端口转发，不猜测代理配置。

## Open Problems and Risks

- 远端没有 Git，当前代码同步依赖路径和哈希清单；后续需要建立 source manifest，而不是直接复制整个工作树。
- 本地环境未安装 `pytest`，因此本轮只能完成 py_compile/import smoke；远端正式验证需使用 finetooling 环境。
- 远端磁盘是高占用状态；R5 已按小批下载、上传、校验并清理 staging，旧 `P_dev` 已清理，后续不得下载大文件，派生数据必须预算在实测余量内或使用可验证的临时路径。本轮 P9d 输出约 2.3 MB。
- R8 P0 的 1,000-episode paired audit 和 S_train 300-episode calibration 输出分别位于远端 `results/processed/r8_predopt_audit_20260818/`、`r8_predopt_calibration_pw10/`、`r8_predopt_calibration_pw20/`、`r8_predopt_calibration_fw20/`；本地摘要在 `experiments/EXP-20260818_r8_optimizer_rl_upgrade/`。三组 score 权重没有被提升。
- CP-SAT/P1 状态：远端 `finetooling` 已固定安装 `ortools==9.9.3963`；R8-P1 1,000-episode矩阵已完成，但 CP-RHO V1 的 time-budget fallback 比例较高且未改善 PredOpt-v2，后续不得直接把它当最终部署策略。
- 视频可以降分辨率，但不能覆盖原视频；建议仅对 R5 新视频生成独立的 640×360 派生版本，在本地转码后再分批上传。低分辨率版本必须有独立 manifest/hash 和 collection ID，不能与 v1 原视频轨迹混称。
- R0a 审计指出 C1 split、causal-v2、资源语义仍存在正式实验风险；这些问题在 R5/R6 前必须由清单和 gate 固化。
- R6 首版因旧模板缺失 lane 和 workload 顶层 source refs 保留为审计 evidence，不作为 canonical；`r6_causal_v2_20260816_fix1` 是唯一当前结果。修正版新增约 440MB，验收后远端约 8.1G 可用；没有下载视频或模型。
- R7-0 的权威300/40 split已从远端同步；此前审计发现的21个下划线截断ID已完成规范化，其中19个使用唯一source前缀、2个使用远端run_id证据。
- 归档恢复测试仍未完成；2026-08-17 用户明确授权对旧 `P_dev` 做范围例外清理，300 条记录已标记 `deleted_after_gate`/`deletion_eligible=true`，其余 300 条 holdout/scheduler 视频仍为 `deletion_eligible=false`，不得继续删除。
- 官方 archive index 的远端直连尝试因上述网络限制已中止；archive index 需要改为本地小型索引获取后同步，不能把失败的直连当作下载完成。
- provenance 回填和清理状态：v1 legacy 64/64 条有效记录且 64/64 含官方 archive 字段；v2 expanded 600/600 条唯一记录仍保留完整 provenance 和官方 archive 字段，其中 300 条 `P_dev` 为 `deleted_after_gate`、300 条 holdout/scheduler 视频仍为 `present`；6 条 640×360 pilot 派生记录仍在独立实验目录且带 `parent_sha256`，不计入 v2 原始视频目标。v1 有 8 条来源清单期望 SHA 冲突警告，未覆盖实际文件。
- R5 最后 batch11 的 19 个视频按官方 archive `uncompressed_size` 合计 2,001,987,030 bytes（1.864496 GiB），已完成容量 gate、本地官方范围下载、远端上传和逐文件 SHA-256/字节数复核。
- `r5_batch02` 已验证远端落盘 26/26（3,133,891,683 bytes），v2 待补视频从 251 减至 225；本地 batch staging 已清理，远端正式视频不删除。
- `r5_batch03` 已验证远端落盘 11/11（2,895,847,835 bytes），v2 待补视频从 225 减至 214；本地 batch staging 已清理，远端正式视频不删除。
- `r5_batch04` 已验证远端落盘 20/20（3,193,062,552 bytes），v2 待补视频从 214 减至 194；本地 batch staging 已清理，远端正式视频不删除。
- `r5_batch05` 已验证远端落盘 30/30（3,083,745,719 bytes），v2 待补视频从 194 减至 164；`I4yye8mUzWg` 的 partial 异常已单文件续传修复，最终 `bad=[]`；本地 batch staging 已清理，远端正式视频不删除。
- `r5_batch06` 已验证远端落盘 35/35（2,777,791,428 bytes），v2 待补视频从 164 减至 129；本地 batch staging 已清理，远端正式视频不删除。
- `r5_batch07` 已验证远端落盘 30/30（2,997,639,724 bytes），v2 待补视频从 129 减至 99；本地 batch staging 已清理，远端正式视频不删除。
- `r5_batch08` 已验证远端落盘 19/19（3,180,615,578 bytes），v2 待补视频从 99 减至 80；本地 batch staging 已清理，远端正式视频不删除。
- `r5_batch09` 已验证远端落盘 28/28（3,180,289,454 bytes），v2 待补视频从 80 减至 52；远端逐文件 SHA-256/字节数 `bad=[]`，本地 staging 已清理。
- `r5_batch10` 已验证远端落盘 33/33（3,119,420,600 bytes），v2 待补视频从 52 减至 19；远端逐文件 SHA-256/字节数 `bad=[]`，本地 staging 已清理。
- `r5_batch11` 已验证远端落盘 19/19（2,001,987,030 bytes），远端逐文件 SHA-256/字节数 `bad=[]`；本地 staging 已清理。
- R5 provenance gate 已通过并完成一次范围例外清理：`video_provenance_v2_expanded_available.jsonl` 为 600 条唯一记录，600/600 含官方 archive 定位和 ffprobe 尺寸/时长字段；其中 300 条 `original_status=deleted_after_gate`/`deletion_eligible=true`，300 条仍为 `present`/`deletion_eligible=false`。删除审计见远端 `results/processed/video_delete_pdev_20260817/`，非目标误删为 0。

## R6 Verified Facts

- 2026-08-16 R6 canonical 远端输出为 `/root/autodl-tmp/scheduler/results/processed/r6_causal_v2_20260816_fix1/`；R6 manifest schema `r6-causal-v2-manifest`，gate=true。输入是只读 768 条 C1 legacy templates/prefixes，输出 4,683 causal prefix rows、9,366×H1/H3/H5 future rows、9,366 train-only resource predictions，以及 20,000/1,000/6,750 C1 train/validation/test-retrospective episodes（743,680/34,160/248,400 jobs）。
- R6 split registry `/root/autodl-tmp/scheduler/data/manifests/video_split_registry_r6_v1_causal_v2.jsonl` 固化 C1 `48/8/8`，以及 32 个 C2 pending candidates + 8 个 backups；候选与 C1 无交集，并排除了 341 个已有 trace IDs、349 个已有 trace hashes。C2 状态仍为 `pending_trace_collection`。
- R6 leakage/causal gate：prefix forbidden-input violations=0，duplicate prefix IDs=0，causal-chain edge errors=0，future probability errors=0，workload forbidden scheduler fields=0；resource fit only 566 C1 train templates / 6,916 train truth nodes，truth rows 保持 engine-only。修正版还把旧模板缺失的 lane 推断为 `cpu/gpu/api`，当前实际为 6,693 GPU / 2,673 CPU。

## R7-0 Boundary Freeze (2026-08-17)

- 已建立 `data/manifests/video_split_registry_r7_v1.json`，固定总量600：`P_dev=300`、`P_holdout_diag=40`、`S_train=120`、`S_val=40`、`T_final=80`、`T_backup=20`。
- 行为预测器和资源预测器共享 `P_dev`；调度器训练/验证/最终测试只使用另外260个 predictor-unseen 视频。`T_final` 在所有模型与策略冻结前保持封存。
- 已将 R7-0 的实验记录写入 `experiments/EXP-20260817_r7_boundary_gate/`，状态为 `passed_id_boundary_pending_trace_collection`；没有猜测视频ID，调度260视频由剩余池按可复现元数据分层顺序分配。
- 已发现旧 `PLAN.md` 中 `U_reservoir=281` 与最近审计不一致，已更正为当前调度池260；旧64-video资源预测器明确标记为legacy baseline，不能冒充300-video主线资源模型。
- 本轮本地控制面、边界清单和生成脚本已更新；控制面同步到远端后逐文件SHA-256一致。
- R7-1资源输入已在远端生成：P_dev覆盖300视频、1,240 runs、17,303 compute rows、1,128 run级静态metadata和300 video级metadata；错误事件574条保留为观测标签，不被改成成功或零值。
- R7-1资源预测器已在远端以正确的`PYTHONPATH=src`模块入口完成：新模型输出位于`resource_predictor_r7_pdev300_20260817_retry2`，开发拟合13,754 train rows + 2,029 validation rows，最终holdout为1,380 rows/40 videos；runtime/load/peak的选定模型分别为`lgb_point_group_q99_interval`、`group_quantile`、`group_quantile`。
- R7-1接口 gate 已完成：`src/tracing/scheduling/event_engine.py` 将 `SchedulerStateView`、`ExecutionTruthProvider`、`FutureProvider` 分离；远端 10/50 deterministic contract smoke 全部通过，验证 H=0 不读 future、预测状态不含 runtime/load/workspace/status、当前真值和 H 外 suffix 变更不影响 state/reveal；事件循环接入及混合 stack collector 修复后，远端 unittest 为32/32，本地为40/40。
- 统一事件循环已做最小接入：远端 `src/tracing/analysis/workload_v02_simulator.py` 在每个 GPU dispatch 前构造 scheduler state，执行 truth 仍由 engine-only provider持有；真实 legacy validation replay 的 10/50 episode 分别产生 1,332/6,744 个 dispatch-state 对，全部匹配。该回放不计入 R7 正式 workload。
- R7 trace manifest 已冻结并同步：`r7_trace_manifest_v1.jsonl` 为 640 行（S_train/S_val 共160个 predictor-unseen 视频 × 2 动态 stack × 2 动态 baseline），`r7_trace_pilot_manifest_v1.jsonl` 为 8 行/2 视频；任务只取公开问题的 question/options，不复制 answer。
- R7 pilot 已在远端完成：8/8 run status success、8/8 trace validator valid；两套 stack 和两种动态 baseline 各覆盖4条。2个解析错误均作为 trace 事件保留，并由成功 retry 恢复；source manifest 无 answer，planner input 无 answer/gold 字段。验收报告为 `results/processed/r7_trace_pilot_20260817/pilot_report.json`，本地副本在 `experiments/EXP-20260817_r7_trace_pilot/remote_pilot_report.json`。
- 首次错误入口产生的0字节pickle已标记为无效审计产物；重试exit=0且models.pkl约14MB。资源报告的development/holdout overlap=0，未来/答案/目标/ video_id feature leakage flags均为false。

## Latest Predictor Data Standard

- 行为预测器最新正式结果：远端 `results/processed/behavior_nn_v1_r2/`，B05 Masked GRU 在 final holdout 固定 epoch=19、3 个 seed 的 joint accuracy 为 0.8101/0.8022/0.8014，均值约 0.8046。
- 主训练/开发数据：300 个 VideoMME 视频、1,240 条 run、17,303 条 role rows、5,889 条 tool rows；role split 为 train 240 视频/988 runs/13,754 rows、validation 30/144/2,029、旧 test 30/108/1,520。B05 只用 train 拟合、validation 评估，旧 test 不用于模型选择。
- 最终未见评估：40 个从未参与开发的视频、120 条成功 run、1,380 条 role rows、526 条 tool rows、766 条 prefix；只在配置冻结后评估。
- 后续 resource predictor、workload 和 scheduler 主线沿用上述 300/40 数据边界；64-video/48/8/768 legacy 只用于经典预测器、R6 和兼容性对照，不与新主线混合。

## Current Mainline

`R0 remote audit → R1 Project OS → R2 src/tracing migration → R3 canonical docs → R4 data/source registries → R5 trace expansion → R6 legacy causal-v2 audit → behavior_nn_v1_r2 freeze → R7 resource/scheduling experiments → R8 optimizer/RL upgrade`。

## Latest Verified Outputs

- `.project/` canonical control plane
- `docs/source_layout.md`
- `data/manifests/remote_inventory_20260813.json`
- `data/manifests/trace_collection_registry.json`
- `data/manifests/source_sync_registry_local.json`
- `data/manifests/source_sync_registry_remote_20260813.json`
- `data/external/videomme/test-00000-of-00001.parquet`
- `data/manifests/videomme_600_source_v2.jsonl`
- `data/manifests/videomme_600_download_v2.jsonl`
- `data/manifests/videomme_600_selection_report_v2.json`
- `data/manifests/videomme_600_size_estimate_v2.json`
- `data/manifests/videomme_official_archive_index_v0_1.json`
- `data/manifests/video_provenance_v1_legacy_core.jsonl`
- `data/manifests/video_provenance_v1_legacy_core_coverage.json`
- `data/manifests/video_provenance_v2_expanded_available.jsonl`
- `data/manifests/video_provenance_v2_expanded_coverage.json`
- `data/manifests/videomme_600_missing_batches_v2.json`
- `data/manifests/videomme_600_upload_batch01_v2.jsonl`
- `data/external/videomme/staging/batch01/report.json`
- `data/manifests/remote_layout_audit_20260815.json`
- `experiments/EXP-20260815_video_640x360_pilot/RESULT.md`
- `experiments/EXP-20260816_r6_causal_v2/RESULT.md`
- `data/manifests/video_split_registry_r6_v1_causal_v2.jsonl`
- `docs/r6_causal_v2_data_contract.md`
- `src/tracing/schema/causal_v2_contract_v0_1.json`
- `docs/video_provenance_and_derivative_policy.md`
- `src/tracing/schema/video_provenance_v0_1.json`
- `experiments/EXP-20260818_r8_optimizer_rl_upgrade/CONFIG.md`
- `experiments/EXP-20260818_r8_optimizer_rl_upgrade/RESULT.md`
- `experiments/EXP-20260818_r8_optimizer_rl_upgrade/p0_predopt_metrics.json`
- `experiments/EXP-20260818_r8_optimizer_rl_upgrade/p0_calibration_summary.json`
- `experiments/EXP-20260818_r8_optimizer_rl_upgrade/cp_rho_matrix_report.json`
- `experiments/EXP-20260818_r8_optimizer_rl_upgrade/scheduler_results.jsonl`
- `experiments/EXP-20260818_r8_optimizer_rl_upgrade/milp_sanity_report.json`
- `experiments/EXP-20260818_r8_optimizer_rl_upgrade/milp_sanity_results.jsonl`

## Next Step

R6 只保留为 legacy reference。P9d future-content、P9e reduced role/family 和 P9f acceptance diagnostic 均已完成；P9e 的 B 是按 validation structure score 选出的当前简化结构参考，P9f 证明其结构误差显著优于 empirical baseline，但 raw probability calibration 未改善，粗粒度 role/family runtime proxy 覆盖不完整且系统性低估长尾，不能据此冻结 scheduler 候选。下一步先设计并确认 resource-aware future-content 或 conditional aggregation 实验，解决 `model_id/node_type/input_scale/cold_warm` 缺失导致的 load/memory 不可辨识问题；在 resource/predictor acceptance 通过前不进入 `S_train/S_val` 只推理集成，不接入正式调度策略，不扩大到正式矩阵，不解封 `T_final`。`Q_local_H5` 与 full-event value、WAIT/RESERVE、R8-P7、RL 冻结和 transition 扩展仍按原边界处理；C2/电梯轨迹继续保持独立 collection。

## Verification

- 2026-08-24 transition simulator pilot：本地 unittest 49/49、JSON/py_compile/shell 校验通过；远端 targeted contract tests 6/6、paired profile-off/profile-on 220+220 rows、event audit 40 rows 均 `passed`，0 failed jobs、0 capacity violations；本地/远端 canonical control-plane 与 experiment summary SHA-256 一致。
- 2026-08-25 Action-Value Audit：local-Q 100 集/15,364 states 通过；full-event follow-up 100 states/717 branches 通过；新增 audit-only forced-action hook 的 py_compile 与定向测试通过；未读取 `T_final`，远端仅做输入只读回收。
- 2026-08-26 RiskAwareScore：本地相关 scheduler/event/transition tests 18/18 通过；远端 RiskAwareScore/PredMPC/PredOpt contract tests 5/5 通过；远端 10 集 paired smoke 40 rows、0 failed jobs；本地/远端 simulator 源码 SHA-256 一致；未读取 `T_final`。
- 2026-09-02 P9d shared causal GRU：本地 py_compile、JSON、shell 和契约测试通过；远端 PyTorch forward/loss/backward 4/4 通过，12 个 run 全部完成；本地恢复后验证 run-manifest 全部 artifact hashes、36 个 prediction files、44,361 rows、概率归一、identity/edge/resource-truth leakage 和 sample coverage 均通过；远端与本地 root metrics/run-manifest hashes 一致，未读取 `T_final`。
- `project_os.py init` 成功。
- 本地 `src/tracing` 全部 Python 文件 `py_compile` 成功。
- `PYTHONPATH=src` 下核心导入 smoke 成功。
- 本地 `PYTHONPATH=src python3 -B -m unittest discover -s tests -v`：40/40 通过。
- 远端 finetooling `PYTHONPATH=src python -B -m unittest discover -s tests -v`：32/32 通过。
- 本地和远端直接执行 `src/tracing/...` 的 validator/simulator CLI help 均通过。
- 远端 py_compile 对真实 `.py` 文件通过；6 个 `._*.py` 是 AppleDouble 元数据，不是可执行源码，已排除且未删除。
- 远端 `src/tracing` 移动后目录和两个关键文件哈希已核验；远端 finetooling import smoke 成功。
- R6 远端 `build_r6_causal_v2.py` smoke 与正式修正版均完成；正式 `r6_gate_report.json` gate=true，所有 H1/H3/H5 scenario probability sums=1，workload summaries 的 forbidden scheduler fields=0。正式输出 hash 与输入 hash 保存在远端 `r6_manifest.json`，本地实验记录见 `experiments/EXP-20260816_r6_causal_v2/`。
- R7-0 本地/远端边界清单 JSON 解析、600计数、21条ID修复、provenance join和零交集检查均已通过。

**R0 执行完成 (2026-09-11):** 按冻结设计在 v3.1 节点本体上完成 Stage 0（六项断言全过）+ smoke（80 run / 932 节点 / 9.0 s）+ 正式全量（15,481 节点 / 36.6 s，CPU，LightGBM 4.7.0 / numpy 2.5.3 / scipy 1.18.1）。裁决 **Core GO PASS**：runtime validation 改善 **74.9%**（bootstrap CI [0.704, 0.824]，≥15% 且下界>0），test 82.1%、holdout 20.2%；校准 `max_τ` = 0.0338/0.0461/0.0161（≤0.05）；load occurrence Brier val 0.0175 / test 0.0033 / holdout 0.0013（固定 logistic 参考 0.106/0.104/0.096），duration PB val 41.7 ms。**memory 降级 descriptive-only**（validation 校准 0.127 > 0.10；覆盖 9,816/15,481，仅 planner/generate/image-qa/image-grid-qa/yolo 有测量）。消融 C0→C4 = 941.4→809.2→809.6→809.6→**380.0**：增益来自 exec_class 与 C4 上下文（baseline + `prefix_model_reuse`）；`workload_scale` 全池零方差（`clip_len` 0/15,481 非空，`query_char_len`/`nested_api_call_count` 恒 0）——接口缺口。holdout 长尾由 2 个 stall（planner 45 min、image-qa 68 min）主导（Inclusive PB 5,264 vs Normal 319.8 ms），指向缺失的执行前 cold/startup 上下文。产物：`experiments/EXP-20260911_p9d_r0_oracle_signature_ceiling/`（`RESULT.md`、`metrics.json` sha256 `832ae291…`、`stage0_audit.json`、逐节点预测、`run_manifest.json`）；门禁记录新增 `.project/EXPERIMENT_GATE.json`（首次创建，含 R0 条目）。未训练 topology/behavior 模型、未接 scheduler，`S_*/T_final` 未动；holdout 评测后不得再改 contract/模型/特征。下一步（建议）：J 系列限定范围启动（runtime 为主、load 为辅、memory/tail 不作可辨识声明），或先做接口扩展议题（真实 token 长度、帧数/分辨率、cold/warm 状态、batch 上下文）。

**R0.1 工作量代理探针 (2026-09-11):** 为回答"工作量大小能否用现有字段补出"，新开 `EXP-20260911_p9d_r0p_workload_proxy_probe`（不碰冻结 holdout）：F0=R0 C4 对照、F1=+trace 代理（`tool_input_len`/`frame_count_before`/`frame_count`，覆盖 7,078 工具节点）、F2=+题目大小（11,889 行）+视频元数据（14,939 行）。结果：validation PB 380.0→384.6→386.0、test 261.1→258.0→278.9——**无有意义改善**（差异在选型/噪声范围、方向不一致）。模型确实使用了这些字段（F2 gain：`tool_input_len` 5,161、`video_duration_s` 3,210、`question_chars` 2,538，仅次于 `exec_class` 134,780 与 `prefix_model_reuse` 8,417），但与已有特征高度冗余、未提升可辨识性。结论：**"工作量大小"不能靠现有记录补出**；真正缺的是模型侧一阶量（真实 prompt/completion token、图像/张量尺寸、批处理/并发状态），只能插桩重采获得；另一种同样成立的解释是给定 `exec_class`+上下文后剩余方差本就以噪声/不可见因素为主。前置约束：P_dev 的 300 个视频已删除，重采需重下约 30GB 或新建视频池。产物：`experiments/EXP-20260911_p9d_r0p_workload_proxy_probe/`（`RESULT.md`、`metrics.json`、`joined_table.jsonl.gz`）；门禁与 INDEX 已登记。未修改任何冻结数据集，`S_*/T_final` 未动。下一步待用户决策：A 进 J（限定范围）/ B 插桩重采议题 / C 小规模插桩 pilot。

**J 系列设计评审与冻结 (2026-09-11):** 在 v3.1 + R0/R0.1 之后起草并冻结 J 系列设计 v3（`docs/p9d_j_series_design.md`）。两轮网页评审：v1 裁决 MODIFY（4 P0：B1 重定义为 frozen backbone + `sg(q(A))` 两段式、废弃旧 ResourceQScore、joint loss 未写死、checkpoint/NI/width 语义）；v2 复核又发现 3 P0（load NI 的 CI 方向写反两处、成功判定不得复用 validation）并明确其余无阻塞。v3 修正全部 P0 并冻结：五变体（J0/B1/J1/J2/J3）+ OracleAttr 诊断、共享 backbone 初始化、统一 attribute-head manifest、`L_total` 固定等权、runtime 单一 primary、load hurdle 分次要、Core success 与 load NI 在冻结后的 `P_dev/test` 一次性判定、NI 逐 endpoint 的 metric+direction+δ manifest（越低越好用 `CI_upper(Δ)<+δ`；越高越好用 `CI_lower(Δ)>−δ`）、checkpoint 先 NI-feasible 再 argmin RuntimeQScore、width 退出正式目标（structure = future length/termination）、T=1、首轮不加 GradNorm/PCGrad、holdout 完全退出正式决策。评审存档 `docs/research/2026-09-11_j_series_design_review.md` 与 `..._review_v2.md`（status=valid）。未改代码/数据/实验，`S_*/T_final` 未动。下一步：本地实现（安装 torch、J 数据管线、J0→backbone→B1→J1/J2/J3，<2 GPU-hour）。

**J 实现规划评审与 v2 修订 (2026-09-11):** J 实现规划 v1（数据管线/代码复用边界/训练协议/环境/门禁）送网页评审，裁决 **MODIFY before accept**，4 个实现级 P0：(1) 资源 join 必须用 `(run_id, node_id)` 复合键，Stage 0 断言两侧唯一、unmatched fail-closed，并区分 `N_unique=15,481` 与 `anchor×future-slot` 展开的监督实例数；(2) `prefix_model_reuse` 需要节点自身 `model_class`，属 future-label 派生——J 中删除（本轮），R0 RESULT 已加口径说明；(3) structure target 固定 `bounded_future_length=min(剩余 compute,5) ∈ {0..5}` + termination，censored 的 `L_H=5` 为窗口内精确长度、全部样本进 CE；(4) patience 与 NI-feasible 选择冲突——取消 early stopping、固定 30 epochs 后从全部 epoch 选 NI-feasible → argmin RuntimeQScore。已修订：`docs/p9d_j_series_design.md`（v3.1：c 仅 `stack/baseline`；长度语义；固定 epoch；固定 bootstrap indices B=1000 全复用；endpoint 成对 mask、空 replicate invalid、<95% fail-closed；正式 device/batch 统一；J0 顺序在 backbone 之后；holdout 默认不入管线；partial 标记）与实现规划 v2 `.scratch/chatgpt_j_impl_plan_v2_zh.md`（新入口 `j_series_train_eval.py` + 小共享模块 `j_series_common.py`，legacy 脚本不动）。评审存档 `docs/research/2026-09-11_j_impl_plan_review.md`（status=valid）。未改代码/数据/实验，`S_*/T_final` 未动。下一步：最终确认（可选）或开工：安装 torch → J 数据管线 → backbone → J0/B1 → J1/J2/J3。

**J 实现规划 v2 验收通过 (2026-09-11):** 最终确认轮裁决 **“J IMPLEMENTATION PLAN v2 ACCEPT（可开工）”**：4 个 P0 全部关闭（复合 join 键与实例数口径、GT-derived `prefix_model_reuse` 移除、`bounded_future_length ∈ {0..5}` + termination、取消早停固定 30 epochs 后 NI-feasible → argmin RuntimeQScore），无新阻塞；评审确认其余保护措施（固定 bootstrap indices、endpoint 成对 mask、空 replicate fail-closed、正式 backend/effective batch 统一、backbone 先于 J0/B1、holdout 默认不入管线、partial 不产出结论、legacy 脚本不动）已足以把风险收敛到实现 bug 与训练稳定性。评审存档 `docs/research/2026-09-11_j_impl_plan_review_v2.md`（status=valid）。下一步：开工实现（安装 torch → J 数据管线 → backbone → J0/B1 → J1/J2/J3 → test 判定）。未改代码/数据/实验，`S_*/T_final` 未动。

**J 本地环境就绪 (2026-09-11):** 按用户要求用 conda 新建独立环境 `scheduler`（`D:\anaconda\envs\scheduler`，Python 3.10.21），并安装 `torch 2.6.0+cu124`（CUDA 12.4 wheel）与 `numpy 2.2.6`。验证：`torch.cuda.is_available()=True`，设备 `NVIDIA GeForce RTX 3060 Laptop GPU`，CUDA 矩阵乘法 smoke 通过。后续 J 系列所有正式 run 固定使用该环境与 device（不混用 CPU/GPU、不跨变体调 batch）。未改代码/数据/实验，`S_*/T_final` 未动。

**J 数据管线完成 (2026-09-11):** `scripts/build_j_dataset.py` 在 conda env `scheduler`（torch 2.6.0+cu124）下运行通过：复合键 `(run_id, node_id)` join、unmatched fail-closed、holdout 默认排除。产物 `results/processed/j_series_dataset_v1/`（登记 `data/manifests/j_series_dataset_v1.json`）：行数 13,754/2,029/1,520；监督槽位（anchor×future-slot）48,343/7,195/5,387 = 60,925；唯一资源节点 14,413（= 15,481 − 1,068 holdout）；`bounded_future_length` 直方图 `{0:2608, 1:1350, 2:1240, 3:1164, 4:1102, 5:9839}`；termination 6,870/994/745；load 计账 zero 55,747 / positive 4,886 / missing 292（合计=60,925）；memory 37,553 / 23,372。Stage 0 断言（键唯一、覆盖、长度/终止/负载计账一致）全部通过。实验目录 `experiments/EXP-20260911_p9d_j_series_joint_resource/`（config、run.sh）与门禁/INDEX 已登记。下一步：`j_series_common.py` + `j_series_train_eval.py`（新入口，legacy 不动）→ Stage 0/smoke → backbone → J0/B1 → J1/J2/J3 → test 判定。未改 v1–v3 数据集与既有实验，`S_*/T_final` 未动。

**J 系列正式结果 (2026-09-11):** 30 epochs ×（backbone 3 seeds + 5 变体 × 3 seeds）在 RTX 3060 本地完成（backbone 312s + 变体 1371.5s + test 13s）。冻结 test 判定一次：**无 Core GO**（load-duration 5% 非劣逐 seed 失败）。Runtime 主端点（Δ vs B1, raw-ms RuntimeQScore）：J2 −49.0/−65.4/−29.9（3/3 seeds CI 上界<0）、J3 −78.1/−77.1/−64.0（3/3 显著更优）；J1 +26.4/+19.9/+73.8、J0 +115.1/+119.4/+147.4（显著更差）。Load Brier：J1/J2/J3 通过，J0 fail；load-duration：J2 仅 seed22 通过（差 2.5% 边际），J3 0/3，判定按 all-seed（worst-seed）读法。J2 seed22 是唯一通过全部逐 seed 判据的 run，已披露但不得作为 NI/Core GO 证据。产物：`experiments/EXP-20260911_p9d_j_series_joint_resource/`（RESULT.md、run_manifest、summaries、artifacts/checkpoints 15 个哈希已核验）；原始运行 `outputs/j_series_joint_resource/`（test_eval.json 冻结）。独立审计（AUD 口径已修正+hash 闭环）与独立评审（P1 已修）记录于 `docs/research/2026-09-11_j_result_audit.md` 与 `docs/research/2026-09-11_j_result_review.md`。结论：属性接口信息是主要增益来源（J2/J3≫J1/J0；与 R0 方向一致），端到端接口（J3）因 load 非劣未过而按负迁移条款拒绝。未接调度器；`S_*/T_final` 未动。

**J 机制诊断 (2026-09-11, exploratory/validation-only):** 针对"runtime/occurrence 大幅改善、load-duration 微降"完成三项判别诊断：D1'（checkpoint 头交换）→ J2/J3 头放到 B1 表示上 +80%~+170%、B1 头放到 J2/J3 表示上 ~+17%、配对后仅差 1–4% ⇒ 表示与头强共适应，非表示丢信息；D2（q(model_class)）熵 0.33→0.22→0.17、top1 0.90→0.93→0.96、边际 KL 0.008–0.017 ⇒ 接口确实变硬（因果未证）；D3（log/raw 分解）log 空间三变体几乎相同（0.108–0.116）、raw 差异小且噪声大 ⇒ 尺度问题不是主因。原"从零 probe"因未过常数基线自查（probe 834 vs B1 头 284 vs 常数 502）弃用。记录：`docs/research/2026-09-11_j_mechanism_analysis.md`；RESULT §6 已引用。结论：时长微降是共享资源头被执行耗时/发生概率主导下的共适应代价，修复方向 = 给 duration 独立/条件化容量（J4a 首选）。待用户决策。

**J4 设计冻结 (2026-09-11):** duration 解耦分支实验。v1 网页评审 MODIFY（P0：test 不可作第二 confirmatory 判定 → 方案 B；P0：checkpoint 选择需加 duration validation gate）→ v2 修订（validation 为正式判定层、test 仅 exploratory second look；两层选择门；H2 改用既有 CI_upper(Δ_runtime vs B1)<0；臂命名/解释降级为 decoupling ablation；hidden=64 冻结；J3 参照 inherited frozen）→ 复核 **ACCEPT（可冻结执行）**。文档 `docs/p9d_j4_duration_branch_design.md`（frozen v2）；评审 `docs/research/2026-09-11_j4_design_review.md`；门禁已登记 `EXP-20260911_p9d_j4_duration_branch`（pending execution）。成本 ≈15 GPU-min。待用户批准执行。

**J4 结果 (2026-09-11, negative):** duration 解耦分支未修复 load-duration 端点。J4a 0/3 seeds 无可行 epoch；J4b 仅 seed11/22 在 ep7–9 可行，但选中 RuntimeQScore 1187/1285（B1 912/895）→ H1 不成立；validation CI（validation_ci.json）：H1 runtime 上界>0、Brier 上界 +0.019/+0.044、duration NI 越界；H2 dur vs J3 上界 +14.2/+15.0 → 不成立；seed33 无 checkpoint。同期 epoch30：J4 runtime 741–798 优于 J3 814–898，但 duration 304–327 差于 J3 260–291 → 存在真实权衡；J4b>J4a（min duration 270.8–288.2 vs 288.0–301.7）说明共享特征携带 duration 信息；**竞争假设被否定**。按预注册负结果规则拒绝该方向（不再拆 hidden）。Stage 0 回归检查 J3 max_abs_delta=0；审计 PASS；评审 CONDITIONAL→4 项 P1 已修；test 仅 exploratory（second look），无 confirmatory 声明。产物：`experiments/EXP-20260911_p9d_j4_duration_branch/`（RESULT.md）与 `outputs/j4_duration_branch/`。下一步选项：J4c 条件化增强，或停止联合训练线回到属性接口 + R0 主线（待用户决策）。

**J duration 退化归因 (2026-09-11, validation-only):** 完成"为什么 J3 的 load-duration 会退化"的收尾诊断（预测分布对比 + 分位数/类别/量级分解 + 配对 bootstrap CI）。结论：**不是零件损坏，而是拟合位置的重新分配**——τ=.90 是退化主源（seed11 +33.2 [23.9,46.5]、seed22 +30.1 [21.4,38.3]、seed33 ≈0），τ=.95 反而 3/3 seeds 改善（−12.0/−7.5/−12.1），中位数不变；J3 的中位数校准其实更好（q50 覆盖率 0.333→0.553/0.363/0.463），分布更收紧。类别上 VL-3B 一致略差、Q3-4B 多数变差、VL-8B 2/3 seeds 明显改善；量级上短加载改善、中段略差；退化结构 seed 间不稳定（seed11 由 top5% 槽位贡献 74% 总差；seed22 广泛小偏移；seed33 总体略好）。⇒ J4 解耦修不好是必然（无单独"坏路径"）。记录：`docs/research/2026-09-11_j_mechanism_analysis.md` §6；GPT+文献建议：`docs/research/2026-09-11_j4_literature_and_gpt_ideas.md`。待用户决定收尾或试修复。

**J 线收束 (2026-09-11, 用户决策 A):** J 系列与 J4 全部结束并归档；不再做 J5 修复实验。最终图景：R0 oracle 上界（runtime val 74.9%/test 82.1% 改善）→ J3 在 deployable 约束下取得 runtime 显著收益（3/3 seeds，相对降低 5.4–8.6%）但 load-duration 以 +5.2%/+5.3% 贴线越界（无 Core GO）→ J4 解耦负结果（拆开更差）→ 归因：分布内拟合重分配（τ=.90 差、τ=.95 好、中位数校准更好），无坏路径可修。记录：`experiments/EXP-20260911_p9d_j_series_joint_resource/RESULT.md` §8、`experiments/EXP-20260911_p9d_j4_duration_branch/RESULT.md`、`docs/research/2026-09-11_j_mechanism_analysis.md`、`docs/research/2026-09-11_j4_literature_and_gpt_ideas.md`；决策：`.project/DECISIONS.md` 2026-09-11 两条；门禁：`EXPERIMENT_GATE.json` 两条记录（J completed / J4 completed_negative）。当前无进行中的实验；`S_*/T_final` 未动，调度器主线（R8-P4/P5）仍封存。下一步待用户选择方向。

**J 预测器集成候选冻结 + 验收审计 (2026-09-11):** 新实验 `EXP-20260911_p9d_j_predictor_acceptance`。选择规则（预注册）：validation RuntimeQScore argmin → **主候选 J3:seed11**（814.4），健壮性 J3:seed22/33，回退 B1:seed11；checkpoint 已复制并哈希。审计（validation）：结构 len_mae 0.0473/term_acc 0.979，内容 7 字段 0.9013，行为 0.9444/0.8396，runtime pb 845.0（覆盖率 .556/.917/.968），load 发生 Brier 0.0135/ECE 0.0289，load 时长 pb 326.9；**memory 头未训练（禁止暴露）**。可辨识性：runtime 误差主导维度=model_class（接口已携带）；**residency 缺失被量化**——validation 加载事件 214/242 全为冷启动，reuse=True 时通常完全不加载，热加载仅 VL-8B n=28；clip_len 100% 缺失。文档：`experiments/EXP-20260911_p9d_j_predictor_acceptance/CANDIDATE.md`；下一步=接口打包 → S_train/S_val 只推理 smoke → 约定调度器级验收标准 → 小规模 PredOpt 重跑。`T_final` 仍封存。

**J 预测器接口打包 + 契约 smoke (2026-09-11):** 新增 `scripts/pack_j_predictor_artifacts.py`，把 J3:seed11 打包为调度器可读的 future artifacts：`j_future_h1/h3/h5.jsonl.gz` + `j_future_h5_layers.jsonl.gz`（layer-H5 侧车，schema `scheduling-future-h5-layer-v1`，校验通过）+ 每步 `resource` 扩展块（runtime/load 分位数与概率）。validation 全部 2,029 anchors，12s；调度器自带 `load_future_artifacts()` 成功读取全部 2,029 节点（mock 目录按 legacy 命名）。记录：`experiments/EXP-20260911_p9d_j_predictor_acceptance/integration_smoke.md`。关键事实：调度器现有策略用自己的资源表，预测器的资源分位数需要新适配层/策略才会被消费；raw_action/execution_lane 为启发式映射。S_* 侧 smoke 待做：S_* traces 位于 `results/raw/r7_trace_full_20260817`（640 runs），缺 anchor feature builder（把事件链转成预测器输入）。

**S_* 推理 smoke 完成 (2026-09-11):** 新增 `scripts/build_sstar_predictor_anchors.py`（复用 P9d 字段派生）+ `scripts/sstar_inference_smoke_audit.py`；在 640 条 S_* trace（`results/raw/r7_trace_full_20260817`）上构建 **9,575 anchors（0 跳过，23s）**，J3:seed11 推理并打包 scheduler artifacts（24s，loader 兼容），对真实执行结果审计（5s，只对比）：next-role 0.912 / next-family 0.803 / runtime pinball 1,045ms / load Brier 0.0225 / 长度误差 0.408；域内参照（P_dev val）为 0.944 / 0.840 / 845 / 0.0135 → **无崩溃但可测退化**（预期 OOD：新栈 stack_a/yolo26n、部分上下文取值 OOV）。分组：s_train≈s_val；star 明显优于 langgraph_react（role 0.947 vs 0.878）。记账：`experiments/EXP-20260911_p9d_j_predictor_acceptance/integration_smoke.md` §5。剩余：调度器级验收标准 + 资源适配层决策 + 小规模 PredOpt 矩阵。

**Forecast-aware scheduling 规划 (2026-09-11, 网页评审):** 核心问题转为"未来行为预测是否影响调度"；建议停止优化 predictor，做 forecast-aware ablation。矩阵：E0 No-Future / E1 Resource-only / E2 Topology-only / E3 Full J3 / E4 Oracle。Primary = mean completion time + deadline 指标；paired bootstrap（1,000 episodes）。资源表作为消融 A/B/C（静态表 / J3 预测 / oracle），memory 不进入主调度。OOD 退化分 predictor 层与 scheduler 层两层报告，不做 S_* 调参。阶段：Phase 0 接口验证（~1 天，100 episodes）→ Phase 1 小 pilot（100 episodes，E0/E2/E3/E4，gate: E3−E0<1% 不推全量）→ Phase 2 正式矩阵（1,000 episodes × 6 策略 × 3 seeds，本地可跑）。规划记录：`docs/research/2026-09-11_forecast_aware_scheduling_plan.md`。待用户确认后启动 Phase 0。

**Forecast-aware Phase 0 通过 (2026-09-11):** J3 artifacts 直接进入调度器矩阵：node_id 免映射、覆盖 8,935/8,935、0 失败；100 eps × 3 策略本地 31–33s。**E2 预览（只换未来结构、资源仍用静态表）：predopt_h5 平均完成时间 152,782ms vs 旧 provider 158,169ms（−3.4%），deadline miss 3.25% vs 4.06%，oracle gap +7.9%**；myopic/oracle 两 provider 逐位一致（接口纯净）。当前策略忽略我们的 `resource` 块 → E3 需资源适配层（opt-in 新策略名），E1 口径待冻结。报告：`experiments/EXP-20260911_forecast_aware_scheduling/PHASE0_REPORT.md`；下一步 Phase 1 pilot（E0/E2/E3/E4，100 eps）。

**Forecast-aware Phase 1 pilot (2026-09-11):** 100 episodes × 5 臂。**核心结论：未来结构预测对调度有显著正向影响**——E2（J 拓扑 + 静态资源）vs E0（无未来）：completion −8,176ms CI[−10,675,−5,771]、queue −9.6%、miss −1.38pp；E4 oracle 相对 E2 仍 −11,207ms（空间大）。**资源臂 E3（J 拓扑 + J 资源）显著变差（+7,507ms）**，原因已量化：加载语义不一致（表=每步条件正加载中位数 vs J=occ×duration；31,884/35,362 步 occ<0.1 被折价）、CPU lane 方向相反、runtime 尺度 0.34、current/future 混用两套尺度。记录：`experiments/EXP-20260911_forecast_aware_scheduling/PHASE1_REPORT.md`；适配层新增 `predopt_h*_jres` 策略（不改既有策略）。下一步：E3 语义标定（不得在 S_* 拟合）→ 重跑 pilot → Phase 2 正式矩阵。

**Forecast-aware Phase 2 正式结果 (2026-09-11, 1000 episodes):** **未来结构预测显著改善调度**——E2（J 拓扑）vs E0（无未来）：completion −17,065ms CI[−18,692,−15,458]、queue −15,575、miss −2.34pp；vs 旧 provider（B05）：−12,583ms CI[−13,962,−11,323]、miss −1.83pp。**资源直接替换显著有害**（E3a vs E2：+16,012ms CI[+14,480,+17,611]；加载块增量≈0）；oracle 仍比 E2 好 25,319ms（E2 距上界 +14.7%）。报告：`experiments/EXP-20260911_forecast_aware_scheduling/PHASE2_REPORT.md`；运行 `outputs/phase2_r7_j_1000/`、`outputs/phase2_r7_b05_1000/`。边界：validation 1,000 集、确定性策略、episode 配对 bootstrap；T_final 封存；未在 S_* 拟合。待独立审计/评审后定稿。

**Forecast-aware Phase 2 定稿 (2026-09-11):** 通过独立审计（15/15 Δ+CI 复算一致、行数/manifest sha 闭环）与独立评审（CONDITIONAL→补丁并入）。补丁要点：新增 §2b J/B05 制品形态对照（J=1 场景 prob 1.0、每节点步数 {0:1272,...,5:5223}、空步 14.2%；B05=3 场景×5 步）→ 跨 provider 结论降级为"制品级"；多重比较注（Bonferroni 下主效应仍显著）；单预测器种子边界（仅 J3:seed11）；`load_threshold=0.5` 与 E1 偏离披露；运行命令/环境/manifest sha 链补齐；evictions 注脚。记录：`docs/research/2026-09-11_fas_phase2_audit.md`、`..._review.md`；报告 `experiments/EXP-20260911_forecast_aware_scheduling/PHASE2_REPORT.md`。结论边界：未来结构预测显著改善调度（vs 无未来 −7.9%，vs 旧 provider −6.0% 制品级）；资源直接替换显著有害（+8.1%）且加载块增量≈0；oracle gap 14.7%。后续（§6）：形态匹配对照、seed22/33 稳健性、资源臂域匹配。

**Phase 2 网页版意见 (2026-09-11):** 核心结论成立但需降级——可称"未来拓扑预测提供可利用信息并显著改善前瞻调度"；不可称"资源预测有效"（应为"资源未在当前域偏移/接口下转化"）；E2−legacy 仅 provider/artifact 级。建议顺序：① 形态匹配（J 对齐 B05 规格：固定 H=5/top-k/同成本接口，只改预测内容）+ J3 seed11/22/33 → 归因是否成立；② oracle/J/静态表三者同接口诊断（判断问题在 scheduler/objective 还是 resource interface）；③ 2×2 oracle 交叉定位 14.7% gap。额外威胁：零步回退（14.2%）需 ablation；模板复用需确认 bootstrap 独立单元；artifact 与 workload 同源需 exploratory/sealed；需 deterministic vs stochastic J 对照。记录：`docs/research/2026-09-11_fas_phase2_web_opinion.md`。

**Forecast-aware Phase 3 (2026-09-11, 1000 episodes):** 形态匹配 + gap 分解完成，**修正 Phase 2 归因**。关键：E2 vs J-fixed5（去掉长度截断）**−14,932ms**（长度效应主导）；形态匹配下 J 内容 vs B05-top1 **+1,939ms**（不占优，先前"新 provider −6.0%"应归因长度约定差异）；拓扑 gap（真值拓扑+表 vs E2）−13,538ms；资源 gap（真值资源 @H=5）仅 −669ms 且 queue/miss 变差；**horizon 效应（无界 oracle vs trueopt_h5）−11,112ms 为最大剩余空间**；oracle 当前节点资源无益（+2,381ms）。零步预测验证为真信息（预测 0 → P(实际≤1)=0.999；预测>0 → P(实际=0)=0；实际终止 100% 被捕获）。报告：`experiments/EXP-20260911_forecast_aware_scheduling/PHASE3_REPORT.md`。后续：horizon 实验 / 长度感知策略开关 / 内容对比干净版 / 论文主线改为长度-终止贡献。

**Phase 3 后网页版头脑风暴 (2026-09-11):** 关键判断——**不能下结论"内容预测无价值"**：当前 E2 接口把内容压缩成"节点存在 + 长度"，q(A) 从未被消费；内容头处于"预测成功但无消费通道"状态。建议（不训练）：**C1** oracle-content vs pred-content（固定长度/表，只换属性真值）；**C2** q(A)→期望成本（length-only / +argmax / +expected / +oracle 四臂，用已有表）；消费机制排序：① survival/hazard 加权成本（E[C]=Σ P(T≥h)·c_h，替换 argmax 长度）② cache-aware 未来模拟（冷/热驻留）③ risk-sensitive（p90/CVaR）。H 扩展排序：① 政策侧 stochastic horizon（用 P(T>5) 生存外推，<1 天）② 重训 H=10（需新门禁）③ 自回归不推荐。论文故事建议："未来预测主要通过工作流是否继续/持续多久/风险在哪改善调度；资源替换失败因缺少 execution-state alignment"。记录：`docs/research/2026-09-11_fas_phase3_web_brainstorm.md`。待用户决策。

**Forecast-aware Phase 4（C1/C2+survival，2026-09-11, 1000 episodes）:** 消费通道实验完成，回答"内容无用是否只是通道缺失"——**不是**。C1（真值内容）反而显著更差（+14,222ms）；**去掉模型身份用 lane 均值反而显著更好（−12,334ms）**；C2（q(A) 期望成本）在形态匹配下仅 −2,347ms（远不如 lane 均值）；survival 权重 ≈ argmax 截断（−723ms）但显著优于固定 5 步（−15,655ms）。结论：**内容身份无正价值、身份信息有害、长度/终止结构是唯一主要价值来源、资源在 H=5 无价值**。报告 `experiments/EXP-20260911_forecast_aware_scheduling/PHASE4_REPORT.md`；产物 `outputs/c1c2_*_1000/`。剩余网页规划：Phase 2 cache-aware、Phase 3 H=10（重新立项）。

**Phase 4 后网页版重新规划 (2026-09-11):** 用户批评成立（C1 固定了消费机制）→ 两层结构：**第一层信息价值**（A0 E2 已有 / A1 Oracle H5+表 184,514 / A2 Oracle H5+真值资源 183,845【A1≈A2→资源非瓶颈】/**A3 Oracle H10/20 新增**）；**第二层消费机制**（固定 oracle topology+resource，比较 C0 greedy sum / C1 survival【已 +15.7s】【最高优先】/ C2 risk-sensitive / C3 cache-aware【cost(node|history)】）。优先：① 立即做 Oracle H10/H20（不训练、分钟级、当前最大确定收益 H5→无界 −11.1s）② consumer ablation（最大遗漏）③ 预测拓扑+新 consumer ④ 最后才考虑重训 predictor。论文主线："dominant scheduling signal is future workflow evolution rather than per-node resource estimation"。记录：`docs/research/2026-09-11_fas_phase4_web_replan.md`。Bridge 故障处理：标签页卡死→close-duplicates 清理重复标签→重发；等待上限按用户要求 300s。

**Forecast-aware Phase 5（horizon + consumer，2026-09-11, 1000 episodes）:** ① **H=10≈无界 oracle**：trueopt_h10−trueopt_h5 = −10,231ms（queue −23,864、miss −1.15pp），oracle−h10 仅 −881ms；H20=h10 饱和 → H5→H10 是最大可部署杠杆（需 H10 标签/重训）。② **风险敏感消费转换**：`predopt_h5_risk`（预测 p90 + 条件加载）vs E2（表 p50）= **−10,505ms**（miss −1.01pp）；aligned cache-aware（表 p50）+650ms 无增益；risk 仍不如 length-only（+1,829ms，身份成本有害）。结论更新：资源分位数在**风险消费**下确实转化；此前"资源不转化"只对 p50 替换与错误语义成立。报告 `experiments/EXP-20260911_forecast_aware_scheduling/PHASE5_REPORT.md`。下一步：H=10 预测器（重新立项）或消费方式细化（λ/分位数扫描、长度+lane+风险组合）。

**重训与消费方式规划 (2026-09-11, 网页版):** 结论=**先消费机制、后重训**。理由：horizon（H5→H10≈10s）是主要信息瓶颈；资源精度（oracle≈table）与内容精度（真值反而差）都不是；消费方式是最大未知变量（p90 已 +10.5s）。重训规格：H=10（L∈[0,10]、termination、每slot existence/runtime/load；不扩属性），**不需新数据**（P_dev 重生成 future_window=10），需新实验ID/checkpoint/门禁，不与 H5 混用。消费候选排序：① Quantile/CVaR（λ 扫描 4 档，<1h）② Chance-constrained（deadline 三档）③ Stochastic MPC/rollout（长期）④ hazard 升级 ⑤ cache-aware（暂缓）。防过拟合：70% dev / 30% frozen confirm，或预注册只跑一次。论文主线："future workflow evolution + uncertainty-aware reasoning"。执行顺序：Phase1 消费（1–3天）→ Phase2 H10 重训（≤1周）→ Phase3 cache-aware（后续）。记录：`docs/research/2026-09-11_fas_retrain_consumer_replan.md`。

**Forecast-aware Phase 6（消费+H10+缓存，2026-09-11, 1000 episodes）:** 三阶段完成。**Phase 1 消费方式重大进展**：q95（CVaR 代理）vs E2 = **−17,690ms [−19,238,−16,172]**（miss −1.39pp）；p90 −10,535；λ=0.5/0 与机会约束均更差 → **目前最佳 deployable 配置 = H5 预测器 + q95（180,362ms，−8.9%）**。**Phase 2 H10 重训 = 负迁移**（新实验 `EXP-20260911_p9d_j_h10_predictor`）：数据集/标签/训练全部新登记；H10 模型前 5 步退化（next-family 0.803→0.640、runtime pinball 1046→1691、load dur 888→2181），调度侧 H10q95 vs H5q95 +14,587ms、H10 模型 5 步消费 +33,459ms（虽 H10 内部 10 步优于 5 步）；需不同训练配方（后段降权/分离 head/更长训练/两段式）。**Phase 3 cache-aware 资源消费 = 负**（220,430ms，比 E2 差 +22,379），与网页版"需 residency 数据、后续再做"一致。报告：`experiments/EXP-20260911_forecast_aware_scheduling/PHASE6_REPORT.md`。loader 已修复为通用 `future_h*`（H10 必需）。

**Phase 7A（dev/confirm 分割 + 场景采样 CVaR，2026-09-14）:** validation 已切 700 dev / 300 frozen confirm（seed 20260914，manifest 在 data/manifests/）。新消费器（场景采样 E[F]+κ·CVaR，独立/共单调耦合）在 dev700 上**优于 E2 但均不如 q95**：q95 179,125（−17,229 vs E2）；scen_k50 183,116（−13,238）；comon_k50 185,880（−10,474）；scen_k0 210,116（比 E2 差 +13,761）。q95 − scen_k50 = −3,991 [−4,492, −3,473] 显著。**q95 在冻结 confirm300 上确认：−18,765 [−21,948, −15,932]，miss −1.36pp**。机制：独立采样使多步求和向均值集中、尾部稀释；q95 的'各步同时 p95'（完全相关）在本 workload 更有效；提高 κ 或共单调耦合都不能弥补。报告 experiments/EXP-20260911_forecast_aware_scheduling/PHASE7A_REPORT.md；新分布 artifacts outputs/sstar_predictor_artifacts_dist_sched/。

**Phase 7B/7C（自适应风险 + 分量拆分，2026-09-14，dev700）:** 7B：adapt（仅 slack）≈E2（+874，CI 含 0）；adapt_q（slack+队列压力）184,996（−11,358 vs E2）但仍输给 q95 5,871 [5,331, 6,478]。7C：**rt95（runtime p95 + load p50）= 179,136 ≈ q95（差 +11 [−120,+135]，统计等价）**；ld95（load p95）211,444（比 E2 差 +15,090）。**机制结论：q95 的全部收益来自 runtime 尾部惩罚（完全相关式逐步骤求和）；load 尾部无贡献且单独使用有害。**消费器家族（场景采样/CVaR、共单调、自适应、机会约束、生存、缓存、内容）均未超过 q95。报告 PHASE7BC_REPORT.md。
