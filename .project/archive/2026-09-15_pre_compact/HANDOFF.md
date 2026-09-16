# Handoff

## Last Updated

2026-09-11 (J series formal result completed: no Core GO; J2/J3 runtime significantly better; audits/reviews recorded; next: user decision on follow-up A-D)

## J series completed (2026-09-11)
- Formal run finished: backbone (3 seeds, 30 epochs) + J0/B1/J1/J2/J3 (3 seeds, 30 epochs) + single frozen test determination. Wall ~28.3 GPU-min on RTX 3060 Laptop.
- Verdict: **no Core GO** (all-seed/worst-seed reading): J2/J3 fail load-duration non-inferiority (5%); J2 1/3 seeds pass (seed22 margin 2.5%, disclosed, not usable as evidence); J3 0/3.
- Runtime primary (Delta vs B1): J2 -49.0/-65.4/-29.9, J3 -78.1/-77.1/-64.0 (all 3 seeds CI upper < 0); J1 +26.4/+19.9/+73.8, J0 +115.1/+119.4/+147.4 (significantly worse).
- Load Brier: J1/J2/J3 pass; J0 fails. Interface endpoints: no collapse; J0/B1 interface deltas exactly 0.
- Artifacts: `experiments/EXP-20260911_p9d_j_series_joint_resource/` (RESULT.md, run_manifest.json, summaries/, artifacts/checkpoints/ 15 hashes verified); raw runs `outputs/j_series_joint_resource/` (test_eval.json frozen, do not re-run test).
- Audits/reviews: `docs/research/2026-09-11_j_result_audit.md` (audit inconsistent -> closed), `docs/research/2026-09-11_j_result_review.md` (review modify -> applied). P1: unregistered >=2/3 aggregation + seed22 non-disclosure -> fixed in RESULT; >=2/3 recorded as post-hoc secondary only.
- Code: `scripts/build_j_dataset.py`, `scripts/j_series_common.py`, `scripts/j_series_train_eval.py` (new; legacy untouched); run.sh stages build|stage0|smoke|backbone|variants|test.
- Next options (user decision, not started): A extra seeds (44/55/66) for stability; B pre-registered load-duration hurdle revision + new gate; C J4 weighting (cos(g_R,g_B)~-0.1..-0.19); D stop joint-training claim and record negative result.

## J dataset v1 built and registered
- Builder `scripts/build_j_dataset.py` (conda env `scheduler`, torch 2.6.0+cu124): composite `(run_id, node_id)` join, unmatched fail-closed, holdout excluded by default. Output `results/processed/j_series_dataset_v1/`, registry `data/manifests/j_series_dataset_v1.json`.
- Counts: rows 13,754/2,029/1,520; supervision slots 48,343/7,195/5,387 = 60,925; unique resource nodes 14,413 (= 15,481 - 1,068 holdout); bounded_future_length histogram {0:2608,1:1350,2:1240,3:1164,4:1102,5:9839}; termination 6,870/994/745; load accounting zero 55,747 / positive 4,886 / missing 292; memory 37,553 / 23,372. Stage 0 assertions passed (key uniqueness, coverage, length/termination/load accounting consistency).
- Experiment registered: `experiments/EXP-20260911_p9d_j_series_joint_resource/` (config.json, run.sh); EXPERIMENT_GATE + INDEX updated (status in_progress).
- Next: implement `scripts/j_series_common.py` + `scripts/j_series_train_eval.py` (new entry, legacy scripts untouched), then Stage 0/smoke -> backbone (3 seeds) -> J0/B1 -> J1/J2/J3 -> frozen test evaluation. No scheduler integration; `S_*/T_final` untouched.

## J local environment ready (conda env `scheduler`)
- Per user instruction: conda env `scheduler` at `D:\anaconda\envs\scheduler` (Python 3.10.21); `torch 2.6.0+cu124` (CUDA 12.4 wheel) + `numpy 2.2.6` installed.
- Verified: `torch.cuda.is_available()=True`, device `NVIDIA GeForce RTX 3060 Laptop GPU`, CUDA matmul smoke passed. All formal J runs will use this env/device with a unified batch (no CPU/GPU mixing, no per-variant batch tuning).
- Next: build the J dataset pipeline (`build_j_dataset.py`: composite `(run_id, node_id)` join with uniqueness/fail-closed assertions, `bounded_future_length in {0..5}` + termination, holdout excluded by default), then `j_series_common.py` / `j_series_train_eval.py`, then Stage 0 -> smoke -> backbone -> J0/B1 -> J1/J2/J3. No scheduler integration; `S_*/T_final` untouched.

## P9d J implementation plan v2 ACCEPTED (ready to implement)
- Final confirmation round: all four P0s closed (composite `(run_id, node_id)` resource join with uniqueness/fail-closed assertions and unique-node vs supervision-pair counts; GT-derived `prefix_model_reuse` removed from J context; `bounded_future_length in {0..5}` + termination BCE with exact within-window censoring; early stopping removed, fixed 30 epochs then NI-feasible -> argmin RuntimeQScore). No new blockers; verdict: "J IMPLEMENTATION PLAN v2 ACCEPT".
- Protections confirmed: fixed validation bootstrap indices (B=1000, reused across epochs/variants/seeds), endpoint-paired masks, empty-replicate fail-closed with >=95% valid rule, unified formal device/batch, backbone before J0/B1, holdout excluded from the default pipeline, partial-completion labeling, legacy scripts untouched.
- Docs: `docs/p9d_j_series_design.md` (v3.1), `.scratch/chatgpt_j_impl_plan_v2_zh.md`, review `docs/research/2026-09-11_j_impl_plan_review_v2.md`.
- Next: start implementation — install torch locally (RTX 3060 6GB), build the J dataset pipeline (composite-key join), train backbone (3 seeds), then J0/B1/J1/J2/J3 and the frozen test evaluation. No scheduler integration; `S_*/T_final` untouched.

## P9d J implementation plan review (completed; 4 blockers fixed)
- v1 impl plan review verdict: MODIFY before accept. P0s: (1) resource join must use composite `(run_id, node_id)` keys with uniqueness assertions and separate counts for 15,481 unique nodes vs anchor×future-slot supervision instances; (2) `prefix_model_reuse` is future-label-derived — removed from J context (R0 RESULT annotated); (3) structure target fixed to `bounded_future_length in {0..5}` + termination BCE (censored L_H=5 is an exact within-window label); (4) early-stopping patience conflicts with NI-feasible selection — removed; fixed 30 epochs then NI-feasible → argmin RuntimeQScore.
- Adopted protections: new entry `j_series_train_eval.py` + small shared `j_series_common.py` (legacy scripts untouched); fixed validation bootstrap indices (B=1000) reused across epochs/variants/seeds; endpoint-paired masks; empty replicates invalid; ≥95% valid-replicate rule; unified formal device/batch; J0 runs after backbone; holdout not built by default; partial completion labeling.
- Design doc updated to v3.1: `docs/p9d_j_series_design.md`. Impl plan v2: `.scratch/chatgpt_j_impl_plan_v2_zh.md`. Review: `docs/research/2026-09-11_j_impl_plan_review.md`.
- Next: optional final confirmation round, else start implementation (install torch → J dataset pipeline → backbone → J0/B1 → J1/J2/J3). No code/data/experiment changes yet; `S_*/T_final` untouched.

## P9d J-series design frozen (v3; two review rounds)
- v1 review: 4 P0 (B1 redefine, ResourceQScore incompatible with hurdle load, joint loss undefined, checkpoint/NI/width semantics). v2 check: 3 more P0 (two inverted CI directions in load NI / NI manifest, and Core success must be determined on the frozen P9d/test rather than validation). v3 fixes all seven; reviewer confirmed no other architecture blockers.
- Frozen design: `docs/p9d_j_series_design.md`. Reviews: `docs/research/2026-09-11_j_series_design_review{,_v2}.md`. Key contracts: five variants (J0/B1/J1/J2/J3) + OracleAttr diagnostic sharing one backbone init and one attribute-head manifest; runtime single primary (`RuntimeQScore=mean(PB.50/.90/.95)`); load hurdle secondary with per-endpoint NI manifest (`CI_upper(Δ)<+δ` for lower-is-better, `CI_lower(Δ)>−δ` for higher-is-better); checkpoint NI-feasible then argmin RuntimeQScore; test-only success determination; width retired from training objectives; T=1; no GradNorm/PCGrad round 1; holdout out of all formal decisions.
- Next: implement locally — install torch (missing; local RTX 3060 6GB), build the J data pipeline (v3.1 labels + measured resource join), then run J0 probe → backbone → B1 → J1/J2/J3 with matched seeds 11/22/33 (<2 GPU-hour estimated). No scheduler integration; `S_*/T_final` untouched.

## P9d R0.1 workload proxy probe (completed; no gain)
- Question: can workload-size signal be recovered from fields already recorded in the existing P9d traces? Feature sets: F0 = R0 C4 (reference), F1 = +`tool_input_len`/`frame_count_before`/`frame_count` (7,078 tool nodes), F2 = F1 + task size (11,889 rows) + video metadata (14,939 rows). Protocol: train fit / validation selection / test diagnostic; the frozen holdout was deliberately not used.
- Result: no meaningful gain — validation PB 380.0→384.6→386.0, test PB 261.1→258.0→278.9 (differences within selection/noise, direction inconsistent). The proxies are used by the model (F2 gain: `tool_input_len` 5,161, `video_duration_s` 3,210, `question_chars` 2,538 vs `exec_class` 134,780 and `prefix_model_reuse` 8,417) but are redundant with existing features.
- Conclusion: workload size cannot be recovered from existing records; the missing information is model-side first-order data (true prompt/completion tokens, image/tensor sizes, batch/concurrency), requiring collector instrumentation and re-collection. Constraint: the 300 P_dev videos were deleted, so re-collection needs ~30 GB re-download or a new video pool. Alternative reading: with exec_class + context given, the residual variance is mostly noise/unobserved factors.
- Artifacts: `experiments/EXP-20260911_p9d_r0p_workload_proxy_probe/` (`RESULT.md`, `metrics.json`, `stage0_audit.json`, `joined_table.jsonl.gz`, `run_manifest.json`); EXPERIMENT_GATE and experiments/INDEX updated. No frozen dataset was modified; `S_*/T_final` untouched.
- Next: user decision — (A) J-series in the bounded scope, (B) instrumented re-collection project (pool/approval question), or (C) small instrumented pilot on a feasible video set.

## P9d R0 completed (design frozen v3; Core GO PASS)
- Execution: Stage 0 (6/6 assertions) + smoke (80 runs / 932 nodes / 9.0 s) + full run (15,481 nodes / 36.6 s, CPU, LightGBM 4.7.0). Artifacts under `experiments/EXP-20260911_p9d_r0_oracle_signature_ceiling/` (`RESULT.md`, `metrics.json` sha256 `832ae291…`, `stage0_audit.json`, node table, per-split predictions, `run_manifest.json`); gate record `.project/EXPERIMENT_GATE.json` (created with this experiment).
- Results: runtime validation improvement **74.9%** (bootstrap CI [0.704, 0.824]), test 82.1%, holdout 20.2%; calibration `max_τ` = 0.0338/0.0461/0.0161; load occurrence Brier val 0.0175 / holdout 0.0013 (fixed logistic 0.106 / 0.096), duration PB val 41.7 ms; memory calibration 0.127 (validation) → **descriptive-only**; ablation C0→C4 = 941.4→809.2→809.6→809.6→**380.0** (`workload_scale` zero-variance; `prefix_model_reuse` + baseline are the main context gains); holdout inclusive tail dominated by 2 stalls (45 min / 68 min) vs normal-subset PB 319.8 ms.
- Findings: the interface is empirically identifiable for runtime (+load) but lacks workload-size descriptors and cold-start/stall pre-execution context; the frozen holdout has been consumed once and can no longer be used for selection or contract changes.
- Next (proposed): J-series in the bounded scope (runtime primary; load secondary; no memory/tail identifiability claims), or a separate interface-extension round (real token length, frame count/resolution, cold/warm state, batch context). No training, no scheduler integration; `S_*/T_final` untouched.

## P9d R0 design frozen (v1→v3 review loop; final ACCEPT)
- Round history: v1 review → 3 P0s (load leakage / G3 coverage math / memory semantics). v2 check → original 3 P0s closed + new P0 (`load_ms=0` semantics) + 11 P1s. v3 check → load accounting error (raw scan included the 482 merged nested generates; 15,887 > 15,481) + missing occurrence-model pre-registration. Final round → load counts aligned to the v3.1 node ontology, hurdle contract and occurrence pre-registration closed, all P1 pre-registered, no new P0s; verdict **“R0 DESIGN v3 ACCEPT（可冻结）”** (only an implementation note: main LightGBM config selection uses the 3-seed mean validation `PB_primary`; worst seed reported only — folded into design §5).
- Frozen design: `docs/p9d_r0_oracle_signature_ceiling_design.md`. Reports: `docs/research/2026-09-11_p9d_r0_design_review{,_v2,_v3,_final}.md` (+ exports). Evidence: `.scratch/v3_load_accounting.{py,json}`, `.scratch/chatgpt_p9d_r0_design_brief_v3_zh.md`.
- Key frozen numbers: 15,481 resource-applicable nodes; load accounting planner 7,211 (67/5,960/1,184), tool 7,078 (0/6,742/336), post-loop generation 1,192 (9/530/653); nested 482 excluded; memory worker-request reset semantics; residency_state unusable (100% null) → `prefix_model_reuse` proxy.

## P9d v3.1 contract freeze + dataset rebuild (completed)
- Builder `scripts/build_p9d_topology_dataset.py` adds `--label-policy v2|v3` (v2 default unchanged). v3 = verified serial-control-flow chain: nested answer calls merged into their Summarizer node (`nested_calls` + composite `resource_signature`, `R_node = R_total`), failed planner + retry kept as chain nodes, terminal `answer` is a non-resource marker excluded from the H=5 slots, width pinned to 1 as a deterministic compatibility field, and all seriality/signal/nested/path/ontology gates fail-closed.
- Evidence: 12/12 local unittests (4 new v3 tests + LF newline regression); rebuild `results/processed/topology_predictor_p9d_v3` with 0 violations (1,360 runs, 16,841 chain nodes, 15,481 edges, 482 nested merges, 502/502 retry adjacency, 6,630/6,630 tool-step-id and planner-name matches, 1,360 terminal markers, width histogram `{1: 65,275}`); rows/splits/videos/runs identical to v2; features decompressed SHA-256 identical to v2 on all four splits; v2→v3 predecessor corrections = 7,117.
- Registry: `data/manifests/topology_predictor_p9d_v3.json` (file + report hashes; `dataset_root` = `/Volumes/Lenovo/scheduler/...`, same volume mounted as `F:` on Windows — the rebuild ran directly on the external disk). Contract: `docs/p9d_topology_label_contract_v3.md` (frozen). Review: `docs/research/2026-09-10_p9d_v3_contract_review.md`. Invalidated rebuilds kept for audit: `topology_predictor_p9d_v3_invalidated_20260910_v2diff_bug`, `..._crlf`.
- Next: design the R0 oracle-signature resource upper-bound experiment on v3 labels (not started). No training, no scheduler integration; `S_*/T_final` untouched.

## P9d v3 contract review (completed; revisions implemented in v3.1, see above)
- Draft `.scratch/chatgpt_p9d_v3_contract_brief_zh.md` sent to bound conversation in Chinese (attachment first failed to deliver; standalone `upload` resend succeeded, full answer received). Report: `docs/research/2026-09-10_p9d_v3_contract_review.md` (`status=valid`) + `..._export.md`.
- Verdict: MODIFY then accept. Direction endorsed (v2 fake-parallel restored to real serial chain). Required contract changes: (1) reframe edges as verified top-level serial control-flow (not file order), add independent Seriality gate with fail-closed on multi-tool/fork/new workflows; (2) merged Summarizer uses composite resource signature (no `effective_model_id` override, no runtime double counting); (3) final answer = terminal/non-resource event, post-loop generate is the compute node; (4) width=1 = deterministic compatibility field; drop width prediction as a task; reframe research as bounded future execution-trace forecasting.
- New P0s: freeze schedulable/resource node ontology before R0; restate holdout as contract-inspected (not untouched); keep scheduler-side v3 migration as its own later gate after predictor freeze.
- Prototype audit evidence: 1,360 runs all single-chain (8,289/8,048/504 nodes; no forks), 7,117 predecessor corrections vs v2 (5,575 tool / 1,177 post-loop gen / 365 retry), 6,630 planner→tool pairs with 0 name mismatches. R7 scheduler pool seriality check: 648 runs also single-chain (0 multi-tool steps, 0 non-prev-step parents, 0 step skips). No code/data/training/scheduler changes; `S_*/T_final` untouched. Evidence: `.scratch/p9d_v3_prototype.py`, `.scratch/p9d_v3_diffs.py`, `.scratch/p9d_v3_prototype.json`, `.scratch/p9d_v3_diffs.json`, `.scratch/r7_seriality_check.py`, `.scratch/r7_seriality_check.json`.

## P9d step-structure audit (completed, read-only; v3 pending decision)
- User challenged v2's step-level expansion rule; read-only audit over all 1,360 P9d-pool runs confirmed 3 flow categories (langgraph_react 596 / star 596 / st_fixed 168), step chain only (0 multi-parent), and trace-internal linkage: tool `standard_tool_call.id=compat-*-step-N` matches its step 100%; planner `parsed_decision.tool_name` matches same-step tool 100%; `retry_of` 502 events all same-step; st_fixed is a fixed `[ImageQA, Summarizer]` sequence.
- v2 labels make same-step planner+tool siblings: 18,028 such pairs across all 13,754 train rows (5,856 rows affected); tool predecessors are 91.6% previous-step tools, only 8.4% planners. R7 `job_templates_r7_v02` uses all-events-of-parent-step and misses the same edge. Conclusion: layer/width labels have systematic fake-parallel inflation; a category-aware v3 reconstruction is needed before trusting width/edge metrics. No code/data/training changes; `S_*/T_final` untouched. Evidence: `.scratch/p9d_step_structure_audit.{py,json}`, `.scratch/p9d_step_supp.{py,json}`.

## P9d v2 rebuild review (completed, conditional go)
- Bound conversation `6a9822da-e278-83e9-9c1a-675923acda0e` reopened via `switch-conversation` (new_background_page); `doctor 8/8`, `select-model` High `already_visible` (`high_ui_mapping`), brief `.scratch/chatgpt_p9d_v2_review_brief.md` sent with `--verified-high --confirm-send`; `wait` completed, `export-conversation` (7 messages) + `save-report` (`status=valid`, 49,270 chars) to `docs/research/2026-09-08_p9d_v2_rebuild_review.md`.
- Verdict (research input, not decision): C MODIFY/accept-as-policy, A ACCEPT, D ACCEPT-labels MODIFY-loss, E ACCEPT 4+, B ACCEPT-for-R0; P0-1 explicit `cycle count=0` required before final freeze, P0-2 fallback-edge semantics scoped to edge-sensitive claims only. R0 must be framed as oracle upper bound conditional on frozen v2 contract. No code/training/scheduler changes, `S_*/T_final` untouched.

## P9d v2 rebuild (completed, pending independent review)
- Builder changes + 3 new tests, 8/8 local unittest pass; v2 rows match v1 per split; intra-layer edges 2,307 -> 0; 16,971 invented edges dropped; workload_scale on 68,731 nodes; termination censored/terminated split recorded.
- Independent reviewer task could not run (subagent auth token expired); recorded as pending in DECISIONS.md. No model training, no scheduler integration, S_*/T_final untouched.
- Control-plane sync: local .project + builder + tests + v2 manifest + research docs verified byte-identical on external-disk primary copy. Remote sync pending (SSH ECONNREFUSED 2026-09-07); remote remains last-known-good backup.
- 2026-09-08 workspace move: working tree is now /Volumes/Lenovo/scheduler. Bidirectional reconcile done: 493 local-only files + 6 local-newer files copied to external; 8 external-newer files (7 scripts + v2 provenance manifest with 8/17 deletion records) backfilled to local; 7 same-byte large files skipped. 46 mp4 files (~3.37GB, incl. staging tmp) held pending user decision. No deletes anywhere.

## Status

`completed/verified`: 用户已确认把远端 `/root/autodl-tmp/scheduler` 完整同步到已有外接盘目录 `/Volumes/Lenovo/scheduler`；同名文件以远端为主，不使用 `--delete`，目标目录独有文件保留。远端总量约 `84G`（rsync source total=`89,707 MB`），外接盘同步后目录约 `111G`、卷剩余约 `782GiB`；`.project/` 控制面已同步。同步阶段 SHA-256 join 逐项检查远端 `45,983` 个 regular files，`MISMATCH_COUNT=0`；随后仅在本地和外接盘的 `.project/` 追加本次迁移与独立执行记录，因此这几份控制面文件相对远端原始版本的差异是有意的、可审计的。

`completed/verified`: 用户已授权推进并完成 P9f。范围限定为冻结 P9e checkpoint 的 train-only 概率/条件校准、P_dev/test 与 P_holdout_diag 分层误差审计，以及 `role/action_family + layer/width` 的粗粒度 future-cost/resource 可辨识性与聚合诊断；不重新训练、不覆盖 P9d/P9e、不使用 `S_train/S_val` 拟合或选择、不读取 `T_final`，不接入正式调度策略。结构比较 gate 通过，但 calibration、resource 和 model-aware gate 未通过，scheduler integration 保持关闭。执行环境为 `/root/autodl-tmp/scheduler`。

`completed/verified`: 用户授权在不覆盖 P9d 四字段实验的前提下完成 P9e reduced future-node role/family 消融。沿用固定 `P_dev/train→validation→test` 与 `P_holdout_diag/holdout`、seeds=`11/22/33`；Stage 0、真实数据 smoke、远端训练和输出契约审计均完成。执行环境为 `/root/autodl-tmp/scheduler`；不使用 `S_train/S_val` 训练标签，不读取 `T_final`，不接入调度器。原始 trace、旧 R7 artifact 和既有 P9d 结果保持只读。

## Current Operation

- 2026-09-04 remote-to-external sync（已完成并验证）：远端 `root@connect.westc.seetacloud.com:12469:/root/autodl-tmp/scheduler` 与外接盘 `/Volumes/Lenovo/scheduler` 完成内容级增量同步。SHA-256 清单筛出并传输了 `501` 个文件、约 `58,382 kB`；没有使用 `--delete`，没有删除目标独有文件。同步阶段外接盘与远端全源 regular-file SHA-256 验收为 `45,983 checked / 0 mismatch`；随后更新后的 `.project/PROJECT/STATE/PLAN/DECISIONS/HANDOFF` 已逐项与本地控制面一致。中途一次强制全量预传在根目录小文件阶段安全中断，最终增量同步从完整清单重新完成，不影响最终结果。

- 2026-09-03 P9f preflight（已完成）：远端 P9f/P9d 契约测试均为 `4/4`，Python 编译检查通过；P9d Stage 0 通过（18,683 rows、视频/trace split 交叉 0、因果输入违规 0、raw_action 标识泄漏 0、终止/截断标记完整）；真实 8-row P9d batch 已成功加载 B/seed22 checkpoint，CUDA forward 输出 shapes 为 `layer=(8,6)`、`width=(8,5,5)`、`content=[(8,5,5,5),(8,5,5,8)]`。
- 2026-09-03 P9f long operation（已完成）：字段命名修正后的脚本/测试已同步远端；远端 `4/4` 回归测试和编译通过。执行 `cd /root/autodl-tmp/scheduler && sh experiments/EXP-20260903_p9f_predictor_acceptance_audit/run.sh` 正常退出，重新生成 P9f 四个输出；远端运行结果为 `completed_diagnostic`。本轮只加载冻结 P9e B checkpoint，不创建 checkpoint，不修改旧实验目录或读取 `S_*`/`T_final`。
- 2026-09-03 P9f final verification（已完成）：本地 JSON 有限值检查、manifest 输出哈希检查和节点数量 `_count` 字段审计通过；`stage0_audit.json`、`cost_profiles.json`、`metrics.json`、`run_manifest.json` 四个输出的本地/远端 SHA-256 均匹配。P9f 结构 gate 为 true，calibration/resource/model-aware/scheduler gate 关闭；正式结果见 `experiments/EXP-20260903_p9f_predictor_acceptance_audit/RESULT.md`。
- 2026-09-03 P9e execution（已完成）：远端 Stage 0 通过（18,683 行、视频/trace split 交叉 0、输入违规 0、raw_action 标识泄漏 0、终止/截断标记完整）；固定 P9d split、seeds 11/22/33 的 `N0/A/B/D` 共 12 个 run 全部完成，根 `metrics.json`/`run_manifest.json` 已生成。B 按 validation `layer_count_mae + full-H width_vector_mae` 选中；holdout 不参与改选。
- 2026-09-03 P9e recovery/verification（已完成）：首次进程在 A/seed11 的评估处因变量复用 bug 中止；修复后通过本地/远端测试并使用 `--resume` 安全续跑，首轮中止不计入结果。实际数据 B/D forward/loss/backward smoke、远端 P9e `4/4` 测试、本地相关测试 `4`（`3` passed、`1` skipped）和实验清单中 62 个非 checkpoint 输出的逐文件 SHA-256 均通过；独立预测契约审计为 0 errors。12 个 checkpoint 保留在远端，不误称为本地已恢复。
- 2026-09-03 P9e reduced role/family experiment（已完成）：新增独立脚本 `scripts/p9d_future_role_family_multitask.py`、契约测试、配置和实验目录；只把未来节点字段从 `node_type/raw_action/model_id/action_family` 改为 `role/action_family`，保留结构头和 `next_role/next_family_if_execute` 辅助头。旧 P9d 四字段脚本/实验不改。远端与本地 root metrics、run manifest、source hash 和预测结果已核对；manifest 中继承的旧文件名键已修正，不改变训练结果或 artifact hashes。
- 2026-09-03 final sync/verification（已完成）：P9e 脚本、配置、run manifest、metrics、RESULT/meta、`experiments/INDEX.md` 和四份 Project OS canonical 文档的本地/远端 SHA-256 均匹配；旧四字段实验未被覆盖。远端训练进程已退出，12/12 run 与根级状态均为完成。

- 2026-09-03 ChatGPT Web follow-up（已完成）：向既有会话 `https://chatgpt.com/c/6a9822da-e278-83e9-9c1a-675923acda0e` 发送脱敏 brief，讨论首轮实验变体、三头损失/输出契约、oracle/predicted topology 诊断、数据切分、非劣性标准、失败模式和停止条件。brief 仅含 shared causal GRU、structure/content decoder、next-step auxiliary head、`P_dev/P_holdout_diag` 边界等抽象信息，不含本地路径、凭据、原始 trace 或私有日志；回答已完整返回，研究记录为 `docs/research/2026-09-03_next_step_auxiliary_experiment_plan.md`。网页建议仍作为研究输入，P9e 按用户本轮确认独立执行。

- 2026-09-06 ChatGPT Web topology-resource follow-up（已完成）：同一绑定会话发送脱敏 brief，讨论未来 DAG 拓扑→资源接口可辨识性（结构可用但 role/family 无法辨识 load/memory、校准恶化、runtime 低估长尾）。发送前核验 GPT-5.6 Sol + High（high_ui_mapping，无限制提示），回答完整返回并通读核验；研究记录为 `docs/research/2026-09-06_topology_resource_interface.md`（`status=valid`）。核心建议（研究输入，非决策）：引入资源等价签名 `z=(exec_class, model_resource_class, input_scale, mode)` + 运行时上下文的两段式接口，先做 oracle-signature 可辨识性上界实验；引用的 Vulcan/Vidur/vLLM/DistServe/ServerlessLLM 等需本地独立核验。另按更新后的 skill 新建 `~/.venvs/chatgpt-bridge`（Python 3.13.5 + `mcp==1.12.2`）并复用；未改代码、未训练、未接 scheduler，`S_train/S_val/T_final` 继续封存。

- 2026-09-03 remote control-plane sync（已完成）：`STATE.md`、`HANDOFF.md` 和 `docs/research/2026-09-03_next_step_auxiliary_experiment_plan.md` 已通过 SSH 临时文件分块传输、逐文件 SHA-256 校验后替换远端目标；三份本地/远端哈希分别为 `6b03703a4b2335e40d692f657a2aea19ce5bfb9892c23782af21755d0c8c42fb`、`a546700cf00f192fd958afc7845ccba16fc8a069ddc18c55e28d8e11bcb454b3`、`375a7b6c50aaef101a223d421cd9e6f4b8662e85a9dacd2f6d79f31c622edca6`，均匹配。默认/旧版 SCP 入口仍返回 connection closed，未将其误记为成功。

- 2026-09-02 ChatGPT Web new-session research：新建空白会话入口，已核验可读页面、URL `https://chatgpt.com/`、title `ChatGPT` 和可见“高”推理控件；用户确认后提交了只包含抽象模型结构、数据边界和评审问题的 brief，没有上传本地路径、凭据、原始 trace、私有日志或未公开标识。会话最终 URL 为 `https://chatgpt.com/c/6a9822da-e278-83e9-9c1a-675923acda0e`，title 为“架构设计评估”，回答已完整返回并完成回答后页面核验。研究记录为 `docs/research/2026-09-02_future_dag_structure_content_decoder.md`，含 VERIFIED/INFERENCE/UNVERIFIED 标签；结论不自动改代码或提升正式策略。

- Formal experiments completed: `EXP-20260902_p9d_topology_empirical_baseline` and `EXP-20260902_p9d_topology_tabular`.
- Fit boundary: `P_dev/train` only; empirical candidate schemas and tabular LightGBM configurations were fit independently from the same train split.
- Selection/evaluation boundary: `P_dev/validation` selected empirical schema/model configuration; `P_dev/test` was diagnostic; `P_holdout_diag/holdout` was evaluated only after each selection froze.
- Target/output: H=5 identity-free DAG-layer signatures with top-3 scenarios; no node IDs, edges, execution truth or resource truth in model keys or predictions.
- Remote dataset: `/root/autodl-tmp/scheduler/results/processed/topology_predictor_p9d_v1`; expected manifest SHA-256 `c921f6ee4dbdee1b01af102ad6d02b29a843f91392357ce7fad78cf24df84630`.
- Local source checks passed: `py_compile`, JSON validation, `sh -n`, empirical baseline tests 5/5 and tabular comparator tests 4/4.
- Completed remote action: empirical baseline and tabular comparator each used a unique remote experiment directory; source/data/output SHA-256 were checked, remote unit tests passed, and the small metrics/prediction outputs were recovered locally. Shared GRU remote training completed in its unique directory; 16 MB of artifacts were recovered locally and independently audited.
- Final verification: shared GRU `run_manifest.json` and `RESULT.md`, plus the updated `.project` canonical documents, are local canonical records; root metrics/run-manifest and all source/config hashes match the remote copy. Local `py_compile`, JSON and shell checks passed; remote PyTorch contract tests passed 4/4; 36 prediction files/44,361 rows had 0 normalization, shape, identity/edge/resource-truth leakage or sample-coverage violations.
- Explicitly out of scope: scheduler integration, `S_train/S_val` labels, `T_final`, raw data/trace changes and R7 policy changes.

## Goal

验证“预测模型 + 有限滚动优化”能否兑现 `TrueOpt-H5` 的未来信息收益：用可部署的预测状态做滚动决策，用 full-event 只做小规模验收，不把 full-event 标签直接当作专家策略。

## Latest Recovery Point

- 用户确认 topology predictor 必须归入 predictor block。正式训练/验证改用行为与资源预测器的 `P_dev=300` / `P_holdout_diag=40` 视频边界；`S_train/S_val/T_final` 只在模型冻结后用于 scheduler 集成评估。
- 新计划已写入 `.project/PLAN.md` 的 `R8-P9d`，持久决策已追加到 `.project/DECISIONS.md`，状态说明已写入 `.project/STATE.md`。现有 `EXP-20260901_topology_predictor_baseline` 保留为 scheduler-side diagnostic，不作为正式 learned topology predictor。
- 阶段 0 标签门禁结果：远端主机 `autodl-container-41d846924e-d62d0601`、执行根目录 `/root/autodl-tmp/scheduler` 已核验。核心模板与扩展 snapshot 的 P_dev 视频并集为 300 个：核心 64 个覆盖 train/validation/test=`52/7/5`，扩展 236 个覆盖=`188/23/25`，与冻结 registry 的 `240/30/30` 一致；P_holdout_diag 未混入这两个集合。holdout 原始 trace 覆盖 40 个视频、120 个 run。
- 阶段 0 结构结果：扩展 collection 的 472 个 trace 和 holdout 的 120 个 trace 全部 JSON 有效；扩展 `parent_refs=5665`、holdout `parent_refs=1057`，均 `missing_parent_refs=0`、`forward_parent_refs=0`、重复 event id 为 0。每条 trace 都有同一步多事件（最大 fanout=4），按 parent step 映射到事件级边后每条仅保留一个 root 无前驱。现有核心模板 640 条/64 视频可读，但其旧生成结果是链式 predecessor；正式 topology label 应统一从原始 trace 保留 parent edges 重建，不能直接把旧 R7 模板当作 P_dev 全量标签。
- 阶段 0 锚点结果：远端 P_dev 行为样本的 role/tool run_id 均能在原始 trace 集合找到（role 17,303 行/1,240 runs；tool 5,889 行/1,113 runs；缺失 0）；样本已有 current/target source event id，可按 `video_id + run_id + prefix/current_node` 对齐。P_holdout 的 120 条 trace 及行为数据位于同一 `final_holdout_v1` collection。
- 完成结果（2026-09-02）：远端输出 `/root/autodl-tmp/scheduler/results/processed/topology_predictor_p9d_v1` 约 2.3 MB。train/validation/test/holdout anchors=`13,754/2,029/1,520/1,380`，videos=`240/30/30/40`，runs=`988/144/108/120`；每个 split features/labels sample_id 完全一致、无重复、无 unmatched anchor。重建 `18,683` raw events、`17,323` graph nodes、`32,934` edges，missing/forward parent refs=`0/0`；非显式 parent 的链式补边 `1,227`。独立 causal audit 的 forbidden model-input key violation=`0`，与 S_train/S_val/T_final/T_backup video intersection=`0`，layer offset/empty-layer error=`0`。远端数据与 manifest SHA-256 已记录在 `data/manifests/topology_predictor_p9d_v1.json`；原始 trace、旧 R7 artifact、T_final 未覆盖/解封。
- 经验基线完成结果（2026-09-02）：正式实验 `EXP-20260902_p9d_topology_empirical_baseline` 选择 `current_full`。validation 的精确拓扑 top-1/top-3 覆盖约束在低支持度下仍有限；test/holdout 的 layer/node count bias 均为负，holdout 条件键全部走全局回退。详见实验 `metrics.json`，不把该结果冒充 learned predictor。
- tabular 完成结果（2026-09-02）：正式实验 `EXP-20260902_p9d_topology_tabular` 选择 `lgbm_small`。结构层数/宽度/节点总量相对 empirical baseline 明显改善，但原型字段在 holdout 仍有分布偏移，完整 topology signature 覆盖不足；详见实验 `metrics.json`。
- Shared GRU recovery: `EXP-20260902_p9d_shared_causal_gru` 已按 `P_dev/train → P_dev/validation → P_dev/test → P_holdout_diag/holdout` 完成 4 个预注册变体×3 seeds；validation topology primary metric 选择 `topology_only`，holdout 上 `shared_multitask` 的结构-行为平衡更好，但不事后改选。
- P9e final result: 选择 `B`；validation structure score=`0.12847`，holdout structure score/exact=`0.27986/0.82923`，未来节点 role/action-family F1=`0.90151/0.40360`。两字段总 F1 不与旧四字段总 F1直接比较；共同 action-family 未提升。预测输出尚未接 scheduler；不使用 `S_*` labels，不猜测 `execution_lane=unknown`，不读取 `T_final`。
- P9f final result: frozen holdout learned B mean layer/width/node MAE=`0.1442/0.1357/0.3807`、exact=`0.8292`，三项结构误差均低于 empirical baseline；train-fit temperature 的四头 holdout NLL 均变差，保留 raw probabilities；role/family runtime proxy coverage=`0.8449/0.8435/0.9152`，top-1 runtime MAE 均值=`118,798.7 ms`且系统性低估。当前输出契约不含 `model_id/node_type/input_scale/cold_warm`，load/memory 不可辨识。
- Current recovery point: P9f 已完成并恢复至本地独立目录 `experiments/EXP-20260903_p9f_predictor_acceptance_audit/`；四个输出和远端哈希已核对一致。P9d/P9e/P9f 及 PredMPC 等最新实验目录现已同步到外接盘 `/Volumes/Lenovo/scheduler/experiments/`，12 个 P9e checkpoint 也已在外接盘可用；远端原副本保持不变。
- Next action: 先设计并确认 resource-aware future-content 或 conditional aggregation 实验，解决长尾 runtime 和 model-aware load/memory 的可辨识性；在该 resource contract 与 predictor acceptance 通过前，不推理到 `S_train/S_val`，不接 scheduler，不读取 `T_final`。
- 2026-09-02 模型候选只读核对：300-video 固定 validation 上 B05/B06/B07/B10 joint accuracy 分别为 `0.8802±0.0019`、`0.8798±0.0005`、`0.8782±0.0007`、`0.8768±0.0011`，B01 XGBoost 为 `0.8501`；40-video final holdout 上 B05 为 `0.8046±0.0048`，B01 为 `0.7406`。因此 B05 是当前已验证的总体行为冠军，B01 是当前已验证的纯传统 ML 冠军。旧 64-video 多步 GNN 未稳定超过当时 v0.4，B10 XGB+GRU residual 也未超过 B05，故二者只保留为后续条件式消融，不作为首轮主模型。
- 2026-09-02 联合模型规划（尚未授权执行）：以统一的 pre-execution anchor 构造每个样本，只允许已完成 prefix 与当前节点静态身份进入输入；共享一个 B05 风格单向 GRU 表示，连接 next-role、execute-gated family 和 H=5 identity-free topology decoder。首轮比较 aligned behavior-only、topology-only、shared multi-task、shared+first-layer consistency；不加 residual/GNN/Transformer。训练沿用 `P_dev` 视频边界与 seeds 11/22/33，`P_holdout_diag` 仅在模型和 epoch 冻结后使用；预测 gate 通过后才推理到 `S_*` 做 scheduler 诊断，`T_final` 继续封存。
- 接口口径：模型输入是当前可见 prefix/current-node 状态、行为/资源预测器的冻结当前预测（正式版）；未来 DAG 边和 `predecessor_node_ids` 只用于训练标签/审计。模型输出是 H=5 的 top-3 概率场景，每个场景为可含多节点的 identity-free DAG layers，节点原型包含 `node_type/raw_action/model_id/execution_lane/action_family`，不输出真实 successor ID、边或执行资源真值。
- 宽度语义已在 R7 True DAG 诊断范围内核验：按现有审计相同的 `predecessor_node_ids` BFS/H=5 口径，1,546 个 unique candidate nodes 对应 4,048 个非空真实后继层，其中 2,831 层宽度大于 1（69.94%）；宽度分布为 1:1217、2:1510、3:1170、4:146、5:5，最大宽度 5。1,386 个非终止候选节点中 997 个至少包含一个多节点层（71.93%）。这里的 width 是同一因果深度的真实后继节点数，不是物理时刻并发数；该证据仅来自 R7 scheduler diagnostic 数据，尚不能外推到正式 predictor `P_dev/P_holdout_diag`。
- 解释口径：最小分枝例为当前节点 A 之后，B/C 都只依赖 A，而汇总节点 D 同时依赖 B/C；因此未来第 1 层是 {B,C}、宽度 2，第 2 层是 {D}、宽度 1。即使单 GPU 按 B→C→D 顺序执行，DAG 层宽度仍为 2；旧 Pred 若只展开一个 next event，会漏掉同层另一个未来工作节点。
- 用户理解确认：topology/width 是 behavior 的结构补充，使 Pred 在同一 H 层范围内预测更完整的未来节点集合及依赖关系，并为逐节点资源预测和 future-cost 聚合提供结构；它不在推理时读取真实未来，也不保证自动更准。若作为共享编码器的额外 head，可视为辅助任务；若其输出直接进入调度评分，则属于核心中间预测量。
- 宽度的动机回溯：最初目标是解释同一时刻 True/Pred 的差异；分解已显示候选过滤/current score 不是主因，Pred 的 `future_h5` 是固定 5 个 synthetic event steps，而 True 使用 5 个后继 DAG layers。True 平均后继节点数为 9.1858，77.8287% 的 candidate occurrence 中 True 后继数大于 Pred event 数，且 92.45% 的 candidate future MAE 由 future 项主导。因此引入 width 是为了描述一层中可能包含的多个未来工作节点、修复未来工作量覆盖不足的假设，不是因为一个物理时间点必然有多个活动节点；若真实 True 语义是线性事件，则应改为预测未来节点数/终止长度，不保留 width。
- 重叠边界：topology 与 behavior 共享当前 prefix/current node 的部分上下文；behavior 预测下一步 role/tool，topology 预测未来多层 DAG 的层数、宽度和节点原型。正式 topology 可把 behavior 的冻结概率作为可选上游特征，但必须保留无 behavior-feature 消融，并在 P_dev 使用 OOF/交叉拟合特征，避免训练内预测造成泄漏。
- 概念确认：用户将 topology 理解为在 behavior 预测基础上加入 DAG 展开；当前解释确认该理解基本正确，但实现上应是共享当前输入/可选行为输出、另设 topology structure head，而不是把行为预测器的单步输出直接复制成多步结果。
- 架构确认：behavior 与 topology 在系统层面互补并共同服务 scheduler；推荐模型形态为共享当前 prefix 编码器、分开的 behavior/topology heads，topology 可读取 behavior 概率作为上游特征，但保留 behavior-only/topology-only/combined 消融，不把两个不同标签任务压成一个不可审计的输出。
- 当前最小结合方案（建议、尚未授权实现）：保持现有 B05 role/family 两个冻结 GRU 不变，导出 soft `p(next_role)` 与 execute-gated `p(family)`；topology 模型读取同一可见 prefix/context 加这些概率，输出 top-3 identity-free DAG-layer scenarios；随后由冻结资源预测器对场景节点估价并聚合 future cost。P_dev 训练侧必须使用 OOF/交叉拟合行为概率，并保留 topology-only 与 topology+behavior-probability 消融。只有该软连接在未见数据上证明增益后，才考虑新建共享 encoder、分离 behavior/topology heads 的联合多任务模型。
- 对称帮助的候选最终架构（建议、尚未授权实现）：同一 causal prefix encoder 同时连接 behavior head 与 topology head；behavior loss 和 topology loss 都反向更新共享表示，topology 第一层的 role/family 边际分布与 behavior 输出增加可校准的一致性约束。推理时两个 head 从同一当前可见输入并行预测，不把任一 head 的 hard top-1 作为另一 head 的必需输入；是否用 topology 首层边际修正最终 behavior 概率，必须由未见数据消融决定，避免循环误差。
- Residual interaction 资料调研状态：仅形成待核验问题，尚无有效外部研究报告。需要比较独立 heads、首层一致性 loss、单次 gated cross-task feature residual 与更复杂迭代交互；在来源核验前不把 residual connection 提升为架构决策。
- Residual interaction 原始资料结论：Cross-Stitch Networks（CVPR 2016）和 NDDR-CNN（CVPR 2019）支持可学习的跨任务隐藏特征混合；PAD-Net（CVPR 2018）与 MTI-Net（ECCV 2020）支持利用初始任务预测做注意力/残差精炼；Taskology（CVPR 2021）支持显式任务一致性损失；GradNorm（ICML 2018）、Multi-Objective MTL（NeurIPS 2018）和 PCGrad（NeurIPS 2020）说明联合任务可能产生损失失衡或梯度冲突。对当前 300-video 边界的最小候选顺序为 shared encoder+separate heads → 首层/终止一致性 loss → 单次零门控初始化的双向 feature residual；只有观察到梯度冲突或 loss 失衡时再加 PCGrad/GradNorm。
- ChatGPT Web delegation：2026-09-02 仅在 Codex In-app Browser 尝试打开 `https://chatgpt.com/`；两次页面加载超时，浏览器连接未呈现可验证页面。失败分类为 `page_load_failure`；brief 未提交，下一步需用户恢复网页版浏览器可访问性后再重试。
- ChatGPT Web retry：用户表示页面可用后重新接管现有标签；URL/title 可读，但 DOM、截图和 visible-DOM 均超时。失败分类仍为 `page_load_failure`；brief 未提交，需页面状态可读后再验证 Sol High。
- Brief handoff：用户选择不再由 Codex 自动发送，改为手动复制 brief 到网页版 GPT；brief 只包含项目架构摘要、数据边界和待评审问题，不含本地路径、凭据、原始数据或私有日志。

- 本轮仅新增本地 post-hoc 诊断产物：
  `experiments/EXP-20260901_topology_predictor_baseline/artifacts/old_pred_true_decomposition/`
  和 `.../pred_true_layer_decomposition/`。
- 两次 decomposition 使用同一份 action audit（SHA-256
  `265d753474874ae3a86afc239b8a90f8ed2ac21d2ef4c19815acca42025967c9`），各覆盖 15,364
  decisions/44,690 candidates；过滤、priority/tie-break、scored set 和加法契约均通过。
- 已验证：current 预测逐 candidate 不变，差异只在 future H5；新 layer 的 tie-aware top-1
  为 71.0817%，旧版为 66.0570%，但未来拓扑规模仍低于 True DAG，不能据此提升正式策略。
- 下一恢复动作：评审两个 `metrics.json` 与 candidate/decision 明细；若继续，只能先确认
  future-cost calibration 或 learned train-only topology predictor 的独立方案。保持远端、
  `T_final`、WAIT/RESERVE、抢占和多 GPU 封存。

## Verified Evidence

- `EXP-20260825_action_value_audit`：15,364 个 common dispatch states；PredOpt-H5 local-Q top-1=63.37%，Myopic=62.89%；PredOpt-H5 mean local-Q regret=18,404.7 ms，略高于 Myopic 的18,249.4 ms。
- full-event follow-up：100 个 common states、717 个 strict-feasible branches、0 failures、100/100 reference consistency；full-event top-1 为 Myopic/PredOpt-H5/PredOpt-v2-H5/local TrueOpt-H5 = 32%/32%/33%/40%。
- 当前结论：`Q_local_H5` 不是 full-event value proxy；WAIT/RESERVE 暂缓，Prefetch、PPO、preemption、多 GPU 契约和 `T_final` 继续封存。
- canonical evidence：`experiments/EXP-20260825_action_value_audit/RESULT.md`、`metrics.json`、`rollout_audit/metrics.json`；输入来自 `.scratch/action_value_audit_remote_inputs_20260825/`，哈希已记录。
- `EXP-20260826_pred_mpc_rolling_horizon` final visible-window smoke：6 policies × 10 episodes = 60 rows；每个 policy 160/160 jobs completed，0 failed jobs；remote output `/root/autodl-tmp/scheduler/results/processed/EXP-20260826_pred_mpc_rolling_horizon/smoke_10_visible_window/`，local copy `experiments/EXP-20260826_pred_mpc_rolling_horizon/artifacts/smoke_10_visible_window/`。
- Final smoke mean completion: Myopic `176,937.1 ms`; PredOpt-H5 `175,860.0 ms`; CP-RHO-H5 `183,881.3 ms`; PredMPC-H3 `186,010.2 ms`; PredMPC-H5 `187,089.5 ms`; PredMPC-H5-no-terminal `185,218.1 ms`。PredMPC-H5 is `+10,152.4 ms`/`+5.74%` vs Myopic and `+11,229.5 ms` vs PredOpt-H5。
- First three smoke outputs remain under `artifacts/smoke_10/`, `artifacts/smoke_10_corrected/`, and `artifacts/smoke_10_visible_artifacts/` as invalidated diagnostics; final reported numbers come only from `smoke_10_visible_window` after removing template-suffix access, including the visible completion check.

## Literature Finding

网页版研究页面两次语义读取超时，未提交研究 brief；本轮改用公开原始论文/官方页面核验。主流“好动作”来源分为：Decima 的 RL rollout reward、Tiresias 的 Gittins/LAS priority、Optimus/Gavel/Pollux 的 performance model + allocation optimization，以及 Themis 的 bid/auction。结论是当前 `force action + Myopic continuation` 适合作为 diagnostic counterfactual，不应直接当作全局专家标签。

## Replanned Experiment

拟创建唯一目录：`experiments/EXP-20260826_pred_mpc_rolling_horizon/`。这是对原 P8c 的路线调整：P8c 的 full-event audit 保留为验收子实验，不再作为主算法。

### Main question

在不使用 execution truth、不读取未来真实到达、不增加 WAIT/RESERVE/抢占/多 GPU 语义的条件下，预测信息能否通过滚动优化转化为更好的当前动作？

### Policy contract

1. 候选仍是当前严格可行的 `(ready_node, free_gpu)`，默认单节点单 GPU，不改变动作空间。
2. 对每个候选动作复制一个 scheduler-visible predicted state，用已有 runtime/load/cache/future-artifact 预测推进有限事件 rollout；不调用 `ExecutionTruthProvider`。
3. 以 Decima 风格的全局 episode 目标作为评价契约：`J_episode = mean_j(C_j - A_j) = (1/N) * integral active_jobs(t) dt`，越低越好；failed jobs 和容量违规是硬门。
4. 滚动评分采用 `predicted active-job area + predicted terminal continuation`。terminal continuation 在预测模型中继续运行已知工作，不用真实 completion；它用于减少固定短 horizon 的截断偏差。
5. 每次只执行第一步并重新观测，命名为 `PredMPC-3`/`PredMPC-5`（分别使用 3/5 个预测 dispatch events，并匹配 H3/H5 future artifacts）。当前 `CP-RHO` 保持不变，作为旧实现对照；不把现有 `CP-RHO` 的 scalar future-cost objective 误称为真正 MPC。

### Comparison matrix

- Baselines: `Myopic`、`PredOpt-H5`、现有 `CP-RHO-H5`。
- New policy: `PredMPC-3`、`PredMPC-5`。
- Ablation: `PredMPC-5-no-terminal`，用于验证收益是否来自 terminal continuation。
- Privileged audit only: `local TrueOpt-H5`/小样本 full-event Q；不用于训练或调参。
- 已有 PPO/R8-P4 结果作为 RL reference，不在本轮重跑 PPO；Tiresias/Gittins、Themis auction 暂不直接移植，因为当前 trace 是非抢占 DAG node、单 GPU slot，直接移植会同时改变问题语义。

### Execution gates

1. `S_train` 小样本只用于冻结 terminal continuation 的实现参数、dispatch depth 和 solver budget；先跑 10–30 episode smoke，随后最多 300 集用于方案选择。
2. smoke 必须通过：所有 policy 正常结束、0 failed jobs、0 capacity violations、forced-reference consistency、预测策略无 truth/future leakage、episode area 与 JCT 恒等式核对。
3. 方案冻结后在未参与选择的固定 `S_val` 上跑 1,000 集 paired comparison；按 action width、cache hit/miss、deadline slack、model/role 分层。
4. 记录 mean/p95 JCT、mean queue、deadline miss、makespan、GPU utilization、planning latency、fallback rate、terminal depth 和 action agreement；不使用未经校准的任意加权总分。
5. 最终候选和配置冻结后，才申请解封 `T_final`；本轮任何 smoke/main 都不读取 `T_final`。

## Current Experiment (2026-08-31)

`experiments/EXP-20260831_aligned_h5_score_contract/` 已完成并标记为
`passed_diagnostic`。正式本地命令为该目录的 `run.sh`；本轮已按相同参数完成三阶段：

1. `action_value_audit/`：100 episodes、15,364 decisions、aligned truth contract、0 failed jobs、无缺失 artifact。
2. `rollout_audit/`：100 sampled states、774 candidate branches、0 failures、reference consistency=1.0。
3. `smoke_10/`：5 policies × 10 episodes、50 rows、每策略 160/160 jobs completed、0 failed jobs；最大模拟峰值显存 32,756MB/32,760MB。

结论：统一评分契约修复了 Predicted/Truth 的语义不一致，但没有实质缩小 full-event gap；
`aligned_predopt_h5` 不提升为正式候选。精确数值见实验根目录 `metrics.json`，解释见
`RESULT.md`。预检目录 `artifacts/preflight_1/` 仅作执行前证据保留。

## Boundary

- Root: `/Users/liuqiuwei/Documents/调度`；canonical execution copy: `/root/autodl-tmp/scheduler`。
- Allowed in this operation: 新建唯一正式实验目录、增加 opt-in layer-H5 生成/评分脚本和测试、使用本地冻结输入做契约/回归验证、更新 canonical docs；远端连接与同步暂不执行。
- Out of scope: 修改 R7/T_final 输入、解封 T_final、重采 trace、下载视频、加入 WAIT/RESERVE、修改多 GPU/抢占语义、重做 PPO。

## Current Operation

- 2026-09-02 只读契约与结构核对：当前 B05 行为预测器的目标只有 `next_role` 和 execute-gated `family_label`，没有未来层数、DAG 深度、并行宽度或终止步数预测头。当前 topology predictor 也不是神经网络训练模型，而是从 480 个 `r7_s_train` 模板按 `predecessor_node_ids` 重建每个节点最多 5 层的 DAG signature，再按 6 个条件字段统计 signature 频次；预测时对 `r7_s_val` 取频次最高的 3 个 topology scenarios。旧 `future_h5` 的 5 是固定 rollout horizon；当前 `future_h5_layers` 不是 B05 行为头直接输出。未修改代码、数据、旧 artifact 或远端状态。

- 2026-09-02 规划阶段：用户希望在本地比较统计模型、传统机器学习、序列神经网络和图结构模型，但明确本轮不训练。已确认本地有 640 个 R7 模板/8,935 个节点，Apple M1 可用 Metal；当前 Python 3.9.13 未安装 `torch`、`sklearn`、`scipy` 等训练依赖。候选路线暂按“任务/泄漏契约 → 统计基线 → tabular ML → B05 风格序列多头 → 条件满足后 GNN/图解码器 → H5 调度验收”规划，不安装依赖、不创建实验目录、不修改模型代码。

- 2026-09-02 远端同步预检/纠偏：已核验 `root@connect.westc.seetacloud.com:12469` 的主机身份为 `autodl-container-41d846924e-d62d0601`，远端路径 `/root/autodl-tmp/scheduler`；`finetooling` 提供 Python 3.10.20、PyTorch 2.12.0+cu130、CUDA 和 RTX 4080 SUPER 32,760 MiB，且 `sklearn/scipy/networkx/xgboost/lightgbm` 可导入。首个实际 rsync 因 `.project/` 尾斜杠把内容写入远端根目录旧式控制文件，未更新远端真正的 `.project/`；未删除文件、未触碰代码/数据/权重。后续需用显式 `.project/` 目标修正并重新 SHA-256 复核。远端已有旧版 R7/B05 数据，但尚未有当前匹配的 `job_templates_r7_v02` 副本；不能直接混用旧数据。
- 2026-09-02 控制面同步完成：显式同步到 `/root/autodl-tmp/scheduler/.project/` 后，本地与远端 `AGENTS.md`、`.project/` canonical 文件及 archive 的 SHA-256 均一致。首个尾斜杠错误还把本地版本写入了远端根目录的旧式控制文件；未删除文件或触碰代码、数据、权重。根目录 legacy 副本不作为控制面，其中 `HANDOFF.md` 仍是第一次同步时的旧副本；后续只以 `.project/` 为准，额外写入已在交付说明中保留记录。
- 2026-09-02 训练数据门禁：远端 `r7_workload_20260817/job_templates_r7_v02.jsonl` 与本地匹配，SHA-256 为 `e0a5b12c9ba6f483a3cdf8abfaf012e3a55d59043f41cc4aab2d247cce09a464`；远端 R7 candidate role/tool 样本与本地匹配；B05 seed 11/22/33 checkpoint 均已存在且与 manifest 记录一致。训练/验证实验目录尚未创建。
- 2026-09-02 数据集一致性澄清：计划中的 topology predictor 训练/验证使用同一份 R7 模板源和同一套 `predecessor_node_ids`→未来 DAG 层目标生成器，按模板/视频隔离为 `r7_s_train` 480/120 videos 与 `r7_s_val` 160/40 videos；B05 行为模型的 `P_dev_train_only` 数据集不是 topology 标签集，只作为冻结的上游特征来源。B05 在 R7 candidate 上生成的 role/tool 样本与当前 R7 640 模板匹配，且 manifest 禁止 future events、target labels 和 resource truth 作为特征；不混用远端旧版 `scheduling_future_v1`。
- 2026-09-02 视频集合核对：R7 topology 的 `S_train` 120 + `S_val` 40 个视频，与行为/资源预测器的 `P_dev` 300 + `P_holdout_diag` 40 个视频交集为 0；因此不是同一批训练视频。行为/资源预测器只是对 R7 这 160 个 predictor-unseen 视频运行冻结推理，输出可按 R7 模板配对使用。
- 2026-09-02 数据归类修正：用户确认 topology predictor 属于 predictor block，不应使用 scheduler 专用 `S_train/S_val` 作为正式训练/验证数据。正式方案应复用行为/资源预测器的 `P_dev` 300-video development 与 `P_holdout_diag` 40-video final holdout 视频边界；topology 仍使用自己的未来 DAG 标签。当前 `EXP-20260901_topology_predictor_baseline` 的 R7 `S_train/S_val` 结果降级为 scheduler-side diagnostic baseline；重新训练前必须为 `P_dev/P_holdout_diag` 对应视频构建并核验 topology labels，未开始训练或改代码。

- 当前长步骤已完成：`EXP-20260901_topology_predictor_baseline` 只读取 R7 模板的 `predecessor_node_ids` 生成训练目标，输出不含 successor/node identity 的 `future_h5_layers`；B05 checkpoint 不在此 baseline 中使用。
- 验收已通过：fit/predict split=480/160 templates，sidecar 可加载且无泄漏，topology audit、同一 100 集 action audit 与 10 集 smoke 均通过；结果标记为 diagnostic，不扩大正式矩阵。

- 2026-09-01 本轮只读恢复：核对 `PROJECT.md`、`STATE.md`、`PLAN.md`、`DECISIONS.md`、`HANDOFF.md`，并运行 `project_os.py audit .`。六类 canonical 文件均存在；audit 的失败项来自多个历史/新实验目录的 contract 缺失、INDEX 注册缺失，以及 decomposition 实验缺少 `seed`，不是控制面文件缺失。
- 本轮未执行远端连接、同步、安装、模型/视频/trace 访问；仅在本地受控输入上完成了 topology baseline 代码、实验产物和 canonical 文档更新。

- 2026-09-01 本地进行 `EXP-20260901_pred_true_score_decomposition`：输入为只读的 `experiments/EXP-20260831_aligned_h5_score_contract/artifacts/action_value_audit/decision_records.jsonl`；目标是对同一 `episode_id + decision_index + state_hash` 下的候选过滤、Pred/True current/future/total score、排序、top-1 和 regret 逐项对比。
- 已完成只读预览：15,364 decisions、44,690 candidate rows；raw/model/strict 三层候选计数在记录中完全一致，尚未发现过滤差异；按完整 aligned score tuple 计算的 Pred/True 确定性 top-1 一致 10,017/15,364（65.1979%），两两排序翻转 29,683/75,964（39.0751%）。该预览不是最终交付，需由正式 decomposition artifacts 复核。
- 正式结果：`experiments/EXP-20260901_pred_true_score_decomposition/`；逐 candidate/decision 明细、top mismatch、score metrics 和 topology metrics 已生成。输入哈希、state/action 数、逐动作 raw score、候选计数、过滤一致性和原策略 top-1 重现均独立复核通过；58/58 unittest 通过。
- 关键发现：过滤、priority/tie-break 和 total 加和不是问题；candidate future mean absolute error=38,531.9ms，current=7,650.2ms，92.45% candidate 由 future 误差主导。Pred `future_h5` 是固定 5 synthetic event steps，True 是后继 5 DAG layers；Qwen3-4B planner 真值平均后继 node=9.1858，77.8287% candidate occurrence 真值后继数大于预测 event 数。
- 当前已获用户确认，已完成 future artifact 生成与 opt-in H5 评分路径；旧策略和旧 artifact 不变。先使用不依赖模型的 train-only empirical baseline，不把它冒充为 learned/formal topology predictor。
- 2026-09-01 当前操作已完成：独立实验 `EXP-20260901_h5_layer_contract_repair` 实现 `future_h5_layers`（每个 scenario 的每一层可含多个预测节点）和 opt-in `aligned_predopt_h5_layer`，同时保留旧 `future_h5`/`aligned_predopt_h5`。100 集 audit 覆盖 15,364 decisions/44,690 candidates，10 集 smoke 为 3 policies、160 jobs/policy、0 failed jobs，full unittest 60/60；新 layer policy 与旧 aligned Pred 完全相同，因为当前仅为 unary projection。
- 最终验证：layer artifact 8,935/8,935 rows 可加载且契约校验通过，旧 `future_h5` SHA-256 未变；`py_compile`、`sh -n` 和 60/60 unittest 均通过。正式状态为 `passed_diagnostic_unary_projection`，不是 topology predictor 完成。
- 外接盘只读发现（2026-09-01）：`/Volumes/Lenovo/scheduler/results/processed/behavior_nn_v1_r2/final_holdout/dataset/` 含 role 17,163 行、tool 5,853 行；`final_holdout/evaluation/B05/seed_{11,22,33}/checkpoint.pt` 三个 checkpoint 的 SHA-256 与现有 B05 artifact manifest 一致。manifest 明确 `fit_split=P_dev_train_only`、未使用 target labels/future events/resource truth，且 `test_targets_used=false`。
- 外接盘的 `/Volumes/Lenovo/scheduler/results/processed/r7_scheduling_future_20260817/` 含 640 templates/runs、8,935 nodes 的 B05 artifact；其 `source_templates_sha256=e0a5...` 与本地 `.scratch/action_value_audit_remote_inputs_20260825/job_templates_r7_v02.jsonl` 完全一致。外接盘当前的 `workload_v0_2_smoke_fixed_20260812` 是后来替换的 768-template 版本，不能与该 artifact 混用。
- 外接盘没有独立 learned topology/layer predictor；`scheduling_future_b05_artifacts.py` 的 B05 输出仍是逐事件 role/family beam `steps`，不能直接宣称多节点 layer prediction。当前实验已用匹配的 640 DAG templates 完成 empirical topology width/layer baseline；B05 checkpoint 仍保留给后续独立 learned stage。
- 当前执行计划：`EXP-20260901_topology_predictor_baseline` 已完成并登记；若要继续，只能在评审该诊断基线后另行确认 learned train-only topology predictor，不自动扩大正式矩阵。
- 第 1 阶段已通过（2026-09-01）：B05 dataset、三 seed checkpoint、R7 candidate samples 和 manifest 的本地副本均与外接盘源文件逐项 SHA-256 一致；B05 manifest 为 `P_dev_train_only`，R7 candidate coverage 为 640/640 runs、8,935/8,935 nodes。
- 恢复条件：若继续 learned stage，需先明确模型输入、训练目标和独立验收；在此之前保持当前 baseline 产物、正式矩阵和 `T_final` 不变。

- Local root: `/Users/liuqiuwei/Documents/调度`.
- Remote execution copy: `root@connect.westc.seetacloud.com:12469:/root/autodl-tmp/scheduler`（源/备份，后续新实验不依赖它）。
- Preferred independent local execution copy: `/Volumes/Lenovo/scheduler`（含代码、数据、模型、实验产物和 `.project/`）。
- Remote identity verified with local project key `~/.ssh/id_ed25519_opencode_westc_12469`; host responded as `autodl-container-41d846924e-d62d0601`, Python `3.12.3`.
- Completed in-scope: create `EXP-20260826_pred_mpc_rolling_horizon`, add opt-in `PredMPC-3/5` and smoke runner, run the final 10-episode staged validation.
- Completed: `RiskAwareScore` is implemented at `src/tracing/analysis/workload_v02_simulator.py::choose_action`; local related tests 18/18 and remote targeted tests 5/5 pass. The policy keeps hard feasibility filtering, then ranks `(ready_node, free_gpu)` using urgency, GPU/cache-transition risk, and waiting age.
- Completed remote run: synced only the approved simulator/test/experiment files after correcting an initial source-hash mismatch; final local/remote simulator SHA-256 matches. Ran 10 paired validation episodes with `myopic,risk_aware,predopt_h5,trueopt_h5`; 40 rows, 0 failed jobs, artifacts recovered locally. Final `.project/` and experiment metrics hashes also match local/remote.
- Control-plane audit: the new `EXP-20260826_risk_aware_score` contract is complete, but the overall Project OS audit still reports pre-existing missing contract files/registrations in older experiment directories; those unrelated legacy findings were not modified in this turn.
- Boundary correction completed: rollout, active-job area, and visible completion checks use known jobs and the current ready-node candidate window; continuation cost comes from frozen future artifacts, not template suffix nodes. The first three smokes remain invalidated diagnostics; final visible-window rerun is accepted for smoke only.
- New decision direction (2026-08-28): first align the TrueOpt-H5 and PredOpt-H5 score contract before attributing their gap to predictor quality. Preserve the legacy policies as read-only baselines; add only opt-in aligned variants after confirmation. The aligned contract will share candidate set, hard priority semantics, current-action inclusion, H5 successor scope, cache/memory transition model, and tie-break order, changing only truth versus predicted cost inputs.
- Local feasibility check (2026-08-31): required templates, episodes, future artifacts, and existing audit records are present locally. `PYTHONPATH=src` imports the simulator and action audit; the rollout audit requires `PYTHONPATH=src:.` because it imports the `scripts` package. Local Python is 3.9.13; `pytest`, `scipy`, `torch`, and `yaml` are absent. The aligned H5 simulator/audit path is CPU-only; future predictor experiments should use the external-disk copy after a local dependency environment is prepared, without requiring the remote host.
- Measurement boundary (2026-08-31): no new real-GPU measurement is required for the aligned H5 score-contract experiment. Reuse the frozen transition profile and existing simulator overlay; new measurement is only needed for an uncovered GPU/model or changed runtime semantics such as batch, prefetch, checkpoint/resume, concurrent multi-GPU, or deployment-latency claims.
- Aligned H5 implementation (2026-08-31): `aligned_predopt_h5` and `aligned_trueopt_h5` are opt-in only. They share current-action inclusion, H=5 DAG-layer scope, cache-aware resident-plus-workspace advancement, hard priority ordering, and tie-breaks; only predicted versus execution-truth cost inputs differ. The audit supports `--truth-contract aligned_h5` while legacy defaults remain unchanged.
- Aligned H5 action audit (2026-08-31): 100 episodes yielded 15,364 decisions. `aligned_trueopt_h5` is internally self-consistent at 100.00% top-1/0 ms regret; `aligned_predopt_h5` is 66.06% top-1 with 7,978.7 ms mean regret; old `predopt_h5` is 66.03%/8,065.5 ms; Myopic is 69.54%/7,659.2 ms under the aligned truth contract. This is diagnostic evidence only; full-event rollout is still open.
- Aligned H5 full-event audit (2026-08-31): 100 sampled common states and 774 strict-feasible branches completed with 0 failures and 100% reference consistency. Full-event top-1/regret were `aligned_predopt_h5` 31%/3,280.0 ms, old `predopt_h5` 31%/3,263.0 ms, Myopic 32%/3,295.1 ms, and `aligned_trueopt_h5` 34%/2,851.2 ms. The shared score contract did not materially close the full-event gap; H5 surrogate versus full-episode continuation remains the leading explanation.
- Aligned H5 smoke (2026-08-31): 5 policies × 10 episodes = 50 rows; each policy completed 160/160 jobs with 0 failed jobs. Mean completion was Myopic 176,937.1ms, old PredOpt-H5 175,963.2ms, aligned PredOpt-H5 177,058.3ms, old TrueOpt-H5 160,220.6ms, aligned TrueOpt-H5 162,234.5ms. Max simulated peak memory was 32,756MB against 32,760MB capacity; 0 capacity violations.
- Out of scope now: `T_final`, WAIT/RESERVE, preemption, multi-GPU, PPO rerun, raw data/trace changes.
- Acceptance status: baseline generation, sidecar contract/load, topology audit, 100-episode/15,364-decision aligned action audit, 10-episode/50-row smoke, full unittest, `py_compile` and shell syntax checks passed. Exact numeric facts and source hashes are in `experiments/EXP-20260901_topology_predictor_baseline/metrics.json`.
- Reproducibility check initially caught an aggregator field-location mismatch (`episodes_sha256` is owned by the action run manifest); the aggregator now cross-checks the real action/smoke manifests and completed successfully. This correction did not change experiment artifacts or policy results.
- Final Project OS audit still reports pre-existing contract/registration findings in older experiment directories; the current topology-baseline directory has its required control files, logs, artifacts and root metrics. The audit's prefix-key registration is now explicit in `experiments/INDEX.md`.

- 2026-09-03 ChatGPT Web 发送恢复点：已按用户要求准备脱敏研究 brief `.scratch/chatgpt_web_edge_experiment_brief.md`，仅包含 A/B/D 与 A+E/B+E/D+E 边预测实验设计问题，不含控制面中的私密会话 URL、凭据或原始数据。此前误选了内置浏览器；重新核对 `control-chrome` 后确认，用户点名 Chrome 时只能使用 Chrome，不能回退内置浏览器，也不能用 AppleScript、shell 或其他脚本替代。复核显示本机 Google Chrome 152.0.7977.75 正在运行，原生 host manifest 正确，但 Chrome 连接列表仍只暴露内置浏览器，唯一 `Default` profile 未检测到当前 Codex 扩展；因此尚未确认附件或正文已发送，也不能据此断言找错了 Chrome。恢复条件：用户在 Chrome 的 Settings → Computer use 中核对/启用 ChatGPT 浏览器扩展后，复用已确认的发送授权，核验 GPT-5.6 Sol + High，再发送 brief 并保存网页研究报告。
- 2026-09-03 ChatGPT Web 路由纠正：更新后的 `chatgpt-web-research` 已明确以 attach-only Chrome CDP bridge 为首选，禁止自动回退内置浏览器。bridge 脚本、MCP 虚拟环境、Chrome `127.0.0.1:9222` CDP 端点和本地缓存的 `chrome-devtools-mcp@1.8.0` 均存在；沙箱内首次预检因 Node 访问本机 CDP 被 `EPERM` 拒绝，受限只读连接后预检成功。当前 Chrome 中可见的是另一 ChatGPT 会话，不是控制面绑定的“架构设计评估”；对绑定 URL 的 `new-page` 导航返回 10 秒超时，未执行 upload/send，不能把 brief 发到当前不匹配会话。后续只沿同一 Chrome CDP bridge 恢复，先核验精确会话和 Sol + High，再发送。
- 2026-09-03 Chrome 精确会话导航核对：bridge 的 `new_page_operation` 固定调用底层 `new_page` 打开 `https://chatgpt.com/`，忽略命令行传入的 `--conversation-url`；因此本次新增的页面是 ChatGPT 首页，不能据此认为已打开控制面绑定的具体会话。`inspect --conversation-url` 只能从已经打开的页面中选择精确 URL。未修改 bridge 代码（根目录 AGENTS 要求先请示），未执行 upload/send；若继续，需要用户在 Chrome 打开绑定 URL，或先授权修复 bridge 的精确导航能力。

## Open Gate

当前 canonical `STATE.md` 和 `EXP-20260818_r8_optimizer_rl_upgrade/RESULT.md` 都仍把 R8-P4/PPO 标为未完成；`PLAN.md` 的 `[ ] R8-P4` 与现有证据一致，本轮不擅自改成完成。PredMPC 是独立的 opt-in 规划分支，不解封 PPO 或 `T_final`。

## Next Action

按以下顺序恢复（优先使用 `/Volumes/Lenovo/scheduler`）：
1. 在本地准备并锁定 Python/PyTorch/测试依赖环境，先做导入、checkpoint 加载和数据路径 smoke；不依赖远端。
2. 设计并确认 resource-aware future-content 或 conditional aggregation 的输出字段、训练标签和 acceptance gate，重点覆盖 `model_id/node_type/input_scale/cold_warm` 与长尾 runtime。
3. 继续沿用 `P_dev/train→validation→test→P_holdout_diag/holdout`，不使用 `S_train/S_val` labels，不以 holdout 事后改选模型或校准；预测器冻结前不接 scheduler，不加入 WAIT/RESERVE、抢占或多 GPU，不解封 `T_final`。

## J mechanism diagnostics done (2026-09-11; validation-only)
- D1' head-swap: J2/J3 head on B1 rep +80%~+170%; B1 head on J2/J3 rep ~+17%; matched pairs only 1-4% worse -> strong co-adaptation, not lost duration info.
- D2: q(model_class) entropy 0.33/0.35/0.31 -> 0.22/0.24/0.21 (J2) -> 0.17/0.19/0.15 (J3); top1 up to 0.95-0.96; marginal KL 0.008-0.017 -> interface hardened (causality unproven).
- D3: log-space duration pinball nearly identical across B1/J2/J3 (0.108-0.116); raw-space diffs small/noisy -> log/raw mismatch not the driver.
- From-scratch probe invalidated by sanity check (834 vs B1 head 284 vs constant-mean 502); replaced by head-swap. Record: docs/research/2026-09-11_j_mechanism_analysis.md.
- Awaiting user decision: J4a duration adapter/separate branch (preferred), J4b duration conditioning, J4c loss weighting/PCGrad.

## J4 design frozen (2026-09-11)
- `docs/p9d_j4_duration_branch_design.md` frozen v2: J4a (duration-decoupled head), J4b (decoupled + frozen shared feature); references J3/B1 inherited frozen.
- Governance: validation = official determination tier; test only exploratory second look (no confirmatory Core GO); two-layer checkpoint selection with duration gate pinned to frozen B1 validation values.
- Review: v1 MODIFY -> v2 ACCEPT (docs/research/2026-09-11_j4_design_review.md). Gate registered EXP-20260911_p9d_j4_duration_branch (pending execution). Cost ~15 GPU-min.
- Next: user approval -> Stage0 (J3 regression check + J4 routing assertions) -> smoke -> J4a/J4b x 3 seeds x 30 epochs -> validation verdict -> test exploratory.

## J4 completed (negative) — 2026-09-11
- Ran per frozen v2 design: stage0 (J3 regression max_abs_delta=0; J4 routing assertions pass) -> smoke -> J4a/J4b x 3 seeds x 30 epochs -> validation verdict -> test exploratory.
- Result: duration decoupling did NOT repair load-duration. J4a 0/3 feasible; J4b feasible only ep7-9 (runtime 1187/1285 vs B1 912/895); seed33 infeasible. Validation CI (validation_ci.json): H1 fail, H2 fail.
- Trade-off: at ep30 J4 runtime 741-798 (better than J3 814-898) but duration 304-327 (worse than J3 260-291); J4b > J4a => shared features carry duration info; competition hypothesis rejected.
- Per pre-frozen negative rule: reject the decoupling branch; do not keep splitting hidden.
- Governance: validation official tier; test exploratory only (test_exploratory.json, no confirmatory claim); J3/B1 inherited frozen; config sha b97879f8...; audit PASS (docs/research/2026-09-11_j4_result_audit.md); review CONDITIONAL->4 P1 applied (docs/research/2026-09-11_j4_result_review.md).
- Control plane: EXPERIMENT_GATE entry completed_negative; experiments/INDEX.md row added; artifacts outputs/j4_duration_branch/ + experiments/EXP-20260911_p9d_j4_duration_branch/RESULT.md.
- Next: user decision — J4c conditioning (model-side/execution-time context) or stop the joint-training line and consolidate (attribute interface + R0 mainline).

## J duration-degradation root cause closed (2026-09-11, validation-only)
- Distribution comparison B1 vs J3 + per-tau/class/magnitude decomposition with paired video bootstrap.
- Result: degradation is a fit re-allocation, not damage: tau=.90 worse (seed11 +33.2 [23.9,46.5], seed22 +30.1 [21.4,38.3], seed33 ~0), tau=.95 better in 3/3 seeds (-12.0/-7.5/-12.1), median unchanged; J3 median calibration actually better (q50 coverage 0.333 -> 0.553/0.363/0.463); spread slightly tighter.
- Class: VL-3B consistently slightly worse; Q3-4B mostly worse; VL-8B clearly better in 2/3 seeds. Magnitude: short loads better, mid-range worse. Structure seed-dependent (one seed outlier-dominated, one broad, one slightly better).
- Implication: no isolated broken pathway -> J4 decoupling could not have fixed it; J2/J3 runtime gain and duration drift come from the same representation-head co-adaptation.
- Records: docs/research/2026-09-11_j_mechanism_analysis.md (sec 6); GPT/literature: docs/research/2026-09-11_j4_literature_and_gpt_ideas.md.
- Awaiting user decision: close out (write-up) or try the cheap fixes (conditional hurdle + model-class gating / distributional duration).

## J line closed (2026-09-11, user decision A)
- No more repair experiments (J5 not started). Final picture: R0 oracle ceiling -> J3 runtime gain significant (3/3 seeds, 5.4-8.6% relative) but load-duration over the 5% NI line (+5.2%/+5.3%) -> no Core GO -> J4 decoupling negative -> root cause: fit re-allocation inside the duration distribution (tau=.90 worse, tau=.95 better in 3/3, median calibration actually better), no isolated broken pathway.
- Record: J RESULT sec 8; J4 RESULT; mechanism analysis sec 6; literature/GPT note; DECISIONS 2026-09-11 (two entries); EXPERIMENT_GATE J completed / J4 completed_negative; INDEX rows updated.
- Project state: no active experiment. S_*/T_final sealed; scheduler mainline (R8-P4 PPO, R8-P5 T_final) still open but unchanged.
- Next-step options (ask user): (1) evidence synthesis / write-up of the predictor line (zero compute); (2) new trace collection with system-side fields (residency/cold-start, workload scale) + new sealed eval set to unlock load-duration and confirmatory claims; (3) resume scheduler mainline (R8-P4/P5) needs RL seed inconsistency and budget resolution.

## J predictor integration candidate frozen + audited (2026-09-11)
- New experiment EXP-20260911_p9d_j_predictor_acceptance: selection rule pre-registered, primary J3:seed11 (validation RuntimeQScore 814.4), robustness J3:seed22/33, fallback B1:seed11; checkpoints copied + sha256.
- Audit (validation): structure len_mae 0.0473 / term_acc 0.979; content 0.9013; behavior 0.9444/0.8396; runtime pb 845 (coverage .556/.917/.968); load occ Brier 0.0135 ECE 0.0289; load dur pb 326.9. J3 >= B1 on all endpoints.
- Boundaries: memory head untrained (never expose); load duration has cold-start assumption + ~5% trade-off vs B1; no residency input.
- Identifiability: runtime error dominated by model_class (interface carries it); validation load events 214/242 cold-start, reuse=True implies no load, warm reloads only VL-8B n=28; clip_len 100% missing.
- Docs: experiments/EXP-20260911_p9d_j_predictor_acceptance/CANDIDATE.md + acceptance_audit.json; gate + INDEX + STATE updated.
- Next: interface packing to scheduler contract -> S_train/S_val inference-only smoke -> agree scheduler-level acceptance criteria -> small PredOpt matrix rerun. T_final sealed.

## J predictor interface packing + contract smoke (2026-09-11)
- New packer scripts/pack_j_predictor_artifacts.py: J3:seed11 -> outputs/j_predictor_artifacts/validation (j_future_h1/h3/h5 + j_future_h5_layers; 2029 anchors; 12s; layer schema validated).
- Scheduler loader smoke passed: load_future_artifacts() read all 2029 nodes; step carries legacy fields + resource extension (runtime/load quantiles).
- Key integration facts: existing scheduler policies use their own resource tables (resource block needs a new adapter/policy); raw_action/execution_lane are heuristic mappings; single deterministic scenario.
- S_* smoke pending: S_* traces at results/raw/r7_trace_full_20260817 (640 runs); need an anchor feature builder (event chain -> predictor input schema; P9d builder is the template).
- Docs: experiments/EXP-20260911_p9d_j_predictor_acceptance/integration_smoke.md; gate/STATE updated. T_final sealed.

## S_* inference smoke passed (2026-09-11)
- Anchor builder (reuses P9d derivations): 640 S_* runs -> 9575 anchors, 0 skipped, 23s; OOV: event fields ~0%, model_id 6%, stack_id 49%, temporal_scope/planner_model OOV (UNK).
- Packed J3:seed11 artifacts for S_* (24s, layer validated, loader compatible): outputs/sstar_predictor_artifacts/.
- Audit vs execution truth (5s, no fitting): role 0.912 / family 0.803 / runtime pinball 1045ms / load Brier 0.0225 / length err 0.408; in-domain ref 0.944 / 0.840 / 845 / 0.0135 -> OOD degradation but no collapse; star > langgraph.
- Docs: experiments/EXP-20260911_p9d_j_predictor_acceptance/integration_smoke.md sec 5; gate/STATE updated.
- Next: agree scheduler-level acceptance criteria; decide runtime-resource adapter (keep memory/lookup table); small PredOpt matrix rerun. T_final sealed.

## Forecast-aware scheduling plan received (2026-09-11, web review)
- Direction restated by user: does future agent-behavior prediction affect scheduling? Web plan: stop predictor tuning; run a forecast-aware ablation.
- Arms E0 (no future) / E1 (resource-only) / E2 (topology-only; most important) / E3 (full J3) / E4 (oracle); primary = mean completion time + deadline metrics; paired bootstrap 1000 episodes.
- Resource table kept as ablation A/B/C; memory excluded from main scheduling; OOD degradation reported at predictor and scheduler layers.
- Phases: Phase 0 interface validation (~1 day, 100 episodes, node-id mapping + resource-block consumption) -> Phase 1 pilot (100 eps, gate E3-E0>=1%) -> Phase 2 formal matrix (1000 eps x 6 arms x 3 seeds).
- Plan doc: docs/research/2026-09-11_forecast_aware_scheduling_plan.md. Awaiting user confirmation to start Phase 0.

## Forecast-aware Phase 0 passed (2026-09-11)
- Interface: node_id identical (no mapping), coverage 8935/8935 template nodes, scheduler-compatible dir outputs/sstar_predictor_artifacts_sched/prediction_artifacts (b05_* names + manifest).
- Runs: 10-eps smoke passed; 100-eps J3 vs B05 both passed, 0 failed jobs, myopic/oracle bit-identical across providers.
- E2 preview (topology-only; resource block currently ignored by policies): predopt_h5 completion 152,782ms (J3) vs 158,169ms (B05) = -3.4%; deadline miss 3.25% vs 4.06%; oracle gap +7.9%; local cost ~32s per 100 eps x 3 policies.
- Next: Phase 1 pilot needs resource adapter for E3 (opt-in new policy name) and E1 definition freeze; then 100-eps pilot E0/E2/E3/E4.
- Docs: experiments/EXP-20260911_forecast_aware_scheduling/PHASE0_REPORT.md; gate + INDEX + STATE updated. T_final sealed.

## Forecast-aware Phase 1 pilot done (2026-09-11)
- 100 eps x 5 arms (myopic / predopt_h1_jres / predopt_h5 / predopt_h5_jres / oracle). Headline: E2-E0 completion -8,176ms CI[-10,675,-5,771] (significant; future topology helps scheduling); E3-E2 +7,507ms (resource substitution hurts); E4-E2 -11,207ms (oracle gap).
- E3 regression root cause quantified: load semantics (table = median positive load per step vs J = occ*duration; 90% of steps occ<0.1), CPU lane direction reversed, runtime scale 0.34, current/future scale mixing.
- Adapter: workload_v02_simulator adds _jres_step_cost / predicted_future_cost_jres and policies predopt_h1/h3/h5_jres (additive only).
- Report: experiments/EXP-20260911_forecast_aware_scheduling/PHASE1_REPORT.md; gate/STATE updated.
- Next: calibrate E3 semantics (no S_* fitting) -> rerun pilot -> Phase 2 formal 1000-episode matrix x 3 seeds. T_final sealed.

## Forecast-aware Phase 2 done (2026-09-11, 1000 episodes)
- Result: future topology prediction significantly improves scheduling: J vs no-future -17,065ms CI[-18,692,-15,458], miss -2.34pp; J vs legacy B05 -12,583ms CI[-13,962,-11,323], miss -1.83pp.
- Resource substitution significantly harms: E3a vs E2 +16,012ms; load block adds nothing (E3b vs E3a ~0).
- Oracle gap remains 25,319ms (E2 is 14.7% above oracle).
- Runs: outputs/phase2_r7_j_1000 (5 arms), outputs/phase2_r7_b05_1000 (legacy control); report experiments/EXP-20260911_forecast_aware_scheduling/PHASE2_REPORT.md.
- Governance: validation episodes only; T_final sealed; no S_* fitting; deterministic policies; pending independent audit/review before final claims.

## Forecast-aware Phase 2 finalized (2026-09-11)
- Audited (15/15 deltas+CIs independently recomputed; line counts; manifest sha chain 0d0c9668.../aa943b60...) and reviewed (CONDITIONAL -> patches applied).
- Patches: artifact shape comparison added (J single-scenario + 14.2% zero-step vs B05 3x5; legacy claim downgraded to artifact-level); multiple-comparison note; single-seed boundary; load_threshold and E1 deviation; run commands/environment; evictions footnote.
- Final claims: future topology prediction improves scheduling (-7.9% vs no-future; -6.0% vs legacy artifact-level); direct resource substitution harms (+8.1% vs E2); oracle gap 14.7%.
- Next options (sec 6): shape-matched control; J3 seed22/33 robustness; domain-matched resource arm. T_final sealed.

## Phase 2 web opinion received (2026-09-11)
- Claims: support 'future topology prediction provides usable info and improves lookahead scheduling'; do NOT claim resource prediction works (E3a failed); E2-vs-legacy is provider/artifact-level only.
- Priority: (1) shape-matched J provider (fixed H=5, top-k scenarios, same cost interface) + J3 seed11/22/33; (2) oracle/J/static-table diagnostic under one interface; (3) 2x2 topology/resource oracle cross to locate the 14.7% gap.
- Extra validity threats: zero-step fallback (14.2%) needs ablation; template reuse vs bootstrap independence; artifact-workload same-source generalization; deterministic vs stochastic J comparison.
- Doc: docs/research/2026-09-11_fas_phase2_web_opinion.md. Awaiting user decision on which to start.

## Forecast-aware Phase 3 done - attribution corrected (2026-09-11)
- Length effect dominates: E2 vs J-fixed5 -14,932ms; matched content J vs B05-top1 +1,939ms (J not superior); earlier provider-level -6.0% explained by length-truncation convention.
- Gap decomposition: topology gap (truth topo + table vs E2) -13,538ms; resource gap at H=5 only -669ms (queue/miss worse); horizon effect (oracle unbounded vs trueopt_h5) -11,112ms = largest headroom.
- Oracle current-node resource does not help (+2,381ms). Zero-length predictions verified informative (P(actual<=1|pred=0)=0.999; P(actual=0|pred>0)=0.000; actual=0 always caught).
- Report: experiments/EXP-20260911_forecast_aware_scheduling/PHASE3_REPORT.md; runs outputs/phase3_*; stats .scratch/phase3_stats.py; zero-length check .scratch/zero_length_check.py.
- Next options: horizon experiments; length-aware policy switch; clean content comparison on L=5 subset; paper mainline = length/termination contribution. T_final sealed.

## Phase 3 web brainstorm received (2026-09-11)
- Key: cannot conclude content has no value; the E2 interface consumes only node-exists+length, q(A) is never consumed (no channel).
- Phase 1 (~1 day, no training): C1 oracle-content vs pred-content; C2 q(A)->expected cost (length-only / +argmax / +expected / +oracle); survival/hazard cost replacing argmax length.
- Phase 2 (~1 day): cache-aware future simulation comparison (length-only / topology+cache / topology+resource+cache).
- Phase 3 (re-register): H=10 (not 20); autoregressive rollout not recommended.
- Paper story: uncertainty-aware workflow-continuation planning, not predictor accuracy; resource substitution fails due to execution-state alignment.
- Doc: docs/research/2026-09-11_fas_phase3_web_brainstorm.md. Awaiting user decision.

## Forecast-aware Phase 4 done - content channel test (2026-09-11)
- C1 (oracle content) worse than predicted: +14,222ms -> content identity has no positive value; the 'missing channel' hypothesis is rejected (not an artifact of unconsumed q(A)).
- length-only (drop model identities, lane-average costs) BETTER than E2: -12,334ms -> identity-specific costing hurts.
- C2 expected-cost vs J-fixed5 (shape-matched): -2,347ms -> small gain from consuming q(A) vs top-1, far below lane-average.
- survival vs E2: -723ms (equivalent), survival vs J-fixed5: -15,655ms -> length/survival is the value driver.
- Report: experiments/EXP-20260911_forecast_aware_scheduling/PHASE4_REPORT.md; runs outputs/c1c2_*_1000; artifacts outputs/c1c2/j_*; builder scripts/build_c1c2_artifacts.py.
- Remaining (web plan): Phase 2 cache-aware comparison; Phase 3 H=10 (re-register). T_final sealed.

## Phase 4 web re-plan received (2026-09-11)
- User critique accepted: C1 fixed the content->cost mapping, so it couldn't test 'content valuable but mis-consumed'.
- Layer 1 (information value): A0 E2 (exists) / A1 Oracle H5+table (exists, 184,514) / A2 Oracle H5+truth resource (exists, 183,845; A1~A2 -> resource not the bottleneck) / A3 Oracle H10/20 (NEW).
- Layer 2 (consumption): fix oracle topology+resource, compare consumers C0 greedy sum / C1 survival (+15.7s already, top priority) / C2 risk-sensitive / C3 cache-aware (cost(node|history)).
- Priorities: 1) Oracle H10/H20 immediately (no training, minutes); 2) consumer ablation; 3) predicted topology + new consumer; 4) retrain predictor last.
- Paper line: 'dominant scheduling signal is future workflow evolution rather than per-node resource estimation'. Doc: docs/research/2026-09-11_fas_phase4_web_replan.md.
- Bridge note: duplicate stuck tab closed via close-duplicates --confirm-close, re-sent brief; wait timeout kept at 300s per user.

## Forecast-aware Phase 5 done - horizon + consumers (2026-09-11)
- Horizon: trueopt_h10 - trueopt_h5 = -10,231ms [-11,115,-9,363]; oracle - trueopt_h10 = -881ms -> H=10 saturates (H20 identical to H10).
- Consumers: predopt_h5_risk (predicted p90) - E2 = -10,505ms [-11,539,-9,511] (miss -1.01pp); aligned cache-aware (table p50) +650ms; risk vs length-only +1,829ms.
- Conclusion update: predicted resource quantiles DO convert under risk-sensitive consumption; earlier 'resources do not convert' holds only for p50 substitution with mismatched semantics.
- Runs: outputs/phase5_oracle_horizon_1000, outputs/phase5_consumers_1000, phase5_smoke50; report experiments/EXP-20260911_forecast_aware_scheduling/PHASE5_REPORT.md; stats .scratch/phase5_stats.py.
- Next options: H=10 predictor (re-register, new labels/gate) to make the 10.2s deployable; consumer refinement (lambda/quantile scan, length+lane+risk combo); paper mainline update. T_final sealed.

## Retrain vs consumer - web plan received (2026-09-11)
- Order: consumption mechanisms FIRST (main line), H=10 retrain SECOND, resource quality LAST. Do not retrain now (risk of optimizing the wrong target).
- Retrain spec: H=10 labels (L in [0,10], termination, per-slot existence/runtime/load; no attribute expansion); no new data needed (regenerate P_dev with future_window=10); new experiment ID/checkpoint/gate; not comparable-mixed with H5.
- Consumer candidates: (1) Quantile/CVaR cost with lambda scan {0,.25,.5,1} (<1h); (2) chance-constrained with deadline 1.2x/1.5x/2x; (3) stochastic MPC/rollout (long-term); (4) hazard upgrade; (5) cache-aware (defer, needs data).
- Overfitting control: 70% development / 30% frozen confirm split, or pre-register lambda/alpha/horizon and run once.
- Paper line: 'future workflow evolution + uncertainty-aware reasoning'.
- Doc: docs/research/2026-09-11_fas_retrain_consumer_replan.md. Awaiting user decision to start Phase 1.

## Forecast-aware Phase 6 done - consumers / H10 / cache (2026-09-11)
- Phase 1: q95 consumer is the best deployable config: 180,362ms vs E2 198,052 (-17,690ms CI[-19,238,-16,172]; miss -1.39pp); p90 -10,535; lam50/lam0/cc all worse.
- Phase 2: H10 retrain negative transfer (EXP-20260911_p9d_j_h10_predictor). New v3_h10 + j_h10 datasets registered; H10 model first-5 audit degrades (family .640 vs .803; runtime pinball 1691 vs 1046); scheduler: H10 q95 vs H5 q95 +14,587ms; H10 model with 5-step consumer +33,459ms.
- Phase 3: cache-aware predicted-resource consumer 220,430ms (+22,379 vs E2) -> defer until residency data exists.
- Loader fix: load_future_artifacts now carries any b05_future_h* file (H10 support); runner resume semantics noted (use fresh output dirs).
- Reports: experiments/EXP-20260911_forecast_aware_scheduling/PHASE6_REPORT.md; stats .scratch/phase6_stats.py. T_final sealed.
- Next options: (a) H10 with a different recipe (later-slot down-weight / separate head / longer training / two-stage); (b) confirm q95 consumer on the 70/30 confirm split; (c) paper write-up around length+horizon+risk-consumption.

## GPT literature consult - Phase 6 follow-up (2026-09-14)
- Bridge restored: Chrome for Testing must run with --proxy-server=http://127.0.0.1:7897 (direct chatgpt.com times out); profile ~/.chrome-chatgpt-bridge; conversation 6a9822da-e278-83e9-9c1a-675923acda0e.
- Sent Phase 6 results + consumption critique; GPT reply received (export .scratch/export_conversation2.md). Key: consumer is the bottleneck, not predictor precision; prioritize scenario-sampling+CVaR (A), adaptive risk (B), queue-aware rollout (D); H10 negative = representation dilution -> try H10-lite (H5 backbone + auxiliary horizon loss); must split 70/30 dev/confirm before further consumer tuning.
- GPT literature list (Clockwork/Orca/Sarathi/DistServe/AlpaServe/Tiresias/Themis/Gavel/Pollux) partially inaccurate/old; cross-checked and curated a better list in docs/research/2026-09-14_fas_phase6_gpt_literature.md (STS TCC2024, CVaR scheduling AAAI2023, PAL 2024, Dancer TCC2025, UniSched TC2024, Cuckoo SoCC2025, Llumnix OSDI2024, ARES 2025, MC-SF 2025, LTR NeurIPS2024, NexusSched 2025).
- Planned (awaiting approval): Step0 70/30 split -> Phase 7A scenario-CVaR -> 7B adaptive risk -> 7D light MPC -> H10-lite.

## Phase 7A done - split + scenario CVaR consumer (2026-09-14)
- Step 0: data/manifests/validation_split_dev700_confirm300.json (seed 20260914); runner --episode-ids-file; dev-only tuning, confirm used once.
- Distribution pack: outputs/sstar_predictor_artifacts_dist_sched (per-step model_probabilities + per-row length_probabilities).
- New consumers: predopt_h5_scen_k{0,50,100} (independent sampling), predopt_h5_comon_k{50,100,200} (comonotone). Dev700: all beat E2 except scen_k0; q95 still best (179,125 vs E2 196,354). q95 - scen_k50 = -3,991 [-4,492,-3,473].
- Confirm300 (first use): q95 - E2 = -18,765 [-21,948,-15,932], miss -1.36pp -> q95 confirmed as champion.
- Mechanism: multi-step summing under independent sampling concentrates (tail diluted); per-step p95 sum assumes perfectly correlated worst case and wins in this workload.
- Next: 7B adaptive risk (lambda by slack/queue), 7C component split (runtime vs load), 7D queue-aware rollout, H10-lite. Report: PHASE7A_REPORT.md.

## Phase 7B/7C done - adaptive risk + component split (2026-09-14, dev700)
- 7B: adapt (slack only) ~= E2 (+874, CI crosses 0); adapt_q (slack + queue pressure) 184,996 = -11,358 vs E2 but +5,871 worse than q95.
- 7C: rt95 (runtime p95 + load p50) 179,136 == q95 (+11 [-120,+135]); ld95 (load p95) 211,444 = +15,090 worse than E2.
- Mechanism: the entire q95 gain is the runtime-tail penalty summed over predicted steps; load tail contributes nothing and hurts alone.
- Consumer families tried and all lose to q95: scenario sampling+CVaR, comonotone, adaptive risk, chance-constrained, survival, cache-aware, content.
- Next options: 7D queue-aware rollout (last structural consumer idea, low EV) or H10-lite predictor (H5 backbone + auxiliary horizon loss). Report: PHASE7BC_REPORT.md.

