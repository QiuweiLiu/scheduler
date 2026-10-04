# 真实档位校准 + profile 契约复验（网页版独立审核）— ACCEPT；activation gate → PASS

## 来源与范围

网页版 GPT（GPT-5.6 Sol + High；generation 11）对固定提交 `cc9793ae270c66b43d94d4de373cbf39e56b428e`
做静态复验（前两轮：619a6d3 NEEDS CHANGE 3×P1 → b8a34b6 NEEDS CHANGE 1×P1 → cc9793a ACCEPT）。

**结论：ACCEPT — P0 0 / P1 0；`profile activation gate: PARTIAL / NOT PASSED → PASS`。**

闭环证据链（审核方原文）：真实档位实测 → committed profile → SHA-pinned extension config →
evidence-based profile contract → real-dev activation → synthetic isolated path →
measured interference cell actually consumed。

审核方明确：**该轮冻结，不再做 activation/profile 契约低收益修改；可进入正式主表测量阶段。**
本地测试数字仍记为 UNVERIFIED execution report。

审核回复原文如下（未改动）。

---

炜，裁定：**ACCEPT**。

**P0 = 0，P1 = 0。**

上一轮唯一阻断项已经完整闭环：

- Phase B 的 synthetic workload 已全部切换到真实 role/shape 词表：4B=`planner`、8B=`videotool_spatial`、3B=`answer_generation`；每个 synthetic GPU node 也补齐了 `gpu_model` 证据。fileciteturn133file0
- synthetic episode 现在与真实 Phase A 一样经过 `apply_real_workload_profile_contract()`，因此 GPU identity class、stratum coverage 等 fail-closed 门没有被绕过。fileciteturn134file0
- Hermes Phase B 实际出现 `prefetch_start`，并产生 `node_interference_start(extra_ms=33.8)`；`failed_jobs=0` 且 `fail_closed_reasons={}`。fileciteturn135file0
- 这个 `33.8 ms` 能精确对应 fixed commit 内 v7 的实测单元  
  `Qwen3-VL-8B-Instruct | videotool_spatial <- Qwen2.5-VL-3B-Instruct`，不是 fallback、默认值或旧档位残留。fileciteturn136file0
- Phase A 真实 dev 30 集仍保持四臂 `episodes_with_failed_jobs=0`、`fail_closed_reasons={}`，且 Parrot / QLM / Hermes / Torpor 的关键机制计数均保持激活。fileciteturn135file0
- fixed commit `cc9793ae...` 本身只修改 audit script 与重跑后的 report，对 committed v7 没有再次变更。fileciteturn132file0

因此我明确裁定：

**`profile activation gate: PARTIAL / NOT PASSED → PASS`**

现在这条证据链已经是闭合的：

`真实档位实测 → committed profile → SHA-pinned extension config → evidence-based profile contract → real-dev activation → synthetic isolated path → measured interference cell actually consumed`

所以这一轮可以冻结，不建议继续在 activation/profile 契约上做低收益修改。**可以进入正式主表测量阶段。**

本地“477 tests 0 failure + 10 environment errors”仍按你的定义记为 **UNVERIFIED execution report**；它不影响本次基于固定提交的代码/产物审查结论。