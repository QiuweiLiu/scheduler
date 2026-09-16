# Transition-profile simulator pilot

Status: **PASSED**

The simulator now has an opt-in transition overlay. Measured cold-load costs
replace only the applicable non-resident load cost, measured unload/eviction
proxies are charged only when eviction occurs, and the existing node runtime,
prefetch semantics, and full-node recompute preemption remain otherwise intact.
The default/off condition reports zero transition-profile cost.

## Execution gate

The frozen stress input contains 150 episodes with three GPU-capacity classes:
55 `[32760, 32760]`, 50 `[24576, 24576]`, and 45 mixed. Because the measured
profile covers only the RTX 4080 SUPER class, strict validation rejected the
mixed full-set attempt. The formal paired run therefore uses a deterministic,
source-order-preserving subset of the 55 `[32760, 32760]` episodes. No
24576-MB GPU was treated as covered by this profile.

Both paired conditions completed 55 episodes × 4 policies = 220 rows, covering
2,320 jobs per condition, with 0 failed jobs, 0 capacity violations, and a
maximum simulated GPU peak of 32,756 MB. A separate profile-off diagnostic over
all 150 source episodes also passed (600 rows).

## Paired result

| Policy | Off mean completion (ms) | On mean completion (ms) | Delta (ms) | Delta (%) |
|---|---:|---:|---:|---:|
| round_robin | 842,352.05 | 864,688.36 | 22,336.32 | 2.65 |
| myopic | 840,401.10 | 850,267.75 | 9,866.65 | 1.17 |
| predopt_h5 | 840,395.29 | 850,381.34 | 9,986.05 | 1.19 |
| oracle | 590,630.33 | 599,920.32 | 9,289.98 | 1.57 |

Across the profile-on paired run, the simulator charged 30,751,273.877 ms of
profile load cost and 982,906.773 ms of eviction cost, with 7,229 load hits
and 6,466 eviction-model hits. The event audit independently streamed 102,722
remote event lines; all 1,773 `model_load_start` events used
`load_source=transition_profile`, and event load/eviction totals reconciled to
the summary within floating-point rounding.

## Scope and limitations

- This is a single-GPU-slot simulator overlay; it does not add GPU-set
  allocation, cross-card communication, parallel runtime scaling, or multi-GPU
  preemption.
- The four measured models have no exposed checkpoint/restore API. Preemption
  remains full-node recompute; checkpoint costs were not fabricated.
- The measured load/eviction values come from the two-repeat RTX 4080 SUPER
  transition benchmark and are not a provider-wide cache-manager contract.
- The 55-episode result is a simulator integration pilot, not R8-P7's 1,000
  episode validation and not `T_final`. `T_final` remained untouched.
- The 450 MB raw event sample remains on the remote result directory; only its
  compact stream-audit summary was retained locally.

Canonical artifacts:

- `metrics.json`
- `artifacts/paired_rtx4080_super/`
- `artifacts/diagnostic_transition_off_150/`
- `artifacts/event_audit_summary.json`
