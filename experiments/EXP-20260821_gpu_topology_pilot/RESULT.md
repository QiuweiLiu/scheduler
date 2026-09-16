# EXP-20260821_gpu_topology_pilot — Result

## Gate

**PASSED.** The remote matrix produced 240 episodes × 4 policies = 960 rows;
8,192/8,192 jobs completed, with zero failed jobs and zero capacity violations.
The eight-cell event audit (one episode per cell) completed and its gzip passed
integrity checking. R7 and `T_final` were not modified/read.

Full numeric data is in [`metrics.json`](metrics.json) and
[`gpu_topology_report.json`](gpu_topology_report.json).

## Myopic versus Oracle reference

Times are mean completion times in milliseconds. The Oracle here is the existing
truth-informed ranking reference, not a proof of global optimality.

| Cell | Myopic completion | Oracle completion | Myopic deadline miss | Myopic gap to Oracle |
|---|---:|---:|---:|---:|
| G4-S0 | 111,726 | 107,879 | 1.3% | 3.6% |
| H4-S0 | 112,352 | 108,627 | 1.3% | 3.4% |
| G8-S0 | 98,702 | 99,291 | 0.0% | -0.6% |
| H8-S0 | 98,782 | 99,556 | 0.0% | -0.8% |
| G4-S4 | 570,644 | 429,843 | 81.2% | 32.8% |
| H4-S4 | 570,164 | 431,087 | 81.2% | 32.3% |
| G8-S4 | 308,879 | 244,730 | 48.3% | 26.2% |
| H8-S4 | 308,752 | 245,476 | 48.5% | 25.8% |

## Findings

1. Increasing GPU count has a large effect under pressure. Relative to the
   previous 2-GPU controls, Myopic completion falls from about 214.7 s to
   111.7 s with 4 GPUs, and to 98.7 s with 8 GPUs in S0. Under S4 pressure it
   falls from about 1,097.6 s to 570.6 s with 4 GPUs and 308.9 s with 8 GPUs.
2. The extra 4→8 GPU benefit is workload-dependent: about 11.7% in S0 but
   45.9% in S4. Eight GPUs are underutilized in S0 (about 19% mean utilization)
   but remain highly occupied in S4 (about 91%), so S4 is the meaningful
   high-concurrency setting for an 8-GPU comparison.
3. Heterogeneous capacity has little effect on completion in this range. H4
   differs from G4 by about +0.6% in S0 and -0.1% in S4; H8 differs from G8 by
   less than 0.1% in both conditions. It does affect cache churn: RR eviction
   counts are higher on H4/H8 than on equal-capacity G4/G8.
4. More GPUs widen the action space in baseline: the Myopic event-audit action
   width median is 3 for G4-S0 and 8 for G8-S0. Under S4 the ready-node limit
   dominates, with medians around 10 for both G4 and G8.
5. PredOpt-H5 remains tied with Myopic in the S4 cells. Adding GPUs reduces
   queueing but does not make the current finite-horizon score influence the
   selected action.
6. GPU truth runtime tails stay essentially unchanged (audit P50 about 6.08 s,
   P90 about 25.67 s), confirming that the topology effect comes from
   concurrency/queueing rather than changed node compute time.

## Conclusion

The two-GPU workload was not sufficient to expose the full effect of GPU count.
For the scheduling study, 4-GPU and 8-GPU S4 workloads are both useful: 4 GPU
retains stronger contention, while 8 GPU tests whether the policy scales when the
action space is wider. Heterogeneous capacity should be treated as a secondary
cache/resource-stratification factor, not the main source of completion-time gap.

This remains a topology calibration pilot, not the final `T_final` claim.

