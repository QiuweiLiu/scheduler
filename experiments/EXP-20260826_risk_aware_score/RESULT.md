# EXP-20260826-risk-aware-score Result

## Status

Implementation, local contract tests and the bounded remote paired smoke passed.

## Hypothesis

A transparent action score inspired by the thesis—urgency minus waiting-age compensation plus GPU/cache-transition risk—may use current scheduler-visible information more effectively than the existing current-runtime/future-cost ordering.

## Scope

The policy is opt-in and only ranks the current strict-feasible `(ready_node, free_gpu)` candidates. It keeps one GPU slot per node, does not add WAIT/RESERVE, and does not inspect future DAG suffixes or execution truth. `T_final` remains sealed.

## Planned acceptance

- all policies complete the same 10 validation episodes;
- zero failed jobs and zero capacity violations;
- report mean/p95 completion, queue, deadline miss and action-width-related diagnostics separately;
- do not treat this smoke as a method-selection or final-test result.

## Result

The four policies completed the same 10 validation episodes with 160/160 jobs per policy and zero failed jobs. `risk_aware` had the lowest mean completion time among the three deployable policies in this smoke: 174,733.3 ms, versus Myopic 176,937.1 ms and PredOpt-H5 175,860.0 ms. The paired per-episode comparison was 6/10 wins against Myopic and 5/10 against PredOpt-H5, so the direction is promising but not stable evidence.

The improvement came with a trade-off: RiskAwareScore had higher mean job queue time (106,960.8 ms) and more GPU evictions (20.8) than Myopic (85,629.5 ms and 18.2). Its deadline miss rate was 0.0% in this smoke, compared with 0.625% for Myopic and 1.25% for PredOpt-H5. The strict-feasible rate recorded by the policy was 1.0 for every decision; no simulated OOM/failed-job path was observed.

## Interpretation and limits

This is evidence that a simple, explainable current-state action score can change the scheduling order in a useful direction on this small sample. It does not show that the score is globally optimal, that the paper's weights transfer, or that future prediction has been successfully exploited: this first RiskAwareScore version does not consume future artifacts. It is a diagnostic pilot, not a 300/1,000-episode selection result, and `T_final` remains sealed.
