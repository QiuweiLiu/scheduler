> Paired per-episode comparison of each frozen baseline vs the main line (pdrs_resident = F0 + residency actions). Cross-interface: baselines do not carry the residency actions; use for end-to-end performance only, not mechanism attribution.

| baseline | mean_delta_vs_main_ms | mean_pct | p95_delta_vs_main_ms | p95_pct | miss_delta | makespan_delta_ms |
|---|---|---|---|---|---|---|
| parrot_appfifo | +2889.4 | +4.19% | -598.7 | -0.47% | +0.0032 | +2469.8 |
| qlm_queue | +4220.5 | +6.12% | +7881.4 | +6.14% | +0.0033 | +678.2 |
| llmsched | +1780.7 | +2.58% | +2084.7 | +1.62% | +0.0027 | +3052.6 |
| hermes_gittins | +1191.8 | +1.73% | +2516.4 | +1.96% | +0.0023 | +3021.1 |
| torpor_lifecycle | +4363.0 | +6.33% | +3702.0 | +2.88% | +0.0079 | -1629.3 |
| fcfs | +5165.3 | +7.49% | +5326.0 | +4.15% | +0.0078 | +719.8 |
