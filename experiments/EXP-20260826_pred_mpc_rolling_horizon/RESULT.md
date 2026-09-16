# EXP-20260826 Result

## Status

Visible-window 10-episode smoke passed.

## Interpretation

The opt-in predicted rolling-horizon policy completed all 160 jobs per policy with 0 failed jobs. On this small paired sample, `PredMPC-H3` was 9,073 ms (+5.13%) slower than Myopic, `PredMPC-H5` was 10,152 ms (+5.74%) slower, and `PredMPC-H5-no-terminal` was 8,281 ms (+4.68%) slower. Existing `PredOpt-H5` remained 1,077 ms (-0.61%) better than Myopic and 11,230 ms faster than `PredMPC-H5`. Terminal continuation did not help this implementation: H5 was 1,871 ms slower than H5-no-terminal.

The first three smokes are retained as invalidated diagnostic artifacts. Review found that those versions could traverse the full internal job/template list. The final implementation restricts predicted rollout, active-job area, and visible completion checks to known jobs and the current ready-node candidate window, with continuation cost obtained from frozen future artifacts rather than template suffix nodes. The reported numbers are from the final visible-window rerun.

## Limitations

- This is only a 10-episode smoke, not a method-selection result.
- The rollout covers the current ready-node window; it does not yet synthesize a predicted DAG successor graph.
- The terminal continuation is a frozen-artifact cost proxy, not full-event optimal value.
- No stress matrix, 300-episode selection, 1,000-episode validation, or `T_final` run has been performed.
