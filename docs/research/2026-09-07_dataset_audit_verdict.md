---
research_backend: "ChatGPT Web"
requested_model: "GPT-5.6 Sol"
requested_reasoning: "High"
model_verification_basis: "high_ui_mapping"
conversation_id: null
conversation_generation: "1"
conversation_url: null
verified_before_submit: true
verified_after_response: true
timestamp: "2026-09-07T15:54+08:00"
status: "valid"
---

# Research Report

## 6. assistant
v2 plan audit verdict

Overall verdict:

The three original revisions should be frozen, but Findings C–E reveal that the original experimental assumptions need one additional layer of safeguards.

My updated priority ordering is:

Finding C (event dependency reconstruction) → elevate to P0 foundation issue.

Finding A (longest-path re-layering) → correct, but only after C.

Finding D (horizon censoring) → add explicit censoring treatment before formal topology/resource experiments.

Finding E (width=5) → cap or merge; do not keep an effectively unlearnable class.

Finding B (workload_scale) → accept; resolution/frame-count does not block J1.

The most important change is:

The R0 oracle-resource experiment should be gated on the event graph contract, because otherwise the "oracle topology" itself is not a trustworthy oracle.

Finding A — layer semantics
Verdict: MODIFY
Labels

[VERIFIED — from audit result]
Current labels are BFS successor shells, not longest-path DAG depth. 4.22% of layers contain intra-layer dependency edges.

[VERIFIED — from mechanism description]
The problematic edges are introduced by reconstruction policy:

parentless same-step chaining;

step-parent expansion to all events in a step;

BFS on the resulting graph.

[INFERENCE]
The fix order should be dependency reconstruction → longest-path re-layering.

I agree with the proposed order.

Do not run longest-path on the current graph.

Reason:

Longest-path computes:

depth(v)=1+
u∈parents(v)
max
	​

depth(u)

but it assumes:

parents(v)

is already correct.

If the graph contains invented edges:

a→b

then longest-path faithfully preserves the wrong dependency.

It would convert a reconstruction bug into a more confidently wrong DAG.

Correct sequence
Step 1 — reconstruct event dependencies

Current:

step dependency→event dependency

with:

step
i
	​

→{all events in step
i
	​

}

This is too aggressive.

The proposed:

step
i
	​

→last(event
i
	​

)

is a much safer interpretation.

Step 2 — longest-path layer assignment

After that:

parent(u,v)⇒layer(u)<layer(v)

becomes guaranteed.

Should R0 oracle experiment wait?
Verdict: ACCEPT with gate

Yes.

I would add:

R0-oracle is only valid after the event graph contract is frozen.

Otherwise:

oracle topology

is not actually oracle.

The resource experiment would mix:

resource prediction error;

topology-label construction error.

That destroys the diagnostic purpose.

Literature relevance

This follows the same principle used in workflow/DAG systems: graph semantics must be correct before scheduling/resource estimation. Workflow systems such as WfCommons/WfBench treat workflow structure and dependencies as explicit ground truth rather than reconstructing them from ambiguous execution traces.

WfCommons documentation

Finding B — workload_scale
Verdict: ACCEPT
Labels

[VERIFIED — from audit]
Raw traces contain:

clip window;

query length;

nested API calls.

[VERIFIED — from contract]
Query text itself remains excluded.

[INFERENCE]
These three fields are sufficient as a first v2 workload-scale contract.

I would not block J1 on missing resolution/frame-count.

Why resolution/frame-count is not mandatory yet

The question is:

Is raw visual input size required to predict the resource target?

Not:

Does resolution affect computation in principle?

Of course resolution can matter.

However:

resource=f(model,input,implementation)

and many video-agent systems normalize input internally:

fixed frame sampling;

resizing;

embedding extraction;

compressed representations.

Therefore:

clip_length

may already explain much of the variance.

Current recommended contract

Freeze:

workload_scale=(clip_len,query_char_len,nested_api_count)

Do not call it complete physical input scale.

Use:

workload-scale descriptor

instead.

Future extension

Resolution/frame count should become an ablation only if:

Var(R∣workload_scale)

remains high.

Do not add it preemptively.

Vidur is a useful analogy: it does not assume one universal "input size"; it models performance from workload/configuration variables that explain measured behavior.

Vidur — MLSys 2024

Finding C — all event edges are reconstructed
Verdict: ACCEPT as a P0 revision
Labels

[VERIFIED — from audit]
There are zero observed event-level parent references.

[VERIFIED — consequence]
All event edges are derived from policy.

[INFERENCE]
This changes the importance of dependency reconstruction fundamentally.

This is not a minor data-cleaning issue anymore.

It is the foundation of the benchmark.

Previously the assumption was:

trace→event DAG

with some ambiguity.

The actual situation is:

trace→step semantics→event DAG

The middle transformation is part of the scientific contribution.

Why this matters for the predictor

Your topology target is:

future DAG

not:

future event sequence.

Therefore the reconstruction function:

G:raw trace→DAG

defines the labels.

If G changes, then:

layer count changes;

width changes;

matching changes;

resource aggregation changes.

Required freeze condition

Before any formal experiment:

Freeze:

step-parent interpretation;

same-step ordering rule;

orphan event handling;

layer construction.

Then regenerate labels.

Do not train J1/J2/J3 on the old labels.

Finding D — truncated horizons
Verdict: MODIFY
Labels

[VERIFIED — from audit]
17.4% samples are truncated_at_horizon.

[VERIFIED — consequence]
A layer-count label of 5 is censored:

L=5

means:

either

true termination=5

or

true termination>5.

[INFERENCE]
A separate termination signal is the correct fix.

Current problem

Your model currently treats:

layer_count=5

as a normal categorical class.

But for truncated examples:

P(L=5)

contains mixed meanings.

This creates label noise.

I would not simply downweight layer loss

Why?

Because the uncertainty is structured, not random.

Downweighting says:

this example is less important.

But the correct statement is:

this label is partially observed.

Recommended formulation

Introduce:

termination_status

with states:

{terminated,censored}.

Then:

terminal examples supervise exact future depth;

truncated examples supervise:

L≥5

rather than:

L=5.

This is standard survival-analysis style thinking.

A related concept is right-censoring: the observation ends before the event is observed.

Literature

This is a classical survival-analysis concept rather than a DAG-specific method. The key idea is that censored observations should contribute inequality constraints, not false exact labels.

A recent ML treatment example:

DeepHit: A Deep Learning Approach for Survival Analysis With Competing Risks

(Older than your preferred window, but the concept is methodological rather than a new architecture claim.)

Finding E — width class 5 has only 2 samples
Verdict: ACCEPT modification: cap at 4
Labels

[VERIFIED — from audit]
Width=5 has 2 examples.

[INFERENCE]
Keeping class 5 creates an almost impossible classification target.

This is not a harmless rare class.

Your width distribution:

1:15418
2:22455
3:14669
4:2143
5:2

means:

P(width=5)≈0.004%

The classifier cannot learn meaningful decision boundaries.

Should width=5 remain as a placeholder?

No.

A structural placeholder should represent a meaningful uncertainty state.

Width=5 currently represents:

two real examples;

possible truncation effects;

tail behavior.

It is not a stable class.

Recommended options
Preferred

Merge:

width≥4

into:

width=4+

or:

width_bucket=4+

Then:

classes={1,2,3,4+}.

This preserves the structural fact:

very wide layer.

without pretending you can distinguish width 4 and width 5.

Do not simply delete width=5 examples

That changes the topology distribution.

Merge them.

Clean bills
Verdict: ACCEPT

The following are positive:

Split reconciliation

The split counts:

train 13,754;

validation 2,029;

test 1,520;

holdout 1,380;

are consistent.

This supports the data-boundary claim.

Anchor distribution

Uniform prefix anchors are important.

No evidence of:

anchor=early only

bias.

This matters because otherwise the future prediction task would secretly become:

near termination

prediction.

Parent integrity

Zero missing/forward references are good.

However:

[INFERENCE]

This does not validate edge semantics.

It only validates the reconstructed graph is internally consistent.

Updated v2 freeze checklist

Before J1/J2/J3:

Item	Status
J0 probe	✅ freeze
workload_scale fields	✅ freeze
resolution/frame-count	postpone
step-parent semantics	🔴 must freeze
same-step chaining	🔴 must freeze
event DAG reconstruction	🔴 regenerate labels
longest-path layers	after reconstruction
width classes	change to 4+
truncation handling	add termination/censoring semantics
R0 oracle resource	wait until DAG labels regenerated
Final verdict summary
Finding	Verdict	Reason
A. BFS vs longest-path	MODIFY	longest-path is correct, but only after edge reconstruction
B. workload_scale	ACCEPT	sufficient first version; no resolution blocker
C. all edges reconstructed	ACCEPT as P0	elevates label generation to core methodology
D. truncated horizons	MODIFY	treat as censored labels, not exact depth=5
E. width=5	ACCEPT modification	merge into width 4+

The biggest conceptual shift is:

The next formal experiment is no longer just “joint topology × resource learning.” It is first a label-contract stabilization experiment.

Once the event DAG, censoring semantics, and width ontology are frozen, the J1/J2/J3 comparison becomes scientifically interpretable. Before that, improvements or regressions could simply reflect target-definition noise.
