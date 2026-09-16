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
timestamp: "2026-09-06T23:19+08:00"
status: "valid"
---

# Research Report

## 5. assistant
Review verdict

Overall, the plan is directionally strong, but I would not freeze it yet. I found four issues that materially affect whether the first J1/J2/J3 experiment can support the claim “resource prediction improved”:

J1 is a valid baseline for J2/J3, but its stated causal interpretation is too strong. It simultaneously learns a resource head and changes the shared encoder, so it cannot by itself isolate “representation became resource-aware.”

The proposed Z is not yet demonstrated to be sufficient or minimal. In the attached sample, input_scale is absent and execution_lane is explicitly unknown; more importantly, the sample contains a same-layer dependency, so the present “DAG layer” is not an antichain.

Case-A joining is conceptually sound, but only under stricter rules about execution context, matching, and how model_resource_class is constructed.

The current code is a good base for the tensor/matching contract, but its experiment protocol is not yet compatible with the new plan: it opens holdout for every variant/seed, selects checkpoints using structure only, renormalizes top-3 scenario probabilities, and has no detached path corresponding to J2.

My recommendation is therefore approve the J1→J2→J3 architecture with revisions to the experimental contract, not approve the package unchanged.

1. Does J1 already confound representation learning with interface learning?
Verdict: [INFERENCE] Yes, for the question currently written; no, for its role as the J2 baseline.

The plan currently describes J1 as asking whether genuine resource supervision makes the shared representation “resource-aware.” Resource loss in J1 is allowed to update the shared encoder. 

review_joint_training_plan

That experiment simultaneously changes:

E
θ
	​

→E
θ
′
	​


and learns

H
R
	​

.

So if J1 beats the frozen legacy resource system, you cannot tell whether the improvement came from:

a better/new resource head;

adaptation of the shared GRU;

the new slot/layer representation;

or all three.

However, this does not invalidate J1→J2→J3. The comparisons remain clean if you reinterpret the questions:

J1:resource prediction without predicted-signature conditioning
J2−J1:incremental information value of q(Z)
J3−J2:incremental value of resource gradients flowing through Z.
Recommended diagnostic

Add a non-formal frozen-encoder probe, not a fourth formal stage:

J0
probe
	​

:sg(E
existing
	​

(X))→H
R
	​

.

Then:

J0 → J1 measures the value of representation adaptation;

J1 → J2 measures interface information;

J2 → J3 measures end-to-end interface learning.

This preserves your maximum of three formal stages.

[VERIFIED] Recent multi-task literature supports the underlying concern that joint optimization can change task representations in nontrivial ways and that average multitask optimization can under-optimize individual tasks; FAMO explicitly starts from this problem. 
NeurIPS 会议录

FAMO — NeurIPS 2023

So I would change the wording of J1, not remove J1.

2. Is Z=(exec_class,model_resource_class,input_scale,execution_mode) minimal and sufficient?
Verdict: [UNVERIFIED] Not yet. It is a good hypothesis, but the attached schema exposes one missing gate and one unavailable field.

Your plan correctly separates intrinsic node characteristics from scheduler-supplied GPU/cache/batch/contention state. 

review_joint_training_plan

 This separation is methodologically strong.

Vidur provides direct recent evidence that inference performance depends on concrete model/configuration/workload properties and uses profiling plus predictive modeling rather than coarse task semantics. 
MLSys 会议录
 DistServe similarly demonstrates that even different execution phases of the same LLM—prefill and decode—have sufficiently different characteristics to justify separate resource allocation and parallelism. 
USENIX

Vidur — MLSys 2024

DistServe — OSDI 2024

But your actual sample reveals two problems.

2.1 You need an explicit resource-domain gate

The future labels contain nodes such as:

Qwen model API calls;

cpu-metadata-adapter-v1;

video tools;

yet execution_lane is explicitly "unknown" / "not_in_trace". 

p9d_data_sample_redacted

For GPU load/memory, you first need to know whether the target is even a GPU-resource-consuming node.

So either:

exec_class

must explicitly contain:

{GPU, CPU, API,…},

or Z needs a separate:

resource_domain.

Without that gate, peak GPU memory is not a well-defined target for all future nodes.

I would therefore conceptualize the minimal candidate as:

Z=(resource_domain, model/implementation_class, work_scale, optional phase)
	​


with semantic role/family outside the minimal resource contract.

2.2 input_scale does not exist in the sample

The attached future label contains model_id, node_type, event_type, raw_action, etc., but I do not see a future-node token count, frame count, resolution bucket, tensor size, or equivalent work-size label. 

p9d_data_sample_redacted

So:

[VERIFIED from attachment] model_resource_class is potentially constructible from the existing labels.

[VERIFIED from attachment] exec_class/mode may potentially be derived from combinations of event_type, node_type, raw_action, and model metadata.

[UNVERIFIED] input_scale is currently labelable.

This matters enormously because the field that is probably most useful for runtime/activation memory may currently be unavailable.

2.3 “Minimal” must be experimentally defined

There is no paper that establishes your four fields as the unique minimal sufficient statistic.

The appropriate definition is empirical:

A field is required if deleting it materially worsens held-out resource prediction after conditioning on all remaining fields and legitimate runtime context.

So I would preregister a train/dev ablation:

Z

versus

Z∖{z
j
	​

}.

If removing exec_class changes nothing once model_resource_class is present, then exec_class is redundant.

If removing input_scale destroys runtime q95 accuracy, it is required.

That is what “minimal” should mean in this paper.

3. Should hard per-node expansion be rejected?
Verdict: [INFERENCE] Reject it as the mainline interface now; retain it only as a diagnostic baseline.

Given future family F1 around 0.40, the pipeline

argmaxfamily
i
	​

→profile(family
i
	​

)

turns one categorical error into a potentially very large continuous error.

That is particularly dangerous for your already-observed heavy runtime tail.

The plan's proposed J2 mechanism is much better:

q
i
	​

(Z)→H
R
	​

(q
i
	​

(Z),c)

using the full probability vector rather than argmax. 

review_joint_training_plan

I agree with that choice.

But I would not permanently rule out hard expansion because family F1 and future resource-signature accuracy are different quantities. It is possible that:

F1(family)=0.40

while a coarser resource class such as

{CPU, smallGPU, mediumGPU, largeGPU}

is predicted much more accurately.

So:

hard role/family expansion: reject;

hard Z expansion: later negative-control/ablation;

soft q(Z) marginalization: first formal design.

That recommendation is an INFERENCE from your observed errors, not a claim directly established by Vidur/DistServe.

4. Is Case-A joining leakage-free?
Verdict: [INFERENCE] The node-ID join itself is sound, but there are three important caveats.

The plan says to join each future node offline to its later measured runtime/load/peak-memory using label-side identity and then discard identity before model input. 

review_joint_training_plan

Your sample actually supports that workflow:

model input explicitly says future events/resource truth are excluded; 

p9d_data_sample_redacted

future labels retain node_id;

label_contract.node_identity_retained_in_label=true. 

p9d_data_sample_redacted

So:

node_idused to construct Y

is not leakage.

Using:

node_idinside X

would be.

This is normal supervised-target construction.

Caveat A — actual future runtime context

The plan says GPU placement/cache/batch/contention are scheduler-provided context. That is correct in principle. 

review_joint_training_plan

But you must distinguish:

c
available/candidate
	​


from

c
realized future
	​

.

The actual future GPU chosen, future batch, future co-runners, or future cache state created by intervening scheduling decisions cannot be injected into the causal future predictor merely because it is present beside the measurement.

That would be execution-truth leakage.

A measured runtime is really:

R
i
	​

=f(Z
i
	​

,c
i
realized
	​

)+ϵ.

If c
i
realized
	​

 varies but cannot be supplied at prediction time, the target contains irreducible conditional variance.

This distinction should be explicit in Case A before it is frozen.

Caveat B — model_resource_class must not be defined from the target itself

Suppose you define:

large-resource class = nodes whose observed peak memory > 10 GB.

and then predict:

model_resource_class

and use that variable to predict peak memory.

That is circular target engineering.

Safe definitions include static/model/configuration information or training-only profiles not using the target node's own held-out measurement.

If resource-class definitions depend on empirical resource data, construct them train-only/cross-fitted.

Caveat C — matching cannot use resource truth

This is especially important in your implementation.

The existing matching cost uses the future content probabilities and detaches the matching decision before applying CE; one assignment is reused across both content fields. 

p9d_future_role_family_multitask

That principle is correct.

When Z is introduced, the one identity-free assignment should be based on the preregistered semantic/signature fields, never runtime/load/memory.

Otherwise you can accidentally match:

slow predicted slot ↔ slow GT node

simply because their runtime values are close.

That would inflate resource accuracy.

5. Important matching ambiguity that the current plan does not resolve

This is one of the largest issues I found.

The current matching uses only:

role + action_family

because CONTENT_FIELDS = ("role", "action_family"). 

p9d_future_role_family_multitask

Once resource targets are attached, two same-layer GT nodes can have identical role/family but different:

models;

resource classes;

input sizes;

measured resources.

Your current code already audits duplicate role/family signatures instead of silently deduplicating them. 

p9d_future_role_family_multitask

So for joint training, the plan needs to specify:

C
match
	​

=C
semantic
	​

+C
Z
	​


using identity-safe categorical resource-signature attributes for which labels exist.

Resource measurements themselves stay excluded.

Hard identifiability case

If two nodes remain identical in all permitted identity-free matching attributes:

Z
i
	​

=Z
j
	​

,semantic
i
	​

=semantic
j
	​


but

R
i
	​


=R
j
	​

,

then their individual resource labels are not identifiable under the identity-free contract.

In that case you should not pretend there is a uniquely correct slot↔resource pairing. That subset needs set-level/distributional or aggregate treatment.

This is not a neural-network weakness; it is a target-identification issue.

6. A more serious data-schema discovery: your current “layers” contain same-layer dependencies
Verdict: [VERIFIED — needs clarification before formal experiment]

The sample's layer_offset=2 contains action:3 and api_call:4, but api_call:4 lists action:3 as a predecessor. Both are therefore in the same layer. 

p9d_data_sample_redacted

So the current label definition is not a topological-depth layering/antichain partition.

This has two consequences.

First:

width
l
	​


cannot even be interpreted as the number of mutually dependency-free nodes in that layer. Your existing warning that width is not simultaneous physical execution remains correct, but the semantics are weaker still: it is essentially cardinality of a BFS/step shell.

Second, treating a layer as an unordered set loses a dependency relation that exists within that set.

The code's Stage-0 audit checks that layer offsets are contiguous and that parent information comes from raw_trace.parent_step_ids, but it does not check:

layer(parent)<layer(child).

p9d_future_role_family_multitask

This is not necessarily fatal if your intended definition is explicitly “event-level BFS shell” and edges are outside the current predictor. But then I would not describe (L,widths) as a DAG skeleton.

I would describe it as:

identity-free future layered profile / BFS-layer profile.

If the paper needs “future DAG skeleton,” explicit dependency prediction becomes more important than the plan currently implies.

This issue existed before joint resource training, but joint resource aggregation makes the distinction much more consequential.

7. Code shapes, masks, and matching versus the plan
What is already consistent
[VERIFIED] H=5 fixed layer/slot tensorization is compatible

The model has:

layer head: HORIZON+1;

five width heads;

width classes 1..MAX_WIDTH;

a flattened HORIZON × MAX_WIDTH slot embedding;

per-content-head outputs reshaped to [B,HORIZON,MAX_WIDTH,C]. 

p9d_future_role_family_multitask

That is directly compatible with adding:

q(Z):[B,5,W
max
	​

,C
Z
	​

]

and resource quantiles:

[B,5,W
max
	​

,R,Q].
[VERIFIED] Structure mask semantics match the plan

Width loss is evaluated only where:

width_target>0

and inactive layers have no width class. 

p9d_future_role_family_multitask

This matches the plan's:

layer CE + active-layer width CE; mask derived from L/W.

[VERIFIED] Matching is shared across content attributes

The training loss constructs one joint cost, chooses one permutation, then applies it to every content field. 

p9d_future_role_family_multitask

This is correct for the identity-free set contract.

One terminology correction:

[VERIFIED] it is not literally implementing the Hungarian algorithm. For width ≤ 5 it enumerates all permutations and chooses the exact minimum.

That optimizes the same one-to-one assignment objective, so this is not a scientific flaw. But the formal document should say:

exact minimum-cost bipartite assignment (exhaustive for W≤5)

rather than claiming the implementation calls Hungarian.

[VERIFIED] Next-step family is correctly gated

The next-step family loss only applies under behavior_family_mask. 

p9d_future_role_family_multitask

8. Code/plan mismatches that should be fixed in the experimental specification
8.1 J2/J3 gradient wording is internally inconsistent

The stage table correctly says:

J2: detached;

J3: full flow. 

review_joint_training_plan

But the network section says the resource head receives a “detached signature distribution in J2/J3.” 

review_joint_training_plan

J3 obviously cannot be both detached and full-flow.

I would treat this as a documentation error, but it needs correction before preregistration.

8.2 Current soft conditioning does not implement J2

Current variant D computes:

softmax(layer_logits),softmax(width_logits)

and passes them through the conditioning trunk with no detach. 

p9d_future_role_family_multitask

So it implements:

topology probabilities → content, joint gradient.

It does not currently implement:

detached resource-signature probabilities → resource head.

That is expected because this is the previous model, but it means J2 cannot simply reuse D's gradient semantics unchanged.

8.3 The existing checkpoint-selection rule is wrong for a resource-primary experiment

This is probably the most important code/protocol mismatch after holdout.

For every non-N0 variant, current early stopping/checkpoint selection uses:

layer_count_MAE+width_vector_MAE

only. 

p9d_future_role_family_multitask

If that rule is retained for J1/J2/J3, you are training a resource-primary experiment but selecting checkpoints according to topology.

That can easily select:

epoch 12 with best shape;

instead of epoch 18 with substantially better resource score and still acceptable shape.

Recommendation

[INFERENCE] Preregister validation checkpoint selection as:

minimize validation ResourceQScore among checkpoints satisfying fixed structure/behavior preservation constraints.

Or, if you want a simpler rule:

select by ResourceQScore only and apply NI gates afterward.

But do not silently keep structure-only selection.

9. Holdout handling is not currently as frozen as the plan claims
Verdict: [VERIFIED — protocol mismatch]

The plan says any holdout-dependent choice invalidates the claim. 

review_joint_training_plan

Yet _train_one evaluates:

validation;

test;

holdout

for every variant × seed immediately after each checkpoint is frozen. 

p9d_future_role_family_multitask

Programmatically the final variant is still selected from validation, so this is not direct training leakage.

But it does expose all holdout results to the analyst before the joint architecture comparison is finished.

Even more subtly, Stage-0 reads train, validation, test, and holdout labels before training and aggregates layer/width histograms and ontology values across them. 

p9d_future_role_family_multitask +1

So the strict statement:

“the holdout is unopened until J1/J2/J3 architecture freeze”

is currently false.

Recommended interpretation

Automated split-integrity checks on holdout IDs/hashes are fine.

But holdout:

target histograms;

ontology values;

prediction metrics;

should not become visible during development/model selection if you want a genuinely frozen 40-video acceptance set.

This is a must-fix experiment-orchestration issue, even though it does not require changing the network architecture.

10. ResourceQScore versus memory false-safe
Verdict: [INFERENCE] Keep ResourceQScore as primary efficacy endpoint; make memory false-safe a hard safety gate.

The plan defines:

ResourceQScore=mean
r
	​

2
Pinball
.50
	​

+Pinball
.95
	​

	​

.

review_joint_training_plan

That is a reasonable primary score after resource-wise train-only normalization.

Quantile regression is appropriate when you care about both typical predictions and upper-tail uncertainty rather than just point MAE. Recent ICML work explicitly motivates quantile-based prediction intervals for precisely this reason. 
Proceedings of Machine Learning Research

Relaxed Quantile Regression — ICML 2024

But a mean across runtime/load/memory can hide an unacceptable memory failure.

Therefore:

ResourceQScore

should answer:

did resource prediction improve overall?

while:

MemoryFalseSafe

answers:

is this model safe to expose to a hard memory feasibility check?

A candidate can win the first and still be rejected by the second.

I would not make false-safe dominate the scalar objective, because that encourages extremely conservative predictions. Use it as a disqualifying gate.

Important metric inconsistency

The plan predicts only:

q
.50
	​

,q
.95
	​


but also lists “interval coverage.” 

review_joint_training_plan +1

With only q50 and q95 you do not have a conventional two-sided prediction interval.

You can report:

P(Y≤
q
^
	​

.95
	​

)

as q95 upper coverage/exceedance calibration.

If you actually want nominal 90% interval coverage, you need something such as:

[q
.05
	​

,q
.95
	​

].

Recent robust conformal work also reinforces that interval coverage under distribution shift is a distinct property that should be evaluated rather than inferred from in-domain calibration. 
Proceedings of Machine Learning Research

Ai & Ren — ICML 2024 robust conformal inference

11. Another missing metric distinction: oracle structure versus end-to-end structure

The present plan just says ResourceQScore.

That is underspecified.

You need two resource paths analogous to the existing content metrics.

Your current code already reports oracle-structure and predicted-structure content separately. 

p9d_future_role_family_multitask

I would preserve exactly that diagnostic philosophy:

ResourceQScore
oracle structure

answers:

given the correct node cardinalities/matching universe, can Hr estimate resources?

while:

ResourceQScore
predicted structure

answers:

how good is the deployable topology×resource interface?

If only oracle structure is reported, a missing future node is never charged to the resource predictor.

Given structure exact-match ≈0.83, that difference may not be enormous, but it should still be explicit.

12. Top-3 scenario probabilities have a hidden calibration problem
[VERIFIED]

Current scenario construction multiplies layer/width probabilities, retains three candidates, then computes:

raw_total = sum(top3 probabilities)
scenario_probability = probability / raw_total

so the displayed top-3 probabilities always renormalize to approximately 1. 

p9d_future_role_family_multitask

That means they are:

P(s
i
	​

∣s∈top3),

not the original total probability mass:

P(s
i
	​

).

If the true top-3 mass was only 0.62, this code turns it into 1.0.

For structure-only display this may be tolerable.

For uncertainty-weighted future resource aggregation, it is dangerous because residual scenario uncertainty disappears.

Recommendation

The formal resource-interface contract should distinguish:

raw scenario probability;

top-3 retained mass;

conditional normalized weight used within top-3;

residual probability outside top-3.

Otherwise “no calibration collapse” in J3 is not actually testable at the scenario level.

13. Partial-label fallback is methodologically sound
Verdict: [VERIFIED concept, INFERENCE for your implementation]

If some measured resources exist but others do not, masking resource loss where measurements are absent is cleaner than manufacturing role/family medians.

The recent CVPR 2024 Joint-Task Regularization paper explicitly studies multi-task learning where examples are not fully labelled for all target tasks, validating partially labelled MTL as a legitimate problem formulation. It does not prove your particular masking scheme is optimal, but it supports the general design. 
Open Access

Joint-Task Regularization for Partially Labeled Multi-Task Learning — CVPR 2024

The attached sample contains no actual runtime/load/peak-memory fields, so Case-A availability itself remains UNVERIFIED from these three attachments. 

p9d_data_sample_redacted

Final answers to the five review questions
Question	Verdict
1. J1→J2→J3 isolation?	Approve with wording change. J1 is a valid no-interface-conditioning baseline, but does not isolate representation learning. Add a frozen-encoder diagnostic probe outside the three formal stages.
2. Is Z minimal/sufficient?	UNVERIFIED. Good starting hypothesis, but add/encode resource domain; input_scale is not present in the sample; exact minimality requires ablation.
3. ResourceQScore vs false-safe?	Keep ResourceQScore primary; false-safe hard gate. Also fix q50/q95 vs “interval coverage” inconsistency.
4. Case-A join leakage/circularity?	Join itself is valid. Hard requirements: label-side IDs only; no realized future scheduling context as causal input; resource classes cannot be derived circularly from each node's resource target; matching cannot use resource truth.
5. Code/schema consistent?	Mostly structurally compatible, but not formally ready. Shapes/masks/set matching are good; J2 detach absent by design; current checkpoint selection, holdout access, top-3 probability handling, matching definition after adding Z, and layer semantics need revision.
Pre-registration stop issues

I would classify the following as must resolve before the formal J1 run:

Severity	Issue	Why
P0	Define whether same-layer dependencies are intentional	Current sample is not a true leveled DAG/antichain representation
P0	Define allowed runtime context c	Prevent realized-future scheduling truth from leaking into Hr
P0	Confirm actual measured resource labels exist and their measurement semantics	Sample alone contains none
P0	Remove holdout target/metric visibility before architecture freeze	Current implementation exposes holdout during development
P0	Change checkpoint-selection contract from structure-only	Otherwise resource-primary training is selected by the wrong target
P1	Define matching cost after Z is added	Role/family-only assignment may be ambiguous
P1	Resolve J2/J3 detach wording contradiction	J3 must allow gradient flow
P1	Define q95 upper coverage vs true interval coverage	q50/q95 alone is not an interval
P1	Preserve top-3 residual probability mass	Needed for uncertainty-aware resource aggregation
P2	Rename “Hungarian” to exact minimum-cost assignment if documenting current implementation	Current code enumerates permutations; objective is equivalent for W≤5

The central architecture idea survives this review. I would keep J1/J2/J3. The biggest risks are now not network design but target identifiability, runtime-context leakage, matching ambiguity, and experimental protocol. If those are fixed, J2 vs J1 and J3 vs J2 become genuinely informative experiments rather than just another multi-head score comparison.
