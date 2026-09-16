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
timestamp: "2026-09-06T11:46+08:00"
status: "valid"
---

# Research Report

## 5. assistant
Joint training design for topology + resource prediction
Executive Summary

[INFERENCE] The first formal joint-training experiment should not ask “can one giant loss make everything better?” It should isolate three increasingly coupled mechanisms:

Stage	Resource input	Resource gradient path	Question answered
J1 — Joint-Independent	shared causal encoder + future slot/layer query + runtime context	resource → shared encoder	Does adding genuine resource supervision make the shared representation more resource-aware?
J2 — Detached Soft-Conditioned	J1 + stopgrad(P(resource_signature))	no resource gradient through content/signature head	Do predicted topology/content features actually contain useful resource information?
J3 — End-to-End Soft-Conditioned	same as J2	resource → signature/content → shared encoder	Can resource supervision reshape the semantic interface and improve real resource accuracy?

My default expected winner is J2 or J3, but J3 should only be considered valid if the resource loss is grounded in independent measured resource truth. If future-node “labels” are merely role/family medians or outputs of the old resource predictor, J3 is optimizing a proxy rather than real runtime/load/memory.

The most important change from the previous design is therefore:

Do not fabricate continuous future-resource truth from role/family.
	​


If a future node later appears as an executed/current node with an actual measurement, its future target may be joined offline to that measurement. Using future identity for label construction only is not leakage; providing that identity or resource value as an input would be leakage.

If those measurements genuinely do not exist, train the resource mapping on the measured current/executed nodes and train topology on future nodes as a partially labelled multitask problem. Resource-equivalence-class profiles can be used as weak supervision/teacher distributions, but not as evidence that true future-resource prediction improved. Recent work explicitly recognizes partially labelled multi-task learning as a legitimate setting; it does not require hallucinating missing labels. 
Open Access CVF

1. What exactly should be jointly learned?

The conceptual model should be:

h
t
	​

=E(X
≤t
	​

)

with four objective groups:

H
S
	​

(h
t
	​

)→future structure
H
C
	​

(h
t
	​

)→future node semantics/resource signatures
H
R
	​

(⋅)→(runtime,load,memory)
H
B
	​

(h
t
	​

)→immediate next-step role/family.

The existing next-step behavior task stays as the local auxiliary anchor, unchanged across J1/J2/J3.

The critical intermediate variable should no longer be only:

(role, family).

It should be a resource-oriented signature such as

Z
i
	​

=(exec_class, model_resource_class, input_scale, execution_mode),

where labels are included only if actually available and causally predictable.

Then:

p(R
i
	​

∣X,c
i
	​

)=
z
∑
	​

p(z
i
	​

=z∣X)p(R
i
	​

∣z,c
i
	​

),

where c
i
	​

 is execution context such as GPU class, resident model/cache state, batch context and other state that is known to the scheduler rather than guessed by the topology predictor.

[VERIFIED] This factorization is consistent with recent GPU/LLM systems. Vidur constructs performance models from experimental profiling and predictive models and explicitly notes that inference performance depends on configuration choices such as model parallelism, batching and scheduling. USHER similarly uses a GPU model-resource-requirement estimator and then separately considers batch size, model placement and interference in scheduling. 
proceedings.mlsys.org
+1

So resource supervision should teach:

“given this resource-relevant operation and execution context, what will it consume?”

rather than:

“somehow infer the entire GPU environment from a future semantic label.”

2. First formal experiment: only three stages

Keep everything else fixed: same causal GRU, same splits, same three seeds, same optimizer/training budget, same structure/content representation, same next-step auxiliary coefficient.

J1 — Joint-Independent

Define an independent future resource slot representation

r
lk
	​

=f
R
	​

(h
t
	​

,e
l
	​

,e
k
	​

,c),

where e
l
	​

,e
k
	​

 are layer/slot embeddings and c is legitimate runtime context.

Critically:

H
R
	​


does not read content probabilities.

Train:

L=L
S
	​

+L
C
	​

+L
R
	​

+λ
B
	​

L
B
	​

.

Resource gradients are allowed into the shared GRU through h
t
	​

.

What J1 tests

[INFERENCE]

J1 answers:

Does genuine resource supervision cause the common causal representation to retain information useful for resource prediction, even before explicitly connecting topology content to resource?

If J1 already substantially improves resource prediction, some useful information existed in the causal prefix but was not being optimized for previously.

J2 — Detached soft resource-signature conditioning

Now give the resource head:

r
lk
	​

=f
R
	​

(h
t
	​

,e
l
	​

,e
k
	​

,c,sg[q
lk
	​

(Z)]),

where sg denotes stop-gradient.

Do not use:

argmaxq(Z).

Use the entire probability vector.

For example:

q(Z)=[0.1 VLM
small
	​

,0.6 VLM
medium
	​

,0.3 VLM
large
	​

].

The resource head sees uncertainty instead of pretending the semantic prediction is certain.

What J2 tests

J2 vs J1 isolates:

information value of predicted topology/content
	​


without letting the resource objective rewrite those predictions.

This is arguably the most important experiment.

If

J2≫J1,

your topology/content interface contains useful information.

If

J2≈J1,

then making the architectures more tightly coupled is unlikely to solve the problem unless the content/signature predictor itself improves.

J3 — Full end-to-end soft conditioning

Only change:

sg[q(Z)]→q(Z).

Now

∇L
R
	​


is allowed through the resource-signature/content head and shared encoder.

Everything else must remain identical.

J3 vs J2 therefore isolates:

value of resource-driven semantic representation learning
	​

.

This is the cleanest possible test of your “joint training should make resources better” hypothesis.

Important restriction

[INFERENCE — hard methodological requirement]

Do not treat J3 as a valid true-resource experiment if L
R
	​

 uses only profile-derived pseudo-labels.

Otherwise the resource loss can simply teach the topology model:

“predict whichever semantic label makes my own lookup table easiest to reconstruct.”

That might increase proxy-resource accuracy while moving farther from actual runtime/memory.

3. The most important issue: where do future resource labels come from?

There are three qualitatively different cases.

Case A — later execution measurements exist

This is the preferred solution.

Suppose at prefix t, node v is future.

Later in the same development trace, v executes and you have:

(runtime
v
	​

,load
v
	​

,peakmem
v
	​

).

You may use the real node identity offline to join:

future target(v)↔resource measurement(v).

Then discard the identifier.

Training input remains:

X
≤t
	​

.

Training target becomes:

(S
future
	​

,C
future
	​

,R
future
	​

).
Leakage status

[INFERENCE]

This is not future leakage.

Supervised learning always uses future truth as the target. Leakage occurs only if that measurement, future ID, or information derived from it becomes part of the predictor input.

This gives you the strongest possible resource experiment.

Case B — measured resources exist only for nodes when they are current/executed

Then do partially labelled multi-task learning rather than inventing labels.

You effectively have two kinds of training record.

Future-prediction record:

X
≤t
	​

→(S,C),

with no L
R
	​

.

Measured-node record:

(X
≤t
	​

,Z,c)→R.

Only evaluate the losses that have labels:

L=m
S
	​

L
S
	​

+m
C
	​

L
C
	​

+m
R
	​

L
R
	​

+m
B
	​

L
B
	​

.

This is scientifically cleaner than replacing missing R with a median lookup.

[VERIFIED] Recent CVPR work explicitly studies multi-task learning where not every example has labels for every task, motivated by exactly the impracticality of complete multi-task annotation. 
Open Access CVF

Consequence

Your resource model learns:

p(R∣Z,c)

from measured executed nodes.

The future topology predictor learns:

p(Z
future
	​

∣X).

At inference the two are composed:

p(R
future
	​

∣X,c)=
Z
∑
	​

p(R∣Z,c)p(Z∣X).

This is likely the cleanest interface for your current data regime.

Case C — no node-level measurements exist

Then there is no real supervised future-resource target.

You may construct:

p
~
	​

(R∣Z,c)

from empirical profile tables.

But this is weak supervision.

I would rank candidate labels:

measured node resource>resource-equivalence profile>role/family profile>legacy predictor output
	​


for scientific evidential value.

Resource-equivalence profiles

If true measurements are unavailable, these are the only surrogate I would seriously consider.

For each training partition only:

F
R∣Z,c
train
	​

.

Do not collapse them to a median. Preserve something like:

(q
.05
	​

,q
.50
	​

,q
.95
	​

)

or an empirical distribution.

Leakage protection

Every fold/profile must be constructed only from its training videos.

Never use:

P_dev validation/test measurements to build train profiles;

the 40-video frozen holdout;

scheduler videos;

final sealed test;

whole-dataset statistics.

If a profile is used to evaluate itself, cross-fitting should be used.

Role/family medians

Keep these strictly as a baseline.

[INFERENCE]

They should not become the new training label, because your existing result already says they throw away exactly the dimensions needed for identifiability.

4. Resource loss contract

I would make the new resource head distributional rather than point-only.

Recent ICML work explicitly motivates quantile regression/prediction intervals because point predictions do not communicate the magnitude and frequency of errors needed for consequential decisions. 
Proceedings of Machine Learning Research

For each node predict:

R
i
	​

={T
i
	​

,U
i
	​

,M
i
	​

},

where:

T: runtime;

U: GPU/load measure;

M: peak memory.

For each resource predict:

τ={0.05,0.50,0.95}.

Tensor contract:

resource_quantiles:
[B, Lmax, Kmax, 3 resources, 3 quantiles]

Thus each node exposes:

q
^
	​

.05
	​

,
q
^
	​

.50
	​

,
q
^
	​

.95
	​

.

Use a monotonic train-only transformation for heavy-tailed quantities:

g
T
	​

(T)=log(1+T)
g
M
	​

(M)=log(1+M)

and no logarithm for a naturally bounded load measure.

Then robust-scale each resource from training statistics only.

Resource loss:

L
R
	​

=
3
1
	​

r∈{T,U,M}
∑
	​

3
1
	​

τ∈{.05,.50,.95}
∑
	​

meanρ
τ
	​

(y
r
′
	​

−
q
^
	​

r,τ
	​

).

This is ordinary pinball/quantile loss conceptually.

Matching

This detail matters a lot.

Resource truth must not participate in Hungarian matching.

First determine the layer-wise identity-free assignment from predicted semantic/resource-signature attributes:

π
l
∗
	​

=Hungarian(C
semantic
	​

).

Then use exactly that assignment for:

content CE;

resource regression;

content metrics;

resource metrics.

Do not rematch nodes to minimize runtime error.

Otherwise the evaluator could pair:

“the slow predicted node” with “the slow true node”

even though the semantic predictor actually assigned it incorrectly.

That would inflate resource performance.

5. Complete loss contract

I recommend four normalized task groups.

Structure
L
S
	​

=
2
1
	​

CE(
L
^
,L)+
2
1
	​

L
1
	​

l≤L
∑
	​

CE(
W
^
l
	​

,W
l
	​

).

No separate node-existence loss if existence is deterministically derived from L,W.

Future content/resource signature

First perform one layer-wise matching.

Then:

L
C
	​

=
∣A∣
1
	​

a∈A
∑
	​

CE
a
	​


over defined attributes.

I would explicitly separate generic semantics from the resource signature internally, but retain one averaged content objective.

For example:

A={node_type,family,model_resource_class,input_scale,execution_mode}.

Do not force raw_action into the resource path unless ablation demonstrates it contributes incremental information.

Resource

As above:

L
R
	​

=mean quantile loss.
Immediate behavior anchor

Retain the previously validated:

L
B
	​

=
2
1
	​

L
next−role
	​

+
2
1
	​

L
execute−family
	​

.

No layer-1 consistency term in this experiment.

Total loss

My preregistered first choice:

L=
3+λ
B
	​

L
S
	​

+L
C
	​

+L
R
	​

+λ
B
	​

L
B
	​

	​

.
	​


Use the same λ
B
	​

 that already succeeded in the previous topology+behavior joint experiment.

If that earlier experiment did not freeze one, preregister:

λ
B
	​

=1.

So the first experiment effectively has equal task-group weighting after within-task normalization.

6. Equal weighting vs Kendall-style uncertainty weighting vs PCGrad-style gradient surgery
Equal weighting

[INFERENCE — recommended first experiment]

Use equal normalized task-group weights first.

Not because equal weighting is universally best, but because it answers the architecture question cleanly.

All four losses have already been normalized internally, so scale mismatch is substantially reduced.

If J3 fails, you then know the problem is genuinely interaction/optimization—not that you simultaneously changed the architecture and weighting rule.

[VERIFIED] Recent multitask optimization research continues to show that naive average-loss optimization can under-optimize some tasks and that gradient conflicts can materially affect performance. FAMO proposes dynamic task weighting for balanced loss reduction; FairGrad similarly starts from the problem of conflicting gradients and task imbalance. 
NeurIPS 会议论文集
+1

Kendall-style uncertainty weighting

The original method predates your requested citation window, so I am intentionally not citing the original paper.

[INFERENCE] Consider this only if, after normalization:

one loss systematically dominates numerically;

regression labels genuinely have very different noise levels;

optimization curves indicate certain tasks are being under/overweighted.

I would be particularly cautious here.

If your resource labels are noisy, uncertainty weighting may learn:

“resource is uncertain, therefore downweight resource.”

But resource improvement is the primary goal of this experiment.

That may produce an apparently well-balanced model that simply gives up on the task you care about.

Recent ICML work also cautions that adaptive MTL weighting can misbehave when tasks have different irreducible/noise levels, because high-loss/noisy tasks distort balancing rules. 
Proceedings of Machine Learning Research

So do not use uncertainty weighting in J1/J2/J3.

PCGrad-style gradient surgery

Again, the original PCGrad paper falls outside your requested source window, so I am not using it as a source.

Use the method family only after observing:

∇L
R
⊤
	​

∇L
S
	​

<0

or

∇L
R
⊤
	​

∇L
C
	​

<0

consistently, and a practical pattern such as:

J3 improves resource but violates the structure/content non-inferiority gate.

Recent CVPR/ICML multitask work continues to document gradient conflict and compare gradient manipulation with weighting strategies. 
Open Access CVF
+1

Interpretation

If J3 gives:

resource ↑;

structure ↓;

behavior ↓;

then gradient surgery is a logical later experiment.

If J3 gives:

resource unchanged;

topology unchanged;

there is nothing for PCGrad to solve.

The problem is probably inadequate resource information.

7. The diagnostic that separates “bad resource head” from “bad topology features”

This should be preregistered.

Oracle-resource-signature diagnostic

Train/evaluate the same resource mapping with:

Z
i
	​

=Z
i
GT
	​


rather than predicted q(Z
i
	​

).

The resource head still sees legitimate runtime context.

Call it:

R
oracle-sig
	​

.

This is not deployable.

Then compare four conditions:

Observation	Diagnosis
Oracle signature poor	resource head/profile/context/labels are inadequate
Oracle strong, J1 poor	resource mapping needs semantic identity
Oracle strong, J2 ≫ J1	topology/content features are informative
Oracle strong, J2 ≈ J1	predicted signature is too noisy/uninformative
J3 > J2	resource loss improves the semantic representation
J3 < J2 while resource training loss falls	end-to-end coupling is overfitting/distorting semantics

This one diagnostic can prevent a lot of wasted architectural work.

Vidur is useful precedent conceptually: its strong resource predictions rely on operator profiling and predictive modeling tied to actual inference configurations, rather than expecting a high-level predictor to recover missing execution physics. 
MLSys 会议录

8. What metrics actually establish RESOURCE improvement?

Do not make runtime MAE the sole endpoint.

Llumnix explicitly motivates dynamic scheduling by heterogeneous/unpredictable resource requirements and poor tail latency, reinforcing the importance of looking beyond averages. 
USENIX

Primary resource score

I recommend a single preregistered aggregate proper score:

ResourceQScore=
3
1
	​

r
∑
	​

2
1
	​

[Pinball
.50
	​

(r)+Pinball
.95
	​

(r)]
	​


using the training-normalized resource scales.

Lower is better.

Why only .50 and .95 in the primary metric?

Because they directly correspond to:

typical prediction quality;

scheduling-risk tail.

Keep .05 for interval coverage.

Primary success criterion

For stage A→B:

Δ=ResourceQScore
B
	​

−ResourceQScore
A
	​

.

Require the upper bound of a paired 95% source-video bootstrap CI to be:

CI
upper
	​

(Δ)<0
	​

.

Thus you need evidence of a real improvement, not merely a better mean over three seeds.

9. Resource safety/diagnostic metrics

Report these separately.

Runtime

q50 raw-ms MAE;

q95 pinball loss;

q95 exceedance:

P(T>
q
^
	​

.95,T
	​

);

mean error among the top true-runtime decile;

relative/absolute underprediction in that decile.

The q95 exceedance should ideally be near 5%.

Memory

Report:

q50 MAE;

q95 pinball;

q95 exceedance;

false-safe rate.

For a memory budget B:

FalseSafe=P(
q
^
	​

.95,M
	​

≤B∧M>B).

The budgets must be hardware/profile budgets defined independently of scheduler evaluation.

Do not tune them on scheduler data.

If you do not have an independent memory budget in the predictor dataset, do not fabricate false-safe; report upper-tail exceedance instead.

Load

Report:

q50 MAE;

q95 pinball;

interval coverage;

tail-decile MAE.

Intervals

For

[
q
^
	​

.05
	​

,
q
^
	​

.95
	​

],

report:

Coverage
90
	​

=P(q
.05
	​

≤y≤q
.95
	​

)

and average interval width.

Recent ICML/AISTATS work directly supports prediction intervals/quantile approaches when point predictions are insufficient and studies uncertainty-aware/conformal interval construction. 
Proceedings of Machine Learning Research
+1

For distribution shift specifically, Ai & Ren 2024 develop robust conformal inference under covariate and conditional shifts; this supports treating holdout coverage as a distinct uncertainty question rather than assuming source-domain calibration survives. 
Proceedings of Machine Learning Research

10. Structure/content/behavior preservation gates

The new objective is resource improvement, but it cannot simply destroy the already-good topology predictor.

I would pre-register hard non-inferiority gates.

Structure

Primary:

StructureExact.

Require:

CI
lower
95%
	​

(StructureExact
new
	​

−StructureExact
reference
	​

)>−0.02
	​


absolute.

Also report layer/width/node MAE, but do not create four independent primary gates.

Status

[UNVERIFIED] The 2-percentage-point margin is a proposed practical margin for your experiment, not a literature constant.

Future content/resource signature

Because resource conditioning relies on it, require:

CI
lower
	​

(SignatureMacroF1
new
	​

−SignatureMacroF1
reference
	​

)>−0.02.

This is more important now than generic raw-action F1.

Immediate next-step behavior

Keep the previously discussed margins:

δ
role
	​

=0.01
	​


and

δ
family
	​

=0.02
	​

.

Thus both must satisfy:

CI
lower
	​

(Δ)>−δ.

Again these are [UNVERIFIED practical preregistration margins], not universal thresholds.

Topology uncertainty

Because downstream scheduling consumes top-3 scenarios, also monitor:

raw structure NLL;

true-shape top-3 coverage;

top-3 probability mass.

I would add a soft gate:

Top3Coverage
new
	​

≥Top3Coverage
reference
	​

−0.02.

A model that improves hard shape accuracy while destroying scenario probabilities can still become worse for uncertainty-aware scheduling.

11. Data protocol

Keep the existing 300-video development division fixed.

Do not repartition because the previous experiments have already established that boundary.

Use:

existing P_dev train → fitting;

existing P_dev validation → early stopping/hyperparameter decisions;

existing P_dev test → formal J1/J2/J3 comparison;

same 3 fixed seeds for every stage.

All stage comparisons must use the same seeds and same examples.

Then select the architecture.

Only after the architecture, loss contract, resource signature and inference rule are frozen should the 40-video holdout be evaluated once.

Do not use the 40-video results to:

choose J2 vs J3;

alter resource classes;

alter quantiles;

modify loss weights;

add gradient surgery;

calibrate intervals.

Scheduler data and sealed final-test data stay completely outside this loop.

12. Calibration rule for this experiment

I would keep this experiment intentionally simple.

Categorical structure/content

Use raw probabilities.

Report:

NLL;

Brier;

ECE;

top-3 coverage.

Do not introduce a new temperature/Dirichlet calibration variable during J1/J2/J3.

You already have evidence that the previous train-fit calibration did not transfer.

Continuous resources

Use native predicted quantiles first.

Report source-validation/test coverage and then frozen-holdout coverage.

Do not conformalize until after the architecture comparison.

[INFERENCE]

Otherwise you will be unable to distinguish:

“the resource model became better”

from

“a post-hoc interval adjustment became wider.”

Conformal/robust interval methods are good candidates for a later uncertainty experiment, especially because recent work explicitly handles distribution shift. 
Proceedings of Machine Learning Research

13. Stage promotion rules

I would make the formal decision tree very simple.

J1 → J2

Promote J2 if:

ResourceQScore significantly improves versus J1, or at minimum shows a stable practically meaningful effect;

structure NI passes;

behavior NI passes;

resource tail/safety metrics do not deteriorate.

Interpretation:

predicted semantic/resource-signature information adds resource value.

J2 → J3

Promote J3 only if:

measured resource labels, not merely proxy labels, supervise the relevant future-resource loss;

ResourceQScore improves over J2;

structure NI passes;

resource-signature content NI passes;

next-step NI passes;

probability/interval calibration does not materially collapse.

If J3 improves resource by 3% but loses 6 pp of structure exact-match:

reject J3.

That is negative transfer, not a successful joint predictor.

14. Failure-mode and stop table
Failure	Observable symptom	Interpretation	Pre-registered response
Proxy circularity	excellent loss against profile labels, no gain against independent measurements	model learned the lookup table	stop true-resource claim
Resource head inadequate	oracle-signature resource prediction remains poor	target/context/resource model is bottleneck	stop joint topology changes
Topology features uninformative	oracle strong, J2≈J1	predicted signature carries little resource information	stop before J3 unless signature task changes
Negative transfer	J3 resource improves but structure/behavior NI fails	resource gradients damage shared representation	reject J3; retain J2
Cardinality dominates semantics	shape stays strong while signature F1 falls	encoder capacity goes to easy structure task	stop further structural weighting
No benefit from gradient flow	J3≈J2	resource gradients do not improve interface	keep detached interface
Resource-gradient domination	rapid L_R improvement + broad CE deterioration	objective-scale/gradient conflict	do not change weights mid-experiment; record failure
Adaptive weighting hides resource task	total MTL metrics improve while resource stalls	noisy resource objective was downweighted	reject weighting as solution
Tail failure	median MAE improves, q95 error/exceedance worsens	mean-focused improvement unsafe for scheduling	reject candidate
Memory false-safe rise	more predicted-safe nodes actually exceed budget	unsafe uncertainty estimate	reject candidate for hard memory constraints
Calibration drift	F1/MAE improves but NLL/top-3/coverage worsens sharply	scores become less trustworthy	reject uncertainty-aware deployment
Pseudo-label-only J3	future L_R comes only from class profiles	no independent continuous supervision	classify as surrogate experiment only
Holdout-dependent choice	architecture/weights changed after 40-video evaluation	holdout contaminated	invalidate holdout claim
15. How I would interpret possible outcomes

Three results are especially informative.

Result A
OracleSig≫J2>J1

This is the encouraging result.

It means:

resource mapping is learnable;

predicted resource signature helps;

remaining gap is future content/signature accuracy.

Then improving the topology/content interface is worthwhile.

Result B
OracleSig≈J1≈J2

but all are poor.

Then your resource model/data/context is the actual bottleneck.

Changing GRU architecture or graph decoding is unlikely to help.

Result C
J3>J2

on resources but structure/content regress.

This says end-to-end learning is working, but tasks conflict.

Only after this result would I investigate task weighting or gradient surgery.

Recent MTL work gives good reason to treat such conflict as a real optimization issue rather than assuming average loss always finds a good compromise. FAMO, FairGrad, and CVPR 2024 task-priority work all address different versions of this problem. 
NeurIPS 会议论文集
+2
Proceedings of Machine Learning Research
+2

16. What I would actually preregister

The first formal paper-quality experiment would therefore be:

Contract	Fixed choice
Encoder	existing shared causal GRU
Horizon	identity-free H=5
Behavior head	retained auxiliary/local anchor
Structure	layer count + active-layer widths
Content	unordered matched future nodes
Resource signature	execution class + model-resource class + input scale + execution mode, where labels exist
Resource context	execution state known independently at scheduling time
Matching	semantic/signature Hungarian only; resource truth excluded
Resource prediction	q05/q50/q95 runtime/load/memory
Resource loss	normalized pinball
J1	independent resource head
J2	J1 + detached soft resource signature
J3	J2 with resource→content gradient
Loss weights	equal normalized S/C/R groups; existing behavior auxiliary weight frozen
Primary resource metric	ResourceQScore
Structure gate	exact-shape NI, −2 pp
Signature gate	Macro-F1 NI, −2 pp
Behavior gates	role −1 pp; family −2 pp
Tail metrics	q95 pinball + exceedance
Memory safety	false-safe where legitimate budgets exist
Interval metric	90% coverage + width
Categorical calibration	raw NLL/Brier/ECE/top-3 coverage
Seeds	existing fixed 3 seeds
Development	fixed 300-video train/val/test split
Holdout	40 videos, once after model freeze
Scheduler/final test	never used for fitting/selection

My strongest recommendation is actually about labels rather than architecture:

If you can recover real future-node resource truth from later executions within P_dev, do that target-side join. If you cannot, do masked/partially-labelled training using measured current-node resources. Do not promote role/family medians into “ground truth” just to make a fully joint loss possible.

That distinction determines whether your next experiment can legitimately show resource prediction improvement, or only improved consistency with an existing proxy.

The recent evidence supporting this design comes mainly from three directions: Vidur/USHER show that useful resource models are grounded in profiling and execution-relevant configuration rather than coarse semantics; recent MTL work shows that shared objectives can suffer task imbalance and gradient conflicts; and recent uncertainty work supports distributional/interval evaluation rather than point-MAE-only evaluation. 
Proceedings of Machine Learning Research
+3
MLSys 会议录
+3
USENIX
+3

Primary sources: VIDUR — MLSys 2024
 · USHER — OSDI 2024
 · Llumnix — OSDI 2024
 · FairGrad — ICML 2024
 · FAMO — NeurIPS 2023
 · Fine-grained robust conformal inference — ICML 2024
 · Relaxed Quantile Regression — ICML 2024
