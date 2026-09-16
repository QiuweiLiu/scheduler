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
timestamp: "2026-09-06T02:20+08:00"
status: "valid"
---

# Research Report

## 5. assistant
Research report: topology → resource interface for future-DAG GPU scheduling
Executive Summary

[INFERENCE — primary recommendation] Your present bottleneck should be treated as an interface identifiability problem, not primarily as a topology-prediction problem. The topology predictor appears able to answer “how many future nodes and how are they structurally arranged?”, but role + action_family does not contain enough information to answer “what GPU resources will this concrete future operation consume?”

I recommend changing the conceptual interface from:

future node→(role,family)→(runtime,load,memory)

to:

future node→P(resource signature)→P(R∣resource signature,runtime context)
	​


where the topology predictor predicts only intrinsic, future-node properties, while the scheduler/resource layer supplies execution-time context.

A useful split is:

z
i
intrinsic
	​

=(exec_class, model_resource_class, input_scale, execution_mode)

versus

c
t
runtime
	​

=(GPU, residency, cache, batching, contention, parallelism).

Then:

p(R
i
	​

∣X,c
t
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
t
	​

),

with R
i
	​

=(runtime,load,memory,setup/loading).

Do not require the topology predictor to infer cache/warm state or GPU placement. Those are states of the scheduling system, not intrinsic properties of the future DAG node.

This decomposition is strongly consistent with recent systems literature. Vidur explicitly models LLM performance using model/configuration/workload information and operator profiling; vLLM shows memory scales dynamically with sequence/model/decoding characteristics; ServerlessLLM makes checkpoint locality part of scheduling; DistServe separates execution phase and hardware/parallelism choices; Vulcan composes pipeline structure with explicit operator configurations, placement and profiles rather than semantic task names alone. 
USENIX
+4
MLSys 会议录
+4
DOI
+4

The most important practical conclusion is therefore:

Keep the good identity-free DAG structure predictor. Do not force it to recover exact semantic node identities just to satisfy the resource predictor. Instead, introduce a smaller resource-equivalence signature and marginalize resource estimates over its uncertainty.

1. What recent top systems work actually does
Key finding

[VERIFIED] I did not find a paper in the surveyed 2023–2026 OSDI/SOSP/NSDI/MLSys/EuroSys/TPDS/TC literature that performs exactly your pipeline:

causal prefix → unknown future agent DAG → identity-free predicted future nodes → per-node GPU resource prediction.

The closest papers start with a known/constructed operator graph or known request/model, then attach resource estimates using operator/model/configuration/profile information. This distinction matters: your project is combining two research problems that those systems normally keep separate.

Work	Venue	Resource interface actually used	Relevance
Vulcan	NSDI 2024	Pipeline operator + configuration + placement + profiling	Closest graph/pipeline analogue
Vidur	MLSys 2024	Model specification + request dimensions + phase + batch/config + hardware profiles	Strongest runtime-model precedent
vLLM/PagedAttention	SOSP 2023	Model/sequence/decoding-dependent KV state	Strong memory evidence
DistServe	OSDI 2024	Prefill/decode phase + workload + GPU allocation/parallelism	Execution phase matters
ServerlessLLM	OSDI 2024	Model checkpoint + locality/storage hierarchy + GPU	Cold/warm/loading context
Llumnix	OSDI 2024	Dynamic request state and in-memory state	Runtime state matters
Cilantro	OSDI 2023	Learned resource→performance mapping + load shift + confidence bounds	Uncertainty-aware resource scheduling
Mudi	EuroSys 2025	Profiled latency/interference + batching/resource scaling	Contention matters
Aker	TC 2025	Concrete kernel pair/configuration → duration predictor	Low-level implementation determines runtime
GraphAGILE	TPDS 2023	GNN model specification + graph metadata → execution instructions	Graph structure alone is insufficient
Strata	OSDI 2026	Context/KV cache state + loading latency + batches	Very recent cache-state evidence

Vulcan, for example, jointly searches pipeline construction, configuration and physical placement and uses profiling to estimate latency/resource effects. It does not try to map a semantic operator label like “detect” directly to resource cost. 
USENIX

Vulcan — NSDI 2024

Vidur is particularly relevant. It combines experimental profiling with predictive modeling of LLM operators and exposes configuration variables including model, GPU, tensor/pipeline parallelism, request lengths and scheduler configuration. Its profiling machinery explicitly varies prefill chunk size, KV-cache size and batch size. 
MLSys 会议录
+2
GitHub
+2

Vidur — MLSys 2024

The 2023 SOSP vLLM/PagedAttention paper likewise shows that KV-cache memory grows and shrinks dynamically and that gains depend on sequence length, model size and decoding algorithm. 
DOI

vLLM / PagedAttention — SOSP 2023

ServerlessLLM adds another variable that your reduced node contract cannot encode: checkpoint locality. Its scheduler explicitly chooses a server based on checkpoint locality to reduce startup time; its official documentation states that loading time depends on model size and hardware bandwidth. 
USENIX
+1

ServerlessLLM — OSDI 2024

And the newest evidence strengthens rather than weakens this conclusion: OSDI 2026 Strata reports that cache-loading latency and cache state have to enter scheduling decisions for long-context LLM serving. 
USENIX

Strata — OSDI 2026

What that implies for your interface

[INFERENCE] Do not ask the topology model to reproduce everything that affects runtime. Instead, define an explicit boundary:

predict what is unknowable about the future node
	​


and

look up what is already known about the runtime system
	​

.

That separation is more defensible scientifically and easier to train.

2. Minimal resource-identifiable node contract

There is an important distinction between an exact model identity and a resource-equivalence class.

You probably do not need the former.

I would define the smallest candidate contract as:

z
i
	​

=(e
i
	​

,m
i
	​

,x
i
	​

,ϕ
i
	​

)

where:

e
i
	​

: execution/node class — e.g. detector inference, VLM inference, LLM inference, CPU transform, API call;

m
i
	​

: model/resource class — models sharing approximately the same weight footprint and compute characteristics may share a class;

x
i
	​

: input-work scale — tokens, frames, resolution, tensor-size bucket, batch-equivalent work;

ϕ
i
	​

: execution mode if relevant — e.g. prefill/decode, precision/backend, algorithm variant.

Then the scheduler supplies:

c
t
	​

=(GPU type,resident models,cache state,available memory,current load,co−runners,parallelism).

The estimator becomes:

p(R
i
	​

∣z
i
	​

,c
t
	​

).
Why these fields?

[VERIFIED] Recent systems work repeatedly conditions performance on this kind of information:

DistServe gives prefill and decode different allocation/parallelism because they have different characteristics. 
USENIX

Vidur explicitly incorporates model/configuration, batch and request dimensions. 
MLSys 会议录
+1

vLLM's memory behavior depends on sequence/model/decoding properties. 
DOI

ServerlessLLM makes model locality and hardware bandwidth relevant to startup latency. 
USENIX
+1

EuroSys 2025 Mudi explicitly profiles latency under resource interference rather than treating the workload label as sufficient. 
EuroSys 2025
+1

TC 2025 Aker's duration predictor operates on concrete fused-kernel configurations. 
IEEE Xplore

Is exact model ID required?

[INFERENCE — important qualification] No, not provably.

Suppose

Z=(role,family)

and omitted information is

U=(model,input,hardware,cache,…).

For resource target Y,

Var(Y∣Z)=E[Var(Y∣Z,U)∣Z]+Var(E[Y∣Z,U]∣Z).

If two executions with the same Z have different conditional resource means because U differs, then

Var(E[Y∣Z,U]∣Z)>0,

and some irreducible error is created by discarding U.

So the correct statement is not:

“model ID is mathematically mandatory.”

It is:

Whatever variables separate resource-distinct executions must either be included in the node signature or supplied as runtime context.

If all detect nodes used exactly the same detector, resolution, GPU and cache situation, then role+family could work perfectly well.

Once those conditions vary, it cannot generally identify resource usage.

I found no recent top-systems evidence suggesting that a coarse semantic category remains sufficient under heterogeneous models, request sizes and cache/hardware states. The literature instead consistently introduces those execution-specific variables.

3. Three possible topology → resource interfaces
Design I — coarse conditional aggregation

Topology predictor produces something like:

N
execute
	​

,N
plan
	​

,N
aggregate
	​

,...

or layer-wise prototype histograms, and a direct estimator predicts aggregate future demand:

p(C
future
	​

∣shape,prototype counts).

[INFERENCE] Advantages: small sample complexity, resilient to bad per-node family identification, and appropriate if the scheduler needs only “how much work is coming.”

[INFERENCE] Weakness: poor at nonlinear effects such as model loading, GPU-specific placement, memory feasibility and tails. A median runtime lookup is a particularly weak version because mixtures of fast and slow implementations collapse into one number.

This approach probably explains why a coarse proxy can cover many nodes while still severely underestimating expensive tails.

Design II — hard full per-node expansion

For every predicted node:

z
^
i
	​

=argmaxp(z
i
	​

∣X)

then

R
^
i
	​

=f(
z
^
i
	​

,c
t
	​

).

All nodes are passed explicitly into the scheduler.

[INFERENCE] Advantages: ideal when placement/cache/memory decisions depend on individual operations.

[INFERENCE] Weakness: it converts semantic classification errors directly into potentially enormous resource errors.

With future-family prediction still weak, this would be risky. A single misclassification from a light model to a heavy-model class—or vice versa—can dominate the entire H=5 horizon.

I would not use hard top-1 node expansion as your next mainline interface.

Design III — uncertainty-preserving resource-signature expansion

This is my recommendation.

For structural scenario s, predicted node i has:

q
is
	​

(z)=P(z
i
	​

=z∣X,s).

Then:

p(R
i
	​

∣X,s,c
t
	​

)=
z
∑
	​

q
is
	​

(z)p(R
i
	​

∣z,c
t
	​

).

For topological scenarios:

p(R
future
	​

∣X,c
t
	​

)=
s
∑
	​

P(s∣X)p(R
future
	​

∣s,c
t
	​

).

That means the scheduler never has to pretend:

“future node #3 definitely uses family X/model Y.”

Instead it gets:

“node #3 has 0.55 probability of resource class A, 0.30 B, 0.15 C; here is its resulting resource distribution.”

Why this is a particularly good fit here

[INFERENCE]

You already have the attractive combination:

short H=5 horizon;

strong structural prediction;

top-3 structural scenarios;

weaker content identification.

That makes marginalization computationally conceptually manageable and removes much of the need for hard identity prediction.

It also converts family/model uncertainty into resource uncertainty, which is actually the quantity the scheduler cares about.

4. What should be aggregated into “future cost”?

I would avoid compressing everything immediately into one scalar.

The resource interface should expose something like:

C
future
	​

=[E(W
GPU
	​

),Q
0.95
	​

(W
GPU
	​

),P(memory violation),E(T
setup
	​

),Q
0.95
	​

(T
setup
	​

),load profile].

Where W
GPU
	​

 is GPU service demand rather than wall-clock DAG completion time.

Runtime

A safe scheduler-independent statistic is:

W=
i
∑
	​

T
i
	​

.

This is total service work. It does not mean the DAG will take W milliseconds.

Structural critical path

With dependency edges:

CP=
path p
max
	​

i∈p
∑
	​

T
i
	​


can be useful as a lower-bound-style structural feature.

Without explicit edges, layer counts/widths alone generally cannot uniquely reconstruct it.

Memory

Memory must be treated differently.

You should not compute:

memory
l
	​

=
i∈layer
l
	​

∑
	​

memory
i
	​


and call that required GPU memory.

Layer width is DAG-layer cardinality, not simultaneous physical execution. Actual concurrency is chosen by the scheduler.

Instead evaluate candidate mappings such as:

P(M
i
	​

>M
g
free
	​

∣z
i
	​

,c
t
	​

)

for node i on GPU g.

This distinction is central to preserving the logical-DAG/physical-scheduling boundary.

Tail risk

[VERIFIED/INFERENCE] Systems literature heavily emphasizes tail latency/SLOs, and Cilantro explicitly incorporates confidence bounds in scheduling; Llumnix targets unpredictable resource requirements and tail latency. 
USENIX
+1

Therefore I would expose at least expected demand and a high quantile/CVaR-like future cost rather than only a median/mean.

The exact risk objective is an UNVERIFIED design choice, but ignoring the tail is poorly aligned with the evidence and your observed error pattern.

5. Calibration under holdout shift

There are actually two separate calibration problems:

calibrating topology/content probabilities;

calibrating continuous resource uncertainty.

They should not be conflated.

Temperature scaling

[VERIFIED] Temperature scaling can work across some real distribution shifts—an ICML 2024 study found strong results for VLMs calibrated in one domain and tested in another. But this is empirical evidence in a particular model family, not a universal robustness guarantee. 
Proceedings of Machine Learning Research

Tu et al., ICML 2024 calibration study

[INFERENCE] Therefore the fact that temperature scaling degrades your frozen holdout is not surprising enough to warrant “fixing” it. Raw probabilities are the correct incumbent if they give better pre-registered held-out NLL.

One caveat: if “train-fit temperature” literally means the calibrator was fitted on the same examples used to train the predictor, that is not the ideal calibration protocol. Calibration should be estimated on held-out development examples.

Dirichlet calibration

[VERIFIED] Dirichlet calibration is a more expressive multiclass calibration mapping than a single temperature and improved log loss/Brier/ECE across the datasets studied in the original NeurIPS paper. 
NeurIPS 会议录

Dirichlet calibration — NeurIPS 2019

But:

[VERIFIED] that paper does not establish distribution-shift robustness.

[INFERENCE] With modest video-level data and several heads, its extra flexibility creates more opportunity for calibration overfit. I would not promote it to the next main experiment merely because temperature scaling failed.

Conformal methods

This is more promising for the resource interface.

[VERIFIED] Recent work provides conformal prediction approaches explicitly addressing distribution shifts. Ai & Ren's ICML 2024 method distinguishes covariate shift from conditional shift and combines reweighting with distributionally robust protection. Other work constructs prediction regions under covariate shift, and Gibbs & Candès study online conformal inference under arbitrary time-varying shifts. 
Proceedings of Machine Learning Research
+2
OUP Academic
+2

Fine-grained robust conformal inference — ICML 2024

However:

Conformal prediction is primarily a coverage/uncertainty-set mechanism, not a magic probability-calibration transform.

So its natural role here is:

runtime
i
	​

→[L
i
	​

,U
i
	​

]

or

memory
i
	​

→[L
i
	​

,U
i
	​

]

rather than trying to transform every node-family probability.

[INFERENCE] For your experiment, all conformal/calibration choices should be developed solely by pseudo-shifts/video-level splits inside the development pool. The frozen holdout should remain evaluation-only.

The multi-head problem

There is another issue that head-level calibration does not solve.

Suppose:

P(s)=P(L)P(W∣L)
i
∏
	​

P(z
i
	​

∣⋯).

Even if every individual head is calibrated, it does not follow that the resulting whole-DAG scenario probability is calibrated.

[INFERENCE] Therefore, if the scheduler consumes the top-3 scenarios, evaluate:

top-1 and top-3 true-scenario coverage;

scenario-level NLL/Brier if scenario probabilities are defined;

probability mass retained by top-3;

calibration of the actual scheduler-visible scenario probability.

I regard this as more important than applying another per-head temperature.

6. Public benchmarks: is there an existing one?
Exact future-DAG + resource prediction benchmark

[VERIFIED SEARCH FINDING, not an absolute nonexistence proof]

I found no public benchmark among the surveyed recent systems/agent literature that simultaneously provides:

a causal agent prefix;

the unknown future DAG as prediction target;

per-future-node measured GPU runtime/load/memory;

an evaluation specifically for forecasting those resources.

There are, however, three neighboring benchmark families.

WfCommons — closest graph + resource benchmark

WfFormat represents task graphs/dependencies together with per-task runtime, I/O sizes, memory, energy and machine information. WfBench can also generate realistic DAG structures with configurable GPU/memory/CPU work. 
WfCommons
+1

WfCommons documentation

This is excellent precedent for your resource contract format, but it does not ask a model to predict an unseen future workflow from a causal prefix.

Vidur-Bench — closest LLM resource benchmark

Vidur explicitly ships a benchmark suite for LLM inference workloads and evaluates end-to-end behavior from detailed operator-performance models. 
MLSys 会议录

It has resource realism but no agent future DAG.

Agent benchmarks

AgentBench and VideoWebArena provide agent/environment trajectories and video-agent tasks, respectively, but their published benchmark objectives concern agent reasoning/task success rather than per-node GPU resource forecasting. 
ICLR 会议记录
+1

PeakBench — unusually relevant new preprint

A very recent August 25, 2026 preprint, PeakBench, explicitly combines executable multi-tool workflows, execution-grounded dependencies and measured resource profiles and separates logical planning from resource-constrained scheduling. 
arXiv

PeakBench preprint

[VERIFIED source / UNVERIFIED peer-review status] This is the closest agent-specific public benchmark I found, but it is currently an arXiv preprint and still does not perform your exact causal-prefix future-DAG/resource joint forecasting problem.

That benchmark gap may itself help motivate your work.

7. Implications for your next experiment

I would change the research question from:

“Can future family prediction become accurate enough to feed the resource predictor?”

to:

“What is the smallest identity-free future-node resource signature that makes runtime/load/memory conditionally predictable?”

That is a substantially cleaner experiment.

Experiment R0 — identifiability upper bound

[INFERENCE]

On development data only, give the resource estimator the true resource-relevant signature for each node:

(exec_class,model_resource_class,input_scale,execution_mode)

plus known runtime context.

This is an oracle-interface diagnostic, not deployable prediction.

The question is:

If the resource predictor knows everything we reasonably believe it should know, does runtime/load/memory become predictable?

If no, stop expanding the topology predictor. The problem is then probably profiling noise, missing context or the resource model itself.

This experiment is extremely important because otherwise you could spend weeks improving future content labels that fundamentally cannot fix resource error.

Experiment R1 — contract ablation

Compare progressively richer contracts, still only on predictor-development videos:

C
0
	​

=(role,family)
C
1
	​

=C
0
	​

+exec/node class
C
2
	​

=C
1
	​

+model resource class
C
3
	​

=C
2
	​

+input scale
C
4
	​

=C
3
	​

+execution mode.

Runtime context such as actual candidate GPU/cache/residency should be supplied rather than predicted whenever causally available at scheduling time.

Primary question

Does adding a field significantly reduce:

Var(R∣contract)

and, particularly, tail underprediction?

If model_resource_class gives a dramatic reduction and action_family gives almost none, that is direct empirical evidence for redesigning the topology/content interface.

Experiment R2 — aggregation comparison

Freeze the best contract and compare:

A0: existing coarse role/family median proxy.

A1: signature-conditioned aggregate distribution.

A2: hard per-node top-1 expansion.

A3: uncertainty-marginalized per-node expansion across predicted resource signatures and top-3 DAG scenarios.

My predicted ordering is:

A3>A1>A0

with A2 potentially unstable because content errors get hardened.

[UNVERIFIED] That ordering is a hypothesis, not something supported directly by an existing paper.

Experiment R3 — uncertainty

For runtime/memory, compare point estimates against quantile/conformal-style intervals developed entirely within the development pool.

Important resource-side metrics should include:

mean/median absolute error;

high-runtime quantile error;

severe-underprediction rate;

interval coverage;

interval width;

for memory, false-safe rate: predicting a node fits when it actually exceeds the candidate budget.

I would give memory false-safe errors much more weight than symmetric MAE when determining whether the estimate is safe enough to expose as a hard scheduling constraint.

8. Concrete recommended interface

If I had to freeze a conceptual contract today, I would use:

TopologyScenario
s
	​

=(p
s
	​

,L
s
	​

,W
s
	​

,{Q
is
	​

(Z)}
i=1
N
s
	​

	​

,E
s
	​

?)

where:

Z=(exec_class,model_resource_class,input_scale,mode).

The resource service receives:

TopologyScenario
s
	​

+RuntimeContext
t
	​


and returns for every node/GPU candidate:

ResourceEstimate
ig
	​

={E[T],Q
90
	​

[T],Q
95
	​

[T],E[M],Q
95
	​

[M],E[load],E[setup]}.

The future aggregator can then produce:

FutureResourceSummary={E[GPUwork],Q
95
	​

[GPUwork],memory risk,setup/load risk,critical path estimate?}.

[INFERENCE] This is materially better than exposing just one future_cost scalar because it keeps the predictor/resource boundary independent of the scheduler's final objective weights.

It also preserves your identity-free requirement: model_resource_class can mean “resource-equivalent VLM-medium” rather than “specific future node ID 847 using model X.”

9. Stop criteria

I would pre-register the following.

Stop expanding future-node semantics if oracle resource signatures do not substantially improve resource prediction. Then semantic prediction is not the limiting factor.

Do not use hard per-node expansion if predicted-signature marginalization fails to outperform the coarse aggregate or if error tails become substantially worse.

Do not use predicted memory as a hard feasibility condition if its false-safe rate remains material.

Do not add explicit DAG edges solely for resource estimation unless edge information improves a scheduler-independent target such as critical-path/resource-envelope estimation. Width alone should remain a structural cardinality variable.

Do not introduce Dirichlet/conformal/etc. merely because temperature scaling failed. Require improvement on pre-registered development pseudo-holdouts before allowing any calibration method to touch the frozen holdout.

Do not change the resource contract based on scheduler outcomes. Predictor/resource-interface selection should finish before scheduler data enters the loop.

10. Uncertainties

[UNVERIFIED] It remains unknown whether your dataset contains enough variation within each proposed model_resource_class × input_scale combination to estimate useful conditional distributions. If not, the theoretically correct contract may still be statistically too sparse.

[UNVERIFIED] model_resource_class may be sufficient without exact model identity, but the appropriate grouping has to be empirically demonstrated. Two nominally similar models can have very different memory behavior; Vidur's broader findings and vLLM's memory results make this plausible, but they do not define the right classes for your particular CV/video-agent stack. 
MLSys 会议录
+1

[UNVERIFIED] The degraded calibrated holdout NLL does not by itself prove distribution shift. It can also come from a small calibration sample, calibrator overfit, different class balance or noisy holdout estimates.

[INFERENCE] If future content F1 remains low even for the reduced resource-signature categories, an aggregate distribution conditioned on structure may ultimately outperform any attempt to explicitly identify every future node. That would not invalidate the topology predictor—it would simply mean topology is predictable at a higher abstraction level than content.

Overall conclusion

The literature changes how I would frame your current result.

[VERIFIED] Modern ML/LLM systems do not generally estimate resource cost from high-level task semantics alone. They repeatedly use model/operator implementation, request/input dimensions, hardware/configuration, cache/locality state and profiling. 
Cloud and Distributed Systems Lab
+4
USENIX
+4
MLSys 会议录
+4

[INFERENCE] So the weak future-family result does not imply the successful H=5 topology predictor is a dead end. It instead suggests the topology→resource boundary is currently asking the content head to solve the wrong identification problem.

My preferred architecture is therefore:

causal prefix→top-3 future DAGs→P(resource signature per node)
	​


then, separately,

P(resource signature)+current execution context→P(runtime,load,memory)
	​


followed by uncertainty-aware aggregation.

If R0 shows that exec class + resource-model class + input scale + runtime context collapses the current resource error substantially, I would consider that much stronger evidence for the next architecture revision than trying to push generic action-family F1 from roughly 0.4 to some arbitrarily higher value.

Primary sources

Vulcan — NSDI 2024

Vidur — MLSys 2024

vLLM / PagedAttention — SOSP 2023

DistServe — OSDI 2024

ServerlessLLM — OSDI 2024

Llumnix — OSDI 2024

Cilantro — OSDI 2023

Strata — OSDI 2026

Aker — IEEE Transactions on Computers 2025

GraphAGILE — IEEE TPDS 2023

EuroSys 2025 accepted papers

WfCommons

Robust conformal inference — ICML 2024

VLM calibration under shift — ICML 2024

PeakBench — August 2026 preprint
