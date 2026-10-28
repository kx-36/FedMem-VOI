# FedMem-VOI: synthetic reference primitives

Designated repository: [FedMem-VOI](https://github.com/kx-36/FedMem-VOI), for the
scoped reference example. This local candidate does not attest upload completion.

This is a **small prospective-method example**, not the empirical FedMem-VOI
learner, an ALFWorld adapter, or a benchmark reproduction. No benchmark
experiments for the proposed method are claimed. Every demo/test input is
hand-written synthetic data. No network/model calls or dependencies are needed.

Python 3.10+:

```sh
python3 demo.py
python3 -m unittest discover -s tests -v
```

The implementation covers temperature-one masked softmax, known uniform feasible
logging probabilities, deterministic cost guards/masks, the two-stage sequential
DR arithmetic in V8 Eq. (10), and direct paired Unknown identification endpoints.
The forced-Local branch returns the **real observed local outcome** and retains
probability-one dummy assignments and zero acquisition costs. Native failure 0
is observed; Unknown has no numeric training label. Assigned transport failures
are retained. Withholding does not erase sunk acquisition charges. No sender Q
or realized outcome is interpreted as individual uplift.

`procedure_decision` implements the prospective V9 shared-cohort gate and
method-specific technical-fit routing as a **pure decision function**. Freeze a
nonempty slot/client/whole-block-role registry, finite collection boundary and
matched learner list before collection. Validate the entire returned registry
once at that boundary. Missing/NotRun slots or required assigned outcomes,
costs, histories, masks, probabilities or versions select ordinary Local for all
matched learners, before any fit. Actual transport failures with observed native
outcomes and explicit unresolved source symbols remain valid. No counterfactual
or unqueried source response is required. Zero feedback explicitly rejects and
needs a separately declared algorithm.

On an eligible cohort, the pre-fit decision permits fitting all rows. Before
evaluation, each learner's predeclared technical flag independently selects its
frozen actor or Local. A valid iterate at its fixed finite limit is valid
completion. Technical failure discards the entire partial actor; no retry,
replacement or partial checkpoint selection is authorized. Once evaluation
starts, this function refuses routing: runtime failures remain assigned records,
not permission for retrospective Local reruns. Fixed no-learning controls are
outside this matched-learner gate.

The caller enforces the single frozen boundary, immutable cohort snapshot,
predeclared finite fit/technical-validity rules and policy binding before any
native evaluation. This library implements neither fitting nor the ordinary
Local executor. All planned evaluation tasks/repetitions and actual native
outcomes/Unknown remain; coincident fallback policies do not fabricate equal
observations. All spent feedback/training/traffic costs survive fallback,
including missing costs as missing/bounded; ordinary Local adds zero external
acquisition bytes but retains normal inference costs. No input records are
mutated or discarded. `cohort_decision` remains a low-level supplied-row check;
use the registered procedure gate for prospective qualification.

**Completeness acceptance does not establish conditional DR identification.**
With fair action A, Y=A, and observation only when A=1, a one-row cohort passes
only at A=1. A deterministic target choosing 1 with zero critics has original
propensity score 2 on accepted cohorts, although target reward is 1. The original
complete-data identity remains an independently fixed-policy evaluation claim;
the eligible-cohort score is a declared training objective, with no conditional
unbiasedness/gradient claim. Primary direct procedure evaluation retains every
allocated repetition, including fallback and Unknown. Tests include this finite
counterexample.

This is an explicit-input arithmetic interface, not an operational logging
schema. Stage maps represent a history-specific mask/policy/critic snapshot;
callers own lawful histories, immutable payload versions, action semantics,
status-dependent mask reconstruction and complete cost ledgers. Actual
serialization, receipt validation and permission enforcement are absent. The
cost unit must be consistent across input costs, caps and multiplier. Uniform
logger probabilities are checked against the known design, never estimated.
Underflow that would erase feasible softmax support raises an error.

Sequential DR assumes complete outcomes, common guards/masks, sequential support
and exchangeability/consistency, stable transition laws and independently fixed
policies/nuisances for evaluation. Tests verify arithmetic identities in finite
synthetic worlds; they cannot validate these scientific assumptions, model
quality, individual treatment effects, independence or generalization. This
release neither fits critics nor constructs cross-fitting folds. Direct paired
bounds retain all explicit assigned tasks with declared client weights and are
**identification bounds, not sampling confidence intervals**. No IPW is applied
to direct evaluation.

See `SCOPE.md` and `PROVENANCE.md` before interpreting or redistributing this
candidate. New original example code is MIT licensed; excluded legacy material is not relicensed. Private verification
records and the release archive are local-only and excluded from the archive.

For a real federated implementation, validate each client's assigned registry
locally and combine only bounded eligibility/control flags to enforce the common
gate. Raw feedback, histories, local critics and residual heads remain local;
only the declared shared parameters and necessary control flags cross clients.
Count control traffic in the training communication ledger, separately from
runtime acquisition traffic. This example is a single-process arithmetic and
routing library, not a distributed implementation or a claim of raw-feedback
export permission. Its in-memory fixtures do not specify a transport protocol.
