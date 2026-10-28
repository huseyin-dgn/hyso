# Evidence-based experiment planning — Stage 3B.3B

`advise(source, request=...)` now returns read-only, deterministic decision plans
alongside the existing native observations and record-local GoalEvaluations.
Nothing is executed, authorized, reserved, moved, trained or benchmarked.

## Contract and API

The existing Finding contract is extended through registered `PlannedFinding`,
which carries a typed immutable `DecisionPlan`. AdvisorReport accepts the additive
`planning_only` status and authority `native_planning_rules_v1; not_authorized`.
Existing JSON formats/defaults remain readable; old observation-only and goal
reports can still be reproduced by `validate_advisor_report(result, source)`.
Source analysis, serializers, native bindings and compare_evidence are unchanged.

```python
from hyso.analysis import UnifiedAnalysisReport
from hyso.advisor import AdvisorRequest, advise, validate_advisor_report

source = UnifiedAnalysisReport.from_json("existing-analysis.json")
report = advise(source, request=AdvisorRequest(priority="memory"))
report.to_json("proposed-plans.json")
report.to_markdown("proposed-plans.md")
validate_advisor_report(report, source)
```

All plans explicitly contain decision/rationale, source records, related goals,
budget references, missing evidence and known risks. Experiment candidates also
require a changed parameter, exactly one changed variable, fixed conditions,
remeasurement metrics and acceptance conditions. There is no executable patch,
factory reference, device handle, numeric benefit estimate or confidence score.

`authorization=not_authorized`, `readiness=blocked`, `cost=unknown`,
`expected_effect=unknown`, `verified_effect=not_measured` are enforced values in
v1, including on JSON reload. Creating a plan is not approving a run. Budget IDs
are references to requests, not proof of runtime enforcement. Deployment limits
are not experiment budgets. Candidate support describes its baseline rationale,
not the success, safety or deployability of the unexecuted modification.

## Closed rule set — all version 1

| Rule ID | Decision and required native basis |
|---|---|
| plan.latency-stage | Review retained P03/P04 stage attribution, using eligible E2E/forward observations without subtracting unrelated captures. No causal diagnosis. |
| plan.batch-throughput | A CPU FP32/FP64 completed-throughput record needs native device, precision, batch, shape, model-state and protocol bindings plus explicit recorded backend. Otherwise review its retained recipe. |
| plan.precision-quality | Review individual validated P08 points and retained P28 quality links. No precision change is selected as a runnable experiment. |
| plan.memory-domain | Inspect retained P10 RSS/P11 logical accounting separately; no RSS-minus-logical attribution or invented memory recipe. |
| plan.accuracy-preservation | Validate P26 baseline/candidate/cohort relations and caller criteria before claiming quality preservation. No numeric tolerance comes from priority. |
| plan.hard-goal-consistency | Resolve an empty hard-goal interval before selecting any optimization candidate. Local goal outcomes remain intact. |

An eligible throughput baseline can support a **conditional batch experiment**.
Its proposed variable is `input.batch_size`; finite new points must be chosen
explicitly by the caller. No batch values or gains are invented. Model state,
device, precision, backend, preprocessing/input semantics and timing method must
remain fixed apart from the selected batch axis. Latency, completed throughput,
memory, quality and bounded failure/stability outcomes must be checked again.

The candidate remains blocked pending retained P06 review, explicit points and
experiment budgets, candidate capability, native associations, linked memory/
quality acceptance, independent confirmation and the future ComparisonContract.
Missing these prerequisites is never presented as completed acceptance. Positive
tests demonstrate a valid conditional proposal, not a successful optimization.

Retained P06 is outside resolver v1: its sweep results are not fabricated or
converted into new batch recommendations. It is reviewed before considering a
new sweep. A missing backend or model-state association prevents even a batch
candidate and produces an evidence review instead. Risky GPU experiment paths
are outside v1. A proposed CPU batch experiment is not a generic capability claim
for arbitrary changed shapes, models or batches.

P08 point precision/batch contexts remain separate. Captured BF16 and FP32 timing
values remain observations; no numerical speedup, winner certification or generic
BF16 acceleration recommendation is produced. Unsupported INT8 remains so.
Precision accuracy effects require linked cohort evidence; current P29 deltas do
not supply independent timing confirmation.

## Goal and priority behavior

Automatic considers only rules with relevant retained evidence. Zero candidates
is valid. It creates no user goal or target. Advanced requests use unchanged
GoalEvaluation results: only supported observed_missed objectives can motivate
related optimization plans. Met goals do not cause a new target. Unknown or
unsupported goals retain their existing clarification actions. Contradictory hard
goals block experiment selection. Different captures meeting different goals do
not establish a jointly acceptable deployment.

Priority changes presentation order only. The dependency order is retained review
first, association validation second, separately authorized measurement third,
conditional experiment last. Domain preferences break ties: memory emphasizes
memory review; preserve_accuracy emphasizes precision/quality prerequisites;
speed emphasizes latency/throughput. Candidate membership, IDs, evidence and hard
acceptance do not change with priority. Rule IDs and content-based finding IDs
are deterministic for the same source/request.

Current legacy MEASURE_NEXT findings remain intact, ordered alongside the new
plans. Every new plan states which decision obtaining its evidence enables. It
does not indiscriminately rerun missing P01–P35 features. A review action may
explain a subsequent capture dependency, but does not authorize it or invent its
cost/capability.

## Outputs and limits

The same AdvisorReport exports a compact Markdown view and complete JSON.
Markdown summarizes key metrics and shows plan rationale/change/verification/
risks; JSON retains every original observation and full provenance/plan fields.
Review findings use insufficient_evidence for the decision despite their cited
individual metrics being valid. A conditional candidate uses supported only for
its local baseline rationale and always remains blocked/not_authorized.

Planner v1 produces no HYPOTHESIS, DIAGNOSIS or VERIFIED_IMPROVEMENT. Their absence
is deliberate: stage waits, utilization, apparent deltas or logical bytes do not
establish a causal mechanism. No finding automatically changes category. P18
qualified-roof limits, P29 noncertified improvement, P31 noncertified 7200-second
acceptance and B4 OPEN remain unchanged.

NOT IMPLEMENTED: executor, authorization grant, model/config mutation, numeric
benefit/cost/confidence estimation, independent effect certification, general
batch/precision/model ComparisonContract, deployment or quality-drop acceptance
and actual runtime resource enforcement. Existing provenance gaps cannot be
repaired by caller declarations or by equal hash strings.

Focused and retained-evidence checks:

```text
python -B -m unittest discover -s tests -p "test_advisor*.py" -v
python -B tools/integration/advisor_planning_acceptance.py --source <analysis.json> --observations-baseline <3B.2.json> --goals-baseline <3B.3A.json> --output <new-directory>
```

Positive fixtures add clearly labeled test-only producer metadata to retained
samples. They do not execute a workload or invent a real performance acceptance.
Negative fixtures test missing associations, wrong model/batch/precision/domain,
GPU claims, unsupported INT8, hard-goal conflict, unknown quality/memory/budget,
fake speedups, invented diagnosis and unauthorized status changes.
