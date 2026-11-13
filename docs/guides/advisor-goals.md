# Record-local goal evaluation — Stage 3B.3A

Stage 3B.3B adds [conditional experiment planning](advisor-planning.md) while
preserving these GoalEvaluation outcomes. Legacy goals_evaluated reports remain
readable and reproducible; new advise() results use planning_only.

`advise(report, request=...)` now evaluates explicit goals against **one eligible
native capture at a time**. No training, inference, device query, optimization,
factory import, runtime budget enforcement or independent improvement is added.
Automatic and priority-only reports retain the 3B.2 JSON and Markdown behavior.

## Explicit goals and captured contexts

```python
from hyso.analysis import UnifiedAnalysisReport
from hyso.advisor import (
    AdvisorRequest, Metric, NativeMetricResolver, UserGoal, advise, goal_context,
)

source = UnifiedAnalysisReport.from_json("existing-analysis.json")
resolver = NativeMetricResolver(source)
# The caller explicitly selects an existing record, not a model factory.
scope, protocol = goal_context(resolver, "exact-P01-record-id", Metric.RESIDENT_FORWARD_P95)
goal = UserGoal(
    "p95", Metric.RESIDENT_FORWARD_P95, "<=", 1.0, "ms",
    scope=scope, protocol=protocol, constraint="hard",
)
result = advise(source, request=AdvisorRequest([goal], priority="speed"))
result.to_json("goals.json")
result.to_markdown("goals.md")
```

`goal_context()` is an explicit convenience call returning existing MetricScope
and EvaluationProtocol objects. It creates no target, does not run a workload
and does not repair missing facts. The caller must choose the actual threshold.
Without scope/protocol completeness a goal remains `needs_scope`.

The helper retains record model/workload/dataset labels as selectors, with their
original declared/missing authorities still present in the EvidenceRef. Matching
these labels is not native association or deployment proof. No report-context
metadata is used to fill absent record fields.

The minimum supported goal identities are:

| UserGoal metric | Native quantity | Scope |
|---|---|---|
| RESIDENT_FORWARD_P95 | P01 exact quantile .95 | Resident inference capture |
| E2E_P95 | P03 read-start to post-completion P95 | Serial pipeline capture; not P01 |
| IMAGE_THROUGHPUT | P02 completed images / completion-window seconds | Recorded batch and protocol; not camera FPS |
| ACCURACY | P26 test accuracy fraction | Recorded test cohort; not macro F1 |
| PROCESS_RSS_PEAK | P10 sampled process-RSS peak | Explicit request phase `process`; not inference-only RAM or CUDA |
| LOGICAL_TENSOR_BYTES | P11 parameter+buffer logical bytes | Explicit request phase `static`; not runtime/physical memory |

The additive request phases `process` and `static` prevent assigning unrecorded
training/inference phases to process-wide or static measurements. They do not
change any source EvidenceScope or binding. Asking for inference/training-only
RSS with a phase-unbound process observation remains insufficient evidence.

P01 backend settings can be explicitly selected through
`native-settings-sha256:<canonical-settings-hash>`. This is a literal fingerprint
of recorded settings, **not** a guessed backend name or a repaired native binding.
For example the retained MNIST P02 payload lacks an explicit backend selector;
its throughput is observable in Automatic mode, but a complete backend-bound
goal cannot be accepted by silently inserting `pytorch_eager`.

The goal protocol uses exact ID/version, window, evaluator and averaging. Its
`sample_policy` is canonical JSON containing the native metric identity, native
protocol SHA and recorded sample count. This explicitly distinguishes P95 from
P50 and captures the native timing/sample policy. Free text such as "1000 samples"
is not parsed as an equivalent protocol. Description is human explanation only.
This initial engine supports exact captured profiles; it does not implement a
general language for looser or minimum-sample protocol requirements.

## Independent result axes

- Invalid mathematical input raises AdvisorError at UserGoal construction;
  invalid values never become accepted goals. Tiny positive latency is valid.
- `input_validity=needs_scope` preserves missing request fields.
- `evidence_support=supported` means one matching native record supports a local
  threshold observation. It is not joint model/deployment acceptance.
- `insufficient_evidence` covers missing metrics, incomplete/incompatible scope
  or protocol and ambiguous multiple captures. The engine never selects the best
  capture, averages unrelated values or fills absent measurements with zero.
- `unsupported_capability` covers unsupported goal metrics, estimated goal
  assessment and baseline-relative comparisons requiring ComparisonContract.
- `observed_met` / `observed_missed` compare the goal with one eligible retained
  value only. `unknown` means no admissible single-record decision.
- `feasibility` remains **unknown**, including when a record meets the threshold.
  No future device, deployment, possible optimization or global impossibility
  is inferred from a historical observation.

Only declared unit scales are converted: s/ms and byte/GB/GiB use exact rational
arithmetic. Accuracy accepts fraction [0,1]; percent is a rendering only. No
FPS/images/s, RSS/VRAM, static/runtime memory or metric/statistic conversion occurs.
Native-derived empirical statistics are admissible; estimated evidence is not
silently treated as a measured observation.

Each explicitly supplied scope field must match the capture selector, including
phase, device, precision, batch, shape and all supplied identity/clock fields.
Nonnative associations remain listed as unproven. Multiple matching records
produce unknown even if they happen to show the same value. A stronger paired
comparison still uses unchanged 3A compare_evidence; no new general comparison
mechanism is part of this stage.

Static `RequestValidation` remains separate. Same-scope hard P95 <=1 ms and
>=2 ms forms an empty interval. Each goal can still have its own local observed
outcome; their conjunction is not accepted. Different batch/device/phase/metric
goals do not become one contradictory variable. High throughput with low RAM
and zero accuracy drop is not declared impossible without appropriate evidence.
Soft/hard intent and priority remain independent; priority does not alter goals,
threshold decisions, evidence support, provenance or future feasibility.

## Serialization and reports

AdvisorReport adds `evaluation_status="goals_evaluated"` and authority
`native_goal_rules_v1; record_local_only`. All existing fields/defaults and old
JSON remain supported. New reports retain complete GoalEvaluation entries,
source EvidenceRefs, reasons and `record-local-goal@1` provenance. Source report
and native serializers are unchanged. `validate_advisor_report(result, source)`
reproduces goal decisions against the source; rehashed invented outcomes fail.
Historical 3B.2 observations_only Advanced reports are reproduced using their
original observation-only behavior, rather than silently reclassifying them.

Markdown shows the requested target, native value, evidence record/path, local
outcome and provenance limits. Conditional MEASURE_NEXT entries request scope
clarification or suitable native evidence; they do not launch any operation.
No overall PASS score or deployment certificate is generated.

NOT IMPLEMENTED: resource-budget assessment/enforcement, goal optimization,
EXPERIMENT_CANDIDATE rules, independent improvement, relative accuracy-drop
acceptance, FPS/energy/VRAM goals, confidence scores, executors and automatic
model/config changes. P08 observations remain isolated point descriptions.
P18/P29/P31 acceptance limits and B4 OPEN remain unchanged.

Focused tests and retained-evidence acceptance:

```text
python -B -m unittest discover -s tests -p "test_advisor*.py" -v
python -B tools/integration/advisor_goal_acceptance.py --source <analysis.json> --baseline-advisor <3B.2-advisor.json> --output <new-directory>
```
