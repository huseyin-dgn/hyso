# Retained result reporting

`hyso.reporting` presents existing results through a versioned
`hyso.presentation_report` contract. The report keeps source snapshots, original
statuses, units, scopes, chart data and existing Advisor presentation output.
Use the existing result objects as inputs.

```python
from hyso.reporting import create_report, feature_explanations

def export_results(output_dir, training_result, unified_report, comparison):
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = create_report(
        training_result, unified_report, comparison,
        language="tr", level="technical", title="HYSo sonuçları")
    report.to_json(output_dir / "report.json")
    report.to_markdown(output_dir / "report.md")
    report.to_html(output_dir / "report.html")
    return report
```

`create_report()` accepts TrainingResult, UnifiedAnalysisReport,
ModelComparisonResult and AdvisorReport. EvidenceIndex and AnalysisExecutionResult
also work with an explicit `context=AnalysisContext(...)`. Analysis uses the
existing unified report and Advisor APIs. Execution outcomes remain visible in
their own status table, including unsupported, failed and skipped rows. The
P01–P35 evidence table retains the original report's native-validation statuses.

## Explanation levels and language

| Option | Content |
|---|---|
| `level="beginner"` | Metric definitions, intended use, recorded values and interpretation limits |
| `level="technical"` | Numerical observations, charts, scope, protocol details and existing Advisor suggestions |
| `level="expert"` | Native evidence, source hashes, original protocol, compatibility, uncertainty and limitations |
| `language="tr"` | Turkish editorial explanations and existing Turkish Advisor labels |
| `language="en"` | English editorial explanations and existing English Advisor labels |

Machine-readable source codes, identifiers and original technical diagnostic
text retain their recorded spelling. `feature_explanations(language=...)`
supplies definitions, purpose, units, limits, dependencies, executor coverage and
guide links for every existing P01–P35 catalog entry. Catalog membership only
describes a feature. Actual execution and evidence come from retained records.

Metric names reuse existing typed Advisor labels where available. Remaining
scalar fields keep their native path alongside the feature name. Numeric units
retain the source convention. Training history accuracy uses percent, while
P26 accuracy and F1 use fractions. Comparison differences preserve their native
units, relative-percent changes and separate percentage-point changes.

Existing Advisor actions supply performance-suggestion links. Suggestions remain
attached to their source report and its prerequisites. The reporting layer
preserves existing comparison gates, arithmetic, claims and recommendation rules.
Profiler observations can identify recorded costs for review. Causal bottleneck
diagnoses require the original evidence and compatible scope.

## Charts and numerical boundaries

| Available retained data | HTML chart |
|---|---|
| TrainingResult loss history | Training/validation loss curves |
| TrainingResult accuracy history | Separate percent-accuracy curves |
| UX observer intervals | Separate interval curve in seconds |
| Validated P01 raw forward samples | Distribution histogram for each recorded repeat |
| P02/P22 rate measurements | Completed-throughput bars using recorded images/s |
| P09 allocator, P10 RSS, P11 logical/file bytes | Separate domain-specific bars |
| P19 native epoch durations | Native training-time bars |
| Eligible model comparison differences | Baseline/candidate bars under the retained comparison contract |
| Available structure comparison totals | Registered-count/size bars under their structure snapshot scope |

HTML embeds deterministic Matplotlib SVG charts and its stylesheet. It opens
locally with all graphics in one file. JSON retains exact source snapshots and
chart series. Markdown provides numerical tables, explanation text, protocol
scope and links to existing Advisor suggestions. HTML supplies the visual plots.
On narrow screens, tables and figures retain readable widths inside horizontal
scroll regions. Chart epoch axes use integer ticks.

Missing observations retain null values and their source statuses. Validation
gaps remain gaps in curves. Unknown or unsupported data supplies no bar. An
incompatible comparison supplies no difference chart. A short P01 capture may
still have a raw-sample histogram while unavailable P95/P99 values remain null.
Histogram binning describes those samples and adds no tail-eligibility or
confidence-interval claim.

P09 allocated/reserved counters describe GPU allocator scope. P10 covers process
RSS and recorded HWM, including runtime and allocator effects. P11 covers
logical tensor size, with serialized file bytes in another domain. These bars
remain separate. Native P19 timing and TrainingResult observer intervals have
separate figures and domains. Structure counts describe recorded registrations.
Raw sources retain their original task, device, precision and comparison limits.

## Experiment history

```python
from hyso.reporting import report_experiment

def export_history(manager, experiment_id, path):
    report = report_experiment(manager, experiment_id, language="en", level="expert")
    report.to_html(path)
    return report
```

`report_experiment()` uses the existing history/detail pagination and result
readers. It reads all pages and verifies workflow artifacts through
ExperimentManager.open_result(). Missing, moved, mismatched, unreadable and
invalid artifacts retain their statuses and produce no result section or chart.
Persisted run completion and current file availability are shown separately.
Repeated links to the same artifact are read once. Identical result sections
can reference several artifact IDs.

Schema1 histories show their recorded runs and scalars. Their scalar names and
units remain caller records, with no reconstruction of native measurements.
Schema2 histories can additionally render verified workflow result artifacts.
History reads retain the current schema and append no artifact-check rows.
The existing explicit migration and recovery APIs still apply.

Execution artifacts use the producer run's recorded caller context. The display
uses the experiment ID as its workload label and `unspecified` where task metadata
was never recorded. Original native payloads and bindings remain available.
Reading comparison JSON reruns its existing pure consistency derivation.

## Export contract and examples

`PresentationReport.to_json()`, `.to_markdown()` and `.to_html()` return text and
optionally write to a path. Writes preserve the established atomic/no-clobber
export behavior. Set `overwrite=True` explicitly when replacing an export.
`PresentationReport.from_json()` checks format/version and presentation integrity,
with a 128 MiB bound. Presentation JSON is a separate display format. Native
result readers continue to accept their original formats and validation rules.

HTML escapes source text and uses local links for Advisor suggestions. It has
no external scripts, stylesheets, font downloads or image dependencies. The
presentation digest detects changed exported content. Original evidence readers
and acquisition contracts determine measurement eligibility and authenticity.

See the [examples index](../../examples/README.md) for executable reporting
scripts. `retained_results.py` has a small synthetic stored training fixture and
can also consume saved training, unified-analysis and comparison JSON.
`experiment_history.py` consumes an existing local experiment store. Their tests
forbid model forwards, training, benchmarking, CUDA queries and checkpoint loading.
The default fixture is visibly labeled as synthetic demonstration history.
