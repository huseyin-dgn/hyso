# Stage 13.5.15 Reporting format and API matrix

Date 2026-11-16, Europe/Istanbul. Evidence under
`%LOCALAPPDATA%/hyso-stage13-5-15-20261116` is abbreviated QA.
The complete 37-signature inventory and CLI help text are in `QA/api-inventory.json`.

## Typed input and reopening contracts

| Input | create_report support | Additional requirement / scope |
| --- | --- | --- |
| TrainingResult | Supported | Existing validated retained training/session history |
| UnifiedAnalysisReport | Supported | Existing validated evidence report |
| EvidenceIndex | Supported | Explicit AnalysisContext |
| AnalysisExecutionResult | Supported | Explicit AnalysisContext and existing workflow reader |
| ModelComparisonResult | Supported | Legacy schema1 or explicit strict schema2 |
| GraphEvaluationResult | Supported | Association schema1/2, current process-local view |
| GraphComparisonResult | Supported | Legacy schema1/2 or strict schema3 |
| AdvisorReport | Supported | Existing typed Advisor result |
| RecommendationReport | Supported | Existing typed manual recommendation presentation |
| EvaluationResult | Direct input rejected | Adapt through EvidenceRecord and EvidenceIndex/analysis |
| PresentationReport | Direct input rejected | Use its own export methods or from_json |
| Untyped dict/list, JSON string/path | Direct input rejected | Use the corresponding validated typed reader |

The repository defines TrainingResult in `hyso/training/session.py`.
`TrainingResults` is absent from the inspected runtime API.

`create_report(*results, language='tr', level='technical', title='HYSo', context=None)`
returns PresentationReport. Languages are tr/en, levels beginner/technical/expert,
and title is a nonblank string of at most 512 characters. EvidenceIndex/execution
requires an explicit AnalysisContext. Other metadata comes from the source type.
Duplicate identical sources are deduplicated while first occurrence/order is
retained. Empty presentation input yields an explicit empty presentation.

PresentationReport schema1 has format `hyso.presentation_report`, language, level,
title, catalog, ordered sources, optional experiment, presentation_only=true,
authoritative_evidence=false and content_sha256.

## Export and field projection

| Field | JSON | Markdown / static HTML |
| --- | --- | --- |
| Scalar accuracy and supported averages | Exact source scalar and raw payload | Metric table at every level |
| Macro precision/recall/F1, weighted F1 | Exact source scalar/path | Supported native scalar rows at every level |
| Per-class metrics, supports, confusion matrix | Full raw native evaluation | Complete raw JSON in expert mode |
| Sample count, class names/order, split | Full source/native payload | Available scalar/scope rows, complete expert raw source |
| Evaluation ID and model-state/checkpoint SHA | Full native source | Complete expert raw source and recorded scopes |
| Registered architecture/structure | Explicit structure binding/comparison | Available metadata metrics, complete expert raw source |
| Comparison status and metric differences | Native/strict effective rows | Eligibility rows with preserved nulls and limits |
| Strict policy/version | Raw schema2/schema3 and policy | Expert raw source, scopes/limitations |
| Graph source/receiver/inference binding | Separate graph_evidence and raw association | Separate evidence metric rows and expert raw association |
| Missing/null versus numeric zero | JSON null and numeric zero | Not recorded / Kaydedilmemiş versus 0 |
| Experiment/run/artifact identity and state | Full report_experiment history snapshot | History/status/artifact tables, expert raw experiment |
| Chart sample arrays | Full chart data | HTML embedded SVG, Markdown refers to JSON chart data |
| Presentation integrity SHA | content_sha256 | Footer SHA256 |

Expert Markdown fenced JSON and HTML preformatted source blocks were parsed back
and compared with the source payloads. Every native source scalar in both model
sections was checked by its source path. Twenty scalar checks in the main QA run
match, alongside full native payload equality. Model A has accuracy 0.75 and
model B accuracy 1.0. The compatible accuracy difference is +0.25.

Beginner/technical levels provide their defined projections. Full per-class,
confusion, identity and protocol interchange uses JSON or the expert raw sections.
Presentation formats retain the source's quality/authority scope.

## Files and errors

| API | File behavior |
| --- | --- |
| to_json(path=None, overwrite=False) | Deterministic UTF-8 JSON string, optional atomic file export |
| to_markdown(path=None, overwrite=False) | Markdown string, optional atomic file export |
| to_html(path=None, overwrite=False) | Self-contained HTML/SVG string, optional atomic file export |
| from_json(path) | Bounded 128 MiB schema/digest validation, returns PresentationReport |
| Default export to existing file | FileExistsError, existing bytes retained |
| overwrite=True | Explicit replacement through the existing exporter |
| Missing parent, directory, unsafe path | Existing exporter/path validation error |
| Invalid source type/language/level/title | ReportError |
| Changed stored metric without valid presentation digest | Integrity rejection |
| Wrong schema or authoritative_evidence=true | Reader rejection |

Presentation digest validation verifies its supplied presentation body.
Independently rehashed presentation data retains presentation authority only.
Artifact verification supplies the independently recorded expected byte identity.
Native source validation uses the Evaluation/Comparison readers.

## Escaping correction

Two original public probes failed for titles `Name *literal*` and
`Name _literal_`. Markdown interpreted those delimiters while JSON/HTML retained
the literal title. The `_md` helper now encodes asterisks, backslashes, tildes and
underscore delimiter boundaries. Intraword underscores preserve identifiers
such as `retained_scalar`.

The original two probe bodies match their final ASTs. Initial red, first green
and final regression XML/logs are retained in QA. The initial broad underscore
escape caused one existing pagination/scalar test to fail. The boundary correction
passes that original assertion. HTML escaping, literal special model/dataset
labels, local links and JSON/expert source roundtrip also pass.

## CLI compatibility and backlog

| Actual command | Accepted source/output | Result |
| --- | --- | --- |
| reports render INPUT --format json/markdown/html | Training, analysis, execution, index, comparison, Advisor/recommendation | Supported through closed typed readers |
| reports render strict comparison schema2 | Existing strict policy retained in the report | PASS |
| reports render EvidenceIndex/execution | Requires --context | Missing context exits 2 |
| reports render Graph association/comparison | Dispatcher lacks Graph formats | Exits 2, Python Reporting supports them |
| reports render PresentationReport | Dispatcher lacks presentation format | Use Python from_json/export methods |
| reports experiment --root STORE ID | Existing history, no migration/repair | Supported |
| compare BASELINE CANDIDATE --features ... | Retained execution/UnifiedAnalysisReport, JSON/Markdown | Historical default policy |
| compare --context-policy strict | Option absent | Parser exits 2, Python API selects strict |
| experiments list/history/run/artifacts | Existing store | Read-only history/detail commands |
| experiments verify/open | Existing registered artifact | 0 verified, 3 unavailable/mismatch, 4 unsupported kind |

Source module-entry help, strict report, typed reopening and real Graph-format
rejection were executed. Both installed packages also rendered a saved strict
result through their module entry point. Exact commands and exit codes are in
`QA/cli-entry-proof.json`.

Concrete backlog items are Graph/presentation CLI dispatch, explicit CLI strict
selection, an EvaluationResult adaptation recipe and clearer migration/artifact
ownership guidance. The current CLI preserves its actual supported scope.


## Inspected public signatures

| API | Signature |
| --- | --- |
| `PresentationReport.from_json` | `(path: 'str \| Path') -> 'PresentationReport'` |
| `PresentationReport.to_json` | `(self, path: 'str \| Path \| None' = None, *, overwrite: 'bool' = False) -> 'str'` |
| `PresentationReport.to_markdown` | `(self, path: 'str \| Path \| None' = None, *, overwrite: 'bool' = False) -> 'str'` |
| `PresentationReport.to_html` | `(self, path: 'str \| Path \| None' = None, *, overwrite: 'bool' = False) -> 'str'` |
| `PresentationReport.to_dict` | `(self) -> 'dict[str, Any]'` |
| `ExperimentManager.migrate_store` | `(root_dir)` |
| `ExperimentManager.create_experiment` | `(self, name: 'str', *, request_id: 'str \| None' = None) -> 'dict[str, _typing.Any]'` |
| `ExperimentManager.get_experiment` | `(self, id)` |
| `ExperimentManager.list_experiments` | `(self, *, name: 'str \| None' = None, kind: 'str \| None' = None, status: 'str \| None' = None, after_seq: 'int \| None' = None, limit: 'int' = 100) -> 'list[dict[str, _typing.Any]]'` |
| `ExperimentManager.create_run` | `(self, experiment_id, kind, config, provenance=None, *, redact_paths=(), retry_of_run_id=None, resume_artifact_id=None, request_id=None)` |
| `ExperimentManager.create_workflow_run` | `(self, experiment_id: 'str', stage: 'str', *, model_ids: '_typing.Sequence[str]', architectures: 'dict[str, ModelBuilder] \| None' = None, training_config: 'TrainingConfig \| None' = None, analysis: 'AnalysisDryRun \| None' = None, comparison: 'ModelComparisonResult \| None' = None, dataset: 'dict[str, _typing.Any] \| None' = None, device: 'str \| None' = None, precision: 'str \| None' = None, protocol: 'dict[str, _typing.Any] \| None' = None, request_id: 'str \| None' = None) -> 'ExperimentRun'` |
| `ExperimentManager.get_run` | `(self, id)` |
| `ExperimentManager.get_run_details` | `(self, run_id: 'str', *, after_metric_seq: 'int \| None' = None, after_artifact_seq: 'int \| None' = None, limit: 'int' = 100) -> 'dict[str, _typing.Any]'` |
| `ExperimentManager.list_runs` | `(self, experiment_id=None, *, kind=None, status=None, after_seq=None, limit=100)` |
| `ExperimentManager.get_experiment_history` | `(self, experiment_id: 'str', *, after_seq: 'int \| None' = None, limit: 'int' = 100) -> 'dict[str, _typing.Any]'` |
| `ExperimentManager.get_artifact` | `(self, artifact_id)` |
| `ExperimentManager.list_artifacts` | `(self, run_id=None, *, after_seq=None, limit=100)` |
| `ExperimentManager.link_artifact` | `(self, run_id, artifact_id, *, role='input')` |
| `ExperimentManager.verify_artifact` | `(self, artifact_id, *, path=None)` |
| `ExperimentManager.open_result` | `(self, artifact_id: 'str', *, path: 'str \| _PathLike[str] \| None' = None) -> 'dict[str, _typing.Any]'` |
| `ExperimentManager.close` | `(self) -> 'None'` |
| `ExperimentManager.recover_interrupted` | `(self, run_id, expected_revision, reason)` |
| `ExperimentManager.reconcile_storage` | `(self, *, limit=1000)` |
| `ExperimentRun.start` | `(self, *, request_id=None)` |
| `ExperimentRun.finish` | `(self, reason='completed', *, request_id=None)` |
| `ExperimentRun.fail` | `(self, code, *, failure_domain='workload', workload_outcome='failed', request_id=None)` |
| `ExperimentRun.interrupt` | `(self, reason='interrupted', *, request_id=None)` |
| `ExperimentRun.record_result` | `(self, result: 'TrainingResult \| AnalysisExecutionResult \| ModelComparisonResult', *, input_artifact_ids: '_typing.Sequence[str]' = ()) -> 'dict[str, _typing.Any]'` |
| `ExperimentRun.add_artifact` | `(self, path, *, kind='checkpoint', ownership='reference', role='output', request_id=None, metadata=None, retain_locator=False, max_size_bytes=1073741824)` |
| `ExperimentRun.log_metric` | `(self, *, name, value, unit, source, epoch=None, step=None, step_kind=None, event_id=None, observed_at_utc=None)` |
| `ExperimentRun.log_metrics` | `(self, records)` |
| `ExperimentRun.read_metrics` | `(self, name=None, *, after_seq=None, limit=1000)` |
| `ExperimentRun.to_dict` | `(self)` |
| `create_report` | `(*results: 'Any', language: 'str' = 'tr', level: 'str' = 'technical', title: 'str' = 'HYSo', context: 'AnalysisContext \| None' = None) -> 'PresentationReport'` |
| `report_experiment` | `(manager: 'ExperimentManager', experiment_id: 'str', *, language: 'str' = 'tr', level: 'str' = 'technical', title: 'str' = 'HYSo') -> 'PresentationReport'` |
| `feature_explanations` | `(*, language: 'str' = 'tr') -> 'list[dict[str, Any]]'` |
| `ExperimentManager` | `(root_dir: 'str \| _PathLike[str]', *, max_metrics_per_run: 'int' = 1000000) -> 'None'` |
