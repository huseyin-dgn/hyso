# Web retirement function inventory

A: web-only implementation removed. B: unique/general engineering logic retained independently. C: delegated Core capability already exists and remains unchanged; only its Studio wrapper is removed. This classification describes capabilities, not moving C source into another wrapper.

All ten runtime Python files and five static files were hashed/backed up before removal. The five browser/UI test modules are retired, not counted as Python regressions. Original source is in the external QA before/ snapshot; historical reports remain in the repository.

| File | Function | Class | Resolution |
| --- | --- | --- | --- |
| `hyso/studio/analytics.py` | `graph_diagnosis` | B | model_observations.explain_diagnostic |
| `hyso/studio/analytics.py` | `compare_architectures` | B | architecture_comparison.compare_architectures |
| `hyso/studio/analytics.py` | `presentation` | C | reporting.create_report; presentation wrapper retired |
| `hyso/studio/analytics.py` | `record_metrics` | C | Native report sources/metrics; wrapper retired |
| `hyso/studio/analytics.py` | `conditions` | B | model_observations._conditions |
| `hyso/studio/analytics.py` | `comparability` | B | model_observations._comparability |
| `hyso/studio/analytics.py` | `compare` | C | analysis.compare_models retained binary evidence API |
| `hyso/studio/analytics.py` | `recommendation` | B | model_observations._recommendation |
| `hyso/studio/analytics.py` | `_series` | B | model_observations._series |
| `hyso/studio/analytics.py` | `diagnose` | B | model_observations._diagnose / diagnose_results |
| `hyso/studio/analytics.py` | `native_items` | C | advisor.recommend original native recipes; UI translation wrapper retired |
| `hyso/studio/analytics.py` | `normalize` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/catalog.py` | `component_catalog` | C | ModelRegistry.list/get and graph_ops metadata unchanged; UI copy retired |
| `hyso/studio/catalog.py` | `capabilities` | C | analysis.feature_catalog and existing CLI help; endpoint inventory historical |
| `hyso/studio/jobs.py` | `JobManager.__init__` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/jobs.py` | `JobManager.preflight` | B | Core preflight retained; bounded CPU admission -> model_checks |
| `hyso/studio/jobs.py` | `JobManager.submit` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/jobs.py` | `JobManager.cancel` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/jobs.py` | `JobManager._run` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/jobs.py` | `JobManager.execute` | B | Core delegates retained; CPU test/count orchestration -> model_checks |
| `hyso/studio/jobs.py` | `JobManager._test` | B | model_checks.run_model_checks |
| `hyso/studio/jobs.py` | `JobManager._test.observe` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/jobs.py` | `JobManager._retain` | C | TrainingResult.log_to / ExperimentRun.add_artifact / record_result; no automatic association |
| `hyso/studio/jobs.py` | `JobManager.close` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/knowledge.py` | `bilingual` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/knowledge.py` | `example_config` | B | examples/models/component-configs.json (Core JSON; no UI metadata) |
| `hyso/studio/knowledge.py` | `enrich` | B | English engineering notes -> component-engineering-reference.md; UI enrichment retired |
| `hyso/studio/runtime.py` | `validate_cpu_options` | C | TrainingConfig/TrainingSession native loader/target validation; browser JSON limits retired |
| `hyso/studio/runtime.py` | `validate_cpu_options.check` | C | Native TensorDataset/Trainer input contracts; no web feature-list conversion |
| `hyso/studio/runtime.py` | `GraphTensorOutput.__init__` | B | models/output.py: GraphTensorOutput (validated, shared parameters) |
| `hyso/studio/runtime.py` | `GraphTensorOutput.forward` | B | models/output.py: GraphTensorOutput.forward |
| `hyso/studio/runtime.py` | `graph_mapping` | B | models/output.py: explicit output/task guard |
| `hyso/studio/runtime.py` | `tensor_features` | C | Caller torch.Tensor/TensorDataset; no implicit CPU conversion |
| `hyso/studio/server.py` | `contract` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/server.py` | `parse_json` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/server.py` | `parse_json.pairs` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/server.py` | `parse_json.invalid` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/server.py` | `parse_json.check` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/server.py` | `StudioServer.server_bind` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/server.py` | `StudioServer.__init__` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/server.py` | `StudioServer.process_request` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/server.py` | `StudioServer.process_request_thread` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/server.py` | `StudioServer.server_close` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/server.py` | `Handler.setup` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/server.py` | `Handler.log_message` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/server.py` | `Handler.send` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/server.py` | `Handler.authorized` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/server.py` | `Handler.do_OPTIONS` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/server.py` | `Handler.do_GET` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/server.py` | `Handler.do_POST` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/server.py` | `Handler.handle_request` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/service.py` | `fields` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/service.py` | `checked_text` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/service.py` | `model_hash` | B | model_checks.config_digest (exact supplied JSON identity) |
| `hyso/studio/service.py` | `model_hash.numbers` | A | Retired JS integral-number normalization; Python uses exact JSON content identity |
| `hyso/studio/service.py` | `utc_timestamp` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/service.py` | `record_context` | C | AnalysisContext / EvidenceScope explicit caller declarations |
| `hyso/studio/service.py` | `restricted` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/service.py` | `restricted.nested_selector` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/service.py` | `restore_record` | C | Native constructors; experiments.workflow and analysis._native readers |
| `hyso/studio/service.py` | `junit_result` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/service.py` | `StudioService.__init__` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/service.py` | `StudioService.preview` | C | ModelBuilder/GraphModelBuilder.preflight; disabled-node UI state retired |
| `hyso/studio/service.py` | `StudioService.save_model` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/service.py` | `StudioService.snippet` | C | Builder.from_config/build public API and Python examples |
| `hyso/studio/service.py` | `StudioService.resolve_model` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/service.py` | `StudioService.publish_version` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/service.py` | `StudioService.architecture_comparison` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/service.py` | `StudioService.overview` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/service.py` | `StudioService.overview.recent` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/service.py` | `StudioService.job_records` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/service.py` | `StudioService.settings` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/service.py` | `StudioService.import_record` | C | ExperimentManager artifacts/workflow results; no Studio-store migration |
| `hyso/studio/service.py` | `StudioService.records` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/service.py` | `StudioService.comparison` | C | analysis.compare_models / compare_model_structures; binary APIs unchanged |
| `hyso/studio/service.py` | `StudioService.recommendations` | B | model_observations.suggest_model_corrections + native Advisor |
| `hyso/studio/service.py` | `StudioService.report` | C | reporting.create_report / report_experiment |
| `hyso/studio/service.py` | `StudioService.experiments` | C | ExperimentManager unchanged |
| `hyso/studio/service.py` | `StudioService.close` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/store.py` | `canonical` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/store.py` | `digest` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/store.py` | `now` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/store.py` | `Store.__init__` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/store.py` | `Store.list` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/store.py` | `Store.get` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/store.py` | `Store.put` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/store.py` | `Store.close` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/store.py` | `Store.publish` | A | HTTP/browser, UI persistence, task queue or display contract; retire |
| `hyso/studio/__main__.py` | `main` | A | HTTP/browser, UI persistence, task queue or display contract; retire |

All functions and event handlers in `static/app.js`, `editor.js`, `presentation.js`, plus index.html/app.css, are A: DOM, visual editor, browser import/export, interaction history, localization or HTTP transport. Architectural source-of-truth remains Core. Each complete static source was inspected in the preceding accepted work and retained in the before/ snapshot; all are removed from runtime/distribution.

## Twelve required engineering capabilities

| Capability | Python resolution |
| --- | --- |
| Graph named-output adaptation | `models.output.GraphTensorOutput`; explicit output/task, shared parameters |
| Structural validation | Existing Builder preflight, with thin schema dispatch |
| Forward / backward / strict roundtrip | `model_checks.run_model_checks`; source copy, synthetic CPU fixture |
| Exact parameters | `model_checks.count_parameters`; bounded explicit construction, no forward |
| Architecture differences | `architecture_comparison.compare_architectures`; existing structure comparison unchanged |
| Diagnostic interpretation / candidate | `model_observations.explain_diagnostic/suggest_model_corrections`; copies only |
| Training association | Existing TrainingResult.log_to / ExperimentManager workflow |
| Evaluation / benchmark association | Existing artifact ownership/provenance + typed EvidenceRecord |
| Binary evidence comparison | Existing analysis.compare_models; no replacement |
| Evidence-linked advice | Existing advisor.recommend/advise + retained pure threshold observations |

No queue, HTTP endpoint, Studio storage schema, or browser technology is required by the new public entry points. Historical SQLite remains external and unmodified. JUnit import and mutable draft/version/position/settings storage are web product features, not new Core algorithms.

### hyso/studio/static/app.js

A ? `api`, `notice`, `safe`, `element`, `button`, `field`, `input`, `payload`, `snapshot`, `checkpoint`, `changed`, `edit`, `replace`, `schedulePreview`, `validate`, `translatePage`, `renderNavigation`, `showPage`, `workspaceGuide`, `technicalDetails`, `signature`, `modelReference`, `referenceFilters`, `modelChoice`, `selectedChoice`, `lookupChoice`, `modelLabel`, `chooseModel`, `modelSelector`, `renderContext`, `copyModel`, `saveDraft`, `renderOverview`, `renderHome`, `renderPalette`, `uniqueId`, `defaultParam`, `addNode`, `transformPlane`, `positionOf`, `drawGraph`, `drawEdges`, `parameterField`, `renderInspector`, `renderOutputs`, `renderDiagnostics`, `diagnosticText`, `focusGraphNode`, `renderKnowledge`, `showKnowledge`, `pythonCode`, `executionRequest`, `startOperation`, `testWorkflow`, `jobWizard`, `options`, `metricName`, `metricValue`, `renderMetricTable`, `renderRecord`, `renderRecords`, `refreshJobs`, `payloadForJob`, `comparisonTable`, `renderComparison`, `update`, `lineChart`, `importRecord`, `analyze`, `renderReportSections`, `renderSettings`, `download`, `textDialog`, `refreshModels`, `fit`, `reportMetrics`, `displayReport`. All anonymous DOM/Promise/event callbacks in this file are also A.

### hyso/studio/static/editor.js

A ? `renameNode`, `importDraft`, `resolveNodeSettings`, `parseJson`, `encodeJson`, `emit`, `recordUploadBody`. All anonymous DOM/Promise/event callbacks in this file are also A.

### hyso/studio/static/presentation.js

A ? `valueText`, `constraintText`, `modelSupport`, `recommendationGroups`, `newClassifier`. All anonymous DOM/Promise/event callbacks in this file are also A.
