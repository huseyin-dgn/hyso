# Stage 13.5.15 ExperimentManager and artifact matrix

Evidence under `%LOCALAPPDATA%/hyso-stage13-5-15-20261116` is abbreviated QA.
All stores in this acceptance are newly created QA-only stores.

## Public lifecycle contract

| Operation | Actual public API / behavior |
| --- | --- |
| Open/create | ExperimentManager(root_dir, max_metrics_per_run=...) |
| Initial schema | New store uses schema1 |
| Migration | Close handles, migrate_store(root), reopen as schema2 |
| Experiment | create_experiment(name, request_id=None), get/list/history |
| Run | create_run(experiment_id, kind, config, ...), get/list/details |
| Workflow | create_workflow_run(..., stage, model_ids, comparison/training_config/analysis, ...) |
| Run states | created, running, completed, failed, interrupted |
| Transitions | start(), finish(), fail(), interrupt(), revision/ownership checks |
| Typed record | run.record_result(TrainingResult/AnalysisExecutionResult/ModelComparisonResult) |
| Scalar metric | log_metric/log_metrics, finite numeric values and explicit axes |
| Artifact | run.add_artifact(path, kind='checkpoint', ownership='reference', ...) |
| Verify/open | verify_artifact(id, path=None), open_result(id, path=None) |
| Relationships | link_artifact(run_id, artifact_id, role='input') |
| Close/reopen | close(), ExperimentManager(same_root), explicit recovery when required |

EvaluationResult, Graph associations/comparisons and PresentationReport use
ordinary artifact registration. The typed workflow reader whitelist contains
TrainingResult, AnalysisExecutionResult and ModelComparisonResult. Unknown kinds
can verify their bytes and return unsupported from open_result.

## Main measured lifecycle

`examples/reporting/reporting_experiment_acceptance.py` executes two actual
four-sample CPU evaluations, strict/legacy comparisons, three Graph authority
views, a strict Graph incompatibility, reporting and a new store lifecycle.
The fixture TrainingResult is a separately labeled retained training record.
New fit count is zero.

The main final directory is `QA/acceptance-final/`.

| Record | Persisted outcome |
| --- | --- |
| Model A evaluation run | completed, managed evaluation JSON plus checkpoint reference |
| Model B evaluation run | completed, managed evaluation JSON plus checkpoint reference |
| Comparison workflow run | completed, strict schema2 result and explicit two evaluation input links |
| Retained training workflow run | completed, fixture history/result |
| Registered artifacts | 12 |
| Verification before close | 12 verified |
| Verification after reopen | 12 verified |
| Run snapshots after reopen | 4/4 exact equality |
| Typed strict/training reopening | Exact source JSON/payload equality |
| Main report / experiment report | JSON, Markdown and static HTML |

Each evaluation EvidenceRecord reads the actual exported evaluation file with
its expected SHA. Those record artifact hashes equal the corresponding stored
artifact observations. Workflow managed result serialization has its own byte
hash, alongside the typed result's canonical content hash.

## Artifact and record acceptance matrix

| Input/scenario | Real observed contract | Evidence |
| --- | --- | --- |
| Managed copy | Verifies registered managed payload | New acceptance and artifact regressions |
| Reference with retained locator | Verifies current referenced file | New main lifecycle and regressions |
| Reference with redacted locator | unavailable until explicit path supplied | Existing artifact regressions |
| Alternate path supplied to managed copy | Managed destination governs verification | Existing contract, scope documented |
| Changed payload | mismatch, immutable expected SHA retained | Four new ownership/damage cases and regressions |
| Removed payload | missing, result absent | Four new ownership/damage cases and workflow regressions |
| Wrong supplied reference file | mismatch | Artifact regressions / prior accepted behavior |
| Unknown experiment/run/artifact ID | ExperimentError not_found | Three new ID cases |
| Same request ID, same inputs | Same identity, no duplicate ingest | New artifact case and workflow regressions |
| Same request ID, different inputs | request_conflict | New artifact case |
| Same filename, different directories/bytes | Distinct IDs and hashes | New artifact case |
| Empty binary artifact | Size 0 and verified empty-byte hash | New artifact case |
| Relative/absolute source path | Same regular-file path validation | New artifact case and regressions |
| Directory/missing source/invalid ownership/role | Validation/ingest rejection | Existing artifact negative cases |
| Symlink/junction/non-regular file | unsafe/unreadable rejection | Existing native Windows junction and simulated symlink regressions |
| Managed path traversal / invalid checksum | Safety/SQLite constraints reject | Existing artifact regressions |
| Publication collision | Existing file retained, ingest failure | Existing no-clobber regression |
| Mid-ingest source change | source_changed / ingest failure | Existing streaming regression |
| Ingest or commit uncertainty | Explicit incomplete/failed/unknown state | Existing artifact/storage regressions |
| Hash-valid malformed JSON | invalid_result, result=None | Five new typed-artifact cases |
| Wrong typed schema / edited metric / partial accounting | invalid_result | Five new typed-artifact cases |
| Wrong direct result type | typed_workflow_result_required before writes | New preflight case |
| Comparison result/context mismatch | comparison_result_mismatch before writes | New preflight case and workflow regressions |
| Null scalar log value | invalid_metric rejection | New preflight case |
| Empty evaluation presentation | Source null metrics, supports/counts 0 | New Reporting case |
| Created/running/failed run after reopen | Same original state, completed count 0 | Three new state cases |
| Explicit migration and rollback | Original core records/backup retained, schema fencing | Full migration/storage selection |

Run completion is an explicit lifecycle event. Reporting and reopening retain
persisted states. File availability is a separate current observation. A ready
artifact's later disappearance is reported through its availability/verification
status. Recording a result supplies its existing validation and typed scope.
Workflow dataset/device/protocol labels retain declared association authority.
Result matching uses the supported training/plan/comparison tracking checks.

## Transactions, authority and recovery

Run state changes use guarded SQLite transactions. Artifact reservation, staging,
publication, ready state, links and metric logging have their existing separate
transactions. A filesystem and database operation can require explicit recovery.
The selected regressions verify migration rollback, no-clobber publication,
streaming outside a long SQLite transaction and uncertain commit behavior.

Expected artifact identity is immutable. Each verify_artifact observation appends
a check. get_run_details, report_experiment and open_result read without appending
verification events. Artifact SHA establishes its registered bytes. Native
producer/checkpoint/receiver facts retain their respective independent contracts.

Imported Graph association JSON has unverified source/receiver authority and
unavailable binding even after artifact SHA verification. A reopened presentation
retains historical status snapshots with authoritative_evidence=false. Reconstructing
the Graph association uses the actual current process-local view.

Public create/run/artifact requests accept request IDs for idempotency. Explicit
recover_interrupted and reconcile_storage provide the existing recovery paths.
Automatic run completion, workload replay and automatic store migration have
execution count zero in the acceptance workflow.


## Individual new acceptance case IDs

| Pytest ID | Outcome |
| --- | --- |
| `test_reporting_experiment_acceptance::test_report_title_preserves_literal_markdown_delimiters[*]` | PASS |
| `test_reporting_experiment_acceptance::test_report_title_preserves_literal_markdown_delimiters[_]` | PASS |
| `test_reporting_experiment_acceptance::test_expert_formats_retain_complete_native_sources_and_exact_values[tr]` | PASS |
| `test_reporting_experiment_acceptance::test_expert_formats_retain_complete_native_sources_and_exact_values[en]` | PASS |
| `test_reporting_experiment_acceptance::test_empty_evaluation_preserves_null_and_zero` | PASS |
| `test_reporting_experiment_acceptance::test_unsupported_direct_report_inputs_are_rejected[dictionary]` | PASS |
| `test_reporting_experiment_acceptance::test_unsupported_direct_report_inputs_are_rejected[evaluation]` | PASS |
| `test_reporting_experiment_acceptance::test_unsupported_direct_report_inputs_are_rejected[presentation]` | PASS |
| `test_reporting_experiment_acceptance::test_export_no_clobber_and_explicit_overwrite[json]` | PASS |
| `test_reporting_experiment_acceptance::test_export_no_clobber_and_explicit_overwrite[md]` | PASS |
| `test_reporting_experiment_acceptance::test_export_no_clobber_and_explicit_overwrite[html]` | PASS |
| `test_reporting_experiment_acceptance::test_html_and_markdown_special_labels_are_literal` | PASS |
| `test_reporting_experiment_acceptance::test_special_model_identity_is_preserved_in_order_and_escaped` | PASS |
| `test_reporting_experiment_acceptance::test_invalid_presentation_json_rejected[json]` | PASS |
| `test_reporting_experiment_acceptance::test_invalid_presentation_json_rejected[schema]` | PASS |
| `test_reporting_experiment_acceptance::test_invalid_presentation_json_rejected[metric]` | PASS |
| `test_reporting_experiment_acceptance::test_invalid_presentation_json_rejected[authority]` | PASS |
| `test_reporting_experiment_acceptance::test_rehashed_presentation_retains_presentation_authority_only` | PASS |
| `test_reporting_experiment_acceptance::test_strict_and_legacy_reporting_remain_distinct` | PASS |
| `test_reporting_experiment_acceptance::test_graph_three_authority_views_and_second_interpreter` | PASS |
| `test_reporting_experiment_acceptance::test_reopen_never_completes_unfinished_or_failed_run[created]` | PASS |
| `test_reporting_experiment_acceptance::test_reopen_never_completes_unfinished_or_failed_run[running]` | PASS |
| `test_reporting_experiment_acceptance::test_reopen_never_completes_unfinished_or_failed_run[failed]` | PASS |
| `test_reporting_experiment_acceptance::test_artifact_failure_status_and_expected_hash_are_persistent[changed-copy]` | PASS |
| `test_reporting_experiment_acceptance::test_artifact_failure_status_and_expected_hash_are_persistent[changed-reference]` | PASS |
| `test_reporting_experiment_acceptance::test_artifact_failure_status_and_expected_hash_are_persistent[missing-copy]` | PASS |
| `test_reporting_experiment_acceptance::test_artifact_failure_status_and_expected_hash_are_persistent[missing-reference]` | PASS |
| `test_reporting_experiment_acceptance::test_empty_same_named_relative_and_idempotent_artifacts` | PASS |
| `test_reporting_experiment_acceptance::test_unknown_identifiers_fail[experiment]` | PASS |
| `test_reporting_experiment_acceptance::test_unknown_identifiers_fail[run]` | PASS |
| `test_reporting_experiment_acceptance::test_unknown_identifiers_fail[artifact]` | PASS |
| `test_reporting_experiment_acceptance::test_hash_valid_invalid_typed_artifact_is_rejected[json]` | PASS |
| `test_reporting_experiment_acceptance::test_hash_valid_invalid_typed_artifact_is_rejected[schema]` | PASS |
| `test_reporting_experiment_acceptance::test_hash_valid_invalid_typed_artifact_is_rejected[metric]` | PASS |
| `test_reporting_experiment_acceptance::test_hash_valid_invalid_typed_artifact_is_rejected[evaluation]` | PASS |
| `test_reporting_experiment_acceptance::test_hash_valid_invalid_typed_artifact_is_rejected[partial]` | PASS |
| `test_reporting_experiment_acceptance::test_result_type_context_and_null_scalar_fail_before_writes` | PASS |
| `test_reporting_experiment_acceptance::test_cli_renders_saved_strict_result[json]` | PASS |
| `test_reporting_experiment_acceptance::test_cli_renders_saved_strict_result[markdown]` | PASS |
| `test_reporting_experiment_acceptance::test_cli_renders_saved_strict_result[html]` | PASS |
| `test_reporting_experiment_acceptance::test_cli_invalid_inputs_and_missing_features_are_explicit[missing]` | PASS |
| `test_reporting_experiment_acceptance::test_cli_invalid_inputs_and_missing_features_are_explicit[bad-json]` | PASS |
| `test_reporting_experiment_acceptance::test_cli_invalid_inputs_and_missing_features_are_explicit[unsupported]` | PASS |
| `test_reporting_experiment_acceptance::test_cli_invalid_inputs_and_missing_features_are_explicit[no-context]` | PASS |
| `test_reporting_experiment_acceptance::test_cli_invalid_inputs_and_missing_features_are_explicit[strict-flag]` | PASS |
| `test_reporting_experiment_acceptance::test_cli_artifact_mismatch_and_unsupported_kind_are_distinct` | PASS |
| `test_reporting_experiment_acceptance::test_retained_training_and_advisor_typed_inputs` | PASS |
| `test_reporting_experiment_acceptance::test_full_qa_lifecycle` | PASS |
