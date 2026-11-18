# Stage 13.5.14 evidence compatibility matrix

The acceptance plan uses small deterministic CPU QA tensors. Every fresh result
uses the ordinary evaluator and its isolated forward receiver. The configured
`test` label in these fixtures denotes QA data. Measured outcomes and individual pytest IDs follow the initial plan below.

| Scenario and real input | Expected result | Evidence source | Protected invariant |
| --- | --- | --- | --- |
| Perfect, wrong, collapsed, imbalanced, absent predicted class, zero support, ten classes | Exact integer counts and independent rational metrics | Supplied predictions and actual Identity receiver logits | Confusion rows, columns, supports, macro/micro/weighted arithmetic |
| Binary single logit, zero threshold, two-class multiclass logits | Explicit task-specific decisions | Actual CPU evaluator | Class map and threshold |
| Binary two logits, wrong width/target, NaN/Inf, empty, under/over collated batch | Reject malformed input, empty retains null metrics | Public evaluator | Complete finite sample accounting |
| Same stream, changed architecture/weights, different or equal accuracy | COMPARABLE with descriptive differences | EvaluationResult through EvidenceIndex and ModelComparisonSource | Model changes explicitly permitted, global winner null |
| Changed input, order, targets, class names/order, split, task, batch, count, zero division | INCOMPATIBLE with null differences | Fresh native results | Exact ordered tensor and protocol identity |
| Schema1, supplied predictions, absent producer, incoherent checkpoint/state, reused ID | Insufficient evidence or reader rejection as applicable | Native reader and comparison | Native consistency and capture reuse |
| Changed declared dataset/preprocessing labels with identical observed tensors | Historical policy permits COMPARABLE, stronger requested rejection checked separately | EvidenceScope and P26 policy | Preserve policy while exposing acceptance gap |
| Correct/wrong checkpoint, state mutation, copied model, receipt expiry | Exact native receipt or missing/reject | Trusted QA checkpoint restore | Process-local object/state association |
| Default Graph, explicit supported CPU receiver, imported/copied/expired association | limited, verified, unverified/unavailable respectively | Actual Graph evaluator and a second interpreter | Separate source, receiver and execution authority |
| Graph hooks, subclass, mutation, topology/output changes | Reject verification | Existing receiver negative regressions | Closed trusted profile |
| Same config with changed weights, equal parameter count with changed activation/topology/head | Independent architecture, registered structure and quality observations | Existing comparison APIs | Metadata equality conveys its recorded scope |
| Schema1/2/3 reads and tampered metrics/raw/accounting/structure | Preserve valid versions, reject inconsistent statements | EvaluationResult reader | Exact schema and content consistency |
| JSON/Markdown/static HTML and fresh QA ExperimentManager | Source values retained, valid hash verified, wrong bytes mismatch | Existing Reporting and artifact APIs | Presentation and artifact integrity |
| Wheel/sdist isolated short positive and negative tests | Target imports and 164 source module hashes match | Existing stage 13.5.13 distributions | Installed/source code equivalence |

Strict requirement probes remain visible when they fail. They cannot be converted
into a PASS by changing the expected historical policy status.


## Exact fixture inputs

The arithmetic reference uses integer pair counts and fractions. Every metric
fixture is run with zero_division 0 and 1, finite logits −4/+4, an actual Identity
receiver and batches of three. Configured classes include every listed index.

| Fixture | Targets | Predictions | Classes |
| --- | --- | --- | --- |
| perfect | [0,1,2,0] | [0,1,2,0] | 3 |
| wrong | [0,1,2,0] | [1,2,0,2] | 3 |
| collapse | [0,1,2,0] | [1,1,1,1] | 3 |
| imbalanced | nine 0, one 1 | eight 0, then 1,0 | 3 |
| missing_prediction | [0,1,2,2] | [0,1,1,1] | 3 |
| zero_support | [0,0,1,1] | [0,2,1,2] | 4 |
| ten_classes | range(10) twice | range(10), then ten 0 | 10 |

Native comparison inputs are x=[−2,−1,1,2] with shape [4,1] and int64
labels [0,0,1,1], classes [a,b] and batch sequence [2,2]. Baseline Linear logits
are [0,x−1.5], candidate logits [0,x]. The equal-accuracy candidate uses
[0,x+1.5] through a Sequential wrapper. Binary native inputs use floating labels
and one logit. The threshold/tie fixtures use logits [−1,0,1] with labels [0,1,1]
and explicit multiclass logits [[1,1],[0,2]] with labels [0,1].

| Native change | Actual input or envelope edit | Expected/observed native outcome |
| --- | --- | --- |
| input | first input −2→−3 | INCOMPATIBLE, differences null |
| order | reverse x and y | INCOMPATIBLE, differences null |
| target | first label 0→1 | INCOMPATIBLE, differences null |
| count | truncate to three samples | INCOMPATIBLE, differences null |
| classes | [a,b]→[b,a] | INCOMPATIBLE, differences null |
| vocabulary | [a,b]→[a,other] | INCOMPATIBLE, differences null |
| split | test→validation | INCOMPATIBLE, differences null |
| task | multiclass→binary with a valid single-logit model | INCOMPATIBLE, differences null |
| batch | [2,2]→[3,1], identical content hashes | INCOMPATIBLE, differences null |
| zero | zero_division 0→1 | INCOMPATIBLE, differences null |
| schema1/supplied | omit native producer or use hard-label result | INSUFFICIENT_EVIDENCE, differences null |
| duplicate ID | independently produce then rehash with copied evaluation UUID | INCOMPATIBLE, differences null |
| wrong scope | supply all-zero native state/checkpoint hash | EvidenceError |
| incoherent JSON | modify metric/raw/count/batches/producer/checkpoint/stream/structure and rehash source | ValueError |
| wrong artifact SHA | expected SHA all zeros | EvidenceError |

Graph fixtures use an actual supported stem/branch/add/head CPU Graph,
x=arange(64).reshape(4,1,4,4).float()/64, labels [0,1,0,1] and batches [2,2].
Trusted checkpoint save/load precedes opt-in capture. Dataset/preprocessing
mismatch probes change one declaration while retaining exactly these tensors.
Source/receiver/association objects stay live during the comparison.

## Measured acceptance outcomes

Source has 69 successful new contract cases and four failed requirement cases.
Selected existing regressions add 317 successful cases. Each installed target
repeats 69 successful cases and the same four failures. Errors, skips and xfails
are zero in the final groups. Native statuses in this ledger are established by
the assertions in the linked test sources. The XML, logs and complete serialized
failed-comparison evidence are retained under the external QA root.

| Test function | Actual observed outcome and invariant |
| --- | --- |
| independent_metrics_on_actual_receiver | complete, every independent arithmetic/accounting check passes |
| explicit_binary_and_two_class_multiclass | complete, correct task-specific matrix |
| invalid_task_width_targets_and_nonfinite | ValueError, task/width/target/input guard |
| nonfinite_output_with_finite_input_is_rejected | ValueError, actual nonfinite logit rejection |
| collation_cannot_under_or_over_account | ValueError, consumed count equals expected |
| empty_and_class_order_are_explicit | empty/null/zero accounting and configured class ordering |
| actual_native_chain_compares_changed_architecture | COMPARABLE, +0.25 accuracy difference, null global winner |
| equal_accuracy_different_per_class_has_no_winner | COMPARABLE, zero accuracy difference, null global winner |
| actual_incompatible_evidence_withholds_numbers | INCOMPATIBLE and null differences |
| missing_native_producer_is_insufficient | INSUFFICIENT_EVIDENCE and null differences |
| declared_scope_historical_policy_boundary | COMPARABLE with differences, exposes the historical policy boundary |
| copied_evaluation_identity_is_rejected | INCOMPATIBLE and null differences |
| incoherent_json_rejected_with_rehashed_envelope | ValueError, reader consistency enforcement |
| schema_roundtrip_immutability_and_artifact_hash | exact schema1/2/3 roundtrip, FrozenInstanceError, wrong artifact hash rejected |
| checkpoint_receipt_is_object_state_and_lifetime_bound | native→missing on copied/changed state, weak receipt collected |
| wrong_native_model_or_checkpoint_scope_is_rejected | EvidenceError, caller scope cannot override native identity |
| graph_actual_receiver_default_and_process_loss | source verified, default limited/opt-in verified, imported and expired unavailable |
| graph_json_in_second_interpreter_cannot_restore_authority | imported source/receiver unverified, inference binding unavailable |
| graph_context_comparison_rejects_declared_mismatch | top INCOMPATIBLE/null, nested legacy COMPARABLE/numeric |
| structure_equality_survives_weight_change_without_receiver_authority | registered metadata equal, state hashes and quality differ |
| architecture_metadata_separates_weights_activation_head_and_topology | distinct configuration hashes, topology/activation equal counts, head count differs |
| reporting_and_experiment_artifacts_in_fresh_qa_store | source values retained, correct artifacts verified, wrong reference bytes mismatch |
| graph_reporting_preserves_limited_and_imported_authority | limited/unavailable retained in independent presentation sections |
| requested_generic_declared_mismatch_rejection | FAIL, actual COMPARABLE and numeric differences |
| requested_graph_incompatibility_has_no_nested_numeric_difference | FAIL, numeric differences in embedded legacy result |

The [69-case source](../../tests/test_evaluation_comparison_acceptance.py) and
[four ordinary failing requirements](../../tests/test_evaluation_comparison_requirements.py)
are committed. Both modules use normal pytest assertions and collection.

### Individual source case IDs

| Source pytest ID | Result |
| --- | --- |
| `test_evaluation_comparison_acceptance.py::test_independent_metrics_on_actual_receiver[perfect-0]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_independent_metrics_on_actual_receiver[perfect-1]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_independent_metrics_on_actual_receiver[wrong-0]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_independent_metrics_on_actual_receiver[wrong-1]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_independent_metrics_on_actual_receiver[collapse-0]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_independent_metrics_on_actual_receiver[collapse-1]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_independent_metrics_on_actual_receiver[imbalanced-0]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_independent_metrics_on_actual_receiver[imbalanced-1]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_independent_metrics_on_actual_receiver[missing_prediction-0]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_independent_metrics_on_actual_receiver[missing_prediction-1]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_independent_metrics_on_actual_receiver[zero_support-0]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_independent_metrics_on_actual_receiver[zero_support-1]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_independent_metrics_on_actual_receiver[ten_classes-0]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_independent_metrics_on_actual_receiver[ten_classes-1]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_explicit_binary_and_two_class_multiclass[binary-logits0-targets0-expected0]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_explicit_binary_and_two_class_multiclass[multiclass-logits1-targets1-expected1]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_invalid_task_width_targets_and_nonfinite[targets0-logits0-binary]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_invalid_task_width_targets_and_nonfinite[targets1-logits1-multiclass]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_invalid_task_width_targets_and_nonfinite[targets2-logits2-multiclass]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_invalid_task_width_targets_and_nonfinite[targets3-logits3-multiclass]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_invalid_task_width_targets_and_nonfinite[targets4-logits4-multiclass]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_invalid_task_width_targets_and_nonfinite[targets5-logits5-multiclass]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_nonfinite_output_with_finite_input_is_rejected[nan-logit]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_nonfinite_output_with_finite_input_is_rejected[inf-logit]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_collation_cannot_under_or_over_account[-1]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_collation_cannot_under_or_over_account[1]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_empty_and_class_order_are_explicit` | PASS |
| `test_evaluation_comparison_acceptance.py::test_actual_native_chain_compares_changed_architecture[False]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_actual_native_chain_compares_changed_architecture[True]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_equal_accuracy_different_per_class_has_no_winner` | PASS |
| `test_evaluation_comparison_acceptance.py::test_actual_incompatible_evidence_withholds_numbers[input]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_actual_incompatible_evidence_withholds_numbers[order]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_actual_incompatible_evidence_withholds_numbers[target]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_actual_incompatible_evidence_withholds_numbers[count]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_actual_incompatible_evidence_withholds_numbers[classes]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_actual_incompatible_evidence_withholds_numbers[vocabulary]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_actual_incompatible_evidence_withholds_numbers[split]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_actual_incompatible_evidence_withholds_numbers[task]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_actual_incompatible_evidence_withholds_numbers[batch]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_actual_incompatible_evidence_withholds_numbers[zero]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_missing_native_producer_is_insufficient[schema1]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_missing_native_producer_is_insufficient[supplied]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_declared_scope_historical_policy_boundary[dataset_id]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_declared_scope_historical_policy_boundary[preprocessing_id]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_copied_evaluation_identity_is_rejected` | PASS |
| `test_evaluation_comparison_acceptance.py::test_incoherent_json_rejected_with_rehashed_envelope[metric]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_incoherent_json_rejected_with_rehashed_envelope[tiny_metric]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_incoherent_json_rejected_with_rehashed_envelope[raw]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_incoherent_json_rejected_with_rehashed_envelope[count]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_incoherent_json_rejected_with_rehashed_envelope[batch]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_incoherent_json_rejected_with_rehashed_envelope[producer]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_incoherent_json_rejected_with_rehashed_envelope[checkpoint]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_incoherent_json_rejected_with_rehashed_envelope[stream]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_incoherent_json_rejected_with_rehashed_envelope[structure]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_schema_roundtrip_immutability_and_artifact_hash[1]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_schema_roundtrip_immutability_and_artifact_hash[2]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_schema_roundtrip_immutability_and_artifact_hash[3]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_checkpoint_receipt_is_object_state_and_lifetime_bound` | PASS |
| `test_evaluation_comparison_acceptance.py::test_wrong_native_model_or_checkpoint_scope_is_rejected[model_state_sha256]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_wrong_native_model_or_checkpoint_scope_is_rejected[checkpoint_sha256]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_graph_actual_receiver_default_and_process_loss[False]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_graph_actual_receiver_default_and_process_loss[True]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_graph_json_in_second_interpreter_cannot_restore_authority` | PASS |
| `test_evaluation_comparison_acceptance.py::test_graph_context_comparison_rejects_declared_mismatch[dataset_id]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_graph_context_comparison_rejects_declared_mismatch[preprocessing_id]` | PASS |
| `test_evaluation_comparison_acceptance.py::test_structure_equality_survives_weight_change_without_receiver_authority` | PASS |
| `test_evaluation_comparison_acceptance.py::test_architecture_metadata_separates_weights_activation_head_and_topology` | PASS |
| `test_evaluation_comparison_acceptance.py::test_reporting_and_experiment_artifacts_in_fresh_qa_store` | PASS |
| `test_evaluation_comparison_acceptance.py::test_graph_reporting_preserves_limited_and_imported_authority` | PASS |
| `test_evaluation_comparison_requirements.py::test_requested_generic_declared_mismatch_rejection[dataset_id]` | FAIL |
| `test_evaluation_comparison_requirements.py::test_requested_generic_declared_mismatch_rejection[preprocessing_id]` | FAIL |
| `test_evaluation_comparison_requirements.py::test_requested_graph_incompatibility_has_no_nested_numeric_difference[dataset_id]` | FAIL |
| `test_evaluation_comparison_requirements.py::test_requested_graph_incompatibility_has_no_nested_numeric_difference[preprocessing_id]` | FAIL |

## Security requirement findings

| Field | Generic comparison | Graph effective rows | Graph embedded legacy rows | Report |
| --- | --- | --- | --- | --- |
| dataset_id differs | COMPARABLE, +0.25 accuracy and +0.2666666666666666 macro-F1 | INCOMPATIBLE, differences null | COMPARABLE, numeric zero differences | visible values null, charts empty, raw legacy differences retained |
| preprocessing_id differs | COMPARABLE, +0.25 accuracy and +0.2666666666666666 macro-F1 | INCOMPATIBLE, differences null | COMPARABLE, numeric zero differences | visible values null, charts empty, raw legacy differences retained |

These four final failed cases leave overall acceptance PARTIAL and gate G FAIL.
The main report records the root causes, proposal boundaries, per-gate decisions,
installed repeats, history preservation and Git closure.
