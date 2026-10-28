# Native model structure comparison — 3B.4D.2

`compare_model_structures(baseline, candidate)` compares two validated
`ModelStructureSnapshot` objects. It reads only recorded snapshot metadata; it
does not accept models, introspect live objects, load checkpoints, execute
forward/backward, change modes, transfer tensors or train/benchmark anything.

```python
from hyso import compare_model_structures, ModelStructureComparison

# Existing snapshots from the actual capture_model_structure producer.
result = compare_model_structures(baseline_snapshot, candidate_snapshot)
for row in result.summary_differences:
    print(row.metric, row.baseline.value, row.candidate.value,
          row.difference, row.percent_change)
for path in result.changed_paths:
    for field in path.configuration_changes:
        print(path.path, field.field, field.baseline.value,
              field.candidate.value, field.status)
result.to_json("structure-comparison.json")  # Exclusive creation, no overwrite.
reopened = ModelStructureComparison.from_json("structure-comparison.json")
assert reopened.content_sha256 == result.content_sha256
```

The same API and frozen result/view classes are available from
`hyso.structure_comparison`. This stage supplies data for future reports, not a
table renderer, HTML/GUI, architecture audit or redesigned TR/EN presentation.

## Versioned result and sources

Format: `hyso.model_structure_comparison`, schema **1**. Policy:
`literal_registered_path_structure_v1`, version **1**. Matching is strictly by
registered path and registration name. The actual snapshot format, schema,
producer method/version, classification version, Torch version and content
SHA-256 are retained separately in typed `baseline` and `candidate` references.
Their order is never normalized or swapped.

Both inputs are revalidated through the existing snapshot constructor. The
result embeds both original metadata-only source snapshots under
`source_snapshots`, allowing its reader to revalidate their hashes and recompute
every difference rather than trust a serialized assertion. The full result,
including sources, references, policy and limitations, has canonical JSON and
`content_sha256` using the existing HYSo codec. Typed collections are immutable;
`to_dict()` is detached. `baseline_snapshot` and `candidate_snapshot` reopen the
retained sources without any live-model access.

Producer method/version and classification semantics must agree. Unsupported
snapshot versions are rejected by the existing reader. A difference in recorded
Torch version is separately listed in `provenance_differences`; comparing schema-1
metadata is not a cross-runtime behavioral or execution-equivalence certificate.
Unknown or inconsistent input/result schemas and rehashed invented differences
are rejected with `ModelStructureComparisonError`.

## Numeric differences

`summary_differences` includes unique parameter elements, trainable/frozen
elements, parameter tensor count/logical bytes, buffer tensor count/elements/
logical bytes, cross-category object count, union tensor logical bytes, unique
modules and registered module paths. Global counts come directly from validated
unique-object summaries; they are never summed over repeated module paths.

`module_type_differences` and `category_differences` separately compare unique
module and expanded-path counts. `dtype_differences` provides per parameter/buffer
dtype tensor count, elements and logical bytes. An absent bucket in an exhaustive
known distribution is an observed zero. If some tensors have unknown dtype,
known dtype buckets are incomplete: unknown tensors might belong to them, so
their complete counts/differences/percentages are withheld. The `dtype=None`
bucket explicitly counts unassigned observations, not a known numeric dtype.

For each `StructureNumericDifference`:

- `difference = candidate - baseline`, as an exact signed integer.
- `percent_change = 100 * difference / baseline`, separately, using decimal
  arithmetic before float conversion. A zero baseline yields null relative
  percent while retaining the absolute difference. Nonfinite relative results
  are withheld with an explicit reason.
- Both source quantities must be native complete totals. Unsupported/missing
  totals yield `status=insufficient_evidence`, null difference and null percent.
- `known_subtotal_difference` is separately reported; it is incomplete and
  never substituted for a global total or given a global relative percentage.

For example, 72 to 41 elements gives -31 and approximately -43.05556 percent.
This describes element counts only. There is no preferred direction,
performance/quality effect, optimization verdict or verified improvement.

## Literal paths, configuration and tensors

`added_paths` and `removed_paths` list exact registered module paths. There is
no rename/reparenting heuristic: `features.0` and `backbone.conv1` remain removed
and added paths, even if their metadata happens to match.

`changed_paths` retains both qualified types, metadata changes and individual
static `configuration_changes`. Each `StructureFieldDifference` has its field
name, baseline/candidate observations and status. Thus `features.conv1` can
expose `out_channels`, 32, 64, `changed`. Container type changes, module/category
changes and registry-name ordering are explicit static facts; no execution
order is inferred from them.

`tensor_differences` is keyed by owner path, parameter/buffer kind and local
registration name. It compares registration/tensor presence, buffer persistence,
qualified tensor type, shape, dtype, device/layout, requires_grad, element/byte
metadata, stride and storage offset. Snapshot-local tensor/storage IDs are
never compared as cross-model identity. Matching registration locations do not
establish semantic equivalence between differently typed modules or weights.

Observations distinguish `native`, `absent`, `unsupported` and `missing`.
An observed native null configuration value is different from unknown metadata.
Field statuses are `added`, `removed`, `changed` or `unresolved`. Unknown fields
may appear as unresolved rows in `changed_paths`/`tensor_differences`; those rows
are not observed changes and are excluded from `has_observed_changes`.
Custom modules remain visible with known registered facts and explicit unknown
configuration semantics. Their class names do not certify a model family.

## Shared/tied and storage relations

`alias_differences` has three distinct relation kinds:

- `module_object`: multiple registered paths target one Module object inside
  the source snapshot.
- `tensor_object`: multiple parameter/buffer registration endpoints target one
  Tensor/Parameter object inside the source snapshot.
- `storage`: multiple distinct tensor objects share one observed storage handle
  inside the source snapshot. Each group's `members` preserves the endpoint
  partition for each distinct object. One tied Parameter is not mislabeled as
  multiple objects sharing storage.

Relations are normalized to sorted `RegistrationEndpoint(owner_path, kind,
name)` partitions **within each source**, then compared on common observed
endpoints. Local `m0`/`t0`/`s0` labels and raw memory identities have no cross-model
meaning. Pure ID renumbering therefore produces no structural differences.
There is no quadratic pairwise alias expansion.

`baseline_groups` and `candidate_groups` retain the complete observed source
sharing partitions, including relations involving unmatched/new endpoints.
`matching_scope=common_registered_endpoints` makes the difference scope explicit.
Removed/added groups describe changes on those assessed common endpoints.
Unmatched endpoints are listed separately. A removed path is not automatically
called broken weight tying; extending a tied group with a newly added endpoint
does not fabricate a change among the existing endpoints. If storage identity
is unavailable, the affected endpoints are excluded from both sides of that
relation comparison and listed as unassessed. A known sharing relation is not
declared absent merely because the other snapshot lacks evidence.

## Coverage, scope and limits

`evidence_gaps` retains side, owner path, kind/name, unavailable fields, status
and reason. `coverage` is `partial` when recorded configuration/tensor/storage
evidence or alias alignment is incomplete; otherwise it is
`complete_recorded_metadata`. This never means a complete model behavior audit.
`has_observed_changes=False` means no changes among the assessed recorded facts;
it is not an equivalence certificate, especially with partial coverage.

The same snapshot can be compared with itself and yield zero differences.
That does not establish two independent captures or models. Equal structure
does not imply equal weights, outputs or live-model origin. Imported JSON
consistency does not authenticate its original producer. Structure content
hashes remain distinct from model-state and checkpoint SHA-256 identities.

Functional operations, forward graph/order, dynamic control flow, runtime
shapes, activation memory and MAC/FLOPs are out of scope. Logical bytes do not
measure VRAM, RSS, physical unique storage bytes or checkpoint file size.
Requires_grad does not prove training behavior. Counts and static differences
do not prove speed, lower VRAM, accuracy, generalization, optimization success
or `VERIFIED_IMPROVEMENT`. Actual YOLO/LLM support is not certified by synthetic
registered-component fixtures.

Expanded observations/differences have a hard 500,000-entry budget and canonical
JSON a 64 MiB budget, including both metadata sources. `max_entries` and
`max_bytes` keyword arguments may lower those bounds. Exceeding either fails
explicitly; no truncated comparison is returned. Alias comparison is bounded
independently of unique-object counts, so repeated shared paths cannot bypass
the budget.

Old snapshot schema/hashes, Advisor 3B.4A/B/C, P26/P11 schemas and TR/EN report
views are unchanged. No new evidence adapter or automatic association is added.
The comparison format is rejected by existing P26/P11 evidence importers.

Focused tests: `tests/test_model_structure_comparison.py`, using actual D.1
producer snapshots and prohibited introspection/workload/device paths during
comparison. Fixtures cover CNN, RNN/GRU/LSTM, Transformer and custom detector-like
or embedding/linear components without running their forward methods.
