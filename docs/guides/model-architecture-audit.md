# Static Model Architecture Audit — 3B.5A

`audit_model_architecture(snapshot)` reads a single validated
`ModelStructureSnapshot`. It does not accept a live model, factory, path or
checkpoint. It runs no forward/backward, model transfer, training or benchmark.

```python
from hyso import (
    capture_model_structure, audit_model_architecture, ModelArchitectureAudit,
)

# An existing trusted, quiescent object; no external model code is loaded.
snapshot = capture_model_structure(model)
audit = audit_model_architecture(snapshot)
for finding in audit.findings:
    code = finding.code
    kind = finding.classification.value
    path = finding.path
    observed = [(fact.field, fact.status, fact.value)
                for fact in finding.observations]
    expected = None if finding.expected is None else finding.expected.value
audit.to_json('architecture-audit.json')  # exclusive; never overwrites
restored = ModelArchitectureAudit.from_json('architecture-audit.json')
```

The result uses the existing structure contract style: frozen typed views,
detached plain JSON, canonical serialization, SHA-256, explicit resource limits,
and a reader that rederives results from retained source evidence. There is no
Advisor severity score, confidence probability, model-wide PASS/FAIL or automatic
optimization recommendation.

## Version and closed rule catalog

The wire format is `hyso.model_architecture_audit`, schema **1**. Policy
`registered_static_architecture_v1`, version **1**, admits only structure schema
1, producer `registered_nn_module_metadata` version 1 and classification version
1. The complete fixed catalog always runs; callers cannot suppress a gap or
replace a rule without a new versioned policy.

| Rule ID | Version | Scope |
|---|---:|---|
| `linear_shapes` | 1 | Exact standard `torch.nn.Linear` |
| `convolution_shapes` | 1 | Exact standard Conv1d/Conv2d/Conv3d |
| `parameter_inventory` | 1 | Unique parameter totals, fractions, distributions and direct owner shares |
| `registration_relationships` | 1 | Local module/tensor/storage registrations and cycles |
| `coverage_gaps` | 1 | Unassessed semantics, configuration, tensors and storage |

Findings use three separate classes:

* `STATIC_INCONSISTENCY`: a known contradiction between supported recorded
  metadata fields or a supported integer domain constraint.
* `OBSERVATION`: a recorded quantity, derived ratio or registration relationship.
  A large or completely frozen model, tied weights or a cycle is not inherently
  a defect.
* `COVERAGE_GAP`: evidence or a rule needed for an assessment is unavailable.
  Missing information never becomes evidence that the model is correct.

## Shape rules and false-positive boundaries

The producer's exact type identity, qualified type and category must support the
rule. A custom subclass is not treated as its standard base class. Even a custom
type that spoofs a standard qualified name retains the producer's `custom`
category and does not acquire a shape rule. Imported metadata still requires
external trust in the original producer; names are not cryptographic provenance.

For standard Linear, weight shape must be `[out_features, in_features]`, and a
present bias must have shape `[out_features]`. Dimensions must be native integer
metadata with nonnegative values. Zero-size dimensions remain valid static
metadata; they do not establish that a runtime workload will succeed.

For normal Conv1d/2d/3d, weight shape must be
`[out_channels, in_channels / groups, *kernel_size]`, and a present bias must have
shape `[out_channels]`. Groups must be a positive integer, channels nonnegative
integers divisible by groups, and kernel dimensions nonnegative integers of the
correct rank. The expected shape is derived only after those prerequisites
hold. Invalid divisibility never produces an expectation using floor division.
Transposed convolutions have a different layout and receive no normal-Conv rule.

Unsupported or missing configuration, non-native tensor metadata, lazy tensors,
custom tensor subclasses and absent registered weight metadata produce gaps or
`insufficient_evidence` checks. Plain values of types unsupported by the rule are
not coerced. Sparse/quantized metadata is not used for a definite shape mismatch
even if a shape was recorded. Reliable meta-device shapes can be checked while
missing materialized storage is reported independently.

Only fields required by a check are prerequisites. An unrelated unavailable Conv
configuration field does not erase a valid shape check, but its own coverage gap
remains. Known contradictions and missing evidence can coexist in one module.
Neither supported shape checks nor module registration establish input/output
runtime shapes, forward semantics or general model correctness.

## Quantities and relationships

Global parameter/trainable/frozen totals reuse the snapshot's unique tensor
object accounting. Complete totals and `known_subtotal` remain separate. Fractions
are derived only from complete totals with a nonzero denominator; otherwise they
are null with an explicit reason. Module/category/dtype distributions retain
their actual source values, including unsupported quantity statuses.

Direct owner shares describe unique parameters registered immediately on a
supported owner. They do not count descendants or executed layers and are not
performance density. Tied parameters can appear in multiple owner shares; those
shares must not be summed. Global totals still count each tensor object once.

Repeated noncyclic paths of one module are reported once per local object. Tied
parameters use distinct registrations on unique module records, so expanding a
shared module's paths does not invent extra tied parameters. Parameter/buffer
sharing, distinct tensor objects sharing storage, and registered cycles are
separate observations. Local IDs never identify objects in other snapshots.
Storage sharing proves neither byte-range overlap nor physical memory usage.

## Programmatic evidence and coverage

The JSON contains `source_snapshot`, `audit_policy`, `summary`, `findings`,
`coverage`, `limitations` and `content_sha256`. The immutable source is embedded
once; every finding binds to its content SHA.

Each typed `ArchitectureAuditFinding` includes a stable content-derived finding
ID and code, class, rule/version, module/tensor ID, path and related registration
paths, observations, optional expected/derived values, evidence status,
limitation and a safe metadata-review suggestion. `AuditEvidence.field` is an
actual dotted path into the retained snapshot (list indexes are decimal), and
`.value` returns a detached value. Derived values carry a named method and the
actual source fields used. No model code is imported to follow a path.

`module_assessments` expose typed per-module checks with `consistent`,
`inconsistent`, `insufficient_evidence` or `not_applicable` status. A known invalid
prerequisite makes its dependent shape expectation not applicable. Coverage
separates total unique modules, registered paths, shape-rule eligible modules,
examined supported modules, consistent/inconsistent modules, insufficient
modules and modules without a rule. It retains unassessed IDs and per-rule
counts. It also counts individual check states and retains IDs with insufficient
checks, including modules that have a separate known contradiction. Unassessed
IDs include these partially assessed modules. A module without a rule is never
silently counted as consistent.

Coverage can be `partial`, `no_applicable_shape_modules` or
`recorded_fields_assessed_for_v1_rules`. Even the last status covers only the
named v1 checks. Findings are ordered by the versioned catalog, then deterministic
source/check traversal. Duplicate findings are rejected rather than hidden.

## Reader, limits and compatibility

The reader validates the embedded snapshot through its existing constructor,
reruns the complete audit and compares the entire rederived canonical result:
policy, observed/expected values, source identities, finding IDs/order, coverage,
summary, limitations and hash. Rehashing an invented finding cannot validate it.
Malformed snapshots, duplicate JSON keys, extra fields and unsupported versions
are rejected. A wholly fabricated, self-consistent source cannot be authenticated
from its own hashes; imported consistency is not live-producer authenticity.

The source retains the existing 16 MiB snapshot limit. Audit limits default to
500,000 derived entries and 64 MiB canonical JSON, with no silent truncation.
`max_entries` and `max_bytes` may be lowered for creation. Exports are exclusive.
Typed results/findings/policies/checks are frozen; returned dictionaries and
observation values are detached. No timestamps or RNG are used during auditing.

P26/P11 schemas, D.1/D.2 capture/comparison semantics, general/P26 comparison
engines and old canonical JSON/hashes are unchanged. No latency, FLOPs, GPU memory,
gradient/overfitting/learning diagnosis, optimization success, model superiority,
P29 acceptance, real YOLO/LLM qualification, Model Builder, GUI or end-user report
is added. Registration order is not execution order, and functional or dynamic
forward operations may be absent.

## Acceptance checks

`tests/test_architecture_audit.py` uses actual owned synthetic CPU modules and
the real structure producer. Its CNN has 122 parameter elements (120 trainable,
2 frozen): two modules receive consistent shape checks and four explicitly have
no applicable shape rule. This produces observations and coverage gaps, without
a defect or model-wide correctness claim. Separate intentionally inconsistent
owned Linear/Conv fixtures produce the corresponding static contradiction codes.

Tests exercise producer -> snapshot -> audit -> JSON -> reader, grouped/normal
convolutions, shape mismatches, missing/custom/lazy/meta metadata, tied/shared/
cyclic registrations, partial totals, deterministic output, rehashed corruption,
immutability, model values/buffers/gradients/hooks/code/modes/RNG preservation and
explicit prohibitions on live execution during capture/audit. Existing evidence,
comparison, canonical artifact and MNIST TR/EN regressions remain separate checks.
