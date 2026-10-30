# P29 accuracy–speed–memory acceptance

**P29 = IMPLEMENTED within finite CPU FP32/FP64 classification scope.**
Quality, resident inference latency and one explicitly selected memory domain are
three independent dimensions. This does not certify a universally better model,
significant speedup, model-exclusive RSS, GPU memory or a physical memory saving.

## Original gap and change

The original roadmap (`a0672d5`) requests compatible quality/speed/memory results
and visible uncertainty. The previous quality audit accepted a narrower CPU
quality–latency implementation and explicitly excluded memory Pareto. Its P29
derivation used only quality and mean latency. That historical acceptance is kept;
it is not evidence of a three-dimensional comparison.

The revision adds a P29-only memory adapter and validation in `hyso/_quality_memory.py`,
called by `compare_performance_quality`. Other performance implementations and
P24/P27/P28 calculations are unchanged. Memory uses the existing P11/P10 readers
and collectors, rather than a new profiler or synthesized aggregate.

```python
from hyso import compare_performance_quality
from hyso.performance import MemoryProtocol

# No extra forward for memory: logical registered Tensor object bytes, P11.
result = compare_performance_quality(baseline, candidate, cohort,
                                    memory_metric="logical_model_bytes")

# Explicit opt-in: one extra guarded inference per precision-owned model copy.
rss_result = compare_performance_quality(
    baseline, candidate, cohort,
    memory_metric="cpu_rss_sampled_peak_bytes",
    memory_protocol=MemoryProtocol(interval_seconds=.01, max_samples=10000),
    max_memory_regression=.05)
```

P29 timing remains the existing measured latency protocol, with raw repeats.
It does not derive P02 completion throughput from reciprocal mean latency.
P26 labeled evaluation and P28 precision execution guards are retained; mixed
precision, BF16/INT8 timing and CUDA evaluation are not added here.

## Memory domains and identity

| Selection | Source and meaning |
| --- | --- |
| `logical_model_bytes` (default) | P11 `parameter_buffer_bytes`, unique registered Tensor object convention; includes registered buffers, excludes input/activation/gradients and adapter witness buffers |
| `cpu_rss_sampled_peak_bytes` | P10 native process RSS samples around a separate single inference; includes the process, observer, runtime and allocator history |
| `None` | Explicitly not requested; null/unsupported |
| `cuda_allocated_peak_bytes` | Null/unsupported: CPU P29 has no matched P09 allocator capture |
| `cuda_reserved_peak_bytes` | Null/unsupported, distinct from allocated |
| `physical_vram_bytes` | Null/unsupported; no physical-device attribution |

There is no fallback from a missing RSS or GPU measurement to logical bytes.
No memory fields are added together. Distinct Tensor views can count separately
despite shared storage; logical bytes are not physical RAM/VRAM. RSS is neither
model-only memory nor an exact window peak. RSS windows are sequential observations
of the same process and can differ because of cache, retained Python objects,
working-set policy and unrelated activity. No causal attribution is inferred.

Each memory record is bound to its benchmark run ID, model weights and structure,
input/cohort hashes, host, device, runtime, precision and benchmark protocol.
Raw P10/P11 results are revalidated on import. Logical totals are independently
checked against registered tensor byte/alias witnesses. RSS must describe one
completed synchronous inference (or no execution if the backend is unavailable),
and both sampling configurations must match the requested protocol. Foreign
models/runtimes, relabeled domains, invalid aggregates and mismatched protocols
are rejected. Registered state/input guards bracket memory capture on owned
precision copies; caller model, gradients, modes and RNG are preserved.

Hashes detect inconsistency, not signed execution authenticity. Recomputing an
entire fabricated evidence envelope is outside that integrity guarantee.

## Pareto and regression contract

New sources carry `comparison_version=2` inside the existing schema_version=1
quality result. Candidate dominance requires quality >= baseline, latency <=
baseline and selected-domain bytes <= baseline, with at least one strict
improvement. The reverse rule defines baseline dominance; equality and trade-offs
yield `neither`. No score, arbitrary weighting or quality gain is invented.

`analysis.pareto_dominates` in version 2 refers to these three dimensions.
The prior two-dimensional result is separately labeled
`quality_speed_pareto_dominates`. If either memory value is missing,
`pareto_dominates=null` and `pareto_status=unsupported_missing_memory` even if
quality/latency favor one model. `winner` is always null: recorded Pareto dominance
is descriptive and does not establish significance or environmental stability.
Memory-only dominance does not imply improved quality or speed.

Memory delta is candidate minus baseline, in bytes; relative change is that delta
divided by baseline. Regression is strictly `relative_change > max_memory_regression`.
With a zero baseline the byte delta remains defined but relative change and the
relative regression decision are null with an explicit reason. Existing quality
and latency thresholds remain separate and unchanged.

## Backward compatibility

- Existing public arguments and top-level result schema_version=1 remain valid.
  Memory options are additive keyword arguments.
- Sources without a comparison version retain the original two-dimensional
  derivation and field semantics. They never acquire invented memory observations.
- Memory-aware records require version 2; dropping the version while retaining
  memory fields is rejected rather than silently downgrading the comparison.
- Archived real P29 JSON reopens unchanged. Its CSV re-export was byte-identical
  to the archived CSV. CSV remains a flattened summary with the same five columns;
  new records add memory fields and provenance rows. Canonical raw reopening is JSON.

## Verification and decision

- P29 memory/legacy quality tests: **102 PASS** (101-case run plus the separately
  added regression-threshold boundary case, distinct test identities).
- Related P26, P10/P11, benchmark compatibility/state and public API regression:
  **273 PASS**.
- Packaging configuration and real wheel/sdist content checks: **2 PASS**.
  The explicit packaging inventory was stale at 67 modules; it now includes the
  P29 helper and twelve already-present accepted modules, matching all 80 sources.
  Those other modules' implementations were not changed.
- **377 distinct targeted/regression/package tests PASS**, zero final failures.
  This is not a full-suite or broad release-matrix claim. Existing deprecation
  warnings remain visible.
- A source-matching wheel was installed without dependencies into a fresh local
  target and ran a real CPU P29 comparison successfully, with CUDA uninitialized.
- Native logical fixture: quality 0 -> 1, logical model bytes 56 -> 24,
  delta -32 bytes / -57.142857%. The stored timing also supported observed candidate
  Pareto dominance in that run; `winner` remained null. This is a tiny fixture,
  not evidence of an application speedup.
- Native RSS fixture: both separate windows completed one inference and retained
  actual native samples. No absolute/model-exclusive peak or memory-saving claim
  follows from their difference. Sampler cleanup and unavailable-backend handling pass.
- Independent reference cases cover strict dominance, equality, non-dominance and
  three-way trade-offs; additional checks cover zero baseline, threshold equality,
  shared objects/views, missing data, CPU/GPU domain confusion, incompatible
  evaluation/model/runtime/protocol and source/analysis tampering.

An initial malformed-source test exposed a validation-order KeyError when a P11
source was passed as RSS; raw source type validation now precedes sampling-protocol
access. A later test-only assertion placement error was corrected. Final results
above do not count intermediate repeated runs again.

P12 remains isolated in its separate worktree. Existing private evidence and user
changes are protected. No GPU workload, long test, global dependency change,
commit or push was performed. P31's 7200-second acceptance remains deferred to the
final validation after feature development; P34/P35 are outside this change.

## 2026-10-29 trained-model completion

The historical 377-test acceptance above is retained. The roadmap's old PARTIAL
entry predates this accepted CPU implementation. Full trained MNIST evaluation
exposed an additional operational gap: the original adapter submitted the entire
cohort as one evaluation/timing/RSS batch.

An additive `batch_size` argument now streams all quality samples, including the
tail, and benchmarks the same explicitly identified first ordered batch for both
models. Large encoded cohorts require this explicit choice; no automatic giant
batch, new RAM quota or artificial batch ceiling is introduced. This resident
batch latency is not full-test-set latency or throughput. Memory capture uses
the identical resident batch when a forward is needed.

Top-level schema_version remains 1. Explicit batching uses comparison_version 3,
with requested/effective batch size, selected sample IDs and content identity.
Model/state/runtime/precision/input/cohort/protocol binding remains mandatory;
the memory binding additionally includes the batch selection hash. Existing
version 1/2 analysis and archived JSON/CSV bytes retain their original semantics.
Version 3 reports accuracy and macro-F1 deltas independently, alongside the
existing selected-quality regression, latency and same-domain memory decisions.

The [completion report](../../outputs/p29-tradeoff-completion-20261029/final-report.md)
records the two trained checkpoints, the full ordered 10,000-sample test cohort,
three fresh CPU process pairs per precision, raw latency repeats, selected P11
logical model bytes, descriptive three-axis Pareto and seeded paired resampling.
The audit's conditional intervals are distinct from the API's unsupported
population/timing significance fields; no winner is inferred. Fixed A-then-B
measurement order and shared machine activity remain limitations. Prior data
from other timing protocols is not substituted into these comparisons.

CPU FP32/FP64 remains the verified scope. RSS, logical object bytes, CUDA
allocated/reserved and physical VRAM are distinct domains; selecting logical
bytes does not certify an RSS or GPU saving. No new CUDA workload or inherited
GPU authorization is used. B4 and the separate CPU65536 native issue remain OPEN.
