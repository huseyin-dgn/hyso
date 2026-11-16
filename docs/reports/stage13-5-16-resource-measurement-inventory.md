# Stage 13.5.16 performance and resource API inventory

The inventory was captured before accepted measurement. `QA/api-inventory.json`
contains 524 actual top-level/method definitions and signatures across 23 inspected
source files. Signatures were extracted from the current source AST, with source
SHA-256, line numbers and docstrings. The table gives the selected execution and
evidence boundary for each surface. [Acceptance report](stage13-5-16-cpu-cuda-performance-acceptance.md)

| Entry point | Visibility | Measurement class | Device and precision | Timer and synchronization | Metric and unit | Execution safety/status | P01–P35 relationship and trust |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BenchmarkRunner.run | Public | Measured resident forward and derived service rate | CPU fp32/fp64, CUDA fp32/amp_fp16 | perf_counter_ns or selected-stream events with end-event sync | ms, images/s, parameter counts | Used CPU, CUDA safety-blocked | P01/P09 namespace, new QA evidence |
| analyze_latency | Public | Derived order statistics with TailPolicy | Saved CPU/CUDA results | Offline saved samples | ms and eligibility | Used CPU snapshots | P01 derived evidence |
| measure_throughput + ThroughputProtocol | Public | Measured completion windows | CPU fp32/fp64, CUDA fp32/amp_fp16 | perf_counter_ns, synchronous CPU or bounded CUDA completion/terminal event | images/s, batches, elapsed s | Used CPU | P02 measured evidence |
| measure_memory + MemoryProtocol | Public | Native process observations | CPU process, device-independent RSS | perf_counter_ns timestamped OS sampling | RSS/HWM byte | Used finite trusted CPU work | P10 process evidence |
| measure_footprint | Public | Structural logical bytes and actual artifact bytes | Dense CPU model, opaque file observation | No runtime timer for structural counters | parameters, logical byte, file byte | Used with QA checkpoint expected SHA | P11 structural/resource evidence |
| measure_utilization + UtilizationProtocol | Public | Sampled CPU process time and optional NVIDIA telemetry | CPU counters, read-only NVIDIA CLI | perf_counter_ns CPU intervals, separate tool-wall domain | percent, core equivalents, sample gaps | Inspected, fixture utilization NOT_MEASURED | P16 observational evidence |
| profile_operators / profile_layers | Public | Instrumented isolated eager profile | Pinned WorkloadRecipe CPU/CUDA precision contract | Profiler CPU us, CUDA trace/event domain and worker synchronization | operator/layer us, captured events, bounded memory fields | Inspected, NOT_RUN here | P13/P14 instrumented evidence |
| analyze_flops | Public | Analytic supported ATen operation model from captured calls | Recipe or OperatorProfileResult | Offline derivation, recipe input may launch capture | MACs/FLOPs and coverage | Inspected, fixture counts NOT_MEASURED | P12 structural estimates with coverage |
| measure_training | Public | Trainer-integrated timing | Trainer-supported CPU/CUDA contract | perf_counter_ns host stages and selected-stream CUDA events | step/epoch s, samples and counters | Inspected, direct QA clone timings used | P19 training evidence |
| measure_training_detailed / analyze_training_throughput | Public | Instrumented stage timing and derived throughput | Supported Trainer/precision contract | Host ns and selected-stream event domain | stage s and images/s | Inspected, NOT_RUN here | P20/P22 scoped instrumented evidence |
| measure_training_memory / measure_training_memory_isolated | Public | Logical tensors, RSS and supported allocator counters | Bounded native training profiles | Boundary observations, scoped allocator reset in owned isolated window | byte by separate memory domain | Inspected, CUDA saved-activation gate retained | P23 Basic and experimental full scopes |
| compare_results | Public | Recorded-condition compatibility and information | Validated CPU/CUDA snapshots | Offline, no new timer | compatibility fields, independent memory eligibility | Used six CPU pairs | Architecture/state differences allowed by design |
| EvidenceRecord / EvidenceIndex / compare_evidence | Public | Validated source projection and explicit scope comparison | Typed result metadata | Source-native clocks retained | feature/unit/domain/context | Used with file SHA verification | Consistency and caller-declared associations |
| advise / performance difference contracts | Public | Explanations and descriptive recommendations | Saved typed evidence | Offline | metric direction and limited suggestions | Existing report adapter used, zero model edits | Advisor evidence remains scoped |
| create_report / PresentationReport / report_experiment | Public | Presentation and existing-store projection | Saved typed results, tr/en language | No intrinsic latency metric, QA host timer around call/export | JSON/Markdown/HTML and raw provenance | Used both languages | presentation_only true, authoritative_evidence false |
| ExperimentManager create_run / log_metric / add_artifact / verify_artifact | Public | Lifecycle, explicit scalar logging and immutable expected hashes | QA-only local store | Caller measurement and logged axes | value/unit/source, SHA256 and status | Four completed runs, 128 checks | Store association and artifact consistency |
| hyso analysis list / show / plan / run | CLI | Catalog, planning and explicitly admitted execution | Selected feature/source/device policy | Delegates to actual operation timer | Feature-specific metrics | Help inspected, execution remains selected Python QA | Same native feature catalog, no P01–P35 suite launch |
| hyso reports / experiments / compare | CLI | Saved report/store/condition operations | Retained files and store | Offline or delegated producer | Exports, verification and comparison | All help paths pass installed probes | Existing command-specific boundaries |
| _telemetry.cpu_observe / gpu_observe | Private | OS process-time and read-only CLI adapters | Windows/Linux/NVIDIA adapters | Process-time and CLI query clocks | counters, percent and byte | Inspected, no direct public calls | Implementation support only |
| _device_diagnostics / _profile_worker / _Snapshot | Private | Provenance, capture guards and state checks | Exact implementation profiles | Source-specific synchronization | Validation, inventory and cleanup facts | Inspected, invoked only through public CPU APIs | Private implementation boundary |

## Actual source entry points

The benchmark exposes BenchmarkRunner, BenchmarkResult, compare_results and
StateMutationError. The class constructor takes explicit device/precision,
one input source, warmup_iterations, measure_iterations, repeats, seed and
optional model metadata. CPU fp32/fp64 and CUDA fp32/amp_fp16 are its accepted
precision strings. Model movement stays caller-controlled. Source code supplies
the exact constructor signature. [Benchmark runner](../../hyso/benchmark/runner.py),
[result and comparison](../../hyso/benchmark/result.py)

CPU timing wraps the model forward. CUDA event timing drains the device at
repeat start, records selected-stream start/end events and synchronizes the
end event. Memory probing is an explicit separate forward with allocator peak
reset side effects. The base result's optional CPU RSS/FLOPs fields have explicit
unavailable statuses. Actual RSS comes from the separate resource API.
[Inference APIs](../../hyso/performance/inference.py), [memory APIs](../../hyso/performance/memory.py)

measure_memory(workload=None, protocol=MemoryProtocol()) records native current
working-set RSS and lifetime HWM using GetProcessMemoryInfo on this host.
The sampler's observed maximum and lifetime HWM retain distinct scopes.
It has bounded observation counts and a trusted synchronous callable contract.
The external stage controller supplies the child deadline used in this QA.
measure_footprint records unique tensor-object logical bytes and opaque artifact
length/hash. Tracemalloc has zero use in this acceptance.
[Memory implementation](../../hyso/performance/memory.py)

ProfilingProtocol bounds isolated capture counts, trace/result sizes and optional
resource limits. profile_operators accepts a WorkloadRecipe. profile_layers and
analyze_flops can derive from a validated OperatorProfileResult. FLOP/MAC models
operate over observed ATen calls with operator coverage and a versioned counting
convention. Instrumented layer/operator timings retain their observer overhead
and attribution limits. No profiler recipe was executed for these fixtures,
so fixture FLOPs/MACs are NOT_MEASURED.
[Profiling APIs](../../hyso/performance/profiling.py), [operation models](../../hyso/performance/_operation_models.py)

TrainingTimingProtocol and stage-detail APIs instrument existing Trainer
operations. Host timing, CUDA event timing, loader wait, optimizer attempts and
completed-image throughput have separate scopes. This task measures small
forward/backward and optimizer operations directly on its dedicated CPU QA clone,
with its raw samples retained as supplements.
[Training timing](../../hyso/performance/training.py), [stage detail](../../hyso/performance/training_details.py)

TrainingMemoryProtocol uses the in-process never-reset policy. The isolated API
has an explicit separate scope for owned reset windows. Basic logical tensor,
RSS and allocator observations retain their supported native profiles.
Production CUDA saved-activation instrumentation remains disabled and B4 remains
open. Current GPU admission was blocked before any CUDA context or worker.
[Training memory](../../hyso/performance/training_memory.py),
[isolated training memory](../../hyso/performance/isolated_training_memory.py)

measure_utilization samples process CPU times and optional read-only NVIDIA
telemetry with interval/count bounds. GPU utilization belongs to its physical
device observation window. This stage retains the read-only safety observation,
with model-attributed utilization NOT_MEASURED.
[Utilization API](../../hyso/performance/utilization.py), [private adapters](../../hyso/performance/_telemetry.py)

## Reporting and measurement support boundary

EvidenceRecord.from_json requires feature_id and expected_sha256. Its native
protocol_sha256 comes from the native source protocol/configuration. The external
QA protocol hash is retained in the QA summary and store config. Applying that
external hash as a native protocol declaration produces native_scope_mismatch,
as preserved by the initial excluded setup attempt.
[Evidence contract](../../hyso/analysis/evidence.py)

Existing native P01/P02/P10/P11 result classes feed the analysis and presentation
adapters. Forward/backward, SGD step, checkpoint IO and construction samples in
this QA driver have no newly invented native result type. Their JSON summaries
carry measured units, raw arrays and limitations. ExperimentManager registers
these artifacts through its existing generic artifact support.
[Report adapter](../../hyso/reporting/report.py), [store API](../../hyso/experiments/manager.py)

compare_results validates shared timing conditions and returns compatibility.
It allows different architecture and weight identities. Same-model regression
requires an explicit state identity policy. EvidenceScope supplies the tested
native state-SHA mismatch rejection. The six accepted architecture pairs share
hardware profile, precision, batch/shape, timer, warmup and repetition settings.
Physical host identity and uncontrolled clock/load/cache/thermal conditions
remain limitations of those recorded profiles.
[Comparison implementation](../../hyso/benchmark/result.py)

The CLI's performance entry is the existing analysis feature catalog/planning/
execution surface. Root, analysis, reports, experiments and compare help routes
exit zero on both installed packages. API inventory and installed logs retain
the exact available options. The inspected command names and options come from the actual parser.
[Analysis CLI](../../hyso/cli/analysis.py)

## Observation categories

Structural metadata includes exact parameter counts and logical parameter bytes.
FLOPs/MACs are NOT_MEASURED for these fixtures. Runtime performance includes raw
forward samples, completed-image windows and separate train-clone operations.
Resources include process RSS, sampled RSS maximum, process lifetime HWM and
checkpoint file lengths. Workflow overhead includes construction, checkpoint
IO, evaluation, comparison and report export. Native typed consistency,
caller-declared association, QA supplement, retained historical observation,
NOT_MEASURED and BLOCKED_SAFETY retain their explicit evidence levels.

`QA/correctness-proof.json` validates the numeric projections and unit formulas.
`QA/cuda-decision.json` binds the CUDA block to the actual read-only current
inventory and inspected history. The [matrix](stage13-5-16-benchmark-matrix.md)
contains all resource values, counts and negative cases.
