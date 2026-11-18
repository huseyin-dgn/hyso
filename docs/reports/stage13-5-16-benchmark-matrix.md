# Stage 13.5.16 benchmark matrix

QA root and frozen settings are defined in the
[acceptance report](stage13-5-16-cpu-cuda-performance-acceptance.md).
Only `cpu-complete-1` and `cpu-complete-2` contribute to these tables.

## Protocol and raw samples

Protocol SHA-256 `22751e3e1e4b843832ab9822df201b97f50e46ce78f65e905d3c46090915b5dc`. Source revision
`1791814e3190215136c288ea55b1af25fffbd61e`, seed 13516, CPU fp32, one Torch
thread. Each row has an independent native result ID. Warm repeats share the
same model/input and cache history. The two complete child PIDs are
30504 and 20668.

Each run stores model-specific `inference-b1.json`, `inference-b8.json`,
`inference-b32.json`, matching throughput/tail files, resource/footprint files,
checkpoint files and evaluation JSON. `summary.json` contains raw workflow
sample arrays. Times use milliseconds, file/memory sizes use bytes and rates
use completed images per second. Sample SD uses ddof 1.

## Warm inference distributions

| Run | Model | Batch | n | Min ms | p50 ms | p90 ms | p95 ms | Mean ms | SD ms | Serial service images/s | Completion images/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | sequential | 1 | 120 | 0.1436 | 0.152 | 0.17488 | 0.187475 | 0.157117 | 0.013398 | 6364.697146 | 5728.82602 |
| 1 | sequential | 8 | 120 | 0.2889 | 0.3041 | 0.42178 | 0.496945 | 0.332112 | 0.063515 | 24088.223117 | 23349.596976 |
| 1 | sequential | 32 | 120 | 0.3251 | 0.35665 | 0.43609 | 0.492485 | 0.373724 | 0.060071 | 85624.647412 | 83308.384208 |
| 1 | graph | 1 | 120 | 0.312 | 0.3403 | 0.42045 | 0.521235 | 0.365613 | 0.066051 | 2735.136244 | 2572.358295 |
| 1 | graph | 8 | 120 | 0.4697 | 0.5178 | 0.58059 | 0.63171 | 0.535465 | 0.065946 | 14940.285546 | 14799.809452 |
| 1 | graph | 32 | 120 | 0.5883 | 0.66995 | 0.76001 | 0.82368 | 0.684873 | 0.057966 | 46723.968422 | 43974.870194 |
| 2 | sequential | 1 | 120 | 0.1509 | 0.1557 | 0.21643 | 0.27148 | 0.172331 | 0.041159 | 5802.79211 | 5763.494983 |
| 2 | sequential | 8 | 120 | 0.2856 | 0.29735 | 0.39306 | 0.443865 | 0.322248 | 0.054946 | 24825.638678 | 23600.03835 |
| 2 | sequential | 32 | 120 | 0.3242 | 0.3589 | 0.40945 | 0.506865 | 0.37205 | 0.056809 | 86009.9449 | 81705.604494 |
| 2 | graph | 1 | 120 | 0.3096 | 0.34395 | 0.40497 | 0.46108 | 0.359319 | 0.060167 | 2783.041075 | 2643.561935 |
| 2 | graph | 8 | 120 | 0.4892 | 0.50925 | 0.57254 | 0.601405 | 0.522118 | 0.035366 | 15322.197075 | 14644.392952 |
| 2 | graph | 32 | 120 | 0.6217 | 0.6408 | 0.72438 | 0.745115 | 0.656423 | 0.039444 | 48749.028828 | 41216.841373 |

Pooled n equals 120 per cell, with three warm repeats of 40 measured forwards.
P90/P95 are descriptive interpolated order statistics with correlated samples.
The built-in strict tail-policy result has insufficient_samples. P95 for
operations with fewer than 100 samples is omitted from the QA summary.
Serial service rate is `batch × n × 1000 / sum(latency_ms)`.
Completion rate comes from three distinct bounded 40-batch windows, with each
completed-image ledger independently checked. Their measurement scopes differ.

## First forward of a fresh model per geometry

| Run | Model | Batch | First forward ms | n |
| --- | --- | --- | --- | --- |
| 1 | sequential | 1 | 0.698 | 1 |
| 1 | sequential | 8 | 1.4868 | 1 |
| 1 | sequential | 32 | 1.1913 | 1 |
| 1 | graph | 1 | 1.927 | 1 |
| 1 | graph | 8 | 1.0963 | 1 |
| 1 | graph | 32 | 2.3208 | 1 |
| 2 | sequential | 1 | 0.9328 | 1 |
| 2 | sequential | 8 | 1.3084 | 1 |
| 2 | sequential | 32 | 0.9884 | 1 |
| 2 | graph | 1 | 0.5593 | 1 |
| 2 | graph | 8 | 0.8743 | 1 |
| 2 | graph | 32 | 1.5302 | 1 |

These values include first forward dispatch for a fresh instance in an already
imported process. Process cold-start latency remains NOT_MEASURED. A single
observation supplies no dispersion or tail estimate.

## Construction, training and workflow distributions

| Run | Model | Operation | n | Min ms | p50 ms | p90 ms | Mean ms | SD ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | sequential | construction | 5 | 2.6633 | 2.739 | 3.68452 | 3.02752 | 0.708078 |
| 1 | sequential | forward_backward | 20 | 0.7523 | 0.8057 | 1.05966 | 0.86075 | 0.127423 |
| 1 | sequential | optimizer_step | 20 | 0.0886 | 0.0919 | 0.11876 | 0.09945 | 0.015326 |
| 1 | sequential | checkpoint_save | 5 | 4.1481 | 5.2915 | 5.39342 | 4.89546 | 0.628424 |
| 1 | sequential | checkpoint_load | 5 | 9.6785 | 9.9609 | 10.86386 | 10.17456 | 0.716969 |
| 1 | sequential | evaluation | 3 | 6.3543 | 6.607 | 14.27596 | 9.718167 | 5.608967 |
| 1 | graph | construction | 5 | 4.5258 | 4.693 | 5.10068 | 4.8172 | 0.262155 |
| 1 | graph | forward_backward | 20 | 1.066 | 1.18645 | 1.35764 | 1.223235 | 0.143282 |
| 1 | graph | optimizer_step | 20 | 0.0948 | 0.09965 | 0.13578 | 0.1085 | 0.018401 |
| 1 | graph | checkpoint_save | 5 | 26.5942 | 28.0771 | 28.58344 | 27.78496 | 0.88173 |
| 1 | graph | checkpoint_load | 5 | 21.8883 | 22.9957 | 24.21678 | 23.07608 | 1.101194 |
| 1 | graph | evaluation | 3 | 10.6709 | 10.6779 | 10.70686 | 10.687633 | 0.023186 |
| 2 | sequential | construction | 5 | 2.6036 | 2.7946 | 3.63334 | 3.04854 | 0.560757 |
| 2 | sequential | forward_backward | 20 | 0.7465 | 0.8016 | 0.85181 | 0.8028 | 0.051325 |
| 2 | sequential | optimizer_step | 20 | 0.0874 | 0.09355 | 0.10449 | 0.09623 | 0.010456 |
| 2 | sequential | checkpoint_save | 5 | 4.4143 | 5.2898 | 5.80446 | 5.2349 | 0.604422 |
| 2 | sequential | checkpoint_load | 5 | 9.0985 | 9.7291 | 10.82164 | 10.013 | 0.766672 |
| 2 | sequential | evaluation | 3 | 7.0718 | 7.1156 | 7.4452 | 7.238333 | 0.251468 |
| 2 | graph | construction | 5 | 4.5574 | 4.7821 | 4.89222 | 4.73642 | 0.164248 |
| 2 | graph | forward_backward | 20 | 1.1441 | 1.33885 | 1.73434 | 1.408775 | 0.204617 |
| 2 | graph | optimizer_step | 20 | 0.1062 | 0.11915 | 0.20303 | 0.141025 | 0.062743 |
| 2 | graph | checkpoint_save | 5 | 26.2996 | 27.7182 | 31.52534 | 28.66602 | 2.636407 |
| 2 | graph | checkpoint_load | 5 | 20.9893 | 22.5466 | 23.44342 | 22.3987 | 1.075429 |
| 2 | graph | evaluation | 3 | 10.0995 | 12.6629 | 12.96034 | 11.932367 | 1.598158 |

Forward/backward includes model forward, cross-entropy and backward. Gradient
zeroing happens before that timer. Optimizer step includes only SGD.step after
real gradients are computed. Two untimed update warmups precede twenty recorded
updates. This dedicated clone changes state and the inference model retains
its original state hash. Evaluation includes the native classification API,
four loader batches, sample accounting and result construction.

Checkpoint save includes native checkpoint metadata and atomic file publication.
Checkpoint load includes the trusted expected-SHA computation, strict native
restore and device/schema checks. Post-load equality checks occur after timing.
Exact CPU outputs match the original model. These timings carry the actual
QA directory filesystem/cache state and five same-sized unique checkpoint files.

## Reporting and comparison overhead

| Run | Operation | n | Min ms | p50 ms | p90 ms | Mean ms | SD ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | report create + JSON/MD/HTML | 3 | 3402.2458 | 3500.2627 | 3530.31318 | 3480.111433 | 70.00028 |
| 2 | report create + JSON/MD/HTML | 3 | 3491.5904 | 3693.9038 | 3911.7854 | 3717.25 | 238.192347 |
| 1 | compare batch 1 | 5 | 0.5841 | 0.7019 | 0.73062 | 0.67294 | 0.066852 |
| 1 | compare batch 8 | 5 | 0.5156 | 0.6162 | 0.68418 | 0.60818 | 0.078231 |
| 1 | compare batch 32 | 5 | 0.5273 | 0.5813 | 0.60514 | 0.5687 | 0.039054 |
| 2 | compare batch 1 | 5 | 0.5483 | 0.5584 | 0.70294 | 0.60698 | 0.092916 |
| 2 | compare batch 8 | 5 | 0.5273 | 0.551 | 0.60532 | 0.56188 | 0.039714 |
| 2 | compare batch 32 | 5 | 0.5399 | 0.5551 | 0.62508 | 0.57494 | 0.051082 |

Each report sample creates one expert English report over 16 existing analysis
records and exports JSON, Markdown and HTML to new filenames. Turkish exports
are produced from the same analysis without rerunning measurement. Resource,
store-registration and export overhead stay outside the inference timer.

## Descriptive architecture differences

| Run | Batch | Graph minus sequential p50 ms | Graph minus sequential completion images/s | Recorded timing conditions |
| --- | --- | --- | --- | --- |
| 1 | 1 | 0.1883 | -3156.467725 | compatible |
| 1 | 8 | 0.2137 | -8549.787524 | compatible |
| 1 | 32 | 0.3133 | -39333.514014 | compatible |
| 2 | 1 | 0.18825 | -3119.933048 | compatible |
| 2 | 8 | 0.2119 | -8955.645398 | compatible |
| 2 | 32 | 0.2819 | -40488.763121 | compatible |

All six native compatibility reports pass the recorded CPU timing-condition
checks. Architecture, parameter count and operator structure remain comparison
variables. P50 differences and completion-rate differences are signed Graph
minus sequential observations. General model ranking and causal optimization
claims remain unestablished.

## Resource observations

| Run | Model | RSS n | Baseline RSS byte | Final RSS byte | Delta byte | Sampled max RSS byte | Lifetime HWM byte | Parameter byte | Checkpoint byte |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | sequential | 4 | 608448512 | 608464896 | 16384 | 608464896 | 666046464 | 21864 | 24776 |
| 1 | graph | 7 | 610738176 | 610811904 | 73728 | 610811904 | 677081088 | 21352 | 26752 |
| 2 | sequential | 5 | 609673216 | 609734656 | 61440 | 609734656 | 666714112 | 21864 | 24776 |
| 2 | graph | 9 | 611479552 | 611565568 | 86016 | 611565568 | 677912576 | 21352 | 26752 |

RSS is current-process Windows working set. Baseline is the start of a separate
100-forward batch32 resource window. Delta is final minus baseline. Sampled max
is the largest recorded RSS sample. Lifetime HWM is the native process-wide
high-water counter including prior work. A model-exclusive interval peak remains
NOT_MEASURED. Sampling adds observer overhead. Raw timestamps and actual gaps
are saved in memory.json. CPU utilization and all CUDA allocator fields remain
NOT_MEASURED. Logical parameter bytes and serialized checkpoint bytes have
independent accounting scopes.

## Negative acceptance matrix

| ID | Case | Observed handling | Evidence and limits |
| --- | --- | --- | --- |
| 1 | Zero repetitions | Rejected | Constructor count validation, three new cases |
| 2 | Negative/invalid duration | Rejected | Negative/zero saved latency and QA summary |
| 3 | NaN/Inf latency | Rejected | Finite native schema and QA summary checks |
| 4 | Wrong time unit | PARTIAL | Wrong timer rejected, ms is fixed by schema, coherent relabeling has no independent unit attestation |
| 5 | Warmup mixed into samples | Rejected | Sample count mismatch and controlled-clock forward ledger |
| 6 | Wrong batch | Rejected | Native shape/batch mismatch |
| 7 | Different shape | Incompatible | Recorded-condition comparison |
| 8 | Different precision | Incompatible | Actual CPU fp32/fp64 result comparison |
| 9 | Different device | Rejected | Native EvidenceScope mismatch |
| 10 | Different state | Rejected within explicit identity scope | Native state SHA constraint, general architecture comparator permits differing states |
| 11 | Missing synchronization | Rejected | Required timing metadata |
| 12 | Missing provenance | Rejected | Required result provenance group |
| 13 | Empty benchmark | Rejected | Empty measured rows |
| 14 | Timeout/partial benchmark | Rejected | Partial result schema and CPU controller timeout probe |
| 15 | Missing memory counter | Rejected or unavailable | Required group removal rejected, unsupported counters retain null/reason in existing tests |
| 16 | Unsupported CUDA | Rejected | CPU-only monkeypatched availability test with zero native CUDA |
| 17 | Crash/abnormal exit | PARTIAL | CPU exit 7 and crash-shaped untyped payload rejected, native crash injection NOT_TESTED |
| 18 | Missing artifact | Observed missing/rejected | Installed reference smoke and EvidenceRecord missing-path check |
| 19 | Changed hash | Observed mismatch/rejected | Installed reference smoke and expected-SHA reader check |
| 20 | Historical JSON live authority | Presentation only | Raw dictionaries rejected and saved presentation authority remains false |

The source selection passes 358 cases, including 25 new cases. Three additional
CPU-only controller probes retain actual timeout/null exit code, explicit exit 7
and zero exit without a completed summary. Their acceptance is false. Those
manual probes and the 212 audit assertions remain separate counts. CUDA-native
timeout/crash execution has count zero.

## Excluded preparation records

`cpu-1` and `cpu-final-1` exit 1 on protocol-scope association and loader-type
errors respectively. Both lack a complete terminal summary. Their partial raw
files remain in QA and contribute zero accepted cells. Corrections and exact
script/lock versions are retained. The frozen measurement protocol SHA is equal
across both successful child runs.
