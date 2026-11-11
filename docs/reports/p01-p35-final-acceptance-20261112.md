# HYSo P01–P35 final acceptance — 2026-11-12

**70/70 PASS in the declared acceptance scopes: Model A 35/35; Model B 35/35.**

| Model | PASS | Actual uninterrupted P31 work (seconds) | P32 | suite_complete |
|---|---|---|---|---|
| A | 35/35 | 7200.0032194 | PASS | true |
| B | 35/35 | 7200.0012437 | PASS | true |

Model A is a composite evidence continuation: preserved prior PASS results plus the new layer, P31 and P32 results and offline CUDA provider reassessment. Its scoped child manifests retain their original `suite_complete=false` values; no historical manifest was rewritten. Model B is a new native full run with terminal `suite_complete=true`.

## Scope and exclusions

- P23 accepts **Basic memory only**: logical parameter/gradient/optimizer bytes, RSS/HWM and owned CUDA allocator counters. Extended saved-activation lifetime remains experimental/PARTIAL; production CUDA saved-activation hooks remain disabled.
- P30 uses a **separate task-appropriate trained image/video adapter**. It does not certify Model A/B MNIST camera-recognition quality.
- P17 covers finite declared candidates, not a global capacity maximum. Precision, pipeline, energy and device results apply only to their recorded protocols; board energy is not model-exclusive.
- P32 is descriptive memory-trend analysis at warmups 0/60/300 seconds, not proof of leak absence.
- P31 durations are separate closed work windows; the historical approximately 723-second low-memory stop contributes no time.

## Corrections and validation

The retained control CV failure was 27.2792681% against the unchanged 25% limit. Its physical jitter source could not be uniquely determined. Fixed CPU affinity, ABOVE_NORMAL priority and predeclared 20 warmups produced control/instrumented CVs of 6.0990%/15.9926% for A and 8.9676%/13.2566% for B. Five pairs, five analyzed windows per arm/worker, two priming windows, the 25% CV limit and 0.5 relative uncertainty limit were retained. No timing window was discarded. Calibration uses synthetic Shared, not model-specific CNN overhead.

CUDA ONNX acceptance now requires the complete 3-warmup + 10-measured ledger and reads coverage from the canonical evidence assessment. The earlier 276 tests + 63 subtests receipt was verified; six targeted adapter tests passed. A native provider evidence was reassessed offline without another native run; B passed its actual native provider run.

## Retained evidence index

Raw measurements stay outside Git. Resolve the following aliases under `%LOCALAPPDATA%`; no user-specific absolute paths or large raw payloads are copied here:

- `FINAL`: `hyso-full-resume-20261111-v2` — `final-results.md`, `final-results.json`, `provider-correction.json`, resource admission and preservation records.
- `A_PRIOR`: `hyso-full-resume-20261111-v1/Model-A` and `hyso-full-models-20261111-v1/Model-A` — prior manifests, stage results and preserved raw evidence.
- `A_LAYER`: `FINAL/Model-A-layer/stages/06.layer`.
- `A_LONG`: `FINAL/Model-A-long/stages` — `10.stability` journal/result/supervisor and `11.memory` trend results.
- `B`: `FINAL/Model-B` — full manifest, final matrix, stage results and all native stage evidence.

Authoritative detailed final-results.json SHA256: `86a014ef432ed96a552bfc35a9c65c3bb78714c117e72708ce7b3c8051795bde`.

The detailed local index binds each feature to exact artifact paths and SHA256 values, including original FAIL/BLOCKED records and new results. All three new controller preservation checks passed; production source/checkpoint/weight preservation passed. No long-run tests were repeated for this Git documentation step.

## Model feature matrices

### Model A

| Feature | Status | Evidence scope |
|---|---|---|
| P01 Inference Latency — P50/P95/P99 | PASS | A_PRIOR |
| P02 Real Throughput / FPS | PASS | A_PRIOR |
| P03 End-to-End Latency | PASS | A_PRIOR |
| P04 Preprocessing / Postprocessing Time | PASS | A_PRIOR |
| P05 Cold Start & Model Loading | PASS | A_PRIOR |
| P06 Batch Size Sweep | PASS | A_PRIOR |
| P07 Resolution Sweep | PASS | A_PRIOR |
| P08 Precision Benchmark | PASS | A_PRIOR |
| P09 GPU Peak Memory | PASS | A_PRIOR |
| P10 CPU Peak RAM | PASS | A_PRIOR |
| P11 Model Footprint | PASS | A_PRIOR |
| P12 FLOPs / MACs | PASS | A_LAYER + A_PRIOR native capture |
| P13 Layer-wise Profiler | PASS | A_LAYER + A_PRIOR native capture |
| P14 Operator / Kernel Profiler | PASS | A_LAYER + A_PRIOR native capture |
| P15 CPU–GPU Transfer Analysis | PASS | A_PRIOR |
| P16 GPU / CPU Utilization | PASS | A_PRIOR |
| P17 Maximum Feasible Batch Size | PASS | A_PRIOR |
| P18 Arithmetic Intensity & Roofline Analysis | PASS | A_PRIOR |
| P19 Epoch / Training Step Time | PASS | A_PRIOR |
| P20 Forward / Backward / Optimizer Breakdown | PASS | A_PRIOR |
| P21 DataLoader Bottleneck Analysis | PASS | A_PRIOR |
| P22 Training Throughput | PASS | A_PRIOR |
| P23 Training Memory Analysis | PASS | A_PRIOR |
| P24 Time to Target Accuracy | PASS | A_PRIOR |
| P25 Gradient / AMP Diagnostics | PASS | A_PRIOR |
| P26 Accuracy / Precision / Recall / F1 | PASS | A_PRIOR |
| P27 Robustness Benchmark | PASS | A_PRIOR |
| P28 Quantization / Precision Accuracy Drop | PASS | A_PRIOR |
| P29 Accuracy–Speed–Memory Trade-off & Regression | PASS | A_PRIOR |
| P30 Real Camera FPS & Dropped Frames | PASS | A_PRIOR |
| P31 Long-Running Stability | PASS | A_LONG |
| P32 Memory Leak / Memory Growth Analysis | PASS | A_LONG |
| P33 Power & Energy per Inference | PASS | A_PRIOR |
| P34 Multi-Device Comparison | PASS | A_PRIOR |
| P35 Execution Provider Fallback Detection | PASS | A_PRIOR + FINAL/provider-correction.json (offline reassessment) |

### Model B

| Feature | Status | Evidence scope |
|---|---|---|
| P01 Inference Latency — P50/P95/P99 | PASS | B/stages |
| P02 Real Throughput / FPS | PASS | B/stages |
| P03 End-to-End Latency | PASS | B/stages |
| P04 Preprocessing / Postprocessing Time | PASS | B/stages |
| P05 Cold Start & Model Loading | PASS | B/stages |
| P06 Batch Size Sweep | PASS | B/stages |
| P07 Resolution Sweep | PASS | B/stages |
| P08 Precision Benchmark | PASS | B/stages |
| P09 GPU Peak Memory | PASS | B/stages |
| P10 CPU Peak RAM | PASS | B/stages |
| P11 Model Footprint | PASS | B/stages |
| P12 FLOPs / MACs | PASS | B/stages |
| P13 Layer-wise Profiler | PASS | B/stages |
| P14 Operator / Kernel Profiler | PASS | B/stages |
| P15 CPU–GPU Transfer Analysis | PASS | B/stages |
| P16 GPU / CPU Utilization | PASS | B/stages |
| P17 Maximum Feasible Batch Size | PASS | B/stages |
| P18 Arithmetic Intensity & Roofline Analysis | PASS | B/stages |
| P19 Epoch / Training Step Time | PASS | B/stages |
| P20 Forward / Backward / Optimizer Breakdown | PASS | B/stages |
| P21 DataLoader Bottleneck Analysis | PASS | B/stages |
| P22 Training Throughput | PASS | B/stages |
| P23 Training Memory Analysis | PASS | B/stages |
| P24 Time to Target Accuracy | PASS | B/stages |
| P25 Gradient / AMP Diagnostics | PASS | B/stages |
| P26 Accuracy / Precision / Recall / F1 | PASS | B/stages |
| P27 Robustness Benchmark | PASS | B/stages |
| P28 Quantization / Precision Accuracy Drop | PASS | B/stages |
| P29 Accuracy–Speed–Memory Trade-off & Regression | PASS | B/stages |
| P30 Real Camera FPS & Dropped Frames | PASS | B/stages |
| P31 Long-Running Stability | PASS | B/stages |
| P32 Memory Leak / Memory Growth Analysis | PASS | B/stages |
| P33 Power & Energy per Inference | PASS | B/stages |
| P34 Multi-Device Comparison | PASS | B/stages |
| P35 Execution Provider Fallback Detection | PASS | B/stages |
