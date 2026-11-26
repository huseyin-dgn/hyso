# U15–U18 product experience acceptance

This separate, serial CPU/offline campaign reads retained evidence. It never
constructs or executes a model, loads weights, queries CUDA, builds an engine,
installs packages or compiles C++. Existing U01–U14 and CPU/CUDA acceptance
directories are sealed as preserved evidence, including failed attempts.

```powershell
.\run_final_acceptance.ps1 -ProductExperience -PrepareOnly
.\run_final_acceptance.ps1 -ProductExperience
.\run_final_acceptance.ps1 -ProductExperience -Status
.\run_final_acceptance.ps1 -ProductExperience -Resume
```

Only an explicitly named failed/blocked case can be retried, for example:

```powershell
.\run_final_acceptance.ps1 -ProductExperience -Resume -Revalidate 'tests/test_product_experience_scenarios.py::test_U17'
```

Preparation is not acceptance. Results live under
`outputs/product-experience-acceptance-20261126`, separately from U01–U14.
The existing pre-import Windows Job, host campaign lock, controller seals,
source/artifact hashes, heartbeat, memory limits and scope-bound cleanup apply.
Unknown cleanup or ownership still stops execution. No automatic retries.

| Case | Check | Workload |
| --- | --- | --- |
| U15 | Public aliases, `__all__`, resolved signatures, return types, `py.typed` metadata | None |
| U16 | Python help, CLI `--help`, all 70 offline topics in four languages, invalid language rejection | None |
| U17 | Retained U08 comparison integrity and U09 Advisor evidence; 35 metric explanations; JSON/Markdown/HTML in four languages | None |
| U18 | 21 components, 35 performance features, ten deployment technologies, evidence vs source presence | None |

The machine-readable inventory is
`docs/acceptance/product-experience-inventory.json`. Each component links source,
public API, tests, guide and retained CPU/CUDA evidence. P-feature rows include
localized measurement interpretation and exact bounded CUDA artifact references.
No general hardware certification follows from those records.

Autocomplete checks cover discovery, inline annotations and runtime hint resolution.
No real IDE/IntelliSense session or external type checker was run. Dynamic metadata,
factories and legacy inputs still use `Any`; this is not strict whole-package typing
certification. Existing legacy import signatures and `hyso.models.__all__` remain.

The offline guide, report, Advisor and comparison presentation support tr/en/de/es.
Python docstrings and argparse help are primarily English. Free-form retained source
text remains unchanged. Translation never changes measurements, units or evidence IDs.
The Advisor recommendation resolver supports P01/P02/P03/P08/P10/P11/P26; other
features have explanations/measurement requests, without invented optimization gains.
Measured, derived, estimated, unavailable, unsupported and missing observations keep
their own recorded statuses. Suggestions are manual and do not modify user models.
Incomparable cohorts/protocols do not establish a winner.

## Export and deployment preinventory

No HYSo-owned conversion API currently implements any of these ten technologies.
External framework capability is not HYSo verification.

| Technology | Existing HYSo-related path | Missing verification/work |
| --- | --- | --- |
| PyTorch → ONNX | U13 external `torch.onnx.export` factory; actual ORT CPU/CUDA node events and P35 placement | Public converter, paired output equivalence, conversion performance |
| ONNX → TensorRT | No observed integration | Converter, engine admission, loader, inference/equivalence/performance |
| PyTorch → TensorRT | No observed integration | Explicit converter and admitted runtime evidence |
| TorchScript / LibTorch C++ | No validated deployment path; Benchmark rejects scripted models | Artifact provenance and actual C++ execution |
| torch.export | No observed integration | Export/load/equivalence/analysis integration |
| AOTInductor C++ | No observed integration | Artifact provenance and actual C++ compilation/execution |
| FP32 → FP16 | P28/U07 native AMP quality/operator observations | Converted artifact, target loader and equivalence |
| FP32 → INT8 | P28 CPU dynamic Linear recipe | General deployment converter and target/equivalence/performance evidence |
| OpenVINO | No observed integration | Converter, runtime, equivalent outputs, analysis |
| TFLite | No observed integration | Converter, runtime, equivalent outputs, analysis |

P35 host/provider observations are not a conversion benchmark or cross-runtime
equivalence test. No C++ runtime support is claimed from a serialized model alone.
The inventory records dependencies, tests and missing work independently per row.

`full_cuda_acceptance=false` remains. P18 advanced counters, P23 CUDA saved activation
lifetime/B4, P31 CUDA 7200s and P33 advanced energy/thermal remain deferred/closed.
This preparation does not reopen them or repeat historical acceptance workloads.
