# E01–E12 model export and deployment

First release: TensorRT is **experimental and unvalidated**. E03/E04 and dependent
E13 native scenarios remain BLOCKED/NOT_TESTED because DLL RECORD provenance is
unresolved. This does not block the independently accepted CPU scope. Native C++,
AOTInductor, OpenVINO and TFLite are outside the first release. FP16 CPU P35 node
coverage remains limited; numerical validation has a separate status.
For end-user setup, use the [installation guide](first-release.md); the acceptance
commands later in this document are developer-only.

`hyso.export` is an explicit engineering API, independent of Analyzer/Advisor.
It does not automatically modify or optimize a caller's model. Exports use a
separate inference copy and preserve registered tensor/alias/device/dtype state,
gradients, per-module mode, Python configuration, input content and RNG. Trusted
Python models remain trusted code; deepcopy is not a security sandbox. Opaque
unregistered model attributes are rejected rather than silently omitted.

```python
import hyso
config = hyso.export.ExportConfig(atol=1e-5, rtol=1e-4)
result = hyso.export.onnx(model, example_input, "model.onnx", config=config)
result.verify()
saved = hyso.export.ExportResult.from_json("model.onnx.export.json")
```

The current resident export profile is dense, finite same-dtype floating CPU
inputs, an external eager Module, ModelBuilder, or supported GraphBuilder tensor
outputs. A Tensor, flat positional tensor tuple, or named GraphBuilder mapping is
accepted. Model/input/artifact ceilings are explicit. There is no implicit model
download, package installation, dtype conversion or source device migration.
The Python CPU API is synchronous trusted code; its byte admission is not an OS
quota or a guaranteed timeout for arbitrary user forwards. Use the owned acceptance
controller for process memory/time containment. Heavy native work always uses the
existing pre-import Windows Job worker and bounded resource admission.
Logical validation/transfer outputs are capped at 32 MiB; complete reloadable
metadata/evidence is capped at 16 MiB. P35's own graph/evidence bounds also apply.
Exceeding these bounds is an explicit resource rejection, never a partial PASS.

| API | Artifact | Execution evidence |
| --- | --- | --- |
| `onnx` | Checked ONNX; symbolic axes and external-data manifest | Actual ORT CPU nodes and same-run outputs; opt-in owned CUDA validation |
| `tensorrt` | TensorRT 10.x serialized engine | Owned GPU build/load/execute and ONNX CPU reference equivalence when inputs supplied |
| `pytorch_tensorrt` | Checked ONNX → engine, linked provenance | Reuses the ONNX artifact after source model/input identity checks |
| `libtorch(method="trace"/"script")` | TorchScript `.pt` | Real Python reload/forward; opt-in owned CUDA; C++ is separate |
| `torch_export` | ExportedProgram `.pt2` | Real save/load/module execution and native dynamic constraints |
| `aotinductor` | Native compilation/runtime deferred to next release | Platform guard retained; never pretends `.pt2` is compiled C++ |

Torch 2.5.1 exporter signatures are used. ONNX `backend="legacy"` explicitly sets
`dynamo=False`. `backend="dynamo"` is opt-in, requires onnxscript, disables the
legacy fallback and records the actual capture strategy report. Unsupported
version/strategy or conversion failure is explicit. Inference export is not a
training graph. Opsets are bounded to the implemented 13–20 option profile.
Example names must match input/output arity. Dynamic shapes use:

```python
dynamic_shapes = {"image": {0: {"name": "batch", "min": 1, "max": 8}}}
```

ONNX symbolic axes do not enforce min/max ranges at runtime. ExportedProgram uses
native Dim constraints. Trace covers example branches, not arbitrary data-dependent
control flow; script may reject unsupported source constructs. Trace warnings are
retained in metadata. TorchScript is the legacy LibTorch deployment approach.

Nine statuses stay independent: dependencies, admission, conversion, artifact,
runtime loading, runtime execution, equivalence, preservation and cleanup.
`deployment_verified` requires all nine PASS plus finite same-shape/dtype outputs
within the explicit elementwise `atol + rtol * abs(reference)` bounds. A checked
artifact with absent ORT or a disabled runtime request is not deployment PASS.
Small-sample equivalence is not cohort quality/performance validation (E09–E11).
P35 host node events are not GPU kernel timing. Unexpected CPU execution is rejected.

Artifacts use marker-owned sibling staging. Existing paths are refused by default;
explicit overwrite backs up and can restore only the selected main/own metadata.
External data files receive unique generation names and individual hashes. The
metadata sidecar is the multi-file commit point; readers must use/verify it. Main
file visibility alone is not a completed generation. Successful overwritten old
external files remain retained; no unrelated directory or unverified generation is
deleted. Intermediate ONNX retention is mandatory in this E04 profile; requesting
cleanup is rejected explicitly. No process or directory is adopted retroactively.

NativeConfig requires a physical GPU UUID and pre-admission canonical DLL hashes.
The worker validates them before and after execution, uses no-breakaway Job
ownership, timeout, bounded host/process limits, the existing allocator fraction
ceiling, and a capped TensorRT GPU allocator. Workspace is a separate tactic budget,
not total VRAM. TensorRT 10.x fixed shapes or complete min/opt/max profiles and FP32
or hardware-capable FP16 flags are accepted. Static signed symmetric INT8 QDQ is
also accepted; dynamic integer operators and implicit calibrators are excluded. FP16 is an engine
configuration, not proof every layer used FP16 instructions. Engine compatibility
is the recorded GPU/compute capability/TRT/platform, not cross-device portability.
Native failure/OOM/unknown cleanup stops safely with no automatic retry.
Native exports also take a nonblocking per-user/GPU file lock. The acceptance
campaign additionally uses the existing host-wide campaign lock; neither implies
exclusive ownership of third-party GPU workloads.

`examples/export/libtorch` supplies CMake and a bounded float32 C++ loader. It reads
the serialized model plus raw input/reference files, runs a real forward and emits
finite/equivalence/device metadata. `--cuda` requires a compatible CUDA LibTorch SDK.
Python `torch.jit.load` never counts as C++ acceptance. CMake alone is insufficient;
missing configured compiler/SDK leaves C++ validation BLOCKED. No compilation has
been run during engineering. Torch 2.5 provides experimental private
`torch._inductor.aot_compile`; newer `aoti_compile_and_package` is absent here.
No Linux compiler adapter or Windows AOT support is advertised as validated.

The offline `export.overview` topic is localized tr/en/de/es. Python help and
technical errors remain English with stable `ExportError.code`/`message_key`.
Full CLI, Advisor, report localization and autocomplete expansion remains E12.
`py.typed`, public `__all__`, typed signatures and base guide lookup are included now.

## Separate user acceptance

```powershell
.\run_final_acceptance.ps1 -ExportAcceptance -PrepareOnly
.\run_final_acceptance.ps1 -ExportAcceptance
.\run_final_acceptance.ps1 -ExportAcceptance -Status
.\run_final_acceptance.ps1 -ExportAcceptance -Resume
```

Revalidate only explicit failures/blocks, for example `-Resume -Revalidate E02_CUDA`
or `-Resume -Revalidate 'tests/test_export_acceptance_scenarios.py::test_E05_CPP'`.
Normal resume starts unstarted independent cases and retains PASS. Unknown cleanup,
source/DLL mismatches and native crashes stop; no hidden risky retry occurs.
Native C++ and AOTInductor are deferred to the next release and excluded from
this campaign. Existing CMake scaffolding and platform guards remain in source.
CPU cases: E01, E02_CPU, E05_CPU, E06_EP. CUDA groups:
ADMISSION, E02_CUDA, E05_CUDA, E03_TRT, E04_TRT. Workloads are tiny and serial.
Missing dependencies/platforms are BLOCKED and do not prevent independent cases.
Evidence is separate in `outputs/export-acceptance-20261127`.

Original 658 CPU / 36 CUDA / 30 user PASS records and their failures/seals remain
historical evidence, not automatic acceptance of these new sources. The U18 export
inventory is the preserved pre-engineering snapshot; this document and the new
campaign describe E01–E06. `full_cuda_acceptance=false` and P18 advanced counters,
P23 saved activation/B4, P31 CUDA 7200s and P33 advanced energy remain closed.
E07–E12 development is described below. E13 real native acceptance remains separate.

Backend references: [Torch 2.5 export](https://pytorch.org/docs/2.5/export.html),
[Torch 2.5 AOTInductor](https://pytorch.org/docs/2.5/torch.compiler_aot_inductor.html),
[TensorRT 10 Python API](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/python-api-docs.html).

## E07–E12 deployment API

These are engineering capabilities, not a GPU acceptance certificate. No engine,
CUDA acceptance, GPU benchmark, package installation or historical campaign rerun
was performed while implementing them. Native C++ and AOTInductor remain excluded.

```python
from hyso import export as ex

source = ex.onnx(model, example, "model.onnx")
half = ex.fp16(source, "model.fp16.onnx", example_input={"input0": example})
dynamic = ex.quantize(source, "model.int8.onnx", example_input={"input0": example})
static = ex.quantize(
    source, "model.qdq.onnx", mode="static",
    calibration=user_calibration_iterator, dataset_id="user-dataset-version",
    representative=True, example_input={"input0": example})

reference = ex.Runtime(model)
candidate = ex.Runtime(dynamic)
validation = ex.validate(reference, candidate, user_cohort, precision="int8")
quality = ex.validate(reference, candidate, user_labelled_cohort,
                      precision="int8", labelled=True, top_k=2)
baseline = ex.benchmark(reference, example, warmup=2, iterations=10)
measured = ex.benchmark(candidate, example, warmup=2, iterations=10)
comparison = ex.compare(baseline, measured, validation=validation)
ex.save(comparison, "comparison.json")
loaded = ex.load("comparison.json")
text = ex.render(loaded, format="html", language="tr")
advice = ex.advise(loaded, language="de")  # also hyso.advisor.export_advice
# hyso.reporting.render_export shares the same renderer.
```

Resident execution accepts dense finite CPU inputs under existing byte budgets.
Each eager runtime uses the existing isolated model/state/RNG guard. Script and
ExportedProgram runtime loading is explicit trusted code. Flat multiple inputs and
outputs are supported; named ONNX feeds are matched by name. Cohorts may vary
supported dynamic dimensions; actual runtime shape/profile errors remain explicit.
Data-dependent output shapes are outside the TensorRT profile.

FP16 uses `onnxruntime.transformers.float16` with its default FP32 operation block
list plus caller additions. Input/output types stay FP32 by default; explicit
`keep_io_types=False` changes them. No automatic source-model dtype migration takes
place. The graph is mixed precision; neither FP16 weights nor a builder flag proves
every operation executes FP16 instructions. ORT's CPU precision cast rewrites can
prevent complete P35 node mapping. Actual numerical output metrics are retained,
but runtime execution/deployment stays BLOCKED in that case. Cross-dtype numerical
validation is separate from the existing exact-dtype deployment seal.

Dynamic INT8 is limited to selected MatMul/Gemm operations; it does not claim CNN
Conv quantization. Static INT8 uses MinMax calibration, signed symmetric QDQ and
selected Conv/MatMul/Gemm. No quantized operation means an explicit unsupported
error. Each calibration batch must contain all graph inputs with matching dtype,
rank, static dimensions, finite values and optional min/opt/max profile bounds.
Calibration accepts 2–64 batches by default, at most 32 MiB total; content hashes,
dataset ID and the caller's representativeness declaration are retained. Synthetic
test fixtures are explicitly nonrepresentative. TensorRT accepts this static QDQ
path without an implicit INT8 calibration flag; native behavior is NOT_TESTED here.

Validation reports absolute/relative max error, MAE, RMSE, cosine, shape/dtype and
NaN/Inf checks per output. Default atol/rtol: FP32=1e-5/1e-4,
FP16=1e-3/1e-2, INT8=0.1/0.1. These are configurable screening bounds, not universal
quality guarantees. Relative error uses a 1e-30 denominator floor. Unlabelled
classification scores can report prediction agreement, never accuracy/F1. Labelled
single-label multiclass cohorts additionally report top-1/top-k, macro precision,
recall and F1 across all score classes (zero division=0). Numerical and quality
statuses are independent; default allowed accuracy/F1 drop is 0.01. Empty,
oversized, invalid or truncated cohorts cannot PASS.

Benchmark statistics reuse `hyso.benchmark.result`. CPU timings cover the resident
runtime call including dispatch/output wrapping, excluding load, warmup, transfers,
validation and source isolation. Loading/setup has a separate duration. Throughput
is serial batch service rate, not concurrent server throughput. Percentiles are
linear interpolations of the retained samples. CPU RAM is optional sampled whole-
process RSS, not incremental model memory or a peak. Missing RAM/VRAM, transfers,
build or end-to-end observations are `null` with unavailable scope/reasons.
TensorRT native measurement uses the existing owned Job, exact engine/runtime/GPU
binding, bounded allocator, retained receipts and synchronized host enqueue timing.
It is not GPU-kernel-only timing. Native GPU code has not been executed here.

Comparison requires identical source identity, input content/shape/dtype, batch,
hardware, driver identity for GPU, timing boundary, warmup and sample policy.
Mismatches produce BLOCKED comparability and `speedup=null`. Precision may differ
while I/O remains comparable. Cross-boundary CPU vs GPU measurements cannot yield
a speedup claim. Whole-process RSS differences are not model-memory savings.

## Optional installations

Choose the intended CPU/CUDA Torch distribution first using the official
[PyTorch installer](https://pytorch.org/get-started/locally/). HYSo does not switch it.

| Extra | Direct additions | Scope |
| --- | --- | --- |
| deployment / deployment-cpu | onnx 1.20.0, onnxruntime 1.23.2 | CPU export/reference |
| deployment-gpu | onnx 1.20.0, onnxruntime-gpu 1.23.2 | ORT GPU, separate environment |
| tensorrt | onnx 1.20.0, tensorrt-cu12 10.13.3.9 | TRT full builder/runtime; CPU reference separately |
| onnx-dynamo | onnx 1.20.0, onnxscript 0.5.6 | Explicit modern Torch exporter |
| precision | onnx 1.20.0, onnxruntime 1.23.2 | FP16 converter and dynamic/static INT8 |

Do not install `onnxruntime` and `onnxruntime-gpu` together: they own the same Python
namespace. `precision` selects CPU ORT; a GPU environment should instead use its
single ORT GPU distribution, which contains the same conversion tools. TensorRT
does not require torch-tensorrt, PyCUDA, CuPy, nvcc or C++ headers in this API path.
Windows x64/Python 3.10 is the inspected native target. Extras on other platforms
do not imply a supported owned native execution adapter. Platform wheel and driver
compatibility must be checked before installation. HYSo import never installs
packages or accesses the internet.

Commands below are user-run installation examples, never executed by this task:

```powershell
conda create --name hyso_trt --clone torch_gpu  # only if this environment is absent
conda run -n hyso_trt python -m pip install ".[tensorrt,deployment-cpu]"
# For source checkout rather than a published HYSo release:
conda run -n hyso_trt python -m pip install "tensorrt-cu12==10.13.3.9" "onnxruntime==1.23.2"
conda run -n hyso_trt python -m pip check
```

Do not launch a second installer while one is running. Do not delete NumPy backup
directories automatically: `~umpy-*.dist-info` / `~umpy.libs` are evidence of an
interrupted/partial uninstall and require a separately authorized package repair.
During initial inspection, hyso_trt had NumPy 2.2.2 metadata plus 2.2.6 backup
metadata, and no TensorRT distribution metadata. Later read-only inspection found
10.13.3.9 binding/libs metadata, no top-level meta package, and all four native DLL
sizes differing from RECORD (each actual file was 9760 bytes larger). This task
did not start an installer. The discrepancy's cause has not been established;
the verifier rejects it without broadening the allowlist. Native readiness is BLOCKED.

TensorRT 10.13.3 officially lists CUDA 12.1 Update 1, Windows R537+, Python sample
versions 3.10–3.12 and PyTorch >=2.0. Driver 591.74 meets that CUDA-12 branch
requirement. NVIDIA lists ONNX **1.16.0** as tested for this TensorRT release;
ONNX 1.20.0 in this project is therefore not an officially tested pair. Parser and
operator compatibility must be validated on actual artifacts. No package downgrade
or runtime switch was performed. Sources:
[10.13.3 release notes](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.13.3.html),
[10.13 allocator docstrings](https://github.com/NVIDIA/TensorRT/blob/release/10.13/python/docstrings/infer/pyCoreDoc.h),
[10.13 Python binding](https://github.com/NVIDIA/TensorRT/blob/release/10.13/python/src/infer/pyCore.cpp),
[ORT quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

## CLI and E13 preparation

```powershell
hyso export --help
hyso export dependencies
hyso export inspect model.onnx.export.json
hyso export fp16 model.onnx.export.json --example inputs.npz --output half.onnx
hyso export quantize model.onnx.export.json --mode dynamic --example inputs.npz --output int8.onnx
hyso export quantize model.onnx.export.json --mode static --calibration batch1.npz batch2.npz --dataset-id my-data-v1 --output qdq.onnx
hyso export benchmark model.onnx.export.json --example inputs.npz --execute --trust-artifact --iterations 10
hyso export validate model.onnx.export.json --candidate int8.onnx.export.json --example inputs.npz --precision int8 --execute --trust-artifact
hyso export compare baseline.json candidate.json
hyso export report comparison.json --format html --language es --output comparison.html
hyso export advise comparison.json --language tr
hyso export convert --factory my_models:create_model --example inputs.npz --backend onnx --execute --trust-factory --output model.onnx

# Metadata-only E13 preparation. Choose a new output directory for each snapshot.
python -B tools/export_acceptance/development.py --prepare-only --conda-env hyso_trt --output outputs/e13-prepare-review

# Prepare an isolated owned campaign for the selected native interpreter.
$envs = conda env list --json | ConvertFrom-Json
$trtEnv = @($envs.envs | Where-Object { (Split-Path $_ -Leaf) -eq 'hyso_trt' })
if ($trtEnv.Count -ne 1) { throw 'hyso_trt must resolve uniquely' }
$trtPython = Join-Path $trtEnv[0] 'python.exe'
python -B tools/export_acceptance/runner.py --prepare-only --runtime-python $trtPython --campaign-root outputs/e13-owned-review --development-native
```

The selected interpreter/site applies only to native engine/development groups;
the original CPU/ORT runtime stays separately identified. Native DLL inventory
requires all three matching TRT distributions, SHA-256/size matching RECORD,
canonical paths inside the selected site and stable file identities. Reparse,
escaping, duplicate or mismatching members fail closed. Values are discovered,
never embedded as machine-specific paths/UUIDs/hashes in product source.

`development.py` does not alter old plans/checkpoints, import Torch/TRT, execute
tests or start GPU work. It inventories CPU/GPU cases, missing dependencies,
precision scope, existing budgets and hashes of preserved evidence. Changed export
sources require a reviewed evidence transition before real acceptance. The new
isolated campaign does not automatically adopt old PASS as current-source proof.
Do not run `--resume` until dependency repair and that transition are reviewed;
otherwise unrelated old PASS tests could be scheduled in the fresh campaign.

## Final E13 profile and dependency closure

The final profile reuses `tools/export_acceptance/runner.py --final` and the
existing Job/controller/checkpoint engine. It binds the previous 45 CPU engineering
JUnit cases to their recorded source hashes; it never invents owned acceptance
receipts for them. Only new closure/control-plane CPU cases, a fresh admission in
the selected TRT interpreter, and nine native cases are scheduled. Historical
E02_CUDA/E05_CUDA and P01–P35/U01–U18 cases are excluded from this selection.

```powershell
.\run_final_acceptance.ps1 -ExportFinalAcceptance -PrepareOnly
.\run_final_acceptance.ps1 -ExportFinalAcceptance
.\run_final_acceptance.ps1 -ExportFinalAcceptance -Status
.\run_final_acceptance.ps1 -ExportFinalAcceptance -Resume
```

The default Conda environment is `hyso_trt`. `-RuntimeEnvironment`, `-RuntimePython`
and `-AcceptanceRoot` provide explicit portable selectors. Status reads retained
state without environment discovery. A first user-started run prepares under the
existing manager lock if needed, then checks dependency readiness before GPU
admission. A failed or blocked native case requires explicit `-Resume -Revalidate
<exact-case>`; ordinary Resume never retries it. Source/dependency changes require
PrepareOnly and retain old receipts as historical. Unknown ownership/cleanup or
native safety violations stop the existing controller.

Native cases: E03_TRT, E04_TRT, E07_TRT_FP16, E08_TRT_QDQ, E09_TRT_COHORT,
E10_TRT_BENCHMARK, E11_TRT_COMPARISON, E03_TRT_DYNAMIC and E04_TRT_FP16.
The dynamic case must execute batch min/opt/max 1/2/4, not repeat one fixed shape.
The QDQ case uses explicitly nonrepresentative synthetic engineering calibration.
Task-quality tests use only their supplied labels and do not certify a real cohort.

Current closure: all three TRT 10.13.3.9 distributions and CUDA runtime 12.9.79
metadata exist. The four TRT DLLs still differ from RECORD in both size and SHA256.
Their NVIDIA signatures are valid; each PE certificate table is 9760 bytes and
starts at the recorded unsigned-size boundary. That explains the size delta, but
does not make the whole-file RECORD match. Neither signature trust nor stripped-
certificate hashes override RECORD. The final profile reports every discrepancy
and emits no usable partial native allowlist.

Torch's CUDA 12.1 cudart and the installed CUDA 12.9 cudart share a basename and
have different bytes. Each has independent valid RECORD provenance. The final
profile admits only the selected Torch cudart, keeps its site first, and explicitly
passes the recorded ORT reference site to both controller and nested worker. The
third-party TensorRT loader may load the alternate cudart by absolute path; actual
module paths/hashes/file identities are checked after that import and before any
Builder/engine call. Unexpected libraries stop the owned worker. Co-installation
or a compatible driver does not prove which runtime Windows loaded.

The two obsolete NumPy 2.2.6 remnants were moved outside `hyso_trt` into quarantine.
Active NumPy 2.2.2 files and identities are unchanged and its small CPU calculation
passes. TensorRT remains dependency BLOCKED: virtual unsigned-image hashing matches
the four RECORD entries, but the original DLL wheel is unavailable to establish
whether NVIDIA shipped that mismatch. Neither local RECORD edits nor signature-based
allowlist exceptions are permitted. No download or reinstall is required for this diagnosis.

E13 CPU acceptance can proceed independently with the same owned controller:

```powershell
.\run_final_acceptance.ps1 -ExportFinalAcceptance -CpuOnly -PrepareOnly
.\run_final_acceptance.ps1 -ExportFinalAcceptance -CpuOnly
.\run_final_acceptance.ps1 -ExportFinalAcceptance -CpuOnly -Status
.\run_final_acceptance.ps1 -ExportFinalAcceptance -CpuOnly -Resume
```

This scope uses a separate evidence directory, schedules no GPU/admission groups,
and does not discover or import TensorRT. It retains the 45 source-bound engineering
proofs and schedules only the new CPU scope regressions; the preceding 24 readiness
results remain historical evidence. It does not certify TensorRT, reclassify the
unsupported FP16 CPU P35 mapping, or change native safety gates. TensorRT stays
available but unvalidated; its removal or a release policy change needs a separate
decision. PrepareOnly executes no acceptance scenarios.
