# First release: installation, workflows and limits

HYSo 0.2.0 is a Python library for building PyTorch classification models,
explicit training/evaluation, P01–P35 performance evidence, model comparisons,
read-only advice and reports. Analyzer and Advisor do not rewrite the caller's
source code, architecture or weights. ModelBuilder constructs a requested model;
Trainer changes weights when explicitly run; Export creates a separate inference
artifact under its preservation and no-clobber contract.

## Install into a new environment

The release profile is Python >=3.10,<3.13 with Torch >=2.5.1,<2.6 and torchvision
>=0.20.1,<0.21. The verified release host is Windows x64 with Python 3.10.
Python 3.11/3.12 and other operating systems are not independently certified here.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install dist/hyso-0.2.0-py3-none-any.whl
.venv\Scripts\hyso --help
.venv\Scripts\hyso-guide --language en --topic getting-started.first-run
```

Choose the Torch distribution before installing HYSo. For an independent CUDA
12.1 environment use the matching official pair:

```powershell
python -m venv .venv-cuda
.venv-cuda\Scripts\python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
.venv-cuda\Scripts\python -m pip install dist/hyso-0.2.0-py3-none-any.whl
```

These commands follow the [official Torch 2.5.1 distribution table](https://pytorch.org/get-started/previous-versions/).
CUDA build, driver, cuDNN, provider, device and actual DLL identity still need to
match the selected workflow. Installing CUDA Torch alone is not GPU acceptance.
Do not run developer acceptance scripts as a normal installation step. No import
automatically installs a package, fetches model weights or requests a workload.

## Optional dependencies

| Extra | Dependencies | Intended scope |
|---|---|---|
| none | Torch/torchvision, NumPy, Matplotlib, Seaborn, scikit-learn, tqdm, torchsummary | Base Python/CPU workflows |
| deployment or deployment-cpu | ONNX 1.20.0, ONNX Runtime 1.23.2 | ONNX checking and CPU deployment |
| deployment-gpu | ONNX 1.20.0, ONNX Runtime GPU 1.23.2 on Windows/Linux | Separate ORT CUDA environment; actual admission required |
| onnx-dynamo | ONNX 1.20.0, onnxscript 0.5.6 | Explicit Dynamo ONNX export; no hidden fallback |
| precision | ONNX 1.20.0, ONNX Runtime 1.23.2 | CPU ONNX FP16 and dynamic/static INT8 tools |
| tensorrt | ONNX 1.20.0, TensorRT CUDA 12 package 10.13.3.9 on Windows/Linux | Experimental and unvalidated; native acceptance blocked |

For a built local wheel, install extras using, for example,
`python -m pip install "./dist/hyso-0.2.0-py3-none-any.whl[deployment-cpu,onnx-dynamo]"`.
For a published version, use `python -m pip install "hyso[deployment-cpu]==0.2.0"`
only after that version is actually published.

**Install exactly one ONNX Runtime distribution per environment.** Do not combine
deployment-gpu with deployment/deployment-cpu/precision. The GPU distribution
contains the conversion tools too; install ONNX plus that distribution alone for
its precision tools. See the [official ORT package rule](https://onnxruntime.ai/docs/get-started/with-python.html).
pip extras cannot express mutual exclusion, so a separate environment is the
supported way to switch. Do not repair a shared namespace by copying DLLs or
altering RECORD. Import discovery from `hyso.export.availability()` is not proof
of a functioning native backend.

## Public API and examples

| Workflow | Public entry point | Usage and boundaries |
|---|---|---|
| Tiny CPU model/training | `ModularCNN`, `Trainer`, `TrainingConfig` | [README](../../README.md), [training settings](training-config.md) |
| External PyTorch model | `BenchmarkRunner(model, ...)`, explicit native capture | [Python workflow](python-first-workflow.md), [analysis](unified-analysis.md) |
| Torchvision model | `torchvision.models.resnet18(weights=None)` passed as an ordinary Module | No weight download; pretrained weights require the caller's explicit request |
| ModelBuilder | `hyso.ModelBuilder` | [Registry and Builder](model-registry-builder.md); commercial component set |
| GraphBuilder | `hyso.GraphModelBuilder` | [Named DAGs](custom-cnn-graph-builder.md); select `GraphTensorOutput` for a tensor classifier |
| Evaluation | `hyso.evaluation.evaluate_classification` | [File image example](file-image-classification.md); explicit labels/split |
| Checkpoint/resume | `Trainer.fit(save_last_to=...)`, fresh trainer with `resume_from=...` | [CPU Graph and resume](python-first-workflow.md#graph-training-and-saved-results); same original total epochs and exact supported config/loader required |
| P01–P35 | `hyso.performance`, `hyso.analysis` | [Catalog and evidence](analysis-evidence-contract.md), offline guide `analysis.p01` through `analysis.p35` |
| Model comparison | `analysis.ModelComparisonSource`, `analysis.compare_models` | [Comparison](two-model-comparison.md); saved compatible evidence, no inferred global winner |
| Advisor | `hyso.advisor.recommend`, `hyso.advisor.advise` | [Advisor reporting](advisor-reporting.md); advice does not run experiments |
| Reports | `hyso.reporting.create_report` | [Reporting](result-reporting.md); tr/en/de/es, JSON/Markdown/HTML |
| Export | `hyso.export` | [Complete export API](model-export.md); ONNX, TorchScript, ExportedProgram |
| Terminal | `hyso --help`, `hyso export --help` | [CLI](cli.md); explicit execution/trust flags |

## A small explicit deployment example

Install the CPU deployment extra first. Use a fresh output directory; output
names and sidecars are refused if they already exist.

```python
from pathlib import Path
import torch
from hyso import export as ex

out = Path("deployment-demo")
out.mkdir()  # A new directory owned by this explicit example.
model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2)).eval()
x = torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2) / 8
source = ex.onnx(model, x, out / "model.onnx")
script = ex.libtorch(model, x, out / "model.pt")
program = ex.torch_export(model, x, out / "model.pt2")
candidate = ex.Runtime(source)
reference = ex.Runtime(model)
validation = ex.validate(reference, candidate, [x], dataset_id="synthetic-example")
baseline = ex.benchmark(reference, x, warmup=1, iterations=3)
measured = ex.benchmark(candidate, x, warmup=1, iterations=3)
comparison = ex.compare(baseline, measured, validation=validation)
ex.save(comparison, out / "comparison.json")
for language in ("tr", "en", "de", "es"):
    for format, suffix in (("json", "json"), ("markdown", "md"), ("html", "html")):
        (out / f"comparison-{language}.{suffix}").write_text(
            ex.render(comparison, format=format, language=language), encoding="utf-8")
```

Three benchmark samples demonstrate usage, not a performance claim. Cross-runtime
comparison can remain incompatible if workload or measurement conditions differ.
Read its status and numerical validation separately. FP16 conversion is mixed
precision; CPU P35 may lack complete original-node mapping. INT8 static QDQ requires
bounded, caller-supplied calibration feeds matching named graph inputs. Set
`representative=True` only for genuinely representative data; batch count does not
prove that. See [precision/calibration examples](model-export.md#e07e12-deployment-api).

## Support and troubleshooting

Verified historical scopes remain separate: 658 CPU/36 GPU base acceptance,
30/30 user acceptance, 4 CPU/3 GPU export acceptance, 45 export engineering proofs,
24 readiness regressions and 7/7 E13 CpuOnly owned acceptance. They describe their
recorded snapshots and are not added into a new universal PASS total.
`full_cuda_acceptance=false` remains in force. New release installation smoke
results are listed separately in the release preparation report.

TensorRT E03/E04 and dependent native E13 cases are BLOCKED/NOT_TESTED. TensorRT
code is retained and optional; no production-validated TensorRT support is claimed.
Native C++, AOTInductor, OpenVINO and TFLite are outside this first release.
P18 advanced counters, P23 CUDA saved activation/B4, P31 long CUDA runs and P33
advanced energy remain deferred or unsupported under the existing support policy.

For an import error, check the active interpreter with `python -m pip show hyso`
and `python -m pip check`. Run outside the checkout to avoid source shadowing.
Missing ONNX dependencies produce a dependency failure or an explicitly unverified
runtime stage; a generated artifact alone is not deployment PASS. Install only the
extra needed in that environment. Unknown CUDA DLL identity, incomplete ownership,
cleanup failures and resource-limit violations remain safety stops with no native
automatic retry. See [support policy](release-support.md) and [export contracts](model-export.md).
