# HYSo 0.2.0 release candidate preparation

Unpublished 0.2.0 candidate finalized on 2026-11-28. No tag or release has been
created. See [changelog](CHANGELOG.md), [installation and workflow
guide](docs/guides/first-release.md) and the packaged `hyso-guide` for usage.

The first release combines PyTorch classification model construction, explicit
training/evaluation, bounded measurements, P01–P35 evidence interpretation,
pairwise comparisons, read-only Advisor and multilingual reports. Explicit export
supports ONNX, TorchScript, ExportedProgram and ONNX precision transformations.
Analyzer/Advisor do not rewrite the user's model. Construction, training and export
operate only under their separately invoked contracts.

## Compatibility and migration

The dependency profile is Python >=3.10,<3.13, Torch >=2.5.1,<2.6 and torchvision
>=0.20.1,<0.21. Actual release validation is Windows x64/Python 3.10 with Torch
2.5.1+cpu; retained bounded CUDA evidence uses Torch 2.5.1+cu121. Python 3.11/3.12
fit the upstream Torch family but are not independently certified. Other operating
systems and devices are not certified by these results.

The former open-ended Torch >=2.2.2/torchvision >=0.17.2 dependency bounds are
narrowed for this release to the exercised API family. Create a fresh environment;
do not upgrade or repair a working CUDA/TensorRT environment for this release.
Canonical Python APIs and CLI entry points remain intact. Commercial packages
retain the reviewed component/profile projection; development M3/M4 families are
not silently added. No automatic package install or model download is added.

`deployment` remains a CPU alias; use `deployment-cpu` or `deployment-gpu`
explicitly. Install one ONNX Runtime distribution per environment. CPU precision
tools use `precision`; the GPU distribution already includes those tools, so do
not combine GPU deployment with CPU extras. `onnx-dynamo` is explicitly optional.
TensorRT is a separate experimental extra and is not a base dependency.

## Evidence scopes and known limits

| Retained evidence | Outcome | Scope |
|---|---|---|
| Base product acceptance | 658 CPU PASS / 36 GPU PASS | Recorded models/runtime/protocol only |
| U01–U18 user acceptance | 30/30 PASS | Recorded user journeys |
| Earlier export acceptance | 4 CPU PASS / 3 GPU PASS | Includes E02_CUDA/E05_CUDA and admission |
| Export engineering | 45 CPU PASS | Source-bound engineering proofs |
| Dependency/readiness regression | 24 PASS | Earlier readiness snapshot |
| E13 CpuOnly owned acceptance | 7/7 PASS | CPU scope only; no native TRT promotion |

These scopes overlap and are not combined into one new PASS total. New packaging,
installed API and clean installation results are separate release-preparation
checks. Historical evidence files are immutable; documentation changes do not
retroactively certify every artifact or current source file.

- TensorRT E03/E04 and dependent E13 native scenarios remain BLOCKED/NOT_TESTED
  because the four installed DLLs disagree with wheel RECORD. No hash/signature
  exception is accepted. TensorRT is experimental and unvalidated, with no claim
  of production-validated support.
- FP16 CPU P35 may fail original-node coverage despite separate numerical checks.
  Cast insertion/rewrite is not relabeled as complete node mapping.
- Native C++, AOTInductor, OpenVINO and TFLite are outside the first release.
- P18 advanced hardware counters, P23 CUDA saved activation/B4, P31 CUDA long
  runs and P33 advanced energy remain under their existing deferred/unsupported
  safety policy. `full_cuda_acceptance=false`.
- Synthetic calibration, short smoke tests and sampled comparisons are not real
  task-quality or general hardware/performance guarantees.

## Release preparation and publication

The version remains 0.2.0. Installed API inspection covers 26 public entries;
actual Pylance completion/signature interaction remains NOT_TESTED. Wheel/sdist
payload verification covers 190 Python modules, packaged guides and py.typed.
Clean base and optional CPU smoke evidence is recorded separately in
[the final preparation report](docs/reports/release-preparation-f01-f06-20261128.md).
No historical P01–P35, U01–U18 or E01–E13 acceptance suite is rerun for this step.

F07/PyPI still requires account/project authorization and confirmation that the
0.2.0 version can be published. F08/GitHub Release still requires the reviewed
commit and tag. Artifacts and checksums are prepared locally; commit, push, tag
and publication are separate owner actions. No prerelease or version change is
applied. Experimental TensorRT native validation stays outside the validated CPU
release scope, with its existing BLOCKED/NOT_TESTED results retained.

The final API usability pass expands the earlier 26-entry inspection to the
shipped public surface. Builder component help includes actual defaults, scalar
bounds, channel/group relationships and output geometry. Literal overloads expose
registered keyword names; training option maps have static key definitions.
See [the API/IDE report](docs/reports/public-api-ide-20261128.md) for the final
installed-package and actual IDE verification results; the earlier NOT_TESTED
entry above describes the previous F01 snapshot, not an implied new IDE result.
