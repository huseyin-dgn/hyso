# Changelog

## 0.2.0 — 2026-11-28 (unpublished)

- Public ModelBuilder/GraphModelBuilder, Trainer/TrainingConfig, evaluation,
  benchmark, model comparison and read-only Advisor/reporting workflows.
- Explicit ONNX, TorchScript and ExportedProgram export; ONNX FP16/INT8,
  user-supplied calibration, cohort validation, deployment benchmarks and
  compatible cross-runtime comparisons. Export CLI and saved-result reporting.
- Four-language offline user guide and tr/en/de/es JSON/Markdown/HTML reports.
- Separate CPU/GPU deployment extras and explicit Torch distribution selection.
- Include the shared native identity helper required by installed CPU ONNX
  runtime verification in the commercial package inventory.
- Installed API and release wheel/sdist checks are recorded in the release
  preparation evidence, separately from historical product acceptance.

TensorRT remains experimental and unvalidated. Its DLL RECORD integrity blocker
is not waived. Native C++, AOTInductor, OpenVINO and TFLite are outside the first
release. FP16 CPU P35 original-node mapping is limited. Advanced deferred CUDA
profiles remain excluded; `full_cuda_acceptance=false`.

Version stays 0.2.0 during preparation. This entry is not a publication or tag.

Release preparation verifies installed public API signatures/docstrings, typed
package data, the commercial wheel/sdist inventory and offline documentation.
Clean Windows/Python 3.10 CPU installation evidence is kept separately from the
historical acceptance suites. Actual Pylance completion/signature interaction
remains NOT_TESTED; Python introspection is not an IDE certification.
