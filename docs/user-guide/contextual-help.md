# UX-03: Hover documentation and offline contextual help

HYSo 0.2.0 now pairs its public Python documentation with the same packaged
four-language guide delivered by UX-02. Reading help never starts a factory,
training run, measurement or CUDA query. Only an explicit `docs open` command
asks the operating system to open a browser.

```console
hyso docs list
hyso docs show models.graph-builder --lang tr
hyso docs open models.graph-builder --lang tr
hyso docs open analysis.p23 --lang de
hyso docs open analysis.p23 --lang de --dry-run
```

`--lang` / `--language` accepts `tr`, `en`, `de`, `es` and defaults to `tr`.
`list` returns stable IDs and localized titles. `show` extracts the selected
language and shared examples from the actual bundled HTML as inert text.
`open --dry-run` prints the URI without opening a browser. Existing `--json`
envelopes, error codes, `--debug`, and the standalone `hyso-guide` remain available.
Invalid/missing IDs and unknown languages fail rather than opening another topic.

Python callers can resolve the current installation path without importing HYSo
or Torch:

```python
from hyso_guide import guide_uri, list_topics, show_topic, topic_for_api

topic = topic_for_api("hyso.GraphModelBuilder")
uri = guide_uri(topic, lang="tr")   # Returns a URI; does not open it.
text = show_topic("analysis.p23", lang="de")["text"]
```

The URI uses the installed `user-guide.html`, the language query and the exact
fragment, for example `?lang=de#analysis.p23`. No installation directory or
unpublished HTTPS documentation address is hardcoded in hover documentation.
The four languages and examples remain physically inside one offline HTML file.

## Documented API scope

The authoritative symbol inventory is
[`api-topics.json`](../../hyso/data/api-topics.json). Its schema is
`hyso.api_topics`, version 1, independently of existing model/checkpoint/report
schemas. It records 1,643 symbol/topic bindings, including aliases and component
names, with 775 distinct source declarations across 91 source files. Of these,
90 files received documentation/type metadata changes; `hyso_guide.py` also
implements the offline reader helpers.

| User operation | Real API | Guide topic |
| --- | --- | --- |
| Sequential model construction | `ModelBuilder`, `BuiltClassifier`, `BuiltSequential` | `models.model-builder` |
| Connected graph construction | `GraphModelBuilder`, `GraphModel`, `GraphTensorOutput` | `models.graph-builder` |
| Component discovery and validation | `ModelRegistry`, `ComponentSpec`, `ParameterSpec` | `models.support` and the relevant `components.*` topic |
| Training preparation | `TrainingConfig`, `TrainingSession`, `TrainingResult` | `training.configuration` |
| Explicit training | `Trainer` | `training.trainer` |
| Analyzer workflow | `plan_analysis`, `preflight_analysis`, `execute_analysis`, plans, bindings and evidence contracts | `analyzer.workflow` |
| Native measurements | Public `hyso.performance` functions, protocols and result types; `BenchmarkRunner` / `BenchmarkResult` | Relevant `analysis.pXX` topic |
| P23 isolated training memory | `measure_training_memory_isolated`, `TrainingMemoryRecipe`, isolation policy/result | `analysis.p23` |
| Model comparison | `compare_models`, `ModelComparisonSource`, `ModelComparisonResult` and advisory comparison contracts | `comparison.models` |
| Architecture and structure | `compare_architectures`, `capture_model_structure`, `compare_model_structures`, `audit_model_architecture` and retained types | `comparison.architecture` |
| Advisor | `advise`, `recommend`, request/goal/finding contracts and display functions | `advisor.recommendations` |
| Metric interpretation | `Metric`, `MetricScope`, native resolution and shared result exports | `metrics.interpretation` |
| Reporting | `create_report`, `report_experiment`, `PresentationReport`, `feature_explanations` | `reporting.outputs`, `metrics.interpretation` |
| Experiment history | `ExperimentManager`, `ExperimentRun` and their public lifecycle/artifact/metric methods | `experiments.manager` |

“Analyzer”, “Comparison”, “Advisor” and “Reporting” describe workflows/namespaces;
this work does not invent classes with those names. The inventory includes all
callable exports of `hyso.analysis`, `hyso.advisor`, `hyso.reporting` and
`hyso.benchmark`, all 85 source-registry component names and four graph fusion
names. Each retained measurement mapping identifies an existing guide topic;
shared result-export methods refer to metric interpretation rather than falsely
claiming a unique measurement scope.

## Boundaries and missing documentation

No requested HYSo-owned workflow or registered constructor remains without a
topic mapping in this audited scope. PyTorch-owned constructors (21 distinct
upstream classes) and inherited `torch.nn.Module` methods keep their upstream
docstrings and signatures; HYSo does not overwrite vendor documentation. Their
HYSo component usage is discoverable through the registry/topic inventory.
Private implementation utilities outside the public facades are not claimed as
documented public APIs.

Dynamic inputs intentionally retain `Any` where a narrower static contract
would be inaccurate. A resolved type hint is not a proof that every possible
caller input is valid. Native validators, unavailable-status fields, resource
budgets and authorization gates remain authoritative. `TrainingConfig` already
uses `TYPE_CHECKING`-only Torch/Dataset imports; runtime `get_type_hints` for its
three preparation methods needs those names supplied in the namespace. Static
editors can follow the existing imports without moving Torch imports to runtime.

Development-only model descriptions remain labeled as such in the guide;
their topic mappings do not install those models into the commercial wheel.
Four-language help does not expand the technical reporting API beyond its
existing Turkish/English support. B4 remains unresolved and GPU admission is
unchanged.

## Editor verification and limits

Docstrings use structured argument, return, error and example sections with type
annotations. The existing `py.typed` marker and package version are preserved.
Python 3.10 syntax parsing and hint resolution are checked without constructing
models. Jedi static inference resolves the root imports for `ModelBuilder`,
`GraphModelBuilder`, `TrainingConfig`, `Trainer`, `ExperimentManager`, and the P23
function to their documented definitions and correct guide topics.

These checks are useful editor metadata evidence, not a Pylance or PyCharm GUI
test. Pylance is installed locally, but its editor hover UI was not exercised;
PyCharm UI behavior was not verified. Neither editor's clickable-link behavior
is claimed. The plain topic IDs and offline CLI commands remain the supported
route regardless of editor link rendering. No `file:` URL dependent on a
particular installation is embedded in a docstring.

## Validation

[`test_contextual_help.py`](../../tests/test_contextual_help.py) checks docstring
fields, annotations, real facade/component coverage, exact model/analysis topic
bindings, generated constructor-example argument binding, all 70 topics in all
four languages, invalid IDs, browser opt-in, dry-run, errors and installed-wheel
access. Examples are parsed/bound, never executed as documentation code.

The UX-03 QA receipt records the exact test runs, static editor results,
executable-AST comparison against the starting commit, and preservation hashes.
Existing CLI, guide and selected packaging regressions are included. No GPU,
P01–P35 acceptance workload or full suite is part of this validation.

UX-03 result: **128 distinct CPU/static/API/CLI cases passed, 0 failed**.
The initial selected regressions passed 125 cases; the subsequent affected-guide
and packaging run passed 48, and the final contextual-help run passed all 27
cases after the last reader/type changes. These overlapping runs are counted
once per test identity. The final run also verifies real Manager return shapes
in a temporary metadata-only store, without starting training.

The executable ASTs of the 90 documentation-only source files match starting
commit `d3f70cf60183a73e48d6e0a86f20c09f280b2cdc` after excluding docstrings, type
annotations and the added typing imports. Parameter names/defaults, algorithms
and native guards are preserved. All 47 protected user-file hashes, 19 artifact
hashes and prior QA/report hashes match the UX-03 baseline. The existing central
HTML, topic registry and `py.typed` bytes are unchanged.
