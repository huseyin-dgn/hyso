# Stage 13.5.12 — ModelBuilder and Graph architecture compatibility

Execution: 2026-11-14; final closure: 2026-11-15 (Europe/Istanbul). Decision: **PASS within the explicitly tested CPU scope**. No runtime implementation change was required. This acceptance establishes construction, execution and integration contracts; it establishes no learning-quality threshold, universal component combination support or GPU qualification.

Companion evidence: [component matrix](stage13-5-12-component-matrix.md), [API and usability inventory](stage13-5-12-usability-inventory.md), [executable tests](../../tests/test_builder_graph_compatibility.py), [constructor metadata and fixtures](../../tests/fixtures/builder-graph-compatibility-v1.json), [five-scenario CPU example](../../examples/models/builder_compatibility_acceptance.py).

`QA` below means `C:\Users\hdgn5\AppData\Local\hyso-stage13-5-12-20261114`. All test temporary directories, builds, installations, experiment SQLite files, checkpoints, XML/logs and measured evidence live there. Python is `C:\Users\hdgn5\.conda\envs\torch_gpu\python.exe`. CPU fp32, one Torch thread, no CUDA initialization; selected geometry regressions also use CPU fp64. No external dataset download was performed.

## 1. Starting Git and protection audit

- Repository: `C:\Users\hdgn5\OneDrive\Belgeler\GitHub\hyso`.
- Branch: `feature/hyso-models-next`.
- Starting HEAD: `888175a6b79c0930064d7b3b9988dcc244a3ec90`.
- Starting parent: `a1b4d3e17474c5421865489adb3e24ea0c7c559e`.
- Index empty. Existing unrelated state: 2 tracked modifications plus 45 untracked files, 47 files in total. No unexpected difference from the previous closure's 746-file manifest.
- `QA/baseline.json` records raw file SHA-256 values, exact status, index and eleven existing commit/parent records before work. `protected47.json` preserves the separate dirty/forensic lists; overlapping protection groups are not additive file counts.
- Reviewed stage 13.5.6–13.5.11 acceptance/closure reports, the graph/component/architecture guides and dataset acceptance boundaries. The earlier partial receiver report remains historical; this stage uses the completed 13.5.11 closure as its baseline.

The continuation request reused this same live work and QA directory. Its expected historical HEAD remains the start above. By arrival, the verified three-file test/example commit had already advanced HEAD to `a5f3ea6a02d20805a858cfa634e7ec61962a3aa2`; four documentation paths were staged for final review. This is explained stage-owned progress, not an unrelated dirty-file discrepancy. Existing XML/logs, copied source hashes, inventory, 54-row matrix, CPU acceptance, archive digests, installed byte inventories and CLI proof were recovered and checked; the 316 tests and broad regressions were not needlessly rerun. Closure adds only documentation/proof detail and fixes generated blank-line whitespace.

## 2. Inspected implementation and public surfaces

The model directory inventory includes `builder.py`, `registry.py`, `graph.py`, `graph_ops.py`, `output.py`, `components.py`, `sampling.py`, `advanced.py`, `blocks.py`, `attention.py`, `WithSE.py`, `bcam.py`, `regularization.py`, `modular.py`, `fit.py`, `_profile.py`, `_development.py`, `m3_attention.py`, `m3_residual.py`, `architectures.py`, `m4_blocks.py`, `m4_registry.py`, `m4_config.py`, `m4_reparam.py`, `mobile_architectures.py`, `reparam_architectures.py` and `__init__.py` under `hyso/models`.

Connected contracts examined/tested include `hyso/architecture_comparison.py`, `architecture_audit.py`, `_architecture_rules.py`, `_architecture_coverage.py`, `structure.py`, `structure_comparison.py`, `model_checks.py`, `model_observations.py`, `training/config.py`, `checkpoint.py`, `evaluation.py`, `_evaluation_evidence.py`, `_evaluation_structure.py`, `_graph_receiver.py`, `analysis/graph_evidence.py`, `analysis/model_comparison.py`, reporting, ExperimentManager, public exports and CLI model commands. Private helpers implement contracts; their inspection does not promote them to public API.

## 3. Complete discovered component inventory

Source development Registry: **85** names = **72 non-model components + 13 complete-model profiles**. Installed commercial wheel and sdist: **57** names = **56 non-model components + modular_cnn**. The 28 deliberately excluded source-only names consist of 16 M3/M4 blocks and 12 named families. `_profile.py` selects development according to the presence of `_development`; these catalogs must not be conflated.

Four fusion operations, `add`, `multiply`, `concat`, `weighted_sum`, are separate Graph operators and are absent from both Registries. The commercial runtime file count remains **164**, a different quantity from catalog size.

The companion matrix gives all 85 real constructors, public registration, role, input/output conditions, fixture counts, signatures, JSON parameter whitelist/defaults/bounds/dependencies and execution limits. It separately lists **89 owned top-level class definitions**, including **25 without a direct Registry constructor alias**. Helpers, metadata/base classes, adapters, model wrappers and legacy standalone classes are not all executable components. No Builder support is inferred from their existence. `QA/full-inventory.json` retains the machine-readable discovery and measured properties; the committed fixture retains the complete Registry snapshot and commercial name set.

## 4. ModelBuilder support and API contract

The default constructor `ModelBuilder((C,H,W), num_classes)` produces a schema1 sequential classifier with an implicit global-average/linear classification output. `sequential(..., head=...)` explicitly selects schema2 classification, regression or embedding heads. `modular()` and source-only `named()` profiles construct complete models; complete families cannot be inserted as ordinary feature blocks.

Supported feature categories include standard/grouped/dilated/transpose/depthwise/separable/factorized convolution, available normalization, activations, residual/attention blocks, regularization, pooling and sampling. Exact names and conditional restrictions are in the matrix. Flatten is terminal schema2 feature usage. Linear Builder has no external skip edges, branch IDs or fusion operations; repeated component names are legal sequential entries.

`add()` validates the entire candidate chain transactionally; invalid scalar parameters, channel transitions and head composition leave the original configuration intact. `to_config()` returns detached plain data. `to_json/from_json` roundtrip architecture, not weights. Rebuilding the same config reproduces structure; equal random weights require an explicit seed or checkpoint. Tested seed reproducibility does not imply identical weights from successive unseeded builds.

Input is fixed positive CHW with dimensions at most 8192, dynamic batch N. Sequential chains allow at most 64 blocks and a conservative 10M parameter bound; development named families use separate conservative 200M bounds. Models construct on CPU; dtype/device movement is explicit via normal Torch APIs. Metadata estimates and activation warnings are not total RAM guarantees.

## 5. GraphModelBuilder support and API contract

Graph architecture JSON is `hyso.model_config` schema3 with named inputs, stable node IDs, ordered references and explicitly named outputs. Graph `.add()` validates component parameters and identifier rules; missing references, cycles, reachability and composed shapes complete at `preview`, `from_config` or `build`. A valid intermediate builder is not a finalized valid architecture.

Observed general DAG execution: sequential, branching, residual add, concat, three convolution branches, unequal branch depths, nested branching, multiply and learnable weighted sum. All nine new topology cases verify finite real forward/backward, input/used-parameter gradients, parameter registration, optimizer change, strict checkpoint state equality and equal restored eval outputs.

Limits include 64 nodes, 256 edges, fan-in 8, fan-out 16, 1–8 inputs and outputs; declared input layout is NCHW or NC with fixed non-batch geometry. Add/multiply require matching shapes rather than implicit broadcasting; concat obeys non-axis shape compatibility and excludes batch-axis concatenation. Closed configs reject arbitrary callables, dotted-import plugins and unknown fields.

Raw Graph supports multiple named inputs and dict outputs. The raw two-input Graph was differentiated and strictly restored. `GraphTensorOutput` selects an existing output and bridges a **single-input** Graph into Tensor classification workflows; it rejects multi-input Graph and incompatible binary output widths. No general multi-input TrainingSession bridge is certified.

## 6. Valid CNN combinations and bounded matrix

All 72 source non-model samples executed through Registry, ModelBuilder and unary Graph: **216 measured component/engine cases**. These are small concrete parameterizations, not all parameter combinations. Commercial installed profiles execute 56 × 3 = **168 positive component/engine cases**; the remaining 48 explicitly assert unsupported development names.

A deterministic four-factor universe contains 3 convolution choices (standard/grouped/depthwise), 3 activations (ReLU/SiLU/GELU), 2 normalizations (BatchNorm/GroupNorm) and 3 pooling choices (max/average/adaptive average): **54 combinations**. The 18 executed cases cover every pair of factor values. **36 are NOT_TESTED** because this is a bounded pairwise schedule. The exact rows and reason are committed in the component matrix and saved in `QA/combination-matrix.json`.

Closure matched all 18 actual source XML test IDs to their measured factor vectors and PASS outcome. The companion matrix lists those IDs and all six factor-pair sets: **45/45 distinct factor-value pairs covered**. `QA/pairwise-test-id-proof.json` binds this mapping to the source XML SHA-256. No extra combination or test-count inflation was introduced.

Additional topology/integration cases cover residual+normalization, branch+add/concat, Graph+classification head, both builders+TrainingSession, Graph+checkpoint and Graph+evaluation. These cases are separate from the 54-row universe. No exhaustive global architecture universe or total global untested-combination count is claimed. No UNKNOWN row is promoted to SUPPORTED.

## 7. Invalid configurations and failure phases

New negative tests cover missing reference, cycle, self-loop, unreachable unnamed node, missing output, add/concat shape mismatch, unary fan-in violation, head feature mismatch, duplicate node ID and batch concat axis. Scalar/composition cases include zero channels/kernel/dilation, invalid group divisibility, integer masquerading as boolean, invalid GroupNorm/PReLU/pixel-shuffle parameters, zero resize scale, zero classes and unknown component.

Invalid finalized Graphs produce structured bilingual diagnostics and are rejected before forward. Existing bounds/config/security regressions add unknown fields, malformed JSON, graph limits and unsupported node kinds. Complete named-model profiles are explicitly invalid Graph nodes; commercial absent families are unsupported.

Unreachable nodes without a named output are invalid. An independently named side branch is valid; when the loss selects only the other output, its parameters correctly have no gradients. This expected absence is not a gradient failure.

## 8. Shape inference and mathematical checks

Independent convolution expected spatial size is `floor((input + 2*padding - dilation*(kernel-1) - 1)/stride) + 1`, compared against metadata and real forward for square/asymmetric kernels, stride, padding, dilation and groups in fp32/fp64 (8 cases). Max/average pooling floor and ceil geometry have four independent cases. Existing regressions cover transpose convolution, sampling, flatten, normalization and head shape rules.

Geometry cases include grayscale, RGB and seven channels; 5×7, 16×20, 9×11 and 32×24 inputs; 1, 2, 5 and 3 output logits. Dynamic batch N=1/2/5 works while changed declared spatial shape rejects. A one-logit model is trained/evaluated under explicit binary task in a separate test; two logits alone do not imply binary-task API selection.

Symbolic BatchNorm validity can include a warning: concrete train N=1,H=W=1 fails with a model configuration error, while eval with running statistics succeeds. Both builders exercise this runtime-dependent boundary. It is not a missed unconditional constructor rejection. For all 72 measured non-model fixtures, Graph and sequential conservative parameter estimates bound actual counts (`QA/budget-bound-proof.json`).

## 9. Forward/backward execution

The shared new test helper uses deterministic seeded input, real weighted mean-square loss and finite-value checks. It compares output shapes, registers every parameter in the model and checks input plus every used trainable parameter gradient. Real classification cross-entropy is also exercised in the CPU example and TrainingSession flows. Tests do not replace forward with metadata preview.

All 216 source component/engine cases, nine Graph topologies, four geometry models and 18 pairwise models execute this real state/gradient path. Parameter-free operators still receive input-gradient checks. Stochastic components are restored/equal-compared in eval mode, not assumed bit-identical in train mode.

## 10. Optimizer and strict state checks

SGD performs an actual update when parameters exist; at least one registered parameter must differ from its pre-step clone. Gradients and state values must remain finite. Parameter-free cases have no required optimizer update. Unselected side-output parameters are handled separately, as above.

Every shared-helper case saves through `save_training_checkpoint`, checks the SHA-256 on load into a newly built model, verifies exact state keys/tensor equality and bit-equal eval outputs. This model-state test does not claim optimizer/epoch resume for every component. Existing training-checkpoint regressions separately cover the supported training-state contracts.

## 11. Five real CPU acceptance scenarios

The committed example executed outside the repository. Actual counts are measured from built models, not estimates:

| Scenario | API and structure | Input → output | Parameters / trainable | Outcome |
| --- | --- | --- | --- | --- |
| Basic | ModelBuilder: conv2d + relu + implicit classifier | [2,1,8,8] → [2,3] | 55 / 55 | forward, cross-entropy, finite backward, SGD change, strict restore PASS |
| Multi-layer | ModelBuilder: conv_norm_act, max pool, conv2d, silu | [2,1,8,8] → [2,3] | 207 / 207 | same checks PASS |
| Branching | Graph stem, pointwise/identity branches, add, classification head; GraphTensorOutput | [2,1,8,8] → [2,3] | 79 / 79 | same checks PASS |
| Negative | two-node Graph cycle | no execution | not allocated for training | preview invalid with cycle diagnostic PASS |
| End-to-end | multi-layer sequential and branching Graph | batches [2,1,8,8] → [2,3] | 207 / 207; 79 / 79 | one CPU epoch each, restore, actual evaluation, comparison, reporting and Manager PASS |

`QA/cpu-acceptance/summary.json` and each architecture/probe/training/evaluation file contain measured evidence. These are compatibility scenarios, not a benchmark or dataset acceptance renewal.

## 12. Training, evaluation, reporting and ExperimentManager

Two distinct explicitly synthetic four-sample tensors serve as train and test, with labels [0,1,2,0], batch size 2, CPU fp32, SGD lr=0.01, one epoch, seeds 13512/13513. Each TrainingSession records **2 applied updates**. Fresh restores validate architecture/settings/context/seed metadata and expected checkpoint digest before actual classification evaluation. Evaluation accuracy was 0.50 sequential and 0.25 Graph on four synthetic test samples; these values establish no quality or superiority claim.

Sequential checkpoint SHA-256: `e4826dd49500d1e0fb49d31ddae624dae0a92ad6a99f1a5e3beb448340cd87f8`.
Graph checkpoint SHA-256: `c018be6e9bdd7190062da46eda2c8f9fefe0ef4b42b5fc6439d50ee5fdf01def`.

Both native evaluations feed P26 ModelComparison with comparable quality metrics and retained context. The result explicitly keeps acceptance NOT_EVALUATED, global winner null and statistical superiority not established. Existing reporting produces JSON, Markdown and static HTML from TrainingResults, unified analysis, Graph association and ModelComparison.

A fresh QA-only ExperimentManager store is explicitly migrated using the existing API; no user store is opened for writing. Experiment `cd8731b4a3b44b2cbba2117a8d75ef71` contains five completed training/evaluation/report runs. All **15 registered artifact SHA-256 checks verified**. The full example exercises existing APIs without a new storage layer. Binary TrainingSession/evaluation has a separate one-epoch new test; regression/embedding head construction does not add tasks to the classification training API.

## 13. Graph v2 and actual receiver compatibility

The supported branching example saves with `graph_contract=True`, restores with SHA/metadata guards and runs `evaluate_graph_classification(..., capture_structure=True, capture_receiver_runtime=True)`. Source v2, actual receiver runtime and source-to-receiver relation all report **verified** in the original supported CPU process.

Serialized association reload correctly loses live receiver authority; JSON cannot recreate a receipt. Graph's legacy P26 checkpoint evidence remains **missing**. Sequential generic native checkpoint evidence is **native**; these two statuses must not be merged.

General Graph construction is broader than the closed v2 trusted profile in `_evaluation_evidence._graph_contract`. A valid depthwise Graph strictly works under ordinary state serialization but v2 save and receiver capture are deliberately rejected. Default evaluation still runs with v2/legacy Graph checkpoint missing. No component-inventory status grants universal v2 trust. Existing receiver/default/golden regressions preserve opt-in boundaries and the legacy reporting golden digest `eb66ee58d77155529d83db358cb5e0567fa8882c44ce9d7d9ee10e6664cc7b6e`.

Every inventory row now separates direct constructor, sequential, Graph and v2 applicability. Closure checked the 10 eligible Registry fixture contracts plus four existing fusion topology contracts against the actual typed validator; all admitted conditionally, and other concrete components rejected. This is metadata/runtime-structure admission with a 1M reference parameter bound, exact types, no hooks/aliasing and little-endian CPU fp32/fp64 constraints. It is **not** new per-component receiver verification. `QA/graph-v2-component-proof.json` records the exact allowed set, outcomes and unchanged contract source hash.

## 14. Architecture, structure, quality and observations

`compare_architectures` compares validated architecture/config metadata and optional actual parameter counts. Same architecture with deliberately changed weights remains architecture-equivalent. Activation, topology, channels, head and convolution-count variations are not declared equivalent. Exact example counts agree with model counts (207/79); no architecture winner is invented.

`capture_model_structure` / `compare_model_structures` inspect registered structure; same registered metadata is not equal weights, actual DAG execution or a receiver identity proof. P26 ModelComparison separately compares context-bound measured quality metrics. Model observations/diagnoses remain hypotheses with `UNVERIFIED_SOURCE` authenticity; they do not upgrade evidence. `run_model_checks` on both fresh concrete model types passes its isolated checks without promoting observations to native authority.

## 15. Technical public API quality

The [usability inventory](stage13-5-12-usability-inventory.md) records **25 actual public method signatures**, annotations and docstrings; **16** inspected methods have method docstrings. ModelBuilder has stronger annotations than Graph's largely unannotated methods. Architecture, evaluation, checkpoint and association version numbers describe distinct formats. Public exports/typing and source CLI operations remain covered by regressions and packaging.

Technical outcome: explicit bounded data contracts and usable Python integration within the tested scope. Sparse Graph annotations and differing validation phases are documented limitations. This is not a claim of complete static typing, a new public private-helper contract or a universal restore API.

## 16. Beginner usability and deferred work

The inventory separates technical API assessment from beginner experience. The model recipe functions contain 10 physical source lines/7 call expressions each; the full integration `run` has 308 lines/144 calls including audit/storage/reporting. Its 31 imported symbols are not the minimum burden for building a model. Graph requires stable IDs, ordered references, named outputs and adapter selection; the minimum sequential recipe is much shorter.

Backlog: profile-aware discovery, validation-phase vocabulary, format/version guide, output/task recipes, optional safe dimension inference, Graph hints/docstrings, batch-statistics warnings, precise diagnostics, checkpoint ownership/restore guide and reusable ExperimentManager recipes. Beginner usability remains an improvement area; **inventory completion** is PASS, not a claim that all UX issues are solved. No broad API simplification was performed.

## 17. Bugs found

No reproducible runtime contract violation was found in the executed scope. Delayed statistical validation, profile exclusions, complete-family Graph rejection and the narrower trusted profile are existing explicit boundaries, not bugs suppressed to obtain green tests. No fabricated before/after failure is claimed. Untested parameter/architecture space can still contain defects.

## 18. Changes made

- Added one 316-case profile-aware test module and its immutable metadata/sample fixture.
- Added the standalone five-scenario synthetic CPU acceptance example with an outside-repository output guard.
- Clarified the existing guide's correct commercial 57 count by explicitly contrasting source development 85 and the four separate fusion operators.
- Added this acceptance report, component matrix and usability inventory.

No runtime module, native/Core contract, dependency, profile allowlist, build metadata or prior report changed. No fix commit or protected-file exception was necessary.

## 19. Unsupported and unexecuted scope

- 36/54 bounded combinations NOT_TESTED; all larger spatial/channel/depth/topology/parameter Cartesian products are unenumerated and NOT_TESTED.
- All-component CUDA, mixed precision, long training, learning quality, speed, memory peaks and historical P01–P35 native/CUDA benchmark suites NOT_TESTED.
- Named families: all 13 metadata/profile/node boundaries checked; existing bounded M3 state/forward and M4 small fp32/fp64 gradient/JSON/weights regressions run. All family variants, long TrainingSession and optimizer/evaluation acceptance for each family NOT_TESTED.
- Commercial excluded development names UNSUPPORTED. Ordinary Graph support does not imply trusted-v2 support; depthwise v2 is explicitly UNSUPPORTED.
- Multi-input raw Graph tested; multi-input Tensor adapter/training bridge UNSUPPORTED. Linear fusion/branch API UNSUPPORTED. Arbitrary plugin JSON, automatic NHWC conversion, symbolic spatial dimensions and implicit dtype/device movement NOT IMPLEMENTED in these Builder interfaces.
- Standalone sigmoid/tanh/softmax Registry entries MISSING, regardless of internal Torch operations. Unregistered standalone classes/helpers are inventoried without invented Builder wiring.
- Regression/embedding head forward/backward/restore tested; regression/metric-learning TrainingSession tasks not added. All historical example scripts were not executed; five new scenarios were.

## 20. Test counts, identities and outcomes

| Evidence file under QA | Main tests passed | Additional subtests | Warnings |
| --- | ---: | ---: | ---: |
| regression.xml / regression.log | 1433 | 10 | 53 |
| architecture-release.xml / architecture-release.log | 316 | 0 | 1 |
| previous-boundary.xml / previous-boundary.log | 13 | 0 | 0 |
| package.xml / package.log | 6 | 0 | 0 |
| **Unique main test IDs** | **1768** | **10 separately** | **54** |
| installed-wheel.xml / installed-wheel.log | 316 repeated | 0 | 1 |
| installed-sdist.xml / installed-sdist.log | 316 repeated | 0 | 1 |

Failures, errors, skips and XFAIL: **0** in every selected run. Warning summaries remain in raw logs; existing AMP deprecation warnings were not suppressed. Initial 275/313-case development passes are retained as logs and excluded from the final unique count. Installed passes are repeats, not new unique tests.

`QA/test-id-proof.json` compares actual `(classname, function name with parameter ID)` entries: **all previous 865 retained**, **316 genuinely new**, **587 newly selected existing tests**. The only normalization removes changed isolated QA directory prefixes from the byte-identical 13-test previous-boundary copy and installed new-test copies; original raw XML IDs are retained. Repo test names/parameter IDs were not normalized away. `QA/test-source-proof.json` records the exact test/fixture source SHA-256 values and unchanged selected regression source hashes.

The 1433 regression tests cover Graph/linear builders, output adapter, checkpoint v2, receiver identity/runtime, schema1/2/3 evaluation, TrainingSession, dataset integration, architecture/quality/registered-structure comparison, reporting, advisor, ExperimentManager, AUDIT-01, CLI and bounded named-family checks. No claim is made that every repository test was run. `QA/regression-selection.json` records the exact selectors. `QA/run_checks.py`, `installed_runner.py` and `install_matrix.py` preserve runner/env details; JUnit `acceptance` properties contain actual measurements.

Installed runs use identical copied new test/fixture bytes, assert imported package location, then assert commercial Registry57. Their 316 passes include 48 expected unsupported component/engine guards plus 12 excluded-family guards, not execution of absent modules. Source positive component execution is 216; installed positive component execution is 168 per package. No skip conceals this distinction.

## 21. Fresh wheel/sdist and installed API

The six packaging tests build fresh archives, validate allowlist/exclusions, metadata, dependency policy, static/public typing surfaces and byte-identical module content. No new runtime module: all **164** allowlisted Python files match source and the starting baseline; the **10** development exclusions remain excluded. No Studio/frontend, user SQLite/model/checkpoint, build cache, secret or unapproved dependency is included.

Archives under `QA/package-tmp/packaging0/dist`:

| Archive | SHA-256 |
| --- | --- |
| hyso-0.2.0-py3-none-any.whl | 4d7f67890f8bda00c722ba7278745a7122ceeebba7c5273e7a442b51b074cf35 |
| hyso-0.2.0.tar.gz | dbbe1d8270500701a215aff107a00c718057d3b6fd9cd22994ceef4701937c2b |

Wheel and sdist installed independently into fresh QA targets using offline `pip --no-index --no-deps --no-build-isolation --no-compile`; runtime dependencies are the existing environment's dependencies, not independently reinstalled. Each target runs all 316 new test IDs. Four additional console/module `--help` checks pass (both entry paths for both packages; no workload), recorded in `QA/cli-proof.json`. Installed tests are launched in isolated Python with an asserted installed `hyso` path; no source-package fallback is allowed.

These archives certify runtime content. Documentation finalized afterward is not asserted byte-identical to an archived sdist report snapshot. Builds and package installs never write into the repository.

## 22. SHA-256 preservation

`QA/precommit-preservation.json` and final preservation/manifest record:

| Protected group | Unchanged |
| --- | ---: |
| Starting unrelated dirty/user files | 47/47 |
| Historical native files | 71/71 |
| Separately identified forensic files | 47/47 |
| External stage-6 evidence | 6/6 |
| External stage-11 evidence | 11/11 |
| User database/WAL/SHM/lock/model/checkpoint inventory | 197/197 |
| Explicit Core/checkpoint/evidence contracts | 6/6 |

The six explicit contracts are `hyso/evaluation.py`, `_graph_receiver.py`, `_evaluation_evidence.py`, `_evaluation_structure.py`, `checkpoint.py`, and `analysis/graph_evidence.py`. The groups overlap; counts are not summed. Of 746 initial tracked/nonignored files, **745 retain raw bytes**. The sole changed existing file is the scoped guide clarification. Six new files produce a 752-file final manifest. All initial runtime files and prior acceptance reports retain their raw hashes. Git blob line-ending normalization is distinct from these worktree raw-byte SHA checks.

## 23. Local commit and Git evidence

Test/example/fixture commit: **`a5f3ea6a02d20805a858cfa634e7ec61962a3aa2`**, parent **`888175a6b79c0930064d7b3b9988dcc244a3ec90`**, message `test: validate CNN builder and graph architecture compatibility`. Exact three paths are the test, fixture and example linked at the start.

The immediately following documentation commit is `docs: record stage 13.5.12 architecture compatibility acceptance`; it contains exactly the existing guide and these three stage reports. Its full SHA/parent/file list, which cannot be self-embedded in this report's commit without changing that SHA, are recorded externally in `QA/commit-docs.json`, `commits.json` and `final-manifest.json` and supplied in the final handoff.

Before each commit: exact explicit staged paths, cached diff snapshot, whitespace check, protected hashes and successful test/package evidence verified. After each: full SHA, exact parent and paths, worktree hashes and empty index verified. Final Git fsck and the prior eleven parent records are checked; remaining status is exactly the original 47-file user state. No reset/clean/restore/rebase/amend/bulk add, history rewrite or push was used. No next-stage work started.

## 24. Interpretation and reproducibility limits

PASS means observed contracts in the disclosed CPU/profile/fixture scope. Catalog discovery, valid bounded combinations, deliberate rejection and missing coverage are separate findings. A green test asserting UNSUPPORTED is not positive component execution. Architecture/registered structure identity is not weight identity, inference equivalence, training quality or global superiority. Strict state restore is not automatically trusted Graph capture or full workflow resume.

Source, installed and example evidence is retained under the single fresh QA root with raw logs, XML, metadata, hashes and scripts. `QA/evidence-index.json` binds primary evidence hashes and `final-manifest.json` binds final repo state; no artifacts from historical evidence were rewritten. Broader GPU, family, parameter and usability acceptance requires future work.

## 25. Independent final acceptance gates

| Gate | Result | Exact accepted scope |
| --- | --- | --- |
| A — Component Inventory | PASS | Actual 85/57 catalogs, four Graph operators, constructor constraints and unregistered class discovery with explicit limits |
| B — ModelBuilder Compatibility | PASS | 72 concrete source components, metadata/profile family guards, geometry and 18 bounded combinations |
| C — GraphModelBuilder Compatibility | PASS | Unary catalog nodes, nine DAG topologies, raw multi-input boundary, negative structures and strict state paths |
| D — Shape Validation | PASS | Independent conv/pool math, dynamic batch/fixed geometry, head/edge constraints and disclosed runtime statistics condition |
| E — Forward / Backward / Optimizer | PASS | Finite used gradients, input gradients and actual parameter updates in the measured CPU cases |
| F — Training / Evaluation / Checkpoint | PASS | Short CPU sessions, fresh strict restore, actual evaluation, scoped receiver opt-in and reporting/Manager; unsupported trust rejects |
| G — Architecture Comparison | PASS | Same config/different weights, five architecture variations, exact counts, registered-structure and quality/observation boundaries |
| H — Public API / Usability Inventory | PASS | Measured signatures/recipes, separate technical/beginner findings and workflow backlog; no UX-completeness claim |
| I — Packaging / Installed Tests | PASS | Fresh wheel/sdist, 164 unchanged runtime modules, 316 tests per installed commercial profile and both CLI entry paths |
| J — Native / User Data Preservation | PASS | All protected raw SHA-256 groups unchanged |
| K — Git Integrity | PASS | Two scoped local commits, prior history retained, explicit staging, empty final index and original unrelated status |

All eleven continuation gates pass only within those stated scopes. The 36 unexecuted bounded combinations and the larger unenumerated architecture space remain NOT_TESTED. There is no runtime fix, no new protected-file authorization, no quality approval and no push. Stop at stage 13.5.12.
