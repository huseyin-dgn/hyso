# Stage 13.5.6 — Final Python-first audit and release readiness

Audit date: **2026-11-14, Europe/Istanbul**. Scope: inspection and verification only. No feature, application/test-code fix, dependency change, migration, staging, commit or push was performed. This report is the only new repository file produced by the audit. New audit outputs are outside the checkout at `%LOCALAPPDATA%/hyso-stage13-5-6-final-audit`; earlier evidence was read, not rewritten.

**Decision: PARTIAL.** Web retirement, supported Python CPU workflows, distribution integrity and preservation checks pass. A reproducible error-handling defect remains in the newly introduced architecture comparison API. Successful existing suites do not cover that malformed-input path. No automatic correction was applied.

## 1. Git integrity and change inventory

| Check | Observed result |
| --- | --- |
| Branch | `feature/hyso-models-next`, expected |
| HEAD | `d325c613caa2718ea4010bffbe7560219707843d`, expected |
| Initial modified tracked files | 10 |
| Initial untracked files | 70 |
| Tracked deletions | 0 |
| Index | Empty |
| `git diff --check` | Exit 0; no whitespace errors |

Git emits existing LF-to-CRLF conversion notices for several working-tree files. These are notices, not diff-check failures. No line-ending conversion was applied by this audit.

Every initial status entry was classified and hashed: **27 transition entries and 53 pre-existing user-work entries**, with no unclassified entry. The complete path/status/classification/SHA-256 inventory is in `integrity-audit.json`.

| Modified tracked file | Classification and verification |
| --- | --- |
| `README.md` | Transition documentation; matches final transition manifest |
| `commercial-profile.json` | Four new independent Python modules; matches manifest; development exclusions unchanged |
| `docs/guides/python-api.md` | Transition guide link; matches manifest |
| `examples/README.md` | Python example instructions; matches manifest |
| `tests/test_commercial_distribution.py` | Web-free installed-package checks; matches manifest |
| `tests/test_packaging.py` | Web-free package inventory/checks and formatting; matches manifest |
| `hyso/models/graph.py` | Pre-existing Graph work, outside retirement edits; exact task-start bytes preserved |
| `docs/guides/custom-cnn-graph-builder.md` | Pre-existing guide change; exact task-start bytes preserved |
| `docs/guides/p23-forensic-plan.md` | Pre-existing forensic change; protected hash preserved |
| `tools/forensics/native_admission.py` | Pre-existing forensic change; protected hash preserved |

The untracked inventory consists of **21 transition files** and **49 pre-existing files**. These include the four new Python modules, Python tests/examples, current/retired guides and transition reports, alongside the preserved patches, forensic scripts/configuration and historical reports. Their presence is accounted for; unrelated user work was not removed or absorbed into the transition.

All **29 final transition-manifest files** match their recorded hashes. All **651 tracked task-start entries** match the expected final state: authorized transition files use their final hashes; the intended removal of Studio additions in MANIFEST.in/setup.cfg is independently equal to committed pre-web content. No unexpected tracked change was found.

The 20 retired Studio source/test files were untracked before removal, so their absence does **not** appear as Git `D` entries. All 20 are absent and their earlier source backups/manifests remain available. Historical reports were not deleted. After writing this report, it is one additional intentional untracked file; the index remains empty.

## 2. Complete web retirement

| Check | Result |
| --- | --- |
| `hyso/studio` directory | Absent |
| Expected 15 Studio runtime/static and five Studio test sources | All 20 absent |
| Active Python Studio/frontend/server imports | None found by AST inspection |
| Runtime HTML/CSS/JavaScript files | None |
| Packaging Studio/static/frontend requirements | None |
| `python -m hyso.studio` | Expected failure: `No module named hyso.studio` |
| TCP 127.0.0.1:8765 | No connection accepted; no listener observed |

Import inspection included Studio, Playwright, Streamlit, Gradio, Flask, FastAPI and standard-library HTTP-server imports. Core result reporting can still generate HTML; retaining that Python export capability is intentional and does not constitute a web product. The audit did not start or stop a server.

## 3. Python API and CLI checks

The fresh smoke run reused the focused independent Python tests with one CLI-heavy test deselected, rather than rerunning the full regression suites. It performed real, bounded CPU work in a new audit-only temporary directory.

| API/capability | Fresh or preserved evidence |
| --- | --- |
| ModelRegistry | Validation/construction through builders and 60 retained component-config preflights |
| ModelBuilder | Fresh CNN forward; schema1 and schema2 checks |
| GraphModelBuilder | Branching schema3 graph, named outputs and actual forward/backward |
| GraphTensorOutput | Shared parameter identity, autograd, float64 propagation, strict state loading and invalid-name/task/multi-input rejection |
| TrainingConfig / TrainingSession / Trainer | Real one-epoch example and four-epoch small training-observation fixture; native Trainer delegation |
| Evaluation | Actual classification inference over four synthetic samples; no supplied-prediction substitution |
| BenchmarkRunner | CPU, one warmup, three measured iterations, one repeat |
| Model Checks | Structure/forward/backward/strict serialization PASS on isolated supplied weights; preservation, actual-failure and resource-skip tests |
| Architecture Comparison | Valid two-model exact counts and allocation-free comparison pass; malformed-input defect documented below |
| Advisor / model observations | Existing report interpretation and real training/check observations; no automatic change |
| ExperimentManager | One finished example run, logged training metrics, report artifact and history read-back in an isolated example store |
| Reporting | Existing Python report creation/JSON artifact from native results; broader prior reporting tests reviewed |

Fresh example observations: linear output `[4,2]`; one-epoch loss **0.7239033282**, training accuracy **50%**; inference sample count **4**, accuracy **0.5**; benchmark mean **0.4246333333 ms**, P95 **0.56553 ms**; architecture parameter counts **16 and 19**. These tiny synthetic results verify API operation, not model quality, sustained throughput or native P01–P35 acceptance. The example created/migrated only its own empty ExperimentManager fixture store, never the user workspace.

Source `python -m hyso --help` succeeded and exposed the existing seven command groups. Four additional checks ran from an audit directory outside the checkout against the previous real installations:

| Installation | Entry | Exit |
| --- | --- | ---: |
| Wheel | Installed `bin/hyso.exe --help` | 0 |
| Wheel | `python -m hyso --help` | 0 |
| Sdist | Installed `bin/hyso.exe --help` | 0 |
| Sdist | `python -m hyso --help` | 0 |

Both installations' 161 Python source files match the current checkout. PYTHONPATH was supplied only to the child processes to select those existing target installations; the user environment was not installed into or modified. Earlier successful seven-group help, dry-run and CLI regressions were also reviewed.

## 4. Test evidence

| Existing transition group reviewed | Successful tests |
| --- | ---: |
| Builder/schema/component/CLI regression | 676 |
| Training/checkpoint/evaluation/benchmark | 1,203 |
| Analysis/comparison/Advisor/experiments/reporting/discovery | 850 |
| Independent Python transition tests | 19 |
| Wheel/sdist/commercial installation | 10 |
| **Existing unique test total** | **2,758** |

The 146 successful subtests are separate. In particular, evidence.xml has 850 testcase elements and a suite header of 996 including those 146 subtests; adding that header as 996 independent tests would double-count them. Existing final archive-content (3) and architecture (2) repeats were not added to the total.

Fresh focused result: **18 passed, 1 deselected**, no product-test failure in `smoke-final.xml`. The deselected test's CLI/import checks were covered by current source/AST checks, current launcher checks and unchanged prior evidence. The first audit invocation had **17 passes and one fixture-setup error** because its audit basetemp parent did not yet exist; this was an audit harness path error. It was retained in `smoke.xml`, and the run was repeated in a separate valid audit path without changing application/test code.

Ruff passed for the four new Python modules, focused tests/example and modified packaging tests. Existing GradScaler deprecation warnings remain. The two malformed-input audit probes below are **reproduced open defects**, not passing negative tests. No CUDA/native acceptance suite or long training was run, and the complete suites were not needlessly repeated.

## 5. Packaging verification

All six previously recorded archives—three wheel/sdist pairs—still exist. Each archive's SHA-256 matches its evidence record, and all **161 packaged Python sources** match current files byte-for-byte. Each contains **zero Studio services, HTML, CSS or JavaScript**, and no SQLite database, checkpoint or generated bytecode. M3/M4/development exclusions match the unchanged committed policy.

Previous real wheel/sdist build/install/CPU tests passed, and their exact source snapshots remain current. Current installed CLI checks also pass. Therefore no rebuild or complete package-suite repeat was needed. Archive-specific results/paths are in `integrity-audit.json`; original archive evidence was not overwritten.

## 6. Data, hashes and Core public contracts

| Protected item | Final audit |
| --- | --- |
| User SQLite models | **3 preserved** |
| User SQLite logical records | **12 preserved**, exact kind/id/payload equality with transition-start snapshot |
| Forensic/debug files | **47/47** SHA-256 matches |
| Native source files | **71/71** SHA-256 matches |
| External checkpoint/acceptance files | **6/6** SHA-256 matches |
| Pre-existing tracked Core Python files | **167/167** byte-identical to approved transition-start state |
| Manifested historical report/capability files | **19/19** SHA-256 matches |

SQLite was opened with `mode=ro`; no service/store instance, migration, repair, test result or export was applied to it. The three previously invalid architectures remain unchanged, including their existing disconnected-node/channel issues. Their invalidity is not a new transition regression and does not justify silently repairing user data.

P01–P35 protection is established for the tracked historical reports and the checkpoint/acceptance files covered by the existing manifests, including the external acceptance matrices. No historical acceptance capture was rerun or rewritten. This is a manifest-based statement, not an unsupported assertion about every unmanifested external file.

Public Core implementation bytes, existing API identity/schema regression evidence and fresh supported CPU flows agree. No existing ModelBuilder/Graph/Trainer/evaluation/benchmark/Advisor/ExperimentManager/reporting/CLI contract change was found. The open finding concerns the new architecture-comparison boundary rather than a modified legacy Core API.

## 7. Open findings and limits

### AUDIT-01 — Medium: malformed comparison inputs lose Core diagnostics

Affected code: `hyso/architecture_comparison.py`, `compare_architectures`, raw nodes/head metadata extraction after `preflight_model`.

The function documents that invalid descriptions retain diagnostics and do not receive runtime counts. Two supplied dictionary fixtures demonstrate a gap:

1. Start with a valid schema1 ModelBuilder configuration and add `head="invalid"`. Core preflight returns `valid=False` and an `INVALID_BLOCK_ORDER` diagnostic. Architecture comparison instead raises **`TypeError: string indices must be integers`** while reading the head as a dictionary.
2. Start with a valid schema3 graph and replace `nodes` with `None`. Core preflight returns `valid=False` with **`nodes must be a list`**. Architecture comparison instead raises **`TypeError: 'NoneType' object is not iterable`** while extracting mechanisms.

Minimal reproducer for the first case:

```python
from copy import deepcopy
from hyso import ModelBuilder
from hyso.model_checks import preflight_model
from hyso.architecture_comparison import compare_architectures

valid = ModelBuilder((1, 2, 2), 2).to_config()
invalid = deepcopy(valid)
invalid["head"] = "invalid"
assert preflight_model(invalid)["valid"] is False
compare_architectures([valid, invalid])  # Reproduced TypeError
```

Impact: the new API cannot reliably return its promised invalid-model diagnostics for malformed container fields. Supported valid comparisons still work. No model, data or package was altered by the probes. Full Core diagnostics and exact exceptions are retained in `edge-case-audit.json`. **No fix was made.**

### Declared limitations, not newly discovered regressions

- Single-Tensor adapter workflows do not provide multi-input/multi-loss orchestration or general regression/embedding training.
- Model checks use bounded CPU synthetic fixtures and a sum-of-outputs gradient objective; PASS is not a training-loss, dataset-quality or complete gradient-coverage guarantee.
- Full registered-model internal topology can remain unavailable in configuration-only comparison.
- Existing invalid saved user architectures still need explicit user-directed correction before execution.
- Existing warning cleanup and a broader runtime/platform matrix remain future work; this audit did not certify CUDA/native behavior or publish a release.

## 8. Stage 13.5.6 decision

| Area | Decision |
| --- | --- |
| Git/preservation integrity | PASS |
| Web removal and closed 8765 port | PASS |
| Supported Python CPU API and CLI operation | PASS |
| Current wheel/sdist integrity | PASS |
| Data/native/forensic/history protection | PASS |
| New comparison API malformed-input handling | **PARTIAL — AUDIT-01 open** |
| **Overall final release readiness** | **PARTIAL** |

The supported Python-first transition is operational and preserved. Unqualified final release-readiness approval is withheld until AUDIT-01 is addressed and its negative cases are verified in a separately authorized change. A high test count does not close an uncovered reproduced defect.

## 9. Recommended next Python priorities

1. Resolve AUDIT-01 with explicit handling of malformed containers while retaining original Core diagnostics; add both reproduced negative cases to focused tests.
2. Run that focused validation plus current source/archive checks, then reassess final readiness. Do not repeat unrelated native/long suites for this boundary fix.
3. Expand graph dataset/head/loss and trained-model provenance examples before stronger model-comparison claims.
4. Design multi-input/multi-loss support separately, with explicit contracts and budgets; do not imply it exists today.
5. Keep legacy architecture export/migration and warning/platform work explicit and independent of user-data preservation.

These are recommendations only. No new implementation, automated correction, 13.5.7 work, git reset/clean/restore, commit or push was performed.
