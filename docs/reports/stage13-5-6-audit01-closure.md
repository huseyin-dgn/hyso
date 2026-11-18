# Stage 13.5.6 — AUDIT-01 closure

Date: **2026-11-14, Europe/Istanbul**. Decision: **PASS within the declared Python-first/short-CPU scope**. AUDIT-01 is closed by the focused change, reproduced regressions, fresh package checks and preservation evidence below. This does not publish a release or certify additional platforms/native workloads.

Branch: `feature/hyso-models-next`. HEAD: `d325c613caa2718ea4010bffbe7560219707843d`. Both match the expected starting state. No commit, push, staging, reset/clean/restore, web restoration or 13.5.7 work occurred.

## 1. Root cause

`compare_architectures()` called the existing Core preflight but continued extracting raw node/head metadata even when `valid=False`. Rejected containers were treated as iterable node lists or head dictionaries. This either replaced the original Core diagnostic with a Python TypeError or exposed partial structural/cost metadata as if it described a validated architecture.

Core validation itself was working. No second validator, schema change, new dependency or unrelated Core rewrite was required.

## 2. Original reproduced failures

Both requested cases were executed on the original source **before** the fix:

| Input | Original Core result | Original comparison result |
| --- | --- | --- |
| Valid schema1 ModelBuilder dictionary with `head="invalid"` added | `valid=False`, `INVALID_BLOCK_ORDER`, original unknown-field context | `TypeError: string indices must be integers` |
| Valid schema3 dictionary with `nodes=None` | `valid=False`, `GRAPH_STRUCTURE_ERROR`, `nodes must be a list` | `TypeError: 'NoneType' object is not iterable` |

The entire original diagnostic dictionaries and exceptions are retained in external QA `original-reproduction.json`. Eight complete valid comparison outputs were also captured before editing, covering schema1/2/3 combinations with optional exact counting both off and on.

## 3. Minimal correction

Only one existing source file changed: `hyso/architecture_comparison.py`.

Immediately after Core preflight, an invalid result now creates a column in the **existing** output format and continues to the next model. It retains the requested name, configuration digest and original diagnostics. Unvalidated input/output geometry, nodes/edges/branches, parameter counts/bounds/deltas and activation estimates remain unavailable. Mechanism/category maps remain empty because no trustworthy metadata was extracted. No rejected `head`, `nodes`, `params` or connection container is inspected by that path, and no invalid model is built or counted.

The existing geometry-equality boolean additionally requires available input metadata; two unknown inputs are not reported as matching merely because both are null. The valid-model extraction and counting path remains unchanged. There is no catch-all exception handler or silent conversion to successful validation.

Two files were added: `tests/test_architecture_comparison.py` and this report. Model Checks, both Builder implementations, Trainer, Evaluation, BenchmarkRunner, dependency/profile files and previous tests/reports were not modified.

## 4. Preserved Core diagnostics and public behavior

Regression assertions compare **complete diagnostic dictionaries** directly with Core preflight: code, layer/node/index context, expected/actual fields, messages and suggestions. Multiple invalid models retain their own diagnostics. Mixed valid/invalid selections retain the valid columns and counts; an invalid baseline cannot create a numerical delta.

`fixed-reproduction.json` confirms that the two original cases now return their original Core diagnostics rather than TypeError, while valid companion models still produce exact counts. The same checks passed against both fresh installed distribution types.

The public signature, result format `hyso.architecture_comparison`, schema_version 1 and existing column keys are unchanged. **8/8 complete valid outputs match their pre-fix captures exactly** (`valid-golden-before.json`, `valid-golden-after.json`). No second schema was introduced.

## 5. New regression coverage

The new file contains **34 parameterized cases**:

- Fifteen malformed/invalid configurations, each checked with exact counting disabled and enabled: schema1 string head; schema2 string/null/list head and null/list head parameters; schema3 null/string/dictionary nodes, null/string node rows, null/list parameters, null input references and a semantic head-feature mismatch.
- Mixed comparisons with two valid and three independently invalid models, under both count modes.
- An invalid first model with a valid exact-count companion and no invented delta.
- Valid schema1/2/3 return-key/signature/count compatibility.

Invalid-only tests forbid Tensor/module allocation, retain original inputs, require exact Core diagnostics and reject fabricated count/cost/structure values. Mixed tests compare valid columns with the corresponding all-valid result.

**Before fix:** 33 failed, 1 passed (`red.xml`). The failures include both original TypeErrors and rejected partial structural/cost values. **After fix:** all 34 passed (`green.xml`). This is real before/after evidence; no expected-failure markers were added.

## 6. Actual validation results

| Fresh run | Passed |
| --- | ---: |
| New AUDIT-01 regressions | 34 |
| Builder / GraphModelBuilder / schema1/2/3 / components / builder CLI | 676 |
| Python-first CPU/model-state/adapter/training/evaluation/benchmark/Advisor/experiment/report and CLI smoke | 19 |
| Fresh wheel/sdist build/install/commercial checks | 10 |
| **Unique fresh successful tests** | **739** |

No failures, errors or skips occurred in these final runs. Existing GradScaler deprecation warnings remain; unrelated Core was not edited to suppress them. Ruff passed for the changed source and new test file. `git diff --check` passed. The signature/column compatibility tests and eight full valid-output comparisons passed.

The prior 2,758-test transition evidence remains historical and unchanged; it is not falsely presented as a full rerun after this patch. This change reran the relevant Builder regressions, focused Python/CLI workflows and distribution checks. No long training or P01–P35 native/CUDA acceptance suite was started.

## 7. Fresh wheel/sdist verification

The old archives were **not** accepted as current source evidence. New distributions were built in the isolated QA `package-temp` tree from the corrected source. Both packaging test modules produced a wheel/sdist pair: **four new archives**, each with **161 Python modules and zero web/static files**.

Every packaged Python source was compared byte-for-byte with the final checkout. Archive SHA-256 values and paths are in `archive-proof-final.json`. No Studio services, HTML/CSS/JavaScript, user SQLite, checkpoints or bytecode are present. The commercial profile, support files, exclusions and dependencies are unchanged.

Separate isolated Python processes imported the freshly installed commercial **wheel** and **sdist**, verified their import roots and signature, and ran both original AUDIT-01 cases. Each preserved `INVALID_BLOCK_ORDER` / `GRAPH_STRUCTURE_ERROR` dictionaries, kept valid companion counts and withheld invalid counts/deltas. These checks did not initialize CUDA. Installed entry-point/public API checks also passed in the distribution suite.

## 8. Git, hashes and user-data preservation

The starting snapshot covers **722 tracked/untracked files**, preserving existing modified/untracked user work. The only changed pre-existing file is `hyso/architecture_comparison.py`; only the new focused test and this closure report were added. The index is empty and branch/HEAD are unchanged.

| Protected item | Final result |
| --- | --- |
| Forensic/debug files | **47/47** SHA-256 matches |
| Native files | **71/71** SHA-256 matches |
| External checkpoint/acceptance files | **6/6** SHA-256 matches |
| Previous acceptance/audit reports | Unchanged; included in the starting file snapshot |
| Existing Core public implementations | Unchanged; all starting paths except the intended comparison source retain their hashes |
| User workspace raw files | **5/5** unchanged SHA-256, including the SQLite file |

The preceding final audit verified **three user models and 12 records**. Per this task's restriction, no SQLite connection/query, data read-back, migration or write was started. Raw file hashes alone were compared at task start/end to prove this fix did not change that previously verified workspace state. Counts were not independently queried again. Existing user weights and P01–P35 manifested evidence were neither loaded nor rewritten.

QA evidence is outside the repository and user workspace at `%LOCALAPPDATA%/hyso-stage13-5-6-audit01-closure-20261114`. It includes the baseline, original source copy, source patch, red/green/Core/smoke/package XML, valid-output captures, original/fixed reproductions and fresh archive proof. Previous evidence files were not overwritten.

## 9. Remaining open issues

No verified AUDIT-01 failure remains in the tested supported configuration space. Declared Python-first limitations are unchanged: bounded synthetic CPU checks, explicit single-Tensor/classification adaptation, no general multi-input/multi-loss orchestration and no inference of speed/quality from architecture metadata. Existing invalid user drafts were not repaired. Other platform/runtime certification and warning cleanup remain outside this closure.

The earlier [PARTIAL audit](stage13-5-6-final-python-audit.md) is preserved byte-for-byte with its original finding and decision. This report records the subsequent closure; it does not rewrite that history.

## 10. Final 13.5.6 decision

**PASS for AUDIT-01 closure and the declared Python-first release-readiness scope.** The original failure was reproduced, Core diagnostics are retained, valid behavior remains identical, focused/regression tests pass, new web-free distributions pass, and preservation checks match.

No commit or push was made. No 13.5.7 or web work was started. Work stops at this closure.
