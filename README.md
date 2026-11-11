# HYSo acceptance status

Model A and Model B completed P01–P35 with **35/35 PASS each** in the declared scopes. See the [final acceptance summary](docs/reports/p01-p35-final-acceptance-20261112.md) for actual P31 durations, evidence locations and limitations. P23 extended saved-activation remains outside acceptance; P30 uses a separate image adapter.

The integration-candidate instructions below are preserved historical material; their unvalidated/pending status statements describe that earlier snapshot, not the final acceptance result.

# HYSo P13 — Native capture integration v1 (review first)

This package extends `a795e93` on `feature/hyso-models-next`. It is an *unvalidated-on-Windows* integration candidate. It has not run Nsight Systems or Model B in this environment. No native command is executed by applying it.

**New files**: P13 native controller, instrumented Model B worker, post-Job provenance finalizer, CPU-only tests.

**Modified files**: `registry.py`, `core.py`, `reporting.py`, `safety.py`, `cli.py`, `handlers.py` in `tools/full_acceptance`.

## Apply and inspect (no GPU)

Copy the ZIP to `Downloads`, then from HYSo repository root:

```powershell
$zip = "$env:USERPROFILE\Downloads\HYSo_P13_native_integration_v1.zip"
$pkg = Join-Path $env:LOCALAPPDATA "hyso-p13-native-integration-v1"
if (Test-Path $pkg) { throw "Use a fresh extraction directory: $pkg" }
Expand-Archive -LiteralPath $zip -DestinationPath $pkg
& "C:\Users\hdgn5\.conda\envs\torch_gpu\python.exe" "$pkg\apply_p13_native_v1.py" (Get-Location).Path
if ($LASTEXITCODE -ne 0) { throw "P13 installation failed" }
git diff --check
if ($LASTEXITCODE -ne 0) { throw "Git diff check failed" }
git diff --stat
& "C:\Users\hdgn5\.conda\envs\torch_gpu\python.exe" -m pytest -q `
    tools/full_acceptance/tests/test_p13_capture_native.py `
    tools/full_acceptance/tests/test_p13_capture_scopes.py `
    tests/test_module_activity.py `
    tools/full_acceptance/tests/test_infrastructure.py
```

**Do not yet run the native capture.** First review the new worker and run the offline test selection. The actual installed Nsight CLI options and GPU behavior have not been tested here. There is no pre-approved GPU run.

## Native plan, after an explicit review/approval

This is documentation, not a requested execution instruction:

- Create a new `p13-nsys-plan.json` with the *measured* SHA256 of the installed x64 `nsys.exe`, `tool` `{path, sha256, bytes}`, `protocol: trained_B_3_forward_module_nvtx`.
- `--preflight --output <fresh-prepared> --p13-nsys-plan <plan> --continue-independent` binds the tool and the source snapshot.
- A **new authorized child manifest** with only `--authorize 06.p13_capture` is required. Its Job-owned stage generates native report and offline preliminary analysis.
- After the acceptance controller has closed with clean preservation, run the new CPU-only finalizer via `python -m tools.full_acceptance.p13_capture_finalize --manifest <child-manifest> --approved-sha256 <hash> --output <newdir>`.
- The finalizer binds the **real** outer Job receipt to the raw capture and emits `p13-evidence-configuration.json`. Only then does a second separate `06.layer` CPU calibration manifest consume `--p13-evidence` for the original P13 PASS criterion.

Do **not** re-run P18, P31/P32 or open P23 CUDA saved-activation hooks. Preserve all prior failures and avoid modifying the user's models/checkpoints. No push without approval.

## Limitations

This package does not certify native capture compatibility, nor alter the original P13 %/latency criteria. The preliminary native stage can PASS while the overall P13 feature stays PARTIAL. The post-Job finalizer fails closed if the source/checkpoint/worker/Job evidence is missing or inconsistent. Binary `.nsys-rep` relation to SQLite remains an attestation of the approved capture chain, not independently reconstructed by the offline reader.
