# HYSo G2 — first real native producer bridge (P02 and P11 only)

G2 is additive to the already applied **G1** `hyso.analysis.feature_plan`. It is not a replacement for any existing HYSo measurement/acceptance pipeline. Production `hyso.performance.measure_throughput` (P02) and `hyso.performance.measure_footprint` (P11) remain the measurement authorities; this bridge neither re-implements their math nor certifies all 35 features.

**Security limitations**: This code executes *trusted*, caller-supplied Python `nn.Module` in the **same process**, with explicit consent. Limits on warmup, batch count, repeats, and input bytes are **not** a hard time or total RAM limit. An untrusted model could execute arbitrary Python code or hang/crash the process; there is no sandbox, hard watchdog, OS worker cap, or tamper-proof source guard in G2. Do not use untrusted models. No automatic CUDA fallback, model export, training, checkpoint overwrite, file export, or system-wide performance guarantee.

Only P02 and P11 can be explicitly authorized; any other feature in a G1 full plan is returned `NOT_EXECUTED`, not silently claimed `PASS`. G1 blocked prerequisites fail before native calls. Model IDs must match exactly. No implicit execution permission is inherited from G1. `NativeBridgeReport.acceptance` remains `NOT_EVALUATED`. Two-model comparisons are descriptive observations only and require matching P02 input content, shape, dtype, measured device, and protocol. Neither speedup nor an independently verified improvement is certified.

## Example: one trusted local CPU model

```python
import torch
from torch import nn
from hyso.analysis.feature_plan import ModelContext, FeaturePlanRequest, plan_features
from hyso.analysis.native_bridge import (
    LiveCPUModel, TrustedCPUExecutionPolicy, execute_scoped_cpu,
)

model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
inputs = torch.ones(2, 1, 2, 2)
plan = plan_features(FeaturePlanRequest(
    mode='selected', models=(ModelContext('example', input_contract=True),),
    features=('P02', 'P11'),
))
policy = TrustedCPUExecutionPolicy(
    explicit_trusted_inprocess_consent=True,
    warmup_iterations=1, batch_count=2, repeats=1,
)
report = execute_scoped_cpu(
    plan, (LiveCPUModel('example', model, inputs),),
    authorized_features=('P02', 'P11'), policy=policy,
)
print(report.to_dict())
```

## Verification

From HYSo repo root (Windows, `torch_gpu`):

```powershell
$py = 'C:\Users\hdgn5\.conda\envs\torch_gpu\python.exe'
& $py -B -m unittest discover -s tests -p test_feature_plan.py -v
& $py -B -m unittest discover -s tests -p test_native_bridge.py -v
& $py -B -m unittest discover -s tests -p test_native_bridge_integration.py -v
```

The first two suites are offline contract tests (G2 native results are **doubles** in the second suite); the final suite calls the **real P02 and P11 producers on trusted CPU toy models**. None tests GPU or external models. Run later source-pinned, bounded **native workers** for other features; don't run P31 7200 s during this development stage.

**Next missing work**: trusted Windows Job child execution and hard resource/timeout policy; selected real-model adapters by task (classification/detection/segmentation); typed evidence exports; complete feature-by-feature routing; full comparison eligibility/provenance and unified human-facing advisor. The initial two producers are not an assertion that the other 33 are DONE.
