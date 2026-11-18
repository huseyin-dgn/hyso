# G3 — Windows Job-bounded trusted CPU model bridge (P02/P11 only)

G3 is **additive**. It does not change any existing G1/G2 behavior, benchmark schema, model code, checkpoints, or weights. It does not automatically run any feature. The G3 controller has no explicit Torch or factory import; the existing top-level `hyso` package may still import Torch when imported normally. User factory execution is worker-only.

## What is supported

- G1 `FeaturePlanRequest` in `single`, `selected`, `automatic`, or `full` mode.
- **Explicit** per-feature authorization for a subset of **P02 throughput** and **P11 model footprint**. The other 33 remain `NOT_EXECUTED` and are not silently treated as successes.
- One or two **trusted** CPU PyTorch classification factories with `source_file`, exact SHA-256, a named top-level factory function, and bounded JSON-only kwargs.
- Each factory is imported and called only **inside a separate Windows worker**, not the controlling HYSo Python process. It returns `(torch.nn.Module, CPU NCHW input tensor)`; P11 ignores the input.
- Actual P02/P11 measurement is delegated to G2 and the **existing native HYSo producers**. The controller checks the worker completion, capture count and hashes, comparisons, and missing/omitted features again. Two-model differences remain observational, without winner/verified-improvement claims.
- The owner creates a Windows child **suspended**, configures a single-child Job Object with `KILL_ON_JOB_CLOSE`, `ACTIVE_PROCESS_LIMIT=1`, hard process+job **committed-memory limits**, assigns the suspended process, and resumes it only afterward. A strict wall-time deadline terminates the entire job. No PID-name killing.
- The worker refuses direct invocation without confirmed Job membership and limits **before importing Torch or factory code**. On a non-Windows host, G3 refuses execution; it does *not* revert to in-process G2.

## Critical boundaries

This is **NOT a security sandbox** for malicious or untrusted Python. Job memory/child count/time limits cannot prevent worker code from writing accessible files, reading secrets, using the network, changing other files or interacting with other processes. Only run trusted, reviewed factory code and trusted model/data files. Factory source hashing is an integrity check, *not* a trust or dependency-authenticity certificate. The called factory may itself import other modules.

For the intended read-only HYSo analysis product, existing user model objects are not passed into G3. The factory constructs a **separate worker-owned instance**. G3 does not intentionally alter source or checkpoint bytes, but it cannot promise what arbitrary trusted Python does. Separate restricted Windows credentials, OS isolation, file access controls, network restrictions, or a VM/container boundary are needed before accepting hostile models.

Hard memory limit concerns **Windows committed process/job memory**, not RSS, CUDA VRAM, or a model-only allocation. Only CPU/native PyTorch P02/P11 are tested here. No GPU, native CUDA, generic ONNX, training, camera, 7200-second run, or automatic optimization is authorized. The 35-feature release gate is unchanged.

## Example

```python
from dataclasses import replace
from pathlib import Path
import hashlib

from hyso.analysis.feature_plan import ModelContext, FeaturePlanRequest
from hyso.analysis.isolated_bridge import (
    TrustedFactoryBinding, IsolatedCPUExecutionPolicy, execute_isolated_cpu,
)

factory = Path(r"C:\my-trusted-code\my_factory.py").resolve()
source_sha = hashlib.sha256(factory.read_bytes()).hexdigest()
request = FeaturePlanRequest(
    'selected',
    (ModelContext('model-A', input_contract=True, trusted_factory=True),),
    features=('P02', 'P11'),
)

result = execute_isolated_cpu(
    request,
    (TrustedFactoryBinding('model-A', str(factory), source_sha, factory_name='build'),),
    authorized_features=('P02', 'P11'),
    policy=IsolatedCPUExecutionPolicy(
        explicit_authorization=True,
        timeout_seconds=45,
        job_memory_limit_bytes=2*1024**3,
        warmup_iterations=1, batch_count=2, repeats=1,
    ),
)
print(result.status, result.reason)
# Only if result.status == 'COMPLETED', inspect the scoped native captures.
# Acceptance stays NOT_EVALUATED even after a completed worker run.
```

Your factory is a **reviewed and trusted Python file** defining `build(**kwargs)` which returns `(model, example_input)`. The factory must create its own CPU model instance. Do not pass an arbitrary pickle, unknown checkpoint deserializer, or untrusted URL to it.

## Verification and known gaps

```powershell
$py = 'C:\Users\hdgn5\.conda\envs\torch_gpu\python.exe'
& $py -B -m unittest discover -s tests -p test_isolated_bridge.py -v
& $py -B -m unittest discover -s tests -p test_isolated_bridge_windows.py -v
```

Offline tests do not certify Windows Job behavior. The second suite runs a real separate Torch CPU worker **only on Windows**, on a bundled trusted toy factory; it may take longer than the earlier G2 tests. No unreviewed source or native GPU work is launched by these tests. Avoid adding your real models until this native smoke passes and resource limits are reviewed. A native crash root cause from earlier HYSo acceptance work remains unresolved; successful G3 tests do not close it.

If a worker times out, exits abnormally, produces no valid terminal, or cannot be contained/cleaned, G3 must report `FAILED` or raise an explicit error. Never interpret exit code 0 alone as valid measurements.

G3 does **not** grant authorization or execute all 35 modules. Later work must add carefully scoped adapters, metric/result validation, model integrity before/after checks, broader platform support, and a stronger isolation boundary for hostile model code.
