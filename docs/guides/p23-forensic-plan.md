# P23 safety closure and separate forensic plan — 2026-11-09

P23 remains **PARTIAL** for the requested CUDA saved-activation scope. The
historical `c-native-crash.json` reopens as native_crash, exit `0xC0000409`,
no worker terminal receipt and no completed measurement; its owned Job reports
zero active processes. SHA256:
`b22337f5f90baba8d6cab488201dec51d1612be0a4fee0a3d0d8ff4a86d25e9a`.
The pinned hook-free control has clean child/outer receipts. It proves the
control's success, not a root cause or a cure for the hook-enabled path.
No native call stack or crash dump in these retained records establishes a
specific failing instruction. A hook/autograd/runtime interaction is a hypothesis.
The previously fixed Python tensor-version check is not proved to cause this
native exit. No crash was reproduced by this task.

| Domain | Meaning and exclusions |
|---|---|
| Parameter | Registered tensor logical bytes; frozen bytes are a subset |
| Gradient | Currently materialized gradients; absent and zeroed gradients differ |
| Optimizer | Native SGD/Adam/AdamW tensor state, including lazy creation; excludes Python heap/EMA/scaler/scheduler |
| Saved payload | CPU autograd saved-slot logical bytes and deduplicated storage extents, including registered aliases; not all activations or physical memory |
| Process RSS | Main process working set sampled through native backend; includes runtime/observer and can miss transients |
| Private commit | Separate OS Job lifetime peaks, including startup/import/factory and descendants for Job aggregate; not RSS or fit-only memory |
| CUDA allocated/reserved | Selected process/device native allocator counters; owned reset-window peaks only when explicitly admitted; not physical VRAM |
| Artifact | Serialized file length, not process/training memory; no RAM estimate from checkpoint size |

Do not add these domains. Logical views and components can alias. A numeric zero
is valid for an observed empty gradient/optimizer component; missing CPU/CUDA,
activation or artifact observations use null + reason. Unsupported raw tensors
do not become measured component totals. Diagnostics expose private commit and
artifact coverage independently; the v1 JSON/CSV wire formats remain unchanged.

Safe changes bound even parameterless module traversal and recipe container
depth/node/cycle handling, validate saved-recorder budgets and enforce CPU-only
admission at the recorder itself. Worker recipe/device mismatch is rejected
before CUDA queries. Isolated result readers validate storage extents, aliases,
deduplicated totals, source epoch/microbatch/attempt timing and CPU/CUDA boundary
identity, even after a payload is rehashed. OS Job ownership/deadline/cleanup and
native-crash propagation remain mandatory; failed observations carry no completed
measurement. Sampler shutdown waits at most five seconds and records an explicit
error if a native RSS query remains blocked; it cannot safely cancel that native
query in-process and cannot publish a successful observation in that case.
The in-process API still has no hard workload timeout and is only
for finite trusted work; a bounded observer is not a workload memory quota.

The production CUDA saved-activation gate remains disabled. CPU fault tests and
archived result reading do not enable it. Thirty-one pinned historical P23
payloads were revalidated with model/worker/CUDA execution sentinels; no execution
was reached. The legacy in-process fixture and saved native crash still reopen.

For a future separately authorized forensic experiment:

1. Preserve these raw records first. Identify one exact hypothesis and one tiny
   finite workload; pin current source, interpreter, Torch/CUDA DLLs, checkpoint,
   input, GPU UUID and driver. Never execute the old consumed manifests.
2. Prepare a fresh forensic snapshot with hooks still disabled. Inspect existing
   WER/dump availability read-only; do not silently edit WER/registry policy or
   install a debugger. A dump-collection configuration change needs its own scope.
3. Use the existing `tools/forensics/p23_preflight.py` contract: fresh single
   inventory, review every compute PID, ordered GPU samples, host/disk availability
   and exact run/manifest identities. Unknown workloads block the experiment.
   This policy is necessary, not authorization. Do not invent telemetry or wait
   until admission happens to pass.
4. Require exact inner/outer owned Windows Jobs, read-back limits, hard deadlines,
   bounded logs/results, one attempt and no retry. Preserve child/outer receipts
   and any native stack/dump on failure, then verify zero owned active processes.
5. A hook-enabled native experiment requires a **new reviewed forensic runner and
   explicit approval for its exact manifest/hash and hook-enabled workload**.
   No such runner or approval is supplied here; ordinary production flags cannot
   enable the disabled path. Do not patch the production gate to run a probe.

The executable preflight validator command is in
`remaining-acceptance-next-steps.md`. It only validates independently collected
evidence and starts no native workload. Hook-free controls are historical evidence;
they are not rerun automatically and cannot close the activation requirement.
