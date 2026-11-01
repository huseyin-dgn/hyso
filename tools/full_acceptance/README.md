# HYSo full-system acceptance tooling

`tools/run_full_acceptance.py` is the single entrypoint. It adapts existing HYSo
measurement/worker/result APIs and retains original acceptance criteria. It does
not implement the proposed user analysis API or a new advisor/report engine.

Preparation and execution are different operations:

```powershell
$py = 'C:\Users\hdgn5\.conda\envs\torch_gpu\python.exe'
& $py tools/run_full_acceptance.py --preflight --output outputs/full-acceptance-NEW
# The command prints the exact manifest SHA256 and generates commands.ps1.
& $py tools/run_full_acceptance.py --manifest outputs/full-acceptance-NEW/manifest.json --approved-sha256 '<EXACT_SHA256>' --smoke
& $py tools/run_full_acceptance.py --manifest outputs/full-acceptance-NEW/manifest.json --approved-sha256 '<EXACT_SHA256>' --execute
```

Use the concrete paths/hash in the generated `commands.ps1`. `--execute` is
one-shot. Repeated execution is rejected even after an interrupted/failed run;
prepare a new manifest rather than deleting STARTED or retrying native work.
Exit0 means every required experiment produced a real result and none failed;
exit2 means incomplete approval/prerequisites/dependencies; exit1 means failure.
Feature acceptance still has its own scoped PASS/PARTIAL/BLOCKED/FAIL table.

The first-delivery defaults approve only metadata preflight, the explicit
reviewed CPU regression list, bounded trained CPU inference and pure memory/replay
analysis. These do not run until the operator invokes the entrypoint. Native GPU,
training, ownership profiling/calibration, camera/video, provider sessions and
7200-second P31 start as `NOT_AUTHORIZED`. Approval is never inherited through
dependencies or previous P07/P08/P28 tasks. Readiness is reported independently
from authorization. Camera/video additionally require a task-appropriate trained
adapter; Model B MNIST is not silently substituted.

To approve exact reviewed experiments, create a **new** run directory/manifest:

```powershell
& $py tools/run_full_acceptance.py --manifest '<PARENT_MANIFEST>' --approved-sha256 '<PARENT_SHA256>' `
  --authorize '04.cuda,05.training,06.layer,06.roofline,07.energy,07.devices,09.provider_cpu,09.provider_cuda,10.stability' `
  --approval-text '<ACTUAL_OPERATOR_APPROVAL_FOR_THESE_EXPERIMENTS>' --output '<NEW_EXCLUSIVE_DIRECTORY>'
# Use the newly printed manifest/hash with --execute. This command itself runs nothing.
```

For camera approval, prepare a new parent with `--camera-config camera.json`, then
approve `08.video`/`08.camera` only after reviewing that adapter. Required camera
config fields: `factory` (`module:callable`), `factory_source` (`path`,`sha256`),
`checkpoint` (`path`,`sha256`), `input_contract`, `output_contract`, `preprocessing`,
`provenance`; optional `kwargs`, `camera_index`, and `video` (`path`,`sha256`). The
trusted factory must return an existing HYSo-compatible frame processor with an
identity declaring trained origin. Its source/checkpoint/video bytes are pinned;
the factory is trusted code, not a hostile-code sandbox. Appropriate labeled
camera quality data remains a separate requirement for accuracy claims.

GPU UUID and optional ONNX runtimes are discovered from retained local capture
metadata only. Physical GPU execution is checked using the observed Torch UUID
at native admission. `--gpu-uuid`, `--ort-cpu` and `--ort-cuda` override selection
when creating a new manifest. No runtime/driver/package install, device setting,
clock lock or memory-fraction setting is performed. NCU may require existing
performance-counter privileges; the suite never changes those privileges.

The P18 helper bundles the preserved reference CUDA source/PTX listed in
`counters/origin.json`, adapting a new owned worker without the historical
allocator fraction. No native binary is compiled or run by preparation/smoke.
Future captures are new independent target/compute/memory sessions, two warmups
and three actual samples each, with an unchanged10% compute spread criterion
and independent output/FFMA checks. The fixed SM86 kernel recipe is supported
only on its declared target; this is a kernel compatibility constraint, not a
hardware batch or memory-capacity claim. These references do not measure trained
CNN DRAM traffic or establish physical root cause/headroom.

The execution plan lists exact per-experiment commands, source tree/manifest
hashes, models/device/input/work, explicit protocol/watchdog rationale, stop
rules and output paths. Windows Jobs are kill-on-close, with **no RAM/VRAM quota,
allocator fraction or default process-count cap**. OS LowMemoryResourceNotification
is monitored. Finite evidence/log and native producer deadlines remain explicit.
The P23 isolated API retains its native4-process lifecycle limit; it is not a
RAM/VRAM/batch quota. Safe regression fixtures may exercise their own small
constructed limit readbacks; they are not physical OOM/crash experiments.

```powershell
# Independent terminal; no live source verification required to observe/stop.
& $py tools/run_full_acceptance.py --manifest '<MANIFEST>' --status
& $py tools/run_full_acceptance.py --manifest '<MANIFEST>' --stop
& $py tools/run_full_acceptance.py --manifest '<MANIFEST>' --package '<NEW_ARCHIVE_OUTSIDE_OUTPUT>.zip'
```

Stop requests bind to the exact manifest hash. The controller terminates/reaps
only its current owned Job tree; it never kills by process name or follows a
user-supplied PID. If assignment fails before admission, only the newly-created
waiting Popen handle is terminated. Ctrl+C also closes owned work. Raw committed
prefixes and failure receipts remain. Full completion cannot be inferred from
child exit, a short smoke, missing terminal, partial sensor coverage or source
reopen. B4 and CPU65536 remain OPEN; their reproducers cannot be authorized as
ordinary experiment IDs and require separate controlled forensic plans.

The15 requested top-level artifacts are produced as preparation/incomplete
reports with explicit null/missing reasons. They become execution reports only
after real results. Retained historical feature acceptance is a separate column;
it is never counted as a fresh35/35 native run. JUnit represents unexecuted
experiments with explicit `NOT_AUTHORIZED`/`BLOCKED_*` skip types for format
interoperability; JSON statuses remain distinct from pytest SKIP/PASS.

Run the tooling's safe fault-injection tests without any native experiment:

```powershell
python -B -m unittest discover -s tools/full_acceptance/tests -v
```
