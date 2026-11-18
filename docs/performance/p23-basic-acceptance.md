# P23 basic training-memory acceptance — 2026-11-11

The current acceptance scope is **basic training memory**, independently of
activation-lifetime tracing. Model 1 means the existing trained Model A (19,890
parameters); Model 2 means trained Model B (5,166 parameters). Both original
checkpoints, their six completed epochs, configuration/state hashes and real
MNIST train/validation split identities must be verified before validation.

The previous expanded P23 full-scope decision remains historical **PARTIAL**.
This scope change does not complete, fix or accept CUDA saved-activation lifetime,
pack/unpack/release hooks, native hook forensics or activation crash reproduction.
Those features remain experimental and production CUDA admission remains disabled.
Previous forensic snapshots, reports and dumps must be preserved. The other 34
feature decisions are not changed by this acceptance.

## Criteria declared before native basic-memory validation

| ID | Requirement for each model | Required evidence |
| --- | --- | --- |
| B1 | Exact checkpoint/config/dataset/split identity | Read-only SHA256 and loaded state/parameter checks |
| B2 | Parameter, gradient and actual native optimizer tensor bytes | Independent live-object oracle at every in-process boundary; lazy zero-to-state and gradient materialization/clear lifecycle |
| B3 | Current process RSS and sampled maximum | Native OS samples, independent native working-set readback and raw/summary agreement; null/reason if backend invalid |
| B4 | Native RSS lifetime peak when available | HWM labeled process-lifetime; sampled RSS maximum never labeled an exact fit peak |
| B5 | CUDA allocated/reserved/peak allocated/peak reserved | Real production CUDA counters for both models, exact readbacks and valid counter inequalities |
| B6 | Reset ownership and measurement boundaries | In-process reset forbidden; isolated resets only inside owned windows, with before/after/end counters and worker PID; parent allocator state unchanged |
| B7 | Training evolution unchanged | Bit-exact model/gradient/optimizer/Trainer state and Python/NumPy/Torch CPU/CUDA RNG versus the same bounded ordinary fit |
| B8 | Failure cleanup | Controlled Python forward error for each model; recorder/lock/sampler removed, isolated Job active processes zero, failed measurement null |
| B9 | API/state compatibility | Existing signatures/defaults, schema1 payloads/checksums and CSV columns retained; old artifacts reopen unchanged |
| B10 | Advanced mechanisms excluded | Saved-hook/adapter sentinels on basic fits, saved_activation_lifetime=False; CUDA guard regression passes |
| B11 | Preservation and honest result | Existing evidence/checkpoints and other feature results unchanged; no PASS unless all applicable CPU/CUDA criteria for both models are met |

CPU CUDA metrics must be `value=null, status=not_applicable, reason=CPU_training`.
Unmeasured applicable metrics use `unsupported` or `unavailable` plus a reason.
A measured empty gradient/optimizer state can legitimately be numeric zero.

Validation uses bounded **9 train / 5 validation examples, batch 4, two epochs**,
FP32 and the existing fresh AdamW training recipe semantics. This verifies memory
accounting and observer behavior on both trained architectures. It is not a
full dataset, production-batch capacity, throughput, AMP, multi-stream or complete
P01–P35 acceptance run. Those runs need their own current manifests and criteria.

## API contract

Both existing result classes add `basic_metrics()`, an offline projection with
its own `hyso.performance.p23_basic_memory` schema1 identity. `metrics` contains
end-boundary parameter/gradient/optimizer bytes, their baseline and sampled
extrema, current/sampled-peak/lifetime-peak RSS and four CUDA metrics, each with
`value`, `unit`, `status`, `reason` and `scope`.

`status=complete` describes available basic domains, not feature acceptance or a
physical memory claim. Failed isolated workers return `status=not_executed` with
null metrics. In-process peaks retain process/device lifetime scope. Explicitly
reset isolated peaks report the maximum of measured owned training windows;
they exclude factory startup and are not an activation peak.

Existing `to_dict()`, JSON/CSV, protocol/recipe constructors and Trainer/checkpoint
state remain unchanged. Their historical full-scope `partial`/activation labels
are retained for compatibility; use `basic_metrics()` for the revised scope.
The observer performs no cache emptying, in-process peak reset or automatic replay.

## Result

**P23 Basic: PASS. Model A CPU/CUDA: PASS. Model B CPU/CUDA: PASS.**

The retained 109 targeted CPU tests are joined by two real production CUDA
acceptance suites on the same trained checkpoints. Each CUDA model has 76 live
logical/counter oracles, six owned peak-reset windows, bit-exact state/RNG replay,
isolated parent allocator preservation, JSON/CSV reopening and controlled Python
failure cleanup. Saved-activation hooks remain disabled. B1?B11 pass for this
bounded FP32 basic scope, not for experimental activation lifetime or full P01?P35.

The earlier fixed-RAM and workload-budget resource stops remain historical
records. Basic uses workload RAM/commit/VRAM estimates plus OS margins, owned Jobs
and live pressure monitoring. It has no forensic fixed 4 GiB host-free floor,
30-second freshness rule, exact-hash experiment permission or GPU PID review.
The new run began with 4.849 GiB available RAM, 7.893 GiB commit headroom and
5435 MiB free VRAM, and both admitted CUDA suites completed with zero active
owned processes. One test-factory device-placement setup error was preserved,
corrected without changing production Trainer code, and followed by one controlled
corrected A run; no uncontrolled retry or native crash reproduction occurred.

See the [final scoped acceptance report](../reports/p01-p35-final-acceptance-20261112.md).
Earlier CPU and resource-stop reports remain retained local historical evidence;
their PARTIAL decisions are not rewritten by the later Basic PASS result.
