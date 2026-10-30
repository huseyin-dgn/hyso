# P35 review — 2026-10-30

**P35 PASS within the verified ONNX Runtime CPU/CUDA node-placement scope.**
Genuine archived runtime profiles establish CPU execution, CUDA execution,
observed mixed CUDA/CPU placement mismatch and an internal transfer event.
Constructed protocol traces are separate regression evidence. No new GPU run
is needed for this scoped acceptance.

The [14-part audit](../../outputs/p35-provider-fallback-20261030/final-report.md)
records the unchanged original criteria, exact source identities, regression
results, legacy compatibility and preservation. The
[guide](../guides/performance-providers.md) defines the new evidence assessment,
clock/coverage rules and unsupported fields. The
[older acceptance report](p35-acceptance.md) remains unchanged as historical
evidence.

This acceptance does not establish trained Model A/B ONNX deployment: their
checkpoint bytes are verified, but a matching trained ONNX export was not found
in the inspected sources. No conversion was performed. Other backends,
arbitrary fusion mapping, complete static assignment, physical UUID execution
attestation, DMA byte counters, device timing, error-retry transitions and causal
performance effects remain unverified or unsupported with reasons.

P30/P31/P33/P34 remain PARTIAL; P32 remains PASS within verified CPU/offline scope.
B4 and CPU65536 remain OPEN. P01–P29 acceptance decisions are untouched. P31's
7200-second run is not started. P35 ends here; no aggregate acceptance inventory,
integration tests, unified API/reporting system, builder or UI is developed.
