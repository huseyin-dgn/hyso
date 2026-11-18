# P34 device comparisons

`compare_devices` keeps the serial owned-worker adapter. New results use
`comparison_version=3`; RuntimeResult schema v1 and the public call signature
are unchanged. Versions 1 and 2 retain their exact original analysis, hashes
and canonical JSON/CSV envelopes.

Offline replay needs no model construction, sensor query or CUDA context:

```python
from hyso.performance import RuntimeResult

result = RuntimeResult.from_json("comparison.json")
diagnostics = result.device_diagnostics(baseline_device="cpu")
```

The baseline defaults to CPU when present, otherwise the first recorded device.
An explicit baseline must be in the recording. Each candidate is compared with
that baseline; no automatic winner is selected. Diagnostics are included in v3
analysis and survive lossless JSON/CSV reopening. Diagnostics computed for an
old result are a separate view and do not modify its original envelope.

The existing validation checks physical UUID mapping, actual residency and
precision, successful completion, sample count, model/input immutability,
request binding and Job cleanup. New captures also record exact local host
window and per-forward boundaries, ordering, an independently observed loaded
state hash, and origin/preprocessing metadata. Local durations are compared;
absolute timestamps from different workers are never subtracted. The clock
instance identifies the reporting worker, not the CUDA device clock.

The model hash includes structure, class definitions and state. The new loaded
state hash is separately calculated as canonical JSON SHA256 of named state-dict
tensor hashes, each binding dtype, shape and contiguous raw bytes. This algorithm
is not interchangeable with every older `state_sha256` field in other features.

Factories still return `(CPU module, CPU floating NCHW tensor)`. Such captures
claim only the recorded model state, input content and factory/configuration;
they do not establish a trained-checkpoint origin. The worker also accepts an
optional third plain dictionary:

```python
origin = dict(
    kind="trained_checkpoint",  # or "controlled_fixture"
    model_id="reviewed model identity",
    checkpoint_path="C:/absolute/checkpoint.pth",
    checkpoint_sha256="<64 lower-case hex digits>",
    preprocessing_id="reviewed preprocessing description",
    preprocessing_sha256="<64 lower-case hex digits>",
    expected_loaded_state_sha256="<optional observed-state digest>",
)
return model, inputs, origin
```

Checkpoint fields are required for `trained_checkpoint` and are unused for a
controlled fixture. The worker checks the checkpoint's bytes before and after
measurement; when supplied, the expected loaded-state hash must match. Factories
are trusted executable code. Their training/preprocessing declarations are
recorded as declarations, not independently authenticated training history.
Input content, module source and actual loaded state are observed separately.
Replay validates the recorded evidence without requiring original local files.
Hashes detect inconsistency; they do not provide signed authenticity.

Comparability includes model/state/checkpoint, actual input, factory entrypoint
and configuration, preprocessing, shape/batch, precision/backend/runtime, CPU
threads and execution flags, warmup/sample count/seed, output shape, and timing
scope. Differences are named and yield `not_comparable + reason` with null ratios.
Missing trained-checkpoint evidence prevents trained-model acceptance, while an
explicitly scoped same-state fixture comparison can still be descriptive.
Legacy v1 lacks the independent execution mapping; its original analysis remains
available but stricter diagnostics do not promote it to a verified comparison.

Mean and P50 describe completed batch forwards. P95/P99 require at least 1,000
recorded samples (at least ten observations in the P99 tail). Otherwise they
are null with an insufficient-sample reason. This policy does not prove sample
independence. Repeated-run distributions and confidence intervals are null when
not measured; a single serial capture has no superiority guarantee.

For comparable pairs only:

- Latency difference = candidate mean minus baseline mean, in ms per batch.
- Percentage difference = 100 times that difference divided by baseline mean.
- Latency speedup = baseline mean divided by candidate mean; above one means
  the candidate has lower observed latency.
- Throughput ratio = candidate completed images/s divided by baseline images/s.

Forward latency includes host launch and terminal CUDA synchronization; the
pre-drain occurs outside it. Initial transfers, model loading and warmup are
excluded. Output validation and heartbeat writes are inside the instrumented
throughput window and outside forward latency. Completed forwards/s and images/s
have separate counts. Throughput is not inferred by inverting mean latency.
Cold-start is not measured and reaching a thermal steady state is not asserted.

RSS snapshots, CUDA allocated/reserved window peaks and logical model bytes keep
their own unit/domain/scope. Missing memory is not zero. Private commit is
explicitly unavailable in this capture, and RSS never substitutes for it.
Process/runtime and allocator values are shown side by side without a model
memory superiority claim; only matching logical-model domains are eligible for
the corresponding byte difference. Energy and quality remain null unless a
separate matching adapter records them; P34 does not borrow P33 power data.

No allocator fraction, RAM quota or artificial batch cap is introduced.
Existing ownership, finite deadlines/evidence bounds, OS pressure monitoring,
cleanup and crash-stop remain active. New GPU/risky native measurements need a
separate exact manifest and explicit approval. Old GPU approvals and the deferred
P31 manifest do not authorize them.
