# P23 acceptance: training memory analysis

**P23 = COMPLETED for the original roadmap's measured scopes.** This covers
registered parameter/gradient/native-optimizer accounting, optional CPU eager
saved-for-backward evidence, native process RSS observations and supported CUDA
allocator/window peaks. CUDA saved-activation tracing remains disabled and
unsupported. Neither all activation storage nor a physical activation peak is
claimed on either device.

The original P23 row in
`a0672d5:docs/roadmap/hyso-performance-backlog-v1.1.md` requires separate
components and measured peaks, logical bytes distinguished from process and
allocator bytes, and native snapshots/probes within the same training boundary.
It does not require an ownership graph or exact physical allocation attribution
to every component. CPU activation evidence is not substituted for CUDA
activation evidence, and separate executions are not merged into one capture.

The review used a new worktree based on verified remote HEAD
`4dd4c3d33f932393a3ef2d3fe85e42e8d3c479f3`. Existing production sources and
tests were preserved. The previously written training-memory guide was present
in local commit `47a007f` but absent from this remote base; it is included here
with the final acceptance matrix and OOM interpretation guidance.

## Acceptance matrix

| Original requirement | Result | Concrete evidence |
| --- | --- | --- |
| 1. Parameter, gradient, optimizer and activation shown separately | PASS, with explicit device coverage | `test_real_logical_gradient_and_lazy_optimizer` checks actual tensor bytes for SGD/Adam/AdamW and FP32/FP64. A saved native CPU run has 60 saved-for-backward records, with distinct logical/storage summaries; CUDA activation stays unsupported. |
| 2. Logical bytes separate from allocator/RSS | PASS | `test_mixed_frozen_tied_view_and_shared_storage` checks ties, views and deduplication. `test_stage_pairs_and_independent_memory_changes_preserve_exports` checks separate domains. Native CPU saved logical peak 176 byte, saved storage peak 160 byte and nonparameter saved storage peak 128 byte remain different quantities. None is a physical activation peak. |
| 3. Correct CPU RSS and CUDA allocated/reserved scope | PASS | CPU tests observe native process RSS; `test_rss_sampler_bound_and_midrun_missing_backend` verifies missing-backend handling. Recorded CUDA core and isolated results reopen with allocated/reserved fields. RSS excludes worker processes; allocator counters exclude non-allocator/device-wide VRAM. Invalid RSS/high-water observations remain unsupported. |
| 4. Real training-boundary baseline/peak observations | PASS, according to each domain's scope | Paired boundary queries bracket forward/backward/step/zero-grad. Native RSS maxima are labeled sampled. The saved hook-free CUDA run has 64 boundary records and four true allocator peak windows, each with an owned-worker reset, previous/baseline counters and completed end counters. No parent peak reset is used. |
| 5. Gradient lifecycle and optimizer-state changes | PASS | `test_accumulation_gradient_lifecycle_and_attempt_state`, `test_sgd_no_momentum_and_zero_grad_false` and native logical tests distinguish accumulation, absent gradients, retained zeroed gradients and lazy optimizer state. Recorded CUDA AdamW state grows from 0 to 224 logical byte while parameters remain 104 byte. Historical AMP overflow records retain skipped attempts. |
| 6. Model/RNG/gradient/optimizer/hook/CUDA state preserved | PASS, relative to the requested execution | Current baseline/observed CPU replay compares model, optimizer, gradients and RNG. Direct activation tests verify original objects, aliasing, native in-place version rejection and cleanup after failure. Saved isolated CPU/CUDA acceptance verifies one recipe fit and parent state preservation; the hook-free CUDA record explicitly preserves parent allocator counters and RNG. |
| 7. Native CPU/CUDA acceptance and JSON/CSV compatibility | PASS, within supported scopes | Saved CPU activation and hook-free CUDA core/window records validate through current result classes. Legacy schema fixtures and historical CUDA success/failure records reopen. JSON reopening, legacy CSV columns and boundary/window/activation rows were checked independently. Unsupported and failed outcomes do not become measurements. |

## Evidence and verification

- **112 targeted CPU/protocol tests passed** across `test_training_memory.py`,
  `test_saved_activation.py` and `test_isolated_training_memory.py`. Thirty-three
  worker-dependent tests were deselected; existing native evidence was reviewed
  first instead of launching another worker/GPU experiment.
- The previous CPU acceptance's **84 distinct passing tests** were checked
  against its XML records, including its five explicitly recovered setup errors.
  Those historical tests are not added to the new test count.
- The saved native CPU activation run has 60 pack events, 48 unpack events,
  60 native release events and zero live saved payloads at completion. Native
  RSS samples remain associated with that CPU fit. All nine rehashed activation
  tamper checks reject modified evidence using this saved result.
- A saved hook-free CUDA acceptance contains both in-process observation and
  an owned isolated fit. Both record exact training-state/RNG agreement; the
  isolated case also records unchanged parent allocator state. Its four window
  peaks are 68166656, 68168704, 68170752 and 68168704 allocated byte; reserved
  peak is 69206016 byte in every window. These are whole allocator-window
  observations for that workload, not activation bytes or physical VRAM.
- The historical **21-case hook-free CUDA matrix** reopens: 18 optimizer/AMP/
  selected-stream combinations, one no-reset case, one controlled OOM and one
  timeout. These are historical native executions, not new current-device runs.
  The original native-crash artifact also reopens as `native_crash`, with no
  completed measurement.
- Saved CPU activation, CUDA core and CUDA window results were exported again:
  JSON payloads reopen unchanged; CSV columns and observation values retain
  their existing formats (236, 64 and 68 rows respectively). The legacy
  in-process fixture also remains supported. Seventy-one protected existing
  files retained their hashes.

No new GPU workload, isolated worker, profiler or crash reproduction was run.
The production modules and tests match the previously reviewed `47a007f` tree;
this closure adds user-facing documentation only.

## Retained limits and crash handling

The native CUDA `0xC0000409` crash root cause remains unresolved. The independently
fixed saved-tensor version-check defect is not presented as its cause or cure.
The CUDA activation gate still rejects that profile before allocator or saved
hook setup; its regression test verifies this with forbidden-call sentinels.
The safe alternative is the already recorded hook-free CUDA allocator path.

CPU saved-payload lifetimes describe autograd slots, including any saved
parameters/buffers. They do not cover every activation or prove final storage
deallocation. CPU RSS extrema remain sampled; exact CPU per-stage RSS peaks,
physical activation peaks, CUDA activation attribution and hidden-stream
attribution remain unsupported/null. These limits are visible in each result.

An in-process measurement intentionally performs the caller's one requested fit;
preservation means the same training evolution as the baseline, with observer
cleanup afterward. Isolated profiling constructs a separate workload and does
not serialize or train the caller's live model. Only an explicitly owned CUDA
worker with `allow_peak_reset=True` resets its own peak counters.

Advanced profiling remains opt-in. RAM/job-memory limits and CUDA allocator
fractions default to unset, and no device-occupancy admission gate was added.
OOM counters, logical component growth and allocator peaks can guide an
investigation; they do not establish a unique physical OOM cause or a guaranteed
memory saving. Raw `partial` coverage statuses are retained, rather than changed
to claim unmeasured domains.

See [Training memory analysis](training-memory.md) for API usage and scope details.
