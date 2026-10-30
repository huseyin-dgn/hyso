# P30 recorded video and camera measurements

`measure_streaming(StreamSource(...), WorkloadRecipe(...), directory, protocol=...)`
uses an owned Windows Job, OpenCV capture and a bounded application FIFO. The
directory must be new. A trusted factory returns a processor with `identity`,
`process(frame)` and `close()`. `FrameInferenceProcessor` runs CPU inference on
an owned model copy after explicit BGR uint8 → RGB float32 /255 and bilinear
resize. Select a model whose input, normalization and output contracts match
that preprocessing; an MNIST checkpoint does not match this RGB adapter.

The original `max_frames=16` default remains a finite smoke experiment. It is
not a hardware capacity limit. Callers can select larger positive frame counts,
FIFO capacities, resize dimensions and finite timeouts. The former 10,000-frame,
16-slot, 3,600-second and 16 MiB/frame ceilings are removed. The queue remains
bounded by the selected capacity. P30 has no implicit RAM/VRAM quota; the parent
observes Job commit peaks and Windows low-memory notifications and stops the
whole Job on OS resource pressure. Startup, heartbeat and duration watchdogs,
process ownership and cleanup remain active. `max_journal_bytes` is an explicit
evidence budget, default 128 MiB, distinct from memory capacity. Existing generic
runtime JSON/CSV readers retain their 128 MiB input guard.

New captures record lifecycle version 2. Received events include successful
read start/end, ID assignment, frame shape/bytes and decoded content SHA256.
Enqueue, dequeue, backpressure, processing stage brackets, completion and drop
events share the header's worker `perf_counter_ns` clock ID. Media PTS is a
separate millisecond clock. Controller resource monitoring has a separate named
clock domain. No media/host/controller duration is inferred across domains.

The analysis distinguishes:

- `video_metadata_fps`: the file/driver declaration, not observed throughput.
- `capture_read_completion_rate_per_second`: successful reads divided by the
  first-read-start to last-read-end window. Producer stalls, pacing and journal
  overhead can affect this window.
- `completed_processing_fps`: completed frames divided by first-read-start to
  final lifecycle event, available only for verified complete runs.
- `inference_calls_per_active_second`: completed calls divided by summed host
  inference brackets; this is not camera FPS or a sustained model benchmark.
- `latency_ms`: read, queue wait (enqueue→dequeue), preprocess, inference,
  optional postprocess, processing service and receive→completion distributions.
  P50/P95 linearly interpolate at `(n-1)*q`; no sample gives null with a reason.
- `drop_rate`: application drops / received frames. `late_rate`: late completed
  frames / processed frames. Late means receive→completion strictly exceeds the
  configured deadline. A dropped frame is not also a late completion.
- `queue_occupancy`: FIFO time-weighted mean and final depth, excluding the
  consumer's in-flight frame and producer's admission wait. Maximum depth and
  paired backpressure duration are separate fields.

Display FPS, sensor FPS and upstream/driver losses are null with explicit
reasons. The adapter does not present frames. Postprocessing is unsupported
unless the processor supplies its actual stage bracket. Output/state validation
overhead remains in processing service but outside model inference brackets.
These are descriptive observations, not an optimization or bottleneck proof.

Each committed prefix validates `received = processed + dropped + pending`,
contiguous unique IDs, FIFO capacity/order, source/recipe/clock association,
stage ordering and terminal accounting. A crash may lose the final buffered
journal rows (chunks contain at most 64 rows); an aborted prefix is explicitly
not full-run accounting. Pending or unknown work is never invented as a drop.

`RuntimeResult.from_json` reopens legacy results with their original derived
analysis. New P30 CSV files use the existing canonical runtime envelope and
support `RuntimeResult.from_csv`, including empty lists, nulls and failed setup.
The original flattened P30 CSV format stays an export format for legacy results;
it has no lossless reader. Raw journal reopen uses the same hash-chain validator.

For the local video audit, run `tools/p30/acceptance.py --video ABSOLUTE_PATH
--output NEW_DIRECTORY` with the CPU-capable project interpreter. It independently
decodes to EOF in an owned probe, then measures both policies on the full file
using an explicitly untrained, deterministic three-channel fixture. It writes
separate protocols, JSON/CSV, per-frame lifecycle and per-event conservation
tables. Synthetic regression fixtures do not establish real-video acceptance.

Live camera access requires separate explicit consent before opening a device.
The P30 processing adapter is CPU-only. New CUDA workloads require their own
manifest, SHA256 and authorization; historical approvals are not reused.
