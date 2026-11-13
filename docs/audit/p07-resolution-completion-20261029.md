# P07 resolution sweep acceptance ? 2026-10-29

**CPU 8/8 PASS; CUDA 8/8 PASS. P07 PASS in the verified scope below.**

The same trained Model B ModularCNN (widths4/8/16,5166 parameters) completed
5x7,9x13,320x320,416x416,640x640,640x384,384x640 and960x960.
Input was exact synthetic contiguous NCHW [1,1,H,W],FP32; every measured
output satisfied finite [1,10] on its selected device. Checkpoint SHA256:
`8279ea41b6b26d9d5e0454327d4548e78fa89ae70a7cd6b7948e76ed6446aaef`.
No interpolation,model changes,training or quality claim was introduced.

Each point retained3 repeats of50 latency samples with10 warmup forwards per
repeat,and a separate native completion-wall throughput protocol of3x20 batches.
CPU wall timing and CUDA event timing remain separate domains. Allocator probes
are separate single-forward observations,not model-only physical VRAM.

The approved v3 GPU manifest was
`1c29f38e9cdcebfd4c79331578885899a251c5e1d9c4a538883b02a6a08909b4`.
Fresh resource/PID/source gates passed. One GPU worker completed in19.703s,
exit0,owned Job active0; no OOM,timeout,native crash or retry. CPU was not rerun.

The production fix stops further point admission after known CPU allocator
RuntimeError or MemoryError,retaining failed/out_of_memory and subsequent
not_run rows. Public API and result schemas remain unchanged. Existing evidence:
53 focused tests passed; the3-case regression failed before and passed after
the fix. No tests or benchmarks were rerun for this commit.

The earlier CPU helper's post-export reader error(exit1) and its successful
native-constructor offline recovery remain preserved. Earlier blocked GPU
admissions,old reports and all failure evidence were retained,including the
reports-before-cuda-v3 archive. No historical acceptance matrix was rewritten.

**B4 ROOT_CAUSE_UNRESOLVED; reliability risk remains OPEN.** This scoped P07
validation does not close B4,approve a stable release,establish independent
improvement,or validate all models/devices/shapes.

Local ignored evidence directory: `outputs/p07-resolution-completion-20261029/`.
Large raw outputs/checkpoints/data/temporary scripts are deliberately not committed.

| Evidence | SHA256 |
|---|---|
| cpu-resolution-results.json | `f4fc2169b95082221d0e1c88b2ef0c52b2f24a2358feaa3a1a07af44fbdfbb57` |
| gpu-resolution-results-v3.json | `3ee5e3c058b2ccb0da22647b84f72aa16107e0b6ba826b48f62b58f342835c2d` |
| focused-tests.xml | `74e34528cf350a18d84f560cbfcc698e3ad6281f8960acb6414e7d381a19c94b` |
| cuda-v3-verification.json | `c0579e95fdae3a22f4b4f5621b1118b68821b488eb1f709a65d9390b4fa0f09b` |
