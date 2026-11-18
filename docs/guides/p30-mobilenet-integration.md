# HYSo P30 — genuine pretrained MobileNetV3-Small / OpenCV camera and video adapter

## Contract completion, 2026-11-09

The existing adapter is retained. Configuration and processor now use one exact
contract: BGR uint8 HWC frames, normalized FP32 `[1,3,224,224]` model input,
FP32 `[1,1000]` logits, ImageNet-1K single-label task, ordered category-list hash,
model identity, checkpoint hash and official preprocessing. Acceptance checks
these fields against the actual retained worker header, as well as source kind,
location, provenance and the recorded video's hash. `trained=True` by itself is
insufficient. No classification quality is inferred from unlabeled video; this
is a pretrained real-frame inference/stream-throughput workload. A task such as
person detection needs its own suitable model and label/output semantics.

The local official checkpoint and `vtest.avi` are available and rehashed:

- checkpoint: `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`
- video: `45cddc9490be69345cbdab64ca583be65987e864ca408038e648db99e10516cf`

`--offline --contract-only --config <NEW_PATH>` verifies existing assets and strict
CPU checkpoint loading, then creates a new configuration without video decoding,
camera access or inference. Existing configurations are never overwritten. The
new adapter source hash requires a fresh config/manifest; do not execute old
manifests. This task's config is
`%LOCALAPPDATA%/hyso-p30-assets/p30-contract-20261109-v1.json`.

`--camera-provenance '<actual physical device/source identity>'` is separate from
the video's provenance. It is unset in the prepared contract because no physical
camera was verified. This operator attestation is not independent sensor or
driver identity proof. A file capture cannot establish physical camera success.
P30 promotion now requires both separately authorized successful video and camera
stages, each with its own trained-source criterion and zero active Job processes.

Startup exceptions close both processor and capture. Producer release exceptions
notify the consumer and retain a failed, unclosed terminal rather than leaving
the consumer waiting indefinitely. The outer Job watchdog still bounds blocked
native reads/joins. Existing host frame timestamps, media PTS, bounded queue,
drop/backpressure accounting and journal conservation remain intact; they are
application observations, not exposure timestamps or upstream camera loss.

CPU tests load the genuine checkpoint but use explicitly synthetic frames for
inference, invalid-input and lifecycle injection. These tests are not real video
or physical camera acceptance. P30 remains **PARTIAL**. See
`remaining-acceptance-next-steps.md` for separate video/camera preparation and
explicit authorization commands.

## Purpose and boundaries

This setup is limited to P30, a **read-only real video/camera streaming acceptance**.
It does not train, modify user models, change weights, mark P30 as PASS, or launch
P31. The factory uses official TorchVision ImageNet-1K MobileNetV3-Small weights
and official preprocessing, and processes CPU frame-by-frame. A classifier
provides a bona fide model inference workload, **not** proof of classification
accuracy on an arbitrary video. Metrics are application queue/inference metrics,
not camera sensor or driver frame loss.

Verified upstream: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html

Default public recorded sample: https://github.com/opencv/opencv/blob/4.x/samples/data/vtest.avi

## Install files (do not overwrite any existing work)

Extract the archive at the **HYSo repository root**. It installs three new files:

- `tools/p30_mobilenet_adapter.py` (trusted, pinned factory)
- `tools/p30_mobilenet_prepare.py` (asset downloader + configuration generator)
- `docs/guides/p30-mobilenet-integration.md` (this file)

Your existing manifest/previous experiments are not modified. The downloaded
weights/video and config are stored **outside the repo**, under
`%LOCALAPPDATA%\hyso-p30-assets` by default.

## Prepare official model and real video

In PowerShell at the HYSo repo root:

```powershell
$py = "C:\Users\hdgn5\.conda\envs\torch_gpu\python.exe"
& $py -B tools/p30_mobilenet_prepare.py
```

Downloads official pretrained ~10 MB weights with PyTorch check_hash and a
real recorded public OpenCV sample. Pin full SHA256 for all assets. Validates
checkpoint architecture strictly and runs **one CPU frame** as a smoke test,
not a P30 acceptance. It does not initialize CUDA, retrain, or overwrite your
original MNIST checkpoint. Internet access is needed for first download.

For your own recorded real video instead of OpenCV:

```powershell
& $py -B tools/p30_mobilenet_prepare.py --video "C:\path\to\video.mp4"
```

For a different physical camera index use `--camera-index 1`. If the config
already exists and values differ, supply `--config` with a NEW filename. To
avoid downloads use `--offline` after both assets are already available.

**Important:** 08.camera requires an actually accessible physical camera;
08.video alone can use the supplied recorded real video. Do not substitute
synthetic frames or a virtual camera and claim genuine capture evidence.

## Prepare a new manifest (no long run)

Once setup succeeds, create a **fresh** acceptance manifest from the current
source tree, not from 1 November's old pinned snapshot:

```powershell
$py = "C:\Users\hdgn5\.conda\envs\torch_gpu\python.exe"
$config = Join-Path $env:LOCALAPPDATA "hyso-p30-assets\p30-camera-config.json"
$new = Join-Path $env:LOCALAPPDATA "hyso-acceptance\p30-preflight-new-unique-id"
& $py -B tools/run_full_acceptance.py --preflight --continue-independent --camera-config $config --output $new
```

Use a NEW unique path each time. Review manifest SHA256, preflight outcome and
source assets before authorizing 08.video or 08.camera. The preparation step
can be sizeable because the project snapshots source and computes hashes.
**Do not automatically approve or execute** stages or launch P31. A separate
explicit operator authorization is required to run native acceptance.

## Evidence scope

The factory checks SHA256 before loading weights, uses
`torch.load(weights_only=True)` with strict model keys, loads fully on CPU,
preprocesses real BGR frames using the TorchVision ImageNet-1K recipe, and
returns two measured stages, `preprocess` and `inference`. HYSo performs
worker ownership, queue accounting, journal hashing, and cleanup validation.
The factory returns `identity.trained=True` only after verified official
pretrained weights are loaded.

**Still unavailable:** P13 CUDA kernel-to-module attribution, P23 blocked
CUDA saved-activation native probe, or any classification quality claim about
unlabeled videos. P31/P32 are separate.
