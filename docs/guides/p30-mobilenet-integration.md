# HYSo P30 — genuine pretrained MobileNetV3-Small / OpenCV camera and video adapter

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
