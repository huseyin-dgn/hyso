"""Prepare a real, pretrained P30 image/video adapter without any native acceptance run.

Downloads only public upstream sample assets when absent. It never writes a user
checkpoint, changes any existing HYSo source, trains a model, or invokes P31.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request

VIDEO_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi"
WEIGHT_NAME = "mobilenet_v3_small-047dcff4.pth"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download_file(url, dest, max_bytes):
    """Atomic bounded download; never overwrite existing files."""
    dest = Path(dest)
    if dest.exists():
        print("Existing video reused, SHA256 will be pinned:", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(prefix=dest.name + ".", suffix=dest.suffix, dir=dest.parent, delete=False) as f:
            temp = Path(f.name)
            request = urllib.request.Request(url, headers={"User-Agent": "HYSo-P30-preparation/1.0"})
            with urllib.request.urlopen(request, timeout=45) as response:
                if response.status != 200:
                    raise RuntimeError("official sample video fetch returned HTTP " + str(response.status))
                total = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError("official video exceeds bounded download size")
                    f.write(chunk)
        if temp.stat().st_size < 10000:
            raise ValueError("video source file is unexpectedly small")
        # Before publication, validate that OpenCV can decode the sample video.
        validate_video(temp)
        if dest.exists():
            raise FileExistsError("destination created during download; do not overwrite")
        temp.rename(dest)
        temp = None
        return dest
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def validate_video(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("cannot decode real recorded video: " + str(path))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, image = cap.read()
        if frames < 120 or not ok or image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("video must provide >=120 decodable BGR frames")
        print("Video verified:", frames, "declared frames; first frame:", list(image.shape))
        return image
    finally:
        cap.release()


def prepare(args):
    # Do not initialize CUDA; all setup validation is on CPU.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    import torch
    import torchvision
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    import cv2

    repo = Path(__file__).resolve().parents[1]
    module_path = Path(__file__).with_name("p30_mobilenet_adapter.py").resolve(strict=True)
    if not (repo / "hyso" / "performance" / "streaming.py").is_file():
        raise RuntimeError("run the bundle from the HYSo repository root, with tools/*.py in tools/")
    if args.camera_index < 0 or args.camera_index > 32:
        raise ValueError("camera index must be 0..32")
    root = (Path(args.asset_dir) if args.asset_dir else Path(os.environ.get("LOCALAPPDATA", Path.home())) / "hyso-p30-assets").resolve()
    root.mkdir(parents=True, exist_ok=True)

    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    if weights.url.rsplit("/", 1)[-1] != WEIGHT_NAME:
        raise RuntimeError("the locally installed torchvision uses unexpected official weight URL")
    if args.offline:
        checkpoint = root / WEIGHT_NAME
        if not checkpoint.is_file():
            raise FileNotFoundError("offline mode requires pretrained official weights at " + str(checkpoint))
    else:
        # Torch downloader performs hash prefix verification when downloading.
        # Cache is pinned to our new, separate asset directory.
        print("Obtaining official pretrained weights:", weights.url)
        torch.hub.load_state_dict_from_url(weights.url, model_dir=str(root),
                                           map_location="cpu", check_hash=True,
                                           progress=True, weights_only=True)
        checkpoint = root / WEIGHT_NAME
    cp_hash = sha256(checkpoint)
    if not cp_hash.startswith("047dcff4"):
        raise ValueError("checkpoint does not match the official ImageNet weight SHA256 prefix")
    # Verify the state dict matches the requested architecture (never train or save).
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    base = mobilenet_v3_small(weights=None)
    base.load_state_dict(state, strict=True)
    print("Strict pretrained MobileNetV3-Small state_dict: verified")
    del base, state

    if args.video:
        video = Path(args.video).resolve(strict=True)
        provenance = "User-supplied real recorded video; original path and SHA256 pinned"
    else:
        if args.offline and not (root / "vtest.avi").is_file():
            raise FileNotFoundError("offline mode requires a video at " + str(root / "vtest.avi"))
        video = root / "vtest.avi"
        if not video.exists():
            print("Downloading public OpenCV real recorded sample video:", VIDEO_URL)
            download_file(VIDEO_URL, video, max_bytes=48*1024*1024)
        provenance = "OpenCV opencv/4.x samples/data/vtest.avi real recorded sample; " + VIDEO_URL
    frame = None if args.contract_only else validate_video(video)

    if __package__:
        from .p30_mobilenet_adapter import contract, build_processor
    else:
        from p30_mobilenet_adapter import contract, build_processor
    config = dict(
        factory="p30_mobilenet_adapter:build_processor",
        factory_source=dict(path=str(module_path), sha256=sha256(module_path)),
        checkpoint=dict(path=str(checkpoint), sha256=cp_hash),
        **contract(weights),
        provenance=provenance,
        kwargs=dict(checkpoint_path=str(checkpoint), checkpoint_sha256=cp_hash),
        camera_index=args.camera_index,
        camera_provenance=args.camera_provenance,
        video=dict(path=str(video), sha256=sha256(video)),
    )
    target = (Path(args.config) if args.config else root / "p30-camera-config.json").resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        old = json.loads(target.read_text(encoding="utf-8-sig"))
        if old != config:
            raise FileExistsError("existing config differs, refuse to overwrite: " + str(target))
        print("Existing identical configuration retained:", target)
    else:
        target.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if not args.skip_smoke and not args.contract_only:
        sys.path.insert(0, str(module_path.parent))
        processor = build_processor(checkpoint_path=str(checkpoint), checkpoint_sha256=cp_hash)
        try:
            stages = processor.process(frame)
            assert [s["name"] for s in stages] == ["preprocess", "inference"]
            assert processor.identity["trained"] is True
            print("Real recorded frame -> pretrained CPU inference: SMOKE PASS")
        finally:
            processor.close()
    print("\nP30 ASSETS PREPARED (not P30 acceptance PASS)")
    print("Torch:", torch.__version__, "Torchvision:", torchvision.__version__, "OpenCV:", cv2.__version__)
    print("Config:", target)
    print("Checkpoint SHA256:", cp_hash)
    print("Video SHA256:", config["video"]["sha256"])
    print("Video:", video)
    print("Camera index selected (not yet validated):", args.camera_index)
    print("Model/code/checkpoint are read-only inside HYSo worker. No native acceptance has run.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", help="existing real video file (otherwise download OpenCV's real recorded vtest.avi)")
    p.add_argument("--camera-index", type=int, default=0, help="physical OpenCV camera index, default 0")
    p.add_argument("--camera-provenance", help="operator's physical camera identity/source; required before live acceptance")
    p.add_argument("--asset-dir", help="separate output directory for official assets/config")
    p.add_argument("--config", help="output config path (existing differing files never overwritten)")
    p.add_argument("--offline", action="store_true", help="do not download; assets must be present")
    p.add_argument("--skip-smoke", action="store_true", help="skip one real-video-frame CPU smoke")
    p.add_argument("--contract-only", action="store_true", help="existing assets only; no video decode, camera or inference")
    args = p.parse_args()
    if args.contract_only and not args.offline:
        p.error('--contract-only requires --offline')
    prepare(args)


if __name__ == "__main__":
    main()
