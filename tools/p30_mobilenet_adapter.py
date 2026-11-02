"""HYSo P30 trusted, read-only adapter for an official pretrained MobileNetV3-Small.

No download, training, optimizer, checkpoint save or user model modification occurs
in the native P30 worker. The caller pins checkpoint and source SHA256 in the
fresh acceptance manifest. CPU processing and provenance are explicit.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import time


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_processor(*, checkpoint_path: str, checkpoint_sha256: str):
    """Factory used by full_acceptance.workloads:frame_processor inside owned worker.

    The official torch.hub weight file is loaded with weights_only=True and
    strict state-dict validation. No network is used here.
    """
    import torch
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    from hyso.performance.streaming import FrameInferenceProcessor

    ckpt = Path(checkpoint_path).resolve(strict=True)
    if ckpt.suffix.lower() != ".pth" or not ckpt.is_file():
        raise ValueError("explicit pretrained .pth checkpoint required")
    if len(checkpoint_sha256) != 64 or _sha256(ckpt) != checkpoint_sha256:
        raise ValueError("pinned pretrained checkpoint SHA256 mismatch")
    # The official 0.20.x URL embeds this SHA256 prefix. Full digest is pinned
    # separately in the configuration; prefix alone is not used for ownership.
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    if weights.url.rsplit("/", 1)[-1] != ckpt.name:
        raise ValueError("checkpoint is not the canonical official MobileNetV3-Small filename")
    if not checkpoint_sha256.startswith("047dcff4"):
        raise ValueError("official torchvision pretrained weight hash prefix mismatch")

    model = mobilenet_v3_small(weights=None)
    try:
        state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("PyTorch must support torch.load(weights_only=True)") from exc
    if not isinstance(state, dict) or not state:
        raise ValueError("nonempty pretrained state dict required")
    model.load_state_dict(state, strict=True)
    model.eval()

    class MobileNetProcessor(FrameInferenceProcessor):
        def __init__(self):
            super().__init__(model, size=(224, 224))
            # Canonical torchvision preprocessing includes resize (short edge
            # 256), center crop 224, RGB [0,1] normalization. The base class's
            # simple resize is NOT used by the overridden process().
            self.transform = weights.transforms()
            self.identity.update(
                trained=True,
                model_id="torchvision_mobilenet_v3_small_imagenet1k_v1",
                origin={"kind": "official_pretrained_checkpoint", "source": weights.url,
                        "checkpoint_sha256": checkpoint_sha256},
                checkpoint_sha256=checkpoint_sha256,
                input_contract={"format": "OpenCV BGR uint8", "shape": "H,W,3", "source": "real_camera_or_video"},
                output_contract={"task": "ImageNet-1K classification", "shape": [1, 1000],
                                 "class_count": 1000, "labeled_quality_claim": False},
                preprocessing={"color": "BGR_to_RGB", "dtype": "float32", "scale": "divide_255",
                               "resize_short_edge": 256, "center_crop": [224, 224],
                               "mean_rgb": [0.485, 0.456, 0.406],
                               "std_rgb": [0.229, 0.224, 0.225],
                               "recipe": "torchvision.IMAGENET1K_V1.transforms"},
                inference_scope="CPU_real_frame; throughput_not_classification_accuracy",
            )

        def process(self, frame):
            if frame is None or frame.dtype.name != "uint8" or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("expected nonempty OpenCV BGR uint8 HWC frame")
            if not frame.size:
                raise ValueError("empty camera frame")
            a = time.perf_counter_ns()
            x = torch.from_numpy(frame[:, :, ::-1].copy()).permute(2, 0, 1).float().div_(255.0)
            x = self.transform(x).unsqueeze(0)
            b = time.perf_counter_ns()
            with torch.inference_mode():
                output = self.model(x)
            c = time.perf_counter_ns()
            self.output_check(output, 1, device=torch.device("cpu"))
            if tuple(output.shape) != (1, 1000) or not torch.isfinite(output).all().item():
                raise ValueError("invalid ImageNet classifier output")
            self.guard.check(full=True)
            return [dict(name="preprocess", start_ns=a, end_ns=b),
                    dict(name="inference", start_ns=b, end_ns=c)]

    return MobileNetProcessor()
