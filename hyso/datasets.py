"""Explicit small file-backed classification splits using torchvision ImageFolder.

No downloading, random splitting, augmentation or training. PNG/JPEG, strict RGB
or grayscale source channels, deterministic resize/ToTensor and optional explicit
normalization. Content checks detect exact decoded/preprocessed cross-split reuse,
not semantic near-duplicates or independent dataset origin.
"""

from dataclasses import dataclass
from functools import partial
import hashlib
import io
import json
import math
from pathlib import Path
import warnings

from PIL import Image, UnidentifiedImageError, __version__ as pillow_version
import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets, transforms, __version__ as torchvision_version

from .model_checks import preflight_model


class ImageDatasetError(ValueError):
    """Invalid file/data/model contract; original Core diagnostics are retained."""

    def __init__(self, message, *, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _read_image(path, *, channels, expected=None):
    path = Path(path)
    try:
        with path.open("rb") as stream:
            raw = stream.read(32 * 1024**2 + 1)
        if len(raw) > 32 * 1024**2:
            raise ImageDatasetError(f"Image exceeds 32 MiB file budget: {path}")
        if expected is not None and _digest(raw) != expected[str(path)]:
            raise ImageDatasetError(f"Image changed since dataset validation: {path}")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format not in ("PNG", "JPEG"):
                    raise ImageDatasetError(
                        f"Unsupported image format {image.format}: {path}"
                    )
                if image.width * image.height > 16_777_216:
                    raise ImageDatasetError(
                        f"Decoded image exceeds 16M-pixel budget: {path}"
                    )
                mode = "L" if channels == 1 else "RGB"
                if image.mode != mode:
                    raise ImageDatasetError(
                        f"Channel mismatch: expected {mode} ({channels}), got {image.mode}: {path}"
                    )
                image.load()
                return image.copy(), _digest(raw)
    except ImageDatasetError:
        raise
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise ImageDatasetError(f"Cannot decode image {path}: {error}") from error


def _checked_loader(path, *, channels, expected):
    return _read_image(path, channels=channels, expected=expected)[0]


@dataclass(frozen=True)
class ImageClassificationSplits:
    """Standard ImageFolder datasets plus a defensive content/preprocessing manifest."""

    train: datasets.ImageFolder
    val: datasets.ImageFolder | None
    test: datasets.ImageFolder | None
    _manifest_json: str

    @property
    def classes(self):
        return tuple(self.to_dict()["classes"])

    def to_dict(self):
        return json.loads(self._manifest_json)

    def make_loaders(self, training, *, seed):
        """Reuse TrainingConfig loaders; explicitly seed independent CPU generators."""
        settings = training.to_dict()
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ImageDatasetError("seed must be a nonnegative integer below 2**63")
        if (
            settings["device"] != "cpu"
            or settings["precision"] != "fp32"
            or settings["task"] != "multiclass"
        ):
            raise ImageDatasetError(
                "This image pipeline supports CPU fp32 multiclass loaders"
            )
        train, val = training.make_loaders(self.train, self.val)
        test = (
            None
            if self.test is None
            else DataLoader(
                self.test,
                batch_size=settings["batch_size"],
                shuffle=False,
                num_workers=0,
            )
        )
        for offset, loader in enumerate((train, val, test)):
            if loader is not None:
                loader.generator = torch.Generator(device="cpu").manual_seed(
                    (seed + offset) % 2**63
                )
                if isinstance(loader.sampler, RandomSampler):
                    loader.sampler.generator = loader.generator
        return train, val, test

    def validate_model(self, config, *, output_name=None):
        """Reuse Core preflight, then match dataset geometry/class vocabulary."""
        report = preflight_model(config)
        if not report["valid"]:
            raise ImageDatasetError(
                "Invalid model: " + report["diagnostics"][0]["message_en"],
                diagnostics=report["diagnostics"],
            )
        if config.get("schema_version") == 3:
            if len(config["inputs"]) != 1:
                raise ImageDatasetError("Image classification requires one graph input")
            if output_name not in config["outputs"]:
                raise ImageDatasetError(
                    "Select an explicit graph classification output"
                )
            output = report["outputs"][output_name]
            if output["feature_meaning"] != "classification":
                raise ImageDatasetError("Selected output must be a classification head")
            shape = next(iter(config["inputs"].values()))["shape"]
            width = output["shape"][-1]
        else:
            if output_name is not None:
                raise ImageDatasetError(
                    "Named output selection is only for graph models"
                )
            if (
                config.get("kind") == "sequential"
                and config["head"]["name"] != "classification_head"
            ):
                raise ImageDatasetError("Only classification heads are supported")
            shape = config.get("input_shape")
            width = report["output_shape"][-1]
        preprocessing = self.to_dict()["preprocessing"]
        expected = [preprocessing["channels"], *preprocessing["image_size"]]
        if shape != expected:
            raise ImageDatasetError(
                f"Dataset/model input channel or shape mismatch: dataset {expected}, model {shape}"
            )
        if width != len(self.classes):
            raise ImageDatasetError(
                f"Class count mismatch: dataset {len(self.classes)}, model {width}"
            )
        return report


def load_image_splits(
    root,
    *,
    image_size=(8, 8),
    channels=3,
    mean=None,
    std=None,
    max_samples_per_split=128,
):
    """Validate train/optional val/test folders without mixing or modifying files.

    Every present split must have the same nonempty class folders. Train defines
    the sorted vocabulary. No implicit validation/test split is created. A changed
    source file is rejected when read again. Strict source modes are RGB/L; no
    implicit channel conversion. Files <=32 MiB, Pillow decompression guards.
    """
    root = Path(root).resolve()
    if channels not in (1, 3) or type(channels) is not int:
        raise ImageDatasetError("channels must be 1 or 3")
    if (
        not isinstance(image_size, (tuple, list))
        or len(image_size) != 2
        or any(type(v) is not int or v < 1 for v in image_size)
    ):
        raise ImageDatasetError("image_size must contain two positive integers (H, W)")
    if type(max_samples_per_split) is not int or max_samples_per_split < 1:
        raise ImageDatasetError("max_samples_per_split must be positive")
    if channels * image_size[0] * image_size[1] > 1_048_576:
        raise ImageDatasetError("Preprocessed image exceeds 1M-element CPU budget")
    if (mean is None) != (std is None):
        raise ImageDatasetError("Provide both normalization mean and std, or neither")
    if mean is not None:
        if (
            not isinstance(mean, (list, tuple))
            or not isinstance(std, (list, tuple))
            or len(mean) != channels
            or len(std) != channels
            or any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in [*mean, *std]
            )
            or any(v <= 0 for v in std)
        ):
            raise ImageDatasetError(
                "Normalization requires finite per-channel mean and positive std"
            )
    if not root.is_dir() or not (root / "train").is_dir():
        raise ImageDatasetError(
            "Expected an existing dataset root with train/class directories"
        )
    unknown = {p.name for p in root.iterdir() if p.is_dir()} - {"train", "val", "test"}
    if unknown:
        raise ImageDatasetError(f"Unsupported split directories: {sorted(unknown)}")
    operations = [
        transforms.Resize(
            tuple(image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float32),
    ]
    if mean is not None:
        operations.append(transforms.Normalize(mean, std))
    transform = transforms.Compose(operations)
    preprocessing = dict(
        channels=channels,
        image_size=list(image_size),
        resize="bilinear",
        antialias=True,
        tensor="ToTensor_CHW_float32_0_1_before_normalization",
        mean=None if mean is None else list(mean),
        std=None if std is None else list(std),
        pillow_version=pillow_version,
        torchvision_version=torchvision_version,
    )
    classes = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
    if len(classes) < 2:
        raise ImageDatasetError(
            "Multiclass training requires at least two nonempty classes"
        )
    class_to_idx = {name: index for index, name in enumerate(classes)}
    expected_hashes, seen, records, counts, splits = {}, {}, [], {}, {}
    for split in ("train", "val", "test"):
        directory = root / split
        if not directory.exists():
            splits[split] = None
            continue
        if not directory.is_dir():
            raise ImageDatasetError(f"Split path must be a directory: {directory}")
        if any(p.is_file() for p in directory.iterdir()):
            raise ImageDatasetError(
                f"Images must be inside class directories: {directory}"
            )
        actual = sorted(p.name for p in directory.iterdir() if p.is_dir())
        if actual != classes:
            raise ImageDatasetError(
                f"Class mapping mismatch in {split}: expected {classes}, got {actual}"
            )
        paths = []
        for name in classes:
            files = sorted(p for p in (directory / name).rglob("*") if p.is_file())
            if not files:
                raise ImageDatasetError(f"Empty class {name} in {split}")
            paths.extend((name, path) for path in files)
        if len(paths) > max_samples_per_split:
            raise ImageDatasetError(
                f"{split} exceeds explicit sample budget {max_samples_per_split}"
            )
        for name, path in paths:
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ImageDatasetError(
                    f"Linked/outside dataset image is unsupported: {path}"
                )
            if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                raise ImageDatasetError(f"Unsupported image extension: {path}")
            image, raw_hash = _read_image(path, channels=channels)
            try:
                tensor = transform(image)
            except (ValueError, RuntimeError) as error:
                raise ImageDatasetError(
                    f"Preprocessing failed for {path}: {error}"
                ) from error
            if not torch.isfinite(tensor).all():
                raise ImageDatasetError(
                    f"Preprocessing produced nonfinite values: {path}"
                )
            pixel_hash = _digest(
                (image.mode + str(image.size)).encode() + image.tobytes()
            )
            tensor_hash = _digest(tensor.contiguous().numpy().tobytes())
            for identity in (
                "raw:" + raw_hash,
                "pixels:" + pixel_hash,
                "tensor:" + tensor_hash,
            ):
                prior = seen.get(identity)
                if prior is not None and prior[0] != split:
                    raise ImageDatasetError(
                        f"Split leakage: {path} duplicates {prior[1]}"
                    )
                seen[identity] = (split, str(path))
            expected_hashes[str(path)] = raw_hash
            records.append(
                dict(
                    path=path.relative_to(root).as_posix(),
                    split=split,
                    class_index=class_to_idx[name],
                    sha256=raw_hash,
                    decoded_sha256=pixel_hash,
                    tensor_sha256=tensor_hash,
                )
            )
        dataset = datasets.ImageFolder(
            str(directory),
            transform=transform,
            loader=partial(
                _checked_loader, channels=channels, expected=expected_hashes
            ),
        )
        if dataset.class_to_idx != class_to_idx:
            raise ImageDatasetError(f"ImageFolder class mapping changed in {split}")
        counts[split] = len(dataset)
        splits[split] = dataset
    identity = dict(
        classes=classes,
        class_to_idx=class_to_idx,
        preprocessing=preprocessing,
        samples=records,
    )
    manifest = dict(
        format="hyso.image_classification_files",
        schema_version=1,
        dataset_id=_digest(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ),
        split_counts=counts,
        **identity,
    )
    return ImageClassificationSplits(
        splits["train"],
        splits["val"],
        splits["test"],
        json.dumps(manifest, sort_keys=True),
    )
