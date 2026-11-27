import typing as _typing
# augmentation.py
from hyso.models.fit import CEMO
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import os



# Transform builder

def build_transforms(img_size: int | tuple[int, int]=(224,224), augment: _typing.Literal['basic', 'advanced']="basic", extra_ops: _typing.Iterable[_typing.Callable] | None=None, channels: int=3) -> 'transforms.Compose':
    """Compose the legacy image-training transforms without loading any images.

    Args:
        img_size: Positive resize H/W, or one integer for a square. Default: (224,224).
        augment: basic applies resize, horizontal flip, rotation, color jitter and
            tensor conversion. advanced also adds vertical flip, random crop,
            blur, perspective and erasing. None is rejected. Default: 'basic'.
        extra_ops: Optional iterable of transforms applied to the tensor after
            augmentation and before normalization. Default: None.
        channels: Supported image profile, 1 (grayscale) or 3 (RGB). Default: 3.

    Returns:
        torchvision.transforms.Compose; applying it performs random augmentation.

    Raises:
        ValueError: augment is not basic/advanced or native transform settings fail.

    Notes:
        RGB uses ImageNet mean/std; grayscale uses mean=std=0.5. Callers control
        RNG seeding. This helper does not download data or change model weights.
    """
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    H, W = img_size

    ops = []

    # Grayscale işlemi
    if channels == 1:
        ops.append(transforms.Grayscale(num_output_channels=1))


    # Basic augment

    basic = [
        transforms.Resize((H, W)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ]


    # Advanced augment

    kernel_size = max(3, int(round(max(H,W)/64)) | 1)
    advanced = [
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomResizedCrop((H, W), scale=(0.8,1.0)),
        transforms.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(0.1,2.0)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomErasing(p=0.3)
    ]

    # Augment seçimi
    if augment == "basic":
        ops += basic
    elif augment == "advanced":
        ops += basic + advanced
    else:
        raise ValueError("augment sadece 'basic' veya 'advanced' olabilir.")

    # Kullanıcı ekstra ops eklemek isterse
    if extra_ops:
        ops.extend(extra_ops)

    # Normalize
    if channels == 3:
        ops.append(transforms.Normalize(mean=[0.485,0.456,0.406],
                                        std=[0.229,0.224,0.225]))
    else:
        ops.append(transforms.Normalize(mean=[0.5], std=[0.5]))

    return transforms.Compose(ops)



# Dataset loader

def _validation_transform(img_size, channels):
    # Validation deliberately excludes training augmentation and extra_ops.
    ops = [transforms.Grayscale(num_output_channels=1)] if channels == 1 else []
    size = (img_size, img_size) if isinstance(img_size, int) else img_size
    ops.extend([transforms.Resize(size), transforms.ToTensor()])
    if channels == 3:
        ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]))
    else:
        ops.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    return transforms.Compose(ops)


def get_loaders(train_path: str | os.PathLike[str], val_path: str | os.PathLike[str] | None=None, batch_size: int=32, img_size: int | tuple[int, int]=(224,224), augment: _typing.Literal['basic', 'advanced']="basic", extra_ops: _typing.Iterable[_typing.Callable] | None=None, num_workers: int=0, channels: int=3) -> 'tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]':
    """Create ImageFolder train/validation loaders with a shared class vocabulary.

    Args:
        train_path: Existing image directory with one subdirectory per class. Required.
        val_path: Optional separate validation directory. None splits training images
            using a random 20% validation subset, with at least one sample. Default: None.
        batch_size: Positive native DataLoader batch size. Default: 32.
        img_size: Resize H/W or one integer for a square. Default: (224,224).
        augment: basic or advanced training transforms; validation has only resize,
            tensor conversion and normalization. None is rejected. Default: 'basic'.
        extra_ops: Tensor transforms added to training only, before normalization.
            Validation never applies them. Default: None.
        num_workers: Native loader worker count; zero runs in process. Default: 0.
        channels: Supported image profile, 1 (grayscale) or 3 (RGB). Default: 3.

    Returns:
        (train_loader, val_loader). Training shuffles; validation does not.

    Raises:
        ValueError: Validation contains unknown classes, an automatic split has
            fewer than two source samples, or native loader/transform settings fail.
        OSError: An image directory cannot be read.

    Notes:
        Separate validation labels are remapped to the training vocabulary. Split
        indices use the current torch RNG; callers must set their seed explicitly.
        No data download or training is performed.
    """

    # ---- transform ----
    train_transform = build_transforms(img_size=img_size, augment=augment, extra_ops=extra_ops, channels=channels)
    val_transform = _validation_transform(img_size, channels)

    # ---- dataset ----
    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    if val_path:
        val_dataset = datasets.ImageFolder(val_path, transform=val_transform)
        unknown = set(val_dataset.classes) - set(train_dataset.classes)
        if unknown:
            raise ValueError(f"Unknown validation classes: {sorted(unknown)}; "
                             f"training classes: {train_dataset.classes}")
        # ImageFolder assigns local indices. Rewrite both samples and metadata
        # into the training vocabulary, including known validation subsets.
        remap = {index: train_dataset.class_to_idx[name]
                 for name, index in val_dataset.class_to_idx.items()}
        val_dataset.samples = [(path, remap[target]) for path, target in val_dataset.samples]
        val_dataset.imgs = val_dataset.samples
        val_dataset.targets = [target for _, target in val_dataset.samples]
        val_dataset.classes = list(train_dataset.classes)
        val_dataset.class_to_idx = dict(train_dataset.class_to_idx)
    else:
        # split train dataset
        if len(train_dataset) < 2:
            raise ValueError("Validation split requires at least two samples for nonempty train and validation sets")
        val_size = max(1, int(0.2 * len(train_dataset)))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        validation_source = datasets.ImageFolder(train_path, transform=val_transform)
        val_dataset = torch.utils.data.Subset(validation_source, val_dataset.indices)

    # ---- loader ----
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
