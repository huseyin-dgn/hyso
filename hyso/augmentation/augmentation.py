# augmentation.py
from hyso.models.fit import CEMO
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import os


# ----------------------
# Transform builder
# ----------------------
def build_transforms(img_size=(224,224), augment="basic", extra_ops=None, channels=3):
    """
    img_size: tuple (H, W)
    augment: 'basic' veya 'advanced'
    extra_ops: Liste şeklinde ekstra augment işlemleri
    channels: 1 veya 3 (default 3)
    """
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    H, W = img_size

    ops = []

    # Grayscale işlemi
    if channels == 1:
        ops.append(transforms.Grayscale(num_output_channels=1))

    # --------------------
    # Basic augment
    # --------------------
    basic = [
        transforms.Resize((H, W)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ]

    # --------------------
    # Advanced augment
    # --------------------
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


# ----------------------
# Dataset loader
# ----------------------
def get_loaders(train_path, val_path=None, batch_size=32, img_size=(224,224), augment="basic", extra_ops=None, num_workers=0, channels=3):
    """
    train_path: training klasörü (class subfolders olmalı)
    val_path: validation/test klasörü (opsiyonel, yoksa train içerisinden split yapılır)
    batch_size: batch size
    img_size: tuple (H,W) veya int
    augment: 'basic' veya 'advanced'
    extra_ops: list, kullanıcı ek opsiyonları
    num_workers: DataLoader için
    channels: 1 veya 3
    """  

    # ---- transform ----
    train_transform = build_transforms(img_size=img_size, augment=augment, extra_ops=extra_ops, channels=channels)
    val_transform = build_transforms(img_size=img_size, augment="basic", channels=channels)  # artık channels geçiliyor

    # ---- dataset ----
    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    if val_path:
        val_dataset = datasets.ImageFolder(val_path, transform=val_transform)
    else:
        # split train dataset
        val_size = int(0.2 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # ---- loader ----
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
