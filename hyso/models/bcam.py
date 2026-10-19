# bcam.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from ..callbacks.callbacks import EarlyStopping, ModelCheckpoint, LRSchedulerCallback
from .WithSE import EMA  # SE-block ve EMA
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torchsummary import summary
import numpy as np
import hashlib
import json
import os
import operator
from pathlib import Path

from ..checkpoint import save_training_checkpoint, load_training_checkpoint, TrainingCheckpointError
from ..checkpoint import _scheduler_state

__all__ = ["EBO", "ELO"]

def _resume_type(obj):
    return type(obj).__module__ + "." + type(obj).__qualname__


def _resume_transform_plan(transform):
    """Only stateless torchvision v1 operations using the restored global RNG."""
    from torchvision import transforms as T
    from enum import Enum

    if transform is None:
        return None
    fields = {
        T.Compose: ("transforms",), T.ToTensor: (),
        T.Resize: ("size", "interpolation", "max_size", "antialias"),
        T.CenterCrop: ("size",), T.Normalize: ("mean", "std", "inplace"),
        T.Grayscale: ("num_output_channels",),
        T.RandomHorizontalFlip: ("p",), T.RandomVerticalFlip: ("p",),
        T.RandomRotation: ("degrees", "interpolation", "expand", "center", "fill"),
        T.ColorJitter: ("brightness", "contrast", "saturation", "hue"),
        T.RandomResizedCrop: ("size", "scale", "ratio", "interpolation", "antialias"),
    }
    if type(transform) not in fields:
        raise TrainingCheckpointError("unsupported custom/stateful transform: " + _resume_type(transform))
    names = fields[type(transform)]
    # Reject added state, instance overrides and registered hooks, even on allowed types.
    base = vars(nn.Module()) if isinstance(transform, nn.Module) else {}
    if (set(vars(transform)) != set(base) | set(names)
            or any(vars(transform)[key] != value for key, value in base.items())):
        raise TrainingCheckpointError("checkpoint transforms must have standard stateless configuration")

    def plain(value):
        if isinstance(value, Enum):
            return value.value
        if type(value) in (list, tuple):
            return [plain(item) for item in value]
        if value is None or type(value) in (str, bool, int, float):
            return value
        raise TrainingCheckpointError("unsupported transform configuration value")

    if type(transform) is T.Compose and type(transform.transforms) not in (list, tuple):
        raise TrainingCheckpointError("Compose transforms must be a fixed list or tuple")
    config = ({"transforms": [_resume_transform_plan(item) for item in transform.transforms]}
              if type(transform) is T.Compose else {name: plain(getattr(transform, name)) for name in names})
    return {"type": _resume_type(transform), "config": config}


def _resume_dataset_plan(dataset):
    from torch.utils.data import TensorDataset, Subset
    from torchvision.datasets import ImageFolder
    from torchvision.datasets.folder import default_loader, pil_loader, IMG_EXTENSIONS
    from torchvision import get_image_backend, __version__ as vision_version
    from PIL import __version__ as pillow_version

    if type(dataset) is TensorDataset:
        tensors = []
        for tensor in dataset.tensors:
            if tensor.device.type != "cpu" or tensor.layout != torch.strided or tensor.requires_grad:
                raise TrainingCheckpointError("checkpoint dataset tensors must be dense CPU tensors without gradients")
            raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
            tensors.append({"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                            "sha256": hashlib.sha256(memoryview(raw)).hexdigest()})
        return {"tensors": tensors}  # Preserve the original TensorDataset plan exactly.
    if type(dataset) is Subset:
        try:
            indices = [operator.index(index) for index in dataset.indices]
        except (TypeError, ValueError) as error:
            raise TrainingCheckpointError("Subset indices must be integers") from error
        if any(index < 0 or index >= len(dataset.dataset) for index in indices):
            raise TrainingCheckpointError("Subset indices must be within the underlying dataset")
        return {"dataset": {"type": _resume_type(dataset), "indices": indices,
                            "base": _resume_dataset_plan(dataset.dataset)}}
    if type(dataset) is not ImageFolder:
        raise TrainingCheckpointError("checkpoint dataset must be TensorDataset, ImageFolder or Subset")
    if (dataset.loader not in (default_loader, pil_loader) or get_image_backend() != "PIL"
            or dataset.target_transform is not None or dataset.extensions != IMG_EXTENSIONS):
        raise TrainingCheckpointError("ImageFolder checkpointing requires standard PIL loading and no target_transform")
    classes = dataset.classes
    mapping = dataset.class_to_idx
    if (type(classes) is not list or any(type(name) is not str for name in classes)
            or len(set(classes)) != len(classes)
            or mapping != {name: index for index, name in enumerate(classes)}
            or any(type(index) is not int for index in mapping.values())):
        raise TrainingCheckpointError("ImageFolder classes/class_to_idx are inconsistent")
    root = Path(dataset.root).resolve()
    if not root.is_dir():
        raise TrainingCheckpointError("ImageFolder dataset root is missing or moved: " + str(root))
    try:
        directories = [path for path in root.iterdir() if path.is_dir()]
        if any(path.is_symlink() for path in directories):
            raise TrainingCheckpointError("ImageFolder symlink directories are unsupported")
        if any(path.name not in mapping for path in directories):
            raise TrainingCheckpointError("ImageFolder class directory list changed")
        # Scan names as well as saved samples: additions to an already constructed dataset must fail.
        discovered = set()
        for directory in directories:
            for current, subdirs, files in os.walk(str(directory)):
                if any(Path(current, name).is_symlink() for name in subdirs):
                    raise TrainingCheckpointError("ImageFolder symlink directories are unsupported")
                for name in files:
                    if name.lower().endswith(IMG_EXTENSIONS):
                        discovered.add(str(Path(current, name).absolute()))
        manifest = []
        for path, label in dataset.samples:
            source = Path(path).absolute()
            relative = source.relative_to(root)
            if source.is_symlink() or label != mapping.get(relative.parts[0]) or type(label) is not int:
                raise TrainingCheckpointError("ImageFolder sample path/label mapping is inconsistent")
            stat = source.stat()
            manifest.append([relative.as_posix(), label, stat.st_size, stat.st_mtime_ns])
        if (discovered != {str(Path(path).absolute()) for path, _ in dataset.samples}
                or dataset.targets != [label for _, label in dataset.samples]
                or dataset.imgs != dataset.samples):
            raise TrainingCheckpointError("ImageFolder image list/targets changed; reconstruct the dataset")
    except (OSError, ValueError) as error:
        raise TrainingCheckpointError("ImageFolder dataset path is missing, moved or inaccessible: " + str(root)) from error
    # Hash plain metadata, never decoded images. This is not a content-integrity hash.
    digest = hashlib.sha256(json.dumps(manifest, ensure_ascii=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"dataset": {"type": _resume_type(dataset), "root": root.as_posix(),
                        "classes": classes, "class_to_idx": mapping, "samples": len(manifest),
                        "manifest_sha256": digest, "transform": _resume_transform_plan(dataset.transform),
                        "torchvision": vision_version, "pillow": pillow_version,
                        "torch": str(torch.__version__), "default_dtype": str(torch.get_default_dtype()),
                        "loader": "pil_loader" if dataset.loader is pil_loader else "default_loader"}}


def _resume_loader_plan(loader):
    """Bounded epoch replay: no worker, sampler or dataset state is omitted."""
    from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler, default_collate
    if (type(loader) is not DataLoader
            or loader.num_workers != 0 or loader.generator is not None
            or loader.batch_size is None or loader.collate_fn is not default_collate or loader.timeout != 0
            or type(loader.batch_sampler) is not BatchSampler
            or type(loader.sampler) not in (RandomSampler, SequentialSampler)
            or getattr(loader.sampler, "generator", None) is not None
            or getattr(loader.sampler, "replacement", False)
            or len(loader) == 0):
        raise TrainingCheckpointError(
            "epoch checkpointing requires nonempty CPU DataLoader(TensorDataset/ImageFolder/Subset), num_workers=0, "
            "standard batch/samplers and no explicit generators")
    plan = {"batches": len(loader), "batch_size": loader.batch_size, "drop_last": loader.drop_last,
            "sampler": _resume_type(loader.sampler), "sampler_samples": len(loader.sampler)}
    plan.update(_resume_dataset_plan(loader.dataset))
    return plan


def _resume_plan(model, optimizer, scheduler, train_loader, val_loader,
                 epochs, accum_steps, use_ema, use_amp, amp_enabled, grad_clip):
    if type(model) is not EBO or type(model.model) is not ELO:
        raise TrainingCheckpointError("epoch checkpointing requires the standard EBO/ELO model")
    if type(optimizer) not in (optim.Adam, optim.AdamW, optim.SGD):
        raise TrainingCheckpointError("checkpointing supports standard Adam, AdamW or SGD optimizers")
    if scheduler is None:
        scheduler_config = None
    elif type(scheduler) is optim.lr_scheduler.OneCycleLR:
        dynamic = {"last_epoch", "_step_count", "_get_lr_called_within_step", "_last_lr"}
        scheduler_config = {key: value for key, value in _scheduler_state(scheduler).items() if key not in dynamic}
    else:
        fields = {
            optim.lr_scheduler.StepLR: ("step_size", "gamma"),
            optim.lr_scheduler.CosineAnnealingLR: ("T_max", "eta_min"),
            optim.lr_scheduler.ReduceLROnPlateau:
                ("mode", "factor", "patience", "threshold", "threshold_mode", "cooldown", "min_lrs", "eps"),
        }
        if type(scheduler) not in fields:
            raise TrainingCheckpointError("custom schedulers are not supported for epoch checkpointing")
        scheduler_config = {key: getattr(scheduler, key) for key in fields[type(scheduler)]}
    if scheduler is not None and scheduler.last_epoch != 0:
        raise TrainingCheckpointError("checkpoint setup requires a fresh scheduler; resume restores its saved counter")
    groups = [{key: value for key, value in group.items() if key != "params"}
              for group in optimizer.state_dict()["param_groups"]]
    plan = {"version": 1, "epochs": epochs, "accum_steps": accum_steps, "grad_clip": grad_clip,
            "use_ema": bool(use_ema), "use_amp": bool(use_amp), "amp_enabled": bool(amp_enabled),
            "device_type": next(model.parameters()).device.type, "num_classes": model.num_classes,
            "model": repr(model.model), "optimizer_type": _resume_type(optimizer), "optimizer_groups": groups,
            "scheduler_type": _resume_type(scheduler) if scheduler is not None else None,
            "scheduler_config": scheduler_config, "train_loader": _resume_loader_plan(train_loader),
            "val_loader": _resume_loader_plan(val_loader) if val_loader is not None else None}
    try:
        # Stable plain metadata: tuple-valued optimizer options become lists.
        return json.loads(json.dumps(plan, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise TrainingCheckpointError("checkpoint training plan requires plain finite configuration values") from error


def _validate_resume_progress(progress, plan):
    if progress["metrics"] != {"ebo_fit": plan}:
        raise TrainingCheckpointError("EBO training plan mismatch: keep total epochs, accumulation, "
                                      "optimizer/scheduler, AMP/EMA and dataset/loader configuration unchanged")
    completed = progress["completed_epochs"]
    if not 0 <= completed <= plan["epochs"]:
        raise TrainingCheckpointError("completed_epochs exceeds the total epoch budget")
    if progress["scheduler"] is not None:
        state = progress["scheduler"]["state_dict"]
        config = {key: state[key] for key in plan["scheduler_config"]}
        if json.loads(json.dumps(config, allow_nan=False)) != plan["scheduler_config"]:
            raise TrainingCheckpointError("saved scheduler configuration/budget disagrees with the training plan")
        if plan["scheduler_type"] == "torch.optim.lr_scheduler.OneCycleLR":
            steps = (plan["train_loader"]["batches"] + plan["accum_steps"] - 1) // plan["accum_steps"]
            if state["total_steps"] != steps * plan["epochs"]:
                raise TrainingCheckpointError("OneCycle total_steps disagrees with the original epoch budget")
            attempted = completed * steps
            valid_count = 0 <= state["last_epoch"] <= attempted if plan["amp_enabled"] else state["last_epoch"] == attempted
        else:
            expected = (0 if plan["scheduler_type"] == "torch.optim.lr_scheduler.ReduceLROnPlateau"
                        and plan["val_loader"] is None else completed)
            valid_count = state["last_epoch"] == expected
        if not valid_count:
            raise TrainingCheckpointError("saved scheduler counter disagrees with completed_epochs")
    history = progress["history"]
    expected = {"train_loss", "train_acc", "val_loss", "val_acc"}
    if type(history) is not dict or set(history) != expected:
        raise TrainingCheckpointError("resume history must contain the four EBO metric lists")
    for key in expected:
        count = completed if key.startswith("train_") or plan["val_loader"] is not None else 0
        if type(history[key]) is not list or len(history[key]) != count:
            raise TrainingCheckpointError("resume history length disagrees with completed_epochs: " + key)
        if any(type(value) not in (int, float) for value in history[key]):
            raise TrainingCheckpointError("resume history values must be numeric: " + key)

# -------------------------
# CBAM Block
# -------------------------
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(channels, max(1, channels // reduction))
        self.fc2 = nn.Linear(max(1, channels // reduction), channels)
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        mx = self.max_pool(x).view(b, c)

        avg_out = self.fc2(F.silu(self.fc1(avg)))
        max_out = self.fc2(F.silu(self.fc1(mx)))
        scale = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * scale

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_scale = torch.sigmoid(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_scale


# -------------------------
# Bottleneck Residual Block + CBAM (depthwise optional)
# -------------------------
def _find_group_count(channels, preferred=8):
    g = min(preferred, channels)
    while g > 1:
        if channels % g == 0:
            return g
        g -= 1
    return 1

def stochastic_depth(x, drop_prob: float, training: bool):
    if not training or drop_prob <= 0.0:
        return x
    keep_prob = 1.0 - drop_prob
    shape = [x.shape[0]] + [1] * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    return x.div(keep_prob) * torch.floor(random_tensor)

class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 use_projection=False, bottleneck_ratio=0.25,
                 p_drop=0.0, att_reduction=16, norm_type='auto',
                 use_depthwise=False):
        """
        use_depthwise: if True, use depthwise separable conv for the 3x3 conv path (faster, less params)
        """
        super().__init__()
        self.p_drop = p_drop
        self.use_depthwise = use_depthwise
        mid_channels = max(1, int(out_channels * bottleneck_ratio))

        def make_norm(ch):
            if norm_type == 'batch':
                return nn.BatchNorm2d(ch)
            else:  # group veya auto
                g = _find_group_count(ch)
                return nn.GroupNorm(g, ch)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = make_norm(mid_channels)

        if use_depthwise:
            # depthwise separable: depthwise 3x3 then pointwise 1x1
            self.dw_seq = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, groups=mid_channels, bias=False),
                make_norm(mid_channels),
                nn.SiLU(),
                nn.Conv2d(mid_channels, mid_channels, 1, bias=False),
                make_norm(mid_channels)
            )
            # keep placeholders for compatibility
            self.conv2 = nn.Identity()
            self.bn2 = nn.Identity()
        else:
            self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, bias=False)
            self.bn2 = make_norm(mid_channels)
            self.dw_seq = None

        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = make_norm(out_channels)

        if use_projection or in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                make_norm(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.attention = CBAMBlock(out_channels, reduction=att_reduction)

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.silu(self.bn1(self.conv1(x)))
        if self.use_depthwise and (self.dw_seq is not None):
            out = self.dw_seq(out)
        else:
            out = F.silu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = self.attention(out)
        out = stochastic_depth(out, self.p_drop, self.training)
        out = out + identity
        return F.silu(out)


# -------------------------
# CNN + Residual + CBAM
# -------------------------
class ELO(nn.Module):
    def __init__(self, input_channels=3, num_classes=10,
                 conv_channels=[64,128,256,512], residual_per_block=2,
                 p_drop=0.0, att_reduction=16, norm_type='auto',
                 use_depthwise=False):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, conv_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(conv_channels[0]),
            nn.SiLU()
        )

        self.blocks = nn.ModuleList()
        in_ch = conv_channels[0]
        for stage_idx, out_ch in enumerate(conv_channels):
            for i in range(residual_per_block):
                stride = 2 if (i == 0 and stage_idx > 0) else 1
                use_proj = (in_ch != out_ch) or (stride != 1)
                self.blocks.append(
                    BottleneckResidualBlock(in_ch, out_ch, stride=stride, use_projection=use_proj,
                                            p_drop=p_drop, att_reduction=att_reduction, norm_type=norm_type,
                                            use_depthwise=use_depthwise)
                )
                in_ch = out_ch

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_norm = nn.LayerNorm(conv_channels[-1])
        self.head = nn.Sequential(
            nn.Linear(conv_channels[-1], conv_channels[-1] // 2),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(conv_channels[-1] // 2, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc_norm(x)
        return self.head(x)


# -------------------------
# BCAMWithFit + Callbacks
# -------------------------
class EBO(nn.Module):
    def __init__(self, input_channels=3, num_classes=10,
                 conv_channels=[32,64,128,256], residual_per_block=2,
                 p_drop=0.0, att_reduction=16, norm_type='auto', device=None,
                 use_depthwise=False):
        super().__init__()
        self.model = ELO(
            input_channels=input_channels,
            num_classes=num_classes,
            conv_channels=conv_channels,
            residual_per_block=residual_per_block,
            p_drop=p_drop,
            att_reduction=att_reduction,
            norm_type=norm_type,
            use_depthwise=use_depthwise
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # ensure model placed on device early to avoid device mismatch
        self.model.to(self.device)
        self.num_classes = num_classes
        self._callbacks = []
        self.optimizer = None

    def forward(self, x):
        return self.model(x)

    # -------------------------
    # Callbacks & Optimizer
    # -------------------------
    def callbacks(self, optimizer="Adam", optimizer_params=None,
                  scheduler=None, scheduler_params=None,
                  model_checkpoint=False, earlystop=None,
                  lr=1e-3, weight_decay=1e-4):
        # ensure model on device
        try:
            self.model.to(self.device)
        except Exception:
            pass

        # --- Optimizer ---
        opt_params = optimizer_params or {"lr": lr, "weight_decay": weight_decay}
        if optimizer.lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), **opt_params)
        elif optimizer.lower() == "sgd":
            if "momentum" not in opt_params:
                opt_params["momentum"] = 0.9
            self.optimizer = optim.SGD(self.model.parameters(), **opt_params)
        else:
            # default to AdamW if unknown for better generalization
            if optimizer.lower() == "adamw":
                self.optimizer = optim.AdamW(self.model.parameters(), **opt_params)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer}")

        # --- Scheduler ---
        if scheduler:
            if scheduler.lower() == "steplr":
                sched = optim.lr_scheduler.StepLR(self.optimizer, **(scheduler_params or {"step_size":5,"gamma":0.5}))
            elif scheduler.lower() == "reduceonplateau":
                sched = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **(scheduler_params or {"patience":3,"factor":0.5}))
            elif scheduler.lower() == "cosineannealinglr":
                sched = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **(scheduler_params or {"T_max":10}))
            elif scheduler.lower() == "onecycle":
                # OneCycle will be set up in fit because it needs steps_per_epoch
                sched = None
            else:
                raise ValueError(f"Unsupported scheduler: {scheduler}")
            if sched is not None:
                self.lr_scheduler = LRSchedulerCallback(sched)
                self._callbacks.append(self.lr_scheduler)

        # --- ModelCheckpoint ---
        if model_checkpoint:
            self.model_checkpoint = ModelCheckpoint(filepath="best_bcam_model.pth", verbose=True)
            self._callbacks.append(self.model_checkpoint)

        # --- EarlyStopping ---
        if earlystop:
            self.early_stopping = EarlyStopping(
                patience=earlystop.get("patience",7),
                delta=earlystop.get("delta",0.0),
                verbose=earlystop.get("verbose",True),
                restore_best_weights=True
            )
            self._callbacks.append(self.early_stopping)

    # -------------------------
    # Improved Fit Fonksiyonu (AMP, AdamW, OneCycle, grad_clip, accum_steps)
    # -------------------------
    def fit(self, train_loader, val_loader=None, epochs=10, lr=1e-3,
            optimizer_name="adamw", weight_decay=1e-4, use_ema=False,
            use_amp=True, grad_clip=None, accum_steps=1,
            scheduler_name=None, scheduler_params=None, max_lr=None, *,
            resume_from=None, save_last_to=None):
        """
        optimizer_name: 'adamw' | 'adam' | 'sgd'
        scheduler_name: 'onecycle' | 'steplr' | 'reduceonplateau' | 'cosine'
        max_lr: required for onecycle
        accum_steps: positive Python integer; bool is not accepted
        use_amp: mixed precision when training on CUDA
        resume_from/save_last_to: opt-in full-state epoch checkpoints.
        epochs remains the total original budget, including completed epochs.
        Checkpointing accepts TensorDataset, PIL ImageFolder and their Subsets with
        standard loaders/samplers, num_workers=0 and bounded torchvision v1 transforms.
        Explicit generators and custom callbacks are unsupported; the original plan
        must match. Image identity uses paths/size/mtime, not image-content hashing.
        """
        if isinstance(accum_steps, bool) or not isinstance(accum_steps, int):
            raise TypeError("accum_steps must be a positive integer; bool is not accepted")
        if accum_steps <= 0:
            raise ValueError("accum_steps must be a positive integer")

        checkpointing = resume_from is not None or save_last_to is not None
        if checkpointing:
            if type(epochs) is not int or epochs <= 0:
                raise TrainingCheckpointError("checkpoint epochs must be a positive total epoch budget")
            if any(type(callback) is not LRSchedulerCallback for callback in self._callbacks):
                raise TrainingCheckpointError("stateful/custom callbacks are not supported for epoch checkpointing")

        # device sync
        self.model.to(self.device)

        # create optimizer if not set
        if self.optimizer is None:
            if optimizer_name.lower() == "adamw":
                self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            elif optimizer_name.lower() == "adam":
                self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            elif optimizer_name.lower() == "sgd":
                self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # scheduler setup (OneCycle needs steps_per_epoch, so prepare later if requested)
        scheduler = None
        if scheduler_name:
            sname = scheduler_name.lower()
            sp = scheduler_params or {}
            if sname == "onecycle":
                if max_lr is None:
                    raise ValueError("max_lr must be provided for OneCycleLR")
                steps_per_epoch = (len(train_loader) + accum_steps - 1) // accum_steps
                scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=epochs, **sp)
            elif sname == "steplr":
                scheduler = optim.lr_scheduler.StepLR(self.optimizer, **(sp or {"step_size":5,"gamma":0.5}))
            elif sname == "reduceonplateau":
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **(sp or {"patience":3,"factor":0.5}))
            elif sname == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **(sp or {"T_max":epochs}))
            else:
                raise ValueError(f"Unsupported scheduler: {scheduler_name}")

            self.lr_scheduler = LRSchedulerCallback(scheduler)
            if self.lr_scheduler not in self._callbacks:
                self._callbacks.append(self.lr_scheduler)

        # fit owns stepping for the latest scheduler configured by callbacks()
        # or this fit call. Scheduler entries are not stepped in the callback loop.
        scheduler_callback = getattr(self, "lr_scheduler", None)
        if checkpointing and scheduler_callback is not None and type(scheduler_callback) is not LRSchedulerCallback:
            raise TrainingCheckpointError("custom scheduler callbacks are not supported for epoch checkpointing")
        scheduler = scheduler_callback.scheduler if scheduler_callback is not None else None

        # criterion
        criterion = nn.BCEWithLogitsLoss() if self.num_classes == 1 else nn.CrossEntropyLoss()
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        # EMA setup
        if use_ema:
            if not hasattr(self, "ema"):
                self.ema = EMA(self.model, decay=0.999)
            try:
                self.ema.ema_model.to(self.device)
                self.ema.ema_model.eval()
            except Exception:
                pass
            print("✅ EMA is enabled for training.")

        # AMP setup
        amp_available = use_amp and next(self.model.parameters()).device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=amp_available)

        plan = (_resume_plan(self, self.optimizer, scheduler, train_loader, val_loader, epochs,
                             accum_steps, use_ema, use_amp, amp_available, grad_clip)
                if checkpointing else None)

        # A resumed run must not consume a sample/RNG or touch BN for presentation.
        if resume_from is None:
            sample = next(iter(train_loader))[0]
            input_size = tuple(sample.shape[1:])
            print("\n===== Model Summary =====")
            try:
                try:
                    summary(self.model, input_size)
                except Exception:
                    print(self.model)
            except:
                print("Could not print detailed summary, using fallback")
            print("=========================\n")

        # Restore RNG last, after component construction and before the next iterator.
        start_epoch = 0
        if resume_from is not None:
            progress = load_training_checkpoint(
                resume_from, self, optimizer=self.optimizer, scheduler=scheduler, scaler=scaler,
                ema=self.ema if use_ema else None, map_location="cpu", restore_rng=True,
                validate_metadata=lambda metadata: _validate_resume_progress(metadata, plan))
            start_epoch = progress["completed_epochs"]
            history = progress["history"]
            self.model.train(val_loader is None)

        for epoch in range(start_epoch, epochs):
            # TRAIN
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Train", leave=False)
            self.optimizer.zero_grad()
            window_samples = 0
            epoch_batches = len(train_loader)
            for step, (inputs, targets) in enumerate(loop):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                device_type = inputs.device.type
                with torch.amp.autocast(device_type=device_type, enabled=amp_available):
                    outputs = self.model(inputs)
                    if self.num_classes == 1:
                        outputs = outputs.view(-1)
                        targets_proc = targets.view(-1)
                        loss = criterion(outputs, targets_proc.float())
                        preds = (torch.sigmoid(outputs) > 0.5).long()
                        batch_correct = (preds == targets_proc.long()).sum().item()
                    else:
                        if targets.ndim > 1:
                            targets_proc = targets.view(-1)
                        else:
                            targets_proc = targets
                        loss = criterion(outputs, targets_proc)
                        preds = outputs.argmax(dim=1)
                        batch_correct = (preds == targets_proc).sum().item()

                # Sum sample losses, then normalize once over the actual window.
                # This also handles unequal microbatches and a partial tail.
                batch_size = inputs.size(0)
                window_samples += batch_size
                scaler.scale(loss * batch_size).backward()

                if (step + 1) % accum_steps == 0 or step + 1 == epoch_batches:
                    scaler.unscale_(self.optimizer)
                    for parameter in self.model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(window_samples)
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    optimizer_updates = []
                    if isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
                        # GradScaler.step() may return None for both a real step
                        # and a skipped step. Observe the optimizer itself.
                        scale_before_step = scaler.get_scale()
                        handle = self.optimizer.register_step_post_hook(
                            lambda *args: optimizer_updates.append(True)
                        )
                        try:
                            scaler.step(self.optimizer)
                        finally:
                            handle.remove()
                    else:
                        scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                    window_samples = 0
                    # per-step scheduler for OneCycle
                    # Fused optimizers can delegate the overflow skip to their
                    # kernel while still invoking step hooks. Scale backoff
                    # distinguishes that case from an applied update.
                    if optimizer_updates and scaler.get_scale() >= scale_before_step:
                        scheduler_callback.step()

                # EMA update
                if use_ema:
                    try:
                        self.ema.update(self.model)
                    except Exception:
                        pass

                running_loss += loss.item() * batch_size
                total += batch_size
                correct += batch_correct
                loop.set_postfix(loss=running_loss / total if total>0 else 0.0,
                                 acc=100.*correct/total if total>0 else 0.0)

            train_loss = running_loss / total if total>0 else 0.0
            train_acc = 100.*correct/total if total>0 else 0.0
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # VALIDATION
            val_loss_final = None
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                all_preds = []
                all_targets = []
                with torch.no_grad():
                    for inputs, targets in tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Val", leave=False):
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        # use EMA model for inference if available
                        if use_ema and hasattr(self, "ema"):
                            try:
                                self.ema.ema_model.to(self.device)
                                self.ema.ema_model.eval()
                                out = self.ema.ema_model(inputs)
                            except Exception:
                                out = self.model(inputs)
                        else:
                            out = self.model(inputs)

                        if self.num_classes == 1:
                            out = out.view(-1)
                            targets_proc = targets.view(-1)
                            loss = criterion(out, targets_proc.float())
                            preds = (torch.sigmoid(out) > 0.5).long()
                            val_correct += (preds == targets_proc.long()).sum().item()
                            all_preds.extend(preds.cpu().numpy().ravel().tolist())
                            all_targets.extend(targets_proc.cpu().numpy().ravel().tolist())
                        else:
                            if targets.ndim > 1:
                                targets_proc = targets.view(-1)
                            else:
                                targets_proc = targets
                            loss = criterion(out, targets_proc)
                            preds = out.argmax(dim=1)
                            val_correct += (preds == targets_proc).sum().item()
                            all_preds.extend(preds.cpu().numpy().ravel().tolist())
                            all_targets.extend(targets_proc.cpu().numpy().ravel().tolist())

                        batch_size = inputs.size(0)
                        val_loss += loss.item() * batch_size
                        val_total += batch_size

                val_loss_final = val_loss / val_total if val_total > 0 else 0.0
                val_acc = 100.*val_correct/val_total if val_total>0 else 0.0
                history["val_loss"].append(val_loss_final)
                history["val_acc"].append(val_acc)

                # print metrics + sklearn report
                try:
                    all_preds_arr = np.array(all_preds).ravel().astype(int)
                    all_targets_arr = np.array(all_targets).ravel().astype(int)
                except Exception:
                    all_preds_arr = np.array(all_preds).ravel()
                    all_targets_arr = np.array(all_targets).ravel()

                print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss_final:.4f}, Acc: {val_acc:.2f}%")
                try:
                    print(classification_report(all_targets_arr, all_preds_arr))
                except Exception:
                    pass

                try:
                    cm = confusion_matrix(all_targets_arr, all_preds_arr)
                    plt.figure(figsize=(6,5))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                    plt.title(f"Confusion Matrix - Epoch {epoch+1}")
                    plt.xlabel("Predicted")
                    plt.ylabel("True")
                    plt.show()
                except Exception:
                    pass

            # Epoch schedulers have one owner, with or without validation.
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                if val_loss_final is not None:
                    if not np.isfinite(val_loss_final):
                        raise ValueError("ReduceLROnPlateau requires a finite validation loss")
                    scheduler_callback.step(val_loss_final)
            elif scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
                scheduler_callback.step()

            if val_loader:
                # Other callback handling is unchanged.
                for cb in self._callbacks:
                    if isinstance(cb, EarlyStopping):
                        try:
                            cb(val_loss_final, self.model)
                        except Exception:
                            try:
                                cb.step(val_loss_final)
                            except Exception:
                                pass
                        if getattr(cb, "early_stop", False):
                            print(f"⏹ Early stopping at epoch {epoch+1}")
                            if getattr(cb, "restore_best_weights", False):
                                try:
                                    cb.restore_weights(self.model)
                                except Exception:
                                    pass
                            return history
                    elif isinstance(cb, ModelCheckpoint):
                        try:
                            cb(epoch, self.model, val_loss_final)
                        except Exception:
                            try:
                                cb.on_epoch_end(epoch, self.model, val_loss_final)
                            except Exception:
                                pass

            if save_last_to is not None:
                save_training_checkpoint(
                    save_last_to, self, optimizer=self.optimizer, scheduler=scheduler, scaler=scaler,
                    ema=self.ema if use_ema else None, completed_epochs=epoch + 1,
                    history=history, metrics={"ebo_fit": plan}, include_rng=True)

        return history
