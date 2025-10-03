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

__all__ = ["EBO", "ELO"]

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
            scheduler_name=None, scheduler_params=None, max_lr=None):
        """
        optimizer_name: 'adamw' | 'adam' | 'sgd'
        scheduler_name: 'onecycle' | 'steplr' | 'reduceonplateau' | 'cosine'
        max_lr: required for onecycle
        accum_steps: gradient accumulation steps
        use_amp: mixed precision if cuda available
        """
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
                steps_per_epoch = len(train_loader)
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
        amp_available = use_amp and torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler(enabled=amp_available)

        # model summary
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

        for epoch in range(epochs):
            # TRAIN
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Train", leave=False)
            self.optimizer.zero_grad()
            for step, (inputs, targets) in enumerate(loop):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                device_type = "cuda" if torch.cuda.is_available() else "cpu"
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

                loss = loss / accum_steps
                scaler.scale(loss).backward()

                if (step + 1) % accum_steps == 0:
                    if grad_clip is not None:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                    # per-step scheduler for OneCycle
                    if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
                        try:
                            scheduler.step()
                        except Exception:
                            pass

                # EMA update
                if use_ema:
                    try:
                        self.ema.update(self.model)
                    except Exception:
                        pass

                batch_size = inputs.size(0)
                running_loss += loss.item() * accum_steps * batch_size
                total += batch_size
                correct += batch_correct
                loop.set_postfix(loss=running_loss / total if total>0 else 0.0,
                                 acc=100.*correct/total if total>0 else 0.0)

            train_loss = running_loss / total if total>0 else 0.0
            train_acc = 100.*correct/total if total>0 else 0.0
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # VALIDATION
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

                # scheduler step for ReduceLROnPlateau
                if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    try:
                        scheduler.step(val_loss_final)
                    except Exception:
                        pass
                elif scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
                    try:
                        scheduler.step()
                    except Exception:
                        pass

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

                # CALLBACKS (robust)
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
                    elif isinstance(cb, LRSchedulerCallback):
                        try:
                            cb.step(val_loss_final)
                        except Exception:
                            try:
                                cb.step()
                            except Exception:
                                pass

        return history
