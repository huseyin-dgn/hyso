# -------------------------
# Temel PyTorch ve yardımcı kütüphaneler
# -------------------------
# Temel PyTorch ve yardımcı kütüphaneler
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from tqdm import tqdm

# Callbacks
from hyso.callbacks.callbacks import EarlyStopping, ModelCheckpoint, LRSchedulerCallback

# -------------------------
# EMA Wrapper
# -------------------------
__all__ = ["CNNSENormalWithFit", "EMA"]

class EMA:
    def __init__(self, model, decay=0.999, device=None):
        # model can be either the model instance or a nn.Module
        # copy model to create ema model and stop grads
        self.ema_model = copy.deepcopy(model)
        self.decay = decay
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        # keep track of device preference
        self.device = device

    def to(self, device):
        try:
            self.ema_model.to(device)
            self.device = device
        except Exception:
            pass

    def update(self, model):
        # update ema params from model (works whether model is wrapper or plain)
        with torch.no_grad():
            # align params by name/zip
            for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

# -------------------------
# SEBlock
# -------------------------
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced)
        self.fc2 = nn.Linear(reduced, channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()
        y = x.mean(dim=(2, 3))
        y = self.act(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(n, c, 1, 1)
        return x * y

# -------------------------
# Bottleneck + Depthwise + SE + StochasticDepth
# -------------------------
class BottleneckSEBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        bottleneck_ratio: float = 0.25,
        norm_layer=nn.BatchNorm2d,
        activation=nn.SiLU,
        drop_prob: float = 0.0,
        se_reduction: int = 8,
    ):
        super().__init__()
        mid_channels = max(1, int(out_channels * bottleneck_ratio))
        self.drop_prob = drop_prob
        self.use_projection = (in_channels != out_channels) or (stride != 1)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = norm_layer(mid_channels)
        self.act1 = activation()

        self.dwconv = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, groups=mid_channels, bias=False)
        self.bn2 = norm_layer(mid_channels)
        self.act2 = activation()

        self.conv_pw = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = norm_layer(out_channels)

        self.se = SEBlock(out_channels, reduction=se_reduction)

        if self.use_projection:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                norm_layer(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.final_act = activation()

    def stochastic_depth(self, x, residual):
        # x = main path, residual = shortcut
        if not self.training or self.drop_prob == 0.0:
            return x + residual
        if torch.rand(1).item() < self.drop_prob:
            # drop main path -> return only residual
            return residual
        return x + residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.dwconv(out)))
        out = self.bn3(self.conv_pw(out))
        out = self.se(out)
        out = self.stochastic_depth(out, identity)
        out = self.final_act(out)
        return out

# -------------------------
# CNNSENormalWithFit + EMA
# -------------------------
class CEMO(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
        conv_channels=[64, 128, 256],
        residuals_per_stage: int = 2,
        se_reduction: int = 8,
        p_drop: float = 0.1,
        dropout_fc: float = 0.3,
        use_ema: bool = False,
        ema_decay: float = 0.999,
        device: str = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_ema = use_ema
        # set device but DO NOT move model here (we will ensure move in callbacks/fit)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, conv_channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(conv_channels[0]),
            nn.SiLU()
        )

        # Stages
        self.stages = nn.ModuleList()
        in_ch = conv_channels[0]
        total_blocks = sum(residuals_per_stage for _ in conv_channels)
        block_id = 0

        for out_ch in conv_channels:
            blocks = []
            for j in range(residuals_per_stage):
                stride = 2 if (j == 0 and in_ch != out_ch) else 1
                drop_prob = p_drop * (block_id / max(1, total_blocks - 1))
                blocks.append(BottleneckSEBlock(in_ch, out_ch, stride, se_reduction=se_reduction, drop_prob=drop_prob))
                in_ch = out_ch
                block_id += 1
            self.stages.append(nn.Sequential(*blocks))

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout_fc = nn.Dropout(dropout_fc)
        self.fc = nn.Linear(conv_channels[-1], num_classes)

        # default criterion: allow override by user later
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # storage for callbacks/optimizer if user uses model-level API
        self._callbacks = []
        self.optimizer = None

        # EMA (create but don't assume device yet)
        if self.use_ema:
            try:
                self.ema = EMA(self, decay=ema_decay, device=self.device)
            except Exception:
                self.ema = EMA(self, decay=ema_decay)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout_fc(x)
        return self.fc(x)

    # -------------------------
    # Model-level callbacks (so CEMO.callbacks(...) works)
    # -------------------------
    def callbacks(self, optimizer="Adam", lr=1e-3, weight_decay=1e-4,
                  model_checkpoint=False, earlystop=None, scheduler=None, scheduler_params=None):
        # ensure model & submodules moved to desired device BEFORE creating optimizer
        try:
            self.to(self.device)
        except Exception:
            pass

        # if EMA present, ensure ema model is on same device
        if hasattr(self, "ema"):
            try:
                self.ema.to(self.device)
                self.ema.ema_model.eval()
            except Exception:
                pass

        # Optimizer
        if optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        # Scheduler
        if scheduler:
            if scheduler.lower() == "steplr":
                sched = torch.optim.lr_scheduler.StepLR(self.optimizer, **(scheduler_params or {"step_size":5,"gamma":0.5}))
            elif scheduler.lower() == "reduceonplateau":
                sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **(scheduler_params or {"patience":3,"factor":0.5}))
            else:
                raise ValueError(f"Unsupported scheduler: {scheduler}")
            self._callbacks.append(LRSchedulerCallback(sched))

        # ModelCheckpoint
        if model_checkpoint:
            self._callbacks.append(ModelCheckpoint(filepath="best_cemo_model.pth", verbose=True))

        # EarlyStopping
        if earlystop:
            self._callbacks.append(EarlyStopping(
                patience=earlystop.get("patience",7),
                delta=earlystop.get("delta",0.0),
                verbose=earlystop.get("verbose",True),
                restore_best_weights=True
            ))

    # -------------------------
    # Fit metodu (güncellendi: cihaz senkronizasyonu)
    # -------------------------
    def fit(self, train_loader, val_loader=None, epochs=5, lr=1e-3, weight_decay=1e-4, use_ema=False):
        # Ensure model is on right device
        try:
            self.to(self.device)
        except Exception:
            pass

        # ensure EMA model is also on device if available
        if use_ema and hasattr(self, "ema"):
            try:
                self.ema.to(self.device)
                self.ema.ema_model.eval()
            except Exception:
                pass
        elif hasattr(self, "ema"):
            # even if not using EMA for this run, keep ema_model on device to avoid mismatches later
            try:
                self.ema.to(self.device)
            except Exception:
                pass

        # if optimizer not provided via callbacks(), create a default one (params already on device)
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        # use model's criterion
        criterion = getattr(self, "criterion", torch.nn.CrossEntropyLoss())

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(epochs):
            # TRAIN
            self.train()
            running_loss, correct, total = 0.0, 0, 0
            loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Train", leave=False)
            for x, y in loop:
                # move batch to same device as model
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                out = self(x)  # now model weights and input tensor are on same device
                if y.ndim > 1:
                    y_proc = y.view(-1)
                else:
                    y_proc = y
                loss = criterion(out, y_proc)
                loss.backward()
                self.optimizer.step()

                # EMA update (if available)
                if hasattr(self, "ema"):
                    try:
                        self.ema.update(self)
                    except Exception:
                        pass

                batch_size = x.size(0)
                running_loss += loss.item() * batch_size
                total += batch_size
                correct += (out.argmax(dim=1) == y_proc).sum().item()
                avg_loss = running_loss / total if total > 0 else 0.0
                acc = 100. * (correct / total) if total > 0 else 0.0
                loop.set_postfix(loss=avg_loss, acc=acc)

            train_loss = running_loss / total if total > 0 else 0.0
            train_acc = 100. * (correct / total) if total > 0 else 0.0
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # VALIDATION
            if val_loader:
                self.eval()
                val_loss, val_correct, val_total = 0.0, 0, 0
                all_preds, all_targets = [], []
                loop_val = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Val", leave=False)
                with torch.no_grad():
                    for x, y in loop_val:
                        x, y = x.to(self.device), y.to(self.device)
                        if hasattr(self, "ema"):
                            try:
                                self.ema.to(self.device)
                                self.ema.ema_model.eval()
                                out = self.ema.ema_model(x)
                            except Exception:
                                out = self(x)
                        else:
                            out = self(x)

                        if y.ndim > 1:
                            y_proc = y.view(-1)
                        else:
                            y_proc = y

                        loss = criterion(out, y_proc)
                        preds = out.argmax(dim=1)
                        batch_size = x.size(0)
                        val_loss += loss.item() * batch_size
                        val_total += batch_size
                        val_correct += (preds == y_proc).sum().item()
                        all_preds.extend(preds.cpu().numpy().ravel().tolist())
                        all_targets.extend(y_proc.cpu().numpy().ravel().tolist())
                        avg_val_loss = val_loss / val_total if val_total > 0 else 0.0
                        val_acc_running = 100. * (val_correct / val_total) if val_total > 0 else 0.0
                        loop_val.set_postfix(loss=avg_val_loss, acc=val_acc_running)

                val_loss_final = val_loss / val_total if val_total > 0 else 0.0
                val_acc = 100. * (val_correct / val_total) if val_total > 0 else 0.0
                history["val_loss"].append(val_loss_final)
                history["val_acc"].append(val_acc)

                print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss_final:.4f}, Acc: {val_acc:.2f}%")

                # normalize arrays for sklearn
                try:
                    all_preds_arr = np.array(all_preds).ravel().astype(int)
                    all_targets_arr = np.array(all_targets).ravel().astype(int)
                except Exception:
                    all_preds_arr = np.array(all_preds).ravel()
                    all_targets_arr = np.array(all_targets).ravel()

                try:
                    from sklearn.metrics import classification_report, confusion_matrix
                    print(classification_report(all_targets_arr, all_preds_arr))
                    cm = confusion_matrix(all_targets_arr, all_preds_arr)
                    import seaborn as sns
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(6,5))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                    plt.title(f"Confusion Matrix - Epoch {epoch+1}")
                    plt.xlabel("Predicted")
                    plt.ylabel("True")
                    plt.show()
                except Exception as e:
                    print("Couldn't produce sklearn report/plot:", e)

                # CALLBACKS (robust calls)
                for cb in self._callbacks:
                    if isinstance(cb, EarlyStopping):
                        try:
                            cb(val_loss_final, self)
                        except Exception:
                            try:
                                cb.step(val_loss_final)
                            except Exception:
                                pass
                        if getattr(cb, "early_stop", False):
                            print(f"⏹ Early stopping at epoch {epoch+1}")
                            if getattr(cb, "restore_best_weights", False):
                                try:
                                    cb.restore_weights(self)
                                except Exception:
                                    try:
                                        cb.restore_best(self)
                                    except Exception:
                                        pass
                            return history
                    elif isinstance(cb, ModelCheckpoint):
                        try:
                            cb(epoch, self, val_loss_final)
                        except Exception:
                            try:
                                cb.on_epoch_end(epoch, self, val_loss_final)
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
