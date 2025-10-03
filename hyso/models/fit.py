import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Modeller
from hyso.models.WithSE import CEMO, EMA  # SE-block tabanlı model ve EMA

# Callbacks
from hyso.callbacks.callbacks import EarlyStopping, ModelCheckpoint, LRSchedulerCallback

# Yardımcı kütüphaneler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torchsummary import summary

# Summary helper
# -----------------------
def model_summary(model, input_size):
    """Simple summary print like Keras."""
    try:
        summary(model, input_size)
    except:
        print(model)
        print("Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))


# -----------------------
# CNNSENormalWithFit + Fit ve Callbacks
# -----------------------
class CNNSENormalWithFit(CEMO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = None
        self.early_stopping = None
        self.model_checkpoint = None
        self.lr_scheduler = None
        self._callbacks = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    # -------------------------
    # Callbacks & Optimizer
    # -------------------------
    def callbacks(self, optimizer="Adam", optimizer_params=None,
                  scheduler=None, scheduler_params=None,
                  model_checkpoint=False, earlystop=None,
                  lr=1e-3, weight_decay=1e-4):

        # --- Optimizer ---
        opt_params = optimizer_params or {"lr": lr, "weight_decay": weight_decay}
        if optimizer=="Adam":
            self.optimizer = optim.Adam(self.parameters(), **opt_params)
        elif optimizer=="SGD":
            if "momentum" not in opt_params:
                opt_params["momentum"] = 0.9
            self.optimizer = optim.SGD(self.parameters(), **opt_params)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        # --- Scheduler ---
        if scheduler:
            if scheduler=="StepLR":
                sched = optim.lr_scheduler.StepLR(self.optimizer, **(scheduler_params or {"step_size":5,"gamma":0.5}))
            elif scheduler=="ReduceLROnPlateau":
                sched = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **(scheduler_params or {"patience":3,"factor":0.5}))
            elif scheduler=="CosineAnnealingLR":
                sched = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **(scheduler_params or {"T_max":10}))
            else:
                raise ValueError(f"Unsupported scheduler: {scheduler}")
            self.lr_scheduler = LRSchedulerCallback(sched)
            self._callbacks.append(self.lr_scheduler)

        # --- ModelCheckpoint ---
        if model_checkpoint:
            self.model_checkpoint = ModelCheckpoint(filepath="best_model.pth", verbose=True)
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
    # Fit Fonksiyonu
    # -------------------------
    def fit(self, train_loader, val_loader=None, epochs=10, lr=1e-3, use_ema=False):
        self.to(self.device)
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)

        criterion = nn.BCEWithLogitsLoss() if self.num_classes == 1 else nn.CrossEntropyLoss()
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        # -----------------------
        # EMA setup
        # -----------------------
        if use_ema:
            if not hasattr(self, "ema"):
                self.ema = EMA(self, decay=0.999)
            print("✅ EMA is enabled for training.")

        # -----------------------
        # Model summary
        # -----------------------
        sample = next(iter(train_loader))[0]
        input_size = tuple(sample.shape[1:])
        print("\n===== Model Summary =====")
        model_summary(self, input_size)
        print("=========================\n")

        for epoch in range(epochs):
            # ------------------------
            # TRAIN
            # ------------------------
            self.train()
            running_loss, correct, total = 0, 0, 0
            loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Train", leave=False)
            for inputs, targets in loop:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(inputs)

                if self.num_classes == 1:
                    targets = targets.float().unsqueeze(1)
                    loss = criterion(outputs, targets)
                    preds = (torch.sigmoid(outputs) > 0.5).long()
                else:
                    loss = criterion(outputs, targets)
                    preds = outputs.argmax(dim=1)

                loss.backward()
                self.optimizer.step()

                # EMA update
                if use_ema:
                    self.ema.update(self)

                running_loss += loss.item() * inputs.size(0)
                total += targets.size(0)
                correct += (preds == targets).sum().item()
                loop.set_postfix(loss=running_loss/total, acc=100.*correct/total)

            train_loss = running_loss / total
            train_acc = 100.*correct/total
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # ------------------------
            # VALIDATION
            # ------------------------
            if val_loader:
                self.eval()
                val_loss, correct, total = 0, 0, 0
                all_preds, all_targets = [], []
                loop_val = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Val", leave=False)
                with torch.no_grad():
                    for inputs, targets in loop_val:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)

                        # EMA prediction
                        if use_ema:
                            outputs = self.ema.ema_model(inputs)
                        else:
                            outputs = self(inputs)

                        if self.num_classes == 1:
                            targets = targets.float().unsqueeze(1)
                            loss = criterion(outputs, targets)
                            preds = (torch.sigmoid(outputs) > 0.5).long()
                        else:
                            loss = criterion(outputs, targets)
                            preds = outputs.argmax(dim=1)

                        val_loss += loss.item() * inputs.size(0)
                        total += targets.size(0)
                        correct += (preds == targets).sum().item()
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                        loop_val.set_postfix(loss=val_loss/total, acc=100.*correct/total)

                val_loss /= total
                val_acc = 100.*correct/total
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                      f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
                print(classification_report(all_targets, all_preds))

                # Confusion Matrix
                cm = confusion_matrix(all_targets, all_preds)
                plt.figure(figsize=(6,5))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.title(f"Confusion Matrix - Epoch {epoch+1}")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.show()

                # Callback çağrıları
                for cb in self._callbacks:
                    if isinstance(cb, EarlyStopping):
                        cb(val_loss, self)
                        if cb.early_stop:
                            print(f"Early stopping at epoch {epoch+1}")
                            if cb.restore_best_weights:
                                cb.restore_weights(self)
                            return history
                    elif isinstance(cb, ModelCheckpoint):
                        cb(epoch, self, val_loss)
                    elif isinstance(cb, LRSchedulerCallback):
                        cb.step(val_loss)

        return history
