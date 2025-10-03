import torch

class EarlyStopping:
    def __init__(self, patience=7, delta=0.0, verbose=True, restore_best_weights=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = model.state_dict()
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_weights(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class ModelCheckpoint:
    def __init__(self, filepath="best_model.pth", verbose=True):
        self.filepath = filepath
        self.verbose = verbose
        self.best_loss = float('inf')

    def __call__(self, epoch, model, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.filepath)
            if self.verbose:
                print(f"Model saved at epoch {epoch+1} with val_loss {val_loss:.4f}")


class LRSchedulerCallback:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def step(self, val_loss=None):
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()
