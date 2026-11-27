from __future__ import annotations

import copy

import torch

class EarlyStopping:
    """Track validation-loss improvements and retain a copy of the best weights.

    Args:
        patience: Consecutive non-improving validation calls before early_stop is set. Default: 7.
        delta: Minimum validation-loss decrease needed to count as an improvement. Default: 0.0.
        verbose: Print callback progress messages when True. Default: True.
        restore_best_weights: Preference used by Trainer to restore the retained best state;
            restore_weights itself restores whenever a state exists. Default: True.

    Notes:
        This callback is mutable. Construction performs no validation or training.
        Trainer decides whether to call restore_weights using restore_best_weights.
    """
    def __init__(self, patience: int=7, delta: float=0.0, verbose: bool=True, restore_best_weights: bool=True) -> None:
        """Initialize EarlyStopping from the supplied settings or evidence.

        Args:
            patience: Consecutive non-improving validation calls before early_stop is set. Default: 7.
            delta: Minimum validation-loss decrease needed to count as an improvement. Default: 0.0.
            verbose: Print callback progress messages when True. Default: True.
            restore_best_weights: Preference used by Trainer to restore the retained best state;
                restore_weights itself restores whenever a state exists. Default: True.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        """Update the best state or non-improvement counter; set early_stop at patience.

        Args:
            val_loss: Current validation loss; a lower value is better. Required.
            model: Caller-owned torch.nn.Module; must meet the operation device, dtype and input contract.
                Required.

        Returns:
            None; updates this callback in place.
        """
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_weights(self, model: torch.nn.Module) -> None:
        """Load the retained best state into model, if a best state has been observed.

        Args:
            model: Caller-owned torch.nn.Module; must meet the operation device, dtype and input contract.
                Required.

        Returns:
            None. Without a retained best state this is a no-op.
        """
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class ModelCheckpoint:
    """Save a model state_dict whenever validation loss strictly improves.

    Args:
        filepath: Checkpoint filename for best model state_dict; callback writes on each strict
            validation-loss improvement. Default: 'best_model.pth'.
        verbose: Print callback progress messages when True. Default: True.

    Notes:
        This mutable callback writes to filepath on improvement and may overwrite
        that file. It saves weights only, without optimizer, scheduler or RNG state.
        Use the checkpoint API for resumable training state.
    """
    def __init__(self, filepath: str="best_model.pth", verbose: bool=True) -> None:
        """Initialize ModelCheckpoint from the supplied settings or evidence.

        Args:
            filepath: Checkpoint filename for best model state_dict; callback writes on each strict
                validation-loss improvement. Default: 'best_model.pth'.
            verbose: Print callback progress messages when True. Default: True.
        """
        self.filepath = filepath
        self.verbose = verbose
        self.best_loss = float('inf')

    def __call__(self, epoch: int, model: torch.nn.Module, val_loss: float) -> None:
        """Save model.state_dict() to filepath after a strict validation-loss improvement.

        Args:
            epoch: Zero-based epoch index, used only in checkpoint progress text. Required.
            model: Caller-owned torch.nn.Module; must meet the operation device, dtype and input contract.
                Required.
            val_loss: Current validation loss; a lower value is better. Required.

        Returns:
            None; may write or overwrite the configured file.
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.filepath)
            if self.verbose:
                print(f"Model saved at epoch {epoch+1} with val_loss {val_loss:.4f}")


class LRSchedulerCallback:
    """Step an explicit native scheduler, passing validation loss for plateau mode.

    Args:
        scheduler: Existing native PyTorch scheduler; ReduceLROnPlateau consumes val_loss, other
            schedulers step without a metric. Required.

    Notes:
        Call step at the scheduler's intended boundary. Trainer owns the stepping
        of its configured scheduler; do not also step it through a second callback.
    """
    def __init__(self, scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau) -> None:
        """Initialize LRSchedulerCallback from the supplied settings or evidence.

        Args:
            scheduler: Existing native PyTorch scheduler; ReduceLROnPlateau consumes val_loss, other
                schedulers step without a metric. Required.
        """
        self.scheduler = scheduler

    def step(self, val_loss: float | None=None) -> None:
        """Advance the scheduler once, using val_loss only for ReduceLROnPlateau.

        Args:
            val_loss: Current validation loss; a lower value is better. Default: None.

        Returns:
            None; mutates the supplied scheduler.
        """
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()
