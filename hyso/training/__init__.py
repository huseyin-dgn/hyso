"""Model-independent supervised training."""

from .trainer import Trainer
from .config import TrainingConfig, TrainingConfigError
from .session import TrainingSession, TrainingResult

__all__ = ["Trainer", "TrainingConfig", "TrainingConfigError", "TrainingSession", "TrainingResult"]
