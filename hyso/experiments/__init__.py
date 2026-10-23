"""Explicit local experiment tracking; no training or benchmark orchestration."""

from .manager import ExperimentError, ExperimentManager

__all__ = ["ExperimentManager", "ExperimentError"]
