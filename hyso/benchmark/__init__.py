"""CPU/CUDA inference benchmarking, independent of the training lifecycle."""

from .result import BenchmarkResult
from .runner import BenchmarkRunner

__all__ = ["BenchmarkRunner", "BenchmarkResult"]
