"""CPU/CUDA inference benchmarking, independent of the training lifecycle."""

from .result import BenchmarkResult, compare_results
from .runner import BenchmarkRunner

__all__ = ["BenchmarkRunner", "BenchmarkResult", "compare_results"]
