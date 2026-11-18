"""HYSo public entry point.

Use ``ModularCNN`` and ``Trainer`` for new classification workflows. Domain
namespaces expose measurement, evidence and advisory APIs; importing them never
requests a training run or benchmark. Legacy canonical import paths are retained.
See README.md and docs/guides/python-api.md for CPU-first examples and limits.
"""
from typing import TYPE_CHECKING as _TYPE_CHECKING
from importlib import import_module as _import_module

from . import models, callbacks, augmentation
from .evaluation import EvaluationResult, evaluate_predictions, evaluate_classification, evaluation_capabilities
from .quality import (EvaluationCohort, QualityExperimentResult, measure_time_to_quality,
                      evaluate_robustness, evaluate_precision_quality, compare_performance_quality)
from .structure import ModelStructureSnapshot, ModelStructureError, capture_model_structure
from .structure_comparison import ModelStructureComparison, ModelStructureComparisonError, compare_model_structures
from .architecture_audit import ModelArchitectureAudit, ModelArchitectureAuditError, audit_model_architecture

if _TYPE_CHECKING:
    from .training import TrainingConfig as TrainingConfig, TrainingConfigError as TrainingConfigError
    from .training import TrainingSession as TrainingSession, TrainingResult as TrainingResult
    from .models.registry import ModelRegistry as ModelRegistry, ModelConfigError as ModelConfigError
    from .models.builder import ModelBuilder as ModelBuilder
    from .models.graph import GraphModelBuilder as GraphModelBuilder
    from .models.modular import ModularCNN as ModularCNN
    from .training import Trainer as Trainer
    from .benchmark import BenchmarkRunner as BenchmarkRunner
    from .experiments import ExperimentManager as ExperimentManager
    from . import training as training, benchmark as benchmark, performance as performance
    from . import analysis as analysis, advisor as advisor, experiments as experiments
    from . import reporting as reporting

_LAZY_EXPORTS = {
    'TrainingConfig': ('.training', 'TrainingConfig'),
    'TrainingConfigError': ('.training', 'TrainingConfigError'),
    'TrainingSession': ('.training', 'TrainingSession'),
    'TrainingResult': ('.training', 'TrainingResult'),
    'ModelRegistry': ('.models.registry', 'ModelRegistry'),
    'ModelConfigError': ('.models.registry', 'ModelConfigError'),
    'ModelBuilder': ('.models.builder', 'ModelBuilder'),
    'GraphModelBuilder': ('.models.graph', 'GraphModelBuilder'),
    'ModularCNN': ('.models.modular', 'ModularCNN'),
    'Trainer': ('.training', 'Trainer'),
    'BenchmarkRunner': ('.benchmark', 'BenchmarkRunner'),
    'ExperimentManager': ('.experiments', 'ExperimentManager'),
    **{name: ('.' + name, None) for name in
       ('training', 'benchmark', 'performance', 'analysis', 'advisor', 'experiments', 'reporting')},
}


def __getattr__(name: str):
    """Resolve public convenience aliases without instantiating any component."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module_name, symbol = _LAZY_EXPORTS[name]
    module = _import_module(module_name, __name__)
    value = module if symbol is None else getattr(module, symbol)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public names in Python help and interactive completion."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
