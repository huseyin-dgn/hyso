"""HYSo public entry point.

Use ``ModularCNN`` and ``Trainer`` for new classification workflows. Domain
namespaces expose measurement, evidence and advisory APIs; importing them never
requests a training run or benchmark. Legacy canonical import paths are retained.
See README.md and docs/guides/python-api.md for CPU-first examples and limits.
"""
from typing import TYPE_CHECKING as _TYPE_CHECKING, Literal as _Literal, Any as _Any
from importlib import import_module as _import_module

from . import models, callbacks, augmentation
from .evaluation import EvaluationResult, evaluate_predictions, evaluate_classification, evaluation_capabilities
from .quality import (EvaluationCohort, QualityExperimentResult, measure_time_to_quality,
                      evaluate_robustness, evaluate_precision_quality, compare_performance_quality)
from .structure import ModelStructureSnapshot, ModelStructureError, capture_model_structure
from .structure_comparison import ModelStructureComparison, ModelStructureComparisonError, compare_model_structures
from .architecture_audit import ModelArchitectureAudit, ModelArchitectureAuditError, audit_model_architecture

class Device:
    """Discover plain device selectors without querying or allocating hardware.

    Returns:
        Strings accepted by existing device validators and JSON serialization.
    Raises:
        ValueError: A CUDA index is not a nonnegative integer.
    Example:
        config = TrainingConfig(device=Device.cpu())

    HYSo Guide: training.configuration
    Offline: hyso docs show training.configuration --lang en
    Open: hyso docs open training.configuration --lang en
    """

    @staticmethod
    def cpu() -> _Literal['cpu']:
        """Return the CPU selector without inspecting the runtime.

        Returns:
            The plain string 'cpu'.
        Raises:
            No additional validation or hardware access.
        Example:
            Device.cpu()

        HYSo Guide: training.configuration
        Offline: hyso docs show training.configuration --lang en
        Open: hyso docs open training.configuration --lang en
        """
        return 'cpu'

    @staticmethod
    def cuda(index: int = 0) -> str:
        """Format an indexed CUDA selector; execution still requires native admission.

        Args:
            index: Nonnegative CUDA device index; bool is not an index.
        Returns:
            A plain 'cuda:index' string without hardware queries or allocation.
        Raises:
            ValueError: index is not a nonnegative integer.
        Example:
            selector = Device.cuda(1)

        HYSo Guide: training.configuration
        Offline: hyso docs show training.configuration --lang en
        Open: hyso docs open training.configuration --lang en
        """
        if type(index) is not int or index < 0:
            raise ValueError('CUDA index must be a nonnegative integer')
        return f'cuda:{index}'


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
    from . import evaluation as evaluation, quality as quality, checkpoint as checkpoint
    from . import export as export

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
       ('training', 'benchmark', 'performance', 'analysis', 'advisor', 'experiments', 'reporting','export')},
}

# Existing root names plus the existing lazy aliases. The models namespace keeps
# its independent frozen legacy wildcard contract.
__all__ = [
    'Device','ModelBuilder','GraphModelBuilder','ModelRegistry','ModelConfigError','ModularCNN',
    'Trainer','TrainingConfig','TrainingConfigError','TrainingSession','TrainingResult','BenchmarkRunner','ExperimentManager',
    'EvaluationResult','EvaluationCohort','QualityExperimentResult','evaluate_predictions','evaluate_classification',
    'evaluation_capabilities','measure_time_to_quality','evaluate_robustness','evaluate_precision_quality','compare_performance_quality',
    'ModelStructureSnapshot','ModelStructureError','capture_model_structure','ModelStructureComparison',
    'ModelStructureComparisonError','compare_model_structures','ModelArchitectureAudit','ModelArchitectureAuditError','audit_model_architecture',
    'models','callbacks','augmentation','evaluation','quality','checkpoint','benchmark','training','performance',
    'analysis','advisor','reporting','experiments','structure','structure_comparison','architecture_audit','export',
]


def __getattr__(name: str) -> _Any:
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
