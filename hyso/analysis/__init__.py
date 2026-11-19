"""Evidence, pure selection and separately authorized native CPU execution."""
from typing import TYPE_CHECKING as _TYPE_CHECKING

from ._codec import EvidenceError
from .evidence import EvidenceScope, EvidenceRecord, EvidenceIndex, compare_evidence
from .report import AnalysisContext, UnifiedAnalysisReport
from .feature_plan import ModelContext, FeaturePlanRequest, FeaturePlan, plan_features
from .catalog import AnalysisFeature, feature_catalog, PROFILES
from .execution import (AnalysisExecutionError, CPUAnalysisPolicy, AnalysisPlan, AnalysisDryRun,
                        AnalysisExecutionResult, plan_analysis, preflight_analysis, execute_analysis)
from .native_bridge import LiveCPUModel, TrustedCPUExecutionPolicy, execute_scoped_cpu
from .isolated_bridge import TrustedFactoryBinding, IsolatedCPUExecutionPolicy, execute_isolated_cpu
from .model_comparison import ModelComparisonError, ModelComparisonSource, ModelComparisonResult, compare_models
from .sources import (ArtifactSource, OperationPolicy, NativeBinding, RetainedSource,
                      InferenceSource, PipelineSource, RecipeSource, EvaluationSource,
                      QualitySource, TrainingSource, ProtocolSource, RooflineSource, RuntimeSource)

__all__ = ['EvidenceError', 'EvidenceScope', 'EvidenceRecord', 'EvidenceIndex',
           'compare_evidence', 'AnalysisContext', 'UnifiedAnalysisReport',
           'ModelContext', 'FeaturePlanRequest', 'FeaturePlan', 'plan_features',
           'AnalysisFeature', 'feature_catalog', 'PROFILES', 'AnalysisExecutionError',
           'CPUAnalysisPolicy', 'AnalysisPlan', 'AnalysisDryRun', 'AnalysisExecutionResult',
           'plan_analysis', 'preflight_analysis', 'execute_analysis',
           'LiveCPUModel', 'TrustedCPUExecutionPolicy', 'execute_scoped_cpu',
           'TrustedFactoryBinding', 'IsolatedCPUExecutionPolicy', 'execute_isolated_cpu',
           'ModelComparisonError', 'ModelComparisonSource', 'ModelComparisonResult', 'compare_models']
__all__ += ['ArtifactSource', 'OperationPolicy', 'NativeBinding', 'RetainedSource',
            'InferenceSource', 'PipelineSource', 'RecipeSource', 'EvaluationSource',
            'QualitySource', 'TrainingSource', 'ProtocolSource', 'RooflineSource', 'RuntimeSource']

__all__ += ['MultiHeadGraphEvaluationResult', 'evaluate_graph_heads', 'compare_graph_heads', 'GraphHeadSpec']

if _TYPE_CHECKING:
    from .graph_heads import GraphHeadSpec as GraphHeadSpec


def __dir__():
    return sorted(set(globals()) | set(__all__))


def __getattr__(name):
    # evaluation imports analysis._codec while its public types initialize.
    if name in ('MultiHeadGraphEvaluationResult', 'evaluate_graph_heads', 'compare_graph_heads', 'GraphHeadSpec'):
        from importlib import import_module
        value = getattr(import_module('.graph_heads', __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)
