"""Evidence, pure selection and separately authorized native CPU execution."""
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

__all__ = ['EvidenceError', 'EvidenceScope', 'EvidenceRecord', 'EvidenceIndex',
           'compare_evidence', 'AnalysisContext', 'UnifiedAnalysisReport',
           'ModelContext', 'FeaturePlanRequest', 'FeaturePlan', 'plan_features',
           'AnalysisFeature', 'feature_catalog', 'PROFILES', 'AnalysisExecutionError',
           'CPUAnalysisPolicy', 'AnalysisPlan', 'AnalysisDryRun', 'AnalysisExecutionResult',
           'plan_analysis', 'preflight_analysis', 'execute_analysis',
           'LiveCPUModel', 'TrustedCPUExecutionPolicy', 'execute_scoped_cpu',
           'TrustedFactoryBinding', 'IsolatedCPUExecutionPolicy', 'execute_isolated_cpu',
           'ModelComparisonError', 'ModelComparisonSource', 'ModelComparisonResult', 'compare_models']
