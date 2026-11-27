"""Advisor contracts and read-only native observations; no workload execution."""
from ._json import AdvisorError

def export_advice(result, *, language='en') -> dict:
    """Read saved deployment evidence; never execute or alter a model."""
    from ..export.presentation import advise
    return advise(result,language=language)
from .contracts import (
    AdvisorReport, AdvisorRequest, Consistency, EvaluationProtocol,
    EvidenceBinding, EvidenceRef, EvidenceSupport, Feasibility, Finding,
    FindingKind, GoalEvaluation, InputValidity, Metric, MetricScope,
    ObservedGoal, Priority, RequestValidation, ResourceBudget, RuleVersion, UserGoal,
)
from .validation import validate_request
from .resolver import (EvidenceEligibility, MetricResolution, NativeMetric,
                       NativeMetricResolver, NativeObservation, ResolvedMetric)
from .engine import advise, validate_advisor_report
from .presentation import build_display_model, to_display_dict, to_display_json, to_markdown
from .goals import evaluate_goal, evaluate_goals, goal_context
from .planning import DecisionPlan, PlannedFinding
from .recommendations import RecommendationReport, recommend
from .comparison import (ComparisonContract, CapturedCPUComparisonContract, P26QualityComparisonContract, P26BoundQualityComparisonContract,
                         ComparisonReason, ComparisonResult, ComparisonStatus, compare_records)
from .performance import MetricDifference, PerformanceComparison, compare_performance
from .profiles import (BoundEvaluationReference, P26ModelProfileComparisonContract,
                       UnifiedModelComparison, compare_p26_model_profiles)

__all__ = [
    'export_advice',
    'AdvisorError', 'AdvisorReport', 'AdvisorRequest', 'Consistency',
    'EvaluationProtocol', 'EvidenceBinding', 'EvidenceRef', 'EvidenceSupport',
    'Feasibility', 'Finding', 'FindingKind', 'GoalEvaluation', 'InputValidity',
    'Metric', 'MetricScope', 'ObservedGoal', 'Priority', 'RequestValidation',
    'ResourceBudget', 'RuleVersion', 'UserGoal', 'validate_request',
    'EvidenceEligibility', 'MetricResolution', 'NativeMetric', 'NativeMetricResolver',
    'ResolvedMetric', 'advise', 'validate_advisor_report',
    'NativeObservation',
    'evaluate_goal', 'evaluate_goals', 'goal_context',
    'DecisionPlan', 'PlannedFinding',
    'RecommendationReport', 'recommend',
    'ComparisonContract', 'ComparisonReason', 'ComparisonResult', 'ComparisonStatus', 'compare_records',
    'CapturedCPUComparisonContract',
    'P26QualityComparisonContract',
    'P26BoundQualityComparisonContract',
    'MetricDifference', 'PerformanceComparison', 'compare_performance',
    'BoundEvaluationReference', 'P26ModelProfileComparisonContract',
    'UnifiedModelComparison', 'compare_p26_model_profiles',
    'build_display_model', 'to_display_dict', 'to_display_json', 'to_markdown',
]
