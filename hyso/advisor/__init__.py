"""Advisor contracts and read-only native observations; no workload execution."""
from ._json import AdvisorError
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
from .goals import evaluate_goal, evaluate_goals, goal_context
from .planning import DecisionPlan, PlannedFinding
from .comparison import ComparisonContract, ComparisonReason, ComparisonResult, ComparisonStatus, compare_records

__all__ = [
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
    'ComparisonContract', 'ComparisonReason', 'ComparisonResult', 'ComparisonStatus', 'compare_records',
]
