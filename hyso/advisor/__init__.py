"""Immutable Advisor contracts and static request validation; no advise engine."""
from ._json import AdvisorError
from .contracts import (
    AdvisorReport, AdvisorRequest, Consistency, EvaluationProtocol,
    EvidenceBinding, EvidenceRef, EvidenceSupport, Feasibility, Finding,
    FindingKind, GoalEvaluation, InputValidity, Metric, MetricScope,
    ObservedGoal, Priority, RequestValidation, ResourceBudget, RuleVersion, UserGoal,
)
from .validation import validate_request

__all__ = [
    'AdvisorError', 'AdvisorReport', 'AdvisorRequest', 'Consistency',
    'EvaluationProtocol', 'EvidenceBinding', 'EvidenceRef', 'EvidenceSupport',
    'Feasibility', 'Finding', 'FindingKind', 'GoalEvaluation', 'InputValidity',
    'Metric', 'MetricScope', 'ObservedGoal', 'Priority', 'RequestValidation',
    'ResourceBudget', 'RuleVersion', 'UserGoal', 'validate_request',
]
