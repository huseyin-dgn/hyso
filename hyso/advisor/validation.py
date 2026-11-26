"""Static interval consistency only. No performance or feasibility inference."""
import typing as _typing
from fractions import Fraction

from ._json import AdvisorError
from .contracts import (AdvisorRequest, Consistency, GoalEvaluation, InputValidity, Metric,
                        RequestValidation, _METRICS)


def validate_request(request: '_typing.Any') -> 'RequestValidation':
    """Find impossible hard intervals for exactly matching, complete goal contexts.

    Soft goals, priorities, budgets and different metrics cannot create a hard
    contradiction here. A missing match or conflict is no feasibility proof.

    Args:
        request (_typing.Any): Validated advisory/selection request; does not authorize execution. Required.

    Returns:
        The retained validate request value, without recomputing measurements.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.
        RuntimeError: A requested native workload fails; inspect retained state before retrying.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        validate_request(request)

    HYSo Guide: advisor.recommendations
    Offline: hyso docs show advisor.recommendations --lang en
    Open: hyso docs open advisor.recommendations --lang en
    """
    if type(request) is not AdvisorRequest:
        raise AdvisorError('typed_request_required')
    evaluations = tuple(GoalEvaluation(
        g.goal_id, g.input_validity,
        reasons=tuple('missing:' + name for name in g.missing_scope_fields)
    ) for g in request.goals)
    groups = {}
    for goal in request.goals:
        if goal.constraint != 'hard' or goal.input_validity != InputValidity.VALID:
            continue
        key = (goal.metric, goal.scope, goal.protocol, goal.value_basis, goal.baseline_id)
        groups.setdefault(key, []).append(goal)
    conflicts = []
    for goals in groups.values():
        # All admitted metrics are nonnegative. Accuracy/drop also have exact
        # mathematical upper bounds; these are domains, not performance targets.
        lower = Fraction(0)
        upper = (Fraction(1) if goals[0].metric == Metric.ACCURACY else
                 Fraction(100) if goals[0].metric == Metric.ACCURACY_DROP else None)
        lower_strict = upper_strict = False
        for goal in goals:
            # Exact decimal arithmetic: no float tolerance or guessed unit conversion.
            value = Fraction(str(goal.target)) * Fraction(str(_METRICS[goal.metric][0][goal.unit]))
            if goal.operator in ('>=', '>', '=='):
                strict = goal.operator == '>'
                if lower is None or value > lower:
                    lower, lower_strict = value, strict
                elif value == lower:
                    lower_strict |= strict
            if goal.operator in ('<=', '<', '=='):
                strict = goal.operator == '<'
                if upper is None or value < upper:
                    upper, upper_strict = value, strict
                elif value == upper:
                    upper_strict |= strict
        if lower is not None and upper is not None and (
                lower > upper or lower == upper and (lower_strict or upper_strict)):
            conflicts.extend(goal.goal_id for goal in goals)
    return RequestValidation(
        evaluations, Consistency.INCONSISTENT if conflicts else Consistency.NO_CONFLICT,
        conflicting_goal_ids=tuple(sorted(conflicts)),
        reasons=('empty_hard_constraint_interval',) if conflicts else
                ('No direct interval contradiction identified; joint feasibility was not evaluated.',))
