"""Readable export from a single AdvisorReport, with no evidence resolution."""
from pathlib import Path
import html

from ._json import AdvisorError
from .contracts import AdvisorReport, FindingKind
from .resolver import NativeObservation, NativeMetric
from .planning import PlannedFinding


def render_markdown(report, path=None):
    if type(report) is not AdvisorReport:
        raise AdvisorError('typed_advisor_report_required')
    esc = lambda value: html.escape(str(value), quote=False).replace('\r', ' ').replace('\n', ' ')
    lines = ['# HYSo Advisor — existing evidence', '',
             'Source: `' + report.source_report_id + '`',
             'Source canonical SHA-256: `' + report.source_report_sha256 + '`',
             'Mode: ' + report.request.mode + '; priority: ' + (report.request.priority.value if report.request.priority else 'unspecified') + '.',
             'Status: ' + report.evaluation_status + '. No workload or model change was executed.', '']
    has_goal_results = report.evaluation_status == 'goals_evaluated' or report.evaluation_status == 'planning_only' and bool(report.request.goals)
    if has_goal_results:
        evaluations = {g.goal_id: g for g in report.goal_evaluations}
        metrics = {f.resolved_metric.reference: f.resolved_metric for f in report.findings
                   if type(f) is NativeObservation}
        lines += ['## Goals — record-local observations', '',
                  'These results do not certify deployment acceptance or future feasibility.', '']
        for goal in report.request.goals:
            evaluation = evaluations[goal.goal_id]
            lines.append('- ' + esc(goal.goal_id) + ': ' + esc(goal.metric.value) + ' ' +
                         esc(goal.operator) + ' ' + esc(goal.target) + ' ' + esc(goal.unit) +
                         ' (' + goal.constraint + ').')
            lines.append('  - Input: ' + evaluation.input_validity.value + '; evidence: ' +
                         (evaluation.evidence_support.value if evaluation.evidence_support else 'not_evaluated') +
                         '; observed: ' + evaluation.observed.value + '; future feasibility: ' + evaluation.feasibility.value + '.')
            for ref in evaluation.evidence:
                metric = metrics.get(ref)
                value = (format(metric.value, '.12g') + ' ' + ref.unit) if metric is not None else 'unavailable'
                if metric is not None and ref.unit == 'fraction':
                    value += ' (' + format(metric.value * 100, '.8g') + '%)'
                lines.append('  - Recorded value: ' + value + '; record `' + ref.record_id + '`; path `' + ref.metric_path + '`.')
            lines += ['  - ' + esc(reason) for reason in evaluation.reasons if report.evaluation_status != 'planning_only'
                      or not reason.startswith(('unproven_associations:', 'observed_value:', 'target:', 'scope_and_protocol:'))]
        lines += ['', 'Cross-goal consistency: ' +
                  (report.request_validation.consistency.value if report.request_validation else 'not_evaluated') + '.', '']
    observations = [f for f in report.findings if f.kind == FindingKind.OBSERVATION]
    lines += ['## Observations', '']
    for finding in observations:
        if type(finding) is NativeObservation:
            m = finding.resolved_metric
            if report.evaluation_status == 'planning_only':
                # Keep the readable view compact; all 18 native observations and
                # full reference/binding details remain in the same report JSON.
                if m.metric in (NativeMetric.FORWARD_P50, NativeMetric.FORWARD_P99, NativeMetric.E2E_P50,
                                NativeMetric.PRECISION_THROUGHPUT, NativeMetric.RSS_CURRENT):
                    continue
                lines.append('- ' + esc(finding.statement.partition(' Measurement context: ')[0]))
                native = m.native_context
                scope = ', '.join(str(native[k]) for k in ('device', 'precision') if native.get(k) is not None)
                if native.get('batch') is not None:
                    scope += ', batch ' + str(native['batch'])
                lines.append('  - ' + esc(scope) + '; source `' + m.reference.record_id[:12] + '`; source-local, not a verified improvement.')
                continue
            lines.append('- ' + esc(finding.statement.partition(' Measurement context: ')[0]))
            native = m.native_context
            summary = ', '.join(k + '=' + str(native[k]) for k in (
                'measurement_scope', 'device', 'batch', 'precision', 'sample_count', 'split', 'averaging')
                if native.get(k) is not None)
            lines.append('  - Scope: ' + esc(summary) + '.')
        else:
            lines.append('- ' + esc(finding.statement))
        for ref in finding.evidence:
            lines.append('  - Record `' + ref.record_id + '`; path `' + ref.metric_path + '`; domain `' + ref.domain + '`; status `' + ref.status + '`.')
        lines.append('  - Limit: source-local description; associations may be declared/missing; no independent improvement certified.')
    if not observations:
        lines.append('No eligible supported measurements were found; missing data is not zero.')
    lines += ['', '## Evidence limits', '']
    limits = report.evidence_limitations
    if report.evaluation_status == 'planning_only':
        limits = tuple(v for v in limits if any(key in v for key in (
            'Supported means', 'P05 record', 'Planner v1', 'No diagnosis', 'Hard goal', 'Goal ')))
    lines += ['- ' + esc(v) for v in limits]
    lines += ['', '## Risks and acceptance boundaries', '']
    lines += ['- ' + esc(v) for v in report.risks]
    if report.evaluation_status == 'planning_only':
        planned = [f for f in report.findings if type(f) is PlannedFinding]
        candidates = [f for f in planned if f.kind == FindingKind.EXPERIMENT_CANDIDATE]
        lines += ['', '## What to consider next', '',
                  'These are proposals only. Nothing is authorized or executed. No improvement is guaranteed.', '']
        if not candidates:
            lines.append('No optimization experiment is sufficiently supported yet. Start with the evidence reviews below.')
        for finding in planned:
            p = finding.plan
            lines += ['', '- ' + esc(finding.statement), '  - Why: ' + esc(p.rationale),
                      '  - Decision this enables: ' + esc(p.decision),
                      '  - Type: ' + finding.kind.value + '; prerequisite stage: ' + p.stage + '; authorization: ' + p.authorization + '.']
            if p.change_parameter:
                lines.append('  - Change: ' + esc(p.change_parameter) + '; finite points must be explicitly selected.')
            for title, values in (('Hold fixed', p.fixed_conditions), ('Missing evidence', p.missing_evidence),
                                  ('Verify with', p.remeasure_metrics), ('Risks', p.risks), ('Accept only if', p.acceptance_conditions)):
                if values:
                    lines.append('  - ' + title + ': ' + esc('; '.join(values)))
            if p.goal_ids:
                lines.append('  - Related goals: ' + esc(', '.join(p.goal_ids)))
    lines += ['', '## Conditional next evidence', '']
    lines += ['- ' + esc(f.statement) for f in report.findings if f.kind == FindingKind.MEASURE_NEXT and
              (report.evaluation_status != 'planning_only' or type(f) is not PlannedFinding)]
    if not has_goal_results and (report.request.goals or report.request.budgets):
        lines += ['', 'Advanced goal/resource acceptance was not evaluated. Static consistency: ' +
                  (report.request_validation.consistency.value if report.request_validation else 'not_evaluated') + '.']
    lines += ['', 'Full per-observation limitations, exact hashes and original binding labels are retained in JSON.', '']
    raw = '\n'.join(lines)
    if path is not None:
        with Path(path).open('x', encoding='utf-8', newline='\n') as target:
            target.write(raw)
    return raw
