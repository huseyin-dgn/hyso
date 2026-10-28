"""Readable export from a single AdvisorReport, with no evidence resolution."""
from pathlib import Path
import html

from ._json import AdvisorError
from .contracts import AdvisorReport, FindingKind
from .resolver import NativeObservation


def render_markdown(report, path=None):
    if type(report) is not AdvisorReport:
        raise AdvisorError('typed_advisor_report_required')
    esc = lambda value: html.escape(str(value), quote=False).replace('\r', ' ').replace('\n', ' ')
    lines = ['# HYSo Advisor — existing evidence', '',
             'Source: `' + report.source_report_id + '`',
             'Source canonical SHA-256: `' + report.source_report_sha256 + '`',
             'Mode: ' + report.request.mode + '; priority: ' + (report.request.priority.value if report.request.priority else 'unspecified') + '.',
             'Status: ' + report.evaluation_status + '. No workload or model change was executed.', '']
    observations = [f for f in report.findings if f.kind == FindingKind.OBSERVATION]
    lines += ['## Observations', '']
    for finding in observations:
        if type(finding) is NativeObservation:
            m = finding.resolved_metric
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
    lines += ['- ' + esc(v) for v in report.evidence_limitations]
    lines += ['', '## Risks and acceptance boundaries', '']
    lines += ['- ' + esc(v) for v in report.risks]
    lines += ['', '## Conditional next evidence', '']
    lines += ['- ' + esc(f.statement) for f in report.findings if f.kind == FindingKind.MEASURE_NEXT]
    if report.request.goals or report.request.budgets:
        lines += ['', 'Advanced goal/resource acceptance was not evaluated. Static consistency: ' +
                  (report.request_validation.consistency.value if report.request_validation else 'not_evaluated') + '.']
    lines += ['', 'Full per-observation limitations, exact hashes and original binding labels are retained in JSON.', '']
    raw = '\n'.join(lines)
    if path is not None:
        with Path(path).open('x', encoding='utf-8', newline='\n') as target:
            target.write(raw)
    return raw
