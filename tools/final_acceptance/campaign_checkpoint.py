"""Per-node reconciliation. A call PASS alone is never sufficient."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path

from common import STATUSES, atomic_json, read_json, read_journal, utc


def aggregate(nodeid, phases, *, interrupted=None):
    rows = [r for r in phases if r.get('kind', 'phase') == 'phase']
    main = [r for r in rows if not r.get('is_subtest', False)]
    by_phase = {r.get('when'): r for r in main}
    failures = [r for r in rows if r.get('outcome') == 'failed']
    duration = sum(float(r.get('duration', 0)) for r in main if isinstance(r.get('duration',0),(int,float)) and math.isfinite(r.get('duration',0)))
    result = dict(nodeid=nodeid, status='NOT_RUN', reason_code='INCOMPLETE_PHASES',
                  phases=main, subtests=[r for r in rows if r.get('is_subtest') and not r.get('is_cleanup')],
                  cleanups=[r for r in rows if r.get('is_cleanup')],
                  duration_seconds=duration)
    stages=[r.get('when') for r in main]
    malformed=any(r.get('when') not in ('setup','call','teardown') or r.get('outcome') not in ('passed','failed','skipped')
        or not isinstance(r.get('duration',0),(int,float)) or not math.isfinite(r.get('duration',0)) or r.get('duration',0)<0 for r in rows)
    malformed=malformed or len(stages)!=len(set(stages)) or stages!=sorted(stages,key=lambda p:('setup','call','teardown').index(p) if p in ('setup','call','teardown') else 99)
    if malformed:
        result.update(status='ERROR',reason_code='QUARANTINED_PHASE_SCHEMA',evidence_quarantined=True)
    elif failures:
        error = failures[0]
        text = error.get('traceback', error.get('longrepr', '')) or ''
        status = 'ERROR' if error['when'] != 'call' else 'FAIL'
        reason = {'setup': 'SETUP_ERROR', 'teardown': 'TEARDOWN_ERROR'}.get(error['when'], 'ASSERTION_FAILURE')
        if 'ModuleNotFoundError' in text or 'ImportError' in text:
            reason = 'ENVIRONMENT_ERROR'
        result.update(status=status, reason_code=reason, failure=error)
    elif any(r['outcome'] == 'skipped' for r in rows):
        skipped = next(r for r in rows if r['outcome'] == 'skipped')
        result.update(status='SKIP', reason_code='EXPLICIT_TEST_SKIP', skip_reason=skipped.get('traceback', ''))
    elif all(by_phase.get(p, {}).get('outcome') == 'passed' for p in ('setup', 'call', 'teardown')):
        result.update(status='PASS', reason_code='ALL_PHASES_AND_SUBTESTS_PASSED')
    elif interrupted:
        result['reason_code'] = interrupted
        if interrupted in ('GROUP_TIMEOUT', 'NO_PROGRESS_TIMEOUT'):
            result['status'] = 'TIMEOUT'
    result['phases_complete'] = all(p in by_phase for p in ('setup', 'call', 'teardown'))
    return result


def journal_results(path, receipt=None, *, allow_verified_prefix=False,event_limit=None):
    events, issues = read_journal(path)
    if event_limit is not None:events=events[:event_limit]
    groups = defaultdict(list)
    starts = {}
    for row in events:
        if row.get('nodeid'):
            if row['kind'] == 'test_start':
                starts[row['nodeid']] = row
            if row['kind'] == 'phase':
                groups[row['nodeid']].append(row)
    results = {}
    reason = (receipt or {}).get('monitor_reason')
    for node in set(starts) | set(groups):
        result = aggregate(node, groups[node], interrupted=reason)
        header = starts.get(node) or next(iter(groups[node]), {})
        for key in ('attempt_id', 'group', 'source_commit', 'environment_id', 'test_source_sha256'):
            result[key] = header.get(key)
        result['evidence_file'] = str(Path(path))
        result['phase_evidence_sha256'] = groups[node][-1].get('sha256') if groups[node] else header.get('sha256')
        result['finished_utc']=(groups[node][-1].get('time_utc') if groups[node] else header.get('time_utc'))
        if any(i['reason']=='EVIDENCE_INVALID' for i in issues) and not allow_verified_prefix:
            result.update(status='NOT_RUN',reason_code='JOURNAL_INTEGRITY_UNRESOLVED')
        if receipt and receipt.get('source_integrity') is False:
            result.update(status='NOT_RUN', reason_code='EVIDENCE_INVALID')
        if result['status'] == 'PASS' and receipt and not receipt.get('cleanup_verified', False):
            result.update(status='NOT_RUN', reason_code='CLEANUP_UNVERIFIED')
        if result['status'] == 'PASS' and receipt and receipt.get('schema_version',1)>=2 and not receipt.get('payload_admitted'):
            result.update(status='NOT_RUN',reason_code='PAYLOAD_NOT_ADMITTED')
        results[node] = result
    return results, issues


class Checkpoint:
    def __init__(self, root, inventory, baseline):
        self.root = Path(root)
        self.path = self.root / 'checkpoint.json'
        self.data = read_json(self.path)
        if self.data is None:
            self.data = dict(schema_version=1, created_utc=utc(), baseline=baseline,
                inventory={r['nodeid']: r for r in inventory}, results={}, attempts={},
                forensic_verification_reserved=False, interruptions=[], retries=[],
                completed_operations={}, processed_stop_requests=[])
        if self.data.get('schema_version') != 1:
            raise ValueError('unsupported checkpoint schema')
        if len(self.data['inventory']) != len(set(self.data['inventory'])):
            raise ValueError('duplicate inventory')

    def save(self):
        self.data['updated_utc'] = utc()
        atomic_json(self.path, self.data)

    def states(self):
        result = {}
        for node, entry in self.data['inventory'].items():
            current = self.data['results'].get(node, dict(nodeid=node, status='NOT_RUN', reason_code='PENDING'))
            if entry.get('risk_class'):
                current = dict(nodeid=node, status='BLOCKED', reason_code=entry.get('reason_code', 'BLOCKED_ADMISSION'),
                               risk_class=entry['risk_class'], reason=entry.get('risk_reason'))
            elif node in self.data.get('inventory_holds',{}):
                hold=self.data['inventory_holds'][node]
                current=dict(current,status=hold.get('status','BLOCKED'),reason_code=hold['reason_code'],
                    previous_result_status=current['status'],collection_hold=hold)
                if hold.get('failure'):current['failure']=hold['failure']
            if current['status'] not in STATUSES:
                raise ValueError('unknown terminal status')
            result[node] = current
        return result

    def counts(self):
        counts = Counter(r['status'] for r in self.states().values())
        return {key: counts[key] for key in STATUSES}

    def pending(self):
        return [node for node, state in self.states().items() if state['status'] == 'NOT_RUN' and
                not self.data['inventory'][node].get('risk_class')]
