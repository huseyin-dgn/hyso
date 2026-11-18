"""Carry exact completed stage evidence into a new, one-shot child run.

Never resume a consumed manifest, rerun an attempted stage, or hide a failure.
Only the explicitly recognized, clean counter-permission denial is containable.
"""
from pathlib import Path
import copy
from .core import read, record, sha, verify, selected_arm, digest


def bound(ref):
    if record(ref['path'])!=ref:raise ValueError('continuation artifact changed')
    return Path(ref['path'])


def counter_block(m):
    value=m.get('known_counter_permission_block') or m.get('historical_counter_permission_block')
    if value is None:return None
    if value['gpu_uuid']!=m['hardware']['gpu_uuid']:raise ValueError('counter block device mismatch')
    text=bound(value['error_artifact']).read_text(encoding='utf-8-sig')
    if 'ERR_NVGPUCTRPERM' not in text:raise ValueError('counter permission evidence missing')
    return 'NVIDIA counter permission denied (ERR_NVGPUCTRPERM); no automatic repeat of known denied capture'


def completed_rows(m):
    c=m.get('continuation')
    if c is None:return {}
    path=bound(c['parent_manifest']);p=verify(path,expected=c['parent_manifest']['sha256'],live=False)
    state=read(bound(c['parent_terminal']));report=read(bound(c['parent_stage_results']))
    if (state.get('controller_active') is not False or state.get('preservation_status')!='PASS'
            or state.get('finalization_errors') or state.get('run_id')!=p['run_id']
            or state.get('manifest_sha256')!=sha(path)
            or report.get('manifest_sha256')!=sha(path) or report.get('run_id')!=p['run_id']):
        raise ValueError('parent execution not cleanly finalized')
    if (selected_arm(m)!=selected_arm(p) or m['assets']!=p['assets']
            or m['hardware']!=p['hardware'] or m['protocols']!=p['protocols']):
        raise ValueError('continuation model/device/assets/protocol mismatch')
    for rel,h in p['source_snapshot']['files'].items():
        if rel.startswith('hyso/') and m['source_snapshot']['files'].get(rel)!=h:
            raise ValueError('continuation production source changed')
    retry_ids=c.get('retry_ids',[])
    if retry_ids:
        if retry_ids!=['06.roofline']:raise ValueError('only explicit controlled P18 retry is supported')
        approval=read(bound(c['retry_authorization']))
        if (approval.get('parent_manifest')!=c['parent_manifest'] or approval.get('stage')!='06.roofline'
                or approval.get('gpu_uuid')!=m['hardware']['gpu_uuid'] or approval.get('maximum_new_attempts')!=1
                or not approval.get('user_request') or approval.get('automatic_retry') is not False):
            raise ValueError('controlled retry authorization mismatch')
    rows={}
    for row in report['experiments']:
        if not row.get('attempted'):continue
        ident=row['id'];e=next(e for e in p['experiments'] if e['id']==ident)
        sup=row.get('supervisor',{})
        if (not e['authorization']['approved'] or read(Path(e['output'])/'supervisor-receipt.json')!=sup
                or sup.get('manifest_sha256')!=sha(path) or sup.get('native_crash') is not False
                or sup.get('reason') is not None or sup.get('error') is not None
                or sup.get('cleanup',{}).get('active_processes')!=0):
            raise ValueError('parent stage has unsafe or unverified supervisor')
        terminal=read(Path(e['output'])/'result.json')
        if terminal.get('manifest_sha256')!=sha(path) or terminal.get('id')!=ident:
            raise ValueError('parent stage terminal binding mismatch')
        if any(row.get(k)!=terminal.get(k) for k in ('status','criteria','evidence','safety_stop','metrics')):
            raise ValueError('parent worker/controller result mismatch')
        if row.get('safety_stop'):
            if (ident!='06.roofline' or row['safety_stop']!='nested_counter_capture_failed_exit_1'
                    or counter_block(m) is None or sup.get('exit_code')!=1
                    or sup.get('resources',{}).get('low_memory_observed') is not False):
                raise ValueError('unknown unsafe parent boundary cannot be continued')
        for ref in row.get('evidence',[]):bound(ref)
        if ident in retry_ids:
            if row['status']!='FAIL' or not row.get('safety_stop') or counter_block(m) is None:
                raise ValueError('controlled P18 retry requires original permission-denied failure')
            continue
        rows[ident]=dict(copy.deepcopy(row),source_execution=dict(parent_manifest=c['parent_manifest'],
            terminal=record(Path(e['output'])/'result.json'),supervisor=record(Path(e['output'])/'supervisor-receipt.json'),
            carried_without_execution=True))
    if sorted(rows)!=sorted(c['completed_ids']):raise ValueError('completed stage set changed')
    return rows
