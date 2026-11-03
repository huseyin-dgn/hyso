"""Read-only P13 Nsight evidence. Host lexical ownership, CUPTI kernel time.

Configuration pins the export and binary report independently. A capture receipt
is an attestation from the approved supervisor, not something this reader invents
from timestamps or a successful profiler exit. See docs/guides/p13-nsys-evidence.md.
"""
import json
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
import re
import sqlite3

from ._operator_activity import _sha, _contains, _launch_owner


def _pinned(ref):
    path = Path(ref['path']).resolve()
    if not re.fullmatch('[0-9a-f]{64}', ref['sha256']) or _sha(path) != ref['sha256']:
        raise ValueError('P13 artifact SHA256 mismatch')
    return path


def analyze(configuration, *, max_rows=20000):
    """Analyze pinned artifacts without importing Torch, executing or exporting.

    Missing capture provenance yields PARTIAL even if every kernel maps. Invalid
    identities/schema/boundaries raise ValueError; incomplete attribution remains
    explicit in the returned ledgers. Caller-supplied counts are never trusted.
    """
    if type(max_rows) is not int or not 1 <= max_rows <= 100000:
        raise ValueError('invalid P13 row budget')
    try:
        return _analyze(configuration, max_rows)
    except (KeyError, TypeError, OSError, sqlite3.Error, UnicodeError) as error:
        raise ValueError('invalid or unsupported P13 evidence') from error


def _analyze(cfg, budget):
    refs = [cfg['sqlite'], cfg['report']]
    path, report = map(_pinned, refs)
    if path.stat().st_size > 256*1024**2:
        raise ValueError('P13 SQLite byte budget exceeded')
    def stable_export():
        if any(Path(str(path)+suffix).exists() for suffix in ('-wal','-shm','-journal')):
            raise ValueError('P13 requires a closed standalone SQLite export')
    stable_export()
    gp = cfg['global_pid']
    if type(gp) is not int or gp <= 0 or gp & 0xffffff:
        raise ValueError('invalid encoded P13 process identity')
    pid = (gp >> 24) & 0xffffff
    labels = cfg['forward_labels']
    if (not isinstance(labels, list) or not 1 <= len(labels) <= 100
            or any(type(v) is not str or not v for v in labels) or len(set(labels)) != len(labels)):
        raise ValueError('invalid measured forward ledger')
    prefix = 'hyso.p13.module/'
    capture = None
    if cfg.get('capture'):
        receipt = _pinned(cfg['capture']); refs.append(cfg['capture'])
        if receipt.stat().st_size > 1024**2: raise ValueError('capture receipt budget exceeded')
        def unique(pairs):
            result = {}
            for k, v in pairs:
                if k in result: raise ValueError('duplicate receipt key')
                result[k] = v
            return result
        capture = json.loads(receipt.read_text(encoding='utf-8'), object_pairs_hook=unique)
        if (any(type(capture[k]) is not int for k in ('version','global_pid','worker_pid','nsys_exit_code'))
                or type(capture['supervisor']['cleanup']['active_processes']) is not int):
            raise ValueError('invalid P13 receipt identity types')
        if (capture['format'] != 'hyso.performance.p13.nsys_capture' or capture['version'] != 1
                or capture['global_pid'] != gp or capture['worker_pid'] != pid
                or capture['forward_labels'] != labels
                or capture['sqlite_sha256'] != cfg['sqlite']['sha256']
                or capture['report_sha256'] != cfg['report']['sha256']
                or capture['status'] != 'completed' or capture['nsys_exit_code'] != 0
                or not isinstance(capture['output_equal'],list)
                or len(capture['output_equal']) != len(labels)
                or any(v is not True for v in capture['output_equal'])
                or capture['cudnn_tf32_disabled'] is not True
                or capture['authorization']['approved'] is not True
                or capture['supervisor']['owned_process_tree'] is not True
                or capture['supervisor']['process_terminated'] is not True
                or capture['supervisor']['cleanup']['active_processes'] != 0):
            raise ValueError('incomplete P13 capture provenance')
        for key in ('checkpoint', 'input', 'worker_source'):
            _pinned(capture[key]); refs.append(capture[key])
        if not capture['gpu_uuid'] or not capture['authorization']['basis']:
            raise ValueError('missing P13 device/authorization identity')
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA query_only=ON')
        steps = [0]
        def progress():
            steps[0] += 1
            return int(steps[0] > 20000)
        db.set_progress_handler(progress, 1000)
        def rows(table, columns, where=''):
            if not set(columns) <= {r[1] for r in db.execute(f'PRAGMA table_info({table})')}:
                raise ValueError('missing P13 schema: ' + table)
            result = [dict(r) for r in db.execute(f'SELECT rowid AS evidence_id,{",".join(columns)} FROM {table} {where} LIMIT ?', (budget+1,))]
            if len(result) > budget: raise ValueError('P13 evidence row budget exceeded')
            return result
        meta = {r['name']: r['value'] for r in rows('META_DATA_EXPORT', ['name','value'],
                "WHERE name IN ('EXPORT_PRODUCT_NAME','EXPORT_PRODUCT_VERSION','EXPORT_SCHEMA_VERSION')")}
        if meta.get('EXPORT_PRODUCT_NAME') != 'NVIDIA Nsight Systems' or not meta.get('EXPORT_SCHEMA_VERSION','').startswith('3.'):
            raise ValueError('unsupported P13 Nsight schema')
        names_rows = rows('StringIds', ['id','value'])
        names = {r['id']: r['value'] for r in names_rows}
        if len(names) != len(names_rows): raise ValueError('duplicate string identity')
        diagnostics = rows('DIAGNOSTIC_EVENT', ['severity','text','globalPid'])
        target_warnings = [r for r in diagnostics if r['severity'] > 1
                           and (r['globalPid'] in (gp,None,0) or r['globalPid'] < 0)]
        for message in ('CUDA injection initialized successfully.', 'NVTX injection initialized successfully.'):
            if not any(r['globalPid']==gp and message in r['text'] for r in diagnostics):
                raise ValueError('missing target-process injection evidence')
        ranges = rows('NVTX_EVENTS', ['start','end','text','textId','globalTid','endGlobalTid','domainId','eventType'])
        ranges = [dict(r, label=r['text'] or names.get(r['textId'],'')) for r in ranges
                  if type(r['globalTid']) is int and r['globalTid'] & ~0xffffff == gp]
        def interval(r):
            if type(r['start']) is not int or type(r['end']) is not int or r['start'] < 0 or r['end'] <= r['start']:
                raise ValueError('incomplete or invalid P13 interval')
        phases = []
        for i, label in enumerate(labels):
            matches = [r for r in ranges if r['label'] == label]
            if len(matches) != 1: raise ValueError('missing or duplicate P13 forward')
            p = matches[0]; interval(p)
            if p['eventType'] != 59 or phases and p['start'] < phases[-1]['end']:
                raise ValueError('unsupported or overlapping P13 forwards')
            if p['endGlobalTid'] not in (None,p['globalTid']):
                raise ValueError('cross-thread P13 forward range')
            phases.append(dict(p, forward_index=i))
        modules = []
        captured_modules = [r for r in ranges if r['label'].startswith(prefix)]
        for r in ranges:
            if not r['label'].startswith(prefix): continue
            interval(r)
            if r['eventType'] != 59 or not r['label'][len(prefix):]:
                raise ValueError('unsupported P13 module range')
            overlapping = [p for p in phases if r['start'] < p['end'] and p['start'] < r['end']]
            if not overlapping: continue
            if len(overlapping) != 1 or not _contains(overlapping[0], r) or r['globalTid'] != overlapping[0]['globalTid']:
                raise ValueError('module crosses P13 forward boundary/thread')
            if r['domainId'] != overlapping[0]['domainId'] or r['endGlobalTid'] not in (None,r['globalTid']):
                raise ValueError('unsupported cross-domain/thread P13 module range')
            modules.append(dict(r, nvtx_id=r['evidence_id'], identity_status='recorded',
                                module_path=r['label'][len(prefix):], forward_index=overlapping[0]['forward_index']))
        # Validate the entire module stack, including ranges with no direct launch.
        for p in phases:
            stack = []
            selected = sorted((m for m in modules if m['forward_index'] == p['forward_index']), key=lambda r:(r['start'],-r['end']))
            for m in selected:
                while stack and stack[-1]['end'] <= m['start']: stack.pop()
                if stack and (not _contains(stack[-1],m) or (stack[-1]['start'],stack[-1]['end']) == (m['start'],m['end'])):
                    raise ValueError('ambiguous or crossing P13 module nesting')
                stack.append(m)
        launches = []
        tables = ['CUPTI_ACTIVITY_KIND_RUNTIME']
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='CUPTI_ACTIVITY_KIND_DRIVER'").fetchone():
            tables.append('CUPTI_ACTIVITY_KIND_DRIVER')
        for table in tables:
            for r in rows(table, ['start','end','globalTid','correlationId','nameId','returnValue']):
                if type(r['globalTid']) is not int or r['globalTid'] & ~0xffffff != gp: continue
                name = names.get(r['nameId'],'')
                # Graph/cooperative launches are visible but deliberately unsupported.
                if 'Launch' not in name: continue
                interval(r)
                ps = [p for p in phases if r['start'] < p['end'] and p['start'] < r['end']]
                if ps and (len(ps) != 1 or not _contains(ps[0],r) or r['globalTid'] != ps[0]['globalTid']):
                    raise ValueError('launch crosses P13 forward boundary/thread')
                launches.append(dict(r, table=table, name=name, forward_index=ps[0]['forward_index'] if ps else None))
        contexts = rows('TARGET_INFO_CUDA_CONTEXT_INFO', ['processId','deviceId','contextId'])
        if len(modules)*len(launches) > 5000000:
            raise ValueError('P13 ownership analysis complexity budget exceeded')
        by_correlation = defaultdict(list)
        for a in launches: by_correlation[a['correlationId']].append(a)
        kernels = []; outside = []; unresolved_outside = []; seen = set()
        for k in rows('CUPTI_ACTIVITY_KIND_KERNEL', ['start','end','deviceId','contextId','streamId','correlationId','globalPid','shortName']):
            if k['globalPid'] != gp: continue
            interval(k)
            if (any(type(k[v]) is not int or k[v] < 0 for v in ('deviceId','contextId','streamId'))
                    or k['shortName'] not in names):
                raise ValueError('invalid P13 kernel metadata')
            if not any(c['processId']==pid and c['deviceId']==k['deviceId'] and c['contextId']==k['contextId'] for c in contexts):
                raise ValueError('P13 kernel context identity missing')
            key = tuple(k[v] for v in k if v != 'evidence_id')
            if key in seen: raise ValueError('duplicate P13 kernel')
            seen.add(key)
            ps = [p for p in phases if k['start'] < p['end'] and p['start'] < k['end']]
            candidates = by_correlation[k['correlationId']] if type(k['correlationId']) is int and k['correlationId'] > 0 else []
            launch = candidates[0] if len(candidates)==1 else None
            if not ps and (launch is None or launch['forward_index'] is None):
                outside.append(k)
                if launch is None or launch['returnValue'] != 0:
                    unresolved_outside.append(k)
                continue
            owner = None; chain = []; reason = 'missing_or_ambiguous_launch_correlation'
            if len(ps) != 1 or not _contains(ps[0],k): reason = 'kernel_crosses_or_escapes_forward'
            elif launch is not None:
                if launch['forward_index'] != ps[0]['forward_index']: reason = 'launch_forward_mismatch'
                elif launch['returnValue'] != 0: reason = 'failed_launch'
                elif not re.fullmatch(r'(?:cudaLaunchKernel|cuLaunchKernel)(?:_v\d+)?',launch['name']): reason = 'unsupported_launch_API'
                elif launch['start'] > k['start']: reason = 'kernel_precedes_launch'
                else:
                    owner, chain, reason = _launch_owner(modules,launch)
            kernels.append(dict(k, module_nvtx_id=owner, module_chain_nvtx_ids=chain,
                launch_table=launch['table'] if launch else None, launch_id=launch['evidence_id'] if launch else None,
                forward_index=ps[0]['forward_index'] if len(ps)==1 else None,
                duration_us=(k['end']-k['start'])/1000, reason=reason, status='mapped' if reason is None else 'unsupported'))
        used = {(k['launch_table'],k['launch_id']) for k in kernels}
        cardinality = Counter((k['launch_table'],k['launch_id']) for k in kernels)
        for k in kernels:
            if k['launch_id'] is not None and cardinality[k['launch_table'],k['launch_id']] != 1:
                k.update(module_nvtx_id=None,module_chain_nvtx_ids=[],status='unsupported',reason='ambiguous_launch_kernel_cardinality')
        missing = [a for a in launches if a['forward_index'] is not None and (a['table'],a['evidence_id']) not in used]
    for ref in refs: _pinned(ref)
    stable_export()
    mapped = sum(k['status']=='mapped' for k in kernels)
    attribution_complete = (bool(kernels) and mapped == len(kernels) and not missing and not target_warnings and not unresolved_outside
                            and all(any(k['forward_index']==p['forward_index'] for k in kernels) for p in phases))
    complete = attribution_complete and capture is not None
    return dict(format='hyso.performance.layer_profile.nsys', schema_version=1, feature_ids=['P13'],
        status='complete' if complete else 'partial', attribution_complete=attribution_complete,
        provenance_verified=capture is not None, capture=capture, artifacts=refs, export_metadata=meta,
        global_pid=gp, worker_pid=pid, forwards=phases, modules=modules, kernels=kernels,
        captured_module_range_count=len(captured_modules),
        captured_distinct_module_count=len({r['label'][len(prefix):] for r in captured_modules}),
        measured_kernel_count=len(kernels), mapped_kernel_count=mapped,
        out_of_scope_kernel_count=len(outside), out_of_scope_kernels=outside,
        unresolved_out_of_scope_kernels=unresolved_outside,
        missing_kernel_launches=missing, target_warnings=target_warnings,
        limitations=['Observed synchronized forward scopes only; out-of-scope kernels excluded from coverage.',
                     'CUPTI duration only; NVTX is lexical host ownership, never kernel time.',
                     'No CPU ownership, P14 operator attribution, baseline latency or perturbation claim.',
                     'Binary report/export relationship is attested by the pinned capture receipt; binary decoding is not performed.'])
