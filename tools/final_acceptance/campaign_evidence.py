"""One acceptance boundary for durable events, request bindings and owned receipts."""
from __future__ import annotations

from collections import Counter,defaultdict
from pathlib import Path
import math

from common import REPO,ROOT,digest,file_hash,read_json,read_journal,new_json,utc
from campaign_checkpoint import journal_results
from campaign_collection import canonical_nodeid,collection_dependencies
from campaign_preflight import same_process

POLICY_VERSION=3


def terminal_phases(value):
    phases=value.get('phases',[])
    return bool(value.get('phases_complete') or (
        any(r.get('when')=='setup' and r.get('outcome') in ('failed','skipped') for r in phases)
        and any(r.get('when')=='teardown' for r in phases)))


def provisional_progress(cp,attempt_id=None):
    pending=set(cp.pending())
    rows=[v for n,v in cp.data.get('provisional_results',{}).items()
        if n in pending and (attempt_id is None or v.get('attempt_id')==attempt_id)
        and v.get('provisional_until_cleanup') and terminal_phases(v) and not v.get('evidence_quarantined')
        and not v.get('provisional_evidence_warning')]
    counts=Counter(v['status'] for v in rows)
    return dict(provisional_source_counts=dict(counts),provisional_completed=len(rows),
        provisional_new_PASS=counts['PASS'],provisional_remaining_tests=max(0,len(pending)-len(rows)),
        definitive_remaining_tests=len(pending),recovered_from_prior_attempts=cp.data.get('recovered_outcomes',{}),
        acceptance_note='Provisional results are not finalized PASS; final acceptance requires bound receipt and owned cleanup.')


def stream_audit(path):
    records,issues=read_journal(path)
    result=dict(records=records,issues=issues,scope='VALID',reason=None,
        verified_prefix_events=len(records),verified_prefix_head=records[-1]['sha256'] if records else '0'*64)
    if not issues:return result
    if all(i['reason']=='INTERRUPTED_LAST_WRITE' for i in issues):
        result.update(scope='LOCAL',reason='INTERRUPTED_LAST_WRITE')
        return result
    # Exact repair policy for the recorded schema-1 fsync instrumentation failure.
    # Do not bridge the fork, recompute old records, or accept anything in its tail.
    try:
        import json
        all_rows=[json.loads(line) for line in Path(path).read_bytes().splitlines()]
        tail=all_rows[len(records):]
        known=bool(len(records)>1 and len(tail)==3
            and [r['kind'] for r in tail]==['framework_error','session_finish','framework_error']
            and records[-1].get('schema_version')==1
            and records[-1].get('nodeid','').endswith('test_ingest_failures_block_completion_preserve_source[fsync]')
            and records[-1].get('when')=='call' and records[-1].get('outcome')=='passed')
        for index,row in enumerate(tail,len(records)+1):
            body={k:v for k,v in row.items() if k!='sha256'}
            known=known and row.get('schema_version')==1 and row['sequence']==index
            known=known and row['sha256']==digest(body) and row['previous_sha256']==records[-1]['previous_sha256']
            known=known and all(row.get(k)==records[-1].get(k) for k in ('attempt_id','group','source_commit','environment_id'))
        for row in (tail[0],tail[-1]):
            trace=row.get('traceback','')
            known=known and all(s in trace for s in ('PermissionError','common.py','os.fsync(out.fileno())','test_experiment_artifacts.py'))
        if known:
            result.update(scope='LOCAL',reason='QUARANTINED_LEGACY_FSYNC_WRITER_TAIL',quarantined_tail_events=len(tail))
            return result
    except (ValueError,KeyError,IndexError,TypeError):
        pass
    result.update(scope='GLOBAL',reason='UNEXPLAINED_JOURNAL_INTEGRITY_VIOLATION')
    return result


def reconcile_attempt(folder, *, baseline=None, receipt=None, provisional=False):
    folder=Path(folder);request=read_json(folder/'request.json',{})
    receipt=receipt or read_json(folder/'controller.json')
    stream=stream_audit(folder/'events.jsonl')
    global_errors=[];local=[]
    if stream['scope']=='GLOBAL':global_errors.append(stream['reason'])
    if stream['scope']=='LOCAL':local.append(dict(reason=stream['reason'],issues=stream['issues']))
    records=stream['records'];inventory=request.get('inventory',{})
    event_limit=None
    writer=read_json(folder/'journal-terminal.json')
    if writer:
        count=writer.get('committed_sequence')
        if (type(count) is not int or count<0 or count>len(records)
            or writer.get('committed_head_sha256')!=(records[count-1]['sha256'] if count else '0'*64)
            or writer.get('attempt_id')!=request.get('attempt_id') or not writer.get('closed')):
            global_errors.append('WRITER_TERMINAL_MISMATCH')
        elif writer.get('writer_failed'):
            event_limit=count
            local.append(dict(reason='QUARANTINED_WRITER_IO_FAILURE',uncommitted_events=len(records)-count))
    elif receipt and request.get('evidence_protocol')=='hyso.event-journal.v2':
        local.append(dict(reason='WRITER_TERMINAL_UNAVAILABLE',parent_sealed=bool(receipt.get('terminal_evidence_sha256'))))
        if not receipt.get('monitor_reason'):
            starts_at=[i for i,r in enumerate(records) if r['kind']=='test_start']
            if starts_at:event_limit=starts_at[-1]
    values,issues=journal_results(folder/'events.jsonl',receipt,allow_verified_prefix=stream['scope']!='GLOBAL' or provisional,event_limit=event_limit)
    evidence_sha=file_hash(folder/'events.jsonl') if (folder/'events.jsonl').is_file() else None
    selected=set(request.get('selected_nodes',[]))
    blocked=set(request.get('blocked_nodes',[]))
    identity={k:request.get(k) for k in ('attempt_id','source_commit','environment_id','group')}
    if receipt:
        if receipt.get('request_sha256') and receipt['request_sha256']!=file_hash(folder/'request.json'):
            global_errors.append('REQUEST_HASH_MISMATCH')
        if receipt.get('bootstrap_sha256') and (not (folder/'bootstrap.py').is_file()
                or file_hash(folder/'bootstrap.py')!=receipt['bootstrap_sha256']):
            global_errors.append('BOOTSTRAP_BINDING_MISMATCH')
        if receipt.get('journal_sha256') and receipt['journal_sha256']!=file_hash(folder/'events.jsonl'):
            global_errors.append('JOURNAL_FILE_HASH_MISMATCH')
        if receipt.get('worker_terminal_sha256') and (not (folder/'journal-terminal.json').is_file()
                or file_hash(folder/'journal-terminal.json')!=receipt['worker_terminal_sha256']):
            global_errors.append('WRITER_TERMINAL_HASH_MISMATCH')
        if receipt.get('source_integrity') is not True:global_errors.append('SOURCE_INTEGRITY_UNVERIFIED')
        cleanup=receipt.get('cleanup') or {}
        recovery=receipt.get('cleanup_proof') or {}
        dead_owner=bool(recovery.get('kind')=='OWNER_DEATH_KILL_ON_CLOSE'
            and recovery.get('kill_on_close') and recovery.get('no_breakaway') and recovery.get('identities')
            and all(identity and not same_process(identity) for identity in recovery['identities']))
        if not (receipt.get('cleanup_verified') is True and (cleanup.get('active_processes')==0 or dead_owner)):
            global_errors.append('OWNED_CLEANUP_UNVERIFIED')
        if not receipt.get('payload_admitted'):global_errors.append('PAYLOAD_ADMISSION_UNVERIFIED')
        limits=receipt.get('limits',{})
        if limits!={'process_bytes':3*1024**3,'job_bytes':6*1024**3,'max_processes':48}:
            global_errors.append('RESOURCE_LIMIT_RECEIPT_MISMATCH')
        identities=receipt.get('identities') or {}
        launch=receipt.get('launch_binding') or {}
        admission=read_json(folder/'admission-receipt.json',{})
        admit=read_json(folder/'admit.json',{})
        ready=read_json(folder/'interpreter-ready.json',{})
        if (not identities.get('exact_job_membership') or identities!=admission.get('identities')
            or identities!=read_json(folder/'owner-identities.json')
            or identities.get('launcher')!=launch.get('identity')
            or identities.get('nonce_sha256')!=digest(request.get('token'))
            or ready.get('token')!=request.get('token') or admit.get('token')!=request.get('token')
            or ready.get('pid')!=identities.get('interpreter',{}).get('pid')
            or admit.get('pid')!=ready.get('pid') or not admit.get('readback_verified')
            or ready.get('parent_pid')!=identities.get('parent_pid')
            or (ready.get('pid')!=identities.get('launcher',{}).get('pid')
                and ready.get('parent_pid')!=identities.get('launcher',{}).get('pid'))
            or (ready.get('pid')==identities.get('launcher',{}).get('pid')
                and ready.get('parent_pid')!=read_json(folder/'launch-receipt.json',{}).get('owner',{}).get('pid'))):
            global_errors.append('PROCESS_IDENTITY_OR_ADMISSION_MISMATCH')
        terminal_hash=receipt.get('terminal_evidence_sha256')
        if terminal_hash:
            path=folder/'terminal-evidence.json'
            terminal=read_json(path,{})
            if (not path.is_file() or file_hash(path)!=terminal_hash
                or terminal.get('journal_sha256')!=file_hash(folder/'events.jsonl')
                or terminal.get('verified_prefix_head')!=stream['verified_prefix_head']
                or terminal.get('verified_prefix_events')!=stream['verified_prefix_events']):
                global_errors.append('TERMINAL_EVIDENCE_MISMATCH')
        for row in records:
            if row['kind']=='worker_start' and (row.get('pid')!=identities.get('interpreter',{}).get('pid')
                    or row.get('profile')!=request.get('profile') or any(row.get(k)!=v for k,v in identity.items())):
                global_errors.append('WORKER_IDENTITY_CONTEXT_MISMATCH')
    elif not provisional:
        global_errors.append('CONTROLLER_RECEIPT_MISSING')
    selection=read_json(folder/'selection.json',{})
    actual_selected=set(selection.get('selected',[]))
    if not actual_selected<=selected:global_errors.append('UNREVIEWED_SELECTION_EXECUTED')
    if baseline:
        if request.get('source_commit')!=baseline['commit']:global_errors.append('SOURCE_COMMIT_MISMATCH')
        if request.get('collection_dependencies_sha256')!=collection_dependencies(baseline['source_sha256']):
            global_errors.append('PRODUCT_DEPENDENCY_SNAPSHOT_CHANGED')
    runtimes=[r for r in records if r['kind']=='runtime']
    if request.get('profile')=='cpu' and request.get('mode') in ('tests','collect') and records and not any(
            r.get('torch_version')=='2.5.1+cpu' and r.get('cuda_initialized') is False for r in runtimes):
        global_errors.append('CPU_RUNTIME_UNVERIFIED')
    sessions=[r for r in records if r['kind']=='session_finish']
    if any(r.get('cuda_initialized') is True for r in sessions):global_errors.append('UNAUTHORIZED_CUDA_INITIALIZATION')
    if request.get('profile') in ('stdlib','infrastructure') and any(r.get('torch_imported') for r in sessions):
        global_errors.append('TORCH_FREE_PROFILE_VIOLATION')
    starts=Counter(r['nodeid'] for r in records if r['kind']=='test_start')
    node_rows=defaultdict(list)
    for row in records:
        if row.get('nodeid'):node_rows[row['nodeid']].append(row)
    for node,value in values.items():
        rows=node_rows[node]
        if node not in selected or node not in actual_selected or node in blocked or node not in inventory:
            global_errors.append('TEST_EXECUTED_OUTSIDE_APPROVED_SELECTION')
            continue
        expected_sha=inventory[node]['test_source_sha256']
        bad=starts[node]!=1 or any(any(r.get(k)!=v for k,v in identity.items()) for r in rows)
        bad=bad or any(r.get('test_source_sha256')!=expected_sha for r in rows)
        bad=bad or value.get('evidence_quarantined',False)
        bad=bad or any(r.get('raw_nodeid') and canonical_nodeid(r['raw_nodeid'])!=node for r in rows)
        if baseline and baseline['source_sha256'].get(inventory[node]['path'])!=expected_sha:
            global_errors.append('TEST_SOURCE_SNAPSHOT_MISMATCH')
        if bad:
            value.update(status='ERROR',reason_code='QUARANTINED_NODE_EVIDENCE',evidence_quarantined=True)
            local.append(dict(nodeid=node,reason='QUARANTINED_NODE_EVIDENCE'))
        value.update(provisional_until_cleanup=provisional,controller_evidence=str(folder/'controller.json'),
            evidence_sha256=evidence_sha,evidence_prefix_head=stream['verified_prefix_head'])
    if global_errors and not provisional:
        for value in values.values():value.update(status='NOT_RUN',reason_code='EVIDENCE_INVALID')
    return dict(results=values,stream={k:v for k,v in stream.items() if k!='records'},events=records,
        global_errors=sorted(set(global_errors)),local_issues=local,request=request,receipt=receipt)


def accept_attempt(cp,folder,*,receipt=None,recovered=False):
    """Materialize final source outcomes only through the same bound proof gate."""
    folder=Path(folder)
    audit=reconcile_attempt(folder,baseline=cp.data['baseline'],receipt=receipt)
    request=audit['request'];attempt_id=request['attempt_id']
    summary=dict(policy_version=POLICY_VERSION,attempt_id=attempt_id,generated_utc=utc(),
        journal_sha256=file_hash(folder/'events.jsonl'),stream=audit['stream'],
        global_errors=audit['global_errors'],local_issues=audit['local_issues'],recovered=recovered,
        accepted_counts={},accepted_nodeids=[],newly_accepted_nodeids=[],incomplete_nodeids=[])
    target='infrastructure_results' if request['profile']=='infrastructure' else 'results'
    outcomes=cp.data.setdefault(target,{})
    if not audit['global_errors']:
        for node,value in audit['results'].items():
            if node in cp.data.get('retired_tests',{}):
                continue  # Historic events remain valid; retired failures never re-enter active results.
            if value.get('evidence_quarantined'):
                cp.data.setdefault('inventory_holds',{})[node]=dict(nodeid=node,
                    reason_code='QUARANTINED_NODE_EVIDENCE',evidence=str(folder/'events.jsonl'))
                continue
            terminal=terminal_phases(value)
            if value['status']!='NOT_RUN' and terminal:
                previous=outcomes.get(node,{})
                if previous.get('status')!=value['status'] or previous.get('evidence_sha256')!=value['evidence_sha256']:
                    summary['newly_accepted_nodeids'].append(node)
                value.update(provisional_until_cleanup=False,accepted_utc=previous.get('accepted_utc') or utc(),
                    recovered_from_verified_prefix=recovered)
                outcomes[node]=value
                summary['accepted_nodeids'].append(node)
                cp.data.setdefault('provisional_results',{}).pop(node,None)
            else:
                summary['incomplete_nodeids'].append(node)
                if outcomes.get(node,{}).get('status','NOT_RUN')=='NOT_RUN':
                    outcomes[node]=dict(value,status='NOT_RUN',reason_code='INCOMPLETE_PHASES',resume_incomplete=True)
        if any(i['reason'] in ('QUARANTINED_WRITER_IO_FAILURE','WRITER_TERMINAL_UNAVAILABLE') for i in audit['local_issues']):
            for node in summary['incomplete_nodeids']:
                cp.data.setdefault('inventory_holds',{})[node]=dict(nodeid=node,
                    reason_code='QUARANTINED_OBSERVER_IO',evidence=str(folder/'events.jsonl'))
    summary['accepted_counts']=dict(Counter(outcomes[n]['status'] for n in summary['accepted_nodeids']))
    summary['newly_accepted_counts']=dict(Counter(outcomes[n]['status'] for n in summary['newly_accepted_nodeids']))
    path=ROOT/'attempt-reconciliation'/(attempt_id+'-'+summary['journal_sha256'][:12]+'-v'+str(POLICY_VERSION)+'.json')
    if not path.exists():new_json(path,summary)
    cp.data.setdefault('attempt_acceptance',{})[attempt_id]=dict(summary,evidence=str(path))
    if audit['local_issues']:
        cp.data.setdefault('evidence_quarantine',{})[attempt_id]=dict(evidence=str(path),issues=audit['local_issues'])
    cp.save()
    if audit['global_errors']:
        raise RuntimeError('EVIDENCE_INVALID: '+', '.join(audit['global_errors'])+'; '+str(path))
    return audit,summary


def recover_completed_attempts(cp):
    recovered=Counter()
    for attempt_id,attempt in cp.data['attempts'].items():
        if attempt.get('mode')!='tests' or attempt.get('profile') not in ('cpu','stdlib'):
            continue
        folder=Path(attempt['directory'])
        if not (folder/'events.jsonl').is_file() or not ((folder/'controller.json').is_file() or attempt.get('recovery')):continue
        sha=file_hash(folder/'events.jsonl')
        previous=cp.data.get('attempt_acceptance',{}).get(attempt_id,{})
        if previous.get('journal_sha256')==sha and previous.get('policy_version')==POLICY_VERSION and not previous.get('global_errors'):
            continue
        if attempt.get('journal_sha256') and attempt['journal_sha256']!=sha:
            raise RuntimeError('EVIDENCE_INVALID: existing raw attempt changed')
        audit,summary=accept_attempt(cp,folder,receipt=attempt.get('recovery') if not (folder/'controller.json').is_file() else None,recovered=True)
        recovered.update(summary['newly_accepted_counts'])
    nodes={n for a in cp.data.get('attempt_acceptance',{}).values() if a.get('recovered') for n in a['accepted_nodeids']}
    cp.data['recovered_outcomes']=dict(Counter(cp.data['results'][n]['status'] for n in nodes
        if cp.data['results'].get(n,{}).get('recovered_from_verified_prefix')))
    cp.save()
    return dict(recovered)
