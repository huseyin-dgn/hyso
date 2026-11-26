"""Opaque pytest identities, bounded inventory migration and fail-closed quarantine.

No pytest imports or test execution. Parameter IDs are never decoded, case folded,
trimmed or path-normalized. Only the path before the first :: is normalized.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from common import REPO, ROOT, atomic_json, digest, file_hash, new_json, read_json, read_journal, utc


def collection_dependencies(snapshot):
    return digest({n:sha for n,sha in snapshot.items()
        if not n.startswith('tools/final_acceptance/') and n!='run_final_acceptance.ps1'})


def canonical_nodeid(node):
    if not isinstance(node,str) or not node:
        raise ValueError('EVIDENCE_INVALID: empty/non-string node identity')
    path,separator,opaque=node.partition('::')
    path=path.replace('\\','/')
    if path.startswith('/') or ':' in path or any(p in ('','..','.') for p in path.split('/')) or '\x00' in path:
        raise ValueError('EVIDENCE_INVALID: unsafe node path')
    return path+separator+opaque


def profile_for(node):
    if node.startswith(('package:wheel::','package:sdist::')):
        return 'package'
    path=canonical_nodeid(node).split('::',1)[0]
    if path.startswith('tools/final_acceptance/tests/'):
        return 'infrastructure'
    if path.startswith('tests/forensics/'):
        return 'stdlib'
    if path=='tools/full_acceptance/tests/test_cuda_dispatch.py':return 'stdlib'
    if path.startswith(('tests/','tools/full_acceptance/tests/','tools/p31/')):
        return 'cpu'
    return 'quarantine'


def migrate_nodeids(cp, prepared, *, reviewed_changes=None):
    """Invert only the documented old corruption of the sealed original manifest."""
    wanted={r['nodeid']:r for r in prepared}
    changes=reviewed_changes or {}
    approved_additions={r['nodeid'] for r in changes.get('additions',[])}
    for rule in changes.get('retirements',[]):
        node=rule['nodeid'];path=node.split('::')[0]
        if node in wanted or file_hash(REPO/path)!=rule['source_after_sha256']:
            raise RuntimeError('EVIDENCE_INVALID: retired test source differs from reviewed removal')
        if node in cp.data['inventory']:
            entry=cp.data['inventory'][node];previous=cp.data['results'].get(node,{})
            if entry.get('risk_class') or entry['test_source_sha256']!=rule['source_before_sha256'] or previous.get('status')!='FAIL':
                raise RuntimeError('EVIDENCE_INVALID: retirement does not match original failed test')
            archived=dict(nodeid=node,status='RETIRED',previous_status=previous['status'],
                reason=rule['reason'],policy=rule,original_inventory=entry,original_result=previous,retired_utc=utc())
            cp.data.setdefault('retired_tests',{})[node]=archived
            new_json(ROOT/'retired-history'/('retired-'+digest(archived)+'.json'),archived)
            for bucket in ('inventory','results','provisional_results','inventory_holds'):
                cp.data.get(bucket,{}).pop(node,None)
        elif node in cp.data.get('retired_tests',{}) and cp.data['retired_tests'][node].get('policy')!=rule:
            raise RuntimeError('EVIDENCE_INVALID: missing historical retirement record')
    aliases=defaultdict(list)
    for node in wanted:
        aliases[node.replace('\\','/')].append(node)
    mapping={}
    for old,entry in cp.data['inventory'].items():
        if old in wanted:
            continue
        options=aliases.get(old,[])
        if len(options)!=1:
            raise RuntimeError('EVIDENCE_INVALID: ambiguous/unreviewed checkpoint identity transition')
        node=options[0]; current=wanted[node]
        if any(entry.get(k)!=current.get(k) for k in ('path','profile','test_source_sha256','risk_class')):
            raise RuntimeError('EVIDENCE_INVALID: identity alias changed source, profile or risk')
        mapping[old]=node
    added=set(wanted)-{mapping.get(n,n) for n in cp.data['inventory']}
    if not added<=approved_additions or len(cp.data['inventory'])+len(added)!=len(wanted) or len(set(mapping.values()))!=len(mapping):
        raise RuntimeError('EVIDENCE_INVALID: inventory migration changed or merged identities')
    if mapping or added:
        for bucket in ('results','provisional_results','inventory_holds'):
            values=cp.data.get(bucket,{})
            for old,node in mapping.items():
                if old in values:
                    if node in values:raise RuntimeError('EVIDENCE_INVALID: checkpoint result collision')
                    value=dict(values.pop(old),nodeid=node,checkpoint_nodeid_before_migration=old)
                    values[node]=value
        cp.data['inventory']=wanted
        cp.data.setdefault('nodeid_migrations',[]).append(dict(time_utc=utc(),aliases=mapping,
            reason='Only frozen manifest identities inverted; raw evidence and parameter strings untouched'))
    cp.data['nodeid_schema_version']=2
    return mapping


def quarantine_reason(row):
    node=row['nodeid'].lower()
    markers={m.lower() for m in row.get('markers',[])}
    if markers & {'cuda','gpu','native','stress','failfast','slow'} or any(
            word in node for word in ('actual_cuda','actual_native','failfast','allocation_stress','7200')):
        return 'BLOCKED_ADMISSION', 'POTENTIALLY_RISKY_UNREVIEWED'
    return 'BLOCKED_UNREVIEWED_NODE', 'UNREVIEWED_NO_AUTOMATIC_EXECUTION'


def discover_unreviewed_files(cp,roots):
    """Names/hashes only. Unknown files are never imported by source collection."""
    known={r['path'] for r in cp.data['inventory'].values()}
    known.update(n for n in cp.data['baseline']['source_sha256'] if Path(n).name=='conftest.py')
    q=cp.data.setdefault('quarantined_inventory',{})
    for name in roots:
        base=REPO/name
        for path in {*base.rglob('test*.py'),*base.rglob('conftest.py')}:
            relative=path.relative_to(REPO).as_posix()
            if relative in known:continue
            node=relative+'::[UNREVIEWED_FILE]'
            safe_path=not path.is_symlink() and path.resolve().is_relative_to(REPO.resolve())
            q['file::'+node]=dict(nodeid=node,path=relative,status='BLOCKED',
                reason_code='BLOCKED_UNREVIEWED_FILE',owner_profile=profile_for(node),
                risk_class='UNREVIEWED_NO_AUTOMATIC_IMPORT',
                test_source_sha256=file_hash(path) if safe_path and path.stat().st_size<=1024**2 else None,
                evidence='Static filename discovery; no import, collection or execution')


def reconcile_collection(cp, rows, profile, *, evidence, expected_nodes=None, errors=None):
    """Known exact nodes stay eligible; unknowns/duplicates/missing nodes are held locally."""
    if profile not in ('cpu','stdlib','infrastructure'):
        raise ValueError('invalid collection profile')
    inventory=cp.data['infrastructure_inventory'] if profile=='infrastructure' else cp.data['inventory']
    expected=set(expected_nodes) if expected_nodes is not None else {n for n,r in inventory.items() if r['profile']==profile}
    normalized=[]
    for original in rows:
        row=dict(original)
        node=row['nodeid']
        row['raw_nodeid']=row.get('raw_nodeid',node)
        row['nodeid']=node if node.startswith('package:') else canonical_nodeid(node)
        if row.get('path') and row['path'].replace('\\','/')!=row['nodeid'].split('::',1)[0]:
            raise RuntimeError('EVIDENCE_INVALID: collection path disagrees with node identity')
        normalized.append(row)
    counts=Counter(r['nodeid'] for r in normalized)
    duplicates={n:c for n,c in counts.items() if c>1}
    found=set(counts)
    unknown=sorted(found-set(inventory))
    missing=sorted(expected-found)
    foreign=[]; quarantined=[]
    holds=cp.data.setdefault('inventory_holds',{})
    q=cp.data.setdefault('quarantined_inventory',{})
    observed_files={}
    for row in normalized:
        node=row['nodeid']; actual_profile=profile_for(node)
        if node in inventory:
            entry=inventory[node]
            captured=row.get('test_source_sha256')
            if entry['path'] not in observed_files:observed_files[entry['path']]=file_hash(REPO/entry['path'])
            current=observed_files[entry['path']]
            if current!=entry['test_source_sha256'] or captured and captured!=current:
                raise RuntimeError('EVIDENCE_INVALID: known test source hash changed')
        if actual_profile!=profile:
            foreign.append(dict(nodeid=node,actual_profile=actual_profile,collected_in=profile))
            # Routing never authorizes execution through the wrong runtime profile.
            if node in expected:
                holds[node]=dict(nodeid=node,reason_code='BLOCKED_PROFILE_MISMATCH',evidence=evidence)
        if node in duplicates or node not in inventory:
            reason,risk=quarantine_reason(row)
            if node in duplicates:reason='BLOCKED_DUPLICATE_NODE'
            q[profile+'::'+node]=dict(row,status='BLOCKED',reason_code=reason,risk_class=risk,
                owner_profile=actual_profile,collected_profile=profile,evidence=evidence)
            quarantined.append(node)
            if node in expected:holds[node]=dict(nodeid=node,reason_code=reason,evidence=evidence)
        elif node in expected and actual_profile==profile:
            holds.pop(node,None)
    for node in missing:
        related_errors=[e for e in errors or [] if not e.get('collection_nodeid')
            or e['collection_nodeid'].split('::',1)[0].replace('\\','/')==node.split('::',1)[0]]
        holds[node]=dict(nodeid=node,reason_code='BLOCKED_COLLECTION_MISSING',evidence=evidence,
            collection_errors=related_errors,previous_status=cp.data.get('results',{}).get(node,{}).get('status'))
        if related_errors:
            holds[node].update(status='ERROR',reason_code='SETUP_ERROR',
                failure=dict(when='setup',traceback='\n'.join(e.get('traceback','') for e in related_errors)))
    # A duplicate must remain held even if a later row had otherwise been eligible.
    for node in expected & set(duplicates):
        holds[node]=dict(nodeid=node,reason_code='BLOCKED_DUPLICATE_NODE',evidence=evidence)
    result=dict(schema_version=2,profile=profile,expected=len(expected),collected=len(rows),unique=len(found),
        missing=missing,unknown=unknown,duplicates=duplicates,foreign_profiles=foreign,
        quarantined=sorted(set(quarantined)),eligible=sorted((expected&found)-set(holds)),
        status='PASS' if not (missing or unknown or duplicates or foreign or errors) else 'PARTIAL',
        evidence=evidence,parameter_ids_are_opaque=True)
    cp.data.setdefault('collection_reconciliation',{})[profile]=result
    atomic_json(ROOT/'quarantined_tests.json',list(q.values()))
    atomic_json(ROOT/'collection-reconciliation'/('latest-'+profile+'.json'),result)
    return result


def cached_collection(cp,profile):
    """Reuse a sealed collection only when its commit and every input file still match."""
    candidates=[a for a in cp.data['attempts'].values() if a.get('mode')=='collect' and a.get('profile')==profile
                and a.get('status')=='FINISHED']
    for attempt in reversed(candidates):
        folder=Path(attempt['directory']); receipt=read_json(folder/'controller.json',{})
        if not (receipt.get('cleanup_verified') and receipt.get('source_integrity') and receipt.get('payload_admitted')
                and receipt.get('monitor_reason') is None and receipt.get('exit_code') in (0,5)):
            continue
        request=read_json(folder/'request.json',{})
        if request.get('source_commit')!=cp.data['baseline']['commit'] or request.get('profile')!=profile:
            continue
        captured=request.get('collection_dependencies_sha256')
        if not captured:
            captured=cp.data.get('reviewed_collection_baselines',{}).get(attempt.get('source_fingerprint'))
        if captured!=collection_dependencies(cp.data['baseline']['source_sha256']):continue
        if receipt.get('request_sha256')!=file_hash(folder/'request.json'):
            raise RuntimeError('EVIDENCE_INVALID: cached collection request changed')
        path=folder/'collected.json'
        if not path.is_file():continue
        sha=file_hash(path)
        if attempt.get('collected_sha256') and attempt['collected_sha256']!=sha:
            raise RuntimeError('EVIDENCE_INVALID: cached collection changed')
        events,issues=read_journal(folder/'events.jsonl')
        if any(i['reason']=='EVIDENCE_INVALID' for i in issues):
            raise RuntimeError('EVIDENCE_INVALID: cached collection journal changed')
        rows=read_json(path,[])
        if not any(e['kind']=='collection_finished' and e.get('collected')==len(rows) for e in events):continue
        hashes={r['path']:r['test_source_sha256'] for r in request.get('inventory',{}).values()
                if r.get('profile')==profile}
        if any(not (REPO/p).is_file() or file_hash(REPO/p)!=sha for p,sha in hashes.items()):continue
        sessions=[e for e in events if e['kind']=='session_finish']
        if not sessions or sessions[-1].get('cuda_initialized') is True:continue
        if profile=='stdlib' and sessions[-1].get('torch_imported') is not False:continue
        if profile=='cpu' and not any(e['kind']=='runtime' and e.get('torch_version')=='2.5.1+cpu'
                and e.get('cuda_initialized') is False for e in events):continue
        errors=[e for e in events if e['kind'] in ('collection_error','framework_error','integrity_error')]
        if any(e['kind']=='integrity_error' for e in errors):
            raise RuntimeError('EVIDENCE_INVALID: cached collection integrity error')
        return dict(folder=str(folder),rows=rows,events=events,errors=errors,
            evidence=dict(path=str(path),sha256=sha,controller=str(folder/'controller.json'),
                applicability='Original commit and file hashes revalidated offline; no new capture-time metadata is invented'))
    return None
