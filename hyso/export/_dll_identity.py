"""Strict DLL comparison and transportable, hash-bound failure evidence."""
import hashlib
import json
import os
from pathlib import Path
from .contracts import ExportError
from .artifacts import file_hash, safe_file
from .._native_identity import canonical_path, file_identity


def compare(actual, allowed, identities):
    canonical={canonical_path(p):h for p,h in allowed.items()}
    if len(canonical)!=len(allowed):raise ExportError('DLL_INTEGRITY','ambiguous duplicate canonical manifest path')
    allowed=canonical
    rows=[]
    for path, value in actual.items():
        key=canonical_path(path)
        candidates=[p for p in allowed if Path(p).name.lower()==Path(key).name.lower()]
        expected=key if key in allowed else candidates[0] if len(candidates)==1 else None
        fields=[]
        if key not in allowed:fields.extend(('manifest_membership','path'))
        if allowed.get(key)!=value['sha256']:fields.append('sha256')
        expected_id=identities.get(key)
        for field in ('file_id','bytes'):
            if not expected_id or expected_id.get(field)!=value.get(field):fields.append(field)
        rows.append(dict(dll_name=Path(key).name,actual_path=key,expected_path=expected,
            expected_path_candidates=candidates,actual_sha256=value['sha256'],expected_sha256=allowed.get(expected),
            expected_candidates=[dict(path=p,sha256=allowed[p],identity=identities.get(p)) for p in candidates],
            actual_identity={k:value[k] for k in ('file_id','bytes')},expected_identity=identities.get(expected),
            manifest_member=key in allowed,mismatched_fields=fields,
            reason='identity mismatch' if fields else 'matched'))
    return dict(rows=rows,rejected=not actual or any(r['mismatched_fields'] for r in rows),
                reason='empty loaded inventory' if not actual else 'strict pre-admission comparison')


def failure_bundle(folder, controller):
    """Read before staging cleanup; bind diagnostics to the owned worker receipt."""
    records={}
    for name in ('request.json','controller.json','identity-failure.json','stdout.txt','stderr.txt'):
        path=safe_file(folder,name)
        if not path.exists():continue
        raw=path.read_bytes()
        if len(raw)>2*1024**2:raise ExportError('RESULT_INTEGRITY','oversized native failure evidence')
        records[name]=dict(sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode('utf-8'))
    request=json.loads(records['request.json']['text'])
    if request['token']!=controller['token'] or json.loads(records['controller.json']['text'])!=controller:
        raise ExportError('RESULT_INTEGRITY','failure controller/request mismatch')
    diagnostic=records.get('identity-failure.json')
    if diagnostic:
        data=json.loads(diagnostic['text'])
        if (data['pid']!=controller['launcher_pid'] or data['token']!=controller['token'] or
            data['request_sha256']!=records['request.json']['sha256']):
            raise ExportError('RESULT_INTEGRITY','DLL failure identity does not match owned request')
    return dict(schema_version=1,token=request['token'],records=records)


def retain(folder,bundle):
    """Exclusive publication outside staging; never overwrite prior evidence."""
    root=Path(folder)
    if root.is_symlink() or os.name=='nt' and root.lstat().st_file_attributes & 0x400:
        raise ExportError('OWNERSHIP','refusing linked failure destination')
    from uuid import UUID
    UUID(bundle['token'])
    path=safe_file(root,'native-failure-'+bundle['token']+'.json')
    raw=(json.dumps(bundle,sort_keys=True,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode('utf-8')
    with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    sha=hashlib.sha256(raw).hexdigest()
    if file_hash(path)!=sha:raise ExportError('RESULT_INTEGRITY','retained native failure changed')
    return dict(path=str(path),sha256=sha)


def verify_retained(record):
    path=Path(record['path'])
    if path.is_symlink() or file_hash(path)!=record['sha256']:raise ExportError('RESULT_INTEGRITY','retained failure hash mismatch')
    bundle=json.loads(path.read_text(encoding='utf-8'))
    for value in bundle['records'].values():
        if hashlib.sha256(value['text'].encode('utf-8')).hexdigest()!=value['sha256']:
            raise ExportError('RESULT_INTEGRITY','retained failure member hash mismatch')
    return bundle
