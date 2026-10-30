"""Offline P30 evidence and preexisting-work preservation checks; no workloads."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit',required=True,type=Path)
    parser.add_argument('--run',required=True,type=Path);args=parser.parse_args()
    audit=args.audit.resolve();run=args.run.resolve()
    from hyso.performance import RuntimeResult
    from hyso.performance.runtime_result import _hash
    from hyso.performance.stability import _read_journal
    from hyso.performance.streaming import _analyze_stream
    before=json.loads((audit/'preservation-before.json').read_text(encoding='utf-8'))
    allowed={'hyso/performance/streaming.py','hyso/performance/stability.py',
             'hyso/performance/runtime_result.py','tests/runtime_support.py'}
    unchanged={path:sha(ROOT/path)==digest for path,digest in before.items() if path not in allowed}
    assert all(unchanged.values()),[p for p,ok in unchanged.items() if not ok]
    results={}
    for policy in ('block','drop_newest'):
        r=RuntimeResult.from_json(run/(policy+'.json'));d=r.to_dict()
        assert RuntimeResult.from_csv(run/(policy+'.csv')).to_dict()==d
        assert _read_journal(run/policy)==d['evidence']
        a=d['analysis'];assert a['status']=='completed'
        e=d['evidence'];states={};pending=set();processed=dropped=0
        for row in e['rows']:
            frame=row.get('frame_id')
            if row['event']=='received':states[frame]=row['content_sha256'];pending.add(frame)
            elif row['event']=='processed':processed+=1;pending.remove(frame)
            elif row['event']=='dropped':dropped+=1;pending.remove(frame)
            assert len(states)==processed+dropped+len(pending)
        assert not pending and len(states)==a['received']
        results[policy]=dict(received=a['received'],processed=processed,dropped=dropped,
            conservation_prefixes_verified=len(e['rows']),json_csv_journal_identical=True)
    # Use the untouched original analyzer as an independent legacy schema oracle.
    original=subprocess.check_output(['git','show','HEAD:hyso/performance/streaming.py'],cwd=ROOT)
    snapshot=audit/'original-streaming.py';snapshot.write_bytes(original)
    module=types.ModuleType('hyso.performance._p30_legacy');module.__package__='hyso.performance'
    sys.modules[module.__name__]=module;exec(compile(original,str(snapshot),'exec'),module.__dict__)
    value=copy.deepcopy(RuntimeResult.from_json(run/'block.json').to_dict());e=value['evidence']
    e['header'].pop('lifecycle_version');e['request']['protocol'].pop('max_journal_bytes')
    previous=_hash(e['header']);cursor=0
    for index,chunk in enumerate(e['chunks']):
        n=chunk['rows'];previous=_hash(dict(schema_version=1,run_id=e['header']['run_id'],index=index,
            previous_sha256=previous,rows=e['rows'][cursor:cursor+n]));cursor+=n;chunk['sha256']=previous
    e['terminal']['chunks']=e['chunks'];e['terminal']['last_chunk_sha256']=previous
    value['analysis']=module._analyze_stream(e)
    assert _analyze_stream(e)==value['analysis']
    value['source_sha256']=_hash(dict(feature_id='P30',evidence=e))
    legacy=RuntimeResult(value);legacy_path=audit/'legacy-oracle.json';legacy.to_json(legacy_path)
    assert RuntimeResult.from_json(legacy_path).to_dict()==value
    result=dict(unchanged_preexisting_files=len(unchanged),all_preexisting_unrelated_hashes_match=True,
        allowed_existing_edits=sorted(allowed),legacy_original_analyzer_exact_analysis_compatibility=True,
        legacy_oracle_scope='original_analyzer_applied_to_new_real_video_events_not_a_historical_camera_run',
        policies=results,B4='OPEN',CPU65536='OPEN')
    with (audit/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
