"""Offline preparation audit: AST, metadata, paths, manifests and checkpoint data only."""
from __future__ import annotations

import ast
import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,REPO,LEGACY,atomic_json,read_json,source_snapshot,utc


def review():
    checks=[]
    def record(name,good,detail):
        checks.append(dict(name=name,status='PASS' if good else 'FAIL',detail=detail))
    files=list(Path(__file__).parent.rglob('*.py'))
    for path in files:
        payload=path.read_text(encoding='utf-8-sig')
        ast.parse(payload,filename=str(path))
        compile(payload,str(path),'exec')
    record('Python syntax',True,dict(files=len(files),execution='No code objects executed'))
    ps=read_json(ROOT/'preparation/powershell-syntax.json',{})
    record('PowerShell syntax',ps.get('error_count')==0,ps)
    cp=read_json(ROOT/'checkpoint.json',{})
    inventory=cp.get('inventory',{})
    record('Checkpoint schema',cp.get('schema_version')==1 and bool(inventory),dict(schema=cp.get('schema_version')))
    record('Frozen source fingerprint',source_snapshot()==cp['baseline']['source_sha256'],dict(commit=cp['baseline']['commit']))
    rows=[r for name in ('passed','failed','error','skipped','blocked','timeout','not_run')
          for r in read_json(ROOT/(name+'_tests.json'),[])]
    nodes=[r['nodeid'] for r in rows]
    changes=read_json(Path(__file__).with_name('retired-tests.json'),{})
    frozen=read_json(LEGACY/'collected-tests.json',[])
    expected_active=len(frozen)+len(changes.get('additions',[]))-len(changes.get('retirements',[]))
    record('Unique source reconciliation',len(nodes)==len(set(nodes))==len(inventory)==expected_active
           and set(nodes)==set(inventory),dict(unique=len(set(nodes)),counts=dict(Counter(r['status'] for r in rows))))
    risks=read_json(LEGACY/'blocked-nodeids.json',{})
    blocked={r['nodeid'] for r in rows if r['status']=='BLOCKED'}
    record('Risk exclusions',len(risks)==13 and set(risks)<=blocked,dict(risky=len(risks)))
    plan=read_json(ROOT/'preparation/execution-plan.json',{})
    assigned=[n for g in plan.get('source_groups',[]) for n in g['nodes']]
    pending={r['nodeid'] for r in rows if r['status']=='NOT_RUN' and inventory[r['nodeid']]['profile']=='cpu'}
    record('Serial source assignment',set(assigned)==pending and len(assigned)==len(set(assigned))
        and not set(assigned)&set(risks),dict(groups=len(plan.get('source_groups',[])),assigned=len(assigned),orphans=len(pending-set(assigned))))
    package=read_json(ROOT/'package_acceptance.json',{})
    record('Separate package denominator',package.get('planned_cases')==36 and len(package.get('cases',[]))==36,
           dict(planned=package.get('planned_cases'),counts=package.get('test_counts')))
    matrix=read_json(ROOT/'P01-P35-acceptance.json',[])
    record('35 separate features',len(matrix)==35 and len({r['analysis_number'] for r in matrix})==35,
        dict(decisions=dict(Counter(r['decision'] for r in matrix))))
    p23=next(r for r in matrix if r['analysis_number']=='P23')
    record('P23 B4 scope',p23['decision']=='PARTIAL' and p23['CUDA_validation']['current']=='BLOCKED_B4_UNRESOLVED',p23['CUDA_validation'])
    p31=read_json(ROOT/'preparation/P31-historical-integrity.json',{})
    record('P31 historical integrity',len(p31.get('checks',{}))==2 and all(r['status']=='PASS' for r in p31.get('checks',{}).values()),
        dict(records=len(p31.get('checks',{})),new_run=False))
    protection=read_json(ROOT/'user_file_preservation.json',{})
    original=protection.get('original_user_files',{})
    record('47 original user files',len(original)==47 and all(r['unchanged'] for r in original.values())
        and protection.get('user_files_unchanged') is True,dict(original_files=len(original)))
    protected=protection.get('protected_model_and_data_inputs',{})
    record('Original model and data hashes',len(protected)==4 and all(r['status']=='PASS' for r in protected.values()),dict(files=len(protected)))
    env=read_json(ROOT/'environment.json',{})['prepared_runtime']
    record('Declared runtime paths and metadata',not env['missing'],dict(missing=env['missing'],
        cpu_torch=env['torch_cpu_distribution'],build=env['versions'].get('build'),runtime_imports='DEFERRED'))
    required=['FINAL_REPORT_TR.md','FINAL_REPORT_EN.md','FINAL_REPORT.html','EXECUTIVE_SUMMARY_TR.md',
        'summary.json','test_inventory.json','test_reconciliation.json','P01-P35-acceptance.json',
        'P01-P35-acceptance.csv','package_acceptance.json','public_api_acceptance.json','cli_acceptance.json',
        'forensic_acceptance.json','progress.json','environment.json','resource_usage.json','interruptions.json',
        'retry_history.json','checkpoint.json','user_file_preservation.json','evidence_manifest.json','campaign.log',
        'COMPLETE_EVIDENCE.zip','COMPLETE_EVIDENCE.sha256.json']
    record('Required preparation reports',all((ROOT/n).is_file() for n in required),dict(files=len(required)))
    issues=[]
    with zipfile.ZipFile(ROOT/'COMPLETE_EVIDENCE.zip') as archive:
        manifest=json.loads(archive.read('evidence_manifest.json'))['contents']
        if set(archive.namelist())!=set(manifest)|{'evidence_manifest.json'}:issues.append('manifest content set differs')
        for name,binding in manifest.items():
            h=hashlib.sha256()
            with archive.open(name) as stream:
                for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk)
            if h.hexdigest()!=binding['sha256']:issues.append(name)
        record('ZIP contents SHA-256 manifest',not issues,dict(entries=len(manifest),issues=issues))
    after=read_json(ROOT/'checkpoint.json',{})
    record('Audit performed no test execution',set(after.get('attempts',{}))==set(cp['attempts']) and 'torch' not in sys.modules and 'pytest' not in sys.modules,
        dict(attempts=len(cp['attempts']),pytest_imported='pytest' in sys.modules,torch_imported='torch' in sys.modules))
    result=dict(generated_utc=utc(),status='PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL',
        scope='STATIC_PREPARATION_ONLY; runtime behavior is not certified by this audit',checks=checks)
    atomic_json(ROOT/'preparation/static-review.json',result)
    return result


if __name__=='__main__':
    result=review()
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result['status']=='PASS' else 1)
