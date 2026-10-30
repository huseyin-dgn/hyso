"""Acceptance execution state, honest missing metrics and non-certifying reports."""
from pathlib import Path
import collections,csv,json,xml.etree.ElementTree as ET
from .core import read,publish,record,now,sha
from .registry import FEATURE_EXPERIMENTS

OBSERVED={'PASS','PARTIAL','FAIL','UNSUPPORTED'}
MISSING_STATES={'NOT_AUTHORIZED','PENDING','BLOCKED_PREREQUISITE','BLOCKED_DEPENDENCY','NOT_RUN_STOPPED','ABORTED'}

def initial_results(m):
    return {e['id']:dict(id=e['id'],stage=e['stage'],status='PENDING' if e['authorization']['approved'] else 'NOT_AUTHORIZED',
        attempted=False,reason='awaiting_explicit_execute' if e['authorization']['approved'] else e['authorization']['basis'],
        metrics=[],criteria={},evidence_directory=None) for e in m['experiments']}

def prerequisites(m,e):
    missing=[]
    for key in e['requires']:
        if key=='gpu':
            if not m['hardware'].get('gpu_uuid'):missing.append('explicit physical GPU UUID missing; recorded UUID is selection intent, validated natively only after approval')
        elif key.startswith('ort_'):
            path=m['hardware'].get(key)
            if not path or not Path(path,'onnxruntime').is_dir():missing.append(key+' runtime_search_path missing')
        elif key=='camera_adapter':
            cfg=m['optional'].get('camera_adapter',{}).get('configuration',{})
            required=('factory','factory_source','checkpoint','input_contract','output_contract','preprocessing','provenance')
            if any(k not in cfg for k in required):missing.append('task-appropriate trained camera adapter/config/checkpoint missing; MNIST not assumed appropriate')
        elif key=='video':
            cfg=m['optional'].get('camera_adapter',{}).get('configuration',{}).get('video',{})
            if not cfg.get('path') or not Path(cfg['path']).is_file():missing.append('real local video required')
        elif key=='ncu_plan':
            cfg=m['optional'].get('ncu_plan',{}).get('configuration',{})
            if not cfg.get('tool') or not Path(cfg['tool']['path']).is_file():missing.append('installed pinned Nsight Compute required; native counter privileges checked at capture')
    return missing

def decision(m,e,rows,*,stopped=False):
    ready=prerequisites(m,e)
    if not e['authorization']['approved']:return 'NOT_AUTHORIZED',e['authorization']['basis'],ready
    if stopped:return 'NOT_RUN_STOPPED','earlier experiment failed/stopped; no native continuation or retry',ready
    if ready:return 'BLOCKED_PREREQUISITE','; '.join(ready),ready
    bad=[x for x in e['depends'] if rows[x]['status']!='PASS']
    if bad:return 'BLOCKED_DEPENDENCY','required successful dependency: '+', '.join(bad),ready
    return 'READY',None,[]

def validate_metrics(row):
    for metric in row.get('metrics',[]):
        if metric.get('kind') not in ('measured','derived','missing'):raise ValueError('metric kind required')
        if metric.get('value') is None and (metric['kind']!='missing' or not metric.get('reason')):raise ValueError('null+reason required')
        if metric.get('value') is not None and metric['kind']=='missing':raise ValueError('missing metric has numeric value')

def matrix(m,rows):
    old=read(m['prior_audit']['P01-P35-acceptance-matrix.json']['path']);out=[]
    promotions={'P18':('06.roofline','P18_calibration_screen'),'P30':('08.camera','real_camera_trained'),
        'P31':('10.stability','P31_uninterrupted_7200'),'P33':('07.energy','trained_energy_full'),
        'P34':('07.devices','trained_matched_CPU_GPU')}
    for feature in old['features']:
        ident=feature['id'];ids=FEATURE_EXPERIMENTS[ident];selected=[rows[x] for x in ids]
        current=feature['decision'];basis='retained historical scoped decision; no automatic promotion from stage exit'
        if ident in promotions:
            stage,key=promotions[ident];row=rows[stage]
            if row.get('attempted') and row['status']=='PASS' and row.get('criteria',{}).get(key) is True and row.get('supervisor',{}).get('cleanup',{}).get('active_processes')==0:
                current='PASS';basis='new exact criterion satisfied in declared scope; historical failed sessions unchanged'
            elif row['status']=='FAIL':current='FAIL';basis='new required acceptance criterion actually failed; historical scope retained separately'
        execution='FAIL' if any(r['status']=='FAIL' for r in selected) else 'ABORTED' if any(r['status']=='ABORTED' for r in selected) else (
            'OBSERVED' if all(r['status'] in OBSERVED for r in selected) else 'INCOMPLETE')
        out.append(dict(id=ident,name=feature['name'],prior_scoped_decision=feature['decision'],decision=current,
            decision_basis=basis,current_execution=execution,experiments={x:rows[x]['status'] for x in ids},
            retained_scope=feature['decision_scope'],prior_decision_source=feature['decision_source'],
            current_evidence=[r.get('evidence_directory') for r in selected if r.get('evidence_directory')],
            limitations=feature['unsupported_metrics']))
    return out

def write_reports(m,rows,*,state='PREPARED',preparation=None):
    out=Path(m['output']);preparation=preparation or {}
    for row in rows.values():validate_metrics(row)
    complete=all(rows[e['id']].get('attempted') and rows[e['id']]['status'] in OBSERVED and e['authorization']['approved'] for e in m['experiments'])
    # Even a completed suite can retain PARTIAL/FAIL/unsupported feature scopes.
    counts=dict(collections.Counter(r['status'] for r in rows.values()));features=matrix(m,rows)
    publish(out/'stage-results.json',dict(schema_version=1,run_id=m['run_id'],manifest_sha256=sha(out/'manifest.json'),
        updated_at=now(),suite_complete=complete,status_counts=counts,experiments=list(rows.values()),preparation=preparation))
    cols=['id','name','prior_scoped_decision','decision','decision_basis','current_execution','experiments','retained_scope','prior_decision_source','current_evidence','limitations']
    with (out/'P01-P35-final-acceptance-matrix.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,cols);writer.writeheader()
        for f in features:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in f.items()})
    publish(out/'P01-P35-final-acceptance-matrix.json',dict(features=features,scope='retained acceptance vs current execution separated',suite_complete=complete))
    groups={'model-evidence':['03.cpu','04.cuda'],'training-evidence':['05.training'],'cpu-gpu-comparison':['07.devices'],
        'energy-evidence':['07.energy'],'memory-evidence':['05.training','10.stability','11.memory'],
        'provider-evidence':['09.provider_cpu','09.provider_cuda']}
    for name,ids in groups.items():
        observed=[rows[x] for x in ids if rows[x].get('attempted')]
        publish(out/(name+'.json'),dict(schema_version=1,status='OBSERVED' if observed else 'MISSING',
            reason=None if observed else 'required experiment has not run; no synthetic native substitute',
            value=observed if observed else None,experiments={x:rows[x]['status'] for x in ids},
            assets=m['assets'] if name=='model-evidence' else None,preparation_smoke=preparation.get('smoke_cpu') if name=='model-evidence' else None))
    publish(out/'native-forensic-summary.json',dict(**m['forensic'],new_reproducer_runs=0,
        historical_review=[m['prior_audit']['P01-P35-acceptance-matrix.json']],
        observed_native_failures=[r for r in rows.values() if r.get('supervisor',{}).get('native_crash')],
        interpretation='root causes remain OPEN; successful short tests do not close B4 or CPU65536'))
    write_junit(out,rows,preparation)
    partial=[f['id'] for f in features if f['decision']=='PARTIAL'];fc=collections.Counter(f['decision'] for f in features)
    lines=['# HYSo full-system acceptance — '+('EXECUTION REPORT' if complete else 'PREPARATION / INCOMPLETE EXECUTION'),'',
        '**Suite complete: '+str(complete).lower()+'.** State: `'+state+'`. A prepared manifest, successful smoke or process exit is not full native acceptance.','',
        'Manifest SHA256: `'+sha(out/'manifest.json')+'`. Source tree: `'+m['source_snapshot']['tree_sha256']+'`.','',
        'Retained/current scoped feature decisions: `'+json.dumps(dict(fc))+'`. These are not35 newly executed PASS results.','',
        '| Experiment | Status | Attempted | Reason |','|---|---|---|---|']
    for e in m['experiments']:
        row=rows[e['id']];lines.append('| '+e['id']+' | '+row['status']+' | '+str(row.get('attempted',False))+' | '+str(row.get('reason') or '') .replace('|','/')+' |')
    lines+=['','## All35 features','','| ID | Feature | Retained | Current decision | New execution |','|---|---|---|---|---|']
    for f in features:lines.append('| '+f['id']+' | '+f['name']+' | '+f['prior_scoped_decision']+' | '+f['decision']+' | '+f['current_execution']+' |')
    lines+=['','## Evidence and limits','',
        '- Exact commands, device/model/input/work/stop/cleanup: `manifest.json`; actual commands and Job/low-memory receipts are in each stage directory.',
        '- Metric records explicitly use measured/derived/missing. Missing values remain null+reason. Native canonical raw JSON/CSV/trace/journals remain authoritative.',
        '- Preflight/smoke results: '+json.dumps(preparation,ensure_ascii=False),
        '- Missing prerequisites: '+json.dumps({e['id']:prerequisites(m,e) for e in m['experiments'] if prerequisites(m,e)},ensure_ascii=False),
        '- B4 OPEN/ROOT_CAUSE_UNRESOLVED; CPU65536 OPEN. No automatic reproducer or retry. Historical P18 compute screen FAIL remains recorded; new calibration uses unchanged10% screen.',
        '- Camera model/data: explicit task-appropriate trained adapter is required. MNIST camera suitability is not assumed.',
        '- Matched CPU/GPU experiments share checkpoint/input/precision/batch/warmup/samples/completion scope. Independent P28/P29/P31 records are not joined to fabricate speedup.',
        '- RSS/private commit, CUDA allocated/reserved and logical/artifact bytes remain separate. No unverified winner, model-exclusive energy, causal leak or leak-free guarantee.',
        '- P31 requires one actual completed>=7200s work span, valid terminal/close/heartbeat/accounting/gaps and inner+outer cleanup. Short/aborted runs are not concatenated.',
        '- P13 CPU calibration alone does not fill CUDA attribution. P23 original measured scopes and newer PARTIAL source closure are preserved. P14/P15/P17/P24/P27/P28/P29 breadth not newly exercised stays historical, not fresh native acceptance.',
        '- P18 reference kernels are explicitly controlled, non-trained calibration references; trained-CNN DRAM/capacity/headroom is not inferred.',
        '- `test-results.xml` uses explicit NOT_AUTHORIZED/BLOCKED skipped types for JUnit interoperability; JSON statuses distinguish these from test SKIP and never count them as PASS.',
        '- Preservation: `preservation-verification.json`. No checkpoint, previous training artifact or source/user dirty file is overwritten.',
        '','## Remaining work','',
        'Review `execution-plan.md` and `commands.ps1`; create a newly authorized child manifest for explicitly approved native stages. Supply a trained camera adapter/appropriate data and any missing runtime/hardware. Execute once, inspect raw failures; fix only proved defects and prepare a fresh source-pinned run. Full acceptance remains incomplete until every required experiment has a real terminal result.']
    (out/'final-report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return complete

def write_junit(out,rows,preparation):
    suite=ET.Element('testsuite',name='HYSo-full-acceptance',tests=str(len(rows)),failures=str(sum(r['status']=='FAIL' for r in rows.values())),
        skipped=str(sum(r['status'] not in ('PASS','FAIL') for r in rows.values())))
    properties=ET.SubElement(suite,'properties');ET.SubElement(properties,'property',name='suite_complete',value=str(all(r.get('attempted') and r['status'] in OBSERVED for r in rows.values())).lower())
    for ident,r in rows.items():
        case=ET.SubElement(suite,'testcase',classname='acceptance.experiment',name=ident,time=str(r.get('supervisor',{}).get('wall_seconds',0)))
        if r['status']=='FAIL':ET.SubElement(case,'failure',message=r.get('reason') or 'acceptance criterion failed')
        elif r['status']!='PASS':ET.SubElement(case,'skipped',type=r['status'],message=r.get('reason') or r['status'])
        ET.SubElement(case,'system-out').text=json.dumps(dict(status=r['status'],attempted=r.get('attempted',False),criteria=r.get('criteria',{})))
    # Include actual regression case records; no summing historical prior runs.
    root=ET.Element('testsuites');root.append(suite)
    accepted=Path(out)/'infrastructure-tests-accepted.xml'
    unit=accepted if accepted.exists() else Path(out)/'infrastructure-tests.xml'
    if unit.exists():root.append(ET.parse(unit).getroot())
    smoke=preparation.get('smoke_cpu')
    if smoke:
        s=ET.SubElement(root,'testsuite',name='HYSo-preparation-smoke',tests='1',failures='0' if smoke['status']=='PASS' else '1')
        case=ET.SubElement(s,'testcase',classname='acceptance.preparation',name='trained_Model_B_one_CPU_forward',time=str(smoke.get('supervisor',{}).get('wall_seconds',0)))
        if smoke['status']!='PASS':ET.SubElement(case,'failure',message=smoke.get('reason') or 'smoke failure')
    for row in rows.values():
        if row.get('evidence_directory'):
            xml=Path(row['evidence_directory'])/'tests.xml'
            if xml.is_file():
                original=ET.parse(xml).getroot()
                for node in original.findall('testsuite') if original.tag=='testsuites' else [original]:root.append(node)
    ET.ElementTree(root).write(Path(out)/'test-results.xml',encoding='utf-8',xml_declaration=True)
