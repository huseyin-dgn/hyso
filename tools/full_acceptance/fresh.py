"""Current-run acceptance adapters. Historical decisions never certify these runs.

Only the acceptance harness calls these adapters; production APIs are unchanged.
"""
from pathlib import Path
from .core import read, publish, record, sha, selected_arm


def arms(m):
    return [selected_arm(m)] if 'model_run' in m else ['A', 'B']


def authorize_auxiliary(m, ident, operation, root):
    if 'fresh_acceptance' not in m:
        raise PermissionError('fresh auxiliary scope missing')
    base=Path(m['output'])
    if ident=='__p13_finalize_stage':
        parent='06.p13_capture';expected=base/'preparation/p13-stage-finalizer'
        required='p13_finalize_stage'
    elif ident in ('__tradeoff_0','__tradeoff_1','__tradeoff_2'):
        parent='03.cpu';expected=base/'stages/03.cpu'/selected_arm(m)/('tradeoff-trial-'+ident[-1])
        required='tradeoff_trial'
    else:raise PermissionError('unknown auxiliary operation')
    if operation!=required or Path(root).resolve()!=expected.resolve() or not next(
            e for e in m['experiments'] if e['id']==parent)['authorization']['approved']:
        raise PermissionError('auxiliary parent authorization or routing mismatch')


def inference_extra(m, root, device, model):
    import torch
    from hyso.performance import (ColdStartRecipe, ColdStartProtocol, measure_cold_start,
        BatchSearchProtocol, find_max_batch_size, measure_footprint)
    from hyso._evaluation_evidence import model_hash
    from .handlers import save, UnsafeCaptureError
    arm = selected_arm(m); folder = Path(root)/arm; refs = []
    weights = folder/'trained-weights.pt'
    state = model_hash(model)
    torch.save(model.state_dict(), weights)
    refs.append(save(folder, 'weights-origin', dict(checkpoint=m['assets']['models'][arm],
        state_sha256=state, weights=record(weights), device=device, model=arm)))
    refs.append(save(folder, 'weight-only-footprint', measure_footprint(model,
        artifact_path=weights, artifact_format='pytorch_weights_only_state_dict', expected_sha256=sha(weights))))
    recipe_kw = dict(model_kwargs=dict(manifest_path=str(Path(m['output'])/'manifest.json'), arm=arm),
        input_shape=(1,1,28,28), device=device, factory_search_path=str(Path(m['source_snapshot']['root'])/'tools'),
        artifact_path=str(weights), artifact_sha256=sha(weights), seed=m['protocols']['inference']['seed'])
    cold = measure_cold_start(ColdStartRecipe('full_acceptance.workloads:architecture_only', **recipe_kw),
        protocol=ColdStartProtocol(**m['fresh_acceptance']['cold']))
    refs.append(save(folder, 'cold-start', cold))
    rows = cold.to_dict()['raw_evidence']['trials']
    if not all(r['status']=='measured' and r['process_terminated'] and r['exit_code']==0
               and r['phases']['artifact_load']['status']=='measured' for r in rows):
        raise UnsafeCaptureError('fresh_cold_worker_incomplete_no_retry')
    precisions = ['fp32','fp64'] if device=='cpu' else ['fp32','amp_fp16']
    for precision in precisions:
        recipe=ColdStartRecipe('full_acceptance.workloads:architecture_only', precision=precision, **recipe_kw)
        value=find_max_batch_size(recipe, protocol=BatchSearchProtocol(**m['fresh_acceptance']['capacity']))
        refs.append(save(folder, 'capacity-'+precision, value))
        d=value.to_dict()
        if d['measurements']['status']!='complete' or any(
                r['status']!='feasible' or not r['process_terminated'] for r in d['raw_evidence']['candidates']):
            raise UnsafeCaptureError('finite_capacity_worker_incomplete_no_retry')
    if model_hash(model)!=state or sha(m['assets']['models'][arm]['path'])!=m['assets']['models'][arm]['sha256']:
        raise ValueError('selected inference source state changed')
    if device=='cpu':
        from .safety import supervise
        for trial in range(3):
            e=dict(id='__tradeoff_'+str(trial),handler='tradeoff_trial',outer_seconds=180)
            value=supervise(m,e,output=folder/('tradeoff-trial-'+str(trial)))
            refs.append(save(folder,'tradeoff-supervisor-'+str(trial),value))
            if value['status']!='PASS' or value['supervisor']['cleanup']['active_processes']!=0:
                raise UnsafeCaptureError('independent_tradeoff_worker_incomplete_no_retry')
    return refs


def tradeoff_trial(m, root):
    """Independent process pair, selected trained CNN vs channels-last clone.

    This evaluates a memory layout option. Equal quality or no significant winner
    is a valid measured result; no improvement is required or invented.
    """
    import copy, torch
    from hyso import EvaluationCohort, compare_performance_quality
    from hyso._evaluation_evidence import model_hash
    from .workloads import configure, load_model, data
    from .handlers import result, save
    configure(m); arm=selected_arm(m); model=load_model(m,arm); before=model_hash(model)
    candidate=copy.deepcopy(model).to(memory_format=torch.channels_last)
    if model_hash(candidate)!=before:raise ValueError('layout clone changed trained weights')
    x,y=data(m);cohort=EvaluationCohort(x,y,sample_ids=[f'mnist:test:{i:05}' for i in range(len(x))],
        class_names=[str(i) for i in range(10)],split='test',preprocessing_id=m['assets']['dataset']['preprocessing'])
    refs=[]
    for precision in ('fp32','fp64'):
        value=compare_performance_quality(model,candidate,cohort,precision=precision,iterations=20,
            batch_size=m['protocols']['inference']['batch'],memory_metric='logical_model_bytes',independent_samples=False)
        refs.append(save(root,precision,value))
    if model_hash(model)!=before or model_hash(candidate)!=before:raise ValueError('tradeoff source mutated')
    refs.append(save(root,'identity',dict(model=arm,checkpoint=m['assets']['models'][arm],state_sha256=before,
        candidate='same trained weights, channels_last memory layout',process_independent_pair=True,
        within_pair_latency_iterations_independent=False)))
    return result(evidence=refs,criteria=dict(selected_trained_comparison=True))


def finalize_p13_stage(m, root):
    from .p13_capture_finalize import build
    from .handlers import result
    out=Path(m['output'])/'p13-finalized'
    build(Path(m['output'])/'manifest.json',sha(Path(m['output'])/'manifest.json'),out,completed_stage=True)
    return result(evidence=[record(out/'p13-evidence-configuration.json'),record(out/'p13-finalization.json')],
        criteria=dict(completed_capture_provenance=True),native_execution=False)


# Fresh mapping: replay only checks historical portability, never fresh acceptance.
MAPPING={**{f'P{i:02}':['03.cpu','04.cuda'] for i in range(1,9)},
    'P09':['04.cuda'],'P10':['03.cpu'],'P11':['03.cpu','04.cuda'],
    'P12':['06.layer'],'P13':['06.p13_capture','06.layer'],'P14':['06.p13_capture','06.layer'],
    'P15':['04.cuda'],'P16':['03.cpu','04.cuda'],'P17':['03.cpu','04.cuda'],'P18':['06.roofline'],
    **{f'P{i:02}':['05.training'] for i in range(19,26)},
    'P26':['03.cpu','04.cuda'],'P27':['03.cpu'],'P28':['03.cpu','04.cuda'],
    'P29':['03.cpu'],'P30':['08.video','08.camera'],'P31':['10.stability'],
    'P32':['11.memory'],'P33':['07.energy'],'P34':['07.devices'],
    'P35':['09.provider_cpu','09.provider_cuda']}


INFERENCE_FILES={'P01':['benchmark','latency'],'P02':['throughput'],'P03':['pipeline'],
    'P04':['pipeline','stages'],'P05':['cold-start'],'P06':['batch-sweep'],
    'P07':['resolution-sweep'],'P08':['precision-sweep'],'P09':['benchmark'],
    'P10':['process-memory'],'P11':['weight-only-footprint','weights-origin'],
    'P16':['utilization','utilization-work'],'P17':['capacity-fp32'],
    'P26':['quality'],'P27':['robustness'],'P28':['quality']}


def artifact_problem(ident, path, data):
    name=path.stem
    if name in ('batch-sweep','resolution-sweep','precision-sweep'):
        points=data['raw_evidence']['points']
        if not points or any(r['status']!='measured' for r in points):return 'unmeasured sweep point'
    if ident=='P01' and name=='latency' and any(x['value'] is None for x in data['measurements']):
        return 'latency quantile sample policy not satisfied'
    if ident=='P02' and name=='throughput' and data['measurements']['value']<=0:return 'no measured throughput'
    if ident=='P09' and name=='benchmark':
        if data['cuda_memory']['status']!='measured' or data['cuda_memory']['peak_allocated_bytes']<=0:
            return 'CUDA allocator not measured'
    if ident=='P10' and name=='process-memory':
        if any(data['measurements'][k]['value'] is None for k in ('current_rss_bytes','sampled_peak_rss_bytes','current_lifetime_hwm_bytes')):
            return 'native RSS/HWM unavailable'
    if ident=='P16' and name=='utilization' and data['measurements']['status']!='measured':
        return 'utilization domain incomplete'
    if ident=='P16' and name=='utilization-work' and data['completed_calls']<=0:return 'no utilization workload completed'
    if ident=='P12' and name=='operations':
        # Analytic counts are useful only when the declared observed calls are modeled.
        if data['measurements'].get('status') not in ('complete','modeled','measured'):
            return 'operation-count coverage incomplete'
    return None


def independent_layer_subset(ident, stage, row):
    # Partial raw measurements remain useful, but never certify a failed
    # required stage as full feature acceptance.
    return False


def matrix(m, rows):
    old=read(m['prior_audit']['P01-P35-acceptance-matrix.json']['path'])
    arm=selected_arm(m);out=[]
    for feature in old['features']:
        ident=feature['id'];ids=MAPPING[ident];selected=[rows[x] for x in ids]
        refs=[];problems=[];metrics=[];checks=[]
        pending=any(not r.get('attempted') for r in selected)
        failed=any(r['status'] in ('FAIL','ABORTED') and not independent_layer_subset(ident,s,r) for s,r in zip(ids,selected))
        for stage,row in zip(ids,selected):
            sup=row.get('supervisor',{})
            clean=sup.get('cleanup',{}).get('active_processes')
            subset=independent_layer_subset(ident,stage,row)
            if not row.get('attempted') or (row['status']!='PASS' and not subset) or clean!=0 or (sup.get('exit_code')!=0 and not subset):
                problems.append(stage+': '+row['status']+'; '+str(row.get('reason')))
                continue
            if not next(e for e in m['experiments'] if e['id']==stage)['authorization']['approved']:
                problems.append(stage+': not authorized');continue
            metrics.extend(row.get('metrics',[]))
            folder=Path(row['evidence_directory']);files=[]
            if stage in ('03.cpu','04.cuda'):
                names=list(INFERENCE_FILES.get(ident,[]))
                if ident=='P17':names+=['capacity-fp64' if stage=='03.cpu' else 'capacity-amp_fp16']
                if ident=='P28':names+=['quality-'+p for p in m['protocols']['inference'][
                    'precisions_cpu' if stage=='03.cpu' else 'precisions_cuda'] if p!='fp32']
                files=[folder/arm/(name+'.json') for name in names]
                if ident=='P15':files=list(folder.glob('transfer-*.json'))
                if ident=='P29':files=[folder/arm/f'tradeoff-trial-{i}'/(p+'.json') for i in range(3) for p in ('fp32','fp64')]
            elif stage=='06.layer':
                names=['operations'] if ident=='P12' else ['operator-profile','operations'] if ident=='P14' else ['ownership-calibration','verified-cuda-modules']
                files=[folder/(name+'.json') for name in names]
            else:
                files=[Path(r['path']) for r in row.get('evidence',[]) if Path(r['path']).suffix=='.json']
            if not files or not all(p.is_file() for p in files):
                problems.append(stage+': required current artifacts missing');continue
            if stage=='05.training':
                if not row.get('criteria',{}).get('P23_basic_each_device'):problems.append('training CPU/CUDA basic coverage incomplete')
                if ident=='P24':files=[folder/'cpu/time-to-target.json']
            for p in files:
                if not p.is_file():problems.append('missing '+str(p));continue
                ref=record(p)
                if not any(r.get('path')==ref['path'] and r.get('sha256')==ref['sha256'] for r in row.get('evidence',[])):
                    problems.append('unbound or changed current evidence '+str(p))
                refs.append(ref)
                if p.stem=='operator-profile':continue  # producer validated; preserve exact raw trace hash
                data=read(p)
                problem=artifact_problem(ident,p,data)
                if problem:problems.append(stage+': '+problem)
                if ident=='P05' and not all(t['status']=='measured' and t['exit_code']==0 for t in data['raw_evidence']['trials']):problems.append('cold trial incomplete')
                if ident=='P17' and data['measurements']['status']!='complete':problems.append('finite capacity incomplete')
            checks.append(dict(stage=stage,owned_job_cleanup_active_processes=clean,
                exit_code=sup.get('exit_code'),criteria=row.get('criteria',{}),model=arm,
                device=next(e['device'] for e in m['experiments'] if e['id']==stage)))
        verdict='NOT_RUN' if pending else 'FAIL' if failed else 'PARTIAL' if problems else 'PASS'
        scope=m['fresh_acceptance']['feature_scopes'].get(ident,'selected trained Model '+arm+'; manifest finite protocol and actual measured API scope')
        out.append(dict(id=ident,name=feature['name'],prior_scoped_decision=feature['decision'],decision=verdict,
            decision_basis='; '.join(problems) if problems else 'current authorized measurements and clean owned workers',
            current_execution='INCOMPLETE' if pending else 'OBSERVED',experiments={x:rows[x]['status'] for x in ids},
            retained_scope=scope,prior_decision_source=feature['decision_source'],current_evidence=refs,
            limitations=[scope],model=arm,checkpoint=m['assets']['models'][arm],checks=checks,metrics=metrics,
            workload_identity=m.get('optional',{}).get('camera_adapter',{}).get('configuration') if ident=='P30' else
                dict(kind='controlled_reference_kernels') if ident=='P18' else m['assets']['models'][arm]))
    return out
