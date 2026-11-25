"""Real tiny CPU producers, retained compatibility and fail-closed comparison."""
import copy
import hashlib
import json

import pytest
import torch

from hyso.analysis import (AnalysisContext,EvidenceIndex,EvidenceRecord,EvidenceScope,
    ModelComparisonSource,ModelComparisonResult,compare_models)
from hyso.advisor import (CapturedCPUComparisonContract,ComparisonContract,NativeMetric,
    NativeMetricResolver,compare_records,PerformanceComparison)
from hyso.checkpoint import save_training_checkpoint,load_training_checkpoint
from hyso.performance import capture_cpu_comparison,ThroughputProtocol,measure_footprint
from hyso.performance.comparison_capture import _digest


def model(outputs=2):return torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(4,outputs))
def inputs():return torch.arange(8,dtype=torch.float32).reshape(2,1,2,2)/8
def source(name,values,**scope):
    records=[EvidenceRecord.from_native(v.to_dict(),feature_id=f,scope=EvidenceScope(model_id=name,**scope)) for f,v in values.items()]
    report=EvidenceIndex(records).analyze(context=AnalysisContext(workload_id='CPU-capture-test',task='classification',model={'id':name},dataset={}))
    return ModelComparisonSource(name,report)


@pytest.fixture(scope='module')
def pair(tmp_path_factory):
    folder=tmp_path_factory.mktemp('cpu-capture'); result={}
    for name,outputs in [('A',2),('B',3)]:
        m=model(outputs);path=folder/(name+'.pth')
        save_training_checkpoint(path,m,include_rng=False)
        load_training_checkpoint(path,m,capture_evidence=True,expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        result[name]=capture_cpu_comparison(m,inputs(),footprint_artifact_path=path)
    return result


def test_real_three_producers_exact_samples_and_public_comparison(pair,tmp_path):
    comparison=compare_models(source('A',pair['A']),source('B',pair['B']),features=('P01','P02','P11'))
    data=comparison.to_dict()
    assert len(data['metrics'])==8
    assert all(r['status']=='COMPARABLE' and r['difference'] is not None for r in data['metrics']),data['metrics']
    logical=next(r for r in data['metrics'] if r['metric']==NativeMetric.MODEL_LOGICAL_BYTES.value)
    assert logical['difference']['difference']==20
    assert data['claims']['acceptance']=='NOT_EVALUATED' and data['claims']['global_winner'] is None
    comparison.to_json(tmp_path/'comparison.json')
    assert ModelComparisonResult.from_json(tmp_path/'comparison.json').to_dict()==data
    for values in pair.values():
        assert values['P01'].to_dict()['actual']['samples']==1000
        assert values['P02'].to_dict()['actual']['completed_batches']==1000
        for value in values.values():
            raw=value.to_dict();proof=raw['provenance']['comparison_capture']
            assert proof['checkpoint']['status']=='native'
            assert proof['model_state_sha256']==proof['post_model_state_sha256']
            path=tmp_path/(raw['result_id']+'.json');value.to_json(path)
            assert type(value)(json.loads(path.read_text(encoding='utf-8'))).to_dict()==raw


def test_model_modes_gradients_inputs_and_rng_are_preserved():
    m=model();m[0].eval();m[1].weight.grad=torch.ones_like(m[1].weight);x=inputs()
    before={k:v.clone() for k,v in m.state_dict().items()};grad=m[1].weight.grad;version=grad._version
    rng=torch.get_rng_state().clone();x_before=x.clone();modes=[s.training for s in m.modules()]
    capture_cpu_comparison(m,x,latency_iterations=8,throughput_protocol=ThroughputProtocol(batch_count=8,repeats=1))
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
    assert [s.training for s in m.modules()]==modes
    assert m[1].weight.grad is grad and grad._version==version and torch.equal(grad,torch.ones_like(grad))
    assert torch.equal(x,x_before) and torch.equal(torch.get_rng_state(),rng) and not torch.cuda.is_initialized()


@pytest.mark.parametrize('feature',['P01','P02','P11'])
@pytest.mark.parametrize('damage',['sample','protocol','state','input','environment','run','checkpoint'])
def test_tampered_receipts_cannot_be_imported(pair,feature,damage):
    value=pair['A'][feature];raw=value.to_dict();proof=raw['provenance']['comparison_capture']
    if damage=='sample':raw['result_id']='00000000-0000-0000-0000-000000000001'
    elif damage=='protocol':raw['protocol']['scope']='other_clock_or_boundary'
    elif damage=='state':proof['post_model_state_sha256']='a'*64
    elif damage=='input':proof['bound_input']['sha256']='b'*64
    elif damage=='environment':proof['environment']['torch']='different-runtime'
    elif damage=='run':proof['run_id']='00000000-0000-0000-0000-000000000002'
    else:proof['checkpoint']['model_state_sha256']='c'*64
    # P11's bound input is context, but must still be part of receipt integrity.
    with pytest.raises(ValueError):type(value)(raw)


def test_footprint_is_an_exact_inventory_without_fake_forwards_or_samples(monkeypatch):
    from hyso.benchmark import BenchmarkRunner
    def deny(*a,**kw):pytest.fail('P11 must not run a forward/timing protocol')
    monkeypatch.setattr(BenchmarkRunner,'run',deny);monkeypatch.setattr(torch.nn.Module,'_call_impl',deny)
    a=capture_cpu_comparison(model(),inputs(),features=('P11',))
    b=capture_cpu_comparison(model(3),inputs(),features=('P11',))
    result=compare_models(source('A',a),source('B',b),features='P11').to_dict()
    assert all(r['status']=='COMPARABLE' for r in result['metrics'][:3])
    assert result['metrics'][-1]['status']=='INSUFFICIENT_EVIDENCE'
    assert 'observations' not in a['P11'].to_dict()['actual']
    assert a['P11'].to_dict()['provenance']['comparison_capture']['bound_input']['role']=='bound_workload_context_only'


@pytest.mark.parametrize('kind',['protocol','input','precision'])
def test_real_mismatched_protocols_or_inputs_are_not_comparable(pair,kind):
    x=inputs();kwargs=dict(features=('P01',))
    if kind=='protocol':kwargs['latency_iterations']=1001
    elif kind=='input':x=x+1
    else:x=x.double()
    other=capture_cpu_comparison(model().double() if kind=='precision' else model(),x,**kwargs)
    result=compare_models(source('A',{'P01':pair['A']['P01']}),source('B',other),features='P01').to_dict()
    assert all(r['status']=='INCOMPATIBLE' and r['difference'] is None for r in result['metrics'])


def test_short_throughput_retains_sample_threshold(pair):
    short=capture_cpu_comparison(model(),inputs(),features=('P02',),throughput_protocol=ThroughputProtocol(batch_count=3,repeats=1))
    result=compare_models(source('A',{'P02':pair['A']['P02']}),source('B',short),features='P02').to_dict()
    assert result['metrics'][0]['status'] in ('INCOMPATIBLE','INSUFFICIENT_EVIDENCE')
    assert result['metrics'][0]['difference'] is None


def test_old_results_are_not_upgraded_and_legacy_contract_stays_strict(pair):
    legacy=measure_footprint(model()); assert 'comparison_capture' not in legacy.to_dict()['provenance']
    result=compare_models(source('old',{'P11':legacy}),source('new',{'P11':pair['A']['P11']}),features='P11').to_dict()
    assert all(r['status']!='COMPARABLE' for r in result['metrics'])
    old=legacy.to_dict()
    with pytest.raises(TypeError):capture_cpu_comparison(old,inputs(),features=('P11',))
    assert legacy.to_dict()==old


def test_caller_dataset_declarations_do_not_become_native_origin(pair):
    result=compare_models(source('A',pair['A'],dataset_id='caller-supplied'),source('B',pair['B'],dataset_id='caller-supplied'),features='P11').to_dict()
    assert all(r['status']=='INSUFFICIENT_EVIDENCE' for r in result['metrics'])


def test_checkpoint_mutation_invalidates_real_load_association(pair,tmp_path):
    m=model();p=tmp_path/'model.pth';save_training_checkpoint(p,m,include_rng=False)
    load_training_checkpoint(p,m,capture_evidence=True)
    with torch.no_grad():m[1].weight.add_(1)
    current=capture_cpu_comparison(m,inputs(),features=('P11',),footprint_artifact_path=p)
    assert current['P11'].to_dict()['provenance']['comparison_capture']['checkpoint']['status']=='missing'
    result=compare_models(source('A',{'P11':pair['A']['P11']}),source('B',current),features='P11').to_dict()
    assert result['metrics'][-1]['status']=='INSUFFICIENT_EVIDENCE'


def test_cuda_request_rejected_before_any_cuda_query(monkeypatch):
    def deny(*a,**kw):pytest.fail('CUDA initialization/query reached')
    monkeypatch.setattr(torch.cuda,'is_available',deny);monkeypatch.setattr(torch.cuda,'_lazy_init',deny)
    with pytest.raises(ValueError,match='CPU_throughput'):
        capture_cpu_comparison(model(),inputs(),throughput_protocol=ThroughputProtocol(device='cuda:0'))


def test_explicit_v2_contract_roundtrip_and_legacy_gate(pair):
    a=source('A',{'P11':pair['A']['P11']});b=source('B',{'P11':pair['B']['P11']})
    ar=NativeMetricResolver(__import__('hyso.analysis',fromlist=['UnifiedAnalysisReport']).UnifiedAnalysisReport(a.to_dict()['report']))
    br=NativeMetricResolver(__import__('hyso.analysis',fromlist=['UnifiedAnalysisReport']).UnifiedAnalysisReport(b.to_dict()['report']))
    ai=a.to_dict()['report']['index']['records'][0]['record_id'];bi=b.to_dict()['report']['index']['records'][0]['record_id']
    contract=CapturedCPUComparisonContract(metric=NativeMetric.MODEL_LOGICAL_BYTES,allowed_changes=('model',))
    assert CapturedCPUComparisonContract.from_dict(contract.to_dict())==contract
    assert compare_records(ar,ai,br,bi,contract=contract).status.value=='COMPARABLE'
    assert compare_records(ar,ai,br,bi,contract=ComparisonContract(metric=NativeMetric.MODEL_LOGICAL_BYTES,allowed_changes=('model',))).status.value=='INSUFFICIENT_EVIDENCE'


def test_capture_readers_and_comparison_never_execute(pair,monkeypatch):
    import hyso.performance.comparison_capture as module
    def deny(*a,**kw):pytest.fail('retained comparison invoked live operation')
    monkeypatch.setattr(module,'_environment',deny)
    monkeypatch.setattr(module,'_source_identity',deny)
    monkeypatch.setattr(module,'_clock_identity',deny)
    monkeypatch.setattr(module,'capture_cpu_comparison',deny)
    monkeypatch.setattr(torch.nn.Module,'_call_impl',deny)
    result=compare_models(source('A',pair['A']),source('B',pair['B']),features=('P01','P02','P11')).to_dict()
    assert all(r['status']=='COMPARABLE' for r in result['metrics'])


def test_duplicate_capture_is_not_independent_evidence(pair):
    result=compare_models(source('A',pair['A']),source('other-label',pair['A']),features=('P01','P02','P11')).to_dict()
    assert all(r['status']=='INCOMPATIBLE' and r['difference'] is None for r in result['metrics'])


def test_total_forward_budget_rejects_before_measurement(monkeypatch):
    from hyso.benchmark import BenchmarkRunner
    monkeypatch.setattr(BenchmarkRunner,'run',lambda *a,**kw:pytest.fail('budget bypass'))
    with pytest.raises(ValueError,match='total_forward_budget'):
        capture_cpu_comparison(model(),inputs(),latency_iterations=4096,repeats=4096)


def test_throughput_queue_protocol_difference_is_preserved(pair):
    other=capture_cpu_comparison(model(),inputs(),features=('P02',),throughput_protocol=ThroughputProtocol(
        warmup_iterations=2,batch_count=1000,repeats=1,queue_depth=2))
    result=compare_models(source('A',{'P02':pair['A']['P02']}),source('B',other),features='P02').to_dict()
    assert result['metrics'][0]['status']=='INCOMPATIBLE'


def test_wrong_artifact_does_not_gain_checkpoint_association(tmp_path):
    m=model();checkpoint=tmp_path/'model.pth';foreign=tmp_path/'other.pth'
    save_training_checkpoint(checkpoint,m,include_rng=False);save_training_checkpoint(foreign,model(3),include_rng=False)
    load_training_checkpoint(checkpoint,m,capture_evidence=True)
    a=capture_cpu_comparison(m,inputs(),features=('P11',),footprint_artifact_path=checkpoint)
    b=capture_cpu_comparison(m,inputs(),features=('P11',),footprint_artifact_path=foreign)
    result=compare_models(source('A',a),source('B',b),features='P11').to_dict()
    assert result['metrics'][-1]['status']=='INSUFFICIENT_EVIDENCE'


def test_observed_revision_clock_cohort_and_explicit_unknowns(pair):
    import sys
    from pathlib import Path
    for values in pair.values():
        for feature,value in values.items():
            raw=value.to_dict();proof=raw['provenance']['comparison_capture'];o=proof['observation']
            assert proof['schema_version']==3
            for field in ('source','model_source'):
                assert o[field]['status']=='observed'
                for module,sha in o[field]['files'].items():
                    assert hashlib.sha256(Path(sys.modules[module].__file__).read_bytes()).hexdigest()==sha
                assert o[field]['revision_sha256']==_digest(o[field]['files'])
            assert o['clock']['monotonic'] and o['clock']['name']=='perf_counter_ns'
            assert o['hardware']['status']=='observed'
            assert o['started_unix_ns']>0 and o['finished_unix_ns']>0
            assert proof['end_ns']>=proof['start_ns']
            assert o['cohort']['scope']=='ordered_resident_input_batch_only'
            assert o['cohort']['input_sha256']==proof['bound_input']['sha256']
            assert o['cohort']['role']==('bound_workload_context_only' if feature=='P11' else 'measured_resident_input')
            assert o['precision']=='fp32' and o['protocol_sha256']==_digest(raw['protocol'])
            assert o['missing_fields']==['dataset_origin','preprocessing']


@pytest.mark.parametrize('field',['source','clock','hardware','cohort','protocol','precision',
    'preprocessing','dataset_origin','missing_fields','timestamp'])
def test_resigned_inconsistent_observation_is_rejected(pair,field):
    value=pair['A']['P11'];raw=value.to_dict();proof=raw['provenance']['comparison_capture'];o=proof['observation']
    if field=='source':o['source']['revision_sha256']='a'*64
    elif field=='clock':o['clock']['identity_sha256']='b'*64
    elif field=='hardware':o['hardware']['identity_sha256']='c'*64
    elif field=='cohort':o['cohort']['input_sha256']='d'*64
    elif field=='protocol':o['protocol_sha256']='e'*64
    elif field=='precision':o['precision']='fp64'
    elif field in ('preprocessing','dataset_origin'):o[field]={'status':'observed','sha256':'f'*64}
    elif field=='missing_fields':o['missing_fields']=[]
    else:o['started_unix_ns']=True
    proof['receipt_sha256']=_digest({k:v for k,v in proof.items() if k!='receipt_sha256'})
    with pytest.raises(ValueError):type(value)(raw)


@pytest.mark.parametrize('field',['source','clock','hardware'])
def test_recorded_source_clock_hardware_mismatch_is_incompatible(pair,field):
    import hyso.performance.comparison_capture as module
    original=pair['B']['P11'];raw=original.to_dict();proof=raw['provenance']['comparison_capture'];o=proof['observation']
    if field=='source':
        key=next(iter(o['source']['files']));o['source']['files'][key]='a'*64
        o['source']['revision_sha256']=_digest(o['source']['files'])
    elif field=='clock':
        o['clock']['implementation']='other_monotonic_clock'
        o['clock']['identity_sha256']=_digest({k:v for k,v in o['clock'].items() if k!='identity_sha256'})
    else:
        proof['environment']['host']='other-host'
        o['hardware']=module._hardware_identity(proof['environment'])
    proof['receipt_sha256']=_digest({k:v for k,v in proof.items() if k!='receipt_sha256'})
    other=type(original)(raw)
    result=compare_models(source('A',{'P11':pair['A']['P11']}),source('B',{'P11':other}),features='P11').to_dict()
    assert all(r['status']=='INCOMPATIBLE' and r['difference'] is None for r in result['metrics'])


def legacy_receipt(value):
    """In-memory schema-2 fixture, never an upgrade or an edit to retained evidence."""
    raw=value.to_dict();proof=raw['provenance']['comparison_capture'];proof.pop('observation')
    proof.update(schema_version=2,producer='hyso.performance.capture_cpu_comparison.v1')
    proof['receipt_sha256']=_digest({k:v for k,v in proof.items() if k!='receipt_sha256'})
    return type(value)(raw)


def test_schema2_reader_and_original_comparison_policy_remain_supported(pair):
    a={k:legacy_receipt(v) for k,v in pair['A'].items()}
    b={k:legacy_receipt(v) for k,v in pair['B'].items()}
    result=compare_models(source('A',a),source('B',b),features=('P01','P02','P11')).to_dict()
    assert all(r['status']=='COMPARABLE' for r in result['metrics'])


@pytest.mark.parametrize('reverse',[False,True])
def test_mixed_schema2_and_schema3_never_invent_new_evidence(pair,reverse):
    a=source('A',{k:legacy_receipt(v) for k,v in pair['A'].items()});b=source('B',pair['B'])
    result=compare_models(b,a,features=('P01','P02','P11')) if reverse else compare_models(a,b,features=('P01','P02','P11'))
    assert all(r['status']!='COMPARABLE' and r['difference'] is None for r in result.to_dict()['metrics'])


def test_unavailable_source_is_explicit_and_never_comparable(monkeypatch):
    import hyso.performance.comparison_capture as module
    def unavailable(modules):
        files={name:None for name in sorted(set(modules))}
        return dict(status='missing',scope='installed_file_snapshot_not_loaded_code_attestation',
            files=files,revision_sha256=_digest(files))
    monkeypatch.setattr(module,'_source_identity',unavailable)
    a=capture_cpu_comparison(model(),inputs(),features=('P11',));b=capture_cpu_comparison(model(3),inputs(),features=('P11',))
    o=a['P11'].to_dict()['provenance']['comparison_capture']['observation']
    assert o['missing_fields']==['checkpoint','dataset_origin','model_source','preprocessing','source']
    result=compare_models(source('A',a),source('B',b),features='P11').to_dict()
    assert all(r['status']=='INSUFFICIENT_EVIDENCE' and r['difference'] is None for r in result['metrics'])


@pytest.mark.parametrize('field',['source','clock'])
def test_source_or_clock_change_during_capture_publishes_nothing(monkeypatch,field):
    import hyso.performance.comparison_capture as module
    from hyso.benchmark.runner import StateMutationError
    name='_source_identity' if field=='source' else '_clock_identity';real=getattr(module,name);calls=0
    def changed(*args):
        nonlocal calls
        calls+=1;value=real(*args)
        if calls>(2 if field=='source' else 1):value=dict(value,changed=True)
        return value
    monkeypatch.setattr(module,name,changed)
    with pytest.raises(StateMutationError,match='no result published'):
        capture_cpu_comparison(model(),inputs(),features=('P11',))


def test_all_new_CPU_producers_do_not_query_CUDA(monkeypatch):
    def deny(*a,**kw):pytest.fail('CUDA query/initialization reached')
    for name in ('is_available','_lazy_init','device_count','current_device'):
        monkeypatch.setattr(torch.cuda,name,deny)
    values=capture_cpu_comparison(model(),inputs(),latency_iterations=8,
        throughput_protocol=ThroughputProtocol(batch_count=3,repeats=1))
    assert set(values)=={'P01','P02','P11'}


def test_missing_hardware_is_explicit_and_withholds_comparison(monkeypatch):
    import hyso.performance.comparison_capture as module
    real=module._environment
    monkeypatch.setattr(module,'_environment',lambda:dict(real(),host=''))
    a=capture_cpu_comparison(model(),inputs(),features=('P11',));b=capture_cpu_comparison(model(3),inputs(),features=('P11',))
    o=a['P11'].to_dict()['provenance']['comparison_capture']['observation']
    assert o['hardware']['status']=='missing' and 'hardware' in o['missing_fields']
    result=compare_models(source('A',a),source('B',b),features='P11').to_dict()
    assert all(r['status']=='INSUFFICIENT_EVIDENCE' and r['difference'] is None for r in result['metrics'])


@pytest.mark.parametrize('feature',['P01','P02'])
def test_both_short_captures_cannot_bypass_default_sample_requirement(feature):
    kwargs=dict(features=(feature,),latency_iterations=999,
        throughput_protocol=ThroughputProtocol(batch_count=999,repeats=1))
    a=capture_cpu_comparison(model(),inputs(),**kwargs);b=capture_cpu_comparison(model(3),inputs(),**kwargs)
    result=compare_models(source('A',a),source('B',b),features=feature).to_dict()
    assert all(r['status']=='INSUFFICIENT_EVIDENCE' and r['difference'] is None for r in result['metrics'])
