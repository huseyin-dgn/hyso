"""Real CPU P12/P13/P14 observations and analytic reference counts."""
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from hyso.performance import (ColdStartRecipe,ProfilingProtocol,ProfilingError,
    OperatorProfileResult,LayerProfileResult,OperationCountResult,
    profile_operators,profile_layers,analyze_flops)


def recipe(factory='spatial',shape=(2,2,5,7),precision='fp32',**kwargs):
    return ColdStartRecipe('profiling_factories:'+factory,factory_search_path=Path(__file__).parent,
                           input_shape=shape,precision=precision,model_kwargs=kwargs)


def protocol(**kw):
    return ProfilingProtocol(**dict(dict(warmup_iterations=1,measure_iterations=2,timeout_seconds=45,
        cpu_memory_limit_bytes=3*1024**3,input_limit_bytes=32*1024**2),**kw))


@pytest.fixture(scope='module')
def captured(tmp_path_factory):
    path=tmp_path_factory.mktemp('profile')/'trace.json'
    return profile_operators(recipe('Shared',unused=True),protocol=protocol(),trace_path=path)


@pytest.mark.parametrize('groups,bias',[(1,False),(1,True),(2,False),(2,True)])
def test_actual_convolution_counts(groups,bias):
    r=analyze_flops(recipe(groups=groups,bias=bias),protocol=protocol()).to_dict()
    expected=2*4*5*7*(2//groups)*9
    m=r['measurements']
    assert m['total_macs_per_batch']==expected
    assert m['total_flops_per_batch']==2*expected+(2*4*5*7 if bias else 0)
    assert m['total_flops_per_sample']==m['total_flops_per_batch']/2
    assert m['status']=='complete' and m['coverage']['unknown_calls']==0


def test_conv1d_depthwise():
    m=analyze_flops(recipe('conv1'),protocol=protocol()).to_dict()['measurements']
    assert m['total_macs_per_batch']==2*4*35*3


@pytest.mark.parametrize('shape',[(1,2,3,4),(3,2,8,9)])
def test_batch_resolution_scoped_counts(shape):
    m=analyze_flops(recipe(shape=shape,bias=False),protocol=protocol()).to_dict()['measurements']
    expected=shape[0]*4*shape[2]*shape[3]*2*9
    assert m['total_macs_per_batch']==expected
    assert len(m['per_forward'])==2 and all(r['total_flops']==2*expected for r in m['per_forward'])


@pytest.mark.parametrize('kind',['linear','mm','matmul','bmm','addmm'])
def test_matrix_models(kind):
    d=analyze_flops(recipe('Matrix',shape=(2,1,5,7),kind=kind),protocol=protocol()).to_dict()
    m=d['measurements'];macs=2*5*5*7 if kind=='bmm' else 2*3*35
    assert m['total_macs_per_batch']==macs
    assert m['total_flops_per_batch']==2*macs+(6 if kind in ('linear','addmm') else 0)


def test_unknown_is_not_zero_and_normalization_explicit():
    d=analyze_flops(recipe('Matrix',shape=(2,1,5,7),kind='unknown'),protocol=protocol()).to_dict()
    assert d['measurements']['status']=='partial'
    assert d['measurements']['total_flops_per_batch'] is None
    assert any(r['flops'] is None and r['reason'] for r in d['raw_evidence']['operations'])
    d=analyze_flops(recipe('Norm'),protocol=protocol()).to_dict()
    assert d['measurements']['status']=='partial'
    bn=[r for r in d['raw_evidence']['operations'] if 'batch_norm' in r['operator']]
    assert len(bn)==2 and all(r['flops']==2*2+4*(2*2*5*7) for r in bn)
    assert any('silu' in r['operator'] for r in d['raw_evidence']['operations'] if r['flops'] is None)


@pytest.mark.parametrize('precision',['fp32','fp64'])
@pytest.mark.parametrize('shape,spatial_dims,affine',[
    ((2,3,2,4),1,False),((3,2,3,5),2,True)])
def test_batch_norm_inference_reference_cpu(precision,shape,spatial_dims,affine,tmp_path):
    source=profile_operators(recipe('Norm',shape=shape,precision=precision,channels=shape[1],
        spatial_dims=spatial_dims,affine=affine,activation=False),protocol=protocol())
    d=analyze_flops(source).to_dict();m=d['measurements']
    expected=2*shape[1]+np.prod(shape)*(2+2*int(affine))
    assert m['status']=='complete' and m['total_macs_per_batch']==0
    assert m['total_flops_per_batch']==expected and m['total_flops_per_sample']==expected/shape[0]
    assert m['supported_subtotal_flops']==2*expected
    assert all(r['total_flops']==expected for r in m['per_forward'])
    bn=[r for r in d['raw_evidence']['operations'] if 'batch_norm' in r['operator']]
    assert len(bn)==2 and all(r['flops']==expected for r in bn)
    assert all(r['arguments']['input']['dtype']=='torch.'+('float64' if precision=='fp64' else 'float32') for r in bn)
    assert any('batch_norm' in r['name'] for r in source.to_dict()['raw_evidence']['events'])
    result=OperationCountResult(d);path=tmp_path/'bn.json'
    result.to_json(path);result.to_csv(tmp_path/'bn.csv')
    assert OperationCountResult(json.loads(path.read_text())).to_dict()==d
    assert profile_layers(source).to_dict()['measurements']['status']=='measured'
    from hyso.performance._operation_models import summarize
    old=json.loads(path.read_text());old['protocol'].pop('operation_model_version')
    old['raw_evidence']['operations'],old['measurements']=summarize(
        old['raw_evidence']['aten_calls'],2,shape[0],model_version=1)
    assert OperationCountResult(old).to_dict()['measurements']['status']=='partial'


@pytest.mark.parametrize('precision',['fp32','fp64'])
@pytest.mark.parametrize('kind,keepdim',[('sum',False),('mean',True)])
def test_reduction_reference_cpu(precision,kind,keepdim):
    d=analyze_flops(recipe('Reduction',precision=precision,kind=kind,keepdim=keepdim),protocol=protocol()).to_dict()
    expected=2*2*(5*7-1+int(kind=='mean'))
    assert d['measurements']['status']=='complete'
    assert d['measurements']['total_flops_per_batch']==expected
    assert d['measurements']['total_flops_per_sample']==expected/2
    assert len(d['raw_evidence']['operations'])==2


def test_pooling_activation_policy():
    d=analyze_flops(recipe('pooled'),protocol=protocol()).to_dict()
    assert d['measurements']['total_macs_per_batch']==0
    assert d['measurements']['total_flops_per_batch']==2*2*5*7
    assert d['measurements']['comparisons_per_batch']==2*2*5*7


@pytest.mark.parametrize('factory,flops,comparisons',[('fixed_pool',0,72),('adaptive_pool',216,0)])
def test_nontrivial_pool_windows(factory,flops,comparisons):
    m=analyze_flops(recipe(factory),protocol=protocol()).to_dict()['measurements']
    assert m['total_flops_per_batch']==flops and m['comparisons_per_batch']==comparisons


def test_custom_namespace_not_misidentified_as_linear():
    d=analyze_flops(recipe('Custom',shape=(2,1,5,7)),protocol=protocol()).to_dict()
    assert d['measurements']['total_flops_per_batch'] is None
    assert any(r['operator'].startswith('hyso_profile_custom.linear') and r['flops'] is None
               for r in d['raw_evidence']['operations'])


def test_shared_hierarchy_repeated_calls_and_exclusive(captured):
    d=profile_layers(captured).to_dict();rows=d['raw_evidence']['invocations']
    modules=d['measurements']['modules'];proj=next(m for m in modules if m['path']=='proj')
    assert proj['invocation_count']==6 and set(proj['aliases'])=={'proj','alias','branch.1'}
    assert len(rows)==12
    for row in rows:
        children=[r for r in rows if r['parent_invocation_id']==row['invocation_id']]
        assert row['cpu_total_us']>=sum(c['cpu_total_us'] for c in children)
        assert row['cpu_exclusive_us']==row['cpu_total_us']-sum(c['cpu_total_us'] for c in children)
        assert row['input_shapes'] and row['output_shapes']
        assert row['cuda_activity_us'] is None
    assert d['measurements']['ranking_basis']=='cpu_exclusive_us_instrumented'
    assert d['identity']['source_result_id']==captured.to_dict()['result_id']
    counts=analyze_flops(captured).to_dict()['measurements']
    assert counts['total_macs_per_batch']==840 and counts['total_flops_per_batch']==1820


def test_real_cpu_events_trace_and_cleanup(captured):
    d=captured.to_dict();ops=d['measurements']['operators']
    assert any(r['name']=='aten::conv2d' and r['count']>=6 for r in ops)
    assert any(r['self_cpu_us']>0 for r in ops)
    assert d['measurements']['cuda_activity']['status']=='not_requested'
    assert d['measurements']['cuda_activity']['total_kernel_us'] is None
    assert all(d['raw_evidence']['safety'].values())
    path=Path(d['raw_evidence']['trace']['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==d['raw_evidence']['trace']['sha256']
    assert json.loads(path.read_text())['traceEvents']
    assert d['measurements']['overhead']['perturbation_ms'] is None


def test_operator_self_total_counts_and_csv_preserve_cpu_evidence(captured,tmp_path):
    import csv
    import math
    d=captured.to_dict();events=d['raw_evidence']['events']
    for row in d['measurements']['operators']:
        observed=[e for e in events if e['device_type']=='CPU' and e['scope']==0 and e['name']==row['name']]
        assert observed and row['count']==len(observed)
        assert row['cpu_total_us']==math.fsum(e['cpu_total_us'] for e in observed)
        assert row['self_cpu_us']==math.fsum(e['self_cpu_us'] for e in observed)
        assert 0<=row['self_cpu_us']<=row['cpu_total_us']
        assert row['shapes']==[e['input_shapes'] for e in observed]
        assert row['status']=='measured' and row['unit']=='us'
    path=tmp_path/'operators.csv';captured.to_csv(path)
    with path.open(newline='',encoding='utf-8') as stream:
        rows=list(csv.DictReader(stream))
    assert [json.loads(r['payload']) for r in rows]==d['measurements']['operators']
    assert all(r['row_kind']=='operator_profile' for r in rows)


def test_layer_event_evidence_cpu_unused_memory_and_shared_capture(captured):
    source=captured.to_dict();layers=profile_layers(captured).to_dict();raw=layers['raw_evidence']
    assert source['schema_version']==layers['schema_version']==1
    assert source['protocol']['instrumentation']['evidence_version']==3
    assert layers['protocol']['profile_evidence_version']==3
    assert layers['identity']['source_result_id']==source['result_id']
    assert raw['events']==source['raw_evidence']['events'] and raw['trace_events']==source['raw_evidence']['trace_events']
    assert raw['profile_measurements']['cuda_attribution']==source['measurements']['cuda_attribution']
    events={e['event_id']:e for e in raw['events'] if e['device_type']=='CPU'}
    for invocation in raw['invocations']:
        e=events[invocation['cpu_event_id']]
        assert e['name']==invocation['annotation']
        assert invocation['cpu_total_us']==pytest.approx(e['end_us']-e['start_us'])
        assert invocation['cpu_timing_reason'] is None
        assert invocation['cuda_elapsed_us'] is None and invocation['cuda_elapsed_status']=='not_requested'
        assert invocation['cuda_exclusive_elapsed_us'] is None and invocation['cuda_activity_us'] is None
    unused=next(m for m in layers['measurements']['modules'] if m['path']=='unused')
    assert unused['status']=='not_executed' and unused['cpu_total_us'] is None and unused['invocation_count']==0
    memory=layers['measurements']['layer_memory']
    assert memory['status']=='unsupported' and all(memory[k] is None for k in ('allocation_bytes','retained_bytes','peak_bytes'))
    assert layers['measurements']['overhead']['perturbation_ms'] is None
    assert layers['measurements']['cuda_attribution']['mapped_kernel_count']==0
    assert layers['measurements']['cuda_event_timing']['status']=='not_requested'
    assert source['raw_evidence']['cuda_event_timing']['pairs']==[]


def test_current_rejects_module_event_and_correlation_tamper(captured):
    import copy
    d=captured.to_dict()
    for mutate in (
        lambda p:p['raw_evidence']['events'].append(copy.deepcopy(p['raw_evidence']['events'][0])),
        lambda p:p['raw_evidence']['events'][0].update(parent_id=999999),
        lambda p:p['raw_evidence']['invocations'][1].update(parent_invocation_id=None),
        lambda p:p['raw_evidence']['invocations'][1].update(cpu_total_us=1),
        lambda p:p['raw_evidence']['trace_events'].append(copy.deepcopy(p['raw_evidence']['trace_events'][0])),
        lambda p:p['measurements']['cuda_attribution'].update(mapped_kernel_count=1),
        lambda p:p['raw_evidence']['trace'].update(timestamp_unit='ns'),
        lambda p:p['raw_evidence']['cuda_event_timing'].update(requested=True),
        lambda p:p['raw_evidence']['invocations'][0].update(cuda_elapsed_us=1),
    ):
        broken=copy.deepcopy(d);mutate(broken)
        with pytest.raises(ValueError):OperatorProfileResult(broken)
    layer=profile_layers(captured).to_dict()
    layer['raw_evidence']['invocations'][1]['cpu_total_us']=1
    with pytest.raises(ValueError):LayerProfileResult(layer)


@pytest.mark.parametrize('version',[1,2])
def test_legacy_layer_evidence_reopen(captured,version,tmp_path):
    old=captured.to_dict();old['raw_evidence'].pop('cuda_event_timing')
    fields=('cuda_elapsed_us','cuda_elapsed_status','cuda_elapsed_reason','cuda_exclusive_elapsed_us','cuda_exclusive_elapsed_reason')
    for r in old['raw_evidence']['invocations']:
        for key in fields:r.pop(key)
    if version==1:
        old['protocol']['instrumentation'].pop('evidence_version')
        old['measurements'].pop('cuda_attribution');old['raw_evidence'].pop('trace_events')
        for r in old['raw_evidence']['invocations']:
            r.pop('cpu_event_id');r.pop('cpu_timing_reason')
        for r in old['raw_evidence']['events']:
            for key in ('start_us','end_us','thread_id'):r.pop(key)
    else:old['protocol']['instrumentation']['evidence_version']=2
    legacy=OperatorProfileResult(old);assert legacy.to_dict()==old
    layer=profile_layers(legacy);path=tmp_path/'legacy.json'
    layer.to_json(path);layer.to_csv(tmp_path/'legacy.csv')
    assert LayerProfileResult(json.loads(path.read_text())).to_dict()==layer.to_dict()
    assert 'cuda_event_timing' not in layer.to_dict()['measurements']
    assert analyze_flops(legacy).to_dict()['measurements']['total_flops_per_batch']==1820


@pytest.mark.parametrize('mode',['fail','mutate'])
def test_forward_exception_state_and_hook_cleanup(mode):
    with pytest.raises(ProfilingError) as error:
        profile_operators(recipe('Shared',**{mode:True}),protocol=protocol())
    d=error.value.diagnostics
    assert d['status']==('state_mutation' if mode=='mutate' else 'failed')
    if mode=='fail':assert d['reason']=='RuntimeError: controlled forward failure'
    assert d['process_terminated'] and d['exit_code']==0
    assert d['cleanup']['hooks_preserved'] and d['cleanup']['rng_restored']
    r=profile_operators(recipe('pooled'),protocol=protocol())
    assert r.to_dict()['actual']['measured_forwards']==2


def test_oom_exception_is_exposed_with_worker_cleanup(tmp_path):
    # Inject the exception type; deliberately exhausting real RAM/VRAM is unnecessary.
    (tmp_path/'operator_oom_factory.py').write_text('''
import torch
class Model(torch.nn.Module):
    def forward(self,x):
        raise torch.OutOfMemoryError('controlled operator OOM')
def build():return Model()
''',encoding='utf-8')
    source=ColdStartRecipe('operator_oom_factory:build',factory_search_path=tmp_path,input_shape=(1,1,1,1))
    with pytest.raises(ProfilingError) as error:
        profile_operators(source,protocol=ProfilingProtocol(warmup_iterations=0,measure_iterations=1,timeout_seconds=45))
    d=error.value.diagnostics
    assert d['status']=='failed' and d['reason']=='OutOfMemoryError: controlled operator OOM'
    assert d['process_terminated'] and d['exit_code']==0 and all(d['cleanup'].values())


def test_timeout_and_parent_state():
    model=torch.nn.Linear(2,2);model.weight.grad=torch.ones_like(model.weight)
    hook=model.register_forward_hook(lambda m,a,o:None)
    refs=dict(model._forward_hooks);params=[p.clone() for p in model.parameters()]
    rng=torch.get_rng_state().clone();py=random.getstate();n=np.random.get_state();cuda=torch.cuda.is_initialized()
    with pytest.raises(ProfilingError) as error:
        profile_operators(recipe(),protocol=protocol(timeout_seconds=.01))
    assert error.value.diagnostics['status']=='timed_out'
    assert error.value.diagnostics['process_terminated'] and error.value.diagnostics['exit_code'] is not None
    assert dict(model._forward_hooks)==refs and model.training
    assert all(torch.equal(p,q) for p,q in zip(model.parameters(),params))
    assert torch.equal(model.weight.grad,torch.ones_like(model.weight)) and model.bias.grad is None
    assert torch.equal(torch.get_rng_state(),rng) and random.getstate()==py
    assert np.array_equal(np.random.get_state()[1],n[1]) and torch.cuda.is_initialized()==cuda
    hook.remove()


@pytest.mark.parametrize('kwargs',[dict(measure_iterations=0),dict(warmup_iterations=-1),
    dict(timeout_seconds=0),dict(max_events=0),dict(cpu_memory_limit_bytes=1),
    dict(cuda_memory_fraction=0),dict(cuda_memory_fraction=2)])
def test_invalid_protocol(kwargs):
    with pytest.raises((ValueError,TypeError)): ProfilingProtocol(**kwargs)


def test_immutable_exports_and_tamper(captured,tmp_path):
    for result in (captured,profile_layers(captured),analyze_flops(captured)):
        cls=type(result);d=result.to_dict()
        with pytest.raises((FrozenInstanceError,AttributeError)):result._payload='bad'
        path=tmp_path/(d['result_id']+'.json');result.to_json(path);result.to_csv(path.with_suffix('.csv'))
        assert cls(json.loads(path.read_text())).to_dict()==d
        with pytest.raises(FileExistsError):result.to_json(path)
        with pytest.raises(FileExistsError):result.to_csv(path.with_suffix('.csv'))
        d['actual']['measured_forwards']=999
        with pytest.raises(ValueError):cls(d)
    d=captured.to_dict();d['measurements']['operators'][0]['self_cpu_us']=float('nan')
    with pytest.raises(ValueError):OperatorProfileResult(d)


def test_operation_model_revision_legacy_reopen_and_tamper(captured,tmp_path):
    import copy
    from hyso.performance._operation_models import summarize
    d=analyze_flops(captured).to_dict()
    assert d['protocol']['operation_model_version']==2 and d['schema_version']==1
    old=copy.deepcopy(d);old['protocol'].pop('operation_model_version')
    old['raw_evidence']['operations'],old['measurements']=summarize(
        old['raw_evidence']['aten_calls'],2,2,model_version=1)
    legacy=OperationCountResult(old);legacy.to_json(tmp_path/'legacy.json');legacy.to_csv(tmp_path/'legacy.csv')
    assert OperationCountResult(json.loads((tmp_path/'legacy.json').read_text())).to_dict()==old
    for mutate in (
        lambda p:p['protocol'].update(operation_model_version=999),
        lambda p:p['protocol'].update(operation_model_version=True),
        lambda p:p['raw_evidence']['aten_calls'][0].update(measurement_index=2),
        lambda p:p['raw_evidence']['aten_calls'][0].update(call_index=3),
        lambda p:p['raw_evidence']['aten_calls'][0].update(arguments={}),
        lambda p:p['raw_evidence']['operations'][0].update(flops=0),
        lambda p:p['protocol']['convention'].update(fma_flops=1),
        lambda p:p.update(schema_version=2),
    ):
        broken=copy.deepcopy(d);mutate(broken)
        with pytest.raises(ValueError):OperationCountResult(broken)


def test_no_clobber_trace(captured):
    with pytest.raises(FileExistsError):
        profile_operators(recipe(),protocol=protocol(),trace_path=captured.to_dict()['raw_evidence']['trace']['path'])


def test_evidence_budget_failure_cleanup():
    with pytest.raises(ProfilingError) as error:
        profile_operators(recipe(),protocol=protocol(max_events=1))
    assert 'budget' in error.value.diagnostics['reason']
    assert error.value.diagnostics['process_terminated'] and error.value.diagnostics['cleanup']['hooks_preserved']


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1,None])
def test_invalid_event_duration_not_zero(value):
    from hyso.performance._profile_worker import _duration
    assert _duration(value) is None


def test_missing_activity_and_legacy_labels_are_not_kernels():
    from hyso.performance.profiling import _cuda_metrics
    m=_cuda_metrics([],True)
    assert m['status']=='unsupported' and m['total_kernel_us'] is None and m['kernel_count'] is None
    with pytest.raises(ValueError):
        _cuda_metrics([dict(category='cpu_op',phase='X',name='aten::conv2d',duration_us=1)],True)


def test_experiment_reopen(captured,tmp_path):
    from hyso.experiments import ExperimentManager
    root=tmp_path/'runs';m=ExperimentManager(root);ex=m.create_experiment('profile');m.close()
    ExperimentManager.migrate_store(root);m=ExperimentManager(root)
    run=m.create_run(ex['id'],'benchmark',config={'feature':'P14'});run.start()
    path=tmp_path/'profile.json';captured.to_json(path)
    artifacts=[run.add_artifact(p,kind='performance_result',ownership='copy') for p in
        (path,Path(captured.to_dict()['raw_evidence']['trace']['path']))]
    logged=[]
    for i,row in enumerate(captured.to_dict()['measurements']['operators']):
        if row['self_cpu_us'] is None:
            assert row['status']=='partial'
            # The complete partial observation remains in the copied artifact.
            # Unsupported timing is not a numeric metric or a zero duration.
            continue
        run.log_metric(name='profile.cpu_self',value=row['self_cpu_us'],unit='us',step=i,
                       step_kind='observation',source='performance.operator_profile_v1')
        logged.append((i,row['self_cpu_us']))
    rid=run.id;run.finish();m.close();m=ExperimentManager(root)
    metrics=m.get_run(rid).read_metrics()
    assert logged and [(r['axis_index'],r['value']) for r in metrics]==logged
    assert all(r['axis_kind']=='observation' for r in metrics)
    assert all(m.verify_artifact(a['id'])['status']=='verified' for a in artifacts)
    m.close()
