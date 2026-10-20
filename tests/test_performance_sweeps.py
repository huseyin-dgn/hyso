"""Actual finite CPU sweeps, precision execution and ownership contracts."""
import copy
from dataclasses import FrozenInstanceError
import gc
import json
from pathlib import Path
import random
import re
import weakref

import numpy as np
import pytest
import torch

from hyso.performance import SweepProtocol, SweepResult, PrecisionResult, run_sweep
from hyso.performance import sweeps as engine
from hyso.benchmark import BenchmarkResult, compare_results
from hyso.benchmark.runner import StateMutationError


def factory(shape,dtype,device):
    return torch.randn(shape,dtype=dtype,device=device)


def linear():
    return torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(35,8),torch.nn.ReLU(),torch.nn.Linear(8,2))


def spatial():
    return torch.nn.Sequential(torch.nn.Conv2d(1,2,3,padding=1),torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten())


def plan(**kwargs):
    return SweepProtocol(warmup_iterations=1,measure_iterations=3,repeats=2,**kwargs)


@pytest.mark.parametrize('precision',['fp32','fp64'])
def test_actual_batches_cardinality_warmup(precision):
    result=run_sweep(linear(),factory,axis='batch',points=[1,2,4,8],protocol=plan(precision=precision,sustained_batches=3))
    rows=result.to_dict()['raw_evidence']['points']
    assert result.to_dict()['actual']['measured']==4 and len({r['result_id'] for r in rows})==4
    for point,batch in zip(rows,[1,2,4,8]):
        data=point['result']; raw=data['raw_evidence']
        assert point['shape']==[batch,1,5,7] and data['measurements']['batch_cardinality']==batch
        assert data['actual']['samples']==6 and raw['benchmark']['configuration']['warmup_iterations']==1
        assert raw['throughput']['actual']['completed_images']==6*batch
        assert raw['throughput']['actual']['pending_batches']==0
    a,b=[BenchmarkResult(r['result']['raw_evidence']['benchmark']) for r in rows[:2]]
    assert compare_results(a,b)['status']=='incompatible'


def test_rectangular_shapes_exact_factory_fixed_weights():
    seen=[]
    def source(s,d,v): seen.append(s);return factory(s,d,v)
    result=run_sweep(spatial(),source,axis='resolution',points=[(5,7),(8,8),(9,13)],input_shape=(3,1,5,7),protocol=plan())
    rows=result.to_dict()['raw_evidence']['points']
    assert seen==[(3,1,5,7),(3,1,8,8),(3,1,9,13)] and all(r['status']=='measured' for r in rows)
    assert len({r['result']['provenance']['model']['source_weights_sha256'] for r in rows})==1
    assert rows[-1]['aspect_ratio']==13/9


@pytest.mark.parametrize('axis,points',[
    ('bad',[1]),('batch',[]),('batch',[0]),('batch',[True]),('batch',[1,1]),('batch',[1.5]),
    ('resolution',[(3,)]),('resolution',[(3,0)]),('resolution',[(3,4),(3,4)]),
    ('precision',['half']),('precision',['fp32','fp32'])])
def test_invalid_points(axis,points):
    with pytest.raises((ValueError,TypeError)): run_sweep(linear(),factory,axis=axis,points=points,protocol=plan())


@pytest.mark.parametrize('options',[dict(warmup_iterations=-1),dict(measure_iterations=0),dict(repeats=0),
    dict(sustained_batches=0),dict(queue_depth=1),dict(device='mps'),dict(precision='half'),dict(continue_on_error=1)])
def test_invalid_protocol(options):
    with pytest.raises((ValueError,TypeError)): SweepProtocol(**options)


@pytest.mark.parametrize('continuation',[False,True])
@pytest.mark.parametrize('kind',['invalid','runtime','oom','unsupported'])
def test_failure_continuation_oom_stops(kind,continuation):
    def source(s,d,v):
        if s[0]==2:
            if kind=='invalid':return factory((1,1,5,7),d,v)
            if kind=='oom':raise torch.cuda.OutOfMemoryError('injected protocol test, not physical OOM')
            if kind=='unsupported':raise NotImplementedError('shape unsupported')
            raise RuntimeError('workload error')
        return factory(s,d,v)
    rows=run_sweep(linear(),source,axis='batch',points=[1,2,4],protocol=plan(continue_on_error=continuation)).to_dict()['raw_evidence']['points']
    assert rows[0]['status']=='measured' and rows[1]['status']==('unsupported' if kind=='unsupported' else 'failed')
    assert rows[1]['result'] is None and rows[1]['reason']
    assert rows[2]['status']==('measured' if continuation and kind!='oom' else 'not_run')


def test_invalid_model_shape_not_resized():
    data=run_sweep(linear(),factory,axis='resolution',points=[(4,4),(5,7)],protocol=plan()).to_dict()
    assert [r['status'] for r in data['raw_evidence']['points']]==['failed','measured']


def test_cpu_bf16_real_dynamic_int8_operator():
    rows=run_sweep(linear(),factory,axis='precision',points=['fp32','fp64','bf16','int8_dynamic','amp_fp16'],protocol=plan(sustained_batches=3)).to_dict()['raw_evidence']['points']
    assert [r['status'] for r in rows]==['measured']*4+['unsupported']
    assert rows[2]['result']['provenance']['precision']['output_dtype']=='torch.bfloat16'
    proof=rows[3]['result']['provenance']['precision']
    assert proof['execution_proof']['operators']==['quantized::linear_dynamic'] and proof['execution_proof']['quantized_linear_count']==2
    assert proof['calibration']['status']=='not_applicable' and proof['accuracy_status']=='not_measured_P28'
    assert proof['copy_conversion_ms']>0 and proof['load_cost_ms'] is None
    assert rows[3]['result']['measurements']['sustained_images_per_second']['value']>0


def test_bf16_fallback_nonquantized_graph_unsupported():
    a=run_sweep(torch.nn.Identity(),factory,axis='precision',points=['bf16','int8_dynamic'],protocol=plan()).to_dict()
    assert a['actual']['unsupported']==2


def test_quantization_backend_unavailable(monkeypatch):
    monkeypatch.setattr(torch.backends.quantized,'engine','none')
    assert run_sweep(linear(),factory,axis='precision',points=['int8_dynamic'],protocol=plan()).to_dict()['actual']['unsupported']==1


def test_caller_state_gradients_modes_rng_input():
    model=linear(); model[1].eval(); model[1].weight.grad=torch.ones_like(model[1].weight)
    before=copy.deepcopy(model.state_dict()); modes=[m.training for m in model.modules()]
    gradient=model[1].weight.grad; saved=gradient.clone()
    py=random.getstate(); npstate=np.random.get_state(); cpu=torch.get_rng_state(); tensor=torch.ones(1,1,5,7)
    def source(s,d,v): random.random();np.random.rand();torch.rand(1);return tensor
    assert run_sweep(model,source,axis='precision',points=['fp32','bf16','int8_dynamic'],protocol=plan()).to_dict()['actual']['measured']==3
    assert all(torch.equal(model.state_dict()[k],v) for k,v in before.items()) and [m.training for m in model.modules()]==modes
    assert model[1].weight.grad is gradient and torch.equal(gradient,saved)
    assert random.getstate()==py and np.array_equal(np.random.get_state()[1],npstate[1]) and torch.equal(torch.get_rng_state(),cpu)
    assert torch.equal(tensor,torch.ones_like(tensor))


def test_factory_mutation_fatal_rollback():
    model=linear(); saved=model[1].weight.detach().clone()
    def bad(s,d,v):
        with torch.no_grad():model[1].weight.add_(1)
        return factory(s,d,v)
    with pytest.raises(StateMutationError):run_sweep(model,bad,axis='batch',points=[1,2],protocol=plan())
    assert torch.equal(model[1].weight,saved) and model.training
    assert run_sweep(model,factory,axis='batch',points=[1],protocol=plan()).to_dict()['actual']['measured']==1


def test_point_rng_independent():
    values=[]
    def source(s,d,v):
        x=factory(s,d,v);values.append(x.clone());return x
    run_sweep(linear(),source,axis='precision',points=['fp32','fp64','bf16'],protocol=plan())
    assert torch.equal(values[0],values[2])


def test_gc_disabled_lifetime(monkeypatch):
    refs=[]; original=engine.copy.deepcopy
    def copied(value):
        model=original(value)
        if isinstance(model,torch.nn.Module): refs.append(weakref.ref(model))
        return model
    monkeypatch.setattr(engine.copy,'deepcopy',copied)
    old=gc.isenabled();gc.disable()
    try:
        model=spatial(); result=run_sweep(model,factory,axis='batch',points=[1,2],protocol=plan())
        assert result.to_dict()['actual']['measured']==2 and all(r() is None for r in refs)
        ref=weakref.ref(model);del model;assert ref() is None
    finally:
        if old:gc.enable()


@pytest.fixture
def result():
    return run_sweep(linear(),factory,axis='batch',points=[1,2],protocol=plan())


@pytest.mark.parametrize('kind',['sweep','precision'])
def test_frozen_defensive_json_csv(result,tmp_path,kind):
    if kind=='precision':result=PrecisionResult(result.to_dict()['raw_evidence']['points'][0]['result'])
    data=result.to_dict();data['actual'].clear();assert result.to_dict()['actual']
    with pytest.raises((FrozenInstanceError,AttributeError)):result._payload='bad'
    result.to_json(tmp_path/'a.json');result.to_csv(tmp_path/'a.csv')
    with pytest.raises(FileExistsError):result.to_json(tmp_path/'a.json')
    assert type(result)(json.loads((tmp_path/'a.json').read_text())).to_dict()==result.to_dict()
    assert 'latency_mean_ms' in (tmp_path/'a.csv').read_text()


@pytest.mark.parametrize('case',['count','id','shape','status','reference','axis','precision','nan','samples','int8'])
def test_forged_results_rejected(result,case):
    data=result.to_dict();row=data['raw_evidence']['points'][0]
    if case=='count':data['actual']['measured']=0
    elif case=='id':row['point_id']='bad'
    elif case=='shape':row['shape'][0]=9
    elif case=='status':row['status']='unsupported'
    elif case=='reference':row['result_id']='bad'
    elif case=='axis':data['feature_ids']=['P07']
    elif case=='precision':row['precision']='fp64'
    elif case=='nan':row['aspect_ratio']=float('nan')
    elif case=='samples':row['result']['actual']['samples']=999
    else:row['result']['protocol']['precision']='int8_dynamic'
    with pytest.raises(ValueError):SweepResult(data)


def test_guide_manager_reopen(tmp_path,monkeypatch):
    monkeypatch.setattr('tempfile.mkdtemp',lambda **kwargs:str(tmp_path))
    guide=Path(__file__).resolve().parents[1]/'docs/guides/performance-sweeps.md'; namespace={}
    for block in re.findall(r'```python\n(.*?)```',guide.read_text(encoding='utf-8'),re.S):exec(compile(block,str(guide),'exec'),namespace)
    assert namespace['batches'].to_dict()['actual']['measured']==4


@pytest.mark.parametrize('precision',['fp32','fp64','bf16'])
def test_exact_forward_count_and_output_release(precision):
    outputs=[]; calls=[]
    class Counted(torch.nn.Module):
        def __init__(self):super().__init__();self.conv=torch.nn.Conv2d(1,2,1)
        def forward(self,x):
            calls.append(tuple(x.shape));y=self.conv(x);outputs.append(weakref.ref(y));return y
    model=Counted();old=gc.isenabled();gc.disable()
    try:
        data=run_sweep(model,factory,axis='batch',points=[1,3],protocol=plan(precision=precision,sustained_batches=4)).to_dict()
        assert data['actual']['measured']==2
        per_point=1+2*(1+3)+2*(1+4)+(1 if precision!='bf16' else 0)
        assert len(calls)==2*per_point and calls.count((1,1,5,7))==per_point
        assert all(r() is None for r in outputs)
    finally:
        if old:gc.enable()


def test_formula_safe_error_csv(tmp_path):
    def bad(s,d,v):raise NotImplementedError('=HYPERLINK("bad")')
    result=run_sweep(linear(),bad,axis='batch',points=[1],protocol=plan())
    result.to_csv(tmp_path/'result.csv')
    assert "'=HYPERLINK" in (tmp_path/'result.csv').read_text()
