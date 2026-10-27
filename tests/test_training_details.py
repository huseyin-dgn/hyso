"""Real CPU P20/P22, exact triple replay, failures and generic storage."""
from fixture_support import example_fixture
import copy
import json
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from hyso.training import Trainer
from hyso.performance import (TrainingTimingProtocol, TrainingStageProtocol, TrainingStageResult,
    TrainingThroughputResult, TrainingTimingError, measure_training, measure_training_detailed)
from hyso.performance.training_details import _StageRecorder, MICRO_STAGES, ATTEMPT_STAGES
from trainer_replay_support import equal, seed, snapshot, build, CASES, CRITICAL


def batches():
    return list(zip(torch.linspace(-1,1,15).reshape(5,3).split(2),torch.tensor([0,1,0,1,0]).split(2)))


def make(opt=torch.optim.Adam,epochs=2, *, clip=.1, ema=True):
    m=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.2),torch.nn.Linear(4,2))
    o=opt(m.parameters(),lr=.01)
    s=torch.optim.lr_scheduler.OneCycleLR(o,.02,total_steps=epochs*2,cycle_momentum=False)
    return Trainer(m,optimizer=o,loss_fn=torch.nn.CrossEntropyLoss(),accum_steps=2,
        scheduler=s,grad_clip=clip,ema_decay=.8 if ema else None)


@pytest.mark.parametrize('opt',[torch.optim.SGD,torch.optim.Adam,torch.optim.AdamW])
@pytest.mark.parametrize('epochs',[1,2,3])
@pytest.mark.parametrize('validation',[False,True])
def test_real_triple_equivalence_and_stages(opt,epochs,validation):
    states=[]
    for mode in ('normal','p19','detailed'):
        seed(724)
        t=make(opt,epochs)
        data=batches(); val=batches() if validation else None
        calls=[]
        native=t.fit
        def fit(*a,**kw): calls.append(True); return native(*a,**kw)
        with patch.object(t,'fit',fit):
            if mode=='normal': t.fit(data,val,epochs=epochs)
            elif mode=='p19': measure_training(t,data,val,epochs=epochs)
            else: result=measure_training_detailed(t,data,val,epochs=epochs)
        assert len(calls)==1
        assert not hasattr(t,'_timing_recorder') and not hasattr(t,'_stage_recorder')
        assert not t.optimizer._optimizer_step_pre_hooks and not t.optimizer._optimizer_step_post_hooks
        states.append(snapshot(t))
    equal(states[0],states[1]); equal(states[0],states[2])
    d=result.stages.to_dict(); timing=result.timing.to_dict(); throughput=result.throughput.to_dict()
    assert d['source_p19']==timing==throughput['source_p19']
    for name,count in [('forward',3),('backward',3),('optimizer_execution',2),('scaler_update',2),('ema_update',2),('onecycle_scheduler',2),('zero_grad',3)]:
        assert sum(r['stage']==name and r['status']=='measured' for r in d['observations'])==count*epochs
    assert [r['samples'] for r in timing['observations'] if r['kind']=='window']==[4,1]*epochs
    assert throughput['segment']['completed']['samples']==5*epochs
    assert throughput['segment']['rates']['samples_per_second']['value']==5*epochs/sum(e['training_s'] for e in timing['epochs'])
    assert all(r['host_ms']>0 and r['cuda_event_ms'] is None for r in d['observations'] if r['status']=='measured')
    assert all(r['host_ms'] is None and r['cuda_event_ms'] is None for r in d['observations'] if r['status']=='not_executed')


@pytest.mark.parametrize('opt',[torch.optim.SGD,torch.optim.Adam,torch.optim.AdamW])
@pytest.mark.parametrize('observer',['P19','P20','P25'])
def test_injected_cpu_skip_uses_actual_execution_ledger(monkeypatch,opt,observer):
    from hyso.performance import measure_gradient_diagnostics
    t=make(opt,epochs=1,clip=None)
    assert not t.amp and not t.scaler.is_enabled()
    native_step=t.scaler.step; calls=[]; executions=[]
    def skip_first(*args,**kwargs):
        # CPU protocol fault injection, NOT native AMP overflow. Real Trainer
        # counters/history are never fabricated; the second step is native.
        calls.append(True)
        if len(calls)>1: return native_step(*args,**kwargs)
    monkeypatch.setattr(t.scaler,'step',skip_first)
    handle=t.optimizer.register_step_post_hook(lambda *args:executions.append(True))
    pre=set(t.optimizer._optimizer_step_pre_hooks); post=set(t.optimizer._optimizer_step_post_hooks)
    try:
        measure={'P19':measure_training,'P20':measure_training_detailed,'P25':measure_gradient_diagnostics}[observer]
        with patch.object(t,'fit',wraps=t.fit) as fit:
            result=measure(t,batches(),epochs=1)
        assert fit.call_count==1 and len(calls)==2 and len(executions)==1
        assert (t.microbatch_count,t.attempt_count,t.update_count,t.ema_update_count)==(3,2,1,1)
        assert t.scheduler.last_epoch==1 and len(t.history['train_loss'])==1
        assert set(t.optimizer._optimizer_step_pre_hooks)==pre
        assert set(t.optimizer._optimizer_step_post_hooks)==post
        assert not hasattr(t,'_timing_recorder') and not hasattr(t,'_stage_recorder')
        if observer=='P19':
            timing=result.to_dict(); stages=None
        elif observer=='P20':
            timing=result.timing.to_dict(); stages=result.stages.to_dict()
        else:
            diagnostics=result.to_dict(); timing=diagnostics['source_p19']; stages=diagnostics['source_p20']
            assert diagnostics['summary']['recovered_after_skip_attempt_ids']==[2]
            assert diagnostics['observations'][0]['overflow_evidence']=='unknown'
            assert all(r['scale_before']==r['scale_after']==1. for r in diagnostics['observations'])
            assert diagnostics['instrumentation']['total_overhead_ns'] is None
            assert diagnostics['instrumentation']['cuda_host_scalar_sync'] is False
            assert diagnostics['instrumentation']['scalar_reads']>0
        attempts=[r for r in timing['observations'] if r['kind']=='attempt']
        assert [r['samples'] for r in attempts]==[4,1]
        assert [r['applied'] for r in attempts]==[False,True]
        assert [r['skipped'] for r in attempts]==[True,False]
        assert [r['applied_update_id'] for r in attempts]==[None,1]
        assert timing['instrumentation']['device_clock'] is None
        assert any('instrumented' in text for text in timing['limitations'])
        if stages is not None:
            for name in ('optimizer_execution','ema_update','onecycle_scheduler'):
                rows=[r for r in stages['observations'] if r['stage']==name]
                assert [r['status'] for r in rows]==['not_executed','measured']
                assert rows[0]['host_ms'] is None and rows[0]['cuda_event_ms'] is None and rows[0]['reason']
            assert all(r['aggregation']=='inclusive_per_stage_no_cross_stage_sum' for r in stages['summary'])
    finally:
        handle.remove()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('case',CASES)
def test_detailed_replay_matrix(case):
    outputs=[]
    for observed in (False,True):
        seed(86); t,train,val=build(case)
        if observed: result=measure_training_detailed(t,train,val,epochs=5)
        else: t.fit(train,val,epochs=5)
        outputs.append(snapshot(t))
    equal(*outputs)


@pytest.mark.parametrize('failure',['loader','forward','backward','optimizer','buffer','event','event_in_optimizer'])
def test_failure_cleanup(failure,monkeypatch):
    t=make(); data=batches()
    def fail(*args,**kw): raise RuntimeError('injected failure')
    if failure=='loader':
        original_data=data
        class Loader:
            def __len__(self): return 3
            def __iter__(self):
                yield original_data[0]
                raise RuntimeError('injected failure')
        data=Loader()
    elif failure=='forward': monkeypatch.setattr(t.model,'forward',fail)
    elif failure=='backward': t.model[0].weight.register_hook(fail)
    elif failure=='optimizer': t.optimizer.register_step_pre_hook(fail)
    elif failure=='event': monkeypatch.setattr(_StageRecorder,'_event',fail)
    elif failure=='event_in_optimizer':
        original_event=_StageRecorder._event
        def event(recorder):
            if 'stage:scaler_step_attempt' in recorder.active: fail()
            return original_event(recorder)
        monkeypatch.setattr(_StageRecorder,'_event',event)
    pre=set(t.optimizer._optimizer_step_pre_hooks); post=set(t.optimizer._optimizer_step_post_hooks)
    with pytest.raises((RuntimeError,TrainingTimingError)):
        measure_training_detailed(t,data,epochs=2,stage_protocol=TrainingStageProtocol(max_records=12 if failure=='buffer' else 100000))
    assert not hasattr(t,'_timing_recorder') and not hasattr(t,'_stage_recorder')
    assert set(t.optimizer._optimizer_step_pre_hooks)==pre and set(t.optimizer._optimizer_step_post_hooks)==post


def test_roundtrip_exports_immutable_and_tamper(tmp_path):
    r=measure_training_detailed(make(),batches(),epochs=2)
    for i,result in enumerate((r.stages,r.throughput)):
        result.to_json(tmp_path/f'{i}.json'); result.to_csv(tmp_path/f'{i}.csv')
        assert type(result)(json.loads((tmp_path/f'{i}.json').read_text())).to_dict()==result.to_dict()
        with pytest.raises((AttributeError,TypeError)): result._payload='bad'
        with pytest.raises(FileExistsError): result.to_json(tmp_path/f'{i}.json')
        d=result.to_dict(); d['source_sha256']='bad'
        object.__setattr__(result,'_payload',json.dumps(d))
        with pytest.raises(ValueError): result.to_csv(tmp_path/'bad.csv')
        assert not (tmp_path/'bad.csv').exists()


@pytest.mark.parametrize('mutation',['missing','duration','parent','executed','summary'])
def test_stage_tamper(mutation):
    d=measure_training_detailed(make(),batches(),epochs=2).stages.to_dict()
    row=next(r for r in d['observations'] if r['stage']=='optimizer_execution')
    if mutation=='missing': d['observations'].remove(row)
    elif mutation=='duration': row['host_ms']+=1
    elif mutation=='parent': row['attempt_id']=100
    elif mutation=='executed': row['status']='not_executed'
    else: d['summary'][0]['host_ms']=0
    with pytest.raises(ValueError): TrainingStageResult(d)


def test_checkpoint_resume_and_completed(tmp_path):
    import hyso.training.resume as resume
    outcomes=[]
    for observed in (False,True):
        seed(57); t,train,val=build(CRITICAL); path=tmp_path/f'{observed}.pth'
        original=resume.save_last
        class Stop(Exception): pass
        def cut(*args):
            original(*args)
            if len(args[0].history['train_loss'])==2: raise Stop()
        with patch.object(resume,'save_last',cut),pytest.raises(Stop):
            if observed: measure_training_detailed(t,train,val,epochs=5,save_last_to=path)
            else: t.fit(train,val,epochs=5,save_last_to=path)
        t,train,val=build(CRITICAL)
        if observed: result=measure_training_detailed(t,train,val,epochs=5,resume_from=path,save_last_to=tmp_path/'done.pth')
        else: t.fit(train,val,epochs=5,resume_from=path)
        outcomes.append(snapshot(t))
    equal(*outcomes)
    assert [e['epoch_id'] for e in result.throughput.to_dict()['epochs']]==[3,4,5]
    t,train,val=build(CRITICAL)
    done=measure_training_detailed(t,train,val,epochs=5,resume_from=tmp_path/'done.pth')
    assert done.stages.to_dict()['observations']==[]
    assert done.throughput.to_dict()['segment']['rates']['samples_per_second']['value'] is None


def test_experiment_manager(tmp_path,monkeypatch):
    from hyso.experiments import ExperimentManager
    root=tmp_path/'store'; m=ExperimentManager(root); m.close(); ExperimentManager.migrate_store(root)
    m=ExperimentManager(root); run=m.create_run(m.create_experiment('B9')['id'],'training',{}); run.start()
    t=make(); results=measure_training_detailed(t,batches(),epochs=2)
    before=snapshot(t)
    def forbidden(*a,**kw): raise AssertionError('tracking retrained')
    monkeypatch.setattr(t,'fit',forbidden)
    for name,result in [('P20',results.stages),('P22',results.throughput)]:
        path=tmp_path/(name+'.json'); result.to_json(path); result.to_csv(tmp_path/(name+'.csv'))
        artifact=run.add_artifact(path,kind='report',ownership='copy',metadata=dict(source_segment_id=results.timing.to_dict()['segment_id']))
        assert m.verify_artifact(artifact['id'])['status']=='verified'
        csv_artifact=run.add_artifact(tmp_path/(name+'.csv'),kind='report',ownership='copy',metadata=dict(source_segment_id=results.timing.to_dict()['segment_id']))
        assert m.verify_artifact(csv_artifact['id'])['status']=='verified'
    for e in results.throughput.to_dict()['epochs']:
        metric=e['rates']['samples_per_second']
        if metric['value'] is not None: run.log_metric(name='training.samples',value=metric['value'],unit=metric['unit'],source='P22',epoch=e['epoch_id'])
    with patch.object(run,'log_metric',side_effect=RuntimeError('tracking failed')),pytest.raises(RuntimeError):
        run.log_metric(name='test',value=1,unit='count',source='P20',epoch=1)
    equal(before,snapshot(t))
    run.finish(); identity=run.id; m.close(); m=ExperimentManager(root)
    assert len(m.get_run(identity).read_metrics())==2
    m.close()


def test_early_stopping_only_completed_epochs():
    outputs=[]
    for observed in (False,True):
        seed(13); t,train,val=build(dict(CRITICAL,early='terminal',best=True))
        if observed: result=measure_training_detailed(t,train,val,epochs=5)
        else: t.fit(train,val,epochs=5)
        outputs.append(snapshot(t))
    equal(*outputs)
    assert result.throughput.to_dict()['actual']['epochs']==2


def test_invalid_profile_is_pretraining_and_disabled_has_no_stage_calls(monkeypatch):
    t=make()
    with pytest.raises(TypeError): measure_training_detailed(t,batches(),epochs=2,stage_protocol={})
    assert t._epochs is None and t.microbatch_count==0
    def fail(*a,**kw): raise AssertionError('stage on baseline/P19')
    monkeypatch.setattr(_StageRecorder,'stage_begin',fail)
    t.fit(batches(),epochs=2)
    measure_training(make(),batches(),epochs=2)


def test_deterministic_stage_clock_arithmetic(monkeypatch):
    import hyso.performance.training as module
    ticks=iter(range(100,100000000,100))
    monkeypatch.setattr(module.time,'perf_counter_ns',lambda:next(ticks))
    result=measure_training_detailed(make(epochs=1),batches(),epochs=1)
    for r in result.stages.to_dict()['observations']:
        if r['status']=='measured':
            assert r['host_ms']==(r['end_ns']-r['start_ns'])/1e6
            if r['stage'] not in ('optimizer_execution','scaler_step_attempt'): assert r['host_ms']==.0001


def test_breakdown_guide():
    import re
    path=example_fixture('docs/guides/performance-training-breakdown-throughput.md')
    namespace={}
    for block in re.findall(r'```python\n(.*?)```',path.read_text(encoding='utf-8'),re.S):
        exec(compile(block,str(path),'exec'),namespace)
    assert namespace['throughput']['segment']['completed']['samples']==10
