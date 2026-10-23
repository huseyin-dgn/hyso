"""Phase B: fresh production workers, exact replay and worker-only true peaks."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,os.environ.get('HYSO_TEST_PACKAGE_ROOT',str(Path(__file__).resolve().parents[1])))
import numpy as np
import torch
from hyso.performance import (TrainingMemoryRecipe,IsolatedTrainingMemoryProtocol,TrainingMemoryProtocol,
    measure_training_memory_isolated)
from hyso.performance import isolated_training_memory as module
from training_memory_factories import tiny,state_digest


def seed():random.seed(71);np.random.seed(71);torch.manual_seed(71)


def run_case(root,optimizer,amp,nondefault,overflow,reset=True,activation=False,profile=None):
    name=f'{optimizer}-amp{amp}-stream{nondefault}-overflow{overflow}-reset{reset}'+('-'+profile if profile else '')
    extra=dict(modular=profile in ('modular','early'),early=profile=='early',tied=profile=='tied')
    epochs=5 if profile=='early' else 2
    seed();workload=tiny(device='cuda:0',optimizer=optimizer,amp=amp,nondefault=nondefault,overflow=overflow,**extra)
    stream=workload['stream'] or torch.cuda.current_stream()
    with torch.cuda.stream(stream):workload['trainer'].fit(workload['train_loader'],workload['val_loader'],epochs=epochs)
    stream.synchronize();expected=state_digest(workload['trainer']);torch.cuda.synchronize()
    rng=(copy.deepcopy(random.getstate()),copy.deepcopy(np.random.get_state()),torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone())
    before=dict(torch.cuda.memory_stats())
    path=root/(name+'.state.json')
    recipe=TrainingMemoryRecipe('training_memory_factories:tiny',device='cuda:0',factory_search_path=Path(__file__).parent,
        seed=71,epochs=epochs,kwargs=dict(optimizer=optimizer,amp=amp,nondefault=nondefault,overflow=overflow,state_path=str(path),**extra))
    protocol=IsolatedTrainingMemoryProtocol(timeout_seconds=60,allow_peak_reset=reset,saved_activation_lifetime=activation,memory_protocol=TrainingMemoryProtocol(device='cuda:0'))
    result=measure_training_memory_isolated(recipe,protocol=protocol);d=result.to_dict()
    assert d['outcome']=='success',d
    assert before==dict(torch.cuda.memory_stats())
    assert rng[0]==random.getstate() and rng[1][0]==np.random.get_state()[0] and np.array_equal(rng[1][1],np.random.get_state()[1])
    assert torch.equal(rng[2],torch.get_rng_state()) and torch.equal(rng[3],torch.cuda.get_rng_state())
    assert state_digest(workload['trainer'])==expected
    assert json.loads(path.read_text())==dict(state_sha256=expected,fits=1)
    measurement=d['measurement'];windows=measurement['windows']
    assert [r['samples'] for r in windows]==([6,1,6,1] if extra['modular'] else [4,1,4,1])
    assert sum(r['skipped'] for r in windows)==int(overflow)
    assert measurement['source_p19']['counters']['after']['applied']==4-int(overflow)
    assert measurement['boundaries'][0]['logical']['summary']['optimizer']['logical_bytes']==0
    if optimizer!='sgd':
        if overflow:
            assert next(r for r in measurement['boundaries'] if r['boundary']=='post_attempt')['logical']['summary']['optimizer']['logical_bytes']==0
        assert measurement['boundaries'][-1]['logical']['summary']['optimizer']['logical_bytes']>0
    if activation:
        a=measurement['activation'];s=a['summary']
        assert s['saved_tensors']>0 and s['pack_events']==s['release_events'] and s['unpack_events']>0
        assert s['live_payloads_at_end']==0 and a['events'][-1]['live_storage_bytes']==0
        assert a['physical_activation_peak']['value'] is None
    for window in windows:
        assert window['reset_performed']==reset
        if reset:
            assert window['reset_owner_pid']==d['receipt']['launcher_pid']
            assert window['after_reset']['lifetime_peak_allocated_bytes']==window['after_reset']['allocated_bytes']
            assert window['peak_allocated_bytes']>=window['end']['allocated_bytes']
            assert window['peak_reserved_bytes']>=window['peak_allocated_bytes']
        else:assert window['peak_allocated_bytes'] is None and window['status']=='unsupported'
    if os.environ.get('HYSO_TEST_PACKAGE_ROOT'):
        assert Path(measurement['provenance']['package_source']).is_relative_to(Path(os.environ['HYSO_TEST_PACKAGE_ROOT']))
    result.to_json(root/(name+'.result.json'))
    return dict(id=name,exact_state_rng=True,parent_allocator_preserved=True,result=str(root/(name+'.result.json')))


def activation_extras(root):
    cases=[]
    for profile,amp,overflow in (('modular',False,False),('modular',True,True),('early',False,False),('tied',False,False)):
        cases.append(run_case(root,'adamw',amp,True,overflow,activation=True,profile=profile))
    p=IsolatedTrainingMemoryProtocol(timeout_seconds=60,allow_peak_reset=True,saved_activation_lifetime=True,
        memory_protocol=TrainingMemoryProtocol(device='cuda:0'))
    for failure in ('after_forward','backward',None):
        path=root/('lifetime-'+str(failure)+'.json')
        r=TrainingMemoryRecipe('training_memory_factories:tiny',device='cuda:0',factory_search_path=Path(__file__).parent,
            seed=71,epochs=2,kwargs=dict(failure=failure,lifetime_path=str(path)))
        before=dict(torch.cuda.memory_stats());rng=torch.cuda.get_rng_state().clone()
        d=measure_training_memory_isolated(r,protocol=p).to_dict()
        assert d['outcome']==('workload_exception' if failure else 'success'),d
        assert all(json.loads(path.read_text()).values())
        assert d['receipt']['job']['active_processes']==0
        assert before==dict(torch.cuda.memory_stats()) and torch.equal(rng,torch.cuda.get_rng_state())
        cases.append(dict(id='gc-disabled-'+str(failure),outcome=d['outcome'],weakrefs_released=True))
    seed();w=tiny(device='cuda:0',optimizer='adamw',amp=True,modular=True)
    w['trainer'].fit(w['train_loader'],w['val_loader'],epochs=2);expected=state_digest(w['trainer'])
    checkpoint=root/'resume.pth';state=root/'resume.state.json'
    first=TrainingMemoryRecipe('training_memory_factories:tiny',device='cuda:0',factory_search_path=Path(__file__).parent,
        seed=71,epochs=2,save_last_to=str(checkpoint),kwargs=dict(optimizer='adamw',amp=True,modular=True,interrupt_after_epoch=1))
    assert measure_training_memory_isolated(first,protocol=p).to_dict()['outcome']=='workload_exception'
    second=TrainingMemoryRecipe('training_memory_factories:tiny',device='cuda:0',factory_search_path=Path(__file__).parent,
        seed=71,epochs=2,save_last_to=str(checkpoint),resume_from=str(checkpoint),kwargs=dict(optimizer='adamw',amp=True,modular=True,state_path=str(state)))
    d=measure_training_memory_isolated(second,protocol=p).to_dict();assert d['outcome']=='success',d
    assert json.loads(state.read_text())['state_sha256']==expected
    done=measure_training_memory_isolated(second,protocol=p).to_dict();assert done['outcome']=='success',done
    assert done['measurement']['source_p19']['actual_epochs']==0 and done['measurement']['activation']['summary']['saved_tensors']==0
    cases.append(dict(id='amp-checkpoint-resume-and-completed',exact_state_rng=True))
    return cases


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--admit');p.add_argument('--case',default='smoke');a=p.parse_args()
    if a.admit:
        deadline=time.monotonic()+15
        while not Path(a.admit).exists() and time.monotonic()<deadline:time.sleep(.02)
        assert Path(a.admit).exists()
    assert torch.cuda.is_available();torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.15)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(a.output).parent/(Path(a.output).stem+'-cases');root.mkdir()
    cases=[];activation=a.case=='activation'
    if a.case=='smoke':cases.append(run_case(root,'adamw',False,False,False))
    else:
        for optimizer in ('sgd','adam','adamw'):
            for amp in (False,True):
                for stream in (False,True):
                    for overflow in ((False,True) if amp else (False,)):
                        cases.append(run_case(root,optimizer,amp,stream,overflow,activation=activation))
        cases.append(run_case(root,'adamw',True,False,False,reset=False,activation=activation))
        for failure,outcome in (('oom','oom'),('timeout','timeout')):
            recipe=TrainingMemoryRecipe('training_memory_factories:tiny',device='cuda:0',factory_search_path=Path(__file__).parent,
                epochs=2,kwargs=dict(failure=failure))
            before=dict(torch.cuda.memory_stats());rng=torch.cuda.get_rng_state().clone()
            result=measure_training_memory_isolated(recipe,protocol=IsolatedTrainingMemoryProtocol(timeout_seconds=25 if failure=='timeout' else 60,
                allow_peak_reset=True,saved_activation_lifetime=activation,memory_protocol=TrainingMemoryProtocol(device='cuda:0')))
            d=result.to_dict();assert d['outcome']==outcome,d
            assert before==dict(torch.cuda.memory_stats()) and torch.equal(rng,torch.cuda.get_rng_state())
            assert d['measurement'] is None and d['receipt']['job']['active_processes']==0
            if failure=='timeout':assert 'training_forward_started' in d['receipt']['stdout']
            result.to_json(root/(failure+'.json'));cases.append(dict(id=failure,result=str(root/(failure+'.json'))))
        if activation:cases.extend(activation_extras(root))
    Path(a.output).write_text(json.dumps(dict(status='passed',scenarios=cases,package_source=module.__file__,process_pid=os.getpid()),indent=2),encoding='utf-8')
    print(len(cases),'isolated CUDA cases passed')


if __name__=='__main__':main()
