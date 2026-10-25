"""P13 allocation evidence and explicit matched profiler/control experiment."""
import copy
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from hyso.performance import (ColdStartRecipe,ProfilingProtocol,profile_operators,profile_layers,
    OperatorProfileResult,LayerProfileResult,LayerOverheadProtocol,LayerOverheadResult,measure_layer_overhead,analyze_roofline)
from hyso.performance._layer_memory import memory_events,attribute_memory
from hyso.performance.layer_overhead import _metrics


def recipe(**kw):
    return ColdStartRecipe('layer_diagnostic_factories:Memory',input_shape=(2,2,3,3),
        factory_search_path=Path(__file__).parent,**kw)


def capture(**kw):
    return ProfilingProtocol(warmup_iterations=2,measure_iterations=3,timeout_seconds=45,
        cpu_memory_limit_bytes=3*1024**3,input_limit_bytes=32*1024**2,**kw)


@pytest.fixture(scope='module')
def memory_capture():return profile_operators(recipe(),protocol=capture(profile_memory=True))


@pytest.fixture(scope='module')
def overhead():return measure_layer_overhead(recipe(),protocol=LayerOverheadProtocol(trials=2),capture_protocol=capture())


def test_real_cpu_allocations_are_events_not_peaks_or_rss(memory_capture,monkeypatch,tmp_path):
    before=memory_capture.to_dict();rng=torch.get_rng_state().clone()
    def forbidden(*a,**kw):raise AssertionError('pure layer analysis must not run model/CUDA')
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden)
    result=profile_layers(memory_capture);d=result.to_dict();m=d['measurements']['layer_memory']
    assert before['protocol']['instrumentation']['evidence_version']==4
    assert m['status']=='partial' and m['peak_bytes'] is m['retained_bytes'] is None
    leaf=next(module for module in d['measurements']['modules'] if module['path']=='leaf')['memory']['cpu']
    assert leaf['status']=='measured'
    values=leaf['inclusive']
    assert values['allocation_bytes']>=3*2*36*4 and values['deallocation_bytes']>=3*36*4
    assert values['net_change_bytes']==values['allocation_bytes']-values['deallocation_bytes']
    assert leaf['peak_bytes'] is leaf['retained_bytes'] is None
    root=next(module for module in d['measurements']['modules'] if module['path']=='')['memory']['cpu']
    assert root['inclusive']==values and root['exclusive']['allocation_bytes']==0
    assert m['unassigned_event_count']>=0
    result.to_json(tmp_path/'layers.json');result.to_csv(tmp_path/'layers.csv')
    assert LayerProfileResult(json.loads((tmp_path/'layers.json').read_text())).to_dict()==d
    assert 'memory' in (tmp_path/'layers.csv').read_text()
    assert memory_capture.to_dict()==before and torch.equal(rng,torch.get_rng_state())
    assert analyze_roofline(memory_capture).to_dict()['measurements']['traffic']['status']=='estimated'


@pytest.mark.parametrize('mutation',[
    lambda r:r['raw_evidence']['memory_events'].append(copy.deepcopy(r['raw_evidence']['memory_events'][0])),
    lambda r:r['raw_evidence']['memory_events'][0].update(start_us=float('nan')),
    lambda r:r['raw_evidence']['memory_events'][0].update(bytes=True),
    lambda r:r['measurements']['layer_memory'].update(peak_bytes=100),
    lambda r:r['protocol']['configuration'].update(profile_memory=False),
])
def test_memory_tamper_or_malformed_evidence_rejected(memory_capture,mutation):
    d=memory_capture.to_dict();mutation(d)
    with pytest.raises(ValueError):OperatorProfileResult(d)


def test_no_memory_events_or_missing_annotation_is_unsupported(memory_capture):
    raw=memory_capture.to_dict()['raw_evidence']
    args=(raw['trace_events'],raw['invocations'],raw['module_inventory'],'cpu')
    absent=attribute_memory([],*args)
    assert absent['status']=='unsupported' and absent['peak_bytes'] is None
    unknown=attribute_memory(raw['memory_events'],[],raw['invocations'],raw['module_inventory'],'cpu')
    assert unknown['status']=='unsupported'
    assert unknown['unassigned_event_count']==len(raw['memory_events'])
    assert all(r['domains']['cpu']['inclusive'] is None for r in unknown['invocations'])
    with pytest.raises(ValueError,match='budget'):memory_events({'traceEvents':[dict(name='[memory]')]},0)


def test_real_overhead_separate_owned_workers_and_canonical_exports(overhead,tmp_path):
    d=overhead.to_dict();m=d['measurements'];pairs=d['raw_evidence']['pairs']
    assert m['matched_context'] and m['status'] in ('estimated','unsupported')
    if not m['variance_accepted']:assert m['signed_overhead_percent'] is None
    for pair in pairs:
        source=pair['instrumented'];control=pair['control']
        assert control['worker']['worker_pid']!=source['raw_evidence']['worker']['worker_pid']
        assert control['worker']['process_terminated'] and source['raw_evidence']['worker']['process_terminated']
        assert source['measurements']['overhead']['perturbation_ms'] is None
        assert len(control['windows_ms'])==3
    j=tmp_path/'overhead.json';c=tmp_path/'overhead.csv';overhead.to_json(j);overhead.to_csv(c)
    assert LayerOverheadResult.from_json(j).to_dict()==LayerOverheadResult.from_csv(c).to_dict()==d
    with pytest.raises(FileExistsError):overhead.to_json(j)
    with pytest.raises(FileExistsError):overhead.to_csv(c)
    corrupted=copy.deepcopy(d);corrupted['measurements']['signed_overhead_percent']=999
    with pytest.raises(ValueError):LayerOverheadResult(corrupted)


def _synthetic_windows(overhead,control,instrumented):
    d=overhead.to_dict()
    for pair in d['raw_evidence']['pairs']:
        pair['control']['windows_ms']=control[:]
        source=pair['instrumented']
        source['raw_evidence']['overhead_experiment']['windows_ms']=instrumented[:]
        source['measurements']['overhead']['instrumented_host_forward_windows_ms']=instrumented[:]
    return d


def _derive(d):
    p=d['protocol']
    return _metrics(d['raw_evidence']['pairs'],p['recipe'],p['capture_configuration'],LayerOverheadProtocol(**p['configuration']))


def test_matched_math_variance_and_mismatch_gates_are_not_native_acceptance(overhead):
    d=_synthetic_windows(overhead,[2.,2.,2.],[3.,3.,3.]);m=_derive(d)
    assert m['status']=='estimated' and m['signed_overhead_ms']==1 and m['signed_overhead_percent']==50
    d=_synthetic_windows(overhead,[2.,2.,2.],[1.,1.,1.]);m=_derive(d)
    assert m['signed_overhead_ms']==-1 and m['signed_overhead_percent']==-50
    d=_synthetic_windows(overhead,[1.,10.,1.],[3.,3.,3.]);m=_derive(d)
    assert m['status']=='unsupported' and m['signed_overhead_percent'] is None and 'variation' in m['reason']
    d=_synthetic_windows(overhead,[2.,2.,2.],[3.,3.,3.])
    d['raw_evidence']['pairs'][0]['control']['output_sha256']='0'*64;m=_derive(d)
    assert not m['matched_context'] and m['signed_overhead_percent'] is None


def test_experiment_is_explicit_bounded_and_defaults_have_no_quotas():
    p=asdict(ProfilingProtocol())
    assert p['profile_memory'] is False
    assert all(p[k] is None for k in ('cpu_memory_limit_bytes','input_limit_bytes','cuda_memory_fraction'))
    with pytest.raises(ValueError):ProfilingProtocol(profile_memory=1)
    with pytest.raises(ValueError):LayerOverheadProtocol(trials=1)
    with pytest.raises(ValueError):LayerOverheadProtocol(trials=6)
    with pytest.raises(ValueError):measure_layer_overhead(recipe(),capture_protocol=ProfilingProtocol(warmup_iterations=0))


def test_allocation_free_net_and_ambiguous_siblings():
    inv=[dict(invocation_id=0,module_id='m0',annotation='root',parent_invocation_id=None),
         dict(invocation_id=1,module_id='m1',annotation='leaf',parent_invocation_id=0)]
    inventory=[dict(module_id='m0'),dict(module_id='m1')]
    def span(name,start,end):return dict(category='user_annotation',name=name,pid=1,tid=2,start_us=start,duration_us=end-start)
    trace=[span('root',0,10),span('leaf',1,9)]
    def event(i,size):return dict(trace_index=i,phase='i',start_us=5+i,pid=1,tid=2,bytes=size,device_type=0,device_id=-1)
    events=[event(0,64),event(1,-96)]
    m=attribute_memory(events,trace,inv,inventory,'cpu');leaf=m['modules'][1]['domains']['cpu']
    assert leaf['inclusive']==dict(allocation_bytes=64,deallocation_bytes=96,net_change_bytes=-32,event_count=2)
    assert leaf['retained_bytes'] is leaf['peak_bytes'] is None
    inv.append(dict(invocation_id=2,module_id='m2',annotation='sibling',parent_invocation_id=0))
    inventory.append(dict(module_id='m2'));trace.append(span('sibling',1,9))
    m=attribute_memory(events,trace,inv,inventory,'cpu')
    assert m['status']=='unsupported' and m['unassigned_event_count']==2
    assert all(r['domains']['cpu']['inclusive'] is None for r in m['modules'])


def test_failure_is_propagated_without_partial_success_or_extra_workers(monkeypatch):
    import hyso.performance.layer_overhead as module
    from hyso.performance import ProfilingError
    calls=[];failure=ProfilingError(dict(status='failed',reason='controlled OOM classification; no stress allocation'))
    def failed(*a,**kw):calls.append(kw['experiment_mode']);raise failure
    monkeypatch.setattr(module,'_execute_profile',failed)
    with pytest.raises(ProfilingError) as e:measure_layer_overhead(recipe())
    assert e.value is failure and calls==['control']
