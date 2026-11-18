"""Real CPU storage witnesses, address generations, history reconciliation."""
import copy
import json
from pathlib import Path

import pytest
import torch
from hyso.performance import (ColdStartRecipe,ProfilingProtocol,profile_operators,profile_layers,
    OperatorProfileResult,LayerProfileResult,layer_profiling_capabilities)
from hyso.performance._layer_allocator import _cuda_metrics,_digest


@pytest.fixture(scope='module')
def source():
    recipe=ColdStartRecipe('layer_diagnostic_factories:NestedRetaining',input_shape=(1,1,2,8),factory_search_path=Path(__file__).parent)
    return profile_operators(recipe,protocol=ProfilingProtocol(warmup_iterations=1,measure_iterations=2,
        profile_memory=True,allocation_lifetimes=True,cpu_memory_limit_bytes=3*1024**3,timeout_seconds=45))


def test_real_CPU_retained_storage_alias_and_shared_invocations(source,tmp_path):
    d=source.to_dict();result=profile_layers(source);m=result.to_dict()['measurements']['allocator_diagnostics']
    assert d['protocol']['instrumentation']['record_shapes'] is False
    assert d['protocol']['instrumentation']['evidence_version']==6
    assert m['cuda']['status']=='unsupported'
    leaf=next(x for x in m['modules'] if x['module_id']=='m1')
    assert len(leaf['invocation_ids'])==4
    assert leaf['CPU_retained_new_allocation_bytes_at_exits']==[64]*4
    old=profile_layers(source).to_dict()['measurements']['layer_memory']['invocations']
    for i in leaf['invocation_ids']:
        row=m['cpu']['invocations'][i]
        assert row['status']=='measured'
        assert row['retained_new_allocation_bytes']==64
        assert row['new_allocation_live_peak_bytes']==128
        assert len(row['retained_allocation_trace_indices'])==1
        assert row['global_CPU_allocator_peak_bytes'] is row['physical_RSS_peak_bytes'] is None
    assert any(old[i]['domains']['cpu']['inclusive']['net_change_bytes']!=64 for i in leaf['invocation_ids'])
    for i in (0,3):
        # Two child invocations, first output eventually freed, second kept at root exit.
        assert m['cpu']['invocations'][i]['retained_new_allocation_bytes']==64
    result.to_json(tmp_path/'allocator.json');result.to_csv(tmp_path/'allocator.csv')
    assert LayerProfileResult(json.loads((tmp_path/'allocator.json').read_text())).to_dict()==result.to_dict()
    assert 'allocator' in (tmp_path/'allocator.csv').read_text()
    assert 'CPU_retained_new_allocation_bytes_at_exits' in (tmp_path/'allocator.csv').read_text()
    assert profile_layers(source,memory_analysis_version=2).to_dict()['protocol']['profile_evidence_version']==6


@pytest.mark.parametrize('field',['retained','address','witness','duplicate_witness'])
def test_allocator_evidence_tamper_rejected(source,field):
    d=source.to_dict()
    if field=='retained':d['measurements']['allocator_diagnostics']['cpu']['invocations'][0]['retained_new_allocation_bytes']+=1
    elif field=='address':d['raw_evidence']['memory_events'][0]['address']=-1
    elif field=='duplicate_witness':d['raw_evidence']['allocation_evidence']['storage_exits'].append(copy.deepcopy(d['raw_evidence']['allocation_evidence']['storage_exits'][0]))
    else:d['raw_evidence']['allocation_evidence']['storage_exits']=[]
    with pytest.raises(ValueError):OperatorProfileResult(d)


def test_weak_storage_alias_lifetime_does_not_retain_tensor():
    import weakref,gc
    from hyso.performance._layer_allocator import AllocationRecorder
    r=AllocationRecorder(torch,torch.device('cpu'),100)
    x=torch.ones(8);alias=x.view(2,4);ref=weakref.ref(x.untyped_storage())
    r.observe([x,alias]);assert len(r.refs)==1
    del x;gc.collect();assert ref() is not None
    del alias;gc.collect();assert ref() is None
    r.close()


def test_capability_probe_does_not_initialize_CUDA(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('must not initialize CUDA')
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden)
    c=layer_profiling_capabilities()
    assert c['CUDA_Event_is_kernel_time'] is False and c['external_profiler_required'] is False
    assert c['cuda_kernel_trace'] in ('unsupported_backend_capability','available_requires_actual_trace_validation')
    assert ProfilingProtocol().allocation_lifetimes is False
    with pytest.raises(ValueError):ProfilingProtocol(allocation_lifetimes=True)


def history_fixture():
    history=[dict(action='snapshot'),dict(action='alloc',addr=20,size=512),dict(action='alloc',addr=30,size=1024),
        dict(action='free_requested',addr=30,size=1024),dict(action='free_completed',addr=30,size=1024)]
    points=[dict(invocation_id=None,phase='initial',position=0,allocated_bytes=512,active_blocks_sha256=_digest([(10,512)])),
        dict(invocation_id=0,phase='entry',position=1,allocated_bytes=512,active_blocks_sha256=_digest([(10,512)])),
        dict(invocation_id=0,phase='exit',position=5,allocated_bytes=1024,active_blocks_sha256=_digest([(10,512),(20,512)]))]
    return dict(status='recorded',initial_position=0,initial_blocks=[(10,512)],points=points,history=history)


def test_history_absolute_peak_and_retained_are_not_net_or_physical_peak():
    e=history_fixture();m=_cuda_metrics(e,[dict(parent_invocation_id=None)])['invocations'][0]
    assert m['baseline_allocated_bytes']==512 and m['end_allocated_bytes']==1024
    assert m['peak_allocated_bytes']==2048 and m['retained_new_allocation_bytes']==512
    # Free a pre-existing allocation too: net zero, retained allocation remains512.
    e['history'].append(dict(action='free_requested',addr=10,size=512))
    e['points'][-1].update(position=6,allocated_bytes=512,active_blocks_sha256=_digest([(20,512)]))
    m=_cuda_metrics(e,[dict(parent_invocation_id=None)])['invocations'][0]
    assert m['end_allocated_bytes']==m['baseline_allocated_bytes']
    assert m['retained_new_allocation_bytes']==512


def test_history_loss_or_snapshot_disagreement_rejects_peak():
    for kind in ('loss','counter','hash'):
        e=history_fixture()
        if kind=='loss':e['history'][2]['action']='free_requested'
        elif kind=='counter':e['points'][-1]['allocated_bytes']=999
        else:e['points'][-1]['active_blocks_sha256']='bad'
        with pytest.raises(ValueError):_cuda_metrics(e,[dict(parent_invocation_id=None)])


def test_requested_bytes_are_not_allocator_block_bytes():
    e=history_fixture();e['initial_requested_sizes']=[(10,64)]
    e['history'][1]['size']=64;e['history'][2]['size']=520;e['history'][3]['size']=520
    events=[dict(start_us=1,trace_index=20,address=20,bytes=512,device_type=1,device_id=0),
        dict(start_us=2,trace_index=21,address=30,bytes=1024,device_type=1,device_id=0)]
    m=_cuda_metrics(e,[dict(parent_invocation_id=None)],memory_events=events)
    assert m['invocations'][0]['peak_allocated_bytes']==2048
    assert m['invocations'][0]['retained_new_allocation_bytes']==512
    assert len(m['allocation_block_size_proof'])==2
    with pytest.raises(ValueError):_cuda_metrics(e,[dict(parent_invocation_id=None)],memory_events=events[:1])


def test_new_diagnostic_overhead_is_measured_with_existing_priming_and_uncertainty(tmp_path):
    from hyso.performance import measure_layer_overhead,LayerOverheadProtocol,LayerOverheadResult
    recipe=ColdStartRecipe('layer_diagnostic_factories:NestedRetaining',input_shape=(1,1,2,8),factory_search_path=Path(__file__).parent)
    result=measure_layer_overhead(recipe,protocol=LayerOverheadProtocol(trials=2),capture_protocol=
        ProfilingProtocol(warmup_iterations=2,measure_iterations=3,profile_memory=True,allocation_lifetimes=True,
            cpu_memory_limit_bytes=3*1024**3,timeout_seconds=45))
    d=result.to_dict();m=d['measurements']
    assert m['matched_context'] and m['uncertainty']['pairs']==2
    assert m['observed_mean_difference_ms']>0
    if m['status']=='unsupported':assert m['signed_overhead_percent'] is None
    for pair in d['raw_evidence']['pairs']:
        assert pair['instrumented']['protocol']['instrumentation']['evidence_version']==6
    result.to_json(tmp_path/'allocator-overhead.json');result.to_csv(tmp_path/'allocator-overhead.csv')
    assert LayerOverheadResult.from_csv(tmp_path/'allocator-overhead.csv').to_dict()==d
