"""Real CPU allocator/ownership checks; native memory is a separate scope."""
import copy
import json
from pathlib import Path

import pytest
import torch
from hyso.performance import ColdStartRecipe,ProfilingProtocol,profile_operators,profile_layers,LayerProfileResult,OperatorProfileResult


def capture(factory,kwargs=None,iterations=1):
    recipe=ColdStartRecipe(factory,input_shape=(1,1,2,8),model_kwargs=kwargs,factory_search_path=Path(__file__).parent)
    return profile_operators(recipe,protocol=ProfilingProtocol(warmup_iterations=0,measure_iterations=iterations,
        profile_memory=True,allocation_lifetimes=True,cpu_memory_ownership=True,timeout_seconds=45,
        cpu_memory_limit_bytes=3*1024**3))


@pytest.fixture(scope='module')
def pulse(tmp_path_factory):
    reference=tmp_path_factory.mktemp('cpu-known-peak')/'reference.json'
    return capture('cpu_layer_factories:ShortPeak',dict(reference_path=str(reference))),reference


@pytest.fixture(scope='module')
def shared():return capture('cpu_layer_factories:Shared',iterations=2)


def extended(source):return profile_layers(source).to_dict()['measurements']['allocator_diagnostics']['cpu_peak_ownership']


def test_known_short_peak_includes_live_baseline_without_sampling(pulse,tmp_path,monkeypatch):
    source,path=pulse;reference=json.loads(path.read_text());d=source.to_dict()
    assert d['protocol']['instrumentation']['evidence_version']==7
    def forbidden(*a,**kw):raise AssertionError('CPU analysis must not initialize CUDA')
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden)
    result=profile_layers(source);m=extended(source);row=m['invocations'][0]
    assert row['status']=='measured' and m['unknown_baseline_frees']==0
    assert row['baseline_captured_allocator_bytes']>=128  # input plus its snapshot copy, not zero
    assert row['peak_captured_allocator_bytes']==row['baseline_captured_allocator_bytes']+reference['expected_increment']
    assert reference['expected_increment']==81920
    assert row['end_captured_allocator_bytes']==row['baseline_captured_allocator_bytes']
    assert row['process_memory']['physical_layer_peak_bytes'] is None
    assert len(row['process_memory']['samples'])==2
    assert 'short_peaks_may_be_missed' in row['process_memory']['sampling']
    result.to_json(tmp_path/'cpu-peak.json');result.to_csv(tmp_path/'cpu-peak.csv')
    assert LayerProfileResult(json.loads((tmp_path/'cpu-peak.json').read_text())).to_dict()==result.to_dict()
    assert 'CPU_peak_ownership' in (tmp_path/'cpu-peak.csv').read_text()
    # Measured operator evidence excludes factory/setup allocations; trace memory includes them.
    from hyso.performance._cpu_layer_memory import measured_subtree_events
    assert measured_subtree_events(d['raw_evidence']['events'],d['raw_evidence']['invocations'])==d['raw_evidence']['events']


def test_native_allocation_is_not_PyTorch_allocator_peak(tmp_path):
    path=tmp_path/'native.json';source=capture('cpu_layer_factories:NativePeak',dict(reference_path=str(path)))
    ref=json.loads(path.read_text());row=extended(source)['invocations'][0]
    assert ref['native_bytes']==2*1024**2
    assert row['peak_captured_allocator_bytes']==row['baseline_captured_allocator_bytes']
    assert row['process_memory']['physical_layer_peak_bytes'] is None
    assert any(s['status']=='measured' for s in row['process_memory']['samples'])
    assert row['process_memory']['observed_lifetime_hwm_bytes']>=row['process_memory']['sampled_max_rss_bytes']
    profile_layers(source).to_json(tmp_path/'native-scope.json')


def test_foreign_Tensor_storage_is_a_reference_not_a_captured_allocator_baseline(tmp_path):
    source=capture('cpu_layer_factories:ForeignStorage');row=extended(source)['invocations'][0]
    assert row['untracked_entry_storage_capacity_bytes']==256
    foreign=[s for s in row['reference_graph']['entry'] if s['storage_uid'] in row['untracked_entry_storage_uids']]
    assert len(foreign)==1 and foreign[0]['allocation_generation'] is None
    assert foreign[0]['observed_module_holders']==['m0'] and foreign[0]['exclusive_owner'] is None
    assert foreign[0]['allocation_origin_scope']=='outside_captured_allocation_ledger'
    assert row['peak_captured_allocator_bytes']==row['baseline_captured_allocator_bytes']
    profile_layers(source).to_json(tmp_path/'foreign-storage.json')


def test_real_bounded_address_reuse_keeps_allocation_generations(tmp_path):
    path=tmp_path/'reuse.json';source=capture('cpu_layer_factories:Reuse',dict(reference_path=str(path)))
    ref=json.loads(path.read_text());row=extended(source)['invocations'][0]
    assert row['peak_captured_allocator_bytes']==row['baseline_captured_allocator_bytes']+4096
    addresses=ref['addresses']
    if len(set(addresses))==len(addresses):pytest.skip('no address reuse observed in this bounded native allocator run')
    events=source.to_dict()['raw_evidence']['memory_events']
    reused=next(a for a in addresses if addresses.count(a)>1)
    creations=[e['trace_index'] for e in events if e['address']==reused and e['bytes']>0]
    assert len(creations)>1 and len(set(creations))==len(creations)
    assert profile_layers(source).to_dict()['measurements']['allocator_diagnostics']['cpu']['invocations'][0]['retained_new_allocation_bytes']==0
    profile_layers(source).to_json(tmp_path/'address-reuse.json')


def test_shared_storage_references_are_not_arbitrary_exclusive_owners(shared,tmp_path):
    m=extended(shared);d=profile_layers(shared).to_dict()['measurements']['allocator_diagnostics']
    for i in (0,4):
        exit=m['invocations'][i]['reference_graph']['exit']
        shared_storage=next(s for s in exit if s['ownership']=='shared_observed')
        assert set(shared_storage['observed_module_holders'])=={'m0','m1','m2'}
        assert len(shared_storage['tensor_aliases'])>=2
        assert shared_storage['exclusive_owner'] is None and not shared_storage['complete_reference_graph']
        assert shared_storage['allocation_generation'] is not None
        assert shared_storage['allocation_origin_invocation_id']==i+2
        assert d['cpu']['invocations'][i]['retained_new_allocation_bytes']==64
    # Child producer exit may show one model holder: still never claim exclusivity.
    graph=m['invocations'][1]['reference_graph']['exit']
    assert any(s['ownership']=='single_model_holder_observed' and s['exclusive_owner'] is None for s in graph)
    producer=next(x for x in m['modules'] if x['module_id']=='m1')
    assert len(producer['invocation_ids'])==4
    refs=[e for s in graph for e in s['observed_references'] if e['holder'].get('module_id')=='m1']
    assert refs and all(e['holder']['module_aliases']==['producer','alias'] for e in refs)
    profile_layers(shared).to_json(tmp_path/'cpu-ownership.json');profile_layers(shared).to_csv(tmp_path/'cpu-ownership.csv')


def test_distinct_StorageImpl_interior_view_shares_allocation_without_new_owner(tmp_path):
    source=capture('cpu_layer_factories:SeparateStorageAliases');m=extended(source)
    graph=m['invocations'][0]['reference_graph']['exit']
    interior=next(s for s in graph if s['storage_allocation_offset_bytes']==4)
    assert len(interior['allocation_alias_storage_uids'])==2
    assert set(interior['allocation_module_holders'])=={'m0','m1','m2'}
    assert interior['allocation_ownership']=='shared_observed' and interior['exclusive_owner'] is None
    original=next(s for s in graph if s['storage_uid']!=interior['storage_uid'] and s['allocation_generation']==interior['allocation_generation'])
    assert original['storage_id']!=interior['storage_id']
    assert profile_layers(source).to_dict()['measurements']['allocator_diagnostics']['cpu']['invocations'][0]['retained_new_allocation_bytes']==64
    profile_layers(source).to_json(tmp_path/'interior-alias.json')


@pytest.mark.parametrize('field',['counter','edge','peak','boundary','holder','exclusive','contract'])
def test_invalid_counter_or_ownership_evidence_is_rejected(pulse,field):
    d=pulse[0].to_dict()
    if field=='counter':d['raw_evidence']['memory_events'][0]['total_allocated_bytes']+=1
    elif field=='edge':d['raw_evidence']['allocation_evidence']['cpu_context']['points'][-1]['references'][0]['tensor_uid']=-1
    elif field=='boundary':d['raw_evidence']['allocation_evidence']['cpu_context']['points']=[]
    elif field=='holder':
        edge=d['raw_evidence']['allocation_evidence']['cpu_context']['points'][-1]['references'][0]
        edge['holder']=dict(kind='module_attribute',module_id='invented',module_aliases=['invented'])
    elif field=='exclusive':
        d['measurements']['allocator_diagnostics']['cpu_peak_ownership']['invocations'][0]['reference_graph']['exit'][0]['exclusive_owner']='m0'
    elif field=='contract':
        d['protocol']['configuration']['cpu_memory_ownership']=False;d['protocol']['instrumentation']['evidence_version']=6
    else:d['measurements']['allocator_diagnostics']['cpu_peak_ownership']['invocations'][0]['peak_captured_allocator_bytes']+=1
    with pytest.raises(ValueError):OperatorProfileResult(d)


def test_CPU_mode_is_opt_in_and_never_launches_a_CUDA_recipe():
    assert ProfilingProtocol().cpu_memory_ownership is False
    with pytest.raises(ValueError):ProfilingProtocol(cpu_memory_ownership=True)
    recipe=ColdStartRecipe('cpu_layer_factories:Shared',device='cuda:0',factory_search_path=Path(__file__).parent)
    with pytest.raises(ValueError,match='requires CPU'):
        profile_operators(recipe,protocol=ProfilingProtocol(profile_memory=True,allocation_lifetimes=True,cpu_memory_ownership=True))


def test_untracked_native_storage_spans_do_not_double_count_alias_capacity():
    from hyso.performance._cpu_layer_memory import _witnessed_capacity
    assert _witnessed_capacity([dict(address=100,bytes=64),dict(address=116,bytes=32)])==64
    assert _witnessed_capacity([dict(address=100,bytes=64),dict(address=160,bytes=64)])==124
    assert _witnessed_capacity([])==0


def test_boundary_timestamp_ambiguity_cannot_publish_a_module_peak(pulse):
    from hyso.performance._layer_allocator import derive_allocator
    d=pulse[0].to_dict();raw=copy.deepcopy(d['raw_evidence'])
    root=raw['invocations'][0]
    span=next(e for e in raw['trace_events'] if e['name']==root['annotation'])
    first=next(e for e in raw['memory_events'] if span['start_us']<e['start_us']<span['start_us']+span['duration_us'] and e['bytes']>0)
    first['start_us']=span['start_us']
    m=derive_allocator(raw['allocation_evidence'],raw['memory_events'],raw['trace_events'],raw['invocations'],raw['module_inventory'],'cpu')['cpu_peak_ownership']
    assert m['invocations'][0]['status']=='partial'
    assert m['invocations'][0]['peak_captured_allocator_bytes'] is None
    assert m['modules'][0]['peak_captured_allocator_bytes'] is None


def test_real_CPU_capture_preserves_parent_model_RNG_gradient_optimizer_and_hooks(tmp_path):
    import random
    import numpy as np
    model=torch.nn.Linear(2,2).train();model.weight.grad=torch.ones_like(model.weight)
    optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9);optimizer.step()
    gradient=model.weight.grad;grad=gradient.clone();momentum=optimizer.state[model.weight]['momentum_buffer'];mom=momentum.clone()
    weights={k:v.clone() for k,v in model.state_dict().items()};handle=model.register_forward_hook(lambda *a:None)
    hooks=dict(model._forward_hooks);rng=torch.get_rng_state();py=random.getstate();ns=np.random.get_state();threads=torch.get_num_threads()
    try:
        capture('cpu_layer_factories:ShortPeak',dict(reference_path=str(tmp_path/'parent-check.json')))
        assert torch.equal(rng,torch.get_rng_state()) and random.getstate()==py
        assert np.array_equal(ns[1],np.random.get_state()[1]) and ns[2:]==np.random.get_state()[2:]
        assert model.training and dict(model._forward_hooks)==hooks and torch.get_num_threads()==threads
        assert model.weight.grad is gradient and torch.equal(gradient,grad)
        assert optimizer.state[model.weight]['momentum_buffer'] is momentum and torch.equal(momentum,mom)
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in weights.items())
    finally:handle.remove()
