"""Portable ledger/conversion tests. Real CUDA Events use the bounded native test."""
import copy

import pytest

from hyso.performance._profile_evidence import cuda_event_timings
from hyso.performance.profiling import _layer_metrics


def example():
    rows=[dict(invocation_id=0,measurement_index=0,module_id='m0',parent_invocation_id=None,
        cpu_total_us=100,cpu_exclusive_us=100,cuda_activity_us=None)]
    cfg=dict(version=1,requested=True,device='cuda:0',stream_id=0,host_thread_id=123,
        synchronized=True,scope_reason=None,clock='torch.cuda.Event.elapsed_time_ms',
        scope='owned_eager_single_host_thread_and_stream; unmanaged_native_async_excluded',
        validation='serialized_ledger_and_ms_to_us_only; CUDA_event_objects_not_serialized_or_replayed',
        pairs=[dict(invocation_id=0,measurement_index=0,device='cuda:0',start_event_index=0,end_event_index=1,
            start_stream_id=0,end_stream_id=0,start_thread_id=123,end_thread_id=123,
            start_recorded=True,end_recorded=True,start_complete=True,end_complete=True,
            elapsed_ms=.125,error=None)])
    return rows,cfg


def test_elapsed_ms_to_us_is_separate_from_cpu_and_kernel_time():
    rows,cfg=example();converted=cuda_event_timings(rows,cfg,'cuda:0')
    assert converted[0]['cuda_elapsed_us']==.125*1000==125
    assert converted[0]['cpu_total_us']==100 and converted[0]['cuda_activity_us'] is None
    assert converted[0]['cuda_exclusive_elapsed_us'] is None
    inventory=[dict(module_id='m0',path='',aliases=[''])]
    m=_layer_metrics(inventory,converted,cuda_events=cfg)
    assert m['status']=='measured' and m['modules'][0]['cuda_elapsed_us']==125
    assert m['cuda_layer_status']=='unsupported' and m['cuda_event_timing']['status']=='measured'


@pytest.mark.parametrize('mutation',[
    lambda c:c.update(version=True),lambda c:c.update(synchronized=False),
    lambda c:c.update(device='cuda:1'),lambda c:c.update(clock='CPU_wall_clock'),
    lambda c:c.update(validation='independent_GPU_replay'),lambda c:c['pairs'].clear(),
    lambda c:c['pairs'][0].update(invocation_id=1),
    lambda c:c['pairs'][0].update(measurement_index=1),
    lambda c:c['pairs'][0].update(start_event_index=1),
    lambda c:c['pairs'][0].update(end_event_index=0),
    lambda c:c['pairs'][0].update(end_stream_id=1),
    lambda c:c['pairs'][0].update(end_stream_id=False),
    lambda c:c['pairs'][0].update(end_thread_id=456),
    lambda c:c['pairs'][0].update(device='cuda:1'),
    lambda c:c['pairs'][0].update(elapsed_ms=float('nan')),
    lambda c:c['pairs'][0].update(elapsed_ms=True),
    lambda c:c['pairs'][0].update(elapsed_ms=1e308),
    lambda c:c['pairs'][0].update(start_recorded=False),
])
def test_manipulated_pair_device_stream_and_clock_rejected(mutation):
    rows,cfg=example();mutation(cfg)
    with pytest.raises(ValueError):cuda_event_timings(rows,cfg,'cuda:0')


@pytest.mark.parametrize('elapsed',[0.,-1.,None])
def test_nonpositive_or_missing_time_is_null_and_partial(elapsed):
    rows,cfg=example();cfg['pairs'][0].update(elapsed_ms=elapsed,error='nonpositive_or_nonfinite_CUDA_event_elapsed')
    values=cuda_event_timings(rows,cfg,'cuda:0')
    assert values[0]['cuda_elapsed_us'] is None and values[0]['cuda_elapsed_status']=='invalid'
    m=_layer_metrics([dict(module_id='m0')],values,cuda_events=cfg)
    assert m['status']=='partial' and m['cuda_event_timing']['status']=='partial'


def test_incomplete_pair_has_no_time():
    rows,cfg=example();cfg['pairs'][0].update(end_complete=False,elapsed_ms=None,error='missing_or_incomplete_CUDA_event_pair')
    assert cuda_event_timings(rows,cfg,'cuda:0')[0]['cuda_elapsed_us'] is None


@pytest.mark.parametrize('reason',['multiple_or_changed_CUDA_streams','cross_thread_module_or_dispatch',
    'unverified_custom_CUDA_dispatch','unverified_async_Python_thread','unverified_existing_Python_threads'])
def test_unsupported_scope_never_reports_a_duration(reason):
    rows,cfg=example();cfg.update(scope_reason=reason);cfg['pairs'][0]['elapsed_ms']=None
    values=cuda_event_timings(rows,cfg,'cuda:0')
    assert values[0]['cuda_elapsed_us'] is None and values[0]['cuda_elapsed_reason']==reason
    assert values[0]['cuda_elapsed_status']=='unsupported'


def test_cpu_not_requested_never_initializes_cuda():
    import torch
    before=torch.cuda.is_initialized();rows,cfg=example()
    cfg.update(requested=False,device='cpu',stream_id=None,host_thread_id=None,synchronized=False,pairs=[])
    values=cuda_event_timings(rows,cfg,'cpu')
    assert values[0]['cuda_elapsed_us'] is None and values[0]['cuda_elapsed_status']=='not_requested'
    assert torch.cuda.is_initialized()==before
    cfg['pairs']=example()[1]['pairs']
    with pytest.raises(ValueError):cuda_event_timings(rows,cfg,'cpu')


def test_repeated_invocations_are_aggregated_only_as_observed_inclusive_intervals():
    rows,cfg=example();rows.append(dict(rows[0],invocation_id=1,measurement_index=1))
    cfg['pairs'].append(dict(cfg['pairs'][0],invocation_id=1,measurement_index=1,start_event_index=2,end_event_index=3,elapsed_ms=.25))
    result=cuda_event_timings(rows,cfg,'cuda:0')
    m=_layer_metrics([dict(module_id='m0')],result,cuda_events=cfg)
    assert m['modules'][0]['cuda_elapsed_us']==(.125+.25)*1000
    assert m['modules'][0]['cuda_exclusive_elapsed_us'] is None
    broken=copy.deepcopy(cfg);broken['pairs'][1]['end_event_index']=1
    with pytest.raises(ValueError):cuda_event_timings(rows,broken,'cuda:0')


def test_profiler_resource_quotas_are_opt_in():
    from dataclasses import asdict
    from hyso.performance import ProfilingProtocol,ColdStartRecipe
    defaults=asdict(ProfilingProtocol())
    assert all(defaults[k] is None for k in ('cpu_memory_limit_bytes','input_limit_bytes','cuda_memory_fraction'))
    explicit=ProfilingProtocol(cpu_memory_limit_bytes=256*1024**2,input_limit_bytes=1024,cuda_memory_fraction=.1)
    assert explicit.cpu_memory_limit_bytes==256*1024**2 and explicit.cuda_memory_fraction==.1
    # Precision expansion is outside P13; retain the existing CUDA API contract.
    with pytest.raises(ValueError,match='unsupported precision'):
        ColdStartRecipe('profiling_factories:tiny_cuda',device='cuda:0',precision='fp64')


@pytest.mark.parametrize('limit',[None,256*1024**2])
def test_optional_private_commit_limit_keeps_single_process_job(limit):
    import os
    import json
    from pathlib import Path
    import subprocess
    import sys
    root=Path(__file__).resolve().parents[1]
    code="""
import importlib.util,json,sys,ctypes as c
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
guard=load(sys.argv[1],'guard');layout=load(sys.argv[2],'layout')
limit=None if sys.argv[3]=='None' else int(sys.argv[3]);api,handle=guard._memory_guard(limit)
probe=layout._Job(4*1024**3,4*1024**3,1)
try:
 info=probe.Extended();assert api.QueryInformationJobObject(handle,9,c.byref(info),c.sizeof(info),None)
 print(json.dumps(dict(flags=info.basic.flags,processes=info.basic.processes,process_bytes=info.process_memory,job_bytes=info.job_memory)))
finally:probe.close();api.CloseHandle(handle)
"""
    result=subprocess.run([sys.executable,'-I','-S','-B','-c',code,str(root/'hyso/performance/_batch_worker.py'),
        str(root/'hyso/performance/_training_process.py'),str(limit)],capture_output=True,text=True,timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    assert result.returncode==0,result.stderr
    value=json.loads(result.stdout);assert value['processes']==1 and value['flags'] & 8
    if limit is None:
        assert value['flags'] & 0x300==0 and value['process_bytes']==value['job_bytes']==0
    else:assert value['flags'] & 0x300==0x300 and value['process_bytes']==value['job_bytes']==limit


def test_default_cpu_profile_without_resource_quotas(tmp_path):
    import torch
    from pathlib import Path
    from hyso.performance import (ColdStartRecipe,ProfilingProtocol,profile_operators,profile_layers)
    before=torch.cuda.is_initialized()
    source=profile_operators(ColdStartRecipe('profiling_factories:tiny_cuda',
        factory_search_path=Path(__file__).parent,input_shape=(1,2,3,3)),
        protocol=ProfilingProtocol(warmup_iterations=0,measure_iterations=1,timeout_seconds=45))
    cfg=source.to_dict()['protocol']['configuration']
    assert all(cfg[k] is None for k in ('cpu_memory_limit_bytes','input_limit_bytes','cuda_memory_fraction'))
    assert profile_layers(source).to_dict()['measurements']['status']=='measured'
    assert torch.cuda.is_initialized()==before
