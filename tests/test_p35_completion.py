"""P35 constructed protocol regressions and separately identified native replay."""
import base64
import copy
import hashlib
import os
from pathlib import Path
import random

import numpy as np
import onnx
import pytest
import torch

from hyso.performance import RuntimeResult,ProviderProtocol,WorkloadRecipe,profile_execution_providers
from hyso.performance import providers as module
from hyso.performance.runtime_result import _result
from test_provider_protocol import simulated,rebind,FakeORT


@pytest.fixture
def modern(monkeypatch,tmp_path):
    value,_=simulated(monkeypatch,tmp_path,providers=['CPUExecutionProvider'],expected={'*':'CPUExecutionProvider'},optimization='disabled')
    e=value.to_dict()['evidence'];e['provider_analysis_version']=2;e['graph_analysis_version']=2
    for event in e['report']['trace']:
        event['ph']='X';event['args']['node_index']='0'
    rebind(e)
    return e


def result(e):return _result('P35',e)
def assessment(e):return result(e).provider_diagnostics()['evidence_assessment']


def test_explicit_cpu_and_no_error_fallback(modern):
    d=assessment(modern)
    assert d['explicit_CPU_execution'] and d['node_coverage']['status']=='complete'
    assert d['placement_mismatch']['detected'] is False and not d['findings']
    assert d['execution_error_retry']['detected'] is None
    assert d['performance_impact']['latency_delta_ms'] is None


def test_cpu_in_session_list_unused_is_not_fallback(modern):
    e=modern;r=e['report'];e['providers']=['CUDAExecutionProvider','CPUExecutionProvider'];e['expected']={'*':'CUDAExecutionProvider'}
    r['requested_providers']=e['providers'];r['available_providers']=e['providers'];r['session_providers']=e['providers']
    r['trace'][0]['args']['provider']='CUDAExecutionProvider'
    from hyso.performance.runtime_result import _hash
    r['request_sha256']=_hash({k:e[k] for k in ('recipe','protocol','providers','expected','optimization','runtime_search_path','provider_options')})
    rebind(e);d=assessment(e)
    assert d['observed_execution_providers']==['CUDAExecutionProvider'] and d['placement_mismatch']['detected'] is False
    assert d['physical_device']['value'] is None and not d['findings']


def test_provider_priority_difference_is_configuration_not_execution_error(modern):
    e=modern;r=e['report'];e['providers']=['AzureExecutionProvider','CPUExecutionProvider'];r['requested_providers']=e['providers']
    r['available_providers']=e['providers'];r['session_providers']=list(reversed(e['providers']))
    from hyso.performance.runtime_result import _hash
    r['request_sha256']=_hash({k:e[k] for k in ('recipe','protocol','providers','expected','optimization','runtime_search_path','provider_options')})
    rebind(e);d=assessment(e)
    assert d['provider_priority']['status']=='configuration_order_difference'
    assert d['placement_mismatch']['detected'] is False and d['execution_error_retry']['detected'] is None


def test_partial_durations_are_not_reported_as_total(modern):
    r=modern['report'];event=copy.deepcopy(r['trace'][0]);event['ts']+=10;event.pop('dur');r['trace'].append(event)
    rebind(modern);d=assessment(modern);timing=d['operator_timings'][0]
    assert timing['total_host_node_us'] is None and timing['known_duration_sum_us']==4.5
    assert timing['event_count']==2 and timing['timed_event_count']==1 and timing['reason']
    assert d['placement_mismatch']['detected'] is None
    top=result(modern).provider_diagnostics()['operator_timings'][0]
    assert top['total_host_node_us'] is None and top['known_duration_sum_us']==4.5


@pytest.mark.parametrize('damage',['duplicate_event','conflicting_node_index','bad_node_index','foreign_session_pid'])
def test_broken_profile_identity_rejected(modern,damage):
    r=modern['report'];event=copy.deepcopy(r['trace'][0])
    if damage=='duplicate_event':r['trace'].append(event)
    elif damage=='conflicting_node_index':event['name']='other_kernel_time';event['ts']+=20;r['trace'].append(event)
    elif damage=='bad_node_index':r['trace'][0]['args']['node_index']='-1'
    else:r['trace'].append(dict(cat='Session',name='model_run',pid=-1,ph='X',ts=0,dur=1))
    rebind(modern)
    with pytest.raises(ValueError):result(modern)


@pytest.mark.parametrize('missing',['node_index','ts','ph','provider','trace'])
def test_missing_node_proof_stays_unknown(modern,missing):
    r=modern['report'];event=r['trace'][0]
    if missing=='trace':r['trace']=[]
    elif missing in ('node_index','provider'):event['args'].pop(missing)
    else:event.pop(missing)
    rebind(modern);d=assessment(modern)
    assert d['node_coverage']['status']=='partial_or_unknown'
    assert d['placement_mismatch']['detected'] is None


@pytest.mark.parametrize('damage',['input_hash','source_hash','input_changed','host_clock','profile_unit','sequence','time_order'])
def test_capture_context_corruption_rejected(modern,damage):
    r=modern['report'];c=r['capture_context']
    if damage=='input_hash':r['input_sha256']='f'*64;rebind(modern)
    elif damage=='source_hash':r['source']['module_sha256']=None
    elif damage=='input_changed':c['input_unchanged']=False
    elif damage=='host_clock':c['host_clock']['instance_id']='foreign'
    elif damage=='profile_unit':c['profile_clock']['unit']='ms'
    elif damage=='sequence':c['run_intervals'][0]['index']=1
    else:c['run_intervals'][0]['end_ns']=c['run_intervals'][0]['start_ns']-1
    with pytest.raises(ValueError):result(modern)


def test_duplicate_source_node_names_never_get_assignment(modern):
    r=modern['report'];model=onnx.load_model_from_string(base64.b64decode(r['model_base64']))
    model.graph.node.append(onnx.helper.make_node('Relu',['Y'],['Z'],name='relu'))
    raw=model.SerializeToString();r['model_base64']=base64.b64encode(raw).decode();r['optimized_base64']=r['model_base64']
    r['model_sha256']=hashlib.sha256(raw).hexdigest();r['optimized_sha256']=r['model_sha256']
    r['graph_sha256']=hashlib.sha256(model.graph.SerializeToString()).hexdigest();rebind(modern)
    d=assessment(modern)
    assert not d['node_coverage']['source_names_unique'] and d['placement_mismatch']['detected'] is None
    assert all(x['source_node'] is None for x in d['events'])


@pytest.mark.parametrize('kind',['timeout','crash','partial','missing_report','cleanup'])
def test_incomplete_or_failed_worker_is_not_execution_proof(modern,kind):
    if kind=='cleanup':
        modern['controller']['cleanup']['active_processes']=1
        with pytest.raises(ValueError):result(modern)
        return
    if kind=='timeout':modern['controller']['reason']='heartbeat_timeout'
    elif kind=='crash':modern['controller']['exit_code']=0xc0000005
    elif kind=='partial':modern['report'].update(status='partial',reason='interrupted',failure_kind='ort_error')
    else:modern['report']=None
    d=assessment(modern)
    assert d['status']=='unavailable' and not d['observed_execution_providers']
    assert d['execution_error_retry']['detected'] is None and d['partial_evidence']['reason']


def test_exception_category_does_not_invent_unsupported_provider():
    error=NotImplementedError('operator not implemented;no structured node evidence')
    assert module._failure_kind(error,'execution')=='ort_error'
    assert module._failure_kind(error,'provider_selection')=='unsupported_provider'
    assert module._failure_kind(ValueError('bad feed'),'input_validation')=='workload_error'


def test_public_crash_receipt_has_no_automatic_retry(modern,monkeypatch,tmp_path):
    calls=[];modern['controller']['exit_code']=0xc0000005
    def launch(*args,**kwargs):calls.append(1);return copy.deepcopy(modern)
    monkeypatch.setattr(module,'_launch',launch)
    value=profile_execution_providers(WorkloadRecipe(**modern['recipe']),tmp_path/'never_created',
        providers=['CPUExecutionProvider'],expected={'*':'CPUExecutionProvider'},optimization='disabled')
    assert len(calls)==1 and value.provider_diagnostics()['evidence_assessment']['status']=='unavailable'


def test_json_csv_lossless_and_no_clobber(modern,tmp_path):
    value=result(modern)
    for suffix in ('json','csv'):
        p=tmp_path/('result.'+suffix);getattr(value,'to_'+suffix)(p)
        other=getattr(RuntimeResult,'from_'+suffix)(p)
        assert other.to_dict()==value.to_dict() and other.provider_diagnostics()==value.provider_diagnostics()
        with pytest.raises(FileExistsError):getattr(value,'to_'+suffix)(p)


def test_unsupported_backend_does_not_receive_ORT_placement(modern):
    modern['report']['runtime']['name']='pytorch'
    with pytest.raises(ValueError,match='backend'):result(modern)


def test_native_archive_transfer_boundary_and_no_performance_claim(monkeypatch):
    directory=os.environ.get('HYSO_P35_CAPTURE_ROOT')
    if not directory:pytest.skip('explicit archived native directory required;never starts native inference')
    def forbidden(*a,**kw):raise AssertionError('offline replay executed native operation')
    monkeypatch.setattr(module,'_collect_profile',forbidden);monkeypatch.setattr(module,'_launch',forbidden)
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden)
    rng=torch.get_rng_state().clone();py=random.getstate();nrng=np.random.get_state()
    p=Path(directory)/'cuda-cpu-partition-recovered.json'
    value=RuntimeResult.from_json(p);d=value.provider_diagnostics()['evidence_assessment']
    assert d['placement_mismatch']['detected'] is True and d['node_coverage']['status']=='complete'
    finding=d['findings'][0]
    assert finding['node']=='scaler_cpu' and finding['observed_provider']=='CPUExecutionProvider'
    assert finding['previous_provider'] is None and d['execution_error_retry']['detected'] is None
    transfer=d['transfers'][0]
    assert transfer['op_type']=='MemcpyToHost' and transfer['host_node_duration_us']==425
    assert transfer['transferred_bytes'] is None and transfer['device_dma_duration_us'] is None
    assert d['dataflow_provider_boundaries'][0]['producer_node']=='relu_gpu'
    assert d['performance_impact']['status']=='needs_comparable_experiment' and d['performance_impact']['percentage'] is None
    assert torch.equal(rng,torch.get_rng_state()) and py==random.getstate() and np.array_equal(nrng[1],np.random.get_state()[1])


@pytest.mark.parametrize('version',[True,0,3])
def test_new_evidence_version_is_explicit(modern,version):
    modern['provider_analysis_version']=version
    with pytest.raises(ValueError):result(modern)


@pytest.mark.parametrize('case',['complete','last_run_missing','windows_missing','invalid_phase'])
def test_multirun_coverage_requires_each_profile_window(modern,case):
    from hyso.performance.runtime_result import _hash
    e=modern;r=e['report'];e['protocol']['runs']=2;r['completed_runs']=2
    r['request_sha256']=_hash({k:e[k] for k in ('recipe','protocol','providers','expected','optimization','runtime_search_path','provider_options')})
    context=r['capture_context'];span=copy.deepcopy(context['run_intervals'][0]);span['index']=1
    span['start_ns']=context['run_intervals'][0]['end_ns']+1;span['end_ns']=span['start_ns']+100
    context['run_intervals'].append(span)
    second=copy.deepcopy(r['trace'][0]);second['ts']=200 if case in ('complete','invalid_phase') else 110;r['trace'].append(second)
    if case!='windows_missing':
        for start in (90,190):r['trace'].append(dict(cat='Session',name='model_run',pid=e['controller']['launcher_pid'],ph='B' if case=='invalid_phase' else 'X',ts=start,dur=50))
    rebind(e);d=assessment(e)
    assert d['node_coverage']['events_per_source_node']=={'relu':2}
    if case=='complete':
        assert d['node_coverage']['per_run_association']=='complete' and d['placement_mismatch']['detected'] is False
    else:
        assert d['node_coverage']['status']=='partial_or_unknown' and d['placement_mismatch']['detected'] is None
