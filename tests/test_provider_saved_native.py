"""Read-only revalidation of explicit native P35 captures; never starts ORT/GPU."""
import copy
import os
from pathlib import Path

import pytest

from hyso.performance import RuntimeResult
from hyso.performance.runtime_result import _result, _hash


@pytest.fixture(scope='module')
def native():
    root=os.environ.get('HYSO_P35_CAPTURE_ROOT')
    if not root:pytest.skip('explicit saved native capture directory required; no execution implied')
    partition='cuda-cpu-partition' if (Path(root)/'cuda-cpu-partition.json').is_file() else 'cuda-cpu-partition-recovered'
    return {name:RuntimeResult.from_json(Path(root)/(stem+'.json')) for name,stem in (
        ('cpu','cpu-relu'),('fusion','cpu-fusion'),('cuda','cuda-relu'),
        ('partition',partition))}


def rebind(e):
    r=e['report'];r['trace_sha256']=_hash(r['trace'])
    r['profile_binding']=_hash(dict(session_id=e['token'],model_sha256=r['model_sha256'],
        trace_sha256=r['trace_sha256'],input_sha256=r['input_sha256'],providers=r['session_providers']))


@pytest.mark.parametrize('name',['cpu','fusion','cuda','partition'])
def test_native_completed_execution_cleanup_and_lossless_exports(native,name,tmp_path):
    result=native[name];data=result.to_dict();e=data['evidence'];d=result.provider_diagnostics()
    assert e['report']['status']=='completed' and e['report']['completed_runs']==1
    assert e['controller']['reason'] is None and e['controller']['exit_code']==0
    assert e['controller']['cleanup']['active_processes']==0
    assert d['events'] and all(x['duration_us'] is not None for x in d['events'])
    result.to_json(tmp_path/'result.json');result.to_csv(tmp_path/'result.csv')
    assert RuntimeResult.from_json(tmp_path/'result.json').to_dict()==data
    assert RuntimeResult.from_csv(tmp_path/'result.csv').to_dict()==data


def test_native_cpu_and_accelerated_events_are_distinct_from_provider_lists(native):
    cpu=native['cpu'].provider_diagnostics();cuda=native['cuda'].provider_diagnostics()
    assert cpu['runtime']['version']=='1.23.2' and cuda['runtime']['version']=='1.20.2'
    assert cpu['observed_execution_providers']==['CPUExecutionProvider']
    assert cuda['observed_execution_providers']==['CUDAExecutionProvider']
    assert 'CPUExecutionProvider' in cuda['enabled_session_providers']
    assert all(x['source_node']=='relu' and x['provider']=='CUDAExecutionProvider' for x in cuda['events'])


def test_native_fusion_stays_unmapped(native):
    import base64,onnx
    result=native['fusion'];r=result.to_dict()['evidence']['report']
    source=onnx.load_model_from_string(base64.b64decode(r['model_base64']))
    optimized=onnx.load_model_from_string(base64.b64decode(r['optimized_base64']))
    assert len(source.graph.node)==3 and len(optimized.graph.node)==1
    d=result.provider_diagnostics()
    assert d['source_node_mapping']['status']=='unknown'
    assert all(x['source_node'] is None for x in d['events'])
    assert result.to_dict()['analysis']['fallback_detected'] is None


def test_native_real_partition_fallback_is_not_an_error_retry_claim(native):
    result=native['partition'];d=result.provider_diagnostics()
    assert set(d['observed_execution_providers'])=={'CPUExecutionProvider','CUDAExecutionProvider'}
    events={x['source_node']:x for x in d['events'] if x['source_node']}
    assert events['relu_gpu']['provider']=='CUDAExecutionProvider'
    assert events['scaler_cpu']['provider']=='CPUExecutionProvider'
    assert events['scaler_cpu']['expected_provider']=='CUDAExecutionProvider'
    assert d['expectation_mismatch']['status']=='observed'
    assert result.to_dict()['analysis']['fallback_detected'] is True
    assert any(x['op_type']=='MemcpyToHost' and x['source_node'] is None for x in d['events'])
    assert d['execution_fallback']['detected'] is None


@pytest.mark.parametrize('field',['model_sha256','graph_sha256','optimized_sha256','trace_sha256','session_id','profile_binding'])
def test_real_capture_rejects_foreign_binding(native,field):
    e=native['partition'].to_dict()['evidence'];e['report'][field]='0'*64
    with pytest.raises(ValueError):_result('P35',e)


def test_real_capture_wrong_process_still_rejected_after_trace_rehash(native):
    e=native['partition'].to_dict()['evidence']
    for row in e['report']['trace']:
        if row.get('cat')=='Node':row['pid']+=1
    rebind(e)
    with pytest.raises(ValueError):_result('P35',e)


@pytest.mark.parametrize('damage',['missing_trace','missing_provider','missing_duration'])
def test_real_capture_missing_fields_never_become_complete(native,damage):
    e=native['cpu'].to_dict()['evidence']
    if damage=='missing_trace':e['report']['trace']=[]
    else:
        for row in e['report']['trace']:
            if row.get('cat')=='Node' and row.get('name','').endswith('_kernel_time'):
                if damage=='missing_provider':row['args'].pop('provider')
                else:row.pop('dur')
    rebind(e);result=_result('P35',e);d=result.provider_diagnostics()
    assert result.to_dict()['analysis']['fallback_detected'] is None
    if damage in ('missing_trace','missing_provider'):assert not d['observed_execution_providers']
    else:assert d['operator_timings'][0]['total_host_node_us'] is None
