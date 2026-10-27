"""P34 contracts; modified-copy fixtures below are never native GPU evidence."""
import copy
import json
import os
from pathlib import Path
import uuid

import pytest

from hyso.performance import DeviceProtocol,RuntimeResult,compare_devices
from hyso.performance.devices import _analyze_devices,_cuda_hardware,_timed_forward
from hyso.performance.runtime_result import _hash
from test_deployment_analysis import cpu,recipe

GPU='GPU-00000000-0000-0000-abcd-000000000001'


def rebind(run):
    run['report']['request_sha256']=_hash({k:run[k] for k in ('recipe','protocol','device')})


@pytest.fixture
def pair(cpu):
    # Copy of native CPU capture, with an explicitly simulated CUDA contract.
    # This fixture tests validation arithmetic only. Native acceptance is separate.
    e=cpu.to_dict()['evidence'];e['devices'].append(GPU)
    run=copy.deepcopy(e['runs'][0]);e['runs'].append(run)
    run['token']=str(uuid.uuid4());run['controller']['token']=run['token'];run['report']['token']=run['token']
    run['device']=GPU;run['recipe']['gpu_uuid']=GPU
    ident=run['report']['identity'];ident['device_id']=GPU;ident['device_name']='SIMULATED GPU CONTRACT'
    ident['hardware']=dict(type='cuda',physical_id=GPU,identity_source='torch.cuda.get_device_properties.uuid',
        logical_index=0,visible_device_count=1,allocator_backend='native',total_memory_bytes=1024**3)
    ident['synchronization']='device_drain_before_and_after_each_forward'
    ident['execution'].update(input_device='cuda:0',output_device='cuda:0')
    run['report']['memory'].update(cuda_peak_allocated_bytes=2048,cuda_peak_reserved_bytes=4096)
    rebind(run)
    return e


def test_matching_contract_axes_and_nulls(pair):
    a=_analyze_devices(pair)
    assert a['comparable'] and a['axes']['latency']['comparable'] and a['axes']['throughput']['comparable']
    assert a['axes']['logical_model_bytes']['comparable']
    assert all(not a['axes'][key]['comparable'] for key in
        ('rss_before_bytes','rss_after_bytes','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes','energy','quality'))
    assert a['ranking'] is None
    assert all(r['quality']['value'] is None and r['energy']['value'] is None for r in a['devices'])


@pytest.mark.parametrize('field',['model_sha256','input_sha256','factory_sha256','runtime','shape','settings','configuration'])
def test_incompatible_contract_stays_conditional(pair,field):
    run=pair['runs'][1];ident=run['report']['identity']
    if field=='runtime':ident[field]['torch']='different'
    elif field=='shape':ident[field][2]*=2
    elif field=='settings':ident['execution']['settings']['matmul_allow_tf32']=True
    elif field=='configuration':run['recipe']['configuration']['different']=True;rebind(run)
    else:ident[field]='f'*64
    a=_analyze_devices(pair)
    assert not a['comparable'] and a['differences'] and not a['axes']['latency']['comparable']
    assert a['ranking'] is None


@pytest.mark.parametrize('key',['device_id','precision','backend','worker_pid','physical_id','logical_index',
    'visible_device_count','synchronization','input_device','output_device','output_dtype','input_unchanged',
    'model_unchanged','completed','window_ns','cuda_memory_fraction'])
def test_forged_or_inconsistent_execution_rejected(pair,key):
    r=pair['runs'][1];i=r['report']['identity']
    if key in ('physical_id','logical_index','visible_device_count'):i['hardware'][key]='wrong'
    elif key in ('input_device','output_device','output_dtype'):i['execution'][key]='wrong'
    elif key in i:i[key]='wrong'
    else:r['report'][key]='wrong'
    with pytest.raises(ValueError):_analyze_devices(pair)


@pytest.mark.parametrize('key',['rss_before_bytes','rss_after_bytes','cuda_peak_allocated_bytes',
    'cuda_peak_reserved_bytes','logical_model_bytes'])
def test_missing_memory_never_blocks_latency_or_invents_value(pair,key):
    for r in pair['runs']:r['report']['memory'].pop(key)
    a=_analyze_devices(pair)
    assert a['comparable'] and not a['axes'][key]['comparable']
    assert all(key not in r['memory']['value'] for r in a['devices'])


@pytest.mark.parametrize('unit,scope',[('MiB','registered_parameter_and_buffer_logical_bytes'),('byte','physical_VRAM')])
def test_incompatible_memory_domain_is_not_comparable(pair,unit,scope):
    pair['runs'][1]['report']['memory_domains']['logical_model_bytes']=dict(unit=unit,scope=scope)
    a=_analyze_devices(pair)
    assert a['comparable'] and not a['axes']['logical_model_bytes']['comparable'] and a['ranking'] is None


def test_all_memory_missing_and_inconsistent_allocator(pair):
    pair['runs'][0]['report']['memory']={}
    a=_analyze_devices(pair);assert a['comparable'] and a['devices'][0]['memory']['status']=='unavailable'
    pair['runs'][1]['report']['memory']['cuda_peak_allocated_bytes']=99999
    with pytest.raises(ValueError,match='allocator'):_analyze_devices(pair)


@pytest.mark.parametrize('kind',['missing_report','exit_failed','timed_out','terminal_failed'])
def test_unsuccessful_worker_cannot_compare(pair,kind):
    run=pair['runs'][1]
    if kind=='missing_report':run['report']=None
    elif kind=='exit_failed':run['controller']['exit_code']=42
    elif kind=='timed_out':run['controller']['reason']='heartbeat_timeout'
    else:run['report'].update(status='failed',reason='controlled fixture failure')
    a=_analyze_devices(pair)
    assert not a['comparable'] and a['devices'][1]['latency']['value'] is None


@pytest.mark.parametrize('target',['token','request_sha256','cleanup','limits','protocol'])
def test_binding_cleanup_and_protocol_corruption(pair,target):
    r=pair['runs'][1]
    if target in ('token','request_sha256'):r['report'][target]='wrong'
    elif target=='cleanup':r['controller']['cleanup']['active_processes']=1
    elif target=='limits':r['controller']['memory_limits']['process_memory_bytes']=123
    else:r['protocol']['samples']+=1
    with pytest.raises(ValueError):_analyze_devices(pair)


@pytest.mark.parametrize('devices',[['cuda:0','cuda:1'],[GPU,GPU.upper().replace('GPU-','GPU-')],
    [GPU,GPU],['GPU-00000000-0000-0000'],[0]])
def test_no_logical_alias_or_duplicate_device(devices,tmp_path):
    with pytest.raises(ValueError):compare_devices(recipe(),devices,tmp_path/'run')
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('values',[dict(cuda_memory_fraction=0),dict(cuda_memory_fraction=1.1),
    dict(cuda_memory_fraction=True),dict(process_memory_bytes=0),dict(max_processes=True),
    dict(process_memory_bytes=10,job_memory_bytes=9)])
def test_invalid_explicit_quotas(values):
    with pytest.raises(ValueError):DeviceProtocol(**values)


def test_quotas_default_to_unlimited(cpu):
    p=DeviceProtocol();assert p.cuda_memory_fraction is None and p.process_memory_bytes is None
    assert p.job_memory_bytes is None and p.max_processes is None
    c=cpu.to_dict()['evidence']['runs'][0]['controller']
    assert c['memory_limits']==dict(process_memory_bytes=None,job_memory_bytes=None) and c['process_limit'] is None


def test_cuda_async_completion_order(monkeypatch):
    from hyso.performance import devices
    order=[];ticks=iter([100,450])
    monkeypatch.setattr(devices.time,'perf_counter_ns',lambda:order.append('clock') or next(ticks))
    output,elapsed=_timed_forward(lambda x:order.append('forward') or 'output',None,lambda:order.append('sync'))
    assert order==['sync','clock','forward','sync','clock'] and output=='output' and elapsed==350


def test_runtime_observed_uuid_mismatch_rejected():
    from types import SimpleNamespace
    fake=SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda _:SimpleNamespace(
        uuid='00000000-0000-0000-0000-000000000002'),device_count=lambda:1))
    with pytest.raises(ValueError,match='physical GPU UUID'):_cuda_hardware(fake,GPU)


def test_new_and_legacy_json_csv(cpu,tmp_path):
    # Sanitized-path immutable schema fixtures derived from the pre-change CPU
    # export, with hashes recomputed. They are not fresh native acceptance.
    fixtures=Path(__file__).parent/'fixtures'
    old_json=RuntimeResult.from_json(fixtures/'device-comparison-v1.json')
    old_csv=RuntimeResult.from_csv(fixtures/'device-comparison-v1.csv')
    assert old_json.to_dict()==old_csv.to_dict()
    assert 'comparison_version' not in old_json.to_dict()['evidence']
    # The real pre-change CPU export retains exactly its original analysis/hash.
    saved=os.environ.get('HYSO_P34_EVIDENCE')
    if saved:
        j=RuntimeResult.from_json(Path(saved)/'legacy.json');c=RuntimeResult.from_csv(Path(saved)/'legacy.csv')
        assert j.to_dict()==c.to_dict() and 'comparison_version' not in j.to_dict()['evidence']
    for suffix in ('json','csv'):
        path=tmp_path/('new.'+suffix);getattr(cpu,'to_'+suffix)(path)
        assert getattr(RuntimeResult,'from_'+suffix)(path).to_dict()==cpu.to_dict()


def test_version_cannot_be_guessed(pair):
    pair['comparison_version']=True
    with pytest.raises(ValueError):_analyze_devices(pair)


@pytest.mark.skipif(not os.environ.get('HYSO_P34_NATIVE_RESULT'),reason='explicit saved native CPU/GPU acceptance')
def test_saved_native_cpu_gpu():
    p=Path(os.environ['HYSO_P34_NATIVE_RESULT']);r=RuntimeResult.from_json(p)
    assert RuntimeResult.from_csv(p.with_suffix('.csv')).to_dict()==r.to_dict()
    a=r.to_dict()['analysis'];assert a['comparable']
    cpu,gpu=a['devices'];assert cpu['identity']['hardware']['type']=='cpu' and gpu['identity']['hardware']['type']=='cuda'
    assert cpu['identity']['device_id']!=gpu['identity']['device_id']
    assert all(x['identity']['precision']=='float32' for x in (cpu,gpu))
    assert cpu['identity']['model_sha256']==gpu['identity']['model_sha256']
    assert cpu['identity']['input_sha256']==gpu['identity']['input_sha256']
    assert gpu['memory']['value']['cuda_peak_reserved_bytes']>=gpu['memory']['value']['cuda_peak_allocated_bytes']>0
    assert all(x['report']['cuda_memory_fraction'] is None for x in r.to_dict()['evidence']['runs'])
