"""Offline native gates plus ONE real CPU-only owned rejection/cleanup probe.

No TensorRT build, C++ compiler or CUDA execution is started here. Allocator
simulation results are not native runtime acceptance.
"""
from dataclasses import asdict,replace
import ctypes
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from hyso import export as ex
from hyso.export import native
from hyso.export import _worker
from hyso.export.artifacts import file_hash
from tools.export_acceptance import runner as r

UUID='GPU-00000000-0000-0000-0000-000000000001'
@pytest.fixture
def config(tmp_path):
    file=tmp_path/'declared.dll';file.write_bytes(b'offline identity fixture, not a DLL')
    return ex.NativeConfig(UUID,{str(file):file_hash(file)})

@pytest.mark.parametrize('field,value',[('workspace_bytes',1024**3),('max_gpu_bytes',1024**3),('process_memory_bytes',4*1024**3),
    ('job_memory_bytes',7*1024**3),('max_processes',5),('allocator_fraction',.2),('timeout_seconds',121),('gpu_uuid','cuda:0')])
def test_native_ceiling_and_physical_identity_rejections(config,field,value):
    with pytest.raises(ex.ExportError):replace(config,**{field:value})

def test_pre_admission_hash_mismatch_never_launches(config,tmp_path,monkeypatch):
    Path(next(iter(config.allowed_binaries))).write_bytes(b'modified')
    from hyso.performance import stability
    monkeypatch.setattr(stability,'_run_owned',lambda *a,**k:pytest.fail('bad identity launched worker'))
    with pytest.raises(ex.ExportError,match='DLL_INTEGRITY'):native.run_owned('tensorrt',tmp_path/'worker',{},config)

def test_loaded_binary_hash_path_and_file_id_mismatch_rejected(config,monkeypatch):
    p=next(iter(config.allowed_binaries));canonical=os.path.normcase(str(Path(p).resolve()))
    expected={canonical:dict(file_id=17,bytes=10)}
    for damage in ('hash','path','file_id'):
        row=dict(sha256=config.allowed_binaries[p],file_id=17,bytes=10);path=canonical
        if damage=='hash':row['sha256']='0'*64
        if damage=='path':path=canonical+'.foreign'
        if damage=='file_id':row['file_id']=999
        monkeypatch.setattr(_worker,'loaded_binaries',lambda path=path,row=row:{path:row})
        with pytest.raises(ex.ExportError,match='DLL_INTEGRITY'):_worker.verify_loaded(config,expected)

def test_gpu_operations_refused_before_os_admission(config,monkeypatch):
    monkeypatch.delenv('HYSO_RUNTIME_ADMITTED_PID',raising=False)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('GPU queried before admission'))
    with pytest.raises(ex.ExportError,match='NATIVE_ADMISSION'):_worker._gpu(config,{})

def test_allocator_offline_simulation_caps_requests_and_latches_real_oom():
    class Buffer:
        def __init__(self,n):self.n=n
        def data_ptr(self):return ((id(self)+255)//256)*256
        def numel(self):return self.n
    fake=SimpleNamespace(uint8='uint8',OutOfMemoryError=MemoryError,empty=lambda shape,**kw:Buffer(shape[0]))
    allocator=_worker.gpu_allocator(SimpleNamespace(IGpuAllocator=object),fake,16)
    pointer=allocator.allocate(12,256,0);assert pointer and allocator.used==allocator.peak==12
    assert allocator.allocate(8,256,0)==0 and allocator.used==12 and allocator.denied
    assert allocator.deallocate(pointer) and allocator.used==0
    def oom(*a,**kw):raise MemoryError('simulated, no GPU allocation')
    fake.empty=oom
    assert allocator.allocate(1,256,0)==0 and allocator.fatal_oom

@pytest.mark.parametrize('damage',['flags','breakaway','process','job','count'])
def test_job_readback_offline_simulation_fails_closed(damage):
    from hyso.performance._training_process import _Job
    class Basic(ctypes.Structure):_fields_=[('flags',ctypes.c_ulong),('processes',ctypes.c_ulong)]
    class Extended(ctypes.Structure):_fields_=[('basic',Basic),('process_memory',ctypes.c_size_t),('job_memory',ctypes.c_size_t)]
    def query(handle,kind,value,*args):
        value=value._obj;value.basic.flags=0x2308;value.basic.processes=4;value.process_memory=3;value.job_memory=6
        if damage=='flags':value.basic.flags=0x308
        if damage=='breakaway':value.basic.flags|=0x1000
        if damage=='process':value.process_memory=4
        if damage=='job':value.job_memory=7
        if damage=='count':value.basic.processes=5
        return True
    job=object.__new__(_Job);job.Extended=Extended;job.handle=123;job.api=SimpleNamespace(QueryInformationJobObject=query)
    with pytest.raises(RuntimeError,match='readback mismatch'):job.verify_limits(3,6,4)

def test_real_owned_cpu_rejection_reads_limits_and_cleans_tree(config,tmp_path):
    from hyso.performance.stability import _run_owned
    assert torch.version.cuda is None and not torch.cuda.is_initialized()
    root=_run_owned(tmp_path/'owned-cpu-probe','export',dict(operation='reject_without_GPU',recipe={'gpu_uuid':None},native=asdict(config),
        protocol=dict(process_memory_bytes=3*1024**3,job_memory_bytes=6*1024**3,max_processes=4)),
        startup_timeout=45,heartbeat_timeout=15,max_runtime=20,max_bytes=1024**2)
    controller=json.loads((root/'controller.json').read_text());report=json.loads((root/'report.json').read_text())
    assert controller['reason'] is None and controller['exit_code']==0 and controller['cleanup']['active_processes']==0
    assert controller['limit_readback']['verified'] and controller['limit_readback']['no_breakaway']
    assert controller['limit_readback']['process_memory_bytes']==3*1024**3 and controller['limit_readback']['job_memory_bytes']==6*1024**3
    assert report['status']=='FAIL' and report['code']=='UNSUPPORTED_BACKEND' and report['cuda_initialized'] is False
    assert report['torch_version'].endswith('+cpu') and report['pid']==controller['launcher_pid'] and report['token']==controller['token']

def test_export_campaign_exact_scope_no_implicit_retries():
    inventory=r.cpu_selection();assert len(inventory)==4
    assert {n.rsplit('::test_',1)[1] for n in inventory}=={'E01','E02_CPU','E05_CPU','E06_EP'}
    assert [f for group in r.GROUPS.values() for f in group]==['ADMISSION','E02_CUDA','E05_CUDA','E03_TRT','E04_TRT']
    rows={'PASS':{'status':'PASS'},'FAIL':{'status':'FAIL'},'BLOCKED':{'status':'BLOCKED'}}
    assert r.user.pending_cases(['PASS','FAIL','BLOCKED','NEW'],rows,set())==['NEW']
    assert r.user.pending_cases(['PASS','FAIL','BLOCKED','NEW'],rows,{'FAIL'})==['FAIL','NEW']

def test_independent_gpu_gate_accepts_clean_dependency_blocks_but_not_missing_cases():
    nodes=r.cpu_selection();plan={'cpu_inventory':nodes}
    rows={n:dict(status='BLOCKED' if 'CPP' in n or 'AOT' in n else 'PASS',source_valid=True) for n in nodes}
    assert r.user.cpu_group_ready({'cpu_results':rows},plan)
    rows.pop(next(iter(rows)));assert not r.user.cpu_group_ready({'cpu_results':rows},plan)

def test_native_hold_prevents_normal_failure_classification():
    assert not r.normal_cpu_completion('CPU_REGRESSIONS',{}, {'safety_hold':{'features':['E05_CPP']}},{})

def test_cpu_preparation_does_not_import_optional_runtimes_or_launch(monkeypatch,tmp_path):
    # Dependency inventory is stdlib discovery; it never imports TensorRT.
    import sys
    before=set(sys.modules)
    r.dependencies(tmp_path);r.native_inventory(tmp_path)
    assert not {'tensorrt','torch_tensorrt','onnxruntime'} & (set(sys.modules)-before)

def test_gpu_export_lock_is_nonblocking_and_released(tmp_path,monkeypatch):
    monkeypatch.setenv('LOCALAPPDATA',str(tmp_path))
    with native.gpu_lock(UUID):
        with pytest.raises(ex.ExportError,match='GPU_BUSY'):
            with native.gpu_lock(UUID):pytest.fail('concurrent GPU export admitted')
    with native.gpu_lock(UUID):pass

def test_cpu_native_crash_keeps_prior_pass_but_requires_explicit_retry(tmp_path,monkeypatch):
    # Unit simulation of sealed-reader output, not a runtime acceptance result.
    nodes=['tests/test_export_acceptance_scenarios.py::test_'+n for n in ('E01','E05_CPP','E06_EP')]
    for name in ('E01','E05_CPP'):
        folder=tmp_path/name;folder.mkdir();(folder/'raw.json').write_text('{}')
    (tmp_path/'user-safety-stop.json').write_text(json.dumps(dict(scenario='E05_CPP',code='CPP_NATIVE_CRASH')))
    cp={'cpu_results':{},'results':{},'history':[]}
    def base(cp,folder,request,receipt,plan):
        cp['cpu_results']={n:dict(status='PASS' if n==nodes[0] else 'BLOCKED',source_valid=True,evidence=str(folder/'cpu-results.json'),sha256='pending') for n in nodes}
        (folder/'cpu-results.json').write_text('{}')
    monkeypatch.setattr(r,'BASE_RECORD_CPU',base)
    r.record_cpu(cp,tmp_path,{'selected_nodes':nodes},{'cleanup_verified':True,'source_integrity':True,'cleanup':{'active_processes':0}},{'cpu_inventory':dict.fromkeys(nodes)})
    assert cp['cpu_results'][nodes[0]]['status']=='PASS' and nodes[2] not in cp['cpu_results']
    assert cp['safety_hold']['features']==[nodes[1]] and cp['cpu_results'][nodes[1]]['status']=='BLOCKED'
    assert r.user.pending_cases(nodes,cp['cpu_results'],set())==[nodes[2]]

@pytest.mark.parametrize('clean',[False,True])
def test_cpu_hold_clear_requires_new_verified_cleanup(tmp_path,monkeypatch,clean):
    node='tests/test_export_acceptance_scenarios.py::test_E05_CPP'
    cp={'safety_hold':{'features':[node]},'cpu_results':{node:dict(status='PASS',source_valid=True)},'results':{}}
    monkeypatch.setattr(r.user,'record_cpu',lambda *a:None)
    receipt=dict(cleanup_verified=clean,source_integrity=True,cleanup={'active_processes':0},exit_code=0,monitor_reason=None)
    r.record_cpu(cp,tmp_path,{'selected_nodes':[node]},receipt,{'cpu_inventory':{node:{}}})
    assert ('safety_hold' not in cp)==clean
    assert clean or cp['cpu_results'][node]['status']=='BLOCKED'

def test_export_request_scope_matches_cleanup_and_cpu_runtime_manifest(monkeypatch,tmp_path):
    monkeypatch.setattr(r,'BASE_REQUEST',lambda *a:dict(campaign_id='cuda-functional',profile='cpu'))
    request=r.request_for(tmp_path,tmp_path/'payload','CPU_REGRESSIONS',{'feature_tools':{'ort':{'site':'sealed'}}},{'node':{}})
    assert request['campaign_id']=='export-acceptance' and request['cpu_ort_manifest']=={'site':'sealed'}
    assert Path(request['argv'][0]).name=='cpu_worker.py'

@pytest.mark.parametrize('kind',['raw','prefixed','bytes','ascii_bytes'])
def test_uuid_display_matches_existing_controller_contract(kind):
    from uuid import UUID as Parse
    from tools.cuda_acceptance.identity import gpu_uuid
    value={'raw':str(Parse(UUID.removeprefix('GPU-'))),'prefixed':UUID,'bytes':Parse(UUID.removeprefix('GPU-')).bytes,'ascii_bytes':UUID.encode()}[kind]
    assert _worker.canonical_uuid(value)==gpu_uuid(value)==UUID

def test_output_budget_rejects_expanded_view_before_finite_mask(monkeypatch):
    from hyso.export._model import outputs
    view=torch.ones(1).expand(64*1024**2)
    monkeypatch.setattr(torch,'isfinite',lambda *a,**k:pytest.fail('oversized output reached finite mask'))
    with pytest.raises(ex.ExportError,match='RESOURCE_BUDGET'):outputs(view)
