"""Offline identity/retention/one-shot gates. Never launches a CUDA worker."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import uuid
import pytest
from hyso.export import ExportError,NativeConfig
from hyso.export import _worker,native
from hyso.export import _dll_identity as identity
from hyso.export.artifacts import Transaction,file_hash
from tools.cuda_acceptance import native_identity as shared
from tools.export_acceptance import runner as r
from tools.export_acceptance.diagnostic_gate import validate_ticket


@pytest.fixture
def fixture(tmp_path):
    dll=tmp_path/'onnxruntime.dll';dll.write_bytes(b'offline fixture, not executable')
    key=shared.canonical_path(dll);st=dll.stat()
    config=NativeConfig('GPU-00000000-0000-0000-0000-000000000001',{key:file_hash(dll)})
    ids={key:dict(file_id=st.st_ino,bytes=st.st_size)}
    actual={key:dict(ids[key],sha256=file_hash(dll))}
    return config,ids,actual


@pytest.mark.parametrize('damage',['none','hash','file_id','bytes','foreign','empty'])
def test_exact_identity_and_unexpected_dll(fixture,monkeypatch,damage):
    config,ids,actual=fixture;key=next(iter(actual))
    if damage=='hash':actual[key]['sha256']='0'*64
    if damage in ('file_id','bytes'):actual[key][damage]+=1
    if damage=='foreign':actual[key+'.foreign']=actual.pop(key)
    if damage=='empty':actual={}
    monkeypatch.setattr(_worker,'loaded_binaries',lambda:actual)
    if damage=='none':assert _worker.verify_loaded(config,ids)==actual
    else:
        with pytest.raises(ExportError) as error:_worker.verify_loaded(config,ids)
        assert error.value.code=='DLL_INTEGRITY'
        data=error.value.result
        assert data['rejected']
        if actual:
            row=data['rows'][0]
            assert row['mismatched_fields'] and row['actual_path'] and 'expected_identity' in row
            if damage=='foreign':assert not row['manifest_member']


def test_windows_canonical_path_and_shared_identity(fixture):
    config,ids,actual=fixture;key=next(iter(actual))
    altered=replace(config,allowed_binaries={key.upper():config.allowed_binaries[key]})
    assert not identity.compare(actual,altered.allowed_binaries,ids)['rejected']
    assert shared.file_identity(key)['file_id']==ids[key]['file_id']


@pytest.mark.parametrize('name,selected',[('onnxruntime.dll',True),('cudnn64_9.dll',True),('nvcuda64.dll',True),
    ('torch_cuda.dll',True),('nvrtc64_120_0.dll',False),('nvinfer_10.dll',False),('unrelated.dll',False)])
def test_p35_u13_default_selection_unchanged(name,selected):
    assert shared._selected(name) is selected


def test_export_enumeration_reuses_shared_and_keeps_additional_checks(monkeypatch,tmp_path):
    dll=tmp_path/'nvrtc64_120_0.dll';dll.write_bytes(b'fixture')
    def enumerate_loaded(*,selected):
        assert selected(dll.name) and selected('nvinfer_10.dll') and not selected('unrelated.dll')
        return {shared.canonical_path(dll):file_hash(dll)}
    from hyso import _native_identity as primitives
    monkeypatch.setattr(primitives,'loaded_binaries',enumerate_loaded)
    assert next(iter(_worker.loaded_binaries().values()))['file_id']==dll.stat().st_ino


def test_loaded_file_identity_unavailable_is_safety_failure(monkeypatch):
    from hyso import _native_identity as primitives
    monkeypatch.setattr(primitives,'loaded_binaries',lambda **kw:{'missing.dll':'0'*64})
    with pytest.raises(ExportError,match='DLL_INTEGRITY'):_worker.loaded_binaries()


def make_failure(folder,config,monkeypatch):
    token=str(uuid.uuid4())
    request=dict(operation='dll_diagnostic',native=config.__dict__,token=token,
        attempt_identity='offline-attempt',_binary_file_ids={})
    (folder/'request.json').write_text(json.dumps(request))
    def reject(*args):raise ExportError('DLL_INTEGRITY','offline simulated rejection',result={'rows':[{'mismatched_fields':['sha256']}]})
    monkeypatch.setattr(_worker,'_gpu',reject)
    with pytest.raises(ExportError):_worker.worker(folder,request)
    controller=dict(token=token,launcher_pid=os.getpid(),exit_code=1,reason=None,cleanup={'active_processes':0})
    (folder/'controller.json').write_text(json.dumps(controller))
    return controller


def test_failure_survives_owned_cleanup_with_member_hashes(fixture,tmp_path,monkeypatch):
    config,_,_=fixture
    with Transaction(tmp_path/'model.onnx',False,1024**2) as tx:
        folder=tx.root/'native';folder.mkdir()
        controller=make_failure(folder,config,monkeypatch)
        bundle=identity.failure_bundle(folder,controller)
        record=identity.retain(tmp_path,bundle)
        diagnostic=json.loads(bundle['records']['identity-failure.json']['text'])
        assert diagnostic['pid']==os.getpid() and diagnostic['attempt_identity']=='offline-attempt'
        assert diagnostic['runtime_versions']['torch'] and diagnostic['details']['rows']
    assert not tx.root.exists()
    assert identity.verify_retained(record)==bundle
    with pytest.raises(FileExistsError):identity.retain(tmp_path,bundle)
    Path(record['path']).write_text('{}')
    with pytest.raises(ExportError,match='hash mismatch'):identity.verify_retained(record)


def test_retained_member_tampering_rejected_even_with_new_outer_hash(fixture,tmp_path,monkeypatch):
    config,_,_=fixture;controller=make_failure(tmp_path,config,monkeypatch)
    record=identity.retain(tmp_path,identity.failure_bundle(tmp_path,controller))
    path=Path(record['path']);bundle=json.loads(path.read_text());bundle['records']['identity-failure.json']['text']='{}'
    path.write_text(json.dumps(bundle));record['sha256']=file_hash(path)
    with pytest.raises(ExportError,match='member hash'):identity.verify_retained(record)


@pytest.mark.parametrize('field',['pid','token','request_sha256'])
def test_failure_receipt_forgery_rejected(fixture,tmp_path,monkeypatch,field):
    config,_,_=fixture;controller=make_failure(tmp_path,config,monkeypatch)
    path=tmp_path/'identity-failure.json';data=json.loads(path.read_text());data[field]='forged';path.write_text(json.dumps(data))
    with pytest.raises(ExportError,match='does not match'):identity.failure_bundle(tmp_path,controller)


def test_run_owned_retains_failure_before_exception(fixture,tmp_path,monkeypatch):
    from contextlib import nullcontext
    from hyso.performance import stability
    config,_,_=fixture;folder=tmp_path/'worker';folder.mkdir()
    controller=make_failure(folder,config,monkeypatch)
    monkeypatch.setattr(native,'gpu_lock',lambda *a:nullcontext())
    monkeypatch.setattr(stability,'_run_owned',lambda *a,**k:folder)
    with pytest.raises(ExportError,match='NATIVE_SAFETY_STOP') as error:
        native.run_owned('dll_diagnostic',folder,{},config,failure_directory=tmp_path)
    assert error.value.result['controller']==controller
    identity.verify_retained(error.value.result['retained_failure'])


@pytest.fixture
def ticket(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'ROOT',tmp_path)
    plan={'runtime':{}};cp={'safety_hold':{'features':['E02_CUDA','E05_CUDA']}}
    r.engine.atomic_json(tmp_path/'plan.json',plan);r.engine.atomic_json(tmp_path/'checkpoint.json',cp)
    monkeypatch.setattr(r,'snapshot',lambda:{'source.py':'sha'})
    monkeypatch.setattr(r.engine,'user_snapshot',lambda:{})
    root=tmp_path/'diagnostics'/'one';root.mkdir(parents=True)
    data=dict(root=str(root.resolve()),operation='dll_diagnostic',automatic_retries=0,
        inference_allowed=False,conversion_allowed=False,source_sha256=r.snapshot(),user_files={},
        plan_sha256=file_hash(tmp_path/'plan.json'),checkpoint_sha256=file_hash(tmp_path/'checkpoint.json'))
    r.engine.atomic_json(root/'ticket.json',data)
    r.engine.atomic_json(root/'ticket-seal.json',{'sha256':file_hash(root/'ticket.json')})
    return root/'ticket.json'


@pytest.mark.parametrize('damage',['none','consumed','source','checkpoint','tamper'])
def test_one_shot_diagnostic_gate(ticket,monkeypatch,damage):
    if damage=='consumed':(ticket.parent/'consumed.json').write_text('{}')
    if damage=='source':monkeypatch.setattr(r,'snapshot',lambda:{'source.py':'changed'})
    if damage=='checkpoint':r.engine.atomic_json(r.ROOT/'checkpoint.json',{})
    if damage=='tamper':ticket.write_text('{}')
    if damage=='none':assert validate_ticket(ticket)[1]==ticket.parent
    else:
        with pytest.raises(RuntimeError):validate_ticket(ticket)


def test_source_review_fails_closed_without_exact_review(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'ROOT',tmp_path)
    assert not r.reviewed_dll_transition({'source_sha256':{}},{'source_sha256':{}},{})


@pytest.mark.parametrize('damage',['none','runtime','unreviewed','evidence'])
def test_exact_source_transition_preserves_only_reviewed_delta(tmp_path,monkeypatch,damage):
    monkeypatch.setattr(r,'ROOT',tmp_path)
    old=dict(source_sha256={'gpu.py':'old'},source_fingerprint='old',user_files={},runtime='same')
    new=dict(old,source_sha256={'gpu.py':'new'},source_fingerprint='new')
    proof=tmp_path/'proof.json';proof.write_text('sealed evidence')
    review=dict(old_plan_sha256=r.engine.digest(old),new_sources=new['source_sha256'],new_user_files={},
        reviewed_paths=['gpu.py'],frozen_evidence={str(proof):file_hash(proof)})
    if damage=='runtime':new['runtime']='changed'
    if damage=='unreviewed':review['reviewed_paths']=[]
    if damage=='evidence':proof.write_text('tampered')
    r.engine.atomic_json(tmp_path/'dll-diagnostic-source-review.json',review)
    checked=[];monkeypatch.setattr(r.user,'verify_cached',lambda cp:checked.append(cp))
    if damage=='evidence':
        with pytest.raises(RuntimeError,match='EVIDENCE_CHANGED'):r.reviewed_dll_transition(old,new,{})
    else:
        assert r.reviewed_dll_transition(old,new,{}) is (damage=='none')
        assert bool(checked) is (damage=='none')


@pytest.mark.parametrize('admitted',[True,False])
def test_diagnostic_loads_only_identity_boundary_no_session_or_tensor(tmp_path,monkeypatch,admitted):
    import sys
    from types import SimpleNamespace
    import torch
    site=tmp_path/'site';capi=site/'onnxruntime/capi';capi.mkdir(parents=True)
    provider=capi/'onnxruntime_providers_cuda.dll';provider.write_bytes(b'not an executable')
    key=shared.canonical_path(provider)
    config=NativeConfig('GPU-00000000-0000-0000-0000-000000000001',
        {key:file_hash(provider) if admitted else '0'*64},runtime_search_path=str(site))
    request=dict(operation='dll_diagnostic',native=config.__dict__,token=str(uuid.uuid4()),_binary_file_ids={})
    (tmp_path/'request.json').write_text(json.dumps(request))
    def forbidden(*a,**k):pytest.fail('diagnostic attempted inference/tensor allocation')
    monkeypatch.setattr(torch,'empty',forbidden)
    monkeypatch.setitem(sys.modules,'onnxruntime',SimpleNamespace(__file__=str(site/'onnxruntime/__init__.py'),__version__='fixture',InferenceSession=forbidden))
    monkeypatch.setattr(_worker,'_gpu',lambda *args:(torch,{}))
    monkeypatch.setattr(_worker,'verify_loaded',lambda *args:{key:{'sha256':file_hash(provider)}})
    loaded=[];monkeypatch.setattr(_worker.ctypes,'WinDLL',lambda p:loaded.append(p))
    if not admitted:
        with pytest.raises(ExportError,match='pre-admitted'):_worker.worker(tmp_path,request)
        assert not loaded and (tmp_path/'identity-failure.json').exists()
    else:
        _worker.worker(tmp_path,request)
        report=json.loads((tmp_path/'report.json').read_text())
        assert len(loaded)==1 and report['diagnostic_only']
        assert report['inference_executed'] is False and report['conversion_executed'] is False
