"""E13 closure regressions: metadata/files/dispatch only, no GPU imports or build."""
import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from tools.export_acceptance import final_profile as profile


def installation(tmp_path):
    python=tmp_path/'python.exe';python.write_bytes(b'fake interpreter; never executed')
    site=tmp_path/'Lib/site-packages';site.mkdir(parents=True)
    packages={'tensorrt-cu12':('10.13.3.9',None),'tensorrt-cu12-libs':('10.13.3.9','tensorrt_libs/nvinfer_10.dll'),
      'tensorrt-cu12-bindings':('10.13.3.9','tensorrt_bindings/tensorrt.cp310-win_amd64.pyd'),
      'nvidia-cuda-runtime-cu12':('12.9.79','nvidia/cuda_runtime/bin/cudart64_12.dll'),
      'torch':('2.5.1+cu121','torch/lib/cudart64_12.dll'),'numpy':('2.2.2',None),'onnx':('1.20.0',None)}
    for name,(version,member) in packages.items():
        info=site/(name.replace('-','_')+'-'+version+'.dist-info');info.mkdir()
        (info/'METADATA').write_text('Metadata-Version: 2.1\nName: '+name+'\nVersion: '+version+'\n')
        if member:
            p=site/member;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(name.encode())
            encoded=base64.urlsafe_b64encode(hashlib.sha256(p.read_bytes()).digest()).decode().rstrip('=')
            with (info/'RECORD').open('w',newline='') as stream:csv.writer(stream).writerow([member,'sha256='+encoded,p.stat().st_size])
    return python,site


def test_closure_collects_all_record_mismatches(tmp_path):
    python,site=installation(tmp_path)
    dll=site/'tensorrt_libs/nvinfer_10.dll';dll.write_bytes(dll.read_bytes()+b'changed')
    pyd=site/'tensorrt_bindings/tensorrt.cp310-win_amd64.pyd';pyd.write_bytes(b'X'*pyd.stat().st_size)
    result=profile.closure(python)
    assert result['status']=='BLOCKED' and result['allowed_binaries']=={}
    assert any('RECORD_SIZE' in s for s in result['issues']) and sum('RECORD_SHA256' in s for s in result['issues'])==2
    assert result['native_imports'] is False


def test_closure_verifies_canonical_file_identity_and_provenance(tmp_path):
    python,site=installation(tmp_path);result=profile.closure(python)
    assert result['status']=='READY_FOR_ADMISSION' and result['runtime_verified'] is False
    assert len(result['files'])==4
    for row in result['files']:
        p=Path(row['path']);assert p.is_absolute() and row['file_id']==p.stat().st_ino and row['bytes']==p.stat().st_size
        assert row['sha256']==profile.sha(p) and row['problems']==[]


def test_alternate_cudart_never_enters_allowlist(tmp_path):
    python,_=installation(tmp_path);result=profile.closure(python)
    selected=result['cuda_selection']['selected'];alternatives=result['cuda_selection']['alternatives']
    assert selected['distribution']=='torch' and selected['path'] in result['allowed_binaries']
    assert alternatives and all(r['path'] not in result['allowed_binaries'] for r in alternatives)
    assert result['warnings'] and result['runtime_verified'] is False


def test_numpy_metadata_ambiguity_blocks_but_is_not_deleted(tmp_path):
    python,site=installation(tmp_path);remnant=site/'~umpy-2.2.6.dist-info';remnant.mkdir()
    result=profile.closure(python)
    assert result['status']=='BLOCKED' and 'NUMPY_METADATA_AMBIGUOUS' in result['issues'] and remnant.is_dir()


def test_missing_meta_and_ort_namespace_conflict(tmp_path):
    python,site=installation(tmp_path)
    (site/'tensorrt_cu12-10.13.3.9.dist-info/METADATA').write_text('Metadata-Version: 2.1\nName: other-package\nVersion: 1\n')
    for name in ('onnxruntime','onnxruntime-gpu'):
        d=site/(name+'-1.dist-info');d.mkdir();(d/'METADATA').write_text('Metadata-Version: 2.1\nName: '+name+'\nVersion: 1\n')
    result=profile.closure(python)
    assert 'ORT_CPU_GPU_NAMESPACE_CONFLICT' in result['issues']
    assert any('tensorrt-cu12' in s and 'MISSING' in s for s in result['issues'])


def test_record_cannot_escape_site(tmp_path):
    python,site=installation(tmp_path);outside=tmp_path/'external.dll';outside.write_bytes(b'outside')
    record=site/'tensorrt_cu12_libs-10.13.3.9.dist-info/RECORD';record.write_text('../../external.dll,,\n')
    result=profile.closure(python)
    assert result['status']=='BLOCKED' and any('escapes' in s for s in result['issues'])


def test_matrix_preserves_coverage_block_without_numeric_failure():
    reused={'pass_count':45,'files':{'prior.xml':'digest'}}
    rows=profile.matrix(reused,{'status':'BLOCKED'})
    assert {r['id'] for r in rows if r['device']=='cpu'}>={'E01_CPU','E02_CPU','E05_CPU','E06_CPU','E07_CPU','E08_CPU','E09_CPU','E10_CPU','E11_CPU','E12_CPU'}
    coverage=next(r for r in rows if r['id']=='FP16_CPU_P35_COVERAGE')
    assert coverage['status']=='BLOCKED' and not coverage['product_numeric_defect']
    assert all(r['status']=='BLOCKED' for r in rows if r['device']=='cuda')


def test_prior_45_cpu_evidence_is_source_bound_and_not_rerun():
    result=profile.reuse_cpu_evidence()
    assert len(result['cases'])==45 and result['pass_count']==45 and result['changed_sources']==[]
    assert all(not v['owned_acceptance'] for v in result['cases'].values())


def test_final_profile_excludes_completed_cuda_cases():
    names={n for group in profile.GROUPS.values() for n in group}
    assert {'E03_TRT','E04_TRT','E03_TRT_DYNAMIC','E04_TRT_FP16','E08_TRT_QDQ'}<=names
    assert not names & {'E02_CUDA','E05_CUDA'}
    assert len(names)==10


def test_worker_refuses_unowned_entry_before_execution(tmp_path):
    path=tmp_path/'request.json';path.write_text(json.dumps({'token':'unowned'}))
    with pytest.raises((RuntimeError,FileNotFoundError)):profile.owned_worker(path)


def test_final_profile_import_is_native_free():
    root=Path(__file__).resolve().parents[1]
    script="import sys,runpy; runpy.run_path(sys.argv[1],run_name='readiness_import_only'); assert not {'torch','tensorrt','numpy','onnxruntime'} & set(sys.modules)"
    result=subprocess.run([sys.executable,'-I','-S','-B','-c',script,str(root/'tools/export_acceptance/final_profile.py')],capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr


def test_powershell_final_commands_parse():
    root=Path(__file__).resolve().parents[1]
    command="$t=$null; $e=$null; [System.Management.Automation.Language.Parser]::ParseFile($env:HYSO_EXPORT_PARSE_FILE,[ref]$t,[ref]$e) | Out-Null; if($e.Count){$e | Out-String | Write-Output; exit 1}"
    # Parse only. Running the user acceptance command is explicitly forbidden.
    result=subprocess.run(['powershell','-NoProfile','-Command',command],env=dict(os.environ,HYSO_EXPORT_PARSE_FILE=str(root/'run_final_acceptance.ps1')),capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stdout+result.stderr


def test_status_never_discovers_or_imports_runtime(tmp_path,monkeypatch,capsys):
    state=tmp_path/'state';state.mkdir();(state/'progress.json').write_text(json.dumps({'status':'PREPARED'}))
    monkeypatch.setattr(profile,'ROOT',state)
    monkeypatch.setattr(profile.subprocess,'run',lambda *a,**k:pytest.fail('Status must not launch discovery'))
    assert profile.main(['--campaign-root',str(state),'--status'])==0
    assert 'PREPARED' in capsys.readouterr().out


def test_blocked_start_cannot_reach_acceptance_engine(tmp_path,monkeypatch):
    from types import SimpleNamespace
    python,site=installation(tmp_path);dll=site/'tensorrt_libs/nvinfer_10.dll';dll.write_bytes(b'broken')
    state=tmp_path/'campaign';state.mkdir();(state/'plan.json').write_text(json.dumps({'runtime':{'python':str(python)}}))
    (state/'progress.json').write_text('{}')
    def atomic(path,value):Path(path).write_text(json.dumps(value))
    engine=SimpleNamespace(main=lambda:pytest.fail('native engine reached before readiness'),atomic_json=atomic,read_json=lambda p,default:profile.read(p) if Path(p).is_file() else default)
    monkeypatch.setattr(profile,'ROOT',state);monkeypatch.setattr(profile,'NATIVE_PYTHON',python)
    monkeypatch.setattr(profile,'configure',lambda:engine)
    assert profile.main(['--runtime-python',str(python),'--campaign-root',str(state),'--resume'])==1
    assert not (state/'attempts').exists()
    progress=profile.read(state/'progress.json')
    assert progress['status']=='BLOCKED' and progress['GPU_executed'] is False


def test_unverified_loader_is_rejected_before_builder(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from uuid import UUID
    from hyso import export as ex
    from hyso.export import _worker
    from tools.cuda_acceptance.feature_evidence import SafetyViolation
    from tools.export_acceptance import scenarios
    file=tmp_path/'declared.dll';file.write_bytes(b'offline fixture')
    tools={'ort':{'site':str(tmp_path),'allowed_binaries':{str(file):profile.sha(file)}},
           'tensorrt':{'allowed_binaries':{}},'export_cuda':{'allowed_binaries':{},'binary_identities':{}}}
    ctx=SimpleNamespace(uuid='GPU-'+str(UUID(int=1)),plan={'feature_tools':tools,'runtime':{'binary_expectations':{}}})
    monkeypatch.setitem(sys.modules,'tensorrt',SimpleNamespace(__version__='10.13.3.9'))
    monkeypatch.setattr(scenarios,'gpu_group',lambda ctx:pytest.fail('Builder dispatch reached'))
    monkeypatch.setattr(_worker,'verify_loaded',lambda *args:(_ for _ in ()).throw(ex.ExportError('DLL_INTEGRITY','alternate cudart loaded')))
    with pytest.raises(SafetyViolation,match='PRE_ENGINE_DLL_SELECTION'):profile.gpu_group(ctx)


def test_dynamic_acceptance_requires_distinct_min_opt_max(monkeypatch):
    from tools.export_acceptance import evidence
    monkeypatch.setattr(evidence,'verify_feature',lambda *args:None)
    def data(shapes):return {'development_extension':{'validation':{'numerical':{'status':'PASS'},'batches':[
        {'numerical':{'outputs':[{'reference_shape':[n,2]}]}} for n in shapes]}}}
    with pytest.raises(ValueError,match='min/opt/max'):profile.verify_feature('E03_TRT_DYNAMIC',data([2,2,2]),{})
    profile.verify_feature('E03_TRT_DYNAMIC',data([1,2,4]),{})


def test_e04_fp16_cannot_accept_fp32_chain(monkeypatch):
    from tools.export_acceptance import evidence
    monkeypatch.setattr(evidence,'verify_feature',lambda *args:None)
    with pytest.raises(ValueError,match='FP16'):profile.verify_feature('E04_TRT_FP16',{'precision':{'requested':'fp32'}},{})
    profile.verify_feature('E04_TRT_FP16',{'precision':{'requested':'fp16'},'conversion':'pytorch_to_onnx_to_tensorrt'},{})


def test_changed_cpu_proof_is_not_promoted(monkeypatch):
    original=profile.sha
    monkeypatch.setattr(profile,'sha',lambda p:'changed' if Path(p).as_posix().endswith('hyso/export/runtime.py') else original(p))
    result=profile.reuse_cpu_evidence()
    assert result['pass_count']==0 and result['changed_sources']
    assert all(v['status']=='NOT_TESTED' for v in result['cases'].values())


def test_runtime_sites_keep_selected_torch_and_explicit_ort(tmp_path):
    from tools.cuda_acceptance.runner import runtime_sites
    selected=tmp_path/'selected';reference=tmp_path/'reference';selected.mkdir();reference.mkdir()
    assert runtime_sites({'site':str(selected),'runtime_paths':[str(selected),str(reference)]})==[str(selected),str(reference)]
    with pytest.raises(RuntimeError,match='FIRST'):runtime_sites({'site':str(selected),'runtime_paths':[str(reference),str(selected)]})
    with pytest.raises(RuntimeError,match='UNIQUE'):runtime_sites({'site':str(selected),'runtime_paths':[str(selected),str(selected)]})
    with pytest.raises(RuntimeError,match='INVALID'):runtime_sites({'site':str(selected),'runtime_paths':[str(selected),str(tmp_path/'absent')]})


def test_request_forwards_selected_interpreter_and_reference_sites(tmp_path,monkeypatch):
    from tools.export_acceptance import runner as previous
    from tools.cuda_acceptance import validation
    engine=previous.engine
    for name in ('ROOT','FEATURES','GPU_UUID','prepare','snapshot','request_for','progress','record_cpu','verify_cached','pending_cases','cpu_group_ready','normal_cpu_completion'):
        monkeypatch.setattr(engine,name,getattr(engine,name))
    for name in ('verify_feature','validate_gpu_attempt'):monkeypatch.setattr(validation,name,getattr(validation,name))
    monkeypatch.setattr(engine,'tree_bytes',lambda path:0)
    monkeypatch.setattr(profile,'ROOT',tmp_path/'profile')
    configured=profile.configure()
    plan=profile.read(profile.HISTORY/'plan.json')
    selected=tmp_path/'selected-site';reference=tmp_path/'reference-site';selected.mkdir();reference.mkdir()
    plan['runtime']=dict(plan['runtime'],python=str(tmp_path/'selected-python.exe'),site=str(selected),
                         ort_site=str(reference),runtime_paths=[str(selected),str(reference)])
    plan['dependency_readiness']=dict(python=plan['runtime']['python'],site=str(selected))
    request=configured.request_for(tmp_path/'attempt',tmp_path/'work','03-native-engine',plan)
    assert request['gpu_plan']['runtime']['python']==plan['dependency_readiness']['python']
    assert engine.runtime_sites(request['gpu_plan']['runtime'])[0]==plan['dependency_readiness']['site']
    assert plan['runtime']['ort_site'] in request['gpu_plan']['runtime']['runtime_paths']
    assert request['argv'][1]=='--worker' and request['campaign_id']=='export-acceptance'


def test_outer_worker_uses_controller_receipt_not_nested_environment(tmp_path,monkeypatch):
    import runpy
    from tools.cuda_acceptance import worker
    from tools.export_acceptance import scenarios,evidence
    token='offline-owned-envelope';identity={'pid':os.getpid(),'image':'offline fixture'}
    request=tmp_path/'request.json';request.write_text(json.dumps({'token':token}))
    receipt={'identities':{'exact_job_membership':True,'interpreter':identity,'nonce_sha256':worker.digest(token)},
             'no_breakaway':True,'kill_on_close':True,'limits_readback':True}
    (tmp_path/'admission-receipt.json').write_text(json.dumps(receipt))
    (tmp_path/'admit.json').write_text(json.dumps({'pid':os.getpid(),'token':token,'readback_verified':True}))
    monkeypatch.delenv('HYSO_RUNTIME_ADMITTED_PID',raising=False)
    monkeypatch.setattr(worker,'process_identity',lambda pid:identity)
    monkeypatch.setattr(scenarios,'gpu_group',scenarios.gpu_group);monkeypatch.setattr(evidence,'verify_feature',evidence.verify_feature)
    monkeypatch.setattr(sys,'argv',list(sys.argv))
    calls=[];monkeypatch.setattr(runpy,'run_path',lambda path,**kwargs:calls.append(path))
    profile.owned_worker(request)
    assert len(calls)==1 and Path(calls[0]).name=='gpu_worker.py'
    receipt['identities']['nonce_sha256']='wrong';(tmp_path/'admission-receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError,match='OWNED_WORKER_ADMISSION'):profile.owned_worker(request)
    assert len(calls)==1


def test_cpu_ort_reference_metadata_is_supported_without_gpu_package(tmp_path):
    module=tmp_path/'onnxruntime';module.mkdir();(module/'__init__.py').write_text('# fixture, never imported')
    (module/'capi').mkdir();(module/'capi/onnxruntime_pybind11_state.pyd').write_bytes(b'offline extension fixture')
    info=tmp_path/'onnxruntime-1.23.2.dist-info';info.mkdir()
    (info/'METADATA').write_text('Metadata-Version: 2.1\nName: onnxruntime\nVersion: 1.23.2\n')
    result=profile.reference_runtime(tmp_path)
    assert result['status']=='READY_FOR_ADMISSION' and result['version']=='1.23.2' and not result['runtime_verified']
    (info/'METADATA').write_text('Metadata-Version: 2.1\nName: onnxruntime\nVersion: 1.20.2\n')
    assert profile.reference_runtime(tmp_path)['status']=='READY_FOR_ADMISSION'
    (info/'METADATA').write_text('Metadata-Version: 2.1\nName: onnxruntime\nVersion: 99.0\n')
    assert 'UNSUPPORTED_ORT_REFERENCE_VERSION' in profile.reference_runtime(tmp_path)['issues']
    (info/'METADATA').write_text('Metadata-Version: 2.1\nName: onnxruntime\nVersion: 1.23.2\n')
    other=tmp_path/'onnxruntime_gpu-1.23.2.dist-info';other.mkdir()
    (other/'METADATA').write_text('Metadata-Version: 2.1\nName: onnxruntime-gpu\nVersion: 1.23.2\n')
    assert profile.reference_runtime(tmp_path)['status']=='BLOCKED'


def test_missing_ort_reference_is_blocked_before_build(tmp_path):
    result=profile.reference_runtime(tmp_path)
    assert result['status']=='BLOCKED' and 'ORT_REFERENCE_MODULE_MISSING' in result['issues']


def test_reference_and_cpu_changes_stop_before_execution(tmp_path,monkeypatch):
    from types import SimpleNamespace
    python,site=installation(tmp_path)
    module=site/'onnxruntime';module.mkdir();(module/'__init__.py').write_text('# never imported')
    (module/'capi').mkdir();(module/'capi/onnxruntime_pybind11_state.pyd').write_bytes(b'fixture')
    info=site/'onnxruntime-1.23.2.dist-info';info.mkdir()
    (info/'METADATA').write_text('Metadata-Version: 2.1\nName: onnxruntime\nVersion: 1.23.2\n')
    cpu_python=tmp_path/'cpu.exe';cpu_python.write_bytes(b'CPU fixture')
    cpu=dict(cpu_python=str(cpu_python),cpu_site=str(tmp_path),torch_cpu_distribution='2.5.1+cpu',runtime_paths=[str(tmp_path)],missing=[])
    identity=dict(python=str(cpu_python),python_sha256=profile.sha(cpu_python),site=str(tmp_path),torch='2.5.1+cpu',runtime_paths=[str(tmp_path)],missing=[])
    state=tmp_path/'state';state.mkdir()
    plan=dict(runtime={'python':str(python),'ort_site':str(site)},dependency_readiness=profile.closure(python),
              reference_runtime=profile.reference_runtime(site),cpu_runtime=identity)
    (state/'plan.json').write_text(json.dumps(plan))
    engine=SimpleNamespace(main=lambda:pytest.fail('execution reached after changed reference/CPU'),runtime_plan=lambda:cpu)
    monkeypatch.setattr(profile,'ROOT',state);monkeypatch.setattr(profile,'NATIVE_PYTHON',python)
    monkeypatch.setattr(profile,'configure',lambda:engine)
    args=['--runtime-python',str(python),'--campaign-root',str(state),'--resume']
    cpu['torch_cpu_distribution']='changed-cpu-runtime'
    with pytest.raises(RuntimeError,match='CPU_RUNTIME_SELECTION_CHANGED'):profile.main(args)
    cpu['torch_cpu_distribution']='2.5.1+cpu';(module/'__init__.py').write_text('# changed reference')
    with pytest.raises(RuntimeError,match='ORT_REFERENCE_CHANGED'):profile.main(args)
