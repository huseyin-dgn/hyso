"""Offline checks of the real discovered cudart descriptors; no DLL is loaded."""
import hashlib
import json
from pathlib import Path
import pytest
from hyso.export._dll_identity import compare

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def cuda():
    plan=json.loads((ROOT/'outputs/export-final-acceptance-20261127/plan.json').read_text(encoding='utf-8'))
    choice=plan['dependency_readiness']['cuda_selection']
    selected=choice['selected'];alternate=next(r for r in choice['alternatives'] if Path(r['path']).name.lower()=='cudart64_12.dll')
    assert selected['distribution']=='torch' and alternate['distribution']=='nvidia-cuda-runtime-cu12'
    assert selected['sha256']!=alternate['sha256']
    allowed={selected['path']:selected['sha256']}
    ids={selected['path']:{k:selected[k] for k in ('file_id','bytes')}}
    return selected,alternate,allowed,ids

def descriptor(row):return {k:row[k] for k in ('sha256','file_id','bytes')}

def test_selected_cudart_descriptor_matches(cuda):
    selected,_,allowed,ids=cuda
    assert compare({selected['path']:descriptor(selected)},allowed,ids)['rejected'] is False

def test_same_named_alternate_cuda_is_rejected(cuda):
    selected,alternate,allowed,ids=cuda
    result=compare({alternate['path']:descriptor(alternate)},allowed,ids)
    assert result['rejected'] and result['rows'][0]['expected_path']==selected['path']
    assert {'manifest_membership','path','sha256','file_id','bytes'}<=set(result['rows'][0]['mismatched_fields'])

def test_selected_path_with_wrong_hash_is_rejected(cuda):
    selected,_,allowed,ids=cuda;row=descriptor(selected);row['sha256']=hashlib.sha256(b'offline changed bytes').hexdigest()
    result=compare({selected['path']:row},allowed,ids)
    assert result['rejected'] and 'sha256' in result['rows'][0]['mismatched_fields']

def test_selected_path_with_wrong_file_identity_is_rejected(cuda):
    selected,_,allowed,ids=cuda;row=descriptor(selected);row['file_id']+=1
    result=compare({selected['path']:row},allowed,ids)
    assert result['rejected'] and 'file_id' in result['rows'][0]['mismatched_fields']

def test_numpy_quarantine_is_outside_active_distribution():
    before=json.loads((ROOT/'outputs/tensorrt-blockers-20261127/numpy-before.json').read_text())
    after=json.loads((ROOT/'outputs/tensorrt-blockers-20261127/numpy-after.json').read_text())
    moved=json.loads((ROOT/'outputs/tensorrt-blockers-20261127/numpy-quarantine.json').read_text(encoding='utf-8-sig'))
    env=Path(before['site']).parents[1].resolve()
    assert after['numpy']=='2.2.2' and after['active_changes']==[] and after['cpu_sum']==15.0
    for entry in moved:
        source=Path(entry['source']);target=Path(entry['destination'])
        assert source.parent==Path(before['site']) and not source.exists()
        assert target.is_dir() and not target.resolve().is_relative_to(env)

@pytest.fixture
def cpu_profile(tmp_path,monkeypatch):
    import sys
    from tools.export_acceptance import final_profile as profile,runner as previous
    from tools.cuda_acceptance import validation
    engine=previous.engine
    for name in ('ROOT','FEATURES','GPU_UUID','prepare','snapshot','request_for','progress','record_cpu','verify_cached','pending_cases','cpu_group_ready','normal_cpu_completion'):
        monkeypatch.setattr(engine,name,getattr(engine,name))
    for name in ('verify_feature','validate_gpu_attempt'):monkeypatch.setattr(validation,name,getattr(validation,name))
    monkeypatch.setattr(profile,'CPU_ONLY',True);monkeypatch.setattr(profile,'ROOT',tmp_path/'cpu-campaign')
    monkeypatch.setattr(profile,'closure',lambda *a:pytest.fail('CPU scope must not inspect TensorRT'))
    monkeypatch.setattr(profile.subprocess,'run',lambda *a,**k:pytest.fail('CPU scope must not discover Conda or GPU'))
    monkeypatch.setattr(previous,'snapshot',lambda:{})
    monkeypatch.setattr(engine,'user_snapshot',lambda:{})
    monkeypatch.setattr(engine,'tree_bytes',lambda *a:0)
    monkeypatch.setattr(profile,'reuse_cpu_evidence',lambda:dict(pass_count=45,files={}))
    monkeypatch.setattr(profile,'reference_runtime',lambda *a:dict(status='READY_FOR_ADMISSION',issues=[],version='1.20.2'))
    cpu=dict(cpu_python=sys.executable,cpu_site=str(tmp_path),torch_cpu_distribution='2.5.1+cpu',runtime_paths=[str(tmp_path)],missing=[])
    monkeypatch.setattr(engine,'runtime_plan',lambda:cpu)
    return profile,profile.configure()

def test_cpu_scope_prepares_without_native_discovery(cpu_profile):
    profile,engine=cpu_profile;plan,cp=engine.prepare()
    assert plan['scope']=='cpu-only' and plan['dependency_readiness']['status']=='READY_FOR_CPU_ACCEPTANCE'
    assert not plan['dependency_readiness']['tensorrt_in_scope']
    assert plan['dependency_readiness']['tensorrt_status']=='BLOCKED_UNVERIFIED'
    assert not cp['attempts'] and plan['automatic_retries']==0

def test_cpu_scope_uses_existing_engine_without_gpu_groups(cpu_profile):
    profile,engine=cpu_profile
    from tools.cuda_acceptance import runner
    plan,_=engine.prepare()
    assert engine is runner and engine.FEATURES==plan['groups']=={}
    assert plan['scenario_counts']['GPU']==plan['scenario_counts']['admission']==0
    assert all(r['device']=='cpu' for r in plan['acceptance_matrix'])
    assert next(r for r in plan['acceptance_matrix'] if r['id']=='FP16_CPU_P35_COVERAGE')['status']=='BLOCKED'

def test_cpu_scope_does_not_reschedule_prior_pass_suites(cpu_profile):
    profile,engine=cpu_profile;plan,_=engine.prepare()
    assert plan['reused_cpu']['pass_count']==45
    assert plan['cpu_inventory'] and all('test_cpu_scope_' in node for node in plan['cpu_inventory'])
    assert not any('test_export_final_readiness.py' in node for node in plan['cpu_inventory'])

def test_cpu_scope_request_keeps_owned_cpu_worker(cpu_profile,tmp_path):
    profile,engine=cpu_profile;plan,_=engine.prepare()
    request=engine.request_for(tmp_path/'attempt',tmp_path/'scratch/work','CPU_REGRESSIONS',plan,plan['cpu_inventory'])
    assert request['profile']=='cpu' and not request.get('cuda_authorized',False)
    assert Path(request['argv'][0]).name=='cpu_worker.py' and request['cpu_ort_manifest']==plan['feature_tools']['ort']

def test_cpu_scope_rejects_native_campaign_root(cpu_profile):
    profile,engine=cpu_profile;plan,_=engine.prepare()
    plan.pop('scope');engine.atomic_json(profile.ROOT/'plan.json',plan)
    with pytest.raises(RuntimeError,match='CAMPAIGN_SCOPE_CHANGED'):engine.prepare()

def test_cpu_scope_start_bypasses_native_gate_only(cpu_profile,monkeypatch):
    import sys
    profile,engine=cpu_profile;plan,_=engine.prepare()
    monkeypatch.setattr(profile,'configure',lambda:engine)
    monkeypatch.setattr(engine,'main',lambda:0)
    monkeypatch.setattr(sys,'argv',list(sys.argv))
    assert profile.main(['--cpu-only','--campaign-root',str(profile.ROOT)])==0

def test_cpu_scope_status_requires_no_environment(cpu_profile,capsys):
    profile,engine=cpu_profile;engine.prepare()
    assert profile.main(['--cpu-only','--campaign-root',str(profile.ROOT),'--status'])==0
    assert 'PREPARED' in capsys.readouterr().out
