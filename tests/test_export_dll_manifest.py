"""Offline compiler dependency provenance and explicit recovery regressions."""
import base64
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pytest
from hyso import _native_identity as identity
from hyso.export import NativeConfig,ExportError,native
from tools.export_acceptance import runner as r


@pytest.fixture
def installed(tmp_path):
    site=tmp_path/'site';lib=site/'torch/lib';lib.mkdir(parents=True)
    info=site/'torch-2.5.1.dist-info';info.mkdir()
    (info/'METADATA').write_text('Metadata-Version: 2.1\nName: torch\nVersion: 2.5.1+cu121\n')
    names=['torch_cuda.dll','nvrtc64_120_0.dll','nvrtc-builtins64_121.dll','nvJitLink_120_0.dll']
    rows=[]
    for name in names:
        p=lib/name;p.write_bytes(('offline fixture '+name).encode())
        sha=hashlib.sha256(p.read_bytes()).digest()
        rows.append(['torch/lib/'+name,'sha256='+base64.urlsafe_b64encode(sha).decode().rstrip('='),str(p.stat().st_size)])
    record=info/'RECORD'
    def write():
        with record.open('w',newline='') as stream:csv.writer(stream).writerows(rows)
    write()
    anchor={'path':str(lib/'torch_cuda.dll'),'sha256':identity.file_hash(lib/'torch_cuda.dll')}
    return site,lib,info,rows,write,anchor


def test_record_backed_inventory_matches_export_policy(installed):
    site,lib,_,_,_,anchor=installed
    result=identity.torch_compiler_inventory(site,'2.5.1+cu121',anchor)
    assert len(result['files'])==3
    for path,row in result['files'].items():
        assert identity.export_selected(Path(path).name)
        assert Path(path).parent==lib
        assert row['sha256']==identity.file_hash(path)
        assert result['binary_identities'][path]=={'file_id':Path(path).stat().st_ino,'bytes':Path(path).stat().st_size}
        assert row['provenance']['distribution']=='torch'


@pytest.mark.parametrize('damage',['hash','size','missing_record','extra_dll','outside','version','anchor','reparse','hash_mode'])
def test_unverified_dependency_never_enters_manifest(installed,monkeypatch,damage):
    site,lib,info,rows,write,anchor=installed;version='2.5.1+cu121'
    dll=lib/'nvrtc64_120_0.dll'
    if damage=='hash':dll.write_bytes(b'modified')
    if damage=='size':rows[1][2]='1';write()
    if damage=='missing_record':rows.pop(1);write()
    if damage=='extra_dll':(lib/'nvrtc_unrecorded.dll').write_bytes(b'untrusted')
    if damage=='outside':rows[1][0]='other/nvrtc64_120_0.dll';write()
    if damage=='version':version='unverified-version'
    if damage=='anchor':anchor=dict(anchor,sha256='0'*64)
    if damage=='reparse':
        original=Path.is_symlink
        monkeypatch.setattr(Path,'is_symlink',lambda p:p==dll or original(p))
    if damage=='hash_mode':rows[1][1]='md5=untrusted';write()
    with pytest.raises((ValueError,OSError)):
        identity.torch_compiler_inventory(site,version,anchor)


def test_versioned_filenames_not_machine_specific(installed):
    site,lib,info,rows,write,anchor=installed
    for row in rows[1:]:
        source=site/row[0];target=source.with_name(source.name.replace('120','130').replace('121','130'))
        source.rename(target);row[0]=target.relative_to(site).as_posix()
    write()
    assert len(identity.torch_compiler_inventory(site,'2.5.1+cu121',anchor)['files'])==3


def test_prepared_file_id_change_rejected_before_launch(installed,tmp_path,monkeypatch):
    from hyso.performance import stability
    site,_,_,_,_,anchor=installed;inventory=identity.torch_compiler_inventory(site,'2.5.1+cu121',anchor)
    ids={p:dict(row,file_id=row['file_id']+1) for p,row in inventory['binary_identities'].items()}
    config=NativeConfig('GPU-00000000-0000-0000-0000-000000000001',inventory['allowed_binaries'],dependency_identities=ids)
    monkeypatch.setattr(stability,'_run_owned',lambda *a,**kw:pytest.fail('changed file identity launched native code'))
    with pytest.raises(ExportError,match='file identity differs'):native.run_owned('ort',tmp_path/'worker',{},config)


def test_dll_admission_failure_does_not_claim_conversion_failure(installed,tmp_path,monkeypatch):
    from contextlib import nullcontext
    from hyso.performance import stability
    from test_export_dll_diagnostic import make_failure
    site,_,_,_,_,anchor=installed;inventory=identity.torch_compiler_inventory(site,'2.5.1+cu121',anchor)
    config=NativeConfig('GPU-00000000-0000-0000-0000-000000000001',inventory['allowed_binaries'])
    folder=tmp_path/'worker';folder.mkdir();make_failure(folder,config,monkeypatch)
    monkeypatch.setattr(native,'gpu_lock',lambda *a:nullcontext())
    monkeypatch.setattr(stability,'_run_owned',lambda *a,**kw:folder)
    with pytest.raises(ExportError) as error:native.run_owned('dll_diagnostic',folder,{},config,failure_directory=tmp_path)
    assert error.value.result['failure_stage']=='admission'
    assert error.value.result['stages']['admission']['status']=='FAIL'
    assert error.value.result['stages']['conversion']['status']=='NOT_RUN'


def test_recovery_runs_only_explicit_cases_and_never_retries_hold():
    results={'E02_CUDA':{'status':'BLOCKED'},'E05_CUDA':{'status':'BLOCKED'}}
    assert r.pending_cases(['E02_CUDA','E05_CUDA'],results,set())==[]
    assert r.pending_cases(['E02_CUDA','E05_CUDA'],results,{'E02_CUDA','E05_CUDA'})==['E02_CUDA','E05_CUDA']
    assert r.pending_cases(['E03_TRT','E04_TRT'],{}, {'E02_CUDA','E05_CUDA'})==[]


def test_source_review_rejects_unreviewed_manifest_delta(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'ROOT',tmp_path)
    old=dict(source_sha256={},source_fingerprint='x',user_files={},feature_tools={'ort':'unchanged'})
    new=dict(old,feature_tools=dict(old['feature_tools'],export_cuda={'files':'reviewed'}))
    proof=tmp_path/'evidence';proof.write_text('sealed')
    review=dict(old_plan_sha256=r.engine.digest(old),new_sources={},new_user_files={},reviewed_paths=[],
        new_export_cuda={'files':'reviewed'},frozen_evidence={str(proof):identity.file_hash(proof)})
    r.engine.atomic_json(tmp_path/'dll-manifest-source-review.json',review)
    monkeypatch.setattr(r.user,'verify_cached',lambda cp:None)
    assert r.reviewed_dll_transition(old,new,{})
    new['feature_tools']['export_cuda']={'files':'unreviewed'}
    assert not r.reviewed_dll_transition(old,new,{})
