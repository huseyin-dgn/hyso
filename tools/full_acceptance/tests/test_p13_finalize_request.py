"""Offline authorization and binding faults; finalizer capture is always a double."""
from pathlib import Path
import pytest
from tools.full_acceptance import p13_finalize_request as request


@pytest.fixture
def capture(tmp_path,monkeypatch):
    out=tmp_path/'capture';out.mkdir()
    manifest=out/'manifest.json';request.core.publish(manifest,{'fixture':True})
    m=dict(output=str(out),run_id='offline',source_snapshot=dict(tree_sha256='tree'),
        experiments=[dict(id=stage,authorization=dict(approved=True))
                     for stage in ('02.regression','06.p13_capture')])
    monkeypatch.setattr(request.core,'verify',lambda *a,**k:m)
    return manifest,request.core.sha(manifest),m


def finish(capture):
    manifest,h,m=capture;out=manifest.parent
    request.core.publish(out/'STARTED.json',{})
    request.core.publish(out/'run-status.json',dict(controller_active=False,preservation_status='PASS',
        finalization_errors=[],manifest_sha256=h,run_id=m['run_id'],state='INCOMPLETE'))
    request.core.publish(out/'stage-results.json',dict(experiments=[
        dict(id='02.regression',status='PASS',attempted=True,pytest_exit=0,criteria=dict(safe_regressions=True)),
        dict(id='06.p13_capture',status='PASS',attempted=True)]))
    request.core.publish(out/'preservation-verification.json',dict(all_readable_preserved=True))
    request.core.publish(out/'stages/06.p13_capture/result.json',dict(fixture=True))


def test_waiting_request_cannot_be_authorized_or_executed(capture,tmp_path):
    manifest,h,_=capture;out=tmp_path/'waiting'
    r=request.prepare(manifest,h,out)
    assert r['state']=='WAITING_FOR_COMPLETED_CAPTURE' and not r['authorization']['approved']
    path=out/'manifest.json';sha=request.core.sha(path)
    with pytest.raises(ValueError,match='waiting request'):request.authorize(path,sha,tmp_path/'auth','approval')
    with pytest.raises(ValueError,match='waiting request'):request.execute(path,sha)
    assert not (out/'STARTED.json').exists()


@pytest.mark.parametrize('change',[dict(controller_active=True),dict(preservation_status='FAIL'),
    dict(finalization_errors=[{'phase':'reports'}]),dict(state='FAILED_CONTROLLER'),dict(run_id='other')])
def test_unclean_controller_never_becomes_ready(capture,tmp_path,change):
    finish(capture);manifest,h,_=capture
    p=manifest.parent/'run-status.json';s=request.core.read(p);s.update(change);request.core.publish(p,s)
    with pytest.raises(ValueError,match='clean terminal'):request.prepare(manifest,h,tmp_path/'request')
    assert not (tmp_path/'request').exists()


def test_ready_request_needs_separate_approval_and_is_one_shot(capture,tmp_path,monkeypatch):
    finish(capture);manifest,h,_=capture
    out=tmp_path/'request';request.prepare(manifest,h,out)
    path=out/'manifest.json';sha=request.core.sha(path)
    with pytest.raises(PermissionError):request.execute(path,sha)
    child=tmp_path/'approved';r=request.authorize(path,sha,child,'explicit offline finalizer only')
    assert r['authorization']['approved'] and r['parent_request']['sha256']==sha
    from tools.full_acceptance import p13_capture_finalize
    calls=[]
    def fake_build(path,expected,target):
        calls.append((path,expected));target.mkdir()
        request.core.publish(target/'p13-evidence-configuration.json',dict(fixture=True))
    monkeypatch.setattr(p13_capture_finalize,'build',fake_build)
    childpath=child/'manifest.json';childsha=request.core.sha(childpath)
    request.execute(childpath,childsha)
    assert len(calls)==1
    with pytest.raises(FileExistsError):request.execute(childpath,childsha)
    assert len(calls)==1


def test_changed_capture_artifact_blocks_authorization(capture,tmp_path):
    finish(capture);manifest,h,_=capture
    out=tmp_path/'request';request.prepare(manifest,h,out)
    request.core.publish(manifest.parent/'stages/06.p13_capture/result.json',dict(changed=True))
    path=out/'manifest.json'
    with pytest.raises(ValueError,match='artifact changed'):
        request.authorize(path,request.core.sha(path),tmp_path/'approved','explicit')


def test_no_finalizer_output_inside_consumed_capture(capture):
    manifest,h,_=capture
    with pytest.raises(ValueError,match='capture evidence'):
        request.prepare(manifest,h,manifest.parent/'new-request')


def test_finalizer_requires_successful_strict_cpu_regression(capture,tmp_path):
    finish(capture);manifest,h,_=capture
    path=manifest.parent/'stage-results.json'
    rows=request.core.read(path);rows['experiments'][0]['status']='FAIL'
    request.core.publish(path,rows)
    with pytest.raises(ValueError,match='regression prerequisite'):
        request.prepare(manifest,h,tmp_path/'request')
