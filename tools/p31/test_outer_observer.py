"""External exit/cleanup/failure semantics with bounded OS-boundary doubles."""
import subprocess
import sys
from unittest.mock import patch

import pytest
import start_approved as observer


class Job:
    def __init__(self,*args):pass
    def assign(self,p):pass
    def terminate(self):pass
    def observation(self):return dict(active_processes=0)
    def close(self):pass


@pytest.mark.parametrize('outcome',[0,17,'timeout','interrupt'])
def test_exit_receipt_and_original_exception(tmp_path,outcome):
    original=subprocess.TimeoutExpired('controlled',.1) if outcome=='timeout' else KeyboardInterrupt('controlled')
    class Process:
        pid=123;returncode=None;calls=0
        def wait(self,timeout):
            self.calls+=1
            if self.calls==1 and isinstance(outcome,str):raise original
            self.returncode=outcome if type(outcome) is int else 0
        def kill(self):pass
    with patch('hyso.performance._training_process._Job',Job),patch.object(observer.subprocess,'Popen',return_value=Process()):
        if type(outcome) is int:assert observer.observe(['controlled'],tmp_path,timeout=.1)==outcome
        else:
            with pytest.raises(type(original)) as caught:observer.observe(['controlled'],tmp_path,timeout=.1)
            assert caught.value is original
    receipt=observer.launcher.read(tmp_path/'parent-exit.json')
    assert receipt['cleanup']['active_processes']==0 and receipt['P31']=='PARTIAL_PENDING_REVIEW'
    assert receipt['reason']==('outer_watchdog_timeout' if outcome=='timeout' else
        'outer_interrupted_or_failed' if outcome=='interrupt' else None)


def test_no_explicit_start_cannot_preflight_or_launch():
    with patch.object(sys,'argv',['start_approved.py','--evidence-root','unused']),patch.object(observer.launcher,'preflight') as preflight:
        with pytest.raises(SystemExit) as caught:observer.main()
        assert caught.value.code==2;preflight.assert_not_called()


def test_cleanup_error_does_not_replace_original_timeout(tmp_path):
    original=subprocess.TimeoutExpired('controlled',.1);closed=[]
    class BrokenJob(Job):
        def terminate(self):raise OSError('controlled cleanup fault')
        def close(self):closed.append(True)
    class Process:
        pid=124;returncode=None
        def wait(self,timeout):raise original
    with patch('hyso.performance._training_process._Job',BrokenJob),patch.object(observer.subprocess,'Popen',return_value=Process()):
        with pytest.raises(subprocess.TimeoutExpired) as caught:observer.observe(['controlled'],tmp_path,timeout=.1)
    assert caught.value is original and closed
    receipt=observer.launcher.read(tmp_path/'parent-exit.json')
    assert receipt['cleanup'] is None and receipt['cleanup_errors']
    assert receipt['status']=='INTERRUPTED_OR_FAILED'


@pytest.mark.parametrize('primary', [True,False],ids=['with_primary','without_primary'])
def test_receipt_failure_never_becomes_success_or_masks_original(tmp_path,primary):
    original=KeyboardInterrupt('primary');secondary=OSError('publication failure')
    class Process:
        pid=125;returncode=0;calls=0
        def wait(self,timeout):
            self.calls+=1
            if primary and self.calls==1:raise original
    with patch('hyso.performance._training_process._Job',Job),patch.object(observer.subprocess,'Popen',return_value=Process()),patch.object(observer.launcher,'publish_new',side_effect=secondary):
        with pytest.raises(KeyboardInterrupt if primary else OSError) as caught:observer.observe(['controlled'],tmp_path)
    assert caught.value is (original if primary else secondary)
    assert not (tmp_path/'parent-exit.json').exists()
