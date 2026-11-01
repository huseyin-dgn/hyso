"""Execute only stdlib simulated trees in a freshly sealed Windows candidate.

Run with the candidate's interpreter; no pytest/conftest or CUDA imports.
"""
import argparse
import ctypes as c
from ctypes import wintypes as w
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools/forensics'))
import native_scenario_runner as runner


def exited(pid):
    api = c.WinDLL('kernel32', use_last_error=True)
    api.OpenProcess.argtypes = [w.DWORD,w.BOOL,w.DWORD]
    api.OpenProcess.restype = w.HANDLE
    api.GetExitCodeProcess.argtypes = [w.HANDLE,c.POINTER(w.DWORD)]
    api.GetExitCodeProcess.restype = w.BOOL
    api.CloseHandle.argtypes = [w.HANDLE]
    handle = api.OpenProcess(0x1000, False, pid)
    if not handle:
        if c.get_last_error() == 87:  # PID no longer exists
            return True
        raise OSError(c.get_last_error(), 'owned PID absence verification')
    try:
        code = w.DWORD()
        if not api.GetExitCodeProcess(handle, c.byref(code)):
            raise OSError(c.get_last_error(), 'owned process exit query')
        return code.value != 259
    finally:
        api.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args()
    root = args.candidate.resolve()
    seal = runner.read(root/'seal.json')
    manifest = runner.verify(root, seal['manifest_sha256'])
    outcomes = []
    for mode in ('success', 'timeout', 'outer-timeout', 'log-overflow'):
        work = root/('dry-'+mode)
        process = subprocess.run([sys.executable, '-I', '-S', '-B',
            str(root/'source/native_scenario_runner.py'), '--root',str(root),
            '--work',str(work), '--mode',mode], capture_output=True, timeout=110,
            creationflags=subprocess.CREATE_NO_WINDOW)
        if len(process.stdout)>65536 or len(process.stderr)>65536:
            raise RuntimeError('dry controller log bound violated')
        runner.publish(root/('dry-'+mode+'-controller.json'), dict(exit_code=process.returncode,
            stdout=process.stdout.decode(errors='replace'), stderr=process.stderr.decode(errors='replace')))
        assert process.returncode == 0, process.stderr.decode(errors='replace')
        outer = runner.read(work/'parent-receipt.json')
        assert outer['cleanup']['active_processes'] == 0 and outer['owned_leader_exited']
        assert outer['assignment']['limits'] == runner.expected_limits('outer')
        assert outer['assignment']['assigned'] is True and outer['kill_on_close']
        assert outer['deadline_seconds'] == (15 if mode=='outer-timeout' else 90)
        all_pids = set(outer['owned_pids'])
        if mode == 'outer-timeout':
            assert outer['outcome']=='timeout'
        else:
            child = runner.read(work/'worker-receipt.json')
            assert child['cleanup']['active_processes']==0 and child['owned_leader_exited']
            assert child['assignment']['limits']==runner.expected_limits('child')
            assert child['assignment']['assigned'] is True and child['kill_on_close']
            assert child['deadline_seconds']==(15 if mode=='timeout' else 60)
            all_pids.update(child['owned_pids'])
            expected = {'success':None, 'timeout':'timeout', 'log-overflow':'bounded_log_exceeded'}[mode]
            assert child['outcome']==expected
            assert child['exit_code']==0 if mode=='success' else True
            if mode=='log-overflow':
                assert child['log_bytes']['stdout']==70000
                assert len(child['logs']['stdout'].encode())==65536
        for role in ('parent','worker'):
            identity=runner.read(work/(role+'-identity.json'))
            assert identity['job_verified'] and identity['cuda_imported'] is False
        tree=runner.read(work/'simulated-tree.json')
        assert tree['cuda_imported'] is False
        assert tree['worker_pid'] in all_pids and tree['descendant_pid'] in all_pids
        assert all(exited(pid) for pid in all_pids)
        assert sum(p.stat().st_size for p in work.glob('*.json'))<2*1024**2
        outcomes.append(dict(mode=mode, status='PASS', owned_pids=sorted(all_pids),
            all_owned_pids_exited=True, active_processes=0,
            outer_outcome=outer['outcome'], cuda_workload_started=False))
        print(json.dumps(outcomes[-1]), flush=True)
    # Authorization refusal is evaluated before any native process/attempt marker.
    try:
        runner.controller(root, root/'not-created', 'native')
    except ValueError as error:
        assert 'authorization required' in str(error)
    else:
        raise AssertionError('missing native approval accepted')
    assert not (root/'execution-consumed.json').exists()
    assert not (root/'not-created').exists()
    runner.verify(root, seal['manifest_sha256'])
    assert 'torch' not in sys.modules
    runner.publish(root/'dryrun-summary.json', dict(status='PASS', scenarios=outcomes,
        native_approval_absent_rejected=True, source_runtime_posthash_verified=True,
        cuda_workload_started=False, native_executions=0))


if __name__ == '__main__':
    main()
