"""One sealed Conv2d candidate and stdlib-only Windows isolation dry runs.

Native execution requires a separate explicit approval bound to a new seal.
Never reuse the historical r3 execution directory. This is a trusted harness,
not a hostile-code sandbox. All termination uses newly created owned Jobs.
"""
import argparse
import ctypes as c
from ctypes import wintypes as w
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid

# Isolated interpreters deliberately exclude the checkout and site-packages.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_admission as admission

LIMITS = admission.policy.LIMITS
SCENARIO = 'conv2d_fp32_p01_p02_p09_p16'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path, limit=2*1024**2):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > limit:
        raise ValueError('linked/oversized artifact')
    with path.open('rb') as stream:
        raw = stream.read(limit+1)
    if len(raw) > limit:
        raise ValueError('artifact grew beyond bound')
    return json.loads(raw.decode('utf-8-sig'), object_pairs_hook=admission.policy._unique)


def publish(path, data):
    raw = json.dumps(data, indent=2, allow_nan=False).encode()
    if len(raw) > LIMITS['result_bytes']:
        raise ValueError('artifact budget exceeded')
    with Path(path).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def prepare(destination, r3):
    """Copy current sources, reuse and verify the r3 runtime identity inventory."""
    root = Path(__file__).resolve().parents[2]
    destination, r3 = Path(destination), Path(r3)
    if digest(r3/'manifest.json') != '660a53dbdde6a21901bf6ad2743a3fad9bb51bad57ebd25a5a5dee46ad2444b8':
        raise ValueError('r3 manifest mismatch')
    previous = read(r3/'manifest.json')
    for relative, sha in previous['source_files'].items():
        if digest(r3/'source'/relative) != sha:
            raise ValueError('r3 source mismatch: '+relative)
    for path, sha in previous['runtime_files'].items():
        if digest(path) != sha:
            raise ValueError('r3 runtime mismatch: '+path)
    destination.mkdir()  # exclusive; never replace existing evidence
    source = destination/'source'
    source.mkdir()
    files = list((root/'hyso').rglob('*.py')) + [
        Path(__file__), Path(__file__).with_name('native_admission.py'),
        Path(__file__).with_name('p23_preflight.py')]
    hashes = {}
    for path in files:
        relative = path.relative_to(root) if path.is_relative_to(root/'hyso') else Path(path.name)
        target = source/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[relative.as_posix()] = digest(target)
        target.chmod(0o444)
    runtime = dict(previous['runtime_files'])
    interpreter = read(r3/'seal.json')['interpreter']
    manifest = dict(run_id=str(uuid.uuid4()), scenario=SCENARIO, source_files=hashes,
                    runtime_files=runtime, interpreter=interpreter,
                    site_packages=str(Path(interpreter).parent/'Lib/site-packages'),
                    limits=LIMITS, executions=1, retries=0, baselines=0)
    publish(destination/'manifest.json', manifest)
    publish(destination/'seal.json', dict(run_id=manifest['run_id'],
                                         manifest_sha256=digest(destination/'manifest.json')))
    for name in ('manifest.json', 'seal.json'):
        (destination/name).chmod(0o444)
    return manifest


def verify(root, expected):
    root = Path(root)
    if digest(root/'manifest.json') != expected:
        raise ValueError('manifest identity mismatch')
    manifest = read(root/'manifest.json')
    if manifest['limits'] != LIMITS or manifest['scenario'] != SCENARIO:
        raise ValueError('scenario/limit mismatch')
    if Path(sys.executable).resolve() != Path(manifest['interpreter']).resolve():
        raise ValueError('interpreter identity mismatch')
    for relative, sha in manifest['source_files'].items():
        path = root/'source'/relative
        if path.is_symlink() or not path.resolve().is_relative_to((root/'source').resolve()) or digest(path) != sha:
            raise ValueError('source identity mismatch: '+relative)
    for path, sha in manifest['runtime_files'].items():
        if digest(path) != sha:
            raise ValueError('runtime identity mismatch: '+path)
    return manifest


def verify_bootstrap(manifest):
    for module, relative in ((sys.modules[__name__], 'native_scenario_runner.py'),
                             (admission, 'native_admission.py'),
                             (admission.policy, 'p23_preflight.py')):
        if digest(module.__file__) != manifest['source_files'][relative]:
            raise ValueError('executing bootstrap identity mismatch: '+relative)


def job_class(root):
    module = admission.load(Path(root)/'source/hyso/performance/_training_process.py', 'owned_job')

    class Job(module._Job):
        def __init__(self, job_bytes, processes):
            super().__init__(LIMITS['process_private_commit'], job_bytes, processes)

        def readback(self):
            value = self.Extended()
            if not self.api.QueryInformationJobObject(self.handle, 9, c.byref(value), c.sizeof(value), None):
                raise OSError(c.get_last_error(), 'hard limit readback')
            result = dict(flags=value.basic.flags, process_bytes=value.process_memory,
                          job_bytes=value.job_memory, processes=value.basic.processes)
            return result

        def process_ids(self):
            class Pids(c.Structure):
                _fields_ = [('assigned', w.DWORD), ('listed', w.DWORD),
                            ('pids', c.c_size_t*8)]
            value = Pids()
            if not self.api.QueryInformationJobObject(self.handle, 3, c.byref(value), c.sizeof(value), None):
                raise OSError(c.get_last_error(), 'owned Job PID inventory')
            if value.assigned > 8 or value.listed != value.assigned:
                raise ValueError('incomplete Job PID inventory')
            return list(value.pids[:value.listed])

        def receipt(self, process, expected):
            actual = self.readback()
            if actual != expected:
                raise ValueError('hard limit readback mismatch')
            api = self.api
            api.IsProcessInJob.argtypes = [w.HANDLE, w.HANDLE, c.POINTER(w.BOOL)]
            api.IsProcessInJob.restype = w.BOOL
            member = w.BOOL()
            if not api.IsProcessInJob(w.HANDLE(int(process._handle)), self.handle, c.byref(member)) or not member.value:
                raise ValueError('Job assignment ownership failed')
            # Give the child a QUERY-only handle to verify its exact assigned Job.
            api.GetCurrentProcess.restype = w.HANDLE
            api.DuplicateHandle.argtypes = [w.HANDLE,w.HANDLE,w.HANDLE,c.POINTER(w.HANDLE),w.DWORD,w.BOOL,w.DWORD]
            api.DuplicateHandle.restype = w.BOOL
            remote = w.HANDLE()
            if not api.DuplicateHandle(api.GetCurrentProcess(), self.handle,
                    w.HANDLE(int(process._handle)), c.byref(remote), 4, False, 0):
                raise OSError(c.get_last_error(), 'Job query handle receipt')
            return dict(pid=process.pid, parent_pid=os.getpid(), limits=actual,
                        assigned=True, query_handle=remote.value)
    return Job


def expected_limits(kind):
    return dict(flags=0x2308, process_bytes=LIMITS['process_private_commit'],
                job_bytes=LIMITS[kind+'_job_private_commit'], processes=LIMITS[kind+'_processes'])


def validate_gate(receipt, *, pid, parent_pid, limits, token):
    if (receipt['pid'] != pid or receipt['parent_pid'] != parent_pid
            or receipt['limits'] != limits or receipt['assigned'] is not True
            or receipt['token'] != token or not receipt['query_handle']):
        raise ValueError('missing/invalid Job assignment receipt')


def await_gate(work, role, token):
    path = Path(work)/(role+'-admit.json')
    end = time.monotonic()+10
    while not path.exists():
        if time.monotonic() >= end:
            raise TimeoutError('missing Job assignment receipt')
        time.sleep(.01)
    receipt = read(path)
    kind = 'outer' if role == 'parent' else 'child'
    validate_gate(receipt, pid=os.getpid(), parent_pid=os.getppid(),
                  limits=expected_limits(kind), token=token)
    api = c.WinDLL('kernel32', use_last_error=True)
    api.GetCurrentProcess.restype = w.HANDLE
    api.IsProcessInJob.argtypes = [w.HANDLE,w.HANDLE,c.POINTER(w.BOOL)]
    api.IsProcessInJob.restype = w.BOOL
    member = w.BOOL()
    if not api.IsProcessInJob(api.GetCurrentProcess(), w.HANDLE(receipt['query_handle']), c.byref(member)) or not member.value:
        raise ValueError('child exact Job membership failed')
    # Independently read limits using the duplicated query-only handle.
    Job = job_class(Path(work).parent)
    probe = Job.__new__(Job)
    # Obtain the ctypes layouts from an empty owned Job, never from a user Job.
    temporary = Job(LIMITS[kind+'_job_private_commit'], LIMITS[kind+'_processes'])
    try:
        probe.Extended, probe.api, probe.handle = temporary.Extended, temporary.api, receipt['query_handle']
        if probe.readback() != expected_limits(kind):
            raise ValueError('child hard limit readback mismatch')
    finally:
        temporary.close()
        api.CloseHandle.argtypes = [w.HANDLE]
        api.CloseHandle(w.HANDLE(receipt['query_handle']))
    publish(Path(work)/(role+'-identity.json'), dict(pid=os.getpid(), parent_pid=os.getppid(),
            manifest_sha256=os.environ['HYSO_CANDIDATE_SHA'], job_verified=True,
            cuda_imported='torch' in sys.modules, token=token))


def supervise(root, work, role, mode, token, expected_sha, timeout=None):
    kind = 'outer' if role == 'parent' else 'child'
    budget = LIMITS['outer_timeout_s' if kind == 'outer' else 'child_timeout_s']
    if timeout is not None:  # only shorter dry-run deadlines
        if (type(timeout) not in (int,float) or not math.isfinite(timeout)
                or not 0 < timeout <= budget or mode == 'native'):
            raise ValueError('invalid timeout override')
        budget = timeout
    Job = job_class(root)
    job = Job(LIMITS[kind+'_job_private_commit'], LIMITS[kind+'_processes'])
    p, threads = None, []
    logs = {name: bytearray() for name in ('stdout', 'stderr')}
    counts = dict(stdout=0, stderr=0)
    overflow = threading.Event()
    start = time.monotonic()
    outcome, cleanup, assignment = None, None, None
    owned_pids = set()

    def drain(pipe, name):
        try:
            while True:
                data = pipe.read(8192)
                if not data:
                    break
                counts[name] += len(data)
                logs[name].extend(data[:max(0, 65536-len(logs[name]))])
                if counts[name] > 65536:
                    overflow.set()
        finally:
            pipe.close()

    try:
        if job.readback() != expected_limits(kind) or job.observation()['active_processes'] != 0:
            raise ValueError('initial empty Job/limit readback failed')
        env = dict(os.environ, HYSO_CANDIDATE_SHA=expected_sha,
                   PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='0' if mode == 'native' else '-1')
        p = subprocess.Popen([sys.executable, '-I', '-S', '-B', str(Path(root)/'source/native_scenario_runner.py'),
              '--internal', role, '--root', str(root), '--work', str(work), '--mode', mode,
              '--token', token], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
              env=env, creationflags=subprocess.CREATE_NO_WINDOW)
        job.assign(p)
        assignment = job.receipt(p, expected_limits(kind))
        owned_pids.update(job.process_ids())
        assignment['token'] = token
        for name in logs:
            thread = threading.Thread(target=drain, args=(getattr(p, name), name), daemon=True)
            threads.append(thread)
            thread.start()
        publish(Path(work)/(role+'-admit.json'), assignment)
        while p.poll() is None:
            owned_pids.update(job.process_ids())
            if time.monotonic()-start >= budget:
                outcome = 'timeout'
                break
            if overflow.is_set():
                outcome = 'bounded_log_exceeded'
                break
            paths = list(Path(work).glob('*.json'))
            if len(paths) > 32 or any(path.stat().st_size > LIMITS['result_bytes'] for path in paths):
                outcome = 'artifact_limit_exceeded'
                break
            time.sleep(.01)
    finally:
        try:
            job.terminate()  # only this newly created Job, including descendants
            if p is not None:
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Assignment itself may have failed. This handle is from
                    # our own Popen; never terminate an inventory/user PID.
                    p.kill()
                    p.wait(timeout=5)
            end = time.monotonic()+5
            while True:
                cleanup = job.observation()
                if cleanup['active_processes'] == 0 or time.monotonic() >= end:
                    break
                time.sleep(.01)
            for thread in threads:
                thread.join(timeout=2)
        finally:
            job.close()
            if p is not None and not threads:
                p.stdout.close()
                p.stderr.close()
    if cleanup['active_processes'] != 0 or any(t.is_alive() for t in threads):
        raise RuntimeError('missing tree cleanup evidence')
    if overflow.is_set():
        outcome = 'bounded_log_exceeded'
    result = dict(role=role, pid=None if p is None else p.pid, parent_pid=os.getpid(),
                  assignment=assignment, deadline_seconds=budget, exit_code=None if p is None else p.returncode,
                  outcome=outcome, elapsed_seconds=time.monotonic()-start,
                  cleanup=cleanup, owned_leader_exited=p is not None and p.poll() is not None,
                  owned_pids=sorted(owned_pids),
                  kill_on_close=True, log_bytes=counts,
                  logs={k:bytes(v).decode('utf-8', errors='replace') for k,v in logs.items()})
    publish(Path(work)/(role+'-receipt.json'), result)
    return result


def loaded_identity(root, manifest):
    for name, module in list(sys.modules.items()):
        if name == 'hyso' or name.startswith('hyso.'):
            path = Path(module.__file__).resolve()
            relative = path.relative_to(Path(root)/'source').as_posix()
            if digest(path) != manifest['source_files'][relative]:
                raise ValueError('loaded source identity mismatch')
        if name == 'torch':
            if Path(module.__file__).resolve() != Path(manifest['site_packages'])/'torch/__init__.py':
                raise ValueError('loaded Torch origin mismatch')


def native_worker(root, work, manifest):
    # Called only after child Job receipt, exact membership and full seal checks.
    capture, review = read(Path(work)/'capture.json'), read(Path(work)/'review.json')
    decision = admission.evaluate(capture, review, run_id=manifest['run_id'],
                                  manifest_sha256=os.environ['HYSO_CANDIDATE_SHA'])
    if not decision['eligible']:
        raise ValueError('GPU admission expired or rejected before CUDA import')
    sys.path[:0] = [str(Path(root)/'source'), manifest['site_packages']]
    import torch
    from hyso.benchmark import BenchmarkRunner
    loaded_identity(root, manifest)
    # Check the live clock again after imports, before CUDA initialization.
    if not admission.evaluate(capture, review, run_id=manifest['run_id'],
            manifest_sha256=os.environ['HYSO_CANDIDATE_SHA'])['eligible']:
        raise ValueError('GPU admission expired during imports')
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('native CUDA required')
    torch.cuda.set_per_process_memory_fraction(.15, 0)
    torch.manual_seed(71)
    telemetry = admission.load(Path(root)/'source/hyso/performance/_telemetry.py', 'candidate_telemetry')
    before = telemetry.gpu_observe(admission.policy.GPU_UUID, 2)
    model = torch.nn.Conv2d(1, 3, 3, padding=1).to('cuda:0').eval()
    inputs = torch.randn(2, 1, 7, 9, device='cuda:0', dtype=torch.float32)
    # Reuses the existing benchmark protocol and tiny Conv2d validation dimensions.
    result = BenchmarkRunner(model, device='cuda:0', example_input=inputs, precision='fp32',
             warmup_iterations=1, measure_iterations=3, repeats=1, measure_cuda_memory=True).run().to_dict()
    latencies = [value for row in result['timing']['latency_samples_ms'] for value in row]
    if (len(latencies) != 3 or any(not math.isfinite(v) or v <= 0 for v in latencies)
            or not math.isfinite(result['throughput']['images_per_sec'])
            or result['throughput']['images_per_sec'] <= 0
            or result['cuda_memory']['status'] != 'measured'
            or result['cuda_memory']['counter_reset_performed'] is not True):
        raise RuntimeError('missing/invalid P01/P02/P09 result')
    after = telemetry.gpu_observe(admission.policy.GPU_UUID, 2)
    if any(s['status'] != 'measured' for s in (before, after)):
        raise RuntimeError('missing P16 device-wide sensor samples')
    loaded_identity(root, manifest)
    return dict(scenario=SCENARIO, executions=1, baseline=0, retries=0,
                benchmark=result, device_wide_sensor_samples=[before, after])


def internal(args):
    # -I -S omits our source directory: main restores only the sealed helper path.
    root, work = Path(args.root), Path(args.work)
    await_gate(work, args.internal, args.token)
    manifest = verify(root, os.environ['HYSO_CANDIDATE_SHA'])
    verify_bootstrap(manifest)
    if args.internal == 'parent':
        timeout = 15 if args.mode == 'timeout' else None
        result = supervise(root, work, 'worker', args.mode, args.token,
                           os.environ['HYSO_CANDIDATE_SHA'], timeout)
        if result['outcome'] is not None or result['exit_code'] != 0:
            raise RuntimeError('child stop: '+str(result['outcome']))
    elif args.mode == 'native':
        publish(work/'result.json', native_worker(root, work, manifest))
    else:
        # Stdlib-only descendant; it inherits both Jobs and is intentionally
        # left for the supervisor to clean up after leader exit/timeout.
        child = subprocess.Popen([sys.executable, '-I', '-S', '-B', '-c', 'import time; time.sleep(120)'],
                                  creationflags=subprocess.CREATE_NO_WINDOW)
        publish(work/'simulated-tree.json', dict(worker_pid=os.getpid(), descendant_pid=child.pid,
                                                cuda_imported='torch' in sys.modules))
        if args.mode in ('timeout', 'outer-timeout'):
            time.sleep(120)
        elif args.mode == 'log-overflow':
            sys.stdout.write('x'*70000)
            sys.stdout.flush()
        else:
            publish(work/'result.json', dict(simulated=True, cuda_imported='torch' in sys.modules))
    verify(root, os.environ['HYSO_CANDIDATE_SHA'])


def controller(root, work, mode, approval=None):
    root, work = Path(root), Path(work)
    seal = read(root/'seal.json')
    manifest = verify(root, seal['manifest_sha256'])
    verify_bootstrap(manifest)
    if mode == 'native':
        if approval is None:
            raise ValueError('separate explicit native authorization required')
        value = read(approval)
        if value != dict(run_id=manifest['run_id'], manifest_sha256=seal['manifest_sha256'],
                         scenario=SCENARIO, maximum_executions=1, explicit_user_approval=True):
            raise ValueError('native approval identity mismatch')
        # Admission must already be freshly collected and explicitly reviewed.
        decision = admission.evaluate(read(work/'capture.json'), read(work/'review.json'),
                      run_id=manifest['run_id'], manifest_sha256=seal['manifest_sha256'])
        if not decision['eligible']:
            raise ValueError('native GPU admission NO-GO')
        publish(root/'execution-consumed.json', dict(run_id=manifest['run_id'], time_ns=time.time_ns()))
        publish(work/'native-authorization.json', value)
    else:
        work.mkdir()  # each dry run owns a fresh exclusive directory
    token = str(uuid.uuid4())
    publish(work/'controller-identity.json', dict(pid=os.getpid(), parent_pid=os.getppid(),
            manifest_sha256=seal['manifest_sha256'], run_id=manifest['run_id'],
            source_verified=True, runtime_verified=True, cuda_imported='torch' in sys.modules))
    receipt = supervise(root, work, 'parent', mode, token, seal['manifest_sha256'],
                         timeout=15 if mode == 'outer-timeout' else None)
    verify(root, seal['manifest_sha256'])
    if mode in ('native', 'success'):
        if receipt['outcome'] is not None or receipt['exit_code'] != 0:
            raise RuntimeError('outer stop: '+str(receipt['outcome']))
        child = read(work/'worker-receipt.json')
        if (child['outcome'] is not None or child['exit_code'] != 0
                or child['cleanup']['active_processes'] != 0 or not child['owned_leader_exited']):
            raise RuntimeError('missing child cleanup/timeout evidence')
        for role in ('parent', 'worker'):
            identity = read(work/(role+'-identity.json'))
            if identity['manifest_sha256'] != seal['manifest_sha256'] or not identity['job_verified'] or identity['cuda_imported']:
                raise ValueError('missing/invalid pre-import identity receipt')
        read(work/'result.json')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--r3', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--mode', choices=['success', 'timeout', 'outer-timeout', 'log-overflow', 'native'], default='success')
    parser.add_argument('--approval', type=Path)
    parser.add_argument('--internal', choices=['parent', 'worker'])
    parser.add_argument('--token')
    args = parser.parse_args()
    if args.prepare:
        manifest = prepare(args.prepare, args.r3)
        print(json.dumps(dict(prepared=True, run_id=manifest['run_id'], cuda_workload_started=False)))
    elif args.internal:
        internal(args)
    else:
        print(json.dumps(controller(args.root, args.work, args.mode, args.approval)))


if __name__ == '__main__':
    main()
