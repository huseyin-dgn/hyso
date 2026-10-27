"""Warning-only GPU context with the preserved owned Windows isolation runner.

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
import gpu_admission_policy as admission

LIMITS = admission.policy.LIMITS
SCENARIO = 'conv2d_fp32_p01_p02_p09_p16'
P02_SCENARIO = 'conv2d_fp32_p02_completion'
PIPELINE_SCENARIO = 'conv2d_fp32_p03_p04_pipeline'
TRAINING_SCENARIO = 'linear_fp32_p19_p20_p22_p25_training'
COLD_SCENARIO = 'conv2d_fp32_p05_cold'
CAPACITY_SCENARIO = 'conv2d_fp32_p17_capacity'
PREPARED = {
    PIPELINE_SCENARIO: dict(kind='pipeline', seed=71, input_shape=[2,1,7,9],
        protocol=dict(device='cuda:0', precision='fp32', samples=3, repeats=1, warmup_samples=1, queue_depth=2)),
    TRAINING_SCENARIO: dict(kind='training', seed=71, epochs=1, samples=5, batch=2,
        accum_steps=2, optimizer='AdamW', amp=False, event_buffer=2),
    COLD_SCENARIO: dict(kind='cold', feature='P05', seed=71, input_shape=[2,1,7,9],
        protocol=dict(trials=1, timeout_seconds=40., warmup_iterations=1, warm_samples=2)),
    CAPACITY_SCENARIO: dict(kind='capacity', feature='P17', seed=71, input_shape=[2,1,7,9],
        protocol=dict(candidates=[1,2], maximum_batch=4, cpu_memory_limit_bytes=4*1024**3,
            input_limit_bytes=1024**2, timeout_seconds=20., stop_policy='first_infeasible',
            warmup_iterations=1, measure_iterations=2, repeats=1, cuda_memory_fraction=.15)),
}
for _axis, _feature, _points in (('batch','P06',[1,2]), ('resolution','P07',[[7,9],[9,11]]),
                                ('precision','P08',['fp32','amp_fp16'])):
    PREPARED['conv2d_'+_feature.lower()+'_sweep'] = dict(kind='sweep', axis=_axis,
        feature=_feature, points=_points, seed=71, input_shape=[2,1,7,9],
        protocol=dict(device='cuda:0', precision='fp32', warmup_iterations=1,
            measure_iterations=2, repeats=1, sustained_batches=3, queue_depth=2,
            continue_on_error=False, measure_cuda_memory=False))
BLOCKED = {}
SCENARIOS = (SCENARIO, P02_SCENARIO, *PREPARED)
P02_PROTOCOL = dict(device='cuda:0', precision='fp32', warmup_iterations=1,
                    batch_count=5, duration_seconds=None, repeats=1,
                    queue_depth=2, concurrency=1)


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


def prepare(destination, r3, scenario=SCENARIO):
    """Copy current sources, reuse and verify the r3 runtime identity inventory."""
    root = Path(__file__).resolve().parents[2]
    if scenario not in SCENARIOS:
        raise ValueError('unsupported scenario')
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
        Path(__file__), Path(__file__).with_name('gpu_admission_policy.py'),
        Path(__file__).with_name('native_admission.py'),
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
    manifest = dict(run_id=str(uuid.uuid4()), scenario=scenario, source_files=hashes,
                    runtime_files=runtime, interpreter=interpreter,
                    site_packages=str(Path(interpreter).parent/'Lib/site-packages'),
                    limits=LIMITS, executions=1, retries=0, baselines=0,
                    admission_policy=admission.POLICY_ID)
    if scenario == P02_SCENARIO:
        manifest['p02_protocol'] = dict(P02_PROTOCOL)
    if scenario in PREPARED:
        manifest['scenario_contract'] = PREPARED[scenario]
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
    if (manifest['limits'] != LIMITS or manifest['scenario'] not in SCENARIOS
            or manifest.get('executions') != 1 or manifest.get('retries') != 0
            or manifest.get('baselines') != 0 or manifest.get('admission_policy') != admission.POLICY_ID
            or any(type(manifest.get(key)) is not int for key in ('executions', 'retries', 'baselines'))):
        raise ValueError('scenario/limit mismatch')
    if manifest['scenario'] == P02_SCENARIO and json.dumps(
            manifest.get('p02_protocol'), sort_keys=True, allow_nan=False) != json.dumps(P02_PROTOCOL, sort_keys=True):
        raise ValueError('P02 sealed protocol mismatch')
    if manifest['scenario'] in PREPARED and json.dumps(manifest.get('scenario_contract'),
            sort_keys=True, allow_nan=False) != json.dumps(PREPARED[manifest['scenario']], sort_keys=True):
        raise ValueError('sealed scenario contract mismatch')
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
    for module, relative in ((sys.modules[__name__], 'gpu_scenario_runner.py'),
                             (admission, 'gpu_admission_policy.py'),
                             (admission.legacy, 'native_admission.py'),
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


def process_creation_identity(handle=None):
    api = c.WinDLL('kernel32', use_last_error=True)
    api.GetCurrentProcess.restype = w.HANDLE
    api.GetProcessId.argtypes = [w.HANDLE]; api.GetProcessId.restype = w.DWORD
    api.GetProcessTimes.argtypes = [w.HANDLE]+[c.POINTER(w.FILETIME)]*4
    api.GetExitCodeProcess.argtypes = [w.HANDLE,c.POINTER(w.DWORD)]
    handle = api.GetCurrentProcess() if handle is None else w.HANDLE(handle)
    times = [w.FILETIME() for _ in range(4)]; code = w.DWORD()
    if not api.GetProcessTimes(handle, *[c.byref(t) for t in times]) or not api.GetExitCodeProcess(handle,c.byref(code)):
        raise OSError(c.get_last_error(), 'parent process handle identity')
    if code.value != 259:
        raise ValueError('parent no longer alive')
    return dict(pid=api.GetProcessId(handle), creation_time_100ns=(times[0].dwHighDateTime<<32)|times[0].dwLowDateTime)


def duplicate_parent_handle(api, child):
    remote = w.HANDLE()
    if not api.DuplicateHandle(api.GetCurrentProcess(), api.GetCurrentProcess(),
            w.HANDLE(int(child._handle)), c.byref(remote), 0x1000|0x100000, False, 0):
        raise OSError(c.get_last_error(), 'parent identity handle duplication')
    return remote.value


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


def supervise(root, work, role, mode, token, expected_sha, timeout=None, nested=False):
    kind = 'outer' if role == 'parent' else 'child'
    budget = LIMITS['outer_timeout_s' if kind == 'outer' else 'child_timeout_s']
    if timeout is not None:  # only shorter dry-run deadlines
        if (type(timeout) not in (int,float) or not math.isfinite(timeout)
                or not 0 < timeout <= budget or (mode == 'native' and not nested)):
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
    ready_receipt_ns = []

    def drain(pipe, name):
        try:
            while True:
                data = pipe.read1(8192)
                if not data:
                    break
                counts[name] += len(data)
                logs[name].extend(data[:max(0, 65536-len(logs[name]))])
                if nested and name == 'stdout' and not ready_receipt_ns and b'@@HYSO_READY ' in logs[name] and b'\n' in logs[name]:
                    ready_receipt_ns.append(time.perf_counter_ns())
                if counts[name] > 65536:
                    overflow.set()
        finally:
            pipe.close()

    try:
        if job.readback() != expected_limits(kind) or job.observation()['active_processes'] != 0:
            raise ValueError('initial empty Job/limit readback failed')
        env = dict(os.environ, HYSO_CANDIDATE_SHA=expected_sha,
                   PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='0' if mode == 'native' else '-1')
        p = subprocess.Popen([sys.executable, '-I', '-S', '-B', str(Path(root)/'source/gpu_scenario_runner.py'),
              '--internal', 'nested' if nested else role, '--root', str(root), '--work', str(work), '--mode', mode,
              '--token', token], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
              env=env, creationflags=subprocess.CREATE_NO_WINDOW)
        job.assign(p)
        assignment = job.receipt(p, expected_limits(kind))
        owned_pids.update(job.process_ids())
        assignment['token'] = token
        if nested:
            assignment['request_sha256'] = digest(Path(work)/'nested-request.json')
            assignment['parent_identity'] = process_creation_identity()
            assignment['parent_handle'] = duplicate_parent_handle(job.api, p)
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
    if nested:
        result['ready_receipt_ns'] = ready_receipt_ns[0] if ready_receipt_ns else None
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


def optional_context(path):
    if not Path(path).exists():
        return None
    try:
        return read(path)
    except (json.JSONDecodeError, UnicodeError, OSError):
        return None


def validate_p02_payload(data):
    """Validate the fixed scenario's transport without importing Torch in parent.

    The worker additionally round-trips the full public ThroughputResult schema.
    These checks bind its completion ledger to this sealed five-batch contract.
    """
    try:
        if (data['format'] != 'hyso.performance.throughput' or data['schema_version'] != 1
                or data['feature_ids'] != ['P02']
                or data['protocol']['configuration'] != P02_PROTOCOL
                or data['protocol']['id'] != 'continuous_completion_window'
                or data['protocol']['scope'] != 'resident_forward_completion_wall'
                or data['protocol']['clock'] != 'perf_counter_ns'
                or data['protocol']['synchronization'] != 'bounded_group_completion_and_terminal_event'
                or data['provenance']['input']['shape'] != [2, 1, 7, 9]
                or data['provenance']['input']['device'] != 'cuda:0'
                or data['provenance']['input']['dtype'] != 'torch.float32'
                or data['provenance']['environment']['device'] != 'cuda:0'
                or data['provenance']['environment']['stream_id'] != 0
                or data['requested'] != dict(batch_count=5, duration_seconds=None,
                                            repeats=1, warmup_iterations=1)):
            raise ValueError('P02 protocol/provenance mismatch')
        windows = data['raw_evidence']['windows']
        if type(windows) is not list or len(windows) != 1:
            raise ValueError('P02 window count mismatch')
        row = windows[0]
        expected = dict(repeat_index=0, admitted_batches=5, completed_batches=5,
                        completed_images=10, pending_batches=0, completion_groups=3,
                        terminal_drains=1)
        if any(type(row[k]) is not int or row[k] != v for k, v in expected.items()):
            raise ValueError('P02 completion conservation failure')
        if (row['completed_id_range'] != [0, 4] or type(row['max_pending_batches']) is not int
                or not 1 <= row['max_pending_batches'] <= 2
                or type(row['backpressure_syncs']) is not int or not 0 <= row['backpressure_syncs'] <= 2
                or any(type(row[k]) is not int or row[k] < 0
                       for k in ('start_ns', 'admission_end_ns', 'end_ns'))
                or not row['start_ns'] <= row['admission_end_ns'] <= row['end_ns']
                or row['end_ns'] <= row['start_ns']):
            raise ValueError('P02 completion interval/queue failure')
        elapsed = (row['end_ns']-row['start_ns'])/1e9
        rate = 10/elapsed
        if (not math.isfinite(rate) or rate <= 0
                or row['elapsed_wall_seconds'] != elapsed
                or row['tail_seconds'] != (row['end_ns']-row['admission_end_ns'])/1e9
                or row['images_per_second'] != rate
                or data['actual'] != dict(repeats=1, completed_batches=5, completed_images=10, pending_batches=0)
                or data['measurements'] != dict(value=rate, unit='images/s', status='measured',
                    reason=None, elapsed_wall_seconds=elapsed,
                    aggregation='total_completed_images_over_total_wall_seconds')):
            raise ValueError('P02 wall-clock rate/conservation mismatch')
        if ('warmup' not in data['protocol']['excluded']
                or 'terminal_tail' not in data['protocol']['included']):
            raise ValueError('P02 measurement scope mismatch')
    except (KeyError, TypeError, IndexError, ZeroDivisionError, OverflowError) as error:
        raise ValueError('invalid P02 result transport') from error


def measure_p02_completion(model, input_source):
    """Resident NCHW Tensor, not a loader/callback; public API owns inference mode."""
    from hyso.performance.inference import measure_throughput, ThroughputProtocol
    from hyso.performance.result import ThroughputResult
    result = measure_throughput(model, input_source, protocol=ThroughputProtocol(**P02_PROTOCOL))
    data = ThroughputResult(result.to_dict()).to_dict()
    validate_p02_payload(data)
    return data


def measure_pipeline_pair(torch, contract):
    from hyso.performance.pipeline import (PipelineSample, PipelineStage, PipelineProtocol,
                                          measure_pipeline, analyze_stages)
    from hyso.performance.result import PipelineResult, StageTimingResult
    source_ids, delivered = [], []
    def source():
        index = len(source_ids)
        identifier = 'frame-'+str(index)
        source_ids.append(identifier)
        return PipelineSample(identifier, torch.full(tuple(contract['input_shape']), float(index), dtype=torch.float32))
    def preprocess(frame):
        return PipelineSample(frame.sample_id, frame.payload/255.)
    def transfer(frame):
        return PipelineSample(frame.sample_id, frame.payload.to('cuda:0'))
    def postprocess(frame):
        return PipelineSample(frame.sample_id, frame.payload.mean((2,3), keepdim=True))
    def deliver(frame):
        delivered.append(frame.sample_id)
        return PipelineSample(frame.sample_id, frame.payload.cpu())
    model = torch.nn.Conv2d(1,3,3,padding=1).to('cuda:0').eval()
    pipeline = measure_pipeline(model, source,
        preprocess=PipelineStage('preprocess', preprocess),
        transfer=PipelineStage('transfer', transfer, device='cuda'),
        postprocess=PipelineStage('postprocess', postprocess, device='cuda'),
        deliver=PipelineStage('deliver', deliver), protocol=PipelineProtocol(**contract['protocol']))
    pipeline = PipelineResult(pipeline.to_dict())
    stages = StageTimingResult(analyze_stages(pipeline).to_dict())
    return {'P03':pipeline.to_dict(), 'P04':stages.to_dict()}, dict(source_ids=source_ids, delivered_ids=delivered)


def measure_sweep_case(torch, contract):
    from hyso.performance.sweeps import run_sweep, SweepProtocol
    from hyso.performance.result import SweepResult
    # run_sweep requires a CPU FP32 master and owns per-point device/dtype copies.
    model = torch.nn.Sequential(torch.nn.Conv2d(1,3,3,padding=1),
                                torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()).eval()
    def inputs(shape, dtype, device):
        return torch.randn(shape, dtype=dtype, device=device)
    result = run_sweep(model, inputs, axis=contract['axis'], points=contract['points'],
        input_shape=tuple(contract['input_shape']), protocol=SweepProtocol(**contract['protocol']))
    return {contract['feature']:SweepResult(result.to_dict()).to_dict()}, {}


def training_state_snapshot(torch, trainer):
    """Small fixture state, outside measured training; never runs a forward."""
    import random

    def plain(value):
        if torch.is_tensor(value):
            if value.numel() > 16384:
                raise ValueError('training state tensor budget exceeded')
            return dict(dtype=str(value.dtype), device=str(value.device),
                        shape=list(value.shape), values=value.detach().cpu().tolist())
        if isinstance(value, dict):
            return {str(key):plain(item) for key,item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise ValueError('unsupported training state value')

    state = dict(model=plain(trainer.model.state_dict()),
                 optimizer=plain(trainer.optimizer.state_dict()),
                 rng=dict(python=plain(random.getstate()),
                          torch_cpu=plain(torch.get_rng_state()),
                          cuda_0=plain(torch.cuda.get_rng_state(0))))
    numpy = sys.modules.get('numpy')
    state['rng']['numpy'] = (plain([numpy.random.get_state()[0],
        numpy.random.get_state()[1].tolist(), *numpy.random.get_state()[2:]])
        if numpy is not None else dict(status='not_loaded'))
    raw = json.dumps(state, sort_keys=True, allow_nan=False).encode()
    if len(raw) > 128*1024:
        raise ValueError('training state artifact budget exceeded')
    return dict(state=state, sha256=hashlib.sha256(raw).hexdigest(),
                scope='read_only_snapshot_outside_measured_fit')


def measure_training_bundle(torch, contract):
    from hyso.training import Trainer
    from hyso.performance.gradients import (measure_gradient_diagnostics, GradientDiagnosticsProtocol,
                                            GradientDiagnosticsResult)
    from hyso.performance.training import TrainingTimingProtocol, TrainingPerformanceResult
    from hyso.performance.training_details import (TrainingStageResult, TrainingThroughputResult,
                                                  analyze_training_throughput)
    model = torch.nn.Sequential(torch.nn.Linear(3,4), torch.nn.Dropout(.2), torch.nn.Linear(4,2)).to('cuda:0')
    trainer = Trainer(model, optimizer=torch.optim.AdamW(model.parameters(), lr=.01),
        loss_fn=torch.nn.CrossEntropyLoss(), device='cuda:0', amp=False, accum_steps=2, grad_clip=.1)
    train = list(zip(torch.linspace(-1,1,15).reshape(5,3).split(2), torch.tensor([0,1,0,1,0]).split(2)))
    before = training_state_snapshot(torch, trainer)
    gradient = measure_gradient_diagnostics(trainer, train, epochs=contract['epochs'],
        protocol=GradientDiagnosticsProtocol(max_attempts=8, max_tensor_records=128, chunk_elements=3),
        timing_protocol=TrainingTimingProtocol(event_buffer=contract['event_buffer'], max_records=2000))
    after = training_state_snapshot(torch, trainer)
    data = GradientDiagnosticsResult(gradient.to_dict()).to_dict()
    timing = TrainingPerformanceResult(data['source_p19'])
    stages = TrainingStageResult(data['source_p20'])
    rates = TrainingThroughputResult(analyze_training_throughput(timing).to_dict())
    return {'P19':timing.to_dict(), 'P20':stages.to_dict(), 'P22':rates.to_dict(), 'P25':data}, dict(
        fits=1, before=before, after=after,
        configuration=dict(model='Linear(3,4)/Dropout(0.2)/Linear(4,2)',
            optimizer='AdamW', learning_rate=.01, loss='CrossEntropyLoss',
            device='cuda:0', precision='fp32', amp=False, accum_steps=2,
            grad_clip=.1, seed=contract['seed'], epochs=contract['epochs'],
            samples=5, batches=[2,2,1], activation_hooks=False),
        source_projection='P19/P20 include P25 instrumentation; P22 derived from same P19, not a second fit')


def validate_pipeline_pair(results, frames):
    p, s = results['P03'], results['P04']
    expected_ids = ['frame-'+str(i) for i in range(4)]
    rows = p['raw_evidence']['samples']
    if (p['format'] != 'hyso.performance.pipeline' or s['format'] != 'hyso.performance.stage_timing'
            or p['feature_ids'] != ['P03'] or s['feature_ids'] != ['P04']
            or p['protocol']['configuration'] != PREPARED[PIPELINE_SCENARIO]['protocol']
            or frames != dict(source_ids=expected_ids, delivered_ids=expected_ids)
            or [row['sample_id'] for row in rows] != expected_ids[1:]
            or s['identity']['source_result_id'] != p['result_id']
            or s['raw_evidence'] != p['raw_evidence']
            or p['actual']['completed'] != 3 or p['actual']['pending'] != 0
            or p['actual']['forward_count'] != 4 or p['actual']['warmup_samples'] != 1
            or p['provenance']['environment']['device'] != 'cuda:0'
            or p['provenance']['environment']['stream_id'] != 0
            or p['protocol']['clock'] != 'perf_counter_ns'
            or p['protocol']['scope'] != 'read_start_to_post_completion'
            or s['protocol']['scope'] != 'instrumented_stage_host_and_device'):
        raise ValueError('pipeline identity/order/scope mismatch')
    names = ['source','preprocess','transfer','inference','postprocess','deliver']
    for row in rows:
        if ([stage['name'] for stage in row['stages']] != names
                or row['e2e_ms'] != (row['postprocess_completion_ns']-row['source_start_ns'])/1e6
                or not math.isfinite(row['e2e_ms']) or row['e2e_ms'] <= 0):
            raise ValueError('pipeline E2E/clock mismatch')
        for stage in row['stages']:
            if (stage['sample_id'] != row['sample_id'] or stage['unit'] != 'ms'
                    or stage['status'] != 'measured' or stage['reason'] is not None
                    or not stage['start_ns'] <= stage['launch_end_ns'] <= stage['completion_ns']
                    or stage['host_elapsed_ms'] != (stage['completion_ns']-stage['start_ns'])/1e6
                    or stage['host_launch_ms'] != (stage['launch_end_ns']-stage['start_ns'])/1e6):
                raise ValueError('stage identity/host clock mismatch')
            if stage['name'] in ('transfer','inference','postprocess'):
                if (stage['device'] != 'cuda' or stage['device_event_status'] != 'measured'
                        or not isinstance(stage['device_event_ms'], (int,float))
                        or not math.isfinite(stage['device_event_ms']) or stage['device_event_ms'] <= 0
                        or stage['sync_policy'] != 'end_event_sync' or stage['stream_id'] != 0):
                    raise ValueError('missing CUDA stage completion')
            elif stage['device'] != 'cpu' or stage['device_event_ms'] is not None:
                raise ValueError('CPU stage cannot claim CUDA time')
    for name in names:
        stages = [row['stages'][names.index(name)] for row in rows]
        metric = s['measurements'][name]
        device = [stage['device_event_ms'] for stage in stages if stage['device_event_ms'] is not None]
        if (metric['count'] != 3 or metric['unit'] != 'ms'
                or metric['host_mean_ms'] != math.fsum(stage['host_elapsed_ms'] for stage in stages)/3
                or metric['device_mean_ms'] != (math.fsum(device)/3 if device else None)):
            raise ValueError('stage summary mismatch')


def nested_bootstrap(root, work, manifest):
    """Stdlib only until exact Job admission, live parent handle and full seal."""
    receipt = read(work/'worker-admit.json')
    request = read(work/'nested-request.json')
    try:
        if (digest(work/'nested-request.json') != receipt['request_sha256']
                or digest(work/'request.json') != request['payload_sha256']
                or process_creation_identity(receipt['parent_handle']) != receipt['parent_identity']
                or receipt['parent_identity']['pid'] != os.getppid()):
            raise ValueError('nested request/parent creation identity mismatch')
        origin = Path(request['origin_work'])
        if origin.parent.resolve() != Path(root).resolve():
            raise ValueError('nested controller evidence outside candidate')
        controller = read(origin/'controller-identity.json')
        parent = read(origin/'parent-identity.json')
        worker = read(origin/'worker-identity.json')
        if (worker['pid'] != os.getppid() or worker['parent_pid'] != parent['pid']
                or parent['parent_pid'] != controller['pid'] or worker['token'] != receipt['token']
                or any(v['manifest_sha256'] != os.environ['HYSO_CANDIDATE_SHA'] or v['cuda_imported']
                       for v in (controller,parent,worker))
                or not parent['job_verified'] or not worker['job_verified']
                or not controller['source_verified'] or not controller['runtime_verified']):
            raise ValueError('nested controller/parent/worker chain mismatch')
        kind = request['kind']
        if kind not in ('cold','capacity','dryrun'):
            raise ValueError('unapproved nested worker kind')
        scenario = manifest['scenario']
        if kind != 'dryrun' and PREPARED[scenario]['kind'] != kind:
            raise ValueError('nested scenario mismatch')
        publish(work/'nested-identity.json', dict(pid=os.getpid(), parent=receipt['parent_identity'],
            controller_pid=controller['pid'], job_verified=True, source_verified=True,
            runtime_verified=True, cuda_imported='torch' in sys.modules,
            request_sha256=receipt['request_sha256'], manifest_sha256=os.environ['HYSO_CANDIDATE_SHA']))
    finally:
        api=c.WinDLL('kernel32',use_last_error=True);api.CloseHandle.argtypes=[w.HANDLE]
        api.CloseHandle(w.HANDLE(receipt['parent_handle']))
    if kind == 'dryrun':
        action=request['action']
        if action == 'crash': os._exit(73)
        child=subprocess.Popen([sys.executable,'-I','-S','-B','-c','import time;time.sleep(120)'],creationflags=subprocess.CREATE_NO_WINDOW)
        publish(work/'dryrun-result.json',dict(cuda_imported='torch' in sys.modules,
            descendant_pid=child.pid,status='simulated_oom' if action=='oom' else 'success'))
        if action == 'timeout': time.sleep(120)
        return
    import runpy
    sys.path[:0] = [str(Path(root)/'source'),manifest['site_packages']]
    script = Path(root)/'source/hyso/performance'/('_cold_worker.py' if kind=='cold' else '_batch_worker.py')
    sys.argv = [str(script),str(work/'request.json'),str(work/'output.json')]
    if kind == 'capacity': sys.argv.append(str(work/'ready.json'))
    sys.argv.append(str(Path(root)/'source'))
    code = 0
    try:
        runpy.run_path(str(script),run_name='__main__')
    except SystemExit as error:
        code = error.code
    loaded_identity(root,manifest)
    verify(root,os.environ['HYSO_CANDIDATE_SHA'])
    raise SystemExit(code)


def measure_nested_scenario(torch, contract, root, work, manifest):
    from dataclasses import asdict
    from hyso.performance import cold, capacity
    from hyso.performance.result import ColdStartResult, CapacityResult
    calls = []
    artifact = None
    if contract['kind'] == 'cold':
        artifact=work/'trusted-conv2d-state.pt'
        model=torch.nn.Conv2d(1,3,3,padding=1)
        torch.save(model.state_dict(),artifact)
        del model
    recipe=cold.ColdStartRecipe('torch.nn:Conv2d',model_kwargs=dict(in_channels=1,out_channels=3,kernel_size=3,padding=1),
        input_shape=tuple(contract['input_shape']),device='cuda:0',precision='fp32',seed=71,
        artifact_path=artifact,artifact_sha256=digest(artifact) if artifact else None)

    def launch(kind, config, plan, index, unused_temporary_root):
        maximum=1 if kind=='cold' else len(contract['protocol']['candidates'])
        if len(calls)>=maximum or kind!=contract['kind']:
            raise ValueError('nested execution budget exceeded')
        directory=Path(root)/('nested-'+str(len(calls)+1));directory.mkdir()
        payload=dict(recipe=config,protocol=asdict(plan),cuda_memory_fraction=.15)
        if kind=='capacity':payload['batch']=index
        publish(directory/'request.json',payload)
        publish(directory/'nested-request.json',dict(kind=kind,origin_work=str(work),
            payload_sha256=digest(directory/'request.json')))
        start=time.perf_counter_ns()
        receipt=supervise(root,directory,'worker','native',read(work/'worker-identity.json')['token'],
            os.environ['HYSO_CANDIDATE_SHA'],timeout=plan.timeout_seconds,nested=True)
        end=time.perf_counter_ns()
        calls.append(dict(directory=str(directory),kind=kind,index=index,receipt=receipt))
        if not (directory/'nested-identity.json').is_file():
            raise RuntimeError('nested pre-import admission failed')
        identity=read(directory/'nested-identity.json')
        calls[-1]['identity']=identity
        if identity['cuda_imported'] or not identity['job_verified']:
            raise ValueError('nested pre-import identity invalid')
        status=('timed_out' if receipt['outcome']=='timeout' else 'crashed'
                if receipt['exit_code'] not in (0,1) else 'failed')
        if receipt['outcome'] not in (None,'timeout'):
            raise RuntimeError('nested resource/cleanup gate failure: '+str(receipt['outcome']))
        output=directory/'output.json'
        body=read(output) if receipt['outcome'] is None and output.is_file() else None
        if body is not None and body['worker_pid']!=receipt['pid']:
            raise ValueError('nested result process identity mismatch')
        if kind=='cold':
            ready=[json.loads(line[13:]) for line in receipt['logs']['stdout'].splitlines() if line.startswith('@@HYSO_READY ')]
            if body is not None:
                if len(ready)!=1 or ready[0]['pid']!=receipt['pid']:raise ValueError('cold ready identity mismatch')
                status=body['status']
            else:
                body=dict(worker_pid=receipt['pid'],phases={n:cold._missing(status) for n in cold.PHASES},
                    provenance=dict(device=config['device']),first_output=None,warm_samples_ms=[],artifact=None,error=status)
            ready_ns=receipt['ready_receipt_ns']
            body.update(trial_index=index,status=status,parent_start_ns=start,parent_end_ns=end,
                parent_elapsed_ms=(end-start)/1e6,exit_code=receipt['exit_code'],process_terminated=True,
                launcher_pid=receipt['pid'],termination_policy='owned_windows_job_kill_on_close',
                startup_to_ready_receipt=dict(start_ns=start,end_ns=ready_ns,
                    elapsed_ms=(ready_ns-start)/1e6 if ready_ns is not None else None,unit='ms',clock='parent_perf_counter_ns',
                    status='measured' if ready_ns else 'not_observed',reason=None if ready_ns else status,
                    device='host',backend='python_subprocess',included_operations=['launcher_and_worker_startup',
                        'pre_import_Job_identity_seal_admission','ready_pipe_receipt_and_reader_scheduling']),
                stderr_tail=receipt['logs']['stderr'][-4096:])
            return body
        row=capacity._row(index,config)
        row.update(launcher_pid=receipt['pid'],exit_code=receipt['exit_code'],process_terminated=True,
            execution_seconds=(end-start)/1e9,stderr_tail=receipt['logs']['stderr'][-4096:])
        if (directory/'ready.json').is_file():
            ready=read(directory/'ready.json')
            if ready['pid']!=receipt['pid']:raise ValueError('capacity ready identity mismatch')
            row.update(worker_pid=ready['pid'],worker_phase=ready['phase'],runtime_ready=ready['runtime_ready'])
        if body is not None and receipt['exit_code']==0:
            body.pop('worker_pid')
            if set(body)!={'status','reason','actual_batch','forward_completed','latency_mean_ms','memory','sweep','provenance'}:
                raise ValueError('invalid nested capacity result')
            row.update(body)
        else:row.update(status=status,reason='nested worker '+status)
        row['classification']=capacity._classification(row['status'])
        return row

    token=cold._owned_worker_launcher.set(launch)
    try:
        if contract['kind']=='cold':
            data=cold.measure_cold_start(recipe,protocol=cold.ColdStartProtocol(**contract['protocol']))
            raw=ColdStartResult(data.to_dict()).to_dict()
        else:
            data=capacity.find_max_batch_size(recipe,protocol=capacity.BatchSearchProtocol(**contract['protocol']))
            raw=CapacityResult(data.to_dict()).to_dict()
    finally:cold._owned_worker_launcher.reset(token)
    return {contract['feature']:raw},dict(nested_workers=calls,artifact_sha256=digest(artifact) if artifact else None)


def validate_nested_result(result):
    contract=PREPARED[result['scenario']];kind=contract['kind']
    data=result['public_results'][contract['feature']]
    if data['protocol']['configuration'] != contract['protocol']:
        # Public capacity JSON converts candidate tuple to list.
        raise ValueError('nested public protocol mismatch')
    recipe=data['protocol']['recipe']
    if (recipe['device']!='cuda:0' or recipe['precision']!='fp32' or recipe['seed']!=71
            or recipe['input_shape']!=contract['input_shape'] or recipe['factory']!='torch.nn:Conv2d'):
        raise ValueError('nested public recipe mismatch')
    rows=data['raw_evidence']['trials' if kind=='cold' else 'candidates']
    calls=result['frame_evidence']['nested_workers']
    attempted=[row for row in rows if row['status']!='not_run']
    if len(calls)!=len(attempted) or len(calls)>(1 if kind=='cold' else 2):
        raise ValueError('nested execution count mismatch')
    for row,call in zip(attempted,calls):
        receipt=call['receipt'];identity=call['identity']
        if (receipt['assignment']['limits']!=expected_limits('child') or not receipt['assignment']['assigned']
                or receipt['deadline_seconds']!=contract['protocol']['timeout_seconds']
                or receipt['cleanup']['active_processes']!=0 or not receipt['owned_leader_exited']
                or not receipt['kill_on_close'] or identity['cuda_imported']
                or not all(identity[k] for k in ('job_verified','source_verified','runtime_verified'))
                or row['launcher_pid']!=receipt['pid'] or identity['pid']!=receipt['pid']
                or not row['process_terminated'] or row['exit_code']!=receipt['exit_code']
                or any(v>65536 for v in receipt['log_bytes'].values())):
            raise ValueError('nested ownership/identity/limit/cleanup receipt mismatch')
    if kind=='cold':
        if recipe['artifact_sha256']!=result['frame_evidence']['artifact_sha256']:
            raise ValueError('cold artifact identity mismatch')
        row=rows[0]
        if row['status']=='measured' and (row['worker_pid']!=row['launcher_pid']
                or row['artifact']['sha256']!=recipe['artifact_sha256']
                or row['provenance']['cuda_allocator_fraction']!=.15
                or row['provenance']['forward_count']!=4
                or row['provenance']['cuda_initialized_after_import']
                or row['provenance']['cuda_initialized_after_construction']):
            raise ValueError('cold runtime/forward/artifact mismatch')
    elif [row['batch'] for row in rows]!=contract['protocol']['candidates']:
        raise ValueError('capacity candidate list mismatch')


def validate_prepared_result(result):
    scenario = result['scenario']
    raw = result['public_results']
    expected = (['P03','P04'] if scenario == PIPELINE_SCENARIO else
                ['P19','P20','P22','P25'] if scenario == TRAINING_SCENARIO else [PREPARED[scenario]['feature']])
    if (set(raw) != set(expected) or result['public_result_sha256'] != admission.policy.canonical_hash(raw)
            or result['executions'] != 1 or result['retries'] != 0 or result['baseline'] != 0):
        raise ValueError('public result identity/digest mismatch')
    for feature, data in raw.items():
        if (data['schema_version'] != 1 or data['feature_ids'] != [feature]
                or not data['format'].startswith('hyso.performance.')):
            raise ValueError('public feature/schema mismatch')
    if scenario == PIPELINE_SCENARIO:
        validate_pipeline_pair(raw, result['frame_evidence'])
    elif scenario == TRAINING_SCENARIO:
        timing = raw['P19']
        if any(raw[key]['source_segment_id'] != timing['segment_id'] or raw[key]['source_p19'] != timing
               for key in ('P20','P22','P25')) or result['frame_evidence']['fits'] != 1:
            raise ValueError('training projections must share one real fit')
    elif PREPARED[scenario]['kind'] in ('cold','capacity'):
        validate_nested_result(result)
    else:
        data = raw[expected[0]]
        if (data['protocol']['axis'] != PREPARED[scenario]['axis']
                or data['protocol']['configuration'] != PREPARED[scenario]['protocol']
                or data['requested']['points'] != len(PREPARED[scenario]['points'])):
            raise ValueError('sweep contract mismatch')


def run_prepared_scenario(torch, manifest, decision, root=None, work=None):
    scenario = manifest['scenario']
    contract = PREPARED[scenario]
    fn = measure_pipeline_pair if contract['kind']=='pipeline' else (
        measure_training_bundle if contract['kind']=='training' else measure_sweep_case)
    raw, frames = (measure_nested_scenario(torch,contract,root,work,manifest)
        if contract['kind'] in ('cold','capacity') else fn(torch, contract))
    partial = contract['kind']=='sweep' and any(row['status'] != 'measured'
        for row in raw[contract['feature']]['raw_evidence']['points'])
    if contract['kind']=='cold':partial=raw['P05']['actual']['measured']!=1
    if contract['kind']=='capacity':partial=raw['P17']['measurements']['status']!='complete'
    result = dict(scenario=scenario, status='partial' if partial else 'success',
        executions=1, retries=0, baseline=0, public_results=raw,
        public_result_sha256=admission.policy.canonical_hash(raw), frame_evidence=frames,
        gpu_context=decision, warnings=decision['warnings'],
        measurement_reliability=decision['measurement_reliability']['status'], fixture=contract)
    validate_prepared_result(result)
    return result


def validate_result(result, expected_scenario=None):
    if not isinstance(result, dict):
        raise ValueError('invalid result integrity')
    if result.get('scenario') in PREPARED and result.get('status') in ('success','partial'):
        if expected_scenario is not None and result['scenario'] != expected_scenario:
            raise ValueError('returned scenario mismatch')
        validate_prepared_result(result)
        return
    if result.get('status') == 'controlled_failure':
        if (result.get('reason') not in ('cuda_oom', 'timeout')
                or result.get('measurements', 'missing') is not None
                or result.get('retries') != 0 or result.get('cleanup_required') is not True):
            raise ValueError('invalid controlled failure result')
    elif result.get('status') == 'success':
        scenario = result.get('scenario')
        if (scenario not in SCENARIOS or (expected_scenario is not None and scenario != expected_scenario)
                or result.get('executions') != 1
                or result.get('retries') != 0 or result.get('baseline') != 0
                or not isinstance(result.get('throughput' if scenario == P02_SCENARIO else 'benchmark'), dict)):
            raise ValueError('invalid successful result contract')
        if scenario == P02_SCENARIO:
            validate_p02_payload(result['throughput'])
    else:
        raise ValueError('unrecognized workload result status')


def controlled_timeout(outer, child=None):
    """No measurement survives a deadline; cleanup proof remains mandatory."""
    receipts = [outer] + ([] if child is None else [child])
    if not any(receipt.get('outcome') == 'timeout' for receipt in receipts):
        return None
    for receipt in receipts:
        if (receipt['cleanup']['active_processes'] != 0
                or receipt.get('owned_leader_exited') is not True
                or receipt['assignment']['assigned'] is not True):
            raise ValueError('timeout without verified owned cleanup')
    return admission.controlled_failure('timeout')


def native_worker(root, work, manifest):
    try:
        return run_native_worker(root, work, manifest)
    except Exception as error:
        torch = sys.modules.get('torch')
        oom_type = getattr(getattr(torch, 'cuda', None), 'OutOfMemoryError', None)
        if oom_type is None or not isinstance(error, oom_type):
            raise
        # No retry, reduced batch, fallback or allocator budget escalation.
        return admission.controlled_failure('cuda_oom')


def run_native_worker(root, work, manifest):
    # Called only after child Job receipt, exact membership and full seal checks.
    capture, review = optional_context(Path(work)/'capture.json'), optional_context(Path(work)/'review.json')
    decision = admission.evaluate(capture, review, run_id=manifest['run_id'],
                                  manifest_sha256=os.environ['HYSO_CANDIDATE_SHA'])
    if not decision['eligible']:
        raise ValueError('invalid GPU policy execution contract')
    sys.path[:0] = [str(Path(root)/'source'), manifest['site_packages']]
    import torch
    from hyso.benchmark import BenchmarkRunner
    loaded_identity(root, manifest)
    # Age affects telemetry reliability only; no stale-capture workload gate.
    decision = admission.evaluate(capture, review, run_id=manifest['run_id'],
            manifest_sha256=os.environ['HYSO_CANDIDATE_SHA'])
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('native CUDA required')
    torch.cuda.set_per_process_memory_fraction(.15, 0)
    torch.manual_seed(71)
    if manifest['scenario'] in PREPARED:
        result = run_prepared_scenario(torch, manifest, decision, root=root, work=work)
        loaded_identity(root, manifest)
        return result
    if manifest['scenario'] == P02_SCENARIO:
        model = torch.nn.Conv2d(1, 3, 3, padding=1).to('cuda:0').eval()
        inputs = torch.randn(2, 1, 7, 9, device='cuda:0', dtype=torch.float32)
        data = measure_p02_completion(model, inputs)
        loaded_identity(root, manifest)
        return dict(scenario=P02_SCENARIO, executions=1, baseline=0, retries=0,
                    status='success', throughput=data, gpu_context=decision,
                    warnings=decision['warnings'],
                    measurement_reliability=decision['measurement_reliability']['status'])
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
    sensor_samples = [admission.sensor_metadata(s) for s in (before, after)]
    warnings = set(decision['warnings'])
    if any(s['status'] != 'measured' or any(v['status'] != 'measured'
           for v in s['value']['fields'].values()) for s in sensor_samples):
        warnings.add('gpu_sensor_unavailable_or_unsupported')
    loaded_identity(root, manifest)
    return dict(scenario=SCENARIO, executions=1, baseline=0, retries=0,
                status='success', benchmark=result, device_wide_sensor_samples=sensor_samples,
                gpu_context=decision, warnings=sorted(warnings),
                measurement_reliability='unqualified' if warnings else 'observed')


def internal(args):
    # -I -S omits our source directory: main restores only the sealed helper path.
    root, work = Path(args.root), Path(args.work)
    await_gate(work, 'worker' if args.internal=='nested' else args.internal, args.token)
    manifest = verify(root, os.environ['HYSO_CANDIDATE_SHA'])
    verify_bootstrap(manifest)
    if args.internal == 'nested':
        nested_bootstrap(root,work,manifest)
    elif args.internal == 'parent':
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
        elif args.mode == 'simulated-oom':
            publish(work/'result.json', admission.controlled_failure('cuda_oom', simulated=True))
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
                         scenario=manifest['scenario'], maximum_executions=1, explicit_user_approval=True):
            raise ValueError('native approval identity mismatch')
        work.mkdir(exist_ok=True)
        # Optional GPU context can qualify measurements, never bypass hard gates.
        decision = admission.evaluate(optional_context(work/'capture.json'), optional_context(work/'review.json'),
                      run_id=manifest['run_id'], manifest_sha256=seal['manifest_sha256'])
        if not decision['eligible']:
            raise ValueError('invalid GPU policy execution contract')
        publish(root/'execution-consumed.json', dict(run_id=manifest['run_id'], time_ns=time.time_ns()))
        publish(work/'native-authorization.json', value)
        publish(work/'gpu-context-admission.json', decision)
    else:
        work.mkdir()  # each dry run owns a fresh exclusive directory
    token = str(uuid.uuid4())
    publish(work/'controller-identity.json', dict(pid=os.getpid(), parent_pid=os.getppid(),
            manifest_sha256=seal['manifest_sha256'], run_id=manifest['run_id'],
            source_verified=True, runtime_verified=True, cuda_imported='torch' in sys.modules))
    receipt = supervise(root, work, 'parent', mode, token, seal['manifest_sha256'],
                         timeout=15 if mode == 'outer-timeout' else None)
    verify(root, seal['manifest_sha256'])
    if mode == 'native':
        child_path = work/'worker-receipt.json'
        child = read(child_path) if child_path.exists() else None
        timeout = controlled_timeout(receipt, child)
        if timeout is not None:
            publish(work/'workload-failure.json', timeout)
            receipt.update(workload_status='controlled_failure', failure_reason='timeout')
            return receipt
    if mode in ('native', 'success', 'simulated-oom'):
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
        result = read(work/'result.json')
        if mode in ('native', 'simulated-oom'):
            validate_result(result, expected_scenario=manifest['scenario'])
            receipt['workload_status'] = result['status']
            receipt['failure_reason'] = result.get('reason')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--r3', type=Path)
    parser.add_argument('--scenario', choices=SCENARIOS, help='Preparation only; execution is bound to the sealed manifest')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--mode', choices=['success', 'timeout', 'outer-timeout', 'log-overflow', 'simulated-oom', 'native'], default='success')
    parser.add_argument('--approval', type=Path)
    parser.add_argument('--internal', choices=['parent', 'worker', 'nested'])
    parser.add_argument('--token')
    args = parser.parse_args()
    if args.scenario is not None and not args.prepare:
        parser.error('--scenario is only valid with --prepare')
    if args.prepare:
        manifest = prepare(args.prepare, args.r3, scenario=args.scenario or SCENARIO)
        print(json.dumps(dict(prepared=True, run_id=manifest['run_id'], cuda_workload_started=False)))
    elif args.internal:
        internal(args)
    else:
        print(json.dumps(controller(args.root, args.work, args.mode, args.approval)))


if __name__ == '__main__':
    main()
