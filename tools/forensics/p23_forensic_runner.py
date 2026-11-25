"""One-shot P23 forensic harness. Native execution needs new exact-hash consent.

Trusted diagnostic code, not a hostile-code sandbox. Production gates are never
changed. CPU safety cases cannot select CUDA; native mode cannot select retries.
"""
import argparse
import ctypes as c
from ctypes import wintypes as w
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_scenario_runner as safety
import native_admission as admission
import p23_forensic_windows as windows

LIMITS = admission.policy.LIMITS
SCENARIO = 'p23-tiny-saved-payload-forensic-v1'
CPU_MODES = ('success', 'cpu-recipe', 'exception', 'failfast', 'timeout', 'memory', 'tree', 'logs',
             'collector-timeout', 'collector-limit', 'outer-timeout')
digest, read = safety.digest, safety.read


def publish(path, value):
    """Complete exclusive artifact before making its name visible to readers."""
    path = Path(path)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    safety.publish(temporary, value)
    try:
        if path.exists(): raise FileExistsError(str(path))
        temporary.rename(path)  # Windows rename refuses an existing destination
    finally:
        if temporary.exists(): temporary.unlink()


def transform(source, target):
    """Only three explicit device gates in PRIVATE generated diagnostic copies.

    No detach/copy/sync/ownership or version-check change is proposed as a fix.
    Original guarded files remain present and hash-pinned in the snapshot.
    """
    text = source.read_text(encoding='utf-8')
    if target == 'recipe':
        guard = ('    if protocol.saved_activation_lifetime and device.type=="cuda":\n'
                 '        raise NotImplementedError("CUDA saved-activation adapter disabled: P23 R1 native-crash gate unresolved (0xC0000409)")\n')
        if text.count(guard) != 1:
            raise ValueError('recipe guard source drift')
        text = text.replace(guard, '    # Explicitly authorized PRIVATE P23 diagnostic copy only.\n')
        old = 'from ._saved_activation import '
        if text.count(old) != 1:
            raise ValueError('recorder import source drift')
        text = text.replace(old, 'from ._p23_diagnostic_saved import ')
    else:
        guards = [
            ('        if any(t.device.type != "cpu" for t in (*model.parameters(), *model.buffers())):\n'
             '            raise NotImplementedError("saved-activation recorder is CPU-only; CUDA 0xC0000409 gate unresolved")\n'),
            ('        if tensor.device.type != "cpu":\n'
             '            raise NotImplementedError("saved-activation recorder is CPU-only; CUDA 0xC0000409 gate unresolved")\n')]
        for guard in guards:
            if text.count(guard) != 1:
                raise ValueError('saved recorder guard source drift')
            text = text.replace(guard, '        # Explicitly authorized PRIVATE P23 diagnostic copy only.\n')
    compile(text, str(source), 'exec')
    return text


def prepare(destination, design):
    repo, destination = Path(__file__).resolve().parents[2], Path(destination)
    design = Path(design)
    if digest(design) != 'd4c7a5da1068de8b7ff56a7b81330b47d63a64adb119d58896084e1a84ff15ea':
        raise ValueError('design identity changed')
    previous = read(design)
    destination.mkdir()  # never overwrite or reuse any prior run
    source = destination/'source'
    source.mkdir()
    files = list((repo/'hyso').rglob('*.py'))
    files += [Path(__file__).with_name(name) for name in (
        'p23_forensic_runner.py', 'p23_forensic_worker.py', 'p23_forensic_windows.py',
        'native_scenario_runner.py', 'native_admission.py', 'p23_preflight.py')]
    for path in files:
        relative = path.relative_to(repo) if path.is_relative_to(repo/'hyso') else Path(path.name)
        target = source/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    transforms = []
    for original, generated, kind in (
        ('isolated_training_memory.py', '_p23_diagnostic_recipe.py', 'recipe'),
        ('_saved_activation.py', '_p23_diagnostic_saved.py', 'saved')):
        original, generated = source/'hyso/performance'/original, source/'hyso/performance'/generated
        generated.write_text(transform(original, kind), encoding='utf-8', newline='\n')
        transforms.append(dict(original=str(original.relative_to(source)), original_sha256=digest(original),
                               generated=str(generated.relative_to(source)), generated_sha256=digest(generated)))
    runtime = {row['path']: row['sha256'] for row in previous['runtime_files'].values()}
    site = Path(sys.executable).parent/'Lib/site-packages'
    for package in ('torch', 'numpy'):
        for path in (site/package).rglob('*'):
            if path.is_file() and path.suffix.lower() in ('.py', '.pyd', '.dll'):
                observed = digest(path)
                if str(path) in runtime and runtime[str(path)] != observed:
                    raise ValueError('historical runtime drift: '+str(path))
                runtime[str(path)] = observed
    dbghelp = Path(os.environ['SystemRoot'])/'System32/dbghelp.dll'
    runtime[str(dbghelp)] = digest(dbghelp)
    # Materialization is CPU-only; no training or CUDA context, checkpoint or model.
    result = subprocess.run([sys.executable, '-I', '-S', '-B', str(source/'p23_forensic_worker.py'),
                             '--materialize', str(site)], capture_output=True, text=True,
                            timeout=30, env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1'),
                            creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode or len(result.stdout) > 65536 or len(result.stderr) > 65536:
        raise RuntimeError('CPU input materialization failed: '+result.stderr[:1000])
    inputs = json.loads(result.stdout)
    if inputs['cuda_initialized'] or inputs['torch'] != '2.5.1+cu121':
        raise ValueError('input runtime contract mismatch')
    for path, sha in previous['runtime_files'].items():
        if digest(sha['path']) != sha['sha256']:
            raise ValueError('runtime drift relative to historical identities: '+path)
    hashes = {p.relative_to(source).as_posix(): digest(p) for p in source.rglob('*.py')}
    manifest = dict(format='hyso.p23.executable_forensic.v1', run_id=str(uuid.uuid4()),
        scenario=SCENARIO, authorization=dict(approved=False, separate_exact_hash_required=True),
        interpreter=sys.executable, site_packages=str(site), dbghelp=str(dbghelp),
        source_files=hashes, runtime_files=runtime, input_contract=inputs,
        limits=LIMITS, admission_flow=admission.P23_FLOW,
        dump_max_bytes=4*1024**3, collector_timeout_s=20,
        diagnostic_copies=transforms, production_gate='DISABLED_UNCHANGED',
        native_executions=1, retries=0, baselines=0, fixture=dict(seed=71, epochs=2,
            optimizer='adamw', amp=False, stream='default', reset=True, samples=5,
            batch=2, accumulation=2, model='Linear(3,4)-Dropout(.2)-Linear(4,2)'),
        interpretation='Diagnostic only. Not P23 acceptance or a root-cause/fix claim.',
        dump_scope='Full host memory of this worker only; local/private; no upload or global WER edits')
    publish(destination/'manifest.json', manifest)
    publish(destination/'seal.json', dict(manifest_sha256=digest(destination/'manifest.json')))
    for path in source.rglob('*.py'):
        path.chmod(0o444)
    return manifest


def verify(root, sha, runtime=False):
    root = Path(root)
    if digest(root/'manifest.json') != sha:
        raise ValueError('manifest identity mismatch')
    m = read(root/'manifest.json')
    if (m['scenario'] != SCENARIO or m['format'] != 'hyso.p23.executable_forensic.v1'
            or m['limits'] != LIMITS or m['native_executions'] != 1 or m['retries'] != 0
            or m['baselines'] != 0 or m['production_gate'] != 'DISABLED_UNCHANGED'
            or m.get('admission_flow') != admission.P23_FLOW
            or m['dump_max_bytes'] != 4*1024**3 or m['collector_timeout_s'] != 20
            or Path(m['interpreter']).resolve() != Path(sys.executable).resolve()):
        raise ValueError('forensic policy contract mismatch')
    for relative, expected in m['source_files'].items():
        path = root/'source'/relative
        if path.is_symlink() or not path.resolve().is_relative_to((root/'source').resolve()) or digest(path) != expected:
            raise ValueError('source drift: '+relative)
    # Verify bootstrap actually running from this seal, not just a sealed copy.
    for module in (sys.modules[__name__], safety, admission, admission.policy, windows):
        if digest(module.__file__) != m['source_files'][Path(module.__file__).name]:
            raise ValueError('loaded bootstrap drift')
    if runtime:
        for path, expected in m['runtime_files'].items():
            if digest(path) != expected:
                raise ValueError('runtime drift: '+path)
    else:
        # Stdlib-only safety cases need no GPU library hashing or imports.
        for path in (m['interpreter'], m['dbghelp']):
            if digest(path) != m['runtime_files'][path]: raise ValueError('helper runtime drift')
    return m


def gate(work, role, token):
    # Existing exact Job membership, hard-limit readback, nonce and PID/parent checks.
    safety.await_gate(work, role, token)


def pipes(process):
    logs, counts, threads = {}, {}, []
    overflow = threading.Event()
    def drain(pipe, name):
        logs[name], counts[name] = bytearray(), 0
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
    for name in ('stdout', 'stderr'):
        t = threading.Thread(target=drain, args=(getattr(process, name), name), daemon=True)
        t.start()
        threads.append(t)
    return logs, counts, threads, overflow


def spawn(root, work, role, mode, token, sha, job, debug=False):
    process = subprocess.Popen([sys.executable, '-I', '-S', '-B', str(Path(root)/'source/p23_forensic_runner.py'),
        '--internal', role, '--root', str(root), '--work', str(work), '--mode', mode, '--token', token],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW | (2 if debug else 0),
        env=dict(os.environ, HYSO_CANDIDATE_SHA=sha, CUDA_VISIBLE_DEVICES='0' if mode == 'native' else '-1',
                 PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8'))
    try:
        job.assign(process)
        receipt = job.receipt(process, safety.expected_limits('outer' if role == 'parent' else 'child'))
        receipt['token'] = token
        output = pipes(process)
        publish(Path(work)/(role+'-admit.json'), receipt)
        return process, output
    except BaseException:
        process.kill()  # only this Popen handle if assignment/gating failed
        process.stdout.close()
        process.stderr.close()
        raise


def finish(process, output, job, *, debug=False):
    job.terminate()
    end = time.monotonic()+5
    if debug:
        while time.monotonic() < end:
            event = windows.next_event(process.pid)
            if event:
                exited = event.kind == 5
                windows.close_event_handles(event)
                windows.resume(event, True)
                if exited:
                    break
            elif process.poll() is not None:
                break
    process.wait(timeout=5)
    while job.observation()['active_processes'] and time.monotonic() < end:
        time.sleep(.01)
    for thread in output[2]:
        thread.join(timeout=2)
    cleanup = job.observation()
    if cleanup['active_processes'] or any(t.is_alive() for t in output[2]):
        raise RuntimeError('owned tree/pipe cleanup unverified')
    return cleanup


def collect_dump(root, work, token, sha, worker, job, row, mode, remaining):
    if worker.pid not in job.process_ids() or worker.poll() is not None:
        raise ValueError('dump requires live owned debuggee')
    process, output = spawn(root, work, 'collector', mode, token, sha, job)
    try:
        request = dict(pid=worker.pid, nonce=token, exception=row,
            process_handle=windows.duplicate_to(process, worker._handle),
            job_handle=windows.duplicate_to(process, job.handle), manifest_sha256=sha)
        publish(Path(work)/'dump-request.json', request)
        start = time.monotonic()
        outcome = None
        dump_budget = 1024**2 if mode == 'collector-limit' else 4*1024**3
        timeout = min(2 if mode == 'collector-timeout' else 20, remaining)
        while process.poll() is None:
            if time.monotonic()-start >= timeout:
                outcome = 'collector_timeout'; break
            path = Path(work)/'owned-worker.dmp'
            if path.exists() and path.stat().st_size > dump_budget:
                outcome = 'collector_dump_limit'; break
            if output[3].is_set():
                outcome = 'collector_log_limit'; break
            time.sleep(.01)
        if process.poll() is None:
            process.kill()
        process.wait(timeout=3)
        for t in output[2]: t.join(timeout=2)
        path = Path(work)/'owned-worker.dmp'
        if path.exists() and path.stat().st_size > dump_budget:
            outcome = 'collector_dump_limit'
        result = dict(outcome=outcome, exit_code=process.returncode, pid=process.pid,
                      elapsed_s=time.monotonic()-start, logs={k:bytes(v).decode(errors='replace') for k,v in output[0].items()})
        if outcome is None and process.returncode == 0:
            checked = read(Path(work)/'dump-result.json')
            if checked['nonce'] != token or checked['manifest_sha256'] != sha or checked['pid'] != worker.pid:
                raise ValueError('collector result binding mismatch')
            result['dump'] = checked
        else:
            result['inconclusive_stop'] = True
        publish(Path(work)/'collector-receipt.json', result)
        return result
    finally:
        if process.poll() is None:
            process.kill(); process.wait(timeout=3)


def supervise(root, work, mode, token, sha, outer=False):
    kind, role = ('outer', 'parent') if outer else ('child', 'worker')
    Job = safety.job_class(root)
    job = Job(LIMITS[kind+'_job_private_commit'], LIMITS[kind+'_processes'])
    start, process, output = time.monotonic(), None, None
    budget = LIMITS[kind+'_timeout_s']
    if mode == 'timeout' and not outer: budget = 4
    if mode == 'outer-timeout' and outer: budget = 4
    reason, fault, collector, exited = None, None, None, False
    try:
        if job.readback() != safety.expected_limits(kind):
            raise ValueError('Job quota readback mismatch')
        process, output = spawn(root, work, role, mode, token, sha, job, debug=not outer)
        while True:
            if outer:
                if process.poll() is not None: break
            else:
                event = windows.next_event(process.pid)
                if event:
                    resumed = False
                    try:
                        if event.kind == 1:
                            code = event.data.exception.record.code
                            fatal = (code in (0xc0000409, 0xc0000005, 0xc0000602, 0xc000001d, 0xc00000fd)
                                     or not event.data.exception.first_chance)
                            if fatal and fault is None:
                                fault = windows.exception_row(event)
                                try:
                                    publish(Path(work)/'native-exception.json', dict(nonce=token, manifest_sha256=sha, **fault))
                                    collector = collect_dump(root, work, token, sha, process, job, fault,
                                                             mode, max(.01, budget-(time.monotonic()-start)))
                                finally:
                                    # Finish only the owned Job while the fatal event is
                                    # debug-stopped. Passing it to WER would rotate the
                                    # shared CrashDumps folder and displace prior evidence.
                                    job.terminate()
                                    windows.resume(event, True)
                                    resumed = True
                            else:
                                windows.resume(event, code == 0x80000003 or fatal)
                                resumed = True
                        else:
                            exited = event.kind == 5
                            windows.close_event_handles(event)
                            windows.resume(event, True)
                            resumed = True
                    except BaseException:
                        # Never pass an unhandled fatal exception to system WER.
                        job.terminate()
                        if not resumed:
                            try: windows.resume(event, True)
                            except OSError: pass
                        raise
                    if exited: break
            if time.monotonic()-start >= budget:
                reason = 'timeout'; break
            if output[3].is_set():
                reason = 'log_limit'; break
            artifacts = list(Path(work).glob('*.json')) + list(Path(work).glob('*.jsonl'))
            if len(artifacts) > 32 or any(p.stat().st_size > (1024**2 if p.suffix == '.jsonl' else LIMITS['result_bytes']) for p in artifacts):
                reason = 'artifact_limit'; break
            time.sleep(.005)
    finally:
        try:
            cleanup = finish(process, output, job, debug=not outer and not exited) if process else job.observation()
        finally:
            job.close()
    if output[3].is_set(): reason = 'log_limit'
    receipt = dict(pid=process.pid, parent_pid=os.getpid(), nonce=token, manifest_sha256=sha,
        exit_code=process.returncode, monitor_outcome=reason, fault=fault, collector=collector,
        cleanup=cleanup, kill_on_close=True, limits=safety.expected_limits(kind),
        elapsed_s=time.monotonic()-start, log_bytes=output[1],
        logs={k:bytes(v).decode(errors='replace') for k,v in output[0].items()})
    publish(Path(work)/(role+'-receipt.json'), receipt)
    return receipt


def approval(root, work, m, sha):
    expected = dict(run_id=m['run_id'], manifest_sha256=sha, scenario=SCENARIO,
                    maximum_executions=1, explicit_user_approval=True)
    if read(Path(work)/'authorization.json') != expected:
        raise ValueError('new exact-hash explicit native consent required')


def validate_dump_request(request, worker_admission, token, sha):
    if (request['nonce'] != token or request['manifest_sha256'] != sha
            or worker_admission['token'] != token or request['pid'] != worker_admission['pid']
            or request['exception']['pid'] != request['pid']):
        raise ValueError('collector worker/nonce/seal binding mismatch')


def controller(root, work, mode, sha):
    root, work = Path(root), Path(work)
    if work.resolve().parent != root.resolve() or work.is_symlink():
        raise ValueError('work directory must be a direct, unlinked child of this seal')
    m = verify(root, sha, runtime=mode in ('native', 'cpu-recipe'))
    if mode == 'native':
        approval(root, work, m, sha)
        capture, review = read(work/'capture.json'), read(work/'review.json')
        decision = admission.evaluate_bound(capture, review, run_id=m['run_id'],
            manifest_sha256=sha, stage='controller-preflight',
            nonce=capture['binding']['nonce'], worker_pid=0)
        publish(work/'admission-controller.json', decision)
        if not decision['eligible']: raise ValueError('GPU admission NO-GO')
        publish(root/'execution-consumed.json', dict(time_ns=time.time_ns(), manifest_sha256=sha))
    else:
        if mode not in CPU_MODES: raise ValueError('unrecognized offline case')
        work.mkdir()
    token = str(uuid.uuid4())
    publish(work/'controller-identity.json', dict(pid=os.getpid(), nonce=token, manifest_sha256=sha,
        cuda_imported='torch' in sys.modules, source_verified=True, runtime_verified=True))
    result, error = None, None
    try:
        result = supervise(root, work, mode, token, sha, outer=True)
    except BaseException as exc:
        error = type(exc).__name__+': '+str(exc)
        raise
    finally:
        preserved = True
        try: verify(root, sha, runtime=mode in ('native', 'cpu-recipe'))
        except BaseException: preserved = False
        child = read(work/'worker-receipt.json') if (work/'worker-receipt.json').exists() else None
        completed = (error is None and preserved and result is not None and result['exit_code'] == 0
            and result['monitor_outcome'] is None and child is not None and child['exit_code'] == 0
            and child['monitor_outcome'] is None and child['fault'] is None and (work/'result.json').is_file())
        publish(work/'controller-terminal.json', dict(manifest_sha256=sha, nonce=token,
            active=False, outcome='DIAGNOSTIC_COMPLETE' if completed else 'STOP', error=error,
            preservation='PASS' if preserved else 'FAIL', supervisor=result,
            diagnostic_only=True, p23_acceptance='PARTIAL', cuda_workload_authorized=mode == 'native'))
    return result


def internal(args):
    root, work = Path(args.root), Path(args.work)
    gate(work, args.internal, args.token)
    sha = os.environ['HYSO_CANDIDATE_SHA']
    m = verify(root, sha)
    if args.mode == 'native': approval(root, work, m, sha)
    if args.internal == 'parent':
        supervise(root, work, args.mode, args.token, sha)
    elif args.internal == 'collector':
        request = read(work/'dump-request.json') if (work/'dump-request.json').exists() else None
        deadline = time.monotonic()+5
        while request is None:
            if time.monotonic() > deadline: raise TimeoutError('missing dump request')
            time.sleep(.01)
            if (work/'dump-request.json').exists(): request = read(work/'dump-request.json')
        validate_dump_request(request, read(work/'worker-admit.json'), args.token, sha)
        if args.mode == 'collector-timeout': time.sleep(120)
        if args.mode == 'collector-limit':
            # A STRICTER CPU-only cap exercises overflow without a 4-GiB file.
            with (work/'owned-worker.dmp').open('xb') as stream: stream.truncate(1024**2+1)
            return
        value = windows.write_owned_dump(request, work/'owned-worker.dmp', m['dbghelp'])
        publish(work/'dump-result.json', dict(pid=request['pid'], nonce=args.token,
                    manifest_sha256=sha, sha256=digest(work/'owned-worker.dmp'), **value))
    else:
        import p23_forensic_worker as worker
        worker.run(root, work, args.mode, args.token, sha, m)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--design', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--sha')
    parser.add_argument('--mode', choices=(*CPU_MODES, 'native'), default='success')
    parser.add_argument('--internal', choices=('parent', 'worker', 'collector'))
    parser.add_argument('--token')
    args = parser.parse_args()
    if args.prepare:
        m = prepare(args.prepare, args.design)
        print(json.dumps(dict(run_id=m['run_id'], manifest_sha256=digest(args.prepare/'manifest.json'), cuda_started=False)))
    elif args.internal: internal(args)
    else: print(json.dumps(controller(args.root, args.work, args.mode, args.sha)))


if __name__ == '__main__': main()
