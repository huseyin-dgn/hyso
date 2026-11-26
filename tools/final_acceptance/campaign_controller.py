"""Reuse HYSo's Windows Job; keep all work behind its pre-import admission latch."""
from __future__ import annotations

import ctypes as c
import importlib.util
import os
import subprocess
import time
from pathlib import Path

from common import REPO, ROOT, atomic_json, digest, file_hash, new_json, read_json, read_journal, utc, tree_bytes
from campaign_preflight import process_identity, resources


def runtime_job(config):
    spec = importlib.util.spec_from_file_location('final_owned_job', REPO / 'hyso/performance/_training_process.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    job = module._Job(config['process_memory_bytes'], config['job_memory_bytes'], config['max_processes'])
    actual = job.Extended()
    if not job.api.QueryInformationJobObject(job.handle, 9, c.byref(actual), c.sizeof(actual), None):
        job.close()
        raise OSError('Job limits cannot be read back')
    if not (actual.process_memory == config['process_memory_bytes']
            and actual.job_memory == config['job_memory_bytes']
            and actual.basic.processes == config['max_processes']
            and actual.basic.flags & 0x2308 == 0x2308
            and not actual.basic.flags & 0x1800):
        job.close()
        raise RuntimeError('Job limit/kill-on-close readback mismatch')
    return job


def admit_interpreter(job, launcher, ready, nonce):
    """Bind a Windows venv redirector to the real interpreter before imports."""
    if ready['token'] != nonce:
        raise ValueError('bootstrap nonce mismatch')
    pid = ready['pid']
    api = job.api
    api.OpenProcess.argtypes = [c.c_ulong, c.c_int, c.c_ulong]
    api.OpenProcess.restype = c.c_void_p
    api.IsProcessInJob.argtypes = [c.c_void_p, c.c_void_p, c.POINTER(c.c_int)]
    api.IsProcessInJob.restype = c.c_int
    handle = api.OpenProcess(0x100000 | 0x1000 | 0x400 | 0x100 | 1, False, pid)
    if not handle:
        raise OSError('cannot open admitted interpreter')
    try:
        class Basic(c.Structure):
            _fields_ = [('exit_status', c.c_long), ('peb', c.c_void_p), ('affinity', c.c_size_t),
                        ('priority', c.c_long), ('pid', c.c_size_t), ('parent_pid', c.c_size_t)]
        native = c.WinDLL('ntdll')
        native.NtQueryInformationProcess.argtypes = [c.c_void_p, c.c_ulong, c.c_void_p, c.c_ulong, c.c_void_p]
        native.NtQueryInformationProcess.restype = c.c_long
        basic = Basic()
        if native.NtQueryInformationProcess(handle, 0, c.byref(basic), c.sizeof(basic), None) != 0:
            raise OSError('OS parent identity unavailable')
        if not (basic.pid == pid and basic.parent_pid == ready['parent_pid'] and
                (pid == launcher.pid or basic.parent_pid == launcher.pid)):
            raise ValueError('interpreter is not the exact launcher/direct descendant')
        member = c.c_int()
        if not api.IsProcessInJob(handle, job.handle, c.byref(member)):
            raise OSError('exact Job membership unavailable')
        if not member.value:
            job.assign(type('OwnedHandle', (), {'_handle': handle})())
        if not api.IsProcessInJob(handle, job.handle, c.byref(member)) or not member.value:
            raise RuntimeError('interpreter outside exact Job')
        return dict(launcher=process_identity(launcher.pid), interpreter=process_identity(pid),
                    parent_pid=basic.parent_pid, nonce_sha256=digest(nonce), exact_job_membership=True)
    finally:
        api.CloseHandle(handle)


def exact_job_member(job, handle):
    api = job.api
    api.IsProcessInJob.argtypes = [c.c_void_p, c.c_void_p, c.POINTER(c.c_int)]
    api.IsProcessInJob.restype = c.c_int
    member = c.c_int()
    if not api.IsProcessInJob(handle, job.handle, c.byref(member)):
        raise OSError('exact Job membership unavailable')
    return bool(member.value)


def resume_owned_launcher(job, process):
    """Popen closes its primary thread handle; resume only our suspended Job member."""
    if not exact_job_member(job, process._handle):
        raise RuntimeError('refusing to resume a launcher outside the exact Job')
    native = c.WinDLL('ntdll')
    native.NtResumeProcess.argtypes = [c.c_void_p]
    native.NtResumeProcess.restype = c.c_long
    status = native.NtResumeProcess(process._handle)
    if status != 0:
        raise OSError('NtResumeProcess failed: NTSTATUS 0x%08x' % (status & 0xffffffff))


def verified_cleanup(cleanup, binding, *, launcher_wait_verified, cleanup_error):
    """Proof of termination is independent of successful Python/payload admission.

    Suspended creation closes the redirector race: no descendant can predate the
    launcher's exact Job assignment. All later descendants inherit the same
    no-breakaway Job, even if Python dies before the stdlib latch.
    """
    identity = (binding or {}).get('identity') or {}
    return bool(cleanup_error is None and cleanup
        and type(cleanup.get('active_processes')) is int and cleanup['active_processes'] == 0
        and launcher_wait_verified and binding
        and binding.get('created_suspended') is True
        and binding.get('exact_job_membership_before_resume') is True
        and binding.get('limits_readback') is True and binding.get('no_breakaway') is True
        and type(identity.get('pid')) is int and identity['pid'] > 0
        and identity['pid'] == binding.get('launcher_pid')
        and type(identity.get('creation_filetime')) is int and identity['creation_filetime'] > 0
        and bool(identity.get('image')))


def safe_environment(readonly, work):
    allowed = {'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'PATH', 'PATHEXT', 'COMSPEC',
        'LOCALAPPDATA', 'APPDATA', 'USERPROFILE', 'HOMEDRIVE', 'HOMEPATH',
        'NUMBER_OF_PROCESSORS', 'PROCESSOR_ARCHITECTURE', 'PROGRAMFILES',
        'PROGRAMFILES(X86)', 'PROGRAMDATA', 'ALLUSERSPROFILE'}
    env = {k.upper(): v for k, v in os.environ.items() if k.upper() in allowed}
    if not env.get('SYSTEMROOT'):
        raise RuntimeError('ENVIRONMENT_ERROR: Windows Python startup requires SYSTEMROOT')
    env.update(CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8',
        PYTHONUNBUFFERED='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', MPLBACKEND='Agg',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK='1')
    input_keys = {'HYSO_STAGE10_SAVED_ROOT', 'HYSO_P35_CAPTURE_ROOT', 'HYSO_P34_NATIVE_RESULT',
                  'HYSO_P34_EVIDENCE', 'HYSO_OPTIONAL_ORT_PATH', 'P23_BASIC_ASSETS'}
    if set(readonly) - input_keys:
        raise ValueError('read-only fixture mapping cannot override the execution environment')
    env.update(readonly)
    env.pop('HYSO_NATIVE_GPU_UUID', None)
    env.pop('HYSO_ORT_CUDA_AVAILABLE', None)
    env['P23_TEST_OUTPUT'] = str(work / 'forensic-cpu')
    basic = work / 'p23-basic'
    basic.mkdir(exist_ok=True)
    env['P23_BASIC_OUTPUT'] = str(basic)
    return env


def execute(attempt, request, config, runtime, *, update, stop_requested, integrity):
    """Exactly one attempt. Resource continuation is decided by the scheduler."""
    attempt = Path(attempt)
    work = Path(request['work'])
    work.mkdir(parents=True, exist_ok=False)
    new_json(work / '.hyso-owned-work.json', dict(attempt_id=request['attempt_id'],
             campaign_id=request['campaign_id']))
    # Preserve the existing bootstrap; copy once into this unique attempt.
    bootstrap = attempt / 'bootstrap.py'
    bootstrap.write_bytes((REPO / 'outputs/release-acceptance-20261125/bootstrap.py').read_bytes())
    request = dict(request, runtime_paths=runtime['runtime_paths'])
    new_json(attempt / 'request.json', request)
    job = runtime_job(config)
    started = time.monotonic()
    last_update, last_integrity, last_event_change = started, started, started
    log, journal = attempt / 'stdout.log', attempt / 'events.jsonl'
    reason, error, process, identities, cleanup = None, None, None, None, None
    prior_size = 0
    source_ok = True
    launcher_assigned = False
    launch_binding = None
    launcher_wait_verified = False
    payload_admitted = False
    environment = {}
    minimum_commit = None
    growth_check = started
    peak_growth = 0
    try:
        with log.open('xb') as stream:
            environment = safe_environment(read_json(REPO / 'outputs/release-acceptance-20261125/readonly-test-environment.json', {}), work)
            environment.update(HYSO_FINAL_ATTEMPT_ID=request['attempt_id'], HYSO_FINAL_COMMIT=request['source_commit'], HYSO_FINAL_CAMPAIGN_ID=request['campaign_id'])
            if request.get('package_sha256'):
                environment['HYSO_PACKAGE_SHA256'] = request['package_sha256']
            argv = [runtime['python'], '-I', '-S', '-B', str(bootstrap), str(attempt / 'request.json')]
            if request['mode'] == 'startup-exit-before-latch':
                argv = [runtime['python'], '-I', '-S', '-B', '-c', 'raise SystemExit(73)']
            process = subprocess.Popen(argv, cwd=REPO, env=environment,
                stdout=stream, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW | 0x00000004)  # CREATE_SUSPENDED
            job.assign(process)
            launcher_assigned = True
            launch_binding = dict(identity=process_identity(process.pid), launcher_pid=process.pid, created_suspended=True,
                exact_job_membership_before_resume=exact_job_member(job, process._handle),
                limits_readback=True, no_breakaway=True)
            if not launch_binding['identity'] or not launch_binding['exact_job_membership_before_resume']:
                raise RuntimeError('OWNERSHIP_UNVERIFIED: suspended launcher binding failed')
            new_json(attempt/'launch-receipt.json',dict(schema_version=2,attempt_id=request['attempt_id'],
                owner=process_identity(os.getpid()),source_commit=request['source_commit'],
                launch_binding=launch_binding,kill_on_close=True))
            resume_owned_launcher(job,process)
            deadline = time.monotonic() + 20
            while not (attempt / 'interpreter-ready.json').exists():
                if process.poll() is not None:
                    reason = 'BOOTSTRAP_EXIT_BEFORE_ADMISSION'
                    raise RuntimeError('Python exited before the stdlib admission latch; inspect stdout.log')
                if time.monotonic() > deadline:
                    reason = 'BOOTSTRAP_ADMISSION_TIMEOUT'
                    raise RuntimeError('bootstrap admission latch timeout')
                if stop_requested():
                    raise InterruptedError('user requested stop before admission')
                time.sleep(.02)
            identities = admit_interpreter(job, process, read_json(attempt / 'interpreter-ready.json'), request['token'])
            new_json(attempt / 'owner-identities.json', identities)
            new_json(attempt / 'admission-receipt.json', dict(owner=process_identity(os.getpid()),
                identities=identities, kill_on_close=True, exact_membership=True,
                limits_readback=True, no_breakaway=True, source_commit=request['source_commit']))
            atomic_json(attempt / 'admit.json', dict(pid=identities['interpreter']['pid'],
                token=request['token'], readback_verified=True))
            payload_admitted = True
            while process.poll() is None:
                now = time.monotonic()
                live = resources()
                minimum_commit = min(minimum_commit or live['available_commit_bytes'], live['available_commit_bytes'])
                size = journal.stat().st_size if journal.exists() else 0
                if size != prior_size:
                    last_event_change, prior_size = now, size
                if stop_requested():
                    reason = 'USER_STOP'
                elif live['available_commit_bytes'] < config['stop_available_commit_bytes']:
                    reason = 'BLOCKED_RESOURCE'
                elif live['disk_free_bytes'] < config['minimum_running_disk_bytes']:
                    reason = 'DISK_RESERVE'
                elif log.stat().st_size > config['max_log_bytes_per_attempt'] or size > config['max_journal_bytes_per_attempt']:
                    reason = 'EVIDENCE_BUDGET'
                elif now - started > request['timeout_seconds']:
                    reason = 'GROUP_TIMEOUT'
                elif now - last_event_change > request.get('no_progress_timeout_seconds', request['timeout_seconds']):
                    reason = 'NO_PROGRESS_TIMEOUT'
                if now - last_integrity >= config['source_integrity_interval_seconds']:
                    source_ok = integrity()
                    last_integrity = now
                    if not source_ok:
                        reason = 'EVIDENCE_INVALID'
                if now - growth_check >= config['source_integrity_interval_seconds']:
                    growth = tree_bytes(request['scratch_root']) + tree_bytes(ROOT) - request['initial_evidence_bytes']
                    peak_growth = max(peak_growth, growth)
                    if growth > config['max_workspace_growth_bytes']:
                        reason = 'WORKSPACE_BUDGET'
                    growth_check = now
                if now - last_update >= config['live_progress_interval_seconds']:
                    update(attempt, live, job.observation())
                    last_update = now
                if reason:
                    break
                time.sleep(.25)
    except BaseException as exc:
        import traceback
        error = dict(type=type(exc).__name__, message=str(exc),traceback=traceback.format_exc())
        reason = 'USER_STOP' if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else reason or 'CONTROLLER_ERROR'
    finally:
        cleanup_error = None
        try:
            if process is not None and process.poll() is None and identities is None and (attempt/'interpreter-ready.json').is_file():
                # A venv redirector can spawn before its parent is assigned. Bind the
                # exact latched direct child for cleanup without releasing admission.
                try:
                    identities=admit_interpreter(job,process,read_json(attempt/'interpreter-ready.json'),request['token'])
                except BaseException as exc:
                    cleanup_error=dict(type=type(exc).__name__,message=str(exc),stage='latched_child_cleanup_binding')
            job.terminate()
            if process is not None:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                launcher_wait_verified = process.returncode is not None
            deadline = time.monotonic() + 15
            while True:
                cleanup = job.observation()
                if cleanup['active_processes'] == 0 or time.monotonic() >= deadline:
                    break
                time.sleep(.05)
        except BaseException as exc:
            cleanup_error = dict(type=type(exc).__name__, message=str(exc))
            try:
                cleanup = job.observation()
            except OSError:
                cleanup = None
        finally:
            job.close()
    try:
        source_ok = source_ok and integrity()
    except BaseException as exc:
        source_ok = False
        error = dict(type=type(exc).__name__, message=str(exc))
    terminal_hash=None
    journal_sha=None
    if journal.is_file():
        # This controller is outside the test interpreter and seals the final bytes
        # only after all owned writers have exited. Never edits the event stream.
        with journal.open('rb+') as final_stream:
            os.fsync(final_stream.fileno())
        journal_sha=file_hash(journal)
        final_records,final_issues=read_journal(journal)
        new_json(attempt/'terminal-evidence.json',dict(schema_version=2,
            attempt_id=request['attempt_id'],request_sha256=file_hash(attempt/'request.json'),
            journal_sha256=journal_sha,verified_prefix_events=len(final_records),
            verified_prefix_head=final_records[-1]['sha256'] if final_records else '0'*64,
            journal_issues=final_issues,cleanup=cleanup,source_integrity=source_ok))
        terminal_hash=file_hash(attempt/'terminal-evidence.json')
    receipt = dict(schema_version=3, attempt_id=request['attempt_id'], group=request['group'],
        started_utc=request['started_utc'], finished_utc=utc(), identities=identities,
        monitor_reason=reason, error=error, exit_code=process.returncode if process else None,
        elapsed_seconds=time.monotonic() - started, source_integrity=source_ok,
        minimum_available_commit_bytes=minimum_commit, limits=dict(process_bytes=config['process_memory_bytes'],
        job_bytes=config['job_memory_bytes'], max_processes=config['max_processes']),
        limit_readback_verified=True, assigned_before_imports=payload_admitted, payload_admitted=payload_admitted,
        launch_binding=launch_binding, launcher_assigned=launcher_assigned,
        launcher_wait_verified=launcher_wait_verified,
        child_environment=dict(systemroot_present=bool(environment.get('SYSTEMROOT')),
            required_key_policy='CASE_INSENSITIVE_WINDOWS_ALLOWLIST'),
        kill_on_close=True, cleanup=cleanup, cleanup_verified=verified_cleanup(cleanup,launch_binding,
            launcher_wait_verified=launcher_wait_verified,cleanup_error=cleanup_error),
        bootstrap_sha256=file_hash(bootstrap), request_sha256=file_hash(attempt / 'request.json'),
        cleanup_error=cleanup_error, peak_workspace_growth_bytes=peak_growth, retries_inside_attempt=0,
        journal_sha256=journal_sha,terminal_evidence_sha256=terminal_hash)
    receipt['worker_terminal_sha256']=file_hash(attempt/'journal-terminal.json') if (attempt/'journal-terminal.json').is_file() else None
    new_json(attempt / 'controller.json', receipt)
    update(attempt, resources(), cleanup or {})
    if not receipt['cleanup_verified']:
        raise RuntimeError('OWNED_CLEANUP_UNVERIFIED; stop entire campaign')
    return receipt
