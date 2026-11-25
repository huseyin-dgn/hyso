"""Read-only capture and evidence binding around the unchanged r3 evaluator.

No name allowlist, automatic desktop decisions, process termination or workload.
"""
import copy
import importlib.util
import json
import re
import shutil
import subprocess
import time
from pathlib import Path


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy = load(Path(__file__).with_name('p23_preflight.py'), 'recovery_policy')

# P23's two-stage protocol is separate from the unchanged legacy evaluator.
P23_FLOW = 'hyso.p23.post-import-admission.v1'


def capture_binding(*, run_id, manifest_sha256, stage, nonce, worker_pid):
    policy._text(run_id, 'run_id')
    policy._text(nonce, 'capture nonce')
    if (type(manifest_sha256) is not str or len(manifest_sha256) != 64
            or any(ch not in '0123456789abcdef' for ch in manifest_sha256)):
        raise ValueError('invalid capture manifest SHA256')
    if (stage not in ('controller-preflight', 'worker-final')
            or type(worker_pid) is not int
            or (worker_pid != 0 if stage == 'controller-preflight' else worker_pid <= 0)):
        raise ValueError('invalid capture stage/worker PID')
    return dict(flow=P23_FLOW, run_id=run_id, manifest_sha256=manifest_sha256,
                stage=stage, nonce=nonce, worker_pid=worker_pid)


def bind_capture(raw, **context):
    """Bind a newly collected object once; never relabel old bound evidence."""
    binding = capture_binding(**context)
    if type(raw) is not dict or 'binding' in raw:
        raise ValueError('new unbound capture required')
    for key in ('run_id', 'manifest_sha256'):
        if key in raw and raw[key] != binding[key]:
            raise ValueError('capture identity mismatch: '+key)
    value = copy.deepcopy(raw)
    value.update(binding=binding, run_id=binding['run_id'],
                 manifest_sha256=binding['manifest_sha256'])
    return value


def validate_binding(capture, review, **context):
    expected = capture_binding(**context)
    if (capture.get('binding') != expected or review.get('capture_binding') != expected
            or capture.get('run_id') != expected['run_id']
            or capture.get('manifest_sha256') != expected['manifest_sha256']):
        raise ValueError('capture/review run/manifest/stage/nonce/PID mismatch')


def evaluate_bound(capture, review, *, run_id, manifest_sha256, stage, nonce,
                   worker_pid, now_ns=None):
    validate_binding(capture, review, run_id=run_id, manifest_sha256=manifest_sha256,
                     stage=stage, nonce=nonce, worker_pid=worker_pid)
    inventory = capture['inventory']
    table_pids = inventory['gpu_table_pids']
    if (inventory['gpu_table_query_succeeded'] is not True
            or type(table_pids) is not list or len(table_pids) > 4096
            or any(type(pid) is not int or pid <= 0 for pid in table_pids)
            or len(set(table_pids)) != len(table_pids)
            or sorted(table_pids) != sorted(gpu_table_pids(inventory['gpu_table_raw']))
            or not set(table_pids).issubset(inventory['compute_pids'])):
        raise ValueError('GPU C/G process table incomplete or contains unreviewed PIDs')
    return evaluate(capture, review, run_id=run_id, manifest_sha256=manifest_sha256,
                    now_ns=now_ns)


def refreshed_review(initial, reviewed, fresh, *, run_id, manifest_sha256,
                     nonce, worker_pid):
    """Transfer a workload judgment only for an exactly unchanged GPU identity.

    The initial review belongs to this new run/manifest, not to an authorization.
    Its historical timestamp is never used for the final admission decision.
    New, disappeared or changed GPU PIDs stop the single planned refresh.
    """
    validate_binding(initial, reviewed, run_id=run_id, manifest_sha256=manifest_sha256,
                     stage='controller-preflight', nonce=initial['binding']['nonce'], worker_pid=0)
    expected = capture_binding(run_id=run_id, manifest_sha256=manifest_sha256,
                               stage='worker-final', nonce=nonce, worker_pid=worker_pid)
    if fresh.get('binding') != expected:
        raise ValueError('final capture identity mismatch')
    # Validate the retained workload judgment, without changing any timestamp or
    # treating old telemetry as fresh. Controller/worker admitted it before import.
    validate_review(initial, reviewed)
    if (reviewed['conflicting_workloads']
            or any(row['category'] != 'desktop' for row in reviewed['gpu_processes'])):
        raise ValueError('initial workload review was not eligible')
    before, after = initial['inventory'], fresh['inventory']
    if sorted(before['compute_pids']) != sorted(after['compute_pids']):
        raise ValueError('GPU process set changed; no inherited review')
    for row in reviewed['gpu_processes']:
        if facts(before, row['pid']) != facts(after, row['pid']):
            raise ValueError('GPU process identity changed; no inherited review')
    result = copy.deepcopy(reviewed)
    result.update(capture_binding=expected, inventory_sha256=policy.canonical_hash(after),
                  initial_capture_sha256=policy.canonical_hash(initial),
                  initial_review_sha256=policy.canonical_hash(reviewed))
    return result


def facts(inventory, pid):
    processes = {p['pid']: p for p in inventory['processes']}
    process = processes.get(pid)
    return dict(process=process,
                parent=None if process is None else processes.get(process['parent_pid']),
                nvml_rows=[s for s in inventory['raw_query']
                           if int(s.split(',', 1)[0].strip()) == pid])


def validate_review(capture, review):
    """Operator decisions must bind to exactly one current identity per PID."""
    inventory = capture['inventory']
    rows = review['gpu_processes']
    if sorted(r['pid'] for r in rows) != sorted(inventory['compute_pids']):
        raise ValueError('incomplete or duplicate classification')
    if review['inventory_sha256'] != policy.canonical_hash(inventory):
        raise ValueError('inventory identity mismatch')
    for row in rows:
        identity = facts(inventory, row['pid'])
        if row['identity_sha256'] != policy.canonical_hash(identity):
            raise ValueError('PID identity mismatch')
        if identity['process'] is None or not identity['nvml_rows']:
            raise ValueError('missing CIM/NVML identity')
        restricted = [key for key in ('path', 'command_line')
                      if not identity['process'].get(key)]
        if identity['parent'] is None:
            restricted.append('parent')
        if row.get('restricted_fields') != restricted:
            raise ValueError('restricted Windows identity fields must be explicit')


def evaluate(capture, review, *, run_id, manifest_sha256, now_ns=None):
    validate_review(capture, review)
    value = copy.deepcopy(capture)
    value.update(policy_id=policy.POLICY_ID, run_id=run_id,
                 manifest_sha256=manifest_sha256,
                 workload_review=dict(status='complete', reviewer=review['reviewer'],
                    inventory_sha256=review['inventory_sha256'], gpu_processes=review['gpu_processes'],
                     conflicting_workloads=review['conflicting_workloads']))
    return policy.evaluate(value, run_id=run_id, manifest_sha256=manifest_sha256,
                           now_ns=now_ns)


def collect():
    """One capture, exactly three samples, no polling for an eligible state."""
    start = time.time_ns()
    command = '''[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); $ErrorActionPreference='Stop'; $p=@(Get-CimInstance Win32_Process | ForEach-Object {
      @{pid=[int]$_.ProcessId;name=$_.Name;parent_pid=[int]$_.ParentProcessId;
        path=$_.ExecutablePath;command_line=$_.CommandLine;creation_date=[string]$_.CreationDate}
    }); $o=Get-CimInstance Win32_OperatingSystem;
    @{processes=$p;host_free_bytes=[long]$o.FreePhysicalMemory*1024} | ConvertTo-Json -Depth 6 -Compress'''
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                            capture_output=True, text=True, encoding='utf-8', timeout=12,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError('CIM capture failed: '+result.stderr[:512])
    host = json.loads(result.stdout)
    tool = shutil.which('nvidia-smi')
    if not tool:
        raise RuntimeError('installed NVML CLI unavailable')

    def query(arg):
        p = subprocess.run([tool, arg, '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=4,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        if p.returncode:
            raise RuntimeError('NVML query failed/native crash: '+str(p.returncode))
        if len(p.stdout.encode()) > 1024**2 or len(p.stderr.encode()) > 65536:
            raise ValueError('sensor capture size exceeded')
        return p.stdout.strip().splitlines()

    raw = query('--query-compute-apps=pid,process_name,used_gpu_memory')
    inventory = dict(process_query_succeeded=True, compute_query_succeeded=True,
                     processes=host['processes'], raw_query=raw,
                     compute_pids=[int(row.split(',', 1)[0]) for row in raw])
    samples, gpu_raw = [], []
    for _ in range(3):
        rows = query('--query-gpu=uuid,utilization.gpu,memory.free')
        if len(rows) != 1:
            raise ValueError('exactly one GPU required')
        uuid, busy, free = [s.strip() for s in rows[0].split(',')]
        samples.append(dict(time_ns=time.time_ns(), uuid=uuid, free_mib=int(free),
                            utilization_percent=None if busy in ('N/A', '[N/A]') else int(busy)))
        gpu_raw.extend(rows)
    return dict(collection_start_time_ns=start, checked_time_ns=time.time_ns(),
                host_free_bytes=host['host_free_bytes'],
                disk_free_bytes=shutil.disk_usage(Path(__file__).anchor).free,
                gpu_samples=samples, gpu_raw=gpu_raw, inventory=inventory)


def gpu_table_pids(raw):
    """Fail closed on an unrecognized installed CLI table, including pure G rows."""
    if type(raw) is not str or len(raw.encode('utf-8')) > 1024**2:
        raise ValueError('bounded GPU C/G table required')
    if '| Processes:' not in raw:
        raise ValueError('GPU process table header missing')
    section = raw.split('| Processes:', 1)[1]
    pids = []
    header = False
    empty = False
    for line in section.splitlines()[1:]:
        text = line.strip()
        if not text or text.startswith('+'):
            continue
        if 'No running processes found' in text:
            empty = True
            continue
        if 'PID' in text and 'Type' in text:
            header = True
            continue
        if text.startswith('|') and (not text.strip('| ').strip()
                or 'Usage' in text or '===' in text):
            continue
        match = re.fullmatch(r'\|\s*\d+\s+\S+\s+\S+\s+(\d+)\s+(C\+G|C|G)\s+.+\|', text)
        if not match:
            raise ValueError('unrecognized GPU process table row')
        pids.append(int(match[1]))
    if (not header or (empty and pids) or (not pids and not empty)
            or len(pids) > 4096 or len(pids) != len(set(pids))):
        raise ValueError('incomplete or duplicate GPU process table')
    return pids


def collect_p23():
    """One planned capture; preserve full C/G table as well as compute rows."""
    capture = collect()
    result = subprocess.run([shutil.which('nvidia-smi') or 'nvidia-smi'],
        capture_output=True, text=True, timeout=4,
        creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode or len(result.stderr.encode()) > 65536:
        raise RuntimeError('GPU C/G table query failed')
    capture['inventory'].update(gpu_table_query_succeeded=True,
        gpu_table_pids=gpu_table_pids(result.stdout), gpu_table_raw=result.stdout)
    capture['checked_time_ns'] = time.time_ns()  # never reset capture start
    return capture
