"""Read-only capture and evidence binding around the unchanged r3 evaluator.

No name allowlist, automatic desktop decisions, process termination or workload.
"""
import copy
import importlib.util
import json
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


def facts(inventory, pid):
    processes = {p['pid']: p for p in inventory['processes']}
    process = processes.get(pid)
    return dict(process=process,
                parent=None if process is None else processes.get(process['parent_pid']),
                nvml_rows=[s for s in inventory['raw_query']
                           if int(s.split(',', 1)[0].strip()) == pid])


def evaluate(capture, review, *, run_id, manifest_sha256, now_ns=None):
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
    value = copy.deepcopy(capture)
    value.update(policy_id=policy.POLICY_ID, run_id=run_id,
                 manifest_sha256=manifest_sha256,
                 workload_review=dict(status='complete', reviewer=review['reviewer'],
                     inventory_sha256=review['inventory_sha256'], gpu_processes=rows,
                     conflicting_workloads=review['conflicting_workloads']))
    return policy.evaluate(value, run_id=run_id, manifest_sha256=manifest_sha256,
                           now_ns=now_ns)


def collect():
    """One capture, exactly three samples, no polling for an eligible state."""
    start = time.time_ns()
    command = '''$ErrorActionPreference='Stop'; $p=@(Get-CimInstance Win32_Process | ForEach-Object {
      @{pid=[int]$_.ProcessId;name=$_.Name;parent_pid=[int]$_.ParentProcessId;
        path=$_.ExecutablePath;command_line=$_.CommandLine;creation_date=[string]$_.CreationDate}
    }); $o=Get-CimInstance Win32_OperatingSystem;
    @{processes=$p;host_free_bytes=[long]$o.FreePhysicalMemory*1024} | ConvertTo-Json -Depth 6 -Compress'''
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                            capture_output=True, text=True, timeout=12,
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
