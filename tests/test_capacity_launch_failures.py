"""P17 launch-error containment; no subprocess, model forward or CUDA workload."""
import copy
import errno
import json
from pathlib import Path

import pytest
import torch

from hyso.performance import BatchSearchProtocol, CapacityResult, ColdStartRecipe
from hyso.performance import capacity, cold


@pytest.fixture(autouse=True)
def no_native_work(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('native workload/cleanup must not run in launch-failure tests')
    for name in ('init', '_lazy_init', 'is_available', 'device_count', 'current_device'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(torch.nn.Module, '_call_impl', forbidden)
    monkeypatch.setattr(capacity, '_terminate', forbidden)
    token = cold._owned_worker_launcher.set(None)
    yield
    cold._owned_worker_launcher.reset(token)
    assert not torch.cuda.is_initialized()


def plan():
    return BatchSearchProtocol(candidates=(1, 2, 3), maximum_batch=3,
        cpu_memory_limit_bytes=1024**3, input_limit_bytes=1024**2,
        timeout_seconds=30, warmup_iterations=2, measure_iterations=10, repeats=2)


def launch_error(number):
    if number == 'errno_enomem':
        return OSError(errno.ENOMEM, 'controlled process-start resource failure')
    error = OSError(errno.EACCES if number == 5 else errno.ENOMEM,
                    'controlled process-start failure')
    error.winerror = number
    return error


@pytest.mark.parametrize('number,status', [
    (8, 'resource_limit'), (14, 'resource_limit'), (1450, 'resource_limit'),
    (1455, 'resource_limit'), ('errno_enomem', 'resource_limit'),
    (5, 'failed'), (9999, 'failed')])
def test_launch_failure_returns_unknown_without_oom_or_worker(monkeypatch, tmp_path, number, status):
    streams = []
    def fail(*args, **kwargs):
        streams.extend((kwargs['stdout'], kwargs['stderr']))
        raise launch_error(number)
    monkeypatch.setattr(capacity.subprocess, 'Popen', fail)
    recipe = ColdStartRecipe('torch.nn:Identity').to_dict()
    row = capacity._candidate(recipe, plan(), 2, tmp_path)
    assert row['status'] == status and row['classification'] == 'unknown'
    assert row['worker_pid'] is row['launcher_pid'] is row['exit_code'] is None
    assert row['process_terminated'] and not row['forward_completed']
    assert row['actual_batch'] is row['latency_mean_ms'] is row['sweep'] is None
    assert row['execution_seconds'] >= 0 and 'launch' in row['reason']
    assert all(stream.closed for stream in streams)


def test_previous_success_survives_launch_resource_failure(monkeypatch, tmp_path):
    # Reuse an existing serialized PrecisionResult solely as an opaque success
    # receipt. It is not a new measurement and no model is constructed/run.
    payloads = json.loads((Path(__file__).parent / 'fixtures/advisor-native-v1.json').read_text(encoding='utf-8'))['payloads']
    point = copy.deepcopy(payloads['P08']['raw_evidence']['points'][0]['result'])
    recipe = ColdStartRecipe('hyso.models.modular:ModularCNN', input_shape=(1, 1, 28, 28))
    protocol = plan()
    original = capacity._candidate
    attempts = []
    launches = []
    def fail(*args, **kwargs):
        launches.append(args)
        raise launch_error(1455)
    monkeypatch.setattr(capacity.subprocess, 'Popen', fail)
    def candidate(config, settings, batch, root):
        attempts.append(batch)
        if batch != 1:
            return original(config, settings, batch, root)
        row = capacity._row(batch, config)
        row.update(status='feasible', classification='feasible', actual_batch=1,
            forward_completed=True, latency_mean_ms=point['measurements']['latency_mean_ms']['value'],
            memory=dict(private_commit_limit_bytes=settings.cpu_memory_limit_bytes,
                        input_limit_bytes=settings.input_limit_bytes),
            sweep=copy.deepcopy(point), worker_pid=12341, launcher_pid=12341, exit_code=0,
            worker_phase='runtime_prepared', runtime_ready=True, execution_seconds=.01,
            provenance=dict(device='cpu'))
        return row
    monkeypatch.setattr(capacity, '_candidate', candidate)
    result = capacity.find_max_batch_size(recipe, protocol=protocol)
    data = result.to_dict()
    rows = data['raw_evidence']['candidates']
    assert attempts == [1, 2] and len(launches) == 1
    assert [r['status'] for r in rows] == ['feasible', 'resource_limit', 'not_run']
    assert rows[0]['sweep'] == point
    assert data['measurements']['status'] == 'aborted'
    assert data['measurements']['maximum_observed_feasible_batch'] is None
    assert data['measurements']['prior_observed_feasible_batch'] == 1
    assert data['measurements']['maximum_proven_within_range'] is None
    result.to_json(tmp_path / 'result.json')
    result.to_csv(tmp_path / 'result.csv')
    assert CapacityResult(json.loads((tmp_path / 'result.json').read_text())).to_dict() == data


@pytest.mark.parametrize('exit_code', [0xC0000005, -1073741819])
def test_access_violation_overrides_stale_success_receipt(monkeypatch, tmp_path, exit_code):
    class ExitedWorker:
        pid = 12342
        returncode = exit_code
        def __init__(self, command, **kwargs):
            # Deliberately incomplete success publication must be ignored when
            # the OS process outcome is a native failure. No process is started.
            Path(command[5]).write_text(json.dumps(dict(worker_pid=self.pid, status='feasible')))
            Path(command[6]).write_text(json.dumps(dict(pid=self.pid, phase='runtime_prepared', runtime_ready=True)))
        def poll(self):
            return self.returncode
        def wait(self, timeout):
            return self.returncode
    monkeypatch.setattr(capacity.subprocess, 'Popen', ExitedWorker)
    row = capacity._candidate(ColdStartRecipe('torch.nn:Identity').to_dict(), plan(), 2, tmp_path)
    assert row['status'] == 'crashed' and row['classification'] == 'unknown'
    assert row['exit_code'] == exit_code and row['worker_pid'] == 12342
    assert row['process_terminated'] and not row['forward_completed']
    assert row['actual_batch'] is row['latency_mean_ms'] is row['sweep'] is None
