"""G3 child entrypoint. Never run directly on untrusted user-supplied inputs.

The parent must assign an OS Job and hard limits before resuming this process.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

# In isolated Python mode, the executable script's directory does not appear in
# sys.path. Only the trusted, checked-out HYSo root is added here.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _require_bounded_job():
    """Refuse direct worker invocation without the owner's configured hard limits."""
    if os.name != 'nt':
        raise RuntimeError('G3_worker_requires_Windows_Job')
    import ctypes as C
    from ctypes import wintypes as W
    # Do not import the hyso package before the admission check: its __init__
    # may import Torch. Load the stdlib-only structure definition by file path.
    import importlib.util
    path = Path(__file__).with_name('_windows_job.py')
    spec = importlib.util.spec_from_file_location('_g3_windows_job_limits', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _EXTENDED_LIMIT = module._EXTENDED_LIMIT
    k = C.WinDLL('kernel32', use_last_error=True)
    k.GetCurrentProcess.argtypes = []
    k.GetCurrentProcess.restype = W.HANDLE
    k.IsProcessInJob.argtypes = [W.HANDLE, W.HANDLE, C.POINTER(W.BOOL)]
    k.IsProcessInJob.restype = W.BOOL
    k.QueryInformationJobObject.argtypes = [W.HANDLE, C.c_int, C.c_void_p,
                                             W.DWORD, C.POINTER(W.DWORD)]
    k.QueryInformationJobObject.restype = W.BOOL
    belonging = W.BOOL()
    if not k.IsProcessInJob(k.GetCurrentProcess(), None, C.byref(belonging)) or not belonging.value:
        raise RuntimeError('G3_unbounded_worker_forbidden')
    limit = _EXTENDED_LIMIT()
    size = W.DWORD()
    if not k.QueryInformationJobObject(None, 9, C.byref(limit), C.sizeof(limit), C.byref(size)):
        raise RuntimeError('G3_job_limit_query_failed')
    required = 0x2000 | 0x100 | 0x200 | 0x8
    flags = limit.BasicLimitInformation.LimitFlags
    if flags & required != required or limit.BasicLimitInformation.ActiveProcessLimit != 1:
        raise RuntimeError('G3_insufficient_job_constraints')
    if not 1024**3 <= limit.JobMemoryLimit <= 4 * 1024**3 or limit.ProcessMemoryLimit != limit.JobMemoryLimit:
        raise RuntimeError('G3_job_memory_limit_not_bounded')


# Verify job membership *before* importing HYSo/Torch or user factory code.
_require_bounded_job()

from hyso.analysis.feature_plan import FeaturePlanRequest, ModelContext, plan_features
from hyso.analysis.native_bridge import (LiveCPUModel, TrustedCPUExecutionPolicy,
                                         execute_scoped_cpu)
from hyso.analysis.isolated_bridge import (TrustedFactoryBinding, IsolatedCPUExecutionPolicy,
                                          _canonical, _source_receipt, _read_worker_output)


def main(config_path, output_path):
    req = json.loads(Path(config_path).read_text(encoding='utf-8'))
    if type(req) is not dict or set(req) != {'request','bindings','authorized_features','policy','selected'}:
        raise ValueError('invalid_G3_worker_request')
    r = req['request']
    contexts = tuple(ModelContext(**m) for m in r['models'])
    request = FeaturePlanRequest(r['mode'], contexts, features=r['features'],
                                 goals=r['goals'], capabilities=r['capabilities'])
    policy = IsolatedCPUExecutionPolicy(**req['policy'])
    ids = tuple(req['authorized_features'])
    if not policy.explicit_authorization or not ids or any(x not in ('P02','P11') for x in ids):
        raise PermissionError('worker_not_explicitly_authorized')
    plan = plan_features(request)
    if [f.feature_id for f in plan.features] != req['selected']:
        raise ValueError('worker_replanned_scope_mismatch')
    bindings = tuple(TrustedFactoryBinding(**b) for b in req['bindings'])
    if tuple(b.model_id for b in bindings) != tuple(m.model_id for m in request.models):
        raise ValueError('worker_factory_binding_mismatch')
    constructed = []
    for index, binding in enumerate(bindings):
        if _source_receipt(binding.source_file) != binding.source_sha256:
            raise ValueError('factory_sha256_mismatch_before_import')
        spec = importlib.util.spec_from_file_location('hyso_g3_user_factory_' + str(index), binding.source_file)
        if spec is None or spec.loader is None:
            raise ValueError('factory_module_unavailable')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if _source_receipt(binding.source_file) != binding.source_sha256:
            raise ValueError('factory_source_changed_during_import')
        factory = getattr(module, binding.factory_name, None)
        if not callable(factory):
            raise ValueError('requested_factory_not_callable')
        produced = factory(**binding.kwargs)
        if type(produced) not in (tuple, list) or len(produced) != 2:
            raise TypeError('factory_must_return_model_and_input_pair')
        model, tensor = produced
        constructed.append(LiveCPUModel(binding.model_id, model, tensor))
    bridge_policy = TrustedCPUExecutionPolicy(
        explicit_trusted_inprocess_consent=True,
        warmup_iterations=policy.warmup_iterations,
        batch_count=policy.batch_count, repeats=policy.repeats,
        queue_depth=policy.queue_depth, max_input_bytes=policy.max_input_bytes)
    result = execute_scoped_cpu(plan, tuple(constructed), authorized_features=ids,
                                policy=bridge_policy)
    return result.to_dict()


def _publish(path, payload):
    target = Path(path)
    source = target.with_name(target.name + '.partial')
    with source.open('x', encoding='utf-8') as h:
        h.write(_canonical(payload))
        h.flush(); os.fsync(h.fileno())
    os.replace(source, target)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(2)
    try:
        report = main(sys.argv[1], sys.argv[2])
    except Exception as exc:
        # Do not put the native result in the terminal on a failed run.
        try: _publish(sys.argv[2], dict(status='FAILED', reason=type(exc).__name__, report=None))
        except Exception: pass
        raise SystemExit(1)
    _publish(sys.argv[2], dict(status='COMPLETED', reason=None, report=report))
