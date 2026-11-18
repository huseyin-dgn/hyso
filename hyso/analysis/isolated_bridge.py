"""G3 opt-in, Windows-only, Job-bounded trusted CPU factory bridge.

No model or factory is loaded in this controller process. This is NOT a
sandbox for adversarial Python, a generic checkpoint deserializer, or a
35-feature production acceptance. Only G2's native P02/P11 CPU producers run.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

from .feature_plan import FeaturePlanRequest, plan_features


class IsolatedBridgeError(ValueError):
    pass


_ALLOWED = frozenset(('P02', 'P11'))
_HEX = re.compile(r'[0-9a-f]{64}\Z')
_NAME = re.compile(r'[A-Za-z_][A-Za-z_0-9]*\Z')
_MAX_SOURCE = 512 * 1024
_MAX_RESULT = 4 * 1024 * 1024


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def _source_receipt(path):
    source = Path(path)
    if not source.is_absolute() or source.suffix != '.py' or source.is_symlink() or not source.is_file():
        raise IsolatedBridgeError('trusted_factory_must_be_regular_absolute_python_file')
    if any(parent.is_symlink() for parent in source.parents):
        raise IsolatedBridgeError('symlink_factory_ancestor_forbidden')
    with source.open('rb') as h:
        before = os.fstat(h.fileno())
        if before.st_size <= 0 or before.st_size > _MAX_SOURCE:
            raise IsolatedBridgeError('factory_source_size_out_of_range')
        content = h.read(_MAX_SOURCE + 1)
        after = os.fstat(h.fileno())
    latest = source.stat()
    ident = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    if ident(before) != ident(after) or ident(after) != ident(latest) or len(content) != after.st_size:
        raise IsolatedBridgeError('factory_source_changed_during_verification')
    return hashlib.sha256(content).hexdigest()


@dataclass(frozen=True)
class TrustedFactoryBinding:
    model_id: str
    source_file: str
    source_sha256: str
    factory_name: str = 'build'
    kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if type(self.model_id) is not str or not self.model_id or len(self.model_id) > 160:
            raise IsolatedBridgeError('invalid_model_id')
        if type(self.source_file) is not str or not Path(self.source_file).is_absolute():
            raise IsolatedBridgeError('absolute_factory_source_required')
        if type(self.source_sha256) is not str or not _HEX.fullmatch(self.source_sha256):
            raise IsolatedBridgeError('expected_factory_sha256_required')
        if type(self.factory_name) is not str or not _NAME.fullmatch(self.factory_name):
            raise IsolatedBridgeError('invalid_factory_name')
        if type(self.kwargs) is not dict or any(type(k) is not str or not _NAME.fullmatch(k) for k in self.kwargs):
            raise IsolatedBridgeError('finite_named_factory_kwargs_required')
        def valid(v, depth=0):
            if depth > 4: return False
            if type(v) in (bool, int, str) or v is None: return type(v) is not str or len(v) <= 1024
            if type(v) is float: return __import__('math').isfinite(v)
            if type(v) is list: return len(v) <= 64 and all(valid(x, depth+1) for x in v)
            return False
        if not all(valid(x) for x in self.kwargs.values()):
            raise IsolatedBridgeError('invalid_factory_kwargs')
        if len(_canonical(self.kwargs).encode('utf-8')) > 4096:
            raise IsolatedBridgeError('factory_kwargs_too_large')


@dataclass(frozen=True)
class IsolatedCPUExecutionPolicy:
    explicit_authorization: bool = False
    timeout_seconds: int = 45
    job_memory_limit_bytes: int = 2 * 1024**3
    warmup_iterations: int = 1
    batch_count: int = 2
    repeats: int = 1
    queue_depth: int = 2
    max_input_bytes: int = 8 * 1024**2

    def __post_init__(self):
        if type(self.explicit_authorization) is not bool:
            raise IsolatedBridgeError('explicit_authorization_must_be_boolean')
        for name, low, high in (
            ('timeout_seconds', 5, 90),
            ('job_memory_limit_bytes', 1024**3, 4*1024**3),
            ('warmup_iterations', 0, 5), ('batch_count', 1, 16),
            ('repeats', 1, 3), ('queue_depth', 2, 8),
            ('max_input_bytes', 1, 32*1024**2),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise IsolatedBridgeError('invalid_bounded_policy:' + name)


@dataclass(frozen=True)
class IsolatedCPUReport:
    status: str
    reason: str | None
    worker_receipt: dict | None
    native_bridge_report: dict | None
    requested_feature_ids: tuple
    authorized_feature_ids: tuple
    scope: str = 'trusted_factory_windows_hard_job;not_adversarial_code_sandbox'
    acceptance: str = 'NOT_EVALUATED'

    def to_dict(self):
        return json.loads(_canonical(asdict(self)))


def _check_request(request, bindings, ids, policy):
    if type(request) is not FeaturePlanRequest or type(policy) is not IsolatedCPUExecutionPolicy:
        raise IsolatedBridgeError('typed_G1_request_and_G3_policy_required')
    if not policy.explicit_authorization:
        raise IsolatedBridgeError('explicit_worker_execution_authorization_required')
    if type(bindings) not in (tuple, list) or len(bindings) != len(request.models) or any(
            type(b) is not TrustedFactoryBinding for b in bindings):
        raise IsolatedBridgeError('one_typed_factory_per_model_required')
    if tuple(b.model_id for b in bindings) != tuple(m.model_id for m in request.models):
        raise IsolatedBridgeError('factory_model_identity_mismatch')
    if any(not m.trusted_factory or m.backend != 'pytorch' or m.task != 'classification' for m in request.models):
        raise IsolatedBridgeError('trusted_pytorch_classification_factory_required')
    if type(ids) not in (tuple, list) or not ids or any(
            type(x) is not str or x not in _ALLOWED for x in ids) or len(ids) != len(set(ids)):
        raise IsolatedBridgeError('only_explicit_P02_P11_allowed')
    plan = plan_features(request)
    selected = tuple(x.feature_id for x in plan.features)
    if not set(ids).issubset(selected):
        raise IsolatedBridgeError('authorization_not_in_feature_plan')
    for row in plan.features:
        if row.feature_id in ids and any(m.status != 'PROPOSED' for m in row.models):
            raise IsolatedBridgeError('G1_prerequisites_not_satisfied:' + row.feature_id)
    for binding in bindings:
        if _source_receipt(binding.source_file) != binding.source_sha256:
            raise IsolatedBridgeError('factory_sha256_mismatch:' + binding.model_id)
    return selected


def _read_worker_output(path, selected, authorized, bindings, receipt):
    source = Path(path)
    if source.is_symlink() or not source.is_file() or source.stat().st_size > _MAX_RESULT:
        raise IsolatedBridgeError('missing_or_oversize_worker_terminal')
    def no_duplicates(pairs):
        data = {}
        for key, value in pairs:
            if key in data:
                raise IsolatedBridgeError('duplicate_worker_json_field')
            data[key] = value
        return data
    with source.open('r', encoding='utf-8') as h:
        payload = json.load(h, object_pairs_hook=no_duplicates,
                            parse_constant=lambda _: (_ for _ in ()).throw(IsolatedBridgeError('nonfinite_worker_json')))
    if type(payload) is not dict or set(payload) != {'status','reason','report'}:
        raise IsolatedBridgeError('invalid_worker_terminal_envelope')
    if payload['status'] != 'COMPLETED' or payload['reason'] is not None:
        raise IsolatedBridgeError('worker_reported_failure:' + str(payload['reason'])[:200])
    report = payload['report']
    if type(report) is not dict or report.get('acceptance') != 'NOT_EVALUATED' or report.get('authorization') != 'EXPLICIT_CPU_P02_P11_ONLY':
        raise IsolatedBridgeError('invalid_worker_native_report')
    if (report.get('requested_feature_ids') != list(selected) or
            report.get('executed_feature_ids') != list(authorized)):
        raise IsolatedBridgeError('worker_scope_mismatch')
    if report.get('scope') != 'trusted_inprocess_cpu;not_isolated_no_hard_runtime_memory_limit':
        raise IsolatedBridgeError('worker_G2_scope_mismatch')
    if type(report.get('comparisons')) is not list or type(report.get('omitted_features')) is not list:
        raise IsolatedBridgeError('missing_worker_comparison_or_omissions')
    expected_omitted = [dict(feature_id=f, status='NOT_EXECUTED',
                             reason='not_authorized' if f in _ALLOWED else 'native_adapter_not_implemented')
                        for f in selected if f not in authorized]
    if report['omitted_features'] != expected_omitted:
        raise IsolatedBridgeError('worker_omitted_features_mismatch')
    captures = report.get('captures')
    expected = {(b.model_id, f) for f in authorized for b in bindings}
    if type(captures) is not list or len(captures) != len(expected):
        raise IsolatedBridgeError('worker_capture_count_mismatch')
    for c in captures:
        if type(c) is not dict or (c.get('model_id'), c.get('feature_id')) not in expected:
            raise IsolatedBridgeError('worker_capture_identity_mismatch')
        if c.get('status') != 'MEASURED_NATIVE_SCOPE' or type(c.get('native_result')) is not dict:
            raise IsolatedBridgeError('worker_native_result_not_measured')
        content = _canonical(c['native_result'])
        if hashlib.sha256(content.encode('utf-8')).hexdigest() != c.get('source_sha256'):
            raise IsolatedBridgeError('worker_native_source_hash_mismatch')
        if c['native_result'].get('feature_ids') != [c['feature_id']]:
            raise IsolatedBridgeError('worker_native_feature_mismatch')
        expected.remove((c['model_id'],c['feature_id']))
    if expected:
        raise IsolatedBridgeError('missing_worker_capture')
    from .native_bridge import CapturedNativeResult, _observed_pair
    model_ids = [b.model_id for b in bindings]
    native = {(c['model_id'], c['feature_id']): CapturedNativeResult(
        c['model_id'], c['feature_id'], c['status'], c['source_sha256'],
        _canonical(c['native_result'])) for c in captures}
    derived = []
    if len(model_ids) == 2:
        for feature in authorized:
            from dataclasses import asdict
            derived.append(asdict(_observed_pair(feature, native[(model_ids[0], feature)],
                                                       native[(model_ids[1], feature)])))
    if report['comparisons'] != derived:
        raise IsolatedBridgeError('worker_comparison_not_derived_from_captures')
    return report


def _windows_available():
    return os.name == 'nt'


def execute_isolated_cpu(request, bindings, *, authorized_features, policy=IsolatedCPUExecutionPolicy()):
    """Explicit, fail-closed Windows-only execution; no automatic in-process fallback.

    Parent validates factory file hashes WITHOUT importing or running factory.
    Child is created suspended and assigned a strict one-process Job prior to
    first Python instruction, with job/process commit limits and termination on
    timeout. Distinct worker-created model objects; no original model objects.
    """
    selected = _check_request(request, bindings, authorized_features, policy)
    if not _windows_available():
        raise IsolatedBridgeError('windows_job_backend_required_no_inprocess_fallback')
    from ._windows_job import run_bounded_windows, JobLaunchError
    from dataclasses import asdict as _asdict
    authorized = tuple(authorized_features)
    with tempfile.TemporaryDirectory(prefix='hyso-g3-') as folder:
        root = Path(folder)
        req = dict(request=dict(mode=request.mode, models=[_asdict(m) for m in request.models],
                                features=list(request.features), goals=list(request.goals),
                                capabilities=list(request.capabilities)),
                   bindings=[_asdict(b) for b in bindings],
                   authorized_features=list(authorized),
                   policy=_asdict(policy), selected=list(selected))
        serialized = _canonical(req)
        config = root / 'request.json'
        output = root / 'terminal.json'
        config.write_text(serialized, encoding='utf-8')
        worker = Path(__file__).with_name('_isolated_worker.py').resolve(strict=True)
        try:
            receipt = run_bounded_windows([sys.executable, '-I', '-B', str(worker), str(config), str(output)], root,
                                          timeout_seconds=policy.timeout_seconds,
                                          job_memory_limit_bytes=policy.job_memory_limit_bytes)
            if receipt['status'] != 'process_exited' or receipt['exit_code'] != 0:
                return IsolatedCPUReport('FAILED', receipt['status'] if receipt['status'] != 'process_exited'
                                         else 'worker_nonzero_exit', receipt, None, selected, authorized)
            if not receipt['assigned_before_resume'] or receipt['active_owned_processes'] != 0:
                raise IsolatedBridgeError('worker_ownership_not_confirmed')
            report = _read_worker_output(output, selected, authorized, bindings, receipt)
            return IsolatedCPUReport('COMPLETED', None, receipt, report, selected, authorized)
        except Exception as exc:
            return IsolatedCPUReport('FAILED', type(exc).__name__ + ':' + str(exc)[:200],
                                     None, None, selected, authorized)
