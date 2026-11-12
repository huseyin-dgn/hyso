"""Selection -> metadata preflight -> explicit authorization -> native CPU adapters.

This layer does not implement measurements, train, export models, grant GPU access
or certify acceptance. Trusted in-process execution has no hard timeout/RAM cap.
The existing isolated route retains Windows Job limits without fallback.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
import re
from typing import Any, Sequence

from ._codec import canonical, digest, export
from .catalog import EXECUTOR_FEATURES, PROFILES, feature_catalog
from .feature_plan import FeaturePlanRequest, ModelContext, plan_features
from .native_bridge import LiveCPUModel, TrustedCPUExecutionPolicy, execute_scoped_cpu
from .isolated_bridge import (TrustedFactoryBinding, IsolatedCPUExecutionPolicy,
                              execute_isolated_cpu, _source_receipt)
from .evidence import EvidenceRecord, EvidenceIndex, EvidenceScope
from .report import AnalysisContext, UnifiedAnalysisReport
from ._analysis_resources import memory_availability


STATUSES = ('SUPPORTED','UNSUPPORTED','MISSING_PREREQUISITE','NOT_IMPLEMENTED_IN_EXECUTOR',
            'COMPLETED','FAILED','SKIPPED')


class AnalysisExecutionError(ValueError):
    """Invalid selection, policy, identity, authorization or reused dry-run ticket."""


@dataclass(frozen=True)
class CPUAnalysisPolicy:
    """Explicit small-workload bounds; not a hard in-process resource sandbox.

    P01 uses warmup/latency_measure_iterations/repeats; P02 uses
    warmup/batch_count/repeats/queue_depth. P10 takes exactly one OS snapshot.
    P11 reads logical tensors only. Tail eligibility uses the native default
    TailPolicy unchanged. Availability thresholds are OS estimates, not reserved
    memory or an estimate of arbitrary forward activation usage.
    """
    warmup_iterations: int = 1
    latency_measure_iterations: int = 1000
    batch_count: int = 2
    repeats: int = 1
    queue_depth: int = 2
    quantiles: tuple[float, ...] = (.5,.95,.99)
    max_input_bytes: int = 8*1024**2
    max_model_bytes: int = 256*1024**2
    min_available_memory_bytes: int = 128*1024**2
    min_available_commit_bytes: int = 256*1024**2

    def __post_init__(self) -> None:
        # Validate shared counts using the unchanged legacy policy contract.
        TrustedCPUExecutionPolicy(warmup_iterations=self.warmup_iterations,
            batch_count=self.batch_count,repeats=self.repeats,queue_depth=self.queue_depth,
            max_input_bytes=self.max_input_bytes)
        for name,low,high in (('latency_measure_iterations',1,4096),('max_model_bytes',1,1024**3),
                ('min_available_memory_bytes',1,2**63-1),('min_available_commit_bytes',1,2**63-1)):
            value = getattr(self,name)
            if type(value) is not int or not low <= value <= high:
                raise AnalysisExecutionError('invalid_policy:'+name)
        if (type(self.quantiles) not in (tuple,list) or not self.quantiles
                or any(type(q) not in (int,float) or not 0 < q < 1 for q in self.quantiles)
                or len(set(self.quantiles)) != len(self.quantiles)):
            raise AnalysisExecutionError('invalid_quantiles')
        object.__setattr__(self,'quantiles',tuple(self.quantiles))


@dataclass(frozen=True)
class AnalysisPlan:
    """Pure selection and policy snapshot. to_dict() performs no native preflight."""
    request: FeaturePlanRequest
    device: str
    execution_mode: str
    policy: CPUAnalysisPolicy

    def __post_init__(self) -> None:
        if type(self.request) is not FeaturePlanRequest or type(self.policy) is not CPUAnalysisPolicy:
            raise AnalysisExecutionError('typed_request_and_policy_required')
        if (type(self.execution_mode) is not str or self.execution_mode not in EXECUTOR_FEATURES
                or type(self.device) is not str or not (self.device=='cpu' or re.fullmatch(r'cuda:(0|[1-9][0-9]*)',self.device))):
            raise AnalysisExecutionError('invalid_execution_mode_or_device')
        plan_features(self.request)

    @property
    def selected_features(self) -> tuple[str, ...]:
        return tuple(row.feature_id for row in plan_features(self.request).features
                     if row.selection != 'prerequisite')

    def to_dict(self) -> dict[str,Any]:
        legacy = plan_features(self.request)
        rows = []
        for row in legacy.features:
            info = feature_catalog(row.feature_id).to_dict()
            support = _support(self,row.feature_id)
            rows.append(dict(**info,selection=row.selection,executor_support=support,
                status='SKIPPED' if row.selection=='prerequisite' else support,
                model_assessments=[asdict(m) for m in row.models]))
        data = dict(format='hyso.analysis_plan',schema_version=1,request=asdict(self.request),
            device=self.device,execution_mode=self.execution_mode,policy=asdict(self.policy),
            selected_features=list(self.selected_features),features=rows,
            execution_authorized=False,acceptance='NOT_EVALUATED',
            scope='catalog support is not runtime readiness; dependencies are never auto-executed')
        data['plan_id'] = digest(data)
        return data


def plan_analysis(models: ModelContext | Sequence[ModelContext] | FeaturePlanRequest,
                  features: str | Sequence[str] | None = None, *, profile: str | None = None,
                  capabilities: Sequence[str] = (), device: str = 'cpu',
                  execution_mode: str = 'trusted_cpu', policy: CPUAnalysisPolicy = CPUAnalysisPolicy()) -> AnalysisPlan:
    """Select a feature, list, profile or existing G1 request; never run/probe anything.

    Exactly one of features/profile is required unless a FeaturePlanRequest is
    supplied. Device selection never moves models or enables automatic fallback.
    Profiles choose IDs, not permissions. Existing G1 dependency semantics remain.
    """
    if type(policy) is not CPUAnalysisPolicy or type(execution_mode) is not str or execution_mode not in EXECUTOR_FEATURES:
        raise AnalysisExecutionError('typed_policy_and_known_execution_mode_required')
    if type(device) is not str or not (device=='cpu' or re.fullmatch(r'cuda:(0|[1-9][0-9]*)',device)):
        raise AnalysisExecutionError('device_must_be_cpu_or_explicit_cuda_index')
    if type(models) is FeaturePlanRequest:
        if features is not None or profile is not None or capabilities:
            raise AnalysisExecutionError('request_cannot_be_combined_with_selection_arguments')
        request = models
    else:
        if (features is None) == (profile is None):
            raise AnalysisExecutionError('provide_exactly_one_of_features_or_profile')
        if profile is not None:
            if type(profile) is not str or profile not in PROFILES:
                raise AnalysisExecutionError('unknown_profile')
            ids = PROFILES[profile]
        else:
            ids = (features,) if type(features) is str else features
        contexts = (models,) if type(models) is ModelContext else models
        request = FeaturePlanRequest('selected',contexts,features=ids,capabilities=capabilities)
    plan_features(request)  # Original G1 validation, no model/device inspection.
    return AnalysisPlan(request,device,execution_mode,policy)


def _support(plan, feature):
    if plan.device != 'cpu' or (plan.device=='cpu' and 'gpu' in feature_catalog(feature).required_capabilities):
        return 'UNSUPPORTED'
    return 'SUPPORTED' if feature in EXECUTOR_FEATURES[plan.execution_mode] else 'NOT_IMPLEMENTED_IN_EXECUTOR'


class AnalysisDryRun:
    """Owned metadata review and live bindings. Execution is explicit and one-shot.

    Nothing is imported from factory code during preflight. The common executor
    refreshes metadata/resource preflight before invoking any authorized adapter.
    """
    def __init__(self, plan, bindings, isolated_policy, data):
        self._plan = plan; self._bindings = bindings; self._isolated_policy = isolated_policy
        self._payload = canonical(data); self._used = False

    def to_dict(self) -> dict[str,Any]:
        """Return detached per-model support, reasons, costs, limits and OS estimates."""
        return json.loads(self._payload)

    def tracking_config(self) -> dict[str,Any]:
        """Explicit Manager config projection; full review/receipts stay in artifacts.

        ExperimentManager rejects auth-named snapshot keys. This summary carries
        plan identity, selection and unchanged policy, without approval flags or
        factory kwargs. It neither starts a run nor logs anything automatically.
        """
        plan = self.to_dict()['plan']
        return dict(analysis_plan_id=plan['plan_id'],selected_features=plan['selected_features'],
            models=[m['model_id'] for m in plan['request']['models']],device=plan['device'],
            execution_mode=plan['execution_mode'],policy=plan['policy'],
            scope='plan_and_policy_summary;full_dry_run_and_receipts_retained_in_execution_artifact')


def _binding_preflight(binding, model, policy, feature, mode):
    if model.backend!='pytorch' or model.task!='classification':
        return 'UNSUPPORTED',['this_executor_profile_requires_pytorch_classification'],0
    if mode == 'isolated_cpu':
        if not model.trusted_factory:
            return 'MISSING_PREREQUISITE',['missing_declared_trusted_factory'],0
        try:
            if _source_receipt(binding.source_file) != binding.source_sha256:
                raise AnalysisExecutionError('factory_sha256_mismatch')
        except (ValueError,OSError) as error:
            return 'MISSING_PREREQUISITE',[str(error)],0
        return 'SUPPORTED',[],0
    import torch
    if not isinstance(binding.model,torch.nn.Module):
        return 'MISSING_PREREQUISITE',['torch_module_required'],0
    try:
        tensors = tuple(binding.model.parameters()) + tuple(binding.model.buffers())
        if any(t.device.type!='cpu' for t in tensors):
            return 'UNSUPPORTED',['cpu_only_no_implicit_device_transfer'],0
        unique = {id(t):t for t in tensors}
        size = sum(t.numel()*t.element_size() for t in unique.values())
        if size > policy.max_model_bytes:
            return 'MISSING_PREREQUISITE',['registered_model_bytes_exceed_policy'],size
        if feature in ('P01','P02'):
            x = binding.input_tensor
            if (type(x) is not torch.Tensor or x.device.type!='cpu' or x.dtype!=torch.float32
                    or x.layout!=torch.strided or not x.is_contiguous() or x.requires_grad
                    or x.ndim!=4 or any(d<1 for d in x.shape)):
                return 'MISSING_PREREQUISITE',['bounded_cpu_fp32_no_grad_nchw_input_required'],size
            if any(t.is_floating_point() and t.dtype!=torch.float32 for t in tensors):
                return 'UNSUPPORTED',['inference_adapter_requires_fp32_model'],size
            input_bytes = x.numel()*x.element_size()
            if input_bytes > policy.max_input_bytes:
                return 'MISSING_PREREQUISITE',['input_bytes_exceed_policy'],size
            size += input_bytes
        return 'SUPPORTED',[],size
    except (ValueError,RuntimeError,TypeError) as error:
        return 'MISSING_PREREQUISITE',[type(error).__name__+':'+str(error)[:200]],0


def _binding_metadata(binding):
    if type(binding) is TrustedFactoryBinding:
        return dict(model_id=binding.model_id,source_file=binding.source_file,
                    source_sha256=binding.source_sha256,factory_name=binding.factory_name,kwargs=binding.kwargs)
    import torch
    x = binding.input_tensor
    try:
        tensors = tuple(binding.model.parameters())+tuple(binding.model.buffers()) if isinstance(binding.model,torch.nn.Module) else ()
        inspected = dict(registered_tensor_bytes=sum(t.numel()*t.element_size() for t in {id(t):t for t in tensors}.values()),
            model_devices=sorted({str(t.device) for t in tensors}),
            model_floating_dtypes=sorted({str(t.dtype) for t in tensors if t.is_floating_point()}))
    except (ValueError,RuntimeError,TypeError) as error:
        inspected = dict(inspection_error=type(error).__name__+':'+str(error)[:200])
    return dict(model_id=binding.model_id,
        model_type=type(binding.model).__module__+'.'+type(binding.model).__qualname__,
        **inspected,
        input_shape=list(x.shape) if type(x) is torch.Tensor else None,
        input_dtype=str(x.dtype) if type(x) is torch.Tensor else None,
        input_device=str(x.device) if type(x) is torch.Tensor else None)


def preflight_analysis(plan: AnalysisPlan, bindings: Sequence[LiveCPUModel | TrustedFactoryBinding] = (), *,
                       isolated_policy: IsolatedCPUExecutionPolicy = IsolatedCPUExecutionPolicy()) -> AnalysisDryRun:
    """Inspect metadata/OS headroom only; no forward, training or factory import.

    Missing bindings/prerequisites are per-model rows. Wrong or duplicate model
    identities are request errors. RAM/commit estimates fail closed if unavailable.
    In-process byte/operation limits do not bound arbitrary trusted Python runtime.
    """
    if type(plan) is not AnalysisPlan or type(isolated_policy) is not IsolatedCPUExecutionPolicy:
        raise AnalysisExecutionError('typed_plan_and_isolated_policy_required')
    if type(bindings) not in (tuple,list):
        raise AnalysisExecutionError('bindings_must_be_a_sequence')
    expected_type = LiveCPUModel if plan.execution_mode=='trusted_cpu' else TrustedFactoryBinding
    if any(type(b) is not expected_type for b in bindings):
        raise AnalysisExecutionError('binding_type_does_not_match_execution_mode')
    ids = [b.model_id for b in bindings]
    model_ids = [m.model_id for m in plan.request.models]
    if len(set(ids))!=len(ids) or not set(ids).issubset(model_ids):
        raise AnalysisExecutionError('binding_model_identity_mismatch')
    if plan.execution_mode=='trusted_cpu' and len({id(b.model) for b in bindings})!=len(bindings):
        raise AnalysisExecutionError('distinct_models_must_not_share_one_module')
    if plan.execution_mode=='isolated_cpu':
        for key in ('warmup_iterations','batch_count','repeats','queue_depth','max_input_bytes'):
            if getattr(plan.policy,key)!=getattr(isolated_policy,key):
                raise AnalysisExecutionError('isolated_policy_must_match_plan:'+key)
    lookup = {b.model_id:b for b in bindings}
    legacy = plan_features(plan.request)
    resources = memory_availability() if bindings and plan.device=='cpu' else dict(status='not_queried')
    rows = []
    for feature in legacy.features:
        for context, assessment in zip(plan.request.models,feature.models):
            status = _support(plan,feature.feature_id); reasons = []
            if feature.selection=='prerequisite':
                status='SKIPPED'; reasons=['dependency_only_not_selected;never_auto_execute']
            elif status=='UNSUPPORTED':
                reasons=['selected_device_not_supported_by_this_executor']
            elif assessment.status=='NOT_APPLICABLE':
                status='UNSUPPORTED'; reasons=list(assessment.reasons)
            elif assessment.status!='PROPOSED':
                status='MISSING_PREREQUISITE'; reasons=list(assessment.reasons)
            elif status=='NOT_IMPLEMENTED_IN_EXECUTOR':
                reasons=['native_adapter_not_implemented_in_common_executor']
            elif context.model_id not in lookup:
                status='MISSING_PREREQUISITE'; reasons=['missing_model_binding']
            else:
                status,reasons,size = _binding_preflight(lookup[context.model_id],context,
                    plan.policy,feature.feature_id,plan.execution_mode)
                if status=='SUPPORTED' and plan.execution_mode=='isolated_cpu' and os.name!='nt':
                    status='UNSUPPORTED'; reasons=['windows_job_backend_required_no_fallback']
                if status=='SUPPORTED':
                    demand = 3*size  # Only registered/input/snapshot headroom, NOT activations.
                    if (resources.get('status')!='observed'
                            or resources['available_physical_bytes'] < max(plan.policy.min_available_memory_bytes,demand)
                            or resources['available_commit_bytes'] < max(plan.policy.min_available_commit_bytes,demand)):
                        status='MISSING_PREREQUISITE'; reasons=['OS_memory_or_commit_headroom_not_verified']
            rows.append(dict(model_id=context.model_id,feature_id=feature.feature_id,selection=feature.selection,
                executor_support=_support(plan,feature.feature_id),status=status,reasons=reasons,
                dependencies=list(feature_catalog(feature.feature_id).dependencies),
                cost_class=feature_catalog(feature.feature_id).cost_class,
                executor_scope=feature_catalog(feature.feature_id).executor_scope))
    data = dict(format='hyso.analysis_dry_run',schema_version=1,plan=plan.to_dict(),features=rows,
        bindings=[_binding_metadata(b) for b in bindings],
        resources=resources,isolated_policy=asdict(replace(isolated_policy,explicit_authorization=False))
            if plan.execution_mode=='isolated_cpu' else None,
        execution_authorized=False,acceptance='NOT_EVALUATED',
        trust_scope='reviewed trusted Python only; no adversarial-code sandbox or implicit fallback')
    return AnalysisDryRun(plan,tuple(bindings),isolated_policy,data)


def _invoke(plan, context, binding, feature, isolated_policy):
    request = FeaturePlanRequest('single',(context,),features=(feature,),capabilities=plan.request.capabilities)
    if plan.execution_mode=='isolated_cpu':
        report = execute_isolated_cpu(request,(binding,),authorized_features=(feature,),
                                     policy=replace(isolated_policy,explicit_authorization=True))
        if report.status!='COMPLETED':
            return None,{},dict(status=report.status,reason=report.reason,worker_receipt=report.worker_receipt)
        captures = report.native_bridge_report['captures']
        if len(captures)!=1 or captures[0]['model_id']!=context.model_id or captures[0]['feature_id']!=feature:
            raise AnalysisExecutionError('isolated_capture_identity_mismatch')
        return captures[0]['native_result'],{},dict(status='COMPLETED',worker_receipt=report.worker_receipt)
    if feature in ('P02','P11'):
        p = plan.policy
        legacy_policy = TrustedCPUExecutionPolicy(explicit_trusted_inprocess_consent=True,
            warmup_iterations=p.warmup_iterations,batch_count=p.batch_count,repeats=p.repeats,
            queue_depth=p.queue_depth,max_input_bytes=p.max_input_bytes)
        report = execute_scoped_cpu(plan_features(request),(binding,),authorized_features=(feature,),policy=legacy_policy)
        if len(report.captures)!=1 or report.captures[0].model_id!=context.model_id or report.captures[0].feature_id!=feature:
            raise AnalysisExecutionError('scoped_capture_identity_mismatch')
        return json.loads(report.captures[0].result),{},dict(status='COMPLETED',scope=report.scope)
    if feature=='P01':
        from hyso.benchmark import BenchmarkRunner
        from hyso.performance import analyze_latency
        p = plan.policy
        benchmark = BenchmarkRunner(binding.model,device='cpu',example_input=binding.input_tensor,
            precision='fp32',warmup_iterations=p.warmup_iterations,
            measure_iterations=p.latency_measure_iterations,repeats=p.repeats,measure_cuda_memory=False).run()
        return analyze_latency(benchmark,quantiles=p.quantiles).to_dict(),{'benchmark':benchmark.to_dict()},dict(status='COMPLETED')
    if feature=='P10':
        from hyso.performance import measure_memory
        return measure_memory().to_dict(),{},dict(status='COMPLETED',scope='process_snapshot_only_no_forward')
    raise AnalysisExecutionError('native_adapter_not_implemented')


@dataclass(frozen=True)
class AnalysisExecutionResult:
    """Detached execution outcomes and native evidence, without acceptance claims."""
    _payload: str

    def to_dict(self) -> dict[str,Any]:
        return json.loads(self._payload)

    def to_json(self, path, *, overwrite: bool = False) -> None:
        """Explicit no-clobber export. No files are written by execution itself."""
        export(path,self._payload+'\n',overwrite)

    def to_evidence_index(self) -> EvidenceIndex:
        """Preserve validated native payloads and explicit unavailable reasons."""
        records = []
        for row in self.to_dict()['features']:
            if row['evidence_record'] is not None:
                records.append(EvidenceRecord(row['evidence_record']))
            else:
                status = 'invalid' if row['status']=='FAILED' else 'unsupported' if row['status'] in (
                    'UNSUPPORTED','NOT_IMPLEMENTED_IN_EXECUTOR') else 'not_run'
                records.append(EvidenceRecord.unavailable(row['feature_id'],status=status,
                    reason=';'.join(row['reasons']),scope=EvidenceScope(model_id=row['model_id'])))
        return EvidenceIndex(records)

    def to_unified_report(self, *, context: AnalysisContext) -> UnifiedAnalysisReport:
        """Use the existing evidence report. Advisor interpretation is a separate call."""
        risks = tuple('Executor '+row['model_id']+'/'+row['feature_id']+': '+row['status']+'; '+';'.join(row['reasons'])
                      for row in self.to_dict()['features'] if row['status']!='COMPLETED')
        return self.to_evidence_index().analyze(context=context,risks=risks)


def execute_analysis(dry_run: AnalysisDryRun, *, authorized_features: Sequence[str],
                     explicit_authorization: bool = False) -> AnalysisExecutionResult:
    """Run only explicitly selected AND authorized supported rows, once.

    Authorization never follows dependencies or enables another device/route.
    Ordinary per-feature failures are retained while independent rows continue.
    KeyboardInterrupt/SystemExit propagate. In-process trusted forwards may hang;
    use the existing isolated CPU route where a hard Job deadline is required.
    """
    if type(dry_run) is not AnalysisDryRun or dry_run._used:
        raise AnalysisExecutionError('fresh_dry_run_required;execution_is_one_shot')
    if type(authorized_features) not in (tuple,list) or any(type(f) is not str for f in authorized_features):
        raise AnalysisExecutionError('explicit_feature_id_sequence_required')
    authorized = tuple(authorized_features)
    if len(set(authorized))!=len(authorized) or not set(authorized).issubset(dry_run._plan.selected_features):
        raise AnalysisExecutionError('authorization_must_be_a_unique_subset_of_explicit_selection')
    if type(explicit_authorization) is not bool or (authorized and not explicit_authorization):
        raise AnalysisExecutionError('explicit_execution_authorization_required')
    fresh = preflight_analysis(dry_run._plan,dry_run._bindings,isolated_policy=dry_run._isolated_policy)
    data = fresh.to_dict()
    reviewed = dry_run.to_dict()
    if data['bindings']!=reviewed['bindings'] or data['plan']['plan_id']!=reviewed['plan']['plan_id']:
        raise AnalysisExecutionError('plan_or_binding_metadata_changed;review_a_new_dry_run')
    dry_run._used = True
    bindings = {b.model_id:b for b in dry_run._bindings}
    contexts = {m.model_id:m for m in dry_run._plan.request.models}
    rows = []
    for before in data['features']:
        row = dict(before,evidence_record=None,auxiliary_evidence={},execution_receipt=None,attempted=False)
        feature = row['feature_id']; model_id = row['model_id']
        if row['status']=='SUPPORTED' and feature not in authorized:
            row.update(status='SKIPPED',reasons=['not_authorized'])
        elif row['status']=='SUPPORTED':
            row['attempted'] = True
            try:
                native,aux,receipt = _invoke(dry_run._plan,contexts[model_id],bindings[model_id],feature,dry_run._isolated_policy)
                row['execution_receipt'] = receipt
                if native is None:
                    raise AnalysisExecutionError(receipt.get('reason') or 'native_execution_failed')
                if native.get('feature_ids') != [feature]:
                    raise AnalysisExecutionError('native_feature_identity_mismatch')
                record = EvidenceRecord.from_native(native,feature_id=feature,scope=EvidenceScope(model_id=model_id))
                row.update(evidence_record=record.to_dict(),auxiliary_evidence=aux,status='COMPLETED',reasons=[])
                if feature=='P10' and native['actual']['observations']==0:
                    row.update(status='UNSUPPORTED',reasons=['native_RSS_backend_unavailable'])
                elif feature=='P01' and any(m['value'] is None for m in native['measurements']):
                    row.update(status='MISSING_PREREQUISITE',reasons=['requested_quantile_native_tail_policy_not_met;raw_samples_retained'])
            except Exception as error:
                row.update(status='FAILED',reasons=[type(error).__name__+':'+str(error)[:240]])
        rows.append(row)
    result = dict(format='hyso.analysis_execution',schema_version=1,dry_run=data,
        authorized_features=list(authorized),features=rows,
        status_counts={s:sum(r['status']==s for r in rows) for s in STATUSES},
        acceptance='NOT_EVALUATED',comparison_authority='per_model_observations_only;no_certified_winner',
        reporting_scope='native evidence, execution status and later Advisor interpretation are separate')
    return AnalysisExecutionResult(canonical(result))
