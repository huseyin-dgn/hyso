"""P35 optional ONNX Runtime profiling adapter, not a provider-list heuristic.

Requires optional onnxruntime and onnx in the worker/validation environment.
Disabled optimization, unique names and matching optimized/source node definitions
are required for source-node attribution. Graph analysis v2 resolves only validated
runtime transfer aliases; fusion is conservatively unknown. Legacy results retain
their original graph interpretation unless explicitly reanalyzed as v2 evidence.

Compatibility semantics (RuntimeResult schema1): ``fallback_detected`` and the
same-named analysis status mean *unexpected observed provider execution*, based
on the caller's explicit node expectations. They do not establish an execution
failure, retry, session-creation fallback, or accelerated-provider-to-CPU
transition. A CPU event where Azure was expected is an expectation mismatch.
``session_fallback`` describes configuration differences only; static assignment
remains unknown. These legacy field names are retained so existing serialized
results can still be reopened without rehashing or a schema migration.
"""
import base64
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from .runtime_result import _result, _hash, _int, _number, _name, _digest
from .stability import WorkloadRecipe, _factory, _atomic
from .devices import _launch, _valid_run


@dataclass(frozen=True)
class ProviderProtocol:
    warmup: int = 0
    runs: int = 1
    startup_timeout_seconds: float = 45.
    step_timeout_seconds: float = 15.
    max_runtime_seconds: float = 60.
    process_memory_bytes: int | None = None
    job_memory_bytes: int | None = None
    max_processes: int | None = None

    def __post_init__(self):
        _int(self.warmup,0,100);_int(self.runs,1,100)
        for value,limit in ((self.startup_timeout_seconds,120),(self.step_timeout_seconds,60),
                            (self.max_runtime_seconds,300)):
            _number(value,.05)
            if value>limit:raise ValueError('worker time budget exceeded')
        for limit in (self.process_memory_bytes,self.job_memory_bytes):
            if limit is not None:_int(limit,1)
        if self.max_processes is not None:_int(self.max_processes,1,2**32-1)
        if self.process_memory_bytes is not None and self.job_memory_bytes is not None and self.process_memory_bytes>self.job_memory_bytes:
            raise ValueError('job memory must cover one process')


def profile_execution_providers(recipe, directory, *, providers=None, expected=None, optimization='all',
                                protocol=ProviderProtocol(), runtime_search_path=None, provider_options=None):
    """Factory returns (self-contained ONNX bytes, numpy input-feed dictionary).

    Default: one profiled CPU execution with normal ORT graph optimization.
    No dependency is installed; missing ORT/providers produce unavailable evidence.
    CUDA/device options are explicit opt-in; a physical GPU UUID is optional.
    expected optionally maps unique source node names to provider names. '*' is an explicit
    default. A nonmatching observed provider is evidence, even if that requested
    provider was unavailable; it is never proof of CUDA execution. In schema1,
    ``fallback_detected=True`` means only unexpected_provider_execution. Runtime
    failure/retry and session-creation fallback are not observed by this adapter;
    ``disable_fallback`` prevents automatic run-error session recreation.
    """
    if type(recipe) is not WorkloadRecipe or type(protocol) is not ProviderProtocol:
        raise TypeError('typed recipe and protocol required')
    if providers is not None:
        if type(providers) not in (list,tuple) or not providers or any(type(x) is not str for x in providers) or len(set(providers))!=len(providers):
            raise ValueError('ordered distinct provider names required')
        for provider in providers:_name(provider)
        providers=list(providers)
    expected={} if expected is None else expected
    if type(expected) is not dict:raise ValueError('node expectations must be a mapping')
    expected=dict(expected)
    for node,provider in expected.items():_name(node);_name(provider)
    if optimization not in ('disabled','all'):raise ValueError('unsupported optimization')
    if runtime_search_path is not None:
        path=Path(runtime_search_path)
        if not path.is_absolute() or not path.is_dir():raise ValueError('absolute optional runtime directory required')
        runtime_search_path=str(path)
    provider_options=_options(provider_options,providers,recipe.gpu_uuid)
    run=_launch(recipe,directory,'providers',dict(providers=providers,expected=expected,
        optimization=optimization,runtime_search_path=runtime_search_path,provider_options=provider_options),protocol)
    run['graph_analysis_version']=2
    run['provider_analysis_version']=2
    return _result('P35',run)


def _options(options, providers, gpu_uuid=None):
    options={} if options is None else options
    if type(options) is not dict or any(type(k) is not str or type(v) is not dict for k,v in options.items()):
        raise ValueError('provider options must map provider names to option dictionaries')
    if set(options)-set(providers or ['CPUExecutionProvider']):raise ValueError('options require a selected provider')
    for provider,values in options.items():
        for name,value in values.items():
            if type(name) is not str or type(value) not in (str,int,float,bool) or (type(value) is float and not math.isfinite(value)):
                raise ValueError('plain finite provider option values required')
            if name in ('device_id','gpu_mem_limit'):
                if type(value) not in (str,int) or not str(value).isdigit() or int(value)<(1 if name=='gpu_mem_limit' else 0):
                    raise ValueError('invalid explicit device/memory option')
            if gpu_uuid and provider in ('CUDAExecutionProvider','TensorrtExecutionProvider') and name=='device_id' and int(value)!=0:
                raise ValueError('UUID visibility selects one CUDA device, index must be zero')
    encoded=json.dumps(options,allow_nan=False)
    if len(encoded)>65536:raise ValueError('provider option budget exceeded')
    return json.loads(encoded)


def _is_runtime_transfer(node):
    return node.domain in ('','com.microsoft') and node.op_type in ('MemcpyToHost','MemcpyFromHost')


def _transfer_aliases(model):
    aliases={}
    for node in model.graph.node:
        if not _is_runtime_transfer(node):continue
        if len(node.input)!=1 or len(node.output)!=1 or not node.input[0] or not node.output[0] or node.attribute:
            raise ValueError('unsupported runtime transfer definition')
        if node.output[0] in aliases:raise ValueError('ambiguous runtime transfer output')
        aliases[node.output[0]]=node.input[0]
    for name in aliases:_resolve_transfer(name,aliases)
    return aliases


def _resolve_transfer(name,aliases):
    seen=set()
    while name in aliases:
        if name in seen:raise ValueError('cyclic runtime transfer aliases')
        seen.add(name);name=aliases[name]
    return name


def _canonical_nodes(model,aliases,*,optimized=False):
    # Transfer nodes are semantic identities for source graph association only.
    # Do not erase their actual provider events, durations or raw graph bytes.
    result=[]
    for node in model.graph.node:
        if optimized and _is_runtime_transfer(node):continue
        copy=type(node)();copy.CopyFrom(node)
        for fields in (copy.input,copy.output):
            for i,name in enumerate(fields):fields[i]=_resolve_transfer(name,aliases)
        result.append(copy.SerializeToString())
    return result


def _decode_model(encoded,*,optimized=False):
    import onnx
    raw=base64.b64decode(encoded,validate=True)
    if len(raw)>16*1024**2:raise ValueError('model evidence exceeds budget')
    model=onnx.load_model_from_string(raw)
    # External files and subgraphs require a separate artifact/mapping adapter.
    if any(t.data_location==onnx.TensorProto.EXTERNAL for t in model.graph.initializer):
        raise ValueError('external ONNX data unsupported')
    if any(a.type in (onnx.AttributeProto.GRAPH,onnx.AttributeProto.GRAPHS)
           for node in model.graph.node for a in node.attribute):
        raise ValueError('control-flow subgraphs unsupported')
    checked=model
    if optimized:
        _transfer_aliases(model)
        # ORT serializes its internal Memcpy operators in the standard domain.
        # Check a disposable semantic-identity copy; retain/hash the original.
        checked=type(model)();checked.CopyFrom(model)
        for node in checked.graph.node:
            if _is_runtime_transfer(node):node.op_type='Identity';node.domain=''
    onnx.checker.check_model(checked)
    return raw,model


def _analyze_providers(e):
    version=e.get('provider_analysis_version',1)
    if type(version) is not int or version not in (1,2):raise ValueError('invalid provider evidence analysis version')
    analysis=_analyze_providers_legacy(e)
    if version==2:
        from ._provider_diagnostics import _assessment
        analysis['evidence_assessment']=_assessment(e,analysis,strict=True)
    return analysis


def _analyze_providers_legacy(e):
    ProviderProtocol(**e['protocol']);WorkloadRecipe(**e['recipe'])
    graph_version=e.get('graph_analysis_version',1)
    if type(graph_version) is not int or graph_version not in (1,2):raise ValueError('invalid provider graph analysis version')
    raw_report=e.get('report') or {}
    if 'observation_schema' in raw_report and (type(raw_report['observation_schema']) is not int or raw_report['observation_schema']!=2):
        raise ValueError('unsupported provider observation schema')
    report,reason=_valid_run(e)
    if report is None:
        result=dict(status='unavailable',reason=reason,fallback_detected=None,
            session_fallback=None,node_assignment=dict(status='unknown',reason='no completed execution'),events=[])
        if (e.get('report') or {}).get('observation_schema')==2 and 'failure_kind' in e['report']:
            kind=e['report']['failure_kind']
            if kind not in ('oom','missing_dependency','unsupported_provider','cuda_error','workload_error','ort_error'):
                raise ValueError('invalid provider failure category')
            result['failure_kind']=kind
        return result
    modern=report.get('observation_schema')==2
    _digest(report['input_sha256'])
    _int(report['completed_runs'],1)
    if report['completed_runs']!=e['protocol']['warmup']+e['protocol']['runs']:
        raise ValueError('incomplete session execution ledger')
    if report['runtime']['name']!='onnxruntime':raise ValueError('unsupported profiling backend')
    raw,model=_decode_model(report['model_base64']);opt_raw,optimized=_decode_model(report['optimized_base64'],optimized=graph_version==2)
    if (hashlib.sha256(raw).hexdigest()!=report['model_sha256'] or
            hashlib.sha256(model.graph.SerializeToString()).hexdigest()!=report['graph_sha256'] or
            hashlib.sha256(opt_raw).hexdigest()!=report['optimized_sha256']):
        raise ValueError('model/graph identity mismatch')
    trace=report['trace']
    if type(trace) is not list or _hash(trace)!=report['trace_sha256']:
        raise ValueError('corrupt profiling trace')
    if report['session_id']!=e['token'] or report['profile_binding']!=_hash(dict(
            session_id=e['token'],model_sha256=report['model_sha256'],trace_sha256=report['trace_sha256'],
            input_sha256=report['input_sha256'],providers=report['session_providers'])):
        raise ValueError('trace/session/model binding mismatch')
    requested=e['providers'] if e['providers'] is not None else ['CPUExecutionProvider']
    if report['requested_providers']!=requested or report['optimization']!=e['optimization']:
        raise ValueError('session context mismatch')
    if modern:
        if report['requested_provider_options']!=e['provider_options']:raise ValueError('provider option context mismatch')
        if any(x not in report['available_providers'] for x in requested):raise ValueError('unavailable requested provider claimed successful')
        if any(x not in report['session_providers'] for x in requested):raise ValueError('successful profile lost requested providers')
    source=list(model.graph.node);names=[n.name for n in source]
    unique=all(names) and len(set(names))==len(names)
    if graph_version==2:
        aliases=_transfer_aliases(optimized)
        same_nodes=_canonical_nodes(model,aliases)==_canonical_nodes(optimized,aliases,optimized=True)
    else:
        same_nodes=[n.SerializeToString() for n in source]==[n.SerializeToString() for n in optimized.graph.node]
    can_map=e['optimization']=='disabled' and unique and same_nodes
    if set(e['expected'])-set(names)-{'*'}:raise ValueError('expectation names not in source graph')
    events=[];covered=set();fallback=[];executed=set()
    for event in trace:
        if type(event) is not dict:raise ValueError('malformed trace event')
        if event.get('cat')!='Node':continue
        if type(event.get('name')) is not str:raise ValueError('malformed Node event identity')
        if not event['name'].endswith('_kernel_time'):continue
        if event.get('pid')!=e['controller']['launcher_pid']:
            raise ValueError('profiling trace belongs to another worker process')
        args=event.get('args',{})
        if type(args) is not dict:raise ValueError('malformed Node event arguments')
        if modern:
            if args.get('op_name') is not None:_name(args['op_name'])
            if event.get('ts') is not None:_number(event['ts'])
        provider=args.get('provider');name=event['name'][:-12]
        duration=event.get('dur')
        if duration is not None or not modern:_number(duration,0)
        if not isinstance(provider,str) or provider not in report['session_providers']:
            if not modern or provider not in (None,''):
                raise ValueError('execution event has no valid session provider')
            provider=None
        if provider is not None:executed.add(provider)
        node=next((n for n in source if n.name==name),None) if can_map else None
        mapped=node is not None and node.op_type==args.get('op_name')
        expectation=e['expected'].get(name,e['expected'].get('*')) if mapped else None
        row=dict(source_node=name if mapped else None,event_name=event['name'],provider=provider,
                 expected_provider=expectation,mapping='mapped' if mapped else 'unmapped/unknown',duration_us=duration)
        if modern:row.update(op_type=args.get('op_name'),event_timestamp_us=event.get('ts'),
                             evidence_status='observed' if provider is not None and duration is not None else 'incomplete')
        if mapped and provider is not None:covered.add(name)
        if provider is not None and expectation is not None and expectation!=provider:fallback.append(row)
        events.append(row)
    complete=can_map and bool(names) and covered==set(names) and all(x['mapping']=='mapped' for x in events) and all(
        e['expected'].get(name,e['expected'].get('*')) is not None for name in names) and all(x['duration_us'] is not None for x in events)
    automatic=modern and not e['expected']
    observed=any(x['provider'] is not None for x in events)
    return dict(status='execution_observed' if automatic and observed else 'fallback_detected' if fallback else 'expected_execution_observed' if complete else 'unknown',
        reason=None if complete or (automatic and observed) else 'missing execution coverage, expectations, or unambiguous source/optimized mapping',
        fallback_detected=True if fallback else False if complete else None,
        session_fallback=dict(status='configuration_difference' if any(x not in report['session_providers'] for x in requested)
            else 'requested_providers_present',missing_requested=[x for x in requested if x not in report['session_providers']],
            execution_claim=False),
        node_assignment=dict(status='observed_execution_only',static_assignment=None,
            reason='public session provider API does not expose complete static node placement'),
        events=events,unexpected_executions=fallback,uncovered_source_nodes=sorted(set(names)-covered),
        unused_session_providers=[x for x in report['session_providers'] if x not in executed],
        model_sha256=report['model_sha256'],graph_sha256=report['graph_sha256'],
        runtime=report['runtime'],provider_versions=report['provider_versions'],
        evidence_scope='ORT host Node kernel_time events; no GPU kernel/CUPTI timing claim',
        integrity_scope='hash consistency and session association, not signed authenticity')


class _ProviderFailure(RuntimeError):
    def __init__(self, reason, details):
        super().__init__(reason)
        self.details=details


def _failure_kind(error, phase):
    text=str(error).lower()
    if isinstance(error,MemoryError) or any(x in text for x in
            ('out of memory','failed to allocate memory','cudaerrormemoryallocation')):
        return 'oom'
    if phase=='dependencies' and isinstance(error,(ImportError,ModuleNotFoundError)):return 'missing_dependency'
    if isinstance(error,NotImplementedError) and phase=='provider_selection':return 'unsupported_provider'
    if 'cuda' in text or 'cudnn' in text or 'cublas' in text:return 'cuda_error'
    return 'workload_error' if phase in ('factory','input_validation') else 'ort_error'


def _worker_providers(root, request):
    details=dict(observation_schema=2,runtime=None,available_providers=None,requested_providers=request['providers'],
        requested_provider_options=request.get('provider_options',{}),session_providers=None,completed_runs=0,phase='dependencies')
    try:
        return _collect_profile(root,request,details)
    except Exception as error:
        details['failure_kind']=_failure_kind(error,details['phase'])
        details['error_type']=type(error).__name__
        raise _ProviderFailure(type(error).__name__+': '+str(error),details) from error


def _runtime_info(ort):
    build=getattr(ort,'get_build_info',None)
    return dict(name='onnxruntime',version=ort.__version__,build=build() if callable(build) else None)


def _diagnostics(evidence, analysis):
    """Unambiguous modern names over schema1, without additional executions."""
    report=evidence.get('report') or {};events=analysis['events']
    observed=sorted({x['provider'] for x in events if x['provider'] is not None})
    operators={}
    for event in events:
        key=(event.get('op_type'),event['provider'])
        group=operators.setdefault(key,dict(op_type=key[0],provider=key[1],events=0,timed_events=0,total_host_node_us=0.))
        group['events']+=1
        if event['duration_us'] is not None:
            group['timed_events']+=1;group['total_host_node_us']+=event['duration_us']
    for group in operators.values():
        group['mean_host_node_us']=group['total_host_node_us']/group['timed_events'] if group['timed_events'] else None
        group['known_duration_sum_us']=group['total_host_node_us'] if group['timed_events'] else None
        group['duration_coverage']='complete' if group['timed_events']==group['events'] else 'partial'
        if group['timed_events']!=group['events']:group['total_host_node_us']=None
    session=report.get('session_providers');requested=report.get('requested_providers')
    missing=[x for x in requested if x not in session] if session is not None and requested is not None else None
    result=dict(core_status='observed' if observed else 'unavailable' if analysis['status']=='unavailable' else 'unknown',
        advanced_status='partial',runtime=report.get('runtime'),available_providers=report.get('available_providers'),
        requested_providers=requested,enabled_session_providers=session,observed_execution_providers=observed,
        provider_options=report.get('provider_options'),requested_provider_options=evidence.get('provider_options'),
        configuration_fallback=dict(status='configuration_difference' if missing else 'requested_providers_present'
            if missing is not None else 'unknown',missing_requested=missing,execution_claim=False),
        expectation_mismatch=dict(status='observed' if analysis.get('unexpected_executions') else
            'not_requested' if not evidence['expected'] else
            'not_observed_in_covered_events' if analysis['fallback_detected'] is False else 'unknown',
            events=analysis.get('unexpected_executions',[])),
        execution_fallback=dict(status='unknown',detected=None,
            reason='provider events and expectations do not establish a failure/retry or CUDA-to-CPU transition'),
        source_node_mapping=dict(status='observed' if events and all(x['mapping']=='mapped' for x in events) else 'unknown',
            uncovered_source_nodes=analysis.get('uncovered_source_nodes'),optimization=evidence['optimization']),
        events=events,operator_timings=list(operators.values()),completed_runs=report.get('completed_runs'),
        failure_kind=analysis.get('failure_kind'),failure_phase=report.get('phase'),reason=analysis.get('reason'),
        worker_exit_code=evidence['controller']['exit_code'],controller_outcome=evidence['controller']['reason'],
        device_selection=report.get('device_selection'),
        resource_policy='advisory_unless_explicit_limits_requested',
        limitations=['ORT Node durations are host profiler events, not GPU kernel time or end-to-end inference latency.',
            'Node durations can overlap; their sum is not a session wall-time measurement.',
            'Profiling covers requested warmup/runs and runtime-recorded setup; profiling and publication add overhead.',
            'Optimization is enabled by default; disabled optimization changes inference behavior and is reported.',
            'Fusion, renaming and missing trace fields leave source attribution unknown.',
            'Legacy fallback_detected means expectation mismatch only; execution fallback is not established.',
            'Run-error retry is disabled and session provider loss aborts before inference; no automatic rebenchmark.',
            'Per-provider library versions and physical GPU UUID execution identity may be unavailable.',
            'Owned-worker capture is Windows-only; optional ORT/ONNX are not required for base HYSo imports.'])
    from ._provider_diagnostics import _assessment
    result['evidence_assessment']=analysis.get('evidence_assessment') or _assessment(evidence,analysis)
    result['source_node_mapping']['coverage_status']=(result['evidence_assessment']['node_coverage'] or {}).get('status')
    return result


def _collect_profile(root, request, details):
    if request['runtime_search_path']:sys.path.insert(0,request['runtime_search_path'])
    import numpy as np
    import onnxruntime as ort
    p=ProviderProtocol(**request['protocol'])
    details.update(runtime=_runtime_info(ort),
        available_providers=ort.get_available_providers())
    import onnx
    details['phase']='provider_selection'
    requested=request['providers'] if request['providers'] is not None else ['CPUExecutionProvider']
    details['requested_providers']=requested
    missing=[name for name in requested if name not in details['available_providers']]
    if missing:raise NotImplementedError('requested providers unavailable: '+', '.join(missing))
    details['phase']='factory'
    (raw,feed),source=_factory(request['recipe'])
    if type(raw) is not bytes or type(feed) is not dict or not feed:raise TypeError('ONNX bytes and numpy feeds required')
    encoded=base64.b64encode(raw).decode();_,model=_decode_model(encoded)
    if set(request['expected'])-{n.name for n in model.graph.node}-{'*'}:
        raise ValueError('expectation names not in source graph')
    details['phase']='input_validation'
    for value in feed.values():
        if not isinstance(value,np.ndarray) or value.dtype.hasobject or not np.isfinite(value).all():
            raise ValueError('finite numpy feeds required')
    if sum(v.nbytes for v in feed.values())>64*1024**2:raise ValueError('input budget exceeded')
    def fingerprints():
        return {k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in feed.items()}
    input_before=fingerprints()
    options=ort.SessionOptions();options.enable_profiling=True
    options.profile_file_prefix=str(root/'profile');options.optimized_model_filepath=str(root/'optimized.onnx')
    options.intra_op_num_threads=1;options.inter_op_num_threads=1
    options.graph_optimization_level=(ort.GraphOptimizationLevel.ORT_DISABLE_ALL if request['optimization']=='disabled'
                                      else ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
    configured=_options(request.get('provider_options'),requested,request['recipe'].get('gpu_uuid'))
    selected=[(name,configured[name]) if configured.get(name) else name for name in requested]
    if 'DmlExecutionProvider' in requested:
        options.enable_mem_pattern=False
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    # ORT 1.23.2 exposes this constructor flag through **kwargs. No private session API.
    # Also reject configuration loss before executing anything; never silently benchmark CPU.
    details['phase']='session_initialization'
    session=ort.InferenceSession(raw,sess_options=options,providers=selected,enable_fallback=False)
    session.disable_fallback()
    details['session_providers']=session.get_providers()
    details['provider_options']=session.get_provider_options()
    if any(name not in details['session_providers'] for name in requested):
        details['phase']='session_configuration_difference'
        raise RuntimeError('session did not enable all requested providers; no inference executed')
    details['phase']='execution'
    intervals=[]
    for index in range(p.warmup+p.runs):
        start=time.perf_counter_ns()
        outputs=session.run(None,feed)
        end=time.perf_counter_ns()
        if any(not np.isfinite(value).all() for value in outputs):raise ValueError('nonfinite ONNX output')
        intervals.append(dict(index=index,phase='warmup' if index<p.warmup else 'measured',start_ns=start,end_ns=end))
        details['completed_runs']=index+1
        _atomic(root/'heartbeat.json',dict(index=index))
    details['phase']='profile_publication'
    profile=Path(session.end_profiling())
    if profile.parent!=root or profile.stat().st_size>16*1024**2:raise ValueError('unbounded/foreign profile')
    trace=json.loads(profile.read_text(encoding='utf-8'));trace_hash=_hash(trace)
    model_hash=hashlib.sha256(raw).hexdigest()
    input_after=fingerprints();input_hash=_hash(input_before)
    providers=session.get_providers()
    return dict(observation_schema=2,requested_provider_options=configured,
        capture_context=dict(input_fingerprints=input_before,input_unchanged=input_before==input_after,
            profile_clock=dict(name='ORT_JSON_profile_host_relative',unit='us',origin='runtime_private_uncorrelated_with_host_perf_counter'),
            host_clock=dict(name='perf_counter_ns',unit='ns',instance_id=request['token']+':perf_counter_ns'),
            run_intervals=intervals,method='session.run_return_host_interval;finite_output_validation_after_end'),
        device_selection=dict(gpu_uuid_request=request['recipe'].get('gpu_uuid'),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            scope='ORT provider options/default device; physical UUID execution association not observed'),
        session_id=request['token'],model_base64=encoded,model_sha256=model_hash,
        graph_sha256=hashlib.sha256(model.graph.SerializeToString()).hexdigest(),
        optimized_base64=base64.b64encode((root/'optimized.onnx').read_bytes()).decode(),
        optimized_sha256=hashlib.sha256((root/'optimized.onnx').read_bytes()).hexdigest(),
        input_sha256=input_hash,source=source,runtime=_runtime_info(ort),provider_versions={x:dict(value=None,status='unavailable',
        reason='ORT does not expose individual provider versions; runtime build recorded') for x in providers},
        available_providers=details['available_providers'],requested_providers=requested,
        session_providers=providers,provider_options=session.get_provider_options(),optimization=request['optimization'],
        trace=trace,trace_sha256=trace_hash,profile_binding=_hash(dict(session_id=request['token'],
        model_sha256=model_hash,trace_sha256=trace_hash,input_sha256=input_hash,providers=providers)),
        completed_runs=p.warmup+p.runs)
