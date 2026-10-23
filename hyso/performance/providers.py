"""P35 optional ONNX Runtime profiling adapter, not a provider-list heuristic.

Requires optional onnxruntime and onnx in the worker/validation environment.
Disabled optimization, unique names and matching optimized/source node definitions
are required for source-node attribution. Fusion is conservatively unknown.
"""
import base64
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import sys

from .runtime_result import _result, _hash, _int, _number, _name, _digest
from .stability import WorkloadRecipe, _factory, _atomic
from .devices import _launch, _valid_run


@dataclass(frozen=True)
class ProviderProtocol:
    warmup: int = 1
    runs: int = 2
    startup_timeout_seconds: float = 45.
    step_timeout_seconds: float = 15.
    max_runtime_seconds: float = 60.

    def __post_init__(self):
        _int(self.warmup,0,100);_int(self.runs,1,100)
        for value,limit in ((self.startup_timeout_seconds,120),(self.step_timeout_seconds,60),
                            (self.max_runtime_seconds,300)):
            _number(value,.05)
            if value>limit:raise ValueError('worker time budget exceeded')


def profile_execution_providers(recipe, directory, *, providers, expected, optimization='disabled',
                                protocol=ProviderProtocol(), runtime_search_path=None):
    """Factory returns (self-contained ONNX bytes, numpy input-feed dictionary).

    expected maps unique source node names to provider names. '*' is an explicit
    default. A nonmatching observed provider is evidence, even if that requested
    provider was unavailable; it is never proof of CUDA execution.
    """
    if type(recipe) is not WorkloadRecipe or type(protocol) is not ProviderProtocol:
        raise TypeError('typed recipe and protocol required')
    if type(providers) not in (list,tuple) or not providers or len(set(providers))!=len(providers):
        raise ValueError('explicit ordered distinct providers required')
    for provider in providers:_name(provider)
    if any(provider not in ('CPUExecutionProvider','AzureExecutionProvider','CUDAExecutionProvider') for provider in providers):
        raise ValueError('unsupported execution provider adapter')
    if type(expected) is not dict or not expected:raise ValueError('explicit node expectations required')
    for node,provider in expected.items():_name(node);_name(provider)
    if optimization not in ('disabled','all'):raise ValueError('unsupported optimization')
    if runtime_search_path is not None:
        path=Path(runtime_search_path)
        if not path.is_absolute() or not path.is_dir():raise ValueError('absolute optional runtime directory required')
        runtime_search_path=str(path)
    if any('CUDA' in x or 'Tensorrt' in x for x in providers) and recipe.gpu_uuid is None:
        raise ValueError('GPU providers require explicit physical GPU UUID admission')
    run=_launch(recipe,directory,'providers',dict(providers=list(providers),expected=expected,
        optimization=optimization,runtime_search_path=runtime_search_path),protocol)
    return _result('P35',run)


def _decode_model(encoded):
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
    onnx.checker.check_model(model)
    return raw,model


def _analyze_providers(e):
    ProviderProtocol(**e['protocol']);WorkloadRecipe(**e['recipe'])
    report,reason=_valid_run(e)
    if report is None:return dict(status='unavailable',reason=reason,fallback_detected=None,
        session_fallback=None,node_assignment=dict(status='unknown',reason='no completed execution'),events=[])
    _digest(report['input_sha256'])
    if report['completed_runs']!=e['protocol']['warmup']+e['protocol']['runs']:
        raise ValueError('incomplete session execution ledger')
    if report['runtime']['name']!='onnxruntime':raise ValueError('unsupported profiling backend')
    raw,model=_decode_model(report['model_base64']);opt_raw,optimized=_decode_model(report['optimized_base64'])
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
    if report['requested_providers']!=e['providers'] or report['optimization']!=e['optimization']:
        raise ValueError('session context mismatch')
    source=list(model.graph.node);names=[n.name for n in source]
    unique=all(names) and len(set(names))==len(names)
    same_nodes=[n.SerializeToString() for n in source]==[n.SerializeToString() for n in optimized.graph.node]
    can_map=e['optimization']=='disabled' and unique and same_nodes
    if set(e['expected'])-set(names)-{'*'}:raise ValueError('expectation names not in source graph')
    events=[];covered=set();fallback=[];executed=set()
    for event in trace:
        if type(event) is not dict:raise ValueError('malformed trace event')
        if event.get('cat')!='Node' or not event.get('name','').endswith('_kernel_time'):continue
        if event.get('pid')!=e['controller']['launcher_pid']:
            raise ValueError('profiling trace belongs to another worker process')
        args=event.get('args',{});provider=args.get('provider');name=event['name'][:-12]
        _number(event.get('dur'),0)
        if not isinstance(provider,str) or provider not in report['session_providers']:
            raise ValueError('execution event has no valid session provider')
        executed.add(provider)
        node=next((n for n in source if n.name==name),None) if can_map else None
        mapped=node is not None and node.op_type==args.get('op_name')
        expectation=e['expected'].get(name,e['expected'].get('*')) if mapped else None
        row=dict(source_node=name if mapped else None,event_name=event['name'],provider=provider,
                 expected_provider=expectation,mapping='mapped' if mapped else 'unmapped/unknown',duration_us=event['dur'])
        if mapped:covered.add(name)
        if expectation is not None and expectation!=provider:fallback.append(row)
        events.append(row)
    complete=can_map and bool(names) and covered==set(names) and all(x['mapping']=='mapped' for x in events) and all(
        e['expected'].get(name,e['expected'].get('*')) is not None for name in names)
    return dict(status='fallback_detected' if fallback else 'expected_execution_observed' if complete else 'unknown',
        reason=None if complete else 'missing execution coverage, expectations, or unambiguous source/optimized mapping',
        fallback_detected=True if fallback else False if complete else None,
        session_fallback=dict(status='configuration_difference' if any(x not in report['session_providers'] for x in e['providers'])
            else 'requested_providers_present',missing_requested=[x for x in e['providers'] if x not in report['session_providers']],
            execution_claim=False),
        node_assignment=dict(status='observed_execution_only',static_assignment=None,
            reason='public session provider API does not expose complete static node placement'),
        events=events,unexpected_executions=fallback,uncovered_source_nodes=sorted(set(names)-covered),
        unused_session_providers=[x for x in report['session_providers'] if x not in executed],
        model_sha256=report['model_sha256'],graph_sha256=report['graph_sha256'],
        runtime=report['runtime'],provider_versions=report['provider_versions'],
        evidence_scope='ORT host Node kernel_time events; no GPU kernel/CUPTI timing claim',
        integrity_scope='hash consistency and session association, not signed authenticity')


def _worker_providers(root, request):
    if request['runtime_search_path']:sys.path.insert(0,request['runtime_search_path'])
    import numpy as np
    import onnxruntime as ort
    p=ProviderProtocol(**request['protocol'])
    (raw,feed),source=_factory(request['recipe'])
    if type(raw) is not bytes or type(feed) is not dict or not feed:raise TypeError('ONNX bytes and numpy feeds required')
    encoded=base64.b64encode(raw).decode();_,model=_decode_model(encoded)
    for value in feed.values():
        if not isinstance(value,np.ndarray) or value.dtype.hasobject or not np.isfinite(value).all():
            raise ValueError('finite numpy feeds required')
    if sum(v.nbytes for v in feed.values())>64*1024**2:raise ValueError('input budget exceeded')
    options=ort.SessionOptions();options.enable_profiling=True
    options.profile_file_prefix=str(root/'profile');options.optimized_model_filepath=str(root/'optimized.onnx')
    options.intra_op_num_threads=1;options.inter_op_num_threads=1
    options.graph_optimization_level=(ort.GraphOptimizationLevel.ORT_DISABLE_ALL if request['optimization']=='disabled'
                                      else ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
    selected=[(name,{'device_id':0,'gpu_mem_limit':256*1024**2}) if name=='CUDAExecutionProvider' else name
              for name in request['providers']]
    session=ort.InferenceSession(raw,sess_options=options,providers=selected)
    session.disable_fallback()  # Never silently recreate a session on run failure.
    for index in range(p.warmup+p.runs):
        outputs=session.run(None,feed)
        if any(not np.isfinite(value).all() for value in outputs):raise ValueError('nonfinite ONNX output')
        _atomic(root/'heartbeat.json',dict(index=index))
    profile=Path(session.end_profiling())
    if profile.parent!=root or profile.stat().st_size>16*1024**2:raise ValueError('unbounded/foreign profile')
    trace=json.loads(profile.read_text(encoding='utf-8'));trace_hash=_hash(trace)
    model_hash=hashlib.sha256(raw).hexdigest()
    input_hash=_hash({k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest())
                      for k,v in feed.items()})
    providers=session.get_providers()
    return dict(session_id=request['token'],model_base64=encoded,model_sha256=model_hash,
        graph_sha256=hashlib.sha256(model.graph.SerializeToString()).hexdigest(),
        optimized_base64=base64.b64encode((root/'optimized.onnx').read_bytes()).decode(),
        optimized_sha256=hashlib.sha256((root/'optimized.onnx').read_bytes()).hexdigest(),
        input_sha256=input_hash,source=source,runtime=dict(name='onnxruntime',version=ort.__version__,
        build=ort.get_build_info()),provider_versions={x:dict(value=None,status='unavailable',
        reason='ORT does not expose individual provider versions; runtime build recorded') for x in providers},
        available_providers=ort.get_available_providers(),requested_providers=request['providers'],
        session_providers=providers,provider_options=session.get_provider_options(),optimization=request['optimization'],
        trace=trace,trace_sha256=trace_hash,profile_binding=_hash(dict(session_id=request['token'],
        model_sha256=model_hash,trace_sha256=trace_hash,input_sha256=input_hash,providers=providers)),
        completed_runs=p.warmup+p.runs)
