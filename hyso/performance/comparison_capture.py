"""Opt-in live CPU comparison producer; retained artifacts cannot be upgraded.

The receipt binds newly executed native measurements, their actual protocols,
the resident input and registered model state. It is an integrity record, not a
signature, dataset-origin attestation or guarantee of statistical independence.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
import uuid


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
        allow_nan=False,separators=(',',':')).encode('utf-8')).hexdigest()


def _environment():
    import os
    import torch
    return dict(scope='observed_host_descriptor_not_physical_attestation',host=platform.node(),
        machine=platform.machine(),processor=platform.processor(),platform=platform.platform(),
        python=platform.python_version(),torch=str(torch.__version__),logical_cpus=os.cpu_count(),
        threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
        backend=dict(mkldnn=torch.backends.mkldnn.enabled,
            deterministic=torch.are_deterministic_algorithms_enabled(),
            deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            float32_matmul_precision=torch.get_float32_matmul_precision()))


def _registered_state(model, inputs):
    import torch
    from hyso.benchmark.runner import _profile,_descriptor,_signature,_bits,_counts
    modules,entries=_profile(model,inputs.dtype)
    size=_counts(entries)['parameter_buffer_bytes']
    if size>256*1024**2:raise ValueError('comparison_model_budget')
    descriptor,state=_descriptor(modules,entries)
    gradients=tuple((id(t),None if t.grad is None else
        (_signature(t.grad),hashlib.sha256(_bits(t.grad).numpy().tobytes()).hexdigest()))
        for kind,_,t in entries if kind=='parameter')
    token=(descriptor,state,tuple((id(m),m.training) for _,m in modules),
        tuple((id(t),_signature(t)) for _,_,t in entries),gradients,
        _signature(inputs),hashlib.sha256(_bits(inputs).numpy().tobytes()).hexdigest())
    return descriptor,state,token,size


def _validate_comparison_capture(data):
    """Validate recorded consistency only; never access a model/file/device."""
    capture=data.get('provenance',{}).get('comparison_capture')
    fields={'format','schema_version','authority','producer','capture_id','run_id','payload_sha256',
        'model_state_sha256','post_model_state_sha256','structure_sha256','bound_input','environment',
        'checkpoint','checkpoint_observed_state_sha256','start_ns','end_ns','receipt_sha256'}
    def require(ok):
        if not ok:raise ValueError('invalid_cpu_comparison_capture')
    def sha(v):return type(v) is str and len(v)==64 and all(c in '0123456789abcdef' for c in v)
    require(type(capture) is dict and set(capture)==fields)
    require(sha(capture['receipt_sha256']) and _digest({k:v for k,v in capture.items() if k!='receipt_sha256'})==capture['receipt_sha256'])
    require(capture['format']=='hyso.native_comparison_capture' and type(capture['schema_version']) is int
        and capture['schema_version']==2 and capture['authority']=='native_capture'
        and capture['producer']=='hyso.performance.capture_cpu_comparison.v1')
    require(data.get('feature_ids') in (['P01'],['P02'],['P11']))
    for key in ('capture_id','run_id'):
        try:uuid.UUID(capture[key])
        except (TypeError,ValueError,AttributeError):raise ValueError('invalid_cpu_comparison_capture') from None
    require(type(capture['start_ns']) is int and type(capture['end_ns']) is int
        and 0<=capture['start_ns']<=capture['end_ns'])
    native=json.loads(json.dumps(data,allow_nan=False));native['provenance'].pop('comparison_capture')
    require(sha(capture['payload_sha256']) and _digest(native)==capture['payload_sha256'])
    p=data['provenance'];feature=data['feature_ids'][0]
    require(p['environment'].get('device')=='cpu')
    run=data['raw_evidence']['source_result_id'] if feature=='P01' else data['result_id']
    require(capture['run_id']==run and sha(capture['model_state_sha256'])
        and capture['model_state_sha256']==capture['post_model_state_sha256']==p['model']['weights_sha256']
        and capture['structure_sha256']==_digest(p['model']['structural_descriptor']))
    inp=capture['bound_input']
    require(type(inp) is dict and set(inp)=={'shape','stride','dtype','layout','sha256','role'})
    require(type(inp['shape']) is list and len(inp['shape'])==4
        and all(type(v) is int and v>0 for v in inp['shape'])
        and type(inp['stride']) is list and len(inp['stride'])==4
        and all(type(v) is int and v>=0 for v in inp['stride'])
        and inp['dtype'] in ('torch.float32','torch.float64') and inp['layout']=='torch.strided' and sha(inp['sha256']))
    require(inp['role']==('bound_workload_context_only' if feature=='P11' else 'measured_resident_input'))
    if feature!='P11':
        require(p['input']['shape']==inp['shape'] and p['input']['dtype']==inp['dtype']
            and p['input']['content_sha256']==inp['sha256'])
    env=capture['environment']
    require(type(env) is dict and set(env)=={'scope','host','machine','processor','platform','python','torch',
        'logical_cpus','threads','interop_threads','backend'})
    require(env['scope']=='observed_host_descriptor_not_physical_attestation'
        and all(type(env[k]) is str for k in ('host','machine','processor','platform','python','torch'))
        and env['python'] and env['torch'] and type(env['threads']) is int and env['threads']>0
        and type(env['interop_threads']) is int and env['interop_threads']>0)
    require(env['python']==p['environment'].get('python_version',p['environment'].get('python'))
        and env['torch']==p['environment'].get('torch_version',p['environment'].get('torch')))
    checkpoint=capture['checkpoint']
    require(type(checkpoint) is dict and set(checkpoint)=={'status','sha256','model_state_sha256','source','reason'})
    if checkpoint['status']=='native':
        require(sha(checkpoint['sha256']) and sha(capture['checkpoint_observed_state_sha256'])
            and checkpoint['model_state_sha256']==capture['checkpoint_observed_state_sha256']
            and checkpoint['source']=='hyso.load_training_checkpoint.snapshot_sha256.v1' and checkpoint['reason'] is None)
    else:
        require(checkpoint==dict(status='missing',sha256=None,model_state_sha256=None,
            source=None,reason='no_matching_trusted_load_receipt') and capture['checkpoint_observed_state_sha256'] is None)
    return capture


def capture_cpu_comparison(model, example_input, *, features=('P01','P02','P11'),
        latency_iterations=1000, warmup_iterations=2, repeats=1, throughput_protocol=None,
        footprint_artifact_path=None):
    """Execute new bounded CPU observations and return typed results by feature.

    Existing BenchmarkRunner/measure_throughput/measure_footprint defaults and
    retained JSON remain unchanged. No CUDA, training, worker, checkpoint load
    or automatic serialization occurs. One resident dense NCHW float32/64 input
    and the existing trusted eager-model profile are required. Each native
    protocol stays separate; no speedup is inferred from logical bytes.

    Checkpoint identity comes only from a current capture_evidence=True trusted
    load receipt. Missing receipts remain missing; dataset/split/preprocessing
    origin is not inferred. footprint_artifact_path measures an existing file;
    artifact-size comparison additionally needs that matching load receipt.
    P11's bound input is context only: no forward is run to count its bytes.

    HYSo Guide: comparison.models
    """
    import torch
    from hyso.benchmark import BenchmarkRunner
    from hyso.benchmark.runner import _tensor,_shape,StateMutationError
    from hyso._evaluation_evidence import model_hash,checkpoint_evidence
    from .inference import ThroughputProtocol,measure_throughput,analyze_latency
    from .memory import measure_footprint
    if type(features) not in (tuple,list) or not features or any(type(f) is not str or f not in ('P01','P02','P11') for f in features) or len(set(features))!=len(features):
        raise ValueError('explicit_unique_CPU_features_required')
    if type(example_input) is not torch.Tensor or example_input.device.type!='cpu' or example_input.dtype not in (torch.float32,torch.float64):
        raise ValueError('resident_CPU_float32_or_float64_input_required')
    _shape(tuple(example_input.shape));_tensor(example_input,example_input.dtype,input_tensor=True)
    if example_input.numel()*example_input.element_size()>8*1024**2:raise ValueError('comparison_input_budget')
    for count,minimum in ((latency_iterations,1),(warmup_iterations,0),(repeats,1)):
        if type(count) is not int or not minimum<=count<=4096:raise ValueError('bounded_comparison_counts_required')
    precision='fp64' if example_input.dtype==torch.float64 else 'fp32'
    if throughput_protocol is None:
        throughput_protocol=ThroughputProtocol(device='cpu',precision=precision,warmup_iterations=warmup_iterations,
            batch_count=1000,repeats=repeats)
    if type(throughput_protocol) is not ThroughputProtocol or throughput_protocol.device!='cpu' or throughput_protocol.precision!=precision:
        raise ValueError('matching_CPU_throughput_protocol_required')
    if throughput_protocol.duration_seconds is not None or throughput_protocol.batch_count>4096 or throughput_protocol.repeats>4096 or throughput_protocol.warmup_iterations>4096:
        raise ValueError('finite_bounded_CPU_batch_protocol_required')
    if ('P01' in features and repeats*(latency_iterations+warmup_iterations+1)>16384
            or 'P02' in features and throughput_protocol.repeats*(throughput_protocol.batch_count+throughput_protocol.warmup_iterations)+1>16384):
        raise ValueError('comparison_total_forward_budget')
    if footprint_artifact_path is not None and 'P11' not in features:raise ValueError('artifact_requires_P11')
    descriptor,state,token,size=_registered_state(model,example_input)
    if size>256*1024**2:raise ValueError('comparison_model_budget')
    environment=_environment();start=time.perf_counter_ns();capture_id=str(uuid.uuid4())
    loaded_state=model_hash(model);checkpoint=checkpoint_evidence(model,loaded_state)
    bound_input=dict(shape=list(example_input.shape),stride=list(example_input.stride()),dtype=str(example_input.dtype),
        layout='torch.strided',sha256=token[-1])
    results={}
    for feature in features:
        if feature=='P01':
            native=analyze_latency(BenchmarkRunner(model,device='cpu',example_input=example_input,precision=precision,
                warmup_iterations=warmup_iterations,measure_iterations=latency_iterations,repeats=repeats).run(),quantiles=(.5,.95,.99))
        elif feature=='P02':native=measure_throughput(model,example_input,protocol=throughput_protocol)
        else:native=measure_footprint(model,artifact_path=footprint_artifact_path,
            artifact_format='hyso.training_checkpoint' if footprint_artifact_path is not None else None)
        _,post_state,post_token,_=_registered_state(model,example_input)
        if post_token!=token or _environment()!=environment:
            raise StateMutationError('comparison_capture_state_or_environment_changed; no result published')
        data=native.to_dict();data['provenance']['comparison_capture']=dict(format='hyso.native_comparison_capture',
            schema_version=2,authority='native_capture',producer='hyso.performance.capture_cpu_comparison.v1',
            capture_id=capture_id,run_id=data['raw_evidence']['source_result_id'] if feature=='P01' else data['result_id'],
            payload_sha256=_digest(data),model_state_sha256=state,post_model_state_sha256=post_state,
            structure_sha256=_digest(data['provenance']['model']['structural_descriptor']),
            bound_input=dict(bound_input,role='bound_workload_context_only' if feature=='P11' else 'measured_resident_input'),
            environment=environment,checkpoint=checkpoint,
            checkpoint_observed_state_sha256=loaded_state if checkpoint['status']=='native' else None,
            start_ns=start,end_ns=time.perf_counter_ns())
        proof=data['provenance']['comparison_capture']
        proof['receipt_sha256']=_digest(proof)
        _validate_comparison_capture(data)
        results[feature]=type(native)(data)
    return results
