"""Isolated profiler capture and pure P12/P13 derivation from its plain evidence."""
from dataclasses import asdict,dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

from .cold import ColdStartRecipe,_terminate
from .inference import _base
from .result import (OperatorProfileResult,LayerProfileResult,OperationCountResult,
                     _integer,_positive,_plain,_write_export)
from ._operation_models import CONVENTION,summarize
from ._profile_evidence import validate_events,validate_modules,module_timings,correlate,cuda_event_timings


@dataclass(frozen=True)
class ProfilingProtocol:
    """Capture bounds plus opt-in resource limits; None imposes no resource quota."""
    warmup_iterations:int=1
    measure_iterations:int=3
    timeout_seconds:float=60.
    cpu_memory_limit_bytes:int|None=None
    input_limit_bytes:int|None=None
    cuda_memory_fraction:float|None=None
    max_events:int=20000
    max_trace_bytes:int=16*1024**2
    profile_memory:bool=False
    allocation_lifetimes:bool=False

    def __post_init__(self):
        if type(self.profile_memory) is not bool:raise ValueError("profile_memory must be boolean")
        if type(self.allocation_lifetimes) is not bool or self.allocation_lifetimes and not self.profile_memory:raise ValueError('allocation_lifetimes requires profile_memory=True')
        _integer(self.warmup_iterations);_integer(self.measure_iterations,1);_positive(self.timeout_seconds)
        if self.warmup_iterations+self.measure_iterations>100 or self.timeout_seconds>600:
            raise ValueError('bounded profile: at most100 forwards/600 seconds')
        if self.cpu_memory_limit_bytes is not None:_integer(self.cpu_memory_limit_bytes,256*1024**2)
        if self.input_limit_bytes is not None:_integer(self.input_limit_bytes,1)
        if self.cuda_memory_fraction is not None:
            _positive(self.cuda_memory_fraction)
            if self.cuda_memory_fraction>1:raise ValueError('invalid allocator budget')
        _integer(self.max_events,1);_integer(self.max_trace_bytes,1)
        if self.max_events>100000 or self.max_trace_bytes>64*1024**2:raise ValueError('evidence budget too large')


class ProfilingError(RuntimeError):
    def __init__(self,diagnostics):
        self.diagnostics=_plain(diagnostics)
        super().__init__(json.dumps(self.diagnostics,sort_keys=True))


def _hash(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def profile_operators(recipe,*,protocol=ProfilingProtocol(),trace_path=None):
    """Fresh Windows worker; no live caller model or hooks cross the boundary.

    Owned model uses eval/inference mode. Torch profiler self/total CPU timings
    are instrumented costs, not BenchmarkRunner latency. Only exported kernel
    activities qualify as kernel evidence; legacy event.kernels is not used.
    """
    return _execute_profile(recipe,protocol,trace_path=trace_path)


def _execute_profile(recipe,protocol,*,trace_path=None,experiment_mode=None):
    if type(recipe) is not ColdStartRecipe or type(protocol) is not ProfilingProtocol:
        raise TypeError('expected ColdStartRecipe and ProfilingProtocol')
    if trace_path is not None and Path(trace_path).exists():raise FileExistsError(trace_path)
    receipt=dict(status='failed',reason=None,exit_code=None,worker_pid=None,launcher_pid=None,
                 worker_phase=None,process_terminated=True,stderr_tail='',cleanup={})
    start=time.perf_counter();worker=None
    with tempfile.TemporaryDirectory(prefix='hyso-profile-') as directory:
        root=Path(directory);request=root/'request.json';output=root/'result.json';trace=root/'trace.json';ready=root/'ready.json'
        request.write_text(json.dumps(dict(recipe=recipe.to_dict(),protocol=asdict(protocol),experiment_mode=experiment_mode)),encoding='utf-8')
        command=[sys.executable,'-I','-B',str(Path(__file__).with_name('_profile_worker.py')),
                 str(request),str(output),str(trace),str(ready),str(Path(__file__).resolve().parents[2])]
        with tempfile.TemporaryFile() as stdout,tempfile.TemporaryFile() as stderr:
            try:
                worker=subprocess.Popen(command,stdout=stdout,stderr=stderr,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                receipt['launcher_pid']=worker.pid
                try:worker.wait(timeout=protocol.timeout_seconds)
                except subprocess.TimeoutExpired:
                    receipt.update(status='timed_out',reason='owned worker timeout');_terminate(worker);worker.wait(timeout=10)
                receipt['exit_code']=worker.returncode
                if ready.exists():
                    ready_data=json.loads(ready.read_text(encoding='utf-8'))
                    receipt.update(worker_pid=ready_data['pid'],worker_phase=ready_data['phase'])
                if receipt['status']!='timed_out':
                    if worker.returncode!=0:receipt.update(status='crashed',reason='unexpected worker exit; not OOM classification')
                    elif output.exists():
                        if output.stat().st_size>protocol.max_trace_bytes*2:raise ValueError('result exceeds evidence budget')
                        body=_plain(json.loads(output.read_text(encoding='utf-8')))
                        if body['worker_pid']!=receipt['worker_pid']:raise ValueError('worker identity mismatch')
                        receipt.update(status=body['status'],reason=body['reason'],cleanup=body['cleanup'])
                    else:receipt['reason']='missing worker result'
            finally:
                if worker is not None:
                    if worker.poll() is None:_terminate(worker)
                    worker.wait(timeout=10);receipt['process_terminated']=worker.poll() is not None
                stderr.seek(0,2);size=stderr.tell();stderr.seek(max(0,size-4096))
                receipt['stderr_tail']=stderr.read().decode('utf-8',errors='replace')
        receipt['execution_seconds']=time.perf_counter()-start
        if receipt['status']!='completed':raise ProfilingError(receipt)
        data=body['result']
        if experiment_mode=='control':
            data['worker']=receipt
            return data
        data['raw_evidence']['worker']=receipt
        meta=data['raw_evidence']['trace'];meta['path']=str(Path(trace_path).resolve()) if trace_path is not None else None
        meta['retention']='published' if trace_path is not None else 'generated_not_retained'
        result=OperatorProfileResult(data)
        if trace_path is not None:
            # read_text performs universal-newline conversion on Windows. Keep
            # the original trace bytes (including CRLF) for the recorded SHA.
            contents=trace.read_bytes().decode('utf-8')
            if hashlib.sha256(trace.read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('trace checksum mismatch')
            _write_export(trace_path,contents,False)
        return result


def _source(source,protocol):
    if type(source) is ColdStartRecipe:return profile_operators(source,protocol=protocol).to_dict()
    if type(source) is not OperatorProfileResult:raise TypeError('expected recipe or OperatorProfileResult')
    return OperatorProfileResult(source.to_dict()).to_dict()


def _derived(kind,feature,source,raw,measurements):
    data=_base(kind,feature,dict(id='derive_from_profile',version=1,scope='observed_profile_evidence',
        configuration=source['protocol']['configuration'],recipe=source['protocol']['recipe'],
        convention=CONVENTION if kind=='operation_count' else 'module_inclusive_minus_direct_module_children'),source['provenance'])
    data['identity']['source_result_id']=source['result_id']
    data.update(requested=source['requested'],actual=source['actual'],measurements=measurements,
        raw_evidence=dict(source_result_id=source['result_id'],source_sha256=_hash(source),
            trace=source['raw_evidence']['trace'],**raw))
    data['diagnostics']['limitations']=list(source['diagnostics']['limitations'])
    return data


def analyze_flops(source,*,protocol=ProfilingProtocol()):
    """Analytic observed-ATen operation models. No timing or hardware FLOP counter."""
    s=_source(source,protocol);calls=s['raw_evidence']['aten_calls']
    rows,m=summarize(calls,s['actual']['measured_forwards'],s['provenance']['input']['shape'][0])
    data=_derived('operation_count','P12',s,dict(aten_calls=calls,operations=rows),m)
    data['protocol']['operation_model_version']=2
    return OperationCountResult(data)


def _layer_metrics(inventory,invocations,*,evidence=None,cuda_events=None):
    modules=[]
    for item in inventory:
        calls=[r for r in invocations if r['module_id']==item['module_id']]
        def total(key):
            return math.fsum(r[key] for r in calls) if calls and all(r[key] is not None for r in calls) else None
        measured=bool(calls) and all(r['cpu_total_us'] is not None and r['cpu_exclusive_us'] is not None for r in calls)
        modules.append(dict(item,invocation_count=len(calls),cpu_total_us=total('cpu_total_us'),
            status='measured' if measured else 'not_executed' if not calls else 'partial',
            reason=None if measured else 'no_observed_invocation' if not calls else 'invalid_or_missing_CPU_duration',
            cpu_exclusive_us=total('cpu_exclusive_us'),cuda_activity_us=total('cuda_activity_us')))
        if cuda_events is not None:
            status=('not_requested' if not cuda_events['requested'] else 'not_executed' if not calls else
                    'measured' if all(c['cuda_elapsed_status']=='measured' for c in calls) else
                    'unsupported' if cuda_events['scope_reason'] else 'partial')
            modules[-1].update(cuda_elapsed_us=total('cuda_elapsed_us'),cuda_elapsed_status=status,
                cuda_elapsed_reason=cuda_events['scope_reason'] or ('missing_or_invalid_CUDA_Event_interval' if status=='partial' else None),
                cuda_exclusive_elapsed_us=None)
    ranking=[r['module_id'] for r in sorted((r for r in modules if r['cpu_exclusive_us'] is not None),
                                          key=lambda r:r['cpu_exclusive_us'],reverse=True)]
    complete=all(r['cpu_total_us'] is not None and (evidence is None or r['cpu_exclusive_us'] is not None) for r in invocations)
    result=dict(status='measured' if complete else 'partial',
        reason=None if complete else 'missing_or_invalid_module_cpu_event',
        modules=modules,ranking=ranking,ranking_basis='cpu_exclusive_us_instrumented',unit='us',
        cuda_layer_status='unsupported',cuda_layer_reason='no_verified_kernel_to_module_attribution')
    if evidence is not None:
        result.update(cuda_attribution=evidence['cuda_attribution'],
            layer_memory=dict(status='unsupported',allocation_bytes=None,retained_bytes=None,peak_bytes=None,
                reason='capture_profile_memory_false; allocations_retained_and_peak_are_distinct'),
            overhead=evidence['overhead'])
        if evidence['cuda_attribution']['mapped_kernel_count']:
            result.update(cuda_layer_status='partial',
                cuda_layer_reason='observed_kernel_links_only; complete_layer_activity_not_claimed')
    if cuda_events is not None:
        valid=sum(r['cuda_elapsed_status']=='measured' for r in invocations)
        status=('not_requested' if not cuda_events['requested'] else 'unsupported' if cuda_events['scope_reason'] else
                'measured' if valid==len(invocations) else 'partial')
        result.update(cuda_event_timing=dict(status=status,reason=cuda_events['scope_reason'] or
            ('missing_or_invalid_CUDA_Event_interval' if status=='partial' else None),
            device=cuda_events['device'],stream_id=cuda_events['stream_id'],clock=cuda_events['clock'],unit='us',
            invocation_count=len(invocations),measured_invocations=valid,
            aggregation='sum_of_inclusive_event_intervals_over_invocations; not_kernel_time_or_inference_latency',
            exclusive='unsupported; not_derived_from_inclusive_intervals',validation=cuda_events['validation']),
            cuda_elapsed_ranking=[m['module_id'] for m in sorted((m for m in modules if m['cuda_elapsed_us'] is not None),
                key=lambda m:m['cuda_elapsed_us'],reverse=True)])
        if complete and status in ('partial','unsupported'):
            result.update(status='partial',reason='CUDA_Event_layer_timing_unavailable_or_invalid')
    return result


def profile_layers(source,*,protocol=ProfilingProtocol(),memory_analysis_version=None):
    """Module invocation tree from the same capture; shared calls remain distinct.

    Opt into allocation-event attribution with ProfilingProtocol(profile_memory=True)
    when capturing a recipe. Existing captures cannot retroactively gain memory
    evidence. Ordinary analysis runs no extra forward or overhead experiment.
    memory_analysis_version=2 explicitly reanalyzes existing version4 allocation
    events with chronological observed net high-water; it needs no new capture.
    allocation_lifetimes=True with profile_memory=True opts into new allocation
    cohorts confirmed alive at module exit and native CUDA allocator peak history.
    This disables profiler shape-reference retention, preserves dispatch shapes,
    adds diagnostic overhead, and never claims physical RAM/VRAM peaks.
    """
    if memory_analysis_version is not None and (type(memory_analysis_version) is not int or memory_analysis_version!=2):
        raise ValueError('memory_analysis_version must be None or 2')
    if memory_analysis_version==2 and type(source) is ColdStartRecipe and not protocol.profile_memory:
        raise ValueError('allocation event capture must be requested')
    s=_source(source,protocol);raw=s['raw_evidence']
    extra={};evidence=None;cuda_events=None
    version=s['protocol']['instrumentation'].get('evidence_version',1)
    if memory_analysis_version==2:
        if version<4:raise ValueError('existing capture has no allocation event evidence')
        version=max(version,5)
    if version>=2:
        evidence=dict(cuda_attribution=s['measurements']['cuda_attribution'],overhead=s['measurements']['overhead'])
        extra=dict(events=raw['events'],trace_events=raw['trace_events'],profile_measurements=evidence)
    if version>=3:
        cuda_events=raw['cuda_event_timing'];extra['cuda_event_timing']=cuda_events
    if version>=4:
        extra['memory_events']=raw['memory_events']
        extra['memory_device']=s['provenance']['environment']['device']
    data=_derived('layer_profile','P13',s,
        dict(module_inventory=raw['module_inventory'],invocations=raw['invocations'],**extra),
        _layer_metrics(raw['module_inventory'],raw['invocations'],evidence=evidence,cuda_events=cuda_events))
    if version>=4:
        from ._layer_memory import attribute_memory
        data['measurements']['layer_memory']=attribute_memory(raw['memory_events'],raw['trace_events'],
            raw['invocations'],raw['module_inventory'],s['provenance']['environment']['device'],analysis_version=2 if version>=5 else 1)
        for module in data['measurements']['modules']:
            module['memory']=next(m['domains'] for m in data['measurements']['layer_memory']['modules'] if m['module_id']==module['module_id'])
    if version>=6:
        from ._layer_allocator import derive_allocator
        extra_allocation=raw['allocation_evidence']
        data['raw_evidence']['allocation_evidence']=extra_allocation
        diagnostics=derive_allocator(extra_allocation,raw['memory_events'],raw['trace_events'],raw['invocations'],raw['module_inventory'],s['provenance']['environment']['device'])
        data['measurements']['allocator_diagnostics']=diagnostics
        for module in data['measurements']['modules']:
            module['allocator']=next(m for m in diagnostics['modules'] if m['module_id']==module['module_id'])
    if evidence is not None:data['protocol']['profile_evidence_version']=version
    return LayerProfileResult(data)


def _validate_profile(data,kind):
    p=ProfilingProtocol(**data['protocol']['configuration'])
    r=ColdStartRecipe(**data['protocol']['recipe']).to_dict();raw=data['raw_evidence']
    expected=dict(measured_forwards=p.measure_iterations,warmup_forwards=p.warmup_iterations)
    if data['actual']!=expected or data['requested']!=expected:raise ValueError('invalid profile forward ledger')
    if kind=='operator_profile':
        if data['protocol']['id']!='isolated_torch_profiler':raise ValueError('invalid profiler protocol')
        if not raw['worker']['process_terminated'] or raw['worker']['exit_code']!=0:raise ValueError('worker did not finish normally')
        if not all(raw['safety'].values()):raise ValueError('state/ownership safety not verified')
        validate_events(raw['events'])
        if len(raw['events'])>p.max_events or len(raw['aten_calls'])>p.max_events:raise ValueError('evidence budget exceeded')
        if data['measurements']['operators']!=_operator_metrics(raw['events']):raise ValueError('operator aggregation mismatch')
        overhead=data['measurements']['overhead']
        _positive(overhead['collector_lifecycle_wall_ms'])
        if len(overhead['instrumented_host_forward_windows_ms'])!=p.measure_iterations:raise ValueError('invalid host window ledger')
        for v in overhead['instrumented_host_forward_windows_ms']:_positive(v)
        if overhead['perturbation_ms'] is not None or overhead['perturbation_status']!='not_measured':raise ValueError('unmeasured perturbation cannot be numeric')
        for event in raw['events']:
            for key in ('cpu_total_us','self_cpu_us'):
                v=event[key]
                if v is not None and (type(v) not in (float,int) or v<0):raise ValueError('invalid CPU event duration')
        if data['measurements']['cuda_kernels']!=raw['kernel_events']:raise ValueError('unverified kernel evidence')
        cuda=_cuda_metrics(raw['kernel_events'],r['device']!='cpu')
        if data['measurements']['cuda_activity']!=cuda:raise ValueError('invalid CUDA availability/timing claim')
        status,reason=_profile_status(raw['events'],cuda)
        if (data['measurements']['status'],data['measurements']['reason'])!=(status,reason):raise ValueError('invalid profile status')
        inventory=raw['module_inventory'];invocations=raw['invocations']
        validate_modules(inventory,invocations,p.measure_iterations)
        version=data['protocol']['instrumentation'].get('evidence_version',1)
        if type(version) is not int or version not in (1,2,3,4,5,6):raise ValueError('invalid profiler evidence version')
        if version<4 and p.profile_memory:raise ValueError('memory tracing requires evidence version4')
        if version>=2:
            expected_rows=module_timings(inventory,invocations,raw['events'],p.measure_iterations)
            if version>=3:
                expected_rows=cuda_event_timings(expected_rows,raw['cuda_event_timing'],_event_device(data,r))
            if expected_rows!=invocations:
                raise ValueError('unverified module timing evidence')
            _validate_trace_evidence(raw,data['measurements']['cuda_attribution'],p.max_events)
            if version>=4:
                from ._layer_memory import attribute_memory
                if not p.profile_memory or data['protocol']['instrumentation']['profile_memory'] is not True:raise ValueError('memory tracing not requested')
                _validate_memory_indices(raw,p.max_events)
                if data['measurements']['layer_memory']!=attribute_memory(raw['memory_events'],raw['trace_events'],invocations,inventory,r['device'] if r['device']!='cuda' else 'cuda:0',analysis_version=2 if version>=5 else 1):
                    raise ValueError('unverified allocation attribution')
        if version>=6:
            from ._layer_allocator import derive_allocator
            if not p.allocation_lifetimes:raise ValueError('allocator evidence not requested')
            if data['protocol']['instrumentation']['record_shapes'] is not False or raw['safety'].get('allocation_history_closed') is not True:
                raise ValueError('lifetime capture must disable shape retention and close history')
            if data['measurements']['allocator_diagnostics']!=derive_allocator(raw['allocation_evidence'],raw['memory_events'],raw['trace_events'],invocations,inventory,_event_device(data,r)):
                raise ValueError('invalid allocator lifetime/peak claim')
        mids={m['module_id'] for m in inventory}
        if len(mids)!=len(inventory):raise ValueError('duplicate module identity')
        if [v['invocation_id'] for v in invocations]!=list(range(len(invocations))):raise ValueError('invalid invocation IDs')
        roots=0
        for row in invocations:
            if row['module_id'] not in mids or not 0<=row['measurement_index']<p.measure_iterations:raise ValueError('invalid module association')
            if row['parent_invocation_id'] is None:roots+=1
            elif not 0<=row['parent_invocation_id']<row['invocation_id']:raise ValueError('invalid call-tree parent')
        if roots!=p.measure_iterations:raise ValueError('incomplete module forward ledger')
        for i,c in enumerate(raw['aten_calls']):
            if c['call_index']!=i or not 0<=c['measurement_index']<p.measure_iterations:raise ValueError('invalid dispatch association')
        if version>=2:return
        for row in raw['invocations']:
            children=[x for x in raw['invocations'] if x['parent_invocation_id']==row['invocation_id']]
            value=row['cpu_total_us']
            exclusive=None if value is None or any(x['cpu_total_us'] is None for x in children) else value-sum(x['cpu_total_us'] for x in children)
            if exclusive is not None and exclusive<0:exclusive=None
            if row['cpu_exclusive_us']!=exclusive:raise ValueError('invalid module nesting accounting')
        return
    if data['protocol']['id']!='derive_from_profile':raise ValueError('invalid derived profile protocol')
    if data['identity']['source_result_id']!=raw['source_result_id'] or len(raw['source_sha256'])!=64:
        raise ValueError('missing source identity')
    if kind=='operation_count':
        version=data['protocol'].get('operation_model_version',1)
        if type(version) is not int or version not in (1,2):raise ValueError('invalid operation model version')
        rows,m=summarize(raw['aten_calls'],p.measure_iterations,r['input_shape'][0],model_version=version)
        if raw['operations']!=rows or data['measurements']!=m or data['protocol']['convention']!=CONVENTION:
            raise ValueError('invalid analytic operation-count claim')
    else:
        validate_modules(raw['module_inventory'],raw['invocations'],p.measure_iterations)
        version=data['protocol'].get('profile_evidence_version',1);evidence=None;cuda_events=None
        if type(version) is not int or version not in (1,2,3,4,5,6):raise ValueError('invalid layer evidence version')
        if version<4 and p.profile_memory:raise ValueError('memory tracing requires evidence version4')
        if version>=2:
            expected_rows=module_timings(raw['module_inventory'],raw['invocations'],raw['events'],p.measure_iterations)
            if version>=3:
                cuda_events=raw['cuda_event_timing']
                expected_rows=cuda_event_timings(expected_rows,cuda_events,_event_device(data,r))
            if expected_rows!=raw['invocations']:
                raise ValueError('unverified layer event timings')
            evidence=raw['profile_measurements']
            _validate_trace_evidence(raw,evidence['cuda_attribution'],p.max_events)
            _validate_overhead(evidence['overhead'],p.measure_iterations)
        expected_metrics=_layer_metrics(raw['module_inventory'],raw['invocations'],evidence=evidence,cuda_events=cuda_events)
        if version>=4:
            from ._layer_memory import attribute_memory
            _validate_memory_indices(raw,p.max_events)
            device=r['device'] if r['device']!='cuda' else 'cuda:0'
            if raw['memory_device']!=device or not p.profile_memory:raise ValueError('memory context mismatch')
            expected_metrics['layer_memory']=attribute_memory(raw['memory_events'],raw['trace_events'],raw['invocations'],raw['module_inventory'],device,analysis_version=2 if version>=5 else 1)
            for module in expected_metrics['modules']:
                module['memory']=next(m['domains'] for m in expected_metrics['layer_memory']['modules'] if m['module_id']==module['module_id'])
        if version>=6:
            from ._layer_allocator import derive_allocator
            if not p.allocation_lifetimes:raise ValueError('allocator evidence not requested')
            diagnostics=derive_allocator(raw['allocation_evidence'],raw['memory_events'],raw['trace_events'],raw['invocations'],raw['module_inventory'],device)
            expected_metrics['allocator_diagnostics']=diagnostics
            for module in expected_metrics['modules']:
                module['allocator']=next(m for m in diagnostics['modules'] if m['module_id']==module['module_id'])
        if data['measurements']!=expected_metrics:
            raise ValueError('invalid module totals/ranking')


def _event_device(data,recipe):
    expected='cuda:0' if recipe['device']=='cuda' else recipe['device']
    if data['provenance']['environment']['device']!=expected:raise ValueError('CUDA Event provenance device mismatch')
    return expected


def _validate_overhead(overhead,forwards):
    _positive(overhead['collector_lifecycle_wall_ms'])
    if len(overhead['instrumented_host_forward_windows_ms'])!=forwards:raise ValueError('invalid host windows')
    for v in overhead['instrumented_host_forward_windows_ms']:_positive(v)
    if overhead['perturbation_ms'] is not None or overhead['perturbation_status']!='not_measured':
        raise ValueError('unmeasured overhead cannot be numeric')


def _validate_trace_evidence(raw,attribution,max_events):
    events=raw['trace_events']
    if len(events)>max_events:raise ValueError('trace event budget exceeded')
    if raw['trace'].get('timestamp_unit')!='us' or raw['trace'].get('timestamp_domain')!='single_exported_chrome_trace; native_FunctionEvent_clock_is_separate':
        raise ValueError('invalid profiler timestamp domain')
    if any(type(e['trace_index']) is not int or not 0<=e['trace_index']<raw['trace']['event_count'] for e in events):
        raise ValueError('trace event index outside capture')
    if correlate(events,raw['invocations'])!=attribution:raise ValueError('invalid kernel attribution')
    if 'kernel_events' in raw:
        kernels={k['trace_index']:k for k in raw['kernel_events']}
        trace_kernels=[e for e in events if e['category']=='kernel']
        if len(kernels)!=len(raw['kernel_events']) or len(kernels)!=len(trace_kernels):raise ValueError('kernel ledger mismatch')
        for e in trace_kernels:
            k=kernels.get(e['trace_index'])
            if k is None or any(k[key]!=e[key] for key in ('name','duration_us','start_us','device','stream','raw_correlation','raw_external_id','raw_event_id')):
                raise ValueError('kernel metadata mismatch')


def _cuda_metrics(kernels,requested):
    for k in kernels:
        if k['category']!='kernel' or k['phase']!='X' or type(k['name']) is not str or not k['name']:
            raise ValueError('only actual exported kernel activities qualify')
        if k['duration_us'] is not None and (type(k['duration_us']) not in (int,float) or not math.isfinite(k['duration_us']) or k['duration_us']<0):
            raise ValueError('invalid kernel duration')
    if not requested and kernels:raise ValueError('unrequested device activity')
    valid=all(k['duration_us'] is not None and ('trace_index' not in k or
        (type(k['device']) is int and k['device']>=0 and type(k['stream']) is int and k['stream']>=0
         and k.get('start_us') is not None and type(k['start_us']) in (int,float)
         and math.isfinite(k['start_us']) and k['start_us']>=0)) for k in kernels)
    status=('not_requested' if not requested else 'unsupported' if not kernels else 'measured' if valid else 'partial')
    return dict(status=status,reason='no_exported_CUDA_kernel_activity' if status=='unsupported' else
        ('invalid_kernel_duration_or_identity_or_timestamp' if any('trace_index' in k for k in kernels) else 'invalid_kernel_duration') if status=='partial' else None,
        total_kernel_us=math.fsum(k['duration_us'] for k in kernels) if status=='measured' else None,
        kernel_count=len(kernels) if kernels else None,
        coverage='observed_exported_trace_only; not full GPU coverage',cpu_to_cuda_correlation='not_asserted')


def _profile_status(events,cuda):
    if any(e['cpu_total_us'] is None or e['self_cpu_us'] is None for e in events if e['device_type']=='CPU'):
        return 'partial','invalid_or_missing_CPU_duration'
    if cuda['status'] in ('unsupported','partial'):return 'partial','cuda_kernel_activity_unavailable_or_invalid'
    return 'measured',None


def _operator_metrics(events):
    groups={}
    for event in events:
        if event['device_type']!='CPU' or event['scope']!=0:continue
        group=groups.setdefault(event['name'],[]);group.append(event)
    rows=[]
    for name,events in groups.items():
        def total(key):return math.fsum(e[key] for e in events) if all(e[key] is not None for e in events) else None
        rows.append(dict(name=name,count=len(events),self_cpu_us=total('self_cpu_us'),cpu_total_us=total('cpu_total_us'),
            shapes=[e['input_shapes'] for e in events],unit='us',scope='instrumented_CPU_operator_events',
            status='measured' if all(e['self_cpu_us'] is not None and e['cpu_total_us'] is not None for e in events) else 'partial'))
    return sorted(rows,key=lambda r:r['self_cpu_us'] if r['self_cpu_us'] is not None else -1,reverse=True)


def _validate_memory_indices(raw,budget):
    events=raw['memory_events']
    if len(events)>budget:raise ValueError('memory evidence budget exceeded')
    trace_indices={e['trace_index'] for e in raw['trace_events']}
    for e in events:
        if type(e['trace_index']) is not int or not 0<=e['trace_index']<raw['trace']['event_count'] or e['trace_index'] in trace_indices:
            raise ValueError('invalid memory/trace association')


def layer_profiling_capabilities():
    """Build/API capabilities only; no CUDA initialization or fake native PASS."""
    import torch
    activities=sorted(str(a).split('.')[-1] for a in torch.profiler.supported_activities())
    return dict(torch=str(torch.__version__),supported_activities=activities,
        cuda_kernel_trace='available_requires_actual_trace_validation' if 'CUDA' in activities else 'unsupported_backend_capability',
        kineto_available=bool(torch.profiler.kineto_available()),
        cuda_allocator_history_API=torch.version.cuda is not None and hasattr(torch._C,'_cuda_memorySnapshot') and
            all(hasattr(torch.cuda.memory,k) for k in ('_snapshot','_record_memory_history')),
        allocator_history_runtime_validation='not_performed_by_capability_query',
        kernel_trace_requires='runtime_build_with_CUDA_profiler_activity_and_actual_exported_kernel_events',
        CUDA_Event_is_kernel_time=False,external_profiler_required=False)
