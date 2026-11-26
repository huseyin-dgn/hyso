"""Scope-specific PASS gates, shared by worker and post-cleanup validator.

Native result constructors validate their complete ledgers. These additional
gates require the functional observations promised by this small CUDA scope.
No Torch imports, execution, or acceptance based on a device label alone.
"""
import math
import hashlib
import json
from .identity import gpu_uuid


class SafetyViolation(RuntimeError):pass


def verify_nested_safety(value):
    if isinstance(value,list):
        for row in value:verify_nested_safety(row)
    elif isinstance(value,dict):
        if value.get('failure_kind') in ('oom','device_error'):
            raise SafetyViolation('NESTED_CUDA_OR_RESOURCE_FAILURE')
        cleanup=value.get('cleanup')
        if isinstance(cleanup,dict) and 'active_processes' in cleanup and cleanup['active_processes']!=0:
            raise SafetyViolation('NESTED_OWNED_CLEANUP_UNVERIFIED')
        for key in ('monitor_reason','monitor_outcome'):
            if value.get(key):raise SafetyViolation('NESTED_MONITOR_STOP: '+str(value[key]))
        if 'launcher_pid' in value and 'cleanup' in value and value.get('reason'):
            raise SafetyViolation('NESTED_CONTROLLER_STOP: '+str(value['reason']))
        for child in value.values():verify_nested_safety(child)


def verify_feature(feature,d,runtime):
    def need(value,reason):
        if not value:raise ValueError(feature+': '+reason)
    def number(value,positive=False):
        return type(value) in (int,float) and math.isfinite(value) and (value>0 if positive else value>=0)
    def native_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    def training(source):
        need(source['status']=='measured' and source['actual_epochs']==1 and len(source['epochs'])==1,'one completed training epoch')
        need(source['provenance']['runtime']['device']=='cuda:0','training device')
        need(source['instrumentation']['pending_event_pairs']==0 and source['instrumentation']['event_drains']>0,'completed CUDA training events')
        need(bool(source['observations']),'raw training observations')
        need(source['counters']['after']['applied']>source['counters']['before']['applied'],'real optimizer updates')
    need(isinstance(d,dict) and bool(d),'missing functional payload')
    verify_nested_safety(d)
    if feature=='P29':
        before=d['source_weights_before'];after=d['source_weights_after'];d=d['comparison']
        need(len(before)==2 and before==after,'both original source models unchanged')
        need(before==[d['sources'][s]['benchmark_model_sha256'] for s in ('baseline','candidate')],'original weights bound to both benchmark witnesses')
    if feature in ('P24','P27','P28','P29'):
        need(d['source_sha256']==native_hash(d['sources']),'quality source hash')
    if feature=='ADMISSION':
        need(d['probe']==dict(completed=True,input=1,output=2,elements=1),'completed arithmetic probe')
        need(d['loaded_binaries']==runtime['loaded_binaries'],'runtime identity')
    elif feature in ('Builder','GraphBuilder'):
        need(d['shape']==[2,2] and d['device']=='cuda:0' and d['finite'] is True and d['gradients'] is True,'finite forward and backward')
    elif feature=='P01':
        rows=d['raw_evidence']['latency_samples_ms']
        need(sum(map(len,rows))==1000 and all(number(v,True) for row in rows for v in row),'1000 actual latency samples')
        need(all(m['value'] is not None for m in d['measurements']),'eligible tail measurements')
    elif feature=='P02':
        need(d['actual']['completed_batches']==1000 and d['actual']['pending_batches']==0,'completed throughput workload')
        need(d['measurements']['status']=='measured' and number(d['measurements']['value'],True) and bool(d['raw_evidence']['windows']),'measured completion rate')
    elif feature in ('P03','P04'):
        rows=d['raw_evidence']['samples']
        need(d['actual']['completed']==100 and d['actual']['pending']==0 and len(rows)==100,'100 completed pipeline samples')
        stages=[stage for row in rows for stage in row['stages'] if stage['name'] in ('transfer','inference')]
        need(len(stages)==200 and all(s['device_event_status']=='measured' and number(s['device_event_ms'],True) for s in stages),'completed transfer/inference event intervals')
    elif feature=='P05':
        need(d['controller']['cleanup_verified'] and d['controller']['cleanup']['active_processes']==0 and d['controller']['exit_code']==0,'fresh worker cleanup')
        result=d['result'];need(result['actual']['measured']==1 and result['actual']['trials']==1,'one measured cold trial')
        row=result['raw_evidence']['trials'][0]
        need(row['provenance']['device']=='cuda:0' and row['provenance']['cuda_allocator_fraction']==.15,'cold CUDA allocator policy')
        need(row['provenance']['cuda_initialized_after_import'] is False and row['provenance']['cuda_initialized_after_construction'] is False,'cold initialization order')
        need(all(row['phases'][p]['status']=='measured' and number(row['phases'][p]['elapsed_ms']) for p in ('runtime_prepare','first_inference','warm_reference')),'cold phase measurements')
        need(len(row['warm_samples_ms'])==2 and all(number(v,True) for v in row['warm_samples_ms']),'warm reference samples')
        need(d['seal']['output_sha256']==d['output_sha256'] and d['seal']['uuid']==runtime['uuid'] and d['seal']['loaded_binaries']==runtime['loaded_binaries'],'cold runtime binding')
        need([r['kind'] for r in d['journal']]==['worker_start','cold_result','session_finish'] and d['journal'][1]['output_sha256']==d['output_sha256'],'bound cold journal')
    elif feature in ('P06','P07','P08'):
        points=d['raw_evidence']['points'];need(len(points)==(2 if feature=='P07' else 3),'complete sweep points')
        for point in points:
            result=point['result']
            need(point['status']=='measured' and result['actual']['samples']==20 and result['provenance']['environment']['device']=='cuda:0','measured CUDA sweep point')
            need(bool(result['raw_evidence']) and bool(result['measurements']),'sweep measurement ledger')
            raw=result['raw_evidence'];benchmark=raw.get('benchmark')
            timings=benchmark['timing']['latency_samples_ms'] if benchmark is not None else raw['special']['latency_samples_ms']
            need(sum(map(len,timings))==20 and all(number(v,True) for row in timings for v in row),'actual sweep timing samples')
    elif feature=='P09':
        mem=d['cuda_memory']
        need(mem['status']=='measured' and mem['counter_reset_performed'] and mem['peak_reserved_bytes']>=mem['peak_allocated_bytes']>0,'actual allocator peaks')
    elif feature=='P10':
        rows=d['raw_evidence']['observations']
        need(bool(rows) and all(number(r['rss_bytes'],True) and r['query_end_ns']>=r['query_start_ns'] for r in rows),'native RSS observations')
    elif feature=='P11':
        gpu=d['resident_GPU_source'];logical=d['logical_inventory']
        need(gpu['model']['weights_sha256']==logical['provenance']['model']['weights_sha256'],'GPU model/logical inventory identity')
        need(logical['actual']['tensor_entries']>0 and bool(logical['raw_evidence']['logical']),'actual logical inventory')
        need(number(logical['measurements']['parameter_buffer_bytes']['value'],True),'logical byte measurement')
    elif feature=='P12':
        need(d['actual']['measured_forwards']==2 and bool(d['raw_evidence']['aten_calls']) and bool(d['raw_evidence']['operations']),'observed ATen operations and logical costs')
        need(any(number(r.get('flops'),True) for r in d['raw_evidence']['operations']),'supported observed operation costs')
    elif feature=='P13':
        m=d['measurements']['cuda_event_timing']
        need(m['status']=='measured' and m['measured_invocations']==m['invocation_count']>0,'completed inclusive layer events')
        need(bool(d['raw_evidence']['invocations']),'layer invocation ledger')
        need(all(number(r['cuda_elapsed_us']) for r in d['raw_evidence']['invocations']),'actual inclusive event durations')
    elif feature=='P14':
        from .p14_probe import verify_capture
        verify_capture(d,runtime)
    elif feature=='P15':
        rows=d['raw_evidence']['copies']
        need(bool(rows) and d['actual']['pending_copies']==0,'completed copy ledger')
        need(all(r['equal'] is True and r['terminal_complete'] is True and number(r['device_stream_ms']) and r['transfer_bytes']>0 for r in rows),'copy equality/completion/event measurements')
    elif feature=='P16':
        sensors=d['sensors'];work=d['concurrent_workload'];window=sensors['actual']
        need(work['uuid']==runtime['uuid'] and work['pid']==runtime['process']['pid'] and work['completed']>=2,'owned concurrent inference')
        need(work['first_completion_ns']<=window['start_ns']<window['end_ns']<=work['last_completion_ns'],'observed workload spans sensor window')
        need(work['exclusive'] is False and sensors['measurements']['gpu']['domain']=='physical_device_wide','board scope')
        need(sensors['measurements']['gpu']['fields']['gpu_busy_percent']['status']=='measured','usable GPU busy samples')
        need(all(r['values']['uuid']==runtime['uuid'] for r in sensors['raw_evidence']['gpu_samples'] if r['status']=='measured'),'sensor UUID binding')
    elif feature=='P17':
        binding=d['device_binding'];d=d['result']
        need(binding==dict(index=0,observed_uuid=runtime['uuid'],visible_uuid=runtime['uuid'],admission_uuid=runtime['uuid']),'indexed physical device/admission binding')
        rows=d['raw_evidence']['candidates']
        need(len(rows)==3 and d['actual']['unknown']==0 and all(r['forward_completed'] is True for r in rows),'finite candidate completion evidence')
        need([r['batch'] for r in rows]==[1,2,4] and all(r['status']=='feasible' and r['provenance']['device']=='cuda:0' and r['sweep']['provenance']['environment']['device']=='cuda:0' for r in rows),'actual indexed CUDA forwards')
    elif feature in ('Trainer','P19'):training(d)
    elif feature in ('P20','P22','P23','P25','P21'):
        result=d['basic'] if feature=='P23' else d
        training(result['source_p19'])
        if feature=='P23':
            need(d['saved_activation_lifetime'] is False and d['B4']=='BLOCKED_B4_UNRESOLVED','B4 remains closed')
            need(result['activation']['status']=='unsupported' and result['protocol']['peak_reset_policy']=='never','basic-only memory')
            rows=result['observations'];need(len(rows)>=2 and all(r['cuda']['reserved_bytes']>=r['cuda']['allocated_bytes']>=0 for r in rows),'actual CUDA memory boundaries')
        elif feature=='P22':need(result['segment']['status']=='measured' and result['segment']['completed']['samples']>0,'measured training rate')
        elif feature=='P21':
            rows=result['h2d_transfer']['observations']
            need(len(rows)==4 and all(r['completed'] is True and r['bytes']>0 and number(r['cuda_event_ms']) for r in rows),'four actual completed H2D copies')
        else:
            need(bool(result['observations']) and bool(result['summary']),'functional stage/gradient observations')
            if feature=='P20':need(any(r['stage']=='optimizer_execution' and r['status']=='measured' and number(r['cuda_event_ms']) for r in result['observations']),'observed optimizer stage events')
            if feature=='P25':
                from .p25_evidence import verify_amp
                verify_amp(result)
    elif feature=='P24':
        need(d['sources']['execution_device']=='cuda:0' and len(d['sources']['runs'])==2,'two-seed CUDA experiment')
        for run in d['sources']['runs']:training(run['timing']);need(bool(run['checkpoints']),'actual target evaluation checkpoints')
        need(d['analysis']['censored']==2 and d['analysis']['reached_mean_seconds'] is None,'unreached target remains censored')
    elif feature in ('P26','P27','P28','P29'):
        evaluations=[d] if feature=='P26' else [d['sources'][k] for k in ('clean','perturbed')] if feature=='P27' else [d['sources'][k] for k in ('baseline','candidate')]
        if feature=='P29':evaluations=[b['evaluation'] for b in evaluations]
        for ev in evaluations:
            need(ev['source']['device']=='cuda:0' and bool(ev['raw']) and sum(ev['accounting']['batch_sizes'])==8,'actual eight-sample CUDA quality evaluation')
        if feature=='P27':
            need(d['sources']['corruption']['calls']>0 and d['sources']['clean_cohort']['inputs']!=d['sources']['perturbed_cohort']['inputs'],'actual paired corruption')
        if feature=='P28':
            proof=d['sources']['candidate_precision'];rows=proof['batch_observations']
            version=proof.get('execution_version');allowed=('torch.float16','torch.float32') if version==3 else ('torch.float16',)
            need(type(version) is int and version in (2,3) and proof['policy']=='amp_fp16' and bool(rows) and all(
                r['device']=='cuda:0' and r['finite'] is True and r['native_output_dtype'] in allowed and
                r['autocast_enabled'] is True and r['autocast_dtype']=='torch.float16' and bool(r['operators']) and all(
                    op['operator'] in ('Conv2d','Linear') and op['device']=='cuda:0' and op['weight_dtype']=='torch.float32' and
                    op['input_dtype'] in ('torch.float32','torch.float16') and op['output_dtype']=='torch.float16' and
                    op['autocast_enabled'] is True and op['autocast_dtype']=='torch.float16' for op in r['operators'])
                for r in rows),'actual AMP operator/dtype observations')
            if version==3:
                observed=[r['native_output_dtype'] for r in rows]
                need(proof['verified_execution']==dict(batches=len(rows),native_output_dtype=observed[0] if len(set(observed))==1 else None,
                    native_output_dtypes=observed,operators=[]),'actual mixed-precision output summary')
        if feature=='P29':
            need(d['sources']['comparison_version']==4 and d['analysis']['memory']['status']=='comparable','GPU comparison and allocator evidence')
            proofs=[d['sources'][s]['gpu_provenance'] for s in ('baseline','candidate')]
            need(proofs[0]==proofs[1],'paired GPU provenance')
            for proof in proofs:need(gpu_uuid(proof['physical_gpu_uuid'])==runtime['uuid'] and proof['cuda_build']==runtime['cuda_build'] and proof['torch_version']==runtime['torch_version'],'comparison bound to loaded runtime')
            for side in ('baseline','candidate'):
                bundle=d['sources'][side];p=bundle['performance']
                need(p['model']['weights_sha256']==bundle['benchmark_model_sha256'] and bool(bundle['state_witness']),'comparison weight witness')
                need(bool(p['timing']['latency_samples_ms']) and p['cuda_memory']['counter_reset_performed'],'resident timing and explicit peak capture')
    elif feature=='P30':
        a=d['analysis'];e=d['evidence']
        need(a['status']=='completed' and a['pending']==0 and a['processed']>0 and a['received']==a['processed']+a['dropped'],'stream completion conservation')
        need(e['header']['processor']['device_id']==runtime['uuid'] and bool(e['rows']),'actual GPU stream rows')
        need(e['controller']['cleanup']['active_processes']==0 and e['controller']['memory_limits']==dict(process_memory_bytes=3*1024**3,job_memory_bytes=6*1024**3),'nested cleanup and budget receipt')
    elif feature=='P32':
        a=d['analysis'];e=d['evidence']['stability']['evidence']
        need(a['source_complete'] is True and e['header']['workload']['device_id']==runtime['uuid'],'completed owned trend source')
        series={k:a[k] for k in ('allocated','reserved')}
        for key,s in series.items():
            need(s['status']=='descriptive_fit' and s['n']>=3 and len(s['timeline'])==s['n'],'real '+key+' trend')
            source={r['index']:r['allocator'][key+'_bytes'] for r in e['rows'] if r['allocator'] is not None}
            need(all(p['value']==source[p['index']] for p in s['timeline']),'independent '+key+' source binding')
        need(series['allocated']['max_bytes']>series['allocated']['min_bytes'],'bounded live allocation/release observed')
        need(any(r['allocator']['reserved_bytes']>r['allocator']['allocated_bytes'] for r in e['rows'] if r['allocator']),'allocated/cache distinction observed')
        need(a['allocator_peak']['reset_performed_by_P32'] is False and a['leak_proven'] is False,'no peak-reset/leak claim')
    elif feature=='P34':
        need(d['analysis']['comparable'] and len(d['analysis']['devices'])==2 and all(r['status']=='measured' for r in d['analysis']['devices']),'actual paired device measurements')
        runs=d['evidence']['runs'];need(len(runs)==2,'paired owned runs')
        for run in runs:
            report=run['report'];identity=report['identity']
            need(len(report['latency_ns'])==5 and all(number(v,True) for v in report['latency_ns']) and report['completed']>0,'completed native forward samples')
            need(report['model_unchanged'] is True and report['input_unchanged'] is True and run['controller']['cleanup']['active_processes']==0,'device run preservation and cleanup')
            if run['device']!='cpu':need(identity['hardware']['physical_id']==runtime['uuid'] and identity['execution']['input_device']==identity['execution']['output_device']=='cuda:0','actual device execution identity')
    elif feature=='P35':
        from .native_identity import validate_ort_identity
        validate_ort_identity(d['loaded_runtime'],d['result'],d['planned_runtime'])
        d=d['result']
        a=d['analysis'];need(a['status']=='expected_execution_observed' and bool(a['events']) and all(r['provider']=='CUDAExecutionProvider' for r in a['events']),'actual CUDA provider execution events')
        need(a['fallback_detected'] is False and not a['uncovered_source_nodes'] and a['evidence_assessment']['node_coverage']['status']=='complete','complete CUDA node coverage without CPU placement fallback')
        e=d['evidence'];report=e['report']
        need(report['completed_runs']==4 and report['trace_sha256']==native_hash(report['trace']) and bool(report['trace']),'completed ORT runs and trace hash')
        need(report['runtime']['name']=='onnxruntime' and bool(report['runtime']['version']) and 'CUDAExecutionProvider' in report['session_providers'],'loaded provider runtime')
        need(report['capture_context']['input_unchanged'] is True and len(report['capture_context']['run_intervals'])==4,'actual ORT run intervals/input preservation')
    else:raise ValueError('unreviewed feature: '+feature)
