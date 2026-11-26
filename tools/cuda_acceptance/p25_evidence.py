"""Offline P25 AMP acceptance: native skip is evidence, never a finite update."""
import hashlib
import json
import math


def native_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify_amp(d):
    def need(value,reason):
        if not value:raise ValueError('P25: '+reason)
    def finite(value):return type(value) in (int,float) and math.isfinite(value) and value>=0
    need(d['sha256']==native_hash({k:v for k,v in d.items() if k!='sha256'}),'evidence checksum')
    source=d['source_p19'];stages=d['source_p20'];rows=d['observations']
    need(d['source_sha256']==native_hash(source) and d['source_p20_sha256']==native_hash(stages),'P19/P20 checksums')
    need(stages['source_p19']==source and stages['source_sha256']==d['source_sha256'] and
        d['source_segment_id']==stages['source_segment_id']==source['segment_id'],'P19/P20 segment association')
    need(d['status']=='measured' and d['reason'] is None and source['provenance']['training']['amp'] is True and
        source['provenance']['runtime']['device']=='cuda:0','measured native CUDA AMP episode')
    attempts=[r for r in source['observations'] if r['kind']=='attempt']
    micro=[r for r in source['observations'] if r['kind']=='microbatch']
    need(len(rows)==len(attempts)>0 and len(rows)<=d['protocol']['max_attempts'],'complete attempt ledger')
    need(d['actual']==dict(epochs=source['actual_epochs'],windows=len(rows),attempts=len(rows)),'attempt accounting')
    by_attempt={}
    for row in stages['observations']:
        if row['attempt_id'] is not None:
            stage=by_attempt.setdefault(row['attempt_id'],{})
            need(row['stage'] not in stage,'duplicate native stage')
            stage[row['stage']]=row

    def snapshot(value):
        tensors=value['tensors'];present=[]
        need(value['query_start_ns']<=value['query_end_ns'],'gradient query clock')
        for index,row in enumerate(tensors,1):
            need(row['parameter_id']==index and bool(row['names']),'gradient parameter identity')
            g=row['gradient']
            if g is None:continue
            need(g['device']=='cuda:0' and g['layout']=='torch.strided' and g['dtype'] in ('torch.float16','torch.bfloat16','torch.float32','torch.float64') and
                all(type(n) is int and n>=0 for n in g['shape']) and type(g['elements']) is int and g['elements']==math.prod(g['shape'])>0,'real dense gradient observations')
            need(all(type(g[k]) is bool for k in ('finite','has_nan','has_inf')) and
                g['finite']==(not (g['has_nan'] or g['has_inf'])),'finite/NaN/Inf flags')
            need(g['chunks']==(g['elements']+d['protocol']['chunk_elements']-1)//d['protocol']['chunk_elements'] and
                2*g['chunks']<=g['scalar_reads']<=4*g['chunks'],'gradient readout accounting')
            if g['finite']:
                need(g['status']=='measured' and finite(g['global_l2']) and g['reason'] is None,'finite gradient norm')
            else:
                need(g['status']=='nonfinite' and g['global_l2'] is None and g['reason']=='observed_nan_or_inf','retain nonfinite evidence without zero filling')
            present.append(g)
        need(bool(present),'no observed gradients')
        nonfinite=any(not g['finite'] for g in present)
        norm=dict(value=None,status='nonfinite',reason='observed_nan_or_inf') if nonfinite else dict(
            value=math.hypot(*(g['global_l2'] for g in present)),status='measured',reason=None)
        expected=dict(global_l2=norm,finite=not nonfinite,nan_tensor_count=sum(g['has_nan'] for g in present),
            inf_tensor_count=sum(g['has_inf'] for g in present),parameters=len(tensors),with_gradient=len(present),
            without_gradient=len(tensors)-len(present),unsupported_gradients=0,gradient_elements=sum(g['elements'] for g in present),
            chunks=sum(g['chunks'] for g in present),scalar_reads=sum(g['scalar_reads'] for g in present))
        need(value['summary']==expected,'raw gradient/summary arithmetic')
        return expected

    counters=source['counters'];cursor=counters['before']['applied'];previous_scale=None
    need(all(type(v) is int and v>=0 for endpoint in counters.values() for v in endpoint.values()),'native counter types')
    recovered=[];awaiting=False;applied=skipped=0;changes=dict(increased=0,unchanged=0,decreased=0)
    for index,(row,attempt) in enumerate(zip(rows,attempts),1):
        need(all(type(row[k]) is int and row[k]>0 for k in ('attempt_id','epoch_id','window_id','samples')),'native identity types')
        need(row['attempt_id']==attempt['attempt_id']==counters['before']['attempts']+index,'ordered attempt identity')
        need(all(row[k]==attempt[k] for k in ('epoch_id','window_id','samples','applied','skipped','applied_update_id')),'P19 attempt ownership')
        need(all(type(row[k]) is bool for k in ('applied','skipped','optimizer_executed')) and row['applied']!=row['skipped'],'execution/skip flags')
        need(attempt['update_before']==cursor and attempt['update_after']==cursor+int(row['applied']) and
            row['applied_update_id']==(attempt['update_after'] if row['applied'] else None),'contiguous optimizer update ledger')
        cursor=attempt['update_after']
        batches=[m for m in micro if m['window_id']==row['window_id']]
        need(row['microbatch_ids']==[m['microbatch_id'] for m in batches] and sum(m['samples'] for m in batches)==row['samples'],'gradient/microbatch sample binding')
        linked=by_attempt[row['attempt_id']];execution=linked['optimizer_execution'];step=linked['scaler_step_attempt'];clip=linked['gradient_clipping']
        need(all(s['epoch_id']==row['epoch_id'] and s['window_id']==row['window_id'] for s in linked.values()),'P20 attempt/stage ownership')
        update=linked['scaler_update']
        need(all(s['status']=='measured' and finite(s['cuda_event_ms']) and s['start_ns']<=s['end_ns'] for s in (step,update)) and
            step['end_ns']<=update['start_ns'],'native scaler step/update stage evidence')
        need(row['clipping']==dict(status=clip['status'],reason=clip['reason'],threshold=stages['configuration']['grad_clip']),'clipping/stage binding')
        pre=snapshot(row['preclip'])
        need(linked['gradient_check_normalization']['end_ns']<=row['preclip']['query_start_ns']<=row['preclip']['query_end_ns']<=
            (clip['start_ns'] if clip['status']=='measured' else step['start_ns']),'gradient timing/stage binding')
        need((row['postclip'] is not None)==(d['protocol']['observe_postclip'] and clip['status']=='measured'),'postclip coverage')
        if row['postclip'] is not None:
            snapshot(row['postclip'])
            need(clip['status']=='measured' and clip['end_ns']<=row['postclip']['query_start_ns']<=row['postclip']['query_end_ns']<=step['start_ns'],'postclip timing/stage binding')
        before,after=row['scale_before'],row['scale_after']
        need(finite(before) and finite(after) and min(before,after)>0 and (previous_scale is None or before==previous_scale),'native scale continuity')
        previous_scale=after;change='increased' if after>before else 'decreased' if after<before else 'unchanged'
        need(row['scale_change']==change and row['overflow_cause']=='unknown','scale classification/unproven overflow cause')
        changes[change]+=1
        if row['skipped']:
            need(pre['finite'] is False and row['optimizer_executed'] is False and
                execution['status']=='not_executed' and execution['reason']=='native_scaler_skipped_optimizer' and
                all(execution[k] is None for k in ('start_ns','end_ns','host_ms','cuda_event_ms')),
                'nonfinite attempt needs native optimizer skip, not fabricated execution')
            need(after<before and row['overflow_evidence']=='native_skip_with_nonfinite_gradient','native AMP skip/backoff evidence')
            skipped+=1;awaiting=True
        else:
            need(pre['finite'] is True and row['optimizer_executed'] is True and execution['status']=='measured' and
                finite(execution['cuda_event_ms']) and execution['start_ns']<=execution['end_ns'],'actual finite gradient and native optimizer update')
            need(after>=before and row['overflow_evidence']=='none_observed','successful AMP scale/evidence')
            applied+=1
            if awaiting:recovered.append(row['attempt_id']);awaiting=False
    need(applied>0,'at least one successful finite optimizer update is mandatory')
    expected=dict(attempts=len(rows),applied=applied,skipped=skipped,recovered_after_skip_attempt_ids=recovered,
        recovery_scope='this_segment_only',scale_changes=changes)
    need(d['summary']==expected,'recovery IDs must bind to a later successful native update')
    need(counters['after']['applied']==cursor and counters['after']['attempts']-counters['before']['attempts']==len(rows) and
        counters['after']['microbatches']-counters['before']['microbatches']==len(micro),'Trainer counter consistency')
    need(sum(e['attempts'] for e in source['epochs'])==len(rows) and sum(e['applied'] for e in source['epochs'])==applied and
        sum(e['skipped'] for e in source['epochs'])==skipped,'epoch accounting')
