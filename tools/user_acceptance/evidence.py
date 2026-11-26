"""Parent and admitted-worker checks for functional user journey witnesses."""
import math
from tools.cuda_acceptance.feature_evidence import verify_feature as native_verify, verify_nested_safety


def verify_feature(feature, data, runtime):
    if feature == 'ADMISSION': return native_verify(feature, data, runtime)
    verify_nested_safety(data)
    if feature not in {'U01','U02','U03','U04','U05','U06','U07','U08','U09','U12','U13'}:
        raise ValueError('Unreviewed user scenario: '+feature)
    if not (feature.startswith('U') and data['scenario'] == feature and data['device'] == 'cuda:0'
            and data['status'] == 'PASS' and data['public_api'] and data['checks']
            and all(v is True for v in data['checks'].values()) and data['observations']):
        raise ValueError('User journey requires actual functional witnesses')
    d=data['observations']
    def need(ok, reason):
        if not ok:raise ValueError(feature+': '+reason)
    def forward(value):
        need(value['finite'] is True and value['output_device']=='cuda:0' and value['parameters']>0,
            'completed finite forward')
        need(value['before']==value['after'],'model state/modes/gradients unchanged')
        b=value['benchmark'];times=b['timing']['latency_samples_ms']
        need(len(times)==1 and len(times[0])==5 and all(type(t) in (float,int) and math.isfinite(t) and t>0 for t in times[0]),'five real timing samples')
        need(b['environment']['device']=='cuda:0' and b['cuda_memory']['status']=='measured' and
            b['cuda_memory']['peak_allocated_bytes']>0,'native device and allocator measurement')
    if feature=='U01':forward(d)
    elif feature=='U02':
        need(len(d['rows'])==4 and not d['blocked'],'all four architectures measured')
        for r in d['rows']:forward(r['result'])
    elif feature=='U03':
        need(len(d['rows'])==3,'three builder convolution paths')
        for r in d['rows']:forward(r['result']);need(r['result']['shape']==[2,2],'classification output')
    elif feature=='U04':
        need(len(d['rows'])==2 and d['invalid_link']['message'],'single/multi input and explicit invalid-link error')
        for r in d['rows']:need(r['shape']==[2,2] and r['output_device']=='cuda:0' and r['finite'] and
            bool(r['gradients']) and all(r['gradients']),'actual finite graph backward')
    elif feature=='U06':
        need(all(d[k] is True for k in ('model_equal','optimizer_equal','scheduler_equal','restored_equal')) and
             bool(d['uninterrupted']) and bool(d['resumed']),'model/optimizer/scheduler resume equivalence')
    elif feature=='U07':
        ev=d['evaluation'];need(ev['source']['device']=='cuda:0' and len(ev['raw']['predictions'])==8 and
            sum(ev['accounting']['batch_sizes'])==8,'eight actual labeled predictions')
        t=d['throughput'];need(t['actual']['completed_batches']==3 and t['actual']['pending_batches']==0 and
            t['measurements']['status']=='measured' and t['measurements']['value']>0,'completed throughput')
        native_verify('P28',d['precision'],runtime)
    elif feature=='U08':native_verify('P29',d,runtime)
    elif feature=='U09':
        records=d['source']['index']['records'];ids={r['record_id'] for r in records}
        need(bool(ids) and all(r['validation']['status']=='native_validated' for r in records),'validated retained native source')
        need(d['recommendations']['format']=='hyso.advisor.recommendations' and bool(d['recommendations']['source']), 'bound recommendation source')
        items=d['recommendations']['recommendations']
        need(bool(items) and any(item['retained_record_ids'] for item in items) and all(
            set(item['retained_record_ids'])<=ids and item['estimated_improvement'] is None and
            item['manual_only'] is True and item['verified_effect']=='not_measured' for item in items),
            'traceable manual recommendations without invented effects')
    elif feature=='U12':
        forward(d['measurement'])
        need(d['before']==d['after'] and bool(d['source_sha256']) and
            bool(d['advisor']['source']['index']['records']),'model and source invariance with retained Advisor input')
    if feature == 'U13': native_verify('P35', data['observations'], runtime)
    if feature == 'U05':
        need(len(d['episodes'])==6,'three model families times FP32/AMP')
        for row in data['observations']['episodes']:
            if row['applied'] <= 0 or not row['weights_changed']: raise ValueError('No applied optimizer update')
            if row['amp']: native_verify('P25', row['gradient_result'], runtime)
            else:
                source=row['gradient_result']['source_p19']
                need(source['provenance']['runtime']['device']=='cuda:0' and source['actual_epochs']==1 and
                    source['instrumentation']['pending_event_pairs']==0 and source['instrumentation']['event_drains']>0,
                    'completed real FP32 backward/update events')
                need(all(r['applied'] and not r['skipped'] for r in row['gradient_result']['observations']),'FP32 applied update ledger')
