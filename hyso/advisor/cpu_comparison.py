"""Read-only eligibility over the explicit v2 CPU producer contract."""
from fractions import Fraction

from hyso.performance.comparison_capture import _validate_comparison_capture,_digest
from .resolver import NativeMetric


def capture_conditions(resolver,record_id,contract,side):
    from .comparison import ComparisonReason,_QUANTILES
    reasons=[]
    def reason(code,field):reasons.append(ComparisonReason(code,field,side))
    record=resolver.record(record_id)
    if record is None:
        reason('record_not_found','record_id');return None,None,reasons,False
    resolved=resolver.resolve(record_id,contract.metric)
    if resolved.status!='supported':
        reason('metric_not_observable',contract.metric.value);return None,None,reasons,False
    metric=resolved.resolved
    if resolver.check_metric(metric).status!='supported':
        reason('invalid_native_evidence','metric');return metric,None,reasons,True
    n=record['native_payload']
    capture=(n or {}).get('provenance',{}).get('comparison_capture')
    if not isinstance(capture,dict) or capture.get('schema_version')!=2:
        reason('CPU_v2_capture_required','provenance.comparison_capture');return metric,None,reasons,False
    try:capture=_validate_comparison_capture(n)
    except (KeyError,TypeError,ValueError,AttributeError):
        reason('invalid_CPU_capture','provenance.comparison_capture');return metric,None,reasons,True
    feature=n['feature_ids'][0];p=n['provenance'];checkpoint=capture['checkpoint']
    bound=capture['bound_input'];scope=record['declared_scope']
    for key in ('dataset_id','split','workload_id'):
        if scope[key] is not None:reason('unverified_declared_dataset_context',key)
    if scope['checkpoint_sha256'] is not None and scope['checkpoint_sha256']!=checkpoint['sha256']:
        reason('checkpoint_not_bound_to_capture','checkpoint_sha256')
    count=n['actual']['samples'] if feature=='P01' else n['actual']['completed_batches'] if feature=='P02' else None
    if count is not None and (count<contract.min_samples or contract.metric in _QUANTILES
            and (1-Fraction(str(_QUANTILES[contract.metric])))*count<contract.min_tail_samples):
        reason('sample_requirement_not_met','samples/tail_samples')
    if contract.metric==NativeMetric.ARTIFACT_BYTES:
        artifact=n['raw_evidence']['artifact']
        if checkpoint['status']!='native' or artifact['sha256']!=checkpoint['sha256']:
            reason('native_checkpoint_association_missing','artifact.sha256')
    config=p.get('configuration',n['protocol'].get('configuration',{}))
    native_environment={k:v for k,v in p['environment'].items() if k!='pid'}
    conditions=dict(format=capture['format'],schema_version=2,authority=capture['authority'],
        model=(capture['model_state_sha256'],capture['structure_sha256'],checkpoint['sha256']),
        batch=bound['shape'][0],nonbatch_shape=bound['shape'][1:],precision='fp64' if bound['dtype']=='torch.float64' else 'fp32',
        input_sha256=bound['sha256'],input_dtype=bound['dtype'],input_layout=bound['layout'],
        autocast_enabled=False,protocol=n['protocol'],configuration=config,
        device='cpu',backend=capture['environment']['backend'],runtime=capture['environment'],
        native_runtime=native_environment,unit=metric.reference.unit,domain=metric.reference.domain,
        measurement_scope=metric.native_context['measurement_scope'],
        metric_context={k:v for k,v in metric.native_context.items() if k not in
            ('sample_count','effective_tail_count','elapsed_wall_seconds','batch','precision','model_state_sha256')},
        producer=capture['producer'],input_role=bound['role'])
    return metric,conditions,reasons,False
