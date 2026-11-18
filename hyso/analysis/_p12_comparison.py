"""P12 adapter over native CPU receipts and existing observation/difference APIs."""
from .evidence import EvidenceRecord


POLICY = dict(id='native_cpu_p12_observed_operations_v1', version=1,
              allowed_changes=['model'], scope='analytic_mean_per_observed_batch;not_hardware_work_or_runtime')
METRICS = (('macs_per_batch', 'MAC/batch'), ('logical_flops_per_batch', 'logical_FLOP/batch'))
LIMITS = ['Analytic observed operations only; fewer MACs/FLOPs do not establish lower latency, energy or hardware work.',
          'Only the recorded CPU recipe/input/control flow is compared; no dataset or hardware performance claim.',
          'Native receipt validation establishes recorded consistency, not independent producer authenticity.',
          'Model identity is native registered state plus structure; caller labels are not native identities.',
          'Checkpoint association is optional and is never inferred from model-state or caller hashes.']


def comparison_rows(a, b, record_a, record_b, why_a, why_b):
    from ..advisor.p12_observation import observe_p12
    from ..advisor.performance import _numbers
    observations=[]; natives=[]; reasons=[]; incompatible=False
    for side, record, why in (('baseline',record_a,why_a), ('candidate',record_b,why_b)):
        if why or record is None or record['native_payload'] is None:
            reasons.append(side+':'+(why or 'missing_native_p12_evidence'))
            observations.append(None);natives.append(None)
            continue
        observation=observe_p12(EvidenceRecord(record),version=1).to_dict()
        native=record['native_payload']; binding=native['raw_evidence'].get('p12_binding')
        observations.append(observation);natives.append(native)
        if binding is None:reasons.append(side+':native_p12_binding_missing')
        if observation['availability']!='AVAILABLE':reasons.append(side+':incomplete_operator_coverage')
        device=native['provenance']['environment'].get('device')
        if device is None:reasons.append(side+':native_device_missing')
        elif device!='cpu':
            reasons.append(side+':unsupported_non_CPU_profile');incompatible=True
        version=native['protocol'].get('operation_model_version')
        if version is None:reasons.append(side+':operation_model_version_missing')
        elif version!=3:
            reasons.append(side+':unsupported_operation_model_version');incompatible=True
        scope=native['protocol'].get('scope')
        if scope is None:reasons.append(side+':counting_scope_missing')
        elif scope!='observed_profile_evidence':
            reasons.append(side+':unsupported_counting_scope');incompatible=True
        checkpoint=record['declared_scope'].get('checkpoint_sha256')
        if checkpoint is not None:
            actual=None if binding is None else binding['checkpoint_artifact_sha256']
            if actual is None:reasons.append(side+':native_checkpoint_association_missing')
            elif actual!=checkpoint:
                reasons.append(side+':checkpoint_association_mismatch');incompatible=True
    if all(n is not None for n in natives):
        left,right=natives
        # Known mismatches reject even when one receipt is absent.
        for name, av, bv in (
            ('input',left['provenance']['input'],right['provenance']['input']),
            ('configuration',left['protocol']['configuration'],right['protocol']['configuration']),
            ('scope',left['protocol'].get('scope'),right['protocol'].get('scope')),
            ('precision',left['protocol']['recipe']['precision'],right['protocol']['recipe']['precision']),
            ('convention',left['protocol']['convention'],right['protocol']['convention']),
            ('operation_model_version',left['protocol'].get('operation_model_version'),right['protocol'].get('operation_model_version')),
            ('torch',left['provenance']['environment'].get('torch'),right['provenance']['environment'].get('torch'))):
            if av is not None and bv is not None and av!=bv:
                reasons.append('condition_mismatch:'+name);incompatible=True
        for key in ('source_result_id','source_sha256'):
            av,bv=(n['raw_evidence'].get(key) for n in natives)
            if av and av==bv:
                reasons.append('same_underlying_capture:'+key);incompatible=True
        traces=[(n['raw_evidence'].get('trace') or {}).get('sha256') for n in natives]
        if traces[0] and traces[0]==traces[1]:
            reasons.append('same_underlying_capture:trace_sha256');incompatible=True
        bindings=[n['raw_evidence'].get('p12_binding') for n in natives]
        if all(bindings):
            for key in ('method','threads','execution_mode'):
                if bindings[0][key]!=bindings[1][key]:
                    reasons.append('condition_mismatch:'+key);incompatible=True
    status='INCOMPATIBLE' if incompatible else 'INSUFFICIENT_EVIDENCE' if reasons else 'COMPARABLE'
    rows=[]
    for metric,unit in METRICS:
        difference=None
        if status=='COMPARABLE':
            values=[]
            for source,observation,native in zip((a,b),observations,natives):
                key='total_macs_per_batch' if metric=='macs_per_batch' else 'total_flops_per_batch'
                values.append(dict(value=observation['metrics'][metric]['value'],
                    reference=dict(report_id=source.to_dict()['report']['report_id'],
                        **observation['evidence_reference'],metric_path='measurements.'+key,
                        native_binding=native['raw_evidence']['p12_binding'],unit=unit,domain='logical_operations')))
            change,percent,pp=_numbers(values[0]['value'],values[1]['value'],False)
            direction='increase' if change>0 else 'decrease' if change<0 else 'unchanged'
            difference=dict(baseline=values[0],candidate=values[1],difference=change,percent_change=percent,
                percentage_point_change=pp,preferred_direction='lower',observed_direction=direction,
                metric_effect='unfavorable' if change>0 else 'favorable' if change<0 else 'unchanged')
        comparison=dict(contract=POLICY,eligibility=dict(status=status,reasons=reasons),
                        limitations=LIMITS,source_observations=[None if o is None else o['content_sha256'] for o in observations])
        rows.append(dict(feature_id='P12',metric=metric,preferred_direction='lower',domain='logical_operations',unit=unit,
            scope=POLICY['scope'],status=status,reasons=list(reasons),difference=difference,comparison=comparison))
    return rows
