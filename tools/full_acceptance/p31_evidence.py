"""Strict offline admission of the current acceptance run's P31 series to P32."""
from pathlib import Path
from .core import read, sha


def validated_source(m):
    from hyso.performance import RuntimeResult, reopen_stability
    stage=Path(m['output'])/'stages/10.stability'
    manifest=Path(m['output'])/'manifest.json';digest=sha(manifest)
    if not any(e['id']=='10.stability' and e['authorization']['approved'] is True for e in m['experiments']):
        raise ValueError('P32 requires an authorized P31 source')
    result=read(stage/'result.json');outer=read(stage/'supervisor-receipt.json')
    if (result.get('id')!='10.stability' or result.get('status')!='PASS'
            or result.get('manifest_sha256')!=digest
            or result.get('criteria',{}).get('P31_uninterrupted_7200') is not True
            or outer.get('experiment')!='10.stability' or outer.get('manifest_sha256')!=digest
            or type(outer.get('exit_code')) is not int or outer['exit_code']!=0
            or outer.get('reason') is not None or outer.get('native_crash') is not False
            or outer.get('owned_job_only') is not True
            or type(outer.get('cleanup',{}).get('active_processes')) is not int
            or outer['cleanup']['active_processes']!=0):
        raise ValueError('P32 requires a complete current-run P31 terminal and owned cleanup receipt')
    raw=RuntimeResult.from_json(stage/'stability.json');data=raw.to_dict()
    e=data['evidence'];a=data['analysis'];diag=raw.stability_diagnostics()
    if e.get('header') is None:raise ValueError('P32 source has no workload header')
    protocol=m['protocols']['stability'];identity=e['header']['workload']
    recipe=e['request']['recipe'];terminal=e['terminal'];controller=e['controller']
    if (data['feature_id']!='P31' or e['header'].get('stability_version')!=2
            or e['request']['protocol']!=protocol or protocol['duration_seconds']<7200
            or a['status']!='completed' or a['duration_seconds'] is None
            or a['duration_seconds']<max(7200,protocol['duration_seconds'])
            or terminal is None or terminal['closed'] is not True
            or controller['exit_code']!=0 or controller['reason'] is not None
            or controller['cleanup']['active_processes']!=0
            or not all(diag['continuity'].get(k) is True for k in
                ('requested_duration_satisfied','no_failures_observed','no_sampling_gaps_detected'))
            or recipe['factory']!='full_acceptance.workloads:inference_work'
            or recipe['configuration']!=dict(manifest_path=str(manifest),device='cpu')
            or recipe['search_path']!=str(Path(m['source_snapshot']['root'])/'tools')
            or recipe.get('gpu_uuid') is not None
            or identity.get('kind')!='model_inference' or identity.get('device_id')!='cpu'
            or identity.get('trained') is not True or identity.get('precision')!='fp32'
            or identity.get('input_shape')!=[256,1,28,28]
            or identity.get('checkpoint',{}).get('sha256')!=m['assets']['models']['B']['sha256']):
        raise ValueError('P32 requires validated uninterrupted 7200-second Model B CPU evidence')
    if reopen_stability(stage/'journal').to_dict()['evidence']!=e:
        raise ValueError('P32 source differs from its committed journal')
    # Missing telemetry cannot become complete long-run memory evidence.
    if not e['rows'] or any(r.get('rss') is None or r.get('process_commit',{}).get('value') is None for r in e['rows']):
        raise ValueError('P32 requires complete RSS/private-commit series')
    return raw
