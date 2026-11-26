"""CPU ORT selection and execution evidence; no runtime imports or installation."""
from pathlib import Path
from tools.cuda_acceptance.native_identity import canonical_path, DLLIntegrityError
from tools.final_acceptance.common import file_hash


def canonical_trace_hash(trace):
    # Exactly runtime_result._hash / providers' canonical JSON encoding. Keep
    # this Torch-free so runner validation never imports a GPU runtime.
    from tools.cuda_acceptance.p25_evidence import native_hash
    if type(trace) is not list:raise ValueError('ORT trace must be a canonical event list')
    return native_hash(trace)


def runtime_options(request):
    expected=request['cpu_ort_manifest']
    site=Path(expected['site'])
    if not expected['version'] or not (site/'onnxruntime/__init__.py').is_file():
        raise ValueError('Prepared CPU ORT source unavailable')
    for path,sha in expected['allowed_binaries'].items():
        if file_hash(path)!=sha:raise DLLIntegrityError('Prepared ORT binary changed: '+path)
    return dict(runtime_search_path=str(site)),expected


def verify_cpu_execution(result,identity,expected):
    e=result['evidence'];a=result['analysis'];r=e['report'];controller=e['controller']
    if not (controller['exit_code']==0 and controller['reason'] is None and
            controller['cleanup']['active_processes']==0 and controller['kill_on_close'] is True):
        raise DLLIntegrityError('CPU ORT owned cleanup unverified')
    if not (a['status']=='expected_execution_observed' and a['events'] and
            all(row['provider']=='CPUExecutionProvider' for row in a['events']) and
            a['fallback_detected'] is False and not a['uncovered_source_nodes'] and
            a['evidence_assessment']['node_coverage']['status']=='complete' and
            r['completed_runs']==4 and r['session_providers']==['CPUExecutionProvider']):
        raise ValueError('CPU ORT requires actual complete CPU node execution, not exit zero/provider listing')
    if not (r['trace'] and r['trace_sha256']==canonical_trace_hash(r['trace']) and
            r['capture_context']['input_unchanged'] is True and len(r['capture_context']['run_intervals'])==4):
        raise ValueError('CPU ORT completed trace/input binding missing')
    coverage=a['evidence_assessment']['node_coverage']
    per_run=coverage['per_run_source_event_counts'];nodes=coverage['events_per_source_node']
    if not (coverage['per_run_association']=='complete' and len(per_run)==4 and nodes and
            all(count==4 for count in nodes.values()) and
            all(set(run)==set(nodes) and all(count==1 for count in run.values()) for run in per_run)):
        raise ValueError('CPU ORT requires every source node in each completed run')
    pid=controller['launcher_pid']
    if not (identity.get('pid')==pid and identity.get('admitted_pid')==str(pid) and
            identity.get('visible_uuid')=='-1' and
            canonical_path(identity['source'])==canonical_path(Path(expected['site'])/'onnxruntime/__init__.py') and
            identity['version']==expected['version']==r['runtime']['version']):
        raise DLLIntegrityError('CPU ORT source/version/admitted process mismatch')
    actual=identity['loaded_binaries'];allowed=expected['allowed_binaries']
    if not actual or not any(Path(p).name.lower()=='onnxruntime_pybind11_state.pyd' for p in actual):
        raise DLLIntegrityError('CPU ORT loaded extension missing')
    for path,sha in actual.items():
        if allowed.get(canonical_path(path))!=sha:
            raise DLLIntegrityError('CPU ORT loaded DLL hash/path mismatch: '+path)
