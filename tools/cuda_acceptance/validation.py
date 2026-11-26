"""Fail-closed acceptance of CUDA evidence after controller cleanup, stdlib only."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'final_acceptance'))
from common import read_json,read_journal,digest,file_hash
from tools.cuda_acceptance.identity import parent_binding,gpu_uuid
from tools.cuda_acceptance.feature_evidence import verify_feature,SafetyViolation


def validate_gpu_attempt(folder,request,receipt):
    folder=Path(folder)
    try:
        def require(condition,reason):
            if not condition:raise ValueError(reason)
        require(receipt.get('schema_version')==3,'CONTROLLER_SCHEMA')
        require(receipt.get('source_integrity') is True,'SOURCE_INTEGRITY')
        require(receipt.get('cleanup_verified') is True and receipt.get('cleanup',{}).get('active_processes')==0,'OWNED_CLEANUP')
        require(receipt.get('payload_admitted') and receipt.get('limit_readback_verified'),'ADMISSION_LIMITS')
        budgets=request['gpu_plan']['budgets']
        require(receipt.get('limits')==dict(process_bytes=budgets['process_bytes'],job_bytes=budgets['job_bytes'],max_processes=budgets['max_processes']),'LIMITS_MISMATCH')
        require(not receipt.get('monitor_reason') and receipt.get('exit_code')==0,'ABNORMAL_TERMINATION')
        for field,name in [('request_sha256','request.json'),('bootstrap_sha256','bootstrap.py'),('journal_sha256','events.jsonl'),('worker_terminal_sha256','journal-terminal.json'),('terminal_evidence_sha256','terminal-evidence.json')]:
            require(bool(receipt.get(field)) and file_hash(folder/name)==receipt[field],field)
        ids=receipt['identities'];admission=read_json(folder/'admission-receipt.json');admit=read_json(folder/'admit.json');ready=read_json(folder/'interpreter-ready.json')
        require(ids.get('exact_job_membership') and ids==admission['identities']==read_json(folder/'owner-identities.json'),'JOB_MEMBERSHIP')
        require(ids['nonce_sha256']==digest(request['token']) and admit['token']==ready['token']==request['token'],'NONCE_BINDING')
        require(admit['readback_verified'] and admit['pid']==ready['pid']==ids['interpreter']['pid'],'PID_BINDING')
        require(parent_binding(ids,ready,receipt['launch_binding']),'PARENT_BINDING')
        require(admission.get('no_breakaway') is True and admission.get('kill_on_close') is True and admission.get('limits_readback') is True,'JOB_INHERITANCE')
        rows,issues=read_journal(folder/'events.jsonl');require(not issues and bool(rows),'JOURNAL_INVALID')
        identity={k:request[k] for k in ('attempt_id','group','source_commit','environment_id')}
        require(all(all(r.get(k)==v for k,v in identity.items()) for r in rows),'EVENT_IDENTITY')
        terminal=read_json(folder/'journal-terminal.json');sealed=read_json(folder/'terminal-evidence.json')
        require(terminal.get('closed') and not terminal.get('writer_failed') and terminal['committed_sequence']==len(rows) and terminal['committed_head_sha256']==rows[-1]['sha256'],'WRITER_SEAL')
        require(sealed['verified_prefix_events']==len(rows) and sealed['verified_prefix_head']==rows[-1]['sha256'] and sealed['journal_sha256']==file_hash(folder/'events.jsonl'),'PARENT_SEAL')
        starts=[r for r in rows if r['kind']=='worker_start']
        require(len(starts)==1 and starts[0]['pid']==ids['interpreter']['pid'] and starts[0]['profile']=='cuda-functional','WORKER_BINDING')
        runtime=read_json(folder/'gpu-runtime.json');events=[r for r in rows if r['kind']=='gpu_identity']
        require(len(events)==1 and events[0]['sha256_file']==file_hash(folder/'gpu-runtime.json'),'GPU_RUNTIME_HASH')
        require(runtime['uuid']==request['gpu_plan']['gpu_uuid']==request['gpu_admission']['uuid'] and runtime['driver']==request['gpu_admission']['driver'],'GPU_IDENTITY')
        require(runtime['process']==ids['interpreter'] and runtime['allocator_fraction']==budgets['cuda_allocator_fraction'] and runtime['device']=='cuda:0','RUNTIME_PROCESS')
        require(runtime['cuda_build'] and runtime['driver_binary']['sha256'] and runtime['torch_cuda_binary']['sha256'],'BINARY_IDENTITY')
        expected=request['gpu_plan']['runtime']
        require(gpu_uuid(runtime['torch_uuid_display'])==runtime['uuid'],'TORCH_UUID')
        require(runtime['torch_version']==expected['versions']['torch'] and Path(runtime['torch_source']).resolve()==(Path(expected['site'])/'torch/__init__.py').resolve(),'TORCH_RUNTIME_SOURCE')
        require(Path(ids['launcher']['image']).resolve()==Path(expected['python']).resolve(),'PLANNED_LAUNCHER_IMAGE')
        require(Path(runtime['base_python']).resolve()==Path(ids['interpreter']['image']).resolve() and
            Path(runtime['python']).resolve() in (Path(expected['python']).resolve(),Path(ids['interpreter']['image']).resolve()),'INTERPRETER_IMAGE')
        require(runtime['loaded_binaries']==expected['binary_expectations'] and
            runtime['driver_binary']==runtime['loaded_binaries']['nvcuda.dll'] and
            runtime['torch_cuda_binary']==runtime['loaded_binaries']['torch_cuda.dll'],'LOADED_RUNTIME_BINARIES')
        require(runtime['probe']==dict(completed=True,input=1,output=2,elements=1),'CUDA_PROBE')
        require(rows[-1]['kind']=='gpu_session_finish' and not any(r['kind']=='critical_error' for r in rows),'INCOMPLETE_GPU_SESSION')
        values={}
        active=None;started_features=set()
        for row in rows:
            if row['kind']=='feature_start':
                require(active is None and row['feature'] in request['features'] and row['feature'] not in started_features,'FEATURE_START_BINDING')
                active=row['feature'];started_features.add(active)
            if row['kind']!='feature_end':continue
            feature=row['feature'];require(feature in request['features'] and feature not in values,'UNREVIEWED_OR_DUPLICATE_FEATURE')
            require(active==feature,'FEATURE_END_BINDING');active=None
            path=folder/(feature+'.json');require(file_hash(path)==row['evidence_sha256'],'FEATURE_HASH')
            data=read_json(path)
            require(data['feature']==feature and data['status'] in ('PASS','FAIL','BLOCKED') and data['runtime_sha256']==file_hash(folder/'gpu-runtime.json'),'FEATURE_SCHEMA')
            if data['status']=='PASS':
                verify_feature(feature,data['data'],runtime)
                if feature=='P35':require(data['data']['planned_runtime']==request['gpu_plan']['feature_tools']['ort'],'ORT_PLANNED_RUNTIME_BINDING')
                if feature=='P14':require(data['data']['tool']==request['gpu_plan']['feature_tools']['nsys'],'NSYS_PLANNED_TOOL_BINDING')
            else:require(bool(data.get('reason')),'MISSING_FAILURE_REASON')
            values[feature]=dict(status=data['status'],reason=data.get('reason'),evidence=str(path),sha256=row['evidence_sha256'],scope=data['scope'])
        require(active is None and set(values)==started_features==set(request['features']),'MISSING_FEATURE_EVIDENCE')
        return values,None
    except (KeyError,ValueError,TypeError,OSError,AssertionError,SafetyViolation) as exc:
        return {},'EVIDENCE_OR_SAFETY_INVALID: '+str(exc)
