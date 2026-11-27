"""Bind native export child requests to the parent pre-admission manifest."""
from pathlib import Path
from tools.cuda_acceptance.validation import validate_gpu_attempt as base_validate
from tools.final_acceptance.common import read_json,file_hash

def validate_gpu_attempt(folder,request,receipt):
    values,critical=base_validate(folder,request,receipt)
    if critical or request.get('campaign_id')!='export-acceptance':return values,critical
    try:
        plan=request['gpu_plan']
        allowed=dict(plan['feature_tools']['ort']['allowed_binaries'])
        allowed.update({r['path']:r['sha256'] for r in plan['runtime']['binary_expectations'].values()})
        allowed.update(plan['feature_tools']['tensorrt']['allowed_binaries'])
        dependencies=plan['feature_tools'].get('export_cuda',{})
        allowed.update(dependencies.get('allowed_binaries',{}))
        for feature,row in values.items():
            if feature=='ADMISSION' or row['status']!='PASS':continue
            data=read_json(row['evidence'])['data'];root=Path(data['artifact_root'])
            files=[r for r in data['artifact_files'] if Path(r['path']).name=='request.json']
            if len(files)!=1:raise ValueError('one native child request required')
            path=root/files[0]['path'];native=read_json(path);config=native['native']
            if (config['allowed_binaries']!=allowed or config['gpu_uuid']!=plan['gpu_uuid'] or
                config['runtime_search_path']!=plan['feature_tools']['ort']['site'] or
                config.get('dependency_identities',{})!=dependencies.get('binary_identities',{}) or
                config['process_memory_bytes']>plan['budgets']['process_bytes'] or config['job_memory_bytes']>plan['budgets']['job_bytes'] or
                config['allocator_fraction']>plan['budgets']['cuda_allocator_fraction']):raise ValueError('native child manifest/budget differs from parent admission')
            runtime=data['runtime'];report=runtime.get('report',runtime);controller=runtime.get('controller',data.get('controller'))
            if report['token']!=native['token'] or report['token']!=controller['token'] or report['pid']!=controller['launcher_pid'] or report['request_sha256']!=file_hash(path):raise ValueError('native child process/request mismatch')
            if feature in ('E02_CUDA','E05_CUDA') and native['source_sha256']!=data['artifact_sha256']:raise ValueError('native runtime used another artifact')
            for p,r in report['loaded_binaries'].items():
                if allowed.get(p)!=r['sha256'] or native['_binary_file_ids'].get(p)!={k:r[k] for k in ('file_id','bytes')}:raise ValueError('native loaded hash/path/file identity differs')
                if p in dependencies.get('binary_identities',{}) and dependencies['binary_identities'][p]!={k:r[k] for k in ('file_id','bytes')}:raise ValueError('native compiler DLL differs from preparation identity')
            extension=data.get('development_extension',{})
            receipts=list(extension.get('validation',{}).get('native_receipts',[]))
            if extension.get('benchmark',{}).get('native_receipt'):receipts.append(extension['benchmark']['native_receipt'])
            for item in receipts:
                evidence=Path(item['evidence_root']).resolve()
                if not evidence.is_relative_to(Path(row['evidence']).resolve().parent):raise ValueError('extension receipt escapes owned attempt')
                child=read_json(evidence/'request.json');observed=read_json(evidence/'report.json');control=read_json(evidence/'controller.json')
                if observed!=item['report'] or control!=item['controller']:raise ValueError('extension receipt differs from retained evidence')
                if child['native']!=native['native'] or child['source_sha256']!=data['artifact_sha256'] or observed['request_sha256']!=file_hash(evidence/'request.json'):
                    raise ValueError('extension native request/config/source binding differs')
                if observed['token']!=control['token'] or observed['pid']!=control['launcher_pid'] or control['exit_code']!=0 or control['reason'] is not None or control['cleanup']['active_processes']!=0:
                    raise ValueError('extension ownership/cleanup mismatch')
                for p,r in observed['loaded_binaries'].items():
                    if allowed.get(p)!=r['sha256'] or child['_binary_file_ids'].get(p)!={k:r[k] for k in ('file_id','bytes')}:raise ValueError('extension loaded DLL differs from admission')
            if feature=='E02_CUDA' and Path(report['runtime']['source']).resolve()!=(Path(plan['feature_tools']['ort']['site'])/'onnxruntime/__init__.py').resolve():raise ValueError('native ORT source differs from admitted runtime')
        return values,None
    except (KeyError,ValueError,TypeError,OSError) as exc:return {},'EXPORT_NATIVE_BINDING_INVALID: '+str(exc)
