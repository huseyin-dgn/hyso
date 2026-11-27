"""Shared staged conversion orchestration; errors carry the last real stage."""
from dataclasses import asdict
import platform
import sys
import threading
import torch
from .contracts import STAGES,ExportConfig,ExportError,availability,seal
from .artifacts import Transaction
from ._model import isolated,invoke,outputs,names,specs,equivalence,fingerprint,tensors

_CONVERSION_LOCK=threading.Lock()

def stage(body,name,status='PASS',**evidence):body['stages'][name]=dict(status=status,**evidence)

def convert(model,example,output,kind,backend,config,producer,options):
    if not _CONVERSION_LOCK.acquire(blocking=False):raise ExportError('EXPORT_BUSY','serialize export requests; RNG/capture state is process-wide')
    try:return _convert(model,example,output,kind,backend,config,producer,options)
    finally:_CONVERSION_LOCK.release()

def _convert(model,example,output,kind,backend,config,producer,options):
    if type(config) is not ExportConfig:raise TypeError('ExportConfig required')
    body=dict(schema_version=1,format='hyso.export',conversion=kind,backend=backend,backend_version=None,
        config=asdict(config),options=options,environment=dict(torch=str(torch.__version__),python=sys.version,platform=platform.platform(),cuda_build=torch.version.cuda,execution_device='cpu'),
        stages={n:dict(status='NOT_RUN') for n in STAGES},warnings=[],limitations=['Inference-only copied model; sample equivalence is not whole-cohort certification','Arbitrary user Python is trusted; external/global side effects are not isolated by deepcopy'])
    current='admission'
    try:
        with Transaction(output,config.overwrite,config.max_artifact_bytes) as tx:
            with isolated(model,example,config) as (owned,inputs,identity):
                body['source_model']=identity;body['input_fingerprints']=[fingerprint(t) for t in tensors(inputs)]
                stage(body,'admission',profile='bounded_cpu_inference',max_model_bytes=config.max_model_bytes,max_input_bytes=config.max_input_bytes)
                current='dependencies'
                # Producer controls backend imports and records the exact selected path.
                current_ref=['admission']
                def mark(name,**info):
                    current_ref[0]=name;stage(body,name,**info)
                with torch.no_grad():reference=invoke(owned,inputs)
                outputs(reference)
                result=producer(tx,owned,inputs,reference,body,mark)
                current=current_ref[0]
                if result is not None:
                    current_ref[0]='equivalence'
                    stage(body,'equivalence',comparisons=equivalence(reference,result,config))
                    from .runtime import encode_arrays
                    body['reference_outputs']=encode_arrays([v.detach().cpu().numpy() for v in outputs(reference)])
                    body['actual_outputs']=encode_arrays([v.detach().cpu().numpy() for v in outputs(result)])
                elif config.validate_runtime:
                    stage(body,'runtime_loading','BLOCKED',reason='Optional runtime unavailable; artifact remains conversion-only')
                body['inputs']=specs(tensors(inputs),names(options.get('input_names'),len(tensors(inputs)),'input'))
                body['outputs']=specs(outputs(reference),names(options.get('output_names'),len(outputs(reference)),'output'))
                body.update(artifact_path=str(tx.output),artifact_root=str(tx.parent),artifact_format=kind,artifact_files=tx.inventory())
                main=next(r for r in body['artifact_files'] if r['path']==tx.output.name)
                body.update(artifact_sha256=main['sha256'],artifact_bytes=main['bytes'])
            stage(body,'preservation',source_unchanged=True,rng_restored=True)
            # Cleanup status is sealed only after staging cleanup succeeds below.
            body['stages']['cleanup']=dict(status='NOT_RUN')
            prepared=seal(body)
            tx.publish(prepared)
        stage(body,'cleanup',owned_staging_removed=True)
        final=seal(body)
        # Replace only the metadata committed by our own transaction.
        import os
        temporary=tx.manifest.with_name(tx.manifest.name+'.'+tx.token+'.tmp')
        with temporary.open('x',encoding='utf-8') as f:f.write(final.to_json());f.flush();os.fsync(f.fileno())
        os.replace(temporary,tx.manifest)
        return final
    except Exception as exc:
        current=locals().get('current_ref',[current])[0]
        if isinstance(exc,ExportError) and exc.code=='MODEL_PRESERVATION':current='preservation'
        if isinstance(exc,ExportError) and exc.code=='OWNERSHIP':current='cleanup'
        if isinstance(exc,ExportError) and exc.result:body['failure_evidence']=exc.result
        nested_blocked=isinstance(exc,ExportError) and exc.result and any(r.get('status')=='BLOCKED' for r in exc.result.get('stages',{}).values())
        stage(body,current,'BLOCKED' if isinstance(exc,(ImportError,NotImplementedError)) or nested_blocked else 'FAIL',reason=str(exc))
        body['deployment_verified']=False
        if isinstance(exc,ExportError):exc.result=body;raise
        raise ExportError('CONVERSION_FAILED',type(exc).__name__+': '+str(exc),result=body) from exc
