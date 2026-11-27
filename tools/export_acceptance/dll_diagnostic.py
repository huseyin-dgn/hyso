"""One-shot owned DLL boundary diagnostic; never creates an acceptance PASS."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import uuid

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from tools.export_acceptance import runner as r


def prepare():
    plan=r.engine.read_json(r.ROOT/'plan.json');cp=r.engine.read_json(r.ROOT/'checkpoint.json')
    if r.snapshot()!=plan['source_sha256'] or r.engine.user_snapshot()!=plan['user_files']:
        raise RuntimeError('PREPARE_EXPORT_ACCEPTANCE_FIRST')
    if not cp.get('safety_hold'):raise RuntimeError('DIAGNOSTIC_REQUIRES_EXISTING_SAFETY_HOLD')
    r.configure();r.user.verify_cached(cp)
    root=r.ROOT/'diagnostics'/uuid.uuid4().hex;root.mkdir(parents=True,exist_ok=False)
    ticket=dict(schema_version=1,operation='dll_diagnostic',root=str(root.resolve()),
        plan_sha256=r.engine.file_hash(r.ROOT/'plan.json'),checkpoint_sha256=r.engine.file_hash(r.ROOT/'checkpoint.json'),
        source_sha256=plan['source_sha256'],user_files=plan['user_files'],automatic_retries=0,
        inference_allowed=False,conversion_allowed=False)
    r.engine.atomic_json(root/'ticket.json',ticket)
    r.engine.atomic_json(root/'ticket-seal.json',{'sha256':r.engine.file_hash(root/'ticket.json')})
    print(json.dumps(dict(status='DIAGNOSTIC_PREPARED',ticket=str(root/'ticket.json'),GPU_executed=False)))
    return root/'ticket.json'


def run(ticket_path):
    """Explicit user launch only. Nothing here is invoked by normal resume."""
    from tools.export_acceptance.diagnostic_gate import validate_ticket
    ticket,root,plan=validate_ticket(ticket_path)
    import msvcrt
    lock=(r.engine.CPU_ROOT/'manager.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:
        # Consume before any native launch. Failed diagnostics cannot auto retry.
        with (root/'consumed.json').open('x',encoding='utf-8') as f:json.dump({'pid':os.getpid()},f)
        live=r.engine.resources();config=r.engine.read_json(REPO/'tools/final_acceptance/campaign_config.json')
        if live['available_commit_bytes']<config['start_available_commit_bytes'] or live['disk_free_bytes']<config['minimum_start_disk_bytes']:
            raise RuntimeError('DIAGNOSTIC_RESOURCE_ADMISSION_FAILED')
        gpu=r.engine.gpu_info()
        if gpu['uuid']!=plan['gpu_uuid'] or gpu['free_mib']<plan['budgets']['start_gpu_free_mib']:
            raise RuntimeError('DIAGNOSTIC_GPU_ADMISSION_FAILED')
        from tools.cuda_acceptance.native_identity import ort_inventory
        manifest=plan['feature_tools']['ort']
        if ort_inventory(manifest['site'],plan['runtime']['site'])!=manifest:
            raise RuntimeError('DIAGNOSTIC_PRE_ADMISSION_INVENTORY_CHANGED')
        sys.path.insert(0,plan['runtime']['site'])
        from hyso.export.contracts import NativeConfig,ExportError
        from hyso.export.native import run_owned
        allowed=dict(manifest['allowed_binaries'])
        allowed.update({v['path']:v['sha256'] for v in plan['runtime']['binary_expectations'].values()})
        native=NativeConfig(plan['gpu_uuid'],allowed,runtime_search_path=manifest['site'])
        outcome=dict(status='DIAGNOSTIC_FAILED',acceptance_PASS=False,admission=gpu)
        try:
            report,controller=run_owned('dll_diagnostic',root/'owned-worker',
                {'attempt_identity':root.name},native,failure_directory=root)
            outcome.update(status='IDENTITY_BOUNDARY_VERIFIED',report=report,controller=controller)
        except ExportError as exc:
            outcome.update(code=exc.code,reason=str(exc),evidence=exc.result)
        finally:
            outcome['source_integrity']=(r.snapshot()==ticket['source_sha256'] and r.engine.user_snapshot()==ticket['user_files'])
            outcome['checkpoint_unchanged']=r.engine.file_hash(r.ROOT/'checkpoint.json')==ticket['checkpoint_sha256']
            if not outcome['source_integrity'] or not outcome['checkpoint_unchanged']:outcome['status']='DIAGNOSTIC_SAFETY_INVALID'
            r.engine.atomic_json(root/'result.json',outcome)
            r.engine.atomic_json(root/'result-seal.json',{'sha256':r.engine.file_hash(root/'result.json')})
        print(json.dumps(dict(status=outcome['status'],result=str(root/'result.json'),acceptance_PASS=False)))
        return 0 if outcome['status']=='IDENTITY_BOUNDARY_VERIFIED' else 1
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare-only',action='store_true');g.add_argument('--run-ticket',type=Path)
    a=p.parse_args()
    if a.prepare_only:prepare()
    else:raise SystemExit(run(a.run_ticket))
