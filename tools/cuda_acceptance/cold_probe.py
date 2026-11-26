"""One P05 trial via the existing suspended, no-breakaway Job controller."""
from pathlib import Path
from dataclasses import asdict
import tempfile
import time
import uuid


def measure(ctx):
    from common import read_json,new_json,file_hash,digest,tree_bytes,utc,read_journal
    from campaign_controller import execute
    from campaign_operations import cleanup_owned
    from .runner import REPO,CPU_ROOT,ROOT,snapshot
    from hyso.performance import cold
    folder=ctx.folder/'cold-owned';folder.mkdir()
    scratch=Path(tempfile.mkdtemp(prefix='hyso-cuda-cold-'));work=scratch/'payload'
    protocol=cold.ColdStartProtocol(trials=1,timeout_seconds=60,warmup_iterations=1,warm_samples=2)
    recipe=cold.ColdStartRecipe('tools.cuda_acceptance.workloads:model',input_shape=(2,1,16,16),device='cuda:0')
    request=dict(schema_version=1,mode='tests',profile='cuda-functional-cold',campaign_id='cuda-functional-cold',
        attempt_id=folder.name,group='P05',source_commit=ctx.plan['commit'],environment_id=digest(ctx.plan['runtime']),
        token=uuid.uuid4().hex,started_utc=utc(),work=str(work),scratch_root=str(scratch),
        timeout_seconds=60,no_progress_timeout_seconds=60,initial_evidence_bytes=tree_bytes(ROOT)+tree_bytes(CPU_ROOT),
        gpu_plan=ctx.plan,gpu_admission=ctx.runtime['admission'],recipe=recipe.to_dict(),protocol=asdict(protocol),
        argv=[str(Path(__file__).with_name('cold_worker.py')),str(folder/'request.json')])
    config=dict(read_json(REPO/'tools/final_acceptance/campaign_config.json'),max_processes=4)
    start=time.perf_counter_ns();ready_receipt=[]
    def update(*unused):
        if not ready_receipt and (folder/'cold-ready.json').is_file():ready_receipt.append(time.perf_counter_ns())
    receipt=execute(folder,request,config,dict(python=ctx.plan['runtime']['python'],runtime_paths=[ctx.plan['runtime']['site']]),
        update=update,stop_requested=lambda:False,integrity=lambda:snapshot()==ctx.plan['source_sha256'])
    end=time.perf_counter_ns()
    if not (receipt['cleanup_verified'] and receipt['cleanup']['active_processes']==0 and receipt['source_integrity']
            and receipt['payload_admitted'] and not receipt['monitor_reason'] and receipt['exit_code']==0):
        raise RuntimeError('P05_NESTED_SAFETY_OR_EXECUTION_FAILURE: '+str(folder))
    body=read_json(folder/'output.json');sealed=read_json(folder/'cold-seal.json')
    rows,issues=read_journal(folder/'events.jsonl')
    assert not issues and [r['kind'] for r in rows]==['worker_start','cold_result','session_finish']
    assert rows[1]['output_sha256']==file_hash(folder/'output.json') and rows[1]['seal_sha256']==file_hash(folder/'cold-seal.json')
    for field,name in [('request_sha256','request.json'),('bootstrap_sha256','bootstrap.py'),('journal_sha256','events.jsonl'),
            ('worker_terminal_sha256','journal-terminal.json'),('terminal_evidence_sha256','terminal-evidence.json')]:
        assert receipt[field]==file_hash(folder/name)
    terminal=read_json(folder/'journal-terminal.json')
    assert terminal['closed'] and not terminal['writer_failed'] and terminal['committed_head_sha256']==rows[-1]['sha256'] and terminal['committed_sequence']==len(rows)
    from .identity import parent_binding
    assert parent_binding(receipt['identities'],read_json(folder/'interpreter-ready.json'),receipt['launch_binding'])
    assert sealed['output_sha256']==file_hash(folder/'output.json')
    assert sealed['pid']==body['worker_pid']==receipt['identities']['interpreter']['pid']
    assert ready_receipt,'P05 parent ready receipt missing'
    ready=ready_receipt[0]
    body.update(trial_index=0,parent_start_ns=start,parent_end_ns=end,parent_elapsed_ms=(end-start)/1e6,
        exit_code=0,process_terminated=True,launcher_pid=receipt['identities']['launcher']['pid'],
        termination_policy='owned_windows_job_kill_on_close_no_breakaway',stderr_tail='',
        startup_to_ready_receipt=dict(start_ns=start,end_ns=ready,elapsed_ms=(ready-start)/1e6,
            unit='ms',clock='parent_perf_counter_ns',status='measured',reason=None,device='host',
            backend='python_subprocess',included_operations=['suspended_launch','pre_import_Job_admission','ready_file_receipt']))
    # Reuse the native P05 aggregation/validator, supplying only this retained trial.
    token=cold._owned_worker_launcher.set(lambda *args:body)
    try:result=cold.measure_cold_start(recipe,protocol=protocol).to_dict()
    finally:cold._owned_worker_launcher.reset(token)
    cleanup_owned(work,scratch,'cuda-functional-cold');scratch.rmdir()
    return dict(result=result,controller=receipt,seal=sealed,journal=rows,output_sha256=file_hash(folder/'output.json'))
