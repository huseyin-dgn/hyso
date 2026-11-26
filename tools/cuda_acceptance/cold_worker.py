"""P05 bootstrap remains stdlib-only until the established admission latch."""
from pathlib import Path
import os
import runpy
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'tools/final_acceptance')]
from common import read_json,new_json,file_hash,digest,Journal
from campaign_preflight import process_identity


def run(path,request,journal):
    folder=path.parent
    ids=read_json(folder/'admission-receipt.json');admit=read_json(folder/'admit.json')
    assert ids['no_breakaway'] and ids['kill_on_close'] and ids['limits_readback']
    assert ids['identities']['interpreter']==process_identity(os.getpid())
    assert ids['identities']['exact_job_membership'] and ids['identities']['nonce_sha256']==digest(request['token'])
    assert admit['pid']==os.getpid() and admit['token']==request['token'] and admit['readback_verified']
    journal.emit('worker_start',pid=os.getpid(),profile=request['profile'])
    os.environ['CUDA_VISIBLE_DEVICES']=request['gpu_plan']['gpu_uuid']
    recipe=folder/'recipe.json';output=folder/'output.json'
    new_json(recipe,dict(recipe=request['recipe'],protocol=request['protocol'],cuda_memory_fraction=.15))
    # Exact entry receipt includes admission and script dispatch scheduling.
    new_json(folder/'cold-ready.json',dict(pid=os.getpid(),token=request['token']))
    sys.argv=[str(REPO/'hyso/performance/_cold_worker.py'),str(recipe),str(output),str(REPO)]
    try:runpy.run_path(sys.argv[0],run_name='__main__')
    except SystemExit as exc:
        if exc.code not in (None,0):raise
    import torch
    from tools.cuda_acceptance.identity import gpu_uuid
    from tools.cuda_acceptance.worker import binary_identity
    assert gpu_uuid(torch.cuda.get_device_properties(0).uuid)==request['gpu_plan']['gpu_uuid']
    binaries={name:binary_identity(name) for name in request['gpu_plan']['runtime']['binary_expectations']}
    assert binaries==request['gpu_plan']['runtime']['binary_expectations']
    new_json(folder/'cold-seal.json',dict(output_sha256=file_hash(output),pid=os.getpid(),
        uuid=request['gpu_plan']['gpu_uuid'],loaded_binaries=binaries,
        torch_version=str(torch.__version__),cuda_build=torch.version.cuda))
    journal.emit('cold_result',output_sha256=file_hash(output),seal_sha256=file_hash(folder/'cold-seal.json'))
    journal.emit('session_finish',exit_code=0)


def main():
    path=Path(sys.argv[1]);request=read_json(path)
    identity={k:request[k] for k in ('attempt_id','group','source_commit','environment_id')}
    journal=Journal(path.parent/'events.jsonl',identity)
    try:run(path,request,journal)
    except BaseException:
        if not journal.failed:journal.emit('critical_error',reason=traceback.format_exc())
        raise
    finally:
        journal.close()
        new_json(path.parent/'journal-terminal.json',dict(schema_version=2,**identity,committed_sequence=journal.sequence,
            committed_head_sha256=journal.previous,writer_failed=journal.failed,closed=True,protocol='hyso.event-journal.v2'))


if __name__=='__main__':main()
