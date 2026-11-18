"""Exactly two seconds of the original hook-free GPU recipe; never a long run."""
import argparse
import csv
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--reference',type=Path,required=True)
    args=parser.parse_args();args.evidence.mkdir(parents=True,exist_ok=False)
    import torch
    from hyso.performance import StabilityProtocol,WorkloadRecipe,measure_stability,RuntimeResult
    assert not torch.cuda.is_initialized();before=torch.get_rng_state().clone()
    p=StabilityProtocol(duration_seconds=2,interval_seconds=.2,startup_timeout_seconds=60,
        step_timeout_seconds=6,chunk_rows=2,max_samples=100)
    recipe=WorkloadRecipe('runtime_support:make_gpu',dict(gpu_uuid=args.gpu_uuid),
        search_path=str(REPO/'tests'),gpu_uuid=args.gpu_uuid)
    result=measure_stability(recipe,args.evidence/'journal',protocol=p)
    data=result.to_dict();e=data['evidence'];a=data['analysis'];t=e['terminal']
    reference=json.loads(args.reference.read_text())['evidence']['header']['workload']
    identity=e['header']['workload']
    assert identity['model_sha256']==reference['model_sha256'] and identity['device_id']==reference['device_id']
    assert identity['input_shape']==[2,3,16,16] and identity['cuda_memory_fraction'] is None
    assert a['status']=='completed' and a['duration_seconds']>=2 and a['completed']>0
    assert t['status']=='completed' and t['closed'] is True and (t['work_end_ns']-t['work_start_ns'])/1e9>=2
    assert e['controller']['exit_code']==0 and e['controller']['reason'] is None
    assert e['controller']['cleanup']['active_processes']==0
    assert torch.equal(before,torch.get_rng_state()) and not torch.cuda.is_initialized()
    result.to_json(args.evidence/'result.json');result.to_csv(args.evidence/'result.csv')
    assert RuntimeResult.from_json(args.evidence/'result.json').to_dict()==data
    with (args.evidence/'result.csv').open(newline='',encoding='utf-8') as handle:
        reader=csv.DictReader(handle)
        assert reader.fieldnames==['feature_id','result_id','source_sha256','field','value']
        rows={r['field']:r['value'] for r in reader}
    assert rows['analysis.status']=='completed' and rows['evidence.controller.cleanup.active_processes']=='0'
    diagnostics=result.stability_diagnostics()
    (args.evidence/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    print(json.dumps(dict(status='PASS',scope='two_seconds_only',observed_seconds=a['duration_seconds'],
        steps=a['steps'],completed=a['completed'],model_sha256=identity['model_sha256'],
        parent_rng_preserved=True,parent_cuda_uninitialized=True,active_processes=0,long_run_started=False)),flush=True)


if __name__=='__main__':main()
