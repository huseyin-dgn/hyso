"""One small P31 CUDA acceptance run with explicit test-only resource limits."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from hyso.performance import WorkloadRecipe,StabilityProtocol,measure_stability,RuntimeResult


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True);parser.add_argument('--admit',required=True)
    parser.add_argument('--case',required=True,help='Explicit physical GPU UUID')
    args=parser.parse_args()
    deadline=time.monotonic()+15
    while not Path(args.admit).exists() and time.monotonic()<deadline:time.sleep(.01)
    assert Path(args.admit).exists()
    assert not torch.cuda.is_initialized()
    before=torch.get_rng_state().clone()
    path=Path(args.output)
    recipe=WorkloadRecipe('runtime_support:make_gpu',dict(gpu_uuid=args.case,cuda_memory_fraction=.1),
        search_path=str(Path(__file__).resolve().parent),gpu_uuid=args.case)
    protocol=StabilityProtocol(duration_seconds=2.,interval_seconds=.2,chunk_rows=2,
        process_memory_bytes=4*1024**3,job_memory_bytes=6*1024**3,max_processes=4)
    result=measure_stability(recipe,path.parent/'journal',protocol=protocol)
    data=result.to_dict();analysis=data['analysis'];diagnostics=result.stability_diagnostics()
    assert analysis['status']=='completed',data
    assert analysis['duration_seconds']>=2. and analysis['completed']>0
    assert diagnostics['core_status']=='complete' and diagnostics['advanced_status']=='partial'
    assert diagnostics['step_fluctuation']['step_count']==analysis['steps']
    assert diagnostics['memory']['allocated_bytes']['samples']>=2
    assert diagnostics['memory']['allocated_bytes']['sampled_max']>0
    assert diagnostics['step_fluctuation']['p99_ms'] is None
    assert any(p['observation']['status']=='measured' for p in diagnostics['gpu_utilization_timeline'])
    assert data['evidence']['header']['workload']['cuda_memory_fraction']==.1
    assert data['evidence']['controller']['cleanup']['active_processes']==0
    assert not torch.cuda.is_initialized() and torch.equal(before,torch.get_rng_state())
    result.to_json(path);result.to_csv(path.with_suffix('.csv'))
    assert RuntimeResult.from_json(path).to_dict()==data
    path.with_suffix('.diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='passed',observed_seconds=analysis['duration_seconds'],
        steps=analysis['steps'],completed=analysis['completed'],thermal_samples=analysis['health']['thermal_samples'],
        parent_cuda_uninitialized=True,scope='two_seconds_only_no_hours_claim')))


if __name__=='__main__':main()
